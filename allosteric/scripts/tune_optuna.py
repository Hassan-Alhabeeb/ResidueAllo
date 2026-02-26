"""
Optuna hyperparameter tuning for XGBoost allosteric site prediction.

Optimizes for AUPRC (our real metric given 1.7% class imbalance).
Uses the hybrid dataset (structural + NMA/graph + ESM-2).
Saves best params, retrains final model, evaluates on test set.
Persists to SQLite — crash-safe, can resume if interrupted.
"""

import os
import sys
import numpy as np
import h5py
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import optuna
from optuna.samplers import TPESampler
import json
import time
import logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/../ = allosteric/
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set up dual logging: console + file
LOG_PATH = os.path.join(RESULTS_DIR, f"tune_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


class DualLogger:
    """Writes to both stdout and a log file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8', buffering=1)  # line-buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = DualLogger(LOG_PATH)

# Use the full hybrid dataset by default
DATASET_PATH = os.path.join(MODEL_DIR, "dataset.h5")

N_TRIALS = 100
SEED = 42

# Suppress Optuna's internal logging (we have our own callback)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_dataset(path):
    """Load train/val/test from HDF5."""
    data = {}
    with h5py.File(path, 'r') as f:
        for split in ['train', 'val', 'test']:
            if split in f:
                data[split] = {
                    'X': f[split]['features'][:],
                    'y': f[split]['labels'][:].astype(np.float64)
                }
    return data


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective: maximize AUPRC on validation set."""

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',  # Use AUPRC for early stopping (invariant to probability shifts from scale_pos_weight)
        'tree_method': 'hist',
        'random_state': SEED,

        # Core tree params
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'n_estimators': 2000,  # High, rely on early stopping
        'early_stopping_rounds': 50,

        # Regularization
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 100),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),

        # Class imbalance handling
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, n_neg / max(n_pos, 1)),
    }

    # Try GPU, fallback to CPU only on GPU-specific errors
    try:
        params['device'] = 'cuda'
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    except xgb.core.XGBoostError:
        params['device'] = 'cpu'
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    y_val_prob = model.predict_proba(X_val)[:, 1]
    auprc = average_precision_score(y_val, y_val_prob)

    # Also report AUROC for monitoring
    auroc = roc_auc_score(y_val, y_val_prob)
    trial.set_user_attr('val_auroc', auroc)
    trial.set_user_attr('n_estimators_used', model.best_iteration + 1)
    trial.set_user_attr('device', params['device'])

    return auprc


def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1."""
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def main():
    print("=" * 70)
    print("  Optuna Hyperparameter Tuning for XGBoost")
    print("  Optimizing: AUPRC (validation set)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Log file: {LOG_PATH}")
    print("=" * 70)

    if not os.path.exists(DATASET_PATH):
        print(f"\nERROR: {DATASET_PATH} not found. Run build_dataset.py first!")
        sys.exit(1)

    # Load data
    print(f"\nLoading dataset: {DATASET_PATH}")
    t0 = time.time()
    data = load_dataset(DATASET_PATH)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    X_train, y_train = data['train']['X'], data['train']['y']
    X_val, y_val = data['val']['X'], data['val']['y']
    X_test, y_test = data['test']['X'], data['test']['y']

    n_features = X_train.shape[1]
    n_pos_train = y_train.sum()
    n_neg_train = len(y_train) - n_pos_train
    imbalance_ratio = n_neg_train / max(n_pos_train, 1)

    print(f"\n{'-' * 50}")
    print(f"  Dataset Summary")
    print(f"{'-' * 50}")
    print(f"  Features:         {n_features}")
    print(f"  Train:            {len(X_train):>10,} samples ({n_pos_train:,.0f} pos, {100*n_pos_train/len(X_train):.1f}%)")
    print(f"  Val:              {len(X_val):>10,} samples ({y_val.sum():,.0f} pos, {100*y_val.sum()/len(X_val):.1f}%)")
    print(f"  Test:             {len(X_test):>10,} samples ({y_test.sum():,.0f} pos, {100*y_test.sum()/len(X_test):.1f}%)")
    print(f"  Imbalance ratio:  {imbalance_ratio:.1f}:1 (neg:pos)")
    print(f"  scale_pos_weight: search [1.0, {imbalance_ratio:.1f}]")

    # Check GPU
    print(f"\n{'-' * 50}")
    print(f"  GPU Check")
    print(f"{'-' * 50}")
    try:
        test_model = xgb.XGBClassifier(n_estimators=1, device='cuda', tree_method='hist')
        test_model.fit(X_train[:100], y_train[:100], verbose=False)
        gpu_available = True
        print(f"  CUDA GPU: Available")
    except Exception:
        gpu_available = False
        print(f"  CUDA GPU: Not available (using CPU)")

    # Create/resume study
    print(f"\n{'-' * 50}")
    print(f"  Optuna Study")
    print(f"{'-' * 50}")
    db_path = os.path.join(RESULTS_DIR, "optuna_study.db")
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=SEED),
        study_name='xgboost_allosteric',
        storage=storage,
        load_if_exists=True
    )

    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, N_TRIALS - n_completed)
    if n_completed > 0:
        print(f"  Resuming study: {n_completed} trials already completed")
        print(f"  Best so far: AUPRC = {study.best_value:.4f}")
        print(f"  Remaining:   {n_remaining} trials")
    else:
        print(f"  New study: {N_TRIALS} trials")
    print(f"  Storage:   {db_path}")

    if n_remaining == 0:
        print(f"\n  All {N_TRIALS} trials already completed! Skipping to final retrain.")
    else:
        # Run optimization
        print(f"\n{'-' * 50}")
        print(f"  Running Trials")
        print(f"{'-' * 50}")
        print(f"  {'Trial':>7} | {'AUPRC':>7} | {'AUROC':>7} | {'Trees':>5} | {'Depth':>5} | {'LR':>8} | {'Best':>7} | {'Time':>6} | {'ETA':>8}")
        print(f"  {'-'*7} | {'-'*7} | {'-'*7} | {'-'*5} | {'-'*5} | {'-'*8} | {'-'*7} | {'-'*6} | {'-'*8}")

        start_time = time.time()
        trial_times = []

        def callback(study, trial):
            elapsed = time.time() - start_time
            trial_time = elapsed / (trial.number - n_completed + 1)
            trial_times.append(trial_time)

            best = study.best_value
            auroc = trial.user_attrs.get('val_auroc', 0)
            n_est = trial.user_attrs.get('n_estimators_used', 0)
            depth = trial.params.get('max_depth', 0)
            lr = trial.params.get('learning_rate', 0)

            # ETA based on rolling average of last 5 trials
            recent_avg = np.mean(trial_times[-5:]) if trial_times else trial_time
            trials_left = N_TRIALS - trial.number - 1
            eta_seconds = recent_avg * trials_left

            if eta_seconds > 3600:
                eta_str = f"{eta_seconds/3600:.1f}h"
            elif eta_seconds > 60:
                eta_str = f"{eta_seconds/60:.0f}m"
            else:
                eta_str = f"{eta_seconds:.0f}s"

            # Mark if this trial is the new best
            is_best = " *" if trial.value >= best else ""

            print(f"  {trial.number:>5d}   | {trial.value:.4f}{is_best} | {auroc:.4f} | {n_est:>5d} | {depth:>5d} | {lr:>8.5f} | {best:.4f} | {elapsed:>5.0f}s | ~{eta_str}")

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_remaining,
            callbacks=[callback],
            show_progress_bar=False
        )

        elapsed = time.time() - start_time
        print(f"\n{'-' * 50}")
        print(f"  Tuning Summary")
        print(f"{'-' * 50}")
        print(f"  Total time:    {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"  Avg per trial: {elapsed/max(n_remaining,1):.1f}s")
        print(f"  Best AUPRC:    {study.best_value:.4f}")
        print(f"  Best trial:    #{study.best_trial.number}")

    # Show best params
    print(f"\n{'-' * 50}")
    print(f"  Best Hyperparameters")
    print(f"{'-' * 50}")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.6f}")
        else:
            print(f"  {k:25s}: {v}")

    # Retrain with best params
    print(f"\n{'=' * 70}")
    print(f"  Retraining Final Model with Best Params")
    print(f"{'=' * 70}")
    best_params = study.best_params.copy()

    final_params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'aucpr'],
        'tree_method': 'hist',
        'random_state': SEED,
        'n_estimators': 2000,
        'early_stopping_rounds': 50,
        **best_params
    }

    print(f"\n  Training with early stopping (max 2000 rounds, patience 50)...")
    t0 = time.time()
    try:
        final_params['device'] = 'cuda'
        model = xgb.XGBClassifier(**final_params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=10)
        print(f"  Device: CUDA GPU")
    except xgb.core.XGBoostError:
        final_params['device'] = 'cpu'
        model = xgb.XGBClassifier(**final_params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=10)
        print(f"  Device: CPU")

    train_time = time.time() - t0
    print(f"  Training time: {train_time:.0f}s")
    print(f"  Best iteration: {model.best_iteration + 1} / 2000")

    # Optimal threshold from val
    y_val_prob = model.predict_proba(X_val)[:, 1]
    optimal_threshold = find_optimal_threshold(y_val, y_val_prob)
    print(f"  Optimal threshold (from val F1): {optimal_threshold:.4f}")

    # Evaluate on all splits
    from sklearn.metrics import (
        precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
    )

    print(f"\n{'=' * 70}")
    print(f"  Final Evaluation")
    print(f"{'=' * 70}")

    all_metrics = {}
    for split_name in ['train', 'val', 'test']:
        if split_name not in data:
            continue
        X, y = data[split_name]['X'], data[split_name]['y']
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= optimal_threshold).astype(int)

        auroc = roc_auc_score(y, y_prob)
        auprc = average_precision_score(y, y_prob)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
        mcc = matthews_corrcoef(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        all_metrics[split_name] = {
            'auroc': float(auroc),
            'auprc': float(auprc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'mcc': float(mcc),
            'n_samples': len(y),
            'n_positive': int(y.sum()),
            'threshold': optimal_threshold,
            'confusion_matrix': cm.tolist()
        }

        print(f"\n  {split_name.upper()} (n={len(y):,}, pos={int(y.sum()):,}, threshold={optimal_threshold:.3f})")
        print(f"  {'-' * 45}")
        print(f"    AUROC:     {auroc:.4f}")
        print(f"    AUPRC:     {auprc:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1:        {f1:.4f}")
        print(f"    MCC:       {mcc:.4f}")
        print(f"    TN={cm[0,0]:>7,}  FP={cm[0,1]:>6,}")
        print(f"    FN={cm[1,0]:>7,}  TP={cm[1,1]:>6,}")

    # Compare with previous best
    train_auroc = all_metrics['train']['auroc']
    val_auroc = all_metrics['val']['auroc']
    test_auroc = all_metrics['test']['auroc']
    test_auprc = all_metrics['test']['auprc']
    gap = train_auroc - test_auroc

    print(f"\n  {'-' * 45}")
    print(f"  Overfitting gap: {gap:.3f} (train {train_auroc:.4f} - test {test_auroc:.4f})")
    print(f"  Previous best:   Test AUROC 0.895 | Test AUPRC 0.214")
    print(f"  This model:      Test AUROC {test_auroc:.3f} | Test AUPRC {test_auprc:.3f}")
    auroc_delta = test_auroc - 0.895
    auprc_delta = test_auprc - 0.214
    print(f"  Delta:           AUROC {'+' if auroc_delta >= 0 else ''}{auroc_delta:.3f} | AUPRC {'+' if auprc_delta >= 0 else ''}{auprc_delta:.3f}")

    # Feature importance
    print(f"\n{'=' * 70}")
    print(f"  Top 20 Feature Importances")
    print(f"{'=' * 70}")

    sys.path.insert(0, os.path.dirname(__file__))
    from extract_features import FEATURE_NAMES
    from extract_nma_graph import ALL_FEATURE_NAMES as NMA_GRAPH_NAMES
    from extract_fpocket import FPOCKET_FEATURE_NAMES
    from extract_aaindex import FEATURE_NAMES as AAINDEX_NAMES
    from extract_transfer_entropy import TE_FEATURE_NAMES
    from extract_prs import PRS_FEATURE_NAMES

    all_names = (list(FEATURE_NAMES) + list(NMA_GRAPH_NAMES) +
                 list(FPOCKET_FEATURE_NAMES) + list(AAINDEX_NAMES) +
                 list(TE_FEATURE_NAMES) + list(PRS_FEATURE_NAMES))
    if n_features > len(all_names):
        n_esm = n_features - len(all_names)
        all_names = all_names + [f"esm_pca_{i}" for i in range(n_esm)]

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(20, len(indices))):
        idx = indices[i]
        name = all_names[idx] if idx < len(all_names) else f"feature_{idx}"
        bar = "#" * int(importances[idx] * 200)
        print(f"  {i+1:3d}. {name:30s}: {importances[idx]:.4f} {bar}")

    top_features = {all_names[i] if i < len(all_names) else f"feature_{i}": float(importances[i])
                    for i in indices[:30]}

    # Count NMA/graph features in top 20
    nma_graph_set = set(NMA_GRAPH_NAMES)
    nma_in_top20 = sum(1 for i in indices[:20] if i < len(all_names) and all_names[i] in nma_graph_set)
    print(f"\n  NMA/graph features in top 20: {nma_in_top20}/11")

    # Save model
    print(f"\n{'=' * 70}")
    print(f"  Saving Results")
    print(f"{'=' * 70}")

    model_path = os.path.join(MODEL_DIR, "xgboost_tuned.json")
    model.save_model(model_path)
    model_size = os.path.getsize(model_path) / 1024**2
    print(f"  Model:   {model_path} ({model_size:.1f} MB)")

    # Save results
    results = {
        'model': 'XGBoost Optuna-tuned',
        'dataset': DATASET_PATH,
        'n_features': n_features,
        'n_trials': N_TRIALS,
        'best_auprc_val': study.best_value,
        'optimal_threshold': optimal_threshold,
        'best_params': study.best_params,
        'best_n_estimators': model.best_iteration + 1,
        'tuning_time_seconds': elapsed if n_remaining > 0 else 0,
        'metrics': all_metrics,
        'top_features': top_features,
    }

    results_path = os.path.join(RESULTS_DIR, "xgboost_tuned_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: {results_path}")

    # Save study trials for analysis
    trials_data = []
    for t in study.trials:
        trials_data.append({
            'number': t.number,
            'value': t.value,
            'params': t.params,
            'user_attrs': t.user_attrs,
        })

    trials_path = os.path.join(RESULTS_DIR, "optuna_trials.json")
    with open(trials_path, 'w') as f:
        json.dump(trials_data, f, indent=2, default=str)
    print(f"  Trials:  {trials_path}")

    print(f"\n{'=' * 70}")
    print(f"  DONE!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
