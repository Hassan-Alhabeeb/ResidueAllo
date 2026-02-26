"""
Baseline model training for allosteric site prediction.

FIXES from Opus review:
  - Reduced max_depth (8->5), adjusted scale_pos_weight (sqrt-based)
  - Optimal threshold from val set (not fixed 0.5)
  - Both train+val in eval_set for monitoring
  - verbose=10 for better training visibility
  - Updated STRUCTURAL_DIM to 64
"""

import os
import sys
import numpy as np
import h5py
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    matthews_corrcoef, confusion_matrix, precision_recall_curve
)
import json
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/../ = allosteric/
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(MODEL_DIR, "dataset.h5")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 on the given data."""
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    # F1 = 2 * precision * recall / (precision + recall)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)
    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def train_xgboost(data):
    """Train XGBoost with tuned hyperparams."""
    X_train, y_train = data['train']['X'], data['train']['y']
    X_val, y_val = data['val']['X'], data['val']['y']

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    # Use sqrt-based weight instead of full ratio to reduce memorization
    scale_pos_weight = np.sqrt(n_neg / max(n_pos, 1))

    print(f"Training data: {len(X_train)} samples")
    print(f"  Positive: {n_pos:.0f} ({100*n_pos/len(X_train):.1f}%)")
    print(f"  Negative: {n_neg:.0f} ({100*n_neg/len(X_train):.1f}%)")
    print(f"  scale_pos_weight: {scale_pos_weight:.1f} (sqrt-based)")

    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'aucpr'],
        'scale_pos_weight': scale_pos_weight,
        'max_depth': 5,          # Reduced from 8 to prevent memorization
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 20,  # Increased from 5 to prevent small-leaf memorization
        'gamma': 0.2,            # Increased from 0.1
        'tree_method': 'hist',
        'device': 'cpu',
        'random_state': 42,
        'n_estimators': 1000,
        'early_stopping_rounds': 30,
    }

    # Try GPU, fallback to CPU
    try:
        params_gpu = {**params, 'device': 'cuda'}
        model = xgb.XGBClassifier(**params_gpu)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=10
        )
    except Exception as e:
        print(f"GPU failed ({e}), using CPU...")
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=10
        )

    return model


def evaluate(model, X, y, split_name, threshold=0.5):
    """Evaluate model with given threshold."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auroc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0
    auprc = average_precision_score(y, y_prob) if len(np.unique(y)) > 1 else 0
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    metrics = {
        'split': split_name,
        'n_samples': len(y),
        'n_positive': int(y.sum()),
        'threshold': threshold,
        'auroc': float(auroc),
        'auprc': float(auprc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mcc': float(mcc),
        'confusion_matrix': cm.tolist()
    }

    print(f"\n{'='*40}")
    print(f"  {split_name} Results (threshold={threshold:.3f}):")
    print(f"{'='*40}")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  AUPRC:     {auprc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  MCC:       {mcc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:6d}  FP={cm[0,1]:6d}")
    print(f"    FN={cm[1,0]:6d}  TP={cm[1,1]:6d}")

    return metrics


def feature_importance(model, feature_names):
    """Get top feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\nTop 20 Features:")
    for i in range(min(20, len(indices))):
        idx = indices[i]
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"  {i+1:3d}. {name:30s}: {importances[idx]:.4f}")

    return {feature_names[i] if i < len(feature_names) else f"feature_{i}": float(importances[i])
            for i in indices[:30]}


if __name__ == '__main__':
    print("Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: {DATASET_PATH} not found. Run build_dataset.py first!")
        sys.exit(1)

    data = load_dataset(DATASET_PATH)

    print("\nTraining XGBoost baseline...")
    model = train_xgboost(data)

    # Find optimal threshold on validation set
    y_val_prob = model.predict_proba(data['val']['X'])[:, 1]
    optimal_threshold = find_optimal_threshold(data['val']['y'], y_val_prob)
    print(f"\nOptimal threshold (from val set): {optimal_threshold:.4f}")

    # Evaluate on all splits using optimal threshold
    all_metrics = {}
    for split in ['train', 'val', 'test']:
        if split in data:
            metrics = evaluate(model, data[split]['X'], data[split]['y'], split, threshold=optimal_threshold)
            all_metrics[split] = metrics

    # Save model FIRST (before feature name imports that could fail)
    model_path = os.path.join(MODEL_DIR, "xgboost_hybrid.json")
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    # Feature importance — build names for all feature groups
    sys.path.insert(0, os.path.dirname(__file__))
    from extract_features import FEATURE_NAMES as STRUCT_NAMES
    from extract_nma_graph import ALL_FEATURE_NAMES as NMA_GRAPH_NAMES
    from extract_fpocket import FPOCKET_FEATURE_NAMES
    from extract_aaindex import FEATURE_NAMES as AAINDEX_NAMES
    from extract_transfer_entropy import TE_FEATURE_NAMES
    from extract_prs import PRS_FEATURE_NAMES
    n_total = data['train']['X'].shape[1]
    n_struct = (len(STRUCT_NAMES) + len(NMA_GRAPH_NAMES) + len(FPOCKET_FEATURE_NAMES) +
                len(AAINDEX_NAMES) + len(TE_FEATURE_NAMES) + len(PRS_FEATURE_NAMES))
    n_esm_pca = n_total - n_struct
    esm_pca_names = [f'esm_pca_{i}' for i in range(n_esm_pca)]
    all_feature_names = (list(STRUCT_NAMES) + list(NMA_GRAPH_NAMES) +
                         list(FPOCKET_FEATURE_NAMES) + list(AAINDEX_NAMES) +
                         list(TE_FEATURE_NAMES) + list(PRS_FEATURE_NAMES) + esm_pca_names)
    top_features = feature_importance(model, all_feature_names)

    results = {
        'model': 'XGBoost Hybrid (structural + NMA + FPocket + ESM-2 PCA)',
        'n_features': data['train']['X'].shape[1],
        'optimal_threshold': optimal_threshold,
        'metrics': all_metrics,
        'top_features': top_features,
    }

    results_path = os.path.join(RESULTS_DIR, "xgboost_hybrid_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")
