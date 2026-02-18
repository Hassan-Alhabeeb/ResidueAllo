"""
Build training dataset combining structural features + NMA/graph + FPocket + ESM-2 embeddings.

Supports TWO ESM-2 models:
  - ESM-2 650M (1280-dim) from local extraction
  - ESM-2 3B  (2560-dim) from Kaggle extraction + alignment
  Joint PCA on concatenated [1280+2560]=3840-dim -> ESM_PCA_DIM components.
  If only one model available, PCA on that model's dim alone.

MEMORY OPTIMIZATION:
  ESM embeddings are concatenated per-protein during loading (not stored separately).
  This keeps peak RAM at ~27 GB instead of ~48 GB.

FIXES from Opus review:
  - has_esm tracks per-protein, not global kill switch
  - Assert alignment instead of silent min_len truncation
  - Updated STRUCTURAL_DIM to 64 (Q3 DSSP, no per-protein z-scores)
  - Explicit PCA svd_solver='randomized'
  - NMA+graph features (11-dim) loaded and merged with structural
  - FPocket features (8-dim) loaded and merged with structural
  - ESM zero-padded (not skipped) to keep lists in lockstep across splits
  - Dimension assertions for structural, NMA, and FPocket features
"""

import os
import gc
import time
import argparse
import numpy as np
import pandas as pd
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = r"E:\newyear\research_plan\allosteric\data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "..", "features")
ESM_650M_DIR = os.path.join(FEATURES_DIR, "esm2_embeddings")
ESM_3B_DIR = os.path.join(FEATURES_DIR, "esm2_3b_embeddings")
NMA_DIR = os.path.join(FEATURES_DIR, "nma_graph")
FPOCKET_DIR = os.path.join(FEATURES_DIR, "fpocket")
AAINDEX_DIR = os.path.join(FEATURES_DIR, "aaindex")
TE_DIR = os.path.join(FEATURES_DIR, "transfer_entropy")
OUTPUT_DIR = os.path.join(DATA_DIR, "..", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ESM_650M_DIM = 1280
ESM_3B_DIM = 2560
ESM_JOINT_DIM = ESM_650M_DIM + ESM_3B_DIM  # 3840
ESM_PCA_DIM = 128
STRUCTURAL_DIM = 64  # Base structural features
NMA_GRAPH_DIM = 11   # 6 NMA + 5 graph centrality
FPOCKET_DIM = 8      # FPocket pocket geometry features
AAINDEX_DIM = 6      # AAindex physicochemical properties
TE_DIM = 3           # Transfer entropy (AllosES: nte_score, te_out_sum, te_in_sum)


def parse_args():
    parser = argparse.ArgumentParser(description="Build training dataset")
    parser.add_argument('--esm-650m-only', action='store_true',
                        help='Force 650M-only PCA (ignore 3B embeddings). '
                             'Use this when CASBench evaluation only has 650M.')
    return parser.parse_args()


def load_and_combine():
    """Load all features and labels, combine into arrays."""
    args = parse_args()

    print("=" * 60)
    print("Building Dataset")
    if args.esm_650m_only:
        print("  ** ESM-2 MODE: 650M-only (3B embeddings will be SKIPPED) **")
    print("=" * 60)

    splits_path = os.path.join(PROCESSED_DIR, "train_val_test_splits.csv")
    if not os.path.exists(splits_path):
        print("ERROR: Run cluster_and_split.py first!")
        return

    splits = pd.read_csv(splits_path)
    pdb_to_split = dict(zip(splits['pdb_id'], splits['split']))
    print(f"Total proteins in splits file: {len(pdb_to_split)}")
    print(f"  train: {sum(1 for v in pdb_to_split.values() if v == 'train')}")
    print(f"  val:   {sum(1 for v in pdb_to_split.values() if v == 'val')}")
    print(f"  test:  {sum(1 for v in pdb_to_split.values() if v == 'test')}")

    all_structural = {'train': [], 'val': [], 'test': []}
    all_esm_joint = {'train': [], 'val': [], 'test': []}  # Combined 650M+3B per-protein
    all_labels = {'train': [], 'val': [], 'test': []}

    n_proteins = 0
    n_skipped_no_feat = 0
    n_with_esm650 = 0
    n_without_esm650 = 0
    n_with_esm3b = 0
    n_without_esm3b = 0
    n_with_nma = 0
    n_without_nma = 0
    n_with_fpocket = 0
    n_without_fpocket = 0
    n_with_aaindex = 0
    n_without_aaindex = 0
    n_with_te = 0
    n_without_te = 0
    n_nma_length_mismatch = 0
    n_fpocket_length_mismatch = 0
    n_aaindex_length_mismatch = 0
    n_te_length_mismatch = 0
    n_esm650_length_mismatch = 0
    n_esm3b_length_mismatch = 0
    total_residues = 0
    total_allosteric = 0

    start_time = time.time()
    print(f"\nLoading per-protein features...")

    for pdb_id, split in pdb_to_split.items():
        feat_path = os.path.join(FEATURES_DIR, f"{pdb_id}_features.npz")
        if not os.path.exists(feat_path):
            n_skipped_no_feat += 1
            continue

        feat_data = np.load(feat_path)
        features = feat_data['features']
        labels = feat_data['labels']

        # Validate dimensions
        assert len(features) == len(labels), \
            f"{pdb_id}: feature/label length mismatch: {len(features)} vs {len(labels)}"
        assert features.shape[1] == STRUCTURAL_DIM, \
            f"{pdb_id}: expected {STRUCTURAL_DIM} structural features, got {features.shape[1]}"

        total_residues += len(labels)
        total_allosteric += labels.sum()

        # Load NMA+graph features (11-dim)
        nma_path = os.path.join(NMA_DIR, f"{pdb_id}_nma_graph.npz")
        if os.path.exists(nma_path):
            nma_data = np.load(nma_path)
            nma_feat = nma_data['features']
            assert nma_feat.shape[1] == NMA_GRAPH_DIM, \
                f"{pdb_id}: expected {NMA_GRAPH_DIM} NMA features, got {nma_feat.shape[1]}"
            if len(nma_feat) == len(features):
                features = np.concatenate([features, nma_feat], axis=1)
                n_with_nma += 1
            else:
                print(f"  WARNING: {pdb_id} NMA length mismatch ({len(nma_feat)} vs {len(features)}), padding with zeros")
                features = np.concatenate([features, np.zeros((len(features), NMA_GRAPH_DIM))], axis=1)
                n_without_nma += 1
                n_nma_length_mismatch += 1
        else:
            features = np.concatenate([features, np.zeros((len(features), NMA_GRAPH_DIM))], axis=1)
            n_without_nma += 1

        # Load FPocket features (8-dim)
        fpocket_path = os.path.join(FPOCKET_DIR, f"{pdb_id}_fpocket.npz")
        if os.path.exists(fpocket_path):
            fpocket_data = np.load(fpocket_path)
            fpocket_feat = fpocket_data['features']
            assert fpocket_feat.shape[1] == FPOCKET_DIM, \
                f"{pdb_id}: expected {FPOCKET_DIM} FPocket features, got {fpocket_feat.shape[1]}"
            if len(fpocket_feat) == len(labels):
                features = np.concatenate([features, fpocket_feat], axis=1)
                n_with_fpocket += 1
            else:
                print(f"  WARNING: {pdb_id} FPocket length mismatch ({len(fpocket_feat)} vs {len(labels)}), padding with zeros")
                features = np.concatenate([features, np.zeros((len(labels), FPOCKET_DIM), dtype=np.float32)], axis=1)
                n_without_fpocket += 1
                n_fpocket_length_mismatch += 1
        else:
            features = np.concatenate([features, np.zeros((len(labels), FPOCKET_DIM), dtype=np.float32)], axis=1)
            n_without_fpocket += 1

        # Load AAindex features (6-dim)
        aaindex_path = os.path.join(AAINDEX_DIR, f"{pdb_id}_aaindex.npz")
        if os.path.exists(aaindex_path):
            aaindex_data = np.load(aaindex_path)
            aaindex_feat = aaindex_data['features']
            assert aaindex_feat.shape[1] == AAINDEX_DIM, \
                f"{pdb_id}: expected {AAINDEX_DIM} AAindex features, got {aaindex_feat.shape[1]}"
            if len(aaindex_feat) == len(labels):
                features = np.concatenate([features, aaindex_feat], axis=1)
                n_with_aaindex += 1
            else:
                print(f"  WARNING: {pdb_id} AAindex length mismatch ({len(aaindex_feat)} vs {len(labels)}), padding with zeros")
                features = np.concatenate([features, np.zeros((len(labels), AAINDEX_DIM), dtype=np.float32)], axis=1)
                n_without_aaindex += 1
                n_aaindex_length_mismatch += 1
        else:
            features = np.concatenate([features, np.zeros((len(labels), AAINDEX_DIM), dtype=np.float32)], axis=1)
            n_without_aaindex += 1

        # Load Transfer Entropy features (3-dim)
        te_path = os.path.join(TE_DIR, f"{pdb_id}_te.npz")
        if os.path.exists(te_path):
            te_data = np.load(te_path)
            te_feat = te_data['features']
            assert te_feat.shape[1] == TE_DIM, \
                f"{pdb_id}: expected {TE_DIM} TE features, got {te_feat.shape[1]}"
            if len(te_feat) == len(labels):
                features = np.concatenate([features, te_feat], axis=1)
                n_with_te += 1
            else:
                print(f"  WARNING: {pdb_id} TE length mismatch ({len(te_feat)} vs {len(labels)}), padding with zeros")
                features = np.concatenate([features, np.zeros((len(labels), TE_DIM), dtype=np.float32)], axis=1)
                n_without_te += 1
                n_te_length_mismatch += 1
        else:
            features = np.concatenate([features, np.zeros((len(labels), TE_DIM), dtype=np.float32)], axis=1)
            n_without_te += 1

        # Load ESM-2 embeddings (both models, concatenated per-protein)
        # This avoids storing 650M and 3B in separate arrays (~48 GB peak -> ~27 GB)
        n_res = len(labels)
        esm_alloc_dim = ESM_650M_DIM if args.esm_650m_only else ESM_JOINT_DIM
        esm_joint = np.zeros((n_res, esm_alloc_dim), dtype=np.float32)

        # 650M: fills columns [0:1280]
        esm650_path = os.path.join(ESM_650M_DIR, f"{pdb_id}_esm2.npz")
        if os.path.exists(esm650_path):
            esm650_emb = np.load(esm650_path)['embeddings']
            if len(esm650_emb) == n_res:
                esm_joint[:, :ESM_650M_DIM] = esm650_emb
                n_with_esm650 += 1
            else:
                n_without_esm650 += 1
                n_esm650_length_mismatch += 1
            del esm650_emb
        else:
            n_without_esm650 += 1

        # 3B: fills columns [1280:3840] (skipped if --esm-650m-only)
        if not args.esm_650m_only:
            esm3b_path = os.path.join(ESM_3B_DIR, f"{pdb_id}_esm2_3b.npz")
            if os.path.exists(esm3b_path):
                esm3b_emb = np.load(esm3b_path)['embeddings']
                if len(esm3b_emb) == n_res:
                    esm_joint[:, ESM_650M_DIM:] = esm3b_emb
                    n_with_esm3b += 1
                else:
                    n_without_esm3b += 1
                    n_esm3b_length_mismatch += 1
                del esm3b_emb
            else:
                n_without_esm3b += 1
        # else: 3B columns stay as zeros -> has_esm3b will be False -> trimmed automatically

        all_structural[split].append(features)
        all_esm_joint[split].append(esm_joint)
        all_labels[split].append(labels)
        n_proteins += 1

        # Progress every 500 proteins
        if n_proteins % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  ... loaded {n_proteins} proteins ({elapsed:.0f}s)")

    elapsed = time.time() - start_time
    total_struct_dim = STRUCTURAL_DIM + NMA_GRAPH_DIM + FPOCKET_DIM + AAINDEX_DIM + TE_DIM

    print(f"\n{'-' * 40}")
    print(f"Loading complete ({elapsed:.1f}s)")
    print(f"{'-' * 40}")
    print(f"  Proteins loaded:      {n_proteins}")
    print(f"  Skipped (no features):{n_skipped_no_feat}")
    print(f"  Total residues:       {total_residues:,}")
    print(f"  Allosteric residues:  {total_allosteric:,.0f} ({100*total_allosteric/max(total_residues,1):.1f}%)")
    print(f"  NMA+graph available:  {n_with_nma} / {n_proteins}")
    if n_nma_length_mismatch > 0:
        print(f"    Length mismatches:  {n_nma_length_mismatch}")
    print(f"  FPocket available:    {n_with_fpocket} / {n_proteins}")
    if n_fpocket_length_mismatch > 0:
        print(f"    Length mismatches:  {n_fpocket_length_mismatch}")
    print(f"  AAindex available:    {n_with_aaindex} / {n_proteins}")
    if n_aaindex_length_mismatch > 0:
        print(f"    Length mismatches:  {n_aaindex_length_mismatch}")
    print(f"  Transfer Entropy:     {n_with_te} / {n_proteins}")
    if n_te_length_mismatch > 0:
        print(f"    Length mismatches:  {n_te_length_mismatch}")
    print(f"  ESM-2 650M available: {n_with_esm650} / {n_proteins}")
    if n_esm650_length_mismatch > 0:
        print(f"    Length mismatches:  {n_esm650_length_mismatch}")
    print(f"  ESM-2 3B available:   {n_with_esm3b} / {n_proteins}")
    if n_esm3b_length_mismatch > 0:
        print(f"    Length mismatches:  {n_esm3b_length_mismatch}")

    # Concatenate per-split lists into big arrays
    print(f"\nConcatenating per-split arrays...")
    for split in ['train', 'val', 'test']:
        if all_structural[split]:
            all_structural[split] = np.concatenate(all_structural[split], axis=0)
            all_labels[split] = np.concatenate(all_labels[split], axis=0)
            all_esm_joint[split] = np.concatenate(all_esm_joint[split], axis=0)
            assert len(all_esm_joint[split]) == len(all_structural[split]), \
                f"{split}: ESM/structural row mismatch after concat"
            assert all_structural[split].shape[1] == total_struct_dim, \
                f"{split}: expected {total_struct_dim} structural+NMA+FPocket features, got {all_structural[split].shape[1]}"
            n_res = all_structural[split].shape[0]
            n_pos = all_labels[split].sum()
            print(f"  {split:5s}: {n_res:>10,} residues, {n_pos:>6,.0f} allosteric ({100*n_pos/n_res:.1f}%), "
                  f"structural {all_structural[split].shape}, "
                  f"ESM joint {all_esm_joint[split].shape}")
        else:
            print(f"  {split:5s}: EMPTY")

    gc.collect()

    # Step 1: Normalize structural+NMA features using train statistics
    print(f"\n{'-' * 40}")
    print(f"Step 1: StandardScaler on structural+NMA+FPocket ({total_struct_dim}-dim)")
    print(f"{'-' * 40}")
    if not isinstance(all_structural['train'], np.ndarray) or len(all_structural['train']) == 0:
        print("ERROR: No training data loaded. Check feature files and splits CSV.")
        return
    scaler = StandardScaler()
    all_structural['train'] = scaler.fit_transform(all_structural['train'])
    print(f"  Fitted on train ({all_structural['train'].shape[0]:,} samples)")
    print(f"  Feature means (first 5): {scaler.mean_[:5].round(4)}")
    print(f"  Feature stds  (first 5): {scaler.scale_[:5].round(4)}")

    for split in ['val', 'test']:
        if isinstance(all_structural[split], np.ndarray) and len(all_structural[split]) > 0:
            all_structural[split] = scaler.transform(all_structural[split])
            print(f"  Transformed {split} ({all_structural[split].shape[0]:,} samples)")

    # Check for NaN/inf after scaling
    for split in ['train', 'val', 'test']:
        if isinstance(all_structural[split], np.ndarray):
            n_nan = np.isnan(all_structural[split]).sum()
            n_inf = np.isinf(all_structural[split]).sum()
            if n_nan > 0 or n_inf > 0:
                print(f"  WARNING: {split} has {n_nan} NaN and {n_inf} Inf values -- replacing with 0")
            all_structural[split] = np.nan_to_num(all_structural[split], nan=0.0, posinf=0.0, neginf=0.0)

    scaler_path = os.path.join(OUTPUT_DIR, "feature_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")

    # Step 2: PCA on joint ESM-2 embeddings
    print(f"\n{'-' * 40}")
    print(f"Step 2: PCA on ESM-2 embeddings (650M + 3B joint)")
    print(f"{'-' * 40}")

    has_esm650 = n_with_esm650 > n_proteins * 0.5
    has_esm3b = n_with_esm3b > n_proteins * 0.5
    has_esm = has_esm650 or has_esm3b
    esm_parts = []
    esm_raw_dim = 0

    if has_esm:
        if has_esm650:
            print(f"  ESM-2 650M: {n_with_esm650}/{n_proteins} proteins ({100*n_with_esm650/n_proteins:.0f}%)")
            esm_parts.append('650M')
        else:
            print(f"  ESM-2 650M: {n_with_esm650}/{n_proteins} -- below 50%, zero columns kept")
        if has_esm3b:
            print(f"  ESM-2 3B:   {n_with_esm3b}/{n_proteins} proteins ({100*n_with_esm3b/n_proteins:.0f}%)")
            esm_parts.append('3B')
        else:
            print(f"  ESM-2 3B:   {n_with_esm3b}/{n_proteins} -- below 50%, zero columns kept")

        # If only one model has >50%, drop the zero columns for the other
        if has_esm650 and not has_esm3b:
            # Keep only first 1280 columns (650M)
            esm_raw_dim = ESM_650M_DIM
            for split in ['train', 'val', 'test']:
                if isinstance(all_esm_joint[split], np.ndarray):
                    all_esm_joint[split] = all_esm_joint[split][:, :ESM_650M_DIM]
            print(f"  Trimmed to 650M-only: {ESM_650M_DIM} dims")
        elif has_esm3b and not has_esm650:
            # Keep only last 2560 columns (3B)
            esm_raw_dim = ESM_3B_DIM
            for split in ['train', 'val', 'test']:
                if isinstance(all_esm_joint[split], np.ndarray):
                    all_esm_joint[split] = all_esm_joint[split][:, ESM_650M_DIM:]
            print(f"  Trimmed to 3B-only: {ESM_3B_DIM} dims")
        else:
            esm_raw_dim = ESM_JOINT_DIM
            n_both = min(n_with_esm650, n_with_esm3b)
            print(f"  Both models active: {ESM_JOINT_DIM} dims (overlap ~{n_both}/{n_proteins})")

        print(f"  PCA: {esm_raw_dim} -> {ESM_PCA_DIM} components")

        pca = PCA(n_components=ESM_PCA_DIM, random_state=42, svd_solver='randomized')
        t0 = time.time()
        all_esm_joint['train'] = pca.fit_transform(all_esm_joint['train'])
        print(f"  Fitted on train ({time.time()-t0:.1f}s)")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"  Top 5 component variances: {pca.explained_variance_ratio_[:5].round(4)}")

        for split in ['val', 'test']:
            if isinstance(all_esm_joint[split], np.ndarray) and len(all_esm_joint[split]) > 0:
                all_esm_joint[split] = pca.transform(all_esm_joint[split])
                print(f"  Transformed {split}: {all_esm_joint[split].shape}")

        pca_path = os.path.join(OUTPUT_DIR, "esm2_joint_pca.joblib")
        joblib.dump(pca, pca_path)
        print(f"  PCA saved to {pca_path}")
        gc.collect()
    else:
        print(f"  No ESM model has >50% coverage -- SKIPPING ESM entirely")
        esm_raw_dim = 0
        del all_esm_joint
        gc.collect()

    # Step 3: Combine and save
    print(f"\n{'-' * 40}")
    print(f"Step 3: Combine and save")
    print(f"{'-' * 40}")
    combined = {}
    for split in ['train', 'val', 'test']:
        if isinstance(all_structural[split], np.ndarray):
            if has_esm:
                combined[split] = np.concatenate([all_structural[split], all_esm_joint[split]], axis=1)
            else:
                combined[split] = all_structural[split]
            print(f"  {split:5s}: {combined[split].shape}  "
                  f"(dtype={combined[split].dtype}, "
                  f"size={combined[split].nbytes / 1024**2:.1f} MB)")

    # Free intermediate arrays
    del all_structural
    if has_esm:
        del all_esm_joint
    gc.collect()

    # Verify all splits have same feature count
    dims = [combined[s].shape[1] for s in ['train', 'val', 'test'] if s in combined]
    assert len(set(dims)) == 1, f"Feature dimension mismatch across splits: {dims}"
    total_dim = dims[0]
    print(f"\n  Total feature dimension: {total_dim}")
    print(f"    Structural: {STRUCTURAL_DIM}")
    print(f"    NMA+graph:  {NMA_GRAPH_DIM}")
    print(f"    FPocket:    {FPOCKET_DIM}")
    print(f"    AAindex:    {AAINDEX_DIM}")
    print(f"    TE:         {TE_DIM}")
    if has_esm:
        esm_desc = '+'.join(esm_parts)
        print(f"    ESM PCA:    {ESM_PCA_DIM} (from {esm_desc}, raw {esm_raw_dim}-dim)")

    # Save hybrid dataset
    output_path = os.path.join(OUTPUT_DIR, "dataset.h5")
    print(f"\n  Saving hybrid dataset to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        for split in ['train', 'val', 'test']:
            if split in combined:
                grp = f.create_group(split)
                grp.create_dataset('features', data=combined[split], compression='gzip')
                grp.create_dataset('labels', data=all_labels[split], compression='gzip')
        f.attrs['n_structural_features'] = STRUCTURAL_DIM
        f.attrs['n_nma_graph_features'] = NMA_GRAPH_DIM
        f.attrs['n_aaindex_features'] = AAINDEX_DIM
        f.attrs['n_te_features'] = TE_DIM
        f.attrs['n_fpocket_features'] = FPOCKET_DIM
        f.attrs['n_esm_features'] = ESM_PCA_DIM if has_esm else 0
        f.attrs['esm_raw_dim'] = esm_raw_dim
        f.attrs['has_esm650'] = has_esm650
        f.attrs['has_esm3b'] = has_esm3b
        f.attrs['total_features'] = total_dim
        f.attrs['has_esm'] = has_esm
    file_size = os.path.getsize(output_path) / 1024**2
    print(f"  Done! ({file_size:.1f} MB)")

    # Structural-only version (includes NMA+graph+FPocket, no ESM)
    struct_output = os.path.join(OUTPUT_DIR, "dataset_structural_only.h5")
    print(f"\n  Saving structural+NMA+FPocket dataset to {struct_output}...")
    with h5py.File(struct_output, 'w') as f:
        for split in ['train', 'val', 'test']:
            if split in combined:
                grp = f.create_group(split)
                # Extract just the structural columns (first total_struct_dim)
                grp.create_dataset('features', data=combined[split][:, :total_struct_dim], compression='gzip')
                grp.create_dataset('labels', data=all_labels[split], compression='gzip')
        f.attrs['n_features'] = total_struct_dim
    file_size = os.path.getsize(struct_output) / 1024**2
    print(f"  Done! ({file_size:.1f} MB)")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Dataset build complete ({total_time:.0f}s)")
    print(f"  Hybrid:         {output_path} ({total_dim}-dim)")
    print(f"  Structural+NMA+FPocket: {struct_output} ({total_struct_dim}-dim)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    load_and_combine()
