"""
Extract AAindex physicochemical properties per residue.

Properties chosen to be NON-REDUNDANT with existing features:
  Already have: hydrophobicity (Kyte-Doolittle), charge, mol_weight, is_aromatic, is_polar

New properties (6 features):
  1. vdw_volume       - FAUJ880103 - Van der Waals volume (top-3 in AlloPED paper)
  2. flexibility      - BHAR880101 - Bhaskaran-Ponnuswamy flexibility index
  3. polarity         - GRAR740102 - Grantham polarity (continuous, unlike binary is_polar)
  4. bulkiness        - ZIMJ680102 - Zimmerman bulkiness
  5. helix_propensity - CHOP780201 - Chou-Fasman alpha-helix propensity
  6. sheet_propensity - CHOP780202 - Chou-Fasman beta-sheet propensity

All values are min-max normalized to [0, 1] using the full amino acid range.

Usage:
    python extract_aaindex.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/../ = allosteric/
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "..", "features", "aaindex")
os.makedirs(FEATURES_DIR, exist_ok=True)

# ── AAindex lookup tables (20 standard amino acids) ─────────────────────────
# Values from https://www.genome.jp/aaindex/

# FAUJ880103 - Normalized Van der Waals volume (Fauchere et al., 1988)
VDW_VOLUME = {
    'ALA': 1.00, 'ARG': 6.13, 'ASN': 2.95, 'ASP': 2.78, 'CYS': 2.43,
    'GLN': 3.95, 'GLU': 3.78, 'GLY': 0.00, 'HIS': 4.66, 'ILE': 4.00,
    'LEU': 4.00, 'LYS': 4.77, 'MET': 4.43, 'PHE': 5.89, 'PRO': 2.72,
    'SER': 1.60, 'THR': 2.60, 'TRP': 8.08, 'TYR': 6.47, 'VAL': 3.00,
}

# BHAR880101 - Flexibility index (Bhaskaran-Ponnuswamy, 1988)
FLEXIBILITY = {
    'ALA': 0.357, 'ARG': 0.529, 'ASN': 0.463, 'ASP': 0.511, 'CYS': 0.346,
    'GLN': 0.493, 'GLU': 0.497, 'GLY': 0.544, 'HIS': 0.323, 'ILE': 0.462,
    'LEU': 0.365, 'LYS': 0.466, 'MET': 0.295, 'PHE': 0.314, 'PRO': 0.509,
    'SER': 0.507, 'THR': 0.444, 'TRP': 0.305, 'TYR': 0.420, 'VAL': 0.386,
}

# GRAR740102 - Polarity (Grantham, 1974)
POLARITY = {
    'ALA': 8.1,  'ARG': 10.5, 'ASN': 11.6, 'ASP': 13.0, 'CYS': 5.5,
    'GLN': 10.5, 'GLU': 12.3, 'GLY': 9.0,  'HIS': 10.4, 'ILE': 5.2,
    'LEU': 4.9,  'LYS': 11.3, 'MET': 5.7,  'PHE': 5.2,  'PRO': 8.0,
    'SER': 9.2,  'THR': 8.6,  'TRP': 5.4,  'TYR': 6.2,  'VAL': 5.9,
}

# ZIMJ680102 - Bulkiness (Zimmerman et al., 1968)
BULKINESS = {
    'ALA': 11.50, 'ARG': 14.28, 'ASN': 12.82, 'ASP': 11.68, 'CYS': 13.46,
    'GLN': 14.45, 'GLU': 13.57, 'GLY': 3.40,  'HIS': 13.69, 'ILE': 21.40,
    'LEU': 21.40, 'LYS': 15.71, 'MET': 16.25, 'PHE': 19.80, 'PRO': 17.43,
    'SER': 9.47,  'THR': 15.77, 'TRP': 21.67, 'TYR': 18.03, 'VAL': 21.57,
}

# CHOP780201 - Chou-Fasman alpha-helix propensity
HELIX_PROPENSITY = {
    'ALA': 1.42, 'ARG': 0.98, 'ASN': 0.67, 'ASP': 1.01, 'CYS': 0.70,
    'GLN': 1.11, 'GLU': 1.51, 'GLY': 0.57, 'HIS': 1.00, 'ILE': 1.08,
    'LEU': 1.21, 'LYS': 1.16, 'MET': 1.45, 'PHE': 1.13, 'PRO': 0.57,
    'SER': 0.77, 'THR': 0.83, 'TRP': 1.08, 'TYR': 0.69, 'VAL': 1.06,
}

# CHOP780202 - Chou-Fasman beta-sheet propensity
SHEET_PROPENSITY = {
    'ALA': 0.83, 'ARG': 0.93, 'ASN': 0.89, 'ASP': 0.54, 'CYS': 1.19,
    'GLN': 1.10, 'GLU': 0.37, 'GLY': 0.75, 'HIS': 0.87, 'ILE': 1.60,
    'LEU': 1.30, 'LYS': 0.74, 'MET': 1.05, 'PHE': 1.38, 'PRO': 0.55,
    'SER': 0.75, 'THR': 1.19, 'TRP': 1.37, 'TYR': 1.47, 'VAL': 1.70,
}

# All property tables in order
PROPERTY_TABLES = [
    ('vdw_volume', VDW_VOLUME),
    ('flexibility', FLEXIBILITY),
    ('polarity', POLARITY),
    ('bulkiness', BULKINESS),
    ('helix_propensity', HELIX_PROPENSITY),
    ('sheet_propensity', SHEET_PROPENSITY),
]

FEATURE_NAMES = [name for name, _ in PROPERTY_TABLES]
N_FEATURES = len(FEATURE_NAMES)

# Precompute min-max normalization parameters
NORM_PARAMS = {}
for name, table in PROPERTY_TABLES:
    vals = list(table.values())
    vmin, vmax = min(vals), max(vals)
    NORM_PARAMS[name] = (vmin, vmax - vmin)  # (min, range)

# 3-letter to standard mapping (same as other scripts)
NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER', 'TPO': 'THR',
    'PTR': 'TYR', 'CSE': 'CYS',
}


def process_protein(pdb_id):
    """Extract AAindex features for one protein.

    Returns: (pdb_id, status, message)
    """
    out_path = os.path.join(FEATURES_DIR, f"{pdb_id}_aaindex.npz")
    if os.path.exists(out_path):
        return pdb_id, 'skip', None

    label_path = os.path.join(PROCESSED_DIR, f"{pdb_id}_labels.csv")
    if not os.path.exists(label_path):
        return pdb_id, 'fail', 'no labels'

    labels_df = pd.read_csv(label_path, dtype={'chain': str})
    n_res = len(labels_df)
    features = np.zeros((n_res, N_FEATURES), dtype=np.float32)

    n_unknown = 0
    for i, row in labels_df.iterrows():
        resname = row['resname']
        if resname in NONSTANDARD_MAP:
            resname = NONSTANDARD_MAP[resname]

        for j, (name, table) in enumerate(PROPERTY_TABLES):
            raw = table.get(resname)
            if raw is not None:
                vmin, vrange = NORM_PARAMS[name]
                features[i, j] = (raw - vmin) / vrange if vrange > 0 else 0.0
            else:
                n_unknown += 1
                features[i, j] = 0.5  # neutral default for unknown residues

    np.savez_compressed(out_path, features=features, feature_names=FEATURE_NAMES)

    msg = f"{n_res} residues"
    if n_unknown > 0:
        msg += f", {n_unknown} unknown lookups"
    return pdb_id, 'ok', msg


def main():
    print("=" * 60)
    print("  AAindex Physicochemical Properties Extraction")
    print("=" * 60)
    print(f"  Features ({N_FEATURES}):")
    for name in FEATURE_NAMES:
        print(f"    - {name}")
    print(f"  Output: {FEATURES_DIR}")

    # Get protein list
    splits_path = os.path.join(PROCESSED_DIR, "train_val_test_splits.csv")
    if not os.path.exists(splits_path):
        print("ERROR: splits file not found")
        sys.exit(1)

    splits = pd.read_csv(splits_path)
    pdb_ids = splits['pdb_id'].tolist()
    print(f"\n  Proteins: {len(pdb_ids)}")

    # Check existing
    n_existing = sum(1 for p in pdb_ids if os.path.exists(os.path.join(FEATURES_DIR, f"{p}_aaindex.npz")))
    if n_existing > 0:
        print(f"  Already done: {n_existing} (will skip)")

    # Process with multiprocessing
    start_time = time.time()
    n_ok = 0
    n_skip = 0
    n_fail = 0

    n_workers = min(cpu_count(), 8)
    print(f"  Workers: {n_workers}")
    print()

    with Pool(n_workers) as pool:
        for pdb_id, status, msg in pool.imap_unordered(process_protein, pdb_ids, chunksize=50):
            if status == 'ok':
                n_ok += 1
            elif status == 'skip':
                n_skip += 1
            elif status == 'fail':
                n_fail += 1
                print(f"  FAIL: {pdb_id}: {msg}")

            total_done = n_ok + n_skip + n_fail
            if total_done % 500 == 0:
                elapsed = time.time() - start_time
                print(f"  [{total_done}/{len(pdb_ids)}] ok={n_ok} skip={n_skip} fail={n_fail} ({elapsed:.0f}s)")

    elapsed = time.time() - start_time
    print(f"\n{'-' * 40}")
    print(f"  Done ({elapsed:.1f}s)")
    print(f"  New:     {n_ok}")
    print(f"  Skipped: {n_skip}")
    print(f"  Failed:  {n_fail}")

    # Verify one file
    sample_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('_aaindex.npz')]
    if sample_files:
        sample = np.load(os.path.join(FEATURES_DIR, sample_files[0]))
        print(f"\n  Verification ({sample_files[0]}):")
        print(f"    Shape: {sample['features'].shape}")
        print(f"    Range: [{sample['features'].min():.3f}, {sample['features'].max():.3f}]")
        print(f"    Names: {list(sample['feature_names'])}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
