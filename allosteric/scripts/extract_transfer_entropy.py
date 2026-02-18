"""
Extract Transfer Entropy (TE) features per residue from GNM Kirchhoff matrix.

Based on the AllosES algorithm for allosteric site detection:
  - Hu et al., JCIM 2024, DOI: 10.1021/acs.jcim.4c00544
  - Kamberaj, Proteins 2017, 85:1056-1064

Features (3-dim):
  1. nte_score    — Normalized net transfer entropy (sender vs receiver, [-1, 1])
  2. te_out_sum   — Total allosteric signal SENT by residue i (sum of TE(i->j))
  3. te_in_sum    — Total allosteric signal RECEIVED by residue i (sum of TE(j->i))

Algorithm:
  1. Build GNM Kirchhoff matrix (cutoff = 7A, per AllosES standard)
  2. Eigendecomposition -> eigenvalues lambda_k, eigenvectors u_k
  3. Covariance C(0) = sum_k (1/lambda_k) * u_k @ u_k^T
  4. Time-delayed covariance C(tau) = sum_k exp(-lambda_k * tau) / lambda_k * u_k @ u_k^T
  5. Transfer Entropy TE(j->i) via 4-term log formula (AllosES)
  6. Net TE: NTE(i,j) = TE(j->i) - TE(i->j)
  7. Per-residue features: row/column sums of TE/NTE matrices

Usage:
    python extract_transfer_entropy.py              # Training proteins
    python extract_transfer_entropy.py --casbench   # CASBench proteins
"""

import os
import gc
import time
import argparse
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

import prody
prody.confProDy(verbosity='none')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = r"E:\newyear\research_plan\allosteric"
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
TE_DIR = os.path.join(FEATURES_DIR, "transfer_entropy")
os.makedirs(TE_DIR, exist_ok=True)

# CASBench paths
CASBENCH_DIR = os.path.join(DATA_DIR, "casbench")
CASBENCH_LABELS_DIR = os.path.join(CASBENCH_DIR, "labels")
CASBENCH_FEATURES_DIR = os.path.join(CASBENCH_DIR, "features")

# ── Constants ─────────────────────────────────────────────────────────────────
# Must match extract_features.py / extract_labels.py
AA_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
           'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
           'TYR', 'VAL']
NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER', 'TPO': 'THR',
    'PTR': 'TYR', 'CSE': 'CYS',
}

TE_FEATURE_NAMES = ['nte_score', 'te_out_sum', 'te_in_sum']
TE_DIM = 3

GNM_CUTOFF = 7.0       # AllosES standard for TE (NOT 10A used for GNM fluctuation)
TAU = 5                 # Time lag parameter (AllosES / Kamberaj standard)
MAX_FULL_SIZE = 5000    # Full eigendecomposition below this
N_MODES_TRUNCATED = 200 # For proteins > MAX_FULL_SIZE


def get_residues_and_coords(pdb_path):
    """Parse PDB, return residues info and CA coordinates.
    Identical to extract_nma_graph.py to ensure consistent residue ordering."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = structure[0]

    residues = []
    ca_coords = []
    for chain in model:
        for res in chain:
            if res.id[0] != ' ':
                continue
            resname = res.get_resname()
            if resname in NONSTANDARD_MAP:
                resname = NONSTANDARD_MAP[resname]
            if resname not in AA_LIST:
                continue

            residues.append({
                'chain': chain.id,
                'resnum': res.id[1],
                'resname': resname
            })

            if 'CA' in res:
                ca_coords.append(res['CA'].get_vector().get_array())
            else:
                atoms = list(res.get_atoms())
                if atoms:
                    ca_coords.append(np.mean([a.get_vector().get_array() for a in atoms], axis=0))
                else:
                    ca_coords.append(np.array([0.0, 0.0, 0.0]))

    return residues, np.array(ca_coords)


def compute_te_features(ca_coords):
    """Compute Transfer Entropy features from GNM Kirchhoff matrix.

    Returns np.ndarray of shape (N, 3) with columns:
      [nte_score, te_out_sum, te_in_sum]

    Algorithm follows AllosES (Hu et al., JCIM 2024) and Kamberaj (Proteins 2017).
    """
    n = len(ca_coords)
    features = np.zeros((n, TE_DIM), dtype=np.float32)

    if n < 5:
        return features

    # Step 1: Build GNM and compute eigenmodes
    gnm = prody.GNM('protein')
    gnm.buildKirchhoff(ca_coords, cutoff=GNM_CUTOFF)

    if n > MAX_FULL_SIZE:
        n_modes = min(N_MODES_TRUNCATED, n - 1)
    else:
        n_modes = n - 1

    gnm.calcModes(n_modes=n_modes)

    eigenvalues = gnm.getEigvals()    # (K,)
    eigenvectors = gnm.getArray()     # (N, K)

    # Filter near-zero eigenvalues (disconnected components)
    valid = eigenvalues > 1e-8
    if valid.sum() < 3:
        return features
    eigenvalues = eigenvalues[valid]
    eigenvectors = eigenvectors[:, valid]

    # Step 2: Compute covariance matrices (vectorized, float32 for memory)
    eigenvectors = eigenvectors.astype(np.float32)
    eigenvalues = eigenvalues.astype(np.float32)

    inv_lambda = 1.0 / eigenvalues                                    # (K,)
    exp_decay = np.exp(-eigenvalues * TAU) * inv_lambda                # (K,)

    # C0 = U @ diag(1/lambda) @ U^T  — static covariance (pseudoinverse of K)
    C0 = (eigenvectors * inv_lambda) @ eigenvectors.T                  # (N, N)

    # Ctau = U @ diag(exp(-lambda*tau)/lambda) @ U^T  — time-delayed covariance
    Ctau = (eigenvectors * exp_decay) @ eigenvectors.T                 # (N, N)

    # Step 3: Compute TE matrix using AllosES 4-term formula
    # TE[i,j] = TE(j -> i) = 0.5 * [log(term_a) - log(term_b) - log(term_f) + log(term_g)]
    eps = 1e-30

    C0_diag = np.diag(C0)             # (N,)
    Ctau_diag = np.diag(Ctau)         # (N,)

    C0_ii = C0_diag[:, None]          # (N, 1) — variance of i
    C0_jj = C0_diag[None, :]          # (1, N) — variance of j
    Ctau_jj = Ctau_diag[None, :]      # (1, N) — time-delayed variance of j

    # term_a = C0_jj^2 - Ctau_jj^2   — "free dynamics" of j
    term_a = C0_jj ** 2 - Ctau_jj ** 2

    # term_b = full conditional term (expanded from AllosES formula)
    # = C0_ii * C0_jj^2 + 2*C0_ij*Ctau_jj*Ctau_ij - (Ctau_ij^2 + C0_ij^2)*C0_jj - Ctau_jj^2 * C0_ii
    term_b = (C0_ii * C0_jj ** 2
              + 2 * C0 * Ctau_jj * Ctau
              - (Ctau ** 2 + C0 ** 2) * C0_jj
              - Ctau_jj ** 2 * C0_ii)

    # term_f = C0_jj
    term_f = C0_jj

    # term_g = C0_ii * C0_jj - C0_ij^2  — determinant of 2x2 covariance
    term_g = C0_ii * C0_jj - C0 ** 2

    # Free intermediate NxN matrices immediately
    del C0, Ctau
    gc.collect()

    # TE = 0.5 * (log(a) - log(b) - log(f) + log(g))
    TE = 0.5 * (np.log(np.clip(np.real(term_a), eps, None))
                - np.log(np.clip(np.real(term_b), eps, None))
                - np.log(np.clip(np.real(term_f), eps, None))
                + np.log(np.clip(np.real(term_g), eps, None)))

    del term_a, term_b, term_f, term_g
    gc.collect()

    # Post-processing (matches AllosES)
    TE = np.real(TE).astype(np.float32)
    TE[TE < 0] = 0.0
    np.fill_diagonal(TE, 0.0)

    # Step 4: Per-residue features
    # NTE[i,j] = TE[i,j] - TE[j,i] = TE(j->i) - TE(i->j)
    NTE = TE - TE.T

    # nte_score: normalized net TE (positive = net receiver, negative = net sender)
    nte_sum = NTE.sum(axis=1)  # sum over j for each i
    max_abs = np.max(np.abs(nte_sum))
    if max_abs > 0:
        features[:, 0] = nte_sum / max_abs  # normalized to [-1, 1]
    # else: stays as zeros

    # te_out_sum: total TE sent by residue i = sum_j TE(i->j) = column sum of TE
    features[:, 1] = TE.sum(axis=0)  # axis=0 col sum: sum_j TE[j,i] = sum_j TE(i->j)

    # te_in_sum: total TE received by residue i = sum_j TE(j->i) = row sum of TE
    features[:, 2] = TE.sum(axis=1)  # axis=1 row sum: sum_j TE[i,j] = sum_j TE(j->i)

    # Final safeguard
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def process_single_protein(args):
    """Process a single protein — designed for multiprocessing Pool."""
    pdb_id, pdb_path, label_path, output_path = args

    if os.path.exists(output_path):
        return pdb_id, 'skipped', None

    if not os.path.exists(pdb_path) or not os.path.exists(label_path):
        return pdb_id, 'failed', 'missing files'

    try:
        residues, ca_coords = get_residues_and_coords(pdb_path)
        if len(residues) == 0:
            return pdb_id, 'failed', 'no residues'

        # Compute TE features
        te_feat = compute_te_features(ca_coords)  # (N, 3)

        # Align with labels
        labels_df = pd.read_csv(label_path)

        feat_lookup = {}
        for i, info in enumerate(residues):
            key = (info['chain'], info['resnum'])
            feat_lookup[key] = i

        aligned_features = []
        aligned_labels = []
        for _, lrow in labels_df.iterrows():
            key = (lrow['chain'], lrow['resnum'])
            if key in feat_lookup:
                aligned_features.append(te_feat[feat_lookup[key]])
            else:
                aligned_features.append(np.zeros(TE_DIM, dtype=np.float32))
            aligned_labels.append(lrow['is_allosteric'])

        if len(aligned_features) == 0:
            return pdb_id, 'failed', 'no alignment'

        aligned_features = np.array(aligned_features, dtype=np.float32)
        aligned_labels = np.array(aligned_labels)

        np.savez_compressed(output_path,
                            features=aligned_features,
                            labels=aligned_labels,
                            pdb_id=pdb_id,
                            feature_names=TE_FEATURE_NAMES)

        n_res = len(ca_coords)
        return pdb_id, 'success', f'{len(aligned_labels)} res (N={n_res})'

    except Exception as e:
        return pdb_id, 'failed', str(e)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Transfer Entropy features")
    parser.add_argument('--casbench', action='store_true',
                        help='Process CASBench proteins instead of training proteins')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Transfer Entropy Feature Extraction")
    print("  Algorithm: AllosES (Hu et al., JCIM 2024)")
    print("=" * 60)
    print(f"  Features ({TE_DIM}):")
    for name in TE_FEATURE_NAMES:
        print(f"    - {name}")
    print(f"  GNM cutoff: {GNM_CUTOFF} A")
    print(f"  Time lag (tau): {TAU}")
    print(f"  CPU cores available: {cpu_count()}")

    if args.casbench:
        # CASBench mode
        output_dir = os.path.join(CASBENCH_FEATURES_DIR)
        os.makedirs(output_dir, exist_ok=True)
        pdb_csv = os.path.join(CASBENCH_DIR, "casbench_independent_pdbs.csv")
        if not os.path.exists(pdb_csv):
            print(f"ERROR: {pdb_csv} not found. Run evaluate_casbench.py --phase discover first.")
            return

        pdb_list = pd.read_csv(pdb_csv)
        pdb_list = pdb_list[pdb_list['is_overlap'] == False]
        print(f"\n  Mode: CASBench ({len(pdb_list)} independent proteins)")
        print(f"  Output: {output_dir}")

        tasks = []
        for _, row in pdb_list.iterrows():
            pdb_id = row['pdb_id']
            pdb_path = row['pdb_path']
            label_path = os.path.join(CASBENCH_LABELS_DIR, f"{pdb_id}_labels.csv")
            output_path = os.path.join(output_dir, f"{pdb_id}_te.npz")
            tasks.append((pdb_id, pdb_path, label_path, output_path))
    else:
        # Training mode
        output_dir = TE_DIR
        print(f"\n  Mode: Training proteins")
        print(f"  Output: {output_dir}")

        summary_path = os.path.join(PROCESSED_DIR, "dataset_summary.csv")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            pdb_ids = summary['pdb_id'].tolist()
        else:
            splits_path = os.path.join(PROCESSED_DIR, "train_val_test_splits.csv")
            splits = pd.read_csv(splits_path)
            pdb_ids = splits['pdb_id'].tolist()

        pdb_dir = os.path.join(DATA_DIR, "pdb_files")
        tasks = []
        for pdb_id in pdb_ids:
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            label_path = os.path.join(PROCESSED_DIR, f"{pdb_id}_labels.csv")
            output_path = os.path.join(output_dir, f"{pdb_id}_te.npz")
            tasks.append((pdb_id, pdb_path, label_path, output_path))

    print(f"  Total tasks: {len(tasks)}")

    # Check existing
    n_existing = sum(1 for _, _, _, op in tasks if os.path.exists(op))
    if n_existing > 0:
        print(f"  Already done: {n_existing} (will skip)")

    n_workers = max(1, cpu_count() - 2)  # Leave 2 cores free (TE is CPU-heavy)
    print(f"  Workers: {n_workers}\n")

    start_time = time.time()
    processed = 0
    skipped = 0
    failed = 0
    errors = []

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_single_protein, tasks, chunksize=4):
            pdb_id, status, msg = result
            if status == 'success':
                processed += 1
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    remaining = (len(tasks) - processed - skipped - failed) / max(rate, 0.01)
                    print(f"  Processed {processed} proteins ({elapsed:.0f}s, ~{remaining:.0f}s remaining)")
            elif status == 'skipped':
                skipped += 1
            else:
                failed += 1
                if len(errors) < 10:
                    errors.append(f"{pdb_id}: {msg}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Transfer Entropy Extraction Complete ({elapsed:.0f}s)")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output dir: {output_dir}")
    if errors:
        print(f"  First errors:")
        for e in errors[:5]:
            print(f"    {e}")

    # Verify one file
    sample_files = [f for f in os.listdir(output_dir) if f.endswith('_te.npz')]
    if sample_files:
        sample = np.load(os.path.join(output_dir, sample_files[0]))
        print(f"\n  Verification ({sample_files[0]}):")
        print(f"    Shape: {sample['features'].shape}")
        print(f"    Range: [{sample['features'].min():.4f}, {sample['features'].max():.4f}]")
        print(f"    Names: {list(sample['feature_names'])}")
        feat = sample['features']
        for i, name in enumerate(TE_FEATURE_NAMES):
            col = feat[:, i]
            print(f"    {name}: mean={col.mean():.4f}, std={col.std():.4f}, "
                  f"min={col.min():.4f}, max={col.max():.4f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
