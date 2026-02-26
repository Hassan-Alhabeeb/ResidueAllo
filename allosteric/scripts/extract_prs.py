"""
Extract PRS (Perturbation Response Scanning) features per residue.

Based on:
  - Atilgan & Atilgan, PLoS Comp Bio 2009, DOI: 10.1371/journal.pcbi.1000544
  - Also used by AlloFusion (Huang et al., JCIM 2025)

Features (3-dim):
  1. prs_effectiveness — How much perturbing THIS residue affects ALL others
  2. prs_sensitivity   — How much THIS residue is affected by perturbations elsewhere
  3. prs_eff_sens_ratio — effectiveness / sensitivity (high = allosteric source)

Algorithm:
  1. Build ANM Hessian (cutoff=15A, standard for PRS)
  2. Compute truncated modes (20 slowest for speed)
  3. calcPerturbResponse with turbo=True (analytical, no random forces)
  4. Returns: PRS matrix (NxN), effectiveness (N,), sensitivity (N,)

Usage:
    python extract_prs.py              # Training proteins
    python extract_prs.py --casbench   # CASBench proteins
"""

import os
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/../ = allosteric/
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
PRS_DIR = os.path.join(FEATURES_DIR, "prs")
os.makedirs(PRS_DIR, exist_ok=True)

# CASBench paths
CASBENCH_DIR = os.path.join(DATA_DIR, "casbench")
CASBENCH_LABELS_DIR = os.path.join(CASBENCH_DIR, "labels")
CASBENCH_FEATURES_DIR = os.path.join(CASBENCH_DIR, "features")

# ── Constants ─────────────────────────────────────────────────────────────────
AA_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
           'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
           'TYR', 'VAL']
NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER', 'TPO': 'THR',
    'PTR': 'TYR', 'CSE': 'CYS',
}

PRS_FEATURE_NAMES = ['prs_effectiveness', 'prs_sensitivity', 'prs_eff_sens_ratio']
PRS_DIM = 3

ANM_CUTOFF = 15.0     # Standard ANM cutoff for PRS
N_MODES = 'all (3N-6)'  # All non-trivial modes (PRS needs full covariance)


def get_residues_and_coords(pdb_path):
    """Parse PDB, return residues info and CA coordinates.
    Identical to extract_nma_graph.py / extract_transfer_entropy.py."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = structure[0]

    residues = []
    ca_coords = []
    for chain in model:
        for res in chain:
            resname = res.get_resname()
            if res.id[0] != ' ' and resname not in NONSTANDARD_MAP:
                continue
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
                    ca_coords.append(np.array([99999.0, 99999.0, 99999.0]))

    return residues, np.array(ca_coords)


def compute_prs_features(ca_coords):
    """Compute PRS effectiveness and sensitivity from ANM.

    Returns np.ndarray of shape (N, 3) with columns:
      [prs_effectiveness, prs_sensitivity, prs_eff_sens_ratio]
    """
    n = len(ca_coords)
    features = np.zeros((n, PRS_DIM), dtype=np.float32)

    if n < 5:
        return features

    try:
        # Build ANM and compute ALL non-trivial modes.
        # PRS uses Linear Response Theory which requires the full covariance
        # matrix — truncating to 20 modes discards ~95% of the mechanical physics.
        anm = prody.ANM('protein')
        anm.buildHessian(ca_coords, cutoff=ANM_CUTOFF)
        n_modes = 3 * n - 6  # All non-trivial ANM modes
        if n_modes < 1:
            return features
        anm.calcModes(n_modes=n_modes)

        # PRS: turbo=True uses analytical covariance (fast, no random sampling)
        prs_mat, effectiveness, sensitivity = prody.calcPerturbResponse(anm)

        # effectiveness: shape (N,) — average response of ALL residues when i is perturbed
        # sensitivity: shape (N,) — average response of i when ALL residues are perturbed
        features[:, 0] = effectiveness.astype(np.float32)
        features[:, 1] = sensitivity.astype(np.float32)

        # Ratio: high effectiveness / low sensitivity = allosteric source
        eps = 1e-8
        features[:, 2] = (effectiveness / (sensitivity + eps)).astype(np.float32)

        # Final safeguard
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    except Exception:
        pass  # Features stay as zeros

    return features


def _set_blas_threads(n_threads):
    """Set BLAS thread count for this worker process."""
    import os
    for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS']:
        os.environ[var] = str(n_threads)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n_threads, user_api='blas')
    except ImportError:
        print(f"  WARNING: threadpoolctl not installed! BLAS thread limiting will not work.")
        print(f"  Install with: pip install threadpoolctl")


def _worker_init_threads(n_threads):
    """Pool initializer that sets BLAS threads."""
    _set_blas_threads(n_threads)


def _get_protein_size(pdb_path):
    """Quick count of CA atoms to estimate protein size without full parsing."""
    count = 0
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    count += 1
    except Exception:
        pass
    return count


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

        prs_feat = compute_prs_features(ca_coords)

        # Align with labels
        labels_df = pd.read_csv(label_path, dtype={'chain': str})

        feat_lookup = {}
        for i, info in enumerate(residues):
            key = (info['chain'], info['resnum'])
            feat_lookup[key] = i

        aligned_features = []
        aligned_labels = []
        for _, lrow in labels_df.iterrows():
            key = (lrow['chain'], lrow['resnum'])
            if key in feat_lookup:
                aligned_features.append(prs_feat[feat_lookup[key]])
            else:
                aligned_features.append(np.zeros(PRS_DIM, dtype=np.float32))
            aligned_labels.append(lrow['is_allosteric'])

        if len(aligned_features) == 0:
            return pdb_id, 'failed', 'no alignment'

        aligned_features = np.array(aligned_features, dtype=np.float32)
        aligned_labels = np.array(aligned_labels)

        np.savez_compressed(output_path,
                            features=aligned_features,
                            labels=aligned_labels,
                            pdb_id=pdb_id,
                            feature_names=PRS_FEATURE_NAMES)

        n_res = len(ca_coords)
        return pdb_id, 'success', f'{len(aligned_labels)} res (N={n_res})'

    except Exception as e:
        return pdb_id, 'failed', str(e)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract PRS features")
    parser.add_argument('--casbench', action='store_true',
                        help='Process CASBench proteins instead of training proteins')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  PRS (Perturbation Response Scanning) Feature Extraction")
    print("  Atilgan & Atilgan, PLoS Comp Bio 2009")
    print("=" * 60)
    print(f"  Features ({PRS_DIM}):")
    for name in PRS_FEATURE_NAMES:
        print(f"    - {name}")
    print(f"  ANM cutoff: {ANM_CUTOFF} A")
    print(f"  Modes: {N_MODES}")
    print(f"  CPU cores available: {cpu_count()}")

    if args.casbench:
        output_dir = CASBENCH_FEATURES_DIR
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
            output_path = os.path.join(output_dir, f"{pdb_id}_prs.npz")
            tasks.append((pdb_id, pdb_path, label_path, output_path))
    else:
        output_dir = PRS_DIR
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
            output_path = os.path.join(output_dir, f"{pdb_id}_prs.npz")
            tasks.append((pdb_id, pdb_path, label_path, output_path))

    print(f"  Total tasks: {len(tasks)}")

    n_existing = sum(1 for _, _, _, op in tasks if os.path.exists(op))
    if n_existing > 0:
        print(f"  Already done: {n_existing} (will skip)")

    # Split tasks into normal (<= 2000 residues) and large (> 2000)
    LARGE_THRESHOLD = 2000
    phys_cores = min(max(1, cpu_count() - 2), 48)

    # Quick-scan PDB sizes to split tasks
    normal_tasks = []
    large_tasks = []
    for task in tasks:
        pdb_id, pdb_path, label_path, output_path = task
        if os.path.exists(output_path):
            normal_tasks.append(task)  # will be skipped anyway
            continue
        size = _get_protein_size(pdb_path)
        if size > LARGE_THRESHOLD:
            large_tasks.append((task, size))
        else:
            normal_tasks.append(task)

    print(f"  Normal proteins (<= {LARGE_THRESHOLD} res): {len(normal_tasks)}")
    print(f"  Large proteins (> {LARGE_THRESHOLD} res): {len(large_tasks)}")

    start_time = time.time()
    processed = 0
    skipped = 0
    failed = 0
    errors = []
    total = len(tasks)

    # Phase 1: Normal proteins — many workers, 1 BLAS thread each
    if normal_tasks:
        n_workers = phys_cores
        print(f"\n  Phase 1: Normal proteins — {n_workers} workers x 1 BLAS thread")
        with Pool(processes=n_workers, initializer=_worker_init_threads, initargs=(1,)) as pool:
            for result in pool.imap_unordered(process_single_protein, normal_tasks, chunksize=1):
                pdb_id, status, msg = result
                if status == 'success':
                    processed += 1
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    remaining = (total - processed - skipped - failed) / max(rate, 0.01)
                    print(f"  [{processed + skipped + failed:>5}/{total}] {pdb_id}: {msg} ({elapsed:.0f}s, ~{remaining:.0f}s left)")
                elif status == 'skipped':
                    skipped += 1
                else:
                    failed += 1
                    print(f"  [{processed + skipped + failed:>5}/{total}] {pdb_id}: FAILED - {msg}")
                    if len(errors) < 10:
                        errors.append(f"{pdb_id}: {msg}")

    # Phase 2: Large proteins — fewer workers, more BLAS threads each
    if large_tasks:
        # Sort largest first so they start immediately
        large_tasks.sort(key=lambda x: x[1], reverse=True)
        large_task_list = [t[0] for t in large_tasks]
        large_sizes = [t[1] for t in large_tasks]

        # Use 6 workers x 8 threads = 48 cores fully utilized
        n_large_workers = min(6, len(large_task_list))
        threads_per = max(1, phys_cores // n_large_workers)
        print(f"\n  Phase 2: Large proteins — {n_large_workers} workers x {threads_per} BLAS threads")
        print(f"  Sizes: {large_sizes[:10]}{'...' if len(large_sizes) > 10 else ''}")

        with Pool(processes=n_large_workers, initializer=_worker_init_threads, initargs=(threads_per,)) as pool:
            for result in pool.imap_unordered(process_single_protein, large_task_list, chunksize=1):
                pdb_id, status, msg = result
                if status == 'success':
                    processed += 1
                    elapsed = time.time() - start_time
                    print(f"  [{processed + skipped + failed:>5}/{total}] {pdb_id}: {msg} ({elapsed:.0f}s)")
                elif status == 'skipped':
                    skipped += 1
                else:
                    failed += 1
                    print(f"  [{processed + skipped + failed:>5}/{total}] {pdb_id}: FAILED - {msg}")
                    if len(errors) < 10:
                        errors.append(f"{pdb_id}: {msg}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"PRS Feature Extraction Complete ({elapsed:.0f}s)")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output dir: {output_dir}")
    if errors:
        print(f"  First errors:")
        for e in errors[:5]:
            print(f"    {e}")

    # Verify one file
    sample_files = [f for f in os.listdir(output_dir) if f.endswith('_prs.npz')]
    if sample_files:
        sample = np.load(os.path.join(output_dir, sample_files[0]))
        print(f"\n  Verification ({sample_files[0]}):")
        print(f"    Shape: {sample['features'].shape}")
        print(f"    Range: [{sample['features'].min():.4f}, {sample['features'].max():.4f}]")
        print(f"    Names: {list(sample['feature_names'])}")
        feat = sample['features']
        for i, name in enumerate(PRS_FEATURE_NAMES):
            col = feat[:, i]
            print(f"    {name}: mean={col.mean():.4f}, std={col.std():.4f}, "
                  f"min={col.min():.4f}, max={col.max():.4f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
