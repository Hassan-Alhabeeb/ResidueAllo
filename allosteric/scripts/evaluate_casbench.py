"""
CASBench Blind Evaluation Pipeline for Allosteric Site Prediction.

Evaluates our trained XGBoost model on CASBench — an independent benchmark
of 91 enzyme families with annotated allosteric/catalytic sites.

Phases (run sequentially or individually):
  1. discover  — Scan CASBench, filter overlapping proteins
  2. labels    — Parse allosteric/catalytic annotations into _labels.csv
  3. features  — Extract structural + NMA + graph + AAindex features
  4. fpocket   — Extract FPocket pocket geometry via WSL
  4b. te       — Extract Transfer Entropy features (AllosES algorithm)
  4c. prs      — Extract PRS features (Atilgan & Atilgan 2009)
  4d. mj       — Extract MJ contact energy features (Miyazawa & Jernigan 1996)
  4e. frustration — Extract local frustration features (Ferreiro 2007/2011)
  5. esm2      — Extract ESM-2 650M embeddings (GPU)
  6. predict   — Load saved model, assemble features, predict, evaluate

Usage:
    python evaluate_casbench.py                    # Run all phases
    python evaluate_casbench.py --phase discover    # Run single phase
    python evaluate_casbench.py --phase predict     # Just re-run evaluation
"""

import os
import sys
import re
import glob
import time
import argparse
import traceback
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')


# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/../ = allosteric/
CASBENCH_ROOT = os.path.join(BASE_DIR, "data", "raw", "casbench", "CASBench_Download")
SPLITS_CSV = os.path.join(BASE_DIR, "data", "processed", "train_val_test_splits.csv")

# CASBench output directories
CASBENCH_DIR = os.path.join(BASE_DIR, "data", "casbench")
CASBENCH_LABELS_DIR = os.path.join(CASBENCH_DIR, "labels")
CASBENCH_FEATURES_DIR = os.path.join(CASBENCH_DIR, "features")
CASBENCH_PREDICTIONS_DIR = os.path.join(CASBENCH_DIR, "predictions")
os.makedirs(CASBENCH_LABELS_DIR, exist_ok=True)
os.makedirs(CASBENCH_FEATURES_DIR, exist_ok=True)
os.makedirs(CASBENCH_PREDICTIONS_DIR, exist_ok=True)

# Model artifacts
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add scripts dir to path for imports
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

# ============================================================
# Logging — tee all print() output to both console and log file
# ============================================================

class TeeLogger:
    """Write to both stdout and a log file simultaneously."""

    def __init__(self, log_path, original_stdout):
        self.log_file = open(log_path, 'w', encoding='utf-8')
        self.stdout = original_stdout
        self.log_path = log_path

    def write(self, message):
        self.stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


LOG_DIR = os.path.join(RESULTS_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_active_logger = None


def start_logging(phase_name):
    """Start logging to a phase-specific file. Returns the log path."""
    global _active_logger
    stop_logging()  # close any prior logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"casbench_{phase_name}_{timestamp}.log")
    _active_logger = TeeLogger(log_path, sys.stdout)
    sys.stdout = _active_logger
    print(f"[LOG] Logging to: {log_path}")
    print(f"[LOG] Phase: {phase_name} | Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return log_path


def stop_logging():
    """Stop logging and restore original stdout."""
    global _active_logger
    if _active_logger is not None:
        print(f"[LOG] Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout = _active_logger.stdout
        _active_logger.close()
        _active_logger = None


# ============================================================
# Constants (must match training pipeline)
# ============================================================
AA_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
           'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
           'TYR', 'VAL']
NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER', 'TPO': 'THR',
    'PTR': 'TYR', 'CSE': 'CYS',
}

# 3-letter mixed-case to uppercase (CASBench format)
AA_MIXED_TO_UPPER = {
    'Ala': 'ALA', 'Arg': 'ARG', 'Asn': 'ASN', 'Asp': 'ASP', 'Cys': 'CYS',
    'Gln': 'GLN', 'Glu': 'GLU', 'Gly': 'GLY', 'His': 'HIS', 'Ile': 'ILE',
    'Leu': 'LEU', 'Lys': 'LYS', 'Met': 'MET', 'Phe': 'PHE', 'Pro': 'PRO',
    'Ser': 'SER', 'Thr': 'THR', 'Trp': 'TRP', 'Tyr': 'TYR', 'Val': 'VAL',
}

STRUCTURAL_DIM = 64
NMA_GRAPH_DIM = 11
FPOCKET_DIM = 8
AAINDEX_DIM = 6
TE_DIM = 3
PRS_DIM = 3
MJ_DIM = 2
FRUST_DIM = 7
ESM_650M_DIM = 1280
ESM_3B_DIM = 2560
ESM_JOINT_DIM = ESM_650M_DIM + ESM_3B_DIM  # 3840
ESM_PCA_DIM = 128


# ============================================================
# Phase 1: Discovery & Filtering
# ============================================================

def phase_discover():
    """Scan CASBench directories and filter overlapping proteins."""
    start_logging("discover")
    print("=" * 60)
    print("  Phase 1: Discovery & Filtering")
    print("=" * 60)

    # Load training PDB IDs (uppercase)
    splits = pd.read_csv(SPLITS_CSV)
    training_pdb_ids = set(splits['pdb_id'].str.upper().tolist())
    print(f"  Training set: {len(training_pdb_ids)} proteins")

    # Scan CASBench directories
    cas_entries = sorted([d for d in os.listdir(CASBENCH_ROOT)
                          if d.startswith('cas') and os.path.isdir(os.path.join(CASBENCH_ROOT, d))])
    print(f"  CASBench entries: {len(cas_entries)}")

    records = []
    n_overlap = 0
    n_independent = 0
    n_no_pdb = 0

    for cas_entry in cas_entries:
        cas_path = os.path.join(CASBENCH_ROOT, cas_entry)
        # Each subdirectory that isn't README/alignments is a PDB entry
        pdb_dirs = [d for d in os.listdir(cas_path)
                    if os.path.isdir(os.path.join(cas_path, d))
                    and d not in ('alignments',)]

        for pdb_dir in pdb_dirs:
            pdb_id_lower = pdb_dir
            pdb_id_upper = pdb_dir.upper()
            pdb_subdir = os.path.join(cas_path, pdb_dir)

            # Check for PDB file
            pdb_file = os.path.join(pdb_subdir, f"{pdb_id_lower}.pdb")
            if not os.path.exists(pdb_file):
                n_no_pdb += 1
                continue

            # Check for allosteric sites annotation
            allo_file = os.path.join(pdb_subdir, "ALLOSTERIC_SITES.txt")
            if not os.path.exists(allo_file):
                continue

            # Check overlap with training set
            is_overlap = pdb_id_upper in training_pdb_ids
            if is_overlap:
                n_overlap += 1
            else:
                n_independent += 1

            records.append({
                'cas_entry': cas_entry,
                'pdb_id': pdb_id_upper,
                'pdb_path': pdb_file,
                'pdb_dir': pdb_subdir,
                'is_overlap': is_overlap,
            })

    df = pd.DataFrame(records)

    print(f"\n  Results:")
    print(f"    Total PDB entries found:  {len(df)}")
    print(f"    Overlapping with training: {n_overlap}")
    print(f"    Truly independent:         {n_independent}")
    print(f"    Missing PDB file:          {n_no_pdb}")

    # Save full list (with overlap flag) for reference
    full_csv = os.path.join(CASBENCH_DIR, "casbench_all_pdbs.csv")
    df.to_csv(full_csv, index=False)

    # Save independent-only list
    independent_df = df[~df['is_overlap']].copy()
    independent_csv = os.path.join(CASBENCH_DIR, "casbench_independent_pdbs.csv")
    independent_df.to_csv(independent_csv, index=False)
    print(f"\n  Saved: {independent_csv} ({len(independent_df)} proteins)")

    # Per-family summary
    print(f"\n  Per-family overlap:")
    for cas_entry in cas_entries[:5]:
        sub = df[df['cas_entry'] == cas_entry]
        n_total = len(sub)
        n_ovlp = sub['is_overlap'].sum()
        print(f"    {cas_entry}: {n_total} PDBs, {n_ovlp} overlap, {n_total - n_ovlp} independent")
    if len(cas_entries) > 5:
        print(f"    ... ({len(cas_entries) - 5} more families)")

    # Families fully excluded (all PDBs overlap)
    fully_excluded = 0
    for cas_entry in cas_entries:
        sub = df[df['cas_entry'] == cas_entry]
        if len(sub) > 0 and sub['is_overlap'].all():
            fully_excluded += 1
    print(f"\n  Families fully excluded (100% overlap): {fully_excluded}/{len(cas_entries)}")

    print("=" * 60)
    stop_logging()
    return independent_df


def load_independent_pdbs():
    """Load the saved independent PDB list."""
    csv_path = os.path.join(CASBENCH_DIR, "casbench_independent_pdbs.csv")
    if not os.path.exists(csv_path):
        print("ERROR: Run --phase discover first!")
        sys.exit(1)
    return pd.read_csv(csv_path)


# ============================================================
# Phase 2: Label Extraction
# ============================================================

def parse_sites_file(filepath):
    """Parse ALLOSTERIC_SITES.txt or CATALYTIC_SITES.txt.

    Format: SITE_LABEL ResName Chain Resnum
    Example: ALLOSTERIC_SITE_0 Thr A 106

    Returns: set of (chain, resnum) tuples
    """
    sites = set()
    if not os.path.exists(filepath):
        return sites

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            # parts: [SITE_LABEL, ResName, Chain, Resnum]
            chain = parts[2]
            try:
                # Strip insertion codes (e.g., "106A" -> 106)
                match = re.match(r'-?\d+', parts[3])
                if not match:
                    continue
                resnum = int(match.group())
            except (ValueError, AttributeError):
                continue
            sites.add((chain, resnum))

    return sites


def extract_labels_for_protein(pdb_path, pdb_dir, pdb_id):
    """Parse PDB structure and CASBench annotations to create _labels.csv.

    Returns: (n_residues, n_allosteric, n_catalytic) or None on failure
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception as e:
        return None, str(e)

    model = structure[0]

    # Parse allosteric and catalytic site annotations
    allo_sites = parse_sites_file(os.path.join(pdb_dir, "ALLOSTERIC_SITES.txt"))
    cat_sites = parse_sites_file(os.path.join(pdb_dir, "CATALYTIC_SITES.txt"))

    # Enumerate residues (same logic as extract_features.py)
    rows = []
    for chain in model:
        for res in chain:
            resname = res.get_resname()
            if res.id[0] != ' ' and resname not in NONSTANDARD_MAP:
                continue
            if resname in NONSTANDARD_MAP:
                resname = NONSTANDARD_MAP[resname]
            if resname not in AA_LIST:
                continue

            chain_id = chain.id
            resnum = res.id[1]
            is_allo = 1 if (chain_id, resnum) in allo_sites else 0
            is_cat = 1 if (chain_id, resnum) in cat_sites else 0

            rows.append({
                'chain': chain_id,
                'resnum': resnum,
                'resname': resname,
                'is_allosteric': is_allo,
                'is_active_site': is_cat,
            })

    if not rows:
        return None, "no standard residues"

    df = pd.DataFrame(rows)
    return df, None


def phase_labels():
    """Parse CASBench annotations into _labels.csv format."""
    start_logging("labels")
    print("=" * 60)
    print("  Phase 2: Label Extraction")
    print("=" * 60)

    pdb_list = load_independent_pdbs()
    print(f"  Proteins to process: {len(pdb_list)}")

    n_ok = 0
    n_skip = 0
    n_fail = 0
    total_residues = 0
    total_allosteric = 0
    total_catalytic = 0

    for _, row in pdb_list.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = row['pdb_path']
        pdb_dir = row['pdb_dir']

        out_path = os.path.join(CASBENCH_LABELS_DIR, f"{pdb_id}_labels.csv")
        if os.path.exists(out_path):
            n_skip += 1
            continue

        labels_df, error = extract_labels_for_protein(pdb_path, pdb_dir, pdb_id)
        if labels_df is None:
            n_fail += 1
            if n_fail <= 10:
                print(f"  FAIL: {pdb_id}: {error}")
            continue

        labels_df.to_csv(out_path, index=False)
        n_ok += 1
        n_res = len(labels_df)
        n_allo = labels_df['is_allosteric'].sum()
        n_cat = labels_df['is_active_site'].sum()
        total_residues += n_res
        total_allosteric += n_allo
        total_catalytic += n_cat

        if (n_ok + n_skip) % 500 == 0:
            print(f"  [{n_ok + n_skip + n_fail}/{len(pdb_list)}] ok={n_ok} skip={n_skip} fail={n_fail}")

    print(f"\n  Results:")
    print(f"    New:     {n_ok}")
    print(f"    Skipped: {n_skip}")
    print(f"    Failed:  {n_fail}")
    print(f"    Total residues (new): {total_residues:,}")
    print(f"    Allosteric (new):     {total_allosteric:,} ({100*total_allosteric/max(total_residues,1):.1f}%)")
    print(f"    Catalytic (new):      {total_catalytic:,}")
    print("=" * 60)
    stop_logging()


# ============================================================
# Phase 3: Structural + NMA + Graph + FPocket + AAindex Features
# ============================================================

def _process_single_features(args):
    """Worker function for parallel feature extraction. Must be top-level for pickling."""
    pdb_id, pdb_path, label_path, out_path = args

    if os.path.exists(out_path):
        return pdb_id, 'skip', None

    if not os.path.exists(label_path):
        return pdb_id, 'fail', 'no labels'

    try:
        # Imports inside worker (each subprocess needs its own)
        from extract_features import extract_all_features
        from extract_nma_graph import extract_nma_features, extract_graph_features, get_residues_and_coords
        from extract_aaindex import PROPERTY_TABLES, NORM_PARAMS, NONSTANDARD_MAP as AAINDEX_NONSTANDARD

        labels_df = pd.read_csv(label_path, dtype={'chain': str})

        # --- Structural features (64-dim) ---
        struct_features, res_info = extract_all_features(pdb_id, pdb_path)
        if struct_features is None:
            return pdb_id, 'fail', 'extract_all_features returned None'

        feat_lookup = {}
        for i, info in enumerate(res_info):
            feat_lookup[(info['chain'], info['resnum'])] = i

        # --- NMA + Graph features (11-dim) ---
        residues_nma, ca_coords = get_residues_and_coords(pdb_path)
        nma_feat = extract_nma_features(ca_coords)    # (N, 6)
        graph_feat = extract_graph_features(ca_coords)  # (N, 5)
        nma_graph = np.concatenate([nma_feat, graph_feat], axis=1)  # (N, 11)

        nma_lookup = {}
        for i, info in enumerate(residues_nma):
            nma_lookup[(info['chain'], info['resnum'])] = i

        # --- AAindex features (6-dim) ---
        n_aaindex = len(PROPERTY_TABLES)

        # Align ALL features with labels
        aligned_struct = []
        aligned_nma_graph = []
        aligned_aaindex = []
        aligned_labels = []

        for _, lrow in labels_df.iterrows():
            key = (lrow['chain'], lrow['resnum'])
            resname = lrow['resname']

            if key in feat_lookup:
                aligned_struct.append(struct_features[feat_lookup[key]])
            else:
                aligned_struct.append(np.zeros(STRUCTURAL_DIM))

            if key in nma_lookup:
                aligned_nma_graph.append(nma_graph[nma_lookup[key]])
            else:
                aligned_nma_graph.append(np.zeros(NMA_GRAPH_DIM))

            aa_feat = np.zeros(n_aaindex, dtype=np.float32)
            mapped_resname = AAINDEX_NONSTANDARD.get(resname, resname)
            for j, (name, table) in enumerate(PROPERTY_TABLES):
                raw = table.get(mapped_resname)
                if raw is not None:
                    vmin, vrange = NORM_PARAMS[name]
                    aa_feat[j] = (raw - vmin) / vrange if vrange > 0 else 0.0
                else:
                    aa_feat[j] = 0.5
            aligned_aaindex.append(aa_feat)

            aligned_labels.append(lrow['is_allosteric'])

        aligned_struct = np.array(aligned_struct, dtype=np.float32)
        aligned_nma_graph = np.array(aligned_nma_graph, dtype=np.float32)
        aligned_aaindex = np.array(aligned_aaindex, dtype=np.float32)
        aligned_labels = np.array(aligned_labels)

        combined = np.concatenate([aligned_struct, aligned_nma_graph, aligned_aaindex], axis=1)
        combined = np.nan_to_num(combined, nan=0.0)

        np.savez_compressed(out_path,
                            features=combined,
                            labels=aligned_labels,
                            pdb_id=pdb_id)

        n_res = len(aligned_labels)
        return pdb_id, 'ok', f'{n_res} residues'

    except Exception as e:
        return pdb_id, 'fail', str(e)


def phase_features():
    """Extract structural, NMA, graph, and AAindex features (parallelized)."""
    start_logging("features")
    print("=" * 60)
    print("  Phase 3: Feature Extraction (structural + NMA + graph + AAindex)")
    print("=" * 60)

    from multiprocessing import Pool

    N_WORKERS = min(max(1, os.cpu_count() - 2), 48)

    pdb_list = load_independent_pdbs()
    print(f"  Proteins to process: {len(pdb_list)}")
    print(f"  Workers: {N_WORKERS}")

    # Build task list
    tasks = []
    n_skip_existing = 0
    for _, row in pdb_list.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = row['pdb_path']
        out_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_structural.npz")
        label_path = os.path.join(CASBENCH_LABELS_DIR, f"{pdb_id}_labels.csv")
        tasks.append((pdb_id, pdb_path, label_path, out_path))

    print(f"  Tasks queued: {len(tasks)}")

    n_ok = 0
    n_skip = 0
    n_fail = 0
    fail_msgs = []
    start_time = time.time()

    with Pool(processes=N_WORKERS) as pool:
        for pdb_id, status, msg in pool.imap_unordered(_process_single_features, tasks, chunksize=4):
            if status == 'ok':
                n_ok += 1
            elif status == 'skip':
                n_skip += 1
            elif status == 'fail':
                n_fail += 1
                if len(fail_msgs) < 10:
                    fail_msgs.append(f"{pdb_id}: {msg}")

            total_done = n_ok + n_skip + n_fail
            if total_done % 100 == 0 and total_done > 0:
                elapsed = time.time() - start_time
                new_processed = n_ok  # only count non-skipped for rate
                rate = new_processed / max(elapsed, 1)
                todo = len(tasks) - total_done
                remaining = todo / max(rate, 0.01) if rate > 0 else 0
                print(f"  [{total_done}/{len(tasks)}] "
                      f"ok={n_ok} skip={n_skip} fail={n_fail} "
                      f"({elapsed:.0f}s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"\n  Results ({elapsed:.0f}s):")
    print(f"    New:     {n_ok}")
    print(f"    Skipped: {n_skip}")
    print(f"    Failed:  {n_fail}")
    if fail_msgs:
        print(f"    First errors:")
        for msg in fail_msgs:
            print(f"      {msg}")
    print("=" * 60)
    stop_logging()


# ============================================================
# Phase 4: FPocket Features
# ============================================================

def _process_single_fpocket(args):
    """Worker function for parallel FPocket extraction. One protein per WSL call."""
    pdb_id, pdb_path, label_path, out_path = args

    if os.path.exists(out_path):
        return pdb_id, 'skip', None

    if not os.path.exists(label_path):
        return pdb_id, 'fail', 'no labels'

    try:
        import subprocess
        from extract_fpocket import (
            get_residue_keys, run_fpocket_and_parse, parse_all_output,
            FPOCKET_DIM as FP_DIM
        )

        # Get residue keys
        keys = get_residue_keys(pdb_path)
        if not keys:
            return pdb_id, 'fail', 'no residue keys'

        # Run fpocket (single WSL call, 120s timeout built in)
        raw_output, name_stem = run_fpocket_and_parse(pdb_path)
        pockets, pocket_residues = parse_all_output(raw_output)

        n_residues = len(keys)
        features = np.zeros((n_residues, FP_DIM), dtype=np.float32)

        if pockets:
            n_pockets = len(pockets)
            key_to_idx = {key: i for i, key in enumerate(keys)}
            res_to_pockets = {i: [] for i in range(n_residues)}

            for pocket_num, res_keys in pocket_residues.items():
                if pocket_num not in pockets:
                    continue
                for res_key in res_keys:
                    if res_key in key_to_idx:
                        res_to_pockets[key_to_idx[res_key]].append(pocket_num)

            for i in range(n_residues):
                pocket_list = res_to_pockets[i]
                if not pocket_list:
                    continue
                best_pocket = min(pocket_list)
                p = pockets[best_pocket]
                features[i, 0] = 1.0
                features[i, 1] = p.get('Score', 0.0)
                features[i, 2] = p.get('Druggability Score', p.get('Drug Score', 0.0))
                features[i, 3] = p.get('Volume', p.get('Pocket volume (Monte Carlo)',
                                 p.get('Volume Score', 0.0)))
                features[i, 4] = p.get('Hydrophobicity score', p.get('Hydrophobicity Score', 0.0))
                features[i, 5] = p.get('Polarity score', p.get('Polarity Score', 0.0))
                features[i, 6] = (n_pockets - best_pocket + 1) / max(n_pockets, 1)
                features[i, 7] = len(pocket_list)

        # Align with labels
        labels_df = pd.read_csv(label_path, dtype={'chain': str})
        fp_lookup = {}
        for i, key in enumerate(keys):
            fp_lookup[(key[0], key[1])] = i

        aligned_fp = np.zeros((len(labels_df), FP_DIM), dtype=np.float32)
        for j, lrow in labels_df.iterrows():
            lkey = (lrow['chain'], lrow['resnum'])
            if lkey in fp_lookup:
                aligned_fp[j] = features[fp_lookup[lkey]]

        np.savez_compressed(out_path, features=aligned_fp)

        n_in_pocket = int((aligned_fp[:, 0] > 0).sum())
        return pdb_id, 'ok', f'{len(labels_df)} res, {n_in_pocket} in pockets'

    except subprocess.TimeoutExpired:
        return pdb_id, 'fail', 'timeout (>120s)'
    except Exception as e:
        return pdb_id, 'fail', str(e)


def phase_fpocket():
    """Extract FPocket features via WSL (parallelized — 20 workers)."""
    start_logging("fpocket")
    print("=" * 60)
    print("  Phase 4: FPocket Feature Extraction (parallel)")
    print("=" * 60)

    from multiprocessing import Pool
    import subprocess

    N_WORKERS = min(max(1, os.cpu_count() - 2), 48)

    # Test fpocket
    import platform
    if platform.system() == 'Linux':
        print("  Testing fpocket (native Linux)...")
        result = subprocess.run(
            ["bash", "-c", "fpocket 2>&1 | head -1"],
            capture_output=True, text=True, timeout=10
        )
    else:
        print("  Testing WSL + fpocket...")
        result = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "bash", "-c", "fpocket 2>&1 | head -1"],
            capture_output=True, text=True, timeout=10
        )
    if "POCKET" not in result.stdout.upper():
        print("ERROR: fpocket not available!")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        sys.exit(1)
    print("  fpocket OK")
    print(f"  Workers: {N_WORKERS}")

    pdb_list = load_independent_pdbs()

    # Build task list
    tasks = []
    n_skip_pre = 0
    for _, row in pdb_list.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = row['pdb_path']
        out_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_fpocket.npz")
        label_path = os.path.join(CASBENCH_LABELS_DIR, f"{pdb_id}_labels.csv")
        tasks.append((pdb_id, pdb_path, label_path, out_path))

    print(f"  Tasks queued: {len(tasks)}")

    n_ok = 0
    n_skip = 0
    n_fail = 0
    fail_msgs = []
    start_time = time.time()

    with Pool(processes=N_WORKERS) as pool:
        for pdb_id, status, msg in pool.imap_unordered(_process_single_fpocket, tasks, chunksize=1):
            if status == 'ok':
                n_ok += 1
            elif status == 'skip':
                n_skip += 1
            elif status == 'fail':
                n_fail += 1
                if len(fail_msgs) < 20:
                    fail_msgs.append(f"{pdb_id}: {msg}")

            total_done = n_ok + n_skip + n_fail
            if total_done % 50 == 0 and total_done > 0:
                elapsed = time.time() - start_time
                new_processed = n_ok
                rate = new_processed / max(elapsed, 1)
                todo = len(tasks) - total_done
                remaining = todo / max(rate, 0.01) if rate > 0 else 0
                print(f"  [{total_done}/{len(tasks)}] "
                      f"ok={n_ok} skip={n_skip} fail={n_fail} "
                      f"({elapsed:.0f}s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"\n  Results ({elapsed:.0f}s):")
    print(f"    New:     {n_ok}")
    print(f"    Skipped: {n_skip}")
    print(f"    Failed:  {n_fail}")
    if fail_msgs:
        print(f"    Errors:")
        for msg in fail_msgs:
            print(f"      {msg}")
    print("=" * 60)
    stop_logging()


# ============================================================
# Phase 4b: Transfer Entropy Features
# ============================================================

def _process_single_te(args):
    """Worker function for parallel TE extraction. Must be top-level for pickling on Windows."""
    pdb_id, pdb_path, label_path, out_path = args
    if os.path.exists(out_path):
        return pdb_id, 'skip', None
    if not os.path.exists(label_path):
        return pdb_id, 'fail', 'no labels'
    try:
        from extract_transfer_entropy import (
            compute_te_features, get_residues_and_coords as te_get_residues,
            TE_DIM as TE_FEAT_DIM, TE_FEATURE_NAMES
        )

        residues, ca_coords = te_get_residues(pdb_path)
        if len(residues) == 0:
            return pdb_id, 'fail', 'no residues'

        te_feat = compute_te_features(ca_coords)
        labels_df = pd.read_csv(label_path, dtype={'chain': str})

        feat_lookup = {}
        for i, info in enumerate(residues):
            feat_lookup[(info['chain'], info['resnum'])] = i

        aligned = np.zeros((len(labels_df), TE_FEAT_DIM), dtype=np.float32)
        for j, lrow in labels_df.iterrows():
            key = (lrow['chain'], lrow['resnum'])
            if key in feat_lookup:
                aligned[j] = te_feat[feat_lookup[key]]

        np.savez_compressed(out_path, features=aligned, feature_names=TE_FEATURE_NAMES)
        return pdb_id, 'ok', f'{len(labels_df)} res'
    except Exception as e:
        return pdb_id, 'fail', str(e)


def phase_te():
    """Extract Transfer Entropy features for CASBench proteins."""
    start_logging("te")
    print("=" * 60)
    print("  Phase 4b: Transfer Entropy Feature Extraction")
    print("  Algorithm: AllosES (Hu et al., JCIM 2024)")
    print("=" * 60)

    from multiprocessing import Pool

    pdb_list = load_independent_pdbs()
    print(f"  Proteins to process: {len(pdb_list)}")

    N_WORKERS = max(1, os.cpu_count() - 2)

    tasks = []
    for _, row in pdb_list.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = row['pdb_path']
        label_path = os.path.join(CASBENCH_LABELS_DIR, f"{pdb_id}_labels.csv")
        out_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_te.npz")
        tasks.append((pdb_id, pdb_path, label_path, out_path))

    n_existing = sum(1 for _, _, _, op in tasks if os.path.exists(op))
    if n_existing > 0:
        print(f"  Already done: {n_existing} (will skip)")
    print(f"  Workers: {N_WORKERS}")

    n_ok = 0
    n_skip = 0
    n_fail = 0
    fail_msgs = []
    start_time = time.time()

    with Pool(processes=N_WORKERS) as pool:
        for pdb_id, status, msg in pool.imap_unordered(_process_single_te, tasks, chunksize=4):
            if status == 'ok':
                n_ok += 1
            elif status == 'skip':
                n_skip += 1
            elif status == 'fail':
                n_fail += 1
                if len(fail_msgs) < 10:
                    fail_msgs.append(f"{pdb_id}: {msg}")

            total_done = n_ok + n_skip + n_fail
            if total_done % 200 == 0 and total_done > 0:
                elapsed = time.time() - start_time
                rate = max(n_ok, 1) / max(elapsed, 1)
                remaining = (len(tasks) - total_done) / max(rate, 0.01)
                print(f"  [{total_done}/{len(tasks)}] ok={n_ok} skip={n_skip} fail={n_fail} "
                      f"({elapsed:.0f}s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"\n  Results ({elapsed:.0f}s):")
    print(f"    New:     {n_ok}")
    print(f"    Skipped: {n_skip}")
    print(f"    Failed:  {n_fail}")
    if fail_msgs:
        print(f"    First errors:")
        for msg in fail_msgs:
            print(f"      {msg}")
    print("=" * 60)
    stop_logging()


# ============================================================
# Phase 4c: PRS Features
# ============================================================

def _process_single_prs(args):
    """Worker function for parallel PRS extraction. Must be top-level for pickling on Windows."""
    pdb_id, pdb_path, label_path, out_path = args
    if os.path.exists(out_path):
        return pdb_id, 'skip', None
    if not os.path.exists(label_path):
        return pdb_id, 'fail', 'no labels'
    try:
        from extract_prs import (
            compute_prs_features, get_residues_and_coords as prs_get_residues,
            PRS_DIM as PRS_FEAT_DIM, PRS_FEATURE_NAMES
        )

        residues, ca_coords = prs_get_residues(pdb_path)
        if len(residues) == 0:
            return pdb_id, 'fail', 'no residues'

        prs_feat = compute_prs_features(ca_coords)
        labels_df = pd.read_csv(label_path, dtype={'chain': str})

        feat_lookup = {}
        for i, info in enumerate(residues):
            feat_lookup[(info['chain'], info['resnum'])] = i

        aligned = np.zeros((len(labels_df), PRS_FEAT_DIM), dtype=np.float32)
        for j, lrow in labels_df.iterrows():
            key = (lrow['chain'], lrow['resnum'])
            if key in feat_lookup:
                aligned[j] = prs_feat[feat_lookup[key]]

        np.savez_compressed(out_path, features=aligned, feature_names=PRS_FEATURE_NAMES)
        return pdb_id, 'ok', f'{len(labels_df)} res'
    except Exception as e:
        return pdb_id, 'fail', str(e)


def phase_prs():
    """Extract PRS features for CASBench proteins."""
    start_logging("prs")
    print("=" * 60)
    print("  Phase 4c: PRS Feature Extraction")
    print("  Algorithm: Atilgan & Atilgan, PLoS Comp Bio 2009")
    print("=" * 60)

    from multiprocessing import Pool

    pdb_list = load_independent_pdbs()
    print(f"  Proteins to process: {len(pdb_list)}")

    N_WORKERS = max(1, os.cpu_count() - 2)

    tasks = []
    for _, row in pdb_list.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = row['pdb_path']
        label_path = os.path.join(CASBENCH_LABELS_DIR, f"{pdb_id}_labels.csv")
        out_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_prs.npz")
        tasks.append((pdb_id, pdb_path, label_path, out_path))

    n_existing = sum(1 for _, _, _, op in tasks if os.path.exists(op))
    if n_existing > 0:
        print(f"  Already done: {n_existing} (will skip)")
    print(f"  Workers: {N_WORKERS}")

    n_ok = 0
    n_skip = 0
    n_fail = 0
    fail_msgs = []
    start_time = time.time()

    with Pool(processes=N_WORKERS) as pool:
        for pdb_id, status, msg in pool.imap_unordered(_process_single_prs, tasks, chunksize=4):
            if status == 'ok':
                n_ok += 1
            elif status == 'skip':
                n_skip += 1
            elif status == 'fail':
                n_fail += 1
                if len(fail_msgs) < 10:
                    fail_msgs.append(f"{pdb_id}: {msg}")

            total_done = n_ok + n_skip + n_fail
            if total_done % 200 == 0 and total_done > 0:
                elapsed = time.time() - start_time
                rate = max(n_ok, 1) / max(elapsed, 1)
                remaining = (len(tasks) - total_done) / max(rate, 0.01)
                print(f"  [{total_done}/{len(tasks)}] ok={n_ok} skip={n_skip} fail={n_fail} "
                      f"({elapsed:.0f}s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"\n  Results ({elapsed:.0f}s):")
    print(f"    New:     {n_ok}")
    print(f"    Skipped: {n_skip}")
    print(f"    Failed:  {n_fail}")
    if fail_msgs:
        print(f"    First errors:")
        for msg in fail_msgs:
            print(f"      {msg}")
    print("=" * 60)
    stop_logging()


# ============================================================
# Phase 4d: MJ Contact Energy Extraction
# ============================================================

def _process_single_mj(args):
    """Worker function for parallel MJ extraction."""
    pdb_id, pdb_path, label_path, out_path = args
    if os.path.exists(out_path):
        return pdb_id, 'skip', None
    if not os.path.exists(label_path):
        return pdb_id, 'fail', 'no labels'
    try:
        from extract_mj_energy import extract_mj_features, MJ_DIM as MJ_FEAT_DIM

        features = extract_mj_features(pdb_path, label_path)
        if features is None:
            return pdb_id, 'fail', 'extraction failed'

        np.savez_compressed(out_path, features=features)
        return pdb_id, 'ok', f'{len(features)} res'
    except Exception as e:
        return pdb_id, 'fail', str(e)


def phase_mj():
    """Extract MJ contact energy features for CASBench proteins."""
    start_logging("mj")
    print("=" * 60)
    print("  Phase 4d: MJ Contact Energy Feature Extraction")
    print("  Miyazawa & Jernigan, J Mol Biol 1996")
    print("=" * 60)

    from multiprocessing import Pool

    pdb_list = load_independent_pdbs()
    print(f"  Proteins to process: {len(pdb_list)}")

    N_WORKERS = min(max(1, os.cpu_count() - 2), 48)

    tasks = []
    for _, row in pdb_list.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = row['pdb_path']
        label_path = os.path.join(CASBENCH_LABELS_DIR, f"{pdb_id}_labels.csv")
        out_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_mj.npz")
        tasks.append((pdb_id, pdb_path, label_path, out_path))

    n_existing = sum(1 for _, _, _, op in tasks if os.path.exists(op))
    if n_existing > 0:
        print(f"  Already done: {n_existing} (will skip)")
    print(f"  Workers: {N_WORKERS}")

    n_ok = 0
    n_skip = 0
    n_fail = 0
    fail_msgs = []
    start_time = time.time()

    with Pool(processes=N_WORKERS) as pool:
        for pdb_id, status, msg in pool.imap_unordered(_process_single_mj, tasks, chunksize=4):
            if status == 'ok':
                n_ok += 1
            elif status == 'skip':
                n_skip += 1
            elif status == 'fail':
                n_fail += 1
                if len(fail_msgs) < 10:
                    fail_msgs.append(f"{pdb_id}: {msg}")

            total_done = n_ok + n_skip + n_fail
            if total_done % 200 == 0 and total_done > 0:
                elapsed = time.time() - start_time
                rate = max(n_ok, 1) / max(elapsed, 1)
                remaining = (len(tasks) - total_done) / max(rate, 0.01)
                print(f"  [{total_done}/{len(tasks)}] ok={n_ok} skip={n_skip} fail={n_fail} "
                      f"({elapsed:.0f}s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"\n  Results ({elapsed:.0f}s):")
    print(f"    New:     {n_ok}")
    print(f"    Skipped: {n_skip}")
    print(f"    Failed:  {n_fail}")
    if fail_msgs:
        print(f"    First errors:")
        for msg in fail_msgs:
            print(f"      {msg}")
    print("=" * 60)
    stop_logging()


# ============================================================
# Phase 4e: Local Frustration Extraction
# ============================================================

def _process_single_frust(args):
    """Worker function for parallel frustration extraction."""
    pdb_id, pdb_path, label_path, out_path = args
    if os.path.exists(out_path):
        return pdb_id, 'skip', None
    if not os.path.exists(label_path):
        return pdb_id, 'fail', 'no labels'
    try:
        from extract_local_frustration import extract_frustration_fallback, FRUST_DIM_CONFIG

        features = extract_frustration_fallback(pdb_path, label_path)
        if features is None:
            return pdb_id, 'fail', 'extraction failed'

        labels_df = pd.read_csv(label_path, dtype={'chain': str})
        labels = labels_df['is_allosteric'].values

        np.savez_compressed(out_path, features=features, labels=labels, pdb_id=pdb_id)
        return pdb_id, 'ok', f'{len(features)} res'
    except Exception as e:
        return pdb_id, 'fail', str(e)


def phase_frustration():
    """Extract local frustration features for CASBench proteins."""
    start_logging("frustration")
    print("=" * 60)
    print("  Phase 4e: Local Frustration Feature Extraction")
    print("  Ferreiro et al., PNAS 2007, 2011")
    print("=" * 60)

    from multiprocessing import Pool

    pdb_list = load_independent_pdbs()
    print(f"  Proteins to process: {len(pdb_list)}")

    N_WORKERS = min(max(1, os.cpu_count() - 2), 48)

    tasks = []
    for _, row in pdb_list.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = row['pdb_path']
        label_path = os.path.join(CASBENCH_LABELS_DIR, f"{pdb_id}_labels.csv")
        out_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_frust.npz")
        tasks.append((pdb_id, pdb_path, label_path, out_path))

    n_existing = sum(1 for _, _, _, op in tasks if os.path.exists(op))
    if n_existing > 0:
        print(f"  Already done: {n_existing} (will skip)")
    print(f"  Workers: {N_WORKERS}")

    n_ok = 0
    n_skip = 0
    n_fail = 0
    fail_msgs = []
    start_time = time.time()

    with Pool(processes=N_WORKERS) as pool:
        for pdb_id, status, msg in pool.imap_unordered(_process_single_frust, tasks, chunksize=4):
            if status == 'ok':
                n_ok += 1
            elif status == 'skip':
                n_skip += 1
            elif status == 'fail':
                n_fail += 1
                if len(fail_msgs) < 10:
                    fail_msgs.append(f"{pdb_id}: {msg}")

            total_done = n_ok + n_skip + n_fail
            if total_done % 200 == 0 and total_done > 0:
                elapsed = time.time() - start_time
                rate = max(n_ok, 1) / max(elapsed, 1)
                remaining = (len(tasks) - total_done) / max(rate, 0.01)
                print(f"  [{total_done}/{len(tasks)}] ok={n_ok} skip={n_skip} fail={n_fail} "
                      f"({elapsed:.0f}s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"\n  Results ({elapsed:.0f}s):")
    print(f"    New:     {n_ok}")
    print(f"    Skipped: {n_skip}")
    print(f"    Failed:  {n_fail}")
    if fail_msgs:
        print(f"    First errors:")
        for msg in fail_msgs:
            print(f"      {msg}")
    print("=" * 60)
    stop_logging()


# ============================================================
# Phase 5: ESM-2 Extraction
# ============================================================

def phase_esm2():
    """Extract ESM-2 650M embeddings for CASBench proteins."""
    start_logging("esm2")
    print("=" * 60)
    print("  Phase 5: ESM-2 650M Embedding Extraction")
    print("=" * 60)

    import torch
    import esm
    from extract_esm2 import get_sequence_from_pdb, extract_single_sequence

    pdb_list = load_independent_pdbs()

    # Count how many need processing
    n_skip = 0
    to_process = []
    for _, row in pdb_list.iterrows():
        pdb_id = row['pdb_id']
        out_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_esm2.npz")
        if os.path.exists(out_path):
            n_skip += 1
        else:
            to_process.append(row)

    print(f"  Proteins to process: {len(to_process)} (skip={n_skip})")

    if not to_process:
        print("  All done!")
        print("=" * 60)
        stop_logging()
        return

    # Load ESM-2 model
    print("  Loading ESM-2 model (esm2_t33_650M_UR50D)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model_esm = model_esm.to(device)
    model_esm.eval()

    if device.type == 'cuda':
        model_esm = model_esm.half()
        print("  Using FP16 on GPU")

    print("  Model loaded!")

    n_ok = 0
    n_fail = 0
    start_time = time.time()

    for row in to_process:
        pdb_id = row['pdb_id']
        pdb_path = row['pdb_path']

        try:
            chains = get_sequence_from_pdb(pdb_path)
            if not chains:
                n_fail += 1
                continue

            # Extract per-chain embeddings
            chain_residue_maps = {}
            chain_embeddings = {}

            for chain_id, residues in chains.items():
                seq = ''.join(r['aa1'] for r in residues)
                chain_residue_maps[chain_id] = residues
                emb = extract_single_sequence(
                    seq, model_esm, alphabet, batch_converter, device
                )
                chain_embeddings[chain_id] = emb

            # Align with label file
            label_path = os.path.join(CASBENCH_LABELS_DIR, f"{pdb_id}_labels.csv")
            if not os.path.exists(label_path):
                n_fail += 1
                continue

            labels_df = pd.read_csv(label_path, dtype={'chain': str})

            # Build lookup: (chain, resnum) -> embedding vector
            emb_lookup = {}
            for chain_id, residues in chain_residue_maps.items():
                if chain_id in chain_embeddings:
                    emb = chain_embeddings[chain_id]
                    for i, res in enumerate(residues):
                        if i < emb.shape[0]:
                            emb_lookup[(res['chain'], res['resnum'])] = emb[i]

            # Align with labels (drop missing)
            aligned_emb = []
            aligned_labels = []
            for _, lrow in labels_df.iterrows():
                key = (lrow['chain'], lrow['resnum'])
                if key in emb_lookup:
                    aligned_emb.append(emb_lookup[key])
                    aligned_labels.append(lrow['is_allosteric'])
                else:
                    # Zero-pad missing residues to keep alignment with labels
                    aligned_emb.append(np.zeros(ESM_650M_DIM, dtype=np.float32))
                    aligned_labels.append(lrow['is_allosteric'])

            aligned_emb = np.array(aligned_emb, dtype=np.float32)

            out_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_esm2.npz")
            np.savez_compressed(out_path, embeddings=aligned_emb,
                                labels=np.array(aligned_labels), pdb_id=pdb_id)

            n_ok += 1
            if n_ok % 50 == 0:
                elapsed = time.time() - start_time
                rate = n_ok / elapsed
                remaining = (len(to_process) - n_ok - n_fail) / max(rate, 0.01)
                print(f"  Processed {n_ok}/{len(to_process)} "
                      f"({elapsed:.0f}s, ~{remaining:.0f}s remaining)")

        except Exception as e:
            n_fail += 1
            if n_fail <= 10:
                print(f"  ERROR {pdb_id}: {e}")

    elapsed = time.time() - start_time
    print(f"\n  Results ({elapsed:.0f}s):")
    print(f"    New:     {n_ok}")
    print(f"    Skipped: {n_skip}")
    print(f"    Failed:  {n_fail}")
    print("=" * 60)
    stop_logging()


# ============================================================
# Phase 6: Prediction & Evaluation
# ============================================================

def phase_predict():
    """Load saved model, assemble features, predict, evaluate."""
    start_logging("predict")
    print("=" * 60)
    print("  Phase 6: Prediction & Evaluation")
    print("=" * 60)

    import xgboost as xgb
    import joblib
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, precision_recall_fscore_support,
        matthews_corrcoef, confusion_matrix
    )

    # Load model artifacts
    print("  Loading model artifacts...")
    model_path = os.path.join(MODEL_DIR, "xgboost_hybrid.json")
    scaler_path = os.path.join(MODEL_DIR, "feature_scaler.joblib")
    pca_path = os.path.join(MODEL_DIR, "esm2_joint_pca.joblib")

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)

    # Detect expected dimensions from saved artifacts
    n_scaler_features = scaler.n_features_in_
    n_pca_input = pca.n_features_in_
    n_pca_output = pca.n_components_
    n_model_features = model.n_features_in_
    expected_structural = n_scaler_features  # structural + NMA + FPocket + (maybe AAindex)

    print(f"  Model expects: {n_model_features} features")
    print(f"  Scaler expects: {n_scaler_features} structural features")
    print(f"  PCA expects: {n_pca_input} -> {n_pca_output} ESM features")

    # Safety check: CASBench only has 650M embeddings
    if n_pca_input > ESM_650M_DIM:
        print(f"ERROR: Model was trained on joint 650M+3B embeddings (PCA input={n_pca_input}).")
        print(f"  CASBench only has 650M. Retrain with: python build_dataset.py --esm-650m-only")
        stop_logging()
        sys.exit(1)

    # Determine which optional features are included in the scaler
    # Scaler was fit on structural(64) + NMA(11) + FPocket(8) + maybe AAindex(6) + TE(3) + PRS(3) + MJ(2) + Frust(7)
    base_dim = STRUCTURAL_DIM + NMA_GRAPH_DIM + FPOCKET_DIM  # 83
    has_aaindex_in_scaler = (n_scaler_features >= base_dim + AAINDEX_DIM)
    has_te_in_scaler = (n_scaler_features >= base_dim + AAINDEX_DIM + TE_DIM)
    has_prs_in_scaler = (n_scaler_features >= base_dim + AAINDEX_DIM + TE_DIM + PRS_DIM)
    has_mj_in_scaler = (n_scaler_features >= base_dim + AAINDEX_DIM + TE_DIM + PRS_DIM + MJ_DIM)
    has_frust_in_scaler = (n_scaler_features >= base_dim + AAINDEX_DIM + TE_DIM + PRS_DIM + MJ_DIM + FRUST_DIM)

    if has_frust_in_scaler:
        print(f"  AAindex + TE + PRS + MJ + Frust: included in scaler ({n_scaler_features} = 64+11+8+6+3+3+2+7)")
    elif has_mj_in_scaler:
        print(f"  AAindex + TE + PRS + MJ: included in scaler ({n_scaler_features} = 64+11+8+6+3+3+2)")
    elif has_prs_in_scaler:
        print(f"  AAindex + TE + PRS: included in scaler ({n_scaler_features} = 64+11+8+6+3+3)")
    elif has_te_in_scaler:
        print(f"  AAindex + TE: included in scaler ({n_scaler_features} = 64+11+8+6+3)")
    elif has_aaindex_in_scaler:
        print(f"  AAindex: included in scaler ({n_scaler_features} = 64+11+8+6)")
    else:
        print(f"  AAindex: NOT in scaler ({n_scaler_features} = 64+11+8)")

    # Load saved threshold from training results (match model file)
    if "tuned" in model_path:
        results_path = os.path.join(RESULTS_DIR, "xgboost_tuned_results.json")
    else:
        results_path = os.path.join(RESULTS_DIR, "xgboost_hybrid_results.json")
    threshold = 0.5
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            train_results = json.load(f)
        threshold = train_results.get('optimal_threshold', 0.5)
        print(f"  Threshold (from training): {threshold:.4f}")
    else:
        print(f"  WARNING: No training results found, using threshold=0.5")

    # Load PDB list
    pdb_list = load_independent_pdbs()
    print(f"\n  Evaluating {len(pdb_list)} independent CASBench proteins...")

    # Process each protein
    all_y_true = []
    all_y_prob = []
    per_protein_metrics = []
    per_family_data = {}  # cas_entry -> (y_true_list, y_prob_list)

    n_ok = 0
    n_fail = 0
    n_no_features = 0

    for _, row in pdb_list.iterrows():
        pdb_id = row['pdb_id']
        cas_entry = row['cas_entry']

        # Load structural features (_structural.npz has struct+NMA+AAindex = 81 dim)
        struct_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_structural.npz")
        if not os.path.exists(struct_path):
            n_no_features += 1
            continue

        try:
            struct_data = np.load(struct_path)
            struct_feat = struct_data['features']  # (N, 81) = 64 struct + 11 NMA + 6 AAindex
            labels = struct_data['labels']
            n_res = len(labels)

            # Load FPocket features
            fp_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_fpocket.npz")
            if os.path.exists(fp_path):
                fp_data = np.load(fp_path)
                fp_feat = fp_data['features']
                if len(fp_feat) != n_res:
                    fp_feat = np.zeros((n_res, FPOCKET_DIM), dtype=np.float32)
            else:
                fp_feat = np.zeros((n_res, FPOCKET_DIM), dtype=np.float32)

            # Assemble structural features in correct order for scaler
            # Training pipeline order: structural(64) + NMA(11) + FPocket(8) + AAindex(6) + TE(3)
            # Our _structural.npz has: structural(64) + NMA(11) + AAindex(6) = 81
            # We need to insert FPocket between NMA and AAindex, then add TE
            struct_base = struct_feat[:, :STRUCTURAL_DIM + NMA_GRAPH_DIM]  # (N, 75) = 64+11
            aaindex_part = struct_feat[:, STRUCTURAL_DIM + NMA_GRAPH_DIM:]  # (N, 6)

            # Load TE features (3-dim)
            te_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_te.npz")
            if os.path.exists(te_path):
                te_data = np.load(te_path)
                te_feat = te_data['features']
                if len(te_feat) != n_res:
                    te_feat = np.zeros((n_res, TE_DIM), dtype=np.float32)
            else:
                te_feat = np.zeros((n_res, TE_DIM), dtype=np.float32)

            # Load PRS features (3-dim)
            prs_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_prs.npz")
            if os.path.exists(prs_path):
                prs_data = np.load(prs_path)
                prs_feat = prs_data['features']
                if len(prs_feat) != n_res:
                    prs_feat = np.zeros((n_res, PRS_DIM), dtype=np.float32)
            else:
                prs_feat = np.zeros((n_res, PRS_DIM), dtype=np.float32)

            # Load MJ features (2-dim)
            mj_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_mj.npz")
            if os.path.exists(mj_path):
                mj_data = np.load(mj_path)
                mj_feat = mj_data['features']
                if len(mj_feat) != n_res:
                    mj_feat = np.zeros((n_res, MJ_DIM), dtype=np.float32)
            else:
                mj_feat = np.zeros((n_res, MJ_DIM), dtype=np.float32)

            # Load Frustration features (up to 7-dim)
            frust_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_frust.npz")
            if os.path.exists(frust_path):
                frust_data = np.load(frust_path)
                frust_feat = frust_data['features']
                if len(frust_feat) != n_res:
                    frust_feat = np.zeros((n_res, FRUST_DIM), dtype=np.float32)
                elif frust_feat.shape[1] < FRUST_DIM:
                    # Zero-pad if only configurational (4-dim → 7-dim)
                    pad_cols = FRUST_DIM - frust_feat.shape[1]
                    frust_feat = np.concatenate([frust_feat, np.zeros((n_res, pad_cols), dtype=np.float32)], axis=1)
            else:
                frust_feat = np.zeros((n_res, FRUST_DIM), dtype=np.float32)

            if has_frust_in_scaler:
                # Scaler expects: structural(64) + NMA(11) + FPocket(8) + AAindex(6) + TE(3) + PRS(3) + MJ(2) + Frust(7) = 104
                combined_structural = np.concatenate([struct_base, fp_feat, aaindex_part, te_feat, prs_feat, mj_feat, frust_feat], axis=1)
            elif has_mj_in_scaler:
                # Scaler expects: structural(64) + NMA(11) + FPocket(8) + AAindex(6) + TE(3) + PRS(3) + MJ(2) = 97
                combined_structural = np.concatenate([struct_base, fp_feat, aaindex_part, te_feat, prs_feat, mj_feat], axis=1)
            elif has_prs_in_scaler:
                # Scaler expects: structural(64) + NMA(11) + FPocket(8) + AAindex(6) + TE(3) + PRS(3) = 95
                combined_structural = np.concatenate([struct_base, fp_feat, aaindex_part, te_feat, prs_feat], axis=1)
            elif has_te_in_scaler:
                # Scaler expects: structural(64) + NMA(11) + FPocket(8) + AAindex(6) + TE(3) = 92
                combined_structural = np.concatenate([struct_base, fp_feat, aaindex_part, te_feat], axis=1)
            elif has_aaindex_in_scaler:
                # Scaler expects: structural(64) + NMA(11) + FPocket(8) + AAindex(6) = 89
                combined_structural = np.concatenate([struct_base, fp_feat, aaindex_part], axis=1)
            else:
                # Scaler expects: structural(64) + NMA(11) + FPocket(8) = 83
                combined_structural = np.concatenate([struct_base, fp_feat], axis=1)

            assert combined_structural.shape[1] == n_scaler_features, \
                f"{pdb_id}: assembled {combined_structural.shape[1]} features, scaler expects {n_scaler_features}"

            # Scale structural features
            scaled_structural = scaler.transform(combined_structural)
            scaled_structural = np.nan_to_num(scaled_structural, nan=0.0, posinf=0.0, neginf=0.0)

            # Load ESM-2 embeddings
            esm_path = os.path.join(CASBENCH_FEATURES_DIR, f"{pdb_id}_esm2.npz")
            if os.path.exists(esm_path):
                esm_data = np.load(esm_path)
                esm_emb = esm_data['embeddings']  # (N, 1280)
                if len(esm_emb) != n_res:
                    esm_emb = np.zeros((n_res, ESM_650M_DIM), dtype=np.float32)
            else:
                esm_emb = np.zeros((n_res, ESM_650M_DIM), dtype=np.float32)

            # Zero-pad to joint dimension (650M + 3B) for PCA
            esm_joint = np.zeros((n_res, n_pca_input), dtype=np.float32)
            esm_joint[:, :ESM_650M_DIM] = esm_emb
            # 3B columns stay as zeros (not available for CASBench)

            # PCA transform
            esm_pca = pca.transform(esm_joint)

            # Combine: scaled structural + ESM PCA
            X = np.concatenate([scaled_structural, esm_pca], axis=1)

            assert X.shape[1] == n_model_features, \
                f"{pdb_id}: assembled {X.shape[1]} features, model expects {n_model_features}"

            # Predict
            y_prob = model.predict_proba(X)[:, 1]
            y_true = labels.astype(np.float64)

            # Save predictions
            pred_path = os.path.join(CASBENCH_PREDICTIONS_DIR, f"{pdb_id}_pred.npz")
            np.savez_compressed(pred_path, y_true=y_true, y_prob=y_prob)

            # Accumulate for aggregate metrics
            all_y_true.append(y_true)
            all_y_prob.append(y_prob)

            # Per-protein metrics (need both classes present)
            if len(np.unique(y_true)) > 1:
                prot_auroc = roc_auc_score(y_true, y_prob)
                prot_auprc = average_precision_score(y_true, y_prob)
            else:
                prot_auroc = np.nan
                prot_auprc = np.nan

            per_protein_metrics.append({
                'pdb_id': pdb_id,
                'cas_entry': cas_entry,
                'n_residues': n_res,
                'n_allosteric': int(y_true.sum()),
                'allosteric_pct': float(y_true.mean() * 100),
                'auroc': float(prot_auroc),
                'auprc': float(prot_auprc),
            })

            # Accumulate per-family
            if cas_entry not in per_family_data:
                per_family_data[cas_entry] = {'y_true': [], 'y_prob': []}
            per_family_data[cas_entry]['y_true'].append(y_true)
            per_family_data[cas_entry]['y_prob'].append(y_prob)

            n_ok += 1

        except Exception as e:
            n_fail += 1
            if n_fail <= 10:
                print(f"  ERROR {pdb_id}: {e}")
                traceback.print_exc()

    print(f"\n  Processed: {n_ok} ok, {n_fail} failed, {n_no_features} missing features")

    if n_ok == 0:
        print("ERROR: No proteins successfully evaluated!")
        stop_logging()
        return

    # ---- Aggregate Metrics ----
    print(f"\n{'=' * 60}")
    print(f"  AGGREGATE RESULTS (CASBench Blind Test)")
    print(f"{'=' * 60}")

    all_y_true = np.concatenate(all_y_true)
    all_y_prob = np.concatenate(all_y_prob)
    all_y_pred = (all_y_prob >= threshold).astype(int)

    n_total = len(all_y_true)
    n_pos = int(all_y_true.sum())
    print(f"  Total residues: {n_total:,}")
    print(f"  Allosteric:     {n_pos:,} ({100*n_pos/n_total:.2f}%)")
    print(f"  Proteins:       {n_ok}")

    auroc = roc_auc_score(all_y_true, all_y_prob)
    auprc = average_precision_score(all_y_true, all_y_prob)
    prec, rec, f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='binary', zero_division=0)
    mcc = matthews_corrcoef(all_y_true, all_y_pred)
    cm = confusion_matrix(all_y_true, all_y_pred)

    print(f"\n  Threshold: {threshold:.4f}")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  AUPRC:     {auprc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  MCC:       {mcc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:>8,}  FP={cm[0,1]:>8,}")
    print(f"    FN={cm[1,0]:>8,}  TP={cm[1,1]:>8,}")

    # Random baseline for AUPRC comparison
    random_auprc = n_pos / n_total
    print(f"\n  Random baseline AUPRC: {random_auprc:.4f}")
    print(f"  AUPRC lift over random: {auprc/random_auprc:.1f}x")

    # ---- Per-Protein Metrics ----
    per_prot_df = pd.DataFrame(per_protein_metrics)
    valid_auroc = per_prot_df['auroc'].dropna()
    valid_auprc = per_prot_df['auprc'].dropna()

    print(f"\n  Per-protein metrics ({len(valid_auroc)} proteins with both classes):")
    print(f"    AUROC: mean={valid_auroc.mean():.4f}, median={valid_auroc.median():.4f}, std={valid_auroc.std():.4f}")
    print(f"    AUPRC: mean={valid_auprc.mean():.4f}, median={valid_auprc.median():.4f}, std={valid_auprc.std():.4f}")

    # Save per-protein CSV
    per_prot_path = os.path.join(RESULTS_DIR, "casbench_per_protein.csv")
    per_prot_df.to_csv(per_prot_path, index=False)
    print(f"\n  Saved: {per_prot_path}")

    # ---- Per-Family Metrics ----
    family_metrics = []
    for cas_entry, data in per_family_data.items():
        fam_y_true = np.concatenate(data['y_true'])
        fam_y_prob = np.concatenate(data['y_prob'])
        n_fam_res = len(fam_y_true)
        n_fam_pos = int(fam_y_true.sum())

        if len(np.unique(fam_y_true)) > 1:
            fam_auroc = roc_auc_score(fam_y_true, fam_y_prob)
            fam_auprc = average_precision_score(fam_y_true, fam_y_prob)
        else:
            fam_auroc = np.nan
            fam_auprc = np.nan

        family_metrics.append({
            'cas_entry': cas_entry,
            'n_proteins': len(data['y_true']),
            'n_residues': n_fam_res,
            'n_allosteric': n_fam_pos,
            'allosteric_pct': float(n_fam_pos / max(n_fam_res, 1) * 100),
            'auroc': float(fam_auroc),
            'auprc': float(fam_auprc),
        })

    family_df = pd.DataFrame(family_metrics)
    valid_fam_auroc = family_df['auroc'].dropna()
    valid_fam_auprc = family_df['auprc'].dropna()

    print(f"\n  Per-family metrics ({len(valid_fam_auroc)} families with both classes):")
    print(f"    AUROC: mean={valid_fam_auroc.mean():.4f}, median={valid_fam_auroc.median():.4f}")
    print(f"    AUPRC: mean={valid_fam_auprc.mean():.4f}, median={valid_fam_auprc.median():.4f}")

    # Top 5 and bottom 5 families by AUROC
    fam_sorted = family_df.dropna(subset=['auroc']).sort_values('auroc', ascending=False)
    print(f"\n  Top 5 families by AUROC:")
    for _, r in fam_sorted.head(5).iterrows():
        print(f"    {r['cas_entry']}: AUROC={r['auroc']:.3f}, AUPRC={r['auprc']:.3f} "
              f"({r['n_proteins']} proteins, {r['n_allosteric']} allo)")
    print(f"  Bottom 5 families by AUROC:")
    for _, r in fam_sorted.tail(5).iterrows():
        print(f"    {r['cas_entry']}: AUROC={r['auroc']:.3f}, AUPRC={r['auprc']:.3f} "
              f"({r['n_proteins']} proteins, {r['n_allosteric']} allo)")

    family_path = os.path.join(RESULTS_DIR, "casbench_per_family.csv")
    family_df.to_csv(family_path, index=False)
    print(f"\n  Saved: {family_path}")

    # ---- Homology-Stratified Metrics ----
    homology_path = os.path.join(CASBENCH_DIR, "casbench_homology.csv")
    homology_stratified = {}

    if os.path.exists(homology_path):
        print(f"\n{'=' * 60}")
        print(f"  HOMOLOGY-STRATIFIED RESULTS")
        print(f"{'=' * 60}")

        homology_df = pd.read_csv(homology_path)
        # Merge max_identity into per_prot_df
        per_prot_with_hom = per_prot_df.merge(
            homology_df[['pdb_id', 'max_identity', 'closest_training_pdb']],
            on='pdb_id', how='left'
        )
        # Proteins not in homology file get identity=0 (no hit)
        per_prot_with_hom['max_identity'] = per_prot_with_hom['max_identity'].fillna(0.0)

        # Save enriched per-protein CSV
        per_prot_with_hom.to_csv(per_prot_path, index=False)

        identity_thresholds = [
            ("All independent", 1.01),
            ("<70% identity", 0.70),
            ("<50% identity", 0.50),
            ("<30% identity", 0.30),
        ]

        print(f"\n  {'Label':<22} {'Proteins':>8} {'Residues':>10} {'AUROC':>8} {'AUPRC':>8} {'MCC':>8} {'F1':>8}")
        print(f"  {'-'*74}")

        for label, max_thresh in identity_thresholds:
            # Get PDB IDs that pass this threshold
            passing = per_prot_with_hom[per_prot_with_hom['max_identity'] < max_thresh]['pdb_id'].tolist()
            passing_set = set(passing)

            if not passing_set:
                print(f"  {label:<22} {'0':>8} {'—':>10} {'—':>8} {'—':>8} {'—':>8} {'—':>8}")
                homology_stratified[label] = {'n_proteins': 0}
                continue

            # Pool residues from passing proteins only
            strat_y_true = []
            strat_y_prob = []
            for ppm in per_protein_metrics:
                if ppm['pdb_id'] not in passing_set:
                    continue
                pred_path = os.path.join(CASBENCH_PREDICTIONS_DIR, f"{ppm['pdb_id']}_pred.npz")
                if os.path.exists(pred_path):
                    pred_data = np.load(pred_path)
                    strat_y_true.append(pred_data['y_true'])
                    strat_y_prob.append(pred_data['y_prob'])

            if not strat_y_true:
                print(f"  {label:<22} {'0':>8} {'—':>10} {'—':>8} {'—':>8} {'—':>8} {'—':>8}")
                homology_stratified[label] = {'n_proteins': 0}
                continue

            sy_true = np.concatenate(strat_y_true)
            sy_prob = np.concatenate(strat_y_prob)
            sy_pred = (sy_prob >= threshold).astype(int)
            sn = len(sy_true)
            sn_pos = int(sy_true.sum())

            if len(np.unique(sy_true)) > 1:
                s_auroc = roc_auc_score(sy_true, sy_prob)
                s_auprc = average_precision_score(sy_true, sy_prob)
                s_prec, s_rec, s_f1, _ = precision_recall_fscore_support(
                    sy_true, sy_pred, average='binary', zero_division=0)
                s_mcc = matthews_corrcoef(sy_true, sy_pred)
            else:
                s_auroc = s_auprc = s_f1 = s_mcc = None

            print(f"  {label:<22} {len(passing_set):>8} {sn:>10,} {s_auroc:>8.4f} {s_auprc:>8.4f} {s_mcc:>8.4f} {s_f1:>8.4f}")

            homology_stratified[label] = {
                'n_proteins': len(passing_set),
                'n_residues': int(sn),
                'n_allosteric': int(sn_pos),
                'auroc': float(s_auroc),
                'auprc': float(s_auprc),
                'mcc': float(s_mcc),
                'f1': float(s_f1),
            }

        # Identity distribution summary
        print(f"\n  Identity distribution of evaluated proteins:")
        bins = [(0.0, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.90), (0.90, 1.01)]
        for lo, hi in bins:
            n = ((per_prot_with_hom['max_identity'] >= lo) & (per_prot_with_hom['max_identity'] < hi)).sum()
            label = f"  [{lo:.0%}, {hi:.0%})" if hi <= 1.0 else f"  [{lo:.0%}, 100%]"
            print(f"  {label}: {n} proteins")
    else:
        print(f"\n  NOTE: Homology file not found ({homology_path})")
        print(f"  Run homology_filter.py first for stratified metrics.")

    # ---- Save full results JSON ----
    results = {
        'benchmark': 'CASBench',
        'n_proteins_evaluated': n_ok,
        'n_proteins_failed': n_fail,
        'n_proteins_missing_features': n_no_features,
        'n_total_residues': int(n_total),
        'n_allosteric_residues': int(n_pos),
        'allosteric_pct': float(n_pos / n_total * 100),
        'threshold': float(threshold),
        'aggregate': {
            'auroc': float(auroc),
            'auprc': float(auprc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'mcc': float(mcc),
            'confusion_matrix': cm.tolist(),
            'random_auprc': float(random_auprc),
            'auprc_lift': float(auprc / random_auprc),
        },
        'per_protein_summary': {
            'n_with_both_classes': int(len(valid_auroc)),
            'auroc_mean': float(valid_auroc.mean()),
            'auroc_median': float(valid_auroc.median()),
            'auroc_std': float(valid_auroc.std()),
            'auprc_mean': float(valid_auprc.mean()),
            'auprc_median': float(valid_auprc.median()),
            'auprc_std': float(valid_auprc.std()),
        },
        'per_family_summary': {
            'n_families': int(len(valid_fam_auroc)),
            'auroc_mean': float(valid_fam_auroc.mean()),
            'auroc_median': float(valid_fam_auroc.median()),
            'auprc_mean': float(valid_fam_auprc.mean()),
            'auprc_median': float(valid_fam_auprc.median()),
        },
        'model_info': {
            'model_path': model_path,
            'n_model_features': int(n_model_features),
            'n_scaler_features': int(n_scaler_features),
            'n_pca_input': int(n_pca_input),
            'n_pca_output': int(n_pca_output),
            'has_aaindex': has_aaindex_in_scaler,
            'has_frustration': has_frust_in_scaler,
        },
        'homology_stratified': homology_stratified,
    }

    results_json_path = os.path.join(RESULTS_DIR, "casbench_evaluation.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {results_json_path}")

    print(f"\n{'=' * 60}")
    print(f"  CASBench Blind Evaluation Complete")
    print(f"{'=' * 60}")
    stop_logging()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CASBench Blind Evaluation Pipeline")
    parser.add_argument('--phase', type=str, default='all',
                        choices=['all', 'discover', 'labels', 'features', 'fpocket', 'te', 'prs', 'mj', 'frustration', 'esm2', 'predict'],
                        help='Which phase to run (default: all)')
    args = parser.parse_args()

    phases = {
        'discover': phase_discover,
        'labels': phase_labels,
        'features': phase_features,
        'fpocket': phase_fpocket,
        'te': phase_te,
        'prs': phase_prs,
        'mj': phase_mj,
        'frustration': phase_frustration,
        'esm2': phase_esm2,
        'predict': phase_predict,
    }

    if args.phase == 'all':
        start = time.time()
        for name, func in phases.items():
            print(f"\n\n{'#' * 60}")
            print(f"# Running Phase: {name}")
            print(f"{'#' * 60}\n")
            func()
        total = time.time() - start
        print(f"\n\nTotal pipeline time: {total:.0f}s ({total/60:.1f} min)")
    else:
        phases[args.phase]()


if __name__ == '__main__':
    main()
