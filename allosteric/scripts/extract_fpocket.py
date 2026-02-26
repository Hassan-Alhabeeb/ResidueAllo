"""
Extract FPocket features for all proteins.

Runs FPocket natively on Linux (or via WSL on Windows) on each PDB file,
parses pocket info and residue mappings, and produces per-residue features:

FPocket Features (8-dim):
  1. is_in_pocket (binary)
  2. pocket_score (FPocket score of best pocket this residue belongs to)
  3. pocket_druggability (drug score)
  4. pocket_volume (Monte Carlo volume)
  5. pocket_hydrophobicity (hydrophobicity score)
  6. pocket_polarity (polarity score)
  7. pocket_rank (rank of best pocket, 1=top, normalized by total pockets)
  8. n_pockets (number of pockets this residue belongs to)

Supports both native Linux and Windows+WSL. Auto-detects platform.
Uses multiprocessing on native Linux for parallel extraction.
Aligns residues with extract_features.py using same NONSTANDARD_MAP and filtering.
"""

import os
import sys
import re
import subprocess
import tempfile
import argparse
import platform
import shutil
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
from Bio.PDB import PDBParser
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/../ = allosteric/
DATA_DIR = os.path.join(BASE_DIR, "data")
PDB_DIR = os.path.join(DATA_DIR, "pdb_files")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "..", "features")
FPOCKET_DIR = os.path.join(FEATURES_DIR, "fpocket")
os.makedirs(FPOCKET_DIR, exist_ok=True)

# CASBench paths
CASBENCH_DIR = os.path.join(DATA_DIR, "casbench")
CASBENCH_LABELS_DIR = os.path.join(CASBENCH_DIR, "labels")
CASBENCH_FEATURES_DIR = os.path.join(CASBENCH_DIR, "features")

AA_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
           'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
           'TYR', 'VAL']
NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER', 'TPO': 'THR',
    'PTR': 'TYR', 'CSE': 'CYS',
}

FPOCKET_FEATURE_NAMES = [
    'is_in_pocket', 'pocket_score', 'pocket_druggability',
    'pocket_volume', 'pocket_hydrophobicity', 'pocket_polarity',
    'pocket_rank', 'n_pockets'
]

FPOCKET_DIM = len(FPOCKET_FEATURE_NAMES)

# Auto-detect platform
IS_LINUX = platform.system() == 'Linux'


def get_residue_keys(pdb_path):
    """Get residue keys matching extract_features.py ordering."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = structure[0]

    keys = []
    for chain in model:
        for res in chain:
            resname = res.get_resname()
            if res.id[0] != ' ' and resname not in NONSTANDARD_MAP:
                continue
            if resname in NONSTANDARD_MAP:
                resname = NONSTANDARD_MAP[resname]
            if resname not in AA_LIST:
                continue
            keys.append((chain.id, res.id[1], resname))
    return keys


def win_to_wsl(path):
    """Convert Windows path to WSL path."""
    path = os.path.abspath(path)
    assert path[1] == ':', f"Not a drive path: {path}"
    return f"/mnt/{path[0].lower()}/{path[3:].replace(os.sep, '/')}"


def run_fpocket_native(pdb_path):
    """Run FPocket natively on Linux. Returns (raw_stdout, name_stem)."""
    pdb_name = os.path.basename(pdb_path)
    name_stem = os.path.splitext(pdb_name)[0]
    tmp_dir = f"/tmp/fpocket_work/{name_stem}_{os.getpid()}"

    script = f"""#!/bin/bash
mkdir -p {tmp_dir} && cd {tmp_dir}
rm -rf '{name_stem}_out' '{name_stem}.pdb' 2>/dev/null
cp '{pdb_path}' . && fpocket -f '{pdb_name}' >/dev/null 2>&1
echo '===INFO_START==='
cat '{name_stem}_out/{name_stem}_info.txt' 2>/dev/null
echo '===INFO_END==='
for f in {name_stem}_out/pockets/pocket*_atm.pdb; do
  [ -f "$f" ] || continue
  num=$(echo "$f" | grep -oP 'pocket\\K[0-9]+')
  echo "===POCKET_${{num}}_START==="
  grep ^ATOM "$f" 2>/dev/null
  echo "===POCKET_${{num}}_END==="
done
rm -rf '{tmp_dir}' 2>/dev/null
"""
    try:
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True, text=True, timeout=600
        )
        return result.stdout, name_stem
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def run_fpocket_wsl(pdb_path):
    """Run FPocket via WSL on Windows. Returns (raw_stdout, name_stem)."""
    pdb_name = os.path.basename(pdb_path)
    name_stem = os.path.splitext(pdb_name)[0]
    tmp_dir = f"/tmp/fpocket_work/{name_stem}_{os.getpid()}"
    wsl_pdb_path = win_to_wsl(pdb_path)

    script_content = f"""#!/bin/bash
mkdir -p {tmp_dir} && cd {tmp_dir}
rm -rf '{name_stem}_out' '{name_stem}.pdb' 2>/dev/null
cp '{wsl_pdb_path}' . && fpocket -f '{pdb_name}' >/dev/null 2>&1
echo '===INFO_START==='
cat '{name_stem}_out/{name_stem}_info.txt' 2>/dev/null
echo '===INFO_END==='
for f in {name_stem}_out/pockets/pocket*_atm.pdb; do
  [ -f "$f" ] || continue
  num=$(echo "$f" | grep -oP 'pocket\\K[0-9]+')
  echo "===POCKET_${{num}}_START==="
  grep ^ATOM "$f" 2>/dev/null
  echo "===POCKET_${{num}}_END==="
done
rm -rf '{tmp_dir}' 2>/dev/null
"""
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.sh', prefix='fpocket_',
                                      newline='\n', delete=False,
                                      dir=os.path.dirname(pdb_path))
    try:
        tmp.write(script_content)
        tmp.close()
        wsl_script = win_to_wsl(tmp.name)
        result = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "bash", wsl_script],
            capture_output=True, text=True, timeout=600
        )
        return result.stdout, name_stem
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def run_fpocket_and_parse(pdb_path):
    """Run FPocket, auto-detecting platform."""
    if IS_LINUX:
        return run_fpocket_native(pdb_path)
    else:
        return run_fpocket_wsl(pdb_path)


def run_fpocket_batch_wsl(pdb_paths):
    """Run FPocket on MULTIPLE proteins in a single WSL call (Windows only).

    Returns dict of {name_stem: raw_output_section}.
    """
    if not pdb_paths:
        return {}

    tmp_dir = "/tmp/fpocket_work"
    lines = ["#!/bin/bash", f"mkdir -p {tmp_dir}", f"cd {tmp_dir}"]

    stems = []
    for pdb_path in pdb_paths:
        pdb_name = os.path.basename(pdb_path)
        name_stem = os.path.splitext(pdb_name)[0]
        stems.append(name_stem)
        wsl_pdb_path = win_to_wsl(pdb_path)

        lines.append(f"# --- {name_stem} ---")
        lines.append(f"rm -rf '{name_stem}_out' '{name_stem}.pdb' 2>/dev/null")
        lines.append(f"cp '{wsl_pdb_path}' . && fpocket -f '{pdb_name}' >/dev/null 2>&1")
        lines.append(f"echo '===PROTEIN_{name_stem}_START==='")
        lines.append(f"echo '===INFO_START==='")
        lines.append(f"cat '{name_stem}_out/{name_stem}_info.txt' 2>/dev/null")
        lines.append(f"echo '===INFO_END==='")
        lines.append(f"for f in {name_stem}_out/pockets/pocket*_atm.pdb; do")
        lines.append(f"  [ -f \"$f\" ] || continue")
        lines.append(f"  num=$(echo \"$f\" | grep -oP 'pocket\\K[0-9]+')")
        lines.append(f"  echo \"===POCKET_${{num}}_START===\"")
        lines.append(f"  grep ^ATOM \"$f\" 2>/dev/null")
        lines.append(f"  echo \"===POCKET_${{num}}_END===\"")
        lines.append(f"done")
        lines.append(f"echo '===PROTEIN_{name_stem}_END==='")
        lines.append(f"rm -rf '{name_stem}_out' '{name_stem}.pdb' 2>/dev/null")

    script_content = '\n'.join(lines) + '\n'

    sample_dir = os.path.dirname(pdb_paths[0])
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.sh', prefix='fpocket_batch_',
                                      newline='\n', delete=False, dir=sample_dir)
    try:
        tmp.write(script_content)
        tmp.close()
        wsl_script = win_to_wsl(tmp.name)

        timeout = 120 * len(pdb_paths)
        result = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "bash", wsl_script],
            capture_output=True, text=True, timeout=timeout
        )

        raw = result.stdout
        results = {}
        for stem in stems:
            m = re.search(
                f'===PROTEIN_{stem}_START===\n(.*?)===PROTEIN_{stem}_END===',
                raw, re.DOTALL
            )
            results[stem] = m.group(1) if m else ''

        return results
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def parse_all_output(raw_output):
    """Parse the combined fpocket output."""
    # Parse pocket info
    pockets = {}
    info_match = re.search(r'===INFO_START===\n(.*?)===INFO_END===', raw_output, re.DOTALL)
    if info_match:
        current_pocket = None
        for line in info_match.group(1).split('\n'):
            line = line.strip()
            m = re.match(r'Pocket (\d+)', line)
            if m:
                current_pocket = int(m.group(1))
                pockets[current_pocket] = {}
                continue
            if current_pocket and ':' in line:
                key, _, val = line.partition(':')
                key = key.strip()
                val = val.strip()
                try:
                    pockets[current_pocket][key] = float(val)
                except ValueError:
                    pass

    # Parse pocket residues
    pocket_residues = {}
    for m in re.finditer(r'===POCKET_(\d+)_START===\n(.*?)===POCKET_\1_END===', raw_output, re.DOTALL):
        pocket_num = int(m.group(1))
        residue_keys = set()
        for line in m.group(2).strip().split('\n'):
            if not line.startswith('ATOM'):
                continue
            chain = line[21]
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            resname = line[17:20].strip()
            if resname in NONSTANDARD_MAP:
                resname = NONSTANDARD_MAP[resname]
            residue_keys.add((chain, resnum, resname))
        pocket_residues[pocket_num] = residue_keys

    return pockets, pocket_residues


def process_single_protein(args):
    """Process a single protein — designed for multiprocessing Pool (native Linux)."""
    pdb_id, pdb_path, output_dir, label_dir = args

    output_path = os.path.join(output_dir, f"{pdb_id}_fpocket.npz")
    if os.path.exists(output_path):
        return pdb_id, "skipped"

    if not os.path.exists(pdb_path):
        return pdb_id, "no PDB"

    try:
        keys = get_residue_keys(pdb_path)
        if len(keys) == 0:
            return pdb_id, "no residues"

        raw_output, name_stem = run_fpocket_native(pdb_path)
        pockets, pocket_residues = parse_all_output(raw_output)

        n_residues = len(keys)
        features = np.zeros((n_residues, FPOCKET_DIM), dtype=np.float32)

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
        label_path = os.path.join(label_dir, f"{pdb_id}_labels.csv")
        if not os.path.exists(label_path):
            return pdb_id, "error: no label file"

        labels_df = pd.read_csv(label_path, dtype={'chain': str})
        feat_lookup = {}
        for i, key in enumerate(keys):
            feat_lookup[(key[0], key[1])] = i

        aligned_features = []
        for _, lrow in labels_df.iterrows():
            lkey = (lrow['chain'], lrow['resnum'])
            if lkey in feat_lookup:
                aligned_features.append(features[feat_lookup[lkey]])
            else:
                aligned_features.append(np.zeros(FPOCKET_DIM, dtype=np.float32))
        features = np.array(aligned_features, dtype=np.float32)
        n_residues = len(features)

        np.savez_compressed(output_path, features=features)

        n_in_pocket = int((features[:, 0] > 0).sum())
        return pdb_id, f"ok ({n_residues} res, {n_in_pocket} in pockets)"
    except Exception as e:
        return pdb_id, f"error: {e}"


def process_batch_wsl(pdb_ids_batch, pdb_dir=PDB_DIR, output_dir=FPOCKET_DIR, label_dir=PROCESSED_DIR,
                      pdb_path_map=None):
    """Process a batch of proteins in a single WSL call (Windows). Returns list of (pdb_id, status)."""
    def _get_pdb_path(pid):
        if pdb_path_map and pid in pdb_path_map:
            return pdb_path_map[pid]
        return os.path.join(pdb_dir, f"{pid}.pdb")

    to_process = []
    results = []
    for pdb_id in pdb_ids_batch:
        output_path = os.path.join(output_dir, f"{pdb_id}_fpocket.npz")
        if os.path.exists(output_path):
            results.append((pdb_id, "skipped"))
            continue
        pdb_path = _get_pdb_path(pdb_id)
        if not os.path.exists(pdb_path):
            results.append((pdb_id, "no PDB"))
            continue
        to_process.append(pdb_id)

    if not to_process:
        return results

    residue_keys_map = {}
    pdb_paths = []
    for pdb_id in to_process:
        pdb_path = _get_pdb_path(pdb_id)
        try:
            keys = get_residue_keys(pdb_path)
            if len(keys) == 0:
                results.append((pdb_id, "no residues"))
                continue
            residue_keys_map[pdb_id] = keys
            pdb_paths.append(pdb_path)
        except Exception as e:
            results.append((pdb_id, f"error: residue parsing: {e}"))
            continue

    if not pdb_paths:
        return results

    try:
        batch_output = run_fpocket_batch_wsl(pdb_paths)
    except Exception as e:
        for pdb_id in residue_keys_map:
            results.append((pdb_id, f"error: batch failed: {e}"))
        return results

    for pdb_id, keys in residue_keys_map.items():
        name_stem = pdb_id
        raw_section = batch_output.get(name_stem, '')

        try:
            pockets, pocket_residues = parse_all_output(raw_section)

            n_residues = len(keys)
            features = np.zeros((n_residues, FPOCKET_DIM), dtype=np.float32)

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

            label_path = os.path.join(label_dir, f"{pdb_id}_labels.csv")
            if not os.path.exists(label_path):
                results.append((pdb_id, "error: no label file"))
                continue

            labels_df = pd.read_csv(label_path, dtype={'chain': str})
            feat_lookup = {}
            for i, key in enumerate(keys):
                feat_lookup[(key[0], key[1])] = i

            aligned_features = []
            for _, lrow in labels_df.iterrows():
                lkey = (lrow['chain'], lrow['resnum'])
                if lkey in feat_lookup:
                    aligned_features.append(features[feat_lookup[lkey]])
                else:
                    aligned_features.append(np.zeros(FPOCKET_DIM, dtype=np.float32))
            features = np.array(aligned_features, dtype=np.float32)
            n_residues = len(features)

            output_path = os.path.join(output_dir, f"{pdb_id}_fpocket.npz")
            np.savez_compressed(output_path, features=features)

            n_in_pocket = int((features[:, 0] > 0).sum())
            results.append((pdb_id, f"ok ({n_residues} res, {n_in_pocket} in pockets)"))
        except Exception as e:
            results.append((pdb_id, f"error: {e}"))

    return results


BATCH_SIZE = 10  # Process 10 proteins per WSL call (Windows only)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract FPocket features")
    parser.add_argument('--casbench', action='store_true',
                        help='Process CASBench proteins instead of training proteins')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers (Linux only, default: cpu_count-2)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    if IS_LINUX:
        print("  FPocket Feature Extraction (native Linux, multiprocessing)")
    else:
        print("  FPocket Feature Extraction (Windows + WSL, batched)")
    print("=" * 60)

    # Test fpocket
    if IS_LINUX:
        print("\nTesting fpocket...")
        result = subprocess.run(
            ["bash", "-c", "fpocket 2>&1 | head -1"],
            capture_output=True, text=True, timeout=10
        )
    else:
        print("\nTesting WSL + fpocket...")
        result = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "bash", "-c", "fpocket 2>&1 | head -1"],
            capture_output=True, text=True, timeout=10
        )

    if "POCKET HUNTING" not in result.stdout:
        print("ERROR: fpocket not available!")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        sys.exit(1)
    print("  fpocket OK")

    pdb_path_map = None

    if args.casbench:
        output_dir = CASBENCH_FEATURES_DIR
        label_dir = CASBENCH_LABELS_DIR
        os.makedirs(output_dir, exist_ok=True)
        pdb_csv = os.path.join(CASBENCH_DIR, "casbench_independent_pdbs.csv")
        if not os.path.exists(pdb_csv):
            print(f"ERROR: {pdb_csv} not found. Run evaluate_casbench.py --phase discover first.")
            return

        pdb_list = pd.read_csv(pdb_csv)
        pdb_list = pdb_list[pdb_list['is_overlap'] == False]
        pdb_ids = pdb_list['pdb_id'].tolist()
        pdb_path_map = dict(zip(pdb_list['pdb_id'], pdb_list['pdb_path']))
        pdb_dir = PDB_DIR
        print(f"\n  Mode: CASBench ({len(pdb_ids)} independent proteins)")
        print(f"  Output: {output_dir}")
    else:
        output_dir = FPOCKET_DIR
        label_dir = PROCESSED_DIR
        pdb_dir = PDB_DIR

        summary_path = os.path.join(PROCESSED_DIR, "dataset_summary.csv")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            pdb_ids = summary['pdb_id'].tolist()
        else:
            splits_path = os.path.join(PROCESSED_DIR, "train_val_test_splits.csv")
            splits = pd.read_csv(splits_path)
            pdb_ids = splits['pdb_id'].tolist()
        print(f"\n  Mode: Training proteins")
        print(f"  Output: {output_dir}")

    # Check existing
    existing = sum(1 for p in pdb_ids if os.path.exists(os.path.join(output_dir, f"{p}_fpocket.npz")))
    print(f"  Total proteins: {len(pdb_ids)}")
    print(f"  Already done:   {existing}")
    print(f"  Remaining:      {len(pdb_ids) - existing}")

    start_time = time.time()
    n_done = 0
    n_ok = 0
    n_error = 0
    n_skipped = 0

    if IS_LINUX:
        # Native Linux: use multiprocessing Pool
        # Use physical cores only; cap at 48
        n_workers = args.workers if args.workers > 0 else min(max(1, cpu_count() - 2), 48)
        print(f"  Workers: {n_workers}")

        def _get_pdb_path(pid):
            if pdb_path_map and pid in pdb_path_map:
                return pdb_path_map[pid]
            return os.path.join(pdb_dir, f"{pid}.pdb")

        tasks = [(pid, _get_pdb_path(pid), output_dir, label_dir) for pid in pdb_ids]

        with Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(process_single_protein, tasks, chunksize=4):
                pdb_id, status = result
                n_done += 1
                if status == "skipped":
                    n_skipped += 1
                elif status.startswith("ok"):
                    n_ok += 1
                else:
                    n_error += 1
                    if "error" in status:
                        print(f"  ERROR: {pdb_id}: {status}")

                if n_ok > 0 and n_ok % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (n_ok + n_skipped) / max(elapsed, 1)
                    remaining = (len(pdb_ids) - n_done) / max(rate, 0.01)
                    print(f"  [{n_done:>5}/{len(pdb_ids)}] {n_ok} ok, {n_skipped} skipped, "
                          f"{n_error} errors ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    else:
        # Windows: WSL batched approach
        print(f"  Batch size: {BATCH_SIZE} proteins/WSL call")

        for batch_start in range(0, len(pdb_ids), BATCH_SIZE):
            batch = pdb_ids[batch_start:batch_start + BATCH_SIZE]
            batch_results = process_batch_wsl(batch, pdb_dir=pdb_dir, output_dir=output_dir,
                                              label_dir=label_dir, pdb_path_map=pdb_path_map)

            for pdb_id, status in batch_results:
                n_done += 1
                if status == "skipped":
                    n_skipped += 1
                elif status.startswith("ok"):
                    n_ok += 1
                else:
                    n_error += 1
                    if "error" in status:
                        print(f"  ERROR: {pdb_id}: {status}")

            elapsed = time.time() - start_time
            rate = n_done / max(elapsed, 1)
            remaining = (len(pdb_ids) - n_done) / max(rate, 0.01)
            new_ok = sum(1 for _, s in batch_results if s.startswith("ok"))
            if new_ok > 0:
                sample = [(pid, s) for pid, s in batch_results if s.startswith("ok")][-1]
                print(f"  [{n_done:>5}/{len(pdb_ids)}] batch {batch_start//BATCH_SIZE + 1}: "
                      f"{new_ok} ok, last={sample[0]} {sample[1]} "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  FPocket extraction complete ({elapsed:.0f}s)")
    print(f"  Processed: {n_ok}")
    print(f"  Skipped:   {n_skipped}")
    print(f"  Errors:    {n_error}")
    print(f"  Total:     {n_done}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
