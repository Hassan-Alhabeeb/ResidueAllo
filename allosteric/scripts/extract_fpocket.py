"""
Extract FPocket features for all proteins.

Runs FPocket via WSL on each PDB file, parses pocket info and residue mappings,
and produces per-residue features:

FPocket Features (8-dim):
  1. is_in_pocket (binary)
  2. pocket_score (FPocket score of best pocket this residue belongs to)
  3. pocket_druggability (drug score)
  4. pocket_volume (Monte Carlo volume)
  5. pocket_hydrophobicity (hydrophobicity score)
  6. pocket_polarity (polarity score)
  7. pocket_rank (rank of best pocket, 1=top, normalized by total pockets)
  8. n_pockets (number of pockets this residue belongs to)

Uses WSL to call fpocket (Linux binary) from Windows.
Aligns residues with extract_features.py using same NONSTANDARD_MAP and filtering.
"""

import os
import sys
import re
import subprocess
import tempfile
import numpy as np
import time
from Bio.PDB import PDBParser
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = r"E:\newyear\research_plan\allosteric\data"
PDB_DIR = os.path.join(DATA_DIR, "pdb_files")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "..", "features")
FPOCKET_DIR = os.path.join(FEATURES_DIR, "fpocket")
os.makedirs(FPOCKET_DIR, exist_ok=True)

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


def get_residue_keys(pdb_path):
    """Get residue keys matching extract_features.py ordering."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = structure[0]

    keys = []
    for chain in model:
        for res in chain:
            if res.id[0] != ' ':
                continue
            resname = res.get_resname()
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


def run_fpocket_and_parse(pdb_path):
    """Run FPocket + parse all output in ONE WSL call. Returns (raw_stdout, name_stem).

    Uses a temp bash script file to avoid Windows->WSL argument escaping issues
    that mangle $f and $num shell variables.
    """
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
    # Write to temp file with Unix line endings
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.sh', prefix='fpocket_',
                                      newline='\n', delete=False,
                                      dir=os.path.dirname(pdb_path))
    try:
        tmp.write(script_content)
        tmp.close()
        wsl_script = win_to_wsl(tmp.name)

        result = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "bash", wsl_script],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout, name_stem
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def run_fpocket_batch(pdb_paths):
    """Run FPocket on MULTIPLE proteins in a single WSL call. Much faster.

    Returns dict of {name_stem: raw_output_section}.
    """
    if not pdb_paths:
        return {}

    tmp_dir = "/tmp/fpocket_work"
    lines = [f"#!/bin/bash", f"mkdir -p {tmp_dir}", f"cd {tmp_dir}"]

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

    # Write to temp file
    sample_dir = os.path.dirname(pdb_paths[0])
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.sh', prefix='fpocket_batch_',
                                      newline='\n', delete=False, dir=sample_dir)
    try:
        tmp.write(script_content)
        tmp.close()
        wsl_script = win_to_wsl(tmp.name)

        timeout = 120 * len(pdb_paths)  # Scale timeout with batch size
        result = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "bash", wsl_script],
            capture_output=True, text=True, timeout=timeout
        )

        # Split output by protein
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
    """Parse the combined fpocket output from a single WSL call."""
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


def extract_fpocket_features(pdb_path, residue_keys):
    """Run FPocket and extract per-residue features. Single WSL call."""
    n_residues = len(residue_keys)
    features = np.zeros((n_residues, FPOCKET_DIM), dtype=np.float32)

    # One WSL call for everything
    raw_output, name_stem = run_fpocket_and_parse(pdb_path)
    pockets, pocket_residues = parse_all_output(raw_output)

    if not pockets:
        return features  # No pockets found — all zeros

    # Diagnostic: print pocket keys once (helps verify key names match our code)
    first_pocket = pockets[min(pockets.keys())]
    if first_pocket and not hasattr(extract_fpocket_features, '_printed_keys'):
        extract_fpocket_features._printed_keys = True
        print(f"  [DEBUG] FPocket info keys for first pocket: {list(first_pocket.keys())}")

    n_pockets = len(pockets)

    # Build residue → pocket mapping
    key_to_idx = {}
    for i, key in enumerate(residue_keys):
        key_to_idx[key] = i

    res_to_pockets = {i: [] for i in range(n_residues)}
    for pocket_num, res_keys in pocket_residues.items():
        if pocket_num not in pockets:
            continue
        for res_key in res_keys:
            if res_key in key_to_idx:
                res_to_pockets[key_to_idx[res_key]].append(pocket_num)

    # Assign features per residue
    for i in range(n_residues):
        pocket_list = res_to_pockets[i]
        if not pocket_list:
            continue

        best_pocket = min(pocket_list)
        p = pockets[best_pocket]

        features[i, 0] = 1.0  # is_in_pocket
        features[i, 1] = p.get('Score', 0.0)
        features[i, 2] = p.get('Druggability Score', p.get('Drug Score', 0.0))
        features[i, 3] = p.get('Volume', p.get('Pocket volume (Monte Carlo)',
                         p.get('Volume Score', 0.0)))
        features[i, 4] = p.get('Hydrophobicity score', p.get('Hydrophobicity Score', 0.0))
        features[i, 5] = p.get('Polarity score', p.get('Polarity Score', 0.0))
        features[i, 6] = (n_pockets - best_pocket + 1) / max(n_pockets, 1)  # Normalized rank (1.0=best)
        features[i, 7] = len(pocket_list)

    return features


def process_batch(pdb_ids_batch):
    """Process a batch of proteins in a single WSL call. Returns list of (pdb_id, status)."""
    # Pre-filter: skip existing and missing PDBs
    to_process = []
    results = []
    for pdb_id in pdb_ids_batch:
        output_path = os.path.join(FPOCKET_DIR, f"{pdb_id}_fpocket.npz")
        if os.path.exists(output_path):
            results.append((pdb_id, "skipped"))
            continue
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            results.append((pdb_id, "no PDB"))
            continue
        to_process.append(pdb_id)

    if not to_process:
        return results

    # Get residue keys for each protein (Python-side, fast)
    residue_keys_map = {}
    pdb_paths = []
    for pdb_id in to_process:
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
        keys = get_residue_keys(pdb_path)
        if len(keys) == 0:
            results.append((pdb_id, "no residues"))
            continue
        residue_keys_map[pdb_id] = keys
        pdb_paths.append(pdb_path)

    if not pdb_paths:
        return results

    # Run fpocket on all proteins in ONE WSL call
    try:
        batch_output = run_fpocket_batch(pdb_paths)
    except Exception as e:
        # If batch fails, mark all as error
        for pdb_id in residue_keys_map:
            results.append((pdb_id, f"error: batch failed: {e}"))
        return results

    # Parse each protein's output and save features
    for pdb_id, keys in residue_keys_map.items():
        name_stem = pdb_id
        raw_section = batch_output.get(name_stem, '')

        try:
            pockets, pocket_residues = parse_all_output(raw_section)

            n_residues = len(keys)
            features = np.zeros((n_residues, FPOCKET_DIM), dtype=np.float32)

            if pockets:
                # Debug: print keys once
                first_pocket = pockets[min(pockets.keys())]
                if first_pocket and not hasattr(process_batch, '_printed_keys'):
                    process_batch._printed_keys = True
                    print(f"  [DEBUG] FPocket info keys: {list(first_pocket.keys())}")

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

            output_path = os.path.join(FPOCKET_DIR, f"{pdb_id}_fpocket.npz")
            np.savez_compressed(output_path, features=features)

            n_in_pocket = int((features[:, 0] > 0).sum())
            results.append((pdb_id, f"ok ({n_residues} res, {n_in_pocket} in pockets)"))
        except Exception as e:
            results.append((pdb_id, f"error: {e}"))

    return results


BATCH_SIZE = 10  # Process 10 proteins per WSL call


def main():
    print("=" * 60)
    print("  FPocket Feature Extraction (batched)")
    print("=" * 60)

    # Test WSL + fpocket
    print("\nTesting WSL + fpocket...")
    result = subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "bash", "-c", "fpocket 2>&1 | head -1"],
        capture_output=True, text=True, timeout=10
    )
    if "POCKET HUNTING" not in result.stdout and "fpocket" not in result.stdout.lower():
        print("ERROR: fpocket not available in WSL!")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        sys.exit(1)
    print("  fpocket OK")

    # Load protein list
    import pandas as pd
    splits_path = os.path.join(PROCESSED_DIR, "train_val_test_splits.csv")
    splits = pd.read_csv(splits_path)
    pdb_ids = splits['pdb_id'].tolist()

    # Check existing
    existing = sum(1 for p in pdb_ids if os.path.exists(os.path.join(FPOCKET_DIR, f"{p}_fpocket.npz")))
    print(f"\n  Total proteins: {len(pdb_ids)}")
    print(f"  Already done:   {existing}")
    print(f"  Remaining:      {len(pdb_ids) - existing}")
    print(f"  Batch size:     {BATCH_SIZE} proteins/WSL call")

    start_time = time.time()
    n_done = 0
    n_ok = 0
    n_error = 0
    n_skipped = 0

    # Process in batches
    for batch_start in range(0, len(pdb_ids), BATCH_SIZE):
        batch = pdb_ids[batch_start:batch_start + BATCH_SIZE]
        batch_results = process_batch(batch)

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

        # Progress every batch
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
