"""
Post-process ESM-2 3B embeddings from Kaggle to align with label files.

The Kaggle extraction script saves raw per-chain concatenated embeddings
(PDB chain order). This script aligns them with the _labels.csv files
so that row i of the output corresponds to residue i in the label file,
matching the 650M embeddings format exactly.

Input:  kaggle/esm2_3b_embeddings/{pdb_id}_esm2_3b.npz  (raw, from Kaggle)  -> shape (total_residues, 2560)
Output: features/esm2_3b_embeddings/{pdb_id}_esm2_3b.npz (aligned)         -> shape (n_label_residues, 2560)

Originals in kaggle/ are NOT modified.

Usage:
    python align_esm2_3b.py
"""

import os
import re
import sys
import time
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = r"E:\newyear\research_plan\allosteric\data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PDB_DIR = os.path.join(DATA_DIR, "pdb_files")
RAW_ESM3B_DIR = r"E:\newyear\research_plan\allosteric\kaggle\esm2_3b_embeddings"   # Kaggle output (raw, unaligned)
ESM3B_DIR = r"E:\newyear\research_plan\allosteric\features\esm2_3b_embeddings"      # Aligned output

ESM_DIM = 2560  # ESM-2 3B hidden dimension

# ── AA mappings (must match extract_esm2.py / extract_features.py) ─────────
NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER', 'TPO': 'THR',
    'PTR': 'TYR', 'CSE': 'CYS',
}
AA_LIST = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
           'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
           'TYR', 'VAL'}


def get_chain_residue_map(pdb_path):
    """Extract per-chain residue lists from PDB. Same logic as extract_esm2.py."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = structure[0]

    chains = {}
    for chain in model:
        residues = []
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
            })
        if residues:
            chains[chain.id] = residues

    return chains


def align_single_protein(pdb_id):
    """Align 3B embeddings for one protein against its label file.

    Returns: (success: bool, message: str)
    """
    raw_npz_path = os.path.join(RAW_ESM3B_DIR, f"{pdb_id}_esm2_3b.npz")
    out_npz_path = os.path.join(ESM3B_DIR, f"{pdb_id}_esm2_3b.npz")
    label_path = os.path.join(PROCESSED_DIR, f"{pdb_id}_labels.csv")
    pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")

    if not os.path.exists(raw_npz_path):
        return False, "no npz"
    if not os.path.exists(label_path):
        return False, "no labels"
    if not os.path.exists(pdb_path):
        return False, "no pdb"

    # Skip if already aligned
    if os.path.exists(out_npz_path):
        return True, "already aligned"

    # Load raw embeddings (concatenated chains in PDB order)
    raw = np.load(raw_npz_path)
    raw_emb = raw['embeddings']  # shape: (total_residues, 2560)

    if raw_emb.shape[1] != ESM_DIM:
        return False, f"wrong dim: {raw_emb.shape[1]}"

    # Get per-chain residue mapping from PDB
    chain_map = get_chain_residue_map(pdb_path)
    if not chain_map:
        return False, "no chains in PDB"

    # Split raw embeddings back into per-chain arrays
    # Chain order matches prepare_sequences.py (iterates model.get_chains())
    chain_ids = list(chain_map.keys())
    chain_lengths = [len(chain_map[cid]) for cid in chain_ids]
    total_pdb_residues = sum(chain_lengths)

    if raw_emb.shape[0] != total_pdb_residues:
        return False, f"length mismatch: npz={raw_emb.shape[0]}, pdb={total_pdb_residues}"

    # Build (chain, resnum) -> embedding lookup
    emb_lookup = {}
    offset = 0
    for cid in chain_ids:
        residues = chain_map[cid]
        for i, res in enumerate(residues):
            emb_lookup[(res['chain'], res['resnum'])] = raw_emb[offset + i]
        offset += len(residues)

    # Align with labels (same logic as extract_esm2.py lines 201-236)
    labels_df = pd.read_csv(label_path)
    aligned_emb = []
    n_missing = 0

    for _, lrow in labels_df.iterrows():
        key = (lrow['chain'], lrow['resnum'])
        if key in emb_lookup:
            aligned_emb.append(emb_lookup[key])
        else:
            n_missing += 1

    if len(aligned_emb) == 0:
        return False, "no residues matched"

    aligned_emb = np.array(aligned_emb, dtype=np.float32)  # (N, 2560)

    # Save aligned version to output dir (originals in kaggle/ stay intact)
    np.savez_compressed(out_npz_path, embeddings=aligned_emb)

    msg = f"OK ({aligned_emb.shape[0]} residues"
    if n_missing > 0:
        msg += f", {n_missing} dropped"
    msg += ")"
    return True, msg


def main():
    print("=" * 60)
    print("  Post-Alignment: ESM-2 3B Embeddings")
    print("=" * 60)

    if not os.path.isdir(RAW_ESM3B_DIR):
        print(f"\n  ERROR: Raw directory not found: {RAW_ESM3B_DIR}")
        print(f"  Extract Kaggle results.zip there first.")
        sys.exit(1)

    os.makedirs(ESM3B_DIR, exist_ok=True)
    print(f"\n  Raw input:  {RAW_ESM3B_DIR}")
    print(f"  Aligned output: {ESM3B_DIR}")

    # Find all 3B npz files in raw directory
    npz_files = [f for f in os.listdir(RAW_ESM3B_DIR) if f.endswith('_esm2_3b.npz')]
    npz_files.sort()
    print(f"  Found {len(npz_files)} .npz files")

    if len(npz_files) == 0:
        print("  Nothing to align.")
        return

    # Check how many are already done
    n_already = sum(1 for f in npz_files if os.path.exists(os.path.join(ESM3B_DIR, f)))
    if n_already > 0:
        print(f"  Already aligned: {n_already}/{len(npz_files)} (will skip these)")

    # Process all proteins
    start_time = time.time()
    n_success = 0
    n_fail = 0
    n_dropped_total = 0
    errors = []

    for i, fname in enumerate(npz_files):
        pdb_id = fname.replace('_esm2_3b.npz', '')
        success, msg = align_single_protein(pdb_id)

        if success:
            n_success += 1
            if 'dropped' in msg:
                m = re.search(r'(\d+) dropped', msg)
                if m:
                    n_dropped_total += int(m.group(1))
        else:
            n_fail += 1
            errors.append((pdb_id, msg))

        if (i + 1) % 200 == 0 or (i + 1) == len(npz_files):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(npz_files) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>5}/{len(npz_files)}] "
                  f"OK={n_success} Fail={n_fail} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - start_time
    print(f"\n{'─' * 40}")
    print(f"  Alignment complete ({elapsed:.1f}s)")
    print(f"{'─' * 40}")
    print(f"  Aligned:  {n_success}")
    print(f"  Failed:   {n_fail}")
    print(f"  Total residues dropped (across all proteins): {n_dropped_total}")

    if errors:
        print(f"\n  Failures (first 20):")
        for pdb_id, msg in errors[:20]:
            print(f"    {pdb_id}: {msg}")

    if n_success > 0:
        # Verify one aligned file
        aligned_files = [f for f in os.listdir(ESM3B_DIR) if f.endswith('_esm2_3b.npz')]
        if aligned_files:
            sample_path = os.path.join(ESM3B_DIR, aligned_files[0])
            sample_id = aligned_files[0].replace('_esm2_3b.npz', '')
            sample = np.load(sample_path)
            print(f"\n  Verification ({sample_id}):")
            print(f"    Shape: {sample['embeddings'].shape}")
            print(f"    Dtype: {sample['embeddings'].dtype}")
            print(f"    Range: [{sample['embeddings'].min():.3f}, {sample['embeddings'].max():.3f}]")


if __name__ == '__main__':
    main()
