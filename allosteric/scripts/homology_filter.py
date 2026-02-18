"""
Homology filter: compare CASBench sequences against training sequences via MMseqs2.

Produces casbench_homology.csv with columns:
  pdb_id, max_identity, closest_training_pdb, n_chains_checked

This allows Phase 6 (predict) to report metrics stratified by identity thresholds
(all / <70% / <50% / <30%) to address the homolog leakage concern.

Usage:
    python homology_filter.py
"""

import os
import sys
import subprocess
import tempfile
import shutil
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
import warnings
import time

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = r"E:\newyear\research_plan\allosteric"
CASBENCH_DIR = os.path.join(BASE_DIR, "data", "casbench")
CASBENCH_CSV = os.path.join(CASBENCH_DIR, "casbench_independent_pdbs.csv")
TRAINING_FASTA = os.path.join(BASE_DIR, "data", "processed", "all_sequences.fasta")
MMSEQS_BAT = os.path.join(BASE_DIR, "tools", "mmseqs", "mmseqs", "mmseqs.bat")
OUTPUT_CSV = os.path.join(CASBENCH_DIR, "casbench_homology.csv")

# ── AA mappings (same as all other scripts) ──────────────────────────────────
AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER', 'TPO': 'THR',
    'PTR': 'TYR', 'CSE': 'CYS',
}
MIN_SEQ_LEN = 10


def get_chain_sequences(pdb_path):
    """Extract unique chain sequences from a PDB file.

    Returns dict of {chain_id: sequence_str}. Deduplicates identical chains
    (e.g., homomers) by sequence content.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = structure[0]

    chains = {}
    seen_seqs = set()

    for chain in model:
        residues = []
        for res in chain:
            if res.id[0] != ' ':
                continue
            resname = res.get_resname()
            if resname in NONSTANDARD_MAP:
                resname = NONSTANDARD_MAP[resname]
            if resname not in AA3TO1:
                continue
            residues.append(AA3TO1[resname])

        seq = ''.join(residues)
        if len(seq) < MIN_SEQ_LEN:
            continue
        if seq in seen_seqs:
            continue  # skip duplicate chains (homomers)
        seen_seqs.add(seq)
        chains[chain.id] = seq

    return chains


def main():
    start_time = time.time()
    print("=" * 60)
    print("  Homology Filter: CASBench vs Training")
    print("=" * 60)

    # Verify inputs exist
    for path, label in [(CASBENCH_CSV, "CASBench CSV"), (TRAINING_FASTA, "Training FASTA"),
                        (MMSEQS_BAT, "MMseqs2")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    # Load CASBench protein list
    pdb_list = pd.read_csv(CASBENCH_CSV)
    pdb_list = pdb_list[pdb_list['is_overlap'] == False]
    print(f"  CASBench independent proteins: {len(pdb_list)}")

    # Step 1: Extract CASBench sequences from PDB files
    print("\n  Extracting CASBench sequences from PDB files...")
    casbench_fasta_entries = []  # (header, sequence)
    pdb_chain_map = {}  # pdb_id -> list of chain_ids written

    n_ok = 0
    n_fail = 0
    for _, row in pdb_list.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = row['pdb_path']

        try:
            chains = get_chain_sequences(pdb_path)
            if not chains:
                n_fail += 1
                continue

            chain_ids = []
            for chain_id, seq in chains.items():
                header = f"{pdb_id}_{chain_id}"
                casbench_fasta_entries.append((header, seq))
                chain_ids.append(chain_id)

            pdb_chain_map[pdb_id] = chain_ids
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"    FAIL: {pdb_id}: {e}")

        if (n_ok + n_fail) % 500 == 0:
            print(f"    [{n_ok + n_fail}/{len(pdb_list)}] ok={n_ok} fail={n_fail}")

    print(f"  Extracted: {n_ok} proteins, {len(casbench_fasta_entries)} unique chain sequences")
    if n_fail > 0:
        print(f"  Failed: {n_fail}")

    # Step 2: Write CASBench FASTA
    work_dir = tempfile.mkdtemp(prefix="homology_filter_")
    casbench_fasta_path = os.path.join(work_dir, "casbench.fasta")

    with open(casbench_fasta_path, 'w') as f:
        for header, seq in casbench_fasta_entries:
            f.write(f">{header}\n{seq}\n")
    print(f"  CASBench FASTA: {len(casbench_fasta_entries)} entries")

    # Step 3: Run MMseqs2 search
    print("\n  Running MMseqs2 search...")

    casbench_db = os.path.join(work_dir, "casbenchDB")
    training_db = os.path.join(work_dir, "trainingDB")
    result_db = os.path.join(work_dir, "resultDB")
    tmp_mmseqs = os.path.join(work_dir, "tmp")
    result_tsv = os.path.join(work_dir, "result.tsv")

    os.makedirs(tmp_mmseqs, exist_ok=True)

    # Create databases
    cmd = f'"{MMSEQS_BAT}" createdb "{casbench_fasta_path}" "{casbench_db}"'
    print(f"    {cmd}")
    subprocess.run(cmd, shell=True, check=True, capture_output=True)

    cmd = f'"{MMSEQS_BAT}" createdb "{TRAINING_FASTA}" "{training_db}"'
    print(f"    {cmd}")
    subprocess.run(cmd, shell=True, check=True, capture_output=True)

    # Search: CASBench (query) vs Training (target)
    # --min-seq-id 0.0 to get all hits (we want full identity distribution)
    # -s 7.5 for high sensitivity
    # --max-seqs 1 to only keep the best hit per query
    cmd = (
        f'"{MMSEQS_BAT}" search "{casbench_db}" "{training_db}" "{result_db}" "{tmp_mmseqs}" '
        f'--min-seq-id 0.0 -s 7.5 -e 10'
    )
    print(f"    {cmd}")
    subprocess.run(cmd, shell=True, check=True, capture_output=True, timeout=600)

    # Convert results to TSV
    cmd = (
        f'"{MMSEQS_BAT}" convertalis "{casbench_db}" "{training_db}" "{result_db}" "{result_tsv}" '
        f'--format-output "query,target,fident"'
    )
    print(f"    {cmd}")
    subprocess.run(cmd, shell=True, check=True, capture_output=True)

    # Step 4: Parse results
    print("\n  Parsing results...")
    # Result format: CASBenchPDB_Chain \t TrainingPDB \t identity_fraction
    chain_hits = {}  # (pdb_id, chain) -> (identity, training_pdb)

    if os.path.exists(result_tsv):
        with open(result_tsv) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                query = parts[0]       # e.g., "1NXE_A"
                target = parts[1]      # e.g., "3K8Y"
                fident = float(parts[2])  # e.g., 0.95

                # Parse PDB ID from query header
                if '_' in query:
                    pdb_id = query.rsplit('_', 1)[0]
                else:
                    pdb_id = query

                key = (pdb_id, query)
                if key not in chain_hits or fident > chain_hits[key][0]:
                    chain_hits[key] = (fident, target)

    # Aggregate: for each CASBench PDB, take max identity across all its chains
    pdb_max_identity = {}  # pdb_id -> (max_identity, closest_training_pdb)

    for (pdb_id, _), (fident, target) in chain_hits.items():
        if pdb_id not in pdb_max_identity or fident > pdb_max_identity[pdb_id][0]:
            pdb_max_identity[pdb_id] = (fident, target)

    # Build output DataFrame
    rows = []
    for _, row in pdb_list.iterrows():
        pdb_id = row['pdb_id']
        n_chains = len(pdb_chain_map.get(pdb_id, []))

        if pdb_id in pdb_max_identity:
            max_id, closest = pdb_max_identity[pdb_id]
        else:
            max_id = 0.0
            closest = ''

        rows.append({
            'pdb_id': pdb_id,
            'max_identity': round(max_id, 4),
            'closest_training_pdb': closest,
            'n_chains_checked': n_chains,
        })

    homology_df = pd.DataFrame(rows)
    homology_df.to_csv(OUTPUT_CSV, index=False)

    # Step 5: Report
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Homology Analysis Results")
    print(f"{'=' * 60}")
    print(f"  Total proteins analyzed: {len(homology_df)}")
    print(f"  With hits in training:   {(homology_df['max_identity'] > 0).sum()}")
    print(f"  No hits (identity = 0):  {(homology_df['max_identity'] == 0).sum()}")

    # Distribution at key thresholds
    thresholds = [0.30, 0.50, 0.70, 0.90, 0.95]
    print(f"\n  Identity distribution:")
    print(f"  {'Threshold':<15} {'Above':<10} {'Below':<10} {'% Below':<10}")
    print(f"  {'-'*45}")
    for t in thresholds:
        above = (homology_df['max_identity'] >= t).sum()
        below = (homology_df['max_identity'] < t).sum()
        pct = 100 * below / len(homology_df)
        print(f"  {t:<15.0%} {above:<10} {below:<10} {pct:<10.1f}")

    print(f"\n  Recommended reporting thresholds for Phase 6:")
    for label, t in [("All independent", 1.01), ("<70% identity", 0.70),
                     ("<50% identity", 0.50), ("<30% identity", 0.30)]:
        n = (homology_df['max_identity'] < t).sum()
        print(f"    {label:<20}: {n:>5} proteins")

    print(f"\n  Saved: {OUTPUT_CSV}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'=' * 60}")

    # Cleanup temp directory
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
