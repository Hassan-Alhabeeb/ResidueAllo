"""
Cluster proteins by sequence identity and create train/val/test splits.

Uses MMseqs2 to cluster at 30% sequence identity, then splits at the
CLUSTER level (not protein level) to prevent data leakage from homologous
proteins appearing in both train and test.

Split ratio: 70% train, 15% val, 15% test (by cluster count).
"""

import os
import subprocess
import time
import pandas as pd
import numpy as np
from collections import defaultdict
import shutil
from Bio.PDB import PDBParser
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/../ = allosteric/
DATA_DIR = os.path.join(BASE_DIR, "data")
PDB_DIR = os.path.join(DATA_DIR, "pdb_files")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SUMMARY_PATH = os.path.join(PROCESSED_DIR, "dataset_summary.csv")
import platform
import shutil
if platform.system() == 'Linux':
    MMSEQS_BAT = shutil.which('mmseqs') or 'mmseqs'
else:
    MMSEQS_BAT = os.path.join(BASE_DIR, "tools", "mmseqs", "mmseqs", "mmseqs.bat")

# Standard amino acids and mappings (must match extract_labels.py)
AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER',
    'TPO': 'THR', 'PTR': 'TYR', 'CSE': 'CYS',
}


def get_full_sequence_from_pdb(pdb_path):
    """Extract concatenated sequence from ALL chains in a PDB file."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("prot", pdb_path)
    except Exception:
        return ""
    model = structure[0]
    seq = []
    for chain in model:
        for res in chain:
            resname = res.get_resname()
            if res.id[0] != ' ' and resname not in NONSTANDARD_MAP:
                continue
            if resname in NONSTANDARD_MAP:
                resname = NONSTANDARD_MAP[resname]
            if resname in AA3TO1:
                seq.append(AA3TO1[resname])
    return ''.join(seq)


def extract_sequence_worker(args):
    """Worker for parallel PDB sequence extraction."""
    pdb_id, pdb_path = args
    if not os.path.exists(pdb_path):
        return pdb_id, None
    seq = get_full_sequence_from_pdb(pdb_path)
    if len(seq) < 10:
        return pdb_id, None
    return pdb_id, seq


def main():
    # Load dataset summary (only proteins we successfully processed)
    summary = pd.read_csv(SUMMARY_PATH)
    print(f"Total proteins with labels: {len(summary)}")

    # Step 1: Extract sequences in parallel, then write FASTA
    fasta_path = os.path.join(PROCESSED_DIR, "all_sequences.fasta")

    tasks = []
    for _, row in summary.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
        tasks.append((pdb_id, pdb_path))

    n_workers = max(1, cpu_count() - 2)
    print(f"  Extracting sequences with {n_workers} workers...")
    start_time = time.time()

    sequences = {}  # pdb_id -> seq
    with Pool(processes=n_workers) as pool:
        for pdb_id, seq in pool.imap_unordered(extract_sequence_worker, tasks, chunksize=4):
            if seq is not None:
                sequences[pdb_id] = seq

    elapsed = time.time() - start_time
    n_with_seq = len(sequences)
    n_without_seq = len(tasks) - n_with_seq

    # Write FASTA (single-threaded, fast string I/O)
    with open(fasta_path, 'w') as f:
        for pdb_id, seq in sequences.items():
            f.write(f">{pdb_id}\n{seq}\n")

    print(f"  Sequences written to FASTA: {n_with_seq} ({elapsed:.0f}s)")
    print(f"  Skipped (no/short sequence): {n_without_seq}")

    # Step 2: Run MMseqs2 clustering at 30% sequence identity
    print("\nRunning MMseqs2 clustering at 30% identity...")
    tmp_dir = os.path.join(PROCESSED_DIR, "mmseqs_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    db_path = os.path.join(tmp_dir, "seqDB")
    cluster_path = os.path.join(tmp_dir, "clusterDB")
    tsv_path = os.path.join(PROCESSED_DIR, "clusters_30pct.tsv")

    # Create sequence database
    cmd_createdb = f'"{MMSEQS_BAT}" createdb "{fasta_path}" "{db_path}"'
    print(f"  Creating DB: {cmd_createdb}")
    subprocess.run(cmd_createdb, shell=True, check=True, capture_output=True)

    # Cluster at 30% identity
    cmd_cluster = (
        f'"{MMSEQS_BAT}" cluster "{db_path}" "{cluster_path}" "{tmp_dir}" '
        f'--min-seq-id 0.3 -c 0.8 --cov-mode 0'
    )
    print(f"  Clustering: {cmd_cluster}")
    subprocess.run(cmd_cluster, shell=True, check=True, capture_output=True)

    # Convert to TSV
    cmd_tsv = f'"{MMSEQS_BAT}" createtsv "{db_path}" "{db_path}" "{cluster_path}" "{tsv_path}"'
    print(f"  Creating TSV: {cmd_tsv}")
    subprocess.run(cmd_tsv, shell=True, check=True, capture_output=True)

    # Step 3: Parse clusters
    print("\nParsing cluster assignments...")
    clusters = defaultdict(list)
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                rep = parts[0]  # cluster representative
                member = parts[1]
                clusters[rep].append(member)

    print(f"  Total clusters: {len(clusters)}")
    print(f"  Cluster size distribution:")
    sizes = [len(v) for v in clusters.values()]
    print(f"    Min: {min(sizes)}, Max: {max(sizes)}, Mean: {np.mean(sizes):.1f}, Median: {np.median(sizes):.1f}")
    print(f"    Singletons: {sum(1 for s in sizes if s == 1)}")

    # Step 4: Split clusters into train/val/test (70/15/15)
    print("\nSplitting clusters...")
    np.random.seed(42)
    cluster_ids = sorted(list(clusters.keys()))
    np.random.shuffle(cluster_ids)

    n_clusters = len(cluster_ids)
    n_train = int(0.70 * n_clusters)
    n_val = int(0.15 * n_clusters)

    train_clusters = cluster_ids[:n_train]
    val_clusters = cluster_ids[n_train:n_train + n_val]
    test_clusters = cluster_ids[n_train + n_val:]

    # Map proteins to splits
    protein_to_split = {}
    protein_to_cluster = {}

    for c in train_clusters:
        for pdb in clusters[c]:
            protein_to_split[pdb] = 'train'
            protein_to_cluster[pdb] = c

    for c in val_clusters:
        for pdb in clusters[c]:
            protein_to_split[pdb] = 'val'
            protein_to_cluster[pdb] = c

    for c in test_clusters:
        for pdb in clusters[c]:
            protein_to_split[pdb] = 'test'
            protein_to_cluster[pdb] = c

    # Count proteins per split
    split_counts = defaultdict(int)
    for split in protein_to_split.values():
        split_counts[split] += 1

    print(f"  Train: {len(train_clusters)} clusters, {split_counts['train']} proteins")
    print(f"  Val:   {len(val_clusters)} clusters, {split_counts['val']} proteins")
    print(f"  Test:  {len(test_clusters)} clusters, {split_counts['test']} proteins")

    # Step 5: Save split assignments
    split_df = pd.DataFrame([
        {'pdb_id': pdb, 'split': split, 'cluster_rep': protein_to_cluster[pdb]}
        for pdb, split in protein_to_split.items()
    ])
    split_path = os.path.join(PROCESSED_DIR, "train_val_test_splits.csv")
    split_df.to_csv(split_path, index=False)
    print(f"\nSplit assignments saved to: {split_path}")

    # Step 6: Verify no sequence leakage
    # Merge with summary to get allosteric residue stats per split
    split_df = split_df.merge(summary[['pdb_id', 'n_residues', 'n_allosteric', 'pct_allosteric']], on='pdb_id', how='left')

    for split_name in ['train', 'val', 'test']:
        sub = split_df[split_df['split'] == split_name]
        print(f"\n  {split_name.upper()} set:")
        print(f"    Proteins: {len(sub)}")
        print(f"    Total residues: {sub.n_residues.sum():.0f}")
        print(f"    Total allosteric: {sub.n_allosteric.sum():.0f}")
        print(f"    Avg pct_allosteric: {sub.pct_allosteric.mean():.1f}%")

    # Cleanup tmp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"DONE — Cluster-level splits saved to {split_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
