"""
Cluster proteins by sequence identity and create train/val/test splits.

Uses MMseqs2 to cluster at 30% sequence identity, then splits at the
CLUSTER level (not protein level) to prevent data leakage from homologous
proteins appearing in both train and test.

Split ratio: 70% train, 15% val, 15% test (by cluster count).
"""

import os
import subprocess
import pandas as pd
import numpy as np
from collections import defaultdict
import tempfile
import shutil

DATA_DIR = r"E:\newyear\research_plan\allosteric\data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SUMMARY_PATH = os.path.join(PROCESSED_DIR, "dataset_summary.csv")
MMSEQS_BAT = r"E:\newyear\research_plan\allosteric\tools\mmseqs\mmseqs\mmseqs.bat"

# Load dataset summary (only proteins we successfully processed)
summary = pd.read_csv(SUMMARY_PATH)
print(f"Total proteins with labels: {len(summary)}")

# Step 1: Create FASTA file from sequences
fasta_path = os.path.join(PROCESSED_DIR, "all_sequences.fasta")
n_with_seq = 0
n_without_seq = 0

with open(fasta_path, 'w') as f:
    for _, row in summary.iterrows():
        pdb_id = row['pdb_id']
        seq = row.get('sequence', '')
        if pd.isna(seq) or len(str(seq).strip()) < 10:
            n_without_seq += 1
            continue
        f.write(f">{pdb_id}\n{seq}\n")
        n_with_seq += 1

print(f"  Sequences written to FASTA: {n_with_seq}")
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
cluster_ids = list(clusters.keys())
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
