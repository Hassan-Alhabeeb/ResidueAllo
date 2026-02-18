"""
Download all PDB files needed for AlloBench dataset.
Downloads from RCSB PDB in batches.
"""

import os
import pandas as pd
import urllib.request
import time
import sys

DATA_DIR = r"E:\newyear\research_plan\allosteric\data"
PDB_DIR = os.path.join(DATA_DIR, "pdb_files")
CSV_PATH = os.path.join(DATA_DIR, "raw", "allobench", "AlloBench.csv")

# Load dataset
df = pd.read_csv(CSV_PATH)
pdb_ids = df['allosteric_pdb'].dropna().unique()
print(f"Total unique PDB IDs to download: {len(pdb_ids)}")

# Download PDB files
downloaded = 0
skipped = 0
failed = []

for i, pdb_id in enumerate(pdb_ids):
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4:
        print(f"  Skipping invalid PDB ID: {pdb_id}")
        continue

    output_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")

    # Skip if already downloaded
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        skipped += 1
        continue

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    try:
        urllib.request.urlretrieve(url, output_path)
        downloaded += 1

        if (downloaded) % 50 == 0:
            print(f"  Downloaded {downloaded}/{len(pdb_ids)} (skipped {skipped} existing)")

        # Be nice to RCSB servers
        if downloaded % 10 == 0:
            time.sleep(0.5)

    except Exception as e:
        failed.append((pdb_id, str(e)))

    # Progress
    if (i + 1) % 200 == 0:
        print(f"  Progress: {i+1}/{len(pdb_ids)} checked, {downloaded} downloaded, {skipped} skipped, {len(failed)} failed")

print(f"\n=== DONE ===")
print(f"  Downloaded: {downloaded}")
print(f"  Already existed: {skipped}")
print(f"  Failed: {len(failed)}")

if failed:
    print(f"\n  Failed PDB IDs:")
    for pdb_id, err in failed[:20]:
        print(f"    {pdb_id}: {err}")

    # Save failed list
    with open(os.path.join(DATA_DIR, "failed_pdbs.txt"), 'w') as f:
        for pdb_id, err in failed:
            f.write(f"{pdb_id}\t{err}\n")
