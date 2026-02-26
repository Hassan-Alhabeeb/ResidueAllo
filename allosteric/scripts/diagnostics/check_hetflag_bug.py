"""
Check how many PDB files have non-standard residues (MSE, SEP, TPO, etc.)
that are het-flagged and would be silently dropped by extract_labels.py.

This verifies whether the hetflag filter bug is actually impacting our dataset.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDB_DIR = os.path.join(BASE_DIR, "data", "pdb_files")

NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER',
    'TPO': 'THR', 'PTR': 'TYR', 'CSE': 'CYS',
}


def check_single_pdb(pdb_path):
    """Check one PDB file for het-flagged non-standard residues."""
    pdb_id = os.path.basename(pdb_path).replace('.pdb', '')
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        model = structure[0]

        missed = []
        kept = []
        for chain in model:
            for res in chain:
                hetflag = res.id[0]
                resname = res.get_resname()
                if resname in NONSTANDARD_MAP:
                    if hetflag != ' ':
                        missed.append((chain.id, res.id[1], resname, hetflag))
                    else:
                        kept.append((chain.id, res.id[1], resname))

        return pdb_id, missed, kept
    except Exception as e:
        return pdb_id, None, None


def main():
    pdb_files = sorted([
        os.path.join(PDB_DIR, f) for f in os.listdir(PDB_DIR)
        if f.endswith('.pdb')
    ])
    print(f"Scanning {len(pdb_files)} PDB files...")
    print(f"Looking for non-standard residues: {list(NONSTANDARD_MAP.keys())}")
    print()

    n_workers = max(1, os.cpu_count() - 2)
    print(f"Workers: {n_workers}")

    affected = []
    has_kept = []
    errors = 0
    total_missed = 0
    total_kept = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(check_single_pdb, p): p for p in pdb_files}
        done = 0
        for future in as_completed(futures):
            done += 1
            pdb_id, missed, kept = future.result()

            if missed is None:
                errors += 1
                continue

            if missed:
                affected.append((pdb_id, missed))
                total_missed += len(missed)
            if kept:
                has_kept.append((pdb_id, kept))
                total_kept += len(kept)

            if done % 500 == 0:
                print(f"  [{done}/{len(pdb_files)}] affected so far: {len(affected)}")

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Total PDBs scanned:    {len(pdb_files)}")
    print(f"  Parse errors:          {errors}")
    print(f"  PDBs with DROPPED non-standard residues (het-flagged): {len(affected)}")
    print(f"  Total residues DROPPED: {total_missed}")
    print(f"  PDBs with KEPT non-standard residues (blank hetflag):  {len(has_kept)}")
    print(f"  Total residues KEPT:    {total_kept}")

    if affected:
        print(f"\n--- Affected proteins (residues being silently dropped) ---")
        for pdb_id, missed in sorted(affected):
            residue_summary = {}
            for chain, resnum, resname, hetflag in missed:
                residue_summary[resname] = residue_summary.get(resname, 0) + 1
            summary_str = ", ".join(f"{count}x {name}" for name, count in residue_summary.items())
            print(f"  {pdb_id}: {len(missed)} residues dropped ({summary_str})")

    if has_kept:
        print(f"\n--- Proteins where NONSTANDARD_MAP is working (blank hetflag) ---")
        for pdb_id, kept in sorted(has_kept)[:10]:
            residue_summary = {}
            for chain, resnum, resname in kept:
                residue_summary[resname] = residue_summary.get(resname, 0) + 1
            summary_str = ", ".join(f"{count}x {name}" for name, count in residue_summary.items())
            print(f"  {pdb_id}: {len(kept)} residues mapped ({summary_str})")
        if len(has_kept) > 10:
            print(f"  ... and {len(has_kept) - 10} more")

    print(f"\n{'='*60}")
    if total_missed > 0:
        print(f"VERDICT: Bug is REAL. {total_missed} residues across {len(affected)} proteins are being dropped.")
        print(f"FIX: Change the hetflag filter in extract_labels.py (and extract_features.py, extract_esm2.py)")
    else:
        print(f"VERDICT: Bug exists in theory but does NOT affect your current dataset.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
