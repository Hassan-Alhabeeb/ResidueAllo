"""
Extract per-residue binary labels from AlloBench CSV.

For each protein:
  - Parse allosteric_site_residue column (e.g., ['A-PHE-157', 'A-TYR-156', ...])
  - Label those residues as 1 (allosteric), all others as 0
  - Also extract active_site_residue as a separate label column
  - Save as a clean per-residue CSV

Output: One CSV per protein with columns:
  chain, resnum, resname, is_allosteric, is_active_site

FIXES from Opus review:
  - Normalize PDB IDs before groupby (case sensitivity fix)
  - Handle insertion codes in residue numbers (int crash fix)
  - Filter to standard AA only (match extract_features.py residue set)
  - Add __name__ guard
  - Division-by-zero guard on final summary
"""

import os
import ast
import re
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r"E:\newyear\research_plan\allosteric\data"
PDB_DIR = os.path.join(DATA_DIR, "pdb_files")
CSV_PATH = os.path.join(DATA_DIR, "raw", "allobench", "AlloBench.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")

# Standard amino acids — MUST match extract_features.py AA_LIST
AA_LIST = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
           'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
           'TYR', 'VAL'}

# Non-standard residue -> standard equivalent (common in X-ray structures)
NONSTANDARD_MAP = {
    'MSE': 'MET',  # selenomethionine
    'HYP': 'PRO',  # hydroxyproline
    'SEP': 'SER',  # phosphoserine
    'TPO': 'THR',  # phosphothreonine
    'PTR': 'TYR',  # phosphotyrosine
    'CSE': 'CYS',  # selenocysteine
}


def parse_residue_list(residue_str):
    """
    Parse AlloBench residue format: ['A-PHE-157', 'B-ILE-12', ...]
    Returns set of (chain, resnum) tuples.
    Handles insertion codes by stripping non-digit characters.
    """
    if pd.isna(residue_str) or residue_str == '[]':
        return set()

    try:
        residues = ast.literal_eval(residue_str)
    except (ValueError, SyntaxError):
        residues = re.findall(r"'([^']+)'", residue_str)

    parsed = set()
    for r in residues:
        parts = r.strip().split('-')
        if len(parts) >= 3:
            chain = parts[0]
            resnum_str = parts[-1]
            try:
                # Strip insertion codes: '157A' -> 157
                resnum = int(re.match(r'-?\d+', resnum_str).group())
                parsed.add((chain, resnum))
            except (ValueError, AttributeError):
                pass
        elif len(parts) == 2:
            try:
                chain = parts[0]
                resnum = int(re.match(r'-?\d+', parts[1]).group())
                parsed.add((chain, resnum))
            except (ValueError, AttributeError):
                pass
    return parsed


def parse_active_site_residues(active_str):
    """
    Parse active_site_residue column: [63, 94, 99, 166, ...] (just residue numbers)
    Returns set of residue numbers.
    NOTE: AlloBench active_site_residue has no chain info — this is a data limitation.
    """
    if pd.isna(active_str) or active_str == '[]':
        return set()

    try:
        nums = ast.literal_eval(active_str)
        return set(int(n) for n in nums)
    except (ValueError, SyntaxError):
        nums = re.findall(r'\d+', active_str)
        return set(int(n) for n in nums)


def extract_residues_from_pdb(pdb_path):
    """
    Extract standard amino acid residues from a PDB file (first model only).
    Non-standard residues (MSE, etc.) are mapped to their standard equivalents.
    Returns list of dicts: [{chain, resnum, resname}, ...]
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("prot", pdb_path)
    except Exception:
        return None

    model = structure[0]  # First model only (cleaner than for/break)
    residues = []
    for chain in model:
        for residue in chain:
            hetflag = residue.id[0]
            if hetflag != ' ':
                continue
            resnum = residue.id[1]
            resname = residue.get_resname()

            # Map non-standard to standard
            if resname in NONSTANDARD_MAP:
                resname = NONSTANDARD_MAP[resname]

            # Only keep standard amino acids
            if resname not in AA_LIST:
                continue

            chain_id = chain.id
            residues.append({
                'chain': chain_id,
                'resnum': resnum,
                'resname': resname
            })

    return residues


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading AlloBench CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Total entries: {len(df)}")

    # Normalize PDB IDs BEFORE groupby to avoid case-sensitivity issues
    df['allosteric_pdb'] = df['allosteric_pdb'].str.strip().str.upper()

    # Group by PDB structure
    grouped = df.groupby('allosteric_pdb')
    print(f"  Unique PDB structures: {len(grouped)}")

    processed = 0
    skipped_no_pdb = 0
    skipped_parse_error = 0
    total_residues = 0
    total_allosteric = 0
    total_active = 0

    all_protein_summaries = []

    for pdb_id, group in grouped:
        pdb_id = pdb_id.strip().upper()
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")

        if not os.path.exists(pdb_path):
            skipped_no_pdb += 1
            continue

        residue_list = extract_residues_from_pdb(pdb_path)
        if residue_list is None or len(residue_list) == 0:
            skipped_parse_error += 1
            continue

        # Collect ALL allosteric residues across all entries for this PDB
        all_allosteric = set()
        all_active = set()

        for _, row in group.iterrows():
            allo_res = parse_residue_list(row['allosteric_site_residue'])
            all_allosteric.update(allo_res)

            active_res = parse_active_site_residues(row['active_site_residue'])
            all_active.update(active_res)

        # Create per-residue labels
        records = []
        for res in residue_list:
            chain = res['chain']
            resnum = res['resnum']
            resname = res['resname']

            is_allo = 1 if (chain, resnum) in all_allosteric else 0
            # Active site: no chain info in AlloBench, match by resnum only (documented limitation)
            is_active = 1 if resnum in all_active else 0

            records.append({
                'chain': chain,
                'resnum': resnum,
                'resname': resname,
                'is_allosteric': is_allo,
                'is_active_site': is_active
            })

        res_df = pd.DataFrame(records)

        output_path = os.path.join(OUTPUT_DIR, f"{pdb_id}_labels.csv")
        res_df.to_csv(output_path, index=False)

        n_allo = res_df['is_allosteric'].sum()
        n_active = res_df['is_active_site'].sum()
        n_total = len(res_df)

        total_residues += n_total
        total_allosteric += n_allo
        total_active += n_active

        all_protein_summaries.append({
            'pdb_id': pdb_id,
            'n_residues': n_total,
            'n_allosteric': n_allo,
            'n_active_site': n_active,
            'pct_allosteric': round(100 * n_allo / n_total, 1) if n_total > 0 else 0,
            'sequence': group.iloc[0]['sequence'] if 'sequence' in group.columns else '',
            'organism': group.iloc[0]['organism'] if 'organism' in group.columns else '',
            'target_gene': group.iloc[0]['target_gene'] if 'target_gene' in group.columns else '',
        })

        processed += 1

        if processed % 100 == 0:
            print(f"  Processed {processed} structures...")

    # Save summary
    summary_df = pd.DataFrame(all_protein_summaries)
    summary_path = os.path.join(DATA_DIR, "processed", "dataset_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Processed:          {processed}")
    print(f"  Skipped (no PDB):   {skipped_no_pdb}")
    print(f"  Skipped (parse err):{skipped_parse_error}")
    print(f"  Total residues:     {total_residues}")
    if total_residues > 0:
        print(f"  Total allosteric:   {total_allosteric} ({100*total_allosteric/total_residues:.1f}%)")
        print(f"  Total active site:  {total_active} ({100*total_active/total_residues:.1f}%)")
    else:
        print(f"  Total allosteric:   {total_allosteric}")
        print(f"  Total active site:  {total_active}")
    print(f"  Summary saved to:   {summary_path}")


if __name__ == '__main__':
    main()
