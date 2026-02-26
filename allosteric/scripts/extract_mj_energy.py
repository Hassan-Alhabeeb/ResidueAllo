"""
Extract Miyazawa-Jernigan (MJ) contact energy features per residue.

MJ Features (2-dim):
  1. mj_energy_sum   — sum of MJ2h pairwise energies for all contacts
  2. mj_energy_mean  — mean MJ2h energy per contact (chemical environment quality)

Contact definition:
  - Side-chain centroid distance <= 6.5 A (GLY uses CA)
  - Sequence separation >= 3 (|i-j| >= 3)

Matrix: MJ2h (eij') from Miyazawa & Jernigan, J Mol Biol 1996, 256:623-644, Table V.
Includes hydration/solvation transfer energy term. Most widely used variant.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from Bio.PDB import PDBParser
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDB_DIR = os.path.join(DATA_DIR, "pdb_files")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "..", "features")
MJ_DIR = os.path.join(FEATURES_DIR, "mj_energy")
os.makedirs(MJ_DIR, exist_ok=True)

MJ_DIM = 2
CONTACT_CUTOFF = 6.5  # Angstroms, side-chain centroid distance
SEQ_SEP = 3           # minimum sequence separation |i-j| >= 3

AA_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
           'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
           'TYR', 'VAL']

NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER', 'TPO': 'THR',
    'PTR': 'TYR', 'CSE': 'CYS',
}

# MJ2h matrix amino acid ordering (from 1996 paper)
MJ_AA_ORDER = ['LEU', 'PHE', 'ILE', 'MET', 'VAL', 'TRP', 'CYS', 'TYR',
               'HIS', 'ALA', 'THR', 'GLY', 'PRO', 'ARG', 'GLN', 'SER',
               'ASN', 'GLU', 'ASP', 'LYS']
MJ_AA_INDEX = {aa: i for i, aa in enumerate(MJ_AA_ORDER)}

# MJ2h matrix (eij', kBT units) — Miyazawa & Jernigan 1996, Table V
MJ2H = np.array([
    [-7.37,-7.28,-7.04,-6.41,-6.48,-6.14,-5.83,-5.67,-4.54,-4.91,-4.34,-4.16,-4.20,-4.03,-4.04,-3.92,-3.74,-3.59,-3.40,-3.37],
    [-7.28,-7.26,-6.84,-6.56,-6.29,-6.16,-5.80,-5.66,-4.77,-4.81,-4.28,-4.13,-4.25,-3.98,-4.10,-4.02,-3.75,-3.56,-3.48,-3.36],
    [-7.04,-6.84,-6.54,-6.02,-6.05,-5.78,-5.50,-5.25,-4.14,-4.58,-4.03,-3.78,-3.76,-3.63,-3.67,-3.52,-3.24,-3.27,-3.17,-3.01],
    [-6.41,-6.56,-6.02,-5.46,-5.32,-5.55,-4.99,-4.91,-3.98,-3.94,-3.51,-3.39,-3.45,-3.12,-3.30,-3.03,-2.95,-2.89,-2.57,-2.48],
    [-6.48,-6.29,-6.05,-5.32,-5.52,-5.18,-4.96,-4.62,-3.58,-4.04,-3.46,-3.38,-3.32,-3.07,-3.07,-3.05,-2.83,-2.67,-2.48,-2.49],
    [-6.14,-6.16,-5.78,-5.55,-5.18,-5.06,-4.95,-4.66,-3.98,-3.82,-3.22,-3.42,-3.73,-3.41,-3.11,-2.99,-3.07,-2.99,-2.84,-2.69],
    [-5.83,-5.80,-5.50,-4.99,-4.96,-4.95,-5.44,-4.16,-3.60,-3.57,-3.11,-3.16,-3.07,-2.57,-2.85,-2.86,-2.59,-2.27,-2.41,-1.95],
    [-5.67,-5.66,-5.25,-4.91,-4.62,-4.66,-4.16,-4.17,-3.52,-3.36,-3.01,-3.01,-3.19,-3.16,-2.97,-2.78,-2.76,-2.79,-2.76,-2.60],
    [-4.54,-4.77,-4.14,-3.98,-3.58,-3.98,-3.60,-3.52,-3.05,-2.41,-2.42,-2.15,-2.25,-2.16,-1.98,-2.11,-2.08,-2.15,-2.32,-1.35],
    [-4.91,-4.81,-4.58,-3.94,-4.04,-3.82,-3.57,-3.36,-2.41,-2.72,-2.32,-2.31,-2.03,-1.83,-1.89,-2.01,-1.84,-1.51,-1.70,-1.31],
    [-4.34,-4.28,-4.03,-3.51,-3.46,-3.22,-3.11,-3.01,-2.42,-2.32,-2.12,-2.08,-1.90,-1.90,-1.90,-1.96,-1.88,-1.74,-1.80,-1.31],
    [-4.16,-4.13,-3.78,-3.39,-3.38,-3.42,-3.16,-3.01,-2.15,-2.31,-2.08,-2.24,-1.87,-1.72,-1.66,-1.82,-1.74,-1.22,-1.59,-1.15],
    [-4.20,-4.25,-3.76,-3.45,-3.32,-3.73,-3.07,-3.19,-2.25,-2.03,-1.90,-1.87,-1.75,-1.70,-1.73,-1.57,-1.53,-1.26,-1.33,-0.97],
    [-4.03,-3.98,-3.63,-3.12,-3.07,-3.41,-2.57,-3.16,-2.16,-1.83,-1.90,-1.72,-1.70,-1.55,-1.80,-1.62,-1.64,-2.27,-2.29,-0.59],
    [-4.04,-4.10,-3.67,-3.30,-3.07,-3.11,-2.85,-2.97,-1.98,-1.89,-1.90,-1.66,-1.73,-1.80,-1.54,-1.49,-1.71,-1.42,-1.46,-1.29],
    [-3.92,-4.02,-3.52,-3.03,-3.05,-2.99,-2.86,-2.78,-2.11,-2.01,-1.96,-1.82,-1.57,-1.62,-1.49,-1.67,-1.58,-1.48,-1.63,-1.05],
    [-3.74,-3.75,-3.24,-2.95,-2.83,-3.07,-2.59,-2.76,-2.08,-1.84,-1.88,-1.74,-1.53,-1.64,-1.71,-1.58,-1.68,-1.51,-1.68,-1.21],
    [-3.59,-3.56,-3.27,-2.89,-2.67,-2.99,-2.27,-2.79,-2.15,-1.51,-1.74,-1.22,-1.26,-2.27,-1.42,-1.48,-1.51,-0.91,-1.02,-1.80],
    [-3.40,-3.48,-3.17,-2.57,-2.48,-2.84,-2.41,-2.76,-2.32,-1.70,-1.80,-1.59,-1.33,-2.29,-1.46,-1.63,-1.68,-1.02,-1.21,-1.68],
    [-3.37,-3.36,-3.01,-2.48,-2.49,-2.69,-1.95,-2.60,-1.35,-1.31,-1.31,-1.15,-0.97,-0.59,-1.29,-1.05,-1.21,-1.80,-1.68,-0.12],
], dtype=np.float32)

# Backbone atom names (excluded from side-chain centroid calculation)
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'OXT'}


def get_sidechain_centroid(residue):
    """Compute side-chain centroid. For GLY, return CA position."""
    resname = residue.get_resname()
    if resname in NONSTANDARD_MAP:
        resname = NONSTANDARD_MAP[resname]

    sc_atoms = []
    ca_coord = None
    for atom in residue:
        name = atom.get_name().strip()
        if name == 'CA':
            ca_coord = atom.get_vector().get_array()
        if name not in BACKBONE_ATOMS and not name.startswith('H'):
            sc_atoms.append(atom.get_vector().get_array())

    if resname == 'GLY' or len(sc_atoms) == 0:
        return ca_coord  # GLY or no side-chain atoms found
    return np.mean(sc_atoms, axis=0)


def extract_mj_features(pdb_path, label_path):
    """Extract per-residue MJ contact energy features."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("prot", pdb_path)
    except Exception:
        return None

    model = structure[0]

    # Collect residues matching our standard pipeline
    residues = []
    resnames = []
    centroids = []
    for chain in model:
        for res in chain:
            resname = res.get_resname()
            if res.id[0] != ' ' and resname not in NONSTANDARD_MAP:
                continue
            if resname in NONSTANDARD_MAP:
                resname = NONSTANDARD_MAP[resname]
            if resname not in AA_LIST:
                continue

            centroid = get_sidechain_centroid(res)
            if centroid is None:
                continue

            residues.append((chain.id, res.id[1], resname))
            resnames.append(resname)
            centroids.append(centroid)

    n_res = len(residues)
    if n_res == 0:
        return None

    centroids = np.array(centroids, dtype=np.float64)

    # Compute pairwise MJ energies
    mj_sum = np.zeros(n_res, dtype=np.float32)
    mj_count = np.zeros(n_res, dtype=np.int32)

    # Map residue names to MJ indices
    mj_indices = np.array([MJ_AA_INDEX.get(aa, -1) for aa in resnames], dtype=np.int32)

    # Precompute distance matrix (vectorized, much faster than per-pair np.linalg.norm)
    dist_matrix = cdist(centroids, centroids)

    # Extract chain IDs and resnums for sequence separation check
    chains = [r[0] for r in residues]
    resnums = [r[1] for r in residues]

    for i in range(n_res):
        if mj_indices[i] < 0:
            continue
        for j in range(i + 1, n_res):
            if mj_indices[j] < 0:
                continue
            # Only enforce sequence separation within the same chain
            if chains[i] == chains[j] and abs(resnums[i] - resnums[j]) < SEQ_SEP:
                continue
            if dist_matrix[i, j] <= CONTACT_CUTOFF:
                energy = MJ2H[mj_indices[i], mj_indices[j]]
                mj_sum[i] += energy
                mj_sum[j] += energy
                mj_count[i] += 1
                mj_count[j] += 1

    # Mean energy per contact (avoid div by zero)
    mj_mean = np.where(mj_count > 0, mj_sum / mj_count, 0.0).astype(np.float32)

    # Stack: [mj_energy_sum, mj_energy_mean]
    features = np.column_stack([mj_sum, mj_mean])

    # Align with labels
    labels_df = pd.read_csv(label_path, dtype={'chain': str})
    feat_lookup = {}
    for i, (chain, resnum, _) in enumerate(residues):
        feat_lookup[(chain, resnum)] = i

    aligned = []
    for _, row in labels_df.iterrows():
        key = (row['chain'], row['resnum'])
        if key in feat_lookup:
            aligned.append(features[feat_lookup[key]])
        else:
            aligned.append(np.zeros(MJ_DIM, dtype=np.float32))

    return np.array(aligned, dtype=np.float32)


def process_single_protein(args):
    """Process one protein — for multiprocessing Pool."""
    pdb_id, pdb_path, label_path, output_path = args

    if os.path.exists(output_path):
        return pdb_id, 'skipped', ''

    if not os.path.exists(pdb_path):
        return pdb_id, 'error', 'no PDB file'

    if not os.path.exists(label_path):
        return pdb_id, 'error', 'no label file'

    try:
        features = extract_mj_features(pdb_path, label_path)
        if features is None:
            return pdb_id, 'error', 'feature extraction failed'

        np.savez_compressed(output_path, features=features)
        return pdb_id, 'success', f'{len(features)} residues'
    except Exception as e:
        return pdb_id, 'error', str(e)


def main():
    print("=" * 60)
    print("MJ Contact Energy Feature Extraction")
    print("=" * 60)
    print(f"  Features ({MJ_DIM}):")
    print(f"    - mj_energy_sum (sum of MJ2h pairwise energies)")
    print(f"    - mj_energy_mean (mean energy per contact)")
    print(f"  Contact cutoff: {CONTACT_CUTOFF} A (side-chain centroid)")
    print(f"  Sequence separation: >= {SEQ_SEP}")
    print(f"  CPU cores available: {cpu_count()}")

    summary_path = os.path.join(PROCESSED_DIR, "dataset_summary.csv")
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        pdb_ids = summary['pdb_id'].tolist()
    else:
        splits_path = os.path.join(PROCESSED_DIR, "train_val_test_splits.csv")
        splits = pd.read_csv(splits_path)
        pdb_ids = splits['pdb_id'].tolist()

    print(f"\n  Mode: Training proteins")
    print(f"  Output: {MJ_DIR}")

    tasks = []
    n_existing = 0
    for pdb_id in pdb_ids:
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
        label_path = os.path.join(PROCESSED_DIR, f"{pdb_id}_labels.csv")
        output_path = os.path.join(MJ_DIR, f"{pdb_id}_mj.npz")
        if os.path.exists(output_path):
            n_existing += 1
        tasks.append((pdb_id, pdb_path, label_path, output_path))

    print(f"  Total tasks: {len(tasks)}")
    if n_existing > 0:
        print(f"  Already done: {n_existing} (will skip)")

    # Use physical cores only; cap at 48
    n_workers = min(max(1, cpu_count() - 2), 48)
    print(f"  Workers: {n_workers}\n")

    start_time = time.time()
    processed = 0
    skipped = 0
    failed = 0
    errors = []

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_single_protein, tasks, chunksize=4):
            pdb_id, status, msg = result
            if status == 'success':
                processed += 1
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    remaining = (len(tasks) - processed - skipped - failed) / max(rate, 0.01)
                    print(f"  Processed {processed} proteins ({elapsed:.0f}s, ~{remaining:.0f}s remaining)")
            elif status == 'skipped':
                skipped += 1
            else:
                failed += 1
                if len(errors) < 10:
                    errors.append(f"{pdb_id}: {msg}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"MJ Contact Energy Extraction Complete ({elapsed:.0f}s)")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output dir: {MJ_DIR}")
    if errors:
        print(f"  First errors:")
        for e in errors:
            print(f"    {e}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
