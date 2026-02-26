"""
Modular per-residue feature extraction for allosteric site prediction.

Feature groups (69-dim total):
  1. B-factors (1): raw average per residue (NO per-protein z-score)
  2. SASA (2): absolute and relative
  3. Contact density (4): raw counts at 6/8/10/12 A cutoffs
  4. Secondary structure DSSP (3): Q3 one-hot (H/E/-)
  5. Physicochemical (7): hydrophobicity, charge, MW, aromatic, polar, gly, pro
  6. Position (3): distance to centroid (raw), sequential index, normalized position
  7. Packing geometry (4): HSE up/down (chain-aware), coordination, packing density (all raw)
  8. AA one-hot (20)
  9. Neighborhood composition (20): fraction of each AA within 10A
  Total: 64 features (reduced from 69: DSSP Q3 instead of Q8, no per-protein z-scores)

CRITICAL FIX: All features are saved as RAW values. Normalization happens
ONLY in build_dataset.py via StandardScaler fitted on train split.
This prevents per-protein distribution leakage (primary overfitting cause).

Other fixes from Opus review:
  - Distance matrix computed once and shared
  - Chain-boundary HSE uses fallback direction
  - Pass full Structure to freesasa (structureFromBioPDB needs get_models())
  - DSSP uses Q3 (H/E/-) to match pydssp output
  - Alignment warns on low match rate
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import warnings
import traceback
warnings.filterwarnings('ignore')

try:
    import freesasa
    HAS_FREESASA = True
except ImportError:
    HAS_FREESASA = False
    print("WARNING: freesasa not available. SASA features will be NaN.")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/../ = allosteric/
DATA_DIR = os.path.join(BASE_DIR, "data")
PDB_DIR = os.path.join(DATA_DIR, "pdb_files")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "..", "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

# Standard amino acids
AA_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
           'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
           'TYR', 'VAL']
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# Non-standard -> standard mapping (must match extract_labels.py)
NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER', 'TPO': 'THR',
    'PTR': 'TYR', 'CSE': 'CYS',
}

HYDROPHOBICITY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}
CHARGE = {
    'ALA': 0, 'ARG': 1, 'ASN': 0, 'ASP': -1, 'CYS': 0,
    'GLN': 0, 'GLU': -1, 'GLY': 0, 'HIS': 0, 'ILE': 0,
    'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0,
    'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0, 'VAL': 0
}
MW = {
    'ALA': 89, 'ARG': 174, 'ASN': 132, 'ASP': 133, 'CYS': 121,
    'GLN': 146, 'GLU': 147, 'GLY': 75, 'HIS': 155, 'ILE': 131,
    'LEU': 131, 'LYS': 146, 'MET': 149, 'PHE': 165, 'PRO': 115,
    'SER': 105, 'THR': 119, 'TRP': 204, 'TYR': 181, 'VAL': 117
}
AROMATIC = {'PHE', 'TRP', 'TYR', 'HIS'}
POLAR = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS', 'HIS'}

# DSSP: pydssp outputs only Q3 (H, E, -)
DSSP_CODES = ['H', 'E', '-']
DSSP_TO_IDX = {code: i for i, code in enumerate(DSSP_CODES)}

MAX_ASA = {
    'ALA': 129, 'ARG': 274, 'ASN': 195, 'ASP': 193, 'CYS': 167,
    'GLN': 225, 'GLU': 223, 'GLY': 104, 'HIS': 224, 'ILE': 197,
    'LEU': 201, 'LYS': 236, 'MET': 224, 'PHE': 240, 'PRO': 159,
    'SER': 155, 'THR': 172, 'TRP': 285, 'TYR': 263, 'VAL': 174
}


def get_ca_coords(residues):
    """Get CA coordinates. Residues missing CA are flagged."""
    coords = []
    has_ca = []
    for res in residues:
        if 'CA' in res:
            coords.append(res['CA'].get_vector().get_array())
            has_ca.append(True)
        else:
            # Fallback: use centroid of available atoms
            atoms = list(res.get_atoms())
            if atoms:
                pos = np.mean([a.get_vector().get_array() for a in atoms], axis=0)
                coords.append(pos)
            else:
                coords.append(np.array([99999.0, 99999.0, 99999.0]))
            has_ca.append(False)
    return np.array(coords), has_ca


def compute_dist_matrix(ca_coords):
    """Compute pairwise distance matrix once."""
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def extract_bfactors(residues):
    """Extract RAW B-factors per residue (no per-protein normalization)."""
    bfactors = []
    for res in residues:
        atoms = list(res.get_atoms())
        if atoms:
            avg_b = np.mean([a.get_bfactor() for a in atoms])
        else:
            avg_b = np.nan
        bfactors.append(avg_b)
    return np.array(bfactors).reshape(-1, 1)


def extract_sasa(structure, residues, res_info):
    """Extract SASA using freesasa. Pass full BioPython Structure (has get_models())."""
    n = len(residues)
    if not HAS_FREESASA:
        return np.full((n, 2), np.nan)

    try:
        # structureFromBioPDB expects Bio.PDB.Structure (not Model)
        fs_structure = freesasa.structureFromBioPDB(structure)
        result = freesasa.calc(fs_structure)

        # Map (chain, resnum) -> list of atom indices
        residue_atoms = {}
        for i in range(fs_structure.nAtoms()):
            chain_id = fs_structure.chainLabel(i)
            resnum_str = fs_structure.residueNumber(i)
            m = re.match(r'-?\d+', resnum_str.strip())
            if not m:
                continue
            resnum = int(m.group())
            key = (chain_id, resnum)
            if key not in residue_atoms:
                residue_atoms[key] = []
            residue_atoms[key].append(i)

        sasa_features = []
        for i in range(n):
            chain_id = res_info[i]['chain']
            resnum = res_info[i]['resnum']
            resname = res_info[i]['resname']
            key = (chain_id, resnum)

            abs_sasa = 0.0
            if key in residue_atoms:
                for atom_idx in residue_atoms[key]:
                    abs_sasa += result.atomArea(atom_idx)

            max_asa = MAX_ASA.get(resname, 200)
            rel_sasa = min(abs_sasa / max_asa, 1.0) if max_asa > 0 else 0.0
            sasa_features.append([abs_sasa, rel_sasa])

        return np.array(sasa_features)
    except Exception as e:
        print(f"  WARNING: SASA failed: {e}")
        return np.full((n, 2), np.nan)


def extract_contact_density(dist_matrix, cutoffs=[6.0, 8.0, 10.0, 12.0]):
    """RAW contact counts (no per-protein normalization)."""
    n = dist_matrix.shape[0]
    features = np.zeros((n, len(cutoffs)))
    for j, cutoff in enumerate(cutoffs):
        features[:, j] = (dist_matrix < cutoff).sum(axis=1) - 1  # subtract self
    return features


def extract_dssp(model, pdb_path, residues, res_info):
    """DSSP via pydssp. Process per-chain to handle chain breaks."""
    try:
        import pydssp
        import torch

        dssp_dict = {}

        # Group residues by chain
        chain_residues = {}
        for i, res in enumerate(residues):
            chain_id = res_info[i]['chain']
            if chain_id not in chain_residues:
                chain_residues[chain_id] = []
            chain_residues[chain_id].append((i, res))

        for chain_id, chain_res_list in chain_residues.items():
            coords_list = []
            valid_indices = []

            for idx, res in chain_res_list:
                if 'N' in res and 'CA' in res and 'C' in res and 'O' in res:
                    coords_list.append([
                        res['N'].get_vector().get_array(),
                        res['CA'].get_vector().get_array(),
                        res['C'].get_vector().get_array(),
                        res['O'].get_vector().get_array(),
                    ])
                    valid_indices.append(idx)

            if len(coords_list) < 3:
                continue

            try:
                coords = torch.tensor(np.array(coords_list), dtype=torch.float32)
                ss_arr = pydssp.assign(coords)

                for j, idx in enumerate(valid_indices):
                    if j < len(ss_arr):
                        resnum = res_info[idx]['resnum']
                        dssp_dict[(chain_id, resnum)] = str(ss_arr[j])
            except Exception:
                # Skip this chain if pydssp fails (e.g. tensor size mismatch)
                continue

        return dssp_dict if dssp_dict else None
    except Exception as e:
        print(f"  WARNING: DSSP failed: {e}")
        return None


def dssp_to_onehot(dssp_dict, res_info):
    """Convert DSSP to Q3 one-hot: H, E, - (3-dim)."""
    n = len(res_info)
    features = np.zeros((n, 3))

    if dssp_dict is None:
        features[:, 2] = 1.0  # default to coil
        return features

    for i, info in enumerate(res_info):
        key = (info['chain'], info['resnum'])
        ss = dssp_dict.get(key, '-')
        idx = DSSP_TO_IDX.get(ss, 2)  # default to '-'
        features[i, idx] = 1.0

    return features


def extract_physicochemical(res_info):
    """Physicochemical properties (global fixed normalization)."""
    n = len(res_info)
    features = np.zeros((n, 7))
    for i, info in enumerate(res_info):
        resname = info['resname']
        # Hydrophobicity normalized to [0,1] using known Kyte-Doolittle range
        features[i, 0] = (HYDROPHOBICITY.get(resname, 0.0) + 4.5) / 9.0
        features[i, 1] = CHARGE.get(resname, 0.0)
        features[i, 2] = MW.get(resname, 130) / 204.0
        features[i, 3] = 1.0 if resname in AROMATIC else 0.0
        features[i, 4] = 1.0 if resname in POLAR else 0.0
        features[i, 5] = 1.0 if resname == 'GLY' else 0.0
        features[i, 6] = 1.0 if resname == 'PRO' else 0.0
    return features


def extract_position_features(ca_coords, res_info):
    """Position features — RAW values (no per-protein z-score).
    Sequential index and normalized position are per-chain to avoid
    cross-chain positional leakage in multi-chain complexes."""
    n = len(ca_coords)
    features = np.zeros((n, 3))

    # Raw distance to centroid (Angstroms)
    centroid = ca_coords.mean(axis=0)
    features[:, 0] = np.sqrt(np.sum((ca_coords - centroid)**2, axis=1))

    # Per-chain sequential index and normalized position
    chain_lengths = {}
    for info in res_info:
        chain_lengths[info['chain']] = chain_lengths.get(info['chain'], 0) + 1

    current_chain = None
    chain_idx = 0

    for i in range(n):
        chain = res_info[i]['chain']
        if chain != current_chain:
            current_chain = chain
            chain_idx = 0

        features[i, 1] = chain_idx
        features[i, 2] = chain_idx / max(chain_lengths[chain] - 1, 1)
        chain_idx += 1

    return features


def extract_packing_geometry(ca_coords, dist_matrix, res_info):
    """Packing geometry with chain-aware HSE (no per-protein normalization)."""
    n = len(ca_coords)
    features = np.zeros((n, 4))

    # Build chain membership for boundary detection
    chain_ids = [info['chain'] for info in res_info]

    for i in range(n):
        neighbors_mask = (dist_matrix[i] < 13.0) & (dist_matrix[i] > 0)
        neighbor_indices = np.where(neighbors_mask)[0]

        if len(neighbor_indices) > 0:
            directions = ca_coords[neighbor_indices] - ca_coords[i]

            # Chain-aware direction: only use adjacent residues in SAME chain
            chain_dir = np.array([0.0, 0.0, 1.0])  # fallback for single-residue chains
            if i > 0 and i < n - 1 and chain_ids[i-1] == chain_ids[i] == chain_ids[i+1]:
                # Interior residue: use i-1 to i+1 direction
                chain_dir = ca_coords[i+1] - ca_coords[i-1]
            elif i < n - 1 and chain_ids[i] == chain_ids[i+1]:
                # N-terminus: use i to i+1 direction
                chain_dir = ca_coords[i+1] - ca_coords[i]
            elif i > 0 and chain_ids[i-1] == chain_ids[i]:
                # C-terminus: use i-1 to i direction
                chain_dir = ca_coords[i] - ca_coords[i-1]
            norm = np.linalg.norm(chain_dir)
            if norm > 1e-8:
                chain_dir = chain_dir / norm
            else:
                chain_dir = np.array([0.0, 0.0, 1.0])

            dots = np.dot(directions, chain_dir)
            features[i, 0] = np.sum(dots > 0)   # HSE up (raw count)
            features[i, 1] = np.sum(dots <= 0)  # HSE down (raw count)

        # Coordination number (raw count within 8A)
        features[i, 2] = np.sum((dist_matrix[i] < 8.0) & (dist_matrix[i] > 0))

        # Packing density (inverse average distance to 12 nearest)
        sorted_dists = np.sort(dist_matrix[i])
        k = min(12, len(sorted_dists) - 1)
        if k > 0:
            features[i, 3] = 1.0 / (np.mean(sorted_dists[1:k+1]) + 1e-8)

    return features


def extract_aa_onehot(res_info):
    """One-hot encoding of amino acid type (20-dim)."""
    n = len(res_info)
    features = np.zeros((n, 20))
    for i, info in enumerate(res_info):
        idx = AA_TO_IDX.get(info['resname'], -1)
        if idx >= 0:
            features[i, idx] = 1.0
    return features


def extract_neighborhood_composition(dist_matrix, res_info, cutoff=10.0):
    """Fraction of each AA type within cutoff distance."""
    n = dist_matrix.shape[0]
    features = np.zeros((n, 20))

    for i in range(n):
        neighbors = np.where((dist_matrix[i] < cutoff) & (dist_matrix[i] > 0))[0]
        if len(neighbors) > 0:
            for j in neighbors:
                idx = AA_TO_IDX.get(res_info[j]['resname'], -1)
                if idx >= 0:
                    features[i, idx] += 1
            features[i] /= len(neighbors)

    return features


# ============================================================
# Main extraction pipeline
# ============================================================

def extract_all_features(pdb_id, pdb_path):
    """Extract all features for a single protein."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception:
        return None, None

    model = structure[0]

    residues = []
    res_info = []
    for chain in model:
        for residue in chain:
            resname = residue.get_resname()
            if residue.id[0] != ' ' and resname not in NONSTANDARD_MAP:
                continue

            # Map non-standard to standard
            if resname in NONSTANDARD_MAP:
                resname = NONSTANDARD_MAP[resname]

            if resname not in AA_TO_IDX:
                continue

            residues.append(residue)
            res_info.append({
                'chain': chain.id,
                'resnum': residue.id[1],
                'resname': resname
            })

    if len(residues) == 0:
        return None, None

    # Compute shared data structures ONCE
    ca_coords, _ = get_ca_coords(residues)
    dist_matrix = compute_dist_matrix(ca_coords)

    # Extract each feature group — ALL RAW VALUES
    feat_bfactor = extract_bfactors(residues)                              # (N, 1)
    feat_sasa = extract_sasa(structure, residues, res_info)                 # (N, 2)
    feat_contact = extract_contact_density(dist_matrix)                    # (N, 4)
    dssp_dict = extract_dssp(model, pdb_path, residues, res_info)
    feat_dssp = dssp_to_onehot(dssp_dict, res_info)                       # (N, 3) Q3
    feat_physchem = extract_physicochemical(res_info)                       # (N, 7)
    feat_position = extract_position_features(ca_coords, res_info)         # (N, 3)
    feat_packing = extract_packing_geometry(ca_coords, dist_matrix, res_info)  # (N, 4)
    feat_aa = extract_aa_onehot(res_info)                                  # (N, 20)
    feat_neighbor = extract_neighborhood_composition(dist_matrix, res_info) # (N, 20)

    all_features = np.concatenate([
        feat_bfactor,     # 1
        feat_sasa,        # 2
        feat_contact,     # 4
        feat_dssp,        # 3  (Q3 not Q8)
        feat_physchem,    # 7
        feat_position,    # 3
        feat_packing,     # 4
        feat_aa,          # 20
        feat_neighbor,    # 20
    ], axis=1)  # Total: 64

    # Replace NaN with 0 for safety (NaN from failed SASA/bfactor)
    all_features = np.nan_to_num(all_features, nan=0.0)

    return all_features, res_info


# Feature names for reference
FEATURE_NAMES = (
    ['bfactor'] +
    ['sasa_abs', 'sasa_rel'] +
    ['contacts_6A', 'contacts_8A', 'contacts_10A', 'contacts_12A'] +
    [f'dssp_{c}' for c in DSSP_CODES] +
    ['hydrophobicity', 'charge', 'mol_weight_norm', 'is_aromatic', 'is_polar', 'is_glycine', 'is_proline'] +
    ['dist_to_centroid', 'seqpos', 'seqpos_normalized'] +
    ['hse_up', 'hse_down', 'coordination_num', 'packing_density'] +
    [f'aa_{aa}' for aa in AA_LIST] +
    [f'neighbor_{aa}' for aa in AA_LIST]
)


def process_single_protein(args):
    """Process a single protein — designed for multiprocessing Pool."""
    pdb_id, pdb_path, label_path, feat_path = args

    if os.path.exists(feat_path):
        return pdb_id, 'skipped', None

    if not os.path.exists(pdb_path):
        return pdb_id, 'failed', 'no PDB file'

    if not os.path.exists(label_path):
        return pdb_id, 'failed', 'no label file'

    try:
        features, res_info = extract_all_features(pdb_id, pdb_path)
        if features is None:
            return pdb_id, 'failed', 'no residues'

        labels_df = pd.read_csv(label_path, dtype={'chain': str})

        # Align features with labels by (chain, resnum)
        feat_lookup = {}
        for i, info in enumerate(res_info):
            key = (info['chain'], info['resnum'])
            feat_lookup[key] = i

        aligned_features = []
        aligned_labels = []
        n_matched = 0
        for _, lrow in labels_df.iterrows():
            key = (lrow['chain'], lrow['resnum'])
            if key in feat_lookup:
                aligned_features.append(features[feat_lookup[key]])
                n_matched += 1
            else:
                aligned_features.append(np.zeros(len(FEATURE_NAMES), dtype=np.float32))
            aligned_labels.append(lrow['is_allosteric'])

        if len(aligned_features) == 0:
            return pdb_id, 'failed', 'no alignment'

        # Warn on low match rate
        match_rate = n_matched / len(labels_df)
        if match_rate < 0.90:
            print(f"  WARNING: {pdb_id} matched only {match_rate:.1%} of label residues (rest zero-padded)")

        aligned_features = np.array(aligned_features)
        aligned_labels = np.array(aligned_labels)

        np.savez_compressed(feat_path,
                            features=aligned_features,
                            labels=aligned_labels,
                            pdb_id=pdb_id,
                            feature_names=FEATURE_NAMES)

        return pdb_id, 'success', f'{len(aligned_labels)} res'

    except Exception as e:
        return pdb_id, 'failed', str(e)


def main():
    import time
    from multiprocessing import Pool, cpu_count

    summary = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_summary.csv"))
    print(f"Proteins to process: {len(summary)}")
    print(f"Feature dimensions: {len(FEATURE_NAMES)}")

    tasks = []
    for _, row in summary.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
        label_path = os.path.join(PROCESSED_DIR, f"{pdb_id}_labels.csv")
        feat_path = os.path.join(FEATURES_DIR, f"{pdb_id}_features.npz")
        tasks.append((pdb_id, pdb_path, label_path, feat_path))

    n_existing = sum(1 for _, _, _, fp in tasks if os.path.exists(fp))
    if n_existing > 0:
        print(f"  Already done: {n_existing} (will skip)")

    # Use physical cores only; cap at 48
    n_workers = min(max(1, cpu_count() - 2), 48)
    print(f"  Workers: {n_workers}")
    print(f"  Total tasks: {len(tasks)}\n")

    start_time = time.time()
    newly_processed = 0
    already_done = 0
    failed = 0
    errors = []

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_single_protein, tasks, chunksize=4):
            pdb_id, status, msg = result
            if status == 'success':
                newly_processed += 1
                if newly_processed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = newly_processed / elapsed
                    remaining = (len(tasks) - newly_processed - already_done - failed) / max(rate, 0.01)
                    print(f"  Processed {newly_processed} proteins ({elapsed:.0f}s, ~{remaining:.0f}s remaining)")
            elif status == 'skipped':
                already_done += 1
            else:
                failed += 1
                if len(errors) < 5:
                    errors.append(f"{pdb_id}: {msg}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Feature extraction complete ({elapsed:.0f}s)")
    print(f"  Newly processed: {newly_processed}")
    print(f"  Already done: {already_done}")
    print(f"  Failed: {failed}")
    print(f"  Feature dim: {len(FEATURE_NAMES)}")
    print(f"  Output dir: {FEATURES_DIR}")
    if errors:
        print(f"  First errors:")
        for e in errors:
            print(f"    {e}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
