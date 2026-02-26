"""
Extract Local Frustration features per residue.

Based on the protein frustration principle (Ferreiro et al., PNAS 2007, 2011).
Uses frustrapy if available; falls back to direct MJ2H-based computation.

Features (7-dim with frustrapy, 4-dim with fallback):
  Configurational (4):
    1. frust_config_high_density    — Fraction of highly frustrated contacts
    2. frust_config_neutral_density — Fraction of neutral contacts
    3. frust_config_minimal_density — Fraction of minimally frustrated contacts
    4. frust_config_mean_index      — Mean frustration index across contacts
  Mutational (3, frustrapy only):
    5. frust_mut_high_density       — Fraction of highly frustrated contacts
    6. frust_mut_neutral_density    — Fraction of neutral contacts
    7. frust_mut_minimal_density    — Fraction of minimally frustrated contacts

Frustration index per contact:
    FI = (<E_decoy> - E_native) / std(E_decoy)
Classifications (Ferreiro 2007):
    highly frustrated: FI < -1
    minimally frustrated: FI > 0.78
    neutral: otherwise

Usage:
    python extract_local_frustration.py                       # Training (config only)
    python extract_local_frustration.py --mutational          # Training (config + mutational)
    python extract_local_frustration.py --casbench            # CASBench (config only)
    python extract_local_frustration.py --casbench --mutational  # CASBench (config + mut)
    python extract_local_frustration.py --workers 24          # Limit workers
"""

import os
import gc
import time
import shutil
import tempfile
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from Bio.PDB import PDBParser
from multiprocessing import Pool, cpu_count
import warnings
import glob as globmod

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
FRUST_DIR = os.path.join(FEATURES_DIR, "frustration")
os.makedirs(FRUST_DIR, exist_ok=True)

# CASBench paths
CASBENCH_DIR = os.path.join(DATA_DIR, "casbench")
CASBENCH_LABELS_DIR = os.path.join(CASBENCH_DIR, "labels")
CASBENCH_FEATURES_DIR = os.path.join(CASBENCH_DIR, "features")

# ── Constants ─────────────────────────────────────────────────────────────────
FRUST_DIM_CONFIG = 4
FRUST_DIM_FULL = 7
FRUST_DIM = 7  # Maximum possible (config + mutational)

FRUST_FEATURE_NAMES_CONFIG = [
    'frust_config_high_density',
    'frust_config_neutral_density',
    'frust_config_minimal_density',
    'frust_config_mean_index',
]
FRUST_FEATURE_NAMES_MUT = [
    'frust_mut_high_density',
    'frust_mut_neutral_density',
    'frust_mut_minimal_density',
]
FRUST_FEATURE_NAMES = FRUST_FEATURE_NAMES_CONFIG + FRUST_FEATURE_NAMES_MUT

# Contact parameters (same as extract_mj_energy.py)
CONTACT_CUTOFF = 6.5  # Angstroms, side-chain centroid distance
SEQ_SEP = 3           # minimum sequence separation |i-j| >= 3

# Frustration classification thresholds (Ferreiro et al., PNAS 2007)
FI_HIGHLY_FRUSTRATED = -1.0   # FI < this → highly frustrated
FI_MINIMALLY_FRUSTRATED = 0.78  # FI > this → minimally frustrated

AA_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
           'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
           'TYR', 'VAL']

NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER', 'TPO': 'THR',
    'PTR': 'TYR', 'CSE': 'CYS',
}

BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'OXT'}

# MJ2h matrix (Miyazawa & Jernigan 1996, Table V) — same as extract_mj_energy.py
MJ_AA_ORDER = ['LEU', 'PHE', 'ILE', 'MET', 'VAL', 'TRP', 'CYS', 'TYR',
               'HIS', 'ALA', 'THR', 'GLY', 'PRO', 'ARG', 'GLN', 'SER',
               'ASN', 'GLU', 'ASP', 'LYS']
MJ_AA_INDEX = {aa: i for i, aa in enumerate(MJ_AA_ORDER)}

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

# Precompute background distribution for MJ-based frustration
MJ2H_FLAT = MJ2H.flatten()
MJ2H_MEAN = float(MJ2H_FLAT.mean())
MJ2H_STD = float(MJ2H_FLAT.std())

# ── Check for frustrapy ──────────────────────────────────────────────────────
try:
    import frustrapy
    HAS_FRUSTRAPY = True
except ImportError:
    HAS_FRUSTRAPY = False


# ── Shared PDB parsing ────────────────────────────────────────────────────────

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
        return ca_coord
    return np.mean(sc_atoms, axis=0)


def parse_residues_and_contacts(pdb_path):
    """Parse PDB, compute contacts. Returns residues list and contact pairs with FI.

    Returns:
        residues: list of (chain, resnum, resname) tuples
        contacts: list of (i, j, frustration_index) tuples
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = structure[0]

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
        return [], []

    centroids_arr = np.array(centroids, dtype=np.float64)
    mj_indices = np.array([MJ_AA_INDEX.get(aa, -1) for aa in resnames], dtype=np.int32)

    # Compute pairwise distances
    dist_matrix = cdist(centroids_arr, centroids_arr)

    chains = [r[0] for r in residues]
    resnums = [r[1] for r in residues]

    contacts = []
    for i in range(n_res):
        if mj_indices[i] < 0:
            continue
        for j in range(i + 1, n_res):
            if mj_indices[j] < 0:
                continue
            if chains[i] == chains[j] and abs(resnums[i] - resnums[j]) < SEQ_SEP:
                continue
            if dist_matrix[i, j] <= CONTACT_CUTOFF:
                # Compute frustration index
                e_native = MJ2H[mj_indices[i], mj_indices[j]]
                fi = (MJ2H_MEAN - e_native) / MJ2H_STD
                contacts.append((i, j, float(fi)))

    return residues, contacts


def aggregate_contacts_to_residues(n_res, contacts, mode='config'):
    """Aggregate per-contact frustration to per-residue features.

    Args:
        n_res: number of residues
        contacts: list of (i, j, frustration_index) tuples
        mode: 'config' for configurational (4-dim), 'mut' for mutational (3-dim)

    Returns:
        features: (n_res, 4) for config or (n_res, 3) for mut
    """
    if mode == 'config':
        out_dim = FRUST_DIM_CONFIG
    else:
        out_dim = 3

    # Per-residue accumulators
    fi_sums = np.zeros(n_res, dtype=np.float64)
    fi_counts = np.zeros(n_res, dtype=np.int32)
    n_high = np.zeros(n_res, dtype=np.int32)
    n_neutral = np.zeros(n_res, dtype=np.int32)
    n_minimal = np.zeros(n_res, dtype=np.int32)

    for i, j, fi in contacts:
        # Classify
        if fi < FI_HIGHLY_FRUSTRATED:
            n_high[i] += 1
            n_high[j] += 1
        elif fi > FI_MINIMALLY_FRUSTRATED:
            n_minimal[i] += 1
            n_minimal[j] += 1
        else:
            n_neutral[i] += 1
            n_neutral[j] += 1

        fi_sums[i] += fi
        fi_sums[j] += fi
        fi_counts[i] += 1
        fi_counts[j] += 1

    total = fi_counts.astype(np.float32)
    total_safe = np.where(total > 0, total, 1.0)

    high_density = n_high.astype(np.float32) / total_safe
    neutral_density = n_neutral.astype(np.float32) / total_safe
    minimal_density = n_minimal.astype(np.float32) / total_safe
    mean_fi = np.where(fi_counts > 0, fi_sums / fi_counts, 0.0).astype(np.float32)

    # Zero out residues with no contacts
    no_contacts = (fi_counts == 0)
    high_density[no_contacts] = 0.0
    neutral_density[no_contacts] = 0.0
    minimal_density[no_contacts] = 0.0

    if mode == 'config':
        return np.column_stack([high_density, neutral_density, minimal_density, mean_fi])
    else:
        return np.column_stack([high_density, neutral_density, minimal_density])


# ── Frustrapy-based extraction ────────────────────────────────────────────────


def _densities_to_features(densities, key_to_idx, n_res, include_mean_fi=True):
    """Convert frustrapy FrustrationDensity list to per-residue feature array.

    Args:
        densities: list of FrustrationDensity objects (residue_number, chain_id, rel_*)
        key_to_idx: dict mapping (chain, resnum) -> label index
        n_res: total residues in labels
        include_mean_fi: if True, output 4-dim (with mean FI); else 3-dim

    Note: mean_FI is approximated as a weighted sum of density fractions
    using category centroids (-2.0, 0.0, 1.5). This is a proxy because
    frustrapy's per-contact FI values use spatial proximity assignment
    (5A sphere around CA atoms) that can't be trivially inverted to
    per-residue means from the density results alone.

    Returns:
        (n_res, 4) or (n_res, 3) array
    """
    out_dim = 4 if include_mean_fi else 3
    features = np.zeros((n_res, out_dim), dtype=np.float32)

    for d in densities:
        key = (str(d.chain_id), int(d.residue_number))
        idx = key_to_idx.get(key)
        if idx is None:
            continue
        features[idx, 0] = float(d.rel_highly_frustrated)
        features[idx, 1] = float(d.rel_neutrally_frustrated)
        features[idx, 2] = float(d.rel_minimally_frustrated)

    # For mean FI, we need per-contact data — approximate from densities
    # since frustrapy doesn't directly give per-residue mean FI.
    # Use: mean_FI ≈ high*(-2) + neutral*(0) + minimal*(1.5) as weighted proxy
    if include_mean_fi:
        for d in densities:
            key = (str(d.chain_id), int(d.residue_number))
            idx = key_to_idx.get(key)
            if idx is None:
                continue
            # Weighted estimate from density fractions
            features[idx, 3] = (
                float(d.rel_highly_frustrated) * (-2.0) +
                float(d.rel_neutrally_frustrated) * 0.0 +
                float(d.rel_minimally_frustrated) * 1.5
            )

    return features


def extract_frustrapy_features(pdb_path, label_path, run_mutational=False):
    """Extract frustration features using frustrapy library.

    frustrapy returns:
        result = (pdb_obj, plots_dict, FrustrationDensityResults, single_residue_data)
        FrustrationDensityResults has:
            .densities: list of FrustrationDensity(residue_number, chain_id,
                        rel_highly_frustrated, rel_neutrally_frustrated, rel_minimally_frustrated, ...)
            .frustration_values: np.ndarray of per-contact FI values
            .contact_coordinates: np.ndarray (N_contacts, 3)

    Returns:
        features: (N, 4) or (N, 7) aligned with labels
        n_dim: 4 or 7
    """
    labels_df = pd.read_csv(label_path, dtype={'chain': str})
    n_res = len(labels_df)

    # Build lookup for label alignment
    key_to_idx = {}
    for idx, (_, row) in enumerate(labels_df.iterrows()):
        key_to_idx[(row['chain'], row['resnum'])] = idx

    # Create isolated temp directory
    tmpdir = tempfile.mkdtemp(prefix='frust_')
    try:
        # Copy PDB to temp dir (frustrapy may modify working dir)
        pdb_basename = os.path.basename(pdb_path)
        tmp_pdb = os.path.join(tmpdir, pdb_basename)
        shutil.copy2(pdb_path, tmp_pdb)

        # Run configurational mode — returns (pdb_obj, equiv, density_results, None)
        result = frustrapy.calculate_frustration(
            pdb_file=tmp_pdb,
            mode="configurational",
            results_dir=tmpdir,
            graphics=False
        )
        density_results = result[2]

        if density_results is None or not hasattr(density_results, 'densities'):
            return None, 0

        config_feat = _densities_to_features(
            density_results.densities, key_to_idx, n_res,
            include_mean_fi=True
        )

        if run_mutational:
            result_mut = frustrapy.calculate_frustration(
                pdb_file=tmp_pdb,
                mode="mutational",
                results_dir=tmpdir,
                graphics=False
            )
            mut_density = result_mut[2]

            if mut_density is not None and hasattr(mut_density, 'densities'):
                mut_feat = _densities_to_features(
                    mut_density.densities, key_to_idx, n_res,
                    include_mean_fi=False
                )
                features = np.concatenate([config_feat, mut_feat], axis=1)
                return features.astype(np.float32), FRUST_DIM_FULL
            else:
                return config_feat.astype(np.float32), FRUST_DIM_CONFIG
        else:
            return config_feat.astype(np.float32), FRUST_DIM_CONFIG

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)



# ── MJ-based fallback extraction ─────────────────────────────────────────────

def extract_frustration_fallback(pdb_path, label_path):
    """Compute configurational frustration features using MJ2H matrix directly.

    This is the fallback when frustrapy is not available.
    Produces 4-dim features only (configurational, no mutational).

    Returns:
        features: (N, 4) aligned with labels, or None on failure
    """
    residues, contacts = parse_residues_and_contacts(pdb_path)
    n_res = len(residues)
    if n_res == 0:
        return None

    config_feat = aggregate_contacts_to_residues(n_res, contacts, mode='config')

    # Align with labels
    labels_df = pd.read_csv(label_path, dtype={'chain': str})

    feat_lookup = {}
    for i, (chain, resnum, _) in enumerate(residues):
        feat_lookup[(chain, resnum)] = i

    aligned = []
    for _, row in labels_df.iterrows():
        key = (row['chain'], row['resnum'])
        if key in feat_lookup:
            aligned.append(config_feat[feat_lookup[key]])
        else:
            aligned.append(np.zeros(FRUST_DIM_CONFIG, dtype=np.float32))

    return np.array(aligned, dtype=np.float32)


# ── Worker function ───────────────────────────────────────────────────────────

def process_single_protein(args):
    """Process one protein — for multiprocessing Pool."""
    pdb_id, pdb_path, label_path, output_path, use_frustrapy, run_mutational = args

    if os.path.exists(output_path):
        return pdb_id, 'skipped', None

    if not os.path.exists(pdb_path) or not os.path.exists(label_path):
        return pdb_id, 'failed', 'missing files'

    try:
        if use_frustrapy:
            features, n_dim = extract_frustrapy_features(
                pdb_path, label_path, run_mutational=run_mutational
            )
            if features is None:
                # Frustrapy failed for this protein, try fallback
                features = extract_frustration_fallback(pdb_path, label_path)
                n_dim = FRUST_DIM_CONFIG if features is not None else 0
        else:
            features = extract_frustration_fallback(pdb_path, label_path)
            n_dim = FRUST_DIM_CONFIG if features is not None else 0

        if features is None:
            return pdb_id, 'failed', 'extraction returned None'

        # Save
        labels_df = pd.read_csv(label_path, dtype={'chain': str})
        labels = labels_df['is_allosteric'].values

        feature_names = FRUST_FEATURE_NAMES[:n_dim]
        np.savez_compressed(
            output_path,
            features=features,
            labels=labels,
            pdb_id=pdb_id,
            feature_names=feature_names
        )
        return pdb_id, 'success', f'{len(features)} res, {n_dim}-dim'

    except Exception as e:
        return pdb_id, 'failed', str(e)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Local Frustration features")
    parser.add_argument('--casbench', action='store_true',
                        help='Process CASBench proteins instead of training proteins')
    parser.add_argument('--mutational', action='store_true',
                        help='Also compute mutational frustration (requires frustrapy, slow)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of worker processes (default: cpu_count - 2, cap 24)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Local Frustration Feature Extraction")
    print("  Ferreiro et al., PNAS 2007, 2011")
    print("=" * 60)

    # Decide engine
    if HAS_FRUSTRAPY:
        print(f"  Engine: frustrapy (installed)")
        _USE_FRUSTRAPY = True
    else:
        print(f"  Engine: MJ2H fallback (frustrapy not found)")
        _USE_FRUSTRAPY = False

    if args.mutational:
        if not HAS_FRUSTRAPY:
            print(f"  WARNING: --mutational requires frustrapy. Will produce 4-dim (config only).")
            _RUN_MUTATIONAL = False
        else:
            _RUN_MUTATIONAL = True
            print(f"  Mutational mode: ON (7-dim output)")
    else:
        _RUN_MUTATIONAL = False

    n_dim = FRUST_DIM_FULL if (_USE_FRUSTRAPY and _RUN_MUTATIONAL) else FRUST_DIM_CONFIG
    feature_names = FRUST_FEATURE_NAMES[:n_dim]

    print(f"  Features ({n_dim}):")
    for name in feature_names:
        print(f"    - {name}")
    print(f"  Contact cutoff: {CONTACT_CUTOFF} A")
    print(f"  Sequence separation: >= {SEQ_SEP}")
    print(f"  FI thresholds: highly < {FI_HIGHLY_FRUSTRATED}, minimally > {FI_MINIMALLY_FRUSTRATED}")
    print(f"  MJ2H background: mean={MJ2H_MEAN:.4f}, std={MJ2H_STD:.4f}")
    print(f"  CPU cores available: {cpu_count()}")

    if args.casbench:
        # CASBench mode
        output_dir = CASBENCH_FEATURES_DIR
        os.makedirs(output_dir, exist_ok=True)
        pdb_csv = os.path.join(CASBENCH_DIR, "casbench_independent_pdbs.csv")
        if not os.path.exists(pdb_csv):
            print(f"ERROR: {pdb_csv} not found. Run evaluate_casbench.py --phase discover first.")
            return

        pdb_list = pd.read_csv(pdb_csv)
        pdb_list = pdb_list[pdb_list['is_overlap'] == False]
        print(f"\n  Mode: CASBench ({len(pdb_list)} independent proteins)")
        print(f"  Output: {output_dir}")

        tasks = []
        for _, row in pdb_list.iterrows():
            pdb_id = row['pdb_id']
            pdb_path = row['pdb_path']
            label_path = os.path.join(CASBENCH_LABELS_DIR, f"{pdb_id}_labels.csv")
            output_path = os.path.join(output_dir, f"{pdb_id}_frust.npz")
            tasks.append((pdb_id, pdb_path, label_path, output_path, _USE_FRUSTRAPY, _RUN_MUTATIONAL))
    else:
        # Training mode
        output_dir = FRUST_DIR
        print(f"\n  Mode: Training proteins")
        print(f"  Output: {output_dir}")

        summary_path = os.path.join(PROCESSED_DIR, "dataset_summary.csv")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            pdb_ids = summary['pdb_id'].tolist()
        else:
            splits_path = os.path.join(PROCESSED_DIR, "train_val_test_splits.csv")
            splits = pd.read_csv(splits_path)
            pdb_ids = splits['pdb_id'].tolist()

        pdb_dir = os.path.join(DATA_DIR, "pdb_files")
        tasks = []
        for pdb_id in pdb_ids:
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            label_path = os.path.join(PROCESSED_DIR, f"{pdb_id}_labels.csv")
            output_path = os.path.join(output_dir, f"{pdb_id}_frust.npz")
            tasks.append((pdb_id, pdb_path, label_path, output_path, _USE_FRUSTRAPY, _RUN_MUTATIONAL))

    print(f"  Total tasks: {len(tasks)}")

    n_existing = sum(1 for t in tasks if os.path.exists(t[3]))
    if n_existing > 0:
        print(f"  Already done: {n_existing} (will skip)")

    # Workers
    if args.workers > 0:
        n_workers = min(args.workers, cpu_count())
    else:
        n_workers = min(max(1, cpu_count() - 2), 24)
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
    print(f"Local Frustration Extraction Complete ({elapsed:.0f}s)")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output dir: {output_dir}")
    if errors:
        print(f"  First errors:")
        for e in errors[:5]:
            print(f"    {e}")

    # Verify one file
    sample_files = [f for f in os.listdir(output_dir) if f.endswith('_frust.npz')]
    if sample_files:
        sample = np.load(os.path.join(output_dir, sample_files[0]))
        feat = sample['features']
        print(f"\n  Verification ({sample_files[0]}):")
        print(f"    Shape: {feat.shape}")
        print(f"    Range: [{feat.min():.4f}, {feat.max():.4f}]")
        names = list(sample['feature_names'])
        print(f"    Names: {names}")
        for i, name in enumerate(names):
            col = feat[:, i]
            print(f"    {name}: mean={col.mean():.4f}, std={col.std():.4f}, "
                  f"min={col.min():.4f}, max={col.max():.4f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
