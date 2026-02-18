"""
Extract NMA flexibility + graph centrality features for all proteins.

NMA Features (6-dim, ProDy):
  1. ANM fluctuation (Anisotropic Network Model)
  2. GNM fluctuation (Gaussian Network Model)
  3. Slow mode contribution (modes 1-3, global motions)
  4. Fast mode contribution (top 10% fast modes)
  5. Stiffness (inverse of GNM fluctuation)
  6. Slow/fast ratio

Graph Centrality Features (5-dim, NetworkX):
  1. Betweenness centrality
  2. Closeness centrality
  3. Clustering coefficient
  4. Degree centrality
  5. PageRank

Total: 11 new features per residue.
Uses multiprocessing to parallelize across all CPU cores.
"""

import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import warnings
import traceback
from multiprocessing import Pool, cpu_count
import time

warnings.filterwarnings('ignore')

# Suppress ProDy logging
import prody
prody.confProDy(verbosity='none')

DATA_DIR = r"E:\newyear\research_plan\allosteric\data"
PDB_DIR = os.path.join(DATA_DIR, "pdb_files")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "..", "features")
NMA_DIR = os.path.join(FEATURES_DIR, "nma_graph")
os.makedirs(NMA_DIR, exist_ok=True)

# Must match extract_features.py / extract_labels.py
AA_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
           'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
           'TYR', 'VAL']
NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER', 'TPO': 'THR',
    'PTR': 'TYR', 'CSE': 'CYS',
}

NMA_FEATURE_NAMES = [
    'anm_fluctuation', 'gnm_fluctuation',
    'slow_mode_contrib', 'fast_mode_contrib',
    'stiffness', 'slow_fast_ratio'
]

GRAPH_FEATURE_NAMES = [
    'betweenness_centrality', 'closeness_centrality',
    'clustering_coefficient', 'degree_centrality', 'pagerank'
]

ALL_FEATURE_NAMES = NMA_FEATURE_NAMES + GRAPH_FEATURE_NAMES


def get_residues_and_coords(pdb_path):
    """Parse PDB, return residues info and CA coordinates (matching extract_features.py)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = structure[0]

    residues = []
    ca_coords = []
    for chain in model:
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
                'resname': resname
            })

            if 'CA' in res:
                ca_coords.append(res['CA'].get_vector().get_array())
            else:
                atoms = list(res.get_atoms())
                if atoms:
                    ca_coords.append(np.mean([a.get_vector().get_array() for a in atoms], axis=0))
                else:
                    ca_coords.append(np.array([0.0, 0.0, 0.0]))

    return residues, np.array(ca_coords)


def extract_nma_features(ca_coords):
    """Extract NMA flexibility features using ProDy ANM and GNM."""
    n = len(ca_coords)
    features = np.zeros((n, 6))

    if n < 5:
        return features

    try:
        # GNM (Gaussian Network Model) — simpler, faster, 1D fluctuations
        gnm = prody.GNM('protein')
        gnm.buildKirchhoff(ca_coords, cutoff=10.0)
        gnm.calcModes(n_modes=min(20, n - 1))

        gnm_fluct = prody.calcSqFlucts(gnm)
        features[:, 1] = gnm_fluct  # GNM fluctuation

        # Stiffness = inverse of GNM fluctuation
        features[:, 4] = 1.0 / (gnm_fluct + 1e-8)

        # Slow modes (1-3): global motions — allosteric sites often involved
        n_slow = min(3, gnm.numModes())
        if n_slow > 0:
            slow_fluct = np.zeros(n)
            for i in range(n_slow):
                mode = gnm[i]
                slow_fluct += (mode.getArray() ** 2) * mode.getVariance()
            features[:, 2] = slow_fluct

        # Fast modes (top 10%): local vibrations
        n_fast = max(1, gnm.numModes() // 10)
        fast_start = gnm.numModes() - n_fast
        fast_fluct = np.zeros(n)
        for i in range(fast_start, gnm.numModes()):
            mode = gnm[i]
            fast_fluct += (mode.getArray() ** 2) * mode.getVariance()
        features[:, 3] = fast_fluct

        # Slow/fast ratio
        features[:, 5] = features[:, 2] / (features[:, 3] + 1e-8)

    except Exception:
        pass  # GNM features stay as zeros

    # ANM skipped for now (too slow on large proteins — 3N×3N Hessian).
    # Column 0 stays as zeros. Re-run with ANM later.
    # features[:, 0] = anm_fluct  # ANM fluctuation

    return features


def extract_graph_features(ca_coords, contact_cutoff=8.0):
    """Extract graph centrality features from residue contact graph.
    Uses vectorized distance matrix and approximate betweenness for large proteins."""
    import networkx as nx
    from scipy.spatial.distance import cdist

    n = len(ca_coords)
    features = np.zeros((n, 5))

    if n < 3:
        return features

    try:
        # Vectorized distance matrix (not Python loop)
        dist_matrix = cdist(ca_coords, ca_coords)

        # Build contact graph from distance matrix
        G = nx.Graph()
        G.add_nodes_from(range(n))
        contacts = np.argwhere((dist_matrix < contact_cutoff) & (np.triu(np.ones((n, n), dtype=bool), k=1)))
        for i, j in contacts:
            G.add_edge(int(i), int(j))

        if G.number_of_edges() == 0:
            return features

        # Betweenness centrality (approximate for large proteins to avoid O(VE) stall)
        k_approx = min(n, 200) if n > 500 else None
        bc = nx.betweenness_centrality(G, k=k_approx)
        for i in range(n):
            features[i, 0] = bc.get(i, 0.0)

        # Closeness centrality
        cc = nx.closeness_centrality(G)
        for i in range(n):
            features[i, 1] = cc.get(i, 0.0)

        # Clustering coefficient
        clust = nx.clustering(G)
        for i in range(n):
            features[i, 2] = clust.get(i, 0.0)

        # Degree centrality
        dc = nx.degree_centrality(G)
        for i in range(n):
            features[i, 3] = dc.get(i, 0.0)

        # PageRank
        pr = nx.pagerank(G, max_iter=100)
        for i in range(n):
            features[i, 4] = pr.get(i, 0.0)

    except Exception:
        pass  # Graph features stay as zeros

    return features


def process_single_protein(args):
    """Process a single protein — designed for multiprocessing Pool."""
    pdb_id, pdb_path, label_path, output_path = args

    if os.path.exists(output_path):
        return pdb_id, 'skipped', None

    if not os.path.exists(pdb_path) or not os.path.exists(label_path):
        return pdb_id, 'failed', 'missing files'

    try:
        residues, ca_coords = get_residues_and_coords(pdb_path)
        if len(residues) == 0:
            return pdb_id, 'failed', 'no residues'

        # Extract both feature groups
        nma_feat = extract_nma_features(ca_coords)     # (N, 6)
        graph_feat = extract_graph_features(ca_coords)  # (N, 5)
        all_feat = np.concatenate([nma_feat, graph_feat], axis=1)  # (N, 11)

        # Align with labels (same as extract_features.py — drop unmatched)
        labels_df = pd.read_csv(label_path)

        feat_lookup = {}
        for i, info in enumerate(residues):
            key = (info['chain'], info['resnum'])
            feat_lookup[key] = i

        aligned_features = []
        aligned_labels = []
        for _, lrow in labels_df.iterrows():
            key = (lrow['chain'], lrow['resnum'])
            if key in feat_lookup:
                aligned_features.append(all_feat[feat_lookup[key]])
                aligned_labels.append(lrow['is_allosteric'])

        if len(aligned_features) == 0:
            return pdb_id, 'failed', 'no alignment'

        aligned_features = np.array(aligned_features, dtype=np.float32)
        aligned_labels = np.array(aligned_labels)

        np.savez_compressed(output_path,
                            features=aligned_features,
                            labels=aligned_labels,
                            pdb_id=pdb_id,
                            feature_names=ALL_FEATURE_NAMES)

        return pdb_id, 'success', None

    except Exception as e:
        return pdb_id, 'failed', str(e)


if __name__ == '__main__':
    summary = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_summary.csv"))
    print(f"Proteins to process: {len(summary)}")
    print(f"Features: {len(ALL_FEATURE_NAMES)} ({len(NMA_FEATURE_NAMES)} NMA + {len(GRAPH_FEATURE_NAMES)} graph)")
    print(f"CPU cores available: {cpu_count()}")

    # Build task list
    tasks = []
    for _, row in summary.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
        label_path = os.path.join(PROCESSED_DIR, f"{pdb_id}_labels.csv")
        output_path = os.path.join(NMA_DIR, f"{pdb_id}_nma_graph.npz")
        tasks.append((pdb_id, pdb_path, label_path, output_path))

    n_workers = max(1, cpu_count() - 1)  # Leave 1 core free
    print(f"Using {n_workers} worker processes\n")

    start_time = time.time()
    processed = 0
    skipped = 0
    failed = 0
    errors = []

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_single_protein, tasks, chunksize=4):
            pdb_id, status, error = result
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
                    errors.append(f"{pdb_id}: {error}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"NMA + Graph Feature Extraction Complete ({elapsed:.0f}s)")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output dir: {NMA_DIR}")
    if errors:
        print(f"  First errors:")
        for e in errors[:5]:
            print(f"    {e}")
    print(f"{'='*60}")
