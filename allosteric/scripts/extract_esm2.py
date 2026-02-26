"""
Extract ESM-2 per-residue embeddings for all proteins.

Uses the esm2_t33_650M_UR50D model (650M params, 1280-dim embeddings).
Embeddings are saved per-protein and later reduced via PCA to 128 dimensions.

Handles long sequences (>1022 residues) via overlapping sliding windows
with stride 512, averaging embeddings in overlapping regions.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/../ = allosteric/
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "..", "features")
ESM_DIR = os.path.join(FEATURES_DIR, "esm2_embeddings")
os.makedirs(ESM_DIR, exist_ok=True)

# AA 3-letter to 1-letter mapping
AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# Non-standard -> standard mapping (must match extract_features.py / extract_labels.py)
NONSTANDARD_MAP = {
    'MSE': 'MET', 'HYP': 'PRO', 'SEP': 'SER', 'TPO': 'THR',
    'PTR': 'TYR', 'CSE': 'CYS',
}

MAX_LEN = 1022      # ESM-2 max input length
WINDOW_STRIDE = 512  # Overlap stride for sliding windows


def get_sequence_from_pdb(pdb_path):
    """Extract sequence and residue mapping from PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = structure[0]

    chains = {}
    for chain in model:
        residues = []
        for res in chain:
            resname = res.get_resname()
            if res.id[0] != ' ' and resname not in NONSTANDARD_MAP:
                continue
            # Map non-standard to standard
            if resname in NONSTANDARD_MAP:
                resname = NONSTANDARD_MAP[resname]
            if resname not in AA3TO1:
                continue
            residues.append({
                'chain': chain.id,
                'resnum': res.id[1],
                'resname': resname,
                'aa1': AA3TO1[resname]
            })
        if residues:
            chains[chain.id] = residues

    return chains


def extract_single_sequence(seq, model, alphabet, batch_converter, device):
    """Extract ESM-2 embeddings for a single sequence, handling long sequences
    with overlapping sliding windows."""
    seq_len = len(seq)

    if seq_len <= MAX_LEN:
        # Short sequence — single forward pass
        batch_labels, batch_strs, batch_tokens = batch_converter([("seq", seq)])
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        embeddings = results["representations"][33]
        return embeddings[0, 1:seq_len+1, :].cpu().float().numpy()  # (seq_len, 1280) always float32

    # Long sequence — sliding window with overlap
    emb_sum = np.zeros((seq_len, 1280), dtype=np.float64)
    emb_count = np.zeros(seq_len, dtype=np.int32)

    start = 0
    while start < seq_len:
        end = min(start + MAX_LEN, seq_len)
        # If the last window is too short, shift it back
        if end - start < 100 and start > 0:
            break
        window_seq = seq[start:end]
        window_len = len(window_seq)

        batch_labels, batch_strs, batch_tokens = batch_converter([("win", window_seq)])
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        window_emb = results["representations"][33]
        window_emb = window_emb[0, 1:window_len+1, :].cpu().numpy()

        emb_sum[start:start+window_len] += window_emb
        emb_count[start:start+window_len] += 1

        if end >= seq_len:
            break
        start += WINDOW_STRIDE

    # Verify all positions covered, then average overlapping regions
    if np.any(emb_count == 0):
        uncovered = int(np.sum(emb_count == 0))
        raise RuntimeError(f"{uncovered} residues not covered by any window (seq_len={seq_len})")
    embeddings = emb_sum / emb_count[:, np.newaxis]
    return embeddings.astype(np.float32)


if __name__ == '__main__':
    import esm

    # Load ESM-2 model
    print("Loading ESM-2 model (esm2_t33_650M_UR50D)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model_esm = model_esm.to(device)
    model_esm.eval()

    # Use FP16 on GPU for VRAM efficiency (~3GB instead of ~6GB)
    if device.type == 'cuda':
        model_esm = model_esm.half()
        print(f"  Using FP16 on GPU")

    print("  Model loaded!")

    # Load summary
    summary = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_summary.csv"))
    print(f"Proteins to process: {len(summary)}")

    processed = 0
    skipped = 0
    failed = 0
    n_windowed = 0
    start_time = time.time()

    for idx, row in summary.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = os.path.join(DATA_DIR, "pdb_files", f"{pdb_id}.pdb")

        output_path = os.path.join(ESM_DIR, f"{pdb_id}_esm2.npz")

        # Skip if already done
        if os.path.exists(output_path):
            skipped += 1
            continue

        if not os.path.exists(pdb_path):
            failed += 1
            continue

        try:
            chains = get_sequence_from_pdb(pdb_path)
            if not chains:
                failed += 1
                continue

            # Extract embeddings per chain
            chain_residue_maps = {}
            chain_embeddings = {}

            for chain_id, residues in chains.items():
                seq = ''.join(r['aa1'] for r in residues)
                chain_residue_maps[chain_id] = residues

                used_window = len(seq) > MAX_LEN
                if used_window:
                    n_windowed += 1

                emb = extract_single_sequence(
                    seq, model_esm, alphabet, batch_converter, device
                )
                chain_embeddings[chain_id] = emb

            # Align embeddings with label file residues
            label_path = os.path.join(PROCESSED_DIR, f"{pdb_id}_labels.csv")
            if not os.path.exists(label_path):
                failed += 1
                continue

            labels_df = pd.read_csv(label_path, dtype={'chain': str})

            # Build lookup: (chain, resnum) -> embedding vector
            emb_lookup = {}
            for chain_id, residues in chain_residue_maps.items():
                if chain_id in chain_embeddings:
                    emb = chain_embeddings[chain_id]
                    for i, res in enumerate(residues):
                        if i < emb.shape[0]:
                            emb_lookup[(res['chain'], res['resnum'])] = emb[i]

            # Align with labels — match extract_features.py behavior:
            # Only keep residues found in BOTH labels AND embeddings (drop missing).
            # This ensures row counts match structural features exactly.
            aligned_emb = []
            aligned_labels = []
            n_missing = 0
            for _, lrow in labels_df.iterrows():
                key = (lrow['chain'], lrow['resnum'])
                if key in emb_lookup:
                    aligned_emb.append(emb_lookup[key])
                else:
                    aligned_emb.append(np.zeros(1280, dtype=np.float32))
                    n_missing += 1
                aligned_labels.append(lrow['is_allosteric'])

            if len(aligned_emb) == 0:
                failed += 1
                continue

            aligned_emb = np.array(aligned_emb, dtype=np.float32)  # (N, 1280)

            if n_missing > 0 and n_missing > len(labels_df) * 0.1:
                print(f"  WARNING: {pdb_id}: {n_missing}/{len(labels_df)} residues dropped ({100*n_missing/len(labels_df):.0f}%)")

            np.savez_compressed(output_path, embeddings=aligned_emb,
                                labels=np.array(aligned_labels), pdb_id=pdb_id)

            processed += 1
            if processed % 50 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                remaining = (len(summary) - skipped - processed - failed) / max(rate, 0.01)
                print(f"  Processed {processed} proteins ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, {n_windowed} windowed)")

        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"  ERROR on {pdb_id}: {e}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ESM-2 Embedding Extraction Complete ({elapsed:.0f}s)")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Used sliding window: {n_windowed} chains")
    print(f"  Output dir: {ESM_DIR}")
    print(f"{'='*60}")
