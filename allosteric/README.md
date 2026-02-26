# ResidueAllo: ML-Based Allosteric Binding Site Prediction

Residue-level allosteric site prediction using XGBoost with hybrid structural, dynamic, energetic, and protein language model features. Trained on the largest dataset in this field (2,043 proteins) and blindly evaluated on 2,370 independent proteins from CASBench.

## Key Results

| Metric | Test Set | CASBench (blind) | CASBench <30% identity |
|--------|----------|-------------------|------------------------|
| AUROC | 0.926 | 0.750 | 0.789 |
| AUPRC | 0.401 (23.6x random) | 0.190 (4.3x random) | 0.218 |
| MCC | 0.452 | 0.193 | - |
| F1 | 0.461 | - | - |

## Features (225 per residue)

| Category | Dim | Description |
|----------|-----|-------------|
| Structural | 64 | Secondary structure, solvent accessibility, B-factors, contacts, half-sphere exposure |
| Dynamics | 11 | Normal mode analysis (GNM/ANM) + graph centrality measures |
| Pocket geometry | 8 | FPocket druggability, volume, hydrophobicity, polarity |
| Physicochemical | 6 | AAindex hydrophobicity, volume, charge, flexibility, polarity, molecular weight |
| Transfer Entropy | 3 | Causal information flow between residues (AllosES-inspired) |
| Perturbation Response | 3 | PRS effectiveness, sensitivity, and ratio |
| Contact Energy | 2 | Miyazawa-Jernigan pairwise contact potential statistics |
| Local Frustration | 7 | Configurational frustration index fractions and mean (Ferreiro et al.) |
| Language Model | 128 | ESM-2 (650M parameters) embeddings compressed via PCA |

## Comparison to Existing Methods

| Method | Year | Level | Training Set | Blind Benchmark |
|--------|------|-------|-------------|-----------------|
| DeepAllo | 2025 | Pocket | 207 proteins | None |
| AlloPED | 2025 | Both | 146 proteins | None |
| AlloFusion | 2025 | Residue | 143 proteins | None |
| STINGAllo | 2025 | Residue | 230 proteins | None |
| PASSer | 2023 | Pocket | 207 proteins | CASBench (80.5%) |
| **ResidueAllo** | **2026** | **Residue** | **2,043 proteins** | **CASBench (2,370)** |

## Data Pipeline

1. Collected 2,043 protein structures from AlloBench
2. Labeled 1.7M residues (28,603 allosteric, 1.7% positive rate)
3. Clustered at 30% sequence identity (MMseqs2) to prevent data leakage
4. Split into train (1,600) / validation (204) / test (242) proteins

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch, and the following external tools:
- **FPocket** (v4.2+) for pocket detection
- **MMseqs2** for sequence clustering

## Usage

### Feature Extraction

```bash
# Extract all feature types
python scripts/extract_labels.py
python scripts/cluster_and_split.py
python scripts/extract_features.py          # Structural (64-dim)
python scripts/extract_nma_graph.py         # NMA + graph (11-dim)
python scripts/extract_fpocket.py           # Pocket geometry (8-dim)
python scripts/extract_aaindex.py           # Physicochemical (6-dim)
python scripts/extract_transfer_entropy.py  # Transfer entropy (3-dim)
python scripts/extract_prs.py              # Perturbation response (3-dim)
python scripts/extract_mj_energy.py        # Contact energy (2-dim)
python scripts/extract_local_frustration.py # Local frustration (7-dim)
python scripts/extract_esm2.py             # ESM-2 embeddings (GPU)
```

### Training

```bash
python scripts/build_dataset.py --esm-650m-only
python scripts/train_baseline.py
python scripts/tune_optuna.py              # Bayesian hyperparameter search
```

### CASBench Blind Evaluation

```bash
python scripts/evaluate_casbench.py                    # Run all phases
python scripts/evaluate_casbench.py --phase predict     # Re-run evaluation only
```

## Novel Finding: Pocket Bias in Allosteric Prediction

Per-family stratified analysis on CASBench revealed that 5 out of 83 enzyme families show anti-correlated predictions (AUROC < 0.5). All five have allosteric sites at flat protein-protein interfaces rather than in pockets. Since pocket features account for ~31% of model importance, the model systematically mistakes the active site pocket for the allosteric site in these cases. This failure mode affects all pocket-based allosteric predictors and has not been reported previously.

## Project Structure

```
allosteric/
├── scripts/                  # All extraction, training, and evaluation scripts
├── data/
│   ├── raw/                  # Raw PDB files and CASBench download
│   ├── processed/            # Cleaned structures and split assignments
│   └── casbench/             # CASBench benchmark data and features
├── features/                 # Extracted per-residue feature arrays (.npz)
├── models/                   # Trained XGBoost models and scalers
└── results/                  # Evaluation metrics and Optuna trial logs
```

## Citation

Paper in preparation.

## License

MIT
