# Real Site vs. Model Prediction — Live Demo

Companion web app for my oral defense on residue-level allosteric site prediction.
Interactive 3D viewer showing where the model agrees with experimentally validated
allosteric sites — and where it fails.

**Live:** https://presentationsite-six.vercel.app
**Code:** https://github.com/Hassan-Alhabeeb/ResidueAllo

---

## What the model does

XGBoost trained on **2,043 proteins** (≈10× prior work), 225 features per residue
combining structural geometry, protein dynamics, pocket detection, and ESM-2 language
model embeddings. Blind-tested on **2,370 independent CASBench proteins** across 91
enzyme families — the largest rigorous evaluation of an allosteric site predictor to
date.

## Headline results

| Metric | Test set (242 proteins) | CASBench blind (2,370 proteins) |
|---|---|---|
| AUROC | **0.926** | **0.796** |
| AUPRC | 0.401 | 0.238 |
| MCC | 0.452 | 0.250 |

On proteins with **<30% sequence identity to training**, CASBench AUROC actually
*increased* to 0.789 — evidence the model learned real structural-functional patterns,
not just sequence memorization.

## Novel finding

Per-family stratification across 83 CASBench families revealed **5 enzyme families with
anti-correlated predictions** (AUROC < 0.5 — worse than random). All five share the
same property: the allosteric site sits on a **flat protein-protein interface**, not in
a pocket. FPocket features — which contribute 31% of total model importance — return
null on flat surfaces, so the model defaults to predicting the active-site pocket and
gets the answer backwards. This is a previously unreported failure mode affecting every
pocket-based predictor in the literature.

The site exposes this directly: tap **1HQ6** (AUROC 0.26) or **3W8L** (AUROC 0.34) to
see the model miss the real site and flag the active site instead.

---

## Demo proteins

Five CASBench blind-test proteins — three clean successes, two interface-allostery failures.

| PDB | AUROC | Category | What it demonstrates |
|---|---|---|---|
| 1UU7 | 0.99 | Success | Clean win. Pocket geometry, dynamics, and ESM-2 all agree. |
| 4KSQ | 0.96 | Success | Kinase family — clinically critical, the home of asciminib. |
| 3ME3 | 0.95 | Success | <30% ID to training — real generalization. |
| 1HQ6 | 0.26 | **Failure** | Interface allostery. Novel pocket-bias failure mode. |
| 3W8L | 0.34 | **Failure** | Second interface case. Same failure pattern, different family. |

## View modes

- **Both** (default) — true positive / false negative / false positive + active site overlaid together
- **Ground truth only** — just the experimentally validated allosteric residues
- **Prediction only** — just what the model flagged (green if correct, red if wrong)
- **Probability heatmap** — per-residue probability with no threshold

## Color legend

| Color | Meaning |
|---|---|
| Green | True positive — real allosteric residue the model caught |
| Blue | False negative — real allosteric residue the model missed |
| Red | False positive — model predicted, actually not allosteric |
| Orange | Active / catalytic site |
| Grey | Other residues |

---

## Files

| Path | Purpose |
|---|---|
| `index.html` / `style.css` / `script.js` | Single-page app, 3Dmol.js from CDN |
| `data/manifest.json` | Protein list + threshold |
| `data/*.json` | Per-residue truth + prediction + active-site flags |
| `scripts/generate_data.py` | Regenerates `data/` from the CASBench predictions |
| `scripts/generate_cheatsheet.py` | Regenerates the printable demo cheat-sheet PDF |
| `demo_cheatsheet.pdf` | 2-page cheat-sheet for live Q&A |
| `vercel.json` | Static deploy config |

PDB structures are streamed at runtime from `files.rcsb.org` — nothing is bundled.

## Run locally

```bash
cd presentation_site
python -m http.server 8080
# open http://localhost:8080
```

## Deploy

```bash
vercel --prod       # rebuilds and redeploys to the aliased URL
```

## Regenerate data

```bash
python scripts/generate_data.py         # per-residue JSONs from NPZ predictions
python scripts/generate_cheatsheet.py   # printable PDF cheat-sheet
```

Paths are resolved relative to `../allosteric/data/casbench/`. The decision threshold
is read from `../allosteric/results/xgboost_tuned_results.json` (currently **0.343**,
Optuna-tuned on the validation set).
