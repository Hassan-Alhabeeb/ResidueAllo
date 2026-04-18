"""
Export per-residue predictions + labels to compact JSON for the presentation site.

For each PDB:
  - Loads labels CSV (chain, resnum, resname, is_allosteric, is_active_site)
  - Loads predictions NPZ (y_true, y_prob)
  - Writes JSON: {pdb_id, meta, threshold, residues: [{c, n, r, t, p, a}...]}
"""

import os
import json
import csv
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "allosteric"))
LABELS_DIR = os.path.join(BASE, "data", "casbench", "labels")
PRED_DIR = os.path.join(BASE, "data", "casbench", "predictions")
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RESULTS_JSON = os.path.join(BASE, "results", "xgboost_tuned_results.json")

# (pdb_id, category, family_cas, narrative_title, narrative_blurb)
TARGETS = [
    (
        "1UU7", "win", "cas0003",
        "Clean win — model nails the allosteric pocket",
        "Phosphofructokinase-1 style regulation. High-AUROC case where pocket geometry, dynamics, and ESM-2 all agree on the regulatory site.",
    ),
    (
        "4KSQ", "win", "cas0012",
        "Kinase family — the clinically critical case",
        "Kinases are 35% of drug targets and the domain where allosteric inhibitors (e.g. asciminib for CML) are transforming therapy. Model correctly flags the allosteric pocket away from the ATP site.",
    ),
    (
        "3ME3", "win", "cas0044",
        "Generalization beyond the training distribution",
        "Test protein with <30% identity to anything in training. Strong prediction demonstrates the model learned structure-function patterns, not sequence memorization.",
    ),
    (
        "1HQ6", "fail", "cas0074",
        "Interface allostery — the novel failure mode",
        "Allosteric site sits on a flat protein-protein interface. FPocket returns null features. Model defaults to the active-site pocket and gets it backwards (AUROC 0.25 — worse than random). This is the pocket-bias blind spot affecting the entire field.",
    ),
    (
        "3W8L", "fail", "cas0014",
        "Second interface-allostery case — same blind spot",
        "Different enzyme family, same failure pattern. Any predictor relying on pocket geometry — and most in the literature do — shares this blind spot.",
    ),
]


def load_threshold():
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            return json.load(f).get("optimal_threshold", 0.4)
    return 0.4


def build_entry(pdb_id: str) -> dict:
    labels_path = os.path.join(LABELS_DIR, f"{pdb_id}_labels.csv")
    pred_path = os.path.join(PRED_DIR, f"{pdb_id}_pred.npz")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(labels_path)
    if not os.path.exists(pred_path):
        raise FileNotFoundError(pred_path)

    rows = []
    with open(labels_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    pred = np.load(pred_path)
    y_true = pred["y_true"]
    y_prob = pred["y_prob"]
    if len(rows) != len(y_true):
        raise ValueError(f"{pdb_id}: label count {len(rows)} != pred count {len(y_true)}")

    residues = []
    for row, t, p in zip(rows, y_true, y_prob):
        residues.append({
            "c": row["chain"],
            "n": int(row["resnum"]),
            "r": row["resname"],
            "t": int(row["is_allosteric"]),
            "a": int(row["is_active_site"]),
            "p": round(float(p), 4),
        })

    true_total = sum(res["t"] for res in residues)
    active_total = sum(res["a"] for res in residues)
    return {
        "pdb_id": pdb_id,
        "n_residues": len(residues),
        "n_true_allosteric": true_total,
        "n_active_site": active_total,
        "residues": residues,
    }


def main():
    thr = load_threshold()
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {"threshold": thr, "proteins": []}

    for pdb_id, cat, fam, title, blurb in TARGETS:
        print(f"-> {pdb_id} ({cat})", end=" ")
        data = build_entry(pdb_id)
        data["category"] = cat
        data["family"] = fam
        data["title"] = title
        data["blurb"] = blurb
        out_path = os.path.join(OUT_DIR, f"{pdb_id}.json")
        with open(out_path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        manifest["proteins"].append({
            "pdb_id": pdb_id,
            "category": cat,
            "family": fam,
            "title": title,
            "blurb": blurb,
            "n_residues": data["n_residues"],
            "n_true_allosteric": data["n_true_allosteric"],
        })
        print(f"OK  ({data['n_residues']} residues, {data['n_true_allosteric']} allosteric)")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nThreshold: {thr}")
    print(f"Manifest written to {os.path.join(OUT_DIR, 'manifest.json')}")


if __name__ == "__main__":
    main()
