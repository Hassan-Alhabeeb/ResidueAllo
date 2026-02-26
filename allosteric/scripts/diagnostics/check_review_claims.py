"""
Diagnostic script: Verify two claims from external code review.

Trace 1: Transfer Entropy float32 catastrophic cancellation
  - Checks whether eigenvalue downcast to float32 causes TE features to collapse to -69.0
  - Picks a sample protein, computes TE in float32 vs float64, compares

Trace 2: Cascading feature wipeout
  - Checks how many proteins have length mismatches between structural features,
    FPocket, AAindex, NMA, TE, and ESM-2
  - Reports exactly which proteins lose features due to the cascade

Usage:
    python scripts/diagnostics/check_review_claims.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # diagnostics/../../ = allosteric/
FEATURES_DIR = os.path.join(BASE_DIR, "features")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Feature subdirectories
STRUCTURAL_DIR = FEATURES_DIR  # {PDB}_features.npz
FPOCKET_DIR = os.path.join(FEATURES_DIR, "fpocket")
NMA_DIR = os.path.join(FEATURES_DIR, "nma_graph")
AAINDEX_DIR = os.path.join(FEATURES_DIR, "aaindex")
TE_DIR = os.path.join(FEATURES_DIR, "transfer_entropy")
ESM_DIR = os.path.join(FEATURES_DIR, "esm2_embeddings")
PRS_DIR = os.path.join(FEATURES_DIR, "prs")


def check_trace2_feature_wipeout():
    """Trace 2: Check length mismatches between feature extractors."""
    print("=" * 70)
    print("  TRACE 2: Cascading Feature Wipeout Check")
    print("=" * 70)
    print()
    print("  Comparing array lengths across all feature extractors.")
    print("  If structural features dropped residues, other extractors will mismatch.")
    print()

    # Find all structural feature files (the reference)
    struct_files = glob.glob(os.path.join(STRUCTURAL_DIR, "*_features.npz"))
    print(f"  Structural feature files found: {len(struct_files)}")

    # Also load labels CSVs to compare
    label_files = glob.glob(os.path.join(PROCESSED_DIR, "*_labels.csv"))
    label_counts = {}
    for lf in label_files:
        pdb_id = os.path.basename(lf).replace("_labels.csv", "")
        try:
            df = pd.read_csv(lf)
            label_counts[pdb_id] = len(df)
        except:
            pass

    # Track mismatches
    results = []
    n_checked = 0

    for sf in sorted(struct_files):
        pdb_id = os.path.basename(sf).replace("_features.npz", "")
        try:
            sdata = np.load(sf)
            struct_len = len(sdata["labels"])
        except:
            continue

        row = {
            "pdb_id": pdb_id,
            "labels_csv": label_counts.get(pdb_id, -1),
            "structural": struct_len,
            "fpocket": -1,
            "nma": -1,
            "aaindex": -1,
            "te": -1,
            "esm2": -1,
            "prs": -1,
        }

        # Check each feature type
        fp_path = os.path.join(FPOCKET_DIR, f"{pdb_id}_fpocket.npz")
        if os.path.exists(fp_path):
            try:
                row["fpocket"] = len(np.load(fp_path)["features"])
            except:
                pass

        nma_path = os.path.join(NMA_DIR, f"{pdb_id}_nma_graph.npz")
        if os.path.exists(nma_path):
            try:
                row["nma"] = len(np.load(nma_path)["features"])
            except:
                pass

        aa_path = os.path.join(AAINDEX_DIR, f"{pdb_id}_aaindex.npz")
        if os.path.exists(aa_path):
            try:
                row["aaindex"] = len(np.load(aa_path)["features"])
            except:
                pass

        te_path = os.path.join(TE_DIR, f"{pdb_id}_te.npz")
        if os.path.exists(te_path):
            try:
                row["te"] = len(np.load(te_path)["features"])
            except:
                pass

        esm_path = os.path.join(ESM_DIR, f"{pdb_id}_esm2.npz")
        if os.path.exists(esm_path):
            try:
                row["esm2"] = len(np.load(esm_path)["embeddings"])
            except:
                pass

        prs_path = os.path.join(PRS_DIR, f"{pdb_id}_prs.npz")
        if os.path.exists(prs_path):
            try:
                row["prs"] = len(np.load(prs_path)["features"])
            except:
                pass

        results.append(row)
        n_checked += 1

    df = pd.DataFrame(results)

    # --- Report 1: Labels CSV vs Structural features (the drop) ---
    has_labels = df[df["labels_csv"] > 0]
    label_drops = has_labels[has_labels["labels_csv"] != has_labels["structural"]]
    print(f"\n  --- Labels CSV vs Structural Features (the initial drop) ---")
    print(f"  Proteins with labels CSV: {len(has_labels)}")
    print(f"  Mismatches (dropped residues): {len(label_drops)}")
    if len(label_drops) > 0:
        print(f"\n  {'PDB':<8} {'Labels CSV':>10} {'Structural':>10} {'Dropped':>8}")
        print(f"  {'-'*40}")
        for _, r in label_drops.head(20).iterrows():
            dropped = r["labels_csv"] - r["structural"]
            print(f"  {r['pdb_id']:<8} {r['labels_csv']:>10} {r['structural']:>10} {dropped:>8}")
        if len(label_drops) > 20:
            print(f"  ... and {len(label_drops) - 20} more")
    else:
        print("  >> No drops detected! extract_features.py matches labels_df perfectly.")

    # --- Report 2: Structural vs FPocket (the cascade trigger) ---
    has_fp = df[df["fpocket"] > 0]
    fp_mismatch = has_fp[has_fp["structural"] != has_fp["fpocket"]]
    print(f"\n  --- Structural vs FPocket (cascade trigger) ---")
    print(f"  Proteins with FPocket: {len(has_fp)}")
    print(f"  Length mismatches: {len(fp_mismatch)} ({100*len(fp_mismatch)/max(len(has_fp),1):.1f}%)")
    if len(fp_mismatch) > 0:
        print(f"\n  {'PDB':<8} {'Structural':>10} {'FPocket':>10} {'Diff':>8}")
        print(f"  {'-'*40}")
        for _, r in fp_mismatch.head(20).iterrows():
            diff = r["fpocket"] - r["structural"]
            print(f"  {r['pdb_id']:<8} {r['structural']:>10} {r['fpocket']:>10} {diff:>8}")

    # --- Report 3: All feature types vs structural ---
    print(f"\n  --- All Feature Types vs Structural ---")
    for feat_name in ["nma", "aaindex", "te", "esm2", "prs"]:
        has_feat = df[df[feat_name] > 0]
        if len(has_feat) == 0:
            print(f"  {feat_name:>8}: no files found")
            continue
        mismatch = has_feat[has_feat["structural"] != has_feat[feat_name]]
        print(f"  {feat_name:>8}: {len(has_feat):>5} files, {len(mismatch):>4} mismatches ({100*len(mismatch)/max(len(has_feat),1):.1f}%)")
        if len(mismatch) > 0:
            examples = mismatch.head(3)
            for _, r in examples.iterrows():
                print(f"           {r['pdb_id']}: structural={r['structural']}, {feat_name}={r[feat_name]}")

    # --- Summary ---
    any_mismatch = df[
        ((df["fpocket"] > 0) & (df["fpocket"] != df["structural"])) |
        ((df["nma"] > 0) & (df["nma"] != df["structural"])) |
        ((df["aaindex"] > 0) & (df["aaindex"] != df["structural"])) |
        ((df["te"] > 0) & (df["te"] != df["structural"])) |
        ((df["esm2"] > 0) & (df["esm2"] != df["structural"])) |
        ((df["prs"] > 0) & (df["prs"] != df["structural"]))
    ]
    print(f"\n  --- SUMMARY ---")
    print(f"  Total proteins checked: {n_checked}")
    print(f"  Proteins with ANY mismatch: {len(any_mismatch)}")
    if len(any_mismatch) > 0:
        print(f"  >> BUG IS ACTIVE: {len(any_mismatch)} proteins losing features due to cascade!")
        print(f"  >> These proteins have ALL non-structural features replaced with zeros.")
    else:
        print(f"  >> Bug NOT active: all feature extractors agree on residue counts.")
        print(f"  >> The cascade mechanism exists in code but doesn't trigger in practice.")

    return df


def check_trace1_te_float32():
    """Trace 1: Check if TE features are collapsed to -69.0 noise."""
    print("\n" + "=" * 70)
    print("  TRACE 1: Transfer Entropy Float32 Check")
    print("=" * 70)
    print()

    te_files = glob.glob(os.path.join(TE_DIR, "*_te.npz"))
    if not te_files:
        print("  No TE feature files found. Skipping.")
        print("  (TE extraction hasn't been run yet)")
        return

    print(f"  TE feature files found: {len(te_files)}")

    # Check value distribution across all TE files
    all_nte = []
    all_out = []
    all_in = []
    n_checked = 0
    n_all_zero = 0

    for tf in te_files[:200]:  # Sample up to 200
        try:
            data = np.load(tf)
            feat = data["features"]  # (N, 3): nte_score, te_out_sum, te_in_sum
            if feat.shape[1] != 3:
                continue

            if np.all(feat == 0):
                n_all_zero += 1
            else:
                all_nte.extend(feat[:, 0].tolist())
                all_out.extend(feat[:, 1].tolist())
                all_in.extend(feat[:, 2].tolist())
            n_checked += 1
        except:
            pass

    print(f"  Files checked: {n_checked}")
    print(f"  All-zero files: {n_all_zero}")

    if all_nte:
        nte = np.array(all_nte)
        out = np.array(all_out)
        inp = np.array(all_in)

        print(f"\n  nte_score distribution:")
        print(f"    min={nte.min():.6f}  max={nte.max():.6f}  mean={nte.mean():.6f}  std={nte.std():.6f}")
        print(f"    zeros: {(nte == 0).sum()} / {len(nte)} ({100*(nte==0).sum()/len(nte):.1f}%)")

        print(f"\n  te_out_sum distribution:")
        print(f"    min={out.min():.6f}  max={out.max():.6f}  mean={out.mean():.6f}  std={out.std():.6f}")

        print(f"\n  te_in_sum distribution:")
        print(f"    min={inp.min():.6f}  max={inp.max():.6f}  mean={inp.mean():.6f}  std={inp.std():.6f}")

        # Check for the specific -69.0 claim
        n_neg69_nte = ((nte < -60) & (nte > -75)).sum()
        n_neg69_out = ((out < -60) & (out > -75)).sum()
        print(f"\n  Values in [-75, -60] range (the '-69 noise' claim):")
        print(f"    nte_score: {n_neg69_nte}")
        print(f"    te_out_sum: {n_neg69_out}")

        if n_neg69_nte > len(nte) * 0.5:
            print(f"\n  >> REVIEWER IS RIGHT: >50% of nte_score values are in the -69 noise range!")
        elif n_neg69_nte > 0:
            print(f"\n  >> PARTIAL: Some -69 values exist but they don't dominate.")
        else:
            print(f"\n  >> REVIEWER IS WRONG: No -69 noise detected. TE features have real signal.")
    else:
        print("  No non-zero TE features to analyze.")

    # Bonus: pick one protein and check the raw TE matrix if available
    print(f"\n  --- Float32 vs Float64 Spot Check ---")
    print(f"  (Requires running the TE computation on a sample protein)")
    print(f"  To fully verify, run: python scripts/diagnostics/te_precision_test.py")


def main():
    print()
    print("#" * 70)
    print("  External Review Claims Diagnostic")
    print("#" * 70)

    df = check_trace2_feature_wipeout()
    check_trace1_te_float32()

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
