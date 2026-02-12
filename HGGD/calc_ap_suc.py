#!/usr/bin/env python3
import os
import glob
import csv
import json
import numpy as np

# Mu values used by GraspNet evaluation (must match last dim M in res)
MU = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

# ---- Config ----
OUT_DIR = "dumps"
os.makedirs(OUT_DIR, exist_ok=True)

# You asked specifically for this filename:
RES_PATH_EXACT = "temp_result_scene160_189.npy"

# Outputs (ONLY these three will be written)
CSV_AP_OVERALL = os.path.join(OUT_DIR, "ap_overall.csv")
CSV_S1_OVERALL = os.path.join(OUT_DIR, "success_at_1_overall.csv")
JSON_SUMMARY   = os.path.join(OUT_DIR, "summary.json")


def find_res_path():
    if os.path.exists(RES_PATH_EXACT):
        return RES_PATH_EXACT

    # fallback: try any temp_result*.npy (but warn)
    cands = sorted(glob.glob("temp_result*.npy"))
    if not cands:
        print("ERROR: Could not find:", RES_PATH_EXACT)
        print("Also no temp_result*.npy found in current directory.")
        raise FileNotFoundError("No temp_result*.npy found.")
    print("WARNING: Exact file not found:", RES_PATH_EXACT)
    print("Using fallback candidate:", cands[0])
    return cands[0]


def main():
    res_path = find_res_path()
    print("Using RES_PATH:", res_path)

    res = np.load(res_path)  # expected shape (S, V, K, M)
    print("res shape:", res.shape)

    if res.ndim != 4:
        raise ValueError(f"Expected res.ndim == 4 (S,V,K,M). Got {res.ndim} dims: {res.shape}")

    S, V, K, M = res.shape
    if M != len(MU):
        raise ValueError(f"Expected M={len(MU)} mu values, got M={M}. res shape={res.shape}")
    if K < 1:
        raise ValueError(f"Expected K>=1 (topK dimension), got K={K}. res shape={res.shape}")

    # ---------------------------
    # AP metrics from full (S,V,K,M)
    # ---------------------------
    ap_mu = res.mean(axis=(0, 1, 2))  # (M,)
    ap = float(ap_mu.mean())          # mean over mu

    # ---------------------------
    # Success@1 from top1 only: res[:,:,0,:] -> (S,V,M)
    # ---------------------------
    top1 = res[:, :, 0, :]  # (S, V, M)

    # Binarize (safe if slightly non-binary)
    top1_bin = (top1 > 0.5).astype(np.float32)

    s1_overall_mu = top1_bin.mean(axis=(0, 1))  # (M,)

    # ---------------------------
    # Save AP overall
    # ---------------------------
    with open(CSV_AP_OVERALL, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["AP_mean_over_mu", ap])
        for mu, val in zip(MU, ap_mu):
            w.writerow([f"AP@mu_{mu}", float(val)])
    print("Saved:", CSV_AP_OVERALL)

    # ---------------------------
    # Save Success@1 overall
    # ---------------------------
    with open(CSV_S1_OVERALL, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mu", "success_at_1_overall"])
        for mu, val in zip(MU, s1_overall_mu):
            w.writerow([mu, float(val)])
    print("Saved:", CSV_S1_OVERALL)

    # ---------------------------
    # Save JSON summary
    # ---------------------------
    summary = {
        "res_path": res_path,
        "res_shape": [int(x) for x in res.shape],
        "mu": MU,
        "AP_mean_over_mu": ap,
        "AP_at_mu": {str(mu): float(v) for mu, v in zip(MU, ap_mu)},
        "SuccessAt1_overall_at_mu": {str(mu): float(v) for mu, v in zip(MU, s1_overall_mu)},
        "num_scenes": int(S),
        "num_views": int(V),
        "topK": int(K),
    }
    with open(JSON_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", JSON_SUMMARY)

    # ---------------------------
    # Print key results
    # ---------------------------
    print("\n=== OVERALL AP RESULTS ===")
    print(f"AP (mean over mu): {ap:.6f}")
    for mu, val in zip(MU, ap_mu):
        print(f"AP@{mu:.1f}: {float(val):.6f}")

    print("\n=== OVERALL SUCCESS@1 RESULTS ===")
    for mu, val in zip(MU, s1_overall_mu):
        print(f"Success@1 @ {mu:.1f}: {float(val):.6f}")


if __name__ == "__main__":
    main()
