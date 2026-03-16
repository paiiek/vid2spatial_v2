#!/usr/bin/env python3
"""
Train learned distance→(gain, LPF, wet) mapping for vid2spatial.

Data source: text2hoa training data (17,252 labeled samples).
Produces two models:
  - Polynomial (3rd degree) — default, fastest
  - MLP (2-layer, 32 hidden) — slightly more precise

Output: weights/distance_params_v1.npz
"""
import json
import sys
import os
import math
import numpy as np
from numpy.polynomial import polynomial as P
from pathlib import Path

# ── paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DATA_PATH = Path("/home/seung/mmhoa/text2hoa/archive/intermediate_datasets/text2spatial_v3_boosted4.jsonl")
WEIGHTS_DIR = REPO_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
OUT_NPZ = WEIGHTS_DIR / "distance_params_v1.npz"
OUT_PLOT = WEIGHTS_DIR / "distance_model_diagnostic.png"

# ── constants ──
DIST_MIN, DIST_MAX = 0.6, 6.0  # text2hoa data range
LPF_MIN, LPF_MAX = 800.0, 8000.0
SEED = 42
VAL_RATIO = 0.2


def load_data():
    """Load and preprocess text2hoa JSONL data."""
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            r = json.loads(line)
            if "dist_m" not in r or "gain_db" not in r or "wet_mix" not in r:
                continue
            drr = r.get("room_depth", {}).get("drr_db", None)
            records.append({
                "dist_m": float(r["dist_m"]),
                "gain_db": float(r["gain_db"]),
                "wet_mix": float(r["wet_mix"]),
                "drr_db": float(drr) if drr is not None else None,
            })

    print(f"[data] Loaded {len(records)} records from {DATA_PATH.name}")

    dist_m = np.array([r["dist_m"] for r in records])
    d_rel = np.clip((dist_m - DIST_MIN) / (DIST_MAX - DIST_MIN), 0.0, 1.0)
    gain_lin = 10.0 ** (np.array([r["gain_db"] for r in records]) / 20.0)
    wet = np.array([r["wet_mix"] for r in records])

    # DRR (may have None values)
    drr_mask = np.array([r["drr_db"] is not None for r in records])
    drr = np.array([r["drr_db"] if r["drr_db"] is not None else 0.0 for r in records])

    print(f"[data] d_rel: [{d_rel.min():.3f}, {d_rel.max():.3f}]")
    print(f"[data] gain_lin: [{gain_lin.min():.3f}, {gain_lin.max():.3f}], mean={gain_lin.mean():.3f}")
    print(f"[data] wet: [{wet.min():.3f}, {wet.max():.3f}], mean={wet.mean():.3f}")
    print(f"[data] drr_db: [{drr[drr_mask].min():.2f}, {drr[drr_mask].max():.2f}], "
          f"valid={drr_mask.sum()}/{len(drr_mask)}")

    return d_rel, gain_lin, wet, drr, drr_mask


def hardcoded_gain(d_rel):
    return 1.0 - 0.7 * d_rel


def hardcoded_wet(d_rel):
    return 0.05 + 0.30 * d_rel


def hardcoded_lpf(d_rel):
    log_min = math.log(LPF_MIN)
    log_max = math.log(LPF_MAX)
    return np.exp(log_max - (log_max - log_min) * d_rel)


def train_val_split(n, val_ratio=VAL_RATIO, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    val_n = int(n * val_ratio)
    return idx[val_n:], idx[:val_n]


def fit_polynomial(d_rel, gain_lin, wet, drr, drr_mask, train_idx, val_idx):
    """Fit 3rd-degree polynomials for gain, wet, and LPF."""
    print("\n=== Stage A: Polynomial Fit (deg=3) ===")

    d_tr, d_va = d_rel[train_idx], d_rel[val_idx]
    g_tr, g_va = gain_lin[train_idx], gain_lin[val_idx]
    w_tr, w_va = wet[train_idx], wet[val_idx]

    # Fit gain
    gain_coeffs = P.polyfit(d_tr, g_tr, deg=3)
    g_pred_va = np.clip(P.polyval(d_va, gain_coeffs), 0.3, 1.1)
    gain_mae = float(np.mean(np.abs(g_pred_va - g_va)))
    gain_mae_hc = float(np.mean(np.abs(hardcoded_gain(d_va) - g_va)))
    print(f"  gain MAE: poly={gain_mae:.4f}, hardcoded={gain_mae_hc:.4f} "
          f"(delta={100*(gain_mae-gain_mae_hc)/gain_mae_hc:+.1f}%)")

    # Fit wet
    wet_coeffs = P.polyfit(d_tr, w_tr, deg=3)
    w_pred_va = np.clip(P.polyval(d_va, wet_coeffs), 0.0, 1.0)
    wet_mae = float(np.mean(np.abs(w_pred_va - w_va)))
    wet_mae_hc = float(np.mean(np.abs(hardcoded_wet(d_va) - w_va)))
    print(f"  wet  MAE: poly={wet_mae:.4f}, hardcoded={wet_mae_hc:.4f} "
          f"(delta={100*(wet_mae-wet_mae_hc)/wet_mae_hc:+.1f}%)")

    # LPF from DRR curve
    drr_tr_mask = drr_mask[train_idx]
    drr_tr = drr[train_idx][drr_tr_mask]
    d_tr_drr = d_tr[drr_tr_mask]

    drr_coeffs = P.polyfit(d_tr_drr, drr_tr, deg=3)

    # Derive LPF: DRR → LPF cutoff mapping
    d_eval = np.linspace(0, 1, 100)
    drr_curve = P.polyval(d_eval, drr_coeffs)
    drr_min_data = float(drr[drr_mask].min())
    drr_max_data = float(drr[drr_mask].max())
    drr_norm = np.clip((drr_curve - drr_min_data) / (drr_max_data - drr_min_data), 0, 1)
    lpf_curve = LPF_MIN * (LPF_MAX / LPF_MIN) ** drr_norm
    log_lpf_curve = np.log(lpf_curve)
    lpf_log_coeffs = P.polyfit(d_eval, log_lpf_curve, deg=3)

    lpf_pred = np.exp(P.polyval(d_eval, lpf_log_coeffs))
    print(f"  LPF range: [{lpf_pred.min():.0f}, {lpf_pred.max():.0f}] Hz "
          f"(hardcoded: [{LPF_MIN:.0f}, {LPF_MAX:.0f}])")

    return {
        "gain_coeffs": gain_coeffs,
        "wet_coeffs": wet_coeffs,
        "lpf_log_coeffs": lpf_log_coeffs,
        "drr_coeffs": drr_coeffs,
        "val_gain_mae": gain_mae,
        "val_wet_mae": wet_mae,
        "hc_gain_mae": gain_mae_hc,
        "hc_wet_mae": wet_mae_hc,
    }


def train_mlp(d_rel, gain_lin, wet, drr, drr_mask, train_idx, val_idx, poly_result):
    """Train a small 2-layer MLP: d_rel -> (gain, log_lpf, wet).

    Targets are normalized to [0,1] for balanced training, then the
    denormalization is folded into the output layer weights for
    zero-cost inference.
    """
    print("\n=== Stage B: MLP Training ===")
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("  [skip] torch not available, skipping MLP training")
        return None

    # Prepare targets: [gain_lin, log(lpf_hz), wet]
    lpf_log_coeffs = poly_result["lpf_log_coeffs"]
    log_lpf_target = P.polyval(d_rel, lpf_log_coeffs)

    raw_targets = np.stack([gain_lin, log_lpf_target, wet], axis=1).astype(np.float32)

    # Per-column normalization to [0,1] for balanced MSE
    t_min = raw_targets.min(axis=0)
    t_max = raw_targets.max(axis=0)
    t_range = t_max - t_min
    t_range[t_range < 1e-6] = 1.0
    Y_norm = (raw_targets - t_min) / t_range

    X_all = d_rel.reshape(-1, 1).astype(np.float32)
    X_tr = torch.tensor(X_all[train_idx])
    Y_tr = torch.tensor(Y_norm[train_idx])
    X_va = torch.tensor(X_all[val_idx])
    Y_va = torch.tensor(Y_norm[val_idx])

    # Model (predicts in normalized space)
    hidden = 32
    model = nn.Sequential(
        nn.Linear(1, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 3),
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = criterion(pred, Y_tr)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_va)
                val_loss = criterion(val_pred, Y_va).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  epoch {epoch+1:3d}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

    # Load best
    model.load_state_dict(best_state)
    model.eval()

    # Fold denormalization into output layer:
    # original: y_norm = W2 @ h + b2
    # denorm:   y_raw  = y_norm * t_range + t_min
    #                   = (W2 @ h + b2) * t_range + t_min
    #                   = (W2 * t_range) @ h + (b2 * t_range + t_min)
    sd = model.state_dict()
    W2_norm = sd["2.weight"].numpy()  # [3, H]
    b2_norm = sd["2.bias"].numpy()    # [3]
    W2_denorm = W2_norm * t_range[:, None]
    b2_denorm = b2_norm * t_range + t_min

    W1 = sd["0.weight"].numpy()  # [H, 1]
    b1 = sd["0.bias"].numpy()    # [H]

    # Eval on val set (in original scale)
    with torch.no_grad():
        val_norm = model(X_va).numpy()
    val_out = val_norm * t_range + t_min
    mlp_gain_va = np.clip(val_out[:, 0], 0.3, 1.1)
    mlp_wet_va = np.clip(val_out[:, 2], 0.0, 1.0)
    mlp_gain_mae = float(np.mean(np.abs(mlp_gain_va - gain_lin[val_idx])))
    mlp_wet_mae = float(np.mean(np.abs(mlp_wet_va - wet[val_idx])))

    print(f"\n  MLP val gain MAE: {mlp_gain_mae:.4f} (poly={poly_result['val_gain_mae']:.4f})")
    print(f"  MLP val wet  MAE: {mlp_wet_mae:.4f} (poly={poly_result['val_wet_mae']:.4f})")

    return {
        "mlp_W1": W1.T,           # [1, H] for numpy matmul
        "mlp_b1": b1,             # [H]
        "mlp_W2": W2_denorm.T,    # [H, 3] for numpy matmul (denorm folded)
        "mlp_b2": b2_denorm,      # [3] (denorm folded)
        "val_gain_mae": mlp_gain_mae,
        "val_wet_mae": mlp_wet_mae,
    }


def make_diagnostic_plot(d_rel, gain_lin, wet, poly_result, mlp_result=None):
    """Generate diagnostic comparison plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available, skipping plot")
        return

    d_eval = np.linspace(0, 1, 200)

    # Poly predictions
    poly_gain = np.clip(P.polyval(d_eval, poly_result["gain_coeffs"]), 0.3, 1.1)
    poly_wet = np.clip(P.polyval(d_eval, poly_result["wet_coeffs"]), 0.0, 1.0)
    poly_lpf = np.clip(np.exp(P.polyval(d_eval, poly_result["lpf_log_coeffs"])), 200, 20000)

    # Hardcoded
    hc_gain = hardcoded_gain(d_eval)
    hc_wet = hardcoded_wet(d_eval)
    hc_lpf = hardcoded_lpf(d_eval)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Gain
    ax = axes[0]
    ax.scatter(d_rel, gain_lin, alpha=0.03, s=2, color="gray", label="data")
    ax.plot(d_eval, hc_gain, "r--", lw=2, label=f"hardcoded (MAE={poly_result['hc_gain_mae']:.3f})")
    ax.plot(d_eval, poly_gain, "b-", lw=2, label=f"poly-3 (MAE={poly_result['val_gain_mae']:.3f})")
    if mlp_result:
        # Forward pass for plot
        x = d_eval.reshape(-1, 1)
        h = np.maximum(0, x @ mlp_result["mlp_W1"] + mlp_result["mlp_b1"])
        out = h @ mlp_result["mlp_W2"] + mlp_result["mlp_b2"]
        ax.plot(d_eval, np.clip(out[:, 0], 0.3, 1.1), "g-", lw=2,
                label=f"MLP (MAE={mlp_result['val_gain_mae']:.3f})")
    ax.set_xlabel("d_rel (0=near, 1=far)")
    ax.set_ylabel("gain (linear)")
    ax.set_title("Distance → Gain")
    ax.legend(fontsize=8)
    ax.set_ylim(0.2, 1.2)
    ax.grid(True, alpha=0.3)

    # Wet
    ax = axes[1]
    ax.scatter(d_rel, wet, alpha=0.03, s=2, color="gray", label="data")
    ax.plot(d_eval, hc_wet, "r--", lw=2, label=f"hardcoded (MAE={poly_result['hc_wet_mae']:.3f})")
    ax.plot(d_eval, poly_wet, "b-", lw=2, label=f"poly-3 (MAE={poly_result['val_wet_mae']:.3f})")
    if mlp_result:
        ax.plot(d_eval, np.clip(out[:, 2], 0.0, 1.0), "g-", lw=2,
                label=f"MLP (MAE={mlp_result['val_wet_mae']:.3f})")
    ax.set_xlabel("d_rel (0=near, 1=far)")
    ax.set_ylabel("wet mix")
    ax.set_title("Distance → Reverb Wet")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, alpha=0.3)

    # LPF
    ax = axes[2]
    ax.plot(d_eval, hc_lpf, "r--", lw=2, label="hardcoded")
    ax.plot(d_eval, poly_lpf, "b-", lw=2, label="poly-3 (DRR-derived)")
    if mlp_result:
        ax.plot(d_eval, np.clip(np.exp(out[:, 1]), 200, 20000), "g-", lw=2, label="MLP")
    ax.set_xlabel("d_rel (0=near, 1=far)")
    ax.set_ylabel("LPF cutoff (Hz)")
    ax.set_title("Distance → LPF Cutoff")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(200, 20000)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    print(f"\n[plot] Saved diagnostic plot: {OUT_PLOT}")
    plt.close()


def main():
    print("=" * 60)
    print(" Train Distance → Audio Parameter Mapping")
    print("=" * 60)

    # Load data
    d_rel, gain_lin, wet, drr, drr_mask = load_data()
    n = len(d_rel)

    # Train/val split
    train_idx, val_idx = train_val_split(n)
    print(f"\n[split] train={len(train_idx)}, val={len(val_idx)}")

    # Stage A: Polynomial fit
    poly_result = fit_polynomial(d_rel, gain_lin, wet, drr, drr_mask, train_idx, val_idx)

    # Stage B: MLP training
    mlp_result = train_mlp(d_rel, gain_lin, wet, drr, drr_mask, train_idx, val_idx, poly_result)

    # Stage C: Save
    print("\n=== Stage C: Save Weights ===")
    save_dict = {
        "mode": "poly",
        "gain_coeffs": poly_result["gain_coeffs"],
        "wet_coeffs": poly_result["wet_coeffs"],
        "lpf_log_coeffs": poly_result["lpf_log_coeffs"],
        "val_gain_mae": poly_result["val_gain_mae"],
        "val_wet_mae": poly_result["val_wet_mae"],
        "hc_gain_mae": poly_result["hc_gain_mae"],
        "hc_wet_mae": poly_result["hc_wet_mae"],
        "d_rel_range": np.array([DIST_MIN, DIST_MAX]),
        "version": np.array(1),
    }
    if mlp_result:
        save_dict.update({
            "mlp_W1": mlp_result["mlp_W1"],
            "mlp_b1": mlp_result["mlp_b1"],
            "mlp_W2": mlp_result["mlp_W2"],
            "mlp_b2": mlp_result["mlp_b2"],
            "mlp_val_gain_mae": mlp_result["val_gain_mae"],
            "mlp_val_wet_mae": mlp_result["val_wet_mae"],
        })

    np.savez(OUT_NPZ, **save_dict)
    print(f"  Saved: {OUT_NPZ}")
    print(f"  Keys: {list(save_dict.keys())}")

    # Stage D: Diagnostic plot
    make_diagnostic_plot(d_rel, gain_lin, wet, poly_result, mlp_result)

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"{'':20s} {'Hardcoded':>10s} {'Poly-3':>10s}", end="")
    if mlp_result:
        print(f" {'MLP':>10s}", end="")
    print()
    print(f"{'Gain MAE':20s} {poly_result['hc_gain_mae']:10.4f} {poly_result['val_gain_mae']:10.4f}", end="")
    if mlp_result:
        print(f" {mlp_result['val_gain_mae']:10.4f}", end="")
    print()
    print(f"{'Wet MAE':20s} {poly_result['hc_wet_mae']:10.4f} {poly_result['val_wet_mae']:10.4f}", end="")
    if mlp_result:
        print(f" {mlp_result['val_wet_mae']:10.4f}", end="")
    print()
    print()


if __name__ == "__main__":
    main()
