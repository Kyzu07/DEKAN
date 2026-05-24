#!/usr/bin/env python3
"""
Visualize MTST components from saved NumPy arrays (first 96 points).
Expected files in --dir:
  - z_orig_first96.npy
  - trend_orig_first96.npy
  - seasonal_orig_first96.npy
  - residual_orig_first96.npy
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_1d(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    arr = np.load(path)
    if arr.ndim != 1:
        # tolerate shapes like (96,) or (96,1) or (1,96)
        arr = arr.squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array in {path}, got shape {arr.shape}")
    return arr


def main():
    parser = argparse.ArgumentParser(description="Plot MTST original-scale components")
    parser.add_argument("--dir", type=str, default="",
                        help="Directory containing *_first96.npy files")
    parser.add_argument("--prefix", type=str, default="",
                        help="Optional prefix for output filenames (e.g., run1_)")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively")
    args = parser.parse_args()

    base = args.dir
    z_path        = os.path.join(base, "z_orig_first96.npy")
    trend_path    = os.path.join(base, "trend_orig_first96.npy")
    seasonal_path = os.path.join(base, "seasonal_orig_first96.npy")
    residual_path = os.path.join(base, "residual_orig_first96.npy")

    z        = load_1d(z_path)
    trend    = load_1d(trend_path)
    seasonal = load_1d(seasonal_path)
    residual = load_1d(residual_path)

    # x-axis as time index
    t = np.arange(z.shape[0])

    # -----------------------
    # 1) Overlay figure
    # -----------------------
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(t, z,        label="Original (first var, first sample)")
    ax1.plot(t, trend,    label="Trend")
    ax1.plot(t, seasonal, label="Seasonality")
    ax1.plot(t, residual, label="Residual")
    ax1.set_xlabel("Time (index)")
    ax1.set_ylabel("Value")
    ax1.set_title("MTST components (original scale) — overlay")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=False)

    out_overlay_png = f"{args.prefix}mtst_components_overlay.png"
    out_overlay_pdf = f"{args.prefix}mtst_components_overlay.pdf"
    fig1.tight_layout()
    fig1.savefig(out_overlay_png, dpi=300)
    fig1.savefig(out_overlay_pdf)

    # -----------------------
    # 2) Decomposition-style figure (stacked)
    # -----------------------
    fig2, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, z);         axes[0].set_ylabel("Original")
    axes[1].plot(t, trend);     axes[1].set_ylabel("Trend")
    axes[2].plot(t, seasonal);  axes[2].set_ylabel("Seasonality")
    axes[3].plot(t, residual);  axes[3].set_ylabel("Residual"); axes[3].set_xlabel("Time (index)")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig2.suptitle("MTST components (original scale) — decomposition view")
    fig2.tight_layout(rect=[0, 0, 1, 0.97])

    out_decomp_png = f"{args.prefix}mtst_components_decomposition.png"
    out_decomp_pdf = f"{args.prefix}mtst_components_decomposition.pdf"
    fig2.savefig(out_decomp_png, dpi=300)
    fig2.savefig(out_decomp_pdf)

    print(f"Saved:\n  {out_overlay_png}\n  {out_overlay_pdf}\n  {out_decomp_png}\n  {out_decomp_pdf}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
