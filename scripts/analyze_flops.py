#!/usr/bin/env python3
"""
Generate FLOPs analysis plots and a LaTeX table fragment from the sweep CSVs.

Inputs:
  inference_results/flops/flops_total.csv
  inference_results/flops/flops_by_module.csv

Outputs:
  docs/scalable_vits_for_embedded_systems/figures/flops_vs_resolution.pdf
  docs/scalable_vits_for_embedded_systems/figures/flops_breakdown_at_96.pdf
  docs/scalable_vits_for_embedded_systems/figures/flops_breakdown_at_224.pdf
  docs/scalable_vits_for_embedded_systems/figures/flops_top_modules_table.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Per-family panel for the small-multiples scaling plot.
# Each entry: (panel title, [(timm name, short label)])
FAMILY_PANELS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Swin v1", [
        ("swin_tiny_patch4_window7_224",  "Swin-T"),
        ("swin_small_patch4_window7_224", "Swin-S"),
        ("swin_base_patch4_window7_224",  "Swin-B"),
    ]),
    ("DeiT-3", [
        ("deit3_small_patch16_224",  "DeiT3-S"),
        ("deit3_medium_patch16_224", "DeiT3-M"),
        ("deit3_base_patch16_224",   "DeiT3-B"),
    ]),
    ("ConvNeXt", [
        ("convnext_tiny",  "ConvNeXt-T"),
        ("convnext_small", "ConvNeXt-S"),
        ("convnext_base",  "ConvNeXt-B"),
    ]),
    ("PVT v2", [
        ("pvt_v2_b2", "PVTv2-B2"),
        ("pvt_v2_b3", "PVTv2-B3"),
        ("pvt_v2_b5", "PVTv2-B5"),
    ]),
    ("ResNet-D", [
        ("resnet50d",  "ResNet-50d"),
        ("resnet101d", "ResNet-101d"),
        ("resnet200d", "ResNet-200d"),
    ]),
    ("MobileViT v2", [
        ("mobilevitv2_200", "MobileViTv2-200"),
    ]),
]

# Display order + short labels for the breakdown bars (16 models)
MODEL_DISPLAY = [
    ("swin_tiny_patch4_window7_224",  "Swin-T"),
    ("swin_small_patch4_window7_224", "Swin-S"),
    ("swin_base_patch4_window7_224",  "Swin-B"),
    ("deit3_small_patch16_224",       "DeiT3-S"),
    ("deit3_medium_patch16_224",      "DeiT3-M"),
    ("deit3_base_patch16_224",        "DeiT3-B"),
    ("convnext_tiny",                 "ConvNeXt-T"),
    ("convnext_small",                "ConvNeXt-S"),
    ("convnext_base",                 "ConvNeXt-B"),
    ("pvt_v2_b2",                     "PVTv2-B2"),
    ("pvt_v2_b3",                     "PVTv2-B3"),
    ("pvt_v2_b5",                     "PVTv2-B5"),
    ("resnet50d",                     "ResNet-50d"),
    ("resnet101d",                    "ResNet-101d"),
    ("resnet200d",                    "ResNet-200d"),
    ("mobilevitv2_200",               "MobileViTv2-200"),
]

# Stable category order and palette
CATEGORY_ORDER = [
    "ffn_mlp",
    "attention_qkv",
    "attention_proj",
    "attention_other",
    "conv",
    "downsample",
    "patch_embed",
    "norm",
    "head",
    "other",
]
CATEGORY_LABEL = {
    "ffn_mlp":         "FFN / MLP",
    "attention_qkv":   "Attn QKV",
    "attention_proj":  "Attn Proj",
    "attention_other": "Attn other",
    "conv":            "Conv",
    "downsample":      "Downsample",
    "patch_embed":     "Patch / stem",
    "norm":            "Norm",
    "head":            "Head",
    "other":           "Other",
}
CATEGORY_COLORS = {
    "ffn_mlp":         "#1f77b4",
    "attention_qkv":   "#d62728",
    "attention_proj":  "#ff7f0e",
    "attention_other": "#9467bd",
    "conv":            "#2ca02c",
    "downsample":      "#8c564b",
    "patch_embed":     "#17becf",
    "norm":            "#bcbd22",
    "head":            "#7f7f7f",
    "other":           "#c7c7c7",
}


def plot_flops_vs_resolution(df_total: pd.DataFrame, out: Path) -> None:
    """Small-multiples: one panel per architecture family, linear axes.

    Linear axes make the quadratic (in resolution) / linear (in pixels) growth
    visible directly as a parabola; per-family panels prevent the four mid-size
    families from overlapping into a single mush at $224\\times224$.
    """
    ok = df_total[df_total["status"] == "ok"].copy()

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.4), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # One color per panel for visual cohesion across the figure
    panel_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    variant_styles = [("o", "-", 2.2), ("s", "--", 2.0), ("^", ":", 2.0)]

    for ax, (panel_title, members), color in zip(axes_flat, FAMILY_PANELS, panel_colors):
        for (model, label), (marker, ls, lw) in zip(members, variant_styles):
            sub = ok[ok["model"] == model].sort_values("resolution")
            if sub.empty:
                continue
            x = sub["resolution"].to_numpy(dtype=float)
            y = sub["gmacs"].to_numpy(dtype=float)
            ax.plot(x, y, marker=marker, linestyle=ls, linewidth=lw,
                    markersize=7.0, color=color, label=label)

        ax.set_title(panel_title, fontsize=15, pad=4)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.65)
        ax.legend(loc="upper left", fontsize=12, frameon=False, handlelength=2.4)
        ax.set_xlim(0, 1050)
        # Slight negative y-floor so the sub-1-GMAC points at low resolutions
        # lift off the x-axis instead of overlapping it.
        ax.set_ylim(-15, 360)
        ax.tick_params(axis="both", labelsize=12)

    # Shared axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel("Input resolution  $H = W$  (pixels)", fontsize=13)
    for ax in axes[:, 0]:
        ax.set_ylabel("GMACs", fontsize=13)

    fig.suptitle("Inference cost vs. input resolution  (batch=1, FP32, "
                 "GMACs from fvcore)", fontsize=15, y=1.0)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _stacked_breakdown(df_mod: pd.DataFrame, df_total: pd.DataFrame,
                       resolution: int, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.6))

    n = len(MODEL_DISPLAY)
    y_pos = np.arange(n)
    totals = []

    # Build a [n_models x n_categories] matrix in GMACs
    mat = np.zeros((n, len(CATEGORY_ORDER)), dtype=float)
    for i, (model, _label) in enumerate(MODEL_DISPLAY):
        sub = df_mod[(df_mod["model"] == model) & (df_mod["resolution"] == resolution)]
        if sub.empty:
            totals.append(float("nan"))
            continue
        for j, cat in enumerate(CATEGORY_ORDER):
            row = sub[sub["category"] == cat]
            mat[i, j] = float(row["gmacs"].sum()) if not row.empty else 0.0
        tot_row = df_total[(df_total["model"] == model)
                           & (df_total["resolution"] == resolution)
                           & (df_total["status"] == "ok")]
        totals.append(float(tot_row["gmacs"].iloc[0]) if not tot_row.empty
                      else float("nan"))

    # Plot as fraction of total so bars are comparable across models
    pct = np.zeros_like(mat)
    for i in range(n):
        s = mat[i].sum()
        if s > 0:
            pct[i] = 100.0 * mat[i] / s

    left = np.zeros(n)
    for j, cat in enumerate(CATEGORY_ORDER):
        widths = pct[:, j]
        if not np.any(widths > 0.05):
            continue
        ax.barh(y_pos, widths, left=left,
                color=CATEGORY_COLORS[cat], label=CATEGORY_LABEL[cat],
                edgecolor="white", linewidth=0.4)
        left += widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels([lab for _, lab in MODEL_DISPLAY], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Share of inference GMACs (%)", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="x", labelsize=11)

    # Annotate total GMACs at the right edge of each bar
    for i, t in enumerate(totals):
        if t == t:
            ax.text(101, i, f"{t:.2f} G", va="center", ha="left", fontsize=10,
                    color="#444")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22),
              ncol=5, fontsize=11, frameon=False)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def write_top_modules_table(df_mod: pd.DataFrame, df_total: pd.DataFrame,
                            resolution: int, out: Path) -> None:
    """Top-3 categories per model at the given resolution, as a LaTeX fragment."""
    rows = []
    for model, label in MODEL_DISPLAY:
        sub = df_mod[(df_mod["model"] == model) & (df_mod["resolution"] == resolution)]
        if sub.empty:
            continue
        tot_row = df_total[(df_total["model"] == model)
                           & (df_total["resolution"] == resolution)
                           & (df_total["status"] == "ok")]
        if tot_row.empty:
            continue
        gmacs_total = float(tot_row["gmacs"].iloc[0])
        top = (sub.sort_values("gmacs", ascending=False)
                  .drop_duplicates("category")
                  .head(3))
        cells = []
        for _, r in top.iterrows():
            cells.append(f"{CATEGORY_LABEL.get(r['category'], r['category'])} "
                         f"({float(r['pct_of_total']):.0f}\\%)")
        while len(cells) < 3:
            cells.append("---")
        rows.append((label, gmacs_total, cells))

    lines = []
    lines.append(f"% Auto-generated by scripts/analyze_flops.py at {resolution}x{resolution}")
    lines.append(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l r l l l}")
    lines.append(r"\toprule")
    lines.append(r"Model & GMACs & 1st contributor & 2nd contributor & 3rd contributor \\")
    lines.append(r"\midrule")
    for label, g, cells in rows:
        lines.append(f"{label} & {g:.2f} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="inference_results/flops")
    ap.add_argument("--out-dir",
                    default="docs/scalable_vits_for_embedded_systems/figures")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_total = pd.read_csv(in_dir / "flops_total.csv")
    df_mod = pd.read_csv(in_dir / "flops_by_module.csv")

    plot_flops_vs_resolution(df_total, out_dir / "flops_vs_resolution.pdf")
    _stacked_breakdown(df_mod, df_total, 96,
                       out_dir / "flops_breakdown_at_96.pdf",
                       "GMACs breakdown at $96\\times96$ (TinyFace deployment)")
    write_top_modules_table(df_mod, df_total, 96,
                            out_dir / "flops_top_modules_table.tex")


if __name__ == "__main__":
    main()
