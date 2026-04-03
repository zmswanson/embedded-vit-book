"""Generate plots for pruning + quantization analysis."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import os
import re

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(os.path.dirname(OUT_DIR),
                       "docs", "scalable_vits_for_embedded_systems", "images")
os.makedirs(IMG_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────

pareto = pd.read_csv(os.path.join(OUT_DIR, "pareto_data.csv"))
full_metrics = pd.read_csv(os.path.join(OUT_DIR, "pruned_quantized_full_metrics.csv"))
compression = pd.read_csv(os.path.join(OUT_DIR, "compression_analysis.csv"))


def save(fig, name):
    """Save to both pruning_results/ and docs images/."""
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(IMG_DIR, name.replace(".png", ".pdf")),
                bbox_inches="tight")
    fig.savefig(os.path.join(IMG_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")

# ── Styling ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "font.family": "serif",
})

# ── Color palettes ───────────────────────────────────────────────────────────

# Colors encode the *pruning method*, not the source model
METHOD_COLORS = {
    "Original":       "#888888",
    "Heads 10%":      "#4a90d9",
    "Heads 25%":      "#2ca02c",
    "Heads 50%":      "#ff7f0e",
    "Blocks":         "#d62728",
}

# Markers encode precision
PREC_MARKERS = {"FP32": "o", "FP16": "s", "INT8": "^"}
PREC_SIZES   = {"FP32": 60, "FP16": 60, "INT8": 60}

# Model shapes (marker edge style)
MODEL_EDGE = {
    "deit3_base (LoRA)":        {"edgecolors": "black",  "linewidths": 1.2},
    "deit3_base (KD CVLFace)":  {"edgecolors": "blue",   "linewidths": 1.2},
    "swin_base":                {"edgecolors": "green",  "linewidths": 1.2},
    "swin_base (baseline)":     {"edgecolors": "green",  "linewidths": 1.2},
    "swin_tiny (KD CVLFace)":   {"edgecolors": "red",    "linewidths": 1.2},
}


def classify_method(method_str):
    """Return (method_key, precision) from a pareto method string."""
    m = method_str
    # Determine precision
    if "+ INT8" in m or m == "INT8":
        prec = "INT8"
    elif "+ FP16" in m or m == "FP16":
        prec = "FP16"
    else:
        prec = "FP32"
    # Determine pruning method
    if "Original" in m or m in ("FP16", "INT8"):
        return "Original", prec
    if "50%" in m:
        return "Heads 50%", prec
    if "25%" in m:
        return "Heads 25%", prec
    if "10%" in m:
        return "Heads 10%", prec
    if "block" in m.lower():
        return "Blocks", prec
    return "Original", prec


# ── 1. Pareto front: 2×2 grid per model ──────────────────────────────────────
print("Generating Pareto plots...")

# Map CSV model names to display names and subplot positions
MODEL_PANELS = [
    ("deit3_base (LoRA)",       "DeiT3-B (LoRA)"),
    ("deit3_base (KD CVLFace)", "DeiT3-B (KD CVLFace)"),
    ("swin_base (baseline)",    "Swin-B (Baseline)"),
    ("swin_tiny (KD CVLFace)",  "Swin-T (KD CVLFace)"),
]
# Also match 'swin_base' (without ' (baseline)') for pruned entries
MODEL_ALIASES = {"swin_base": "swin_base (baseline)"}


def make_pareto(x_col, x_label, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    for idx, (model_key, display_name) in enumerate(MODEL_PANELS):
        ax = axes_flat[idx]

        # Filter rows for this model
        model_rows = pareto[
            pareto["model"].apply(
                lambda m: MODEL_ALIASES.get(m, m) == model_key or m == model_key
            )
        ]

        for _, row in model_rows.iterrows():
            mkey, prec = classify_method(row["method"])
            color = METHOD_COLORS[mkey]
            marker = PREC_MARKERS[prec]

            ax.scatter(
                row[x_col], row["rank_at_1"],
                c=color, marker=marker, s=80, alpha=0.85, zorder=3,
                edgecolors="black", linewidths=0.6,
            )

        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel("Rank-1", fontsize=10)
        ax.set_title(display_name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)

    # ─ Shared legend at top ─
    method_handles = [mpatches.Patch(color=c, label=k) for k, c in METHOD_COLORS.items()]
    prec_handles = [mlines.Line2D([], [], marker=PREC_MARKERS[p], color="gray",
                                  linestyle="None", markersize=7, label=p)
                    for p in ["FP32", "FP16", "INT8"]]
    all_handles = method_handles + prec_handles

    fig.legend(handles=all_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), ncol=8,
               framealpha=0.95, fontsize=9, handletextpad=0.4,
               columnspacing=1.0)

    fig.suptitle(title, fontsize=14, y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, filename)


make_pareto("size_MB", "Model Size (MB)",
            "Accuracy vs. Model Size \u2014 Pruned + Quantized",
            "pareto_size_vs_accuracy.png")

make_pareto("compression_ratio", "Compression Ratio (\u00d7)",
            "Accuracy vs. Compression Ratio \u2014 Pruned + Quantized",
            "pareto_compression_vs_accuracy.png")


# ── 2. INT8 quantization impact on pruned models ─────────────────────────────
print("Generating INT8 impact plot...")

# Use full_metrics which has actual per-precision rank@1 from ONNX evaluation
fp32_fm = full_metrics[full_metrics["precision"] == "FP32"].copy()
int8_fm = full_metrics[full_metrics["precision"] == "INT8"].copy()

merged = fp32_fm[["model", "rank@1"]].merge(
    int8_fm[["model", "rank@1"]], on="model", suffixes=("_fp32", "_int8")
)
merged["drop"] = (merged["rank@1_fp32"] - merged["rank@1_int8"]) * 100

# Parse model name into display label and pruning method
def _parse_model_label(m):
    """Convert e.g. 'deit3_base_lora-heads_0.10' to display label."""
    model_map = {
        "deit3_base_lora": "DeiT3-B (LoRA)",
        "deit3_base_kd": "DeiT3-B (KD)",
        "swin_base": "Swin-B",
        "swin_tiny_kd": "Swin-T (KD)",
    }
    for key, display in model_map.items():
        if m.startswith(key + "-"):
            prune_part = m[len(key) + 1:]  # e.g. "heads_0.10" or "blocks_3"
            if prune_part.startswith("heads_"):
                ratio = prune_part.replace("heads_", "")
                prune_label = f"Pruned {int(float(ratio)*100)}% heads"
            elif prune_part.startswith("blocks_"):
                n = prune_part.replace("blocks_", "")
                prune_label = f"Pruned {n} blocks"
            else:
                prune_label = prune_part
            return display, prune_label
    return m, ""

parsed = merged["model"].apply(_parse_model_label)
merged["model_short"] = [p[0] for p in parsed]
merged["prune_method"] = [p[1] for p in parsed]
merged["label"] = merged["model_short"] + "\n" + merged["prune_method"]

# Color by pruning method type
def _method_color(pm):
    if "50%" in pm: return METHOD_COLORS["Heads 50%"]
    if "25%" in pm: return METHOD_COLORS["Heads 25%"]
    if "10%" in pm: return METHOD_COLORS["Heads 10%"]
    if "block" in pm.lower(): return METHOD_COLORS["Blocks"]
    return "#888"

bar_colors = [_method_color(pm) for pm in merged["prune_method"]]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(merged))
bars = ax.bar(x, merged["drop"], color=bar_colors, edgecolor="white", width=0.65)

for bar, val in zip(bars, merged["drop"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(merged["label"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Rank-1 Accuracy Drop (pp)")
ax.set_title("INT8 Quantization Impact on Pruned Models")
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=0, color="black", lw=0.8)
# No legend — labels are self-explanatory

# Ensure enough room above tallest bar for value labels
ymax = merged["drop"].max()
ax.set_ylim(top=ymax * 1.35 if ymax > 0 else 0.5)

fig.tight_layout()
save(fig, "int8_impact_pruned.png")


# ── 3. Pruning method comparison by architecture ─────────────────────────────
print("Generating pruning method comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# -- DeiT3 panel --
deit_data = compression[compression["model"].str.startswith("deit3_base (LoRA)")].copy()
deit_fp32 = deit_data[deit_data["method"].str.endswith("FP32") | (deit_data["method"] == "Original FP32")]

ax = axes[0]
methods = deit_fp32["method"].values
rank1 = deit_fp32["rank_at_1"].values
sizes = deit_fp32["size_MB"].values
bar_colors_d = ["#aec7e8" if "Original" in m else "#1f77b4" for m in methods]

y = np.arange(len(methods))
bars = ax.barh(y, rank1, color=bar_colors_d, edgecolor="white", height=0.6)
for bar, val, sz in zip(bars, rank1, sizes):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f} ({sz:.0f} MB)", va="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels([m.replace(" FP32", "") for m in methods], fontsize=9)
ax.set_xlabel("Rank-1 Identification Rate")
ax.set_title("DeiT3-Base (LoRA)")
ax.set_xlim(0.48, 0.62)
ax.grid(axis="x", alpha=0.3)
ax.axvline(x=0.5834, color="red", ls="--", alpha=0.5, lw=1.2, label="Baseline")
ax.legend(fontsize=9)

# -- Swin panel --
swin_data = compression[
    compression["model"].str.startswith("swin_base")
    & ~compression["model"].str.contains("tiny")
].copy()
swin_fp32 = swin_data[swin_data["method"].str.endswith("FP32") | (swin_data["method"] == "Original FP32")]

ax = axes[1]
methods_s = swin_fp32["method"].values
rank1_s = swin_fp32["rank_at_1"].values
sizes_s = swin_fp32["size_MB"].values
bar_colors_s = ["#a1d99b" if "Original" in m else "#2ca02c" for m in methods_s]

y_s = np.arange(len(methods_s))
bars = ax.barh(y_s, rank1_s, color=bar_colors_s, edgecolor="white", height=0.6)
for bar, val, sz in zip(bars, rank1_s, sizes_s):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f} ({sz:.0f} MB)", va="center", fontsize=9)
ax.set_yticks(y_s)
ax.set_yticklabels([m.replace(" FP32", "") for m in methods_s], fontsize=9)
ax.set_xlabel("Rank-1 Identification Rate")
ax.set_title("Swin-Base")
ax.set_xlim(0.42, 0.62)
ax.grid(axis="x", alpha=0.3)
ax.axvline(x=0.5359, color="red", ls="--", alpha=0.5, lw=1.2, label="Baseline")
ax.legend(fontsize=9)

fig.suptitle("Pruning Method Comparison (FP32)", fontsize=14, y=1.01)
fig.tight_layout()
save(fig, "pruning_method_comparison.png")
# NOTE: DeiT3-B has 5 bars (original + 3 head-pruning ratios + block dropping)
# while Swin-B has 4 bars (original + 10% heads + 25% heads + block dropping).

print("\nAll plots generated successfully.")
