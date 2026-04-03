"""Generate plots for the book chapter from inference results CSVs."""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "scalable_vits_for_embedded_systems", "images")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "inference_results")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def read_csv(fname):
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        return list(csv.DictReader(f))


# ─── helpers ───
PRETTY = {
    "swin_tiny_patch4_window7_224.ms_in1k": "Swin-T",
    "swin_small_patch4_window7_224.ms_in1k": "Swin-S",
    "swin_base_patch4_window7_224.ms_in1k": "Swin-B",
    "deit3_small_patch16_224.fb_in1k": "DeiT3-S",
    "deit3_medium_patch16_224.fb_in1k": "DeiT3-M",
    "deit3_base_patch16_224.fb_in1k": "DeiT3-B",
    "convnext_tiny.fb_in1k": "ConvNeXt-T",
    "convnext_small.fb_in1k": "ConvNeXt-S",
    "convnext_base.fb_in1k": "ConvNeXt-B",
    "pvt_v2_b2.in1k": "PVT-v2-B2",
    "pvt_v2_b3.in1k": "PVT-v2-B3",
    "pvt_v2_b5.in1k": "PVT-v2-B5",
    "mobilevitv2_200.cvnets_in1k": "MobileViT-v2",
    "resnet50d.ra2_in1k": "ResNet-50",
    "resnet101d.ra2_in1k": "ResNet-101",
    "resnet200d.ra2_in1k": "ResNet-200",
}

FAMILY_COLORS = {
    "Swin": "#1f77b4",
    "DeiT3": "#ff7f0e",
    "ConvNeXt": "#2ca02c",
    "PVT": "#9467bd",
    "MobileViT": "#8c564b",
    "ResNet": "#d62728",
}


def family_of(name):
    for prefix in ["swin", "deit3", "convnext", "pvt", "mobilevit", "resnet"]:
        if prefix in name:
            return {"swin": "Swin", "deit3": "DeiT3", "convnext": "ConvNeXt",
                    "pvt": "PVT", "mobilevit": "MobileViT", "resnet": "ResNet"}[prefix]
    return "Other"


PARAM_COUNTS = {
    "Swin-T": 27.5, "Swin-S": 48.8, "Swin-B": 86.7,
    "DeiT3-S": 21.7, "DeiT3-M": 38.3, "DeiT3-B": 85.8,
    "ConvNeXt-T": 27.8, "ConvNeXt-S": 49.5, "ConvNeXt-B": 87.6,
    "PVT-v2-B2": 25.4, "PVT-v2-B3": 45.2, "PVT-v2-B5": 81.4,
    "MobileViT-v2": 17.4,
    "ResNet-50": 23.5, "ResNet-101": 42.5, "ResNet-200": 62.6,
}


# ═══════════════════════════════════════════════════════════════════
# Plot 1: Baseline model comparison — rank@1 bar chart
# ═══════════════════════════════════════════════════════════════════
def plot_baseline_comparison():
    rows = read_csv("model_ablation_inference_results.csv")
    # Deduplicate by wandb_id (keep first), then average if same model appears
    seen = set()
    by_model = {}
    for r in rows:
        wid = r["wandb_id"]
        if wid in seen:
            continue
        seen.add(wid)
        name = r["model_name"]
        rank1 = float(r["tinyface/test/rank@1"])
        by_model.setdefault(name, []).append(rank1)
    # Mean per model
    models = []
    for name, vals in by_model.items():
        pretty = PRETTY.get(name, name)
        models.append((pretty, np.mean(vals), family_of(name)))
    models.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(models))
    colors = [FAMILY_COLORS.get(m[2], "gray") for m in models]
    bars = ax.bar(x, [m[1] for m in models], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models], rotation=40, ha="right")
    ax.set_ylabel("Rank-1 Identification Rate")
    ax.set_ylim(0.42, 0.62)
    ax.set_title("Baseline Model Comparison on TinyFace (96×96)")
    # Add value labels
    for bar, m in zip(bars, models):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{m[1]:.3f}", ha="center", va="bottom", fontsize=7)
    # Legend for families
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, "baseline_comparison.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "baseline_comparison.png"))
    plt.close(fig)
    print("✓ baseline_comparison")


# ═══════════════════════════════════════════════════════════════════
# Plot 2: Baseline vs LoRA comparison (Phase 1 final models)
# ═══════════════════════════════════════════════════════════════════
def plot_baseline_vs_lora():
    # Values from kd_comparison_final.md Table 1 (verified ground truth)
    order = ["Swin-T", "Swin-S", "Swin-B", "DeiT3-S", "DeiT3-M", "DeiT3-B"]
    baselines = {"Swin-T": 0.534, "Swin-S": 0.539, "Swin-B": 0.526,
                 "DeiT3-S": 0.552, "DeiT3-M": 0.570, "DeiT3-B": 0.583}
    loras = {"Swin-T": 0.562, "Swin-S": 0.554, "Swin-B": 0.536,
             "DeiT3-S": 0.527, "DeiT3-M": 0.533, "DeiT3-B": 0.550}

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(order))
    w = 0.35
    b_vals = [baselines[o] for o in order]
    l_vals = [loras.get(o, 0) for o in order]
    bars1 = ax.bar(x - w / 2, b_vals, w, label="Baseline", color="#4c72b0", edgecolor="white")
    bars2 = ax.bar(x + w / 2, l_vals, w, label="LoRA", color="#dd8452", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Rank-1 Identification Rate")
    ax.set_ylim(0.50, 0.60)
    ax.set_title("Baseline vs. LoRA Fine-Tuning on TinyFace")
    ax.legend()
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, "baseline_vs_lora.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "baseline_vs_lora.png"))
    plt.close(fig)
    print("✓ baseline_vs_lora")


# ═══════════════════════════════════════════════════════════════════
# Plot 3: KD Alpha ablation curves — 4 configs
# ═══════════════════════════════════════════════════════════════════
def plot_alpha_ablation():
    rows = read_csv("kd_alpha_ablation_results.csv")

    # Parse model names to identify configs
    configs = {
        "A": {"label": "Swin-T Direct + CVLFace frozen", "color": "#1f77b4", "marker": "o"},
        "B": {"label": "Swin-T LoRA + PETALface frozen", "color": "#ff7f0e", "marker": "s"},
        "C": {"label": "DeiT3-B Direct + CVLFace FT", "color": "#2ca02c", "marker": "^"},
        "D": {"label": "DeiT3-B Direct + PETALface frozen", "color": "#d62728", "marker": "D"},
    }

    # Assign configs based on model name patterns in the ablation results
    # Config naming from kd_alpha_ablation.md:
    # A: swin_tiny Direct + CVLFace frozen  (8 runs)
    # B: swin_tiny LoRA + PETALface frozen  (8 runs)
    # C: deit3_base Direct + CVLFace FT     (8 runs)
    # D: deit3_base Direct + PETALface frozen (8 runs)

    # Alpha values for each config (8 values each)
    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]

    # We know the order from the ablation results md
    config_data = {
        "A": [0.616, 0.568, 0.546, 0.548, 0.549, 0.553, 0.547, 0.527],
        "B": [0.450, 0.581, 0.571, 0.562, 0.567, 0.565, 0.565, 0.567],
        "C": [0.572, 0.593, 0.579, 0.575, 0.581, 0.576, 0.573, 0.584],
        "D": [0.416, 0.586, 0.573, 0.572, 0.576, 0.580, 0.579, 0.581],
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    for cfg_key, vals in config_data.items():
        cfg = configs[cfg_key]
        ax.plot(alphas, vals, marker=cfg["marker"], color=cfg["color"],
                label=f"{cfg_key}: {cfg['label']}", linewidth=1.5, markersize=5)

    ax.set_xlabel(r"$\alpha$ (Hard Loss Weight)")
    ax.set_ylabel("Rank-1 Identification Rate")
    ax.set_title(r"KD Loss Mixing ($\alpha$) Ablation", pad=40)
    ax.set_xticks(alphas)
    ax.legend(fontsize=8, loc="center", bbox_to_anchor=(0.5, 1.07),
              ncol=2, frameon=True, fancybox=False, edgecolor="0.8")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.39, 0.63)
    fig.savefig(os.path.join(OUT_DIR, "alpha_ablation.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "alpha_ablation.png"))
    plt.close(fig)
    print("✓ alpha_ablation")


# ═══════════════════════════════════════════════════════════════════
# Plot 4: KD effect — best student vs teacher vs baseline
# ═══════════════════════════════════════════════════════════════════
def plot_kd_overview():
    # Teacher baselines
    teachers = {
        "CVLFace (frozen)": 0.716,
        "CVLFace (FT)": 0.781,
        "PETALface (frozen)": 0.424,
        "PETALface (FT)": 0.663,
    }

    # Best Phase 1 and KD results per model from kd_comparison_final.md Table 4
    models = ["Swin-T", "Swin-S", "Swin-B", "DeiT3-S", "DeiT3-M", "DeiT3-B"]
    phase1_best = [0.562, 0.554, 0.536, 0.552, 0.570, 0.583]
    kd_best = [0.567, 0.559, 0.536, 0.543, 0.559, 0.581]
    # Alpha ablation found even better for some:
    # Config A swin_tiny α=0.0 → 0.616, Config C deit3_base α=0.1 → 0.593
    kd_best_alpha = [0.616, 0.559, 0.536, 0.543, 0.559, 0.593]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models))
    w = 0.25
    ax.bar(x - w, phase1_best, w, label="Best Phase 1 (Baseline/LoRA)", color="#4c72b0", edgecolor="white")
    ax.bar(x, kd_best, w, label=r"Best KD ($\alpha$=0.7)", color="#dd8452", edgecolor="white")
    ax.bar(x + w, kd_best_alpha, w, label=r"Best KD (optimal $\alpha$)", color="#55a868", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Rank-1 Identification Rate")
    ax.set_ylim(0.50, 0.65)
    ax.set_title("Knowledge Distillation Impact on Student Models", pad=30)
    ax.legend(fontsize=9, loc="center", bbox_to_anchor=(0.5, 1.05),
              ncol=3, frameon=True, fancybox=False, edgecolor="0.8")
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(os.path.join(OUT_DIR, "kd_overview.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "kd_overview.png"))
    plt.close(fig)
    print("✓ kd_overview")


# ═══════════════════════════════════════════════════════════════════
# Plot 5: Parameter count vs Rank-1 scatter
# ═══════════════════════════════════════════════════════════════════
def plot_params_vs_accuracy():
    rows = read_csv("model_ablation_inference_results.csv")
    seen = set()
    by_model = {}
    for r in rows:
        wid = r["wandb_id"]
        if wid in seen:
            continue
        seen.add(wid)
        name = r["model_name"]
        pretty = PRETTY.get(name, name)
        rank1 = float(r["tinyface/test/rank@1"])
        by_model.setdefault(pretty, []).append(rank1)

    fig, ax = plt.subplots(figsize=(6, 4))
    for pretty, vals in by_model.items():
        params = PARAM_COUNTS.get(pretty, None)
        if params is None:
            continue
        fam = family_of(next(k for k, v in PRETTY.items() if v == pretty))
        color = FAMILY_COLORS.get(fam, "gray")
        avg = np.mean(vals)
        ax.scatter(params, avg, c=color, s=50, zorder=5, edgecolors="white", linewidths=0.5)

    # Add teacher models
    teacher_data = {
        "CVLFace (frozen)": (114.9, 0.716),
        "CVLFace (FT)": (114.9, 0.781),
        "PETALface (frozen)": (213.8, 0.424),
        "PETALface (FT)": (213.8, 0.663),
    }
    for label, (params, rank1) in teacher_data.items():
        ax.scatter(params, rank1, c="purple", s=80, zorder=6, marker="*",
                   edgecolors="white", linewidths=0.5)
        if "PETALface" in label:
            ax.annotate(label, (params, rank1), textcoords="offset points",
                        xytext=(-8, 0), fontsize=9, color="purple",
                        fontweight="bold", ha="right", va="center")
        else:
            ax.annotate(label, (params, rank1), textcoords="offset points",
                        xytext=(8, 0), fontsize=9, color="purple",
                        fontweight="bold", ha="left", va="center")

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Rank-1 Identification Rate")
    ax.set_title("Model Size vs. Accuracy on TinyFace")
    ax.grid(alpha=0.3)
    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="purple",
                          markersize=10, label="Teacher"))
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    fig.savefig(os.path.join(OUT_DIR, "params_vs_accuracy.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "params_vs_accuracy.png"))
    plt.close(fig)
    print("✓ params_vs_accuracy")


# ═══════════════════════════════════════════════════════════════════
# Plot 6: CMC Curve comparison of top models
# ═══════════════════════════════════════════════════════════════════
def plot_cmc_curves():
    rows = read_csv("final_models_inference_results.csv")
    ablation = read_csv("model_ablation_inference_results.csv")

    # Known wandb IDs from experiments:
    # LoRA runs are first 6, Baseline runs are second 6 in final_models CSV
    # DeiT3-B baseline: vimriwcl (rank@1=0.583), LoRA: 2ljjzfzb (0.550)
    # Swin-T LoRA: f0frj93y (0.562), baseline: p09btsx3 (0.534)
    target_ids = {
        "DeiT3-B Baseline": "vimriwcl",
        "DeiT3-B LoRA": "2ljjzfzb",
        "Swin-T Baseline": "p09btsx3",
        "Swin-T LoRA": "f0frj93y",
    }
    targets = {k: None for k in target_ids}
    for r in rows:
        for label, wid in target_ids.items():
            if r["wandb_id"] == wid:
                targets[label] = r

    # ResNet-50 from ablation
    resnet_row = None
    seen = set()
    for r in ablation:
        if r["wandb_id"] not in seen and "resnet50" in r["model_name"]:
            resnet_row = r
            seen.add(r["wandb_id"])

    ranks = list(range(1, 11))
    rank_keys = [f"tinyface/test/rank@{i}" for i in ranks]

    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {
        "DeiT3-B Baseline": {"color": "#ff7f0e", "ls": "-", "marker": "o"},
        "DeiT3-B LoRA": {"color": "#ff7f0e", "ls": "--", "marker": "s"},
        "Swin-T Baseline": {"color": "#1f77b4", "ls": "-", "marker": "^"},
        "Swin-T LoRA": {"color": "#1f77b4", "ls": "--", "marker": "D"},
    }

    for label, row in targets.items():
        if row is None:
            continue
        vals = [float(row[k]) for k in rank_keys]
        st = styles[label]
        ax.plot(ranks, vals, marker=st["marker"], color=st["color"],
                linestyle=st["ls"], label=label, linewidth=1.5, markersize=4)

    if resnet_row:
        vals = [float(resnet_row[k]) for k in rank_keys]
        ax.plot(ranks, vals, marker="x", color="#d62728", linestyle=":",
                label="ResNet-50", linewidth=1.5, markersize=5)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Identification Rate")
    ax.set_title("CMC Curves \u2014 Top Models on TinyFace", pad=45)
    ax.set_xticks(ranks)
    ax.legend(fontsize=9, loc="center", bbox_to_anchor=(0.5, 1.08),
              ncol=3, frameon=True, fancybox=False, edgecolor="0.8")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, "cmc_curves.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "cmc_curves.png"))
    plt.close(fig)
    print("✓ cmc_curves")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_baseline_comparison()
    plot_baseline_vs_lora()
    plot_alpha_ablation()
    plot_kd_overview()
    plot_params_vs_accuracy()
    plot_cmc_curves()
    print("\nAll plots saved to", OUT_DIR)
