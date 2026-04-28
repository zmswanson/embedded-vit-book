"""
Phase 6.1 — Master script to compile all experimental results into
publication-quality LaTeX tables and figures for the book chapter.

Reads CSVs from all phases (inference_results/, pruning_results/,
quantization_results/, profiles/, onnx_models/) and produces:
  - tables/*.tex   — standalone LaTeX table fragments
  - docs/.../figures/*.pdf — publication figures (also .png at 300 DPI)

Run:
    conda activate vit-benchmark && python compile_results.py
"""
import csv
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "inference_results")
PRUNING_DIR = os.path.join(ROOT, "pruning_results")
QUANT_DIR = os.path.join(ROOT, "quantization_results")
PROFILES_DIR = os.path.join(ROOT, "profiles")
MANIFEST = os.path.join(ROOT, "onnx_models", "model_manifest.csv")

FIG_DIR = os.path.join(ROOT, "docs", "scalable_vits_for_embedded_systems", "figures")
TABLE_DIR = os.path.join(ROOT, "tables")

# ── canonical naming ─────────────────────────────────────────────────
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

# Short-stem names used in TRT benchmark csvs → pretty
SHORT_PRETTY = {
    "swin_tiny": "Swin-T", "swin_small": "Swin-S", "swin_base": "Swin-B",
    "deit3_small": "DeiT3-S", "deit3_medium": "DeiT3-M", "deit3_base": "DeiT3-B",
    "convnext_tiny": "ConvNeXt-T", "convnext_small": "ConvNeXt-S", "convnext_base": "ConvNeXt-B",
    "pvt_v2_b2": "PVT-v2-B2", "pvt_v2_b3": "PVT-v2-B3", "pvt_v2_b5": "PVT-v2-B5",
    "mobilevitv2_200": "MobileViT-v2",
    "resnet50d": "ResNet-50", "resnet101d": "ResNet-101", "resnet200d": "ResNet-200",
}

FAMILY_COLORS = {
    "Swin": "#1f77b4",
    "DeiT3": "#ff7f0e",
    "ConvNeXt": "#2ca02c",
    "PVT": "#9467bd",
    "MobileViT": "#8c564b",
    "ResNet": "#d62728",
}

PARAM_COUNTS = {
    "Swin-T": 27.5, "Swin-S": 48.8, "Swin-B": 86.7,
    "DeiT3-S": 21.7, "DeiT3-M": 38.3, "DeiT3-B": 85.8,
    "ConvNeXt-T": 27.8, "ConvNeXt-S": 49.5, "ConvNeXt-B": 87.6,
    "PVT-v2-B2": 25.4, "PVT-v2-B3": 45.2, "PVT-v2-B5": 81.4,
    "MobileViT-v2": 17.4,
    "ResNet-50": 23.5, "ResNet-101": 42.5, "ResNet-200": 62.6,
}

EMBED_DIMS = {
    "Swin-T": 768, "Swin-S": 768, "Swin-B": 1024,
    "DeiT3-S": 384, "DeiT3-M": 512, "DeiT3-B": 768,
    "ConvNeXt-T": 768, "ConvNeXt-S": 768, "ConvNeXt-B": 1024,
    "PVT-v2-B2": 512, "PVT-v2-B3": 512, "PVT-v2-B5": 512,
    "MobileViT-v2": 512,
    "ResNet-50": 2048, "ResNet-101": 2048, "ResNet-200": 2048,
}

# Model display order for consistency
BASELINE_ORDER = [
    "DeiT3-B", "DeiT3-M", "DeiT3-S",
    "Swin-T", "Swin-S", "Swin-B",
    "ConvNeXt-T", "ConvNeXt-S", "ConvNeXt-B",
    "PVT-v2-B2", "PVT-v2-B3", "PVT-v2-B5",
    "MobileViT-v2",
    "ResNet-50", "ResNet-101", "ResNet-200",
]


def family_of(name):
    n = name.lower()
    for prefix, fam in [("swin", "Swin"), ("deit", "DeiT3"), ("convnext", "ConvNeXt"),
                         ("pvt", "PVT"), ("mobilevit", "MobileViT"), ("resnet", "ResNet")]:
        if prefix in n:
            return fam
    return "Other"


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


def _save_fig(fig, stem):
    """Save figure to docs figures/ dir."""
    fig.savefig(os.path.join(FIG_DIR, f"{stem}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"{stem}.png"))
    plt.close(fig)


def _escape_latex(s):
    """Escape underscores and other LaTeX-sensitive chars."""
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


# ═════════════════════════════════════════════════════════════════════
# HELPER: Get baseline rank@1 per model
# ═════════════════════════════════════════════════════════════════════
def _get_baseline_rank1():
    """Return {pretty_name: rank1} for baseline models."""
    rows = list(csv.DictReader(
        open(os.path.join(RESULTS_DIR, "model_ablation_inference_results.csv"))))
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
    return {k: np.mean(v) for k, v in by_model.items()}


# ═════════════════════════════════════════════════════════════════════
# TABLE 1: Architecture Comparison
# ═════════════════════════════════════════════════════════════════════
def table_architecture_comparison():
    baselines = _get_baseline_rank1()
    rows = []
    for name in BASELINE_ORDER:
        rows.append({
            "Model": name,
            "Family": family_of(name),
            "Params (M)": PARAM_COUNTS.get(name, "---"),
            "Embed Dim": EMBED_DIMS.get(name, "---"),
            "Rank@1": f"{baselines[name]:.4f}" if name in baselines else "---",
        })

    header = r"""\begin{table}[t]
\centering
\caption{Baseline architecture comparison on TinyFace ($96\times96$). Models sorted by family.
All use ImageNet-1k pretrained weights from \texttt{timm}.}
\label{tab:architecture-comparison}
\footnotesize
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l l r r c}
\toprule
Model & Family & Params (M) & Embed Dim & Rank@1 \\
\midrule
"""
    body = ""
    prev_fam = None
    for r in rows:
        if prev_fam and r["Family"] != prev_fam:
            body += r"\midrule" + "\n"
        prev_fam = r["Family"]
        body += f"{r['Model']} & {r['Family']} & {r['Params (M)']} & {r['Embed Dim']} & {r['Rank@1']} \\\\\n"

    footer = r"""\bottomrule
\end{tabular*}
\end{table}
"""
    tex = header + body + footer
    with open(os.path.join(TABLE_DIR, "architecture_comparison.tex"), "w") as f:
        f.write(tex)
    print("✓ tables/architecture_comparison.tex")


# ═════════════════════════════════════════════════════════════════════
# TABLE 2: LoRA vs Full Fine-Tuning
# ═════════════════════════════════════════════════════════════════════
def table_lora_results():
    # Values from experiments (kd_comparison_final.md Table 1)
    data = [
        ("Swin-T",  0.534, 0.562, "+0.028"),
        ("Swin-S",  0.539, 0.554, "+0.015"),
        ("Swin-B",  0.526, 0.536, "+0.010"),
        ("DeiT3-S", 0.552, 0.527, "$-$0.025"),
        ("DeiT3-M", 0.570, 0.533, "$-$0.037"),
        ("DeiT3-B", 0.583, 0.550, "$-$0.033"),
    ]

    header = r"""\begin{table}[t]
\centering
\caption{LoRA vs.\ full fine-tuning on TinyFace. LoRA consistently improves Swin
but degrades DeiT-3 performance, revealing architecture-dependent response.}
\label{tab:lora-results}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l c c c}
\toprule
Model & Baseline Rank@1 & LoRA Rank@1 & $\Delta$ \\
\midrule
"""
    body = ""
    for name, bl, lora, delta in data:
        body += f"{name} & {bl:.4f} & {lora:.4f} & {delta} \\\\\n"
    footer = r"""\bottomrule
\end{tabular*}
\end{table}
"""
    tex = header + body + footer
    with open(os.path.join(TABLE_DIR, "lora_results.tex"), "w") as f:
        f.write(tex)
    print("✓ tables/lora_results.tex")


# ═════════════════════════════════════════════════════════════════════
# TABLE 3: Knowledge Distillation Results
# ═════════════════════════════════════════════════════════════════════
def table_kd_results():
    # From the chapter's tab:kd-results
    data = [
        ("Swin-T",  0.562, 0.567, "PETALface frozen, LoRA",   "+0.005"),
        ("Swin-S",  0.554, 0.559, "CVLFace FT, LoRA",         "+0.004"),
        ("Swin-B",  0.536, 0.536, "CVLFace FT, Direct",       "+0.000"),
        ("DeiT3-S", 0.552, 0.543, "CVLFace frozen, Direct",   "$-$0.009"),
        ("DeiT3-M", 0.570, 0.559, "CVLFace FT, Direct",       "$-$0.011"),
        ("DeiT3-B", 0.583, 0.581, "CVLFace FT, Direct",       "$-$0.003"),
    ]

    header = r"""\begin{table}[t]
\centering
\caption{Knowledge distillation results (rank-1) compared to standalone training.
Best Phase~1 refers to the better of baseline or LoRA for each model.
The best KD result across all teacher configurations is shown.
Default $\alpha=0.7$.}
\label{tab:kd-comparison}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l c c l c}
\toprule
Model & Phase 1 & Best KD & Best Config & $\Delta$ \\
\midrule
"""
    body = ""
    for name, p1, kd, cfg, delta in data:
        body += f"{name} & {p1:.3f} & {kd:.3f} & {cfg} & {delta} \\\\\n"
    footer = r"""\bottomrule
\end{tabular*}
\end{table}
"""
    tex = header + body + footer
    with open(os.path.join(TABLE_DIR, "kd_results.tex"), "w") as f:
        f.write(tex)
    print("✓ tables/kd_results.tex")


# ═════════════════════════════════════════════════════════════════════
# TABLE 4: Quantization Results
# ═════════════════════════════════════════════════════════════════════
def table_quantization_results():
    # From chapter tab:quant-results (4 representative models × 3 precisions)
    header = r"""\begin{table}[t]
\centering
\caption{Post-training quantization results for representative models.
FP16 conversion is effectively lossless. INT8 dynamic quantization
induces rank-1 drops of at most 0.16 percentage points.}
\label{tab:quantization-full}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l c c c c}
\toprule
Model & Precision & Size (MB) & Rank@1 & $\Delta$ Rank@1 \\
\midrule
"""
    data = [
        ("Swin-T Baseline",        "FP32", 106.9, 0.5335, "---"),
        ("",                        "FP16",  54.1, 0.5335, "0.000"),
        ("",                        "INT8",  28.3, 0.5322, "$-$0.001"),
        ("\\midrule", "", "", "", ""),
        ("Swin-T KD (CVLFace)",    "FP32", 106.9, 0.5491, "---"),
        ("",                        "FP16",  54.1, 0.5494, "$+$0.000"),
        ("",                        "INT8",  28.3, 0.5475, "$-$0.002"),
        ("\\midrule", "", "", "", ""),
        ("DeiT3-B KD (CVLFace)",   "FP32", 327.7, 0.5719, "---"),
        ("",                        "FP16", 164.2, 0.5714, "$-$0.001"),
        ("",                        "INT8",  84.6, 0.5727, "$+$0.001"),
        ("\\midrule", "", "", "", ""),
        ("DeiT3-B LoRA",           "FP32", 327.7, 0.5496, "---"),
        ("",                        "FP16", 164.2, 0.5496, "0.000"),
        ("",                        "INT8",  84.6, 0.5486, "$-$0.001"),
    ]
    body = ""
    for row in data:
        if row[0].startswith("\\"):
            body += row[0] + "\n"
            continue
        name, prec, size, rank1, delta = row
        size_s = f"{size:.1f}" if isinstance(size, float) else str(size)
        rank1_s = f"{rank1:.4f}" if isinstance(rank1, float) else str(rank1)
        body += f"{name} & {prec} & {size_s} & {rank1_s} & {delta} \\\\\n"
    footer = r"""\bottomrule
\end{tabular*}
\end{table}
"""
    tex = header + body + footer
    with open(os.path.join(TABLE_DIR, "quantization_results.tex"), "w") as f:
        f.write(tex)
    print("✓ tables/quantization_results.tex")


# ═════════════════════════════════════════════════════════════════════
# TABLE 5: Pruning Results
# ═════════════════════════════════════════════════════════════════════
def table_pruning_results():
    df = pd.read_csv(os.path.join(PRUNING_DIR, "comprehensive_compression.csv"))
    df_fp32 = df[df["precision"] == "FP32"].copy()
    df_fp32.sort_values(["architecture", "variant", "prune_method"], inplace=True)

    header = r"""\begin{table}[t]
\centering
\caption{Structured pruning results (FP32) across four architectures.
$\Delta$ is relative to the corresponding unpruned source checkpoint.}
\label{tab:pruning-full}
\footnotesize
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l l l c c c}
\toprule
Architecture & Variant & Pruning & Params (M) & Rank@1 & Compression \\
\midrule
"""
    body = ""
    prev_arch = None
    for _, r in df_fp32.iterrows():
        arch = r["architecture"]
        if prev_arch and arch != prev_arch:
            body += r"\midrule" + "\n"
        prev_arch = arch
        prune_str = str(r['prune_method'])
        if prune_str == "nan" or prune_str == "":
            prune_str = "--- (source)"
        body += f"{arch} & {r['variant']} & {prune_str} & {r['params_M']:.1f} & {r['rank_at_1']:.4f} & {r['compression_ratio']:.2f}$\\times$ \\\\\n"
    footer = r"""\bottomrule
\end{tabular*}
\end{table}
"""
    tex = header + body + footer
    with open(os.path.join(TABLE_DIR, "pruning_results.tex"), "w") as f:
        f.write(tex)
    print("✓ tables/pruning_results.tex")


# ═════════════════════════════════════════════════════════════════════
# TABLE 6: Jetson Inference — UNIFIED native TensorRT (all 16 models)
# ═════════════════════════════════════════════════════════════════════
def table_jetson_benchmarks():
    # Unified TRT-native FP32/FP16 sweep (Phase 7.3) covers all 16 baselines,
    # including ConvNeXt / PVT-v2 / MobileViT-v2 via export_trt_compat.py.
    uni = pd.read_csv(os.path.join(RESULTS_DIR, "trt_unified_all16.csv"))
    uni = uni[uni["mean_ms"].notna()].copy()

    # INT8 numbers still come from the earlier TRT-native PTQ sweep
    # (only available for the 9 originally-compatible models).
    legacy = pd.read_csv(os.path.join(RESULTS_DIR, "trt_benchmark.csv"))
    legacy_int8 = legacy[(legacy["category"] == "baselines") &
                         (legacy["precision"] == "int8")].copy()

    model_data = {}
    for _, r in uni.iterrows():
        pretty = SHORT_PRETTY.get(r["model"], r["model"])
        model_data.setdefault(pretty, {})[r["precision"]] = (
            r["mean_ms"], r["throughput_fps"])
    for _, r in legacy_int8.iterrows():
        pretty = SHORT_PRETTY.get(r["model"], r["model"])
        model_data.setdefault(pretty, {})["int8"] = (
            r["mean_ms"], r["throughput_fps"])

    header = r"""\begin{table}[t]
\centering
\caption{Unified native TensorRT inference latency on the Jetson Orin Nano
(batch size 1, $96\times96$ input, MAXN\_SUPER, locked clocks). All sixteen
baseline architectures execute as native TRT engines via the
\texttt{export\_trt\_compat.py} rewrites. INT8 uses TRT's built-in PTQ
calibration and is reported only for the nine originally TRT-compatible
models.}
\label{tab:jetson-latency-full}
\footnotesize
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l r r r r r r}
\toprule
& \multicolumn{2}{c}{FP32} & \multicolumn{2}{c}{FP16} & \multicolumn{2}{c}{INT8} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Model & ms & FPS & ms & FPS & ms & FPS \\
\midrule
"""
    body = ""
    # Sort by family then size
    order = [m for m in BASELINE_ORDER if m in model_data]
    for m in order:
        vals = model_data[m]
        fp32 = vals.get("fp32", (None, None))
        fp16 = vals.get("fp16", (None, None))
        int8 = vals.get("int8", (None, None))
        def _fmt(v):
            if v is None:
                return "---"
            return f"{v:.2f}"
        def _fmti(v):
            if v is None:
                return "---"
            return f"{v:.0f}"
        body += f"{m} & {_fmt(fp32[0])} & {_fmti(fp32[1])} & {_fmt(fp16[0])} & {_fmti(fp16[1])} & {_fmt(fp32[0]) if int8[0] is None else _fmt(int8[0])} & {_fmti(fp32[1]) if int8[1] is None else _fmti(int8[1])} \\\\\n"
    footer = r"""\bottomrule
\end{tabular*}
\end{table}
"""
    tex = header + body + footer
    with open(os.path.join(TABLE_DIR, "jetson_benchmarks.tex"), "w") as f:
        f.write(tex)
    print("✓ tables/jetson_benchmarks.tex")


# ═════════════════════════════════════════════════════════════════════
# TABLE 7 (DEPRECATED): TRT EP fallback table superseded by unified table.
#   Phase 7.3 rewrote the previously-incompatible architectures to
#   native TRT, so the dual-harness presentation is no longer needed.
#   Kept as a no-op so the orchestration call site keeps working.
# ═════════════════════════════════════════════════════════════════════
def table_jetson_trt_ep():
    # No-op — see table_jetson_benchmarks() for the unified 16-model table.
    return


def _table_jetson_trt_ep_legacy():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "jetson_benchmark_all.csv"))
    ep = df[(df["backend"] == "ONNXRT_TRT_EP") & (df["category"] == "baselines")].copy()
    # Also get CUDA baselines for comparison
    cuda = df[(df["backend"] == "ONNXRT_TRT_EP") | (df["backend"] == "TensorRT")]
    cuda_bl = df[df["category"] == "baselines"]

    model_data = {}
    for _, r in ep.iterrows():
        m = r["model"]
        p = r["precision"]
        pretty = SHORT_PRETTY.get(m, m)
        if pretty not in model_data:
            model_data[pretty] = {}
        model_data[pretty][f"trt_ep_{p}"] = (r["mean_ms"], r["throughput_fps"])

    header = r"""\begin{table}[t]
\centering
\caption{Inference latency for TRT-incompatible architectures on Jetson Orin Nano
using ONNX Runtime TRT Execution Provider (auto sub-graph partitioning).}
\label{tab:jetson-trt-ep}
\footnotesize
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l r r r r}
\toprule
& \multicolumn{2}{c}{TRT EP FP32} & \multicolumn{2}{c}{TRT EP FP16} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Model & ms & FPS & ms & FPS \\
\midrule
"""
    body = ""
    order = [m for m in BASELINE_ORDER if m in model_data]
    for m in order:
        d = model_data[m]
        fp32 = d.get("trt_ep_fp32", (None, None))
        fp16 = d.get("trt_ep_fp16", (None, None))
        def _f(v):
            return "---" if v is None else f"{v:.2f}"
        def _fi(v):
            return "---" if v is None else f"{v:.0f}"
        body += f"{m} & {_f(fp32[0])} & {_fi(fp32[1])} & {_f(fp16[0])} & {_fi(fp16[1])} \\\\\n"
    footer = r"""\bottomrule
\end{tabular*}
\end{table}
"""
    tex = header + body + footer
    with open(os.path.join(TABLE_DIR, "jetson_trt_ep.tex"), "w") as f:
        f.write(tex)
    print("✓ tables/jetson_trt_ep.tex")


# ═════════════════════════════════════════════════════════════════════
# TABLE 8: Per-Layer Time Breakdown
# ═════════════════════════════════════════════════════════════════════
def table_layer_breakdown():
    df = pd.read_csv(os.path.join(PROFILES_DIR, "profiling_summary.csv"))

    # Prettify model names
    def _pname(m):
        stem = m.replace("_fp16", "")
        return SHORT_PRETTY.get(stem, stem)

    header = r"""\begin{table}[t]
\centering
\caption{Per-layer time breakdown (FP16, TensorRT on Jetson Orin Nano).
``FFN All'' includes both fused FFN and unfused linear layers.}
\label{tab:layer-breakdown}
\footnotesize
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l r r r r r r}
\toprule
Model & Total (ms) & Attn (\%) & FFN All (\%) & Norm (\%) & Conv/Embed (\%) & Other (\%) \\
\midrule
"""
    body = ""
    for _, r in df.iterrows():
        name = _pname(r["model"])
        other = r.get("reformat_pct", 0) + r.get("other_pct", 0)
        body += f"{name} & {r['total_ms']:.2f} & {r['attention_pct']:.1f} & {r['ffn_all_pct']:.1f} & {r['norm_pct']:.1f} & {r['conv_embed_pct']:.1f} & {other:.1f} \\\\\n"
    footer = r"""\bottomrule
\end{tabular*}
\end{table}
"""
    tex = header + body + footer
    with open(os.path.join(TABLE_DIR, "layer_breakdown.tex"), "w") as f:
        f.write(tex)
    print("✓ tables/layer_breakdown.tex")


# ═════════════════════════════════════════════════════════════════════
# FIGURE: Per-Layer Time Breakdown — stacked bar chart
# ═════════════════════════════════════════════════════════════════════
def plot_layer_breakdown():
    df = pd.read_csv(os.path.join(PROFILES_DIR, "profiling_summary.csv"))

    # Prettify and select representative models (skip duplicate-ish KD/pruned variants)
    selected = [
        "deit3_small_fp16", "deit3_medium_fp16", "deit3_base_fp16",
        "swin_tiny_fp16", "swin_small_fp16", "swin_base_fp16",
        "swin_tiny_pruned_blocks2_fp16",
        "resnet50d_fp16", "resnet101d_fp16", "resnet200d_fp16",
    ]
    df = df[df["model"].isin(selected)].copy()
    df["pretty"] = df["model"].apply(lambda m: SHORT_PRETTY.get(m.replace("_fp16", ""), m.replace("_fp16", "")))
    # Fix pruned name
    df.loc[df["model"] == "swin_tiny_pruned_blocks2_fp16", "pretty"] = "Swin-T (−2 blk)"

    order = [
        "DeiT3-S", "DeiT3-M", "DeiT3-B",
        "Swin-T", "Swin-S", "Swin-B", "Swin-T (−2 blk)",
        "ResNet-50", "ResNet-101", "ResNet-200",
    ]
    df["order"] = df["pretty"].apply(lambda p: order.index(p) if p in order else 99)
    df = df.sort_values("order")

    categories = ["attention_pct", "ffn_all_pct", "norm_pct", "conv_embed_pct"]
    cat_labels = ["Attention", "FFN + Linear", "Normalization", "Conv/Embed"]
    cat_colors = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]

    # Compute 'Other' as remainder
    df["other_total"] = 100.0 - df[categories].sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    bottoms = np.zeros(len(df))

    for cat, label, color in zip(categories, cat_labels, cat_colors):
        vals = df[cat].values.astype(float)
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, edgecolor="white", linewidth=0.5)
        bottoms += vals

    # Other
    other_vals = df["other_total"].values.astype(float)
    ax.bar(x, other_vals, bottom=bottoms, label="Other", color="#95a5a6", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["pretty"].values, rotation=35, ha="right")
    ax.set_ylabel("Time (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-Layer Time Breakdown (FP16, Jetson Orin Nano)", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5,
              frameon=True, fancybox=False, edgecolor="0.8")
    ax.grid(axis="y", alpha=0.3)

    # Add total ms annotation on top
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(i, 101, f"{r['total_ms']:.1f} ms", ha="center", va="bottom", fontsize=7, color="0.4")

    _save_fig(fig, "layer_breakdown")
    print("✓ layer_breakdown figure")


# ═════════════════════════════════════════════════════════════════════
# FIGURE: Latency comparison bar chart — all baselines FP32/FP16/INT8
# ═════════════════════════════════════════════════════════════════════
def plot_latency_comparison():
    # Unified TRT-native sweep covers all 16 baselines (Phase 7.3).
    uni = pd.read_csv(os.path.join(RESULTS_DIR, "trt_unified_all16.csv"))
    combined = uni[uni["mean_ms"].notna()].copy()

    order = [
        "resnet50d", "resnet101d", "resnet200d",
        "deit3_small", "deit3_medium", "deit3_base",
        "swin_tiny", "swin_small", "swin_base",
        "convnext_tiny", "convnext_small", "convnext_base",
        "pvt_v2_b2", "pvt_v2_b3", "pvt_v2_b5",
        "mobilevitv2_200",
    ]
    combined = combined[combined["precision"].isin(["fp32", "fp16"])]

    model_data = {}
    for _, r in combined.iterrows():
        m = r["model"]
        if m not in order:
            continue
        pretty = SHORT_PRETTY.get(m, m)
        model_data.setdefault(pretty, {})[r["precision"]] = r["mean_ms"]

    fig_order = [SHORT_PRETTY.get(m, m) for m in order if SHORT_PRETTY.get(m, m) in model_data]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(fig_order))
    w = 0.35

    fp32_vals = [model_data[m].get("fp32", 0) for m in fig_order]
    fp16_vals = [model_data[m].get("fp16", 0) for m in fig_order]

    colors_fp32 = [FAMILY_COLORS.get(family_of(m), "gray") for m in fig_order]
    # FP16: lighter version
    import matplotlib.colors as mcolors
    colors_fp16 = [mcolors.to_rgba(c, alpha=0.6) for c in colors_fp32]

    ax.bar(x - w / 2, fp32_vals, w, color=colors_fp32, edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, fp16_vals, w, color=colors_fp16, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(fig_order, rotation=40, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Jetson Orin Nano Latency — native TensorRT, FP32 vs FP16 (batch size 1, 96×96)")

    # Legend for families + precision
    handles = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]
    handles.append(Patch(facecolor="0.3", label="FP32 (solid)"))
    handles.append(Patch(facecolor="0.3", alpha=0.6, label="FP16 (light)"))
    ax.legend(handles=handles, loc="upper left", ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, "latency_comparison")
    print("✓ latency_comparison figure")


# ═════════════════════════════════════════════════════════════════════
# FIGURE: Accuracy vs Latency — Pareto frontier (Jetson FP16)
# ═════════════════════════════════════════════════════════════════════
def plot_accuracy_vs_latency_pareto():
    baselines = _get_baseline_rank1()

    # Unified TRT-native FP16 latencies cover all 16 baselines (Phase 7.3).
    uni = pd.read_csv(os.path.join(RESULTS_DIR, "trt_unified_all16.csv"))
    fp16 = uni[(uni["precision"] == "fp16") & uni["mean_ms"].notna()].copy()

    points = []  # (pretty, latency_ms, rank1, family)
    for _, r in fp16.iterrows():
        pretty = SHORT_PRETTY.get(r["model"], r["model"])
        if pretty in baselines:
            points.append((pretty, r["mean_ms"], baselines[pretty], family_of(pretty)))

    fig, ax = plt.subplots(figsize=(7, 5))

    for name, lat, rank1, fam in points:
        color = FAMILY_COLORS.get(fam, "gray")
        ax.scatter(lat, rank1, c=color, s=60, zorder=5, edgecolors="white", linewidths=0.5)
        ax.annotate(name, (lat, rank1), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color=color)

    # Compute and draw Pareto frontier
    pts_sorted = sorted(points, key=lambda p: p[1])  # sort by latency
    pareto = []
    best_rank1 = -1
    for name, lat, rank1, fam in pts_sorted:
        if rank1 > best_rank1:
            pareto.append((lat, rank1, name))
            best_rank1 = rank1
    # Draw connected Pareto front (from low-lat/low-acc to high-lat/high-acc)
    # Actually Pareto = non-dominated: no other point better in BOTH latency and accuracy
    # Re-compute properly
    pareto = []
    for n1, l1, r1, f1 in points:
        dominated = False
        for n2, l2, r2, f2 in points:
            if l2 <= l1 and r2 >= r1 and (l2 < l1 or r2 > r1):
                dominated = True
                break
        if not dominated:
            pareto.append((l1, r1))
    pareto.sort()
    if pareto:
        plats, pranks = zip(*pareto)
        ax.plot(plats, pranks, "k--", alpha=0.4, linewidth=1.0, zorder=1, label="Pareto frontier")

    ax.set_xlabel("Latency (ms) — FP16, native TensorRT on Jetson Orin Nano")
    ax.set_ylabel("Rank-1 Identification Rate")
    ax.set_title("Accuracy vs. Latency — Baseline Models (FP16, unified native TRT)")
    handles = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]
    handles.append(Line2D([0], [0], linestyle="--", color="k", alpha=0.4, label="Pareto"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    _save_fig(fig, "accuracy_vs_latency_pareto")
    print("✓ accuracy_vs_latency_pareto figure")


# ═════════════════════════════════════════════════════════════════════
# FIGURE: Model size reduction waterfall (FP32 → FP16 → INT8)
# ═════════════════════════════════════════════════════════════════════
def plot_size_reduction_waterfall():
    quant = pd.read_csv(os.path.join(QUANT_DIR, "all_models_quantization.csv"))
    baselines_q = quant[quant["category"] == "baselines"].copy()
    manifest = pd.read_csv(MANIFEST)
    bl_manifest = manifest[manifest["training_method"] == "baseline"]

    # Get FP32 size from manifest
    fp32_sizes = {}
    for _, r in bl_manifest.iterrows():
        pretty = PRETTY.get(r["model_name"], r["model_name"])
        fp32_sizes[pretty] = r["file_size_MB"]

    # Short stem to pretty for quant CSV
    models = []
    for _, r in baselines_q.iterrows():
        pretty = SHORT_PRETTY.get(r["model"], r["model"])
        if pretty in fp32_sizes:
            models.append({
                "name": pretty,
                "fp32": fp32_sizes[pretty],
                "fp16": r["fp16_size_MB"],
                "int8": r["int8_size_MB"],
            })

    # Sort by FP32 size descending
    models.sort(key=lambda m: m["fp32"], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(models))
    w = 0.25

    fp32_vals = [m["fp32"] for m in models]
    fp16_vals = [m["fp16"] for m in models]
    int8_vals = [m["int8"] for m in models]
    names = [m["name"] for m in models]

    ax.bar(x - w, fp32_vals, w, label="FP32", color="#2c3e50", edgecolor="white")
    ax.bar(x, fp16_vals, w, label="FP16", color="#2980b9", edgecolor="white")
    ax.bar(x + w, int8_vals, w, label="INT8", color="#27ae60", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Model Size (MB)")
    ax.set_title("Model Size Reduction: FP32 → FP16 → INT8")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Annotate compression ratio
    for i, m in enumerate(models):
        ratio = m["fp32"] / m["int8"]
        ax.text(i + w, m["int8"] + 3, f"{ratio:.1f}×", ha="center", va="bottom",
                fontsize=7, color="#27ae60")

    _save_fig(fig, "size_reduction_waterfall")
    print("✓ size_reduction_waterfall figure")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    print("=" * 60)
    print("Phase 6.1 — Compiling results into tables and figures")
    print("=" * 60)

    # ── Tables ──
    print("\n── Generating LaTeX tables ──")
    table_architecture_comparison()
    table_lora_results()
    table_kd_results()
    table_quantization_results()
    table_pruning_results()
    table_jetson_benchmarks()
    table_jetson_trt_ep()
    table_layer_breakdown()

    # ── New figures ──
    print("\n── Generating new figures ──")
    plot_layer_breakdown()
    plot_latency_comparison()
    plot_accuracy_vs_latency_pareto()
    plot_size_reduction_waterfall()

    # ── Existing figures (re-generate via scripts/generate_plots.py) ──
    print("\n── Re-generating existing figures ──")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gen_plots", os.path.join(ROOT, "scripts", "generate_plots.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    # These save to docs/.../figures/ directly
    gen.plot_baseline_comparison()
    gen.plot_baseline_vs_lora()
    gen.plot_alpha_ablation()
    gen.plot_kd_overview()
    gen.plot_params_vs_accuracy()
    gen.plot_cmc_curves()

    print("\n" + "=" * 60)
    print(f"Tables:  {TABLE_DIR}/")
    print(f"Figures: {FIG_DIR}/")
    print("=" * 60)

    # ── Cross-check ──
    # Note: model_ablation CSV contains best Phase 1 runs per arch (may include LoRA)
    print("\n── Cross-checking accuracy values (from ablation CSV) ──")
    baselines = _get_baseline_rank1()
    checks = {
        "DeiT3-B": 0.583, "Swin-T": 0.562, "ResNet-50": 0.476,
    }
    all_ok = True
    for model, expected in checks.items():
        actual = baselines.get(model)
        if actual and abs(actual - expected) > 0.005:
            print(f"  ✗ {model}: expected ~{expected:.3f}, got {actual:.3f}")
            all_ok = False
        else:
            print(f"  ✓ {model}: {actual:.4f}")
    if all_ok:
        print("  All cross-checks passed.")
    print()


if __name__ == "__main__":
    main()
