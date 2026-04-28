#!/usr/bin/env python3
"""Phase 7.3 — analysis for the unified native-TensorRT Jetson sweep.

Reads:
  inference_results/trt_unified_all16.csv       (headline, all 16 baselines)
  inference_results/model_ablation_inference_results.csv (rank@1)
  inference_results/final_models_inference_results.csv   (rank@1 fallback)

Writes (into docs/scalable_vits_for_embedded_systems/figures/):
  latency_comparison_unified.pdf
  accuracy_vs_latency_pareto_unified.pdf
  jetson_unified_table.tex

Background: the seven architectures previously called "TRT-incompatible"
(ConvNeXt-T/S/B, PVT-v2-B2/B3/B5, MobileViT-v2-200) have been re-exported
through ``export_trt_compat.py``. All seven now build native TRT engines
under TensorRT 10.3 / SM 8.7 with cosine similarity 1.000 vs the originals.
This script consumes the resulting unified ``trt_unified_all16.csv``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "inference_results"
FIG_DIR = ROOT / "docs" / "scalable_vits_for_embedded_systems" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ORDER = [
    "resnet50d", "resnet101d", "resnet200d",
    "deit3_small", "deit3_medium", "deit3_base",
    "swin_tiny", "swin_small", "swin_base",
    "convnext_tiny", "convnext_small", "convnext_base",
    "pvt_v2_b2", "pvt_v2_b3", "pvt_v2_b5",
    "mobilevitv2_200",
]

PRETTY = {
    "resnet50d": "ResNet-50d", "resnet101d": "ResNet-101d", "resnet200d": "ResNet-200d",
    "deit3_small": "DeiT3-S", "deit3_medium": "DeiT3-M", "deit3_base": "DeiT3-B",
    "swin_tiny": "Swin-T", "swin_small": "Swin-S", "swin_base": "Swin-B",
    "convnext_tiny": "ConvNeXt-T", "convnext_small": "ConvNeXt-S", "convnext_base": "ConvNeXt-B",
    "pvt_v2_b2": "PVTv2-B2", "pvt_v2_b3": "PVTv2-B3", "pvt_v2_b5": "PVTv2-B5",
    "mobilevitv2_200": "MobileViTv2-200",
}

PARAMS_M = {
    "resnet50d": 25.6, "resnet101d": 44.6, "resnet200d": 64.7,
    "deit3_small": 21.7, "deit3_medium": 38.3, "deit3_base": 85.8,
    "swin_tiny": 27.5, "swin_small": 49.6, "swin_base": 86.7,
    "convnext_tiny": 28.6, "convnext_small": 50.2, "convnext_base": 87.6,
    "pvt_v2_b2": 25.4, "pvt_v2_b3": 45.2, "pvt_v2_b5": 82.0,
    "mobilevitv2_200": 18.4,
}

FAMILY_OF = {
    **{m: "ResNet" for m in ["resnet50d", "resnet101d", "resnet200d"]},
    **{m: "DeiT-3" for m in ["deit3_small", "deit3_medium", "deit3_base"]},
    **{m: "Swin"   for m in ["swin_tiny", "swin_small", "swin_base"]},
    **{m: "ConvNeXt" for m in ["convnext_tiny", "convnext_small", "convnext_base"]},
    **{m: "PVT-v2" for m in ["pvt_v2_b2", "pvt_v2_b3", "pvt_v2_b5"]},
    **{m: "MobileViT-v2" for m in ["mobilevitv2_200"]},
}

FAMILY_COLORS = {
    "ResNet":       "#9c8f6e",
    "DeiT-3":       "#3b6fa8",
    "Swin":         "#4ba56b",
    "ConvNeXt":     "#c46e6e",
    "PVT-v2":       "#a06fb4",
    "MobileViT-v2": "#d99550",
}

# Per-model annotation offsets (dx_pt, dy_pt, ha) for the Pareto scatter
# tuned to avoid overlaps in the dense 1.3-2.0 ms region.
PARETO_LABEL_OFFSETS = {
    "resnet50d":       ( 8,   5, "left"),
    "resnet101d":      ( 8,   5, "left"),
    "resnet200d":      ( 8,   5, "left"),
    "deit3_small":     (-7, -12, "right"),
    "deit3_medium":    ( 8,   8, "left"),
    "deit3_base":      ( 8,   5, "left"),
    "swin_tiny":       (-7,   9, "right"),
    "swin_small":      ( 8,   5, "left"),
    "swin_base":       ( 8,   5, "left"),
    "convnext_tiny":   ( 8, -12, "left"),
    "convnext_small":  ( 8, -12, "left"),
    "convnext_base":   ( 8, -12, "left"),
    "pvt_v2_b2":       ( 8,   8, "left"),
    "pvt_v2_b3":       ( 8,   8, "left"),
    "pvt_v2_b5":       (-7,   8, "right"),
    "mobilevitv2_200": (-7, -12, "right"),
}

TIMM_TO_SHORT = {
    "swin_tiny_patch4_window7_224.ms_in1k": "swin_tiny",
    "swin_small_patch4_window7_224.ms_in1k": "swin_small",
    "swin_base_patch4_window7_224.ms_in1k": "swin_base",
    "deit3_small_patch16_224.fb_in1k": "deit3_small",
    "deit3_medium_patch16_224.fb_in1k": "deit3_medium",
    "deit3_base_patch16_224.fb_in1k": "deit3_base",
    "convnext_tiny.fb_in1k": "convnext_tiny",
    "convnext_small.fb_in1k": "convnext_small",
    "convnext_base.fb_in1k": "convnext_base",
    "pvt_v2_b2.in1k": "pvt_v2_b2",
    "pvt_v2_b3.in1k": "pvt_v2_b3",
    "pvt_v2_b5.in1k": "pvt_v2_b5",
    "mobilevitv2_200.cvnets_in1k": "mobilevitv2_200",
    "resnet50d.ra2_in1k": "resnet50d",
    "resnet101d.ra2_in1k": "resnet101d",
    "resnet200d.ra2_in1k": "resnet200d",
}


def load_unified() -> pd.DataFrame:
    p = RESULTS / "trt_unified_all16.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run shell_scripts/benchmark_trt_unified.sh on Jetson"
        )
    df = pd.read_csv(p)
    df["precision"] = df["precision"].str.lower()
    df = df[df["mean_ms"].notna()].copy()
    df["mean_ms"] = df["mean_ms"].astype(float)
    return df


def load_rank1() -> dict:
    rank1: dict = {}
    for csv_name in ["model_ablation_inference_results.csv",
                     "final_models_inference_results.csv"]:
        p = RESULTS / csv_name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        col = "tinyface/test/rank@1"
        if "model_name" not in df.columns or col not in df.columns:
            continue
        for _, r in df.iterrows():
            short = TIMM_TO_SHORT.get(r["model_name"])
            if short and short not in rank1 and pd.notna(r[col]):
                rank1[short] = float(r[col])
    return rank1


def plot_latency_comparison(df: pd.DataFrame, out_pdf: Path) -> None:
    fp32 = df[df["precision"] == "fp32"].set_index("model")["mean_ms"]
    fp16 = df[df["precision"] == "fp16"].set_index("model")["mean_ms"]

    models = [m for m in ORDER if m in fp32.index or m in fp16.index]
    x = np.arange(len(models))
    w = 0.38

    fp32_v = [fp32.get(m, np.nan) for m in models]
    fp16_v = [fp16.get(m, np.nan) for m in models]

    base = [FAMILY_COLORS[FAMILY_OF[m]] for m in models]
    light = [mcolors.to_rgba(c, alpha=0.55) for c in base]

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.bar(x - w/2, fp32_v, w, color=base,  edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, fp16_v, w, color=light, edgecolor="white", linewidth=0.5)

    for xi, v in zip(x - w/2, fp32_v):
        if v == v:
            ax.text(xi, v + 0.10, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color="#222")
    for xi, v in zip(x + w/2, fp16_v):
        if v == v:
            ax.text(xi, v + 0.10, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[m] for m in models], rotation=40, ha="right",
                       fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("Mean inference latency (ms)", fontsize=13)
    ax.set_title(r"Jetson Orin Nano — native TensorRT latency "
                 r"(batch=1, $96{\times}96$, MAXN_SUPER, all 16 models)",
                 fontsize=13)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    handles = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]
    handles.append(Patch(facecolor="0.4", label="FP32 (solid)"))
    handles.append(Patch(facecolor="0.4", alpha=0.55, label="FP16 (light)"))
    ax.legend(handles=handles, loc="upper left", ncol=4, fontsize=10.5,
              frameon=False)
    ax.margins(y=0.14)

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def plot_pareto(df: pd.DataFrame, rank1: dict, out_pdf: Path) -> None:
    fp16 = df[df["precision"] == "fp16"].set_index("model")["mean_ms"]
    pts = [(m, float(fp16[m]), float(rank1[m]), FAMILY_OF[m])
           for m in ORDER if m in fp16.index and m in rank1]

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for m, lat, r1, fam in pts:
        ax.scatter(lat, r1, c=FAMILY_COLORS[fam], s=90, zorder=5,
                   edgecolors="white", linewidths=0.7)
        dx, dy, ha = PARETO_LABEL_OFFSETS.get(m, (8, 5, "left"))
        ax.annotate(PRETTY[m], (lat, r1), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10, ha=ha,
                    color=FAMILY_COLORS[fam])

    pareto = []
    for _, l1, r1, _ in pts:
        dominated = any((l2 <= l1 and r2 >= r1 and (l2 < l1 or r2 > r1))
                        for _, l2, r2, _ in pts)
        if not dominated:
            pareto.append((l1, r1))
    pareto.sort()
    if pareto:
        plats, pranks = zip(*pareto)
        ax.plot(plats, pranks, "k--", alpha=0.4, linewidth=1.0, zorder=1)

    # Pad x-axis so right-edge labels (PVTv2-B5) aren't clipped
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(max(0.9, xmin - 0.2), xmax + 0.4)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.005, ymax + 0.008)

    ax.set_xlabel("Mean inference latency (ms) — FP16, native TensorRT",
                  fontsize=13)
    ax.set_ylabel("Rank-1 identification rate", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title("TinyFace accuracy vs. Jetson FP16 latency — "
                 "unified native TensorRT", fontsize=13)
    handles = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]
    handles.append(Line2D([0], [0], linestyle="--", color="k", alpha=0.45,
                          label="Pareto frontier"))
    ax.legend(handles=handles, loc="lower right", fontsize=10.5, frameon=False)
    ax.grid(linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def write_unified_table(df: pd.DataFrame, rank1: dict, out_tex: Path) -> None:
    fp32 = df[df["precision"] == "fp32"].set_index("model")
    fp16 = df[df["precision"] == "fp16"].set_index("model")

    rows = []
    for m in ORDER:
        if m not in fp32.index and m not in fp16.index:
            continue
        l32 = float(fp32.loc[m, "mean_ms"]) if m in fp32.index else np.nan
        l16 = float(fp16.loc[m, "mean_ms"]) if m in fp16.index else np.nan
        fps32 = 1000.0 / l32 if l32 == l32 else np.nan
        fps16 = 1000.0 / l16 if l16 == l16 else np.nan
        sp = (l32 / l16) if (l32 == l32 and l16 == l16) else np.nan
        r1 = rank1.get(m, np.nan)
        rows.append((m, PARAMS_M.get(m, np.nan), l32, fps32, l16, fps16, sp, r1))

    def fnum(v, spec=".2f"):
        return "---" if (v != v) else format(v, spec)

    body = ""
    last_family = None
    for m, p, l32, fps32, l16, fps16, sp, r1 in rows:
        fam = FAMILY_OF[m]
        if fam != last_family and last_family is not None:
            body += "\\addlinespace[2pt]\n"
        last_family = fam
        body += (f"{PRETTY[m]} & {fnum(p, '.1f')} & "
                 f"{fnum(l32)} & {fnum(fps32, '.0f')} & "
                 f"{fnum(l16)} & {fnum(fps16, '.0f')} & "
                 f"{fnum(sp, '.2f')}$\\times$ & {fnum(r1, '.3f')} \\\\\n")

    tex = (
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}} l r r r r r r r}\n"
        "\\toprule\n"
        " & & \\multicolumn{2}{c}{FP32} & \\multicolumn{2}{c}{FP16} & & \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        "Model & Params (M) & ms & FPS & ms & FPS & FP16 speedup & Rank@1 \\\\\n"
        "\\midrule\n"
        f"{body}"
        "\\bottomrule\n"
        "\\end{tabular*}\n"
    )
    out_tex.write_text(tex)
    print(f"Wrote {out_tex}")


def main() -> None:
    df = load_unified()
    rank1 = load_rank1()

    plot_latency_comparison(df, FIG_DIR / "latency_comparison_unified.pdf")
    plot_pareto(df, rank1, FIG_DIR / "accuracy_vs_latency_pareto_unified.pdf")
    write_unified_table(df, rank1, FIG_DIR / "jetson_unified_table.tex")

    print("\nFP16 ranking (ms, ascending):")
    fp16 = df[df["precision"] == "fp16"].set_index("model")["mean_ms"].to_dict()
    for m, v in sorted(fp16.items(), key=lambda kv: kv[1]):
        print(f"  {PRETTY.get(m, m):<18s}  {v:6.2f} ms  ({FAMILY_OF[m]})")


if __name__ == "__main__":
    main()
