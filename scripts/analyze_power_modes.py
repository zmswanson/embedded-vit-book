#!/usr/bin/env python3
"""Phase 7.1 — consolidate Jetson power-mode benchmarks.

Reads:
  inference_results/power_modes/trt_power_sweep.csv
  inference_results/power_modes/onnxrt_trt_ep_power_sweep.csv
  inference_results/power_modes/tegrastats_<label>.log
  inference_results/final_models_inference_results.csv  (rank@1 reference)

Writes:
  inference_results/power_modes/summary.csv
  inference_results/power_modes/summary.md
  docs/scalable_vits_for_embedded_systems/figures/power_mode_latency.pdf
  docs/scalable_vits_for_embedded_systems/figures/power_mode_throughput_per_watt.pdf
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PM_DIR = ROOT / "inference_results" / "power_modes"
FIG_DIR = ROOT / "docs" / "scalable_vits_for_embedded_systems" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# canonical mode order (low → high)
MODE_ORDER = ["15W", "25W", "MAXN_SUPER"]
MODE_BUDGET = {"15W": 15, "25W": 25, "MAXN_SUPER": 25}

# rank@1 lookup: engine-label -> timm model_name in final_models_inference_results.csv
LABEL_TO_TIMM = {
    "swin_tiny":   "swin_tiny_patch4_window7_224.ms_in1k",
    "swin_base":   "swin_base_patch4_window7_224.ms_in1k",
    "deit3_small": "deit3_small_patch16_224.fb_in1k",
    "deit3_base":  "deit3_base_patch16_224.fb_in1k",
    # resnets and KD/pruned/ORT-EP variants are not in that CSV → rank@1 left blank
}


# ---------------------------------------------------------------- tegrastats
_VDD_RE = re.compile(r"VDD_IN (\d+)mW/(\d+)mW")


def parse_tegrastats(path: Path) -> dict:
    inst, avg = [], []
    with path.open() as f:
        for line in f:
            m = _VDD_RE.search(line)
            if m:
                inst.append(int(m.group(1)))
                avg.append(int(m.group(2)))
    if not inst:
        return {"vdd_in_mean_w": np.nan, "vdd_in_p95_w": np.nan, "samples": 0}
    return {
        "vdd_in_mean_w": np.mean(inst) / 1000.0,
        "vdd_in_p95_w":  np.percentile(inst, 95) / 1000.0,
        "samples": len(inst),
    }


# ---------------------------------------------------------------- load data
def load_trt() -> pd.DataFrame:
    df = pd.read_csv(PM_DIR / "trt_power_sweep.csv")
    df["source"] = "trt"
    df["throughput_fps"] = pd.to_numeric(df["throughput_fps"], errors="coerce")
    df["mean_ms"] = pd.to_numeric(df["mean_ms"], errors="coerce")
    df["p99_ms"]  = pd.to_numeric(df["p99_ms"],  errors="coerce")
    return df[["model", "category", "precision", "power_mode_id",
               "power_mode_label", "wattage_budget",
               "mean_ms", "p99_ms", "throughput_fps", "source"]]


def load_ort() -> pd.DataFrame:
    df = pd.read_csv(PM_DIR / "onnxrt_trt_ep_power_sweep.csv")
    df["category"] = "trt_incompatible_ortep"
    df["precision"] = df["precision"].str.lower()
    df["throughput_fps"] = 1000.0 / df["mean_ms"]
    df["source"] = "ortep"
    return df[["model", "category", "precision", "power_mode_id",
               "power_mode_label", "wattage_budget",
               "mean_ms", "p99_ms", "throughput_fps", "source"]]


def load_rank1() -> dict:
    p = ROOT / "inference_results" / "final_models_inference_results.csv"
    df = pd.read_csv(p)
    return dict(zip(df["model_name"], df["tinyface/test/rank@1"]))


# ---------------------------------------------------------------- main
def main():
    long_df = pd.concat([load_trt(), load_ort()], ignore_index=True)
    long_df["power_mode_label"] = pd.Categorical(
        long_df["power_mode_label"], categories=MODE_ORDER, ordered=True
    )
    long_df = long_df.sort_values(["category", "model", "precision", "power_mode_label"])

    # tegrastats per mode (sustained wall power)
    tegra = {}
    for label in MODE_ORDER:
        f = PM_DIR / f"tegrastats_{label}.log"
        if f.exists():
            tegra[label] = parse_tegrastats(f)
        else:
            tegra[label] = {"vdd_in_mean_w": np.nan, "vdd_in_p95_w": np.nan, "samples": 0}
    tegra_df = pd.DataFrame(tegra).T
    tegra_df.index.name = "power_mode_label"
    tegra_df.to_csv(PM_DIR / "tegrastats_summary.csv")
    print("Tegrastats per mode (VDD_IN, sustained wall power):")
    print(tegra_df.to_string(float_format=lambda x: f"{x:6.2f}"))

    # Wide format: one row per (category, model, precision, source); columns per mode
    pivot_mean = long_df.pivot_table(
        index=["category", "source", "model", "precision"],
        columns="power_mode_label",
        values="mean_ms",
        aggfunc="first",
        observed=True,
    ).rename(columns={l: f"mean_ms_{l}" for l in MODE_ORDER})
    pivot_fps = long_df.pivot_table(
        index=["category", "source", "model", "precision"],
        columns="power_mode_label",
        values="throughput_fps",
        aggfunc="first",
        observed=True,
    ).rename(columns={l: f"fps_{l}" for l in MODE_ORDER})
    pivot_p99 = long_df.pivot_table(
        index=["category", "source", "model", "precision"],
        columns="power_mode_label",
        values="p99_ms",
        aggfunc="first",
        observed=True,
    ).rename(columns={l: f"p99_ms_{l}" for l in MODE_ORDER})

    summary = pd.concat([pivot_mean, pivot_fps, pivot_p99], axis=1).reset_index()

    # Reference column = MAXN_SUPER mean_ms; relative slowdown vs MAXN
    ref = "mean_ms_MAXN_SUPER"
    for label in MODE_ORDER:
        if label == "MAXN_SUPER":
            summary["slowdown_vs_MAXN_MAXN_SUPER"] = 1.0
        else:
            summary[f"slowdown_vs_MAXN_{label}"] = summary[f"mean_ms_{label}"] / summary[ref]

    # fps per measured wall watt (use VDD_IN mean across the run for that mode)
    for label in MODE_ORDER:
        watt = tegra[label]["vdd_in_mean_w"]
        summary[f"fps_per_watt_{label}"] = summary[f"fps_{label}"] / watt if watt and not np.isnan(watt) else np.nan

    # Variance metric: how stable is fps across modes? (std / mean over the 3 modes)
    fps_cols = [f"fps_{l}" for l in MODE_ORDER]
    summary["fps_cv_across_modes"] = summary[fps_cols].std(axis=1) / summary[fps_cols].mean(axis=1)

    # rank@1 join (only for the 4 baselines we have)
    rank1 = load_rank1()
    summary["rank1"] = summary["model"].map(lambda m: rank1.get(LABEL_TO_TIMM.get(m, ""), np.nan))

    summary_path = PM_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    print(f"\nWrote {summary_path}")

    # ---------- markdown table for chapter paste ----------
    md_cols = [
        "model", "precision",
        "mean_ms_15W", "mean_ms_25W", "mean_ms_MAXN_SUPER",
        "fps_15W", "fps_25W", "fps_MAXN_SUPER",
        "slowdown_vs_MAXN_15W", "fps_cv_across_modes", "rank1",
    ]
    md_rows = summary[md_cols].copy()

    def fmt(v, spec=".2f"):
        if pd.isna(v):
            return "—"
        return format(v, spec)

    md_lines = [
        "| Model | Prec | Mean ms (15W / 25W / MAXN) | FPS (15W / 25W / MAXN) | Slowdown 15W vs MAXN | FPS CV | Rank@1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in md_rows.iterrows():
        md_lines.append(
            f"| {r['model']} | {r['precision']} | "
            f"{fmt(r['mean_ms_15W'])} / {fmt(r['mean_ms_25W'])} / {fmt(r['mean_ms_MAXN_SUPER'])} | "
            f"{fmt(r['fps_15W'], '.0f')} / {fmt(r['fps_25W'], '.0f')} / {fmt(r['fps_MAXN_SUPER'], '.0f')} | "
            f"{fmt(r['slowdown_vs_MAXN_15W'], '.2f')}× | "
            f"{fmt(r['fps_cv_across_modes'], '.3f')} | "
            f"{fmt(r['rank1'], '.4f')} |"
        )
    tegra_md = ["", "### Tegrastats wall power (VDD_IN, sustained over the sweep)", "",
                "| Mode | mean W | p95 W | samples |", "|---|---|---|---|"]
    for label in MODE_ORDER:
        t = tegra[label]
        tegra_md.append(f"| {label} | {fmt(t['vdd_in_mean_w'])} | {fmt(t['vdd_in_p95_w'])} | {t['samples']} |")

    md_path = PM_DIR / "summary.md"
    md_path.write_text("\n".join(md_lines + tegra_md) + "\n")
    print(f"Wrote {md_path}")

    # ---------- plots ----------
    plot_df = summary.copy()
    plot_df["label"] = plot_df["model"] + " (" + plot_df["precision"] + ")"
    # latency bars
    fig, ax = plt.subplots(figsize=(11, 5))
    n = len(plot_df)
    x = np.arange(n)
    w = 0.27
    colors = {"15W": "#d96666", "25W": "#e0a64a", "MAXN_SUPER": "#4a90c2"}
    for i, label in enumerate(MODE_ORDER):
        ax.bar(x + (i - 1) * w, plot_df[f"mean_ms_{label}"], w, label=label, color=colors[label])
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Mean inference latency (ms)")
    ax.set_title("Jetson Orin Nano latency by power mode (batch=1, 96×96)")
    ax.legend(title="Power mode")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "power_mode_latency.pdf")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'power_mode_latency.pdf'}")

    # throughput / wattage_budget bars (use the configured budget, capped at 25 for MAXN)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, label in enumerate(MODE_ORDER):
        budget = MODE_BUDGET[label]
        ax.bar(x + (i - 1) * w, plot_df[f"fps_{label}"] / budget, w,
               label=f"{label} (÷{budget}W)", color=colors[label])
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Throughput / configured budget (FPS / W)")
    ax.set_title("Jetson Orin Nano throughput-per-watt by power mode")
    ax.legend(title="Power mode")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "power_mode_throughput_per_watt.pdf")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'power_mode_throughput_per_watt.pdf'}")

    # ---------- console summary ----------
    print("\n=== Markdown table ===\n")
    print("\n".join(md_lines + tegra_md))


if __name__ == "__main__":
    main()
