#!/usr/bin/env python3
"""Analyze LoRA sweep results and extract best hyperparameter configs."""

import pandas as pd
import json
import os
import sys

DATA_DIR = ".ignore/.ignore/.wandb_sweep_results"
OUT_DIR = "configs"

sweeps = {
    "swin_lora_adaface": {
        "file": "wandb_report-runs_sweep-vit_book_swin_lora_ada.csv",
        "model": "swin_base_patch4_window7_224.ms_in1k",
        "head_type": "adaface",
        "swept_params": ["lr", "head_scale", "head_margin", "adaface_h", "adaface_t_alpha"],
        "fixed_lora": {"lora_r": 24, "lora_alpha": 120, "lora_dropout": 0.05,
                        "lora_qkv_proj": 1, "lora_train_bias": 0, "weight_decay": 1e-4},
    },
    "swin_lora_arcface": {
        "file": "wandb_report-runs_sweep-vit_book_swin_lora.csv",
        "model": "swin_base_patch4_window7_224.ms_in1k",
        "head_type": "arcface",
        "swept_params": ["lr", "weight_decay", "lora_r", "lora_alpha", "lora_dropout",
                          "lora_qkv_proj", "lora_train_bias", "head_scale"],
        "fixed_adaface": {"head_margin": 1e-6, "adaface_h": 0.2015, "adaface_t_alpha": 0.0438},
    },
    "deit3_lora_adaface": {
        "file": "wandb_report-runs_sweep-vit_book_deit3_lora_ada.csv",
        "model": "deit3_base_patch16_224.fb_in1k",
        "head_type": "adaface",
        "swept_params": ["lr", "head_scale", "head_margin", "adaface_h", "adaface_t_alpha"],
        "fixed_lora": {"lora_r": 4, "lora_alpha": 24, "lora_dropout": 0.1,
                        "lora_qkv_proj": 1, "lora_train_bias": 0, "weight_decay": 2e-6},
    },
    "deit3_lora_arcface": {
        "file": "wandb_report-runs_sweep-vit_book_deit3_lora.csv",
        "model": "deit3_base_patch16_224.fb_in1k",
        "head_type": "arcface",
        "swept_params": ["lr", "weight_decay", "lora_r", "lora_alpha", "lora_dropout",
                          "lora_qkv_proj", "lora_train_bias", "head_scale"],
        "fixed_adaface": {"head_margin": 1e-5, "adaface_h": 0.2015, "adaface_t_alpha": 0.0438},
    },
}

METRIC = "tinyface/eval/rank@1"
METRIC2 = "tinyface/eval/rank@5"

HP_COLS = [
    "lora_r", "lora_alpha", "lora_dropout", "lora_qkv_proj", "lora_train_bias",
    "lr", "weight_decay", "head_scale", "head_margin", "head_type",
    "adaface_h", "adaface_t_alpha", "backbone_dropout",
]

os.makedirs(OUT_DIR, exist_ok=True)

best_configs = {}
summary_lines = []

def add(line=""):
    summary_lines.append(line)
    print(line)

add("# LoRA Sweep Analysis Summary")
add(f"\nGenerated from Phase 1.1 analysis script.\n")

for sweep_key, info in sweeps.items():
    path = os.path.join(DATA_DIR, info["file"])
    df = pd.read_csv(path)
    total_runs = len(df)

    # Clean: handle CSVs with or without State column
    if "State" in df.columns:
        df_clean = df[df["State"] == "finished"].copy()
    else:
        df_clean = df.copy()
    df_clean[METRIC] = pd.to_numeric(df_clean[METRIC], errors="coerce")
    df_clean = df_clean.dropna(subset=[METRIC])
    if "train/acc" in df_clean.columns:
        df_clean["train/acc"] = pd.to_numeric(df_clean["train/acc"], errors="coerce")
    df_clean = df_clean[df_clean[METRIC] > 0]
    successful = len(df_clean)
    failed = total_runs - successful

    # Sort
    df_clean = df_clean.sort_values(METRIC, ascending=False).reset_index(drop=True)

    add(f"\n---\n\n## {sweep_key}")
    add(f"\n- **File**: `{info['file']}`")
    add(f"- **Model**: `{info['model']}`")
    add(f"- **Head**: {info['head_type']}")
    add(f"- **Total runs**: {total_runs} (including header row)")
    add(f"- **Successful runs**: {successful}")
    add(f"- **Failed/NaN runs**: {failed} ({failed/total_runs*100:.1f}%)")

    # Swept vs fixed
    add(f"\n### Swept parameters")
    add(f"`{'`, `'.join(info['swept_params'])}`")

    if "fixed_lora" in info:
        add(f"\n### Fixed LoRA parameters")
        for k, v in info["fixed_lora"].items():
            add(f"- `{k}`: {v}")

    # Top 5 table
    top5 = df_clean.head(min(5, len(df_clean)))
    add(f"\n### Top {len(top5)} Runs\n")

    # Build table header
    header_cols = ["Rank", "Name", "rank@1"]
    if METRIC2 in df_clean.columns:
        header_cols.append("rank@5")
    for col in info["swept_params"]:
        if col in df_clean.columns:
            header_cols.append(col)
    
    add("| " + " | ".join(header_cols) + " |")
    add("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for i, (_, row) in enumerate(top5.iterrows()):
        vals = [str(i+1), str(row.get("Name", "?"))]
        vals.append(f"{row[METRIC]:.4f}")
        if METRIC2 in df_clean.columns:
            r5 = pd.to_numeric(row.get(METRIC2), errors="coerce")
            vals.append(f"{r5:.4f}" if pd.notna(r5) else "N/A")
        for col in info["swept_params"]:
            if col in df_clean.columns:
                v = row[col]
                if pd.notna(v):
                    try:
                        fv = float(v)
                        if fv == int(fv) and abs(fv) < 1000:
                            vals.append(str(int(fv)))
                        elif abs(fv) < 0.001:
                            vals.append(f"{fv:.2e}")
                        else:
                            vals.append(f"{fv:.4g}")
                    except (ValueError, TypeError):
                        vals.append(str(v))
                else:
                    vals.append("N/A")
        add("| " + " | ".join(vals) + " |")

    # HP clustering
    add(f"\n### Hyperparameter Clustering (top 5)")
    for col in info["swept_params"]:
        if col in df_clean.columns:
            vals = pd.to_numeric(top5[col], errors="coerce").dropna()
            if len(vals) > 1:
                add(f"- `{col}`: min={vals.min():.4g}, max={vals.max():.4g}, std={vals.std():.4g}")
            elif len(vals) == 1:
                add(f"- `{col}`: {vals.iloc[0]:.4g} (single value)")

    # Check overall distribution
    add(f"\n### Full Distribution Stats")
    for col in info["swept_params"]:
        if col in df_clean.columns:
            vals = pd.to_numeric(df_clean[col], errors="coerce").dropna()
            if len(vals) > 0:
                add(f"- `{col}`: mean={vals.mean():.4g}, std={vals.std():.4g}, range=[{vals.min():.4g}, {vals.max():.4g}]")

    # Best config
    best = top5.iloc[0]
    config = {
        "model_name": info["model"],
        "head_type": info["head_type"],
        "rank_at_1": round(float(best[METRIC]), 4),
        "source_wandb_name": str(best.get("Name", "unknown")),
    }

    for col in HP_COLS:
        if col in df_clean.columns and pd.notna(best[col]):
            val = best[col]
            try:
                val = float(val)
                if val == int(val) and abs(val) < 10000:
                    val = int(val)
            except (ValueError, TypeError):
                pass
            config[col] = val

    # Merge fixed params from sweep YAML
    if "fixed_lora" in info:
        for k, v in info["fixed_lora"].items():
            if k not in config:
                config[k] = v
    if "fixed_adaface" in info:
        for k, v in info["fixed_adaface"].items():
            if k not in config:
                config[k] = v

    best_configs[sweep_key] = config

    add(f"\n### Selected Best Config")
    add(f"**Run**: `{config['source_wandb_name']}` — **rank@1 = {config['rank_at_1']:.4f}**")

# Cross-sweep comparison
add("\n---\n\n## Cross-Sweep Comparison\n")
add("| Sweep | Head | Best rank@1 | Best Run |")
add("| --- | --- | --- | --- |")
for k, c in best_configs.items():
    add(f"| {k} | {c['head_type']} | {c['rank_at_1']:.4f} | `{c['source_wandb_name']}` |")

# Per-family comparison
add("\n### Per-Family Best")
for family in ["swin", "deit3"]:
    family_configs = {k: v for k, v in best_configs.items() if k.startswith(family)}
    best_key = max(family_configs, key=lambda x: family_configs[x]["rank_at_1"])
    best_c = family_configs[best_key]
    add(f"\n**{family.upper()} family winner**: `{best_key}` (rank@1 = {best_c['rank_at_1']:.4f})")
    runner_ups = {k: v for k, v in family_configs.items() if k != best_key}
    for rk, rv in runner_ups.items():
        delta = best_c["rank_at_1"] - rv["rank_at_1"]
        add(f"- vs `{rk}`: +{delta:.4f} rank@1")

# Instability notes
add("\n### Instability Observations\n")
for sweep_key, info in sweeps.items():
    path = os.path.join(DATA_DIR, info["file"])
    df = pd.read_csv(path)
    total = len(df)
    if "State" in df.columns:
        finished = len(df[df["State"] == "finished"])
        crashed = len(df[df["State"] == "crashed"]) if "crashed" in df["State"].values else 0
        nan_count = total - finished
    else:
        # Minimal CSV — count rows with valid metric as successful
        valid = pd.to_numeric(df[METRIC], errors="coerce").dropna()
        valid = valid[valid > 0]
        finished = len(valid)
        nan_count = total - finished
    if info["head_type"] == "adaface":
        add(f"- **{sweep_key}**: {nan_count}/{total} runs failed ({nan_count/total*100:.0f}%). "
            f"AdaFace sweeps show {'high' if nan_count/total > 0.2 else 'moderate' if nan_count/total > 0.1 else 'low'} failure rate.")
    else:
        add(f"- **{sweep_key}**: {nan_count}/{total} runs failed ({nan_count/total*100:.0f}%).")

# Save JSON
with open(os.path.join(OUT_DIR, "best_lora_configs.json"), "w") as f:
    json.dump(best_configs, f, indent=2)
add(f"\n\n---\n*Configs saved to `configs/best_lora_configs.json`*")

# Save summary markdown
with open(os.path.join(OUT_DIR, "sweep_analysis_summary.md"), "w") as f:
    f.write("\n".join(summary_lines))

print("\n\nDONE. Files saved to configs/")
