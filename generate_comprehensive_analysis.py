#!/usr/bin/env python
"""Generate comprehensive analysis CSVs combining old and new results.

Reads:
  - pruning_results/eval_pruned_quantized.csv     (old: 9 models × 3 precs = 27 rows)
  - pruning_results/eval_comprehensive.csv          (new: 15 models × 3 precs = 45 rows)
  - pruning_results/pruning_results.csv             (pruning metadata)
  - pruning_results/pruning_summary.csv             (summary with baselines)

Writes:
  - pruning_results/comprehensive_full_metrics.csv  (all 24 models × 3 precs = 72 rows)
  - pruning_results/comprehensive_compression.csv   (compression analysis with baselines)
  - pruning_results/comprehensive_pareto.csv        (pareto front data)
"""
import csv
import os
from pathlib import Path

OUT_DIR = "pruning_results"

# ── Source model baselines (unpruned accuracies and ONNX sizes) ───────────
# Architecture → {variant → {rank1, params_M, fp32_mb, fp16_mb, int8_mb}}
BASELINES = {
    "swin_tiny": {
        "Baseline": {
            "rank1": 0.534, "params_M": 27.5,
            "fp32_mb": 106.9, "fp16_mb": 54.1, "int8_mb": 28.3,
        },
        "KD Best (α=0.0)": {
            "rank1": 0.616, "params_M": 27.5,
            "fp32_mb": 106.9, "fp16_mb": 54.1, "int8_mb": 28.3,
        },
    },
    "deit3_small": {
        "Baseline": {
            "rank1": 0.552, "params_M": 21.7,
            "fp32_mb": 83.3, "fp16_mb": 42.0, "int8_mb": 22.4,
        },
    },
    "swin_base": {
        "Baseline": {
            "rank1": 0.526, "params_M": 86.7,
            "fp32_mb": 334.9, "fp16_mb": 168.6, "int8_mb": 87.2,
        },
    },
    "deit3_base": {
        "LoRA": {
            "rank1": 0.583, "params_M": 85.8,
            "fp32_mb": 327.7, "fp16_mb": 164.2, "int8_mb": 84.6,
        },
        "KD CVLFace (α=0.7)": {
            "rank1": 0.581, "params_M": 85.8,
            "fp32_mb": 327.7, "fp16_mb": 164.2, "int8_mb": 84.6,
        },
        "KD Best (α=0.1)": {
            "rank1": 0.593, "params_M": 85.8,
            "fp32_mb": 327.7, "fp16_mb": 164.2, "int8_mb": 84.6,
        },
    },
}

# ── Map experiment names to architecture/variant ──────────────────────────
EXPERIMENT_META = {
    # Old experiments (DeiT3-B LoRA)
    "deit3_base_lora-heads_0.10": ("deit3_base", "LoRA", "10% heads", 84.7),
    "deit3_base_lora-heads_0.25": ("deit3_base", "LoRA", "25% heads", 80.4),
    "deit3_base_lora-heads_0.50": ("deit3_base", "LoRA", "50% heads", 73.3),
    "deit3_base_lora-blocks_3":   ("deit3_base", "LoRA", "3 blocks", 66.2),
    # Old experiments (duplicated by v2 — use LoRA source ckpt, not baseline)
    "deit3_base_kd-heads_0.25":   ("deit3_base", "KD CVLFace (α=0.7)", "25% heads", 80.4),
    "swin_base-heads_0.10":       ("swin_base", "LoRA-source†", "10% heads", 87.1),
    "swin_base-heads_0.25":       ("swin_base", "LoRA-source†", "25% heads", 83.4),
    "swin_base-blocks_5":         ("swin_base", "LoRA-source†", "5 blocks", 73.3),
    "swin_tiny_kd-heads_0.10":    ("swin_tiny", "KD (α=0.7)", "10% heads", 28.9),
    # New experiments
    "swin_tiny_baseline-heads_0.10": ("swin_tiny", "Baseline", "10% heads", 27.09),
    "swin_tiny_baseline-heads_0.25": ("swin_tiny", "Baseline", "25% heads", 26.05),
    "swin_tiny_baseline-blocks_2":   ("swin_tiny", "Baseline", "2 blocks", 23.96),
    "swin_tiny_kd_best-heads_0.10":  ("swin_tiny", "KD Best (α=0.0)", "10% heads", 27.09),
    "swin_tiny_kd_best-heads_0.25":  ("swin_tiny", "KD Best (α=0.0)", "25% heads", 26.05),
    "swin_tiny_kd_best-blocks_2":    ("swin_tiny", "KD Best (α=0.0)", "2 blocks", 23.96),
    "deit3_small_baseline-heads_0.10": ("deit3_small", "Baseline", "10% heads", 20.92),
    "deit3_small_baseline-heads_0.25": ("deit3_small", "Baseline", "25% heads", 19.84),
    "deit3_small_baseline-blocks_2":   ("deit3_small", "Baseline", "2 blocks", 18.06),
    "swin_base_baseline-heads_0.10":   ("swin_base", "Baseline", "10% heads", 84.75),
    "swin_base_baseline-heads_0.25":   ("swin_base", "Baseline", "25% heads", 81.00),
    "swin_base_baseline-blocks_5":     ("swin_base", "Baseline", "5 blocks", 70.95),
    "deit3_base_kd_best-heads_0.10":   ("deit3_base", "KD Best (α=0.1)", "10% heads", 82.94),
    "deit3_base_kd_best-heads_0.25":   ("deit3_base", "KD Best (α=0.1)", "25% heads", 78.61),
    "deit3_base_kd_best-blocks_3":     ("deit3_base", "KD Best (α=0.1)", "3 blocks", 64.43),
}

ARCH_DISPLAY = {
    "swin_tiny": "Swin-T",
    "deit3_small": "DeiT3-S",
    "swin_base": "Swin-B",
    "deit3_base": "DeiT3-B",
}

# ONNX sizes from export logs
ONNX_SIZES = {
    # Old
    "deit3_base_lora-heads_0.10": (317.2, 158.9, 82.0),
    "deit3_base_lora-heads_0.25": (300.6, 150.7, 77.8),
    "deit3_base_lora-heads_0.50": (273.6, 137.1, 71.0),
    "deit3_base_lora-blocks_3":   (246.4, 123.5, 64.0),
    "deit3_base_kd-heads_0.25":   (300.6, 150.7, 77.8),
    "swin_base-heads_0.10":       (327.2, 164.8, 85.1),
    "swin_base-heads_0.25":       (312.6, 157.5, 81.2),
    "swin_base-blocks_5":         (273.9, 137.9, 71.2),
    "swin_tiny_kd-heads_0.10":    (105.3, 53.2, 27.8),
    # New
    "swin_tiny_baseline-heads_0.10": (105.3, 53.2, 27.8),
    "swin_tiny_baseline-heads_0.25": (101.2, 51.2, 26.7),
    "swin_tiny_baseline-blocks_2":   (93.1, 47.0, 24.6),
    "swin_tiny_kd_best-heads_0.10":  (105.3, 53.2, 27.8),
    "swin_tiny_kd_best-heads_0.25":  (101.2, 51.2, 26.7),
    "swin_tiny_kd_best-blocks_2":    (93.1, 47.0, 24.6),
    "deit3_small_baseline-heads_0.10": (80.6, 40.7, 21.7),
    "deit3_small_baseline-heads_0.25": (76.5, 38.6, 20.7),
    "deit3_small_baseline-blocks_2":   (69.6, 35.1, 18.9),
    "swin_base_baseline-heads_0.10":   (327.2, 164.7, 85.1),
    "swin_base_baseline-heads_0.25":   (312.7, 157.5, 81.2),
    "swin_base_baseline-blocks_5":     (273.9, 137.9, 71.2),
    "deit3_base_kd_best-heads_0.10":   (317.2, 158.9, 82.0),
    "deit3_base_kd_best-heads_0.25":   (300.6, 150.7, 77.8),
    "deit3_base_kd_best-blocks_3":     (246.4, 123.5, 64.0),
}


def read_eval_csv(path):
    """Read an evaluation CSV and return list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_full_metrics():
    """Combine old and new eval CSVs into comprehensive_full_metrics.csv."""
    rows = []

    # Old results
    old_path = os.path.join(OUT_DIR, "eval_pruned_quantized.csv")
    if os.path.exists(old_path):
        rows.extend(read_eval_csv(old_path))

    # New results
    new_path = os.path.join(OUT_DIR, "eval_comprehensive.csv")
    if os.path.exists(new_path):
        rows.extend(read_eval_csv(new_path))

    # Enrich with metadata
    out_rows = []
    for row in rows:
        model = row["model"]
        meta = EXPERIMENT_META.get(model)
        if meta is None:
            print(f"  WARNING: Unknown model {model}, skipping")
            continue
        arch, variant, prune_method, params_M = meta
        sizes = ONNX_SIZES.get(model, (0, 0, 0))
        prec = row["precision"]
        if prec == "FP32":
            size_mb = sizes[0]
        elif prec == "FP16":
            size_mb = sizes[1]
        else:
            size_mb = sizes[2]

        out_row = {
            "model": model,
            "architecture": ARCH_DISPLAY.get(arch, arch),
            "variant": variant,
            "prune_method": prune_method,
            "precision": prec,
            "params_M": params_M,
            "size_MB": round(size_mb, 1),
            "rank@1": row["rank@1"],
            "rank@2": row.get("rank@2", ""),
            "rank@3": row.get("rank@3", ""),
            "rank@4": row.get("rank@4", ""),
            "rank@5": row.get("rank@5", ""),
            "auc": row.get("auc", ""),
            "mAP": row.get("mAP", ""),
            "eer": row.get("eer", ""),
            "tpr_at_fpr_1pct": row.get("tpr_at_fpr_1pct", ""),
            "tpr_at_fpr_5pct": row.get("tpr_at_fpr_5pct", ""),
        }
        out_rows.append(out_row)

    # Write
    out_path = os.path.join(OUT_DIR, "comprehensive_full_metrics.csv")
    if out_rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"  Wrote {out_path} ({len(out_rows)} rows)")
    return out_rows


def build_compression_analysis(full_metrics):
    """Build comprehensive_compression.csv with baselines + pruned models."""
    rows = []

    # Add baseline entries (unpruned) for each architecture/variant
    for arch, variants in BASELINES.items():
        display = ARCH_DISPLAY.get(arch, arch)
        for variant, info in variants.items():
            for prec, size_key in [("FP32", "fp32_mb"), ("FP16", "fp16_mb"), ("INT8", "int8_mb")]:
                size = info[size_key]
                ratio = info["fp32_mb"] / size
                rows.append({
                    "architecture": display,
                    "variant": variant,
                    "prune_method": "None",
                    "precision": prec,
                    "params_M": info["params_M"],
                    "size_MB": round(size, 1),
                    "rank_at_1": info["rank1"],
                    "compression_ratio": round(ratio, 2),
                    "accuracy_drop": 0.0,
                    "model_label": f"{display} ({variant})",
                    "method_label": f"Original {prec}",
                })

    # Add pruned model entries
    for row in full_metrics:
        model = row["model"]
        meta = EXPERIMENT_META.get(model)
        if meta is None:
            continue
        arch, variant, prune_method, params_M = meta
        display = ARCH_DISPLAY.get(arch, arch)

        # Get baseline FP32 size for compression ratio
        baseline_info = BASELINES.get(arch, {}).get(variant)
        if baseline_info is None:
            # Try matching variant
            for v, info in BASELINES.get(arch, {}).items():
                baseline_info = info
                break
        if baseline_info is None:
            continue

        fp32_size = baseline_info["fp32_mb"]
        baseline_rank1 = baseline_info["rank1"]
        size = float(row["size_MB"])
        rank1 = float(row["rank@1"])

        rows.append({
            "architecture": display,
            "variant": variant,
            "prune_method": prune_method,
            "precision": row["precision"],
            "params_M": params_M,
            "size_MB": round(size, 1),
            "rank_at_1": rank1,
            "compression_ratio": round(fp32_size / size, 2) if size > 0 else 0,
            "accuracy_drop": round(baseline_rank1 - rank1, 4),
            "model_label": f"{display} ({variant})",
            "method_label": f"Pruned {prune_method} {row['precision']}",
        })

    # Write
    out_path = os.path.join(OUT_DIR, "comprehensive_compression.csv")
    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  Wrote {out_path} ({len(rows)} rows)")
    return rows


def build_pareto(compression_rows):
    """Build comprehensive_pareto.csv for Pareto front plots."""
    rows = []
    for r in compression_rows:
        rows.append({
            "architecture": r["architecture"],
            "variant": r["variant"],
            "model_label": r["model_label"],
            "method": r["method_label"],
            "size_MB": r["size_MB"],
            "compression_ratio": r["compression_ratio"],
            "rank_at_1": r["rank_at_1"],
            "params_M": r["params_M"],
        })

    out_path = os.path.join(OUT_DIR, "comprehensive_pareto.csv")
    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    print("Building comprehensive analysis CSVs...")
    full = build_full_metrics()
    comp = build_compression_analysis(full)
    build_pareto(comp)
    print("\nDone!")
