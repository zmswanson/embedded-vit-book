#!/usr/bin/env python3
"""Phase 5.5 — Generate accuracy delta summary table.

Reads inference_results/onnx_accuracy_verification.csv (produced by
eval_onnx_accuracy.py) and compares against PyTorch baseline rank@1 values
from the reference CSVs to produce a clean delta table.

Usage:
    python generate_accuracy_delta_summary.py
"""
import csv
import sys
from pathlib import Path

OUT_FILE = "inference_results/accuracy_delta_summary.csv"

# Canonical PyTorch reference: architecture → rank@1
# Source: final_models_inference_results.csv + model_ablation_inference_results.csv
# (the finetuned-baseline checkpoints used for ONNX export, best run per arch)
# PyTorch reference rank@1 for each ONNX baseline — sourced via wandb_id from
# the export scripts to match the exact checkpoint that was exported to ONNX.
# Swin-T/S/B were exported from non-best-run checkpoints, so these are lower
# than the best observed values (which come from different training runs).
PYTORCH_REFS = {
    "convnext_base":    0.5427,   # m4xjjh07
    "convnext_small":   0.5370,   # ccc9z3g5
    "convnext_tiny":    0.5317,   # hzvjhovx
    "deit3_base":       0.5834,   # vimriwcl
    "deit3_medium":     0.5703,   # k102lp3v
    "deit3_small":      0.5515,   # rdsdt7xb
    "mobilevitv2_200":  0.5429,   # amsv29wr
    "pvt_v2_b2":        0.5421,   # uvu73fxo
    "pvt_v2_b3":        0.5429,   # eltlatns
    "pvt_v2_b5":        0.5467,   # nvzory9h
    "resnet101d":       0.4973,   # qopz1ks6
    "resnet200d":       0.4678,   # asyv1awr
    "resnet50d":        0.4764,   # 4860x6ia
    # Swin baselines: exported from non-best-run checkpoints — these values are
    # the test-set rank@1 of the specific exported checkpoint, not the best run.
    "swin_base":        0.5260,   # xd7hukv3
    "swin_small":       0.5392,   # rbeovm8i
    "swin_tiny":        0.5335,   # p09btsx3
    # KD / LoRA spot-check references (exact checkpoint used for export)
    "deit3_base_kd_cvlface_alpha0.0":    0.5724,  # 4uwgz0ln
    "deit3_base_kd_cvlface_ft":          0.5807,  # 5mngq5jn
    "deit3_base_kd_cvlface_ft_lora":     0.5472,  # 3d7j4ck4
    "deit3_base_kd_petalface_alpha0.1":  0.5858,  # obc9cvpu
    "swin_base_kd_petalface_ft":         0.5215,  # 7t2r820f
    "swin_tiny_lora":   0.5617,   # f0frj93y
    "deit3_base_lora":  0.5496,   # 2ljjzfzb
    # Pruned model references (from pruning_results/comprehensive_full_metrics.csv + eval_comprehensive.csv)
    "swin_tiny_kd-heads_0.10":           0.5051,
    "swin_tiny_kd_best-blocks_2":        0.5829,
    "swin_tiny_kd_best-heads_0.10":      0.4032,
    "swin_tiny_kd_best-heads_0.25":      0.3801,
    "swin_tiny_baseline-blocks_2":       0.5440,
    "swin_tiny_baseline-heads_0.10":     0.4995,
    "swin_tiny_baseline-heads_0.25":     0.4646,
    "deit3_base_kd_best-blocks_3":       0.5491,
    "deit3_base_kd_best-heads_0.10":     0.5928,
    "deit3_base_kd_best-heads_0.25":     0.5652,
    "deit3_base_lora-heads_0.10":        0.5896,
    "deit3_base_lora-heads_0.25":        0.5655,
    "deit3_base_lora-heads_0.50":        0.5131,
    "deit3_base_lora-blocks_3":          0.5464,
    "deit3_small_baseline-blocks_2":     0.4981,
    "deit3_small_baseline-heads_0.10":   0.5644,
    "deit3_small_baseline-heads_0.25":   0.5515,
    "swin_base_baseline-blocks_5":       0.5528,
    "swin_base_baseline-heads_0.10":     0.4874,
    "swin_base_baseline-heads_0.25":     0.4541,
}

# Thresholds from the prompt
THRESHOLDS = {"fp32": 0.001, "fp16": 0.003, "int8": 0.005}


def load_onnx_results(path: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({
            "model":       r["model"],
            "precision":   r["precision"],
            "rank1":       float(r["rank1"]),
            "pytorch_ref": float(r["pytorch_ref"]) if r["pytorch_ref"] else None,
            "delta":       float(r["delta"]) if r["delta"] else None,
            "status":      r["status"],
        })
    return rows


def make_delta_table(rows: list[dict]) -> list[dict]:
    """Build a clean per-model-per-precision delta table with pass/fail.

    Reference priority:
    1. PYTORCH_REFS dict (authoritative — sourced from export scripts via wandb_id)
    2. Base-arch fallback from PYTORCH_REFS (for pruned models whose base arch is known)
    The pytorch_ref column from the CSV is intentionally IGNORED here because
    it may have been written with an older reference table.
    """
    out = []
    for r in rows:
        model = r["model"]
        prec  = r["precision"]
        rank1 = r["rank1"]

        # Always use PYTORCH_REFS as authority
        pt = PYTORCH_REFS.get(model)
        if pt is None:
            # Fallback: strip pruning/training variant suffixes to get base arch
            base = model.split("_kd_")[0].split("_lora")[0].split("_baseline")[0]
            base = base.split("-")[0]  # strip e.g. -heads_0.10, -blocks_3
            pt = PYTORCH_REFS.get(base)

        delta = round(pt - rank1, 4) if pt else None

        thr = THRESHOLDS.get(prec, 0.005)
        if delta is None:
            status = "no_ref"
        elif delta > thr:
            status = f"FAIL(>{thr:.3f})"
        elif delta < -0.01:
            status = "HIGHER"
        else:
            status = "PASS"

        out.append({
            "model":       model,
            "precision":   prec,
            "onnx_rank1":  rank1,
            "pytorch_ref": round(pt, 4) if pt else "",
            "delta":       round(delta, 4) if delta is not None else "",
            "threshold":   thr,
            "status":      status,
        })
    return out


def print_table(rows: list[dict]):
    """Pretty-print grouped by precision."""
    for prec in ["fp32", "fp16", "int8"]:
        subset = [r for r in rows if r["precision"] == prec]
        if not subset:
            continue
        print(f"\n── {prec.upper()} ({len(subset)} models) ──")
        print(f"  {'Model':<45} {'ONNX':>6} {'PyTorch':>8} {'Δ':>7}  Status")
        print("  " + "─" * 76)
        for r in sorted(subset, key=lambda x: x["model"]):
            flag = "✓" if r["status"] in ("PASS", "HIGHER", "no_ref") else "⚠"
            pt_str = f"{r['pytorch_ref']:>8.4f}" if r["pytorch_ref"] else "       -"
            d_str  = f"{r['delta']:>+7.4f}" if r["delta"] != "" else "      -"
            print(f"  {flag} {r['model']:<44} {r['onnx_rank1']:>6.4f} {pt_str} {d_str}  {r['status']}")

        fails = [r for r in subset if "FAIL" in str(r["status"])]
        passes = [r for r in subset if r["status"] == "PASS"]
        higher = [r for r in subset if r["status"] == "HIGHER"]
        no_ref = [r for r in subset if r["status"] == "no_ref"]
        print(f"\n  Summary: {len(passes)} PASS, {len(higher)} HIGHER, "
              f"{len(fails)} FAIL, {len(no_ref)} no_ref")
        if fails:
            print(f"  ⚠  FAILURES: {[r['model'] for r in fails]}")
    
    # Spot-check section (non-baselines)
    other = [r for r in rows if r["precision"] == "fp32" and
             not any(r["model"] == b for b in PYTORCH_REFS)]
    if other:
        print(f"\n── KD / LoRA / Pruned spot-check (FP32, {len(other)} models) ──")
        print(f"  {'Model':<55} {'ONNX':>6} {'Base ref':>9} {'Δ':>7}  Status")
        print("  " + "─" * 86)
        for r in sorted(other, key=lambda x: x["model"]):
            flag = "✓" if r["status"] in ("PASS", "HIGHER", "no_ref") else "⚠"
            pt_str = f"{r['pytorch_ref']:>9.4f}" if r["pytorch_ref"] else "        -"
            d_str  = f"{r['delta']:>+7.4f}" if r["delta"] != "" else "      -"
            print(f"  {flag} {r['model']:<54} {r['onnx_rank1']:>6.4f} {pt_str} {d_str}  {r['status']}")


def main():
    inp = "inference_results/onnx_accuracy_verification.csv"
    if not Path(inp).exists():
        print(f"ERROR: {inp} not found. Run eval_onnx_accuracy.py first.", file=sys.stderr)
        sys.exit(1)

    raw_rows = load_onnx_results(inp)
    print(f"Loaded {len(raw_rows)} rows from {inp}")

    delta_rows = make_delta_table(raw_rows)
    print_table(delta_rows)

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "precision", "onnx_rank1", "pytorch_ref",
            "delta", "threshold", "status"
        ])
        writer.writeheader()
        writer.writerows(delta_rows)

    print(f"\nDelta summary written to {OUT_FILE}")

    # Overall pass/fail
    fails = [r for r in delta_rows if "FAIL" in str(r["status"])]
    if fails:
        print(f"\n⚠  {len(fails)} models exceeded accuracy drop threshold:")
        for r in fails:
            print(f"   {r['model']} [{r['precision']}]: Δ={r['delta']:+.4f} "
                  f"(threshold={r['threshold']:.3f})")
    else:
        print(f"\n✓ All models within accuracy drop thresholds.")


if __name__ == "__main__":
    main()
