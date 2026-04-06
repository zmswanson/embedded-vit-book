#!/usr/bin/env python3
"""Accuracy verification via ONNX Runtime CUDA EP on the training rig.

Runs TinyFace test-set rank@1 evaluation for ONNX models of all precisions.
Uses existing model_eval utilities so evaluation is identical to PyTorch runs.

Usage:
    # All 16 FP32 baselines + key KD/LoRA/pruned FP32 models
    conda run -n vit-benchmark python eval_onnx_accuracy.py --scope baselines_fp32

    # All INT8 baselines
    conda run -n vit-benchmark python eval_onnx_accuracy.py --scope baselines_int8

    # All FP16 baselines
    conda run -n vit-benchmark python eval_onnx_accuracy.py --scope baselines_fp16

    # KD + LoRA + pruned spot-check (FP32 only)
    conda run -n vit-benchmark python eval_onnx_accuracy.py --scope kd_spot

    # All of the above in sequence
    conda run -n vit-benchmark python eval_onnx_accuracy.py --scope all

    # Single model, e.g.:
    conda run -n vit-benchmark python eval_onnx_accuracy.py --model onnx_models/baselines/swin_tiny.onnx
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.tinyface.dataloader import get_test_loaders, DatasetType
from model_eval.probe_gallery import (
    get_batch_features, get_cosine_similarity, get_rank_k_accuracy,
    MultiLabelMethod,
)

TINYFACE_DIR = "/mnt/data/biometrics/tinyface"
IMG_SIZE = 96
BATCH_SIZE = 32     # DataLoader batch for data loading efficiency; ORT runs BS=1 per image
NUM_WORKERS = 4
OUT_CSV = "inference_results/onnx_accuracy_verification.csv"


# ---------------------------------------------------------------------------
# ONNX-runtime model wrapper — presents the same API as a PyTorch nn.Module
# so it plugs directly into get_batch_features().
# ---------------------------------------------------------------------------
class OrtWrapper:
    """Wraps an ORT InferenceSession to look like a callable PyTorch model."""

    def __init__(self, onnx_path: str, providers=None):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                         providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def to(self, device):
        return self   # no-op: device is implicit in the provider

    def eval(self):
        return self   # no-op

    def __call__(self, images: torch.Tensor):
        # All exported models have static BS=1; loop per image and stack results.
        # images may be on CUDA; ORT needs CPU numpy.
        x = images.detach().cpu().numpy().astype(np.float32)
        results = [self.sess.run(None, {self.input_name: x[i:i+1]})[0]
                   for i in range(x.shape[0])]
        return torch.tensor(np.vstack(results))


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_onnx(onnx_path: str, label: str = None) -> float:
    """Return rank@1 on the TinyFace test set for a single ONNX model."""
    label = label or Path(onnx_path).name
    print(f"  Evaluating {label} ...", end="", flush=True)
    t0 = time.time()

    model = OrtWrapper(str(onnx_path))

    gallery_loader, probe_loader = get_test_loaders(
        TINYFACE_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE,
        num_workers=NUM_WORKERS
    )

    # get_batch_features calls model.to(device).eval() then model(images)
    # OrtWrapper's to/eval are no-ops; feature output: (N, D) torch tensor
    gallery_feats, gallery_labels = get_batch_features(model, gallery_loader,
                                                        device="cpu")
    probe_feats, probe_labels = get_batch_features(model, probe_loader,
                                                    device="cpu")

    sim = get_cosine_similarity(probe_feats, gallery_feats, device="cpu")
    rank_k = get_rank_k_accuracy(sim, probe_labels, gallery_labels, k=1,
                                  multi_label_method=MultiLabelMethod.ALL)
    rank1 = rank_k[0] if isinstance(rank_k, (list, tuple)) else float(rank_k)

    elapsed = time.time() - t0
    print(f"  rank@1 = {rank1:.4f}  ({elapsed:.0f}s)")
    return rank1


# ---------------------------------------------------------------------------
# Model lists per scope
# ---------------------------------------------------------------------------
def models_baselines_fp32():
    return [
        (f"onnx_models/baselines/{p.name}", p.stem, "fp32")
        for p in sorted(Path("onnx_models/baselines").glob("*.onnx"))
    ]

def models_baselines_fp16():
    return [
        (f"onnx_models_fp16/baselines/{p.name}", p.stem, "fp16")
        for p in sorted(Path("onnx_models_fp16/baselines").glob("*.onnx"))
    ]

def models_baselines_int8():
    return [
        (f"onnx_models_int8/baselines/{p.name}", p.stem, "int8")
        for p in sorted(Path("onnx_models_int8/baselines").glob("*.onnx"))
    ]

def models_kd_spot():
    """Key KD / LoRA / pruned models to spot-check at FP32."""
    targets = [
        # KD CVLFace — best DeiT3-B variants
        ("onnx_models/kd_cvlface/deit3_base_kd_cvlface_alpha0.0.onnx",
                                  "deit3_base_kd_cvlface_alpha0.0", "fp32"),
        ("onnx_models/kd_cvlface/deit3_base_kd_cvlface_ft.onnx",
                                  "deit3_base_kd_cvlface_ft", "fp32"),
        ("onnx_models/kd_cvlface/deit3_base_kd_cvlface_ft_lora.onnx",
                                  "deit3_base_kd_cvlface_ft_lora", "fp32"),
        # KD PETALFace — best Swin and DeiT3-B variants
        ("onnx_models/kd_petalface/deit3_base_kd_petalface_alpha0.1.onnx",
                                   "deit3_base_kd_petalface_alpha0.1", "fp32"),
        ("onnx_models/kd_petalface/swin_base_kd_petalface_ft.onnx",
                                   "swin_base_kd_petalface_ft", "fp32"),
        # LoRA
        ("onnx_models/lora/swin_tiny_lora.onnx",    "swin_tiny_lora",    "fp32"),
        ("onnx_models/lora/deit3_base_lora.onnx",   "deit3_base_lora",   "fp32"),
        # Pruned v2 — best models across architectures
        ("onnx_models/pruned_v2/swin_tiny_kd_best-blocks_2.onnx",
                                  "swin_tiny_kd_best-blocks_2",   "fp32"),
        ("onnx_models/pruned_v2/swin_tiny_kd_best-heads_0.10.onnx",
                                  "swin_tiny_kd_best-heads_0.10", "fp32"),
        ("onnx_models/pruned_v2/deit3_base_kd_best-heads_0.10.onnx",
                                  "deit3_base_kd_best-heads_0.10", "fp32"),
        ("onnx_models/pruned_v2/deit3_base_kd_best-blocks_3.onnx",
                                  "deit3_base_kd_best-blocks_3",  "fp32"),
        ("onnx_models/pruned_v2/swin_base_baseline-blocks_5.onnx",
                                  "swin_base_baseline-blocks_5",  "fp32"),
        # Pruned v1 — LoRA pruned
        ("onnx_models/pruned/deit3_base_lora-heads_0.10.onnx",
                               "deit3_base_lora-heads_0.10", "fp32"),
        ("onnx_models/pruned/swin_tiny_kd-heads_0.10.onnx",
                               "swin_tiny_kd-heads_0.10",    "fp32"),
    ]
    return [(p, n, pr) for p, n, pr in targets if Path(p).exists()]


def build_model_list(scope: str):
    if scope == "baselines_fp32":
        return models_baselines_fp32()
    if scope == "baselines_fp16":
        return models_baselines_fp16()
    if scope == "baselines_int8":
        return models_baselines_int8()
    if scope == "kd_spot":
        return models_kd_spot()
    if scope == "all":
        return (models_baselines_fp32() + models_baselines_fp16() +
                models_baselines_int8() + models_kd_spot())
    raise ValueError(f"Unknown scope: {scope}")


# ---------------------------------------------------------------------------
# Reference accuracy lookup
# Source: wandb_id → test rank@1 via the export shell scripts.
# Each value is the test-set rank@1 of the EXACT checkpoint used for ONNX
# export (not the best-ever run), so ONNX FP32 delta should be ≈ 0.0000.
# ---------------------------------------------------------------------------
ONNX_STEM_REF: dict[str, float] = {
    # Baselines (from export_ablation_onnx.sh + export_all_onnx.sh)
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
    # Swin baselines (exported from non-best runs)
    "swin_base":        0.5260,   # xd7hukv3
    "swin_small":       0.5392,   # rbeovm8i
    "swin_tiny":        0.5335,   # p09btsx3
    # LoRA (from export_all_onnx.sh)
    "swin_tiny_lora":   0.5617,   # f0frj93y
    "swin_small_lora":  0.5545,   # ymm0k0ub
    "swin_base_lora":   0.5359,   # ci53ufy8
    "deit3_small_lora": 0.5274,   # cocm8b7d
    "deit3_medium_lora":0.5333,   # dpwn5kb1
    "deit3_base_lora":  0.5496,   # 2ljjzfzb
    # KD CVLFace (key models)
    "deit3_base_kd_cvlface_alpha0.0":       0.5724,  # 4uwgz0ln
    "deit3_base_kd_cvlface_alpha0.8":       0.5759,  # 0hodmg2y
    "deit3_base_kd_cvlface_frozen":         0.5719,  # f63auzxo
    "deit3_base_kd_cvlface_frozen_lora":    0.5469,  # fz8pc2je
    "deit3_base_kd_cvlface_ft":             0.5807,  # 5mngq5jn
    "deit3_base_kd_cvlface_ft_lora":        0.5472,  # 3d7j4ck4
    "swin_tiny_kd_cvlface_alpha0.0":        0.6159,  # zz76v2nu
    "swin_tiny_kd_cvlface_frozen":          0.5491,  # y6ddjqyw
    "swin_tiny_kd_cvlface_ft":              0.5357,  # 5wabjm9e
    # KD PETALFace (key models)
    "deit3_base_kd_petalface_alpha0.1":     0.5858,  # obc9cvpu
    "deit3_base_kd_petalface_alpha0.7":     0.5756,  # mmywymd2
    "swin_base_kd_petalface_ft":            0.5215,  # 7t2r820f
}


def load_pytorch_refs() -> dict:
    """Return {onnx_stem: pytorch_rank1} from the per-checkpoint reference table."""
    return dict(ONNX_STEM_REF)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scope",
                       choices=["baselines_fp32","baselines_fp16",
                                "baselines_int8","kd_spot","all"])
    group.add_argument("--model", help="Path to a single ONNX file")
    parser.add_argument("--out", default=OUT_CSV)
    parser.add_argument("--resume", action="store_true",
                        help="Skip models already in output CSV")
    args = parser.parse_args()

    # Load existing results if resuming
    existing = {}
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        for r in csv.DictReader(open(out_path)):
            existing[(r["model"], r["precision"])] = float(r["rank1"])
        print(f"  Resuming: {len(existing)} existing results loaded")

    # Build list of (onnx_path, model_stem, precision)
    if args.model:
        p = Path(args.model)
        # infer precision from directory name
        prec = "fp32"
        if "fp16" in str(p): prec = "fp16"
        elif "int8" in str(p): prec = "int8"
        models = [(str(p), p.stem, prec)]
    else:
        models = build_model_list(args.scope)

    pytorch_refs = load_pytorch_refs()

    print(f"\n=== Phase 5.5 ONNX Accuracy Verification ===")
    print(f"  Models to evaluate: {len(models)}")
    print(f"  TinyFace dir: {TINYFACE_DIR}")
    print(f"  Output: {args.out}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "precision", "rank1", "pytorch_ref", "delta", "status"]

    # Open in append mode if resuming, write mode otherwise
    mode = "a" if (args.resume and out_path.exists()) else "w"
    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        for onnx_path, model_stem, precision in models:
            key = (model_stem, precision)
            if args.resume and key in existing:
                print(f"  [skip] {model_stem} ({precision})")
                continue

            if not Path(onnx_path).exists():
                print(f"  [missing] {onnx_path}")
                continue

            rank1 = evaluate_onnx(onnx_path, f"{model_stem} [{precision}]")

            # Find pytorch ref — use exact stem first, then basename match
            pt_ref = pytorch_refs.get(model_stem)
            if pt_ref is None:
                # For KD/LoRA/pruned: try to find the base model stem
                base = model_stem.split("_kd_")[0].split("_lora")[0].split("_baseline")[0]
                base = base.split("_kd")[0]
                pt_ref = pytorch_refs.get(base)

            delta = round(pt_ref - rank1, 4) if pt_ref else None

            # Determine pass/fail thresholds
            thresholds = {"fp32": 0.001, "fp16": 0.003, "int8": 0.005}
            thr = thresholds.get(precision, 0.005)
            if delta is None:
                status = "no_ref"
            elif delta > thr:
                status = f"FAIL(>{thr:.3f})"
            elif delta < -0.01:
                status = "HIGHER"   # significantly better (unexpected)
            else:
                status = "PASS"

            row = {
                "model": model_stem,
                "precision": precision,
                "rank1": round(rank1, 4),
                "pytorch_ref": round(pt_ref, 4) if pt_ref else "",
                "delta": round(delta, 4) if delta is not None else "",
                "status": status,
            }
            writer.writerow(row)
            f.flush()

            flag = "✓" if status in ("PASS", "HIGHER", "no_ref") else "⚠"
            pt_str = f"  (ref={pt_ref:.4f}, Δ={delta:+.4f} → {status})" if pt_ref else ""
            print(f"  {flag} {model_stem} [{precision}]: {rank1:.4f}{pt_str}")

    print(f"\n=== Done. Results in {args.out} ===")


if __name__ == "__main__":
    main()
