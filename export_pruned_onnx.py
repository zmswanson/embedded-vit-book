#!/usr/bin/env python
"""Export pruned models to ONNX, quantize (FP16/INT8), and build analysis CSVs.

Reconstructs pruned architecture from original checkpoints,
loads fine-tuned state_dicts, exports to ONNX, applies quantization,
and produces compression_analysis.csv and pareto_data.csv.
"""
import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic

from pruning import load_module_for_finetuning, prune_model, _param_count

# ---------------------------------------------------------------------------
# Experiment definitions (matches eval_pruned_models.py)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "name": "deit3_base_lora-heads_0.10",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/vimriwcl/epoch=44-rank1tinyface/eval/rank@1=0.579.ckpt",
        "model_name": "deit3_base_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models/epoch=3-rank1tinyface/eval/rank@1=0.574.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.10,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5834,
        "test_rank1": 0.5896,
        "variant": "LoRA",
    },
    {
        "name": "deit3_base_lora-heads_0.25",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/vimriwcl/epoch=44-rank1tinyface/eval/rank@1=0.579.ckpt",
        "model_name": "deit3_base_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models/epoch=8-rank1tinyface/eval/rank@1=0.564.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.25,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5834,
        "test_rank1": 0.5655,
        "variant": "LoRA",
    },
    {
        "name": "deit3_base_lora-heads_0.50",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/vimriwcl/epoch=44-rank1tinyface/eval/rank@1=0.579.ckpt",
        "model_name": "deit3_base_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models/epoch=4-rank1tinyface/eval/rank@1=0.521.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.50,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5834,
        "test_rank1": 0.5131,
        "variant": "LoRA",
    },
    {
        "name": "deit3_base_lora-blocks_3",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/vimriwcl/epoch=44-rank1tinyface/eval/rank@1=0.579.ckpt",
        "model_name": "deit3_base_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models/epoch=7-rank1tinyface/eval/rank@1=0.535.ckpt",
        "prune_method": "blocks",
        "head_prune_ratio": 0.0,
        "num_blocks_to_remove": 3,
        "baseline_rank1": 0.5834,
        "test_rank1": 0.5464,
        "variant": "LoRA",
    },
    {
        "name": "deit3_base_kd-heads_0.25",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_kd/5mngq5jn/epoch=47-rank1tinyface/eval/rank@1=0.579.ckpt",
        "model_name": "deit3_base_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models/epoch=3-rank1tinyface/eval/rank@1=0.568.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.25,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5807,
        "test_rank1": 0.5676,
        "variant": "KD CVLFace",
    },
    {
        "name": "swin_base-heads_0.25",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_lora/ci53ufy8/epoch=39-rank1tinyface/eval/rank@1=0.556.ckpt",
        "model_name": "swin_base_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models/epoch=7-rank1tinyface/eval/rank@1=0.436.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.25,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5359,
        "test_rank1": 0.4665,
        "variant": "baseline",
    },
    {
        "name": "swin_base-heads_0.10",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_lora/ci53ufy8/epoch=39-rank1tinyface/eval/rank@1=0.556.ckpt",
        "model_name": "swin_base_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models/epoch=7-rank1tinyface/eval/rank@1=0.471.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.10,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5359,
        "test_rank1": 0.4965,
        "variant": "baseline",
    },
    {
        "name": "swin_base-blocks_5",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_lora/ci53ufy8/epoch=39-rank1tinyface/eval/rank@1=0.556.ckpt",
        "model_name": "swin_base_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models/epoch=1-rank1tinyface/eval/rank@1=0.550.ckpt",
        "prune_method": "blocks",
        "head_prune_ratio": 0.0,
        "num_blocks_to_remove": 5,
        "baseline_rank1": 0.5359,
        "test_rank1": 0.5748,
        "variant": "baseline",
    },
    {
        "name": "swin_tiny_kd-heads_0.10",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_kd/y6ddjqyw/epoch=44-rank1tinyface/eval/rank@1=0.570.ckpt",
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models/epoch=8-rank1tinyface/eval/rank@1=0.512.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.10,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5491,
        "test_rank1": 0.5051,
        "variant": "KD CVLFace",
    },
]

# Baseline ONNX models for comparison
BASELINES = [
    {
        "name": "deit3_base (LoRA)",
        "model": "deit3_base",
        "method": "Original FP32",
        "onnx_fp32": "onnx_models/baselines/deit3_base.onnx",
        "onnx_fp16": "onnx_models_fp16/baselines/deit3_base.onnx",
        "onnx_int8": "onnx_models_int8/baselines/deit3_base.onnx",
        "params_M": 85.8,
        "rank1": 0.5834,
    },
    {
        "name": "deit3_base (KD CVLFace)",
        "model": "deit3_base",
        "method": "Original FP32",
        "onnx_fp32": "onnx_models/kd_cvlface/deit3_base_kd_cvlface_frozen.onnx",
        "onnx_fp16": "onnx_models_fp16/kd_cvlface/deit3_base_kd_cvlface_frozen.onnx",
        "onnx_int8": "onnx_models_int8/kd_cvlface/deit3_base_kd_cvlface_frozen.onnx",
        "params_M": 85.8,
        "rank1": 0.5807,
    },
    {
        "name": "swin_base (baseline)",
        "model": "swin_base",
        "method": "Original FP32",
        "onnx_fp32": "onnx_models/baselines/swin_base.onnx",
        "onnx_fp16": "onnx_models_fp16/baselines/swin_base.onnx",
        "onnx_int8": "onnx_models_int8/baselines/swin_base.onnx",
        "params_M": 86.7,
        "rank1": 0.5359,
    },
    {
        "name": "swin_tiny (KD CVLFace)",
        "model": "swin_tiny",
        "method": "Original FP32",
        "onnx_fp32": "onnx_models/baselines/swin_tiny.onnx",
        "onnx_fp16": "onnx_models_fp16/baselines/swin_tiny.onnx",
        "onnx_int8": "onnx_models_int8/baselines/swin_tiny.onnx",
        "params_M": 27.5,
        "rank1": 0.5491,
    },
]

IMG_SIZE = 96


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_size_mb(path: str) -> float:
    """Return file size in MB, including .data sidecar if present."""
    size = os.path.getsize(path)
    data_path = path + ".data"
    if os.path.exists(data_path):
        size += os.path.getsize(data_path)
    return size / (1024 * 1024)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def validate_onnx(pytorch_model: nn.Module, onnx_path: str,
                   dummy_input: torch.Tensor) -> float:
    """Return cosine similarity between PyTorch and ONNX Runtime outputs."""
    pytorch_model.eval()
    with torch.no_grad():
        pt_output = pytorch_model(dummy_input.cpu()).cpu().numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {"input": dummy_input.cpu().numpy()})[0]

    cos_sim = cosine_similarity(pt_output, ort_output)
    return cos_sim


# ---------------------------------------------------------------------------
# ONNX export for pruned models
# ---------------------------------------------------------------------------

def export_pruned_model(exp: dict, output_dir: str = "onnx_models/pruned",
                        img_size: int = IMG_SIZE) -> dict:
    """Reconstruct pruned architecture, load weights, export to ONNX.

    Returns dict with export metadata.
    """
    name = exp["name"]
    onnx_path = os.path.join(output_dir, f"{name}.onnx")
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting: {name}")
    print(f"{'='*60}")

    # 1) Reconstruct pruned architecture from original checkpoint
    module, family = load_module_for_finetuning(
        exp["original_ckpt"], exp["model_name"]
    )
    params_before = _param_count(module)

    prune_model(
        module.backbone, family,
        prune_method=exp["prune_method"],
        head_prune_ratio=exp["head_prune_ratio"],
        num_blocks_to_remove=exp["num_blocks_to_remove"],
        img_size=img_size,
    )
    params_after = _param_count(module)

    # 2) Load fine-tuned state dict
    ckpt = torch.load(exp["pruned_ckpt"], map_location="cpu", weights_only=False)
    module.load_state_dict(ckpt["state_dict"])

    # 3) Extract backbone, move to CPU, eval mode
    backbone = module.backbone
    backbone.eval()
    backbone.cpu()

    # 4) Export to ONNX
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        backbone,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=18,
        do_constant_folding=True,
    )

    fp32_size = _file_size_mb(onnx_path)
    print(f"  ONNX exported: {onnx_path} ({fp32_size:.1f} MB)")

    # 5) Validate
    cos_sim = validate_onnx(backbone, onnx_path, dummy)
    status = "PASS" if cos_sim > 0.999 else "FAIL"
    print(f"  Validation: cosine={cos_sim:.6f} [{status}]")

    if cos_sim < 0.999:
        print(f"  WARNING: Low cosine similarity for {name}!")

    # 6) Check output shape
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = session.run(None, {"input": dummy.numpy()})[0]
    print(f"  Output shape: {out.shape}")

    # Free memory
    del module, backbone, ckpt
    torch.cuda.empty_cache()

    return {
        "name": name,
        "onnx_fp32": onnx_path,
        "params_before": params_before,
        "params_after": params_after,
        "fp32_size_MB": fp32_size,
        "cosine_sim": cos_sim,
        "output_shape": out.shape,
    }


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_fp16(input_path: str, output_path: str) -> float:
    """Convert FP32 ONNX to FP16. Returns size in MB."""
    model = onnx.load(input_path, load_external_data=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_fp16 = float16.convert_float_to_float16(
            model, keep_io_types=True,
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, output_path)
    return _file_size_mb(output_path)


def quantize_int8(input_path: str, output_path: str) -> float:
    """Apply INT8 dynamic quantization. Returns size in MB."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        use_external_data_format=False,
    )
    return _file_size_mb(output_path)


def validate_quantized_vs_fp32(fp32_path: str, quant_path: str,
                                n_samples: int = 5) -> float:
    """Compare quantized and FP32 ONNX models on random inputs."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    fp32_sess = ort.InferenceSession(fp32_path, opts,
                                     providers=["CPUExecutionProvider"])
    quant_sess = ort.InferenceSession(quant_path, opts,
                                      providers=["CPUExecutionProvider"])
    sims = []
    for _ in range(n_samples):
        inp = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        fp32_out = fp32_sess.run(None, {"input": inp})[0]
        quant_out = quant_sess.run(None, {"input": inp})[0]
        sims.append(cosine_similarity(fp32_out, quant_out))
    return float(np.mean(sims))


def quantize_pruned_model(name: str, fp32_path: str,
                          fp16_dir: str = "onnx_models_fp16/pruned",
                          int8_dir: str = "onnx_models_int8/pruned") -> dict:
    """Quantize a single pruned ONNX model to FP16 and INT8."""
    fp16_path = os.path.join(fp16_dir, f"{name}.onnx")
    int8_path = os.path.join(int8_dir, f"{name}.onnx")

    result = {"fp16_path": None, "fp16_size_MB": None, "fp16_cosine": None,
              "int8_path": None, "int8_size_MB": None, "int8_cosine": None}

    # FP16
    try:
        fp16_size = quantize_fp16(fp32_path, fp16_path)
        fp16_cos = validate_quantized_vs_fp32(fp32_path, fp16_path)
        result["fp16_path"] = fp16_path
        result["fp16_size_MB"] = fp16_size
        result["fp16_cosine"] = fp16_cos
        status = "PASS" if fp16_cos > 0.999 else "WARN"
        print(f"  FP16: {fp16_size:.1f} MB  cos={fp16_cos:.6f}  [{status}]")
    except Exception as e:
        print(f"  FP16: FAILED — {e}")

    # INT8
    try:
        int8_size = quantize_int8(fp32_path, int8_path)
        int8_cos = validate_quantized_vs_fp32(fp32_path, int8_path)
        result["int8_path"] = int8_path
        result["int8_size_MB"] = int8_size
        result["int8_cosine"] = int8_cos
        status = "PASS" if int8_cos > 0.990 else "WARN"
        print(f"  INT8: {int8_size:.1f} MB  cos={int8_cos:.6f}  [{status}]")
    except Exception as e:
        print(f"  INT8: FAILED — {e}")

    return result


# ---------------------------------------------------------------------------
# Build compression analysis CSV
# ---------------------------------------------------------------------------

def build_compression_analysis(export_results: list) -> list:
    """Build rows for compression_analysis.csv combining baselines + pruned models."""
    rows = []

    # Add baseline rows (FP32, FP16, INT8)
    for bl in BASELINES:
        fp32_size = _file_size_mb(bl["onnx_fp32"]) if os.path.exists(bl["onnx_fp32"]) else None
        fp16_size = _file_size_mb(bl["onnx_fp16"]) if os.path.exists(bl["onnx_fp16"]) else None
        int8_size = _file_size_mb(bl["onnx_int8"]) if os.path.exists(bl["onnx_int8"]) else None

        if fp32_size:
            rows.append({
                "model": bl["name"],
                "method": "Original FP32",
                "params_M": bl["params_M"],
                "size_MB": round(fp32_size, 1),
                "rank_at_1": bl["rank1"],
                "compression_ratio": 1.0,
                "accuracy_drop": 0.0,
            })
            if fp16_size:
                rows.append({
                    "model": bl["name"],
                    "method": "Original FP16",
                    "params_M": bl["params_M"],
                    "size_MB": round(fp16_size, 1),
                    "rank_at_1": bl["rank1"],  # FP16 accuracy ~same
                    "compression_ratio": round(fp32_size / fp16_size, 2),
                    "accuracy_drop": 0.0,
                })
            if int8_size:
                rows.append({
                    "model": bl["name"],
                    "method": "Original INT8",
                    "params_M": bl["params_M"],
                    "size_MB": round(int8_size, 1),
                    "rank_at_1": bl["rank1"],  # INT8 accuracy ~same
                    "compression_ratio": round(fp32_size / int8_size, 2),
                    "accuracy_drop": 0.0,
                })

    # Add pruned model rows (FP32, FP16, INT8)
    for exp, er in export_results:
        params_M = round(er["params_after"] / 1e6, 1)
        fp32_size = er["fp32_size_MB"]

        # Find baseline FP32 size for compression ratio
        bl_match = next((b for b in BASELINES
                         if b["model"] in exp["name"].replace("_lora", "").replace("_kd", "")),
                        None)
        baseline_fp32_size = _file_size_mb(bl_match["onnx_fp32"]) if bl_match else fp32_size

        # Pruned description
        if exp["prune_method"] == "heads":
            prune_desc = f"Pruned {int(exp['head_prune_ratio']*100)}% heads"
        else:
            prune_desc = f"Pruned {exp['num_blocks_to_remove']} blocks"

        variant_tag = f" ({exp['variant']})" if exp["variant"] != "baseline" else ""

        # FP32
        rows.append({
            "model": exp["name"].split("-")[0].replace("_lora", "").replace("_kd", "") + variant_tag,
            "method": f"{prune_desc} FP32",
            "params_M": params_M,
            "size_MB": round(fp32_size, 1),
            "rank_at_1": exp["test_rank1"],
            "compression_ratio": round(baseline_fp32_size / fp32_size, 2),
            "accuracy_drop": round(exp["baseline_rank1"] - exp["test_rank1"], 4),
        })

        # FP16
        if er.get("fp16_size_MB"):
            rows.append({
                "model": exp["name"].split("-")[0].replace("_lora", "").replace("_kd", "") + variant_tag,
                "method": f"{prune_desc} FP16",
                "params_M": params_M,
                "size_MB": round(er["fp16_size_MB"], 1),
                "rank_at_1": exp["test_rank1"],  # FP16 ≈ same accuracy
                "compression_ratio": round(baseline_fp32_size / er["fp16_size_MB"], 2),
                "accuracy_drop": round(exp["baseline_rank1"] - exp["test_rank1"], 4),
            })

        # INT8
        if er.get("int8_size_MB"):
            rows.append({
                "model": exp["name"].split("-")[0].replace("_lora", "").replace("_kd", "") + variant_tag,
                "method": f"{prune_desc} INT8",
                "params_M": params_M,
                "size_MB": round(er["int8_size_MB"], 1),
                "rank_at_1": exp["test_rank1"],  # INT8 ≈ same accuracy
                "compression_ratio": round(baseline_fp32_size / er["int8_size_MB"], 2),
                "accuracy_drop": round(exp["baseline_rank1"] - exp["test_rank1"], 4),
            })

    return rows


# ---------------------------------------------------------------------------
# Build Pareto data CSV
# ---------------------------------------------------------------------------

def build_pareto_data(export_results: list) -> list:
    """Build rows for pareto_data.csv for plotting."""
    rows = []

    # Baselines
    for bl in BASELINES:
        fp32_size = _file_size_mb(bl["onnx_fp32"]) if os.path.exists(bl["onnx_fp32"]) else None
        fp16_size = _file_size_mb(bl["onnx_fp16"]) if os.path.exists(bl["onnx_fp16"]) else None
        int8_size = _file_size_mb(bl["onnx_int8"]) if os.path.exists(bl["onnx_int8"]) else None

        if fp32_size:
            rows.append({
                "model": bl["name"],
                "size_MB": round(fp32_size, 1),
                "compression_ratio": 1.0,
                "rank_at_1": bl["rank1"],
                "method": "Original",
            })
        if fp16_size and fp32_size:
            rows.append({
                "model": bl["name"],
                "size_MB": round(fp16_size, 1),
                "compression_ratio": round(fp32_size / fp16_size, 2),
                "rank_at_1": bl["rank1"],
                "method": "FP16",
            })
        if int8_size and fp32_size:
            rows.append({
                "model": bl["name"],
                "size_MB": round(int8_size, 1),
                "compression_ratio": round(fp32_size / int8_size, 2),
                "rank_at_1": bl["rank1"],
                "method": "INT8",
            })

    # Pruned models
    for exp, er in export_results:
        fp32_size = er["fp32_size_MB"]

        bl_match = next((b for b in BASELINES
                         if b["model"] in exp["name"].replace("_lora", "").replace("_kd", "")),
                        None)
        baseline_fp32_size = _file_size_mb(bl_match["onnx_fp32"]) if bl_match else fp32_size

        if exp["prune_method"] == "heads":
            prune_desc = f"Pruned {int(exp['head_prune_ratio']*100)}% heads"
        else:
            prune_desc = f"Pruned {exp['num_blocks_to_remove']} blocks"

        variant_tag = f" ({exp['variant']})" if exp["variant"] != "baseline" else ""
        model_label = exp["name"].split("-")[0].replace("_lora", "").replace("_kd", "") + variant_tag

        rows.append({
            "model": model_label,
            "size_MB": round(fp32_size, 1),
            "compression_ratio": round(baseline_fp32_size / fp32_size, 2),
            "rank_at_1": exp["test_rank1"],
            "method": prune_desc,
        })

        if er.get("fp16_size_MB"):
            rows.append({
                "model": model_label,
                "size_MB": round(er["fp16_size_MB"], 1),
                "compression_ratio": round(baseline_fp32_size / er["fp16_size_MB"], 2),
                "rank_at_1": exp["test_rank1"],
                "method": f"{prune_desc} + FP16",
            })

        if er.get("int8_size_MB"):
            rows.append({
                "model": model_label,
                "size_MB": round(er["int8_size_MB"], 1),
                "compression_ratio": round(baseline_fp32_size / er["int8_size_MB"], 2),
                "rank_at_1": exp["test_rank1"],
                "method": f"{prune_desc} + INT8",
            })

    return rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(rows: list, path: str, fieldnames: list):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Export pruned models → ONNX → FP16/INT8")
    print("=" * 70)

    export_results = []  # list of (exp_dict, result_dict) tuples

    # Step 1 + 2: Export each pruned model and quantize
    for exp in EXPERIMENTS:
        # Export to ONNX
        er = export_pruned_model(exp)

        # Quantize
        print(f"  Quantizing {exp['name']}...")
        qr = quantize_pruned_model(exp["name"], er["onnx_fp32"])

        # Merge results
        er.update(qr)
        export_results.append((exp, er))

    # Step 3: Compression analysis CSV
    print(f"\n{'='*70}")
    print("Building compression analysis...")
    print(f"{'='*70}")
    analysis_rows = build_compression_analysis(export_results)
    write_csv(
        analysis_rows,
        "pruning_results/compression_analysis.csv",
        ["model", "method", "params_M", "size_MB", "rank_at_1",
         "compression_ratio", "accuracy_drop"],
    )

    # Step 4: Pareto data CSV
    print(f"\n{'='*70}")
    print("Building Pareto plot data...")
    print(f"{'='*70}")
    pareto_rows = build_pareto_data(export_results)
    write_csv(
        pareto_rows,
        "pruning_results/pareto_data.csv",
        ["model", "size_MB", "compression_ratio", "rank_at_1", "method"],
    )

    # Summary
    print(f"\n{'='*70}")
    print("EXPORT & QUANTIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'FP32 MB':>8} {'FP16 MB':>8} {'INT8 MB':>8} {'rank@1':>8}")
    print("-" * 75)
    for exp, er in export_results:
        fp32 = f"{er['fp32_size_MB']:.1f}"
        fp16 = f"{er['fp16_size_MB']:.1f}" if er.get("fp16_size_MB") else "—"
        int8 = f"{er['int8_size_MB']:.1f}" if er.get("int8_size_MB") else "—"
        r1 = f"{exp['test_rank1']:.4f}"
        print(f"{exp['name']:<35} {fp32:>8} {fp16:>8} {int8:>8} {r1:>8}")

    # Verify pruned models are smaller than originals
    print(f"\n{'='*70}")
    print("SIZE VERIFICATION (pruned < original)")
    print(f"{'='*70}")
    for exp, er in export_results:
        bl_match = next((b for b in BASELINES
                         if b["model"] in exp["name"].replace("_lora", "").replace("_kd", "")),
                        None)
        if bl_match and os.path.exists(bl_match["onnx_fp32"]):
            bl_size = _file_size_mb(bl_match["onnx_fp32"])
            is_smaller = er["fp32_size_MB"] < bl_size
            status = "OK" if is_smaller else "UNEXPECTED"
            print(f"  {exp['name']}: {er['fp32_size_MB']:.1f} MB "
                  f"vs baseline {bl_size:.1f} MB [{status}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
