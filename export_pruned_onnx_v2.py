#!/usr/bin/env python
"""Export NEW pruned models  to ONNX, quantize (FP16/INT8), build CSVs.

This script handles the 15 new pruning experiments from shell_scripts/prune_comprehensive.sh.
Existing experiments from export_pruned_onnx.py are NOT re-run; the final
comprehensive CSV will be assembled by the eval shell script.
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
# New experiment definitions 
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # ─── Swin-T Baseline (p09btsx3, rank@1=0.534) ───
    {
        "name": "swin_tiny_baseline-heads_0.10",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/p09btsx3/epoch=22-rank1tinyface/eval/rank@1=0.541.ckpt",
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=6-rank1tinyface/eval/rank@1=0.516.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.10,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5335,
        "test_rank1": 0.4995,
        "variant": "Baseline",
        "architecture": "Swin-T",
    },
    {
        "name": "swin_tiny_baseline-heads_0.25",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/p09btsx3/epoch=22-rank1tinyface/eval/rank@1=0.541.ckpt",
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=9-rank1tinyface/eval/rank@1=0.479.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.25,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5335,
        "test_rank1": 0.4646,
        "variant": "Baseline",
        "architecture": "Swin-T",
    },
    {
        "name": "swin_tiny_baseline-blocks_2",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/p09btsx3/epoch=22-rank1tinyface/eval/rank@1=0.541.ckpt",
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=7-rank1tinyface/eval/rank@1=0.547.ckpt",
        "prune_method": "blocks",
        "head_prune_ratio": 0.0,
        "num_blocks_to_remove": 2,
        "baseline_rank1": 0.5335,
        "test_rank1": 0.5440,
        "variant": "Baseline",
        "architecture": "Swin-T",
    },
    # ─── Swin-T KD Best (zz76v2nu, α=0.0, rank@1=0.616) ───
    {
        "name": "swin_tiny_kd_best-heads_0.10",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_kd_alpha_ablation/zz76v2nu/epoch=38-rank1tinyface/eval/rank@1=0.628.ckpt",
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=0-rank1tinyface/eval/rank@1=0.349.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.10,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.6159,
        "test_rank1": 0.4032,
        "variant": "KD (α=0.0)",
        "architecture": "Swin-T",
    },
    {
        "name": "swin_tiny_kd_best-heads_0.25",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_kd_alpha_ablation/zz76v2nu/epoch=38-rank1tinyface/eval/rank@1=0.628.ckpt",
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=1-rank1tinyface/eval/rank@1=0.351.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.25,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.6159,
        "test_rank1": 0.3801,
        "variant": "KD (α=0.0)",
        "architecture": "Swin-T",
    },
    {
        "name": "swin_tiny_kd_best-blocks_2",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_kd_alpha_ablation/zz76v2nu/epoch=38-rank1tinyface/eval/rank@1=0.628.ckpt",
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=0-rank1tinyface/eval/rank@1=0.564.ckpt",
        "prune_method": "blocks",
        "head_prune_ratio": 0.0,
        "num_blocks_to_remove": 2,
        "baseline_rank1": 0.6159,
        "test_rank1": 0.5829,
        "variant": "KD (α=0.0)",
        "architecture": "Swin-T",
    },
    # ─── DeiT3-S Baseline (rdsdt7xb, rank@1=0.552) ───
    {
        "name": "deit3_small_baseline-heads_0.10",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/rdsdt7xb/epoch=47-rank1tinyface/eval/rank@1=0.562.ckpt",
        "model_name": "deit3_small_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=4-rank1tinyface/eval/rank@1=0.591.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.10,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5515,
        "test_rank1": 0.5644,
        "variant": "Baseline",
        "architecture": "DeiT3-S",
    },
    {
        "name": "deit3_small_baseline-heads_0.25",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/rdsdt7xb/epoch=47-rank1tinyface/eval/rank@1=0.562.ckpt",
        "model_name": "deit3_small_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=5-rank1tinyface/eval/rank@1=0.572.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.25,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5515,
        "test_rank1": 0.5515,
        "variant": "Baseline",
        "architecture": "DeiT3-S",
    },
    {
        "name": "deit3_small_baseline-blocks_2",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/rdsdt7xb/epoch=47-rank1tinyface/eval/rank@1=0.562.ckpt",
        "model_name": "deit3_small_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=8-rank1tinyface/eval/rank@1=0.473.ckpt",
        "prune_method": "blocks",
        "head_prune_ratio": 0.0,
        "num_blocks_to_remove": 2,
        "baseline_rank1": 0.5515,
        "test_rank1": 0.4981,
        "variant": "Baseline",
        "architecture": "DeiT3-S",
    },
    # ─── Swin-B Baseline (xd7hukv3, rank@1=0.526) ───
    {
        "name": "swin_base_baseline-heads_0.10",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/xd7hukv3/epoch=37-rank1tinyface/eval/rank@1=0.564.ckpt",
        "model_name": "swin_base_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=7-rank1tinyface/eval/rank@1=0.514.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.10,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5260,
        "test_rank1": 0.4874,
        "variant": "Baseline",
        "architecture": "Swin-B",
    },
    {
        "name": "swin_base_baseline-heads_0.25",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/xd7hukv3/epoch=37-rank1tinyface/eval/rank@1=0.564.ckpt",
        "model_name": "swin_base_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=8-rank1tinyface/eval/rank@1=0.469.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.25,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5260,
        "test_rank1": 0.4541,
        "variant": "Baseline",
        "architecture": "Swin-B",
    },
    {
        "name": "swin_base_baseline-blocks_5",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/xd7hukv3/epoch=37-rank1tinyface/eval/rank@1=0.564.ckpt",
        "model_name": "swin_base_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=0-rank1tinyface/eval/rank@1=0.579.ckpt",
        "prune_method": "blocks",
        "head_prune_ratio": 0.0,
        "num_blocks_to_remove": 5,
        "baseline_rank1": 0.5260,
        "test_rank1": 0.5528,
        "variant": "Baseline",
        "architecture": "Swin-B",
    },
    # ─── DeiT3-B KD Best (j8e97e1o, PETALface α=0.1, rank@1=0.593) ───
    {
        "name": "deit3_base_kd_best-heads_0.10",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_kd_alpha_ablation/j8e97e1o/epoch=47-rank1tinyface/eval/rank@1=0.583.ckpt",
        "model_name": "deit3_base_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=5-rank1tinyface/eval/rank@1=0.591.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.10,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5933,
        "test_rank1": 0.5928,
        "variant": "KD (α=0.1)",
        "architecture": "DeiT3-B",
    },
    {
        "name": "deit3_base_kd_best-heads_0.25",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_kd_alpha_ablation/j8e97e1o/epoch=47-rank1tinyface/eval/rank@1=0.583.ckpt",
        "model_name": "deit3_base_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=3-rank1tinyface/eval/rank@1=0.570.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.25,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5933,
        "test_rank1": 0.5652,
        "variant": "KD (α=0.1)",
        "architecture": "DeiT3-B",
    },
    {
        "name": "deit3_base_kd_best-blocks_3",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_kd_alpha_ablation/j8e97e1o/epoch=47-rank1tinyface/eval/rank@1=0.583.ckpt",
        "model_name": "deit3_base_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models_v2/epoch=8-rank1tinyface/eval/rank@1=0.516.ckpt",
        "prune_method": "blocks",
        "head_prune_ratio": 0.0,
        "num_blocks_to_remove": 3,
        "baseline_rank1": 0.5933,
        "test_rank1": 0.5491,
        "variant": "KD (α=0.1)",
        "architecture": "DeiT3-B",
    },
]

IMG_SIZE = 96


def _file_size_mb(path: str) -> float:
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
    pytorch_model.eval()
    with torch.no_grad():
        pt_output = pytorch_model(dummy_input.cpu()).cpu().numpy()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {"input": dummy_input.cpu().numpy()})[0]
    return cosine_similarity(pt_output, ort_output)


def export_pruned_model(exp: dict, output_dir: str = "onnx_models/pruned_v2") -> dict:
    name = exp["name"]
    onnx_path = os.path.join(output_dir, f"{name}.onnx")
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting: {name}")
    print(f"{'='*60}")

    module, family = load_module_for_finetuning(
        exp["original_ckpt"], exp["model_name"]
    )
    params_before = _param_count(module)

    prune_model(
        module.backbone, family,
        prune_method=exp["prune_method"],
        head_prune_ratio=exp["head_prune_ratio"],
        num_blocks_to_remove=exp["num_blocks_to_remove"],
        img_size=IMG_SIZE,
    )
    params_after = _param_count(module)

    ckpt = torch.load(exp["pruned_ckpt"], map_location="cpu", weights_only=False)
    module.load_state_dict(ckpt["state_dict"])

    backbone = module.backbone
    backbone.eval()
    backbone.cpu()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        backbone, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=18, do_constant_folding=True,
    )

    fp32_size = _file_size_mb(onnx_path)
    cos_sim = validate_onnx(backbone, onnx_path, dummy)
    status = "PASS" if cos_sim > 0.999 else "FAIL"
    print(f"  ONNX: {onnx_path} ({fp32_size:.1f} MB) cos={cos_sim:.6f} [{status}]")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = session.run(None, {"input": dummy.numpy()})[0]

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


def quantize_fp16(input_path: str, output_path: str) -> float:
    model = onnx.load(input_path, load_external_data=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, output_path)
    return _file_size_mb(output_path)


def quantize_int8(input_path: str, output_path: str) -> float:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=input_path, model_output=output_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        use_external_data_format=False,
    )
    return _file_size_mb(output_path)


def validate_quantized_vs_fp32(fp32_path: str, quant_path: str, n_samples: int = 5) -> float:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    fp32_sess = ort.InferenceSession(fp32_path, opts, providers=["CPUExecutionProvider"])
    quant_sess = ort.InferenceSession(quant_path, opts, providers=["CPUExecutionProvider"])
    sims = []
    for _ in range(n_samples):
        inp = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        fp32_out = fp32_sess.run(None, {"input": inp})[0]
        quant_out = quant_sess.run(None, {"input": inp})[0]
        sims.append(cosine_similarity(fp32_out, quant_out))
    return float(np.mean(sims))


def quantize_pruned_model(name: str, fp32_path: str) -> dict:
    fp16_path = os.path.join("onnx_models_fp16/pruned_v2", f"{name}.onnx")
    int8_path = os.path.join("onnx_models_int8/pruned_v2", f"{name}.onnx")

    result = {"fp16_path": None, "fp16_size_MB": None, "fp16_cosine": None,
              "int8_path": None, "int8_size_MB": None, "int8_cosine": None}

    try:
        fp16_size = quantize_fp16(fp32_path, fp16_path)
        fp16_cos = validate_quantized_vs_fp32(fp32_path, fp16_path)
        result.update(fp16_path=fp16_path, fp16_size_MB=fp16_size, fp16_cosine=fp16_cos)
        status = "PASS" if fp16_cos > 0.999 else "WARN"
        print(f"  FP16: {fp16_size:.1f} MB  cos={fp16_cos:.6f}  [{status}]")
    except Exception as e:
        print(f"  FP16: FAILED — {e}")

    try:
        int8_size = quantize_int8(fp32_path, int8_path)
        int8_cos = validate_quantized_vs_fp32(fp32_path, int8_path)
        result.update(int8_path=int8_path, int8_size_MB=int8_size, int8_cosine=int8_cos)
        status = "PASS" if int8_cos > 0.990 else "WARN"
        print(f"  INT8: {int8_size:.1f} MB  cos={int8_cos:.6f}  [{status}]")
    except Exception as e:
        print(f"  INT8: FAILED — {e}")

    return result


def main():
    print("=" * 70)
    print("Export new pruned models → ONNX → FP16/INT8")
    print("=" * 70)

    export_results = []

    for exp in EXPERIMENTS:
        er = export_pruned_model(exp)
        print(f"  Quantizing {exp['name']}...")
        qr = quantize_pruned_model(exp["name"], er["onnx_fp32"])
        er.update(qr)
        export_results.append((exp, er))

    # Summary
    print(f"\n{'='*70}")
    print("EXPORT & QUANTIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<45} {'FP32 MB':>8} {'FP16 MB':>8} {'INT8 MB':>8} {'rank@1':>8}")
    print("-" * 85)
    for exp, er in export_results:
        fp32 = f"{er['fp32_size_MB']:.1f}"
        fp16 = f"{er['fp16_size_MB']:.1f}" if er.get("fp16_size_MB") else "—"
        int8 = f"{er['int8_size_MB']:.1f}" if er.get("int8_size_MB") else "—"
        r1 = f"{exp['test_rank1']:.4f}"
        print(f"{exp['name']:<45} {fp32:>8} {fp16:>8} {int8:>8} {r1:>8}")

    print("\nDone!")


if __name__ == "__main__":
    main()
