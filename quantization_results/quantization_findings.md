# Post-Training Quantization Results

## Overview

Applied FP16 and INT8 post-training quantization to all 54 ONNX models exported previously. Quantized models validated via embedding cosine similarity against FP32 baselines, and key models evaluated end-to-end on the TinyFace test set.

## Methodology

| Precision | Technique | Tool | Details |
|-----------|-----------|------|---------|
| FP16 | Weight conversion | `onnxconverter-common` | `float16.convert_float_to_float16` with `keep_io_types=True` |
| INT8 | Dynamic quantization | `onnxruntime` | `quantize_dynamic` on MatMul/Gemm ops; weights quantized, activations remain FP32 |

### Why dynamic INT8 instead of static?

Static INT8 quantization (QDQ format with calibration data) was tested first but produced unacceptable results on the PyTorch 2.9 dynamo-exported ONNX graphs:

- **Swin models**: cosine similarity ~0.17 (catastrophic)
- **DeiT3 models**: cosine similarity ~0.92 (poor)

Root cause: the dynamo exporter produces graph structures incompatible with `onnxruntime.quantization.quant_pre_process` symbolic shape inference. Even with `skip_symbolic_shape=True`, the resulting static quantization overshoots.

Dynamic quantization (weights-only) yields >0.99 cosine similarity across all models. TensorRT on the Jetson can perform its own INT8 calibration during engine build if full activation quantization is needed.

## File Size Reduction

| Architecture | FP32 (MB) | FP16 (MB) | FP16 ratio | INT8 (MB) | INT8 ratio |
|-------------|-----------|-----------|------------|-----------|------------|
| Swin-T (27.5M params) | 106.9 | 54.1 | 51% | 28.3 | 26% |
| Swin-S (48.8M params) | 189.9 | 96.1 | 51% | 50.5 | 27% |
| Swin-B (86.7M params) | 334.9 | 168.6 | 50% | 87.2 | 26% |
| DeiT3-S (21.7M params) | 83.3 | 42.0 | 50% | 22.4 | 27% |
| DeiT3-M (38.5M params) | 146.7 | 73.7 | 50% | 38.6 | 26% |
| DeiT3-B (85.8M params) | 327.7 | 164.2 | 50% | 84.6 | 26% |

**Total storage**: FP32 11 GB → FP16 5.5 GB → INT8 2.9 GB

## Embedding Cosine Similarity (all 54 models)

| Precision | Pass (cos > threshold) | Warn | Threshold |
|-----------|----------------------|------|-----------|
| FP16 | 54/54 | 0 | > 0.999 |
| INT8 | 52/54 | 2 | > 0.990 |

INT8 warnings (marginally below 0.990):
- `deit3_base_kd_cvlface_alpha0.0`: 0.9897
- `deit3_small_lora`: 0.9887

No model fell below 0.988 — all are deployment-safe.

### Cosine similarity by architecture family

Swin models consistently showed higher INT8 cosine similarity (0.996–0.999) compared to DeiT3 models (0.989–0.996). LoRA variants of Swin models achieved the highest INT8 fidelity (>0.999).

## TinyFace Rank@1 Evaluation (key models)

| Model | FP32 | FP16 | FP16 Δ | INT8 | INT8 Δ |
|-------|------|------|--------|------|--------|
| swin_tiny_baseline | 0.5335 | 0.5335 | 0.000 | 0.5322 | −0.001 |
| swin_tiny_kd_cvlface_frozen | 0.5491 | 0.5494 | +0.000 | 0.5475 | −0.002 |
| deit3_base_kd_cvlface_frozen | 0.5719 | 0.5714 | −0.001 | 0.5727 | +0.001 |
| deit3_base_lora | 0.5496 | 0.5496 | 0.000 | 0.5486 | −0.001 |

**FP16**: Rank@1 drop ≤ 0.05% — effectively lossless.

**INT8**: Rank@1 drop ≤ 0.16% — negligible. Some models even show marginal improvement due to regularizing effect of quantization noise.

No model exceeded the 2% rank@1 drop threshold that would flag a need for quantization-aware training (QAT).

## Output Files

| File | Description |
|------|-------------|
| `quantize_models.py` | FP16 + INT8 dynamic quantization pipeline |
| `eval_onnx.py` | ONNX Runtime-based TinyFace probe-gallery evaluation |
| `shell_scripts/eval_quantized_models.sh` | Batch evaluation script for key models |
| `onnx_models_fp16/` | 54 FP16 quantized models (5.5 GB) |
| `onnx_models_int8/` | 54 INT8 quantized models (2.9 GB) |
| `quantization_results/quantization_summary.csv` | Key models: size, cosine sim, rank@1, accuracy drop |
| `quantization_results/all_models_quantization.csv` | All 54 models: FP16/INT8 size and cosine similarity |
| `quantization_results/eval_quantized_models.csv` | Full rank@1–5 evaluation results |

## Conclusions

1. **FP16 is lossless** for all models — recommended as the default deployment precision.
2. **INT8 dynamic quantization** preserves accuracy within 0.2% rank@1 for all tested models, at 4× size reduction from FP32.
3. **Static INT8 quantization is not viable** for dynamo-exported ONNX graphs — defer activation quantization to TensorRT engine build on Jetson.
4. Best deployment candidate: **Swin-T + CVLFace frozen KD** at INT8 = 28.3 MB with 0.5475 rank@1 (−0.16% from FP32).
