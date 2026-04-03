# Structured Pruning Results

## Overview

Structured pruning was applied to four base models using two strategies:

- **Attention head pruning** — remove entire MHSA heads by magnitude importance, then resize QKV/proj tensors
- **Block dropping** — remove whole encoder blocks (last-first ordering), reducing model depth

All pruned models were fine-tuned for 10 epochs (AdamW, lr=3e-5, cosine annealing) and evaluated on the TinyFace TEST set (rank@1 metric).

## Source Models

| Model | Variant | Params (M) | Baseline rank@1 |
|-------|---------|------------|-----------------|
| deit3_base_patch16_224 | LoRA | 85.8 | 0.5834 |
| deit3_base_patch16_224 | KD CVLFace | 85.8 | 0.5807 |
| swin_base_patch4_window7_224 | baseline | 86.7 | 0.5359 |
| swin_tiny_patch4_window7_224 | KD CVLFace | 27.5 | 0.5491 |

## Pruning Results (TEST Set)

| Experiment | Method | Params Before | Params After | Reduction | rank@1 | Drop |
|------------|--------|--------------|-------------|-----------|--------|------|
| deit3_base (LoRA) | Heads 10% | 87.5M | 84.7M | 3.1% | **0.5896** | **−0.62 pp** |
| deit3_base (LoRA) | Heads 25% | 87.5M | 80.4M | 8.1% | 0.5655 | +1.79 pp |
| deit3_base (LoRA) | Heads 50% | 87.5M | 73.3M | 16.2% | 0.5131 | +7.03 pp |
| deit3_base (LoRA) | Blocks 3/12 | 87.5M | 66.2M | 24.3% | 0.5464 | +3.70 pp |
| deit3_base (KD CVLFace) | Heads 25% | 87.5M | 80.4M | 8.1% | 0.5676 | +1.31 pp |
| swin_base | Heads 25% | 89.1M | 83.4M | 6.4% | 0.4665 | +6.94 pp |
| swin_base | Blocks 5/18 | 89.1M | 73.3M | 17.7% | **0.5748** | **−3.89 pp** |
| swin_tiny (KD CVLFace) | Heads 10% | 29.3M | 28.9M | 1.4% | 0.5051 | +4.40 pp |

Negative drop = accuracy improved after pruning + fine-tuning.

## Accuracy vs. Compression (ONNX, All Precisions)

### DeiT3-Base (LoRA)

| Method | Params (M) | Size (MB) | Compression | rank@1 | Drop |
|--------|-----------|-----------|-------------|--------|------|
| Original FP32 | 85.8 | 327.7 | 1.00× | 0.5834 | — |
| Original FP16 | 85.8 | 164.2 | 2.00× | 0.5834 | — |
| Original INT8 | 85.8 | 84.6 | 3.87× | 0.5834 | — |
| Pruned 10% heads FP32 | 84.7 | 317.2 | 1.03× | 0.5896 | −0.62 pp |
| Pruned 10% heads FP16 | 84.7 | 158.9 | 2.06× | 0.5896 | −0.62 pp |
| Pruned 10% heads INT8 | 84.7 | 82.0 | 4.00× | 0.5896 | −0.62 pp |
| Pruned 25% heads FP32 | 80.4 | 300.6 | 1.09× | 0.5655 | +1.79 pp |
| Pruned 25% heads FP16 | 80.4 | 150.7 | 2.17× | 0.5655 | +1.79 pp |
| Pruned 25% heads INT8 | 80.4 | 77.8 | 4.21× | 0.5655 | +1.79 pp |
| Pruned 50% heads FP32 | 73.3 | 273.6 | 1.20× | 0.5131 | +7.03 pp |
| Pruned 50% heads FP16 | 73.3 | 137.1 | 2.39× | 0.5131 | +7.03 pp |
| Pruned 50% heads INT8 | 73.3 | 71.0 | 4.62× | 0.5131 | +7.03 pp |
| Pruned 3 blocks FP32 | 66.2 | 246.4 | 1.33× | 0.5464 | +3.70 pp |
| Pruned 3 blocks FP16 | 66.2 | 123.5 | 2.65× | 0.5464 | +3.70 pp |
| Pruned 3 blocks INT8 | 66.2 | 64.0 | **5.12×** | 0.5464 | +3.70 pp |

### DeiT3-Base (KD CVLFace)

| Method | Params (M) | Size (MB) | Compression | rank@1 | Drop |
|--------|-----------|-----------|-------------|--------|------|
| Original FP32 | 85.8 | 327.7 | 1.00× | 0.5807 | — |
| Original FP16 | 85.8 | 164.2 | 2.00× | 0.5807 | — |
| Original INT8 | 85.8 | 84.6 | 3.87× | 0.5807 | — |
| Pruned 25% heads FP32 | 80.4 | 300.6 | 1.09× | 0.5676 | +1.31 pp |
| Pruned 25% heads FP16 | 80.4 | 150.7 | 2.17× | 0.5676 | +1.31 pp |
| Pruned 25% heads INT8 | 80.4 | 77.8 | 4.21× | 0.5676 | +1.31 pp |

### Swin-Base

| Method | Params (M) | Size (MB) | Compression | rank@1 | Drop |
|--------|-----------|-----------|-------------|--------|------|
| Original FP32 | 86.7 | 334.9 | 1.00× | 0.5359 | — |
| Original FP16 | 86.7 | 168.6 | 1.99× | 0.5359 | — |
| Original INT8 | 86.7 | 87.2 | 3.84× | 0.5359 | — |
| Pruned 25% heads FP32 | 83.4 | 312.6 | 1.07× | 0.4665 | +6.94 pp |
| Pruned 25% heads FP16 | 83.4 | 157.5 | 2.13× | 0.4665 | +6.94 pp |
| Pruned 25% heads INT8 | 83.4 | 81.2 | 4.12× | 0.4665 | +6.94 pp |
| Pruned 5 blocks FP32 | 73.3 | 273.9 | 1.22× | 0.5748 | −3.89 pp |
| Pruned 5 blocks FP16 | 73.3 | 137.9 | 2.43× | 0.5748 | −3.89 pp |
| Pruned 5 blocks INT8 | 73.3 | 71.2 | **4.70×** | 0.5748 | −3.89 pp |

### Swin-Tiny (KD CVLFace)

| Method | Params (M) | Size (MB) | Compression | rank@1 | Drop |
|--------|-----------|-----------|-------------|--------|------|
| Original FP32 | 27.5 | 106.9 | 1.00× | 0.5491 | — |
| Original FP16 | 27.5 | 54.1 | 1.98× | 0.5491 | — |
| Original INT8 | 27.5 | 28.3 | 3.78× | 0.5491 | — |
| Pruned 10% heads FP32 | 28.9 | 105.3 | 1.02× | 0.5051 | +4.40 pp |
| Pruned 10% heads FP16 | 28.9 | 53.2 | 2.01× | 0.5051 | +4.40 pp |
| Pruned 10% heads INT8 | 28.9 | 27.8 | 3.84× | 0.5051 | +4.40 pp |

## Key Findings

1. **Mild head pruning can improve accuracy.** DeiT3-Base with 10% head pruning achieved 0.5896 rank@1, surpassing the unpruned baseline (0.5834) by 0.62 pp. This suggests a regularization effect from removing redundant attention heads.

2. **Block dropping is highly effective for Swin.** Removing 5 of 18 blocks from Swin-Base *improved* accuracy by 3.89 pp (0.5748 vs 0.5359) while achieving 17.7% parameter reduction. Combined with INT8 quantization, this yields 4.70× compression with better accuracy than the original.

3. **DeiT3 tolerates head pruning better than Swin.** At 25% head removal, DeiT3 drops only 1.79 pp while Swin drops 6.94 pp. DeiT3's uniform block structure may make its heads more independently valuable.

4. **Aggressive pruning (50% heads) degrades significantly.** DeiT3 with 50% heads removed loses 7.03 pp, indicating a practical ceiling around 25% for head pruning.

5. **KD + pruning stacks.** DeiT3-Base (KD CVLFace) with 25% head pruning retains 0.5676 rank@1 — a drop of only 1.31 pp from its KD baseline. The KD-trained model is slightly more robust to pruning than the LoRA variant at the same ratio (1.31 pp vs 1.79 pp drop).

6. **Maximum compression without accuracy loss.** Two configurations actually improve accuracy:
   - DeiT3-Base 10% heads + INT8: **4.00× compression, +0.62 pp** (82 MB)
   - Swin-Base 5 blocks + INT8: **4.70× compression, +3.89 pp** (71.2 MB)

7. **Maximum compression overall.** DeiT3-Base 3 blocks + INT8 achieves **5.12× compression** (64 MB, down from 327.7 MB) with a 3.70 pp accuracy drop.

## Pipeline

- **Pruning**: `pruning.py` — head pruning (magnitude importance) and block dropping (last-first)
- **Fine-tuning**: 10 epochs, AdamW lr=3e-5, weight_decay=0.05, cosine annealing, all params unfrozen
- **ONNX export**: `export_pruned_onnx.py` — reconstructs pruned architecture, loads fine-tuned weights, validates output
- **Quantization**: FP16 (onnxconverter-common), INT8 dynamic (onnxruntime, weights-only)
- **Evaluation**: TinyFace TEST set, rank@1 at k=10

## Output Files

| File | Description |
|------|-------------|
| `pruning_results/pruning_summary.csv` | Per-experiment pruning results (params, rank@1, drop) |
| `pruning_results/compression_analysis.csv` | Full accuracy-vs-compression table (36 rows) |
| `pruning_results/pareto_data.csv` | Pareto-optimal plot data (36 rows) |
| `onnx_models/pruned/*.onnx` | Pruned ONNX models (FP32) |
| `onnx_models_fp16/pruned/*.onnx` | Pruned + FP16 quantized |
| `onnx_models_int8/pruned/*.onnx` | Pruned + INT8 quantized |
