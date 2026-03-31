# Phase 2.3 — CVLFace KD Evaluation Results

TinyFace **test-set** rank@1 accuracy.  All students trained at 96×96 with
ImageNet-pretrained timm backbones.  Teacher is CVLFace ViT-Base (112×112).

## Table 1 — Teacher Baselines

| Model | rank@1 |
|-------|--------|
| CVLFace ViT-Base (frozen / off-the-shelf) | 0.716 |
| CVLFace ViT-Base (fine-tuned on TinyFace) | 0.781 |

## Table 2 — KD with Frozen Teacher vs Phase 1

| Student | Phase 1 Baseline | Phase 1 LoRA | Phase 1 Best | KD Direct (frozen) | KD LoRA (frozen) | KD Best (frozen) | Δ vs Phase 1 Best |
|---------|-----------------|-------------|-------------|-------------------|-----------------|-----------------|------------------|
| swin_tiny | 0.534 | 0.562 | 0.562 | 0.549 | 0.424 | 0.549 | -0.013 |
| swin_small | 0.539 | 0.554 | 0.554 | 0.547 | 0.393 | 0.547 | -0.008 |
| swin_base | 0.526 | 0.536 | 0.536 | 0.529 | 0.361 | 0.529 | -0.007 |
| deit3_small | 0.552 | 0.527 | 0.552 | 0.543 | 0.520 | 0.543 | -0.009 |
| deit3_medium | 0.570 | 0.533 | 0.570 | 0.558 | 0.534 | 0.558 | -0.012 |
| deit3_base | 0.583 | 0.550 | 0.583 | 0.572 | 0.547 | 0.572 | -0.012 |

## Table 3 — KD with Fine-tuned Teacher vs Phase 1

| Student | Phase 1 Baseline | Phase 1 LoRA | Phase 1 Best | KD Direct (ft) | KD LoRA (ft) | KD Best (ft) | Δ vs Phase 1 Best |
|---------|-----------------|-------------|-------------|---------------|-------------|-------------|------------------|
| swin_tiny | 0.534 | 0.562 | 0.562 | 0.536 | 0.415 | 0.536 | -0.026 |
| swin_small | 0.539 | 0.554 | 0.554 | 0.542 | 0.559 | 0.559 | +0.004 |
| swin_base | 0.526 | 0.536 | 0.536 | 0.536 | 0.352 | 0.536 | +0.000 |
| deit3_small | 0.552 | 0.527 | 0.552 | 0.536 | 0.538 | 0.538 | -0.013 |
| deit3_medium | 0.570 | 0.533 | 0.570 | 0.559 | 0.536 | 0.559 | -0.011 |
| deit3_base | 0.583 | 0.550 | 0.583 | 0.581 | 0.547 | 0.581 | -0.003 |

## Table 4 — Overall Comparison: Best KD vs Phase 1 Best

| Student | Phase 1 Best | KD Best (frozen) | KD Best (ft) | Overall KD Best | Teacher | Δ vs Phase 1 |
|---------|-------------|-----------------|-------------|----------------|---------|-------------|
| swin_tiny | 0.562 | 0.549 | 0.536 | **0.549** | frozen | -0.013 |
| swin_small | 0.554 | 0.547 | 0.559 | **0.559** | ft | +0.004 |
| swin_base | 0.536 | 0.529 | 0.536 | **0.536** | ft | +0.000 |
| deit3_small | 0.552 | 0.543 | 0.538 | **0.543** | frozen | -0.009 |
| deit3_medium | 0.570 | 0.558 | 0.559 | **0.559** | ft | -0.011 |
| deit3_base | 0.583 | 0.572 | 0.581 | **0.581** | ft | -0.003 |

## Summary

- **Teacher frozen rank@1**: 0.716
- **Teacher fine-tuned rank@1**: 0.781

- **Students matching/exceeding Phase 1 best**: 1/6

### Notes

- Swin LoRA + KD runs show severe degradation (rank@1 ~0.35–0.42), consistent
  with the Swin LoRA collapse observed in Phase 2.2 training.  These are included
  for completeness but should not be interpreted as representative KD+LoRA results.
- DeiT3 LoRA + KD runs performed normally (no collapse).
- Direct (full fine-tune) KD generally outperformed LoRA KD.
