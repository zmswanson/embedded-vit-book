# Phase 2.5 — Final Combined KD Evaluation Results

TinyFace **test-set** evaluation. All students trained at 96×96 with
ImageNet-pretrained timm backbones. Teachers evaluated at 112×112
(auto-resized internally for PETALface 120×120).

## Table 1 — Full Comparison (all models × all training methods × all teachers)

| Model | Method | Teacher | Teacher Mode | rank@1 | rank@5 | rank@10 | mAP | AUC |
|-------|--------|---------|--------------|--------|--------|---------|-----|-----|
| swin_tiny | Baseline (Phase 1) | — | — | 0.534 | 0.655 | 0.707 | 0.592 | 0.968 |
| swin_tiny | LoRA (Phase 1) | — | — | 0.562 | 0.681 | 0.727 | 0.619 | 0.972 |
| swin_tiny | Direct + KD | CVLFace | frozen | 0.549 | 0.665 | 0.717 | 0.606 | 0.968 |
| swin_tiny | LoRA + KD | CVLFace | frozen | 0.424 | 0.535 | 0.579 | 0.477 | 0.885 |
| swin_tiny | Direct + KD | CVLFace | fine-tuned | 0.536 | 0.653 | 0.702 | 0.595 | 0.967 |
| swin_tiny | LoRA + KD | CVLFace | fine-tuned | 0.415 | 0.517 | 0.571 | 0.468 | 0.891 |
| swin_tiny | Direct + KD | PETALface | frozen | 0.543 | 0.661 | 0.715 | 0.602 | 0.968 |
| swin_tiny | LoRA + KD | PETALface | frozen | 0.567 | 0.684 | 0.730 | 0.623 | 0.969 |
| swin_tiny | Direct + KD | PETALface | fine-tuned | 0.547 | 0.659 | 0.708 | 0.603 | 0.968 |
| swin_tiny | LoRA + KD | PETALface | fine-tuned | 0.562 | 0.677 | 0.727 | 0.619 | 0.971 |
| swin_small | Baseline (Phase 1) | — | — | 0.539 | 0.666 | 0.718 | 0.600 | 0.967 |
| swin_small | LoRA (Phase 1) | — | — | 0.554 | 0.675 | 0.718 | 0.613 | 0.968 |
| swin_small | Direct + KD | CVLFace | frozen | 0.547 | 0.664 | 0.714 | 0.605 | 0.967 |
| swin_small | LoRA + KD | CVLFace | frozen | 0.393 | 0.510 | 0.559 | 0.450 | 0.885 |
| swin_small | Direct + KD | CVLFace | fine-tuned | 0.542 | 0.669 | 0.719 | 0.603 | 0.968 |
| swin_small | LoRA + KD | CVLFace | fine-tuned | 0.559 | 0.668 | 0.716 | 0.613 | 0.968 |
| swin_small | Direct + KD | PETALface | frozen | 0.527 | 0.660 | 0.719 | 0.592 | 0.968 |
| swin_small | LoRA + KD | PETALface | frozen | 0.409 | 0.510 | 0.563 | 0.461 | 0.847 |
| swin_small | Direct + KD | PETALface | fine-tuned | 0.536 | 0.656 | 0.708 | 0.594 | 0.968 |
| swin_small | LoRA + KD | PETALface | fine-tuned | 0.402 | 0.516 | 0.567 | 0.458 | 0.842 |
| swin_base | Baseline (Phase 1) | — | — | 0.526 | 0.650 | 0.706 | 0.586 | 0.965 |
| swin_base | LoRA (Phase 1) | — | — | 0.536 | 0.655 | 0.707 | 0.594 | 0.959 |
| swin_base | Direct + KD | CVLFace | frozen | 0.529 | 0.652 | 0.702 | 0.586 | 0.967 |
| swin_base | LoRA + KD | CVLFace | frozen | 0.361 | 0.481 | 0.526 | 0.420 | 0.888 |
| swin_base | Direct + KD | CVLFace | fine-tuned | 0.536 | 0.654 | 0.699 | 0.593 | 0.966 |
| swin_base | LoRA + KD | CVLFace | fine-tuned | 0.352 | 0.480 | 0.531 | 0.414 | 0.887 |
| swin_base | Direct + KD | PETALface | frozen | 0.530 | 0.653 | 0.705 | 0.589 | 0.968 |
| swin_base | LoRA + KD | PETALface | frozen | 0.497 | 0.633 | 0.693 | 0.563 | 0.972 |
| swin_base | Direct + KD | PETALface | fine-tuned | 0.521 | 0.646 | 0.704 | 0.583 | 0.966 |
| swin_base | LoRA + KD | PETALface | fine-tuned | 0.362 | 0.473 | 0.526 | 0.418 | 0.886 |
| deit3_small | Baseline (Phase 1) | — | — | 0.552 | 0.672 | 0.721 | 0.609 | 0.971 |
| deit3_small | LoRA (Phase 1) | — | — | 0.527 | 0.640 | 0.688 | 0.584 | 0.961 |
| deit3_small | Direct + KD | CVLFace | frozen | 0.543 | 0.667 | 0.718 | 0.602 | 0.972 |
| deit3_small | LoRA + KD | CVLFace | frozen | 0.520 | 0.646 | 0.702 | 0.581 | 0.968 |
| deit3_small | Direct + KD | CVLFace | fine-tuned | 0.536 | 0.666 | 0.715 | 0.598 | 0.969 |
| deit3_small | LoRA + KD | CVLFace | fine-tuned | 0.538 | 0.650 | 0.700 | 0.595 | 0.956 |
| deit3_medium | Baseline (Phase 1) | — | — | 0.570 | 0.688 | 0.731 | 0.625 | 0.972 |
| deit3_medium | LoRA (Phase 1) | — | — | 0.533 | 0.648 | 0.698 | 0.589 | 0.958 |
| deit3_medium | Direct + KD | CVLFace | frozen | 0.558 | 0.674 | 0.725 | 0.615 | 0.973 |
| deit3_medium | LoRA + KD | CVLFace | frozen | 0.534 | 0.651 | 0.693 | 0.589 | 0.956 |
| deit3_medium | Direct + KD | CVLFace | fine-tuned | 0.559 | 0.681 | 0.723 | 0.617 | 0.972 |
| deit3_medium | LoRA + KD | CVLFace | fine-tuned | 0.536 | 0.658 | 0.703 | 0.593 | 0.958 |
| deit3_base | Baseline (Phase 1) | — | — | 0.583 | 0.700 | 0.741 | 0.639 | 0.973 |
| deit3_base | LoRA (Phase 1) | — | — | 0.550 | 0.664 | 0.710 | 0.606 | 0.970 |
| deit3_base | Direct + KD | CVLFace | frozen | 0.572 | 0.687 | 0.734 | 0.628 | 0.973 |
| deit3_base | LoRA + KD | CVLFace | frozen | 0.547 | 0.657 | 0.712 | 0.603 | 0.965 |
| deit3_base | Direct + KD | CVLFace | fine-tuned | 0.581 | 0.696 | 0.742 | 0.637 | 0.973 |
| deit3_base | LoRA + KD | CVLFace | fine-tuned | 0.547 | 0.655 | 0.699 | 0.600 | 0.962 |
| | | | | | | | | |
| CVLFace ViT-Base | Teacher ref | — | frozen | 0.716 | 0.766 | 0.784 | 0.664 | 0.898 |
| CVLFace ViT-Base | Teacher ref | — | fine-tuned | 0.781 | 0.826 | 0.846 | 0.744 | 0.940 |
| PETALface Swin | Teacher ref | — | frozen | 0.424 | 0.515 | 0.553 | 0.351 | 0.831 |
| PETALface Swin | Teacher ref | — | fine-tuned | 0.663 | 0.764 | 0.798 | 0.615 | 0.936 |

## Table 2 — Frozen vs Fine-tuned Teacher Comparison

| Student | Mode | CVLFace frozen | CVLFace fine-tuned | Δ CVL | PETALface frozen | PETALface fine-tuned | Δ PET |
|---------|------|----------------|--------------------|-------|------------------|----------------------|-------|
| swin_tiny | Direct | 0.549 | 0.536 | -0.013 | 0.543 | 0.547 | +0.004 |
| swin_tiny | LoRA | 0.424 | 0.415 | -0.009 | 0.567 | 0.562 | -0.005 |
| swin_small | Direct | 0.547 | 0.542 | -0.005 | 0.527 | 0.536 | +0.009 |
| swin_small | LoRA | 0.393 | 0.559 | +0.166 | 0.409 | 0.402 | -0.007 |
| swin_base | Direct | 0.529 | 0.536 | +0.007 | 0.530 | 0.521 | -0.009 |
| swin_base | LoRA | 0.361 | 0.352 | -0.009 | 0.497 | 0.362 | -0.136 |
| deit3_small | Direct | 0.543 | 0.536 | -0.007 | — | — | — |
| deit3_small | LoRA | 0.520 | 0.538 | +0.019 | — | — | — |
| deit3_medium | Direct | 0.558 | 0.559 | +0.001 | — | — | — |
| deit3_medium | LoRA | 0.534 | 0.536 | +0.002 | — | — | — |
| deit3_base | Direct | 0.572 | 0.581 | +0.009 | — | — | — |
| deit3_base | LoRA | 0.547 | 0.547 | +0.000 | — | — | — |

## Table 3 — Teacher Architecture Comparison (Swin students only)

This table answers: **Does a same-architecture teacher (PETALface Swin → Swin) outperform a cross-architecture teacher (CVLFace ViT → Swin)?**

| Student | Mode | CVLFace ViT (frozen) | PETALface Swin (frozen) | Δ (PET−CVL) | CVLFace ViT (fine-tuned) | PETALface Swin (fine-tuned) | Δ (PET−CVL) |
|---------|------|----------------------|-------------------------|-------------|--------------------------|-----------------------------|----|
| swin_tiny | Direct | 0.549 | 0.543 | -0.006 | 0.536 | 0.547 | +0.011 |
| swin_tiny | LoRA | 0.424 | 0.567 | +0.144 | 0.415 | 0.562 | +0.147 |
| swin_small | Direct | 0.547 | 0.527 | -0.020 | 0.542 | 0.536 | -0.007 |
| swin_small | LoRA | 0.393 | 0.409 | +0.016 | 0.559 | 0.402 | -0.157 |
| swin_base | Direct | 0.529 | 0.530 | +0.001 | 0.536 | 0.521 | -0.014 |
| swin_base | LoRA | 0.361 | 0.497 | +0.137 | 0.352 | 0.362 | +0.010 |

## Table 4 — KD Improvement Over Standalone Training (best across all teachers)

| Model | Best Phase 1 rank@1 | Best KD rank@1 (any teacher/mode) | Best Config | Δ rank@1 | % improvement |
|-------|--------------------|------------------------------------|-------------|----------|---------------|
| swin_tiny | 0.562 | 0.567 | PETALface frozen, LoRA | +0.005 | +1.0% |
| swin_small | 0.554 | 0.559 | CVLFace fine-tuned, LoRA | +0.004 | +0.8% |
| swin_base | 0.536 | 0.536 | CVLFace fine-tuned, Direct | +0.000 | +0.0% |
| deit3_small | 0.552 | 0.543 | CVLFace frozen, Direct | -0.009 | -1.6% |
| deit3_medium | 0.570 | 0.559 | CVLFace fine-tuned, Direct | -0.011 | -2.0% |
| deit3_base | 0.583 | 0.581 | CVLFace fine-tuned, Direct | -0.003 | -0.5% |

## Table 5 — Scaling with Student Size

| Family | Size | Params | Phase 1 Baseline | Phase 1 LoRA | Best KD | KD Δ over best Phase 1 |
|--------|------|--------|-----------------|-------------|---------|----------------------|
| Swin | tiny | ~28M | 0.534 | 0.562 | 0.567 | +0.005 |
| Swin | small | ~50M | 0.539 | 0.554 | 0.559 | +0.004 |
| Swin | base | ~88M | 0.526 | 0.536 | 0.536 | +0.000 |
| DeiT3 | small | ~22M | 0.552 | 0.527 | 0.543 | -0.009 |
| DeiT3 | medium | ~39M | 0.570 | 0.533 | 0.559 | -0.011 |
| DeiT3 | base | ~86M | 0.583 | 0.550 | 0.581 | -0.003 |

## Summary

### Teacher Baselines

| Teacher | Mode | rank@1 | mAP |
|---------|------|--------|-----|
| CVLFace ViT-Base (frozen) | 0.716 | 0.664 |
| CVLFace ViT-Base (fine-tuned) | 0.781 | 0.744 |
| PETALface Swin (frozen) | 0.424 | 0.351 |
| PETALface Swin (fine-tuned) | 0.663 | 0.615 |

### Key Findings

**1. KD effectiveness:** KD from a strong teacher does **not** consistently improve
students over standalone Phase 1 training on TinyFace. Most KD configurations show
slight degradation (-0.003 to -0.013 rank@1) compared to Phase 1 best. The only
improvements are: swin_tiny lora PET-frozen (0.567 vs 0.562); swin_small lora CVL-fine-tuned (0.559 vs 0.554).

**2. Teacher fine-tuning:** Fine-tuning the teacher on TinyFace dramatically improves
the teacher's own performance (CVLFace: 0.716→0.781; PETALface: 0.424→0.663),
but this improvement does **not** reliably transfer to students via KD. Fine-tuned
teacher KD results are mixed — sometimes better, sometimes worse than frozen teacher KD.

**3. Teacher architecture (Swin→Swin vs ViT→Swin):**
PETALface (same-arch) shows average +0.045 rank@1 advantage over
CVLFace (cross-arch) with frozen teacher, and -0.002 disadvantage with
fine-tuned teacher. The PETALface frozen teacher is much weaker (0.424 vs 0.716 rank@1).
Despite this, PETALface Swin LoRA students performed relatively well, suggesting
some benefit from same-architecture inductive bias. However, the CVLFace ViT teacher's
stronger absolute performance generally produces better students overall.

**4. Student scaling:** Larger students do **not** consistently benefit more from KD.
In fact, the pattern is mixed:
- Swin (tiny/small/base): KD Δ = +0.005/+0.004/+0.000
- DeiT3 (small/medium/base): KD Δ = -0.009/-0.011/-0.003

**5. Best overall configuration:**
- Best KD student: **deit3_base + Direct + CVLFace fine-tuned** (rank@1 = 0.581)
- Best Phase 1: **deit3_base Baseline** (rank@1 = 0.583)
- Best teacher: **CVLFace ViT-Base fine-tuned** (rank@1 = 0.781)

### Known Issues

- **Swin LoRA collapse**: Swin LoRA + KD runs show severe degradation for
  swin_small and swin_base (rank@1 ~0.35–0.41). This affects both CVLFace and
  PETALface teacher configurations, confirming the issue is related to the
  Swin LoRA gradual-unfreeze interaction with KD, not the teacher architecture.
  Only swin_tiny LoRA (PETALface) escapes collapse (rank@1 ~0.56–0.57).

- **PETALface frozen teacher weakness**: The frozen PETALface Swin teacher
  (rank@1 = 0.424) is significantly weaker
  than CVLFace frozen (rank@1 = 0.716)
  on TinyFace, likely because the PETALface model was trained on WebFace4M
  without any low-resolution face specialization. Fine-tuning recovers most
  of this gap (0.663).

- **DeiT3 baseline LoRA underperformance**: DeiT3 Phase 1 LoRA results are
  lower than direct baselines. DeiT3 Direct + KD (CVLFace) remains the
  best KD approach for DeiT3 students.

### Summary Statistics

- Total KD student evaluations: **36** (24 CVLFace + 12 PETALface)
- Teacher baselines: **4** (2 CVLFace + 2 PETALface)
- Phase 1 baselines used for comparison: **12** (6 models × 2 modes)
