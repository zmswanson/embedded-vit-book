# KD Alpha Ablation Results

Sweep of `kd_alpha` (hard loss weight) across 8 values for 4 configurations.
`L_total = α * L_CE + (1 - α) * L_feature`

### Table 1: Alpha ablation — Config A (swin_tiny Direct + CVLFace frozen)

| α | Hard Loss | KD Loss | rank@1 | rank@5 | rank@10 | mAP | AUC |
|---|-----------|---------|--------|--------|---------|-----|-----|
| 0.0 | 0.0% | 100.0% | 0.616 | 0.716 | 0.757 | 0.664 | 0.974 |
| 0.1 | 10.0% | 90.0% | 0.568 | 0.674 | 0.725 | 0.621 | 0.970 |
| 0.3 | 30.0% | 70.0% | 0.546 | 0.668 | 0.717 | 0.605 | 0.969 |
| 0.5 | 50.0% | 50.0% | 0.548 | 0.667 | 0.721 | 0.607 | 0.966 |
| 0.7 | 70.0% | 30.0% | 0.549 | 0.665 | 0.717 | 0.606 | 0.968 |
| 0.8 | 80.0% | 20.0% | 0.553 | 0.671 | 0.718 | 0.611 | 0.969 |
| 0.9 | 90.0% | 10.0% | 0.547 | 0.668 | 0.716 | 0.606 | 0.967 |
| 1.0 | 100.0% | 0.0% | 0.527 | 0.652 | 0.700 | 0.588 | 0.970 |

**Optimal α = 0.0** (rank@1 = 0.616)
- vs α=0.7: Δ = +0.067

### Table 2: Alpha ablation — Config B (swin_tiny LoRA + PETALface frozen)

| α | Hard Loss | KD Loss | rank@1 | rank@5 | rank@10 | mAP | AUC |
|---|-----------|---------|--------|--------|---------|-----|-----|
| 0.0 | 0.0% | 100.0% | 0.450 | 0.556 | 0.610 | 0.504 | 0.944 |
| 0.1 | 10.0% | 90.0% | 0.581 | 0.692 | 0.734 | 0.634 | 0.972 |
| 0.3 | 30.0% | 70.0% | 0.571 | 0.686 | 0.725 | 0.625 | 0.972 |
| 0.5 | 50.0% | 50.0% | 0.562 | 0.677 | 0.726 | 0.618 | 0.973 |
| 0.7 | 70.0% | 30.0% | 0.567 | 0.684 | 0.730 | 0.623 | 0.969 |
| 0.8 | 80.0% | 20.0% | 0.565 | 0.678 | 0.730 | 0.621 | 0.968 |
| 0.9 | 90.0% | 10.0% | 0.565 | 0.676 | 0.723 | 0.620 | 0.968 |
| 1.0 | 100.0% | 0.0% | 0.567 | 0.688 | 0.730 | 0.625 | 0.972 |

**Optimal α = 0.1** (rank@1 = 0.581)
- vs α=0.7: Δ = +0.014

### Table 3: Alpha ablation — Config C (deit3_base Direct + CVLFace fine-tuned)

| α | Hard Loss | KD Loss | rank@1 | rank@5 | rank@10 | mAP | AUC |
|---|-----------|---------|--------|--------|---------|-----|-----|
| 0.0 | 0.0% | 100.0% | 0.572 | 0.689 | 0.737 | 0.629 | 0.966 |
| 0.1 | 10.0% | 90.0% | 0.593 | 0.704 | 0.746 | 0.646 | 0.974 |
| 0.3 | 30.0% | 70.0% | 0.579 | 0.697 | 0.741 | 0.635 | 0.975 |
| 0.5 | 50.0% | 50.0% | 0.575 | 0.693 | 0.741 | 0.631 | 0.974 |
| 0.7 | 70.0% | 30.0% | 0.581 | 0.696 | 0.742 | 0.637 | 0.973 |
| 0.8 | 80.0% | 20.0% | 0.576 | 0.692 | 0.742 | 0.632 | 0.973 |
| 0.9 | 90.0% | 10.0% | 0.573 | 0.692 | 0.737 | 0.630 | 0.973 |
| 1.0 | 100.0% | 0.0% | 0.584 | 0.699 | 0.742 | 0.639 | 0.973 |

**Optimal α = 0.1** (rank@1 = 0.593)
- vs α=0.7: Δ = +0.013

### Table 4: Alpha ablation — Config D (deit3_base Direct + PETALface frozen)

| α | Hard Loss | KD Loss | rank@1 | rank@5 | rank@10 | mAP | AUC |
|---|-----------|---------|--------|--------|---------|-----|-----|
| 0.0 | 0.0% | 100.0% | 0.416 | 0.522 | 0.571 | 0.470 | 0.926 |
| 0.1 | 10.0% | 90.0% | 0.586 | 0.697 | 0.742 | 0.640 | 0.974 |
| 0.3 | 30.0% | 70.0% | 0.573 | 0.697 | 0.740 | 0.632 | 0.974 |
| 0.5 | 50.0% | 50.0% | 0.572 | 0.687 | 0.730 | 0.628 | 0.974 |
| 0.7 | 70.0% | 30.0% | 0.576 | 0.692 | 0.737 | 0.632 | 0.972 |
| 0.8 | 80.0% | 20.0% | 0.580 | 0.697 | 0.744 | 0.637 | 0.973 |
| 0.9 | 90.0% | 10.0% | 0.579 | 0.693 | 0.736 | 0.633 | 0.972 |
| 1.0 | 100.0% | 0.0% | 0.581 | 0.694 | 0.738 | 0.637 | 0.973 |

**Optimal α = 0.1** (rank@1 = 0.586)
- vs α=0.7: Δ = +0.010

### Table 5: Optimal α comparison across configs

| Config | Student | Teacher | Mode | Optimal α | rank@1 (optimal) | rank@1 (α=0.7) | Δ (opt vs 0.7) |
|--------|---------|---------|------|-----------|-----------------|----------------|-----------------|
| A | swin_tiny | CVLFace frozen | Direct | 0.0 | 0.616 | 0.549 | +0.067 |
| B | swin_tiny | PETALface frozen | LoRA | 0.1 | 0.581 | 0.567 | +0.014 |
| C | deit3_base | CVLFace fine-tuned | Direct | 0.1 | 0.593 | 0.581 | +0.013 |
| D | deit3_base | PETALface frozen | Direct | 0.1 | 0.586 | 0.576 | +0.010 |

## Summary

**Config A (swin_tiny Direct + CVLFace frozen):**
- Optimal α = 0.0 (rank@1 = 0.616)
- α=0.7 (default): rank@1 = 0.549, gap from optimal: +0.067
- α=1.0 (no KD): rank@1 = 0.527
- α=0.0 (pure KD): rank@1 = 0.616
- Performance spread: 0.089 (max - min across all α)

**Config B (swin_tiny LoRA + PETALface frozen):**
- Optimal α = 0.1 (rank@1 = 0.581)
- α=0.7 (default): rank@1 = 0.567, gap from optimal: +0.014
- α=1.0 (no KD): rank@1 = 0.567
- α=0.0 (pure KD): rank@1 = 0.450
- Performance spread: 0.131 (max - min across all α)

**Config C (deit3_base Direct + CVLFace fine-tuned):**
- Optimal α = 0.1 (rank@1 = 0.593)
- α=0.7 (default): rank@1 = 0.581, gap from optimal: +0.013
- α=1.0 (no KD): rank@1 = 0.584
- α=0.0 (pure KD): rank@1 = 0.572
- Performance spread: 0.021 (max - min across all α)

**Config D (deit3_base Direct + PETALface frozen):**
- Optimal α = 0.1 (rank@1 = 0.586)
- α=0.7 (default): rank@1 = 0.576, gap from optimal: +0.010
- α=1.0 (no KD): rank@1 = 0.581
- α=0.0 (pure KD): rank@1 = 0.416
- Performance spread: 0.170 (max - min across all α)

### Key Findings

1. **Optimal α values**: 0.0, 0.1, 0.1, 0.1 for configs A/B/C/D respectively
   - α=0.7 is NOT the optimal value for any configuration
2. **CVLFace teacher** optimal α: 0.0 (Swin), 0.1 (DeiT3)
   **PETALface teacher** optimal α: 0.1 (Swin), 0.1 (DeiT3)
3. **Swin students** optimal α: 0.0 (CVLFace), 0.1 (PETALface)
   **DeiT3 students** optimal α: 0.1 (CVLFace), 0.1 (PETALface)
4. **Sensitivity analysis**:
   - Config A: mid-range spread (α=0.1–0.9) = 0.022 (robust)
   - Config B: mid-range spread (α=0.1–0.9) = 0.019 (robust)
   - Config C: mid-range spread (α=0.1–0.9) = 0.021 (robust)
   - Config D: mid-range spread (α=0.1–0.9) = 0.013 (robust)
5. **Extreme α behavior**:
   - Config A: α=0.0 (0.616) vs α=1.0 (0.527) → pure KD > no KD
   - Config B: α=0.0 (0.450) vs α=1.0 (0.567) → no KD > pure KD
   - Config C: α=0.0 (0.572) vs α=1.0 (0.584) → no KD > pure KD
   - Config D: α=0.0 (0.416) vs α=1.0 (0.581) → no KD > pure KD
