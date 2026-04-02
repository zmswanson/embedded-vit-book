#!/bin/bash
# KD Alpha Ablation Study
#
# Sweeps kd_alpha on 4 best configs (2 students × 2 teachers) from Phase 2.5
# 8 alpha values × 4 configs = 32 runs

set -euo pipefail

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vit-benchmark

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOGFILE="configs/train_kd_alpha_ablation.log"
echo "=== KD Alpha Ablation ===" | tee "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"

CVLFACE_FT_CKPT="/mnt/data/wandb/checkpoints/vit_book_teacher_finetune/vkt8uhga/teacher_backbone.pt"
PETALFACE_FT_CKPT="/mnt/data/wandb/checkpoints/vit_book_teacher_finetune/t2dm526v/teacher_backbone.pt"

# ──────────────────────────────────────────────
# Config A: Swin + CVLFace (swin_tiny Direct, CVLFace frozen)
# ──────────────────────────────────────────────
SWIN_CVL_MODEL="swin_tiny_patch4_window7_224.ms_in1k"
SWIN_CVL_TEACHER="--teacher_type cvlface_vit_base"
SWIN_CVL_UNFREEZE="--gradual_unfreeze stage --thaw_N_stages 2"
SWIN_CVL_HP="--lr 0.000385 --weight_decay 1e-5 --backbone_dropout 0.165 \
    --head_scale 96 --head_margin 0.329 --adaface_h 0.226 --adaface_t_alpha 0.042"

# ──────────────────────────────────────────────
# Config B: Swin + PETALface (swin_tiny LoRA, PETALface frozen)
# ──────────────────────────────────────────────
SWIN_PET_MODEL="swin_tiny_patch4_window7_224.ms_in1k"
SWIN_PET_TEACHER="--teacher_type petalface_swin"
SWIN_PET_UNFREEZE="--gradual_unfreeze stage --thaw_N_stages 2"
SWIN_PET_HP="--lr 0.000193 --weight_decay 0.0001 --backbone_dropout 0.0 \
    --head_scale 96 --head_margin 0.308 --adaface_h 0.321 --adaface_t_alpha 0.023 \
    --lora_enabled --lora_r 24 --lora_alpha 120 --lora_dropout 0.05 --lora_qkv_proj 3"

# ──────────────────────────────────────────────
# Config C: DeiT3 + CVLFace (deit3_base Direct, CVLFace fine-tuned)
# ──────────────────────────────────────────────
DEIT3_CVL_MODEL="deit3_base_patch16_224.fb_in1k"
DEIT3_CVL_TEACHER="--teacher_type cvlface_vit_base --teacher_finetuned \
    --teacher_finetune_ckpt_path ${CVLFACE_FT_CKPT}"
DEIT3_CVL_UNFREEZE="--gradual_unfreeze block --thaw_N_blocks 8"
DEIT3_CVL_HP="--lr 5.782e-4 --weight_decay 1e-5 --backbone_dropout 0.165 \
    --head_scale 96 --head_margin 0.357 --adaface_h 0.2015 --adaface_t_alpha 0.0438"

# ──────────────────────────────────────────────
# Config D: DeiT3 + PETALface (deit3_base Direct, PETALface frozen)
# ──────────────────────────────────────────────
DEIT3_PET_MODEL="deit3_base_patch16_224.fb_in1k"
DEIT3_PET_TEACHER="--teacher_type petalface_swin"
DEIT3_PET_UNFREEZE="--gradual_unfreeze block --thaw_N_blocks 8"
DEIT3_PET_HP="--lr 5.782e-4 --weight_decay 1e-5 --backbone_dropout 0.165 \
    --head_scale 96 --head_margin 0.357 --adaface_h 0.2015 --adaface_t_alpha 0.0438"

# ──────────────────────────────────────────────
# Common settings (same as Phase 2.2 except kd_alpha varies)
# ──────────────────────────────────────────────
KD_BASE="--img_size 96 --batch_size 8 --max_epochs 50 --seed 73 \
    --head_type adaface --kd_feature_loss cosine \
    --warmup_epochs 8 --unfreeze_every_epochs 5 --rank_k 10 \
    --wandb_project vit_book_kd_alpha_ablation"

ALPHAS=(0.0 0.1 0.3 0.5 0.7 0.8 0.9 1.0)

# ──────────────────────────────────────────────
# Training function
# ──────────────────────────────────────────────
train_alpha() {
    local label=$1
    local student_name=$2
    local alpha=$3
    shift 3
    echo "" | tee -a "$LOGFILE"
    echo "────────────────────────────────────────" | tee -a "$LOGFILE"
    echo "Alpha Ablation [$label]: $student_name  kd_alpha=$alpha" | tee -a "$LOGFILE"
    echo "Start: $(date)" | tee -a "$LOGFILE"

    if python training_pipeline_kd.py \
        --student_model_name "$student_name" \
        --kd_alpha "$alpha" \
        ${KD_BASE} "$@" 2>&1 | tee -a "$LOGFILE"; then
        echo "✓ [$label] alpha=$alpha completed at $(date)" | tee -a "$LOGFILE"
    else
        echo "✗ [$label] alpha=$alpha FAILED at $(date)" | tee -a "$LOGFILE"
        echo "  Continuing to next..." | tee -a "$LOGFILE"
    fi
}

# ══════════════════════════════════════════════
# Config A: Swin + CVLFace — sweep alpha (8 runs)
# ══════════════════════════════════════════════
echo "" | tee -a "$LOGFILE"
echo "══════ Config A: Swin + CVLFace Alpha Ablation ══════" | tee -a "$LOGFILE"
for alpha in "${ALPHAS[@]}"; do
    train_alpha "A-swin-cvl" "$SWIN_CVL_MODEL" "$alpha" \
        ${SWIN_CVL_TEACHER} ${SWIN_CVL_UNFREEZE} ${SWIN_CVL_HP}
done

# ══════════════════════════════════════════════
# Config B: Swin + PETALface — sweep alpha (8 runs)
# ══════════════════════════════════════════════
echo "" | tee -a "$LOGFILE"
echo "══════ Config B: Swin + PETALface Alpha Ablation ══════" | tee -a "$LOGFILE"
for alpha in "${ALPHAS[@]}"; do
    train_alpha "B-swin-pet" "$SWIN_PET_MODEL" "$alpha" \
        ${SWIN_PET_TEACHER} ${SWIN_PET_UNFREEZE} ${SWIN_PET_HP}
done

# ══════════════════════════════════════════════
# Config C: DeiT3 + CVLFace — sweep alpha (8 runs)
# ══════════════════════════════════════════════
echo "" | tee -a "$LOGFILE"
echo "══════ Config C: DeiT3 + CVLFace Alpha Ablation ══════" | tee -a "$LOGFILE"
for alpha in "${ALPHAS[@]}"; do
    train_alpha "C-deit3-cvl" "$DEIT3_CVL_MODEL" "$alpha" \
        ${DEIT3_CVL_TEACHER} ${DEIT3_CVL_UNFREEZE} ${DEIT3_CVL_HP}
done

# ══════════════════════════════════════════════
# Config D: DeiT3 + PETALface — sweep alpha (8 runs)
# ══════════════════════════════════════════════
echo "" | tee -a "$LOGFILE"
echo "══════ Config D: DeiT3 + PETALface Alpha Ablation ══════" | tee -a "$LOGFILE"
for alpha in "${ALPHAS[@]}"; do
    train_alpha "D-deit3-pet" "$DEIT3_PET_MODEL" "$alpha" \
        ${DEIT3_PET_TEACHER} ${DEIT3_PET_UNFREEZE} ${DEIT3_PET_HP}
done

echo "" | tee -a "$LOGFILE"
echo "=== Alpha ablation complete (32 runs) ===" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
