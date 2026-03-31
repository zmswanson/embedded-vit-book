#!/bin/bash
# Train Knowledge-Distilled Student Models — CVLFace Teacher
#
# 24 KD runs total:
# - Part A: 12 runs — CVLFace frozen → 6 students × 2 modes
# - Part B: 12 runs — CVLFace fine-tuned → 6 students × 2 modes
#
# Prerequisites: Run shell_scripts/finetune_cvlface.sh first for Part B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOGFILE="configs/train_kd_cvlface.log"
echo "=== Train CVLFace KD Student Models ===" | tee "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────
# Fine-tuned teacher checkpoint path (fill in after Step 0)
# ──────────────────────────────────────────────
CVLFACE_FT_CKPT="/mnt/data/wandb/checkpoints/vit_book_teacher_finetune/vkt8uhga/teacher_backbone.pt"

# ──────────────────────────────────────────────
# KD common settings
# ──────────────────────────────────────────────
KD_COMMON="--img_size 96 --batch_size 8 --max_epochs 50 --seed 73 \
    --head_type adaface --kd_alpha 0.7 --kd_feature_loss cosine \
    --warmup_epochs 8 --unfreeze_every_epochs 5 --rank_k 10 \
    --wandb_project vit_book_kd"

# ──────────────────────────────────────────────
# Swin baseline (direct) HP
# ──────────────────────────────────────────────
SWIN_DIRECT_HP="--lr 0.000385 --weight_decay 1e-5 --backbone_dropout 0.165 \
    --head_scale 96 --head_margin 0.329 --adaface_h 0.226 --adaface_t_alpha 0.042"

# ──────────────────────────────────────────────
# Swin LoRA HP
# ──────────────────────────────────────────────
SWIN_LORA_HP="--lr 0.000193 --weight_decay 0.0001 --backbone_dropout 0.0 \
    --head_scale 96 --head_margin 0.308 --adaface_h 0.321 --adaface_t_alpha 0.023 \
    --lora_enabled --lora_r 24 --lora_alpha 120 --lora_dropout 0.05 --lora_qkv_proj 3"

# ──────────────────────────────────────────────
# DeiT3 baseline (direct) HP
# ──────────────────────────────────────────────
DEIT3_DIRECT_HP="--lr 5.782e-4 --weight_decay 1e-5 --backbone_dropout 0.165 \
    --head_scale 96 --head_margin 0.357 --adaface_h 0.2015 --adaface_t_alpha 0.0438"

# ──────────────────────────────────────────────
# DeiT3 LoRA HP
# ──────────────────────────────────────────────
DEIT3_LORA_HP="--lr 0.000255 --weight_decay 2e-6 --backbone_dropout 0.0 \
    --head_scale 32 --head_margin 0.227 --adaface_h 0.214 --adaface_t_alpha 0.013 \
    --lora_enabled --lora_r 4 --lora_alpha 24 --lora_dropout 0.1 --lora_qkv_proj 3"

# ──────────────────────────────────────────────
# Training function
# ──────────────────────────────────────────────
train_kd_model() {
    local student_name=$1
    shift
    echo "" | tee -a "$LOGFILE"
    echo "────────────────────────────────────────" | tee -a "$LOGFILE"
    echo "KD Training: $student_name ($@)" | tee -a "$LOGFILE"
    echo "Start: $(date)" | tee -a "$LOGFILE"

    if python training_pipeline_kd.py \
        --student_model_name "$student_name" \
        ${KD_COMMON} "$@" 2>&1 | tee -a "$LOGFILE"; then
        echo "✓ $student_name completed successfully at $(date)" | tee -a "$LOGFILE"
    else
        echo "✗ $student_name FAILED at $(date) (exit code $?)" | tee -a "$LOGFILE"
        echo "  Continuing to next model..." | tee -a "$LOGFILE"
    fi
}

# Helper: run all 6 students with given teacher args
train_all_students() {
    local teacher_args="$1"

    # Swin students (direct)
    for model in swin_tiny_patch4_window7_224.ms_in1k swin_small_patch4_window7_224.ms_in1k swin_base_patch4_window7_224.ms_in1k; do
        train_kd_model "$model" ${teacher_args} --gradual_unfreeze stage --thaw_N_stages 2 ${SWIN_DIRECT_HP}
    done
    # Swin students (LoRA)
    for model in swin_tiny_patch4_window7_224.ms_in1k swin_small_patch4_window7_224.ms_in1k swin_base_patch4_window7_224.ms_in1k; do
        train_kd_model "$model" ${teacher_args} --gradual_unfreeze stage --thaw_N_stages 2 ${SWIN_LORA_HP}
    done
    # DeiT3 students (direct)
    for model in deit3_small_patch16_224.fb_in1k deit3_medium_patch16_224.fb_in1k deit3_base_patch16_224.fb_in1k; do
        train_kd_model "$model" ${teacher_args} --gradual_unfreeze block --thaw_N_blocks 8 ${DEIT3_DIRECT_HP}
    done
    # DeiT3 students (LoRA)
    for model in deit3_small_patch16_224.fb_in1k deit3_medium_patch16_224.fb_in1k deit3_base_patch16_224.fb_in1k; do
        train_kd_model "$model" ${teacher_args} --gradual_unfreeze block --thaw_N_blocks 8 ${DEIT3_LORA_HP}
    done
}

# ══════════════════════════════════════════════
# PART A: CVLFace ViT-Base (frozen) → all 6 students (12 runs)
# ══════════════════════════════════════════════
echo "" | tee -a "$LOGFILE"
echo "══════ PART A: CVLFace Frozen → All Students ══════" | tee -a "$LOGFILE"
train_all_students "--teacher_type cvlface_vit_base"

# ══════════════════════════════════════════════
# PART B: CVLFace ViT-Base (fine-tuned) → all 6 students (12 runs)
# ══════════════════════════════════════════════
echo "" | tee -a "$LOGFILE"
echo "══════ PART B: CVLFace Fine-tuned → All Students ══════" | tee -a "$LOGFILE"
train_all_students "--teacher_type cvlface_vit_base --teacher_finetuned --teacher_finetune_ckpt_path ${CVLFACE_FT_CKPT}"

echo "" | tee -a "$LOGFILE"
echo "=== All CVLFace KD training complete ===" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
