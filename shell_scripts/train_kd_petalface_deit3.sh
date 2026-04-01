#!/bin/bash
# Train Knowledge-Distilled DeiT3 Student Models — PETALface Swin Teacher
#
# 12 KD runs total:
# - Part E: 6 runs — PETALface frozen → 3 DeiT3 students × 2 modes (direct + LoRA)
# - Part F: 6 runs — PETALface fine-tuned → 3 DeiT3 students × 2 modes (direct + LoRA)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOGFILE="configs/train_kd_petalface_deit3.log"
echo "=== Train PETALface KD DeiT3 Student Models ===" | tee "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────
# Fine-tuned teacher checkpoint
# ──────────────────────────────────────────────
PETALFACE_FT_CKPT="/mnt/data/wandb/checkpoints/vit_book_teacher_finetune/t2dm526v/teacher_backbone.pt"

# ──────────────────────────────────────────────
# KD common settings
# ──────────────────────────────────────────────
KD_COMMON="--img_size 96 --batch_size 8 --max_epochs 50 --seed 73 \
    --head_type adaface --kd_alpha 0.7 --kd_feature_loss cosine \
    --warmup_epochs 8 --unfreeze_every_epochs 5 --rank_k 10 \
    --wandb_project vit_book_kd"

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

# Helper: run 3 DeiT3 students × 2 modes with given teacher args
train_deit3_students() {
    local teacher_args="$1"

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
# PART E: PETALface Swin (frozen) → 3 DeiT3 students × 2 modes (6 runs)
# ══════════════════════════════════════════════
echo "" | tee -a "$LOGFILE"
echo "══════ PART E: PETALface Frozen → DeiT3 Students ══════" | tee -a "$LOGFILE"
train_deit3_students "--teacher_type petalface_swin"

# ══════════════════════════════════════════════
# PART F: PETALface Swin (fine-tuned) → 3 DeiT3 students × 2 modes (6 runs)
# ══════════════════════════════════════════════
echo "" | tee -a "$LOGFILE"
echo "══════ PART F: PETALface Fine-tuned → DeiT3 Students ══════" | tee -a "$LOGFILE"
train_deit3_students "--teacher_type petalface_swin --teacher_finetuned --teacher_finetune_ckpt_path ${PETALFACE_FT_CKPT}"

echo "" | tee -a "$LOGFILE"
echo "=== All PETALface DeiT3 KD training complete ===" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
