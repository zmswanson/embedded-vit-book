#!/bin/bash
# Train Knowledge-Distilled Student Models — PETALface Swin Teacher
#
# 12 KD runs total:
# - Part C: 6 runs — PETALface frozen → 3 Swin students × 2 modes (direct + LoRA)
# - Part D: 6 runs — PETALface fine-tuned → 3 Swin students × 2 modes (direct + LoRA)
#
# Prerequisites: Run shell_scripts/finetune_petalface.sh first for Part D

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOGFILE="configs/train_kd_petalface.log"
echo "=== Train PETALface KD Student Models ===" | tee "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────
# Fine-tuned teacher checkpoint path (fill in after finetune_petalface.sh)
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

# Helper: run 3 Swin students × 2 modes with given teacher args
train_swin_students() {
    local teacher_args="$1"

    # Swin students (direct)
    for model in swin_tiny_patch4_window7_224.ms_in1k swin_small_patch4_window7_224.ms_in1k swin_base_patch4_window7_224.ms_in1k; do
        train_kd_model "$model" ${teacher_args} --gradual_unfreeze stage --thaw_N_stages 2 ${SWIN_DIRECT_HP}
    done
    # Swin students (LoRA)
    for model in swin_tiny_patch4_window7_224.ms_in1k swin_small_patch4_window7_224.ms_in1k swin_base_patch4_window7_224.ms_in1k; do
        train_kd_model "$model" ${teacher_args} --gradual_unfreeze stage --thaw_N_stages 2 ${SWIN_LORA_HP}
    done
}

# ══════════════════════════════════════════════
# PART C: PETALface Swin (frozen) → 3 Swin students × 2 modes (6 runs)
# ══════════════════════════════════════════════
echo "" | tee -a "$LOGFILE"
echo "══════ PART C: PETALface Frozen → Swin Students ══════" | tee -a "$LOGFILE"
train_swin_students "--teacher_type petalface_swin"

# ══════════════════════════════════════════════
# PART D: PETALface Swin (fine-tuned) → 3 Swin students × 2 modes (6 runs)
# ══════════════════════════════════════════════
echo "" | tee -a "$LOGFILE"
echo "══════ PART D: PETALface Fine-tuned → Swin Students ══════" | tee -a "$LOGFILE"
train_swin_students "--teacher_type petalface_swin --teacher_finetuned --teacher_finetune_ckpt_path ${PETALFACE_FT_CKPT}"

echo "" | tee -a "$LOGFILE"
echo "=== All PETALface KD training complete ===" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
