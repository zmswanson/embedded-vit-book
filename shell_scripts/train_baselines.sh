#!/bin/bash
# Train Baseline Models (No LoRA)
#
# Full fine-tuning baselines with AdaFace heads for all 6 target models.
# Swin family uses best config from vit_book_adaface sweep (rank@1 = 0.5543).
# DeiT3 family uses model ablation config (no separate DeiT3 AdaFace sweep).
#
# Expected runtime: ~12–24 hours total (sequential, RTX 3090, batch_size=8, 50 epochs)

set -euo pipefail

# Navigate to repo root if invoked from shell_scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOGFILE="configs/train_baselines.log"
echo "=== Train Baseline Models (No LoRA) ===" | tee "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────
# Best configs per family
# ──────────────────────────────────────────────

# Swin AdaFace best config (from vit_book_adaface sweep)
SWIN_LR=0.0003851305720676699
SWIN_WD=0.00001
SWIN_BD=0.165
SWIN_HEAD_SCALE=96
SWIN_HEAD_MARGIN=0.3292590043922424
SWIN_ADAFACE_H=0.2255528043305555
SWIN_ADAFACE_TA=0.04203943269064411

# DeiT3 AdaFace config (from model ablation study)
DEIT3_LR=5.782e-4
DEIT3_WD=1.0e-5
DEIT3_BD=0.165
DEIT3_HEAD_SCALE=96
DEIT3_HEAD_MARGIN=0.357
DEIT3_ADAFACE_H=0.2015
DEIT3_ADAFACE_TA=0.0438

# ──────────────────────────────────────────────
# Common settings (NO LoRA flags)
# ──────────────────────────────────────────────
COMMON_ARGS="--img_size 96 --batch_size 8 --max_epochs 50 --seed 73 \
    --head_type adaface \
    --warmup_epochs 8 --unfreeze_every_epochs 5 --rank_k 10 \
    --wandb_project vit_book_final_baselines"

# ──────────────────────────────────────────────
# Training function with error handling
# ──────────────────────────────────────────────
train_model() {
    local model_name=$1
    shift
    echo "" | tee -a "$LOGFILE"
    echo "────────────────────────────────────────" | tee -a "$LOGFILE"
    echo "Training: $model_name" | tee -a "$LOGFILE"
    echo "Start: $(date)" | tee -a "$LOGFILE"

    if python training_pipeline.py --model_name "$model_name" "$@" $COMMON_ARGS 2>&1 | tee -a "$LOGFILE"; then
        echo "✓ $model_name completed successfully at $(date)" | tee -a "$LOGFILE"
    else
        echo "✗ $model_name FAILED at $(date) (exit code $?)" | tee -a "$LOGFILE"
        echo "  Continuing to next model..." | tee -a "$LOGFILE"
    fi
}

# ──────────────────────────────────────────────
# Swin models (stage-based gradual unfreeze)
# ──────────────────────────────────────────────
for MODEL in \
    swin_tiny_patch4_window7_224.ms_in1k \
    swin_small_patch4_window7_224.ms_in1k \
    swin_base_patch4_window7_224.ms_in1k; do

    train_model "$MODEL" \
        --gradual_unfreeze stage --thaw_N_stages 2 \
        --lr "$SWIN_LR" --weight_decay "$SWIN_WD" --backbone_dropout "$SWIN_BD" \
        --head_scale "$SWIN_HEAD_SCALE" --head_margin "$SWIN_HEAD_MARGIN" \
        --adaface_h "$SWIN_ADAFACE_H" --adaface_t_alpha "$SWIN_ADAFACE_TA"
done

# ──────────────────────────────────────────────
# DeiT3 models (block-based gradual unfreeze)
# ──────────────────────────────────────────────
for MODEL in \
    deit3_small_patch16_224.fb_in1k \
    deit3_medium_patch16_224.fb_in1k \
    deit3_base_patch16_224.fb_in1k; do

    train_model "$MODEL" \
        --gradual_unfreeze block --thaw_N_blocks 8 \
        --lr "$DEIT3_LR" --weight_decay "$DEIT3_WD" --backbone_dropout "$DEIT3_BD" \
        --head_scale "$DEIT3_HEAD_SCALE" --head_margin "$DEIT3_HEAD_MARGIN" \
        --adaface_h "$DEIT3_ADAFACE_H" --adaface_t_alpha "$DEIT3_ADAFACE_TA"
done

echo "" | tee -a "$LOGFILE"
echo "=== Training Complete ===" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
