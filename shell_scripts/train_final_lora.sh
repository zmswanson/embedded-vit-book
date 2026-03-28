#!/bin/bash
# Train Final LoRA+AdaFace Models
#
# Trains all 6 target models using best HP configs from initial sweep analysis.
# Swin family uses swin_lora_adaface config; DeiT3 uses deit3_lora_adaface config.
#
# Expected runtime: ~12–24 hours total (sequential, RTX 3090, batch_size=8, 50 epochs)

set -euo pipefail

# Navigate to repo root if invoked from shell_scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOGFILE="configs/train_final_lora.log"
echo "=== Train Final LoRA+AdaFace Models ===" | tee "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"

# ──────────────────────────────────────────────
# Best configs from initial sweep analysis
# ──────────────────────────────────────────────

# Swin LoRA+AdaFace best config (rank@1 = 0.5814)
SWIN_LR=0.00019313294723680936
SWIN_WD=0.0001
SWIN_BD=0.0
SWIN_HEAD_SCALE=96.0
SWIN_HEAD_MARGIN=0.3076286324927164
SWIN_ADAFACE_H=0.3206180757713378
SWIN_ADAFACE_TA=0.023366334755827948
SWIN_LORA_R=24
SWIN_LORA_ALPHA=120.0
SWIN_LORA_DROPOUT=0.05

# DeiT3 LoRA+AdaFace best config (rank@1 = 0.5891)
DEIT3_LR=0.0002545950184362033
DEIT3_WD=2e-6
DEIT3_BD=0.0
DEIT3_HEAD_SCALE=32.0
DEIT3_HEAD_MARGIN=0.22741634046252604
DEIT3_ADAFACE_H=0.21421962489680813
DEIT3_ADAFACE_TA=0.012527107879321934
DEIT3_LORA_R=4
DEIT3_LORA_ALPHA=24.0
DEIT3_LORA_DROPOUT=0.1

# ──────────────────────────────────────────────
# Common settings
# ──────────────────────────────────────────────
COMMON_ARGS="--img_size 96 --batch_size 8 --max_epochs 50 --seed 73 \
    --head_type adaface --lora_enabled --lora_qkv_proj 3 \
    --warmup_epochs 8 --unfreeze_every_epochs 5 --rank_k 10 \
    --wandb_project vit_book_final_lora"

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
    swin_base_patch4_window7_224.ms_in1k
do
    train_model "$MODEL" \
        --lr $SWIN_LR \
        --weight_decay $SWIN_WD \
        --backbone_dropout $SWIN_BD \
        --head_scale $SWIN_HEAD_SCALE \
        --head_margin $SWIN_HEAD_MARGIN \
        --adaface_h $SWIN_ADAFACE_H \
        --adaface_t_alpha $SWIN_ADAFACE_TA \
        --lora_r $SWIN_LORA_R \
        --lora_alpha $SWIN_LORA_ALPHA \
        --lora_dropout $SWIN_LORA_DROPOUT \
        --gradual_unfreeze stage --thaw_N_stages 2
done

# ──────────────────────────────────────────────
# DeiT3 models (block-based gradual unfreeze)
# ──────────────────────────────────────────────
for MODEL in \
    deit3_small_patch16_224.fb_in1k \
    deit3_medium_patch16_224.fb_in1k \
    deit3_base_patch16_224.fb_in1k
do
    train_model "$MODEL" \
        --lr $DEIT3_LR \
        --weight_decay $DEIT3_WD \
        --backbone_dropout $DEIT3_BD \
        --head_scale $DEIT3_HEAD_SCALE \
        --head_margin $DEIT3_HEAD_MARGIN \
        --adaface_h $DEIT3_ADAFACE_H \
        --adaface_t_alpha $DEIT3_ADAFACE_TA \
        --lora_r $DEIT3_LORA_R \
        --lora_alpha $DEIT3_LORA_ALPHA \
        --lora_dropout $DEIT3_LORA_DROPOUT \
        --gradual_unfreeze block --thaw_N_blocks 8
done

echo "" | tee -a "$LOGFILE"
echo "=== All training complete ===" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
