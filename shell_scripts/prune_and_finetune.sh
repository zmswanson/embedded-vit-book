#!/bin/bash
set -e

if [[ "$(dirname "$0")" == *"shell_scripts"* ]]; then
    cd "$(dirname "$0")/.."
fi

# ============================================================
# Checkpoint paths
# ============================================================
DEIT3_BASE_LORA="/mnt/data/wandb/checkpoints/vit_book_final_baselines/vimriwcl/epoch=44-rank1tinyface/eval/rank@1=0.579.ckpt"
DEIT3_BASE_KD="/mnt/data/wandb/checkpoints/vit_book_kd/5mngq5jn/epoch=47-rank1tinyface/eval/rank@1=0.579.ckpt"
SWIN_BASE="/mnt/data/wandb/checkpoints/vit_book_final_lora/ci53ufy8/epoch=39-rank1tinyface/eval/rank@1=0.556.ckpt"
SWIN_TINY_KD="/mnt/data/wandb/checkpoints/vit_book_kd/y6ddjqyw/epoch=44-rank1tinyface/eval/rank@1=0.570.ckpt"

FT_EPOCHS=10
FT_LR=3e-5
WANDB_PROJECT="vit_book_pruning"
OUT_DIR="pruned_models"

mkdir -p "${OUT_DIR}"

# ============================================================
# Experiment 1–3: deit3_base LoRA — head pruning 10%, 25%, 50%
# ============================================================
for ratio in 0.10 0.25 0.50; do
    echo "========== deit3_base LoRA heads ${ratio} =========="
    python pruning.py \
        --ckpt_path "${DEIT3_BASE_LORA}" \
        --model_name deit3_base_patch16_224.fb_in1k \
        --prune_method heads \
        --head_prune_ratio ${ratio} \
        --finetune_epochs ${FT_EPOCHS} \
        --finetune_lr ${FT_LR} \
        --output_ckpt "${OUT_DIR}/deit3_base_lora_heads_${ratio}" \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name "deit3_base_lora-heads_${ratio}-ft${FT_EPOCHS}"
done

# ============================================================
# Experiment 4: deit3_base LoRA — block pruning 25% (3 of 12)
# ============================================================
echo "========== deit3_base LoRA blocks 3/12 =========="
python pruning.py \
    --ckpt_path "${DEIT3_BASE_LORA}" \
    --model_name deit3_base_patch16_224.fb_in1k \
    --prune_method blocks \
    --num_blocks_to_remove 3 \
    --finetune_epochs ${FT_EPOCHS} \
    --finetune_lr ${FT_LR} \
    --output_ckpt "${OUT_DIR}/deit3_base_lora_blocks_3" \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name "deit3_base_lora-blocks_3-ft${FT_EPOCHS}"

# ============================================================
# Experiment 5: deit3_base KD CVLFace — head pruning 25%
# ============================================================
echo "========== deit3_base KD heads 0.25 =========="
python pruning.py \
    --ckpt_path "${DEIT3_BASE_KD}" \
    --model_name deit3_base_patch16_224.fb_in1k \
    --prune_method heads \
    --head_prune_ratio 0.25 \
    --finetune_epochs ${FT_EPOCHS} \
    --finetune_lr ${FT_LR} \
    --output_ckpt "${OUT_DIR}/deit3_base_kd_heads_0.25" \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name "deit3_base_kd-heads_0.25-ft${FT_EPOCHS}"

# ============================================================
# Experiment 6: swin_base baseline — head pruning 25%
# ============================================================
echo "========== swin_base heads 0.25 =========="
python pruning.py \
    --ckpt_path "${SWIN_BASE}" \
    --model_name swin_base_patch4_window7_224.ms_in1k \
    --prune_method heads \
    --head_prune_ratio 0.25 \
    --finetune_epochs ${FT_EPOCHS} \
    --finetune_lr ${FT_LR} \
    --output_ckpt "${OUT_DIR}/swin_base_heads_0.25" \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name "swin_base-heads_0.25-ft${FT_EPOCHS}"

# ============================================================
# Experiment 7: swin_base baseline — block pruning 25% (5 of 18 in stage 2)
# ============================================================
echo "========== swin_base blocks 5/18 =========="
python pruning.py \
    --ckpt_path "${SWIN_BASE}" \
    --model_name swin_base_patch4_window7_224.ms_in1k \
    --prune_method blocks \
    --num_blocks_to_remove 5 \
    --finetune_epochs ${FT_EPOCHS} \
    --finetune_lr ${FT_LR} \
    --output_ckpt "${OUT_DIR}/swin_base_blocks_5" \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name "swin_base-blocks_5-ft${FT_EPOCHS}"

# ============================================================
# Experiment 8: swin_tiny KD CVLFace — head pruning 10%
# ============================================================
echo "========== swin_tiny KD heads 0.10 =========="
python pruning.py \
    --ckpt_path "${SWIN_TINY_KD}" \
    --model_name swin_tiny_patch4_window7_224.ms_in1k \
    --prune_method heads \
    --head_prune_ratio 0.10 \
    --finetune_epochs ${FT_EPOCHS} \
    --finetune_lr ${FT_LR} \
    --output_ckpt "${OUT_DIR}/swin_tiny_kd_heads_0.10" \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name "swin_tiny_kd-heads_0.10-ft${FT_EPOCHS}"

echo ""
echo "All pruning experiments complete."
