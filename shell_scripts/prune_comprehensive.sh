#!/bin/bash
# Comprehensive pruning — 15 new experiments across all 4 architectures.
# 5 source checkpoints × 3 pruning configs (heads 10%, heads 25%, block dropping).
#
# Already completed (NOT re-run):
#   DeiT3-B Baseline (vimriwcl): heads 10%/25%/50%, blocks 3
#   Swin-B LoRA (ci53ufy8): heads 10%/25%, blocks 5

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate vit-benchmark
cd "$(dirname "$0")/.."

FT_EPOCHS=10
FT_LR=3e-5
PROJECT="vit_book_pruning_v2"
OUTDIR="pruned_models_v2"

mkdir -p "$OUTDIR"

run_prune() {
    local CKPT="$1"
    local MODEL="$2"
    local METHOD="$3"
    local NAME="$4"
    shift 4

    if [ -d "${OUTDIR}/${NAME}" ]; then
        echo "SKIP: ${NAME} already exists"
        return
    fi

    echo ""
    echo "========================================"
    echo "  ${NAME}"
    echo "========================================"
    python pruning.py --ckpt_path "$CKPT" --model_name "$MODEL" \
        --prune_method "$METHOD" \
        --finetune_epochs $FT_EPOCHS --finetune_lr $FT_LR \
        --output_ckpt "${OUTDIR}/${NAME}" \
        --wandb_project $PROJECT --wandb_run_name "$NAME" \
        "$@"
}

# ─── 1. Swin-T Baseline (p09btsx3, rank@1=0.534) ──────────────────────
CKPT="/mnt/data/wandb/checkpoints/vit_book_final_baselines/p09btsx3/epoch=22-rank1tinyface/eval/rank@1=0.541.ckpt"
MODEL="swin_tiny_patch4_window7_224.ms_in1k"

run_prune "$CKPT" "$MODEL" heads "swin_tiny_baseline-heads_0.10" --head_prune_ratio 0.10
run_prune "$CKPT" "$MODEL" heads "swin_tiny_baseline-heads_0.25" --head_prune_ratio 0.25
run_prune "$CKPT" "$MODEL" blocks "swin_tiny_baseline-blocks_2"  --num_blocks_to_remove 2

# ─── 2. Swin-T KD Best (zz76v2nu, α=0.0, rank@1=0.616) ───────────────
CKPT="/mnt/data/wandb/checkpoints/vit_book_kd_alpha_ablation/zz76v2nu/epoch=38-rank1tinyface/eval/rank@1=0.628.ckpt"

run_prune "$CKPT" "$MODEL" heads "swin_tiny_kd_best-heads_0.10" --head_prune_ratio 0.10
run_prune "$CKPT" "$MODEL" heads "swin_tiny_kd_best-heads_0.25" --head_prune_ratio 0.25
run_prune "$CKPT" "$MODEL" blocks "swin_tiny_kd_best-blocks_2"  --num_blocks_to_remove 2

# ─── 3. DeiT3-S Baseline (rdsdt7xb, rank@1=0.552) ────────────────────
CKPT="/mnt/data/wandb/checkpoints/vit_book_final_baselines/rdsdt7xb/epoch=47-rank1tinyface/eval/rank@1=0.562.ckpt"
MODEL="deit3_small_patch16_224.fb_in1k"

run_prune "$CKPT" "$MODEL" heads "deit3_small_baseline-heads_0.10" --head_prune_ratio 0.10
run_prune "$CKPT" "$MODEL" heads "deit3_small_baseline-heads_0.25" --head_prune_ratio 0.25
run_prune "$CKPT" "$MODEL" blocks "deit3_small_baseline-blocks_2"  --num_blocks_to_remove 2

# ─── 4. Swin-B Baseline (xd7hukv3, rank@1=0.526) ─────────────────────
CKPT="/mnt/data/wandb/checkpoints/vit_book_final_baselines/xd7hukv3/epoch=37-rank1tinyface/eval/rank@1=0.564.ckpt"
MODEL="swin_base_patch4_window7_224.ms_in1k"

run_prune "$CKPT" "$MODEL" heads "swin_base_baseline-heads_0.10" --head_prune_ratio 0.10
run_prune "$CKPT" "$MODEL" heads "swin_base_baseline-heads_0.25" --head_prune_ratio 0.25
run_prune "$CKPT" "$MODEL" blocks "swin_base_baseline-blocks_5"  --num_blocks_to_remove 5

# ─── 5. DeiT3-B KD Best (j8e97e1o, PETALface α=0.1, rank@1=0.593) ───
CKPT="/mnt/data/wandb/checkpoints/vit_book_kd_alpha_ablation/j8e97e1o/epoch=47-rank1tinyface/eval/rank@1=0.583.ckpt"
MODEL="deit3_base_patch16_224.fb_in1k"

run_prune "$CKPT" "$MODEL" heads "deit3_base_kd_best-heads_0.10" --head_prune_ratio 0.10
run_prune "$CKPT" "$MODEL" heads "deit3_base_kd_best-heads_0.25" --head_prune_ratio 0.25
run_prune "$CKPT" "$MODEL" blocks "deit3_base_kd_best-blocks_3"  --num_blocks_to_remove 3

echo ""
echo "========================================"
echo "All 15 pruning experiments complete."
echo "========================================"
