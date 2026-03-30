#!/bin/bash
# Run test-set inference on all 12 final models (6 LoRA + 6 baselines)
#
# Produces: inference_results/final_models_inference_results.csv
# Pattern follows: shell_scripts/model_ablation_inference.sh

set -uo pipefail

# Navigate to repo root if invoked from shell_scripts/
if [[ "$(dirname "$0")" == *"shell_scripts"* ]]; then
    cd "$(dirname "$0")/.."
fi

save_path="./inference_results/final_models_inference_results.csv"

# ──────────────────────────────────────────────
# LoRA models (vit_book_final_lora)
# ──────────────────────────────────────────────
model_list=(
    "--wandb_id f0frj93y --model_name swin_tiny_patch4_window7_224.ms_in1k --ckpt_path /mnt/data/wandb/checkpoints/vit_book_final_lora/f0frj93y/epoch=25-rank1tinyface/eval/rank@1=0.576.ckpt"
    "--wandb_id ymm0k0ub --model_name swin_small_patch4_window7_224.ms_in1k --ckpt_path /mnt/data/wandb/checkpoints/vit_book_final_lora/ymm0k0ub/epoch=45-rank1tinyface/eval/rank@1=0.576.ckpt"
    "--wandb_id ci53ufy8 --model_name swin_base_patch4_window7_224.ms_in1k --ckpt_path /mnt/data/wandb/checkpoints/vit_book_final_lora/ci53ufy8/epoch=39-rank1tinyface/eval/rank@1=0.556.ckpt"
    "--wandb_id cocm8b7d --model_name deit3_small_patch16_224.fb_in1k --ckpt_path /mnt/data/wandb/checkpoints/vit_book_final_lora/cocm8b7d/epoch=17-rank1tinyface/eval/rank@1=0.523.ckpt"
    "--wandb_id dpwn5kb1 --model_name deit3_medium_patch16_224.fb_in1k --ckpt_path /mnt/data/wandb/checkpoints/vit_book_final_lora/dpwn5kb1/epoch=26-rank1tinyface/eval/rank@1=0.547.ckpt"
    "--wandb_id 2ljjzfzb --model_name deit3_base_patch16_224.fb_in1k --ckpt_path /mnt/data/wandb/checkpoints/vit_book_final_lora/2ljjzfzb/epoch=8-rank1tinyface/eval/rank@1=0.550.ckpt"
)

# ──────────────────────────────────────────────
# Baseline models (vit_book_final_baselines)
# ──────────────────────────────────────────────
model_list+=(
    "--wandb_id p09btsx3 --model_name swin_tiny_patch4_window7_224.ms_in1k --ckpt_path /mnt/data/wandb/checkpoints/vit_book_final_baselines/p09btsx3/epoch=22-rank1tinyface/eval/rank@1=0.541.ckpt"
    "--wandb_id rbeovm8i --model_name swin_small_patch4_window7_224.ms_in1k --ckpt_path /mnt/data/wandb/checkpoints/vit_book_final_baselines/rbeovm8i/epoch=44-rank1tinyface/eval/rank@1=0.568.ckpt"
    "--wandb_id xd7hukv3 --model_name swin_base_patch4_window7_224.ms_in1k --ckpt_path /mnt/data/wandb/checkpoints/vit_book_final_baselines/xd7hukv3/epoch=37-rank1tinyface/eval/rank@1=0.564.ckpt"
    "--wandb_id rdsdt7xb --model_name deit3_small_patch16_224.fb_in1k --ckpt_path /mnt/data/wandb/checkpoints/vit_book_final_baselines/rdsdt7xb/epoch=47-rank1tinyface/eval/rank@1=0.562.ckpt"
    "--wandb_id k102lp3v --model_name deit3_medium_patch16_224.fb_in1k --ckpt_path /mnt/data/wandb/checkpoints/vit_book_final_baselines/k102lp3v/epoch=44-rank1tinyface/eval/rank@1=0.585.ckpt"
    "--wandb_id vimriwcl --model_name deit3_base_patch16_224.fb_in1k --ckpt_path /mnt/data/wandb/checkpoints/vit_book_final_baselines/vimriwcl/epoch=44-rank1tinyface/eval/rank@1=0.579.ckpt"
)

# ──────────────────────────────────────────────
# Run inference
# ──────────────────────────────────────────────
for model_args in "${model_list[@]}"; do
    # Extract ckpt_path to verify it exists
    ckpt=$(echo "$model_args" | grep -oP '(?<=--ckpt_path )\S+')
    if [ ! -f "$ckpt" ]; then
        echo "SKIP: checkpoint not found: $ckpt"
        continue
    fi

    echo "Running inference: $model_args"
    python inference.py ${model_args} --img_size 96 --batch_size 64 \
        --save_path "${save_path}"

    if [ $? -ne 0 ]; then
        echo "ERROR: inference failed for: $model_args"
        echo "  Continuing to next model..."
    fi
done

echo ""
echo "=== Inference complete ==="
echo "Results saved to: ${save_path}"
if [ -f "${save_path}" ]; then
    echo "Total rows (excluding header): $(tail -n +2 "${save_path}" | wc -l)"
fi
