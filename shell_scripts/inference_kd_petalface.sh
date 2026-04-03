#!/bin/bash
# Evaluate 12 PETALface KD Swin students on TinyFace test set.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="inference_results"
KD_CSV="${RESULTS_DIR}/kd_petalface_results.csv"
CKPT_BASE="/mnt/data/wandb/checkpoints/vit_book_kd"

mkdir -p "$RESULTS_DIR"

# ── Part C: PETALface Frozen teacher → 6 Swin student runs ───────
echo ""
echo "============================================"
echo "  Part C: Swin students trained with FROZEN PETALface teacher"
echo "============================================"

# swin_tiny Direct (frozen)
echo "=== [C] swin_tiny Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/efcezdv2/epoch=43-rank1tinyface/eval/rank@1=0.578.ckpt" \
    --model_name swin_tiny_patch4_window7_224.ms_in1k \
    --wandb_id efcezdv2 \
    --save_path "$KD_CSV"

# swin_tiny LoRA (frozen)
echo "=== [C] swin_tiny LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/vnci86wd/epoch=22-rank1tinyface/eval/rank@1=0.568.ckpt" \
    --model_name swin_tiny_patch4_window7_224.ms_in1k \
    --wandb_id vnci86wd \
    --save_path "$KD_CSV"

# swin_small Direct (frozen)
echo "=== [C] swin_small Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/nfh0d3da/epoch=35-rank1tinyface/eval/rank@1=0.570.ckpt" \
    --model_name swin_small_patch4_window7_224.ms_in1k \
    --wandb_id nfh0d3da \
    --save_path "$KD_CSV"

# swin_small LoRA (frozen)
echo "=== [C] swin_small LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/o1amgdw8/epoch=3-rank1tinyface/eval/rank@1=0.397.ckpt" \
    --model_name swin_small_patch4_window7_224.ms_in1k \
    --wandb_id o1amgdw8 \
    --save_path "$KD_CSV"

# swin_base Direct (frozen)
echo "=== [C] swin_base Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/2vzzetyw/epoch=32-rank1tinyface/eval/rank@1=0.545.ckpt" \
    --model_name swin_base_patch4_window7_224.ms_in1k \
    --wandb_id 2vzzetyw \
    --save_path "$KD_CSV"

# swin_base LoRA (frozen)
echo "=== [C] swin_base LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/dn3t16sl/epoch=37-rank1tinyface/eval/rank@1=0.512.ckpt" \
    --model_name swin_base_patch4_window7_224.ms_in1k \
    --wandb_id dn3t16sl \
    --save_path "$KD_CSV"

# ── Part D: PETALface Fine-tuned teacher → 6 Swin student runs ───
echo ""
echo "============================================"
echo "  Part D: Swin students trained with FINE-TUNED PETALface teacher"
echo "============================================"

# swin_tiny Direct (finetuned)
echo "=== [D] swin_tiny Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/tu0284v9/epoch=32-rank1tinyface/eval/rank@1=0.576.ckpt" \
    --model_name swin_tiny_patch4_window7_224.ms_in1k \
    --wandb_id tu0284v9 \
    --save_path "$KD_CSV"

# swin_tiny LoRA (finetuned)
echo "=== [D] swin_tiny LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/nvjb4zbr/epoch=29-rank1tinyface/eval/rank@1=0.570.ckpt" \
    --model_name swin_tiny_patch4_window7_224.ms_in1k \
    --wandb_id nvjb4zbr \
    --save_path "$KD_CSV"

# swin_small Direct (finetuned)
echo "=== [D] swin_small Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/ek2ut8eu/epoch=31-rank1tinyface/eval/rank@1=0.578.ckpt" \
    --model_name swin_small_patch4_window7_224.ms_in1k \
    --wandb_id ek2ut8eu \
    --save_path "$KD_CSV"

# swin_small LoRA (finetuned)
echo "=== [D] swin_small LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/hcalbado/epoch=3-rank1tinyface/eval/rank@1=0.395.ckpt" \
    --model_name swin_small_patch4_window7_224.ms_in1k \
    --wandb_id hcalbado \
    --save_path "$KD_CSV"

# swin_base Direct (finetuned)
echo "=== [D] swin_base Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/7t2r820f/epoch=39-rank1tinyface/eval/rank@1=0.550.ckpt" \
    --model_name swin_base_patch4_window7_224.ms_in1k \
    --wandb_id 7t2r820f \
    --save_path "$KD_CSV"

# swin_base LoRA (finetuned)
echo "=== [D] swin_base LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/vf34oqq7/epoch=0-rank1tinyface/eval/rank@1=0.333.ckpt" \
    --model_name swin_base_patch4_window7_224.ms_in1k \
    --wandb_id vf34oqq7 \
    --save_path "$KD_CSV"

echo ""
echo "============================================"
echo "  All PETALface KD evaluations complete!"
echo "  Results: ${KD_CSV}"
echo "============================================"
