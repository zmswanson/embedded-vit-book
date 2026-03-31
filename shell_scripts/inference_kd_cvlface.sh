#!/bin/bash
# Phase 2.3: Evaluate KD students + teacher baselines on TinyFace test set.
set -e

RESULTS_DIR="inference_results"
TEACHER_CSV="${RESULTS_DIR}/teacher_baselines.csv"
KD_CSV="${RESULTS_DIR}/kd_cvlface_results.csv"
CKPT_BASE="/mnt/data/wandb/checkpoints/vit_book_kd"
TEACHER_FT_CKPT="/mnt/data/wandb/checkpoints/vit_book_teacher_finetune/vkt8uhga/teacher_backbone.pt"

mkdir -p "$RESULTS_DIR"

# ── Teacher baselines ──────────────────────────────────────────────
echo "=== Teacher frozen ==="
python inference_teacher.py \
    --teacher_type cvlface_vit_base \
    --img_size 112 \
    --batch_size 64 \
    --rank_k 10 \
    --save_path "$TEACHER_CSV"

echo "=== Teacher fine-tuned ==="
python inference_teacher.py \
    --teacher_type cvlface_vit_base \
    --teacher_finetuned \
    --teacher_finetune_ckpt_path "$TEACHER_FT_CKPT" \
    --img_size 112 \
    --batch_size 64 \
    --rank_k 10 \
    --save_path "$TEACHER_CSV"

# ── Part A: CVLFace Frozen teacher → 12 student runs ──────────────
echo ""
echo "============================================"
echo "  Part A: Students trained with FROZEN teacher"
echo "============================================"

# swin_tiny Direct (frozen)
echo "=== [A] swin_tiny Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/y6ddjqyw/epoch=44-rank1tinyface/eval/rank@1=0.570.ckpt" \
    --model_name swin_tiny_patch4_window7_224.ms_in1k \
    --wandb_id y6ddjqyw \
    --save_path "$KD_CSV"

# swin_tiny LoRA (frozen)
echo "=== [A] swin_tiny LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/j5t9bwdw/epoch=2-rank1tinyface/eval/rank@1=0.380.ckpt" \
    --model_name swin_tiny_patch4_window7_224.ms_in1k \
    --wandb_id j5t9bwdw \
    --save_path "$KD_CSV"

# swin_small Direct (frozen)
echo "=== [A] swin_small Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/o8hvv51y/epoch=35-rank1tinyface/eval/rank@1=0.572.ckpt" \
    --model_name swin_small_patch4_window7_224.ms_in1k \
    --wandb_id o8hvv51y \
    --save_path "$KD_CSV"

# swin_small LoRA (frozen)
echo "=== [A] swin_small LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/oz4o3b3x/epoch=1-rank1tinyface/eval/rank@1=0.372.ckpt" \
    --model_name swin_small_patch4_window7_224.ms_in1k \
    --wandb_id oz4o3b3x \
    --save_path "$KD_CSV"

# swin_base Direct (frozen)
echo "=== [A] swin_base Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/i0fg9wl7/epoch=25-rank1tinyface/eval/rank@1=0.537.ckpt" \
    --model_name swin_base_patch4_window7_224.ms_in1k \
    --wandb_id i0fg9wl7 \
    --save_path "$KD_CSV"

# swin_base LoRA (frozen)
echo "=== [A] swin_base LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/fmfagyf4/epoch=0-rank1tinyface/eval/rank@1=0.357.ckpt" \
    --model_name swin_base_patch4_window7_224.ms_in1k \
    --wandb_id fmfagyf4 \
    --save_path "$KD_CSV"

# deit3_small Direct (frozen)
echo "=== [A] deit3_small Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/lkhiumos/epoch=45-rank1tinyface/eval/rank@1=0.545.ckpt" \
    --model_name deit3_small_patch16_224.fb_in1k \
    --wandb_id lkhiumos \
    --save_path "$KD_CSV"

# deit3_small LoRA (frozen)
echo "=== [A] deit3_small LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/wbo6pu5v/epoch=10-rank1tinyface/eval/rank@1=0.543.ckpt" \
    --model_name deit3_small_patch16_224.fb_in1k \
    --wandb_id wbo6pu5v \
    --save_path "$KD_CSV"

# deit3_medium Direct (frozen)
echo "=== [A] deit3_medium Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/6tkurhll/epoch=40-rank1tinyface/eval/rank@1=0.566.ckpt" \
    --model_name deit3_medium_patch16_224.fb_in1k \
    --wandb_id 6tkurhll \
    --save_path "$KD_CSV"

# deit3_medium LoRA (frozen)
echo "=== [A] deit3_medium LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/kfbdjiaz/epoch=24-rank1tinyface/eval/rank@1=0.541.ckpt" \
    --model_name deit3_medium_patch16_224.fb_in1k \
    --wandb_id kfbdjiaz \
    --save_path "$KD_CSV"

# deit3_base Direct (frozen)
echo "=== [A] deit3_base Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/f63auzxo/epoch=40-rank1tinyface/eval/rank@1=0.603.ckpt" \
    --model_name deit3_base_patch16_224.fb_in1k \
    --wandb_id f63auzxo \
    --save_path "$KD_CSV"

# deit3_base LoRA (frozen)
echo "=== [A] deit3_base LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/fz8pc2je/epoch=12-rank1tinyface/eval/rank@1=0.550.ckpt" \
    --model_name deit3_base_patch16_224.fb_in1k \
    --wandb_id fz8pc2je \
    --save_path "$KD_CSV"

# ── Part B: CVLFace Fine-tuned teacher → 12 student runs ─────────
echo ""
echo "============================================"
echo "  Part B: Students trained with FINE-TUNED teacher"
echo "============================================"

# swin_tiny Direct (finetuned)
echo "=== [B] swin_tiny Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/5wabjm9e/epoch=22-rank1tinyface/eval/rank@1=0.558.ckpt" \
    --model_name swin_tiny_patch4_window7_224.ms_in1k \
    --wandb_id 5wabjm9e \
    --save_path "$KD_CSV"

# swin_tiny LoRA (finetuned)
echo "=== [B] swin_tiny LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/s2ceip2w/epoch=1-rank1tinyface/eval/rank@1=0.376.ckpt" \
    --model_name swin_tiny_patch4_window7_224.ms_in1k \
    --wandb_id s2ceip2w \
    --save_path "$KD_CSV"

# swin_small Direct (finetuned)
echo "=== [B] swin_small Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/sxo4xwy2/epoch=36-rank1tinyface/eval/rank@1=0.548.ckpt" \
    --model_name swin_small_patch4_window7_224.ms_in1k \
    --wandb_id sxo4xwy2 \
    --save_path "$KD_CSV"

# swin_small LoRA (finetuned)
echo "=== [B] swin_small LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/ybx1z1y5/epoch=36-rank1tinyface/eval/rank@1=0.591.ckpt" \
    --model_name swin_small_patch4_window7_224.ms_in1k \
    --wandb_id ybx1z1y5 \
    --save_path "$KD_CSV"

# swin_base Direct (finetuned)
echo "=== [B] swin_base Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/jh5v6ltc/epoch=24-rank1tinyface/eval/rank@1=0.560.ckpt" \
    --model_name swin_base_patch4_window7_224.ms_in1k \
    --wandb_id jh5v6ltc \
    --save_path "$KD_CSV"

# swin_base LoRA (finetuned)
echo "=== [B] swin_base LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/i8c854l4/epoch=0-rank1tinyface/eval/rank@1=0.353.ckpt" \
    --model_name swin_base_patch4_window7_224.ms_in1k \
    --wandb_id i8c854l4 \
    --save_path "$KD_CSV"

# deit3_small Direct (finetuned)
echo "=== [B] deit3_small Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/nz4gtmdt/epoch=40-rank1tinyface/eval/rank@1=0.566.ckpt" \
    --model_name deit3_small_patch16_224.fb_in1k \
    --wandb_id nz4gtmdt \
    --save_path "$KD_CSV"

# deit3_small LoRA (finetuned)
echo "=== [B] deit3_small LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/vsypj6cs/epoch=42-rank1tinyface/eval/rank@1=0.548.ckpt" \
    --model_name deit3_small_patch16_224.fb_in1k \
    --wandb_id vsypj6cs \
    --save_path "$KD_CSV"

# deit3_medium Direct (finetuned)
echo "=== [B] deit3_medium Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/zhoquhp7/epoch=41-rank1tinyface/eval/rank@1=0.585.ckpt" \
    --model_name deit3_medium_patch16_224.fb_in1k \
    --wandb_id zhoquhp7 \
    --save_path "$KD_CSV"

# deit3_medium LoRA (finetuned)
echo "=== [B] deit3_medium LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/0gt8dwde/epoch=19-rank1tinyface/eval/rank@1=0.533.ckpt" \
    --model_name deit3_medium_patch16_224.fb_in1k \
    --wandb_id 0gt8dwde \
    --save_path "$KD_CSV"

# deit3_base Direct (finetuned)
echo "=== [B] deit3_base Direct ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/5mngq5jn/epoch=47-rank1tinyface/eval/rank@1=0.579.ckpt" \
    --model_name deit3_base_patch16_224.fb_in1k \
    --wandb_id 5mngq5jn \
    --save_path "$KD_CSV"

# deit3_base LoRA (finetuned)
echo "=== [B] deit3_base LoRA ==="
python inference.py \
    --ckpt_path "${CKPT_BASE}/3d7j4ck4/epoch=14-rank1tinyface/eval/rank@1=0.545.ckpt" \
    --model_name deit3_base_patch16_224.fb_in1k \
    --wandb_id 3d7j4ck4 \
    --save_path "$KD_CSV"

echo ""
echo "============================================"
echo "  All evaluations complete!"
echo "  Teacher results: ${TEACHER_CSV}"
echo "  Student KD results: ${KD_CSV}"
echo "============================================"
