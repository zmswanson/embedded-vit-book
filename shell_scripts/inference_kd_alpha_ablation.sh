#!/bin/bash
# Evaluate alpha ablation checkpoints (32 runs)
# Training completed: 32/32 runs successful

set -uo pipefail

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vit-benchmark

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

save_path="./inference_results/kd_alpha_ablation_results.csv"
CKPT_BASE="/mnt/data/wandb/checkpoints/vit_book_kd_alpha_ablation"

mkdir -p inference_results

SWIN_MODEL="swin_tiny_patch4_window7_224.ms_in1k"
DEIT3_MODEL="deit3_base_patch16_224.fb_in1k"

# ── Config A: Swin + CVLFace (8 runs) ────────────────────────────
echo "=== Config A: Swin + CVLFace ==="

echo "[A] alpha=0.0"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/zz76v2nu/epoch=38-rank1tinyface/eval/rank@1=0.628.ckpt" \
    --wandb_id zz76v2nu --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[A] alpha=0.1"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/kpmq1v0l/epoch=41-rank1tinyface/eval/rank@1=0.576.ckpt" \
    --wandb_id kpmq1v0l --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[A] alpha=0.3"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/i60x81gy/epoch=30-rank1tinyface/eval/rank@1=0.562.ckpt" \
    --wandb_id i60x81gy --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[A] alpha=0.5"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/fyp68bza/epoch=34-rank1tinyface/eval/rank@1=0.558.ckpt" \
    --wandb_id fyp68bza --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[A] alpha=0.7"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/3m21a7os/epoch=44-rank1tinyface/eval/rank@1=0.570.ckpt" \
    --wandb_id 3m21a7os --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[A] alpha=0.8"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/dm9j6n39/epoch=42-rank1tinyface/eval/rank@1=0.554.ckpt" \
    --wandb_id dm9j6n39 --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[A] alpha=0.9"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/7d05whai/epoch=34-rank1tinyface/eval/rank@1=0.574.ckpt" \
    --wandb_id 7d05whai --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[A] alpha=1.0"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/qgskd48h/epoch=21-rank1tinyface/eval/rank@1=0.552.ckpt" \
    --wandb_id qgskd48h --img_size 96 --batch_size 64 --save_path "${save_path}"

# ── Config B: Swin + PETALface (8 runs) ──────────────────────────
echo ""
echo "=== Config B: Swin + PETALface ==="

echo "[B] alpha=0.0"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/zicmj3j9/epoch=34-rank1tinyface/eval/rank@1=0.415.ckpt" \
    --wandb_id zicmj3j9 --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[B] alpha=0.1"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/fnkz16nn/epoch=26-rank1tinyface/eval/rank@1=0.578.ckpt" \
    --wandb_id fnkz16nn --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[B] alpha=0.3"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/3gq0pe9h/epoch=25-rank1tinyface/eval/rank@1=0.591.ckpt" \
    --wandb_id 3gq0pe9h --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[B] alpha=0.5"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/nr3zjr55/epoch=21-rank1tinyface/eval/rank@1=0.556.ckpt" \
    --wandb_id nr3zjr55 --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[B] alpha=0.7"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/lvkbtfaf/epoch=22-rank1tinyface/eval/rank@1=0.568.ckpt" \
    --wandb_id lvkbtfaf --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[B] alpha=0.8"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/it6aidvb/epoch=32-rank1tinyface/eval/rank@1=0.574.ckpt" \
    --wandb_id it6aidvb --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[B] alpha=0.9"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/qtgsq08g/epoch=34-rank1tinyface/eval/rank@1=0.572.ckpt" \
    --wandb_id qtgsq08g --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[B] alpha=1.0"
python inference.py --model_name "$SWIN_MODEL" \
    --ckpt_path "${CKPT_BASE}/y5jnqnsw/epoch=27-rank1tinyface/eval/rank@1=0.578.ckpt" \
    --wandb_id y5jnqnsw --img_size 96 --batch_size 64 --save_path "${save_path}"

# ── Config C: DeiT3 + CVLFace (8 runs) ───────────────────────────
echo ""
echo "=== Config C: DeiT3 + CVLFace ==="

echo "[C] alpha=0.0"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/4uwgz0ln/epoch=45-rank1tinyface/eval/rank@1=0.570.ckpt" \
    --wandb_id 4uwgz0ln --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[C] alpha=0.1"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/j8e97e1o/epoch=47-rank1tinyface/eval/rank@1=0.583.ckpt" \
    --wandb_id j8e97e1o --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[C] alpha=0.3"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/733slwmw/epoch=49-rank1tinyface/eval/rank@1=0.587.ckpt" \
    --wandb_id 733slwmw --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[C] alpha=0.5"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/ke5n9fpd/epoch=49-rank1tinyface/eval/rank@1=0.578.ckpt" \
    --wandb_id ke5n9fpd --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[C] alpha=0.7"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/nkj1775d/epoch=47-rank1tinyface/eval/rank@1=0.579.ckpt" \
    --wandb_id nkj1775d --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[C] alpha=0.8"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/0hodmg2y/epoch=42-rank1tinyface/eval/rank@1=0.601.ckpt" \
    --wandb_id 0hodmg2y --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[C] alpha=0.9"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/qbxp3dql/epoch=43-rank1tinyface/eval/rank@1=0.581.ckpt" \
    --wandb_id qbxp3dql --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[C] alpha=1.0"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/csbzjwt6/epoch=48-rank1tinyface/eval/rank@1=0.583.ckpt" \
    --wandb_id csbzjwt6 --img_size 96 --batch_size 64 --save_path "${save_path}"

# ── Config D: DeiT3 + PETALface (8 runs) ─────────────────────────
echo ""
echo "=== Config D: DeiT3 + PETALface ==="

echo "[D] alpha=0.0"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/l9dfl98a/epoch=46-rank1tinyface/eval/rank@1=0.376.ckpt" \
    --wandb_id l9dfl98a --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[D] alpha=0.1"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/obc9cvpu/epoch=46-rank1tinyface/eval/rank@1=0.603.ckpt" \
    --wandb_id obc9cvpu --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[D] alpha=0.3"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/sc1im6mh/epoch=43-rank1tinyface/eval/rank@1=0.583.ckpt" \
    --wandb_id sc1im6mh --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[D] alpha=0.5"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/wf5djsia/epoch=39-rank1tinyface/eval/rank@1=0.587.ckpt" \
    --wandb_id wf5djsia --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[D] alpha=0.7"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/mmywymd2/epoch=42-rank1tinyface/eval/rank@1=0.595.ckpt" \
    --wandb_id mmywymd2 --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[D] alpha=0.8"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/rqmojnr5/epoch=47-rank1tinyface/eval/rank@1=0.583.ckpt" \
    --wandb_id rqmojnr5 --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[D] alpha=0.9"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/4w0obh1y/epoch=45-rank1tinyface/eval/rank@1=0.587.ckpt" \
    --wandb_id 4w0obh1y --img_size 96 --batch_size 64 --save_path "${save_path}"

echo "[D] alpha=1.0"
python inference.py --model_name "$DEIT3_MODEL" \
    --ckpt_path "${CKPT_BASE}/jewthe02/epoch=48-rank1tinyface/eval/rank@1=0.579.ckpt" \
    --wandb_id jewthe02 --img_size 96 --batch_size 64 --save_path "${save_path}"

echo ""
echo "=== All 32 inference runs complete ==="
echo "Results saved to: ${save_path}"
