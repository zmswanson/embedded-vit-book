#!/bin/bash
# Phase 3.2: Batch export all trained model variants to ONNX format.
#
# Exports backbone-only models (no classification head) using export_onnx.py.
# KD checkpoints are auto-detected; only the student backbone is exported.
# LoRA weights are merged automatically before export.
#
# Output: onnx_models/{baselines,lora,kd_cvlface,kd_petalface}/*.onnx

set -uo pipefail

# Navigate to repo root if invoked from shell_scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vit-benchmark

mkdir -p onnx_models/{baselines,lora,kd_cvlface,kd_petalface}

CKPT_BASELINES="/mnt/data/wandb/checkpoints/vit_book_final_baselines"
CKPT_LORA="/mnt/data/wandb/checkpoints/vit_book_final_lora"
CKPT_KD="/mnt/data/wandb/checkpoints/vit_book_kd"
CKPT_ALPHA="/mnt/data/wandb/checkpoints/vit_book_kd_alpha_ablation"

PASSED=0
FAILED=0
SKIPPED=0

export_model() {
    local ckpt="$1"
    local model_name="$2"
    local output="$3"

    if [ ! -f "$ckpt" ]; then
        echo "  SKIP: checkpoint not found: $ckpt"
        ((SKIPPED++))
        return
    fi

    echo "  Exporting: ${output}"
    if python export_onnx.py \
        --ckpt_path "$ckpt" \
        --model_name "$model_name" \
        --output_path "$output" \
        --img_size 96; then
        # Show size including external data file
        local total_size
        total_size=$(du -ch "$output" "${output}.data" 2>/dev/null | tail -1 | cut -f1)
        echo "  ✓ Done: ${total_size}"
        ((PASSED++))
    else
        echo "  ✗ FAILED: ${output}"
        ((FAILED++))
    fi
    echo ""
}

# ══════════════════════════════════════════════════════════════════
# Baselines (6 models — from vit_book_final_baselines)
# ══════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════"
echo "  Baselines (6 models)"
echo "═══════════════════════════════════════════"

export_model \
    "${CKPT_BASELINES}/p09btsx3/epoch=22-rank1tinyface/eval/rank@1=0.541.ckpt" \
    "swin_tiny_patch4_window7_224.ms_in1k" \
    "onnx_models/baselines/swin_tiny.onnx"

export_model \
    "${CKPT_BASELINES}/rbeovm8i/epoch=44-rank1tinyface/eval/rank@1=0.568.ckpt" \
    "swin_small_patch4_window7_224.ms_in1k" \
    "onnx_models/baselines/swin_small.onnx"

export_model \
    "${CKPT_BASELINES}/xd7hukv3/epoch=37-rank1tinyface/eval/rank@1=0.564.ckpt" \
    "swin_base_patch4_window7_224.ms_in1k" \
    "onnx_models/baselines/swin_base.onnx"

export_model \
    "${CKPT_BASELINES}/rdsdt7xb/epoch=47-rank1tinyface/eval/rank@1=0.562.ckpt" \
    "deit3_small_patch16_224.fb_in1k" \
    "onnx_models/baselines/deit3_small.onnx"

export_model \
    "${CKPT_BASELINES}/k102lp3v/epoch=44-rank1tinyface/eval/rank@1=0.585.ckpt" \
    "deit3_medium_patch16_224.fb_in1k" \
    "onnx_models/baselines/deit3_medium.onnx"

export_model \
    "${CKPT_BASELINES}/vimriwcl/epoch=44-rank1tinyface/eval/rank@1=0.579.ckpt" \
    "deit3_base_patch16_224.fb_in1k" \
    "onnx_models/baselines/deit3_base.onnx"

# ══════════════════════════════════════════════════════════════════
# LoRA models (6 models — from vit_book_final_lora)
# ══════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════"
echo "  LoRA (6 models)"
echo "═══════════════════════════════════════════"

export_model \
    "${CKPT_LORA}/f0frj93y/epoch=25-rank1tinyface/eval/rank@1=0.576.ckpt" \
    "swin_tiny_patch4_window7_224.ms_in1k" \
    "onnx_models/lora/swin_tiny_lora.onnx"

export_model \
    "${CKPT_LORA}/ymm0k0ub/epoch=45-rank1tinyface/eval/rank@1=0.576.ckpt" \
    "swin_small_patch4_window7_224.ms_in1k" \
    "onnx_models/lora/swin_small_lora.onnx"

export_model \
    "${CKPT_LORA}/ci53ufy8/epoch=39-rank1tinyface/eval/rank@1=0.556.ckpt" \
    "swin_base_patch4_window7_224.ms_in1k" \
    "onnx_models/lora/swin_base_lora.onnx"

export_model \
    "${CKPT_LORA}/cocm8b7d/epoch=17-rank1tinyface/eval/rank@1=0.523.ckpt" \
    "deit3_small_patch16_224.fb_in1k" \
    "onnx_models/lora/deit3_small_lora.onnx"

export_model \
    "${CKPT_LORA}/dpwn5kb1/epoch=26-rank1tinyface/eval/rank@1=0.547.ckpt" \
    "deit3_medium_patch16_224.fb_in1k" \
    "onnx_models/lora/deit3_medium_lora.onnx"

export_model \
    "${CKPT_LORA}/2ljjzfzb/epoch=8-rank1tinyface/eval/rank@1=0.550.ckpt" \
    "deit3_base_patch16_224.fb_in1k" \
    "onnx_models/lora/deit3_base_lora.onnx"

# ══════════════════════════════════════════════════════════════════
# KD — CVLFace teacher (24 models: 12 frozen + 12 fine-tuned)
# Plus 2 best alpha ablation models
# ══════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════"
echo "  KD — CVLFace (frozen teacher, 12 students)"
echo "═══════════════════════════════════════════"

# --- Part A: CVLFace Frozen teacher, Direct training ---
export_model \
    "${CKPT_KD}/y6ddjqyw/epoch=44-rank1tinyface/eval/rank@1=0.570.ckpt" \
    "swin_tiny_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_cvlface/swin_tiny_kd_cvlface_frozen.onnx"

export_model \
    "${CKPT_KD}/o8hvv51y/epoch=35-rank1tinyface/eval/rank@1=0.572.ckpt" \
    "swin_small_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_cvlface/swin_small_kd_cvlface_frozen.onnx"

export_model \
    "${CKPT_KD}/i0fg9wl7/epoch=25-rank1tinyface/eval/rank@1=0.537.ckpt" \
    "swin_base_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_cvlface/swin_base_kd_cvlface_frozen.onnx"

export_model \
    "${CKPT_KD}/lkhiumos/epoch=45-rank1tinyface/eval/rank@1=0.545.ckpt" \
    "deit3_small_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_small_kd_cvlface_frozen.onnx"

export_model \
    "${CKPT_KD}/6tkurhll/epoch=40-rank1tinyface/eval/rank@1=0.566.ckpt" \
    "deit3_medium_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_medium_kd_cvlface_frozen.onnx"

export_model \
    "${CKPT_KD}/f63auzxo/epoch=40-rank1tinyface/eval/rank@1=0.603.ckpt" \
    "deit3_base_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_base_kd_cvlface_frozen.onnx"

# --- Part A: CVLFace Frozen teacher, LoRA training ---
export_model \
    "${CKPT_KD}/j5t9bwdw/epoch=2-rank1tinyface/eval/rank@1=0.380.ckpt" \
    "swin_tiny_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_cvlface/swin_tiny_kd_cvlface_frozen_lora.onnx"

export_model \
    "${CKPT_KD}/oz4o3b3x/epoch=1-rank1tinyface/eval/rank@1=0.372.ckpt" \
    "swin_small_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_cvlface/swin_small_kd_cvlface_frozen_lora.onnx"

export_model \
    "${CKPT_KD}/fmfagyf4/epoch=0-rank1tinyface/eval/rank@1=0.357.ckpt" \
    "swin_base_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_cvlface/swin_base_kd_cvlface_frozen_lora.onnx"

export_model \
    "${CKPT_KD}/wbo6pu5v/epoch=10-rank1tinyface/eval/rank@1=0.543.ckpt" \
    "deit3_small_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_small_kd_cvlface_frozen_lora.onnx"

export_model \
    "${CKPT_KD}/kfbdjiaz/epoch=24-rank1tinyface/eval/rank@1=0.541.ckpt" \
    "deit3_medium_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_medium_kd_cvlface_frozen_lora.onnx"

export_model \
    "${CKPT_KD}/fz8pc2je/epoch=12-rank1tinyface/eval/rank@1=0.550.ckpt" \
    "deit3_base_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_base_kd_cvlface_frozen_lora.onnx"

echo "═══════════════════════════════════════════"
echo "  KD — CVLFace (fine-tuned teacher, 12 students)"
echo "═══════════════════════════════════════════"

# --- Part B: CVLFace Fine-tuned teacher, Direct training ---
export_model \
    "${CKPT_KD}/5wabjm9e/epoch=22-rank1tinyface/eval/rank@1=0.558.ckpt" \
    "swin_tiny_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_cvlface/swin_tiny_kd_cvlface_ft.onnx"

export_model \
    "${CKPT_KD}/sxo4xwy2/epoch=36-rank1tinyface/eval/rank@1=0.548.ckpt" \
    "swin_small_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_cvlface/swin_small_kd_cvlface_ft.onnx"

export_model \
    "${CKPT_KD}/jh5v6ltc/epoch=24-rank1tinyface/eval/rank@1=0.560.ckpt" \
    "swin_base_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_cvlface/swin_base_kd_cvlface_ft.onnx"

export_model \
    "${CKPT_KD}/nz4gtmdt/epoch=40-rank1tinyface/eval/rank@1=0.566.ckpt" \
    "deit3_small_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_small_kd_cvlface_ft.onnx"

export_model \
    "${CKPT_KD}/zhoquhp7/epoch=41-rank1tinyface/eval/rank@1=0.585.ckpt" \
    "deit3_medium_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_medium_kd_cvlface_ft.onnx"

export_model \
    "${CKPT_KD}/5mngq5jn/epoch=47-rank1tinyface/eval/rank@1=0.579.ckpt" \
    "deit3_base_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_base_kd_cvlface_ft.onnx"

# --- Part B: CVLFace Fine-tuned teacher, LoRA training ---
export_model \
    "${CKPT_KD}/s2ceip2w/epoch=1-rank1tinyface/eval/rank@1=0.376.ckpt" \
    "swin_tiny_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_cvlface/swin_tiny_kd_cvlface_ft_lora.onnx"

export_model \
    "${CKPT_KD}/ybx1z1y5/epoch=36-rank1tinyface/eval/rank@1=0.591.ckpt" \
    "swin_small_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_cvlface/swin_small_kd_cvlface_ft_lora.onnx"

export_model \
    "${CKPT_KD}/i8c854l4/epoch=0-rank1tinyface/eval/rank@1=0.353.ckpt" \
    "swin_base_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_cvlface/swin_base_kd_cvlface_ft_lora.onnx"

export_model \
    "${CKPT_KD}/vsypj6cs/epoch=42-rank1tinyface/eval/rank@1=0.548.ckpt" \
    "deit3_small_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_small_kd_cvlface_ft_lora.onnx"

export_model \
    "${CKPT_KD}/0gt8dwde/epoch=19-rank1tinyface/eval/rank@1=0.533.ckpt" \
    "deit3_medium_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_medium_kd_cvlface_ft_lora.onnx"

export_model \
    "${CKPT_KD}/3d7j4ck4/epoch=14-rank1tinyface/eval/rank@1=0.545.ckpt" \
    "deit3_base_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_base_kd_cvlface_ft_lora.onnx"

# --- Alpha ablation best models (CVLFace) ---
echo "═══════════════════════════════════════════"
echo "  KD — CVLFace alpha ablation (best models)"
echo "═══════════════════════════════════════════"

# Best: Swin-T + CVLFace α=0.0 → 0.616 rank@1
export_model \
    "${CKPT_ALPHA}/zz76v2nu/epoch=38-rank1tinyface/eval/rank@1=0.628.ckpt" \
    "swin_tiny_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_cvlface/swin_tiny_kd_cvlface_alpha0.0.onnx"

# DeiT3-B + CVLFace α=0.0
export_model \
    "${CKPT_ALPHA}/4uwgz0ln/epoch=45-rank1tinyface/eval/rank@1=0.570.ckpt" \
    "deit3_base_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_base_kd_cvlface_alpha0.0.onnx"

# DeiT3-B + CVLFace α=0.8 (best CVL+DeiT3)
export_model \
    "${CKPT_ALPHA}/0hodmg2y/epoch=42-rank1tinyface/eval/rank@1=0.601.ckpt" \
    "deit3_base_patch16_224.fb_in1k" \
    "onnx_models/kd_cvlface/deit3_base_kd_cvlface_alpha0.8.onnx"

# ══════════════════════════════════════════════════════════════════
# KD — PETALface teacher (12 models: 6 frozen + 6 fine-tuned, Swin only)
# Plus 2 best alpha ablation models
# ══════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════"
echo "  KD — PETALface (frozen teacher, 6 Swin students)"
echo "═══════════════════════════════════════════"

# --- Part C: PETALface Frozen teacher, Direct ---
export_model \
    "${CKPT_KD}/efcezdv2/epoch=43-rank1tinyface/eval/rank@1=0.578.ckpt" \
    "swin_tiny_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_petalface/swin_tiny_kd_petalface_frozen.onnx"

export_model \
    "${CKPT_KD}/nfh0d3da/epoch=35-rank1tinyface/eval/rank@1=0.570.ckpt" \
    "swin_small_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_petalface/swin_small_kd_petalface_frozen.onnx"

export_model \
    "${CKPT_KD}/2vzzetyw/epoch=32-rank1tinyface/eval/rank@1=0.545.ckpt" \
    "swin_base_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_petalface/swin_base_kd_petalface_frozen.onnx"

# --- Part C: PETALface Frozen teacher, LoRA ---
export_model \
    "${CKPT_KD}/vnci86wd/epoch=22-rank1tinyface/eval/rank@1=0.568.ckpt" \
    "swin_tiny_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_petalface/swin_tiny_kd_petalface_frozen_lora.onnx"

export_model \
    "${CKPT_KD}/o1amgdw8/epoch=3-rank1tinyface/eval/rank@1=0.397.ckpt" \
    "swin_small_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_petalface/swin_small_kd_petalface_frozen_lora.onnx"

export_model \
    "${CKPT_KD}/dn3t16sl/epoch=37-rank1tinyface/eval/rank@1=0.512.ckpt" \
    "swin_base_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_petalface/swin_base_kd_petalface_frozen_lora.onnx"

echo "═══════════════════════════════════════════"
echo "  KD — PETALface (fine-tuned teacher, 6 Swin students)"
echo "═══════════════════════════════════════════"

# --- Part D: PETALface Fine-tuned teacher, Direct ---
export_model \
    "${CKPT_KD}/tu0284v9/epoch=32-rank1tinyface/eval/rank@1=0.576.ckpt" \
    "swin_tiny_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_petalface/swin_tiny_kd_petalface_ft.onnx"

export_model \
    "${CKPT_KD}/ek2ut8eu/epoch=31-rank1tinyface/eval/rank@1=0.578.ckpt" \
    "swin_small_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_petalface/swin_small_kd_petalface_ft.onnx"

export_model \
    "${CKPT_KD}/7t2r820f/epoch=39-rank1tinyface/eval/rank@1=0.550.ckpt" \
    "swin_base_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_petalface/swin_base_kd_petalface_ft.onnx"

# --- Part D: PETALface Fine-tuned teacher, LoRA ---
export_model \
    "${CKPT_KD}/nvjb4zbr/epoch=29-rank1tinyface/eval/rank@1=0.570.ckpt" \
    "swin_tiny_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_petalface/swin_tiny_kd_petalface_ft_lora.onnx"

export_model \
    "${CKPT_KD}/hcalbado/epoch=3-rank1tinyface/eval/rank@1=0.395.ckpt" \
    "swin_small_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_petalface/swin_small_kd_petalface_ft_lora.onnx"

export_model \
    "${CKPT_KD}/vf34oqq7/epoch=0-rank1tinyface/eval/rank@1=0.333.ckpt" \
    "swin_base_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_petalface/swin_base_kd_petalface_ft_lora.onnx"

# --- Alpha ablation best models (PETALface) ---
echo "═══════════════════════════════════════════"
echo "  KD — PETALface alpha ablation (best models)"
echo "═══════════════════════════════════════════"

# Best: Swin-T + PETALface α=0.3 → 0.591 rank@1
export_model \
    "${CKPT_ALPHA}/3gq0pe9h/epoch=25-rank1tinyface/eval/rank@1=0.591.ckpt" \
    "swin_tiny_patch4_window7_224.ms_in1k" \
    "onnx_models/kd_petalface/swin_tiny_kd_petalface_alpha0.3.onnx"

# Best: DeiT3-B + PETALface α=0.1 → 0.603 rank@1
export_model \
    "${CKPT_ALPHA}/obc9cvpu/epoch=46-rank1tinyface/eval/rank@1=0.603.ckpt" \
    "deit3_base_patch16_224.fb_in1k" \
    "onnx_models/kd_petalface/deit3_base_kd_petalface_alpha0.1.onnx"

# Best: DeiT3-B + PETALface α=0.7 → 0.595 rank@1
export_model \
    "${CKPT_ALPHA}/mmywymd2/epoch=42-rank1tinyface/eval/rank@1=0.595.ckpt" \
    "deit3_base_patch16_224.fb_in1k" \
    "onnx_models/kd_petalface/deit3_base_kd_petalface_alpha0.7.onnx"

# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════"
echo "  Export Summary"
echo "═══════════════════════════════════════════"
echo "  Passed:  ${PASSED}"
echo "  Failed:  ${FAILED}"
echo "  Skipped: ${SKIPPED}"
echo ""

if [ -d onnx_models ]; then
    echo "ONNX files by category:"
    for dir in onnx_models/*/; do
        count=$(find "$dir" -name "*.onnx" 2>/dev/null | wc -l)
        total=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  $(basename "$dir"): ${count} models (${total})"
    done
    echo ""
    echo "Total ONNX files: $(find onnx_models -name '*.onnx' | wc -l)"
    echo "Total size: $(du -sh onnx_models | cut -f1)"
fi
