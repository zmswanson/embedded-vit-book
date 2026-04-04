#!/bin/bash
# Export ablation study models (ConvNeXt, PVTv2, MobileViTv2, ResNet) to ONNX.
# These were trained during Phase 1 model ablation but never exported.
#
# Output: onnx_models/baselines/{convnext_*,pvt_v2_*,mobilevitv2_*,resnet*}.onnx

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

eval "$(conda shell.bash hook)"
conda activate vit-benchmark

mkdir -p onnx_models/baselines

CKPT_ABLATION="/mnt/data/wandb/checkpoints/model_ablation_study"

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

echo "═══════════════════════════════════════════"
echo "  Ablation models (10 models)"
echo "═══════════════════════════════════════════"

# ConvNeXt
export_model \
    "${CKPT_ABLATION}/hzvjhovx/epoch=34-rank1tinyface/eval/rank@1=0.568.ckpt" \
    "convnext_tiny.fb_in1k" \
    "onnx_models/baselines/convnext_tiny.onnx"

export_model \
    "${CKPT_ABLATION}/ccc9z3g5/epoch=34-rank1tinyface/eval/rank@1=0.556.ckpt" \
    "convnext_small.fb_in1k" \
    "onnx_models/baselines/convnext_small.onnx"

export_model \
    "${CKPT_ABLATION}/m4xjjh07/epoch=17-rank1tinyface/eval/rank@1=0.570.ckpt" \
    "convnext_base.fb_in1k" \
    "onnx_models/baselines/convnext_base.onnx"

# PVTv2
export_model \
    "${CKPT_ABLATION}/uvu73fxo/epoch=24-rank1tinyface/eval/rank@1=0.548.ckpt" \
    "pvt_v2_b2.in1k" \
    "onnx_models/baselines/pvt_v2_b2.onnx"

export_model \
    "${CKPT_ABLATION}/eltlatns/epoch=21-rank1tinyface/eval/rank@1=0.566.ckpt" \
    "pvt_v2_b3.in1k" \
    "onnx_models/baselines/pvt_v2_b3.onnx"

export_model \
    "${CKPT_ABLATION}/nvzory9h/epoch=21-rank1tinyface/eval/rank@1=0.566.ckpt" \
    "pvt_v2_b5.in1k" \
    "onnx_models/baselines/pvt_v2_b5.onnx"

# MobileViTv2
export_model \
    "${CKPT_ABLATION}/amsv29wr/epoch=42-rank1tinyface/eval/rank@1=0.525.ckpt" \
    "mobilevitv2_200.cvnets_in1k" \
    "onnx_models/baselines/mobilevitv2_200.onnx"

# ResNet
export_model \
    "${CKPT_ABLATION}/4860x6ia/epoch=29-rank1tinyface/eval/rank@1=0.502.ckpt" \
    "resnet50d.ra2_in1k" \
    "onnx_models/baselines/resnet50d.onnx"

export_model \
    "${CKPT_ABLATION}/qopz1ks6/epoch=44-rank1tinyface/eval/rank@1=0.512.ckpt" \
    "resnet101d.ra2_in1k" \
    "onnx_models/baselines/resnet101d.onnx"

export_model \
    "${CKPT_ABLATION}/asyv1awr/epoch=33-rank1tinyface/eval/rank@1=0.484.ckpt" \
    "resnet200d.ra2_in1k" \
    "onnx_models/baselines/resnet200d.onnx"

echo ""
echo "═══════════════════════════════════════════"
echo "  Summary: ${PASSED} passed, ${FAILED} failed, ${SKIPPED} skipped"
echo "═══════════════════════════════════════════"
