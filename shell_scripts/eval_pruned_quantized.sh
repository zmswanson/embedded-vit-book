#!/bin/bash
# Phase 4.4: Evaluate all pruned+quantized ONNX models on TinyFace TEST set
# Computes rank@1-5 and ROC metrics (AUC, mAP, EER, TPR@FPR) for each model.
# 8 pruned models × 3 precisions (FP32, FP16, INT8) = 24 evaluations.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

eval "$(conda shell.bash hook)"
conda activate vit-benchmark

CSV_OUT="pruning_results/eval_pruned_quantized.csv"
rm -f "$CSV_OUT"

PASSED=0
FAILED=0

eval_model() {
    local ONNX_PATH="$1"
    local LABEL="$2"
    local PRECISION="$3"

    if [ ! -f "$ONNX_PATH" ]; then
        echo "  SKIP: $ONNX_PATH not found"
        FAILED=$((FAILED + 1))
        return
    fi

    echo ""
    echo "=== Evaluating: $LABEL ($PRECISION) ==="
    if python eval_onnx.py \
        --onnx_path "$ONNX_PATH" \
        --rank_k 5 \
        --batch_size 256 \
        --roc \
        --csv_out "$CSV_OUT" \
        --model_label "$LABEL" \
        --precision_label "$PRECISION"; then
        PASSED=$((PASSED + 1))
    else
        echo "  FAILED: $LABEL ($PRECISION)"
        FAILED=$((FAILED + 1))
    fi
}

MODELS=(
    "deit3_base_lora-heads_0.10"
    "deit3_base_lora-heads_0.25"
    "deit3_base_lora-heads_0.50"
    "deit3_base_lora-blocks_3"
    "deit3_base_kd-heads_0.25"
    "swin_base-heads_0.10"
    "swin_base-heads_0.25"
    "swin_base-blocks_5"
    "swin_tiny_kd-heads_0.10"
)

for MODEL in "${MODELS[@]}"; do
    eval_model "onnx_models/pruned/${MODEL}.onnx" "$MODEL" "FP32"
    eval_model "onnx_models_fp16/pruned/${MODEL}.onnx" "$MODEL" "FP16"
    eval_model "onnx_models_int8/pruned/${MODEL}.onnx" "$MODEL" "INT8"
done

echo ""
echo "========================================"
echo "DONE: ${PASSED} passed, ${FAILED} failed"
echo "Results: ${CSV_OUT}"
echo "========================================"
