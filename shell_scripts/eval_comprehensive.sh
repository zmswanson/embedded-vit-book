#!/bin/bash
# Evaluate all pruned_v2 ONNX models on TinyFace TEST set
# 15 pruned models × 3 precisions (FP32, FP16, INT8) = 45 evaluations.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

eval "$(conda shell.bash hook)"
conda activate vit-benchmark

CSV_OUT="pruning_results/eval_comprehensive.csv"
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

# --- New v2 pruned models (15 models) ---
MODELS_V2=(
    "swin_tiny_baseline-heads_0.10"
    "swin_tiny_baseline-heads_0.25"
    "swin_tiny_baseline-blocks_2"
    "swin_tiny_kd_best-heads_0.10"
    "swin_tiny_kd_best-heads_0.25"
    "swin_tiny_kd_best-blocks_2"
    "deit3_small_baseline-heads_0.10"
    "deit3_small_baseline-heads_0.25"
    "deit3_small_baseline-blocks_2"
    "swin_base_baseline-heads_0.10"
    "swin_base_baseline-heads_0.25"
    "swin_base_baseline-blocks_5"
    "deit3_base_kd_best-heads_0.10"
    "deit3_base_kd_best-heads_0.25"
    "deit3_base_kd_best-blocks_3"
)

for MODEL in "${MODELS_V2[@]}"; do
    eval_model "onnx_models/pruned_v2/${MODEL}.onnx" "$MODEL" "FP32"
    eval_model "onnx_models_fp16/pruned_v2/${MODEL}.onnx" "$MODEL" "FP16"
    eval_model "onnx_models_int8/pruned_v2/${MODEL}.onnx" "$MODEL" "INT8"
done

echo ""
echo "========================================"
echo "DONE: ${PASSED} passed, ${FAILED} failed"
echo "Results: ${CSV_OUT}"
echo "========================================"
