#!/bin/bash
# Evaluate key quantized models on TinyFace test set (rank@1-5)
# Key models to evaulate (based on accuracy-to-size tradeoff):
#   1. swin_tiny_kd_cvlface_frozen  (best accuracy-to-size: 0.616 rank@1)
#   2. deit3_base_kd_cvlface_frozen (0.593 rank@1)
#   3. deit3_base_lora              (0.583 rank@1)
#   4. swin_tiny baseline           (0.562 rank@1, smallest)

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vit-benchmark

CSV_OUT="quantization_results/eval_quantized_models.csv"
rm -f "$CSV_OUT"

eval_model() {
    local ONNX_PATH="$1"
    local LABEL="$2"
    local PRECISION="$3"
    echo "=== Evaluating: $LABEL ($PRECISION) ==="
    python eval_onnx.py \
        --onnx_path "$ONNX_PATH" \
        --rank_k 5 \
        --batch_size 256 \
        --csv_out "$CSV_OUT" \
        --model_label "$LABEL" \
        --precision_label "$PRECISION"
}

# --- swin_tiny baseline ---
eval_model "onnx_models/baselines/swin_tiny.onnx" "swin_tiny_baseline" "FP32"
eval_model "onnx_models_fp16/baselines/swin_tiny.onnx" "swin_tiny_baseline" "FP16"
eval_model "onnx_models_int8/baselines/swin_tiny.onnx" "swin_tiny_baseline" "INT8"

# --- swin_tiny_kd_cvlface_frozen ---
eval_model "onnx_models/kd_cvlface/swin_tiny_kd_cvlface_frozen.onnx" "swin_tiny_kd_cvlface_frozen" "FP32"
eval_model "onnx_models_fp16/kd_cvlface/swin_tiny_kd_cvlface_frozen.onnx" "swin_tiny_kd_cvlface_frozen" "FP16"
eval_model "onnx_models_int8/kd_cvlface/swin_tiny_kd_cvlface_frozen.onnx" "swin_tiny_kd_cvlface_frozen" "INT8"

# --- deit3_base_kd_cvlface_frozen ---
eval_model "onnx_models/kd_cvlface/deit3_base_kd_cvlface_frozen.onnx" "deit3_base_kd_cvlface_frozen" "FP32"
eval_model "onnx_models_fp16/kd_cvlface/deit3_base_kd_cvlface_frozen.onnx" "deit3_base_kd_cvlface_frozen" "FP16"
eval_model "onnx_models_int8/kd_cvlface/deit3_base_kd_cvlface_frozen.onnx" "deit3_base_kd_cvlface_frozen" "INT8"

# --- deit3_base_lora ---
eval_model "onnx_models/lora/deit3_base_lora.onnx" "deit3_base_lora" "FP32"
eval_model "onnx_models_fp16/lora/deit3_base_lora.onnx" "deit3_base_lora" "FP16"
eval_model "onnx_models_int8/lora/deit3_base_lora.onnx" "deit3_base_lora" "INT8"

echo ""
echo "=== DONE ==="
echo "Results saved to $CSV_OUT"
cat "$CSV_OUT"
