#!/bin/bash
# Transfer ONNX models and benchmark scripts to Jetson Orin Nano.
# Usage: bash shell_scripts/transfer_to_jetson.sh
set -euo pipefail

JETSON_HOST="${JETSON_HOST:-zms@jetson-orin}"
REMOTE_DIR="/ssd/vit_benchmark"

echo "=== Transferring to ${JETSON_HOST}:${REMOTE_DIR} ==="

# Create directory tree on Jetson
echo "Creating directory structure..."
ssh "${JETSON_HOST}" "mkdir -p ${REMOTE_DIR}/{onnx_models/{baselines,lora,kd_cvlface,kd_petalface,pruned,pruned_v2},onnx_models_fp16/{baselines,lora,kd_cvlface,kd_petalface,pruned,pruned_v2},onnx_models_int8/{baselines,lora,kd_cvlface,kd_petalface,pruned,pruned_v2},trt_engines,results,logs}"

# Transfer ONNX models (FP32)
echo "Transferring FP32 ONNX models..."
rsync -avz --progress onnx_models/ "${JETSON_HOST}:${REMOTE_DIR}/onnx_models/"

# Transfer FP16 models
echo "Transferring FP16 ONNX models..."
rsync -avz --progress onnx_models_fp16/ "${JETSON_HOST}:${REMOTE_DIR}/onnx_models_fp16/"

# Transfer INT8 models
echo "Transferring INT8 ONNX models..."
rsync -avz --progress onnx_models_int8/ "${JETSON_HOST}:${REMOTE_DIR}/onnx_models_int8/"

# Transfer benchmark script
echo "Transferring benchmark script..."
scp benchmark_jetson.py "${JETSON_HOST}:${REMOTE_DIR}/"

# Verify transfer
echo ""
echo "=== Verifying transfer ==="
ssh "${JETSON_HOST}" "echo 'Disk usage:'; du -sh ${REMOTE_DIR}/onnx_models* 2>/dev/null; echo ''; echo 'Model counts:'; find ${REMOTE_DIR} -name '*.onnx' | wc -l; echo '.onnx files'; find ${REMOTE_DIR} -name '*.onnx.data' | wc -l; echo '.onnx.data files'"

echo ""
echo "=== Transfer complete ==="
