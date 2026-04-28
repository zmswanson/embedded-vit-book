#!/bin/bash
# Phase 7.3 — unified Jetson latency benchmark.
#
# Runs ALL 16 chapter baseline models through ONNX Runtime + TensorRT EP
# (Docker container) at FP32 and FP16. Produces a single CSV with the same
# code path / harness for every model so that the resulting latency numbers
# are directly comparable across all six families.
#
# This replaces the previous dual-path comparison (trtexec native for 9
# models, ORT TRT EP for 7 TRT-incompatible models) with a single fair
# benchmark — see chapter sec:trt-compat for the rationale.
#
# Run on the Jetson as:
#   sudo /ssd/vit_benchmark/benchmark_ortep_unified.sh 2>&1 \
#       | tee /ssd/vit_benchmark/results/ortep_unified.log
set -u
ROOT=/ssd/vit_benchmark
ONNX_DIR=$ROOT/onnx_models/baselines
DOCKER_IMG=dustynv/onnxruntime:1.20.2-r36.4.0
OUT=$ROOT/results/onnxrt_trt_ep_all16.csv
CACHE=$ROOT/results/trt_ep_cache_unified

mkdir -p "$ROOT/results" "$CACHE"
rm -f "$OUT"

# All 16 chapter baselines, in canonical chapter order
MODELS=(
  resnet50d resnet101d resnet200d
  deit3_small deit3_medium deit3_base
  swin_tiny swin_small swin_base
  convnext_tiny convnext_small convnext_base
  pvt_v2_b2 pvt_v2_b3 pvt_v2_b5
  mobilevitv2_200
)

# Lock to MAXN_SUPER for the headline benchmark
echo zms | sudo -S nvpmodel -m 2 >/dev/null 2>&1 || true
sudo jetson_clocks  >/dev/null 2>&1 || true
sleep 5

echo "Running unified ORT TRT EP benchmark on ${#MODELS[@]} models..."

docker run --rm --runtime nvidia --network host \
    -v "$ROOT":/ssd/vit_benchmark \
    "$DOCKER_IMG" \
    python3 /ssd/vit_benchmark/benchmark_onnxrt_trt_ep.py \
      --onnx-dir /ssd/vit_benchmark/onnx_models/baselines \
      --models "${MODELS[@]}" \
      --ep trt \
      --warmup 50 --iterations 500 \
      --output "$OUT" \
      --trt-cache-dir "$CACHE"

echo
echo "Done. Results at: $OUT"
