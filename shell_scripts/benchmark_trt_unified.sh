#!/bin/bash
# Phase 7.3 — unified native TensorRT benchmark for all 16 baseline models.
#
# All 16 models — including ConvNeXt, PVT-v2, MobileViT-v2 — are now exported
# via export_trt_compat.py to use TRT-native ops only (3D-reshape LayerNorm,
# disabled fused attention, manual GroupNorm decomposition). This script
# builds and benchmarks engines for all 16 at MAXN_SUPER, FP32 + FP16.
#
# Run on the Jetson as root:
#   sudo /ssd/vit_benchmark/benchmark_trt_unified.sh 2>&1 \
#       | tee /ssd/vit_benchmark/results/trt_unified.log
set -u
TRTEXEC=/usr/src/tensorrt/bin/trtexec
ROOT=/ssd/vit_benchmark
OUTDIR=$ROOT/results
ENG_DIR=$ROOT/trt_engines/unified
OUT=$OUTDIR/trt_unified_all16.csv
mkdir -p "$OUTDIR" "$ENG_DIR/fp32" "$ENG_DIR/fp16"

# 16-model list. All TRT-native (the 7 originally-incompatible models use
# the trt_compat/ rewrites with cosine ~ 1.0 vs the originals).
# Format: short_name|onnx_path|family
MODELS=(
  "resnet50d|baselines/resnet50d.onnx|resnet"
  "resnet101d|baselines/resnet101d.onnx|resnet"
  "resnet200d|baselines/resnet200d.onnx|resnet"
  "deit3_small|baselines/deit3_small.onnx|deit3"
  "deit3_medium|baselines/deit3_medium.onnx|deit3"
  "deit3_base|baselines/deit3_base.onnx|deit3"
  "swin_tiny|baselines/swin_tiny.onnx|swin"
  "swin_small|baselines/swin_small.onnx|swin"
  "swin_base|baselines/swin_base.onnx|swin"
  "convnext_tiny|trt_compat/convnext_tiny.onnx|convnext"
  "convnext_small|trt_compat/convnext_small.onnx|convnext"
  "convnext_base|trt_compat/convnext_base.onnx|convnext"
  "pvt_v2_b2|trt_compat/pvt_v2_b2.onnx|pvt_v2"
  "pvt_v2_b3|trt_compat/pvt_v2_b3.onnx|pvt_v2"
  "pvt_v2_b5|trt_compat/pvt_v2_b5.onnx|pvt_v2"
  "mobilevitv2_200|trt_compat/mobilevitv2_200.onnx|mobilevit_v2"
)

# Lock to MAXN_SUPER for the headline benchmark.
nvpmodel -m 2
jetson_clocks
sleep 5

# Fresh CSV
echo "model,family,precision,mean_ms,median_ms,p90_ms,p95_ms,p99_ms,throughput_fps,backend" > "$OUT"

build_and_bench() {
  local short=$1 onnx=$2 family=$3 prec=$4
  local prec_flag=""
  local prec_dir="fp32"
  if [ "$prec" = "fp16" ]; then prec_flag="--fp16"; prec_dir="fp16"; fi
  local engine=$ENG_DIR/$prec_dir/${short}_${prec}.engine
  local onnx_path=$ROOT/onnx_models/$onnx

  if [ ! -f "$onnx_path" ]; then
    echo "  MISSING $onnx_path"
    return
  fi

  # Build engine if missing
  if [ ! -f "$engine" ]; then
    echo "  Building $short $prec ..."
    $TRTEXEC \
      --onnx="$onnx_path" \
      $prec_flag \
      --memPoolSize=workspace:2048 \
      --saveEngine="$engine" \
      --skipInference > /dev/null 2>&1
    if [ ! -f "$engine" ]; then
      echo "  BUILD FAILED $short $prec"
      echo "${short},${family},${prec},NaN,NaN,NaN,NaN,NaN,NaN,TRT_NATIVE" >> "$OUT"
      return
    fi
  fi

  # Benchmark
  local log
  log=$($TRTEXEC \
        --loadEngine="$engine" \
        --warmUp=500 \
        --iterations=2000 \
        --noDataTransfers \
        --useCudaGraph \
        --memPoolSize=workspace:2048MiB \
        2>&1)

  local mean median p90 p95 p99 fps
  mean=$(echo "$log"   | grep "GPU Compute Time" | grep -oE "mean = [0-9.]+"               | awk '{print $3}')
  median=$(echo "$log" | grep "GPU Compute Time" | grep -oE "median = [0-9.]+"             | awk '{print $3}')
  p90=$(echo "$log"    | grep "GPU Compute Time" | grep -oE "percentile\(90%\) = [0-9.]+" | awk '{print $3}')
  p95=$(echo "$log"    | grep "GPU Compute Time" | grep -oE "percentile\(95%\) = [0-9.]+" | awk '{print $3}')
  p99=$(echo "$log"    | grep "GPU Compute Time" | grep -oE "percentile\(99%\) = [0-9.]+" | awk '{print $3}')
  fps=$(echo "$log"    | grep "Throughput"       | grep -oE "[0-9.]+ qps"                  | awk '{print $1}')
  [ -z "$mean" ] && mean=NaN && median=NaN && p90=NaN && p95=NaN && p99=NaN && fps=NaN

  echo "${short},${family},${prec},${mean},${median},${p90},${p95},${p99},${fps},TRT_NATIVE" >> "$OUT"
  printf "  %-18s %-4s  mean=%-7s median=%-7s p99=%-7s fps=%s\n" "$short" "$prec" "$mean" "$median" "$p99" "$fps"
}

echo "============================================================"
echo "Unified TRT-native benchmark (MAXN_SUPER, batch=1, 96x96)"
echo "============================================================"
for spec in "${MODELS[@]}"; do
  IFS='|' read -r short onnx family <<< "$spec"
  echo "--- $short ($family) ---"
  build_and_bench "$short" "$onnx" "$family" fp32
  build_and_bench "$short" "$onnx" "$family" fp16
done

echo
echo "Done. Results: $OUT"
