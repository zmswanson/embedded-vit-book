#!/bin/bash
# Phase 7.3 — unified power-mode sweep using ONLY native TensorRT engines.
# Replaces the dual TRT+ONNXRT-EP harness from benchmark_power_modes.sh.
#
# Run on Jetson as:
#   sudo /ssd/vit_benchmark/benchmark_power_modes_unified.sh 2>&1 \
#       | tee /ssd/vit_benchmark/results/power_modes/sweep_unified.log
set -u
TRTEXEC=/usr/src/tensorrt/bin/trtexec
ROOT=/ssd/vit_benchmark
OUTDIR=$ROOT/results/power_modes
mkdir -p "$OUTDIR"

MODES=(
  "0|15W|15"
  "1|25W|25"
  "2|MAXN_SUPER|25"
)

# 14-row TRT-native subset: engine|category|precision|label
# Unified engines from benchmark_trt_unified.sh (TRT-compat for ConvNeXt/PVT-v2/MobileViT).
ENGINES=(
  "$ROOT/trt_engines/baselines/fp16/swin_tiny_fp16.engine|baselines|fp16|swin_tiny"
  "$ROOT/trt_engines/baselines/fp16/swin_base_fp16.engine|baselines|fp16|swin_base"
  "$ROOT/trt_engines/baselines/fp16/deit3_small_fp16.engine|baselines|fp16|deit3_small"
  "$ROOT/trt_engines/baselines/fp16/deit3_base_fp16.engine|baselines|fp16|deit3_base"
  "$ROOT/trt_engines/baselines/fp16/resnet50d_fp16.engine|baselines|fp16|resnet50d"
  "$ROOT/trt_engines/baselines/fp16/resnet200d_fp16.engine|baselines|fp16|resnet200d"
  "$ROOT/trt_engines/baselines/int8/swin_tiny_int8.engine|baselines|int8|swin_tiny"
  "$ROOT/trt_engines/baselines/int8/deit3_base_int8.engine|baselines|int8|deit3_base"
  "$ROOT/trt_engines/kd_cvlface/fp16/swin_tiny_kd_cvlface_alpha0.0_fp16.engine|kd_cvlface|fp16|swin_tiny_kd_a0"
  "$ROOT/trt_engines/kd_petalface/fp16/deit3_base_kd_petalface_alpha0.1_fp16.engine|kd_petalface|fp16|deit3_base_kd_a01"
  "$ROOT/trt_engines/pruned_v2/fp16/swin_tiny_kd_best-blocks_2_fp16.engine|pruned_v2|fp16|swin_tiny_kd_blk2"
  "$ROOT/trt_engines/pruned/fp16/deit3_base_lora-heads_0.10_fp16.engine|pruned|fp16|deit3_base_lora_h10"
  "$ROOT/trt_engines/unified/fp16/convnext_tiny_fp16.engine|baselines|fp16|convnext_tiny"
  "$ROOT/trt_engines/unified/fp16/pvt_v2_b2_fp16.engine|baselines|fp16|pvt_v2_b2"
)

OUT=$OUTDIR/trt_unified_power_sweep.csv
echo "model,category,precision,power_mode_id,power_mode_label,wattage_budget,mean_ms,median_ms,p99_ms,throughput_fps,backend" > "$OUT"

for spec in "${MODES[@]}"; do
  IFS='|' read -r MODE LABEL BUDGET <<< "$spec"
  echo
  echo "============================================================"
  echo "Power mode ${MODE} (${LABEL}, ~${BUDGET}W)"
  echo "============================================================"
  nvpmodel -m "$MODE"
  jetson_clocks
  sleep 5

  TS_LOG="$OUTDIR/tegrastats_unified_${LABEL}.log"
  rm -f "$TS_LOG"
  tegrastats --interval 500 --logfile "$TS_LOG" &
  TS_PID=$!
  sleep 1

  for spec in "${ENGINES[@]}"; do
    IFS='|' read -r engine category precision elabel <<< "$spec"
    if [ ! -f "$engine" ]; then
      echo "MISSING $engine"
      continue
    fi
    log=$($TRTEXEC \
        --loadEngine="$engine" \
        --warmUp=500 \
        --iterations=2000 \
        --noDataTransfers \
        --useCudaGraph \
        --memPoolSize=workspace:2048MiB \
        2>&1)
    mean=$(echo "$log"   | grep "GPU Compute Time" | grep -oE "mean = [0-9.]+"              | awk '{print $3}')
    median=$(echo "$log" | grep "GPU Compute Time" | grep -oE "median = [0-9.]+"            | awk '{print $3}')
    p99=$(echo "$log"    | grep "GPU Compute Time" | grep -oE "percentile\(99%\) = [0-9.]+" | awk '{print $3}')
    fps=$(echo "$log"    | grep "Throughput"       | grep -oE "[0-9.]+ qps"                 | awk '{print $1}')
    if [ -z "$mean" ]; then
      mean=NaN; median=NaN; p99=NaN; fps=NaN
    fi
    echo "${elabel},${category},${precision},${MODE},${LABEL},${BUDGET},${mean},${median},${p99},${fps},TRT_NATIVE" >> "$OUT"
    printf "  %-28s %-4s @ %-11s mean=%s ms  fps=%s\n" "$elabel" "$precision" "$LABEL" "$mean" "$fps"
  done

  kill "$TS_PID" 2>/dev/null || true
  wait "$TS_PID" 2>/dev/null || true
done

echo
echo "Restoring MAXN_SUPER (mode 2)..."
nvpmodel -m 2
jetson_clocks

echo
echo "Done. Results: $OUT"
