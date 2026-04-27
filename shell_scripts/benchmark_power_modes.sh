#!/bin/bash
# Phase 7.1 — sweep representative TRT engines + ONNXRT-EP models across all
# nvpmodel power modes on the Jetson Orin Nano.
#
# Run on the Jetson as:
#   sudo /ssd/vit_benchmark/benchmark_power_modes.sh 2>&1 \
#       | tee /ssd/vit_benchmark/results/power_modes/sweep.log
#
# Discovered modes (Orin Nano Super, Apr 2026):
#   ID=0  15W
#   ID=1  25W
#   ID=2  MAXN_SUPER
set -u
TRTEXEC=/usr/src/tensorrt/bin/trtexec
ROOT=/ssd/vit_benchmark
OUTDIR=$ROOT/results/power_modes
ONNX_DIR=$ROOT/onnx_models/baselines
DOCKER_IMG=dustynv/onnxruntime:1.20.2-r36.4.0
mkdir -p "$OUTDIR"

# Modes (ID|label|wattage_budget). Iterate whatever the host actually has.
MODES=(
  "0|15W|15"
  "1|25W|25"
  "2|MAXN_SUPER|25"
)

# 12-row TRT subset: path|category|precision|label
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
)

# 2 ONNXRT-EP models (FP32 + FP16) — TRT-incompatible architectures
ORT_MODELS=(convnext_tiny pvt_v2_b2)

TRT_OUT=$OUTDIR/trt_power_sweep.csv
ORT_OUT=$OUTDIR/onnxrt_trt_ep_power_sweep.csv

# Fresh outputs (the script is idempotent — wipe & rewrite)
echo "model,category,precision,power_mode_id,power_mode_label,wattage_budget,mean_ms,median_ms,p99_ms,throughput_fps" > "$TRT_OUT"
rm -f "$ORT_OUT"

for spec in "${MODES[@]}"; do
  IFS='|' read -r MODE LABEL BUDGET <<< "$spec"
  echo
  echo "============================================================"
  echo "Power mode ${MODE} (${LABEL}, ~${BUDGET}W)"
  echo "============================================================"
  nvpmodel -m "$MODE"
  jetson_clocks
  sleep 5  # let DVFS / fan / voltage settle

  # tegrastats in background (sustained wall power)
  TS_LOG="$OUTDIR/tegrastats_${LABEL}.log"
  rm -f "$TS_LOG"
  tegrastats --interval 500 --logfile "$TS_LOG" &
  TS_PID=$!
  sleep 1

  # ---------- TRT engines ----------
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
    mean=$(echo "$log"   | grep "GPU Compute Time" | grep -oE "mean = [0-9.]+"            | awk '{print $3}')
    median=$(echo "$log" | grep "GPU Compute Time" | grep -oE "median = [0-9.]+"          | awk '{print $3}')
    p99=$(echo "$log"    | grep "GPU Compute Time" | grep -oE "percentile\(99%\) = [0-9.]+" | awk '{print $3}')
    fps=$(echo "$log"    | grep "Throughput"       | grep -oE "[0-9.]+ qps"               | awk '{print $1}')
    if [ -z "$mean" ]; then
      echo "  ! parse failed for $elabel ($precision); see logs"
      mean=NaN; median=NaN; p99=NaN; fps=NaN
    fi
    echo "${elabel},${category},${precision},${MODE},${LABEL},${BUDGET},${mean},${median},${p99},${fps}" >> "$TRT_OUT"
    printf "  TRT  %-28s %-4s @ %-11s mean=%s ms  fps=%s\n" "$elabel" "$precision" "$LABEL" "$mean" "$fps"
  done

  # ---------- ONNXRT TRT EP (Docker) ----------
  echo "  -- ONNXRT TRT EP container --"
  docker run --rm --runtime nvidia --network host \
      -v "$ROOT":/ssd/vit_benchmark \
      "$DOCKER_IMG" \
      python3 /ssd/vit_benchmark/benchmark_onnxrt_trt_ep.py \
        --onnx-dir /ssd/vit_benchmark/onnx_models/baselines \
        --models "${ORT_MODELS[@]}" \
        --ep trt \
        --warmup 50 --iterations 500 \
        --output /ssd/vit_benchmark/results/power_modes/onnxrt_trt_ep_power_sweep.csv \
        --append \
        --power-mode-id "$MODE" \
        --power-mode-label "$LABEL" \
        --wattage-budget "$BUDGET" \
        --trt-cache-dir /ssd/vit_benchmark/results/power_modes/trt_ep_cache

  # stop tegrastats
  kill "$TS_PID" 2>/dev/null || true
  wait "$TS_PID" 2>/dev/null || true
done

echo
echo "Restoring MAXN_SUPER (mode 2)..."
nvpmodel -m 2
jetson_clocks

echo
echo "Done. Results:"
echo "  $TRT_OUT"
echo "  $ORT_OUT"
echo "  $OUTDIR/tegrastats_*.log"
