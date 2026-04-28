#!/bin/bash
# Phase 7.3 — unified power-mode sweep.
#
# Re-runs the Phase 7.1 model selection (12 TRT-native + 2 TRT-incompatible
# variants) entirely through ONNX Runtime + TensorRT EP so that the
# resulting cross-model latencies are directly comparable within and across
# power modes (15W / 25W / MAXN_SUPER). Replaces the previous mixed-harness
# sweep where 12 models used trtexec native and 2 used ORT EP.
#
# Run on the Jetson as root:
#   sudo /ssd/vit_benchmark/benchmark_ortep_power_modes.sh 2>&1 \
#       | tee /ssd/vit_benchmark/results/power_modes/ortep_unified_sweep.log
set -u
ROOT=/ssd/vit_benchmark
DOCKER_IMG=dustynv/onnxruntime:1.20.2-r36.4.0
OUTDIR=$ROOT/results/power_modes
OUT=$OUTDIR/ortep_power_sweep_all.csv
CACHE=$OUTDIR/trt_ep_cache_unified
mkdir -p "$OUTDIR" "$CACHE"

# Same 14-row selection as the original Phase 7.1 sweep, but every row is
# resolved to an ONNX path so it can be benchmarked through ORT EP.
# Format: subdir|onnx_basename|category|precision|label
ROWS=(
  "baselines|swin_tiny|baselines|fp16|swin_tiny"
  "baselines|swin_base|baselines|fp16|swin_base"
  "baselines|deit3_small|baselines|fp16|deit3_small"
  "baselines|deit3_base|baselines|fp16|deit3_base"
  "baselines|resnet50d|baselines|fp16|resnet50d"
  "baselines|resnet200d|baselines|fp16|resnet200d"
  "baselines|swin_tiny|baselines|int8|swin_tiny"
  "baselines|deit3_base|baselines|int8|deit3_base"
  "kd_cvlface|swin_tiny_kd_cvlface_alpha0.0|kd_cvlface|fp16|swin_tiny_kd_a0"
  "kd_petalface|deit3_base_kd_petalface_alpha0.1|kd_petalface|fp16|deit3_base_kd_a01"
  "pruned_v2|swin_tiny_kd_best-blocks_2|pruned_v2|fp16|swin_tiny_kd_blk2"
  "pruned|deit3_base_lora-heads_0.10|pruned|fp16|deit3_base_lora_h10"
  "baselines|convnext_tiny|trt_incompatible|fp16|convnext_tiny"
  "baselines|pvt_v2_b2|trt_incompatible|fp16|pvt_v2_b2"
)

MODES=(
  "0|15W|15"
  "1|25W|25"
  "2|MAXN_SUPER|25"
)

# Fresh output
rm -f "$OUT"

# We'll let the docker container append rows. The Python script writes a
# header on the first call (file doesn't exist) and appends thereafter.

for mspec in "${MODES[@]}"; do
  IFS='|' read -r MODE LABEL BUDGET <<< "$mspec"
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

  for spec in "${ROWS[@]}"; do
    IFS='|' read -r subdir basename category precision label <<< "$spec"
    onnx_path=/ssd/vit_benchmark/onnx_models/${subdir}/${basename}.onnx
    int8_dir=/ssd/vit_benchmark/onnx_models_int8/${subdir}
    if [ ! -f "$ROOT/onnx_models/${subdir}/${basename}.onnx" ]; then
      echo "MISSING $onnx_path — skipping"
      continue
    fi
    echo "  $label ($category, $precision) @ $LABEL"

    if [ "$precision" = "int8" ]; then
      # INT8 via CUDA EP on pre-quantized ONNX (TRT EP can't ingest
      # quantize_dynamic outputs cleanly; this matches the original sweep).
      docker run --rm --runtime nvidia --network host \
          -v "$ROOT":/ssd/vit_benchmark \
          "$DOCKER_IMG" \
          python3 /ssd/vit_benchmark/benchmark_onnxrt_trt_ep.py \
            --onnx-dir /ssd/vit_benchmark/onnx_models/${subdir} \
            --onnx-int8-dir /ssd/vit_benchmark/onnx_models_int8/${subdir} \
            --models "$basename" \
            --ep cuda \
            --warmup 50 --iterations 500 \
            --output "$OUT" \
            --append \
            --power-mode-id "$MODE" \
            --power-mode-label "$LABEL" \
            --wattage-budget "$BUDGET" \
            --trt-cache-dir "$CACHE"
    else
      fp_flag=( --ep trt )
      [ "$precision" = "fp16" ] && fp_flag+=( )  # FP16 enabled below by passing --models AND restricting model list; the script always benchmarks both fp32 and fp16 when --ep trt; we filter rows in post.
      docker run --rm --runtime nvidia --network host \
          -v "$ROOT":/ssd/vit_benchmark \
          "$DOCKER_IMG" \
          python3 /ssd/vit_benchmark/benchmark_onnxrt_trt_ep.py \
            --onnx-dir /ssd/vit_benchmark/onnx_models/${subdir} \
            --models "$basename" \
            --ep trt \
            --warmup 50 --iterations 500 \
            --output "$OUT" \
            --append \
            --power-mode-id "$MODE" \
            --power-mode-label "$LABEL" \
            --wattage-budget "$BUDGET" \
            --trt-cache-dir "$CACHE/${LABEL}"
    fi
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
echo "Tegrastats: $OUTDIR/tegrastats_unified_*.log"
