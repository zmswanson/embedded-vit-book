#!/usr/bin/env python3
"""Benchmark ONNX models on Jetson using ONNX Runtime with TensorRT EP.

Auto-partitions graphs: TRT handles supported ops (Conv, MatMul),
CUDA handles unsupported ops (LayerNorm, GroupNorm).

Usage (inside Docker container on Jetson):
    python3 benchmark_onnxrt_trt_ep.py --onnx-dir /ssd/vit_benchmark/onnx_models/baselines \
        --models convnext_tiny convnext_small convnext_base pvt_v2_b2 pvt_v2_b3 pvt_v2_b5 mobilevitv2_200

Output: CSV with model, ep (TRT/CUDA), precision, mean_ms, std_ms, p50_ms, p90_ms, p99_ms
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


MODELS = [
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "pvt_v2_b2",
    "pvt_v2_b3",
    "pvt_v2_b5",
    "mobilevitv2_200",
]


def create_session(onnx_path: str, ep: str, trt_fp16: bool = False,
                   trt_cache_dir: str = "/tmp/trt_ep_cache") -> ort.InferenceSession:
    """Create ORT session with specified execution provider."""
    os.makedirs(trt_cache_dir, exist_ok=True)

    if ep == "trt":
        providers = [
            ("TensorrtExecutionProvider", {
                "device_id": 0,
                "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2GB
                "trt_fp16_enable": trt_fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache_dir,
                "trt_detailed_build_log": True,
            }),
            ("CUDAExecutionProvider", {
                "device_id": 0,
            }),
        ]
    elif ep == "cuda":
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
            }),
        ]
    else:
        raise ValueError(f"Unknown EP: {ep}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.log_severity_level = 3  # suppress warnings

    session = ort.InferenceSession(onnx_path, sess_opts, providers=providers)
    return session


def benchmark_session(session: ort.InferenceSession, img_size: int = 96,
                      warmup: int = 50, iterations: int = 200) -> dict:
    """Run warmup + timed iterations. Returns latency statistics."""
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    # Replace dynamic dims with concrete values
    shape = []
    for d in input_shape:
        if isinstance(d, int):
            shape.append(d)
        else:
            shape.append(1)  # batch size
    dummy = np.random.randn(*shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    # Timed iterations
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", required=True, help="Directory with FP32 ONNX models")
    parser.add_argument("--onnx-int8-dir", default=None,
                        help="Directory with pre-quantized INT8 ONNX models (CUDA EP only)")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--output", default="onnxrt_trt_ep_results.csv")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--ep", choices=["trt", "cuda", "both"], default="both",
                        help="Execution provider(s) to benchmark")
    parser.add_argument("--trt-cache-dir", default="/tmp/trt_ep_cache")
    parser.add_argument("--power-mode-label", default=None,
                        help="Optional label tagged onto every result row (e.g. MAXN_SUPER, 15W).")
    parser.add_argument("--power-mode-id", default=None,
                        help="Optional numeric mode ID tagged onto every result row.")
    parser.add_argument("--wattage-budget", default=None,
                        help="Optional wattage budget tagged onto every result row.")
    parser.add_argument("--append", action="store_true",
                        help="Append rows to --output instead of overwriting (header only on create).")
    args = parser.parse_args()

    eps_to_test = ["trt", "cuda"] if args.ep == "both" else [args.ep]
    results = []

    for model_name in args.models:
        onnx_path = os.path.join(args.onnx_dir, f"{model_name}.onnx")
        if not os.path.exists(onnx_path):
            print(f"SKIP {model_name}: {onnx_path} not found")
            continue

        for ep in eps_to_test:
            for fp16 in ([False, True] if ep == "trt" else [False]):
                precision = "FP16" if fp16 else "FP32"
                label = f"{ep.upper()}+{'FP16' if fp16 else 'FP32'}"
                print(f"\n{'='*60}")
                print(f"Benchmarking: {model_name} | EP={ep.upper()} | Precision={precision}")
                print(f"{'='*60}")

                try:
                    cache_dir = os.path.join(args.trt_cache_dir, model_name, precision.lower())
                    session = create_session(onnx_path, ep, trt_fp16=fp16,
                                             trt_cache_dir=cache_dir)

                    # Show which EPs are actually used
                    actual_eps = [p.strip() for p in session.get_providers()]
                    print(f"  Active EPs: {actual_eps}")

                    stats = benchmark_session(session, img_size=args.img_size,
                                              warmup=args.warmup,
                                              iterations=args.iterations)

                    print(f"  Mean: {stats['mean_ms']:.2f} ms  |  "
                          f"P50: {stats['p50_ms']:.2f} ms  |  "
                          f"P90: {stats['p90_ms']:.2f} ms  |  "
                          f"P99: {stats['p99_ms']:.2f} ms")

                    results.append({
                        "model": model_name,
                        "ep": ep.upper(),
                        "precision": precision,
                        **stats,
                    })

                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.append({
                        "model": model_name,
                        "ep": ep.upper(),
                        "precision": precision,
                        "mean_ms": -1,
                        "std_ms": -1,
                        "p50_ms": -1,
                        "p90_ms": -1,
                        "p99_ms": -1,
                        "min_ms": -1,
                        "max_ms": -1,
                        "error": str(e),
                    })

    # INT8 via CUDA EP using pre-quantized INT8 ONNX models
    if args.onnx_int8_dir and "cuda" in eps_to_test:
        for model_name in args.models:
            onnx_path = os.path.join(args.onnx_int8_dir, f"{model_name}.onnx")
            if not os.path.exists(onnx_path):
                print(f"SKIP INT8 {model_name}: {onnx_path} not found")
                continue
            print(f"\n{'='*60}")
            print(f"Benchmarking: {model_name} | EP=CUDA | Precision=INT8")
            print(f"{'='*60}")
            try:
                session = create_session(onnx_path, "cuda", trt_fp16=False)
                actual_eps = [p.strip() for p in session.get_providers()]
                print(f"  Active EPs: {actual_eps}")
                stats = benchmark_session(session, img_size=args.img_size,
                                          warmup=args.warmup,
                                          iterations=args.iterations)
                print(f"  Mean: {stats['mean_ms']:.2f} ms  |  "
                      f"P50: {stats['p50_ms']:.2f} ms  |  "
                      f"P90: {stats['p90_ms']:.2f} ms  |  "
                      f"P99: {stats['p99_ms']:.2f} ms")
                results.append({
                    "model": model_name,
                    "ep": "CUDA",
                    "precision": "INT8",
                    **stats,
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "model": model_name,
                    "ep": "CUDA",
                    "precision": "INT8",
                    "mean_ms": -1, "std_ms": -1, "p50_ms": -1,
                    "p90_ms": -1, "p99_ms": -1, "min_ms": -1, "max_ms": -1,
                    "error": str(e),
                })

    # Tag rows with power-mode metadata if provided
    if args.power_mode_label is not None or args.power_mode_id is not None or args.wattage_budget is not None:
        for r in results:
            if args.power_mode_id is not None:
                r["power_mode_id"] = args.power_mode_id
            if args.power_mode_label is not None:
                r["power_mode_label"] = args.power_mode_label
            if args.wattage_budget is not None:
                r["wattage_budget"] = args.wattage_budget

    # Write CSV
    if results:
        fieldnames = ["model", "ep", "precision", "mean_ms", "std_ms",
                      "p50_ms", "p90_ms", "p99_ms", "min_ms", "max_ms"]
        if args.power_mode_id is not None:
            fieldnames.append("power_mode_id")
        if args.power_mode_label is not None:
            fieldnames.append("power_mode_label")
        if args.wattage_budget is not None:
            fieldnames.append("wattage_budget")
        if any("error" in r for r in results):
            fieldnames.append("error")
        append_mode = args.append and os.path.exists(args.output)
        with open(args.output, "a" if append_mode else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not append_mode:
                writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output} ({'appended' if append_mode else 'written'})")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Model':<20} {'EP':<10} {'Prec':<6} {'Mean':>8} {'P50':>8} {'P90':>8} {'P99':>8}")
    print(f"{'-'*80}")
    for r in results:
        if r.get("mean_ms", -1) > 0:
            print(f"{r['model']:<20} {r['ep']:<10} {r['precision']:<6} "
                  f"{r['mean_ms']:>7.2f}  {r['p50_ms']:>7.2f}  "
                  f"{r['p90_ms']:>7.2f}  {r['p99_ms']:>7.2f}")
        else:
            print(f"{r['model']:<20} {r['ep']:<10} {r['precision']:<6}  FAILED")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
