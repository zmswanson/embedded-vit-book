#!/usr/bin/env python3
"""Benchmark inference on Jetson Orin Nano using TensorRT and ONNX Runtime.

Usage examples:
    # TensorRT engine benchmark
    python3 benchmark_jetson.py --engine_path trt_engines/baselines/fp16/swin_tiny_fp16.engine

    # ONNX Runtime benchmark (CPU or GPU)
    python3 benchmark_jetson.py --onnx_path onnx_models/baselines/swin_tiny.onnx

    # Sweep all engines in a directory
    python3 benchmark_jetson.py --engine_dir trt_engines/ --output_csv results/jetson_trt_benchmarks.csv

    # Sweep all ONNX models
    python3 benchmark_jetson.py --onnx_dir onnx_models/ --output_csv results/jetson_onnx_benchmarks.csv
"""

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Power / thermal monitoring helpers
# ---------------------------------------------------------------------------

POWER_SENSOR_PATHS = [
    # Orin Nano INA sensor paths (VDD_IN = total board power)
    "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon",
    "/sys/bus/i2c/drivers/ina3221x/1-0040/hwmon",
]


def _find_power_hwmon():
    """Find the hwmon directory for the INA power sensor."""
    for base in POWER_SENSOR_PATHS:
        if os.path.isdir(base):
            for d in os.listdir(base):
                if d.startswith("hwmon"):
                    return os.path.join(base, d)
    return None


def read_power_mw():
    """Read current total board power in milliwatts from INA sensor.

    Returns None if sensor is not found.
    """
    hwmon = _find_power_hwmon()
    if hwmon is None:
        return None
    # Try common INA3221 power file names
    for fname in ["power1_input", "in1_input"]:
        fpath = os.path.join(hwmon, fname)
        if os.path.isfile(fpath):
            try:
                with open(fpath) as f:
                    return int(f.read().strip())  # microwatts → keep raw
            except (OSError, ValueError):
                pass
    return None


def read_tegrastats_power():
    """Parse VDD_IN from a single tegrastats sample (mW)."""
    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "100"],
            capture_output=True, text=True, timeout=2,
        )
        line = result.stdout.strip().split("\n")[-1] if result.stdout else ""
        m = re.search(r"VDD_IN\s+(\d+)mW", line)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def read_thermal_c():
    """Read max thermal zone temperature in °C."""
    temps = []
    tz_dir = Path("/sys/devices/virtual/thermal")
    if tz_dir.is_dir():
        for tz in sorted(tz_dir.glob("thermal_zone*")):
            try:
                with open(tz / "temp") as f:
                    temps.append(int(f.read().strip()) / 1000.0)
            except (OSError, ValueError):
                pass
    return max(temps) if temps else None


# ---------------------------------------------------------------------------
# TensorRT benchmark
# ---------------------------------------------------------------------------

def benchmark_trt(engine_path, input_shape=(1, 3, 96, 96),
                  warmup=100, iterations=1000, measure_power=True):
    """Benchmark a TensorRT engine and return performance metrics."""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 – initialises CUDA context

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Determine input/output tensor names and set shapes
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    context.set_input_shape(input_name, input_shape)
    output_shape = tuple(context.get_tensor_shape(output_name))

    # Allocate host + device buffers
    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    # Set tensor addresses
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # Warmup
    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # Timed iterations
    power_samples = []
    latencies = []
    for _ in range(iterations):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        stream.synchronize()

        start = time.perf_counter_ns()
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        end = time.perf_counter_ns()

        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        latencies.append((end - start) / 1e6)  # ns → ms

        if measure_power and len(latencies) % 50 == 0:
            pw = read_tegrastats_power() or read_power_mw()
            if pw is not None:
                power_samples.append(pw)

    latencies = np.array(latencies)
    avg_power_mw = float(np.mean(power_samples)) if power_samples else None

    # Query memory
    free_before, total = cuda.mem_get_info()
    gpu_mem_used_mb = (total - free_before) / 1e6

    result = {
        "engine": Path(engine_path).stem,
        "backend": "TensorRT",
        "input_shape": str(input_shape),
        "output_shape": str(output_shape),
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput_fps": float(input_shape[0] * 1000 / np.mean(latencies)),
        "gpu_mem_used_mb": round(gpu_mem_used_mb, 1),
        "warmup": warmup,
        "iterations": iterations,
    }
    if avg_power_mw is not None:
        result["avg_power_mw"] = round(avg_power_mw, 1)
        result["energy_per_inf_mj"] = round(
            avg_power_mw * float(np.mean(latencies)) / 1000.0, 3
        )

    temp = read_thermal_c()
    if temp is not None:
        result["max_temp_c"] = round(temp, 1)

    # Cleanup
    d_input.free()
    d_output.free()
    del context, engine

    return result


# ---------------------------------------------------------------------------
# ONNX Runtime benchmark
# ---------------------------------------------------------------------------

def benchmark_onnx(onnx_path, input_shape=(1, 3, 96, 96),
                   warmup=100, iterations=1000):
    """Benchmark an ONNX model with ONNX Runtime."""
    import onnxruntime as ort

    # Try GPU providers first, fall back to CPU
    available = ort.get_available_providers()
    providers = []
    for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider"]:
        if p in available:
            providers.append(p)
    providers.append("CPUExecutionProvider")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    active_provider = session.get_providers()[0]

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    h_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        session.run([output_name], {input_name: h_input})

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        session.run([output_name], {input_name: h_input})
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1e6)

    latencies = np.array(latencies)
    output_shape = session.get_outputs()[0].shape

    return {
        "engine": Path(onnx_path).stem,
        "backend": f"ORT-{active_provider}",
        "input_shape": str(input_shape),
        "output_shape": str(output_shape),
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput_fps": float(input_shape[0] * 1000 / np.mean(latencies)),
        "warmup": warmup,
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Directory sweeps
# ---------------------------------------------------------------------------

def sweep_engines(engine_dir, **kwargs):
    """Benchmark all .engine files under engine_dir recursively."""
    results = []
    engines = sorted(Path(engine_dir).rglob("*.engine"))
    print(f"Found {len(engines)} TRT engines under {engine_dir}")
    for eng in engines:
        print(f"  Benchmarking: {eng}")
        try:
            r = benchmark_trt(str(eng), **kwargs)
            results.append(r)
        except Exception as e:
            print(f"    ERROR: {e}")
    return results


def sweep_onnx(onnx_dir, **kwargs):
    """Benchmark all .onnx files under onnx_dir recursively."""
    results = []
    models = sorted(Path(onnx_dir).rglob("*.onnx"))
    print(f"Found {len(models)} ONNX models under {onnx_dir}")
    for m in models:
        print(f"  Benchmarking: {m}")
        try:
            r = benchmark_onnx(str(m), **kwargs)
            results.append(r)
        except Exception as e:
            print(f"    ERROR: {e}")
    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(results, output_csv):
    """Save benchmark results to CSV."""
    if not results:
        print("No results to save.")
        return
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    keys = list(results[0].keys())
    # Ensure all rows have the same keys
    for r in results:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference on Jetson Orin Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--engine_path", help="Path to a single TRT engine")
    group.add_argument("--onnx_path", help="Path to a single ONNX model")
    group.add_argument("--engine_dir", help="Directory of TRT engines (recursive)")
    group.add_argument("--onnx_dir", help="Directory of ONNX models (recursive)")

    parser.add_argument("--output_csv", default=None,
                        help="CSV file to save results")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Number of warmup iterations (default: 100)")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Number of timed iterations (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--input_size", type=int, default=96,
                        help="Input spatial size (default: 96)")
    parser.add_argument("--no_power", action="store_true",
                        help="Skip power measurement")
    args = parser.parse_args()

    input_shape = (args.batch_size, 3, args.input_size, args.input_size)
    common = dict(
        input_shape=input_shape,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    if args.engine_path:
        r = benchmark_trt(args.engine_path, measure_power=not args.no_power,
                          **common)
        results = [r]
    elif args.onnx_path:
        r = benchmark_onnx(args.onnx_path, **common)
        results = [r]
    elif args.engine_dir:
        results = sweep_engines(args.engine_dir,
                                measure_power=not args.no_power, **common)
    elif args.onnx_dir:
        results = sweep_onnx(args.onnx_dir, **common)
    else:
        results = []

    # Print summary
    for r in results:
        print(f"\n{'='*60}")
        print(f"  Model:      {r['engine']}")
        print(f"  Backend:    {r['backend']}")
        print(f"  Mean:       {r['mean_latency_ms']:.3f} ms")
        print(f"  Median:     {r['median_latency_ms']:.3f} ms")
        print(f"  P95:        {r['p95_latency_ms']:.3f} ms")
        print(f"  P99:        {r['p99_latency_ms']:.3f} ms")
        print(f"  Throughput: {r['throughput_fps']:.1f} FPS")
        if "avg_power_mw" in r:
            print(f"  Power:      {r['avg_power_mw']:.0f} mW")
            print(f"  Energy:     {r['energy_per_inf_mj']:.3f} mJ/inference")
        if "max_temp_c" in r:
            print(f"  Temp:       {r['max_temp_c']:.1f} °C")
        print(f"{'='*60}")

    if args.output_csv and results:
        save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
