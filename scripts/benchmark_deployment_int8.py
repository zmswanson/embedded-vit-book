#!/usr/bin/env python3
"""Measure the deployment guide's exact dynamic-INT8 export on the Jetson.

Run in the project's ONNX Runtime GPU container. session.run timings include
host/device transfers and CPU fallback; they are not native TRT compute times.
Profile a separate session so profiling does not affect the timing samples.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations, and repeats must be positive")

    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if digest != args.expected_sha256:
        raise ValueError("The model does not match the locally audited export")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("The ONNX Runtime CUDA provider is unavailable")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def session(profile=False):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        options.enable_profiling = profile
        options.profile_file_prefix = str(args.output.with_suffix(".profile"))
        result = ort.InferenceSession(str(args.model), options, providers=providers)
        if "CUDAExecutionProvider" not in result.get_providers():
            raise RuntimeError("ONNX Runtime failed to initialize CUDA")
        return result

    timed = session()
    input_name = timed.get_inputs()[0].name
    value = np.random.default_rng(0).standard_normal((1, 3, 96, 96)).astype(np.float32)
    feed = {input_name: value}
    for _ in range(args.warmup):
        timed.run(None, feed)

    samples = []
    repeat_means = []
    for _ in range(args.repeats):
        trial = []
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            outputs = timed.run(None, feed)
            trial.append((time.perf_counter_ns() - start) / 1e6)
        if not all(np.isfinite(output).all() for output in outputs):
            raise RuntimeError("Inference produced non-finite outputs")
        samples.extend(trial)
        repeat_means.append(float(np.mean(trial)))
    active_providers = timed.get_providers()
    del timed

    profiled = session(profile=True)
    profiled.run(None, feed)
    profile_path = Path(profiled.end_profiling())
    provider_nodes = Counter()
    provider_ops = defaultdict(Counter)
    for event in json.loads(profile_path.read_text()):
        details = event.get("args", {})
        provider = details.get("provider")
        if event.get("cat") == "Node" and provider:
            provider_nodes[provider] += 1
            provider_ops[provider][details.get("op_name", "unknown")] += 1

    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "deit3_small_baseline-heads_0.10",
        "precision": "ONNX dynamic INT8",
        "model_path": str(args.model),
        "model_sha256": digest,
        "model_bytes": args.model.stat().st_size,
        "onnxruntime_version": ort.__version__,
        "active_providers": active_providers,
        "latency_scope": "session.run including host/device copies and CPU fallback",
        "input_shape": [1, 3, 96, 96],
        "input_seed": 0,
        "warmup": args.warmup,
        "iterations_per_repeat": args.iterations,
        "repeat_means_ms": repeat_means,
        "mean_ms": float(np.mean(samples)),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "std_ms": float(np.std(samples)),
        "profile_path": str(profile_path),
        "profile_provider_node_counts": dict(provider_nodes),
        "profile_provider_op_counts": {k: dict(v) for k, v in provider_ops.items()},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
