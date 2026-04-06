#!/usr/bin/env python3
"""Phase 5.4 — ONNX Runtime per-node profiling for the 7 TRT-incompatible models.

Runs inside the dustynv/onnxruntime Docker container on Jetson:
  docker run --rm --runtime nvidia --network=host \
    -v /ssd/vit_benchmark:/ssd/vit_benchmark \
    dustynv/onnxruntime:1.20.2-r36.4.0 \
    python3 /ssd/vit_benchmark/profile_ort_models.py

Outputs one Chrome-trace JSON per model (ORT default format) plus a parsed
summary CSV at /ssd/vit_benchmark/profiles/ort_profiling_summary.csv.
"""
import json
import csv
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import onnxruntime as ort
except ImportError:
    raise SystemExit("onnxruntime not found — run inside dustynv/onnxruntime Docker image")

BASE = Path("/ssd/vit_benchmark")
ONNX_DIR = BASE / "onnx_models/baselines"
PROFILES_DIR = BASE / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "pvt_v2_b2",
    "pvt_v2_b3",
    "pvt_v2_b5",
    "mobilevitv2_200",
]

WARMUP = 10
ITERS = 100


def categorize_node(name: str, op_type: str) -> str:
    n = (name + " " + op_type).lower()
    if any(x in n for x in ["attention", "softmax", "qkv"]):
        return "attention"
    if any(x in n for x in ["gemm", "matmul", "fc", "linear", "proj", "mlp", "ffn"]):
        return "ffn"
    if any(x in n for x in ["layernorm", "batchnorm", "instancenorm", "norm", "groupnorm"]):
        return "norm"
    if any(x in n for x in ["conv", "patch", "embed", "downsample", "stem"]):
        return "conv_embed"
    if any(x in n for x in ["add", "residual", "skip", "elementwise", "relu", "gelu", "act"]):
        return "residual"
    if any(x in n for x in ["pool", "reduce", "global"]):
        return "pooling"
    if any(x in n for x in ["reshape", "transpose", "gather", "slice", "cast", "flatten"]):
        return "reshape"
    return "other"


def parse_ort_trace(trace_path: str) -> tuple:
    """Parse ORT Chrome-trace JSON. Returns (cats_dict, total_ms, layer_rows)."""
    data = json.loads(Path(trace_path).read_text())

    cats = defaultdict(float)
    total = 0.0
    layer_rows = []

    for event in data:
        # ORT trace events: ph="X" (complete), cat="Node", dur in microseconds
        if event.get("ph") != "X" or event.get("cat") != "Node":
            continue
        dur_us = float(event.get("dur", 0))
        dur_ms = dur_us / 1000.0
        name = event.get("name", "")
        args = event.get("args", {})
        op_type = args.get("op_name", "")

        cat = categorize_node(name, op_type)
        cats[cat] += dur_ms
        total += dur_ms
        layer_rows.append({
            "name": name,
            "op_type": op_type,
            "category": cat,
            "avg_ms": 0,  # will compute per-run average later
            "total_ms": round(dur_ms, 4),
        })

    return cats, total, layer_rows


def run_model(model_name: str) -> dict | None:
    onnx_path = ONNX_DIR / f"{model_name}.onnx"
    if not onnx_path.exists():
        print(f"  [skip] {model_name}: ONNX not found at {onnx_path}")
        return None

    prefix = str(PROFILES_DIR / f"{model_name}_ort")

    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = prefix

    print(f"  [profiling] {model_name} ...")
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    )

    input_name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, 96, 96).astype(np.float32)

    # warmup
    for _ in range(WARMUP):
        sess.run(None, {input_name: x})

    # timed iterations
    for _ in range(ITERS):
        sess.run(None, {input_name: x})

    trace_path = sess.end_profiling()
    print(f"  [done] trace → {trace_path}")
    return trace_path


CATEGORIES_ORDER = ["attention", "qkv_proj", "ffn", "conv_embed", "norm",
                    "residual", "pooling", "reshape", "other"]

summaries = []
all_layer_rows = []

for model_name in MODELS:
    trace_path = run_model(model_name)
    if trace_path is None:
        continue

    cats, total, layer_rows = parse_ort_trace(trace_path)

    row = {"model": f"{model_name}_ort", "total_ms": round(total / ITERS, 3)}
    for cat in CATEGORIES_ORDER:
        t = cats.get(cat, 0.0) / ITERS
        row[f"{cat}_ms"] = round(t, 3)
        row[f"{cat}_pct"] = round(100 * t * ITERS / total, 1) if total else 0.0
    summaries.append(row)

    for lr in layer_rows:
        lr["model"] = model_name
    all_layer_rows.extend(layer_rows)

    breakdown = "  ".join(
        f"{cat}={100*cats.get(cat,0)/total:.0f}%"
        for cat in CATEGORIES_ORDER
        if cats.get(cat, 0) > 0
    )
    print(f"  {model_name:<45} total={total/ITERS:.3f}ms  {breakdown}")

# Write summary CSV
out_csv = PROFILES_DIR / "ort_profiling_summary.csv"
if summaries:
    fields = ["model", "total_ms"]
    for cat in CATEGORIES_ORDER:
        fields += [f"{cat}_ms", f"{cat}_pct"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries)
    print(f"\nORT summary CSV → {out_csv}  ({len(summaries)} models)")
