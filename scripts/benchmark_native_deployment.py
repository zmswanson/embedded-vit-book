#!/usr/bin/env python3
"""Repeat the chapter's native TensorRT timing harness for supplied engines.

An engine label describes its provenance, not its verified internal precision.
Keep the trtexec logs and engine hashes with the measurement.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import subprocess


def clock_settings():
    cpu = {}
    for folder in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq"):
        cpu[folder.parent.name] = {
            key: (folder / key).read_text().strip()
            for key in ("scaling_min_freq", "scaling_max_freq", "scaling_cur_freq")
        }
    gpu = {}
    for folder in Path("/sys/class/devfreq").glob("*gpu*"):
        gpu[folder.name] = {
            key: (folder / key).read_text().strip()
            for key in ("min_freq", "max_freq", "cur_freq")
        }
    assert cpu and all(v["scaling_min_freq"] == v["scaling_max_freq"] for v in cpu.values())
    assert gpu and all(v["min_freq"] == v["max_freq"] for v in gpu.values())
    return {"cpu": cpu, "gpu": gpu}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", action="append", required=True, help="label=engine_path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode = subprocess.check_output(["nvpmodel", "-q"], text=True).strip()
    assert "MAXN_SUPER" in mode, mode
    report = {
        "model": "deit3_small_baseline-heads_0.10",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "nvpmodel": mode,
        "clocks_before": clock_settings(),
        "latency_scope": "native TensorRT GPU compute only, no transfers, CUDA Graph",
        "input_shape": [1, 3, 96, 96],
        "warmup_ms": 500,
        "minimum_iterations_per_repeat": 2000,
        "engines": {},
    }
    for spec in args.engine:
        label, filename = spec.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            parser.error("engine labels must contain only letters, digits, underscores, or hyphens")
        engine = Path(filename)
        data = engine.read_bytes()
        record = {"engine_path": filename, "engine_sha256": hashlib.sha256(data).hexdigest(),
                  "engine_bytes": len(data), "trials": []}
        for trial in range(args.repeats):
            command = [args.trtexec, "--loadEngine=" + filename, "--warmUp=500",
                       "--iterations=2000", "--noDataTransfers", "--useCudaGraph",
                       "--memPoolSize=workspace:2048MiB"]
            output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
            log_path = args.output_dir / f"{label}_{trial + 1}.log"
            log_path.write_text(output)
            line = next(line for line in output.splitlines() if "GPU Compute Time: min" in line)
            timing = {key: float(re.search(r"\b" + key + r" = ([0-9.eE+-]+)", line).group(1))
                      for key in ("min", "max", "mean", "median")}
            timing.update(log_file=log_path.name, command=command)
            record["trials"].append(timing)
            print(label, "trial", trial + 1, "GPU mean ms", timing["mean"], flush=True)
        record["mean_ms"] = mean(t["mean"] for t in record["trials"])
        report["engines"][label] = record
    report["clocks_after"] = clock_settings()
    report["benchmark_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output_dir / "native_deit3_small_heads10.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value["mean_ms"] for key, value in report["engines"].items()}), flush=True)


if __name__ == "__main__":
    main()
