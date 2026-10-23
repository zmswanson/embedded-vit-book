#!/usr/bin/env python3
"""Refresh native GPU-compute latency for the exact deployment engines."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import subprocess

from measure_native_resources import LABELS, clocks, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.engines.read_text())["engines"]
    assert len(records) == 5
    mode = subprocess.check_output(["nvpmodel", "-q"], text=True)
    assert "MAXN_SUPER" in mode
    report = {"measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "nvpmodel": mode.strip(), "clocks_before": clocks(),
              "input_shape": [1, 3, 96, 96], "repeats": 3,
              "latency_scope": "Native TensorRT GPU compute only; no transfers; CUDA Graph",
              "script_sha256": sha(__file__),
              "helper_script_sha256": sha(Path(__file__).with_name("measure_native_resources.py")),
              "engine_manifest_sha256": sha(args.engines), "engines": {}}
    for label, engine in zip(LABELS, records):
        assert sha(engine["engine_path"]) == engine["sha256"]
        trials = []
        for repeat in range(3):
            command = ["/usr/src/tensorrt/bin/trtexec", "--loadEngine=" + engine["engine_path"],
                       "--warmUp=500", "--iterations=2000", "--noDataTransfers",
                       "--useCudaGraph", "--memPoolSize=workspace:2048MiB"]
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
            filename = f"{label}_{repeat + 1}.log"
            (args.output_dir / filename).write_text(output)
            line = next(x for x in output.splitlines() if "GPU Compute Time: min" in x)
            values = {key: float(re.search(r"\b" + key + r" = ([0-9.eE+-]+)", line).group(1))
                      for key in ("min", "max", "mean", "median")}
            trials.append({**values, "command": command, "log_file": filename,
                           "log_sha256": sha(args.output_dir / filename)})
            print(label, repeat + 1, values["mean"], "ms", flush=True)
        report["engines"][label] = {**engine, "trials": trials,
                                    "mean_ms": statistics.mean(t["mean"] for t in trials)}
        report["clocks_after"] = clocks()
        (args.output_dir / "native_recommendations_timing.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
