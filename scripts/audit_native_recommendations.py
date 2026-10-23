#!/usr/bin/env python3
"""Verify native deployment accuracy, RAM, energy traces, and refreshed timing."""

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics

from measure_native_resources import LABELS, integral, sha


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inference_results/native_recommendations_validation"
RESOURCE = ROOT / "inference_results/native_recommendations_resources"
TIMING = OUT / "timing"


def read(path):
    return json.loads(path.read_text())


def locked(state):
    assert state["cpu"] and state["gpu"]
    assert all(x["scaling_min_freq"] == x["scaling_max_freq"] == "1728000" for x in state["cpu"].values())
    assert all(x["min_freq"] == x["max_freq"] == "1020000000" for x in state["gpu"].values())


def main():
    accuracy = read(OUT / "native_recommendations_accuracy.json")
    resources = read(RESOURCE / "native_resources.json")
    timing = read(TIMING / "native_recommendations_timing.json")
    manifest_path = ROOT / "inference_results/deployment_engine_sizes.json"
    engines = read(manifest_path)["engines"]
    assert accuracy["gallery_images"] == 4443 and accuracy["probe_images"] == 3728
    assert accuracy["evaluation_script_sha256"] == sha(ROOT / "scripts/evaluate_native_recommendations.py")
    assert accuracy["inference_helper_sha256"] == sha(ROOT / "scripts/evaluate_native_deployment.py")
    assert resources["script_sha256"] == sha(ROOT / "scripts/measure_native_resources.py")
    assert resources["inference_helper_sha256"] == accuracy["inference_helper_sha256"]
    assert timing["script_sha256"] == sha(ROOT / "scripts/benchmark_native_recommendations.py")
    assert timing["helper_script_sha256"] == resources["script_sha256"]
    assert resources["engine_manifest_sha256"] == timing["engine_manifest_sha256"] == sha(manifest_path)
    assert resources["image_archive_sha256"] == accuracy["image_archive_sha256"]
    assert len(resources["trials"]) == 15
    assert resources["duration_seconds"] == 45 and resources["warmup_seconds"] == 10
    assert resources["sample_interval_seconds"] == .1
    for report in (resources, timing):
        assert "MAXN_SUPER" in report["nvpmodel"]
        locked(report["clocks_before"]); locked(report["clocks_after"])
    reference = read(ROOT / "inference_results/calibrated_native_deit3_small_heads10/comparison.json")
    assert accuracy["engines"]["deit3_small_fp16"]["correct_probes"] == reference["results"][0]["correct_probes"]
    assert accuracy["engines"]["deit3_small_int8"]["correct_probes"] == reference["results"][1]["correct_probes"]
    rows, truth = [], None
    inputs_by_repeat = {}
    for label, engine in zip(LABELS, engines):
        a, t = accuracy["engines"][label], timing["engines"][label]
        assert a["sha256"] == t["sha256"] == engine["sha256"]
        pred = read(OUT / a["probe_predictions_file"])
        if truth is None:
            truth = pred["true_identities"]
        assert len(truth) == 3728 and truth == pred["true_identities"]
        assert len(pred["predicted_identities"]) == 3728
        correct = [x == y for x, y in zip(truth, pred["predicted_identities"])]
        assert correct == pred["correct"] and sum(correct) == a["correct_probes"]
        assert a["rank1"] == a["correct_probes"] / 3728
        assert len(t["trials"]) == 3
        assert t["mean_ms"] == statistics.mean(x["mean"] for x in t["trials"])
        for trial in t["trials"]:
            log = TIMING / trial["log_file"]
            archive = log.with_suffix(log.suffix + ".gz")
            digest = (hashlib.sha256(gzip.decompress(archive.read_bytes())).hexdigest()
                      if archive.exists() else sha(log))
            assert digest == trial["log_sha256"]
        trials = [x for x in resources["trials"] if x["label"] == label]
        assert sorted(x["repeat"] for x in trials) == [1, 2, 3]
        for trial in trials:
            assert trial["sha256"] == engine["sha256"]
            locked(trial["clocks_before"]); locked(trial["clocks_after"])
            assert trial["completed_inferences"] > 0 and trial["completed_inferences"] % 64 == 0
            assert 45 <= trial["elapsed_seconds"] < 46
            assert trial["minimum_observed_gpu_clock_hz"] == 1020000000
            trace_path = RESOURCE / trial["trace_file"]
            assert sha(trace_path) == trial["trace_sha256"]
            samples = [{k: float(v) for k, v in x.items()} for x in
                       csv.DictReader(io.StringIO(gzip.decompress(trace_path.read_bytes()).decode()))]
            assert max(b["time"] - a["time"] for a, b in zip(samples, samples[1:])) < .2
            assert samples[0]["time"] < trial["start"] < trial["end"] < samples[-1]["time"]
            assert all(abs(s["voltage_mV"] * s["current_mA"] / 1e6 - s["module_power_W"]) < 1e-10
                       for s in samples)
            input_key = (trial["input_file"], trial["input_tensor_sha256"])
            assert inputs_by_repeat.setdefault(trial["repeat"], input_key) == input_key
            energy = integral(samples, trial["start"], trial["end"])
            assert abs(energy - trial["module_energy_joules"]) < 1e-8
            assert abs(1000 * energy / trial["completed_inferences"] - trial["module_energy_mJ_per_inference"]) < 1e-9
            active = [s for s in samples if trial["start"] <= s["time"] <= trial["end"]]
            assert len(active) == trial["active_samples"]
            ram = (trial["baseline_mem_available_kib"] - min(s["mem_available_kib"] for s in active)) / 1024
            assert ram == trial["peak_incremental_system_RAM_MiB"]
            assert ram > 0 and trial["baseline_mem_noise_MiB"] < 20
            events = trial["tensor_rt_allocation_events"]
            live = peak = 0
            for event in events:
                live += event["bytes"] if event["action"] == "allocate" else -event["bytes"]
                assert live == event["live_bytes"] and live >= 0
                peak = max(peak, live)
            assert peak == trial["tensor_rt_peak_allocation_bytes"]
        energy = sum(x["module_energy_joules"] for x in trials)
        duration = sum(x["elapsed_seconds"] for x in trials)
        count = sum(x["completed_inferences"] for x in trials)
        rows.append({"label": label, "model": engine["model"], "precision": engine["precision"],
                     "native_rank1": a["rank1"], "correct_probes": a["correct_probes"],
                     "native_gpu_mean_ms": t["mean_ms"], "engine_MiB": engine["size_MiB"],
                     "peak_incremental_system_RAM_MiB": max(x["peak_incremental_system_RAM_MiB"] for x in trials),
                     "incremental_system_RAM_MiB_min": min(x["peak_incremental_system_RAM_MiB"] for x in trials),
                     "baseline_memory_variation_MiB_max": max(x["baseline_mem_noise_MiB"] for x in trials),
                     "tensor_rt_peak_allocation_MiB": max(x["tensor_rt_peak_allocation_MiB"] for x in trials),
                     "peak_host_RSS_MiB": max(x["peak_host_RSS_MiB"] for x in trials),
                     "mean_module_power_W": energy / duration,
                     "module_energy_mJ_per_inference": 1000 * energy / count,
                     "sustained_throughput_per_second": count / duration,
                     "energy_mJ_per_inference_min": min(x["module_energy_mJ_per_inference"] for x in trials),
                     "energy_mJ_per_inference_max": max(x["module_energy_mJ_per_inference"] for x in trials),
                     "engine_sha256": engine["sha256"]})
    evidence = {}
    for folder in (OUT, RESOURCE):
        for p in folder.rglob("*.log"):
            zipped = p.with_suffix(".log.gz")
            if not zipped.exists():
                zipped.write_bytes(gzip.compress(p.read_bytes(), mtime=0))
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.name not in ("deployment_measurements.json", "deployment_measurements.csv") and p.suffix != ".log":
                evidence[str(p.relative_to(ROOT))] = sha(p)
    for file in ("evaluate_native_recommendations.py", "evaluate_native_deployment.py",
                 "measure_native_resources.py", "benchmark_native_recommendations.py",
                 "audit_native_recommendations.py"):
        p = ROOT / "scripts" / file
        evidence[str(p.relative_to(ROOT))] = sha(p)
    report = {"protocol": accuracy["protocol"], "gallery_images": 4443, "probe_images": 3728,
              "input_shape": [1, 3, 96, 96], "repeats": 3, "resource_duration_seconds_per_trial": 45,
              "power_scope": resources["power_scope"], "memory_scope": resources["memory_scope"],
              "latency_scope": timing["latency_scope"], "results": rows, "evidence_sha256": evidence}
    (OUT / "deployment_measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "deployment_measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
