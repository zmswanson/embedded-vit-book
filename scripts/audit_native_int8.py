#!/usr/bin/env python3
"""Verify the calibrated native INT8 experiment and summarize paired results."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "inference_results/calibrated_native_deit3_small_heads10"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(FOLDER / "calibration_manifest.json")
    first = read(FOLDER / "build_provenance.json")
    folder = FOLDER / "required_int8"
    build = read(folder / "build_provenance.json")
    timing = read(folder / "native_deit3_small_heads10.json")
    accuracy = read(folder / "native_deit3_small_heads10_accuracy.json")
    names = read(FOLDER / "int8_layer_names.json")
    assert manifest["count"] == len(manifest["images"]) == 512
    assert len({x["identity"] for x in manifest["images"]}) == 512
    assert all(x["file"].startswith("Training_Set/") for x in manifest["images"])
    assert manifest["test_images_used"] == manifest["test_identity_overlap"] == manifest["heldout_identity_overlap"] == 0
    assert manifest["training_splits"] == list(range(9))
    assert sha(ROOT / "scripts/prepare_native_calibration.py") == manifest["preparation_script_sha256"]
    assert sha(ROOT / "datasets/tinyface/cross_val_splits.json") == manifest["split_file_sha256"]
    assert sha(FOLDER / "opportunistic_build_script.py") == first["build_script_sha256"]
    assert sha(ROOT / "scripts/build_calibrated_native_int8.py") == build["build_script_sha256"]
    assert sha(folder / "build_script.py") == build["build_script_sha256"]
    assert first["calibration_cache_sha256"] == build["calibration_cache_sha256"] == sha(folder / "entropy_calibration.cache")
    assert first["calibration_manifest_sha256"] == build["calibration_manifest_sha256"] == sha(FOLDER / "calibration_manifest.json")
    assert first["onnx_sha256"] == build["onnx_sha256"] == names["onnx_sha256"]
    assert first["external_weights_sha256"] == build["external_weights_sha256"]
    assert build["calibration_tensor_file_sha256"] == manifest["tensor_file_sha256"]
    assert build["calibration_count"] == 512
    assert "OBEY_PRECISION_CONSTRAINTS" in build["builder_flags"]
    layers = read(folder / "engine_inspector.json")["Layers"]
    integer = [x for x in layers if x.get("Weights", {}).get("Type") == "Int8"]
    assert len(integer) == 48
    assert all(any("Int8" in x.get("Format/Datatype", "") for x in layer["Inputs"]) for layer in integer)
    assert all("_i8" in layer["TacticName"] for layer in integer)
    expected = set(names["learned_linear_layer_names"])
    assert len(expected) == 48 and expected == set(build["int8_required_layers"])
    assert all(sum(f"[ONNX Layer: {name}]" in layer.get("Metadata", "") for layer in integer) == 1
               for name in expected)
    assert not any("int8" in json.dumps(layer).lower()
                   for layer in read(FOLDER / "engine_inspector.json")["Layers"])
    assert accuracy["gallery_images"] == 4443 and accuracy["probe_images"] == 3728
    assert sha(ROOT / "scripts/evaluate_native_deployment.py") == accuracy["evaluation_script_sha256"]
    assert sha(ROOT / "scripts/benchmark_native_deployment.py") == timing["benchmark_script_sha256"]
    assert "MAXN_SUPER" in timing["nvpmodel"]
    assert timing["warmup_ms"] == 500 and timing["minimum_iterations_per_repeat"] == 2000
    for phase in ("clocks_before", "clocks_after"):
        assert all(v["scaling_min_freq"] == v["scaling_max_freq"] == "1728000"
                   for v in timing[phase]["cpu"].values())
        assert all(v["min_freq"] == v["max_freq"] == "1020000000"
                   for v in timing[phase]["gpu"].values())
    rows = []
    truth = None
    for label in ("fp16", "calibrated_int8"):
        t, a = timing["engines"][label], accuracy["engines"][label]
        assert t["engine_sha256"] == a["engine_sha256"]
        if label == "calibrated_int8":
            assert t["engine_sha256"] == build["engine_sha256"]
            assert t["engine_bytes"] == build["engine_bytes"]
        assert len(t["trials"]) == 3
        assert t["mean_ms"] == mean(x["mean"] for x in t["trials"])
        assert a["rank1"] == a["correct_probes"] / 3728
        predictions = read(folder / a["probe_predictions_file"])
        if truth is None:
            truth = predictions["true_identities"]
        assert truth == predictions["true_identities"]
        assert sum(x == y for x, y in zip(truth, predictions["predicted_identities"])) == a["correct_probes"]
        assert len(predictions["predicted_identities"]) == 3728
        rows.append({"precision": label, "gpu_compute_mean_ms": t["mean_ms"],
                     "rank1": a["rank1"], "correct_probes": a["correct_probes"],
                     "engine_bytes": t["engine_bytes"], "engine_MiB": t["engine_bytes"] / 1024**2,
                     "engine_sha256": t["engine_sha256"]})
    # Retain raw logs and compressed copies for easy transfer and review.
    for log in FOLDER.rglob("*.log"):
        compressed = log.with_suffix(log.suffix + ".gz")
        if not compressed.exists():
            compressed.write_bytes(gzip.compress(log.read_bytes(), mtime=0))
    evidence = {str(p.relative_to(FOLDER)): sha(p) for p in sorted(FOLDER.rglob("*"))
                if p.is_file() and p.name not in ("comparison.json", "comparison.csv")
                and p.suffix != ".log"}
    report = {"model": timing["model"], "tensorrt_version": build["tensorrt_version"],
              "calibration_images": manifest["count"], "calibration_seed": manifest["seed"],
              "verified_int8_projection_kernels": len(integer),
              "precision_scope": "INT8 learned linear projections; floating-point normalization, attention, and other layers",
              "protocol": accuracy["protocol"], "gallery_images": accuracy["gallery_images"],
              "probe_images": accuracy["probe_images"],
              "latency_scope": timing["latency_scope"], "results": rows,
              "int8_over_fp16_latency_ratio": rows[1]["gpu_compute_mean_ms"] / rows[0]["gpu_compute_mean_ms"],
              "int8_minus_fp16_rank1": rows[1]["rank1"] - rows[0]["rank1"],
              "int8_over_fp16_engine_size_ratio": rows[1]["engine_bytes"] / rows[0]["engine_bytes"],
              "evidence_sha256": evidence}
    (FOLDER / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    with (FOLDER / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in report.items() if k != "evidence_sha256"}, indent=2))


if __name__ == "__main__":
    main()
