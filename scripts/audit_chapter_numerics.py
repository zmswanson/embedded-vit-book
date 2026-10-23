#!/usr/bin/env python3
"""Recompute chapter statistics and recommendation provenance from saved results.

Run from any directory with Python 3. No model inference or training is performed.
Raw results and their historical filenames are preserved.
"""

import csv
import gzip
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/scalable_vits_for_embedded_systems"
SOURCES = {}


def rows(relative):
    path = ROOT / relative
    SOURCES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def json_evidence(relative):
    data = (ROOT / relative).read_bytes()
    SOURCES[relative] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def one(records, **criteria):
    matches = [r for r in records if all(r[k] == v for k, v in criteria.items())]
    assert len(matches) == 1, (criteria, len(matches))
    return matches[0]


def grouped(records, expected):
    result = {}
    for row in records:
        group = result.setdefault(row["model"], {})
        precision = row["precision"].upper()
        assert precision not in group, (row["model"], precision)
        group[precision] = row
    assert len(result) == expected
    return result


def quantization_stats(records, metric, expected):
    groups = grouped(records, expected)
    result = {"models": expected}
    for precision in ("FP16", "INT8"):
        changes = {
            model: Decimal(g[precision][metric]) - Decimal(g["FP32"][metric])
            for model, g in groups.items()
        }
        worst = min(changes, key=changes.get)
        result[precision] = {
            "mean_change": str(mean(changes.values())),
            "mean_absolute_change": str(mean(map(abs, changes.values()))),
            "median_absolute_change": str(median(map(abs, changes.values()))),
            "minimum_change": str(min(changes.values())),
            "maximum_change": str(max(changes.values())),
            "maximum_absolute_change": str(max(map(abs, changes.values()))),
            "largest_drop_model": worst,
            "changes_by_model": {k: str(v) for k, v in changes.items()},
        }
    return result


def main():
    quant_path = "quantization_results/quantization_summary.csv"
    pruned_path = "pruning_results/comprehensive_full_metrics.csv"
    unified_path = "inference_results/trt_unified_all16.csv"
    power_path = "inference_results/power_modes/trt_unified_power_sweep.csv"
    size_path = "quantization_results/all_models_quantization.csv"
    ablation_path = "inference_results/kd_alpha_ablation_results.csv"
    accuracy_path = "inference_results/onnx_accuracy_verification.csv"
    rail_path = "inference_results/power_modes/tegrastats_summary.csv"
    artifact_size_path = "inference_results/deployment_artifact_sizes.csv"
    engine_size_path = "inference_results/deployment_engine_sizes.json"
    compact_latency_path = "inference_results/deployment_int8_locked_latency.json"
    native_compact_dir = "inference_results/native_deit3_small_heads10/"
    native_compact_timing_path = native_compact_dir + "native_deit3_small_heads10.json"
    native_compact_accuracy_path = native_compact_dir + "native_deit3_small_heads10_accuracy.json"
    quant = rows(quant_path)
    pruned = rows(pruned_path)
    native = rows(unified_path)
    power = rows(power_path)
    sizes = rows(size_path)
    ablation = rows(ablation_path)
    accuracy = rows(accuracy_path)
    rail = rows(rail_path)
    artifact_sizes = rows(artifact_size_path)
    engine_sizes = json_evidence(engine_size_path)["engines"]
    deployment_measurements_path = "inference_results/native_recommendations_validation/deployment_measurements.json"
    deployment_measurements = json_evidence(deployment_measurements_path)
    for relative, digest in deployment_measurements["evidence_sha256"].items():
        SOURCES[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert SOURCES[relative] == digest
    assert len(deployment_measurements["results"]) == 5
    assert deployment_measurements["gallery_images"] == 4443
    assert deployment_measurements["probe_images"] == 3728
    assert len(engine_sizes) == 5
    for engine in engine_sizes:
        assert round(engine["size_bytes"] / 1024**2, 1) == engine["size_MiB"]
    native_compact_timing = json_evidence(native_compact_timing_path)
    native_compact_accuracy = json_evidence(native_compact_accuracy_path)
    native_compact_provenance = json_evidence(native_compact_dir + "provenance.json")
    calibrated_int8_dir = "inference_results/calibrated_native_deit3_small_heads10/"
    calibrated_int8 = json_evidence(calibrated_int8_dir + "comparison.json")
    profile_dir = "inference_results/native_int8_profiles/"
    native_int8_profile = json_evidence(profile_dir + "analysis.json")
    for filename, digest in native_int8_profile["evidence_sha256"].items():
        relative = profile_dir + filename
        SOURCES[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert SOURCES[relative] == digest
    assert native_int8_profile["compiled_graphs"]["required_int8"]["layer_types"]["Reformat"] == 51
    assert native_int8_profile["compiled_graphs"]["fp16_selected"]["layer_types"]["Reformat"] == 1
    for filename, digest in calibrated_int8["evidence_sha256"].items():
        relative = calibrated_int8_dir + filename
        SOURCES[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert SOURCES[relative] == digest
    assert calibrated_int8["verified_int8_projection_kernels"] == 48
    assert calibrated_int8["calibration_images"] == 512
    assert calibrated_int8["results"][0]["correct_probes"] == 2102
    assert calibrated_int8["results"][1]["correct_probes"] == 1977
    for filename, digest in native_compact_provenance["evidence_sha256"].items():
        relative = native_compact_dir + filename
        SOURCES[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert SOURCES[relative] == digest
    for filename, digest in native_compact_provenance.get("uncompressed_log_sha256", {}).items():
        archived = ROOT / native_compact_dir / filename
        assert hashlib.sha256(gzip.decompress(archived.read_bytes())).hexdigest() == digest
    for relative, digest in [
        (native_compact_provenance["native_timing_script"], native_compact_timing["benchmark_script_sha256"]),
        (native_compact_provenance["native_accuracy_script"], native_compact_accuracy["evaluation_script_sha256"]),
    ]:
        SOURCES[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert SOURCES[relative] == digest
    assert native_compact_accuracy["gallery_images"] == 4443
    assert native_compact_accuracy["probe_images"] == 3728
    native_compact_rank = native_compact_accuracy["engines"]["fp16"]
    assert native_compact_rank["rank1"] == native_compact_rank["correct_probes"] / 3728
    assert native_compact_rank["engine_sha256"] == native_compact_timing["engines"]["fp16"]["engine_sha256"]
    latency_bytes = (ROOT / compact_latency_path).read_bytes()
    SOURCES[compact_latency_path] = hashlib.sha256(latency_bytes).hexdigest()
    compact_latency = json.loads(latency_bytes)
    for evidence_path in [compact_latency["profile_path"], compact_latency["environment_source"],
                          compact_latency["benchmark_script"]]:
        SOURCES[evidence_path] = hashlib.sha256((ROOT / evidence_path).read_bytes()).hexdigest()
    assert SOURCES[compact_latency["benchmark_script"]] == compact_latency["benchmark_script_sha256"]
    profile_bytes = gzip.decompress((ROOT / compact_latency["profile_path"]).read_bytes())
    assert hashlib.sha256(profile_bytes).hexdigest() == compact_latency["profile_sha256"]
    environment = json.loads((ROOT / compact_latency["environment_source"]).read_text())
    assert "MAXN_SUPER" in environment["nvpmodel"]
    assert environment["cpu"] and all(
        cpu["scaling_min_freq"] == cpu["scaling_max_freq"] == "1728000"
        for cpu in environment["cpu"].values()
    )
    assert environment["gpu"] and all(
        gpu["min_freq"] == gpu["max_freq"] == "1020000000"
        for gpu in environment["gpu"].values()
    )
    fallback = rows("inference_results/onnxrt_trt_ep_all7.csv")
    dynamic_gpu = rows("inference_results/onnxrt_int8_all7.csv")
    q_stats = quantization_stats(quant, "rank_at_1", 4)
    p_stats = quantization_stats(pruned, "rank@1", 24)
    speedups = {
        model: Decimal(g["FP32"]["mean_ms"]) / Decimal(g["FP16"]["mean_ms"])
        for model, g in grouped(native, 16).items()
    }
    native_fp16 = {r["model"]: Decimal(r["mean_ms"]) for r in native if r["precision"] == "fp16"}
    fallback_ratios = {
        r["model"]: Decimal(r["mean_ms"]) / native_fp16[r["model"]]
        for r in fallback if r["ep"] == "TRT" and r["precision"] == "FP16"
    }
    dynamic_ratios = {
        model: Decimal(g["INT8"]["mean_ms"]) / Decimal(g["FP32"]["mean_ms"])
        for model, g in grouped(dynamic_gpu, 7).items()
    }
    assert all(r["backend"] == "TRT_NATIVE" for r in native + power)
    sweep = {}
    for row in power:
        sweep.setdefault((row["model"], row["precision"]), {})[
            row["power_mode_label"]
        ] = row
    assert len(sweep) == 14 and len(power) == 42
    retention = [
        Decimal(g["25W"]["throughput_fps"]) / Decimal(g["MAXN_SUPER"]["throughput_fps"])
        for g in sweep.values()
    ]

    priorities = [
        ("highest observed student accuracy", "swin_tiny_kd"),
        ("smaller faster KD model", "swin_tiny_kd_blocks2"),
        ("lowest latency in uniform FP16 baseline survey", "resnet50d"),
        ("compact native TensorRT model", "deit3_small_fp16"),
        ("smallest native engine, with accuracy and latency tradeoff", "deit3_small_int8"),
    ]
    output_rows = []
    manuscript = (OUT / "bookchapter.tex").read_text()
    for priority, label in priorities:
        result = one(deployment_measurements["results"], label=label)
        engine = one(engine_sizes, model=result["model"], precision=result["precision"])
        assert result["engine_sha256"] == engine["sha256"]
        assert result["engine_MiB"] == engine["size_MiB"]
        ram_low, ram_high = (
            str(Decimal(str(result[key])).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            for key in ("incremental_system_RAM_MiB_min", "peak_incremental_system_RAM_MiB")
        )
        printed_values = (
            f" & {result['native_rank1']:.3f} & {result['native_gpu_mean_ms']:.2f}"
            f" & {result['engine_MiB']:.1f} & {ram_low}--{ram_high}"
            f" & {result['module_energy_mJ_per_inference']:.1f}"
        )
        assert printed_values in manuscript, (label, printed_values)
        output_rows.append(dict(
            priority=priority, model=result["model"], precision=result["precision"],
            rank1=result["native_rank1"],
            accuracy_evaluation="Native TensorRT, chapter gallery/probe protocol without distractors",
            latency_ms=result["native_gpu_mean_ms"], latency_backend="TRT_NATIVE",
            latency_scope="GPU compute only, no transfers",
            native_trt_fp16_mean_ms=result["native_gpu_mean_ms"] if result["precision"] == "FP16" else "",
            engine_file_size_MiB=result["engine_MiB"],
            peak_incremental_system_RAM_MiB=result["peak_incremental_system_RAM_MiB"],
            incremental_system_RAM_MiB_min=result["incremental_system_RAM_MiB_min"],
            mean_module_power_W=result["mean_module_power_W"],
            module_energy_mJ_per_inference=result["module_energy_mJ_per_inference"],
            accuracy_source="inference_results/native_recommendations_validation/native_recommendations_accuracy.json",
            latency_source="inference_results/native_recommendations_validation/timing/native_recommendations_timing.json",
            size_source=engine_size_path, resource_source=deployment_measurements_path,
        ))
    report = {
        "scope": "Reanalysis of saved results, audited export sizes, and new Jetson native accuracy, GPU latency, incremental RAM and per-inference module energy measurements; no retraining.",
        "representative_unpruned_quantization": q_stats,
        "pruned_quantization": p_stats,
        "pruning_figure_sources": {
            prefix: [r for r in pruned if r["precision"] == "FP32" and r["model"].startswith(prefix)]
            for prefix in ("swin_tiny_baseline-", "deit3_small_baseline-",
                           "swin_base_baseline-", "deit3_base_lora-")
        },
        "unified_fp16_speedup": {
            "models": 16, "minimum": str(min(speedups.values())),
            "maximum": str(max(speedups.values())),
            "per_model": {k: str(v) for k, v in speedups.items()},
        },
        "fp16_ort_trt_ep_over_native_ratio": {k: str(v) for k, v in fallback_ratios.items()},
        "dynamic_int8_over_fp32_cuda_ep_ratio": {k: str(v) for k, v in dynamic_ratios.items()},
        "power_sweep": {
            "configurations": 14, "throughput_retained_at_25W_min": str(min(retention)),
            "throughput_retained_at_25W_max": str(max(retention)),
            "input_rail_measurements": rail,
            "power_scope": "Pooled per-mode sweep; not per-model power or a maximum draw.",
        },
        "recommendations": output_rows,
        "native_deployment_validation": deployment_measurements,
        "deployment_artifact_sizes": artifact_sizes,
        "deployment_engine_sizes": engine_sizes,
        "compact_dynamic_int8_latency": compact_latency,
        "compact_native_tensor_rt_timing": native_compact_timing,
        "compact_native_tensor_rt_accuracy": native_compact_accuracy,
        "compact_calibrated_native_int8_comparison": calibrated_int8,
        "native_int8_slowdown_profiles": native_int8_profile,
        "historical_filename_corrections": {
            "deit3_base_lora (pruning)": "No-adapter baseline vimriwcl, not hybrid 2ljjzfzb.",
            "swin_base (early pruning)": "Hybrid LoRA ci53ufy8, not baseline xd7hukv3.",
            "evidence": ["shell_scripts/prune_and_finetune.sh", "shell_scripts/prune_comprehensive.sh"],
        },
        "source_sha256": SOURCES,
    }
    # Explicit checks protect the corrected manuscript bounds and recommendation mapping.
    assert q_stats["INT8"]["minimum_change"] == "-0.0016"
    assert p_stats["INT8"]["minimum_change"] == "-0.0099"
    assert p_stats["INT8"]["median_absolute_change"] == "0.0013"
    assert p_stats["FP16"]["maximum_absolute_change"] == "0.0008"
    assert sum(v > 1 for v in dynamic_ratios.values()) == 6
    assert max(ablation, key=lambda r: Decimal(r["tinyface/test/rank@1"]))["wandb_id"] == "zz76v2nu"
    with (OUT / "deployment_recommendations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0])
        writer.writeheader()
        writer.writerows(output_rows)
    (OUT / "numerical_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Verified 4 unpruned models, 24 pruned models, 16 native architectures, and 14 power-sweep configurations.")
    print("Saved numerical_verification.json and deployment_recommendations.csv.")


if __name__ == "__main__":
    main()
