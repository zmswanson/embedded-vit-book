#!/usr/bin/env python3
"""Evaluate the five deployment engines on the chapter's existing image set.

Uses the same preprocessing and CUDA inference implementation as the earlier
native DeiT3-S evaluation. Gallery distractors are outside this experiment.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from evaluate_native_deployment import EngineRunner, preprocess


LABELS = ("swin_tiny_kd", "swin_tiny_kd_blocks2", "resnet50d",
          "deit3_small_fp16", "deit3_small_int8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.engines.read_text())["engines"]
    assert len(records) == len(LABELS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runners, features = {}, {}
    for label, record in zip(LABELS, records):
        assert hashlib.sha256(Path(record["engine_path"]).read_bytes()).hexdigest() == record["sha256"]
        runners[label] = EngineRunner(record["engine_path"])
        features[label] = []
    with zipfile.ZipFile(args.images) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["image_encoding"] == "Lossless PNG of OpenCV-decoded RGB pixels"
        entries = manifest["images"]
        for index, entry in enumerate(entries):
            image = preprocess(archive.read(entry["file"]))
            for label, runner in runners.items():
                features[label].append(runner(image))
            if (index + 1) % 1000 == 0:
                print("Evaluated", index + 1, "of", len(entries), "images through all five engines", flush=True)
    gallery = np.array([entry["split"] == "Gallery_Match" for entry in entries])
    probes = np.array([entry["split"] == "Probe" for entry in entries])
    assert gallery.sum() == 4443 and probes.sum() == 3728
    identities = np.array([entry["identity"] for entry in entries])
    report = {"measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": manifest["protocol"], "gallery_images": int(gallery.sum()),
              "probe_images": int(probes.sum()), "input_shape": [1, 3, 96, 96],
              "image_archive_sha256": hashlib.sha256(args.images.read_bytes()).hexdigest(),
              "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "inference_helper_sha256": hashlib.sha256(Path(__file__).with_name("evaluate_native_deployment.py").read_bytes()).hexdigest(),
              "engines": {}}
    for label, record in zip(LABELS, records):
        vectors = np.stack(features[label])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        assert np.all(norms > 0) and np.isfinite(vectors).all()
        vectors /= norms
        g, p = vectors[gallery], vectors[probes]
        indices = np.concatenate([np.argmax(p[start:start + 128] @ g.T, axis=1)
                                  for start in range(0, len(p), 128)])
        predicted = identities[gallery][indices]
        truth = identities[probes]
        correct = predicted == truth
        filename = label + "_predictions.json"
        (args.output_dir / filename).write_text(json.dumps(
            {"probe_files": [x["file"] for x in entries if x["split"] == "Probe"],
             "true_identities": truth.tolist(), "predicted_identities": predicted.tolist(),
             "gallery_indices": indices.tolist(), "correct": correct.tolist()}, separators=(",", ":")) + "\n")
        report["engines"][label] = {**record, "rank1": float(correct.mean()),
                                    "correct_probes": int(correct.sum()),
                                    "probe_predictions_file": filename,
                                    "embedding_dimension": int(vectors.shape[1])}
        print(label, "rank-1", float(correct.mean()), "correct probes", int(correct.sum()), flush=True)
    (args.output_dir / "native_recommendations_accuracy.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
