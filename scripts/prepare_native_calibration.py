#!/usr/bin/env python3
"""Prepare deterministic, identity-balanced training-only TensorRT calibration."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    splits = json.loads(args.splits.read_text())
    train_ids = set.union(*(set(splits[f"TRAIN_SET_{i}"]) for i in range(9)))
    heldout_ids = set(splits["TRAIN_SET_9"])
    test_ids = {p.name.split("_", 1)[0] for split in ("Gallery_Match", "Probe")
                for p in (args.dataset / "Testing_Set" / split).rglob("*.jpg")}
    assert train_ids.isdisjoint(heldout_ids | test_ids)
    groups = {}
    for path in sorted((args.dataset / "Training_Set").rglob("*.jpg")):
        identity = path.name.split("_", 1)[0]
        if identity in train_ids:
            groups.setdefault(identity, []).append(path)
    rng = random.Random(args.seed)
    identities = rng.sample(sorted(groups), args.count)
    paths = [rng.choice(groups[identity]) for identity in identities]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensors = np.empty((len(paths), 3, 96, 96), dtype=np.float32)
    entries = []
    for index, path in enumerate(paths):
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((96, 96), Image.Resampling.BILINEAR)
        image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
        tensors[index] = ((image - np.array([.485, .456, .406], dtype=np.float32)[:, None, None])
                          / np.array([.229, .224, .225], dtype=np.float32)[:, None, None])
        entries.append({"file": str(path.relative_to(args.dataset)), "identity": identities[index],
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "tensor_sha256": hashlib.sha256(tensors[index].tobytes()).hexdigest()})
    np.save(args.output_dir / "calibration.npy", tensors, allow_pickle=False)
    manifest = {"dataset": "TinyFace Training_Set", "training_splits": list(range(9)),
                "excluded_validation_split": 9, "test_images_used": 0,
                "test_identity_overlap": len(set(identities) & test_ids),
                "heldout_identity_overlap": len(set(identities) & heldout_ids),
                "selection": "512 distinct training identities, one random image per identity",
                "seed": args.seed, "count": len(paths), "shape": list(tensors.shape),
                "preprocessing": "OpenCV RGB decode; PIL bilinear 96x96; ImageNet normalization",
                "split_file_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
                "preparation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "tensor_file_sha256": hashlib.sha256((args.output_dir / "calibration.npy").read_bytes()).hexdigest(),
                "images": entries}
    (args.output_dir / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "images"}, indent=2))


if __name__ == "__main__":
    main()
