#!/usr/bin/env python3
"""Evaluate native TensorRT engines on an explicitly recorded gallery/probe set.

This follows the chapter's existing Gallery_Match + Probe protocol, without
distractors. It does not claim the complete published TinyFace protocol.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import tensorrt as trt


class EngineRunner:
    def __init__(self, path):
        self.cuda = ctypes.CDLL("libcudart.so.12")
        self.cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.cuda.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.cuda.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.check(self.cuda.cudaSetDevice(0))
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize {path}")
        self.context = self.engine.create_execution_context()
        inputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                  if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        outputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                   if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        assert len(inputs) == len(outputs) == 1
        self.input_name, self.output_name = inputs[0], outputs[0]
        self.context.set_input_shape(self.input_name, (1, 3, 96, 96))
        shape = tuple(self.context.get_tensor_shape(self.output_name))
        assert all(d > 0 for d in shape), shape
        self.input_host = np.empty(3 * 96 * 96, trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
        self.output_host = np.empty(int(np.prod(shape)), trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
        self.input_device, self.output_device, self.stream = (ctypes.c_void_p() for _ in range(3))
        self.check(self.cuda.cudaMalloc(ctypes.byref(self.input_device), self.input_host.nbytes))
        self.check(self.cuda.cudaMalloc(ctypes.byref(self.output_device), self.output_host.nbytes))
        self.check(self.cuda.cudaStreamCreate(ctypes.byref(self.stream)))
        self.context.set_tensor_address(self.input_name, self.input_device.value)
        self.context.set_tensor_address(self.output_name, self.output_device.value)

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"CUDA runtime returned error {code}")

    def __call__(self, image):
        np.copyto(self.input_host, image.ravel(), casting="same_kind")
        self.check(self.cuda.cudaMemcpy(self.input_device, self.input_host.ctypes.data, self.input_host.nbytes, 1))
        if not self.context.execute_async_v3(self.stream.value):
            raise RuntimeError("TensorRT execution failed")
        self.check(self.cuda.cudaStreamSynchronize(self.stream))
        self.check(self.cuda.cudaMemcpy(self.output_host.ctypes.data, self.output_device, self.output_host.nbytes, 2))
        result = self.output_host.astype(np.float32).copy()
        if not np.isfinite(result).all():
            raise RuntimeError("Engine produced non-finite features")
        return result


def preprocess(encoded):
    image = Image.open(io.BytesIO(encoded)).convert("RGB")
    image = image.resize((96, 96), getattr(Image, "Resampling", Image).BILINEAR)
    image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray((image - mean) / std)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--engine", action="append", required=True, help="label=engine_path")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = dict(spec.split("=", 1) for spec in args.engine)
    runners = {label: EngineRunner(path) for label, path in paths.items()}
    features = {label: [] for label in runners}
    with zipfile.ZipFile(args.images) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["image_encoding"] == "Lossless PNG of OpenCV-decoded RGB pixels"
        entries = manifest["images"]
        for index, entry in enumerate(entries):
            image = preprocess(archive.read(entry["file"]))
            for label, runner in runners.items():
                features[label].append(runner(image))
            if (index + 1) % 1000 == 0:
                print("Evaluated images", index + 1, "of", len(entries), flush=True)
    gallery = np.array([entry["split"] == "Gallery_Match" for entry in entries])
    probes = np.array([entry["split"] == "Probe" for entry in entries])
    identities = np.array([entry["identity"] for entry in entries])
    report = {"measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "model": "deit3_small_baseline-heads_0.10", "tensorrt_version": trt.__version__,
              "protocol": manifest["protocol"], "gallery_images": int(gallery.sum()),
              "probe_images": int(probes.sum()),
              "image_archive_sha256": hashlib.sha256(args.images.read_bytes()).hexdigest(),
              "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "preprocessing": "OpenCV-decoded RGB pixels; PIL bilinear resize to 96x96; ImageNet normalization",
              "engines": {}}
    normalized = {}
    for label, vectors in features.items():
        array = np.stack(vectors)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        assert np.all(norms > 0)
        array = array / norms
        normalized[label] = array
        g, p = array[gallery], array[probes]
        predictions = []
        for start in range(0, len(p), 128):
            predictions.extend(np.argmax(p[start:start + 128] @ g.T, axis=1))
        predicted = identities[gallery][predictions]
        correct = predicted == identities[probes]
        entry = {"engine_path": paths[label],
                 "engine_sha256": hashlib.sha256(Path(paths[label]).read_bytes()).hexdigest(),
                 "rank1": float(np.mean(correct)), "correct_probes": int(correct.sum()),
                 "probe_predictions_file": label + "_predictions.json"}
        (args.output_dir / entry["probe_predictions_file"]).write_text(json.dumps(
            {"probe_files": [e["file"] for e in entries if e["split"] == "Probe"],
             "true_identities": identities[probes].tolist(), "predicted_identities": predicted.tolist(),
             "correct": correct.tolist()}, separators=(",", ":")) + "\n")
        report["engines"][label] = entry
        print(label, "native TensorRT rank-1", entry["rank1"], flush=True)
    if "fp16" in normalized:
        report["embedding_cosine_to_native_fp16"] = {
            label: {"mean": float(np.mean(np.sum(array * normalized["fp16"], axis=1))),
                    "minimum": float(np.min(np.sum(array * normalized["fp16"], axis=1)))}
            for label, array in normalized.items() if label != "fp16"
        }
    (args.output_dir / "native_deit3_small_heads10_accuracy.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
