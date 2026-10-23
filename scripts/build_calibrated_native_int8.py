#!/usr/bin/env python3
"""Build an audited TensorRT 10.3 entropy-calibrated INT8/FP16 engine.

Uses the native calibrator supported by the installed JetPack version; no
random activation ranges or evaluation images are used for calibration.
Detailed inspector output records the actual tensor formats in the engine.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorrt as trt


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, tensors, cache, reuse_cache=None):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.tensors = np.load(tensors, mmap_mode="r", allow_pickle=False)
        assert self.tensors.dtype == np.float32 and self.tensors.shape[1:] == (3, 96, 96)
        self.cache = cache
        self.reuse_cache = reuse_cache
        self.index = 0
        self.cuda = ctypes.CDLL("libcudart.so.12")
        self.cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.device = ctypes.c_void_p()
        assert self.cuda.cudaMalloc(ctypes.byref(self.device), self.tensors[0].nbytes) == 0

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        assert len(names) == 1
        if self.index == len(self.tensors):
            return None
        tensor = np.ascontiguousarray(self.tensors[self.index])
        assert self.cuda.cudaMemcpy(self.device, tensor.ctypes.data, tensor.nbytes, 1) == 0
        self.index += 1
        if self.index % 64 == 0:
            print("Calibration images", self.index, flush=True)
        return [self.device.value]

    def read_calibration_cache(self):
        return self.reuse_cache.read_bytes() if self.reuse_cache else None

    def write_calibration_cache(self, cache):
        self.cache.write_bytes(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-int8-linear", action="store_true")
    parser.add_argument("--int8-layer-names", type=Path)
    parser.add_argument("--reuse-calibration-build", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine_path = args.output_dir / "deit3_small_heads10_calibrated_int8.engine"
    if engine_path.exists():
        raise FileExistsError(engine_path)
    manifest_path = args.calibration_dir / "calibration_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tensors = args.calibration_dir / "calibration.npy"
    assert digest(tensors) == manifest["tensor_file_sha256"]
    assert manifest["test_images_used"] == manifest["test_identity_overlap"] == manifest["heldout_identity_overlap"] == 0
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    assert builder.platform_has_fast_int8
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(args.onnx)):
        raise RuntimeError("\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)))
    assert network.num_inputs == 1
    inp = network.get_input(0)
    shape = (1, 3, 96, 96)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2048 * 1024**2)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    constrained = []
    if args.require_int8_linear:
        if not args.int8_layer_names:
            parser.error("--int8-layer-names is required with --require-int8-linear")
        names = json.loads(args.int8_layer_names.read_text())
        assert names["onnx_sha256"] == digest(args.onnx)
        expected = set(names["learned_linear_layer_names"])
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.name in expected:
                assert layer.type == trt.LayerType.MATRIX_MULTIPLY
                layer.precision = trt.int8
                constrained.append(layer.name)
        assert len(constrained) == 48 and set(constrained) == expected, constrained
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        print("INT8-required learned projections", constrained, flush=True)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, shape, shape, shape)
    assert bool(profile)
    config.add_optimization_profile(profile)
    assert config.set_calibration_profile(profile)
    cache = args.output_dir / "entropy_calibration.cache"
    reuse_cache = None
    if args.reuse_calibration_build:
        previous = json.loads((args.reuse_calibration_build / "build_provenance.json").read_text())
        reuse_cache = args.reuse_calibration_build / "entropy_calibration.cache"
        assert digest(reuse_cache) == previous["calibration_cache_sha256"]
        assert digest(args.onnx) == previous["onnx_sha256"]
        assert digest(str(args.onnx) + ".data") == previous["external_weights_sha256"]
        assert digest(manifest_path) == previous["calibration_manifest_sha256"]
        cache.write_bytes(reuse_cache.read_bytes())
    calibrator = Calibrator(tensors, cache, reuse_cache)
    config.int8_calibrator = calibrator
    print("Building calibrated native INT8/FP16 engine", flush=True)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    assert calibrator.index == (0 if reuse_cache else manifest["count"])
    engine_path.write_bytes(bytes(plan))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    inspector = engine.create_engine_inspector()
    inspection = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    (args.output_dir / "engine_inspector.json").write_text(inspection + "\n")
    report = {"built_at_utc": datetime.now(timezone.utc).isoformat(),
              "tensorrt_version": trt.__version__, "platform_has_fast_int8": builder.platform_has_fast_int8,
              "onnx_path": str(args.onnx), "onnx_sha256": digest(args.onnx),
              "external_weights_sha256": digest(str(args.onnx) + ".data"),
              "builder_flags": ["INT8", "FP16"] + (["OBEY_PRECISION_CONSTRAINTS"] if constrained else []),
              "int8_required_layers": constrained, "workspace_MiB": 2048,
              "calibration": "IInt8EntropyCalibrator2, batch 1, measured training-image scales",
              "calibration_count": manifest["count"], "calibration_images_consumed_this_build": calibrator.index,
              "reused_calibration_build": str(args.reuse_calibration_build) if reuse_cache else None,
              "calibration_cache_sha256": digest(cache),
              "calibration_manifest_sha256": digest(manifest_path),
              "calibration_tensor_file_sha256": digest(tensors),
              "engine_sha256": digest(engine_path), "engine_bytes": engine_path.stat().st_size,
              "profiling_verbosity": str(engine.profiling_verbosity),
              "build_script_sha256": digest(__file__), "engine_path": str(engine_path),
              "input_name": inp.name, "input_shape": list(shape)}
    (args.output_dir / "build_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
