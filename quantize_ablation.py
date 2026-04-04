#!/usr/bin/env python3
"""Quantize the 10 ablation study models to FP16 and INT8."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize_models import (
    TinyFaceCalibrationReader, quantize_fp16, quantize_int8,
    validate_quantized, IMG_SIZE,
)

MODELS = [
    "convnext_tiny", "convnext_small", "convnext_base",
    "pvt_v2_b2", "pvt_v2_b3", "pvt_v2_b5",
    "mobilevitv2_200",
    "resnet50d", "resnet101d", "resnet200d",
]

FP32_DIR = "onnx_models/baselines"
FP16_DIR = "onnx_models_fp16/baselines"
INT8_DIR = "onnx_models_int8/baselines"

if __name__ == "__main__":
    tinyface_path = Path("datasets/tinyface/tinyface_path.txt").read_text().strip()
    calib = TinyFaceCalibrationReader(tinyface_path, img_size=IMG_SIZE, num_samples=50)
    print(f"Loaded {len(calib.image_paths)} validation images\n")

    for i, name in enumerate(MODELS, 1):
        fp32 = f"{FP32_DIR}/{name}.onnx"
        print(f"[{i}/{len(MODELS)}] {name}")

        if not Path(fp32).exists():
            data = Path(fp32 + ".data")
            if not data.exists():
                print(f"  SKIP — FP32 not found")
                continue

        # FP16
        fp16_path = f"{FP16_DIR}/{name}.onnx"
        try:
            fp16_size = quantize_fp16(fp32, fp16_path)
            cos = validate_quantized(fp32, fp16_path, calib, n_samples=20)
            status = "PASS" if cos > 0.999 else "WARN"
            print(f"  FP16: {fp16_size:.1f} MB  cos={cos:.6f}  [{status}]")
        except Exception as e:
            print(f"  FP16: FAILED — {e}")

        # INT8
        int8_path = f"{INT8_DIR}/{name}.onnx"
        try:
            int8_size = quantize_int8(fp32, int8_path)
            cos = validate_quantized(fp32, int8_path, calib, n_samples=20)
            status = "PASS" if cos > 0.990 else "WARN"
            print(f"  INT8: {int8_size:.1f} MB  cos={cos:.6f}  [{status}]")
        except Exception as e:
            print(f"  INT8: FAILED — {e}")

        print()

    print("Done.")
