"""Apply FP16 and INT8 post-training quantization to ONNX models.

FP16 — onnxconverter-common float16 conversion.
INT8 — onnxruntime *dynamic* quantization (weights-only).

Note: Static INT8 quantization (QDQ format) produces poor results (<0.20
cosine similarity for Swin models) on ONNX graphs exported by PyTorch 2.9's
dynamo-based exporter.  Dynamic quantization quantizes only weights at export
time and computes activations in FP32 at runtime, yielding >0.99 cosine
similarity.  TensorRT on the Jetson will perform its own INT8 calibration
during engine build if full activation quantization is desired.
"""
import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    quantize_dynamic,
)
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 96


# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------
class TinyFaceCalibrationReader(CalibrationDataReader):
    """Provides calibration images for INT8 static quantization.

    Loads a random subset of TinyFace *training* images and preprocesses them
    identically to the eval transforms in ``datasets/tinyface/dataloader.py``.
    """

    def __init__(self, dataset_path: str, img_size: int = IMG_SIZE,
                 num_samples: int = 500, seed: int = 42):
        training_dir = os.path.join(dataset_path, "Training_Set")
        if not os.path.isdir(training_dir):
            raise FileNotFoundError(f"Training_Set not found at {training_dir}")

        # Collect all image paths
        all_images = []
        for root, _, files in os.walk(training_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_images.append(os.path.join(root, f))

        if len(all_images) == 0:
            raise RuntimeError(f"No images found in {training_dir}")

        rng = np.random.RandomState(seed)
        indices = rng.choice(len(all_images), size=min(num_samples, len(all_images)),
                             replace=False)
        self.image_paths = [all_images[i] for i in indices]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._idx = 0

    def get_next(self):
        if self._idx >= len(self.image_paths):
            return None
        img = Image.open(self.image_paths[self._idx]).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).numpy()  # (1, 3, H, W)
        self._idx += 1
        return {"input": tensor}

    def rewind(self):
        self._idx = 0


# ---------------------------------------------------------------------------
# FP16 quantization
# ---------------------------------------------------------------------------
def quantize_fp16(input_path: str, output_path: str) -> float:
    """Convert an FP32 ONNX model to FP16.  Returns output size in MB."""
    model = onnx.load(input_path, load_external_data=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_fp16 = float16.convert_float_to_float16(
            model, keep_io_types=True,
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return size_mb


# ---------------------------------------------------------------------------
# INT8 dynamic quantization
# ---------------------------------------------------------------------------
def quantize_int8(input_path: str, output_path: str) -> float:
    """Apply INT8 dynamic quantization (weights only).  Returns output size in MB."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        use_external_data_format=False,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return size_mb


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def validate_quantized(fp32_path: str, quant_path: str,
                       calibration_reader: "TinyFaceCalibrationReader",
                       n_samples: int = 20) -> float:
    """Run N real calibration images through both models and return average cosine similarity."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    fp32_sess = ort.InferenceSession(fp32_path, opts,
                                     providers=["CPUExecutionProvider"])
    quant_sess = ort.InferenceSession(quant_path, opts,
                                      providers=["CPUExecutionProvider"])

    calibration_reader.rewind()
    sims = []
    for _ in range(n_samples):
        data = calibration_reader.get_next()
        if data is None:
            break
        fp32_out = fp32_sess.run(None, data)[0]
        quant_out = quant_sess.run(None, data)[0]
        sims.append(cosine_similarity(fp32_out, quant_out))

    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def process_all(fp32_root: str, fp16_root: str, int8_root: str,
                calibration_reader: "TinyFaceCalibrationReader",
                skip_fp16: bool = False, skip_int8: bool = False):
    """Quantize every .onnx file under *fp32_root* into FP16 and INT8 variants."""
    results = []

    onnx_files = sorted(Path(fp32_root).rglob("*.onnx"))
    if not onnx_files:
        print(f"No .onnx files found under {fp32_root}")
        return results

    total = len(onnx_files)
    for i, fp32_file in enumerate(onnx_files, 1):
        rel = fp32_file.relative_to(fp32_root)
        name = fp32_file.stem
        print(f"\n[{i}/{total}] {rel}")

        fp32_size = fp32_file.stat().st_size
        data_file = Path(str(fp32_file) + ".data")
        if data_file.exists():
            fp32_size += data_file.stat().st_size
        fp32_size_mb = fp32_size / (1024 * 1024)

        row = {"model": name, "fp32_size_MB": round(fp32_size_mb, 1)}

        # --- FP16 ---
        if not skip_fp16:
            fp16_path = Path(fp16_root) / rel
            try:
                fp16_size = quantize_fp16(str(fp32_file), str(fp16_path))
                cos_fp16 = validate_quantized(str(fp32_file), str(fp16_path),
                                              calibration_reader)
                row["fp16_size_MB"] = round(fp16_size, 1)
                row["fp16_cosine"] = round(cos_fp16, 6)
                status = "PASS" if cos_fp16 > 0.999 else "WARN"
                print(f"  FP16: {fp16_size:.1f} MB  cos={cos_fp16:.6f}  [{status}]")
            except Exception as e:
                row["fp16_size_MB"] = ""
                row["fp16_cosine"] = ""
                print(f"  FP16: FAILED — {e}")

        # --- INT8 ---
        if not skip_int8:
            int8_path = Path(int8_root) / rel
            try:
                int8_size = quantize_int8(str(fp32_file), str(int8_path))
                cos_int8 = validate_quantized(str(fp32_file), str(int8_path),
                                              calibration_reader)
                row["int8_size_MB"] = round(int8_size, 1)
                row["int8_cosine"] = round(cos_int8, 6)
                status = "PASS" if cos_int8 > 0.990 else "WARN"
                print(f"  INT8: {int8_size:.1f} MB  cos={cos_int8:.6f}  [{status}]")
            except Exception as e:
                row["int8_size_MB"] = ""
                row["int8_cosine"] = ""
                print(f"  INT8: FAILED — {e}")

        results.append(row)

    return results


def write_csv(results: list, csv_path: str):
    if not results:
        return
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {len(results)} rows to {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Post-training quantization (FP16 + INT8) for ONNX models"
    )
    parser.add_argument("--fp32_dir", default="onnx_models",
                        help="Root directory of FP32 ONNX models")
    parser.add_argument("--fp16_dir", default="onnx_models_fp16",
                        help="Output directory for FP16 models")
    parser.add_argument("--int8_dir", default="onnx_models_int8",
                        help="Output directory for INT8 models")
    parser.add_argument("--tinyface_path", default=None,
                        help="TinyFace dataset root (default: read from datasets/tinyface/tinyface_path.txt)")
    parser.add_argument("--calib_samples", type=int, default=50,
                        help="Number of real images for cosine similarity validation")
    parser.add_argument("--csv_out", default="quantization_results/quantization_summary.csv",
                        help="Path for summary CSV")
    parser.add_argument("--skip_fp16", action="store_true")
    parser.add_argument("--skip_int8", action="store_true")
    args = parser.parse_args()

    # Resolve TinyFace path
    tinyface_path = args.tinyface_path
    if tinyface_path is None:
        txt = Path("datasets/tinyface/tinyface_path.txt")
        if txt.exists():
            tinyface_path = txt.read_text().strip()
        else:
            print("ERROR: Cannot find TinyFace path. Pass --tinyface_path.")
            sys.exit(1)

    print(f"FP32 source:  {args.fp32_dir}")
    print(f"FP16 output:  {args.fp16_dir}")
    print(f"INT8 output:  {args.int8_dir}")
    print(f"Validation:   {args.calib_samples} real images from {tinyface_path}")

    # Build calibration reader (used for validation — comparing quantized vs FP32 on real images)
    calib_reader = TinyFaceCalibrationReader(
        tinyface_path, img_size=IMG_SIZE, num_samples=args.calib_samples,
    )
    print(f"Loaded {len(calib_reader.image_paths)} validation images")

    results = process_all(
        fp32_root=args.fp32_dir,
        fp16_root=args.fp16_dir,
        int8_root=args.int8_dir,
        calibration_reader=calib_reader,
        skip_fp16=args.skip_fp16,
        skip_int8=args.skip_int8,
    )

    write_csv(results, args.csv_out)

    # Summary
    print("\n" + "=" * 60)
    print("QUANTIZATION SUMMARY")
    print("=" * 60)
    for r in results:
        line = f"  {r['model']:50s}  FP32={r['fp32_size_MB']:>7.1f}MB"
        if "fp16_size_MB" in r and r["fp16_size_MB"] != "":
            line += f"  FP16={r['fp16_size_MB']:>7.1f}MB (cos={r['fp16_cosine']:.4f})"
        if "int8_size_MB" in r and r["int8_size_MB"] != "":
            line += f"  INT8={r['int8_size_MB']:>7.1f}MB (cos={r['int8_cosine']:.4f})"
        print(line)


if __name__ == "__main__":
    main()
