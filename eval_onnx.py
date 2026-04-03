"""TinyFace probe-gallery evaluation using an ONNX model via ONNX Runtime.

Mirrors the evaluation logic in ``model_eval/tinyface_eval.py`` and
``model_eval/probe_gallery.py`` but replaces the PyTorch model with an
ONNX Runtime inference session.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from datasets.tinyface.dataloader import (
    DatasetType,
    get_eval_loaders,
    get_test_loaders,
    get_tinyface_path,
)
from model_eval.probe_gallery import (
    MultiLabelMethod,
    get_cosine_similarity,
    get_rank_k_accuracy,
    get_roc_metrics,
)


# ------------------------------------------------------------------
# Feature extraction with ONNX Runtime
# ------------------------------------------------------------------

def get_batch_features_onnx(
    session: ort.InferenceSession,
    data_loader,
    use_gpu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features from *data_loader* using an ORT session.

    Processes images one at a time to avoid reshape failures in models
    (e.g. Swin) whose internal ops were traced with batch_size=1.

    Returns (features, labels) as torch tensors on CPU.
    """
    input_name = session.get_inputs()[0].name
    features, labels = [], []

    for images, labels_batch in tqdm(data_loader, desc="Extracting features (ONNX)"):
        for j in range(images.shape[0]):
            inp = images[j:j+1].numpy()
            out = session.run(None, {input_name: inp})[0]
            features.append(torch.from_numpy(out))
        labels.append(labels_batch)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


# ------------------------------------------------------------------
# Rank-k evaluation
# ------------------------------------------------------------------

def evaluate_onnx_rank_k(
    onnx_path: str,
    tinyface_path: str,
    img_size: int = 96,
    batch_size: int = 512,
    num_workers: int = 4,
    dataset_type: DatasetType = DatasetType.TEST,
    rank_k: int = 5,
    multi_label_method: MultiLabelMethod = MultiLabelMethod.MAX,
    use_gpu: bool = False,
    compute_roc: bool = False,
) -> list[float] | tuple[list[float], dict]:
    """Run TinyFace probe-gallery evaluation with an ONNX model.

    Returns a list of rank-k accuracies ``[rank@1, rank@2, ..., rank@k]``.
    If *compute_roc* is True, returns ``(ranks, roc_dict)`` where roc_dict
    contains auc, mAP, eer, eer_threshold, tpr_at_fpr_1pct, tpr_at_fpr_5pct.
    """
    # Load ONNX session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(onnx_path, opts, providers=providers)

    # Data loaders
    if dataset_type == DatasetType.EVAL:
        gallery_loader, probe_loader = get_eval_loaders(
            tinyface_path, batch_size=batch_size, img_size=img_size,
            num_workers=num_workers,
        )
    elif dataset_type == DatasetType.TEST:
        gallery_loader, probe_loader = get_test_loaders(
            tinyface_path, batch_size=batch_size, img_size=img_size,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Extract features
    gallery_features, gallery_labels = get_batch_features_onnx(
        session, gallery_loader, use_gpu=use_gpu,
    )
    probe_features, probe_labels = get_batch_features_onnx(
        session, probe_loader, use_gpu=use_gpu,
    )

    # Compute cosine similarity and rank-k
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    cosine_sim = get_cosine_similarity(
        probe_features, gallery_features, device=device,
    )
    ranks = get_rank_k_accuracy(
        cosine_sim, probe_labels, gallery_labels,
        k=rank_k, multi_label_method=multi_label_method,
    )

    if not compute_roc:
        return ranks

    roc = get_roc_metrics(
        cosine_sim, probe_labels, gallery_labels,
        return_dict=True, multi_label_method=multi_label_method,
    )
    # Keep only scalar metrics
    roc_scalars = {k: float(v) for k, v in roc.items()
                   if not isinstance(v, np.ndarray)}
    return ranks, roc_scalars


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TinyFace probe-gallery evaluation with an ONNX model",
    )
    parser.add_argument("--onnx_path", required=True,
                        help="Path to ONNX model file")
    parser.add_argument("--tinyface_path", default=None,
                        help="TinyFace dataset root (default: auto-detect)")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset", choices=["eval", "test"], default="test")
    parser.add_argument("--rank_k", type=int, default=5)
    parser.add_argument("--gpu", action="store_true",
                        help="Use CUDA execution provider")
    parser.add_argument("--csv_out", default=None,
                        help="Append results to this CSV file")
    parser.add_argument("--model_label", default=None,
                        help="Label for CSV output (default: filename stem)")
    parser.add_argument("--precision_label", default="FP32",
                        help="Precision label for CSV output")
    parser.add_argument("--roc", action="store_true",
                        help="Also compute ROC metrics (AUC, mAP, EER, TPR@FPR)")
    args = parser.parse_args()

    tf_path = args.tinyface_path or get_tinyface_path()
    ds_type = DatasetType.TEST if args.dataset == "test" else DatasetType.EVAL

    print(f"Model:   {args.onnx_path}")
    print(f"Dataset: TinyFace {args.dataset}")

    result = evaluate_onnx_rank_k(
        onnx_path=args.onnx_path,
        tinyface_path=tf_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_type=ds_type,
        rank_k=args.rank_k,
        use_gpu=args.gpu,
        compute_roc=args.roc,
    )

    if args.roc:
        ranks, roc_metrics = result
    else:
        ranks = result
        roc_metrics = None

    for i, r in enumerate(ranks, 1):
        print(f"  rank@{i}: {r:.4f}")

    if roc_metrics:
        print(f"  AUC:            {roc_metrics['auc']:.6f}")
        print(f"  mAP:            {roc_metrics['mAP']:.6f}")
        print(f"  EER:            {roc_metrics['eer']:.6f}")
        print(f"  TPR@FPR=1%:     {roc_metrics['tpr_at_fpr_1pct']:.6f}")
        print(f"  TPR@FPR=5%:     {roc_metrics['tpr_at_fpr_5pct']:.6f}")

    # Optionally append to CSV
    if args.csv_out:
        label = args.model_label or Path(args.onnx_path).stem
        row = {
            "model": label,
            "precision": args.precision_label,
            "onnx_path": args.onnx_path,
        }
        for i, r in enumerate(ranks, 1):
            row[f"rank@{i}"] = round(r, 4)
        if roc_metrics:
            for k in ("auc", "mAP", "eer", "eer_threshold",
                      "tpr_at_fpr_1pct", "tpr_at_fpr_5pct"):
                row[k] = round(roc_metrics[k], 6)

        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"Appended to {args.csv_out}")


if __name__ == "__main__":
    main()
