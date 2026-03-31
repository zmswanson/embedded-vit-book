"""
Evaluate a teacher model (CVLFace / PETALface) on TinyFace test set.

The TinyFace eval dataloaders apply ImageNet normalisation, but teachers
expect [-1, 1] input at 112×112.  This script wraps the teacher so it
un-does ImageNet normalisation on-the-fly before forwarding.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.tinyface.dataloader import get_tinyface_path, DatasetType
from lightning_modules.teacher_wrapper import get_teacher
from model_eval.tinyface_eval import evaluate_model_rank_k, evaluate_model_roc

# ImageNet constants used by eval dataloaders
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _NormConvertWrapper(nn.Module):
    """Wraps a teacher model so it accepts ImageNet-normalised input and
    internally converts to [-1, 1] before forwarding."""

    def __init__(self, teacher: nn.Module):
        super().__init__()
        self.teacher = teacher
        self.register_buffer("img_mean", IMAGENET_MEAN)
        self.register_buffer("img_std", IMAGENET_STD)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Undo ImageNet normalisation → [0, 1]
        x = x * self.img_std + self.img_mean
        # Apply teacher normalisation → [-1, 1]
        x = (x - 0.5) / 0.5
        return self.teacher(x)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate teacher model on TinyFace")
    parser.add_argument("--teacher_type", type=str, required=True,
                        choices=["cvlface_vit_base", "petalface_swin"])
    parser.add_argument("--teacher_finetuned", action="store_true")
    parser.add_argument("--teacher_finetune_ckpt_path", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--rank_k", type=int, default=10)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tinyface_root = get_tinyface_path()

    # Load teacher
    ft_path = args.teacher_finetune_ckpt_path if args.teacher_finetuned else None
    teacher = get_teacher(args.teacher_type, finetune_ckpt_path=ft_path)
    model = _NormConvertWrapper(teacher).to(device).eval()

    mode_str = "finetuned" if args.teacher_finetuned else "frozen"
    print(f"Evaluating {args.teacher_type} ({mode_str}) on TinyFace test set ...")

    # Rank-k
    ranks = evaluate_model_rank_k(
        tinyface_root, model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=4,
        dataset_type=DatasetType.TEST,
        rank_k=args.rank_k,
        device=device,
    )
    # ROC
    roc = evaluate_model_roc(
        tinyface_root, model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=4,
        dataset_type=DatasetType.TEST,
        return_dict=True,
        device=device,
    )

    # Build results dict
    results = {}
    for i, v in enumerate(ranks, start=1):
        results[f"tinyface/test/rank@{i}"] = float(v)
    for k, v in roc.items():
        if isinstance(v, float):
            results[f"tinyface/test/{k}"] = v

    # Print
    teacher_label = f"{args.teacher_type}_{mode_str}"
    print(f"\n=== {teacher_label} ===")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")

    # Save
    save_keys = [
        'tinyface/test/rank@1', 'tinyface/test/rank@2', 'tinyface/test/rank@3',
        'tinyface/test/rank@4', 'tinyface/test/rank@5', 'tinyface/test/rank@6',
        'tinyface/test/rank@7', 'tinyface/test/rank@8', 'tinyface/test/rank@9',
        'tinyface/test/rank@10',
        'tinyface/test/auc', 'tinyface/test/mAP', 'tinyface/test/eer',
        'tinyface/test/tpr_at_fpr_1pct', 'tinyface/test/tpr_at_fpr_5pct',
    ]

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        if not os.path.exists(args.save_path):
            with open(args.save_path, "w") as f:
                f.write("wandb_id,model_name")
                for key in save_keys:
                    f.write(f",{key}")
                f.write("\n")

        with open(args.save_path, "a") as f:
            f.write(f"teacher,{teacher_label}")
            for key in save_keys:
                f.write(f",{results.get(key, '')}")
            f.write("\n")
        print(f"\nResults appended to {args.save_path}")


if __name__ == "__main__":
    main()
