"""Reconstruct trainability for the six evaluated LoRA runs without training.

Run in the vit-benchmark environment from the repository root:
    python scripts/audit_lora_trainability.py

The run IDs match inference_results/final_models_inference_results.csv; adapter
settings were checked against their local wandb run config.yaml files. Models
use random initialization: parameter counts and freeze schedules do not depend
on checkpoint values. No datasets, downloads, or GPU execution are required.
"""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import timm
import torch

from lightning_modules.callbacks import (
    GradualBlockUnfreezeCallback,
    GradualStageUnfreezeCallback,
)
from lightning_modules.timm_id_module import TimmIDModule


RUNS = [
    ("Swin-T", "f0frj93y", "swin_tiny_patch4_window7_224.ms_in1k", 24, 120, 0.05),
    ("Swin-S", "ymm0k0ub", "swin_small_patch4_window7_224.ms_in1k", 24, 120, 0.05),
    ("Swin-B", "ci53ufy8", "swin_base_patch4_window7_224.ms_in1k", 8, 32, 0.05),
    ("DeiT3-S", "cocm8b7d", "deit3_small_patch16_224.fb_in1k", 4, 24, 0.1),
    ("DeiT3-M", "dpwn5kb1", "deit3_medium_patch16_224.fb_in1k", 4, 24, 0.1),
    ("DeiT3-B", "2ljjzfzb", "deit3_base_patch16_224.fb_in1k", 4, 24, 0.1),
]


def main():
    torch.set_num_threads(1)
    rows = []
    for label, run_id, model_name, rank, alpha, dropout in RUNS:
        model = TimmIDModule(
            model_name=model_name, num_classes=2313, head_type="adaface",
            img_size=96, pretrained=False, lora_enabled=True, lora_r=rank,
            lora_alpha=alpha, lora_dropout=dropout,
            lora_apply_qkv=True, lora_apply_proj=True, lora_train_bias=False,
        )
        model.apply_thaw_plan("")
        original = [
            p for name, p in model.backbone.named_parameters()
            if "lora_A" not in name and "lora_B" not in name
        ]
        adapters = [
            p for name, p in model.backbone.named_parameters()
            if "lora_A" in name or "lora_B" in name
        ]
        original_count = sum(p.numel() for p in original)
        adapter_count = sum(p.numel() for p in adapters)
        if label.startswith("Swin"):
            modules = model.get_stages_for_unfreeze()[:2]
            callback = GradualStageUnfreezeCallback(modules, 5, verbose=False)
            starts = [0, 5, 10]
        else:
            modules = model.get_blocks_for_unfreeze()[:8]
            callback = GradualBlockUnfreezeCallback(modules, 5, verbose=False)
            starts = [0] + list(range(5, 41, 5))
        names = {id(m): n for n, m in model.backbone.named_modules()}
        for i, epoch in enumerate(starts):
            callback.on_train_epoch_start(SimpleNamespace(current_epoch=epoch), model)
            trainable = sum(p.numel() for p in original if p.requires_grad)
            rows.append({
                "model": label, "run_id": run_id, "lora_rank": rank,
                "epoch_start_zero_based": epoch,
                "epoch_end_zero_based": starts[i + 1] - 1 if i + 1 < len(starts) else 49,
                "newly_unfrozen_module": names[id(modules[i - 1])] if i else "none",
                "original_backbone_parameters": original_count,
                "trainable_original_backbone_parameters": trainable,
                "trainable_original_backbone_percent": f"{100 * trainable / original_count:.6f}",
                "trainable_adapter_parameters": adapter_count,
                "trainable_head_parameters": sum(p.numel() for p in model.head.parameters()),
                "torch_version": torch.__version__, "timm_version": timm.__version__,
            })
            print(label, epoch, f"{trainable:,}/{original_count:,}",
                  f"({100 * trainable / original_count:.2f}%)", f"adapters={adapter_count:,}")
        del model
    output = Path(__file__).resolve().parents[1] / "docs/scalable_vits_for_embedded_systems/lora_trainability.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
