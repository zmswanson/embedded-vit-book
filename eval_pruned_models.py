#!/usr/bin/env python
"""Evaluate pruned+fine-tuned checkpoints on the TinyFace TEST set.

Because pruned models have non-standard architecture (fewer heads/blocks),
we reconstruct the architecture by re-applying the same pruning config,
then load the fine-tuned state_dict.
"""
import csv
import os
from pathlib import Path

import torch
import lightning as L

from datasets.tinyface.dataloader import get_tinyface_path, DatasetType
from lightning_modules.tinyface_datamodule import TinyFaceDataModule
from lightning_modules.callbacks import TinyFaceEvaluationCallback
from model_eval.probe_gallery import MultiLabelMethod
from pruning import load_module_for_finetuning, prune_model, _param_count


EXPERIMENTS = [
    {
        "name": "deit3_base_lora-heads_0.10",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/vimriwcl/epoch=44-rank1tinyface/eval/rank@1=0.579.ckpt",
        "model_name": "deit3_base_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models/epoch=3-rank1tinyface/eval/rank@1=0.574.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.10,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5834,
        "variant": "LoRA",
    },
    {
        "name": "deit3_base_lora-heads_0.25",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/vimriwcl/epoch=44-rank1tinyface/eval/rank@1=0.579.ckpt",
        "model_name": "deit3_base_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models/epoch=8-rank1tinyface/eval/rank@1=0.564.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.25,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5834,
        "variant": "LoRA",
    },
    {
        "name": "deit3_base_lora-heads_0.50",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/vimriwcl/epoch=44-rank1tinyface/eval/rank@1=0.579.ckpt",
        "model_name": "deit3_base_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models/epoch=4-rank1tinyface/eval/rank@1=0.521.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.50,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5834,
        "variant": "LoRA",
    },
    {
        "name": "deit3_base_lora-blocks_3",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_baselines/vimriwcl/epoch=44-rank1tinyface/eval/rank@1=0.579.ckpt",
        "model_name": "deit3_base_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models/epoch=7-rank1tinyface/eval/rank@1=0.535.ckpt",
        "prune_method": "blocks",
        "head_prune_ratio": 0.0,
        "num_blocks_to_remove": 3,
        "baseline_rank1": 0.5834,
        "variant": "LoRA",
    },
    {
        "name": "deit3_base_kd-heads_0.25",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_kd/5mngq5jn/epoch=47-rank1tinyface/eval/rank@1=0.579.ckpt",
        "model_name": "deit3_base_patch16_224.fb_in1k",
        "pruned_ckpt": "pruned_models/epoch=3-rank1tinyface/eval/rank@1=0.568.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.25,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5807,
        "variant": "KD CVLFace",
    },
    {
        "name": "swin_base-heads_0.25",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_lora/ci53ufy8/epoch=39-rank1tinyface/eval/rank@1=0.556.ckpt",
        "model_name": "swin_base_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models/epoch=7-rank1tinyface/eval/rank@1=0.436.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.25,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5359,
        "variant": "baseline",
    },
    {
        "name": "swin_base-blocks_5",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_final_lora/ci53ufy8/epoch=39-rank1tinyface/eval/rank@1=0.556.ckpt",
        "model_name": "swin_base_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models/epoch=1-rank1tinyface/eval/rank@1=0.550.ckpt",
        "prune_method": "blocks",
        "head_prune_ratio": 0.0,
        "num_blocks_to_remove": 5,
        "baseline_rank1": 0.5359,
        "variant": "baseline",
    },
    {
        "name": "swin_tiny_kd-heads_0.10",
        "original_ckpt": "/mnt/data/wandb/checkpoints/vit_book_kd/y6ddjqyw/epoch=44-rank1tinyface/eval/rank@1=0.570.ckpt",
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "pruned_ckpt": "pruned_models/epoch=8-rank1tinyface/eval/rank@1=0.512.ckpt",
        "prune_method": "heads",
        "head_prune_ratio": 0.10,
        "num_blocks_to_remove": 0,
        "baseline_rank1": 0.5491,
        "variant": "KD CVLFace",
    },
]


def evaluate_pruned_model(exp: dict) -> dict:
    """Reconstruct pruned architecture, load fine-tuned weights, evaluate."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {exp['name']}")
    print(f"{'='*60}")

    # 1) Reconstruct pruned architecture
    module, family = load_module_for_finetuning(
        exp["original_ckpt"], exp["model_name"]
    )
    params_before = _param_count(module)

    prune_model(
        module.backbone, family,
        prune_method=exp["prune_method"],
        head_prune_ratio=exp["head_prune_ratio"],
        num_blocks_to_remove=exp["num_blocks_to_remove"],
        img_size=96,
    )
    params_after = _param_count(module)

    # 2) Load fine-tuned state dict
    ckpt = torch.load(exp["pruned_ckpt"], map_location="cpu", weights_only=False)
    module.load_state_dict(ckpt["state_dict"])
    module.eval()

    # 3) Run test-set evaluation
    tinyface_root = get_tinyface_path()
    dm = TinyFaceDataModule(
        tinyface_root=tinyface_root,
        img_size=96,
        batch_size=512,
    )

    test_cb = TinyFaceEvaluationCallback(
        tinyface_root=tinyface_root,
        img_size=96,
        batch_size=512,
        rank_k=10,
        probe_batch_size=256,
        multi_label_method=MultiLabelMethod.MAX,
        also_log_roc=False,
        eval_on=DatasetType.TEST,
    )

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[test_cb],
        logger=False,
        enable_checkpointing=False,
    )

    results = trainer.validate(model=module, datamodule=dm)
    test_rank1 = results[0].get("tinyface/test/rank@1", float("nan"))

    compression = 1.0 - params_after / params_before
    rank1_drop = exp["baseline_rank1"] - test_rank1

    print(f"  Params: {params_before:,} → {params_after:,} "
          f"(−{compression:.1%})")
    print(f"  Test rank@1: {test_rank1:.4f} "
          f"(baseline: {exp['baseline_rank1']:.4f}, "
          f"drop: {rank1_drop:.4f})")

    # Free GPU memory
    del module, trainer
    torch.cuda.empty_cache()

    return {
        "experiment": exp["name"],
        "model": exp["model_name"].split(".")[0].replace("_patch16_224", "").replace("_patch4_window7_224", ""),
        "variant": exp["variant"],
        "prune_method": exp["prune_method"],
        "head_ratio": exp["head_prune_ratio"],
        "blocks_removed": exp["num_blocks_to_remove"],
        "params_before": params_before,
        "params_after": params_after,
        "compression": f"{compression:.4f}",
        "baseline_rank1": f"{exp['baseline_rank1']:.4f}",
        "test_rank1": f"{test_rank1:.4f}",
        "rank1_drop": f"{rank1_drop:.4f}",
    }


def main():
    csv_path = Path("pruning_results/pruning_summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for exp in EXPERIMENTS:
        result = evaluate_pruned_model(exp)
        all_results.append(result)

    # Write CSV
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved results to {csv_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("PRUNING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<35} {'Prune':<8} {'Compress':<10} "
          f"{'Baseline':<10} {'Pruned':<10} {'Drop':<8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['experiment']:<35} {r['prune_method']:<8} "
              f"{float(r['compression']):>8.1%}  "
              f"{r['baseline_rank1']:>8}  "
              f"{r['test_rank1']:>8}  "
              f"{r['rank1_drop']:>6}")


if __name__ == "__main__":
    main()
