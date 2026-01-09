from __future__ import annotations

import argparse

import torch
torch.set_float32_matmul_precision("high")
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from datasets.tinyface.dataloader import get_tinyface_path, DatasetType
from lightning_modules.tinyface_datamodule import TinyFaceDataModule
from lightning_modules.timm_id_module import TimmIDModule
from lightning_modules.callbacks import (
    TinyFaceEvaluationCallback, GradualBlockUnfreezeCallback, GradualStageUnfreezeCallback
)
from model_eval.probe_gallery import MultiLabelMethod

import os
from math import ceil

from enum import Enum

from logging import getLogger
logger = getLogger(__name__)

# create enum for qkv/proj choices
class LoRATargetChoice(Enum):
    NONE = 0
    QKV = 1
    PROJ = 2
    BOTH = 3

# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser("TinyFace TIMM fine-tuning")

    # Model / data
    p.add_argument("--model_name", type=str, default="swin_base_patch4_window7_224.ms_in1k")
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=73)

    # Optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument(
        "--backbone_dropout", type=float, default=0.0, 
        help="Dropout probability for the backbone embeddings."
    )

    # Head
    p.add_argument("--head_type", type=str, default="linear", choices=["linear", "cosface", "adaface", "arcface"])
    p.add_argument("--head_scale", type=float, help="Scale parameter for the head (used for cosface, adaface, arcface).")
    p.add_argument("--head_margin", type=float, help="Margin parameter for the head (used for cosface, adaface, arcface).")
    p.add_argument("--adaface_h", type=float, help="h parameter for AdaFace head.")
    p.add_argument("--adaface_t_alpha", type=float, help="t_alpha parameter for AdaFace head.")

    # LoRA (optional)
    p.add_argument("--lora_enabled", action="store_true", help="Enable LoRA adapters in the backbone.")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument(
        "--lora_family",
        type=str,
        default="auto",
        choices=["auto", "vit", "swin", "pvt", "mobilevit", "levit", "efficientformer", "cnn"],
        help="Override inferred model family for LoRA target patterns.",
    )
    p.add_argument(
        "--lora_target_regex",
        type=str,
        default=None,
        help="Override LoRA target regex. If set, family-based defaults are ignored.",
    )
    p.add_argument(
        "--lora_qkv_proj", type=int, default=LoRATargetChoice.NONE.value,
        choices=[e.value for e in LoRATargetChoice],
        help="LoRA target choice: 0=none, 1=QKV, 2=proj, 3=both (family-specific)."
    )
    
    p.add_argument(
        "--lora_train_bias", type=int, default=0, choices=[0, 1],
        help="Also train biases (backbone) when LoRA is enabled.",
    )


    # Selective fine-tuning / freezing
    # Comma-separated list, e.g.:--thawed-modules head,all_norm,last_blocks=2,all_attn
    # Valid options are defined in lightning_modules/timm_id_module.py.
    p.add_argument(
        "--thawed_modules",
        type=str,
        default="",
        help=(
            "Comma-separated thaw rules. Examples: 'all_norm,last_blocks=2' or 'no_head,all_attn' or 'all'."
            "Use 'none' to freeze everything (including head)."
        ),
    )

    # Gradual unfreeze
    p.add_argument(
        "--gradual_unfreeze",
        type=str,
        choices=["block", "stage"],
        help="Gradually unfreeze backbone blocks or stages during training.",
    )

    p.add_argument(
        "--unfreeze_every_epochs",
        type=int,
        help="Epoch interval for gradual unfreezing. If None, inferred as max_epochs // num_blocks_to_thaw.",
    )
    p.add_argument(
        "--thaw_N_blocks",
        type=int,
        help="Number of blocks to unfreeze (only for --gradual_unfreeze block). Overrides --thaw_block_pct.",
    )
    p.add_argument(
        "--thaw_block_pct",
        type=float,
        default=1.0,
        help="Fraction of blocks to unfreeze (only for --gradual_unfreeze block).",
    )
    p.add_argument(
        "--thaw_N_stages",
        type=int,
        help="Number of stages to unfreeze (only for --gradual_unfreeze stage). Overrides --thaw_stage_pct.",
    )
    p.add_argument(
        "--thaw_stage_pct",
        type=float,
        default=1.0,
        help="Fraction of stages to unfreeze (only for --gradual_unfreeze stage).",
    )


    # Warmup (only used when gradual_unfreeze is enabled unless explicitly overridden)
    p.add_argument(
        "--warmup_epochs",
        type=int,
        help="Warmup epochs for LR (warmup+cosine). If None and gradual_unfreeze, inferred automatically.",
    )



    # Debug / reporting
    p.add_argument(
        "--print_trainable_params", action="store_true",
        help="Print trainable parameter names (requires_grad=True) and exit.",
    )

    # Rank-k eval
    p.add_argument("--rank_k", type=int, default=10)
    p.add_argument("--probe_batch_size", type=int, default=256)
    p.add_argument(
        "--multi_label_method",
        type=str,
        default="MAX",
        choices=[m.name for m in MultiLabelMethod],
    )
    p.add_argument(
        "--eval_on",
        type=str,
        default="EVAL",
        choices=[d.name for d in DatasetType],
    )
    p.add_argument("--no_roc", action="store_true", default=False)

    # Runtime
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--log_every_n_steps", type=int, default=50)
    p.add_argument("--deterministic", action="store_true", default=False)

    # Paths / W&B
    p.add_argument("--tinyface_root", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="tinyface-timm")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb_model", action="store_true", default=False)

    return p.parse_args()

# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()

    L.seed_everything(args.seed, workers=True)

    tinyface_root = args.tinyface_root or get_tinyface_path()

    # Data
    dm = TinyFaceDataModule(
        tinyface_root=tinyface_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    # Infer number of identities
    train_loader = dm.train_dataloader()
    num_classes = len(train_loader.dataset.subject_ids)

    # Adjust head params defaults
    if args.head_type in ["cosface", "adaface", "arcface"]:
        if args.head_scale is None:
            args.head_scale = 64.0
        if args.head_margin is None:
            args.head_margin = 0.35

    if args.head_type == "adaface":
        if args.adaface_h is None:
            args.adaface_h = 0.33
        if args.adaface_t_alpha is None:
            args.adaface_t_alpha = 0.01

    # If LoRA is enabled but no target specified, default to both QKV and proj
    if args.lora_enabled and args.lora_target_regex is None:
        if args.lora_qkv_proj == LoRATargetChoice.NONE.value:
            args.lora_qkv_proj = LoRATargetChoice.BOTH.value

    if args.lora_train_bias == 0:
        args.lora_train_bias = False
    else:
        args.lora_train_bias = True

    # Model
    model = TimmIDModule(
        model_name=args.model_name,
        num_classes=num_classes,
        head_type=args.head_type,
        head_scale=args.head_scale,
        head_margin=args.head_margin,
        adaface_h=args.adaface_h,
        adaface_t_alpha=args.adaface_t_alpha,
        img_size=args.img_size,
        lr=args.lr,
        warmup_epochs=(0 if not args.gradual_unfreeze else (args.warmup_epochs or 0)),
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        backbone_dropout=args.backbone_dropout,
        thawed_modules=args.thawed_modules,
        verbose_thaw=args.print_trainable_params,
        # LoRA
        lora_enabled=args.lora_enabled,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_family=args.lora_family,
        lora_target_regex=args.lora_target_regex,
        lora_apply_qkv=args.lora_qkv_proj in (LoRATargetChoice.QKV.value, LoRATargetChoice.BOTH.value),
        lora_apply_proj=args.lora_qkv_proj in (LoRATargetChoice.PROJ.value, LoRATargetChoice.BOTH.value),
        lora_train_bias=args.lora_train_bias,

    )

    # W&B config (everything that matters)
    wandb_config = vars(args).copy()
    wandb_config["num_classes"] = num_classes
    wandb_config["tinyface_root"] = tinyface_root

    head_tag = f"-{args.head_type}"
    if args.head_type in ["cosface", "adaface", "arcface"]:
        head_tag += f":s{args.head_scale}:m{args.head_margin}"

    if args.head_type == "adaface":
        head_tag += f":h{args.adaface_h}:ta{args.adaface_t_alpha}"

    lora_tag = ""
    if args.lora_enabled:
        lora_tag = f"-lora(r{args.lora_r}-a{args.lora_alpha}-d{args.lora_dropout})" \
                   f"{',qkv' if args.lora_qkv_proj in (LoRATargetChoice.QKV.value, LoRATargetChoice.BOTH.value) else ''}" \
                   f"{',proj' if args.lora_qkv_proj in (LoRATargetChoice.PROJ.value, LoRATargetChoice.BOTH.value) else ''}" \
                   f"{',' + str(args.lora_target_regex) if args.lora_target_regex else ''}" \
                   f"{',train_bias' if args.lora_train_bias else ''}"

    run_name = args.wandb_run_name or (
        f"{args.model_name}-img{args.img_size}-bs{args.batch_size}-lr{args.lr}" +
        f"{head_tag}{lora_tag}-thawed({args.thawed_modules})"
    )

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        log_model=("best" if not args.no_wandb_model else False),
        config=wandb_config,
    )

    run_id = wandb_logger.experiment.id
    wandb_dir = os.environ.get("WANDB_DIR", "./wandb")
    ckpt_dir = os.path.join(wandb_dir, "checkpoints", args.wandb_project, run_id)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=f"tinyface/{args.eval_on.lower()}/rank@1",
            mode="max",
            save_top_k=1,
            save_last=False, # only best model, change if you want to manually stop
            filename="{epoch}-rank1{"
                     + f"tinyface/{args.eval_on.lower()}/rank@1"
                     + ":.3f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        TinyFaceEvaluationCallback(
            tinyface_root=tinyface_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            rank_k=args.rank_k,
            probe_batch_size=args.probe_batch_size,
            multi_label_method=MultiLabelMethod[args.multi_label_method],
            also_log_roc=not args.no_roc,
            eval_on=DatasetType[args.eval_on],
        ),
    ]

    if args.warmup_epochs is not None and args.warmup_epochs > 0:
        model.warmup_epochs = args.warmup_epochs

    if args.gradual_unfreeze is not None:
        # Infer warmup if not provided:
        # rule: warm up for ~one interval (or at least 1), but not more than 10% of training.
        if args.warmup_epochs is None:
            model.warmup_epochs = min(max(1, every_n_epochs), max(1, args.max_epochs // 10))

        if args.gradual_unfreeze == "block":
            blocks_all = model.get_blocks_for_unfreeze()   # deepest -> shallowest
            num_blocks_total = len(blocks_all)
            if num_blocks_total == 0:
                raise RuntimeError("Gradual unfreeze requested, but no blocks were found.")
            if args.thaw_N_blocks is not None:
                num_blocks_to_thaw = min(args.thaw_N_blocks, num_blocks_total)
            else:
                if not (0.0 < args.thaw_block_pct <= 1.0):
                    raise ValueError("--thaw_block_pct must be in (0.0, 1.0].")

                num_blocks_to_thaw = max(1, int(ceil(args.thaw_block_pct * num_blocks_total)))

            blocks = blocks_all[:num_blocks_to_thaw]  # thaw only this fraction (deepest first)

            # Infer interval if not provided
            if args.unfreeze_every_epochs is None:
                # every_n_epochs = max(1, args.max_epochs // num_blocks_to_thaw)
                every_n_epochs = max(1, (args.max_epochs // 2) // num_blocks_to_thaw)
            else:
                every_n_epochs = max(1, int(args.unfreeze_every_epochs))


            callbacks.append(
                GradualBlockUnfreezeCallback(
                    blocks=blocks,
                    every_n_epochs=every_n_epochs,
                    verbose=True,
                )
            )
        elif args.gradual_unfreeze == "stage":
            stages = model.get_stages_for_unfreeze()   # deepest -> shallowest
            num_stages_total = len(stages)
            if num_stages_total == 0:
                raise RuntimeError("Gradual unfreeze requested, but no stages were found.")

            if args.thaw_N_stages is not None:
                num_stages_to_thaw = min(args.thaw_N_stages, num_stages_total)
            else:
                if not (0.0 < args.thaw_stage_pct <= 1.0):
                    raise ValueError("--thaw_stage_pct must be in (0.0, 1.0].")

                num_stages_to_thaw = max(1, int(ceil(args.thaw_stage_pct * num_stages_total)))

            stages = stages[:num_stages_to_thaw]  # thaw only this fraction (deepest first)

            # Infer interval if not provided
            if args.unfreeze_every_epochs is None:
                every_n_epochs = max(1, (args.max_epochs // 2) // num_stages_total)
            else:
                every_n_epochs = max(1, int(args.unfreeze_every_epochs))

            callbacks.append(
                GradualStageUnfreezeCallback(
                    stages=stages,
                    every_n_epochs=every_n_epochs,
                    verbose=True,
                )
            )

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        precision=args.precision if torch.cuda.is_available() else "32-true",
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=args.deterministic,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
