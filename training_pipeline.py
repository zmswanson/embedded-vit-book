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
from lightning_modules.callbacks import TinyFaceEvaluationCallback
from model_eval.probe_gallery import MultiLabelMethod

import os

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
    p.add_argument("--lora_qkv", action="store_true", help="Apply LoRA to QKV (family-specific).")
    p.add_argument("--lora_proj", action="store_true", help="Apply LoRA to projection (family-specific).")
    p.add_argument(
        "--lora_train_bias",
        action="store_true",
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

    # Debug / reporting
    p.add_argument(
        "--print_trainable_params",
        action="store_true",
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
    p.add_argument("--no_roc", action="store_true")

    # Runtime
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--log_every_n_steps", type=int, default=50)
    p.add_argument("--deterministic", action="store_true")

    # Paths / W&B
    p.add_argument("--tinyface_root", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="tinyface-timm")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb_model", action="store_true")

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

    # LoRA config
    if args.lora_enabled and not (args.lora_qkv or args.lora_proj or args.lora_target_regex):
        # If LoRA is enabled but no target specified, default to both
        args.lora_qkv = True
        args.lora_proj = True

    # Model
    model = TimmIDModule(
        model_name=args.model_name,
        num_classes=num_classes,
        head_type=args.head_type,
        head_scale=args.head_scale,
        head_margin=args.head_margin,
        img_size=args.img_size,
        lr=args.lr,
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
        lora_apply_qkv=args.lora_qkv,
        lora_apply_proj=args.lora_proj,
        lora_train_bias=args.lora_train_bias,

    )

    # W&B config (everything that matters)
    wandb_config = vars(args).copy()
    wandb_config["num_classes"] = num_classes
    wandb_config["tinyface_root"] = tinyface_root

    head_tag = f"-{args.head_type}"
    if args.head_type in ["cosface", "adaface", "arcface"]:
        head_tag += f":s{args.head_scale}:m{args.head_margin}"

    lora_tag = ""
    if args.lora_enabled:
        lora_tag = f"-lora(r{args.lora_r}-a{args.lora_alpha}-d{args.lora_dropout})" \
                   f"{',qkv' if args.lora_qkv else ''}{',proj' if args.lora_proj else ''}" \
                   f"{',' + str(args.lora_target_regex) if args.lora_target_regex else ''}"

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
