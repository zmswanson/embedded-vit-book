"""Training pipeline for knowledge distillation on TinyFace.

CLI wrapper around :class:`KDTimmIDModule` — trains a timm student with
feature-level KD from a frozen (or fine-tuned) teacher.
"""
from __future__ import annotations

import argparse
import os
from enum import Enum
from math import ceil

import torch
torch.set_float32_matmul_precision("high")
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from datasets.tinyface.dataloader import get_tinyface_path, DatasetType
from lightning_modules.tinyface_datamodule import TinyFaceDataModule
from lightning_modules.kd_module import KDTimmIDModule
from lightning_modules.callbacks import (
    TinyFaceEvaluationCallback, GradualBlockUnfreezeCallback, GradualStageUnfreezeCallback,
)
from model_eval.probe_gallery import MultiLabelMethod

from logging import getLogger
logger = getLogger(__name__)


class LoRATargetChoice(Enum):
    NONE = 0
    QKV = 1
    PROJ = 2
    BOTH = 3


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("TinyFace KD Training")

    # Student model / data
    p.add_argument("--student_model_name", type=str, required=True)
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=73)

    # Optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--backbone_dropout", type=float, default=0.0)

    # Head
    p.add_argument("--head_type", type=str, default="adaface",
                    choices=["linear", "cosface", "adaface", "arcface"])
    p.add_argument("--head_scale", type=float, default=None)
    p.add_argument("--head_margin", type=float, default=None)
    p.add_argument("--adaface_h", type=float, default=None)
    p.add_argument("--adaface_t_alpha", type=float, default=None)

    # Teacher
    p.add_argument("--teacher_type", type=str, required=True,
                    choices=["cvlface_vit_base"])
    p.add_argument("--teacher_finetuned", action="store_true",
                    help="Use teacher fine-tuned on TinyFace instead of frozen pretrained")
    p.add_argument("--teacher_finetune_ckpt_path", type=str, default=None,
                    help="Path to fine-tuned teacher checkpoint (required if --teacher_finetuned)")

    # KD
    p.add_argument("--kd_alpha", type=float, default=0.7,
                    help="Weight for hard loss; (1-alpha) for feature KD loss")
    p.add_argument("--kd_feature_loss", type=str, default="cosine",
                    choices=["cosine", "mse"])

    # LoRA (optional)
    p.add_argument("--lora_enabled", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_family", type=str, default="auto",
                    choices=["auto", "vit", "swin", "pvt", "mobilevit",
                             "levit", "efficientformer", "cnn"])
    p.add_argument("--lora_target_regex", type=str, default=None)
    p.add_argument("--lora_qkv_proj", type=int, default=LoRATargetChoice.NONE.value,
                    choices=[e.value for e in LoRATargetChoice])
    p.add_argument("--lora_train_bias", type=int, default=0, choices=[0, 1])

    # Selective fine-tuning
    p.add_argument("--thawed_modules", type=str, default="",
                    help="Comma-separated thaw rules (same as training_pipeline.py)")

    # Warmup
    p.add_argument("--warmup_epochs", type=int, default=None)

    # Gradual unfreeze
    p.add_argument("--gradual_unfreeze", type=str, choices=["block", "stage"], default=None)
    p.add_argument("--unfreeze_every_epochs", type=int, default=None)
    p.add_argument("--thaw_N_blocks", type=int, default=None)
    p.add_argument("--thaw_block_pct", type=float, default=1.0)
    p.add_argument("--thaw_N_stages", type=int, default=None)
    p.add_argument("--thaw_stage_pct", type=float, default=1.0)

    # Debug
    p.add_argument("--print_trainable_params", action="store_true")

    # Eval
    p.add_argument("--rank_k", type=int, default=10)
    p.add_argument("--probe_batch_size", type=int, default=256)
    p.add_argument("--multi_label_method", type=str, default="MAX",
                    choices=[m.name for m in MultiLabelMethod])
    p.add_argument("--eval_on", type=str, default="EVAL",
                    choices=[d.name for d in DatasetType])
    p.add_argument("--no_roc", action="store_true", default=False)

    # Runtime
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--log_every_n_steps", type=int, default=50)
    p.add_argument("--deterministic", action="store_true", default=False)

    # Paths / W&B
    p.add_argument("--tinyface_root", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="tinyface-kd")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb_model", action="store_true", default=False)

    return p.parse_args()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    args = parse_args()

    # Validate
    if args.teacher_finetuned and not args.teacher_finetune_ckpt_path:
        raise ValueError("--teacher_finetune_ckpt_path required when --teacher_finetuned is set")

    L.seed_everything(args.seed, workers=True)

    tinyface_root = args.tinyface_root or get_tinyface_path()

    # Data (student resolution)
    dm = TinyFaceDataModule(
        tinyface_root=tinyface_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    # Infer num_classes
    train_loader = dm.train_dataloader()
    num_classes = len(train_loader.dataset.subject_ids)

    # Head defaults
    if args.head_type in ["cosface", "adaface", "arcface"]:
        if args.head_scale is None:
            args.head_scale = 64.0
        if args.head_margin is None:
            args.head_margin = 0.4

    if args.head_type == "adaface":
        if args.adaface_h is None:
            args.adaface_h = 0.33
        if args.adaface_t_alpha is None:
            args.adaface_t_alpha = 0.01

    # LoRA defaults
    if args.lora_enabled and args.lora_target_regex is None:
        if args.lora_qkv_proj == LoRATargetChoice.NONE.value:
            args.lora_qkv_proj = LoRATargetChoice.BOTH.value

    lora_train_bias = bool(args.lora_train_bias)

    # Model
    model = KDTimmIDModule(
        student_model_name=args.student_model_name,
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
        # Teacher
        teacher_type=args.teacher_type,
        teacher_finetuned=args.teacher_finetuned,
        teacher_finetune_ckpt_path=args.teacher_finetune_ckpt_path,
        # KD
        kd_alpha=args.kd_alpha,
        kd_feature_loss=args.kd_feature_loss,
        # LoRA
        lora_enabled=args.lora_enabled,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_family=args.lora_family,
        lora_target_regex=args.lora_target_regex,
        lora_apply_qkv=args.lora_qkv_proj in (LoRATargetChoice.QKV.value, LoRATargetChoice.BOTH.value),
        lora_apply_proj=args.lora_qkv_proj in (LoRATargetChoice.PROJ.value, LoRATargetChoice.BOTH.value),
        lora_train_bias=lora_train_bias,
    )

    # W&B
    teacher_tag = args.teacher_type
    if args.teacher_finetuned:
        teacher_tag += "-ft"

    head_tag = f"-{args.head_type}"
    if args.head_type in ["cosface", "adaface", "arcface"]:
        head_tag += f":s{args.head_scale}:m{args.head_margin}"
    if args.head_type == "adaface":
        head_tag += f":h{args.adaface_h}:ta{args.adaface_t_alpha}"

    lora_tag = ""
    if args.lora_enabled:
        lora_tag = (
            f"-lora(r{args.lora_r}-a{args.lora_alpha}-d{args.lora_dropout})"
            f"{'qkv' if args.lora_qkv_proj in (LoRATargetChoice.QKV.value, LoRATargetChoice.BOTH.value) else ''}"
            f"{'proj' if args.lora_qkv_proj in (LoRATargetChoice.PROJ.value, LoRATargetChoice.BOTH.value) else ''}"
        )

    run_name = args.wandb_run_name or (
        f"kd-{args.student_model_name}-{teacher_tag}"
        f"-img{args.img_size}-bs{args.batch_size}-lr{args.lr}"
        f"-a{args.kd_alpha}{head_tag}{lora_tag}"
    )

    wandb_config = vars(args).copy()
    wandb_config["num_classes"] = num_classes
    wandb_config["tinyface_root"] = tinyface_root

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
            save_last=False,
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

    # Gradual unfreeze (same logic as training_pipeline.py)
    if args.gradual_unfreeze is not None:
        from lightning_modules.timm_id_module import (
            _collect_block_modules, _collect_stage_modules,
        )
        from lightning_modules._infer_family import _infer_family

        if args.warmup_epochs is None:
            every_n_epochs = 1
            model.warmup_epochs = min(max(1, every_n_epochs), max(1, args.max_epochs // 10))

        family = _infer_family(args.student_model_name)

        if args.gradual_unfreeze == "block":
            block_names = _collect_block_modules(model.backbone, family)
            name_to_mod = dict(model.backbone.named_modules())
            blocks_all = [name_to_mod[n] for n in reversed(block_names) if n in name_to_mod]
            num_blocks_total = len(blocks_all)
            if num_blocks_total == 0:
                raise RuntimeError("No blocks found for gradual unfreeze.")
            if args.thaw_N_blocks is not None:
                num_to_thaw = min(args.thaw_N_blocks, num_blocks_total)
            else:
                num_to_thaw = max(1, int(ceil(args.thaw_block_pct * num_blocks_total)))
            blocks = blocks_all[:num_to_thaw]

            if args.unfreeze_every_epochs is None:
                every_n_epochs = max(1, (args.max_epochs // 2) // num_to_thaw)
            else:
                every_n_epochs = max(1, int(args.unfreeze_every_epochs))

            callbacks.append(GradualBlockUnfreezeCallback(blocks=blocks, every_n_epochs=every_n_epochs))

        elif args.gradual_unfreeze == "stage":
            stage_names = _collect_stage_modules(model.backbone, family)
            name_to_mod = dict(model.backbone.named_modules())
            stages_all = [name_to_mod[n] for n in reversed(stage_names) if n in name_to_mod]
            num_stages_total = len(stages_all)
            if num_stages_total == 0:
                raise RuntimeError("No stages found for gradual unfreeze.")
            if args.thaw_N_stages is not None:
                num_to_thaw = min(args.thaw_N_stages, num_stages_total)
            else:
                num_to_thaw = max(1, int(ceil(args.thaw_stage_pct * num_stages_total)))
            stages = stages_all[:num_to_thaw]

            if args.unfreeze_every_epochs is None:
                every_n_epochs = max(1, (args.max_epochs // 2) // num_to_thaw)
            else:
                every_n_epochs = max(1, int(args.unfreeze_every_epochs))

            callbacks.append(GradualStageUnfreezeCallback(stages=stages, every_n_epochs=every_n_epochs))

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
