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


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser("TinyFace TIMM fine-tuning")

    # Model / data
    p.add_argument("--model-name", type=str, default="swin_tiny_patch4_window7_224.ms_in1k")
    p.add_argument("--img-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=73)

    # Optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.0)

    # Selective fine-tuning / freezing
    # Comma-separated list, e.g.:
    #   --thawed-modules head,all_norm,last_blocks=2,all_attn
    # Valid options are defined in lightning_modules/timm_id_module.py.
    p.add_argument(
        "--thawed-modules",
        type=str,
        default="",
        help=(
            "Comma-separated thaw rules. Examples: "
            "'head,all_norm,last_blocks=2' or 'head,all_attn' or 'full'. "
            "Use 'none' to freeze everything (including head)."
        ),
    )

    # Debug / reporting
    p.add_argument(
        "--print-trainable-params",
        action="store_true",
        help="Print trainable parameter names (requires_grad=True) and exit.",
    )

    # Rank-k eval
    p.add_argument("--rank-k", type=int, default=10)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument(
        "--multi-label-method",
        type=str,
        default="MAX",
        choices=[m.name for m in MultiLabelMethod],
    )
    p.add_argument(
        "--eval-on",
        type=str,
        default="EVAL",
        choices=[d.name for d in DatasetType],
    )
    p.add_argument("--no-roc", action="store_true")

    # Runtime
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--log-every-n-steps", type=int, default=50)
    p.add_argument("--deterministic", action="store_true")

    # Paths / W&B
    p.add_argument("--tinyface-root", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default="tinyface-timm")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb-model", action="store_true")

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

    # Model
    model = TimmIDModule(
        model_name=args.model_name,
        num_classes=num_classes,
        img_size=args.img_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        thawed_modules=args.thawed_modules,
        verbose_thaw=args.print_trainable_params,
    )

    # W&B config (everything that matters)
    wandb_config = vars(args).copy()
    wandb_config["num_classes"] = num_classes
    wandb_config["tinyface_root"] = tinyface_root

    run_name = args.wandb_run_name or (
        f"{args.model_name}-img{args.img_size}-bs{args.batch_size}-lr{args.lr}"
    )

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        log_model=not args.no_wandb_model,
        config=wandb_config,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor=f"tinyface/{args.eval_on.lower()}/rank@1",
            mode="max",
            save_top_k=3,
            save_last=True,
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
