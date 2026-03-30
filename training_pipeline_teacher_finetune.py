"""Training pipeline for fine-tuning a teacher backbone on TinyFace.

Produces:
  - Fine-tuned teacher checkpoint (backbone weights + AdaFace head)
  - Teacher baseline rank@1 on TinyFace (via evaluation callback)

After fine-tuning, the checkpoint can be loaded by
``get_teacher(teacher_type, finetune_ckpt_path=...)`` for KD.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional

import torch
torch.set_float32_matmul_precision("high")
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from torch.optim.lr_scheduler import LambdaLR

from datasets.tinyface.dataloader import (
    get_tinyface_path, DatasetType,
    get_train_loader, get_eval_loaders, get_test_loaders,
    get_training_transforms, get_eval_transforms,
    find_dataset_dir, TinyFaceDataset,
)
from lightning_modules.heads import AdaFaceHead, ArcFaceHead, CosFaceHead
from lightning_modules.teacher_wrapper import (
    _CVLFACE_REPO_ID, _CVLFACE_CACHE,
    _download_cvlface, _load_cvlface_model,
)
from lightning_modules.callbacks import TinyFaceEvaluationCallback
from model_eval.probe_gallery import (
    MultiLabelMethod,
    get_batch_features,
    get_cosine_similarity,
    get_rank_k_accuracy,
    get_roc_metrics,
)

from logging import getLogger
logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Teacher-resolution datamodule
# ---------------------------------------------------------------------------
class TeacherTinyFaceDataModule(L.LightningDataModule):
    """TinyFaceDataModule variant that uses teacher resolution (112×112) and
    teacher normalisation (mean=0.5, std=0.5)."""

    def __init__(
        self,
        tinyface_root: str,
        img_size: int = 112,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.tinyface_root = tinyface_root
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _teacher_train_transforms(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _teacher_eval_transforms(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        tinyfaces_path = find_dataset_dir(self.tinyface_root, "Training_Set")
        train_dataset = TinyFaceDataset(
            tinyfaces_path,
            crossval_splits=list(range(9)),
            transform=self._teacher_train_transforms(),
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        tinyfaces_path = find_dataset_dir(self.tinyface_root, "Training_Set")
        # Use eval split (crossval 9) as probe set for validation
        subject_ids = sorted(list(self._get_crossval_ids(9)))
        subid_to_label_map = {sub_id: idx for idx, sub_id in enumerate(subject_ids)}
        probe_dataset = TinyFaceDataset(
            tinyfaces_path,
            transform=self._teacher_eval_transforms(),
            subid_to_label_map=subid_to_label_map,
        )
        return DataLoader(
            probe_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @staticmethod
    def _get_crossval_ids(split_idx: int):
        from datasets.tinyface.dataloader import _get_crossval_ids
        return _get_crossval_ids(split_idx)


# ---------------------------------------------------------------------------
# Teacher evaluation callback (operates at teacher resolution)
# ---------------------------------------------------------------------------
class TeacherEvaluationCallback(Callback):
    """Evaluation callback that uses teacher normalization ([-1,1]) instead of
    ImageNet normalization."""

    def __init__(
        self,
        tinyface_root: str,
        img_size: int = 112,
        batch_size: int = 64,
        rank_k: int = 10,
        probe_batch_size: int = 256,
        multi_label_method: MultiLabelMethod = MultiLabelMethod.ALL,
        also_log_roc: bool = True,
        eval_on: DatasetType = DatasetType.EVAL,
    ):
        super().__init__()
        self.tinyface_root = tinyface_root
        self.img_size = img_size
        self.batch_size = batch_size
        self.rank_k = rank_k
        self.probe_batch_size = probe_batch_size
        self.multi_label_method = multi_label_method
        self.also_log_roc = also_log_roc
        self.eval_on = eval_on

    def _get_teacher_eval_transforms(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()

        transform = self._get_teacher_eval_transforms()

        if self.eval_on == DatasetType.EVAL:
            # Rebuild eval loaders with teacher transforms
            gallery_loader, probe_loader = self._get_teacher_eval_loaders(transform)
        else:
            gallery_loader, probe_loader = self._get_teacher_test_loaders(transform)

        backbone = pl_module  # TeacherFineTuneModule.forward() returns embeddings

        gallery_features, gallery_labels = get_batch_features(
            backbone, gallery_loader, device=pl_module.device,
        )
        probe_features, probe_labels = get_batch_features(
            backbone, probe_loader, device=pl_module.device,
        )

        cos_sim = get_cosine_similarity(
            probe_features, gallery_features,
            probe_batch_size=self.probe_batch_size,
            device=pl_module.device,
        )

        ranks = get_rank_k_accuracy(
            cos_sim, probe_labels, gallery_labels,
            k=self.rank_k,
            multi_label_method=self.multi_label_method,
        )

        for i, v in enumerate(ranks, start=1):
            pl_module.log(
                f"tinyface/{self.eval_on.name.lower()}/rank@{i}",
                float(v),
                prog_bar=(i == 1),
                on_epoch=True,
                logger=True,
            )

        if self.also_log_roc:
            roc = get_roc_metrics(
                cos_sim, probe_labels, gallery_labels,
                return_dict=True,
                multi_label_method=self.multi_label_method,
            )
            for k, v in roc.items():
                if isinstance(v, float):
                    pl_module.log(
                        f"tinyface/{self.eval_on.name.lower()}/{k}",
                        float(v),
                        on_epoch=True,
                        logger=True,
                    )

    def _get_teacher_eval_loaders(self, transform):
        from datasets.tinyface.dataloader import (
            find_dataset_dir, TinyFaceDataset, _get_crossval_ids,
        )
        from torch.utils.data import DataLoader

        tinyfaces_path = find_dataset_dir(self.tinyface_root, "Training_Set")
        subject_ids = sorted(list(_get_crossval_ids(9)))
        subid_to_label_map = {sub_id: idx for idx, sub_id in enumerate(subject_ids)}

        probe_dataset = TinyFaceDataset(
            tinyfaces_path, transform=transform,
            subid_to_label_map=subid_to_label_map,
        )
        gallery_dataset = TinyFaceDataset(
            tinyfaces_path, transform=transform,
            subid_to_label_map=subid_to_label_map,
        )

        # First image per subject → gallery, rest → probe
        gallery_img_path_indexes = {}
        for idx in range(len(gallery_dataset)):
            _, sub_id = gallery_dataset._get_imgpath_subid(idx)
            if sub_id not in gallery_img_path_indexes:
                gallery_img_path_indexes[sub_id] = idx

        gallery_list = list(gallery_img_path_indexes.values())
        probe_list = list(set(range(len(gallery_dataset))) - set(gallery_img_path_indexes.values()))

        gallery_dataset.img_paths = [gallery_dataset.img_paths[i] for i in gallery_list]
        probe_dataset.img_paths = [probe_dataset.img_paths[i] for i in probe_list]

        probe_loader = DataLoader(probe_dataset, batch_size=self.batch_size, shuffle=False)
        gallery_loader = DataLoader(gallery_dataset, batch_size=self.batch_size, shuffle=False)
        return gallery_loader, probe_loader

    def _get_teacher_test_loaders(self, transform):
        from datasets.tinyface.dataloader import find_dataset_dir, TinyFaceDataset
        from torch.utils.data import DataLoader

        # Test set uses Testing_Set with gallery and probe subsets
        tinyfaces_path = find_dataset_dir(self.tinyface_root, "Testing_Set")

        gallery_dataset = TinyFaceDataset(tinyfaces_path, transform=transform, subset="gallery")
        probe_dataset = TinyFaceDataset(tinyfaces_path, transform=transform, subset="probe")

        gallery_loader = DataLoader(gallery_dataset, batch_size=self.batch_size, shuffle=False)
        probe_loader = DataLoader(probe_dataset, batch_size=self.batch_size, shuffle=False)
        return gallery_loader, probe_loader


# ---------------------------------------------------------------------------
# Lightning module for teacher fine-tuning
# ---------------------------------------------------------------------------
class TeacherFineTuneModule(L.LightningModule):
    """Wraps a teacher backbone with a classification head for TinyFace fine-tuning."""

    def __init__(
        self,
        teacher_type: str = "cvlface_vit_base",
        num_classes: int = 5139,
        head_type: str = "adaface",
        head_scale: float = 64.0,
        head_margin: float = 0.4,
        adaface_h: float = 0.33,
        adaface_t_alpha: float = 0.01,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        label_smoothing: float = 0.0,
        img_size: int = 112,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.label_smoothing = label_smoothing
        self.teacher_type = teacher_type

        # Load pretrained backbone (UNFROZEN for fine-tuning)
        if teacher_type == "cvlface_vit_base":
            _download_cvlface(_CVLFACE_REPO_ID, _CVLFACE_CACHE)
            self.backbone = _load_cvlface_model(_CVLFACE_CACHE)
        else:
            raise ValueError(f"Unknown teacher_type: {teacher_type}")

        # All params trainable
        for p in self.backbone.parameters():
            p.requires_grad = True

        emb_dim = 512  # CVLFace outputs 512-d

        # Classification head
        head_type = head_type.lower()
        if head_type == "adaface":
            self.head = AdaFaceHead(
                emb_dim, num_classes, s=head_scale, m=head_margin,
                h=adaface_h, t_alpha=adaface_t_alpha,
            )
        elif head_type == "arcface":
            self.head = ArcFaceHead(emb_dim, num_classes, s=head_scale, m=head_margin)
        elif head_type == "cosface":
            self.head = CosFaceHead(emb_dim, num_classes, s=head_scale, m=head_margin)
        else:
            raise ValueError(f"Unknown head_type={head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised embeddings (for eval callback)."""
        emb = self.backbone(x)
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        return F.normalize(emb, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        emb = self.backbone(x)
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        logits = self.head(emb, y)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        emb = self(x)
        # No head loss in validation (head needs labels), just log a dummy
        self.log("val/loss", torch.tensor(0.0), prog_bar=False)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        max_epochs = int(self.trainer.max_epochs) if self.trainer is not None else 1
        warmup_epochs = max(0, int(self.warmup_epochs))

        if warmup_epochs > 0:
            min_lr_ratio = 0.001

            def lr_lambda(epoch: int):
                if epoch < warmup_epochs:
                    return float(epoch + 1) / float(warmup_epochs)
                t = (epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
                cosine = 0.5 * (1.0 + math.cos(math.pi * t))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            sched = LambdaLR(opt, lr_lambda=lr_lambda)
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# Checkpoint callback that also saves backbone state_dict separately
# ---------------------------------------------------------------------------
class TeacherCheckpointCallback(Callback):
    """After the best checkpoint is saved, also export the backbone state_dict
    so it can be loaded by ``get_teacher(finetune_ckpt_path=...)``."""

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir

    def on_train_end(self, trainer, pl_module):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, "teacher_backbone.pt")
        torch.save(
            {"backbone_state_dict": pl_module.backbone.state_dict()},
            path,
        )
        logger.info(f"Saved teacher backbone state_dict to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("TinyFace Teacher Fine-Tuning")

    p.add_argument("--teacher_type", type=str, default="cvlface_vit_base",
                    choices=["cvlface_vit_base"])
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=73)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--label_smoothing", type=float, default=0.0)

    p.add_argument("--head_type", type=str, default="adaface",
                    choices=["adaface", "arcface", "cosface"])
    p.add_argument("--head_scale", type=float, default=64.0)
    p.add_argument("--head_margin", type=float, default=0.4)
    p.add_argument("--adaface_h", type=float, default=0.33)
    p.add_argument("--adaface_t_alpha", type=float, default=0.01)

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
    p.add_argument("--wandb_project", type=str, default="vit_book_teacher_finetune")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb_model", action="store_true", default=False)

    return p.parse_args()


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    tinyface_root = args.tinyface_root or get_tinyface_path()

    # Data (teacher resolution + normalisation)
    dm = TeacherTinyFaceDataModule(
        tinyface_root=tinyface_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    # Infer num_classes
    train_loader = dm.train_dataloader()
    num_classes = len(train_loader.dataset.subject_ids)

    # Model
    model = TeacherFineTuneModule(
        teacher_type=args.teacher_type,
        num_classes=num_classes,
        head_type=args.head_type,
        head_scale=args.head_scale,
        head_margin=args.head_margin,
        adaface_h=args.adaface_h,
        adaface_t_alpha=args.adaface_t_alpha,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        img_size=args.img_size,
    )

    # W&B
    run_name = args.wandb_run_name or (
        f"teacher-ft-{args.teacher_type}-img{args.img_size}"
        f"-bs{args.batch_size}-lr{args.lr}-{args.head_type}"
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
        TeacherEvaluationCallback(
            tinyface_root=tinyface_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            rank_k=args.rank_k,
            probe_batch_size=args.probe_batch_size,
            multi_label_method=MultiLabelMethod[args.multi_label_method],
            also_log_roc=not args.no_roc,
            eval_on=DatasetType[args.eval_on],
        ),
        TeacherCheckpointCallback(save_dir=ckpt_dir),
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
