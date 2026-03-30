"""
Knowledge Distillation Lightning Module.

Trains a timm-based student with feature-level KD from a frozen teacher.
The teacher provides embedding supervision; the student retains its own
classification head for TinyFace.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import timm
from torch.optim.lr_scheduler import LambdaLR

from .heads import LinearHead, CosFaceHead, AdaFaceHead, ArcFaceHead
from .lora import LoRAConfig, inject_lora, mark_only_lora_trainable
from .teacher_wrapper import BaseTeacher, get_teacher
from ._infer_family import _infer_family

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _check_img_size_required(model_name: str) -> bool:
    family = _infer_family(model_name)
    if family in {"resnet", "convnext", "pvt"}:
        return False
    elif family in {"vit", "deit", "swin", "mobilevit"}:
        return True
    else:
        raise NotImplementedError(
            f"Image size requirement check not implemented for family '{family}'."
        )


# ---------------------------------------------------------------------------
# KD Module
# ---------------------------------------------------------------------------
class KDTimmIDModule(L.LightningModule):
    """Feature-level knowledge distillation from a frozen teacher into a
    timm-based student backbone + classification head.

    Loss = α * CE(student_logits, y) + (1 - α) * feature_kd_loss
    """

    def __init__(
        self,
        # Student config
        student_model_name: str,
        num_classes: int,
        head_type: str = "adaface",
        head_scale: float = 64.0,
        head_margin: float = 0.4,
        adaface_h: float = 0.33,
        adaface_t_alpha: float = 0.01,
        img_size: int = 96,
        lr: float = 3e-4,
        warmup_epochs: int = 0,
        weight_decay: float = 0.05,
        label_smoothing: float = 0.0,
        backbone_dropout: float = 0.0,
        pretrained: bool = True,
        thawed_modules: str = "head",
        # Teacher config
        teacher_type: str = "cvlface_vit_base",
        teacher_finetuned: bool = False,
        teacher_finetune_ckpt_path: Optional[str] = None,
        # KD params
        kd_alpha: float = 0.7,
        kd_feature_loss: str = "cosine",
        # LoRA (optional)
        lora_enabled: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_family: str = "auto",
        lora_target_regex: Optional[str] = None,
        lora_apply_qkv: bool = True,
        lora_apply_proj: bool = True,
        lora_train_bias: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store hyperparams
        self.student_model_name = student_model_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.kd_alpha = kd_alpha
        self.kd_feature_loss = kd_feature_loss

        # ------------------------------------------------------------------
        # Teacher (frozen, not saved in checkpoint)
        # ------------------------------------------------------------------
        finetune_path = teacher_finetune_ckpt_path if teacher_finetuned else None
        self.teacher: BaseTeacher = get_teacher(teacher_type, finetune_ckpt_path=finetune_path)
        # Ensure teacher is never updated
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------
        # Student backbone
        # ------------------------------------------------------------------
        create_model_kwargs = {"pretrained": pretrained, "num_classes": 0}
        if _check_img_size_required(student_model_name):
            create_model_kwargs["img_size"] = img_size

        self.backbone = timm.create_model(student_model_name, **create_model_kwargs)

        # LoRA injection (optional)
        self.lora_enabled = lora_enabled
        self.lora_train_bias = lora_train_bias
        self.lora_cfg = LoRAConfig(
            enabled=lora_enabled,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            model_name=student_model_name,
            family=lora_family,
            target_regex=lora_target_regex,
            apply_qkv=lora_apply_qkv,
            apply_proj=lora_apply_proj,
        )
        self._lora_replaced: List[str] = []
        if self.lora_cfg.enabled:
            self._lora_replaced = inject_lora(self.backbone, self.lora_cfg)

        student_emb_dim = self.backbone.num_features
        teacher_emb_dim = BaseTeacher.EMB_DIM  # 512

        # ------------------------------------------------------------------
        # Backbone dropout
        # ------------------------------------------------------------------
        self.backbone_dropout = (
            nn.Dropout(backbone_dropout) if backbone_dropout > 0.0 else nn.Identity()
        )

        # ------------------------------------------------------------------
        # Projection layer (student_dim -> teacher_dim) when dims differ
        # ------------------------------------------------------------------
        if student_emb_dim != teacher_emb_dim:
            self.projection: Optional[nn.Linear] = nn.Linear(student_emb_dim, teacher_emb_dim)
        else:
            self.projection = None

        # ------------------------------------------------------------------
        # Classification head (same as TimmIDModule)
        # ------------------------------------------------------------------
        head_type = head_type.lower()
        if head_type == "linear":
            self.head = LinearHead(student_emb_dim, num_classes)
        elif head_type == "cosface":
            self.head = CosFaceHead(student_emb_dim, num_classes, s=head_scale, m=head_margin)
        elif head_type == "adaface":
            self.head = AdaFaceHead(
                student_emb_dim, num_classes, s=head_scale, m=head_margin,
                h=adaface_h, t_alpha=adaface_t_alpha,
            )
        elif head_type == "arcface":
            self.head = ArcFaceHead(student_emb_dim, num_classes, s=head_scale, m=head_margin)
        else:
            raise ValueError(f"Unknown head_type={head_type}")

        # ------------------------------------------------------------------
        # Freeze / thaw state
        # ------------------------------------------------------------------
        self.thawed_modules_arg = thawed_modules
        self._thaw_applied = False

    # ------------------------------------------------------------------
    # Freeze / thaw (reuses TimmIDModule logic via import)
    # ------------------------------------------------------------------
    def on_fit_start(self) -> None:
        if not self._thaw_applied:
            self._apply_thaw_plan()
            self._thaw_applied = True

    def _apply_thaw_plan(self) -> None:
        """Freeze backbone, then apply thaw rules (same as TimmIDModule)."""
        from .timm_id_module import (
            _freeze_all, _thaw_module, _parse_thawed_modules_arg,
            _collect_norm_modules, _collect_attention_modules,
            _collect_block_modules, _collect_stage_modules,
        )

        family = _infer_family(self.student_model_name)

        # Freeze backbone and head
        _freeze_all(self.backbone)
        _freeze_all(self.head)

        # If LoRA, make only LoRA params trainable
        if self.lora_cfg.enabled:
            mark_only_lora_trainable(self.backbone, train_bias=self.lora_train_bias)

        # Projection is always trainable
        if self.projection is not None:
            _thaw_module(self.projection)

        specs = _parse_thawed_modules_arg(self.thawed_modules_arg)

        # Always thaw head unless "no_head"
        include_head = not any(s.key in {"no_head", "nohead"} for s in specs)
        if include_head:
            _thaw_module(self.head)

        norm_names = _collect_norm_modules(self)
        attn_names = _collect_attention_modules(self)
        block_names = _collect_block_modules(self, family)
        stage_names = _collect_stage_modules(self, family)

        import re
        for spec in specs:
            k, v = spec.key, spec.value

            if k in {"head", "default", "no_head", "nohead"}:
                continue

            if k == "all":
                _thaw_module(self.backbone)
                _thaw_module(self.head)
                break

            if k == "all_norm":
                name_to_mod = dict(self.named_modules())
                for n in norm_names:
                    if n in name_to_mod:
                        _thaw_module(name_to_mod[n])
                continue

            if k == "all_attn":
                name_to_mod = dict(self.named_modules())
                for n in attn_names:
                    if n in name_to_mod:
                        _thaw_module(name_to_mod[n])
                continue

            if k in {"last_blocks", "last_block"}:
                if not v or not v.isdigit():
                    raise ValueError("last_blocks requires N")
                n = int(v)
                name_to_mod = dict(self.named_modules())
                for bname in block_names[-n:]:
                    if bname in name_to_mod:
                        _thaw_module(name_to_mod[bname])
                continue

            if k in {"last_stages", "last_stage"}:
                if not v or not v.isdigit():
                    raise ValueError("last_stages requires N")
                n = int(v)
                name_to_mod = dict(self.named_modules())
                for sname in stage_names[-n:]:
                    if sname in name_to_mod:
                        _thaw_module(name_to_mod[sname])
                continue

            if k == "regex":
                if not v:
                    raise ValueError("regex requires a pattern")
                rx = re.compile(v)
                for name, m in self.named_modules():
                    if rx.search(name):
                        _thaw_module(m)
                continue

        # Teacher always frozen (belt & suspenders)
        _freeze_all(self.teacher)

    # ------------------------------------------------------------------
    # Teacher input preparation
    # ------------------------------------------------------------------
    def _prepare_teacher_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert student input (96×96, ImageNet norm) to teacher input
        (112×112, [-1, 1] norm)."""
        mean = IMAGENET_MEAN.to(x.device, x.dtype)
        std = IMAGENET_STD.to(x.device, x.dtype)
        # Undo ImageNet normalisation → [0, 1]
        x_unnorm = x * std + mean
        # Resize to 112×112
        x_resized = F.interpolate(x_unnorm, size=112, mode="bilinear", align_corners=False)
        # Apply teacher normalisation → [-1, 1]
        x_teacher = (x_resized - 0.5) / 0.5
        return x_teacher

    # ------------------------------------------------------------------
    # Forward (inference: returns student embeddings)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns student embeddings. Used by inference.py."""
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch

        # Teacher forward (frozen, different input size/norm)
        x_teacher = self._prepare_teacher_input(x)
        with torch.no_grad():
            teacher_emb = self.teacher(x_teacher)  # [B, 512], L2-normalised

        # Student forward
        student_emb = self.backbone(x)  # [B, student_dim]
        student_emb_drop = self.backbone_dropout(student_emb)
        student_logits = self.head(student_emb_drop, y)

        # Hard loss (CE for TinyFace classification)
        hard_loss = F.cross_entropy(student_logits, y, label_smoothing=self.label_smoothing)

        # Feature KD loss (embedding alignment in teacher's 512-d space)
        projected_emb = self.projection(student_emb) if self.projection is not None else student_emb
        projected_emb = F.normalize(projected_emb, dim=1)
        teacher_emb_norm = F.normalize(teacher_emb, dim=1)

        if self.kd_feature_loss == "cosine":
            kd_loss = 1.0 - F.cosine_similarity(teacher_emb_norm, projected_emb, dim=1).mean()
        else:  # mse
            kd_loss = F.mse_loss(projected_emb, teacher_emb_norm)

        # Combined loss
        loss = self.kd_alpha * hard_loss + (1.0 - self.kd_alpha) * kd_loss

        acc = (student_logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/hard_loss", hard_loss)
        self.log("train/kd_loss", kd_loss)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        max_epochs = int(self.trainer.max_epochs) if self.trainer is not None else 1
        warmup_epochs = max(0, int(getattr(self, "warmup_epochs", 0)))

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
