"""Structured pruning for Swin and DeiT3 vision transformers.

Supports two complementary strategies:

1. **Attention head pruning** — remove entire heads from MHSA layers based
   on magnitude importance, then surgically resize QKV / proj weight tensors.
2. **Block dropping** — remove whole encoder blocks (last-first or
   similarity-based), reducing effective model depth.

Both strategies produce dense sub-networks that run on standard hardware
without sparse kernels.  A brief fine-tuning pass is recommended after
pruning to recover accuracy.
"""
import argparse
import copy
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from lightning_modules._infer_family import _infer_family
from lightning_modules.timm_id_module import TimmIDModule


# =====================================================================
#  Utility helpers
# =====================================================================

def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _param_count_str(model: nn.Module) -> str:
    n = _param_count(model)
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n / 1e3:.1f}K"


def _validate_forward(model: nn.Module, img_size: int = 96) -> torch.Size:
    """Perform a single forward pass and return output shape."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, img_size, img_size, device=device))
    return out.shape


# =====================================================================
#  Attention-module discovery
# =====================================================================

def _find_attention_modules_vit(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return ``[(name, attn_module), ...]`` for every attention layer in a
    flat ViT / DeiT3 model (``model.blocks.*.attn``)."""
    results = []
    for i, block in enumerate(model.blocks):
        results.append((f"blocks.{i}.attn", block.attn))
    return results


def _find_attention_modules_swin(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return ``[(name, attn_module), ...]`` for every attention layer in a
    Swin model (``model.layers.{stage}.blocks.{block}.attn``)."""
    results = []
    for si, stage in enumerate(model.layers):
        for bi, block in enumerate(stage.blocks):
            results.append((f"layers.{si}.blocks.{bi}.attn", block.attn))
    return results


# =====================================================================
#  Head importance scoring
# =====================================================================

def _magnitude_importance_qkv(attn: nn.Module) -> torch.Tensor:
    """Compute per-head importance as the L1 norm over QKV + proj weights.

    Returns a 1-D tensor of shape ``(num_heads,)`` with importance scores.
    """
    num_heads = attn.num_heads
    qkv_w = attn.qkv.weight.data  # (3 * num_heads * head_dim, embed_dim)
    head_dim = qkv_w.shape[0] // (3 * num_heads)

    # Reshape to (3, num_heads, head_dim, embed_dim)
    qkv_3d = qkv_w.view(3, num_heads, head_dim, -1)
    qkv_importance = qkv_3d.abs().sum(dim=(0, 2, 3))  # (num_heads,)

    proj_w = attn.proj.weight.data  # (embed_dim, num_heads * head_dim)
    proj_3d = proj_w.view(-1, num_heads, head_dim)
    proj_importance = proj_3d.abs().sum(dim=(0, 2))  # (num_heads,)

    return qkv_importance + proj_importance


# =====================================================================
#  Head pruning — weight surgery
# =====================================================================

def _prune_linear(linear: nn.Linear, indices: torch.Tensor, dim: int) -> nn.Linear:
    """Return a new ``nn.Linear`` keeping only *indices* along *dim*.

    ``dim=0`` selects output features (rows), ``dim=1`` selects input
    features (columns).
    """
    indices = indices.to(linear.weight.device)
    w = linear.weight.data.index_select(dim, indices)
    b = None
    if linear.bias is not None:
        if dim == 0:
            b = linear.bias.data.index_select(0, indices)
        else:
            b = linear.bias.data.clone()
    new = nn.Linear(w.shape[1], w.shape[0], bias=(b is not None))
    new.weight.data = w
    if b is not None:
        new.bias.data = b
    return new


def _head_indices_to_keep(num_heads: int, head_dim: int,
                          heads_to_keep: List[int]) -> torch.Tensor:
    """Convert head indices to flat feature indices (for QKV row selection)."""
    idx = []
    for h in heads_to_keep:
        idx.extend(range(h * head_dim, (h + 1) * head_dim))
    return torch.tensor(idx, dtype=torch.long)


def _patched_vit_attention_forward(self, x, attn_mask=None):
    """Replacement forward for timm Attention after head pruning.

    Identical to the original except ``reshape(B, N, C)`` is replaced with
    ``reshape(B, N, -1)`` so it works when ``num_heads * head_dim != C``.
    """
    import torch.nn.functional as F
    from timm.layers.attention import maybe_add_mask

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = maybe_add_mask(attn, attn_mask)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, -1)  # ← changed from C to -1
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _prune_attention_heads(attn: nn.Module, heads_to_keep: List[int],
                           family: str) -> None:
    """Prune attention heads **in-place** for a single attention module.

    Resizes ``qkv``, ``proj``, and related parameters (including Swin's
    relative position bias table).  For VIT/DeiT3 models, monkey-patches
    the forward to handle the reduced head dimension correctly.
    """
    num_heads = attn.num_heads
    head_dim = attn.qkv.weight.shape[0] // (3 * num_heads)
    new_num_heads = len(heads_to_keep)

    # QKV: shape (3*H*D, E).  Keep rows for selected heads × 3 chunks.
    qkv_indices = []
    for chunk in range(3):
        offset = chunk * num_heads * head_dim
        for h in heads_to_keep:
            qkv_indices.extend(range(offset + h * head_dim,
                                     offset + (h + 1) * head_dim))
    qkv_idx = torch.tensor(qkv_indices, dtype=torch.long)
    attn.qkv = _prune_linear(attn.qkv, qkv_idx, dim=0)

    # Proj: shape (E, H*D).  Keep columns for selected heads.
    proj_col_idx = _head_indices_to_keep(num_heads, head_dim, heads_to_keep)
    attn.proj = _prune_linear(attn.proj, proj_col_idx, dim=1)

    # Swin: relative position bias table — shape (num_pos, num_heads)
    if family == "swin" and hasattr(attn, "relative_position_bias_table"):
        rpb = attn.relative_position_bias_table.data  # (N, H)
        keep = torch.tensor(heads_to_keep, dtype=torch.long,
                            device=rpb.device)
        attn.relative_position_bias_table = nn.Parameter(
            rpb.index_select(1, keep)
        )

    attn.num_heads = new_num_heads

    # VIT/DeiT3: monkey-patch forward to handle reduced head count
    # (timm Attention uses reshape(B, N, C) which fails after head pruning)
    if family == "vit":
        import types
        attn.forward = types.MethodType(
            _patched_vit_attention_forward, attn
        )


# =====================================================================
#  AttentionHeadPruner
# =====================================================================

class AttentionHeadPruner:
    """Prune attention heads based on importance scores.

    Parameters
    ----------
    model : nn.Module
        The backbone (``timm`` model with ``num_classes=0``).
    family : str
        ``"vit"`` or ``"swin"`` (from ``_infer_family``).
    """

    def __init__(self, model: nn.Module, family: str):
        self.model = model
        self.family = family

    # -----------------------------------------------------------------
    def compute_head_importance(self, method: str = "magnitude"
                                ) -> Dict[str, torch.Tensor]:
        """Return ``{attn_name: importance_tensor}`` for every attention
        layer.  *method* can be ``"magnitude"`` (L1 norm over QKV+proj).
        """
        if method != "magnitude":
            raise ValueError(f"Unsupported importance method: {method}")

        attn_modules = (
            _find_attention_modules_vit(self.model) if self.family == "vit"
            else _find_attention_modules_swin(self.model)
        )
        return {name: _magnitude_importance_qkv(attn)
                for name, attn in attn_modules}

    # -----------------------------------------------------------------
    def prune_heads(self, prune_ratio: float = 0.25,
                    method: str = "magnitude",
                    min_heads: int = 1) -> Dict[str, List[int]]:
        """Globally rank heads and remove the bottom *prune_ratio* fraction.

        Each layer keeps at least *min_heads* heads (guardrail: ≥ 50% is
        enforced on top of that).

        Returns a dict ``{attn_name: [kept_head_indices]}``.
        """
        if not 0.0 < prune_ratio < 1.0:
            raise ValueError(f"prune_ratio must be in (0, 1), got {prune_ratio}")

        importance = self.compute_head_importance(method)

        # Build a flat list of (score, attn_name, head_idx)
        all_heads = []
        for name, scores in importance.items():
            for h in range(scores.numel()):
                all_heads.append((scores[h].item(), name, h))

        # Sort ascending (least important first)
        all_heads.sort(key=lambda t: t[0])

        n_total = len(all_heads)
        n_prune = int(n_total * prune_ratio)

        # Collect which heads to prune, respecting per-layer minimums
        prune_set: Dict[str, set] = {name: set() for name in importance}
        layer_sizes = {name: scores.numel() for name, scores in importance.items()}

        pruned_count = 0
        for score, name, h_idx in all_heads:
            if pruned_count >= n_prune:
                break
            layer_n = layer_sizes[name]
            already_pruned = len(prune_set[name])
            # Keep at least max(min_heads, ceil(layer_n / 2)) heads
            min_keep = max(min_heads, math.ceil(layer_n / 2))
            remaining = layer_n - already_pruned
            if remaining > min_keep:
                prune_set[name].add(h_idx)
                pruned_count += 1

        # Derive keep-lists and perform surgery
        keep_map: Dict[str, List[int]] = {}
        attn_modules = dict(
            _find_attention_modules_vit(self.model) if self.family == "vit"
            else _find_attention_modules_swin(self.model)
        )
        for name, attn in attn_modules.items():
            num_h = layer_sizes[name]
            heads_to_keep = sorted(
                set(range(num_h)) - prune_set[name]
            )
            keep_map[name] = heads_to_keep
            if len(heads_to_keep) < num_h:
                _prune_attention_heads(attn, heads_to_keep, self.family)

        print(f"Head pruning: {pruned_count}/{n_total} heads removed "
              f"({pruned_count / n_total:.1%})")
        return keep_map


# =====================================================================
#  Block dropping
# =====================================================================

class BlockPruner:
    """Remove entire transformer encoder blocks.

    Parameters
    ----------
    model : nn.Module
        The backbone (``timm`` model with ``num_classes=0``).
    family : str
        ``"vit"`` or ``"swin"``.
    """

    def __init__(self, model: nn.Module, family: str):
        self.model = model
        self.family = family

    # -----------------------------------------------------------------
    #  VIT helpers
    # -----------------------------------------------------------------
    def _get_block_count_vit(self) -> int:
        return len(self.model.blocks)

    def _drop_blocks_vit(self, indices_to_remove: List[int]) -> None:
        blocks = list(self.model.blocks)
        remaining = [b for i, b in enumerate(blocks)
                     if i not in set(indices_to_remove)]
        self.model.blocks = nn.Sequential(*remaining)

    # -----------------------------------------------------------------
    #  Swin helpers — only drop blocks WITHIN a stage, never the
    #  patch-merging downsamples between stages.
    # -----------------------------------------------------------------
    def _get_block_counts_swin(self) -> List[int]:
        return [len(stage.blocks) for stage in self.model.layers]

    def _drop_blocks_swin(self, stage_block_map: Dict[int, List[int]]) -> None:
        for stage_idx, block_indices in stage_block_map.items():
            stage = self.model.layers[stage_idx]
            remaining = [b for i, b in enumerate(stage.blocks)
                         if i not in set(block_indices)]
            stage.blocks = nn.Sequential(*remaining)

    # -----------------------------------------------------------------
    #  Similarity-based importance
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _block_similarity_vit(self, img_size: int = 96) -> List[float]:
        """Cosine similarity between each block's input and output.

        High similarity → block is near-identity → safe to drop.
        """
        self.model.eval()
        x = torch.randn(1, 3, img_size, img_size)

        # Forward through patch embed + pos embed (use model's own method)
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)

        sims = []
        for block in self.model.blocks:
            inp = x.clone()
            x = block(x)
            cos = nn.functional.cosine_similarity(
                inp.flatten(), x.flatten(), dim=0
            ).item()
            sims.append(cos)
        return sims

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------
    def prune_blocks(self, num_blocks_to_remove: int,
                     strategy: str = "last_first") -> List:
        """Remove *num_blocks_to_remove* blocks.

        For ViT/DeiT3 this is straightforward.  For Swin, blocks are
        removed from the **largest stage** (stage 2 typically has the most
        blocks), preserving all stages and their downsamples.

        Returns the list/dict of removed indices.
        """
        if self.family == "vit":
            return self._prune_blocks_vit(num_blocks_to_remove, strategy)
        elif self.family == "swin":
            return self._prune_blocks_swin(num_blocks_to_remove, strategy)
        else:
            raise ValueError(f"Unsupported family: {self.family}")

    def _prune_blocks_vit(self, n_remove: int,
                          strategy: str) -> List[int]:
        n_blocks = self._get_block_count_vit()
        if n_remove >= n_blocks:
            raise ValueError(
                f"Cannot remove {n_remove} blocks from {n_blocks}-block model"
            )
        # Guardrail: keep at least 50%
        max_remove = n_blocks // 2
        n_remove = min(n_remove, max_remove)

        if strategy == "last_first":
            to_remove = list(range(n_blocks - n_remove, n_blocks))
        elif strategy == "similarity":
            sims = self._block_similarity_vit()
            ranked = sorted(range(n_blocks), key=lambda i: sims[i],
                            reverse=True)  # most similar first
            to_remove = sorted(ranked[:n_remove])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self._drop_blocks_vit(to_remove)
        print(f"Block pruning (VIT): removed {len(to_remove)}/{n_blocks} "
              f"blocks {to_remove}")
        return to_remove

    def _prune_blocks_swin(self, n_remove: int,
                           strategy: str) -> Dict[int, List[int]]:
        counts = self._get_block_counts_swin()
        # Only prune the largest stage (typically stage 2)
        target_stage = max(range(len(counts)), key=lambda i: counts[i])
        n_in_stage = counts[target_stage]
        # Guardrail: keep at least 50% of the stage
        max_remove = n_in_stage // 2
        n_remove = min(n_remove, max_remove)

        if strategy == "last_first":
            to_remove = list(range(n_in_stage - n_remove, n_in_stage))
        elif strategy == "similarity":
            # Forward through the model up to and through target stage
            raise NotImplementedError(
                "Similarity-based Swin block pruning not yet implemented"
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        stage_map = {target_stage: to_remove}
        self._drop_blocks_swin(stage_map)
        print(f"Block pruning (Swin): removed {len(to_remove)}/{n_in_stage} "
              f"blocks from stage {target_stage}: {to_remove}")
        return stage_map


# =====================================================================
#  High-level prune function
# =====================================================================

def prune_model(
    backbone: nn.Module,
    family: str,
    prune_method: str = "heads",
    head_prune_ratio: float = 0.25,
    num_blocks_to_remove: int = 0,
    block_strategy: str = "last_first",
    img_size: int = 96,
) -> nn.Module:
    """Apply pruning to a backbone and validate the result.

    Parameters
    ----------
    backbone : nn.Module
        timm backbone (num_classes=0).  Should be a **deep copy**.
    family : str
        "vit" or "swin".
    prune_method : str
        "heads", "blocks", or "both".
    head_prune_ratio : float
        Fraction of heads to remove (0–1 exclusive).
    num_blocks_to_remove : int
        Number of blocks to drop (only used if method includes blocks).
    block_strategy : str
        "last_first" or "similarity".
    img_size : int
        Input image resolution for validation.

    Returns
    -------
    nn.Module
        The pruned backbone (same object that was passed in).
    """
    params_before = _param_count(backbone)
    print(f"Before pruning: {_param_count_str(backbone)}")

    if prune_method in ("heads", "both"):
        pruner = AttentionHeadPruner(backbone, family)
        pruner.prune_heads(prune_ratio=head_prune_ratio)

    if prune_method in ("blocks", "both"):
        if num_blocks_to_remove <= 0:
            raise ValueError("num_blocks_to_remove must be > 0 for block pruning")
        pruner = BlockPruner(backbone, family)
        pruner.prune_blocks(num_blocks_to_remove, strategy=block_strategy)

    params_after = _param_count(backbone)
    reduction = 1.0 - params_after / params_before
    print(f"After pruning:  {_param_count_str(backbone)} "
          f"(−{reduction:.1%})")

    # Validate forward pass
    out_shape = _validate_forward(backbone, img_size)
    print(f"Forward pass OK: output shape {out_shape}")

    return backbone


# =====================================================================
#  Checkpoint helpers
# =====================================================================

def _is_kd_checkpoint(ckpt_path: str) -> bool:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return "kd_alpha" in ckpt.get("hyper_parameters", {})


def _has_lora(module: nn.Module) -> bool:
    from lightning_modules.lora import LoRALinear
    return any(isinstance(m, LoRALinear) for m in module.modules())


def _merge_lora(module: nn.Module) -> None:
    """Merge LoRA weights in-place (same as export_onnx.merge_lora_weights)."""
    from lightning_modules.lora import LoRALinear, _get_parent
    for name, m in list(module.named_modules()):
        if not isinstance(m, LoRALinear):
            continue
        merged_weight = m.base.weight.data + m.scaling * (
            m.lora_B.weight.data @ m.lora_A.weight.data
        )
        new_linear = nn.Linear(
            m.base.in_features, m.base.out_features,
            bias=m.base.bias is not None,
        )
        new_linear.weight.data = merged_weight
        if m.base.bias is not None:
            new_linear.bias.data = m.base.bias.data
        parent, attr = _get_parent(module, name)
        setattr(parent, attr, new_linear)


def load_module_for_finetuning(
    ckpt_path: str, model_name: str
) -> Tuple[TimmIDModule, str]:
    """Load a TimmIDModule or KDTimmIDModule checkpoint, merge LoRA if
    present, and return a ``(TimmIDModule, family)`` pair ready for pruning
    and fine-tuning.

    For KD checkpoints the student backbone + head are transplanted into a
    fresh TimmIDModule (no teacher, no projection).
    """
    is_kd = _is_kd_checkpoint(ckpt_path)

    if is_kd:
        from lightning_modules.kd_module import KDTimmIDModule
        kd = KDTimmIDModule.load_from_checkpoint(ckpt_path)
        effective_name = kd.hparams.get(
            "student_model_name", kd.student_model_name
        )
        # Merge LoRA in KD backbone
        if _has_lora(kd.backbone):
            print("Merging LoRA adapters in KD backbone …")
            _merge_lora(kd.backbone)

        # Re-package into a TimmIDModule
        module = TimmIDModule(
            model_name=effective_name,
            num_classes=kd.num_classes,
            head_type=kd.hparams.get("head_type", "adaface"),
            head_scale=kd.hparams.get("head_scale", 64.0),
            head_margin=kd.hparams.get("head_margin", 0.4),
            adaface_h=kd.hparams.get("adaface_h", 0.33),
            adaface_t_alpha=kd.hparams.get("adaface_t_alpha", 0.01),
            img_size=kd.hparams.get("img_size", 96),
            pretrained=False,
            lora_enabled=False,
            thawed_modules="all",
        )
        # Transplant trained weights
        module.backbone = kd.backbone
        module.head = kd.head
    else:
        module = TimmIDModule.load_from_checkpoint(
            ckpt_path, model_name=model_name,
        )
        effective_name = model_name
        if _has_lora(module.backbone):
            print("Merging LoRA adapters …")
            _merge_lora(module.backbone)

    family = _infer_family(effective_name)
    print(f"Loaded module: {effective_name} (family={family}, "
          f"KD={is_kd}, params={_param_count_str(module)})")
    return module, family


def save_pruned_checkpoint(backbone: nn.Module, output_path: str,
                           metadata: Optional[dict] = None) -> None:
    """Save pruned backbone state_dict with optional metadata."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": backbone.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, output_path)
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Saved pruned checkpoint: {output_path} ({size_mb:.1f} MB)")


# =====================================================================
#  Fine-tuning
# =====================================================================

def finetune_pruned_model(
    module: TimmIDModule,
    finetune_epochs: int = 10,
    finetune_lr: float = 3e-5,
    batch_size: int = 512,
    img_size: int = 96,
    wandb_project: str = "vit_book_pruning",
    wandb_run_name: Optional[str] = None,
    output_ckpt: Optional[str] = None,
) -> str:
    """Fine-tune a pruned TimmIDModule and return the best checkpoint path.

    All parameters are unfrozen.  Uses the same training infrastructure as
    ``training_pipeline.py`` (TinyFace data, AdaFace head, evaluation
    callback, W&B logging, ModelCheckpoint).
    """
    import os
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from datasets.tinyface.dataloader import get_tinyface_path, DatasetType
    from lightning_modules.tinyface_datamodule import TinyFaceDataModule
    from lightning_modules.callbacks import TinyFaceEvaluationCallback
    from model_eval.probe_gallery import MultiLabelMethod

    # Override optimiser settings for fine-tuning
    module.lr = finetune_lr
    module.warmup_epochs = 0
    module.thawed_modules_arg = "all"
    module._thaw_applied = False
    module.lora_enabled = False
    # LoRAConfig is frozen; replace with a disabled copy
    from lightning_modules.lora import LoRAConfig
    module.lora_cfg = LoRAConfig(enabled=False)

    tinyface_root = get_tinyface_path()
    dm = TinyFaceDataModule(
        tinyface_root=tinyface_root,
        img_size=img_size,
        batch_size=batch_size,
    )

    run_name = wandb_run_name or f"prune-{module.model_name}-ft{finetune_epochs}"
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=run_name,
        log_model="best",
    )

    run_id = wandb_logger.experiment.id
    wandb_dir = os.environ.get("WANDB_DIR", "./wandb")
    ckpt_dir = output_ckpt or os.path.join(
        wandb_dir, "checkpoints", wandb_project, run_id
    )
    if output_ckpt:
        ckpt_dir = str(Path(output_ckpt).parent)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="tinyface/eval/rank@1",
            mode="max",
            save_top_k=1,
            save_last=False,
            filename="{epoch}-rank1{tinyface/eval/rank@1:.3f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        TinyFaceEvaluationCallback(
            tinyface_root=tinyface_root,
            img_size=img_size,
            batch_size=batch_size,
            rank_k=10,
            probe_batch_size=256,
            multi_label_method=MultiLabelMethod.MAX,
            also_log_roc=False,
            eval_on=DatasetType.EVAL,
        ),
    ]

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        max_epochs=finetune_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=50,
    )

    trainer.fit(module, datamodule=dm)

    # Return best checkpoint path
    best_path = callbacks[0].best_model_path
    best_eval_rank1 = callbacks[0].best_model_score
    print(f"Fine-tuning complete.  Best checkpoint: {best_path}")
    print(f"  Best eval rank@1: {best_eval_rank1:.4f}")

    # ------------------------------------------------------------------
    # Test-set evaluation: reload best weights into the pruned module
    # ------------------------------------------------------------------
    print("Running TEST-set evaluation on best checkpoint …")
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    module.load_state_dict(best_state["state_dict"])
    module.eval()

    test_cb = TinyFaceEvaluationCallback(
        tinyface_root=tinyface_root,
        img_size=img_size,
        batch_size=batch_size,
        rank_k=10,
        probe_batch_size=256,
        multi_label_method=MultiLabelMethod.MAX,
        also_log_roc=False,
        eval_on=DatasetType.TEST,
    )
    test_trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[test_cb],
        logger=wandb_logger,
        enable_checkpointing=False,
    )
    test_results = test_trainer.validate(model=module, datamodule=dm)
    test_rank1 = test_results[0].get("tinyface/test/rank@1", float("nan"))
    print(f"  Test rank@1: {test_rank1:.4f}")

    return best_path, best_eval_rank1, test_rank1


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Structured pruning (+ optional fine-tuning) for "
                    "Swin / DeiT3 backbones"
    )
    parser.add_argument("--ckpt_path", required=True,
                        help="Lightning checkpoint path")
    parser.add_argument("--model_name", required=True,
                        help="timm model name")
    parser.add_argument("--prune_method",
                        choices=["heads", "blocks", "both"],
                        default="heads")
    parser.add_argument("--head_prune_ratio", type=float, default=0.25,
                        help="Fraction of attention heads to remove")
    parser.add_argument("--num_blocks_to_remove", type=int, default=0,
                        help="Number of encoder blocks to drop")
    parser.add_argument("--block_strategy",
                        choices=["last_first", "similarity"],
                        default="last_first")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--output_ckpt", required=True,
                        help="Path to save the pruned (+ fine-tuned) checkpoint")

    # Fine-tuning options
    parser.add_argument("--finetune_epochs", type=int, default=0,
                        help="Fine-tune epochs after pruning (0 = no fine-tuning)")
    parser.add_argument("--finetune_lr", type=float, default=3e-5,
                        help="Learning rate for recovery fine-tuning")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str,
                        default="vit_book_pruning")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    module, family = load_module_for_finetuning(args.ckpt_path,
                                                args.model_name)
    params_before = _param_count(module)

    prune_model(
        module.backbone,
        family=family,
        prune_method=args.prune_method,
        head_prune_ratio=args.head_prune_ratio,
        num_blocks_to_remove=args.num_blocks_to_remove,
        block_strategy=args.block_strategy,
        img_size=args.img_size,
    )

    params_after = _param_count(module)

    if args.finetune_epochs > 0:
        best_ckpt, eval_rank1, test_rank1 = finetune_pruned_model(
            module,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
            batch_size=args.batch_size,
            img_size=args.img_size,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            output_ckpt=args.output_ckpt,
        )
        print(f"Best fine-tuned checkpoint: {best_ckpt}")
        print(f"  eval rank@1={eval_rank1:.4f}  test rank@1={test_rank1:.4f}")

        # Append results to CSV
        csv_path = Path("pruning_results/pruning_results.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = "experiment,model_name,source_ckpt,prune_method,head_prune_ratio,num_blocks_to_remove,params_before,params_after,compression,finetune_epochs,finetune_lr,eval_rank1,test_rank1,best_ckpt\n"
        if not csv_path.exists():
            csv_path.write_text(header)
        with open(csv_path, "a") as f:
            exp_name = args.wandb_run_name or "unnamed"
            compression = 1.0 - params_after / params_before
            f.write(f"{exp_name},{args.model_name},{args.ckpt_path},"
                    f"{args.prune_method},{args.head_prune_ratio},"
                    f"{args.num_blocks_to_remove},{params_before},{params_after},"
                    f"{compression:.4f},{args.finetune_epochs},{args.finetune_lr},"
                    f"{eval_rank1:.4f},{test_rank1:.4f},{best_ckpt}\n")
    else:
        # Save pruned-only backbone checkpoint
        metadata = {
            "source_ckpt": args.ckpt_path,
            "model_name": args.model_name,
            "family": family,
            "prune_method": args.prune_method,
            "head_prune_ratio": args.head_prune_ratio,
            "num_blocks_to_remove": args.num_blocks_to_remove,
            "block_strategy": args.block_strategy,
            "params_before": params_before,
            "params_after": params_after,
        }
        save_pruned_checkpoint(module.backbone, args.output_ckpt,
                               metadata=metadata)


if __name__ == "__main__":
    main()
