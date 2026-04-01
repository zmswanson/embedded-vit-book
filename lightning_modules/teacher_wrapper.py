"""
Teacher model wrappers for knowledge distillation.

BaseTeacher provides an abstract interface. Concrete implementations wrap
specific pretrained face-recognition backbones and expose a frozen forward()
that returns L2-normalised embeddings.

Factory function ``get_teacher()`` returns the appropriate wrapper.
"""
from __future__ import annotations

import os
import sys
import shutil
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseTeacher(nn.Module):
    """Abstract base for frozen teacher models.

    Subclass to add new teachers (e.g., PETALface in Phase 2.4).
    """

    EMB_DIM: int = 512  # all current teachers produce 512-d embeddings
    IMG_SIZE: int = 112  # default — subclasses override as needed

    def __init__(self, finetune_ckpt_path: Optional[str] = None):
        super().__init__()

    @property
    def img_size(self) -> int:
        return self.IMG_SIZE
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: [B, 3, 112, 112] normalised to [-1, 1].
        Output: [B, 512] L2-normalised embeddings."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CVLFace ViT-Base (AdaFace, WebFace4M)
# ---------------------------------------------------------------------------
_CVLFACE_REPO_ID = "minchul/cvlface_adaface_vit_base_webface4m"
_CVLFACE_CACHE = os.path.expanduser("~/.cvlface_cache/minchul/cvlface_adaface_vit_base_webface4m")


def _download_cvlface(repo_id: str, path: str) -> None:
    """Download all files listed in the HuggingFace repo's ``files.txt``."""
    from huggingface_hub import hf_hub_download

    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, "files.txt")
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, "files.txt", local_dir=path, local_dir_use_symlinks=False)
    with open(files_path, "r") as f:
        files = f.read().split("\n")
    for file in [f for f in files if f] + ["config.json", "wrapper.py", "model.safetensors"]:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, local_dir=path, local_dir_use_symlinks=False)


def _load_cvlface_model(path: str) -> nn.Module:
    """Load the CVLFace model from a local directory using ``AutoModel``."""
    from transformers import AutoModel

    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    try:
        model = AutoModel.from_pretrained(path, trust_remote_code=True)
    finally:
        os.chdir(cwd)
        sys.path.remove(path)
    return model


# ---------------------------------------------------------------------------
# PETALface Swin (ArcFace, WebFace4M)
# ---------------------------------------------------------------------------
_PETALFACE_REPO_ID = "kartiknarayan/PETALface"
_PETALFACE_CACHE = os.path.expanduser("~/.petalface_cache")
_PETALFACE_BACKBONE_URL = (
    "https://raw.githubusercontent.com/Kartik-3004/PETALface/main/backbones"
)

# PETALface uses 120×120 input (NOT 112) — its PatchEmbed asserts this.
PETALFACE_IMG_SIZE = 120


def _download_petalface_backbone(cache: str) -> None:
    """Download the PETALface backbone Python files (swin_models + lora_layers)."""
    bb_dir = os.path.join(cache, "backbones")
    os.makedirs(bb_dir, exist_ok=True)
    init_path = os.path.join(bb_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")
    for fname in ("swin_models.py", "lora_layers.py"):
        full = os.path.join(bb_dir, fname)
        if not os.path.exists(full):
            import urllib.request
            url = f"{_PETALFACE_BACKBONE_URL}/{fname}"
            urllib.request.urlretrieve(url, full)


def _download_petalface_weights(cache: str) -> str:
    """Download pretrained PETALface weights from HuggingFace. Returns path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=_PETALFACE_REPO_ID,
        filename="swin_arcface_webface4m/model.pt",
        local_dir=cache,
    )


def _load_petalface_backbone(cache: str):
    """Instantiate PETALface Swin backbone (no LoRA, 120×120)."""
    bb_dir = os.path.join(cache, "backbones")
    if cache not in sys.path:
        sys.path.insert(0, cache)
    try:
        import importlib
        import types

        # Create a synthetic package so relative imports (from .lora_layers)
        # work inside swin_models.py.
        _PKG = "_petalface_bb"
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [bb_dir]
        pkg.__package__ = _PKG
        sys.modules[_PKG] = pkg

        # Load lora_layers as a sub-module of the package
        lora_spec = importlib.util.spec_from_file_location(
            f"{_PKG}.lora_layers",
            os.path.join(bb_dir, "lora_layers.py"),
        )
        lora_mod = importlib.util.module_from_spec(lora_spec)
        lora_mod.__package__ = _PKG
        sys.modules[f"{_PKG}.lora_layers"] = lora_mod
        lora_spec.loader.exec_module(lora_mod)

        # Load swin_models as a sub-module of the same package
        swin_spec = importlib.util.spec_from_file_location(
            f"{_PKG}.swin_models",
            os.path.join(bb_dir, "swin_models.py"),
        )
        swin_mod = importlib.util.module_from_spec(swin_spec)
        swin_mod.__package__ = _PKG
        sys.modules[f"{_PKG}.swin_models"] = swin_mod
        swin_spec.loader.exec_module(swin_mod)

        SwinTransformer = swin_mod.SwinTransformer
    finally:
        for k in list(sys.modules):
            if k.startswith("_petalface_bb"):
                sys.modules.pop(k, None)
        if cache in sys.path:
            sys.path.remove(cache)
    return SwinTransformer(
        lora_rank=4, lora_scale=1,
        img_size=PETALFACE_IMG_SIZE, patch_size=6, in_chans=3, num_classes=512,
        embed_dim=384, depths=[2, 18, 2], num_heads=[8, 16, 16],
        window_size=5, use_lora=False, reso=PETALFACE_IMG_SIZE,
    )


class PETALFaceTeacher(BaseTeacher):
    """Wraps PETALface Swin (ArcFace + WebFace4M) as a frozen teacher.

    Input: ``[B, 3, 120, 120]`` normalised to ``[-1, 1]``.
    Output: ``[B, 512]`` L2-normalised embeddings.

    Note: PETALface requires **120×120** input — its PatchEmbed asserts
    ``img_size == 120``.  This differs from CVLFace (112×112).
    """

    IMG_SIZE = PETALFACE_IMG_SIZE  # 120

    def __init__(
        self,
        finetune_ckpt_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(finetune_ckpt_path)

        cache = cache_dir or _PETALFACE_CACHE
        _download_petalface_backbone(cache)
        _download_petalface_weights(cache)

        self.model = _load_petalface_backbone(cache)

        # Load pretrained weights
        weights_path = os.path.join(cache, "swin_arcface_webface4m", "model.pt")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict, strict=True)

        # Apply fine-tuned weights if provided
        if finetune_ckpt_path is not None:
            state = torch.load(finetune_ckpt_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state["backbone_state_dict"])

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: [B, 3, 120, 120] normalised to [-1, 1].
        Output: [B, 512] L2-normalised embeddings."""
        emb = self.model(x)
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        return F.normalize(emb, dim=1)


class CVLFaceTeacher(BaseTeacher):
    """Wraps ``minchul/cvlface_adaface_vit_base_webface4m`` as a frozen teacher."""

    def __init__(
        self,
        finetune_ckpt_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(finetune_ckpt_path)

        cache = cache_dir or _CVLFACE_CACHE
        _download_cvlface(_CVLFACE_REPO_ID, cache)
        self.model = _load_cvlface_model(cache)

        if finetune_ckpt_path is not None:
            state = torch.load(finetune_ckpt_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state["backbone_state_dict"])

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: [B, 3, 112, 112] normalised to [-1, 1].
        Output: [B, 512] L2-normalised embeddings."""
        emb = self.model(x)
        # Ensure output is a flat [B, D] tensor (some wrappers return tuples)
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        return F.normalize(emb, dim=1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_teacher(
    teacher_type: str,
    finetune_ckpt_path: Optional[str] = None,
    **kwargs,
) -> BaseTeacher:
    """Factory function for teacher models.

    Args:
        teacher_type: One of ``"cvlface_vit_base"`` or ``"petalface_swin"``.
        finetune_ckpt_path: Optional path to fine-tuned teacher checkpoint.
        **kwargs: Forwarded to the teacher constructor.

    Returns:
        A frozen :class:`BaseTeacher` instance.
    """
    if teacher_type == "cvlface_vit_base":
        return CVLFaceTeacher(finetune_ckpt_path=finetune_ckpt_path, **kwargs)
    elif teacher_type == "petalface_swin":
        return PETALFaceTeacher(finetune_ckpt_path=finetune_ckpt_path, **kwargs)
    raise ValueError(f"Unknown teacher_type: {teacher_type}")
