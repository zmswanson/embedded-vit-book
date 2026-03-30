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

    def __init__(self, finetune_ckpt_path: Optional[str] = None):
        super().__init__()

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
        teacher_type: One of ``"cvlface_vit_base"``.
        finetune_ckpt_path: Optional path to fine-tuned teacher checkpoint.
        **kwargs: Forwarded to the teacher constructor.

    Returns:
        A frozen :class:`BaseTeacher` instance.
    """
    if teacher_type == "cvlface_vit_base":
        return CVLFaceTeacher(finetune_ckpt_path=finetune_ckpt_path, **kwargs)
    # Phase 2.4 will add: elif teacher_type == "petalface_swin": ...
    raise ValueError(f"Unknown teacher_type: {teacher_type}")
