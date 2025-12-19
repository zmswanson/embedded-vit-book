# lightning_modules/lora.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from ._infer_family import _infer_family


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear and adds trainable low-rank adapters:
        y = W0 x + (alpha/r) * B(A(dropout(x)))
    where W0 is the frozen base weight, A and B are low-rank matrices,
    r is the rank, and alpha is a scaling factor.

    :param base: The base nn.Linear to wrap (will be frozen).
    :param r: The rank of the LoRA adapters, i.e., the inner dimension of A and B.
    :param alpha: The LoRA scaling factor, typically in range [1, 32].
    :param dropout: Dropout probability applied to input before LoRA adapters.
    """
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # LoRA parameters implementated as two nn.Linear layers--these are
        # effectively low-rank matrices but we can leverage nn.Linear for free.
        self.lora_A = nn.Linear(base.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, base.out_features, bias=False)

        # Common init: A small, B zero so you start exactly at the base model
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        y_lora = self.lora_B(self.lora_A(self.drop(x))) * self.scaling
        return y0 + y_lora


@dataclass(frozen=True)
class LoRAConfig:
    """
    Configuration for LoRA injection, including regex for identifying target modules.
    - If target_regex is not provided (None) OR family="auto", we infer a model
      family (from model_name) and select a family-specific regex.
    - If target_regex is explicitly set, we use it verbatim.

    :param enabled: Whether to enable LoRA injection.
    :param r: The rank of the LoRA adapters.
    :param alpha: The LoRA scaling factor.
    :param dropout: Dropout probability applied to input before LoRA adapters.
    :param model_name: (Optional) timm model name to help auto-infer family.
    :param family: Model family for default regex selection if target_regex is None.
        Choices: "auto", "vit", "swin", "pvt", "mobilevit", "levit",
                 "efficientformer", "cnn"
    :param target_regex: (Optional) regex pattern to match module names for LoRA
        injection. If None, a family-specific default will be used.
    :param apply_qkv: Whether to apply LoRA to QKV layers.
    :param apply_proj: Whether to apply LoRA to projection layers.
    """
    enabled: bool = False
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    model_name: Optional[str] = None
    family: str = "auto"
    target_regex: Optional[str] = None
    # target_regex: str = r"(?:^|\.)(qkv|qkv_proj|qk|kv|q|k|v|proj|out_proj)(?:$|\.)"
    apply_qkv: bool = True
    apply_proj: bool = True


FAMILY_PARTS: dict[str, dict[str, str]] = {
    "vit": {
        "qkv":  r"(?:^|\.)(attn)\.(qkv)(?:$|\.)",
        "proj": r"(?:^|\.)(attn)\.(proj)(?:$|\.)",
    },
    "swin": {
        "qkv":  r"(?:^|\.)(attn)\.(qkv)(?:$|\.)",
        "proj": r"(?:^|\.)(attn)\.(proj)(?:$|\.)",
    },
    "pvt": {
        # qkv-ish for PVT is q and kv
        "qkv":  r"(?:^|\.)(attn)\.(q|kv)(?:$|\.)",
        "proj": r"(?:^|\.)(attn)\.(proj)(?:$|\.)",
    },
    "mobilevit": {
        # qkv-ish for MobileViT is qkv_proj; proj is out_proj
        "qkv":  r"(?:^|\.)(attn)\.(qkv_proj)(?:$|\.)",
        "proj": r"(?:^|\.)(attn)\.(out_proj)(?:$|\.)",
    },
    "levit": {
        "qkv":  r"(?:^|\.)(attn)\.(qkv)(?:$|\.)",
        "proj": r"(?:^|\.)(attn)\.(proj)(?:$|\.)",
    },
    "efficientformer": {
        # efficientformerv2 tends to have q/k/v separate (timm); treat those as qkv-ish
        "qkv":  r"(?:^|\.)(attn)\.(q|k|v)(?:$|\.)",
        "proj": r"(?:^|\.)(attn)\.(proj)(?:$|\.)",
    },
    "cnn": {
        # no-op by default
        "qkv":  r"a^",
        "proj": r"a^",
    },
}


def resolve_target_regex(cfg: LoRAConfig) -> str:
    """
    Decide which regex to use:
    - If cfg.target_regex is explicitly set: use it
    - Else if cfg.family != "auto": use that family
    - Else infer from cfg.model_name
    """
    if cfg.target_regex:
        return cfg.target_regex

    fam = cfg.family.lower()
    if fam == "auto":
        fam = _infer_family(cfg.model_name)

        parts = FAMILY_PARTS.get(fam, FAMILY_PARTS["vit"])

    enabled_patterns: List[str] = []
    if cfg.apply_qkv:
        enabled_patterns.append(parts["qkv"])
    if cfg.apply_proj:
        enabled_patterns.append(parts["proj"])

    # If both are disabled, make it a no-op regex
    if not enabled_patterns:
        return r"a^"

    # Combine as OR
    return "(" + ")|(".join(enabled_patterns) + ")"


def _get_parent(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    """
    Helper function to get the parent module and attribute name for a given module
    name, e.g., "encoder.layer1.attn.qkv" -> (encoder.layer1.attn, "qkv"). Used
    for replacing modules in-place.

    :param root: The root module to start from.
    :param module_name: The full module name (dot-separated).
    :return: A tuple of (parent_module, attribute_name).
    """

    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> List[str]:
    """
    Replace matching nn.Linear modules with LoRALinear wrappers. For example,
    with the default regex, this will replace attention qkv and projection layers
    in ViT-like models.
    """
    if not cfg.enabled:
        return []

    target_regex = resolve_target_regex(cfg)
    rx = re.compile(target_regex)
    replaced: List[str] = []

    # We need a stable list first because we'll be mutating modules.
    candidates = [
        (name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)
    ]
    

    for name, lin in candidates:
        if not rx.search(name):
            continue

        parent, attr = _get_parent(model, name)
        wrapped = LoRALinear(lin, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)
        setattr(parent, attr, wrapped)
        replaced.append(name)

    return replaced


def mark_only_lora_trainable(model: nn.Module, train_bias: bool = False) -> None:
    """
    Freeze everything, then unfreeze LoRA params (and optional bias).
    """
    for p in model.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True
        elif train_bias and n.endswith(".bias"):
            p.requires_grad = True


if __name__ == "__main__":
    # Simple test
    from timm import create_model

    model = create_model("vit_base_patch16_224", pretrained=False)
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"Original model params: {initial_params:,}")

    cfg = LoRAConfig(
        enabled=True, r=4, alpha=8.0, dropout=0.1, 
        model_name="vit_base_patch16_224", 
        apply_qkv=False, apply_proj=True,
    )
    replaced = inject_lora(model, cfg)
    mark_only_lora_trainable(model, train_bias=True)

    print(f"Replaced {len(replaced)} modules with LoRA:")
    for name in replaced:
        print(f" - {name}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable params: {trainable_params:,}")
    print(f"Added {total_params - initial_params:,} LoRA params.")