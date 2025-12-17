import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import timm


# ----------------------------
# Freeze / Thaw helpers
# ----------------------------

_NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.LocalResponseNorm,
)


def _is_norm_module(m: nn.Module) -> bool:
    if isinstance(m, _NORM_TYPES):
        return True
    # timm has some custom norms; catch common names
    cls = m.__class__.__name__.lower()
    return ("layernorm" in cls) or ("batchnorm" in cls) or (cls in {"rmsnorm", "layernorm2d"})


def _is_attention_module(name: str, m: nn.Module) -> bool:
    # Name-based and class-based detection
    n = name.lower()
    cls = m.__class__.__name__.lower()
    if ".attn" in n or n.endswith("attn") or ".attention" in n:
        return True
    if "attention" in cls or cls == "attn":
        return True
    # Some timm blocks name it "self_attn"
    if "attn" in cls or "selfattn" in cls or "self_attn" in n:
        return True
    return False


def _freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def _thaw_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def _natural_key(s: str) -> List[object]:
    # Natural sort key for module names with digits
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


@dataclass(frozen=True)
class ThawSpec:
    key: str
    value: Optional[str] = None


def _parse_thawed_modules_arg(arg: str) -> List[ThawSpec]:
    """
    Accepts tokens like:
      head
      all_norm
      last_blocks=2
      last_blocks:2
      last_blocks_2
      regex=^layers\.3\.
    """
    if not arg:
        return []
    out: List[ThawSpec] = []
    for raw in [t.strip() for t in arg.split(",") if t.strip()]:
        if "=" in raw:
            k, v = raw.split("=", 1)
            out.append(ThawSpec(k.strip().lower(), v.strip()))
            continue
        if ":" in raw:
            k, v = raw.split(":", 1)
            out.append(ThawSpec(k.strip().lower(), v.strip()))
            continue
        m = re.match(r"^(.*)_(\d+)$", raw.strip().lower())
        if m:
            out.append(ThawSpec(m.group(1), m.group(2)))
        else:
            out.append(ThawSpec(raw.strip().lower(), None))
    return out


# ----------------------------
# Model-family block discovery
# ----------------------------

def _infer_family(model_name: str) -> str:
    n = model_name.lower()
    if "convnext" in n:
        return "convnext"
    if "swin" in n:
        return "swin"
    if "deit" in n or re.search(r"\bvit\b", n) or "visiontransformer" in n:
        return "vit"
    if "pvt" in n:
        return "pvt"
    if "mobilevit" in n:
        return "mobilevit"
    if "resnet" in n:
        return "resnet"
    return "generic"


def _collect_block_modules(model: nn.Module, family: str) -> List[str]:
    """
    Returns module names for "blocks" in a stable ordering, best-effort per family.
    """
    names: List[Tuple[Tuple[int, ...], str]] = []

    # ConvNeXt / PVTv2: stages.{s}.blocks.{b}
    if family in {"convnext", "pvt"}:
        rx = re.compile(r"^stages\.(\d+)\.blocks\.(\d+)$")
        for name, m in model.named_modules():
            mm = rx.match(name)
            if mm:
                s, b = int(mm.group(1)), int(mm.group(2))
                names.append(((s, b), name))

    # Swin: layers.{l}.blocks.{b}
    elif family == "swin":
        rx = re.compile(r"^layers\.(\d+)\.blocks\.(\d+)$")
        for name, m in model.named_modules():
            mm = rx.match(name)
            if mm:
                l, b = int(mm.group(1)), int(mm.group(2))
                names.append(((l, b), name))

    # DeiT / ViT: blocks.{i}
    elif family == "vit":
        rx = re.compile(r"^blocks\.(\d+)$")
        for name, m in model.named_modules():
            mm = rx.match(name)
            if mm:
                i = int(mm.group(1))
                names.append(((i,), name))

    # ResNet: layer{1-4}.{i}
    elif family == "resnet":
        rx = re.compile(r"^layer([1-4])\.(\d+)$")
        for name, m in model.named_modules():
            mm = rx.match(name)
            if mm:
                stage = int(mm.group(1))
                blk = int(mm.group(2))
                names.append(((stage, blk), name))

    # MobileViTv2: transformer blocks tend to look like:
    # stages.{s}.{b}.transformer.{t}
    elif family == "mobilevit":
        rx = re.compile(r"^stages\.(\d+)\.(\d+)\.transformer\.(\d+)$")
        for name, m in model.named_modules():
            mm = rx.match(name)
            if mm:
                s, b, t = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
                names.append(((s, b, t), name))

        # Fallback: if no transformer blocks matched, use stages.{s}.{b}
        if not names:
            rx2 = re.compile(r"^stages\.(\d+)\.(\d+)$")
            for name, m in model.named_modules():
                mm = rx2.match(name)
                if mm:
                    s, b = int(mm.group(1)), int(mm.group(2))
                    names.append(((s, b), name))

    else:
        # Generic: treat any module named "blocks.N" or "layers.X.blocks.Y" as blocks if present
        for name, _m in model.named_modules():
            if re.match(r"^blocks\.\d+$", name) or re.match(r"^layers\.\d+\.blocks\.\d+$", name):
                names.append((tuple(int(x) for x in re.findall(r"\d+", name)), name))

    names.sort(key=lambda x: x[0])
    return [n for _, n in names]


def _collect_stage_modules(model: nn.Module, family: str) -> List[str]:
    """
    Returns module names for "stages/layers" containers in stable ordering.
    """
    names: List[Tuple[int, str]] = []

    if family in {"convnext", "pvt"}:
        rx = re.compile(r"^stages\.(\d+)$")
        for name, _m in model.named_modules():
            mm = rx.match(name)
            if mm:
                names.append((int(mm.group(1)), name))

    elif family == "swin":
        rx = re.compile(r"^layers\.(\d+)$")
        for name, _m in model.named_modules():
            mm = rx.match(name)
            if mm:
                names.append((int(mm.group(1)), name))

    elif family == "resnet":
        # Stage containers are layer1..layer4
        for k in [1, 2, 3, 4]:
            if hasattr(model, f"layer{k}"):
                names.append((k, f"layer{k}"))

    elif family == "mobilevit":
        # Stages are stages.0 .. stages.N
        rx = re.compile(r"^stages\.(\d+)$")
        for name, _m in model.named_modules():
            mm = rx.match(name)
            if mm:
                names.append((int(mm.group(1)), name))

    elif family == "vit":
        # No "stage" container; consider the full blocks container
        if hasattr(model, "blocks"):
            names.append((0, "blocks"))

    names.sort(key=lambda x: x[0])
    return [n for _, n in names]


def _collect_attention_modules(model: nn.Module) -> List[str]:
    attn: List[str] = []
    for name, m in model.named_modules():
        if _is_attention_module(name, m):
            # only thaw something that actually has parameters
            has_params = any(True for _ in m.parameters(recurse=False)) or any(True for _ in m.parameters(recurse=True))
            if has_params:
                attn.append(name)
    # stable order
    attn = sorted(set(attn), key=_natural_key)
    return attn


def _collect_norm_modules(model: nn.Module) -> List[str]:
    norms: List[str] = []
    for name, m in model.named_modules():
        if _is_norm_module(m):
            # only thaw norms that own params (e.g., some norms are paramless)
            if any(p.numel() > 0 for p in m.parameters(recurse=False)):
                norms.append(name)
    return sorted(set(norms), key=_natural_key)


def _collect_head_modules(model: nn.Module) -> List[str]:
    """
    Best-effort across timm:
      - ViT/DeiT/Swin/ConvNeXt/PVT/MobileViT often have 'head'
      - ResNet uses 'fc'
      - some have 'classifier'
    """
    candidates = []
    for nm in ["head", "fc", "classifier", "classif", "pre_logits", "head_drop", "fc_norm", "norm", "global_pool"]:
        if hasattr(model, nm):
            candidates.append(nm)

    # Also match by module name prefixes commonly used by timm
    prefixes = ("head", "fc", "classifier")
    for name, _m in model.named_modules():
        if name.startswith(prefixes):
            candidates.append(name)

    # Keep unique, stable
    uniq = sorted(set(candidates), key=_natural_key)
    return uniq


# ----------------------------
# Lightning module
# ----------------------------

class TimmIDModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        img_size: int = 96,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        label_smoothing: float = 0.0,
        pretrained: bool = True,
        thawed_modules: str = "head",
        verbose_thaw: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.pretrained = pretrained
        self.thawed_modules_arg = thawed_modules
        self.verbose_thaw = verbose_thaw

        # timm: many models accept img_size; for ones that don't, timm ignores it
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
        )

        self._thaw_applied = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # Apply freeze/thaw once trainer is set up
    def on_fit_start(self) -> None:
        if not self._thaw_applied:
            self.apply_thaw_plan(self.thawed_modules_arg)
            self._thaw_applied = True

    def apply_thaw_plan(self, thawed_modules: str) -> None:
        specs = _parse_thawed_modules_arg(thawed_modules)
        family = _infer_family(self.model_name)

        # Default: freeze everything, then selectively thaw
        _freeze_all(self.model)

        selected_module_names: List[str] = []

        # Always include head unless user explicitly uses "no_head"
        # (If you truly want "no head", pass thawed_modules including "no_head".)
        include_head = True
        for s in specs:
            if s.key in {"no_head", "nohead"}:
                include_head = False

        if include_head:
            selected_module_names.extend(_collect_head_modules(self.model))

        # Presets
        norm_names = _collect_norm_modules(self.model)
        attn_names = _collect_attention_modules(self.model)
        block_names = _collect_block_modules(self.model, family)
        stage_names = _collect_stage_modules(self.model, family)

        for spec in specs:
            k = spec.key
            v = spec.value

            if k in {"head", "default"}:
                # already included by default
                continue

            if k == "all":
                # thaw everything and stop
                _thaw_module(self.model)
                selected_module_names = ["<ALL>"]
                break

            if k == "all_norm":
                selected_module_names.extend(norm_names)
                continue

            if k == "all_attn":
                selected_module_names.extend(attn_names)
                continue

            if k == "last_attn":
                if not v or not v.isdigit():
                    raise ValueError("last_attn requires N (e.g., last_attn=4)")
                n = int(v)
                if n > 0:
                    selected_module_names.extend(attn_names[-n:])
                continue

            if k in {"last_blocks", "last_block"}:
                if not v or not v.isdigit():
                    raise ValueError("last_blocks requires N (e.g., last_blocks=2)")
                n = int(v)
                if n > 0:
                    selected_module_names.extend(block_names[-n:])
                continue

            if k in {"last_stages", "last_stage"}:
                if not v or not v.isdigit():
                    raise ValueError("last_stages requires N (e.g., last_stages=1)")
                n = int(v)
                if n > 0:
                    selected_module_names.extend(stage_names[-n:])
                continue

            if k == "regex":
                if not v:
                    raise ValueError("regex requires a pattern (e.g., regex=^layers\\.3\\.)")
                rx = re.compile(v)
                for name, _m in self.model.named_modules():
                    if rx.search(name):
                        selected_module_names.append(name)
                continue

            if k in {"no_head", "nohead"}:
                continue

            raise ValueError(
                f"Unknown thaw option '{k}'. Supported: "
                "head, all, all_norm, all_attn, last_attn=N, last_blocks=N, last_stages=N, regex=PATTERN, no_head"
            )

        # Resolve to actual modules and thaw
        if "<ALL>" not in selected_module_names:
            selected_module_names = sorted(set(selected_module_names), key=_natural_key)

            name_to_module: Dict[str, nn.Module] = dict(self.model.named_modules())
            for name in selected_module_names:
                m = name_to_module.get(name, None)
                if m is None:
                    # allow "fc"/"head"/etc as attributes, not necessarily in named_modules()
                    if hasattr(self.model, name):
                        m = getattr(self.model, name)
                    else:
                        continue
                _thaw_module(m)

        # Optional: print a concise summary
        if self.verbose_thaw:
            trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
            self.print(f"[thaw] family={family} thawed_modules='{thawed_modules}'")
            self.print(f"[thaw] trainable tensors: {len(trainable)}")
            if len(trainable) <= 80:
                for n in trainable:
                    self.print(f"  + {n}")
            else:
                for n in trainable[:60]:
                    self.print(f"  + {n}")
                self.print(f"  ... (+{len(trainable)-60} more)")

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
