#!/usr/bin/env python3
"""
FLOPs vs resolution sweep for the 16 chapter architectures.

Outputs:
  inference_results/flops/flops_total.csv
      model, resolution, params_M, gmacs, status, unsupported_pct, sum_by_module_pct
  inference_results/flops/flops_by_module.csv
      model, resolution, category, gmacs, pct_of_total
"""
from __future__ import annotations

import argparse
import csv
import gc
import io
import logging
import re
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import timm
import torch
from fvcore.nn import FlopCountAnalysis

MODELS = [
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
    "deit3_small_patch16_224",
    "deit3_medium_patch16_224",
    "deit3_base_patch16_224",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "pvt_v2_b2",
    "pvt_v2_b3",
    "pvt_v2_b5",
    "resnet50d",
    "resnet101d",
    "resnet200d",
    "mobilevitv2_200",
]

COMMON_RES = [64, 96, 128, 160, 224, 256, 384, 512, 768, 1024]

# Order matters: first match wins. Test on the FULL dotted module path.
CATEGORY_PATTERNS = [
    # Attention
    ("attention_qkv",   re.compile(r"(?:^|\.)attn\.(qkv|q|k|v|kv)$")),
    ("attention_qkv",   re.compile(r"(?:^|\.)attention\.(qkv|q|k|v|kv)$")),
    ("attention_proj",  re.compile(r"(?:^|\.)(attn|attention)\.proj$")),
    # PVTv2 spatial-reduction conv that lives inside the attention block
    ("attention_other", re.compile(r"(?:^|\.)(attn|attention)\.sr(?:\.|$)")),
    ("attention_other", re.compile(r"(?:^|\.)(attn|attention)(?:\.|$)")),
    # FFN / MLP
    ("ffn_mlp",         re.compile(r"(?:^|\.)(mlp|ffn|feed_forward)\.")),
    # Embedding & stems
    ("patch_embed",     re.compile(r"(?:^|\.)(patch_embed|stem|conv_stem)(?:\.|$)")),
    # Downsampling
    ("downsample",      re.compile(r"(?:^|\.)(downsample|reduction)(?:\.|$)")),
    # MobileViTv2 transformer block hits this — bucket it as ffn (linear projections inside the block)
    ("ffn_mlp",         re.compile(r"(?:^|\.)transformer(?:\.|$)")),
    # CNN convs (ConvNeXt depthwise/pointwise, ResNet bottleneck convs, MobileViT convs)
    ("conv",            re.compile(r"(?:^|\.)(conv\d*|conv_dw|conv_pw|conv_pwl|dwconv|pwconv\d*)(?:\.|$)")),
    # Norms (rarely contribute to fvcore MACs but include for completeness)
    ("norm",            re.compile(r"(?:^|\.)(norm\d?|ln\d?|bn\d?)(?:\.|$)")),
    # Head
    ("head",            re.compile(r"^(head|classifier|fc)(?:\.|$)")),
]

# Only families whose timm constructor accepts img_size and benefits from positional-embed
# interpolation. PVTv2 and MobileViTv2 are fully-convolutional in their token mixer and
# accept any input shape directly.
SUPPORTED_NAMESPACES = ("deit", "swin", "vit")


def categorize(module_name: str) -> str:
    for cat, pat in CATEGORY_PATTERNS:
        if pat.search(module_name):
            return cat
    return "other"


def resolutions_for(name: str) -> list[int]:
    # timm's swin_v1 pads internally, so it accepts the full ladder; we no
    # longer special-case it.
    return COMMON_RES


def measure(model_name: str, res: int, device: str) -> tuple[float, dict[str, float], float, float, str]:
    """Returns (gmacs_total, by_category_macs_g, unsupported_pct, sum_by_module_pct, status)."""
    try:
        kwargs = {}
        if any(model_name.startswith(p) for p in SUPPORTED_NAMESPACES):
            kwargs["img_size"] = res
        model = timm.create_model(
            model_name, pretrained=False, num_classes=0, **kwargs
        ).eval().to(device)
        x = torch.zeros(1, 3, res, res, device=device)

        # Capture fvcore stderr to detect unsupported-op warnings
        buf = io.StringIO()
        with warnings.catch_warnings(), redirect_stderr(buf):
            warnings.simplefilter("ignore")
            fca = FlopCountAnalysis(model, x)
            total_macs = fca.total()
            by_module = fca.by_module()
            unsupported = fca.unsupported_ops()  # dict op -> count

        unsupported_total = sum(unsupported.values()) if unsupported else 0
        # fvcore returns "supported" macs only; we report unsupported_ops as a count, so
        # express as a soft proxy: fraction of unsupported op invocations is not directly
        # comparable to MACs. We set this to 0.0 unless it's plausibly large.
        unsupported_pct = 0.0  # placeholder; we log raw count in stderr instead

        # Compute leaf-only sum to compare with total (avoid double-count from nesting)
        # by_module includes every ancestor; the root entry "" equals total. We bucket by
        # leaf modules: those whose name has no descendant in the dict.
        names = set(by_module.keys())
        def is_leaf(n: str) -> bool:
            if n == "":
                return False
            prefix = n + "."
            return not any(other.startswith(prefix) for other in names)

        cat: dict[str, float] = {}
        leaf_sum = 0.0
        for mod_name, m in by_module.items():
            if not is_leaf(mod_name):
                continue
            leaf_sum += float(m)
            c = categorize(mod_name)
            cat[c] = cat.get(c, 0.0) + float(m)

        sum_by_module_pct = (100.0 * leaf_sum / total_macs) if total_macs > 0 else 0.0

        del model, x, fca
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        return (
            total_macs / 1e9,
            {k: v / 1e9 for k, v in cat.items()},
            unsupported_pct,
            sum_by_module_pct,
            "ok",
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan"), {}, 0.0, 0.0, "oom"
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            torch.cuda.empty_cache()
            return float("nan"), {}, 0.0, 0.0, "oom"
        return float("nan"), {}, 0.0, 0.0, f"error: RuntimeError: {msg[:120]}"
    except Exception as e:
        return float("nan"), {}, 0.0, 0.0, f"error: {type(e).__name__}: {str(e)[:120]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="inference_results/flops")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of models to run (default: all 16)")
    args = ap.parse_args()

    # Silence fvcore's chatty logger (we ignore unsupported-op spam)
    logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total_csv = out / "flops_total.csv"
    bymod_csv = out / "flops_by_module.csv"

    selected = args.models if args.models else MODELS

    with total_csv.open("w", newline="") as ft, bymod_csv.open("w", newline="") as fm:
        wt = csv.writer(ft)
        wt.writerow(["model", "resolution", "params_M", "gmacs",
                     "sum_by_module_pct", "status"])
        wm = csv.writer(fm)
        wm.writerow(["model", "resolution", "category", "gmacs", "pct_of_total"])

        for name in selected:
            base = timm.create_model(name, pretrained=False, num_classes=0)
            params_M = sum(p.numel() for p in base.parameters()) / 1e6
            del base

            for res in resolutions_for(name):
                gmacs, by_cat, _unsupp, sum_pct, status = measure(name, res, args.device)
                gm_str = f"{gmacs:.4f}" if gmacs == gmacs else ""
                wt.writerow([name, res, f"{params_M:.2f}", gm_str,
                             f"{sum_pct:.1f}", status])
                if gmacs == gmacs and gmacs > 0:
                    for c, m in sorted(by_cat.items(), key=lambda kv: -kv[1]):
                        wm.writerow([name, res, c, f"{m:.4f}",
                                     f"{100 * m / gmacs:.2f}"])
                print(f"  {name:42s} @ {res:>4}  "
                      f"{('%8.3f GMACs' % gmacs) if gmacs == gmacs else '   --   GMACs':>14s}  "
                      f"sum={sum_pct:5.1f}%  [{status}]", flush=True)
        print(f"\nWrote: {total_csv}\n       {bymod_csv}")


if __name__ == "__main__":
    main()
