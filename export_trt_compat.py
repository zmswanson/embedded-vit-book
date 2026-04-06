"""Export TRT-compatible ONNX models for architectures that fail native TRT conversion.

Handles three model families with specific rewrites:
  - ConvNeXt: Recreates with conv_mlp=True (all-NCHW, no Transpose+LayerNorm patterns)
  - PVTv2: Disables fused attention, replaces LayerNorm with TRT-friendly decomposition
  - MobileViTv2: Replaces GroupNorm1 with TRT-friendly manual normalization

Each model's weights are loaded from a Lightning checkpoint (TimmIDModule or KDTimmIDModule),
remapped into the TRT-compatible architecture, validated against the original, and exported.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning_modules.lora import LoRALinear, _get_parent
from lightning_modules.timm_id_module import TimmIDModule
from export_onnx import merge_lora_weights, _is_kd_checkpoint


# ======================================================================
# Generic helpers
# ======================================================================

def _load_backbone(ckpt_path: str, model_name: str | None = None) -> nn.Module:
    """Load a backbone from a Lightning checkpoint."""
    is_kd = _is_kd_checkpoint(ckpt_path)
    if is_kd:
        from lightning_modules.kd_module import KDTimmIDModule
        module = KDTimmIDModule.load_from_checkpoint(ckpt_path)
        eff_name = module.hparams.get("student_model_name", module.student_model_name)
        print(f"Loaded KD checkpoint – student: {eff_name}")
    else:
        if model_name is None:
            raise ValueError("--model_name required for non-KD checkpoints")
        module = TimmIDModule.load_from_checkpoint(ckpt_path, model_name=model_name)
        eff_name = model_name
        print(f"Loaded checkpoint – model: {eff_name}")

    module.eval().cpu()
    backbone = module.backbone
    if any(isinstance(m, LoRALinear) for m in backbone.modules()):
        merge_lora_weights(backbone)
        print(f"Merged LoRA weights for {eff_name}")
    backbone.eval()
    return backbone


def _validate_outputs(
    orig_model: nn.Module,
    new_model: nn.Module,
    dummy: torch.Tensor,
    label: str = "",
) -> float:
    """Compare two models' outputs — returns cosine similarity."""
    with torch.no_grad():
        out_orig = orig_model(dummy).flatten().numpy()
        out_new = new_model(dummy).flatten().numpy()
    cos = float(
        np.dot(out_orig, out_new)
        / (np.linalg.norm(out_orig) * np.linalg.norm(out_new) + 1e-12)
    )
    status = "PASS" if cos > 0.999 else "FAIL"
    print(f"  [{status}] {label} cosine similarity: {cos:.6f}")
    assert cos > 0.999, f"Validation failed: cosine {cos:.6f}"
    return cos


def _validate_onnx(
    pytorch_model: nn.Module,
    onnx_path: str,
    dummy: torch.Tensor,
) -> float:
    """Compare PyTorch vs ONNX Runtime outputs."""
    import onnxruntime as ort

    with torch.no_grad():
        pt_out = pytorch_model(dummy).flatten().numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy.numpy()})[0].flatten()
    cos = float(
        np.dot(pt_out, ort_out)
        / (np.linalg.norm(pt_out) * np.linalg.norm(ort_out) + 1e-12)
    )
    status = "PASS" if cos > 0.999 else "FAIL"
    print(f"  [{status}] ONNX validation cosine: {cos:.6f}")
    assert cos > 0.999, f"ONNX validation failed: cosine {cos:.6f}"
    return cos


def _export_onnx(model: nn.Module, dummy: torch.Tensor, output_path: str,
                 use_dynamo: bool = False) -> None:
    """Export model to ONNX with opset 17 (native LayerNormalization), legacy TorchScript."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    sz = Path(output_path).stat().st_size / (1024 * 1024)
    data_path = Path(str(output_path) + ".data")
    if data_path.exists():
        sz += data_path.stat().st_size / (1024 * 1024)
    print(f"  Exported: {output_path} ({sz:.1f} MB)")


def _simplify_onnx(onnx_path: str) -> None:
    """Run onnxsim on an ONNX model in-place."""
    try:
        import onnxsim
        model = onnx.load(onnx_path)
        model_sim, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(model_sim, onnx_path)
            print(f"  onnxsim: simplified successfully")
        else:
            print(f"  onnxsim: simplification returned not-ok, keeping original")
    except Exception as e:
        print(f"  onnxsim: {e}")


def _fuse_layernorm_onnx(onnx_path: str) -> None:
    """Fuse decomposed LayerNorm patterns back into single LayerNormalization ops.

    Finds: ReduceMean → Sub → Pow → ReduceMean → Add(eps) → Sqrt → Div → Mul(w) → Add(b)
    Replaces with: LayerNormalization(axis=-1, epsilon, weight, bias)
    Then upgrades opset to 17 (where LayerNormalization is a standard op).
    """
    import onnx_graphsurgeon as gs

    graph = gs.import_onnx(onnx.load(onnx_path))
    fused_count = 0
    nodes_to_remove = []

    for node in graph.nodes:
        # Pattern: starts with ReduceMean
        if node.op != "ReduceMean":
            continue

        try:
            # Follow the exact decomposed LayerNorm pattern
            reduce_mean_1 = node  # ReduceMean (compute mean)

            # Check that ReduceMean output feeds into Sub
            sub_users = [u for u in reduce_mean_1.outputs[0].outputs if u.op == "Sub"]
            if not sub_users:
                continue
            sub_node = sub_users[0]

            # Sub output feeds into: (1) Pow for variance, and (2) Div for final
            sub_out = sub_node.outputs[0]
            pow_users = [u for u in sub_out.outputs if u.op == "Pow"]
            div_users = [u for u in sub_out.outputs if u.op == "Div"]
            if not pow_users or not div_users:
                continue
            pow_node = pow_users[0]
            div_node = div_users[0]

            # Pow(2) → ReduceMean → Add(eps) → Sqrt → Div
            reduce_mean_2_list = [u for u in pow_node.outputs[0].outputs if u.op == "ReduceMean"]
            if not reduce_mean_2_list:
                continue
            reduce_mean_2 = reduce_mean_2_list[0]

            add_eps_list = [u for u in reduce_mean_2.outputs[0].outputs if u.op == "Add"]
            if not add_eps_list:
                continue
            add_eps = add_eps_list[0]

            sqrt_list = [u for u in add_eps.outputs[0].outputs if u.op == "Sqrt"]
            if not sqrt_list:
                continue
            sqrt_node = sqrt_list[0]

            # Verify Sqrt feeds into the same Div node
            div_from_sqrt = [u for u in sqrt_node.outputs[0].outputs if u.op == "Div"]
            if not div_from_sqrt or div_from_sqrt[0] != div_node:
                continue

            # Div → Mul(weight) → Add(bias)
            mul_list = [u for u in div_node.outputs[0].outputs if u.op == "Mul"]
            if not mul_list:
                continue
            mul_node = mul_list[0]

            add_bias_list = [u for u in mul_node.outputs[0].outputs if u.op == "Add"]
            if not add_bias_list:
                continue
            add_bias = add_bias_list[0]

            # Extract weight and bias tensors
            # Mul has two inputs: one from Div, one is weight constant
            weight_input = None
            for inp in mul_node.inputs:
                if inp != div_node.outputs[0]:
                    weight_input = inp
                    break

            bias_input = None
            for inp in add_bias.inputs:
                if inp != mul_node.outputs[0]:
                    bias_input = inp
                    break

            if weight_input is None or bias_input is None:
                continue

            # Extract epsilon from add_eps
            eps_input = None
            for inp in add_eps.inputs:
                if inp != reduce_mean_2.outputs[0]:
                    eps_input = inp
                    break
            if eps_input is None:
                continue

            # Get epsilon value
            if hasattr(eps_input, "values"):
                eps_val = float(eps_input.values.flatten()[0])
            else:
                continue

            # Get axis from ReduceMean
            axes = reduce_mean_1.attrs.get("axes", [-1])
            if isinstance(axes, list):
                axis = axes[0]
            else:
                axis = axes

            # Create LayerNormalization node
            ln_node = gs.Node(
                op="LayerNormalization",
                name=f"LayerNorm_fused_{fused_count}",
                attrs={"axis": axis, "epsilon": eps_val, "stash_type": 1},
                inputs=[reduce_mean_1.inputs[0], weight_input, bias_input],
                outputs=[add_bias.outputs[0]],
            )
            graph.nodes.append(ln_node)

            # Mark old nodes for removal
            nodes_to_remove.extend([
                reduce_mean_1, sub_node, pow_node, reduce_mean_2,
                add_eps, sqrt_node, div_node, mul_node, add_bias,
            ])
            fused_count += 1

        except (IndexError, AttributeError, KeyError):
            continue

    # Remove old nodes
    for n in nodes_to_remove:
        if n in graph.nodes:
            graph.nodes.remove(n)

    if fused_count > 0:
        graph.cleanup().toposort()
        model = gs.export_onnx(graph)
        # Upgrade opset to 17 for native LayerNormalization support
        model.opset_import[0].version = 17
        onnx.save(model, onnx_path)
        print(f"  LayerNorm fusion: fused {fused_count} decomposed patterns → LayerNormalization (opset 17)")
    else:
        print(f"  LayerNorm fusion: no patterns found to fuse")


# ======================================================================
# TRT-native NCHW LayerNorm via 3D reshape + native nn.LayerNorm
# ======================================================================

class TRTLayerNorm2d(nn.Module):
    """Channel-first LayerNorm that flattens spatial dims to 3D and uses
    standard nn.LayerNorm (which TRT handles natively on 3D tensors).

    Converts (B,C,H,W) → (B,HW,C) → LayerNorm(C) → (B,C,H,W).
    The key insight: TRT has native kernels for LayerNormalization on 3D
    (B,N,C) tensors (like Swin/ViT), but NOT for channel-norm on 4D NCHW.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial and move channels to last dim → 3D like Swin tokens
        x = x.flatten(2).transpose(1, 2)          # (B, H*W, C)
        x = self.norm(x)                            # native LayerNorm on 3D
        x = x.transpose(1, 2).reshape(B, C, H, W)  # back to NCHW
        return x

    @classmethod
    def from_layer_norm(cls, ln: nn.Module) -> "TRTLayerNorm2d":
        """Create from nn.LayerNorm or LayerNorm2d, copying weights."""
        C = ln.normalized_shape[0] if hasattr(ln, "normalized_shape") else ln.weight.shape[0]
        eps = ln.eps if hasattr(ln, "eps") else 1e-6
        new = cls(C, eps=eps)
        if hasattr(ln, "weight") and ln.weight is not None:
            new.norm.weight.data.copy_(ln.weight.data)
        if hasattr(ln, "bias") and ln.bias is not None:
            new.norm.bias.data.copy_(ln.bias.data)
        return new


def _replace_layernorm2d(module: nn.Module) -> int:
    """Recursively replace LayerNorm2d with TRTLayerNorm2d."""
    from timm.layers import LayerNorm2d
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LayerNorm2d):
            setattr(module, name, TRTLayerNorm2d.from_layer_norm(child))
            count += 1
        else:
            count += _replace_layernorm2d(child)
    return count


# ======================================================================
# ConvNeXt: conv_mlp=True rewrite + TRT-native norms
# ======================================================================

def _make_convnext_trt(
    model_name: str,
    original_backbone: nn.Module,
    dummy: torch.Tensor,
) -> nn.Module:
    """Create a conv_mlp=True ConvNeXt with TRT-native norms and transfer weights."""
    import timm

    print(f"  Creating conv_mlp=True variant of {model_name}")
    new_model = timm.create_model(model_name, pretrained=False, num_classes=0, conv_mlp=True)
    new_model.eval().cpu()

    # Remap weights: only MLP fc1/fc2 weights need reshaping (2D -> 4D)
    orig_sd = original_backbone.state_dict()
    new_sd = new_model.state_dict()

    remapped = {}
    for key in new_sd:
        if key not in orig_sd:
            raise KeyError(f"Key {key} not found in original state_dict")
        orig_w = orig_sd[key]
        tgt_shape = new_sd[key].shape
        if orig_w.shape == tgt_shape:
            remapped[key] = orig_w
        elif orig_w.ndim == 2 and tgt_shape[-2:] == (1, 1):
            # Linear weight (out, in) -> Conv2d weight (out, in, 1, 1)
            remapped[key] = orig_w.unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(
                f"Cannot remap {key}: {orig_w.shape} -> {tgt_shape}"
            )

    new_model.load_state_dict(remapped, strict=True)
    new_model.eval()

    # Replace LayerNorm2d with TRT-native NCHW normalization (no Transpose)
    n_ln2d = _replace_layernorm2d(new_model)
    print(f"    Replaced {n_ln2d} LayerNorm2d → TRTLayerNorm2d (no Transpose)")

    # Also replace any remaining nn.LayerNorm in the head/norm_pre
    # (NormMlpClassifierHead may contain LayerNorm2d too)
    _validate_outputs(original_backbone, new_model, dummy, "ConvNeXt conv_mlp remap")
    return new_model


# ======================================================================
# PVTv2: Disable fused attention + replace LayerNorm with manual decomposition
# ======================================================================

class TRTLayerNorm(nn.Module):
    """LayerNorm decomposed into primitive ops for TRT compatibility.

    Mathematically equivalent to nn.LayerNorm but exports as
    ReduceMean -> Sub -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Add
    instead of a single LayerNormalization op.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute over the last len(normalized_shape) dims
        dims = list(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        diff = x - mean
        var = (diff * diff).mean(dim=dims, keepdim=True)
        x_norm = diff / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias
        return x_norm

    @classmethod
    def from_layer_norm(cls, ln: nn.LayerNorm) -> "TRTLayerNorm":
        """Create from an existing nn.LayerNorm, copying weights."""
        new = cls(ln.normalized_shape, eps=ln.eps, elementwise_affine=ln.elementwise_affine)
        if ln.elementwise_affine:
            new.weight.data.copy_(ln.weight.data)
            new.bias.data.copy_(ln.bias.data)
        return new


def _replace_layernorms(module: nn.Module) -> int:
    """Recursively replace all nn.LayerNorm with TRTLayerNorm."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, TRTLayerNorm.from_layer_norm(child))
            count += 1
        else:
            count += _replace_layernorms(child)
    return count


def _disable_fused_attn(module: nn.Module) -> int:
    """Set fused_attn=False on all attention modules."""
    count = 0
    for m in module.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
            count += 1
    return count


def _make_pvt_v2_trt(
    model_name: str,
    original_backbone: nn.Module,
    dummy: torch.Tensor,
) -> nn.Module:
    """Make PVTv2 TRT-compatible: disable fused attn.

    With opset 17, nn.LayerNorm exports as native LayerNormalization op,
    so no need to replace LayerNorm — just disable fused SDPA.
    """
    import copy

    print(f"  Patching PVTv2: {model_name}")
    new_model = copy.deepcopy(original_backbone)
    new_model.eval().cpu()

    n_attn = _disable_fused_attn(new_model)
    print(f"    Disabled fused attention in {n_attn} modules")

    _validate_outputs(original_backbone, new_model, dummy, "PVTv2 TRT patch")
    return new_model


# ======================================================================
# MobileViTv2: Replace GroupNorm1 with manual normalization
# ======================================================================

class TRTGroupNorm1(nn.Module):
    """GroupNorm(1, C) decomposed into primitive ops for TRT compatibility.

    Computes mean/var over all non-batch dimensions (equivalent to GroupNorm
    with 1 group), then applies per-channel affine transform.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, ...) — could be 3D (B,C,N) or 4D (B,C,H,W)
        # GroupNorm(1) normalizes over ALL non-batch dims (C + spatial)
        dims = list(range(1, x.ndim))  # [1, 2, ...] — all except batch
        mean = x.mean(dim=dims, keepdim=True)
        diff = x - mean
        var = (diff * diff).mean(dim=dims, keepdim=True)
        x_norm = diff / torch.sqrt(var + self.eps)

        # Per-channel affine: reshape weight/bias to broadcast
        shape = [1, self.num_channels] + [1] * (x.ndim - 2)
        x_norm = x_norm * self.weight.view(shape) + self.bias.view(shape)
        return x_norm

    @classmethod
    def from_group_norm(cls, gn: nn.GroupNorm) -> "TRTGroupNorm1":
        """Create from an existing GroupNorm(1, C), copying weights."""
        assert gn.num_groups == 1, f"Expected 1 group, got {gn.num_groups}"
        new = cls(gn.num_channels, eps=gn.eps)
        if gn.affine:
            new.weight.data.copy_(gn.weight.data)
            new.bias.data.copy_(gn.bias.data)
        return new


def _replace_groupnorm1(module: nn.Module) -> int:
    """Recursively replace GroupNorm(1,...) with TRTGroupNorm1."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm) and child.num_groups == 1:
            setattr(module, name, TRTGroupNorm1.from_group_norm(child))
            count += 1
        else:
            count += _replace_groupnorm1(child)
    return count


def _make_mobilevitv2_trt(
    model_name: str,
    original_backbone: nn.Module,
    dummy: torch.Tensor,
) -> nn.Module:
    """Make MobileViTv2 TRT-compatible: replace GroupNorm1."""
    import copy

    print(f"  Patching MobileViTv2: {model_name}")
    new_model = copy.deepcopy(original_backbone)
    new_model.eval().cpu()

    n_gn = _replace_groupnorm1(new_model)
    print(f"    Replaced {n_gn} GroupNorm1 → TRTGroupNorm1")

    _validate_outputs(original_backbone, new_model, dummy, "MobileViTv2 TRT patch")
    return new_model


# ======================================================================
# Main pipeline
# ======================================================================

MODELS = {
    "convnext_tiny": {
        "timm_name": "convnext_tiny.fb_in1k",
        "ckpt": "/mnt/data/wandb/checkpoints/model_ablation_study/hzvjhovx/epoch=34-rank1tinyface/eval/rank@1=0.568.ckpt",
        "family": "convnext",
    },
    "convnext_small": {
        "timm_name": "convnext_small.fb_in1k",
        "ckpt": "/mnt/data/wandb/checkpoints/model_ablation_study/ccc9z3g5/epoch=34-rank1tinyface/eval/rank@1=0.556.ckpt",
        "family": "convnext",
    },
    "convnext_base": {
        "timm_name": "convnext_base.fb_in1k",
        "ckpt": "/mnt/data/wandb/checkpoints/model_ablation_study/m4xjjh07/epoch=17-rank1tinyface/eval/rank@1=0.570.ckpt",
        "family": "convnext",
    },
    "pvt_v2_b2": {
        "timm_name": "pvt_v2_b2.in1k",
        "ckpt": "/mnt/data/wandb/checkpoints/model_ablation_study/uvu73fxo/epoch=24-rank1tinyface/eval/rank@1=0.548.ckpt",
        "family": "pvt_v2",
    },
    "pvt_v2_b3": {
        "timm_name": "pvt_v2_b3.in1k",
        "ckpt": "/mnt/data/wandb/checkpoints/model_ablation_study/eltlatns/epoch=21-rank1tinyface/eval/rank@1=0.566.ckpt",
        "family": "pvt_v2",
    },
    "pvt_v2_b5": {
        "timm_name": "pvt_v2_b5.in1k",
        "ckpt": "/mnt/data/wandb/checkpoints/model_ablation_study/nvzory9h/epoch=21-rank1tinyface/eval/rank@1=0.566.ckpt",
        "family": "pvt_v2",
    },
    "mobilevitv2_200": {
        "timm_name": "mobilevitv2_200.cvnets_in1k",
        "ckpt": "/mnt/data/wandb/checkpoints/model_ablation_study/amsv29wr/epoch=42-rank1tinyface/eval/rank@1=0.525.ckpt",
        "family": "mobilevitv2",
    },
}


def process_model(
    name: str,
    info: dict,
    output_dir: str,
    img_size: int = 96,
) -> str | None:
    """Process a single model: load, rewrite, validate, export. Returns output path or None."""
    print(f"\n{'='*60}")
    print(f"Processing: {name} ({info['timm_name']})")
    print(f"{'='*60}")

    ckpt = info["ckpt"]
    if not Path(ckpt).exists():
        print(f"  SKIP: checkpoint not found: {ckpt}")
        return None

    # Load original backbone
    original = _load_backbone(ckpt, info["timm_name"])
    dummy = torch.randn(1, 3, img_size, img_size)

    # Create TRT-compatible model
    family = info["family"]
    if family == "convnext":
        trt_model = _make_convnext_trt(info["timm_name"], original, dummy)
    elif family == "pvt_v2":
        trt_model = _make_pvt_v2_trt(info["timm_name"], original, dummy)
    elif family == "mobilevitv2":
        trt_model = _make_mobilevitv2_trt(info["timm_name"], original, dummy)
    else:
        raise ValueError(f"Unknown family: {family}")

    # Export to ONNX — always use legacy (opset 17) to avoid SequenceEmpty/Loop issues
    out_path = str(Path(output_dir) / f"{name}.onnx")
    _export_onnx(trt_model, dummy, out_path)

    # Simplify
    _simplify_onnx(out_path)

    # Validate ONNX
    _validate_onnx(trt_model, out_path, dummy)

    # Report ONNX op stats
    model_onnx = onnx.load(out_path)
    ops = {}
    for node in model_onnx.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    problem_ops = {k: v for k, v in ops.items()
                   if k in ("LayerNormalization", "InstanceNormalization", "GroupNormalization")}
    if problem_ops:
        print(f"  WARNING: remaining problem ops: {problem_ops}")
    else:
        print(f"  No problem ops found (LayerNorm/InstanceNorm/GroupNorm)")
    n_transpose = ops.get("Transpose", 0)
    print(f"  Transpose ops: {n_transpose}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export TRT-compatible ONNX models")
    parser.add_argument(
        "--output_dir",
        default="onnx_models/baselines/trt_compat",
        help="Output directory for ONNX files",
    )
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Specific model names to export (default: all)",
    )
    args = parser.parse_args()

    models_to_process = MODELS
    if args.models:
        models_to_process = {k: v for k, v in MODELS.items() if k in args.models}

    results = {}
    for name, info in models_to_process.items():
        try:
            out = process_model(name, info, args.output_dir, args.img_size)
            results[name] = ("OK", out)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = ("FAIL", str(e))

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, (status, detail) in results.items():
        print(f"  {name}: {status} — {detail}")


if __name__ == "__main__":
    main()
