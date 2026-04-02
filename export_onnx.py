"""Export TimmIDModule / KDTimmIDModule backbone to ONNX format.

Handles plain, LoRA, and knowledge-distillation checkpoints.  LoRA weights
are merged before export so the ONNX graph contains only standard ops.
The exported model produces backbone embeddings (no classification head).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lightning_modules.lora import LoRALinear, _get_parent
from lightning_modules.timm_id_module import TimmIDModule


# ------------------------------------------------------------------
# KD checkpoint detection (mirrors inference.py)
# ------------------------------------------------------------------

def _is_kd_checkpoint(ckpt_path: str) -> bool:
    """Return True if *ckpt_path* was saved by KDTimmIDModule."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    return "kd_alpha" in hparams


# ------------------------------------------------------------------
# LoRA merge
# ------------------------------------------------------------------

def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA adapter weights into base weights, replacing each
    ``LoRALinear`` with a plain ``nn.Linear``.

    The merge formula is: ``W_merged = W_base + (alpha / r) * B @ A``.
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue

        merged_weight = module.base.weight.data + module.scaling * (
            module.lora_B.weight.data @ module.lora_A.weight.data
        )
        new_linear = nn.Linear(
            module.base.in_features,
            module.base.out_features,
            bias=module.base.bias is not None,
        )
        new_linear.weight.data = merged_weight
        if module.base.bias is not None:
            new_linear.bias.data = module.base.bias.data

        parent, attr = _get_parent(model, name)
        setattr(parent, attr, new_linear)

    return model


# ------------------------------------------------------------------
# ONNX validation
# ------------------------------------------------------------------

def validate_onnx(
    pytorch_model: nn.Module,
    onnx_path: str,
    dummy_input: torch.Tensor,
) -> None:
    """Compare PyTorch and ONNX Runtime outputs via cosine similarity."""
    import onnxruntime as ort

    pytorch_model.eval()
    with torch.no_grad():
        pt_output = pytorch_model(dummy_input).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {"input": dummy_input.numpy()})[0]

    cos_sim = float(
        np.dot(pt_output.flatten(), ort_output.flatten())
        / (np.linalg.norm(pt_output) * np.linalg.norm(ort_output))
    )
    print(f"Cosine similarity (PyTorch vs ONNX Runtime): {cos_sim:.6f}")
    assert cos_sim > 0.999, (
        f"ONNX validation failed: cosine similarity {cos_sim:.6f} < 0.999"
    )
    print("ONNX validation passed!")


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

def export_onnx(
    ckpt_path: str,
    output_path: str,
    model_name: str | None = None,
    img_size: int = 96,
    opset_version: int = 18,
    dynamic_batch: bool = True,
) -> None:
    """Load a checkpoint and export its backbone to ONNX.

    Parameters
    ----------
    ckpt_path : str
        Path to a Lightning checkpoint (TimmIDModule or KDTimmIDModule).
    output_path : str
        Destination ``.onnx`` file path.
    model_name : str or None
        timm model name.  Required for non-KD checkpoints (passed to
        ``TimmIDModule.load_from_checkpoint``).  KD checkpoints store
        ``student_model_name`` in their hparams so this can be omitted.
    img_size : int
        Spatial resolution the model was trained on.
    opset_version : int
        ONNX opset (>=14, 17 recommended).
    dynamic_batch : bool
        Whether to mark the batch dimension as dynamic.
    """
    # ------ load checkpoint ------
    is_kd = _is_kd_checkpoint(ckpt_path)

    if is_kd:
        from lightning_modules.kd_module import KDTimmIDModule

        module = KDTimmIDModule.load_from_checkpoint(ckpt_path)
        effective_name = module.hparams.get(
            "student_model_name", module.student_model_name
        )
        print(f"Loaded KD checkpoint – student: {effective_name}")
    else:
        if model_name is None:
            raise ValueError(
                "--model_name is required for non-KD checkpoints"
            )
        module = TimmIDModule.load_from_checkpoint(
            ckpt_path, model_name=model_name
        )
        effective_name = model_name
        print(f"Loaded checkpoint – model: {effective_name}")

    module.eval()
    module.cpu()

    # ------ extract backbone (student only for KD) ------
    backbone = module.backbone

    # ------ merge LoRA if present ------
    has_lora = any(isinstance(m, LoRALinear) for m in backbone.modules())
    if has_lora:
        merge_lora_weights(backbone)
        print(f"Merged LoRA weights for {effective_name}")

    backbone.eval()

    # ------ dummy input ------
    dummy = torch.randn(1, 3, img_size, img_size)

    # ------ export ------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = (
        {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        if dynamic_batch
        else None
    )

    torch.onnx.export(
        backbone,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    onnx_path = Path(output_path)
    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    # The dynamo exporter may store weights in a separate .data file
    data_path = Path(str(output_path) + ".data")
    if data_path.exists():
        onnx_size_mb += data_path.stat().st_size / (1024 * 1024)
    print(f"Exported ONNX model to {output_path}  ({onnx_size_mb:.1f} MB)")

    # ------ validate ------
    validate_onnx(backbone, output_path, dummy)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export TimmIDModule backbone to ONNX"
    )
    parser.add_argument("--ckpt_path", required=True, help="Lightning checkpoint path")
    parser.add_argument(
        "--model_name",
        default=None,
        help="timm model name (required for non-KD checkpoints)",
    )
    parser.add_argument("--output_path", required=True, help="Output .onnx file path")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--opset_version", type=int, default=18)
    parser.add_argument(
        "--no_dynamic_batch",
        action="store_true",
        help="Disable dynamic batch dimension",
    )
    args = parser.parse_args()

    export_onnx(
        ckpt_path=args.ckpt_path,
        output_path=args.output_path,
        model_name=args.model_name,
        img_size=args.img_size,
        opset_version=args.opset_version,
        dynamic_batch=not args.no_dynamic_batch,
    )
