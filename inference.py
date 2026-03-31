import os
import torch
import lightning as L

from datasets.tinyface.dataloader import get_tinyface_path, DatasetType
from lightning_modules.tinyface_datamodule import TinyFaceDataModule
from lightning_modules.timm_id_module import TimmIDModule
from lightning_modules.callbacks import TinyFaceEvaluationCallback
from model_eval.probe_gallery import MultiLabelMethod


def _is_kd_checkpoint(ckpt_path: str) -> bool:
    """Detect whether a checkpoint was saved by KDTimmIDModule."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    return "kd_alpha" in hparams


def run_inference(
    ckpt_path: str,
    model_name: str,
    img_size: int = 96,
    batch_size: int = 512,
):
    # Data
    dm = TinyFaceDataModule(
        tinyface_root=get_tinyface_path(),
        img_size=img_size,
        batch_size=batch_size,
    )

    # Load model *from checkpoint* — auto-detect KD vs standard
    if _is_kd_checkpoint(ckpt_path):
        from lightning_modules.kd_module import KDTimmIDModule
        model = KDTimmIDModule.load_from_checkpoint(ckpt_path)
        # Override validation_step: the KD module's validation_step tries
        # F.cross_entropy on raw embeddings which can error.  The callback
        # handles the actual evaluation, so a no-op is fine.
        model.validation_step = lambda batch, batch_idx: None
    else:
        model = TimmIDModule.load_from_checkpoint(
            ckpt_path,
            model_name=model_name,
        )

    # IMPORTANT: ensure eval mode
    model.eval()

    # Evaluation callback
    eval_cb = TinyFaceEvaluationCallback(
        tinyface_root=get_tinyface_path(),
        img_size=img_size,
        batch_size=batch_size,
        rank_k=10,
        probe_batch_size=256,
        multi_label_method=MultiLabelMethod.MAX,
        also_log_roc=True,
        eval_on=DatasetType.TEST,   # 👈 TEST SET
    )

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[eval_cb],
        logger=False,      # no training logs
        enable_checkpointing=False,
    )

    return trainer.validate(model=model, datamodule=dm)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on TinyFace dataset")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the timm model architecture",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=96,
        help="Input image size",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        required=False,
        help="WandB run ID",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=False,
        help="Path to save the inference results",
    )

    # # LoRA (optional)
    # parser.add_argument("--lora_enabled", action="store_true", help="Enable LoRA adapters in the backbone.")
    # parser.add_argument("--lora_r", type=int, default=8)
    # parser.add_argument("--lora_alpha", type=float, default=16.0)
    # parser.add_argument("--lora_dropout", type=float, default=0.0)
    # parser.add_argument(
    #     "--lora_family",
    #     type=str,
    #     default="auto",
    #     choices=["auto", "vit", "swin", "pvt", "mobilevit", "levit", "efficientformer", "cnn"],
    #     help="Override inferred model family for LoRA target patterns.",
    # )
    # parser.add_argument(
    #     "--lora_target_regex",
    #     type=str,
    #     default=None,
    #     help="Override LoRA target regex. If set, family-based defaults are ignored.",
    # )
    # parser.add_argument("--lora_qkv", action="store_true", help="Apply LoRA to QKV (family-specific).")
    # parser.add_argument("--lora_proj", action="store_true", help="Apply LoRA to projection (family-specific).")

    args = parser.parse_args()

    results = run_inference(
        ckpt_path=args.ckpt_path,
        model_name=args.model_name,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    save_keys = [
        'tinyface/test/rank@1',
        'tinyface/test/rank@2',
        'tinyface/test/rank@3',
        'tinyface/test/rank@4',
        'tinyface/test/rank@5',
        'tinyface/test/rank@6',
        'tinyface/test/rank@7',
        'tinyface/test/rank@8',
        'tinyface/test/rank@9',
        'tinyface/test/rank@10',
        'tinyface/test/auc',
        'tinyface/test/mAP',
        'tinyface/test/eer',
        'tinyface/test/tpr_at_fpr_1pct',
        'tinyface/test/tpr_at_fpr_5pct',
    ]

    if args.save_path:
        if not os.path.exists(args.save_path):
            if not os.path.exists(os.path.dirname(args.save_path)):
                os.makedirs(os.path.dirname(args.save_path))
            with open(args.save_path, "w") as f:
                f.write("wandb_id,model_name")
                for key in save_keys:
                    f.write(f",{key}")
                f.write("\n")

        with open(args.save_path, "a") as f:
            f.write(f"{args.wandb_id},{args.model_name}")
            for key in save_keys:
                f.write(f",{results[0].get(key, '')}")
            f.write("\n")
    else:
        print(f"Inference results: {len(results)}")
        for k, v in results[0].items():
            if k in save_keys:
                print(f"{k}: {v}")
    