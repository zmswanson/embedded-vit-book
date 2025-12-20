import torch
import lightning as L

from datasets.tinyface.dataloader import get_tinyface_path, DatasetType
from lightning_modules.tinyface_datamodule import TinyFaceDataModule
from lightning_modules.timm_id_module import TimmIDModule
from lightning_modules.callbacks import TinyFaceEvaluationCallback
from model_eval.probe_gallery import MultiLabelMethod


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

    # Load model *from checkpoint*
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

    trainer.validate(model, datamodule=dm, ckpt_path=ckpt_path)

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

    args = parser.parse_args()

    run_inference(
        ckpt_path=args.ckpt_path,
        model_name=args.model_name,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )