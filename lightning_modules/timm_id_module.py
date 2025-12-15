import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import timm

class TimmIDModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        img_size: int = 96,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Backbone returns pooled features (embeddings)
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,        # <- no classifier
            global_pool="avg",    # <- pooled embedding
            img_size=img_size,    # <- important for some timm models
        )
        emb_dim = self.backbone.num_features
        self.head = nn.Linear(emb_dim, num_classes)

        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)           # [B, D] typically
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Optional: sanity metric on the (probe) loader Lightning sees
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
