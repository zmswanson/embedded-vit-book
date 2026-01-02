import torch
from lightning.pytorch.callbacks import Callback

from datasets.tinyface.dataloader import (
    DatasetType, get_eval_loaders, get_test_loaders
)

from model_eval.probe_gallery import (
    MultiLabelMethod,
    get_batch_features,
    get_cosine_similarity,
    get_rank_k_accuracy,
    get_roc_metrics,
)

class TinyFaceEvaluationCallback(Callback):
    def __init__(
        self,
        tinyface_root: str,
        img_size: int = 96,
        batch_size: int = 64,
        rank_k: int = 10,
        probe_batch_size: int = 256,
        multi_label_method: MultiLabelMethod = MultiLabelMethod.ALL,
        also_log_roc: bool = True,
        eval_on: DatasetType = DatasetType.EVAL,
    ):
        super().__init__()
        self.tinyface_root = tinyface_root
        self.img_size = img_size
        self.batch_size = batch_size
        self.rank_k = rank_k
        self.probe_batch_size = probe_batch_size
        self.multi_label_method = multi_label_method
        self.also_log_roc = also_log_roc
        self.eval_on = eval_on

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()

        if self.eval_on == DatasetType.EVAL:
            gallery_loader, probe_loader = get_eval_loaders(
                self.tinyface_root,
                batch_size=self.batch_size,
                img_size=self.img_size,
            )
        else:
            gallery_loader, probe_loader = get_test_loaders(
                self.tinyface_root,
                batch_size=self.batch_size,
                img_size=self.img_size,
            )

        # backbone.forward(x) returns embeddings because num_classes=0.
        backbone = pl_module.backbone

        gallery_features, gallery_labels = get_batch_features(
            backbone, gallery_loader, device=pl_module.device
        )
        probe_features, probe_labels = get_batch_features(
            backbone, probe_loader, device=pl_module.device
        )

        cos_sim = get_cosine_similarity(
            probe_features,
            gallery_features,
            probe_batch_size=self.probe_batch_size,
            device=pl_module.device,
        )

        ranks = get_rank_k_accuracy(
            cos_sim,
            probe_labels,
            gallery_labels,
            k=self.rank_k,
            multi_label_method=self.multi_label_method,
        )

        for i, v in enumerate(ranks, start=1):
            pl_module.log(
                f"tinyface/{self.eval_on.name.lower()}/rank@{i}",
                float(v),
                prog_bar=(i == 1),
                on_epoch=True,
                logger=True,
            )

        if self.also_log_roc:
            roc = get_roc_metrics(
                cos_sim,
                probe_labels,
                gallery_labels,
                return_dict=True,
                multi_label_method=self.multi_label_method,
            )
            for k, v in roc.items():
                if isinstance(v, float):
                    pl_module.log(
                        f"tinyface/{self.eval_on.name.lower()}/{k}",
                        float(v),
                        on_epoch=True,
                        logger=True,
                    )


import lightning as L

class GradualBlockUnfreezeCallback(L.Callback):
    def __init__(self, blocks, every_n_epochs=5, verbose=True):
        """
        Args:
            blocks: list of nn.Module objects (ordered deepest → shallowest)
            every_n_epochs: how often to unfreeze the next block
        """
        self.blocks = blocks
        self.every_n_epochs = every_n_epochs
        self.verbose = verbose
        self._next_idx = 0

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch == 0:
            return

        if epoch % self.every_n_epochs != 0:
            return

        if self._next_idx >= len(self.blocks):
            return

        block = self.blocks[self._next_idx]

        for p in block.parameters():
            p.requires_grad = True

        if self.verbose:
            pl_module.print(
                f"[gradual-unfreeze] Epoch {epoch}: unfroze block "
                f"{self._next_idx + 1}/{len(self.blocks)}"
            )

        self._next_idx += 1


class GradualStageUnfreezeCallback(L.Callback):
    def __init__(self, stages, every_n_epochs=5, verbose=True):
        """
        Args:
            stages: list of nn.Module objects (ordered deepest → shallowest)
            every_n_epochs: how often to unfreeze the next stage
        """
        self.stages = stages
        self.every_n_epochs = every_n_epochs
        self.verbose = verbose
        self._next_idx = 0

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch == 0:
            return

        if epoch % self.every_n_epochs != 0:
            return

        if self._next_idx >= len(self.stages):
            return

        stage = self.stages[self._next_idx]

        for p in stage.parameters():
            p.requires_grad = True

        if self.verbose:
            pl_module.print(
                f"[gradual-unfreeze] Epoch {epoch}: unfroze stage "
                f"{self._next_idx + 1}/{len(self.stages)}"
            )

        self._next_idx += 1