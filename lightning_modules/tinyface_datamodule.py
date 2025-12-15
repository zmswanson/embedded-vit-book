import lightning as L

from datasets.tinyface.dataloader import (
    get_tinyface_path,
    get_train_loader,
    get_eval_loaders,
    get_test_loaders,
)

class TinyFaceDataModule(L.LightningDataModule):
    """
    LightningDataModule for TinyFace dataset.
    """
    def __init__(
        self, tinyface_root: str | None = None, img_size: int = 96, 
        batch_size: int = 64, num_workers: int = 4
    ):
        super().__init__()
        self.tinyface_root = tinyface_root or get_tinyface_path()
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return get_train_loader(
            self.tinyface_root,
            batch_size=self.batch_size,
            img_size=self.img_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        # Lightning “val” loader is optional since I perform probe-gallery evaluation.
        # Still return something cheap so Lightning has a validation loop.
        _, probe_loader = get_eval_loaders(
            self.tinyface_root, 
            batch_size=self.batch_size, 
            img_size=self.img_size,
            num_workers=self.num_workers
        )
        
        return probe_loader  # or gallery_loader; not used for rank-k callback below.

    def eval_gallery_probe_loaders(self):
        return get_eval_loaders(
            self.tinyface_root, 
            batch_size=self.batch_size, 
            img_size=self.img_size,
            num_workers=self.num_workers
        )

    def test_gallery_probe_loaders(self):
        return get_test_loaders(
            self.tinyface_root, 
            batch_size=self.batch_size, 
            img_size=self.img_size,
            num_workers=self.num_workers
        )
