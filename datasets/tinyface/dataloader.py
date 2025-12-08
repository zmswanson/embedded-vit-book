import os
import glob
from typing import Optional, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image


# -----------------------------
# Transforms (example)
# -----------------------------
def get_transforms(img_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return train_transform, eval_transform


# -----------------------------
# Training Dataset (ImageFolder-based)
# -----------------------------
class TinyFacesTrainDataset(Dataset):
    """
    Wraps torchvision.datasets.ImageFolder over:
        <base_dir>/tinyface/Training_Set/<subject_id>/*.jpg

    Returns:
        image: Tensor [C,H,W]
        label: int (0..num_classes-1)
        subject_id: str (directory name)
        path: str (absolute image path)
    """
    def __init__(self, root: str, transform=None):
        # Use ImageFolder to walk filesystem & build labels
        self._image_folder = datasets.ImageFolder(root=root)
        self.transform = transform

        self.samples = self._image_folder.samples      # list of (path, class_idx)
        self.classes = self._image_folder.classes      # list of subject_id strings
        self.class_to_idx = self._image_folder.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.loader = self._image_folder.loader        # PIL loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int, str, str]:
        path, label = self.samples[idx]
        img = self.loader(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        subject_id = self.idx_to_class[label]
        return img, label, subject_id, path


# -----------------------------
# Eval Dataset for Gallery / Probe
# -----------------------------
class TinyFacesEvalDataset(Dataset):
    """
    For:
        <base_dir>/tinyface/Testing_Set/Gallery_Match/*.jpg
        <base_dir>/tinyface/Testing_Set/Probe/*.jpg

    Filenames are of the form: <subject_id>_<photo_id>.jpg

    If class_to_idx is given, it is used to keep label indices aligned
    with the training dataset. Otherwise a new mapping is created.
    """
    def __init__(
        self,
        root: str,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root = root
        self.transform = transform

        self.paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
        if len(self.paths) == 0:
            raise RuntimeError(f"No .jpg files found in {root}")

        self.filenames = [os.path.basename(p) for p in self.paths]
        self.subject_ids = [fn.split("_")[0] for fn in self.filenames]

        if class_to_idx is None:
            unique_ids = sorted(set(self.subject_ids))
            self.class_to_idx = {sid: i for i, sid in enumerate(unique_ids)}
        else:
            self.class_to_idx = class_to_idx

        self.labels = [self.class_to_idx[sid] for sid in self.subject_ids]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Any, int, str, str]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[idx]
        subject_id = self.subject_ids[idx]
        return img, label, subject_id, path


# -----------------------------
# Helper: deterministic train/val split
# -----------------------------
def create_tinyfaces_train_val_datasets(
    base_dir: str,
    val_split: float = 0.1,
    seed: int = 42,
    img_size: int = 224,
):
    """
    Creates train/val datasets with a deterministic split from Training_Set.
    Split is based on image indices, not subjects (i.e., random per-image split).
    """

    train_root = os.path.join(base_dir, "tinyface", "Training_Set")
    train_tf, eval_tf = get_transforms(img_size)

    # One dataset just for index generation (no transform needed)
    full_dataset = TinyFacesTrainDataset(train_root, transform=None)
    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)

    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    # Two "views" with different transforms but same underlying indexing
    base_train = TinyFacesTrainDataset(train_root, transform=train_tf)
    base_val = TinyFacesTrainDataset(train_root, transform=eval_tf)

    train_dataset = Subset(base_train, train_indices)
    val_dataset = Subset(base_val, val_indices)

    # class_to_idx mapping shared with eval sets
    class_to_idx = base_train.class_to_idx

    return train_dataset, val_dataset, class_to_idx


def create_tinyfaces_dataloaders(
    base_dir: str = "/mnt/data/biometrics",
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
    img_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    base_dir = os.path.join(base_dir)  # in case you want to override later

    # Train / Val
    train_dataset, val_dataset, class_to_idx = create_tinyfaces_train_val_datasets(
        base_dir=base_dir,
        val_split=val_split,
        seed=seed,
        img_size=img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Gallery / Probe for Testing
    _, eval_tf = get_transforms(img_size=img_size)

    gallery_root = os.path.join(base_dir, "tinyface", "Testing_Set", "Gallery_Match")
    probe_root = os.path.join(base_dir, "tinyface", "Testing_Set", "Probe")

    gallery_dataset = TinyFacesEvalDataset(
        root=gallery_root,
        transform=eval_tf,
        class_to_idx=class_to_idx,
    )

    probe_dataset = TinyFacesEvalDataset(
        root=probe_root,
        transform=eval_tf,
        class_to_idx=class_to_idx,
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    probe_loader = DataLoader(
        probe_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "gallery": gallery_loader,
        "probe": probe_loader,
        "class_to_idx": class_to_idx,
    }


# -----------------------------
# Example usage in a training script
# -----------------------------
if __name__ == "__main__":
    base_dir = "/mnt/data/biometrics"  # so that tinyface/ is under this

    loaders = create_tinyfaces_dataloaders(
        base_dir=base_dir,
        batch_size=64,
        val_split=0.1,
        seed=123,          # fixed for deterministic split
        img_size=224,
        num_workers=8,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    gallery_loader = loaders["gallery"]
    probe_loader = loaders["probe"]

    # Example loop:
    for images, labels, subject_ids, paths in train_loader:
        # your training step here
        pass
