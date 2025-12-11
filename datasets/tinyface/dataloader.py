import os
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from PIL import Image
from json import load as json_load

# Manual seed for reproducibility
np.random.seed(73)
torch.manual_seed(73)

# -----------------------------
# Transforms (example)
# -----------------------------
def get_training_transforms(img_size: int = 224):
    """
    Helper function to generate a base set of transformations for training.
    
    :param img_size: Dimension of the square image to resize to.
    :type img_size: int
    :return: A torchvision.transforms.Compose object with the specified transformations.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform

def get_eval_transforms(img_size: int = 224):
    """
    Helper function to generate a base set of transformations for evaluation.
    Provides the necessary resize, to tensor, and normalization steps.

    :param img_size: Dimension of the square image to resize to.
    :type img_size: int
    :return: A torchvision.transforms.Compose object with the specified transformations.
    """

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return eval_transform


class TinyFaceDataset(Dataset):
    """
    Custom dataset class for loading images from the TinyFace dataset.
    Seperate objects of this class should be created for training, evaluation,
    and testing datasets.

    :param tinyfaces_path: Path to the TinyFace dataset directory.
    :type tinyfaces_path: str
    :param crossval_splits: List of integers representing the cross-validation splits to use.
                            Only use for training and evaluation datasets.
    :type crossval_splits: list of int or int
    :param transform: Optional torchvision.transforms.Compose object to apply to the images.
                      Use get_training_transforms or get_eval_transforms if you don't want to
                      define your own transformations.
    :type transform: torchvision.transforms.Compose or None
    """
    def __init__(self, tinyfaces_path, crossval_splits=None, transform=None):
        self.root_path = Path(tinyfaces_path)
        self.img_paths = [
            img_path for img_path in
            glob.glob(os.path.join(self.root_path, "**", "*.jpg"))
        ]

        if len(self.img_paths) == 0:
            # This handles the Testing_Set gallery and probe where we have a
            # flat directory of images
            self.img_paths = [
                img_path for img_path in 
                glob.glob(os.path.join(self.root_path, "*.jpg"))
            ]
            

        if crossval_splits is not None:
            if isinstance(crossval_splits, int):
                crossval_splits = [crossval_splits]
            elif not isinstance(crossval_splits, list):
                raise ValueError("crossval_splits must be an int or a list of ints")
            
            crossval_path = os.path.join(
                os.path.dirname(__file__), "cross_val_splits.json"
            )

            with open(crossval_path, "r") as f:
                crossval_data = json_load(f)

            # Extract all the subjects for the specified cross validation splits
            self.subject_ids = set([
                subject_id for i in crossval_splits
                for subject_id in crossval_data[f"TRAIN_SET_{i}"]
            ])

            # Only keep the images that contain the selected subject_ids
            self.img_paths = [
                img_path for img_path in self.img_paths
                if os.path.basename(img_path).split("_", 1)[0] in self.subject_ids
            ]

        self.subject_ids = set([
            os.path.basename(img_path).split("_", 1)[0]
            for img_path in self.img_paths
        ])

        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        """
        Load an image from the dataset and apply transformations if specified.
        Extracts the subject_id from the image path.
        
        :param index: Index of the image to load.
        :type index: int
        :return: A tuple of (image, subject_id).
        :rtype: tuple
        """
        if self.img_paths is None:
            return None
        
        # Load image with OpenCV and convert to RGB
        img_array = cvtColor(imread(self.img_paths[index]), COLOR_BGR2RGB)
        # Convert numpy array to PIL Image for torchvision transforms
        img = Image.fromarray(img_array)

        if self.transform is not None:
            img = self.transform(img)
            
        # if the img is not a tensor, convert it to one
        if img is not None and not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)

        subject_id = os.path.basename(self.img_paths[index]).split("_")[0]

        return img, subject_id
    
    def _get_imgpath_subid(self, index):
        """
        Helper function to get the image path and subject_id for a given index.

        :param index: Index of the image.
        :type index: int
        :return: A tuple of (image_path, subject_id).
        :rtype: tuple
        """
        img_path = self.img_paths[index]
        subject_id = os.path.basename(img_path).split("_")[0]

        return img_path, subject_id
    

def find_dataset_dir(tinyfaces_path: str, dataset_name: str):
    """
    Helper function to find the dataset directory within the TinyFace dataset path.
    If the dataset_name is not found in the tinyfaces_path, it will try to find
    the Training_Set directory in the parent directory by scanning the subdirectories
    and checking if the dataset_name is in the subdirectory path.

    :param tinyfaces_path: Path to the TinyFace dataset directory, e.g., /mnt/data/biometrics/tinyface/
    :type tinyfaces_path: str
    :param dataset_name: Name of the dataset to find, e.g., "Training_Set"
    :type dataset_name: str
    :return: The path to the dataset directory, e.g., /mnt/data/biometrics/tinyface/Training_Set/
    :rtype: str
    """
    if not os.path.exists(tinyfaces_path):
        raise FileNotFoundError(f"Path {tinyfaces_path} does not exist")
    
    if dataset_name not in tinyfaces_path:
        if os.path.exists(os.path.join(tinyfaces_path, dataset_name)):
            tinyfaces_path = os.path.join(tinyfaces_path, dataset_name)
        else:
            # try to find the Training_Set directory in the parent directory by
            # scanning the subdirectories (avoiding . and ..)
            dir_found = False
            for sub_dir in os.listdir(tinyfaces_path):
                if sub_dir != "." and sub_dir != "..":
                    sub_dir_path = os.path.join(tinyfaces_path, sub_dir)
                    if os.path.isdir(sub_dir_path) and dataset_name in sub_dir_path:
                        tinyfaces_path = sub_dir_path
                        dir_found = True
                        break

            if not dir_found:
                raise FileNotFoundError(f"Dataset {dataset_name} not found in {tinyfaces_path}")
            
    return tinyfaces_path
    

def get_train_loader(tinyfaces_path: str, batch_size: int = 32, img_size: int = 224):
    """
    Helper function to create a DataLoader for the TinyFace training dataset.
    
    :param tinyfaces_path: Path to the TinyFace dataset directory, e.g., /mnt/data/biometrics/tinyface/
    :type tinyfaces_path: str
    :param batch_size: Number of samples per batch to load.
    :type batch_size: int
    :param img_size: Dimension of the square image to resize to.
    :type img_size: int
    
    :return: A DataLoader object for the training dataset.
    :rtype: torch.utils.data.DataLoader
    """
    tinyfaces_path = find_dataset_dir(tinyfaces_path, "Training_Set")

    train_dataset = TinyFaceDataset(
        tinyfaces_path, crossval_splits=list(range(9)),
        transform=get_training_transforms(img_size)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    return train_loader


def get_eval_loader(tinyfaces_path: str, batch_size: int = 32, img_size: int = 224):
    """
    Helper function to create a DataLoader for the TinyFace evaluation dataset,
    which is specified by the cross-validation split 9 of the TinyFace Training_Set.
    This function handles further splitting the dataset into gallery and probe sets
    and returns two DataLoader objects, one for the gallery and one for the probe set.

    :param tinyfaces_path: Path to the TinyFace dataset directory, e.g., /mnt/data/biometrics/tinyface/
    :type tinyfaces_path: str
    :param batch_size: Number of samples per batch to load.
    :type batch_size: int
    :param img_size: Dimension of the square image to resize to.
    :type img_size: int

    :return: Two DataLoader objects, one for the gallery and one for the probe set.
    :rtype: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
    """
    tinyfaces_path = find_dataset_dir(tinyfaces_path, "Training_Set")

    probe_dataset = TinyFaceDataset(
        tinyfaces_path, crossval_splits=9,
        transform=get_eval_transforms(img_size)
    )

    gallery_dataset = TinyFaceDataset(
        tinyfaces_path, crossval_splits=9,
        transform=get_eval_transforms(img_size)
    )

    # Take the first image from each subject for gallery and the rest for probes
    gallery_img_path_indexes = {}
    for idx in range(len(gallery_dataset)):
        _, sub_id = gallery_dataset._get_imgpath_subid(idx)
        if sub_id not in gallery_img_path_indexes:
            gallery_img_path_indexes[sub_id] = idx

    gallery_list = list(gallery_img_path_indexes.values())
    probe_list = list(set(
        range(len(gallery_dataset))) - set(gallery_img_path_indexes.values()
    ))

    # Manually filter the gallery and probe datasets
    gallery_dataset.img_paths = [gallery_dataset.img_paths[i] for i in gallery_list]
    probe_dataset.img_paths = [probe_dataset.img_paths[i] for i in probe_list]

    probe_loader = DataLoader(
        probe_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return gallery_loader, probe_loader


def get_test_loaders(tinyfaces_path: str, batch_size: int = 32, img_size: int = 224):
    """
    Helper function to create DataLoader objects for the TinyFace test dataset.
    This function handles loading the Gallery_Match and Probe datasets from the Testing_Set.
    The Gallery_Match and Probe datasets are loaded separately and returned as two 
    DataLoader objects.

    :param tinyfaces_path: Path to the TinyFace dataset directory, e.g., 
                            /mnt/data/biometrics/tinyface/. Note that the function
                            handles finding the Gallery_Match and Probe directories
                            within the Testing_Set directory.
    :type tinyfaces_path: str
    :param batch_size: Number of samples per batch to load.
    :type batch_size: int
    :param img_size: Dimension of the square image to resize to.
    :type img_size: int

    :return: Two DataLoader objects, one for the gallery and one for the probe set.
    :rtype: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
    """
    tinyfaces_path = find_dataset_dir(tinyfaces_path, "Testing_Set")
    gallery_path = find_dataset_dir(tinyfaces_path, "Gallery_Match")
    probe_path = find_dataset_dir(tinyfaces_path, "Probe")
    
    gallery_dataset = TinyFaceDataset(
        gallery_path, crossval_splits=None,
        transform=get_eval_transforms(img_size)
    )
    probe_dataset = TinyFaceDataset(
        probe_path, crossval_splits=None,
        transform=get_eval_transforms(img_size)
    )

    gallery_loader = DataLoader(
        gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    probe_loader = DataLoader(
        probe_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return gallery_loader, probe_loader



if __name__ == "__main__":
    # Example usage
    tinyfaces_path = "/mnt/data/biometrics/tinyface/"
    train_loader = get_train_loader(tinyfaces_path, batch_size=32, img_size=224)
    eval_gallery_loader, eval_probe_loader = get_eval_loader(tinyfaces_path, batch_size=32, img_size=224)
    gallery_loader, probe_loader = get_test_loaders(tinyfaces_path, batch_size=32, img_size=224)

    print("Train dataset size:", len(train_loader.dataset))
    print("Eval gallery, probe dataset sizes: ", 
        len(eval_gallery_loader.dataset), " + ",
        len(eval_probe_loader.dataset), " = ",
        len(eval_gallery_loader.dataset) + len(eval_probe_loader.dataset),
        sep=""
    )
    print("Test gallery, probe dataset sizes: ",
        len(gallery_loader.dataset), " + ",
        len(probe_loader.dataset), " = ",
        len(gallery_loader.dataset) + len(probe_loader.dataset),        
        "\n", sep=""
    )

    print("Train subjects:", len(set(train_loader.dataset.subject_ids)))
    print("Eval subjects: ",
        len(eval_gallery_loader.dataset.subject_ids), " (G), ",
        len(eval_probe_loader.dataset.subject_ids), " (P)", sep=""
    )
    print("Test subjects: ",
        len(gallery_loader.dataset.subject_ids), " (G), ",
        len(probe_loader.dataset.subject_ids), " (P)\n", sep=""
    )


    # Print some data to verify
    for batch_idx, (images, subject_ids) in enumerate(train_loader):
        print(f"Batch {batch_idx} - Images shape: {images.shape}, Subject IDs: {subject_ids[:5]}")
        if batch_idx >= 2:
            break

    print()

    for batch_idx, (images, subject_ids) in enumerate(eval_gallery_loader):
        print(f"Eval Gallery Batch {batch_idx} - Images shape: {images.shape}, Subject IDs: {subject_ids[:5]}")
        if batch_idx >= 2:
            break

    for batch_idx, (images, subject_ids) in enumerate(eval_probe_loader):
        print(f"Eval Probe Batch {batch_idx} - Images shape: {images.shape}, Subject IDs: {subject_ids[:5]}")
        if batch_idx >= 2:
            break
        
    print()

    for batch_idx, (images, subject_ids) in enumerate(gallery_loader):
        print(f"Test Gallery Batch {batch_idx} - Images shape: {images.shape}, Subject IDs: {subject_ids[:5]}")
        if batch_idx >= 2:
            break

    for batch_idx, (images, subject_ids) in enumerate(probe_loader):
        print(f"Test Probe Batch {batch_idx} - Images shape: {images.shape}, Subject IDs: {subject_ids[:5]}")
        if batch_idx >= 2:
            break