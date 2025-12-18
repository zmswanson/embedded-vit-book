import cv2
from datasets.tinyface.dataloader import get_tinyface_path, find_dataset_dir
import os
import numpy as np


from typing import List

path = find_dataset_dir(get_tinyface_path(), "Training_Set")

widths: List[int] = []
heights: List[int] = []

# find all image files under path
for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            if img is not None:
                heights.append(img.shape[0])
                widths.append(img.shape[1])

print(f"Collected {len(widths)} image widths and {len(heights)} image heights.")

widths = np.array(widths)
heights = np.array(heights)

print(f"Width stats: min={widths.min()}, max={widths.max()}, mean={widths.mean():.2f}, std={widths.std():.2f}")
print(f"Height stats: min={heights.min()}, max={heights.max()}, mean={heights.mean():.2f}, std={heights.std():.2f}")