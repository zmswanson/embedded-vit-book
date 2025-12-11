from .dataloader import find_dataset_dir
import os
import glob

def find_mismatches(tinyface_path):
    dataset_dir = find_dataset_dir(tinyface_path, "Training_Set")
    mismatch_count = 0
    image_count = 0
    num_subjects = 0
    
    # Walk through each subdirectory (subject id) in the dataset directory
    for root, _, files in os.walk(dataset_dir):
        # For each <subject_id>_<picture_id>.jpg file in the current
        # subdirectory, check if subject_id matches the subdirectory name
        if os.path.basename(root) == "Training_Set":
            continue

        num_subjects += 1

        for file in files:
            if file.endswith(".jpg"):
                subject_id = file.split("_")[0]
                image_count += 1

                if subject_id != os.path.basename(root):
                    # Print the offending path
                    print(os.path.join(root, file))
                    mismatch_count += 1

    print(f"Total mismatches found: {mismatch_count}/{image_count}")
    all_paths = glob.glob(os.path.join(dataset_dir, "**", "*.jpg"))
    print(f"Total images in dataset: {len(all_paths)}")

    print(f"Total subjects: {num_subjects}")

if __name__ == "__main__":
    find_mismatches("/mnt/data/biometrics/tinyface/")