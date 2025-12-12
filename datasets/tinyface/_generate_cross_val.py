def create_crossval_splits(n_splits: int =10):
    import os
    from pathlib import Path
    import glob
    import json
    import numpy as np
    from .dataloader import get_tinyface_path

    rng = np.random.default_rng(73)

    root_path = Path(get_tinyface_path()) / "Training_Set"

    if not root_path.exists():
        raise FileNotFoundError(f"TinyFace Training_Set directory not found at {root_path}")

    unique_ids = np.array(list(set(
        img_path.split(os.sep)[-2] for img_path in glob.glob(os.path.join(root_path, "**", "*.jpg"))
    )))

    # randomly divide the the unique_ids into k sets
    rng.shuffle(unique_ids)
    
    val_sets = np.array_split(unique_ids, n_splits)
    val_dict = {f"TRAIN_SET_{i}": val_sets[i].tolist() for i in range(n_splits)}

    with open(os.path.join(os.path.dirname(__file__), "cross_val_splits.json"), "w") as f:
        json.dump(val_dict, f, indent=4)

    for i, val_set in enumerate(val_sets):
        print(f"TRAIN_SET_{i}: {len(val_set)} unique IDs")


if __name__=="__main__":
    create_crossval_splits()