if __name__=="__main__":
    import os
    from pathlib import Path
    import glob
    import json
    import numpy as np

    rng = np.random.default_rng(73)

    root_path = Path("/mnt/data/biometrics/tinyface/Training_Set/")
    unique_ids = np.array(list(set(
        img_path.split(os.sep)[-2] for img_path in glob.glob(os.path.join(root_path, "**", "*.jpg"))
    )))

    # randomly divide the the unique_ids into k sets
    num_val_sets = 10
    rng.shuffle(unique_ids)
    
    val_sets = np.array_split(unique_ids, num_val_sets)
    val_dict = {f"TRAIN_SET_{i}": val_sets[i].tolist() for i in range(num_val_sets)}

    with open(os.path.join(os.path.dirname(__file__), "cross_val_splits.json"), "w") as f:
        json.dump(val_dict, f, indent=4)

    for i, val_set in enumerate(val_sets):
        print(f"TRAIN_SET_{i}: {len(val_set)} unique IDs")
