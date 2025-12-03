## Install Python Environment
From the project root, run the following:
```bash
make -C install env
```

After the enviroment is created, activate the conda environment: `conda activate vit-benchmark`

Verify the install (runs automatically with env): `make -C install verify`

Cleanup the installation: `make -C install cleanup`

## TinyFace Dataset Download
This repository includes a helper script for downloading and managing the TinyFace dataset.
The script downloads the dataset, extracts it, and saves the dataset root path to `scalable-vits-book/datasets/tinyface_path.txt`

### Download the dataset
From the project root:

``` bash
python datasets/download_tinyface.py -d datasets/tinyface
```

* `-d` specifies where the dataset should be stored
* The script automatically writes the final dataset root path to `tinyface_path.txt`

After running:

``` sql
datasets/
├── download_tinyface.py
└── tinyface_path.txt   ← contains absolute dataset root
```

### Accessing the Dataset Root Path in Code
Here is an example helper to access the dataset path:

``` python
from pathlib import Path

def get_tinyface_root():
    return Path("datasets/tinyface_path.txt").read_text().strip()

tinyface_root = get_tinyface_root()
print("TinyFace located at:", tinyface_root)
```

Any training or evaluation script can now load the dataset root without hard-coding paths.

