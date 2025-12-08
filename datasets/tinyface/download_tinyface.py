#!/usr/bin/env python3
"""
Download and extract the TinyFace dataset to a user-specified directory.
Also writes datasets/tinyface_path.txt to store the dataset root.

Usage:
    python download_tinyface.py -d /path/to/target
"""

import argparse
import os
import sys
import zipfile
import subprocess

# Google Drive file ID for TinyFace
TINYFACE_FILE_ID = "1xTZc7lNmWN33ECO2AKH6FycGdiqIK7W0"
GDRIVE_URL = f"https://drive.google.com/uc?id={TINYFACE_FILE_ID}"
DEFAULT_ZIP_NAME = "tinyface.zip"


def ensure_gdown():
    """Ensure gdown is installed and importable."""
    try:
        import gdown
    except ImportError:
        print("[*] Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "gdown"])
        import gdown
    return gdown


def download_tinyface(dest_dir: str, zip_name: str = DEFAULT_ZIP_NAME):
    """Download tinyface.zip to dest_dir (if not present)."""
    gdown = ensure_gdown()

    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, zip_name)

    if os.path.exists(zip_path):
        print(f"[*] Found existing {zip_path}, skipping download.")
    else:
        print(f"[*] Downloading TinyFace → {zip_path}")
        gdown.download(GDRIVE_URL, zip_path, quiet=False)
        print("[*] Download complete.")

    return zip_path


def extract_zip(zip_path: str, dest_dir: str):
    """Extract the zip file to dest_dir."""
    print(f"[*] Extracting {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print("[*] Extraction complete.")


def write_dataset_path_file(dataset_root: str):
    """
    Write tinyface_path.txt inside scalable-vits-book/datasets/
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_file = os.path.join(script_dir, "tinyface_path.txt")

    with open(path_file, "w") as f:
        f.write(os.path.join(dataset_root, "tinyface") + "\n")

    print(f"[*] Saved dataset root → {path_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download and extract TinyFace dataset.")
    parser.add_argument(
        "-d", "--dest",
        default=".",
        help="Directory to download/extract the dataset (default: current directory)."
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep tinyface.zip after extraction."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dest_dir = os.path.abspath(args.dest)
    print(f"[*] Target download directory: {dest_dir}")

    # Download zip
    zip_path = download_tinyface(dest_dir)

    # Extract
    extract_zip(zip_path, dest_dir)

    # Save dataset root
    write_dataset_path_file(dest_dir)

    # Optionally remove zip
    if not args.keep_zip:
        try:
            os.remove(zip_path)
            print(f"[*] Removed zip file: {zip_path}")
        except OSError as e:
            print(f"[!] Could not remove zip: {e}")

    print("\n[✓] TinyFace dataset ready.")
    print(f"    Root: {dest_dir}")


if __name__ == "__main__":
    main()

