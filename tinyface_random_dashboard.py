#!/usr/bin/env python3
"""
TinyFace Dataset Browser (Streamlit)

Run from repo root:
    streamlit run tinyface_dashboard.py --server.address 0.0.0.0 --server.port 8501
"""

import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import streamlit as st


# -------------------------
# Helpers
# -------------------------

def get_tinyface_root() -> Path:
    """Read datasets/tinyface_path.txt to get the TinyFace root directory."""
    repo_root = Path(__file__).resolve().parent
    path_file = repo_root / "datasets" / "tinyface_path.txt"
    if not path_file.exists():
        st.error(f"tinyface_path.txt not found at {path_file}. "
                 "Run datasets/download_tinyface.py first.")
        st.stop()
    root = Path(path_file.read_text().strip())
    if not root.exists():
        st.error(f"TinyFace root path from tinyface_path.txt does not exist:\n{root}")
        st.stop()
    return root


@st.cache_data(show_spinner=True)
def build_image_index(root: Path,
                      exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")):
    """
    Recursively index all image files under TinyFace root.

    Returns:
        List of dicts with keys: rel_path, full_path, split
    """
    items = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        split = parts[0] if len(parts) > 1 else "(root)"
        items.append(
            {
                "rel_path": str(rel),
                "full_path": str(p),
                "split": split,
            }
        )
    return items


def load_image(path: str) -> Image.Image:
    """Load an image via Pillow."""
    return Image.open(path).convert("RGB")


# -------------------------
# Streamlit UI
# -------------------------

def main():
    st.set_page_config(page_title="TinyFace Browser", layout="wide")
    st.title("TinyFace Dataset Browser")

    # Locate dataset root
    tinyface_root = get_tinyface_root()
    st.caption(f"TinyFace root: `{tinyface_root}`")

    # Build index
    with st.spinner("Indexing images..."):
        index = build_image_index(tinyface_root)

    if not index:
        st.error("No images found under the TinyFace root.")
        st.stop()

    # Sidebar controls
    st.sidebar.header("Filters")

    # Unique splits (top-level directory names)
    splits = sorted({item["split"] for item in index})
    selected_split = st.sidebar.selectbox(
        "Top-level folder (split)",
        options=["(all)"] + splits,
        index=0,
    )

    # Number of images
    max_images = st.sidebar.slider("Number of images to display", 1, 200, 24, step=1)

    # Randomize?
    randomize = st.sidebar.checkbox("Random order", value=True)

    # Optional filename filter
    name_filter = st.sidebar.text_input("Filter by filename substring", value="")

    # Filter index
    filtered = index
    if selected_split != "(all)":
        filtered = [it for it in filtered if it["split"] == selected_split]

    if name_filter:
        nf = name_filter.lower()
        filtered = [it for it in filtered if nf in it["rel_path"].lower()]

    # Sort or shuffle
    if randomize:
        import random
        random.shuffle(filtered)
    else:
        filtered = sorted(filtered, key=lambda x: x["rel_path"])

    # Limit
    filtered = filtered[:max_images]

    st.write(f"Showing **{len(filtered)}** images"
             + ("" if selected_split == "(all)" else f" from split `{selected_split}`"))

    # Layout: grid with 4 columns
    n_cols = 4
    cols = st.columns(n_cols)

    for i, item in enumerate(filtered):
        col = cols[i % n_cols]
        with col:
            img = load_image(item["full_path"])
            st.image(img, caption=item["rel_path"], use_container_width=True)


if __name__ == "__main__":
    main()

