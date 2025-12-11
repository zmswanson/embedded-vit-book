#!/usr/bin/env python3
"""
TinyFace Dataset Browser (Streamlit)

Run from repo root:
    streamlit run tinyface_structured_dashboard.py --server.address 0.0.0.0 --server.port 8501
"""

import re
from pathlib import Path
from typing import Tuple

from PIL import Image
import streamlit as st



# -------------------------
# Helpers
# -------------------------

def get_tinyface_root() -> Path:
    """Read datasets/tinyface_path.txt to get the TinyFace root directory."""
    repo_root = Path(__file__).resolve().parent
    path_file = repo_root / "tinyface_path.txt"
    if not path_file.exists():
        st.error(
            f"tinyface_path.txt not found at {path_file}. "
            "Run datasets/download_tinyface.py first."
        )
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


def list_dir(path: Path):
    """Return (dirs, files) for a given directory."""
    dirs, files = [], []
    for entry in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if entry.is_dir():
            dirs.append(entry)
        else:
            files.append(entry)
    return dirs, files


def strip_markdown_images(content: str) -> str:
    """
    Remove markdown and HTML image tags to avoid Streamlit trying to serve
    missing media files when README.md or other text references images.
    """
    # Remove markdown-style images ![...](...)
    content = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", content)
    # Remove HTML <img> tags
    content = re.sub(r"<img[^>]*>", "", content, flags=re.IGNORECASE)
    # Remove <figure>...</figure> blocks
    content = re.sub(r"<figure[^>]*>.*?</figure>", "", content, flags=re.IGNORECASE | re.DOTALL)
    # Remove leftover empty HTML wrappers (optional)
    content = re.sub(r"<p>\s*</p>", "", content)
    return content


# -------------------------
# Streamlit UI
# -------------------------

def main():
    st.set_page_config(page_title="TinyFace Browser", layout="wide")
    st.title("TinyFace Dataset Browser")

    # Locate dataset root
    tinyface_root = get_tinyface_root()
    st.caption(f"TinyFace root: `{tinyface_root}`")

    # Tabs: Images | Files
    tab_images, tab_files = st.tabs(["🖼 Images", "📁 Files"])

    # -------- Images tab --------
    with tab_images:
        with st.spinner("Indexing images..."):
            index = build_image_index(tinyface_root)

        if not index:
            st.error("No images found under the TinyFace root.")
        else:
            st.sidebar.header("Image Filters")

            splits = sorted({item["split"] for item in index})
            selected_split = st.sidebar.selectbox(
                "Top-level folder (split)",
                options=["(all)"] + splits,
                index=0,
            )

            max_images = st.sidebar.slider(
                "Number of images to display", 1, 200, 24, step=1
            )
            randomize = st.sidebar.checkbox("Random order", value=True)
            name_filter = st.sidebar.text_input(
                "Filter by filename substring", value=""
            )

            filtered = index
            if selected_split != "(all)":
                filtered = [it for it in filtered if it["split"] == selected_split]
            if name_filter:
                nf = name_filter.lower()
                filtered = [it for it in filtered if nf in it["rel_path"].lower()]

            if randomize:
                import random
                random.shuffle(filtered)
            else:
                filtered = sorted(filtered, key=lambda x: x["rel_path"])

            filtered = filtered[:max_images]

            st.write(
                f"Showing **{len(filtered)}** images"
                + ("" if selected_split == "(all)" else f" from split `{selected_split}`")
            )

            n_cols = 4
            cols = st.columns(n_cols)

            for i, item in enumerate(filtered):
                col = cols[i % n_cols]
                with col:
                    img = load_image(item["full_path"])
                    # Grid thumbnails: stretch to column width
                    st.image(img, caption=item["rel_path"], width="stretch")

    # -------- Files tab --------
    with tab_files:
        st.subheader("Directory, File & Image Explorer")

        # Init current directory in session_state
        if "tinyface_current_dir" not in st.session_state:
            st.session_state.tinyface_current_dir = str(tinyface_root)

        cur_dir = Path(st.session_state.tinyface_current_dir)

        # Safety: ensure current dir stays under root
        try:
            cur_dir.relative_to(tinyface_root)
        except ValueError:
            cur_dir = tinyface_root
            st.session_state.tinyface_current_dir = str(tinyface_root)

        top_bar_cols = st.columns([1, 5])
        with top_bar_cols[0]:
            if st.button("⬆️ Up", disabled=(cur_dir == tinyface_root)):
                parent = cur_dir.parent
                if tinyface_root in parent.parents or parent == tinyface_root:
                    st.session_state.tinyface_current_dir = str(parent)
                else:
                    st.session_state.tinyface_current_dir = str(tinyface_root)
                st.rerun()

        with top_bar_cols[1]:
            rel_cur = cur_dir.relative_to(tinyface_root)
            st.markdown(
                f"**Current directory:** `{tinyface_root.name}/{rel_cur}`"
                if rel_cur.parts
                else f"**Current directory:** `{tinyface_root.name}/`"
            )

        dirs, files = list_dir(cur_dir)

        # Split into folders / text files / image files
        text_exts = {".txt", ".md", ".json", ".yaml", ".yml", ".cfg", ".ini"}
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        text_files = [f for f in files if f.suffix.lower() in text_exts]
        image_files = [f for f in files if f.suffix.lower() in image_exts]

        col_dirs, col_files = st.columns(2)

        # --- Directories ---
        with col_dirs:
            st.markdown("**Folders**")
            if not dirs:
                st.caption("No subdirectories.")
            for d in dirs:
                if st.button(f"📁 {d.name}", key=f"dir_{d}"):
                    st.session_state.tinyface_current_dir = str(d)
                    st.rerun()

        # --- Text & image file pickers ---
        with col_files:
            # Text files
            st.markdown("**Text files**")
            if not text_files:
                st.caption("No text/README files in this directory.")
                selected_text_file = None
            else:
                text_options = [f.name for f in text_files]
                default_idx = 0
                # Prefer README* if present
                for i, name in enumerate(text_options):
                    if name.lower().startswith("readme"):
                        default_idx = i
                        break

                selected_text_name = st.selectbox(
                    "Select a text file to view", options=text_options, index=default_idx
                )
                selected_text_file = next(
                    (f for f in text_files if f.name == selected_text_name), None
                )

            st.markdown("---")

            # Image files
            st.markdown("**Image files**")
            if not image_files:
                st.caption("No image files in this directory.")
                selected_image_file = None
            else:
                image_options = [f.name for f in image_files]
                selected_image_name = st.selectbox(
                    "Select an image to preview", options=image_options, key="image_select"
                )
                selected_image_file = next(
                    (f for f in image_files if f.name == selected_image_name), None
                )

        st.markdown("---")

        # ----- Text file preview -----
        if selected_text_file is not None:
            st.markdown(f"### 📄 {selected_text_file.name}")

            try:
                content = selected_text_file.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                st.error(f"Could not read file: {e}")
            else:
                if selected_text_file.suffix.lower() == ".md":
                    # Strip images so Streamlit doesn't try to serve missing media
                    content_no_imgs = strip_markdown_images(content)
                    st.markdown(content_no_imgs)
                else:
                    st.code(content, language="text")

        # ----- Image preview -----
        if selected_image_file is not None:
            st.markdown(f"### 🖼 {selected_image_file.name}")
            try:
                img = load_image(str(selected_image_file))
            except Exception as e:
                st.error(f"Could not load image: {e}")
            else:
                # Bigger, consistent preview (similar to grid): stretch to container
                st.image(img, width=400)


if __name__ == "__main__":
    main()
