#!/usr/bin/env bash
set -e

# 1) Install PyTorch first (CUDA 12.6/ cu126)
pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch torchvision torchaudio

# 2) Install DALI (must match CUDA 12.0)
pip install --extra-index-url https://pypi.nvidia.com \
    nvidia-dali-cuda120

# 3) Now install everything that *depends* on torch
pip install -r pip_requirements.txt
