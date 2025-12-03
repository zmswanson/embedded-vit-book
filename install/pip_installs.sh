#!/usr/bin/env bash
set -e

# 1) Install PyTorch first (CUDA 13.0 / cu130)
pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision torchaudio

# 2) Install DALI (must match CUDA 13.0)
pip install --extra-index-url https://pypi.nvidia.com \
    nvidia-dali-cuda130

# 3) Now install everything that *depends* on torch
pip install -r pip_requirements.txt
