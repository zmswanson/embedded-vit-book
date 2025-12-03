SHELL := /bin/bash

ENV_NAME := vit-benchmark
ENV_FILE := environment.yml
PIP_SCRIPT := pip_installs.sh

.PHONY: env create-env post-install verify clean

# Top-level target
env: create-env post-install verify

# Create or update the conda environment from environment.yml
create-env:
	conda env list | grep -q '$(ENV_NAME)' || conda env create -f $(ENV_FILE)
	conda env update -n $(ENV_NAME) -f $(ENV_FILE) --prune

# Run pip_installs.sh inside the environment
post-install:
	conda run -n $(ENV_NAME) bash $(PIP_SCRIPT)

# Quick sanity check
verify:
	conda run -n $(ENV_NAME) python -c "import torch, lightning as L; \
print('Torch:', torch.__version__); \
print('Lightning:', L.__version__); \
print('CUDA available:', torch.cuda.is_available()); \
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# Remove environment
clean:
	@if [ "$$CONDA_DEFAULT_ENV" = "$(ENV_NAME)" ]; then \
		echo "Environment '$(ENV_NAME)' is currently active in this shell."; \
		echo "Please run 'conda deactivate' and then re-run 'make -C install clean'."; \
		exit 1; \
	fi; \
	conda env remove -n $(ENV_NAME) || true
	echo "Environment '$(ENV_NAME)' removed (or did not exist)."

