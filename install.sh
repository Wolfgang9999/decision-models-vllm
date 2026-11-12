#!/usr/bin/env bash
# Install without Docker into the current Python environment (Python 3.10-3.13, Linux x86_64, CUDA driver for CUDA 13).
# Then: vllm serve baselabs/joey-1 --served-model-name joey-1
set -euo pipefail
cd "$(dirname "$0")"
PIN=$(python3 -c "import json; print(json.load(open('overlay/manifest.json'))['vllm_commit'])")
# The nightly wheel of the pinned commit (same build as the Docker base image).
pip install "https://wheels.vllm.ai/${PIN}/vllm-0.34.0rc1.dev302%2Bg${PIN:0:9}-cp38-abi3-manylinux_2_28_x86_64.whl"
python3 install_overlay.py
echo "Installed. Serve with: vllm serve baselabs/joey-1 --served-model-name joey-1"
