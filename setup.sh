#!/usr/bin/env bash
#
# Reproduce the local AI-weather (NVIDIA earth2studio) environment on another machine.
#
# Target:   Linux + NVIDIA GPU. Built/tested on an RTX 5080 (Blackwell, sm_120).
# Requires: - `conda` on PATH (Miniconda/Anaconda)
#           - a recent NVIDIA driver (>= 570 for Blackwell; >= 550 for CUDA 12.8 in general)
#
# Usage:    bash setup.sh [env_name]            # default env name: earth2
#           CUDA_IDX=https://download.pytorch.org/whl/cu124 bash setup.sh   # other CUDA
#
# Why a script and not requirements.txt: the install ORDER and per-package INDEX URLs
# matter (cu128 wheels, a torchvision re-pin, and swapping cupy/cucim from CUDA-13 to
# CUDA-12). See requirements-freeze.txt for an exact version lock of the working env.

set -euo pipefail

ENV_NAME="${1:-earth2}"
PY_VER="3.12"
TORCH_VER="2.11.0"
TV_VER="0.26.0"
# CUDA 12.8 wheels (needed for Blackwell sm_120). These wheels also include sm_80/86/90,
# so they run on Ampere/Ada/Hopper too. Override CUDA_IDX for a different CUDA if desired.
CUDA_IDX="${CUDA_IDX:-https://download.pytorch.org/whl/cu128}"
# Pin earth2studio to the exact commit this project was built against.
E2S="earth2studio[cyclone,fcn,aurora,pangu] @ git+https://github.com/NVIDIA/earth2studio.git@41b75fb0dd7ea753b4ca49125b3345f5cbf5c64a"

command -v conda >/dev/null || { echo "ERROR: 'conda' not on PATH. Install Miniconda first."; exit 1; }
command -v nvidia-smi >/dev/null || echo "WARN: nvidia-smi not found — is the NVIDIA driver installed?"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "ERROR: conda env '$ENV_NAME' already exists. Remove it first:"
  echo "         conda remove -n $ENV_NAME --all -y"
  exit 1
fi

echo "==> Creating conda env '$ENV_NAME' (Python $PY_VER)"
conda create -y -n "$ENV_NAME" "python=$PY_VER"

# Resolve the env's interpreter without needing 'conda activate' inside the script.
PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
pipi() { "$PY" -m pip install "$@"; }

pipi --upgrade pip

echo "==> [1/4] PyTorch + torchvision from the CUDA index (MUST be installed first)"
pipi "torch==$TORCH_VER" "torchvision==$TV_VER" --index-url "$CUDA_IDX"

echo "==> [2/4] earth2studio (cyclone tracker + FCN + Aurora + Pangu) + cartopy + accelerate"
pipi "$E2S" cartopy accelerate

echo "==> [3/4] Re-pin torchvision to the CUDA build (model extras can pull a CPU/mismatched one)"
pipi "torchvision==$TV_VER" --index-url "$CUDA_IDX" --force-reinstall --no-deps

echo "==> [4/4] Swap cupy/cucim to CUDA-12 builds (the 'cyclone' extra pulls CUDA-13 ones)"
"$PY" -m pip uninstall -y cupy-cuda13x cucim-cu13 nvidia-nvimgcodec-cu13 || true
pipi cupy-cuda12x cucim-cu12

echo "==> Verifying GPU + all imports"
"$PY" - <<'PYCODE'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| arch", torch.cuda.get_arch_list() if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), "CUDA not available — check the NVIDIA driver / CUDA_IDX"
a = torch.randn(2048, 2048, device="cuda"); float((a @ a).sum())   # run a real GPU kernel
from earth2studio.models.px import FCN, Pangu6, Aurora
from earth2studio.models.dx import TCTrackerWuDuan
from earth2studio.data import GFS, ARCO
import cupy, cucim, onnxruntime
print("ONNX providers:", onnxruntime.get_available_providers())
print("OK: earth2studio + FCN/Pangu/Aurora + tracker + cupy/cucim + onnxruntime all import")
PYCODE

cat <<EOF

✅ Done. Environment '$ENV_NAME' is ready.

Run a forecast:
    conda run -n $ENV_NAME python cyclone_track.py
  or:
    conda activate $ENV_NAME && python cyclone_track.py

Notes:
  - 'sfno' is intentionally omitted: it needs NVIDIA 'makani', which isn't on PyPI
    (install from git: pip install "git+https://github.com/NVIDIA/makani.git").
  - On a 16 GB GPU, Pangu runs on CPU and Aurora won't fit — see README.md.
EOF
