#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
if [[ ! -x "$MOLMOACT_ENV/bin/python" ]]; then
    "$CONDA_ROOT/bin/conda" create -p "$MOLMOACT_ENV" python=3.11 -y
fi
python_bin="$MOLMOACT_ENV/bin/python"
"$python_bin" -m pip install --upgrade pip wheel
# SAPIEN 2.2 imports pkg_resources, which newer setuptools releases remove.
"$python_bin" -m pip install setuptools==80.9.0
# Restore the observed GPU1 package set, including CUDA 12.4 PyTorch wheels.
# Editable source packages are installed below rather than as machine-local paths.
"$python_bin" -m pip install --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -r "$VLA_SECURITY_ROOT/environments/molmoact-gpu1-pip-freeze.txt"
"$python_bin" -m pip install --no-deps -e "$TRAP_ROOT/external/molmoact"
"$python_bin" -m pip install --no-deps -e "$TRAP_ROOT"
"$python_bin" -m pip install --no-deps -e "$SIMPLER_ENV_ROOT"
"$python_bin" -m pip install --no-deps -e "$SIMPLER_ENV_ROOT/ManiSkill2_real2sim"
"$python_bin" -c 'import torch, transformers, bitsandbytes, numpy, sapien, trap; assert numpy.__version__ == "1.26.4"; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__, "bnb", bitsandbytes.__version__)'
