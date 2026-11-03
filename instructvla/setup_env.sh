#!/usr/bin/env bash
export VLA_SECURITY_ROOT="${VLA_SECURITY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
# Run only when ready to install a separate InstructVLA Python environment.
set -euo pipefail
source "$VLA_SECURITY_ROOT/scripts/paths.sh"
CONDA=${CONDA_ROOT}/bin/conda
ENV=${INSTRUCTVLA_ENV}
TRAP=${TRAP_ROOT}
SIMPLER=${SIMPLER_ENV_ROOT}

if [[ ! -x "$ENV/bin/python" ]]; then
  "$CONDA" create -p "$ENV" python=3.10 -y
fi
"$ENV/bin/python" -m pip install --upgrade pip wheel
"$ENV/bin/python" -m pip install setuptools==80.9.0
"$ENV/bin/python" -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# Install the inference/SimplerEnv subset. The upstream lockfile also includes
# training, video, and unrelated benchmark packages unavailable on this host.
"$ENV/bin/python" -m pip install \
  numpy==1.26.4 transformers==4.51.0 accelerate==1.3.0 peft==0.13.0 \
  timm==1.0.15 huggingface-hub==0.34.4 einops==0.8.0 \
  opencv-python==4.11.0.86 matplotlib==3.10.0 pyyaml tqdm rich==13.9.4 \
  transforms3d imageio sapien==2.2.2 gymnasium==0.29.1 draccus==0.8.0 \
  gdown GitPython h5py rtree scipy tabulate trimesh
"$ENV/bin/python" -m pip install ruckig==0.12.2 --only-binary=:all:
# The bundled loader selects eager attention. Flash Attention is optional for this path.
"$ENV/bin/python" -m pip install -e "$TRAP" --no-deps
"$ENV/bin/python" -m pip install -e "$SIMPLER" --no-deps
"$ENV/bin/python" -m pip install -e "$SIMPLER/ManiSkill2_real2sim" --no-deps
# Eagle2's nested vision and language configs request FlashAttention2.
"$ENV/bin/python" -m pip install \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' \
  --no-deps
"$ENV/bin/python" -m pip check
