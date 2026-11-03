#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
if [[ -x "$MOLMOACT_ENV/bin/python" ]]; then
    export PATH="$MOLMOACT_ENV/bin:$PATH"
fi
export TRAP_MOLMOACT_MODEL_PATH="${TRAP_MOLMOACT_MODEL_PATH:-$VLA_MODEL_ROOT/MolmoAct-7B-D-Pretrain-RT-1-0812}"
export TRAP_MOLMOACT_DATA="${TRAP_MOLMOACT_DATA:-$VLA_SECURITY_ROOT/datasets/molmoact}"
export PAPER_DATA_ROOT="${PAPER_DATA_ROOT:-$VLA_SECURITY_ROOT/datasets/molmoact_gpu1_paper_direction_20261030}"
export PAPER_MASK_ROOT="${PAPER_MASK_ROOT:-${PAPER_DATA_ROOT}_table_masks}"
export PYTHONPATH="$VLA_SECURITY_ROOT:$VLA_SECURITY_ROOT/scripts:$TRAP_ROOT/src:$SIMPLER_ENV_ROOT:${PYTHONPATH:-}"
export TRAP_MOLMOACT_LOAD_MODE="${TRAP_MOLMOACT_LOAD_MODE:-8bit_skip_vision}"
export TRAP_MOLMOACT_MAX_CROPS="${TRAP_MOLMOACT_MAX_CROPS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$VLA_SECURITY_ROOT/assets/vulkan/nvidia_icd.json}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export DISPLAY="${DISPLAY:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
