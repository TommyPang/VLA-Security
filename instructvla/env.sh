#!/usr/bin/env bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../scripts" && pwd)/paths.sh"
if [[ -x "$INSTRUCTVLA_ENV/bin/python" ]]; then
    export PATH="$INSTRUCTVLA_ENV/bin:$PATH"
fi
export TRAP_INSTRUCTVLA_CKPT="${TRAP_INSTRUCTVLA_CKPT:-$VLA_MODEL_ROOT/instructvla_stage2/checkpoints/step-013500-epoch-01-loss=0.1093.pt}"
export TRAP_INSTRUCTVLA_EAGLE2_DIR="${TRAP_INSTRUCTVLA_EAGLE2_DIR:-$VLA_MODEL_ROOT/InstructVLA_Assets/Eagle2-2B}"
export TRAP_INSTRUCTVLA_EMPTY_ADAPTER="${TRAP_INSTRUCTVLA_EMPTY_ADAPTER:-$VLA_MODEL_ROOT/InstructVLA_Assets/empty_language_adapter}"
export TRAP_INSTRUCTVLA_DATA="${TRAP_INSTRUCTVLA_DATA:-$VLA_SECURITY_ROOT/datasets/instructvla}"
export PYTHONPATH="$VLA_SECURITY_ROOT:$VLA_SECURITY_ROOT/scripts:$TRAP_ROOT/src:$TRAP_ROOT/third_party/instructvla:$SIMPLER_ENV_ROOT:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$VLA_SECURITY_ROOT/assets/vulkan/nvidia_icd.json}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export DISPLAY="${DISPLAY:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
