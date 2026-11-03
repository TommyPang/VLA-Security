#!/usr/bin/env bash
# Shared locations; source before selecting a model environment.
export VLA_SECURITY_ROOT="${VLA_SECURITY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
export TRAP_ROOT="${TRAP_ROOT:-$VLA_SECURITY_ROOT/TRAP}"
export SIMPLER_ENV_ROOT="${SIMPLER_ENV_ROOT:-$VLA_SECURITY_ROOT/SimplerEnv}"
export VLA_MODEL_ROOT="${VLA_MODEL_ROOT:-$VLA_SECURITY_ROOT/models}"
export MOLMOACT_ENV="${MOLMOACT_ENV:-$CONDA_ROOT/envs/trap-molmoact}"
export INSTRUCTVLA_ENV="${INSTRUCTVLA_ENV:-$CONDA_ROOT/envs/trap-instructvla}"
