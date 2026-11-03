"""Portable defaults shared by the experiment entry points."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("VLA_SECURITY_ROOT", Path(__file__).resolve().parents[1]))
TRAP_ROOT = Path(os.environ.get("TRAP_ROOT", PROJECT_ROOT / "TRAP"))
SIMPLER_ENV_ROOT = Path(os.environ.get("SIMPLER_ENV_ROOT", PROJECT_ROOT / "SimplerEnv"))
MODEL_ROOT = Path(os.environ.get("VLA_MODEL_ROOT", PROJECT_ROOT / "models"))
CONDA_ROOT = Path(os.environ.get("CONDA_ROOT", Path.home() / "miniconda3"))
INSTRUCTVLA_ENV = Path(os.environ.get("INSTRUCTVLA_ENV", CONDA_ROOT / "envs/trap-instructvla"))
