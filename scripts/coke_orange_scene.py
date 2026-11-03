"""Shared coke+orange (optional sponge) SimplerEnv layout helpers."""
from __future__ import annotations

from project_paths import PROJECT_ROOT, TRAP_ROOT, SIMPLER_ENV_ROOT, MODEL_ROOT

import os
import sys
from pathlib import Path

import numpy as np
from transforms3d.euler import euler2quat

COKE = "opened_coke_can"
ORANGE = "orange"
SPONGE = "sponge"

POSITIONS = {
    COKE: [-0.25, 0.12],
    ORANGE: [-0.18, 0.35],
    SPONGE: [-0.35, 0.28],
}
ROTATIONS = {
    COKE: euler2quat(np.pi / 2, 0, 0).tolist(),
    ORANGE: [1.0, 0.0, 0.0, 0.0],
    SPONGE: [1.0, 0.0, 0.0, 0.0],
}

INSTR = {
    COKE: "pick coke can",
    ORANGE: "pick orange",
}


def set_vulkan() -> None:
    os.environ.setdefault(
        "VK_ICD_FILENAMES",
        f"{PROJECT_ROOT}/assets/vulkan/nvidia_icd.json",
    )
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    os.environ.setdefault("DISPLAY", "")


def ensure_simpler_path() -> Path:
    root = Path(os.environ.get("SIMPLER_ENV_ROOT", f"{SIMPLER_ENV_ROOT}"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    return root


def build_coke_orange_env(with_sponge: bool = False):
    ensure_simpler_path()
    from simpler_env.utils.env.env_builder import build_maniskill2_env

    distractors = [COKE, ORANGE] + ([SPONGE] if with_sponge else [])
    return build_maniskill2_env(
        "GraspSingleCustomInScene-v0",
        obs_mode="rgbd",
        prepackaged_config=True,
        model_ids=[COKE, ORANGE],
        distractor_model_ids=distractors,
    )


def reset_coke_orange(
    env,
    *,
    designated_target: str,
    seed: int,
    with_sponge: bool = False,
    positions: dict | None = None,
):
    """Both objects on table; designated_target is what env.evaluate() scores."""
    assert designated_target in (COKE, ORANGE)
    positions = positions or POSITIONS
    other = ORANGE if designated_target == COKE else COKE
    distractors = [other] + ([SPONGE] if with_sponge else [])
    distractor_opts = {
        other: {"init_xy": positions[other], "init_rot_quat": ROTATIONS[other]}
    }
    if with_sponge:
        distractor_opts[SPONGE] = {
            "init_xy": positions[SPONGE],
            "init_rot_quat": ROTATIONS[SPONGE],
        }
    return env.reset(
        seed=seed,
        options={
            "model_id": designated_target,
            "model_scale": 1.0,
            "distractor_model_ids": distractors,
            "distractor_model_scales": [1.0] * len(distractors),
            "obj_init_options": {
                "init_xy": positions[designated_target],
                "init_rot_quat": ROTATIONS[designated_target],
            },
            "distractor_obj_init_options": distractor_opts,
        },
    )


def jitter_positions(base_seed: int) -> dict:
    rng = np.random.default_rng(base_seed)
    out = {}
    for k, xy in POSITIONS.items():
        j = rng.uniform(-0.04, 0.04, size=2)
        out[k] = [float(np.clip(xy[0] + j[0], -0.45, -0.10)), float(np.clip(xy[1] + j[1], 0.05, 0.40))]
    return out
