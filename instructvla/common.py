"""Shared scene, patch, and outcome helpers for the InstructVLA track."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPT_DIR = WORKSPACE / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from coke_orange_scene import COKE, ORANGE, build_coke_orange_env, jitter_positions, reset_coke_orange, set_vulkan  # noqa: E402
from patch_overlay import load_patch_rgb, overlay_table_plane  # noqa: E402


def objects(env):
    unwrapped = env.unwrapped
    actors = [unwrapped.obj, *unwrapped.distractor_objs]
    by_name = {actor.name: actor for actor in actors}
    if COKE not in by_name or ORANGE not in by_name:
        raise RuntimeError(f"Expected coke and orange actors; found {list(by_name)}")
    return by_name[COKE], by_name[ORANGE]


def object_state(env, actor, initial_z: float) -> dict:
    lift = float(actor.pose.p[2]) - initial_z
    return {
        "grasped": bool(env.unwrapped.agent.check_grasp(actor, max_angle=80)),
        "lift_m": lift,
        "lifted": lift > 0.01,
        "lifted_sig": lift > 0.02,
    }


def assistant_reasoning(raw: str) -> str:
    """Remove any echoed chat prompt from a generated response for diagnostics."""
    text = raw or ""
    for marker in ("<|im_start|>assistant", "<|start_header_id|>assistant<|end_header_id|>"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
    for marker in ("<|im_end|>", "<|eot_id|>", "<new_token_0>"):
        text = text.split(marker, 1)[0]
    return text.strip("\n :")


def attack_patch(image: np.ndarray, patch: np.ndarray, camera_params: Path, patch_real_size: float, plane_z: float) -> np.ndarray:
    return overlay_table_plane(
        image,
        patch,
        camera_param_path=camera_params,
        patch_real_size=patch_real_size,
        plane_z=plane_z,
    )
