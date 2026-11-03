"""InstructVLA inference using the same CoT path as the TRAP target."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from transforms3d.euler import euler2axangle


TRAP_ROOT = Path(__file__).resolve().parents[1] / "TRAP"
THIRD_PARTY = TRAP_ROOT / "third_party" / "instructvla"


class AdaptiveEnsembler:
    """Small local copy of upstream SimplerEnv's action-chunk ensemble."""

    def __init__(self, horizon: int = 7, alpha: float = 0.1):
        self.history = deque(maxlen=horizon)
        self.alpha = alpha

    def reset(self) -> None:
        self.history.clear()

    def step(self, actions: np.ndarray) -> np.ndarray:
        self.history.append(actions)
        current = np.stack(
            [chunk[i] for i, chunk in zip(range(len(self.history) - 1, -1, -1), self.history)]
        )
        reference = current[-1]
        similarity = (current @ reference) / (np.linalg.norm(current, axis=1) * np.linalg.norm(reference) + 1e-7)
        weights = np.exp(self.alpha * similarity)
        weights /= weights.sum()
        return np.sum(weights[:, None] * current, axis=0)


class InstructVLAPolicy:
    def __init__(
        self,
        checkpoint: str,
        eagle2_dir: str,
        empty_adapter: str,
        *,
        device: str = "cuda:0",
        unnorm_key: str = "fractal20220817_data",
        ensemble: bool = True,
    ) -> None:
        if str(THIRD_PARTY) not in sys.path:
            sys.path.insert(0, str(THIRD_PARTY))
        from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load_vla

        self.vla = load_vla(
            checkpoint,
            llm_backbone_id=eagle2_dir,
            empty_language_adapter_path=empty_adapter,
        )
        # Match the official SimplerEnv wrapper: bf16 VLM, fp32 action head.
        self.vla.vlm = self.vla.vlm.to(torch.bfloat16)
        self.vla = self.vla.to(device).eval()
        self.unnorm_key = unnorm_key
        # GraspSingleCustomInScene-v0 uses google_robot_static. Upstream
        # InstructVLA uses a two-chunk ensemble and relative sticky gripper.
        self.ensemble = AdaptiveEnsembler(horizon=2) if ensemble else None
        self.instruction = ""
        self.previous_gripper_action: float | None = None
        self.sticky_gripper_action = 0.0
        self.sticky_gripper_steps = 0

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.vla.last_response = None
        self.vla.run_index = 0
        self.vla.latent = None
        self.previous_gripper_action = None
        self.sticky_gripper_action = 0.0
        self.sticky_gripper_steps = 0
        if self.ensemble is not None:
            self.ensemble.reset()

    def step(self, rgb: np.ndarray) -> dict:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 uint8 RGB image, got {rgb.shape} {rgb.dtype}")
        generated_now = self.vla.last_response is None or self.vla.run_index % 20 == 0
        raw_actions, normalized_actions, _ = self.vla.predict_action(
            image=Image.fromarray(rgb),
            instruction=self.instruction,
            unnorm_key=self.unnorm_key,
            use_generate=True,
        )
        raw_actions = np.asarray(raw_actions, dtype=np.float32)
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        if raw_actions.ndim != 2 or raw_actions.shape[1] != 7 or normalized_actions.shape != raw_actions.shape:
            raise RuntimeError(f"Unexpected action chunks: raw={raw_actions.shape}, normalized={normalized_actions.shape}")
        selected = self.ensemble.step(raw_actions) if self.ensemble is not None else raw_actions[0]
        axis, angle = euler2axangle(*[float(x) for x in selected[3:6]])
        rotation = np.asarray(axis, dtype=np.float32) * float(angle)
        current_gripper = float(selected[6])
        if self.previous_gripper_action is None:
            relative_gripper = 0.0
            self.previous_gripper_action = current_gripper
        else:
            relative_gripper = self.previous_gripper_action - current_gripper
        if abs(relative_gripper) > 0.5 and self.sticky_gripper_steps == 0:
            self.sticky_gripper_action = relative_gripper
            self.sticky_gripper_steps = 10
            self.previous_gripper_action = current_gripper
        if self.sticky_gripper_steps:
            relative_gripper = self.sticky_gripper_action
            self.sticky_gripper_steps -= 1
            if self.sticky_gripper_steps == 0:
                self.sticky_gripper_action = 0.0
        gripper = np.array([relative_gripper], dtype=np.float32)
        action_vec = np.concatenate([selected[:3], rotation, gripper]).astype(np.float32)
        return {
            "action_vec": action_vec,
            "raw_actions": raw_actions,
            "normalized_actions": normalized_actions,
            "response": self.vla.last_response or "",
            "generated_now": generated_now,
        }
