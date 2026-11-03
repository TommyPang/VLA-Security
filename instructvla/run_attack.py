#!/usr/bin/env python3
"""Launch TRAP with InstructVLA-specific memory and patch-placement settings."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import trap.cli.attack as attack_cli
from trap.attacks import PatchAttackRunner
from trap.cli.attack import main as trap_main
from trap.core.logging import save_patch_image
from trap.losses import trajectory_action_loss
from trap.models.instructvla import InstructVLAAdapter
from trap.patch import TablePlanePatchApplier
from trap.paths import THIRD_PARTY_ROOT
from trap.utils import prepend_sys_path
import trap.attacks.runner as runner_module
import trap.models.instructvla as instructvla_module


class CokeOrangeInstructVLAAdapter(InstructVLAAdapter):
    def load(self) -> None:
        with prepend_sys_path(THIRD_PARTY_ROOT / "instructvla"):
            from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load_vla

            self.vla = load_vla(
                self.config.model["path"],
                llm_backbone_id=self.config.model["llm_backbone_id"],
                empty_language_adapter_path=self.config.model["empty_language_adapter_path"],
            )
        # The official SimplerEnv policy casts only the VLM to bf16.
        self.vla.vlm = self.vla.vlm.to(torch.bfloat16)
        self.vla = self.vla.to(self.device).eval()
        self.freeze_module(self.vla)

        attack = self.config.attack
        camera = Path(attack["camera_param_path"])
        if not camera.exists():
            raise FileNotFoundError(f"Camera calibration is required: {camera}")
        self.patch_applier = TablePlanePatchApplier(
            camera,
            patch_real_size=float(attack.get("patch_real_size", 0.20)),
            plane_z=float(attack.get("plane_z", 0.87)),
        )
        self.patch_applier._homography(*attack.get("patch", {}).get("size", [3, 128, 128])[-2:])
        self.setup_regularizers()

    def preview_step(self, batch, patch: torch.Tensor, step: int, output_dir: str) -> None:
        # The standard preview adds a second VLM+diffusion forward per save step.
        # Closed-loop evaluate.py records the generated CoT and action instead.
        image = batch["images"][0].to(self.device)
        composite = self.patch_applier.apply(image, patch)
        save_patch_image(composite, Path(output_dir) / f"image_step_{step:06d}_composite.png")


def build_adapter(config, device):
    if config.model_name != "instructvla":
        raise ValueError("This launcher only supports InstructVLA")
    return CokeOrangeInstructVLAAdapter(config, device)


def fp32_action_loss(predicted, target):
    # The Stage-2 action model samples in bf16, while rollout targets are fp32.
    # PyTorch's MSE backward expects matching dtypes for both operands.
    return trajectory_action_loss(predicted.float(), target.float())


class ResumePatchAttackRunner(PatchAttackRunner):
    def __init__(self, config):
        cot_weight = os.environ.get("TRAP_INSTRUCTVLA_COT_WEIGHT")
        if cot_weight is not None:
            config.attack["cot_weight"] = float(cot_weight)
        super().__init__(config)
        resume_path = os.environ.get("TRAP_INSTRUCTVLA_RESUME_PATCH")
        if not resume_path:
            return
        if self.patch_param.mode != "pixel":
            raise ValueError("Patch resume requires pixel mode")
        path = Path(resume_path)
        with Image.open(path) as image:
            height, width = self.patch_param.patch.shape[-2:]
            rgb = np.asarray(image.convert("RGB").resize((width, height)), dtype=np.uint8).copy()
        initial = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device, dtype=self.patch_param.patch.dtype) / 255.0
        with torch.no_grad():
            self.patch_param.patch.copy_(initial)
        (self.output_dir / "resume_patch.txt").write_text(str(path.resolve()) + "\n", encoding="utf-8")


def main() -> None:
    if os.environ.get("TRAP_INSTRUCTVLA_DETECT_ANOMALY") == "1":
        torch.autograd.set_detect_anomaly(True)
    runner_module.build_adapter = build_adapter
    instructvla_module.trajectory_action_loss = fp32_action_loss
    attack_cli.PatchAttackRunner = ResumePatchAttackRunner
    trap_main()


if __name__ == "__main__":
    main()
