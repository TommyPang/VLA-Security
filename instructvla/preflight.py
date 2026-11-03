#!/usr/bin/env python3
"""CPU-only check of checkpoint layout, scene assets, and attack data."""
from __future__ import annotations

from scripts.project_paths import PROJECT_ROOT, TRAP_ROOT, SIMPLER_ENV_ROOT, MODEL_ROOT

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("TRAP_INSTRUCTVLA_CKPT", f"{MODEL_ROOT}/instructvla_stage2/checkpoints/step-013500-epoch-01-loss=0.1093.pt")))
    parser.add_argument("--eagle2-dir", type=Path, default=Path(os.environ.get("TRAP_INSTRUCTVLA_EAGLE2_DIR", f"{MODEL_ROOT}/InstructVLA_Assets/Eagle2-2B")))
    parser.add_argument("--empty-adapter", type=Path, default=Path(os.environ.get("TRAP_INSTRUCTVLA_EMPTY_ADAPTER", f"{MODEL_ROOT}/InstructVLA_Assets/empty_language_adapter")))
    parser.add_argument("--dataset", type=Path, default=Path(os.environ.get("TRAP_INSTRUCTVLA_DATA", f"{PROJECT_ROOT}/datasets/instructvla")))
    parser.add_argument("--camera-params", type=Path, default=Path(f"{PROJECT_ROOT}/assets/molmoact/camera_params.npz"))
    args = parser.parse_args()

    checks = {
        "checkpoint": args.checkpoint.is_file(),
        "model_config": (args.checkpoint.parent.parent / "config.json").is_file(),
        "dataset_statistics": (args.checkpoint.parent.parent / "dataset_statistics.json").is_file(),
        "eagle_model": (args.eagle2_dir / "model.safetensors").is_file(),
        "eagle_code": (args.eagle2_dir / "modeling_eagle_chat.py").is_file(),
        "empty_adapter": (args.empty_adapter / "adapter_model.safetensors").is_file(),
        "camera_params": args.camera_params.is_file(),
    }
    config_path = args.checkpoint.parent.parent / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        checks["stage2_config"] = config.get("stage") == "stage2"
        checks["seven_actions"] = config.get("action_dim") == 7
    stats_path = args.checkpoint.parent.parent / "dataset_statistics.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        checks["bridge_stats"] = "bridge_dataset" in stats
        checks["google_robot_stats"] = "fractal20220817_data" in stats
    pairs = list((args.dataset / "GraspCokeOrangeScene-v0").glob("obj_*_tgt_orange/sample_*_input.png"))
    print(json.dumps({"checks": checks, "attack_pairs": len(pairs)}, indent=2))
    if not all(checks.values()):
        raise SystemExit("InstructVLA assets are incomplete; see failed checks above")


if __name__ == "__main__":
    main()
