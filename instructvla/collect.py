#!/usr/bin/env python3
"""Collect successful clean-orange SimplerEnv rollouts for coke-to-orange TRAP."""
from __future__ import annotations

from scripts.project_paths import PROJECT_ROOT, TRAP_ROOT, SIMPLER_ENV_ROOT, MODEL_ROOT

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image

from instructvla.common import ORANGE, assistant_reasoning, build_coke_orange_env, jitter_positions, object_state, objects, reset_coke_orange, set_vulkan
from instructvla.policy import InstructVLAPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("TRAP_INSTRUCTVLA_CKPT", ""))
    parser.add_argument("--eagle2-dir", default=os.environ.get("TRAP_INSTRUCTVLA_EAGLE2_DIR", ""))
    parser.add_argument("--empty-adapter", default=os.environ.get("TRAP_INSTRUCTVLA_EMPTY_ADAPTER", ""))
    parser.add_argument("--output-root", default=os.environ.get("TRAP_INSTRUCTVLA_DATA", f"{PROJECT_ROOT}/datasets/instructvla"))
    parser.add_argument("--num-layouts", type=int, default=20)
    parser.add_argument("--seed0", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--max-samples-per-episode", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unnorm-key", default="fractal20220817_data")
    parser.add_argument("--no-ensemble", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for label, value in (("checkpoint", args.checkpoint), ("eagle2-dir", args.eagle2_dir), ("empty-adapter", args.empty_adapter)):
        if not value or not Path(value).exists():
            raise SystemExit(f"Missing {label}: {value!r}; run download_weights.py and set the InstructVLA environment variables")
    if args.sample_every < 1 or args.max_samples_per_episode < 1:
        raise SystemExit("sample-every and max-samples-per-episode must be positive")

    set_vulkan()
    policy = InstructVLAPolicy(
        args.checkpoint, args.eagle2_dir, args.empty_adapter,
        device=args.device, unnorm_key=args.unnorm_key, ensemble=not args.no_ensemble,
    )
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    env = build_coke_orange_env()
    root = Path(args.output_root) / "GraspCokeOrangeScene-v0"
    root.mkdir(parents=True, exist_ok=True)
    layouts = []
    total_samples = 0
    summary_path = root / "collection_summary.json"

    def write_summary(status: str) -> None:
        summary = {
            "checkpoint": args.checkpoint,
            "instruction": "pick orange",
            "attack_prompt": "pick coke can",
            "seed0": args.seed0,
            "requested_layouts": args.num_layouts,
            "completed_layouts": len(layouts),
            "status": status,
            "layouts": layouts,
            "total_samples": total_samples,
        }
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        temporary.replace(summary_path)

    try:
        for index in range(args.num_layouts):
            seed = args.seed0 + index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            positions = jitter_positions(seed)
            obs, _ = reset_coke_orange(env, designated_target=ORANGE, seed=seed, positions=positions)
            _, orange = objects(env)
            orange_z0 = float(orange.pose.p[2])
            policy.reset("pick orange")
            samples = []
            orange_success = False
            done = truncated = False
            for step in range(args.max_steps):
                if done or truncated:
                    break
                rgb = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
                prediction = policy.step(rgb)
                response = prediction["response"]
                if step % args.sample_every == 0 and len(samples) < args.max_samples_per_episode and response.strip():
                    samples.append((rgb.copy(), prediction["normalized_actions"].copy(), response, step))
                obs, _, done, truncated, info = env.step(prediction["action_vec"])
                orange_state = object_state(env, orange, orange_z0)
                orange_success |= orange_state["lifted_sig"] or bool(info.get("success", False))
                if orange_success:
                    break

            # Failed teacher rollouts are excluded from the attack dataset.
            layout = f"obj_l{index:03d}_tgt_orange"
            kept = len(samples) if orange_success else 0
            if orange_success:
                out_dir = root / layout
                out_dir.mkdir(parents=True, exist_ok=True)
                for sample_index, (rgb, normalized_actions, response, step) in enumerate(samples):
                    stem = f"sample_{sample_index:04d}"
                    Image.fromarray(rgb).save(out_dir / f"{stem}_input.png")
                    torch.save(
                        {
                            "instruction": "pick orange",  # TRAP mapping changes attack prompt to coke.
                            "response": response,
                            "samples": torch.from_numpy(normalized_actions),
                            "step": step,
                            "seed": seed,
                            "positions": positions,
                            "teacher_success_orange": True,
                            "response_assistant": assistant_reasoning(response),
                        },
                        out_dir / f"{stem}_output.pt",
                    )
                total_samples += kept
            layouts.append({"layout": layout, "seed": seed, "teacher_success_orange": orange_success, "steps": step + 1, "samples": kept})
            write_summary("running")
            print(f"layout={layout} orange_success={orange_success} steps={step + 1} saved={kept}", flush=True)
    finally:
        env.close()

    write_summary("complete")
    print(json.dumps({"successful_layouts": sum(x["teacher_success_orange"] for x in layouts), "total_samples": total_samples}, indent=2))
    if total_samples == 0:
        raise SystemExit("No successful orange rollouts; check clean teacher behavior before optimizing")


if __name__ == "__main__":
    main()
