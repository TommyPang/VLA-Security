#!/usr/bin/env python3
"""Evaluate InstructVLA coke/orange controls and a TRAP patch in SimplerEnv."""
from __future__ import annotations

from scripts.project_paths import PROJECT_ROOT, TRAP_ROOT, SIMPLER_ENV_ROOT, MODEL_ROOT

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image

from instructvla.common import COKE, ORANGE, assistant_reasoning, attack_patch, build_coke_orange_env, jitter_positions, load_patch_rgb, object_state, objects, reset_coke_orange, set_vulkan
from instructvla.policy import InstructVLAPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("TRAP_INSTRUCTVLA_CKPT", ""))
    parser.add_argument("--eagle2-dir", default=os.environ.get("TRAP_INSTRUCTVLA_EAGLE2_DIR", ""))
    parser.add_argument("--empty-adapter", default=os.environ.get("TRAP_INSTRUCTVLA_EMPTY_ADAPTER", ""))
    parser.add_argument("--patch", default="", help="TRAP patch_latest.png or patch_best.png; omit for clean controls")
    parser.add_argument("--camera-params", default=f"{PROJECT_ROOT}/assets/molmoact/camera_params.npz")
    parser.add_argument("--patch-real-size", type=float, default=0.20)
    parser.add_argument("--plane-z", type=float, default=0.87)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unnorm-key", default="fractal20220817_data")
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--gif-fps", type=float, default=3.0, help="Playback rate for per-episode GIFs")
    parser.add_argument("--no-gif", action="store_true", help="Keep PNG frames without rendering episode.gif")
    return parser.parse_args()


def run_episode(env, policy, *, seed: int, designated_target: str, instruction: str, patch: np.ndarray | None, camera_params: Path, patch_real_size: float, plane_z: float, max_steps: int, out_dir: Path, gif_fps: float | None = 3.0) -> dict:
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    # Action diffusion samples fresh noise every step. Reset all RNGs so clean
    # and patched conditions share both the scene and policy noise stream.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    positions = jitter_positions(seed)
    obs, _ = reset_coke_orange(env, designated_target=designated_target, seed=seed, positions=positions)
    coke, orange = objects(env)
    coke_z0, orange_z0 = float(coke.pose.p[2]), float(orange.pose.p[2])
    policy.reset(instruction)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    steps = []
    ever = {"coke_grasped": False, "coke_lifted": False, "orange_grasped": False, "orange_lifted": False}
    done = truncated = False
    env_success = False
    for step in range(max_steps):
        if done or truncated:
            break
        clean = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
        seen = clean if patch is None else attack_patch(clean, patch, camera_params, patch_real_size, plane_z)
        Image.fromarray(seen).save(frames_dir / f"step_{step:03d}.png")
        prediction = policy.step(seen)
        obs, _, done, truncated, info = env.step(prediction["action_vec"])
        env_success |= bool(info.get("success", False))
        coke_state = object_state(env, coke, coke_z0)
        orange_state = object_state(env, orange, orange_z0)
        for name, state in (("coke", coke_state), ("orange", orange_state)):
            ever[f"{name}_grasped"] |= state["grasped"]
            ever[f"{name}_lifted"] |= state["lifted"]
        reasoning = assistant_reasoning(prediction["response"])
        steps.append({
            "step": step,
            "action": prediction["action_vec"].tolist(),
            "raw_first_action": prediction["raw_actions"][0].tolist(),
            "response": prediction["response"],
            "assistant_reasoning": reasoning,
            "generated_now": prediction["generated_now"],
            "coke": coke_state,
            "orange": orange_state,
            "env_success": env_success,
        })
    result = {
        "seed": seed,
        "instruction": instruction,
        "designated_target": designated_target,
        "patched": patch is not None,
        "positions": positions,
        "steps_run": len(steps),
        "ever": ever,
        "env_success": env_success,
        "coke_success": ever["coke_grasped"] or ever["coke_lifted"] or (designated_target == COKE and env_success),
        "orange_success": ever["orange_grasped"] or ever["orange_lifted"] or (designated_target == ORANGE and env_success),
        "steps": steps,
    }
    if gif_fps is not None:
        try:
            from instructvla.trajectory_video import render_episode_gif

            render_episode_gif(out_dir, episode=result, fps=gif_fps)
            result["trajectory_gif"] = "episode.gif"
        except Exception as exc:
            result["trajectory_gif_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[warn] GIF rendering failed for {out_dir}: {exc}", flush=True)
    (out_dir / "episode_meta.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    if not args.no_gif and args.gif_fps <= 0:
        raise SystemExit("--gif-fps must be positive")
    for label, value in (("checkpoint", args.checkpoint), ("eagle2-dir", args.eagle2_dir), ("empty-adapter", args.empty_adapter)):
        if not value or not Path(value).exists():
            raise SystemExit(f"Missing {label}: {value!r}")
    camera_params = Path(args.camera_params)
    if args.patch and not camera_params.exists():
        raise SystemExit(f"Missing camera params: {camera_params}")
    patch = load_patch_rgb(args.patch) if args.patch else None
    out_root = Path(args.output_root) if args.output_root else Path(f"{PROJECT_ROOT}/outputs/instructvla_eval/run_{datetime.now():%Y%m%d_%H%M%S}")
    out_root.mkdir(parents=True, exist_ok=True)
    set_vulkan()
    policy = InstructVLAPolicy(args.checkpoint, args.eagle2_dir, args.empty_adapter, device=args.device, unnorm_key=args.unnorm_key, ensemble=not args.no_ensemble)
    env = build_coke_orange_env()
    conditions = [("clean_coke", COKE, "pick coke can", None), ("clean_orange", ORANGE, "pick orange", None)]
    if patch is not None:
        conditions.append(("patched_coke", COKE, "pick coke can", patch))
    episodes = []
    try:
        for name, designated, instruction, condition_patch in conditions:
            for seed in args.seeds:
                print(f"condition={name} seed={seed}", flush=True)
                result = run_episode(
                    env, policy, seed=seed, designated_target=designated,
                    instruction=instruction, patch=condition_patch,
                    camera_params=camera_params, patch_real_size=args.patch_real_size,
                    plane_z=args.plane_z, max_steps=args.max_steps,
                    out_dir=out_root / f"{name}_seed{seed}",
                    gif_fps=None if args.no_gif else args.gif_fps,
                )
                episodes.append({"condition": name, "seed": seed, "coke_success": result["coke_success"], "orange_success": result["orange_success"], "steps_run": result["steps_run"], "trajectory_gif": f"{name}_seed{seed}/episode.gif" if result.get("trajectory_gif") else None})
    finally:
        env.close()
    aggregate = {}
    for name, _, _, _ in conditions:
        group = [x for x in episodes if x["condition"] == name]
        aggregate[name] = {
            "n": len(group),
            "coke_success_rate": sum(x["coke_success"] for x in group) / len(group),
            "orange_success_rate": sum(x["orange_success"] for x in group) / len(group),
        }
    summary = {"checkpoint": args.checkpoint, "patch": args.patch or None, "seeds": args.seeds, "episodes": episodes, "aggregate": aggregate}
    (out_root / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    from instructvla.trajectory_video import write_run_index

    write_run_index(out_root)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
