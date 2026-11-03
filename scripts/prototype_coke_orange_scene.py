#!/usr/bin/env python3
"""Render coke+orange tabletop screenshots (GraspSingleCustomInScene-v0)."""
from __future__ import annotations

from project_paths import PROJECT_ROOT, TRAP_ROOT, SIMPLER_ENV_ROOT, MODEL_ROOT

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coke_orange_scene import (  # noqa: E402
    COKE,
    ORANGE,
    build_coke_orange_env,
    reset_coke_orange,
    set_vulkan,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-dir",
        default=f"{PROJECT_ROOT}/outputs/scene_proto/coke_orange",
    )
    args = ap.parse_args()

    set_vulkan()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    env = build_coke_orange_env(with_sponge=False)
    meta = {"seed": args.seed, "layouts": []}
    for target, tag in ((COKE, "coke"), (ORANGE, "orange")):
        obs, info = reset_coke_orange(env, designated_target=target, seed=args.seed, with_sponge=False)
        instr = env.unwrapped.get_language_instruction()
        rgb = get_image_from_maniskill2_obs_dict(env, obs)
        path = out / f"scene_target_{tag}.png"
        Image.fromarray(rgb).save(path)
        print(f"target={target} instruction={instr!r} distractors={info.get('distractor_model_ids')} -> {path}")
        meta["layouts"].append(
            {"target": target, "instruction": instr, "distractors": info.get("distractor_model_ids"), "image": str(path)}
        )
    env.close()
    (out / "proto_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done → {out}")


if __name__ == "__main__":
    main()
