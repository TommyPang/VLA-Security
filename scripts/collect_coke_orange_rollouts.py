#!/usr/bin/env python3
"""Collect coke+orange SimplerEnv rollouts for TRAP (coke → orange attack).

TRAP mapping semantics (from selection.py + MolmoActDataset):
  .pt["instruction"] = clean-rollout instruction ("pick orange")
  mapping["pick orange"] = "pick coke can"
  → optimizer prompts "pick coke can" while supervising orange CoT/actions.

So we generate MolmoAct under "pick orange" on a coke+orange table and store
instruction="pick orange".
"""
from __future__ import annotations

from project_paths import PROJECT_ROOT, TRAP_ROOT, SIMPLER_ENV_ROOT, MODEL_ROOT

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coke_orange_scene import (  # noqa: E402
    COKE,
    INSTR,
    ORANGE,
    build_coke_orange_env,
    jitter_positions,
    reset_coke_orange,
    set_vulkan,
)


def molmo_prompt(instruction: str) -> str:
    return (
        f"The task is {instruction}. "
        "What is the action that the robot should take. "
        f"To figure out the action that the robot should take to {instruction}, "
        "let's think through it step by step. "
        "First, what is the depth map for this image? "
        "Second, what is the trajectory of the end effector? "
        "Based on the depth map of the image and the trajectory of the end effector, "
        "what is the action that the robot should take?"
    )


def cot_is_usable(text: str) -> bool:
    traj = "The trajectory of the end effector is"
    action = "the action that the robot should take is"
    i = text.find(traj)
    if i < 0:
        return False
    j = text.find(" Based on", i + len(traj))
    if j < 0:
        return False
    return text.lower().find(action, j) >= 0


def parse_action_tokens(model, text: str, min_xyz_norm: float = 0.02):
    act = None
    try:
        if hasattr(model, "parse_action"):
            act = model.parse_action(text, unnorm_key="fractal20220817_data")
    except Exception as exc:
        print(f"[warn] parse_action failed: {exc}")
    if act is None:
        return torch.tensor([[0.05, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32), True
    t = torch.as_tensor(act, dtype=torch.float32)
    flat = t.reshape(-1, t.shape[-1] if t.ndim > 1 else 7)
    boosted = False
    xyz = flat[0, :3]
    n = float(torch.linalg.norm(xyz))
    if n < min_xyz_norm:
        boosted = True
        xyz = torch.tensor([0.05, 0.0, 0.02]) if n < 1e-8 else xyz / n * min_xyz_norm
        flat = flat.clone()
        flat[0, :3] = xyz
        t = flat.reshape(t.shape) if t.ndim > 1 else flat[0]
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t, boosted


def parse_trace(model, text: str):
    try:
        if hasattr(model, "parse_trace"):
            tr = model.parse_trace(text)
            if tr:
                return tr
    except Exception:
        pass
    m = re.search(r"\[\[[\s\S]*?\]\]", text)
    if not m:
        return None
    try:
        import ast

        return ast.literal_eval(m.group(0))
    except Exception:
        return None


def load_molmoact(model_path: str, device: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    mode = os.environ.get("TRAP_MOLMOACT_LOAD_MODE", "4bit_skip_vision")
    skip = ["vision_backbone", "model.vision_backbone", "lm_head"]
    if mode == "4bit_skip_vision":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=skip,
        )
    elif mode in {"8bit", "8bit_skip_vision"}:
        quant = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=skip if mode == "8bit_skip_vision" else None,
        )
    else:
        raise ValueError(f"Unsupported TRAP_MOLMOACT_LOAD_MODE={mode!r}")
    load_kwargs = dict(quantization_config=quant, device_map={"": device})
    if mode.startswith("8bit"):
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, trust_remote_code=True, **load_kwargs
    )
    model.eval()
    return model, processor


def generate_one(model, processor, image: Image.Image, instruction: str, device: str, max_new_tokens: int) -> str:
    prompt = molmo_prompt(instruction)
    text = processor.apply_chat_template(
        [{"role": "user", "content": [dict(type="text", text=prompt)]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(images=[image], text=text, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].size(1) :], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def action_to_env(action_tensor: torch.Tensor, sticky_state: dict) -> np.ndarray:
    """Rough MolmoAct → Google robot env action (same spirit as SimplerEnv policy)."""
    from transforms3d.euler import euler2axangle

    a = action_tensor.detach().cpu().numpy().reshape(-1)[:7].astype(np.float64)
    world = a[:3]
    roll, pitch, yaw = a[3:6]
    ax, ang = euler2axangle(roll, pitch, yaw)
    rot = ax * ang
    grip_open = a[6:7]
    if sticky_state["prev"] is None:
        rel = np.array([0.0])
    else:
        rel = sticky_state["prev"] - grip_open
    sticky_state["prev"] = grip_open
    if abs(float(rel[0])) > 0.5 and not sticky_state["on"]:
        sticky_state["on"] = True
        sticky_state["val"] = rel
    if sticky_state["on"]:
        sticky_state["n"] += 1
        rel = sticky_state["val"]
    if sticky_state["n"] >= 15:
        sticky_state["on"] = False
        sticky_state["n"] = 0
        sticky_state["val"] = 0.0
    return np.concatenate([world, rot, rel])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("TRAP_MOLMOACT_MODEL_PATH", ""))
    ap.add_argument("--output-root", default=os.environ.get("TRAP_MOLMOACT_DATA", f"{PROJECT_ROOT}/datasets/molmoact"))
    ap.add_argument("--env-name", default="GraspCokeOrangeScene-v0")
    ap.add_argument("--num-layouts", type=int, default=20)
    ap.add_argument("--steps-per-layout", type=int, default=12, help="Frames/labels per layout (incl. t=0)")
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--closed-loop", action="store_true", help="Step env with MolmoAct actions between frames")
    ap.add_argument("--match-eval-policy", action="store_true", help="Resize to 256x256 and use unmodified parsed actions, as in molmoact_policy.py")
    ap.add_argument("--teacher-target", choices=["orange", "coke"], default="orange", help="Object the clean teacher is asked to pick; coke enables paper-direction orange-to-coke data")
    args = ap.parse_args()
    if not args.model_path:
        raise SystemExit("Set --model-path or TRAP_MOLMOACT_MODEL_PATH")

    set_vulkan()
    designated = ORANGE if args.teacher_target == "orange" else COKE
    clean_instr = INSTR[designated]
    prompt_instr = INSTR[COKE if designated == ORANGE else ORANGE]
    attack_direction = "coke_to_orange" if designated == ORANGE else "orange_to_coke"

    print(f"Loading MolmoAct from {args.model_path} …")
    model, processor = load_molmoact(args.model_path, args.device)

    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    env = build_coke_orange_env(with_sponge=False)
    saved_total = 0
    meta_layouts = []

    for li in range(args.num_layouts):
        seed = args.seed0 + li
        positions = jitter_positions(seed)
        layout_suffix = "orange" if designated == ORANGE else "coke can"
        layout_name = f"obj_l{li:03d}_tgt_{layout_suffix}"
        out_dir = Path(args.output_root) / args.env_name / layout_name
        out_dir.mkdir(parents=True, exist_ok=True)

        obs, info = reset_coke_orange(
            env, designated_target=designated, seed=seed, with_sponge=False, positions=positions
        )
        sticky = {"prev": None, "on": False, "n": 0, "val": 0.0}
        saved = 0
        layout_success = False
        for t in range(args.steps_per_layout):
            rgb = get_image_from_maniskill2_obs_dict(env, obs)
            image = Image.fromarray(rgb).convert("RGB")
            model_image = Image.fromarray(cv.resize(rgb, (256, 256), interpolation=cv.INTER_AREA)) if args.match_eval_policy else image
            text = generate_one(model, processor, model_image, clean_instr, args.device, args.max_new_tokens)
            if not cot_is_usable(text):
                print(f"[layout {li} t={t}] unusable CoT, skip frame")
                if args.closed_loop:
                    obs, _, done, trunc, _ = env.step(env.action_space.sample() * 0.05)
                    if done or trunc:
                        break
                continue
            action, boosted = parse_action_tokens(model, text, min_xyz_norm=0.0 if args.match_eval_policy else 0.02)
            trace = parse_trace(model, text)
            stem = f"sample_{saved:04d}"
            image.save(out_dir / f"{stem}_input.png")
            payload = {
                "instruction": clean_instr,  # CRITICAL for mapping direction
                "generated_text": text,
                "action": action,
                "trace": trace,
                "target_instruction_used_for_generation": clean_instr,
                "attack_direction": attack_direction,
                "layout": layout_name,
                "step": t,
                "seed": seed,
                "positions": positions,
                "env_model_id": info.get("model_id"),
                "action_xyz_boosted": boosted,
            }
            torch.save(payload, out_dir / f"{stem}_output.pt")
            (out_dir / f"{stem}_cot.txt").write_text(text, encoding="utf-8")
            print(f"[layout {li} t={t}] saved {stem} boosted={boosted} chars={len(text)}")
            saved += 1
            saved_total += 1

            if args.closed_loop or t + 1 < args.steps_per_layout:
                if args.closed_loop:
                    env_act = action_to_env(action, sticky)
                else:
                    env_act = env.action_space.sample() * 0.08
                obs, _, done, trunc, info2 = env.step(env_act)
                info = info2 if isinstance(info2, dict) else info
                layout_success |= bool(info.get("success", False)) if isinstance(info, dict) else False
                if done or trunc:
                    break

        (out_dir / "collection_meta.json").write_text(
            json.dumps(
                {
                    "layout": layout_name,
                    "seed": seed,
                    "positions": positions,
                    "clean_instruction": clean_instr,
                    "attack_direction": attack_direction,
                    "num_samples": saved,
                    "env_success_ever": layout_success,
                    "note": f"pt.instruction={clean_instr} -> mapping prompts {prompt_instr} during TRAP opt",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        meta_layouts.append({"layout": layout_name, "samples": saved, "seed": seed, "env_success_ever": layout_success})

    env.close()
    summary = {
        "env_name": args.env_name,
        "attack_direction": attack_direction,
        "epoch_name_suffix": layout_suffix,
        "total_samples": saved_total,
        "successful_layouts": sum(bool(item["env_success_ever"]) for item in meta_layouts),
        "layouts": meta_layouts,
        "model_path": args.model_path,
        "match_eval_policy": args.match_eval_policy,
        "steps_per_layout": args.steps_per_layout,
    }
    summary_path = Path(args.output_root) / args.env_name / f"collection_summary_{attack_direction}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. total_samples={saved_total} summary={summary_path}")
    if saved_total < 4:
        raise SystemExit("Too few samples")


if __name__ == "__main__":
    main()
