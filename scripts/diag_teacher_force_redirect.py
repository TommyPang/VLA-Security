#!/usr/bin/env python3
"""Diagnose why coke→orange patch fails: AR vs teacher-forced traj, eval vs opt vision.

For each frame (clean + table-plane patched):
  A) Autoregressive generate (eval path: 256 resize + HF processor) — real closed-loop path
  B) Teacher-force GT orange trajectory tokens, then generate action only
  C) Opt-style generate (640×512 + preprocess_torch max_crops=1) — matches TRAP training vision

If B flips toward orange actions but A does not → teacher-forcing gap.
If C redirects but A does not → opt/eval vision mismatch.
If neither redirects → patch too weak / not learned.
"""
from __future__ import annotations

from project_paths import PROJECT_ROOT, TRAP_ROOT, SIMPLER_ENV_ROOT, MODEL_ROOT

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from PIL import Image


def extract_traj(text: str) -> np.ndarray | None:
    m = re.search(r"trajectory of the end effector is\s*(\[\[.*?\]\])", text, re.I | re.S)
    if not m:
        return None
    try:
        return np.asarray(ast.literal_eval(m.group(1)), dtype=float)
    except Exception:
        return None


def mean_dist(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    return float(np.linalg.norm(a[:n] - b[:n], axis=1).mean())


def travel(traj: np.ndarray | None) -> float | None:
    if traj is None or len(traj) < 2:
        return None
    return float(np.linalg.norm(traj[-1] - traj[0]))


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


def load_model(model_path: str, device: str):
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


def generate_eval_path(
    model,
    processor,
    image_rgb: np.ndarray,
    instruction: str,
    device: str,
    max_new_tokens: int,
    *,
    force_traj_text: str | None = None,
) -> dict:
    """Closed-loop-like path: 256×256 + HF processor.generate."""
    image = cv.resize(image_rgb, (256, 256), interpolation=cv.INTER_AREA)
    img = Image.fromarray(image)
    prompt = molmo_prompt(instruction)
    text = processor.apply_chat_template(
        [{"role": "user", "content": [dict(type="text", text=prompt)]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(images=[img], text=text, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    if force_traj_text is None:
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, prompt_len:]
        decoded = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        mode = "AR_eval256"
    else:
        # Prefill through end of trajectory marker + GT traj, then continue for action.
        # Use the model's own depth from a short AR pass? Simpler: splice GT traj into a
        # partial generation by concatenating tokens after depth+traj prefix from GT CoT.
        # We force: generate depth AR first (or use GT depth from orange .pt), then GT traj, then action.
        # Practical approach matching training: feed text up through traj, generate the rest.
        force_ids = processor.tokenizer(
            force_traj_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        pref = torch.cat([inputs["input_ids"], force_ids], dim=1)
        attn = torch.ones_like(pref)
        gen_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        gen_inputs["input_ids"] = pref
        gen_inputs["attention_mask"] = attn
        with torch.inference_mode():
            out = model.generate(**gen_inputs, max_new_tokens=max(64, max_new_tokens // 4), do_sample=False)
        # Full decode from after original prompt (includes forced traj + new action)
        gen = out[:, prompt_len:]
        decoded = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        mode = "TF_traj_then_action_eval256"

    traj = extract_traj(decoded)
    action = None
    parse_ok = False
    try:
        acts = model.parse_action(decoded, unnorm_key="fractal20220817_data")
        if acts:
            action = np.asarray(acts[0], dtype=float).reshape(-1)[:7]
            parse_ok = True
    except Exception as exc:
        action = None
        parse_err = str(exc)
    else:
        parse_err = None

    return {
        "mode": mode,
        "text": decoded,
        "traj": None if traj is None else traj.tolist(),
        "traj_end": None if traj is None else traj[-1].tolist(),
        "traj_travel": travel(traj),
        "parse_ok": parse_ok,
        "parse_err": parse_err,
        "action": None if action is None else action.tolist(),
        "xyz_norm": None if action is None else float(np.linalg.norm(action[:3])),
        "cot_tail": decoded[-280:],
    }


def generate_opt_path(
    model,
    processor,
    image_rgb: np.ndarray,
    instruction: str,
    device: str,
    max_new_tokens: int,
    *,
    resize_hw: tuple[int, int] | None = (256, 256),
) -> dict:
    """TRAP opt vision: preprocess_torch(max_crops=1), optionally after eval-aligned resize."""
    from trap.data.molmoact import get_image_tokens, molmo_prompt as trap_prompt
    from trap.models.molmoact_utils import preprocess_torch

    # image_rgb HxWx3 uint8 → 1xCxHxW float [0,1]
    rgb = image_rgb
    if resize_hw is not None:
        rgb = cv.resize(image_rgb, (resize_hw[1], resize_hw[0]), interpolation=cv.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1) / 255.0
    t = t.unsqueeze(0).to(device)
    image_input = preprocess_torch(t, max_crops=int(os.environ.get("TRAP_MOLMOACT_MAX_CROPS", "1")))
    image_tokens = get_image_tokens(image_input["image_grids"][0])
    prompt_text = processor.apply_chat_template(
        [{"role": "user", "content": [dict(type="text", text=trap_prompt(instruction))]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = processor.tokenizer(
        image_tokens + prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            images=torch.stack(image_input["images"]).to(device),
            pooled_patches_idx=torch.stack(image_input["pooled_patches_idx"]).to(device),
            image_masks=torch.stack(image_input["image_masks"]).to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    decoded = processor.batch_decode(
        generated[:, prompt_ids.size(1) :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    traj = extract_traj(decoded)
    action = None
    parse_ok = False
    try:
        acts = model.parse_action(decoded, unnorm_key="fractal20220817_data")
        if acts:
            action = np.asarray(acts[0], dtype=float).reshape(-1)[:7]
            parse_ok = True
    except Exception:
        pass
    return {
        "mode": f"AR_opt_preprocess_resize{resize_hw}",
        "text": decoded,
        "traj": None if traj is None else traj.tolist(),
        "traj_end": None if traj is None else traj[-1].tolist(),
        "traj_travel": travel(traj),
        "parse_ok": parse_ok,
        "action": None if action is None else action.tolist(),
        "xyz_norm": None if action is None else float(np.linalg.norm(action[:3])),
        "cot_tail": decoded[-280:],
    }


def force_prefix_from_orange_pt(pt: dict) -> str:
    """Build forced prefix: GT depth + trajectory sentence content (matches training teacher force of traj)."""
    text = pt["generated_text"]
    traj_marker = "The trajectory of the end effector is"
    i = text.find(traj_marker)
    if i < 0:
        raise ValueError("no traj marker in orange CoT")
    j = text.find(" Based on", i)
    if j < 0:
        raise ValueError("no Based on after traj")
    # Include depth + traj up to (but not including) " Based on..." so model continues with action.
    # generated_text usually starts with a leading space after BOS decode.
    return text[:j]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("TRAP_MOLMOACT_MODEL_PATH", ""))
    ap.add_argument(
        "--patch",
        default=f"{PROJECT_ROOT}/outputs/molmoact/GraspCokeOrangeScene-v0/coke_to_orange/run_20261021_211630/patch_best.png",
    )
    ap.add_argument("--patch-real-size", type=float, default=0.20)
    ap.add_argument("--camera-params", default=f"{PROJECT_ROOT}/assets/molmoact/camera_params.npz")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    ap.add_argument(
        "--dataset-root",
        default=f"{PROJECT_ROOT}/datasets/molmoact/GraspCokeOrangeScene-v0",
    )
    ap.add_argument(
        "--out",
        default=f"{PROJECT_ROOT}/outputs/offline_redirect/diag_teacher_force.json",
    )
    args = ap.parse_args()
    if not args.model_path:
        raise SystemExit("Need TRAP_MOLMOACT_MODEL_PATH")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from patch_overlay import load_patch_rgb, overlay_table_plane

    instruction = "pick coke can"
    patch = load_patch_rgb(args.patch, size=128)
    print(f"Loading model on {args.device} …")
    model, processor = load_model(args.model_path, args.device)

    results = []
    for seed in args.seeds:
        # layouts: seed0=100 → obj_l000; also resolve by pt["seed"] if needed
        layout = Path(args.dataset_root) / f"obj_l{seed - 100:03d}_tgt_orange"
        if not layout.exists():
            layout = None
            for cand in sorted(Path(args.dataset_root).glob("obj_*_tgt_orange")):
                pt0 = cand / "sample_0000_output.pt"
                if not pt0.exists():
                    continue
                d0 = torch.load(pt0, map_location="cpu", weights_only=False)
                if int(d0.get("seed", -1)) == seed:
                    layout = cand
                    break
        if layout is None:
            print(f"[skip] no layout for seed {seed}")
            continue

        img_path = layout / "sample_0000_input.png"
        pt_path = layout / "sample_0000_output.pt"
        clean = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        pt = torch.load(pt_path, map_location="cpu", weights_only=False)
        orange_traj = np.asarray(pt["trace"][0], dtype=float) if pt.get("trace") else None
        orange_action = np.asarray(pt["action"], dtype=float).reshape(-1)[:7]
        force_text = force_prefix_from_orange_pt(pt)
        patched = overlay_table_plane(
            clean,
            patch,
            camera_param_path=args.camera_params,
            patch_real_size=args.patch_real_size,
        )

        row = {
            "seed": seed,
            "layout": str(layout),
            "orange_gt_traj_end": None if orange_traj is None else orange_traj[-1].tolist(),
            "orange_gt_action": orange_action.tolist(),
            "orange_gt_xyz_norm": float(np.linalg.norm(orange_action[:3])),
            "conditions": {},
        }
        for cond_name, img in [("clean", clean), ("patched", patched)]:
            print(f"\n=== seed{seed} {cond_name} ===")
            ar = generate_eval_path(
                model, processor, img, instruction, args.device, args.max_new_tokens
            )
            ar["dist_to_orange_gt"] = mean_dist(
                None if ar["traj"] is None else np.asarray(ar["traj"]), orange_traj
            )
            if ar["action"] is not None:
                ar["action_l2_vs_orange_gt"] = float(
                    np.linalg.norm(np.asarray(ar["action"]) - orange_action)
                )
            print(
                f"  A AR_eval256: end={ar['traj_end']} dist_orange={ar['dist_to_orange_gt']} "
                f"xyz_n={ar['xyz_norm']} act_l2_orange={ar.get('action_l2_vs_orange_gt')}"
            )
            print(f"    tail: …{ar['cot_tail'][-160:]}")

            tf = generate_eval_path(
                model,
                processor,
                img,
                instruction,
                args.device,
                args.max_new_tokens,
                force_traj_text=force_text,
            )
            tf["dist_to_orange_gt"] = mean_dist(
                None if tf["traj"] is None else np.asarray(tf["traj"]), orange_traj
            )
            if tf["action"] is not None:
                tf["action_l2_vs_orange_gt"] = float(
                    np.linalg.norm(np.asarray(tf["action"]) - orange_action)
                )
            print(
                f"  B TF_traj: end={tf['traj_end']} dist_orange={tf['dist_to_orange_gt']} "
                f"xyz_n={tf['xyz_norm']} parse={tf['parse_ok']} "
                f"act_l2_orange={tf.get('action_l2_vs_orange_gt')}"
            )
            print(f"    tail: …{tf['cot_tail'][-160:]}")

            opt = generate_opt_path(
                model, processor, img, instruction, args.device, args.max_new_tokens
            )
            opt["dist_to_orange_gt"] = mean_dist(
                None if opt["traj"] is None else np.asarray(opt["traj"]), orange_traj
            )
            if opt["action"] is not None:
                opt["action_l2_vs_orange_gt"] = float(
                    np.linalg.norm(np.asarray(opt["action"]) - orange_action)
                )
            print(
                f"  C AR_opt: end={opt['traj_end']} dist_orange={opt['dist_to_orange_gt']} "
                f"xyz_n={opt['xyz_norm']} act_l2_orange={opt.get('action_l2_vs_orange_gt')}"
            )
            print(f"    tail: …{opt['cot_tail'][-160:]}")

            row["conditions"][cond_name] = {"A_AR_eval256": ar, "B_TF_traj": tf, "C_AR_opt": opt}

        # deltas patched vs clean for A and C
        for key in ("A_AR_eval256", "C_AR_opt"):
            tc = row["conditions"]["clean"][key].get("traj")
            tp = row["conditions"]["patched"][key].get("traj")
            shift = mean_dist(
                None if tc is None else np.asarray(tc),
                None if tp is None else np.asarray(tp),
            )
            row["conditions"]["patched"][key]["traj_shift_vs_clean"] = shift
        results.append(row)

    # Verdict summary
    def closer(a, b, margin=5.0):
        return a is not None and b is not None and a + margin < b

    verdict = {
        "n_seeds": len(results),
        "A_patched_closer_to_orange_than_clean": 0,
        "C_patched_closer_to_orange_than_clean": 0,
        "B_patched_has_orange_traj": 0,
        "B_parse_ok_patched": 0,
        "interpretation": [],
    }
    for r in results:
        ca = r["conditions"]["clean"]["A_AR_eval256"]["dist_to_orange_gt"]
        pa = r["conditions"]["patched"]["A_AR_eval256"]["dist_to_orange_gt"]
        cc = r["conditions"]["clean"]["C_AR_opt"]["dist_to_orange_gt"]
        pc = r["conditions"]["patched"]["C_AR_opt"]["dist_to_orange_gt"]
        if closer(pa, ca):
            verdict["A_patched_closer_to_orange_than_clean"] += 1
        if closer(pc, cc):
            verdict["C_patched_closer_to_orange_than_clean"] += 1
        bt = r["conditions"]["patched"]["B_TF_traj"]
        if bt.get("dist_to_orange_gt") is not None and bt["dist_to_orange_gt"] < 15:
            verdict["B_patched_has_orange_traj"] += 1
        if bt.get("parse_ok"):
            verdict["B_parse_ok_patched"] += 1

    n = max(len(results), 1)
    if verdict["A_patched_closer_to_orange_than_clean"] == 0 and verdict["C_patched_closer_to_orange_than_clean"] == 0:
        verdict["interpretation"].append(
            "Neither eval-AR nor opt-AR redirect traj toward orange → patch not flipping CoT (weak / not learned)."
        )
    if verdict["C_patched_closer_to_orange_than_clean"] > verdict["A_patched_closer_to_orange_than_clean"]:
        verdict["interpretation"].append(
            "Opt-preprocess redirects more than eval-256 → vision mismatch (resize/crops); align opt to 256."
        )
    if verdict["A_patched_closer_to_orange_than_clean"] == 0 and verdict["B_parse_ok_patched"] == n:
        verdict["interpretation"].append(
            "AR fails but TF-traj still parses actions → consistent with teacher-forcing gap on traj stage."
        )
    if not verdict["interpretation"]:
        verdict["interpretation"].append("See per-seed numbers; mixed signals.")

    payload = {"verdict": verdict, "patch": args.patch, "patch_real_size": args.patch_real_size, "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # strip huge full texts from disk? keep tails only for size
    slim = json.loads(json.dumps(payload))
    for r in slim["results"]:
        for cond in r["conditions"].values():
            for block in cond.values():
                if isinstance(block, dict) and "text" in block:
                    block["text_len"] = len(block["text"])
                    del block["text"]
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
