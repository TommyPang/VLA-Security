#!/usr/bin/env python3
"""Step 0 gate: offline MolmoAct clean vs patched redirect (no closed-loop ASR).

On held-out coke+orange RGB frames, run MolmoAct with instruction "pick coke can"
and compare CoT trajectories / parsed actions for clean vs corner / table-plane patch.
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


def mean_pointwise_dist(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    return float(np.linalg.norm(a[:n] - b[:n], axis=1).mean())


def load_orange_gt_trajs(dataset_root: Path) -> dict[int, np.ndarray]:
    gt: dict[int, np.ndarray] = {}
    for layout in sorted(dataset_root.glob("obj_*_tgt_orange")):
        for pt in sorted(layout.glob("sample_*_output.pt")):
            d = torch.load(pt, map_location="cpu", weights_only=False)
            if int(d.get("step", -1)) != 0:
                continue
            if not d.get("trace"):
                continue
            gt[int(d["seed"])] = np.asarray(d["trace"][0], dtype=float)
    return gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("TRAP_MOLMOACT_MODEL_PATH", ""))
    ap.add_argument(
        "--frames",
        nargs="+",
        default=[],
        help="Clean RGB paths; default: ASR eval step_000_clean for seeds 100-104",
    )
    ap.add_argument(
        "--patch",
        default=f"{PROJECT_ROOT}/outputs/molmoact/GraspCokeOrangeScene-v0/coke_to_orange/run_20261019_120553/patch_best.png",
    )
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument(
        "--camera-params",
        default=f"{PROJECT_ROOT}/assets/molmoact/camera_params.npz",
    )
    ap.add_argument("--patch-real-size", type=float, default=0.5)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--out",
        default=f"{PROJECT_ROOT}/outputs/offline_redirect/step0_redirect_check.json",
    )
    ap.add_argument(
        "--dataset-root",
        default=f"{PROJECT_ROOT}/datasets/molmoact/GraspCokeOrangeScene-v0",
    )
    ap.add_argument("--modes", nargs="+", default=["clean", "corner", "table_plane"])
    args = ap.parse_args()
    if not args.model_path:
        raise SystemExit("Need TRAP_MOLMOACT_MODEL_PATH or --model-path")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from molmoact_policy import MolmoAct4BitPolicy
    from patch_overlay import load_patch_rgb, overlay_corner, overlay_table_plane

    if not args.frames:
        base = Path(f"{PROJECT_ROOT}/outputs/simpler_eval/coke_to_orange_n10_20261020_114524")
        args.frames = [
            str(base / f"clean_seed{s}" / "frames" / "step_000_clean.png") for s in range(100, 105)
        ]

    gt = load_orange_gt_trajs(Path(args.dataset_root))
    patch = load_patch_rgb(args.patch, size=args.patch_size)
    cam = Path(args.camera_params)
    has_cam = cam.exists()

    print(f"Loading MolmoAct … camera_params={'OK' if has_cam else 'MISSING'}")
    policy = MolmoAct4BitPolicy(args.model_path, device=args.device, max_new_tokens=args.max_new_tokens)
    instruction = "pick coke can"

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for frame_path in args.frames:
        clean = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
        seed_m = re.search(r"seed(\d+)", frame_path)
        seed = int(seed_m.group(1)) if seed_m else None
        variants = {"clean": clean}
        if "corner" in args.modes:
            variants["corner"] = overlay_corner(clean, patch)
        if "table_plane" in args.modes:
            if not has_cam:
                print(f"[skip table_plane] missing {cam}")
            else:
                variants["table_plane"] = overlay_table_plane(
                    clean,
                    patch,
                    camera_param_path=cam,
                    patch_real_size=args.patch_real_size,
                )
                Image.fromarray(variants["table_plane"]).save(
                    out_dir / f"seed{seed}_table_plane.png"
                )

        row = {"frame": frame_path, "seed": seed, "modes": {}}
        for mode, img in variants.items():
            if mode not in args.modes and mode != "clean":
                continue
            if mode == "clean" and "clean" not in args.modes:
                continue
            policy.reset(instruction)
            raw, _ = policy.step(img, instruction)
            text = policy.last_generated_text or ""
            traj = extract_traj(text)
            g = gt.get(seed) if seed is not None else None
            entry = {
                "parse_ok": bool(policy.last_parse_ok),
                "raw_xyz": np.asarray(raw["world_vector"], dtype=float).tolist(),
                "raw_xyz_norm": float(np.linalg.norm(raw["world_vector"])),
                "traj": None if traj is None else traj.tolist(),
                "dist_to_orange_gt": mean_pointwise_dist(traj, g),
                "cot_tail": text[-500:],
                "mentions_orange": "orange" in text.lower(),
                "mentions_coke": "coke" in text.lower() or "can" in text.lower(),
            }
            row["modes"][mode] = entry
            print(
                f"seed={seed} {mode}: parse={entry['parse_ok']} "
                f"xyz_n={entry['raw_xyz_norm']:.3f} "
                f"dist_orange_gt={entry['dist_to_orange_gt']} "
                f"traj_end={None if traj is None else traj[-1].tolist()}"
            )
            print(f"  CoT tail: …{entry['cot_tail'][-220:]}")

        # Shift metrics vs clean
        if "clean" in row["modes"]:
            tc = row["modes"]["clean"].get("traj")
            tc = None if tc is None else np.asarray(tc, dtype=float)
            for mode, entry in row["modes"].items():
                if mode == "clean":
                    continue
                tp = entry.get("traj")
                tp = None if tp is None else np.asarray(tp, dtype=float)
                entry["traj_shift_vs_clean"] = mean_pointwise_dist(tc, tp)
                dc = row["modes"]["clean"].get("dist_to_orange_gt")
                dp = entry.get("dist_to_orange_gt")
                entry["closer_to_orange_than_clean"] = (
                    dp is not None and dc is not None and dp + 5.0 < dc
                )
        results.append(row)

    # Aggregate gate
    def mode_orange_closer(mode: str) -> list[bool]:
        out = []
        for r in results:
            e = r["modes"].get(mode)
            if e and "closer_to_orange_than_clean" in e:
                out.append(bool(e["closer_to_orange_than_clean"]))
        return out

    summary = {
        "n_frames": len(results),
        "instruction": instruction,
        "patch": args.patch,
        "camera_params": str(cam) if has_cam else None,
        "gate": {},
    }
    for mode in args.modes:
        if mode == "clean":
            continue
        flags = mode_orange_closer(mode)
        shifts = [
            r["modes"][mode]["traj_shift_vs_clean"]
            for r in results
            if mode in r["modes"] and r["modes"][mode].get("traj_shift_vs_clean") is not None
        ]
        summary["gate"][mode] = {
            "n": len(flags),
            "n_clearly_closer_to_orange": sum(flags),
            "mean_traj_shift_vs_clean": float(np.mean(shifts)) if shifts else None,
            "clear_redirect": bool(flags) and sum(flags) >= max(1, (len(flags) + 1) // 2),
        }

    payload = {"summary": summary, "results": results}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== Step 0 gate ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
