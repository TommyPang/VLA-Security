#!/usr/bin/env python3
"""Dump SimplerEnv google-robot overhead camera params for TRAP TablePlanePatchApplier.

Writes NPZ with:
  T_c2w  — OpenGL camera-to-world (matches TRAP unproject/project convention)
  K      — OpenCV intrinsics for the policy RGB (640×512 overhead_camera)
  plane_z — scene_table_height (default 0.87)

Also writes side-by-side previews: clean / corner / table-plane.
"""
from __future__ import annotations

from project_paths import PROJECT_ROOT, TRAP_ROOT, SIMPLER_ENV_ROOT, MODEL_ROOT

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=f"{PROJECT_ROOT}/assets/molmoact/camera_params.npz",
        help="Primary NPZ path (opt cwd is TRAP/)",
    )
    ap.add_argument(
        "--also-copy",
        default=f"{PROJECT_ROOT}/assets/molmoact/camera_params.npz",
        help="Extra copy for vla_security assets (empty to skip)",
    )
    ap.add_argument("--preview-dir", default=f"{PROJECT_ROOT}/outputs/debug_patch_applier")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--patch", default=f"{PROJECT_ROOT}/outputs/molmoact/GraspCokeOrangeScene-v0/coke_to_orange/run_20261019_120553/patch_best.png")
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--patch-real-size", type=float, default=0.5)
    ap.add_argument("--plane-z", type=float, default=None, help="Override; default=env.scene_table_height")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from coke_orange_scene import COKE, ORANGE, build_coke_orange_env, reset_coke_orange, set_vulkan
    from patch_overlay import load_patch_rgb, overlay_corner, overlay_table_plane
    from trap.patch.appliers import TablePlanePatchApplier, _project_point

    set_vulkan()
    env = build_coke_orange_env()
    obs, _ = reset_coke_orange(env, designated_target=COKE, seed=args.seed)
    u = env.unwrapped
    params = u.get_camera_params()["overhead_camera"]
    K = np.asarray(params["intrinsic_cv"], dtype=np.float64)
    T_c2w = np.asarray(params["cam2world_gl"], dtype=np.float64)
    plane_z = float(args.plane_z if args.plane_z is not None else u.scene_table_height)

    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    rgb = get_image_from_maniskill2_obs_dict(env, obs)
    assert rgb.shape == (512, 640, 3), rgb.shape

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        T_c2w=T_c2w.astype(np.float32),
        K=K.astype(np.float32),
        plane_z=np.asarray(plane_z, dtype=np.float32),
        camera_name=np.asarray("overhead_camera"),
        image_hw=np.asarray([512, 640], dtype=np.int32),
        note=np.asarray("SimplerEnv google_robot overhead; T_c2w=cam2world_gl"),
    )
    np.savez(out, **payload)
    print(f"Wrote {out} plane_z={plane_z} K=\n{K}\nT_c2w=\n{T_c2w}")

    if args.also_copy:
        also = Path(args.also_copy)
        also.parent.mkdir(parents=True, exist_ok=True)
        np.savez(also, **payload)
        print(f"Also wrote {also}")

    # Verify applier does not fall back
    applier = TablePlanePatchApplier(out, patch_real_size=args.patch_real_size, plane_z=plane_z)
    assert out.exists()
    H = applier._homography(args.patch_size, args.patch_size)
    print(f"TablePlanePatchApplier OK H=\n{H}")

    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    patch = load_patch_rgb(args.patch, size=args.patch_size)
    corner = overlay_corner(rgb, patch)
    table = overlay_table_plane(
        rgb,
        patch,
        camera_param_path=out,
        patch_real_size=args.patch_real_size,
        plane_z=plane_z,
    )
    Image.fromarray(rgb).save(preview_dir / "clean.png")
    Image.fromarray(corner).save(preview_dir / "corner.png")
    Image.fromarray(table).save(preview_dir / "table_plane.png")

    # Annotation: table UV + object projections
    vis = Image.fromarray(rgb.copy())
    dr = ImageDraw.Draw(vis)
    uvs = TablePlanePatchApplier.table_edge_uv
    dr.line([*uvs, uvs[0]], fill=(0, 255, 0), width=2)
    coke = u.obj if u.obj.name == COKE else next(d for d in u.distractor_objs if d.name == COKE)
    orange = u.obj if u.obj.name == ORANGE else next(d for d in u.distractor_objs if d.name == ORANGE)
    for label, actor, col in [("C", coke, (255, 0, 0)), ("O", orange, (255, 165, 0))]:
        uv = _project_point(np.asarray(actor.pose.p, dtype=float), T_c2w, K)
        if uv:
            dr.ellipse([uv[0] - 6, uv[1] - 6, uv[0] + 6, uv[1] + 6], outline=col, width=3)
            dr.text((uv[0] + 8, uv[1]), label, fill=col)
    vis.save(preview_dir / "annotations.png")

    # Side-by-side
    trio = Image.new("RGB", (640 * 3, 512))
    trio.paste(Image.fromarray(rgb), (0, 0))
    trio.paste(Image.fromarray(corner), (640, 0))
    trio.paste(Image.fromarray(table), (1280, 0))
    trio.save(preview_dir / "clean_corner_tableplane.png")
    print(f"Previews → {preview_dir}")

    # Sanity: table-plane must differ from corner
    diff = float(np.abs(table.astype(np.float32) - corner.astype(np.float32)).sum())
    print(f"table_vs_corner L1={diff:.1f} (should be >> 0)")
    if diff < 1e3:
        raise SystemExit("Table-plane overlay looks identical to corner — check camera/plane_z")

    env.close()


if __name__ == "__main__":
    main()
