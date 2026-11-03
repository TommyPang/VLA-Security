#!/usr/bin/env python3
"""Fail fast if table-plane patch occludes coke/orange before expensive opt/eval.

Exit 0 only if every seed keeps object centers clear and disk-overlap under --max-disk-frac.
"""
from __future__ import annotations

from project_paths import PROJECT_ROOT, TRAP_ROOT, SIMPLER_ENV_ROOT, MODEL_ROOT

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-params", default=f"{PROJECT_ROOT}/assets/molmoact/camera_params.npz")
    ap.add_argument("--patch", required=True)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--patch-real-size", type=float, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 120)))
    ap.add_argument("--obj-radius-px", type=int, default=28)
    ap.add_argument("--max-disk-frac", type=float, default=0.10)
    ap.add_argument("--out-dir", default=f"{PROJECT_ROOT}/outputs/offline_redirect/preflight")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from coke_orange_scene import (
        COKE,
        ORANGE,
        build_coke_orange_env,
        jitter_positions,
        reset_coke_orange,
        set_vulkan,
    )
    from patch_overlay import load_patch_rgb, overlay_table_plane
    from trap.patch.appliers import _project_point
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_vulkan()
    env = build_coke_orange_env()
    patch = load_patch_rgb(args.patch, size=args.patch_size)
    rows = []
    fails = []

    for seed in args.seeds:
        positions = jitter_positions(seed)
        obs, _ = reset_coke_orange(env, designated_target=COKE, seed=seed, positions=positions)
        u = env.unwrapped
        rgb = get_image_from_maniskill2_obs_dict(env, obs)
        params = u.get_camera_params()["overhead_camera"]
        K = np.asarray(params["intrinsic_cv"], dtype=np.float64)
        T = np.asarray(params["cam2world_gl"], dtype=np.float64)
        coke = u.obj if u.obj.name == COKE else next(d for d in u.distractor_objs if d.name == COKE)
        orange = u.obj if u.obj.name == ORANGE else next(d for d in u.distractor_objs if d.name == ORANGE)
        uv_c = _project_point(np.asarray(coke.pose.p, dtype=float), T, K)
        uv_o = _project_point(np.asarray(orange.pose.p, dtype=float), T, K)
        patched = overlay_table_plane(
            rgb,
            patch,
            camera_param_path=args.camera_params,
            patch_real_size=args.patch_real_size,
        )
        diff = np.abs(patched.astype(np.int16) - rgb.astype(np.int16)).max(axis=2)
        mask = diff > 12
        h, w = mask.shape

        def disk_stats(uv):
            yy, xx = np.ogrid[:h, :w]
            disk = (xx - uv[0]) ** 2 + (yy - uv[1]) ** 2 <= args.obj_radius_px**2
            return float(mask[disk].mean()), bool(mask[int(uv[1]), int(uv[0])])

        cf, cc = disk_stats(uv_c)
        of, oc = disk_stats(uv_o)
        ok = cf < args.max_disk_frac and of < args.max_disk_frac and not cc and not oc
        row = {
            "seed": seed,
            "coke_disk_frac": cf,
            "orange_disk_frac": of,
            "coke_center_hit": cc,
            "orange_center_hit": oc,
            "ok": ok,
        }
        rows.append(row)
        if not ok:
            fails.append(row)

        if seed == args.seeds[0] or not ok:
            vis = Image.fromarray(patched.copy())
            dr = ImageDraw.Draw(vis)
            for uv, col in [(uv_c, (255, 0, 0)), (uv_o, (255, 165, 0))]:
                r = args.obj_radius_px
                dr.ellipse([uv[0] - r, uv[1] - r, uv[0] + r, uv[1] + r], outline=col, width=2)
            ys, xs = np.where(mask)
            if len(xs):
                dr.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline=(0, 255, 0), width=2)
            tag = "OK" if ok else "COVER"
            dr.text(
                (8, 8),
                f"size={args.patch_real_size} {tag} coke={cf:.3f} orange={of:.3f}",
                fill=(255, 255, 0),
            )
            vis.save(out_dir / f"seed{seed}_size{args.patch_real_size:.2f}.png")

    env.close()
    summary = {
        "patch_real_size": args.patch_real_size,
        "n_seeds": len(args.seeds),
        "n_fail": len(fails),
        "worst_coke": max(r["coke_disk_frac"] for r in rows),
        "worst_orange": max(r["orange_disk_frac"] for r in rows),
        "pass": len(fails) == 0,
        "fails": fails,
    }
    (out_dir / f"preflight_size{args.patch_real_size:.2f}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    if fails:
        raise SystemExit(f"PREFLIGHT FAIL: patch_real_size={args.patch_real_size} occludes objects on {len(fails)} seeds")
    print(f"PREFLIGHT PASS size={args.patch_real_size}")


if __name__ == "__main__":
    main()
