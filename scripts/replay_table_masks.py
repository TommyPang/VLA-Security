#!/usr/bin/env python3
"""Replay saved clean-orange teacher actions and recover exact SimplerEnv table masks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coke_orange_scene import COKE, ORANGE, build_coke_orange_env, reset_coke_orange, set_vulkan
from collect_coke_orange_rollouts import action_to_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--mask-root', required=True)
    ap.add_argument('--limit-layouts', type=int, default=0)
    ap.add_argument('--teacher-target', choices=['orange', 'coke'], default='orange')
    args = ap.parse_args()
    data_root = Path(args.data_root)
    mask_root = Path(args.mask_root)
    layouts = sorted(data_root.glob('*/obj_*'))
    if args.limit_layouts:
        layouts = layouts[:args.limit_layouts]
    designated = ORANGE if args.teacher_target == "orange" else COKE
    set_vulkan()
    env = build_coke_orange_env(with_sponge=False)
    summary = {'layouts': 0, 'verified_frames': 0, 'table_pixels': 0, 'data_root': str(data_root)}
    try:
        for d in layouts:
            pairs = []
            for p in sorted(d.glob('sample_*_output.pt')):
                payload = torch.load(p, map_location='cpu', weights_only=False)
                pairs.append((p, payload))
            if not pairs:
                continue
            steps = [x['step'] for _, x in pairs]
            if steps != list(range(len(pairs))):
                raise RuntimeError(f'{d.name}: noncontiguous steps {steps}')
            seed = pairs[0][1]['seed']
            positions = pairs[0][1]['positions']
            obs, _ = reset_coke_orange(env, designated_target=designated, seed=seed, with_sponge=False, positions=positions)
            arena = next((a for a in env.unwrapped._scene.get_all_actors() if a.name == 'arena'), None)
            if arena is None:
                raise RuntimeError('arena actor not found')
            sticky = {'prev': None, 'on': False, 'n': 0, 'val': 0.0}
            out_dir = mask_root / d.parent.name / d.name
            for i, (p, payload) in enumerate(pairs):
                rgb = get_image_from_maniskill2_obs_dict(env, obs)
                saved = np.asarray(Image.open(d / p.name.replace('_output.pt', '_input.png')).convert('RGB'))
                if rgb.shape != saved.shape or not np.array_equal(rgb, saved):
                    diff = np.abs(rgb.astype(np.int16) - saved.astype(np.int16)) if rgb.shape == saved.shape else None
                    raise RuntimeError(f'{d.name} step={i}: replay RGB mismatch shape={rgb.shape}/{saved.shape} maxdiff={None if diff is None else diff.max()} changed={None if diff is None else np.count_nonzero(np.any(diff != 0, axis=2))}')
                seg = obs['image']['overhead_camera']['Segmentation'][:, :, 1]
                mask = seg == arena.id
                out_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray((mask.astype(np.uint8) * 255)).save(out_dir / p.name.replace('_output.pt', '_tablemask.png'))
                summary['verified_frames'] += 1
                summary['table_pixels'] += int(mask.sum())
                if i + 1 < len(pairs):
                    obs, _, done, trunc, _ = env.step(action_to_env(payload['action'], sticky))
                    if done or trunc:
                        raise RuntimeError(f'{d.name}: episode ended before saved step {i+1}')
            summary['layouts'] += 1
            print(f"{d.name}: verified {len(pairs)} frames", flush=True)
    finally:
        env.close()
    mask_root.mkdir(parents=True, exist_ok=True)
    (mask_root / 'replay_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
