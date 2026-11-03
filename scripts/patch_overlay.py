"""Patch overlays matching TRAP appliers (corner + table-plane)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_patch_rgb(path: str | Path, size: int | None = None) -> np.ndarray:
    """Load patch as uint8 RGB HxWx3, optionally resize to size x size."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def overlay_corner(
    image: np.ndarray,
    patch: np.ndarray,
    *,
    x: int = 0,
    y: int | None = None,
) -> np.ndarray:
    """Paste patch onto image (H,W,3) uint8. y=None → bottom-aligned (TRAP default)."""
    assert image.ndim == 3 and image.shape[2] == 3, image.shape
    assert patch.ndim == 3 and patch.shape[2] == 3, patch.shape
    out = image.copy()
    h, w = out.shape[:2]
    ph, pw = patch.shape[:2]
    y0 = h - ph if y is None else y
    x0 = x
    y1, x1 = min(h, y0 + ph), min(w, x0 + pw)
    if y0 >= h or x0 >= w or y1 <= 0 or x1 <= 0:
        return out
    py0 = max(0, -y0)
    px0 = max(0, -x0)
    y0 = max(0, y0)
    x0 = max(0, x0)
    out[y0:y1, x0:x1] = patch[py0 : py0 + (y1 - y0), px0 : px0 + (x1 - x0)]
    return out


def overlay_table_plane(
    image: np.ndarray,
    patch: np.ndarray,
    *,
    camera_param_path: str | Path,
    patch_real_size: float = 0.5,
    plane_z: float | None = None,
    target_size: tuple[int, int] = (512, 640),
) -> np.ndarray:
    """Project patch onto the table plane (same as TRAP TablePlanePatchApplier)."""
    from trap.patch import TablePlanePatchApplier

    assert image.ndim == 3 and image.shape[2] == 3, image.shape
    assert patch.ndim == 3 and patch.shape[2] == 3, patch.shape
    cam = Path(camera_param_path)
    if not cam.exists():
        raise FileNotFoundError(f"camera_params missing: {cam} (refusing silent corner fallback)")

    kwargs = dict(patch_real_size=float(patch_real_size), target_size=target_size)
    if plane_z is not None:
        kwargs["plane_z"] = float(plane_z)
    applier = TablePlanePatchApplier(cam, **kwargs)

    img_t = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1) / 255.0
    patch_t = torch.from_numpy(np.ascontiguousarray(patch)).float().permute(2, 0, 1) / 255.0
    # Probe: ensure homography builds (no silent corner fallback)
    _ = applier._homography(patch_t.shape[-2], patch_t.shape[-1])
    out_t = applier.apply(img_t, patch_t)
    out = (out_t.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    return out
