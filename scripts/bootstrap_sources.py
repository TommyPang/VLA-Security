#!/usr/bin/env python3
"""Fetch pinned upstream sources and apply the recorded local changes.

This downloads source and simulator assets from Git, but no model weights.
Existing checkouts must have the recorded HEAD; conflicting changes are refused.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(path), *args], check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def checkout(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if not path.exists() or (path.is_dir() and not any(path.iterdir())):
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", spec["url"], str(path)], check=True)
        git(path, "checkout", "--detach", spec["revision"])
    else:
        # A populated submodule has either a .git file or a .git directory.
        if not (path / ".git").exists():
            raise RuntimeError(f"Existing path is not a Git checkout: {path}")
        head = git(path, "rev-parse", "HEAD").stdout.strip()
        if head != spec["revision"]:
            raise RuntimeError(f"Refusing to change existing checkout {path}: HEAD={head}")
    if "patch" in spec:
        patch = root / spec["patch"]
        if git(path, "apply", "--reverse", "--check", str(patch), check=False).returncode == 0:
            print(f"Already patched: {path}")
        else:
            git(path, "apply", "--check", str(patch))
            git(path, "apply", str(patch))
            print(f"Applied {spec['patch']}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=["molmoact", "instructvla", "molmoact2", "poef"],
                        default="molmoact")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    lock = json.loads((root / "upstreams.lock.json").read_text())
    if args.track in {"molmoact", "instructvla"}:
        checkout(root, lock["trap"])
        checkout(root, lock[args.track])
        checkout(root, lock["simpler_env"])
        checkout(root, lock["maniskill2_real2sim"])
    elif args.track == "molmoact2":
        path = checkout(root, lock["molmoact2_wrapper"])
        for name in ["molmoact2_diagnostics.py", "analyze_molmoact2_diagnostics.py",
                     "probe_scene_text.py", "INTERPRETABILITY.md"]:
            target = path / name
            source = root / "diagnostics/molmoact2" / name
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise RuntimeError(f"Refusing to overwrite different existing file: {target}")
            shutil.copy2(source, target)
    else:
        checkout(root, lock["poef"])
    print(f"Pinned source setup complete: {args.track}")


if __name__ == "__main__":
    main()
