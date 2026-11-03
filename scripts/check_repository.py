#!/usr/bin/env python3
"""Check source syntax and keep generated artifacts/secrets out of this repo."""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-untracked", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    flags = ["git", "ls-files", "--cached", "-z"]
    if args.include_untracked:
        flags += ["--others", "--exclude-standard"]
    names = subprocess.check_output(flags, cwd=root).decode().split("\0")
    forbidden = {".mp4", ".gif", ".png", ".jpg", ".jpeg", ".pt", ".pth", ".ckpt",
                 ".safetensors", ".bin", ".npy", ".zip", ".pdf", ".pptx", ".docx", ".log"}
    secrets = [re.compile(p) for p in (
        r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----",
        r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{30,}\b",
        r"\bhf_[A-Za-z0-9]{25,}\b",
        r"\bsk-[A-Za-z0-9_-]{30,}\b",
    )]
    errors = []
    checked = []
    for name in sorted(set(filter(None, names))):
        path = root / name
        if not path.is_file():
            errors.append(f"Non-file or missing entry: {name}")
            continue
        if path.stat().st_size > 1_000_000:
            errors.append(f"File exceeds 1 MB: {name}")
        if path.suffix.lower() in forbidden:
            errors.append(f"Generated/binary artifact: {name}")
        if path.suffix.lower() == ".npz":
            if name != "assets/molmoact/camera_params.npz":
                errors.append(f"Unexpected numeric artifact: {name}")
            checked.append(name)
            continue
        text = path.read_text(encoding="utf-8-sig")
        if any(pattern.search(text) for pattern in secrets):
            errors.append(f"Possible credential in {name}; inspect locally")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=name)
            except SyntaxError as exc:
                errors.append(f"Python syntax: {name}:{exc.lineno}: {exc.msg}")
        checked.append(name)
    if errors:
        raise SystemExit("\n".join(errors))
    total = sum((root / name).stat().st_size for name in checked)
    print(f"Checked {len(checked)} source files: {total:,} bytes; no forbidden artifacts or credential patterns")


if __name__ == "__main__":
    main()
