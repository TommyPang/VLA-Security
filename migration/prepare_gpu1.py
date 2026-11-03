#!/usr/bin/env python3
"""Relocate copied VLA workflow paths for sysarch-gpu1 without moving data."""
from __future__ import annotations

import json
from pathlib import Path


HOME = Path("/home/tommy")
PROJECT = HOME / "vla_security"
EDITABLE_DIRS = (PROJECT / "configs", PROJECT / "scripts", PROJECT / "instructvla")
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".txt"}


def ensure_link(name: str, target: Path) -> None:
    link = PROJECT / name
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"Refusing to replace non-symlink: {link}")
    link.symlink_to(target, target_is_directory=True)


def main() -> None:
    for path in (PROJECT, HOME / "TRAP", HOME / "SimplerEnv"):
        if not path.is_dir():
            raise RuntimeError(f"Missing extracted code tree: {path}")

    for name in ("datasets", "outputs"):
        (PROJECT / name).mkdir(exist_ok=True)
    ensure_link("TRAP", HOME / "TRAP")
    ensure_link("SimplerEnv", HOME / "SimplerEnv")
    ensure_link("datasets_data", PROJECT / "datasets")
    ensure_link("outputs_data", PROJECT / "outputs")

    changed = []
    for folder in EDITABLE_DIRS:
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            content = path.read_text(encoding="utf-8")
            updated = content.replace("/data/tommy", "/home/tommy")
            if updated != content:
                path.write_text(updated, encoding="utf-8")
                changed.append(str(path.relative_to(PROJECT)))

    # On the old host the base conda Python already had huggingface_hub. A new
    # Miniconda install will not, so download with the project environment.
    runner = PROJECT / "instructvla" / "run.sh"
    content = runner.read_text(encoding="utf-8")
    content = content.replace(
        'MODE="${1:-}"',
        'source "$ROOT/instructvla/env.sh"\nMODE="${1:-}"',
        1,
    )
    content = content.replace(
        'exec /home/tommy/miniconda3/bin/python "$ROOT/instructvla/download_weights.py"',
        'exec "$INSTRUCTVLA_ENV/bin/python" "$ROOT/instructvla/download_weights.py"',
    )
    content = content.replace(
        'exec /home/tommy/miniconda3/bin/python "$ROOT/instructvla/preflight.py"',
        'exec /usr/bin/python3 "$ROOT/instructvla/preflight.py"',
    )
    content = content.replace(
        '\nsource "$ROOT/instructvla/env.sh"\nif [[ ! -x',
        '\nif [[ ! -x',
        1,
    )
    runner.write_text(content, encoding="utf-8")
    changed.append(str(runner.relative_to(PROJECT)))

    report = {
        "project": str(PROJECT),
        "trap": str((PROJECT / "TRAP").resolve()),
        "simpler_env": str((PROJECT / "SimplerEnv").resolve()),
        "datasets": str((PROJECT / "datasets_data").resolve()),
        "outputs": str((PROJECT / "outputs_data").resolve()),
        "relocated_files": changed,
        "weights_copied": False,
        "python_environment_copied": False,
    }
    (PROJECT / "GPU1_MIGRATION_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
