#!/usr/bin/env python3
"""Create a fresh symlink subset containing successful teacher layouts only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists() or source == destination or source in destination.parents:
        parser.error("Destination must be a new directory outside the source tree")
    selected = []
    for meta in sorted(source.glob("*/obj_*/collection_meta.json")):
        if json.loads(meta.read_text())["env_success_ever"]:
            selected.append(meta.parent)
    if not selected:
        parser.error("No successful layouts found")
    destination.mkdir(parents=True)
    for layout in selected:
        target = destination / layout.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(layout, target_is_directory=True)
    print(f"Selected {len(selected)} successful layouts into {destination}")


if __name__ == "__main__":
    main()
