#!/usr/bin/env python3
"""Download the exact public checkpoint used on GPU1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads((root / "upstreams.lock.json").read_text())["molmoact_checkpoint"]
    if args.metadata_only:
        print(json.dumps(spec, indent=2))
        return
    from huggingface_hub import snapshot_download
    from project_paths import MODEL_ROOT
    output = args.output_dir or MODEL_ROOT / spec["repo_id"].split("/")[-1]
    snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"], local_dir=output)
    print(output)


if __name__ == "__main__":
    main()
