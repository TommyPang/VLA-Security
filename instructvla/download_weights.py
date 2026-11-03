#!/usr/bin/env python3
"""Download only the Stage-2 checkpoint and assets needed by the TRAP adapter."""
from __future__ import annotations

from scripts.project_paths import PROJECT_ROOT, TRAP_ROOT, SIMPLER_ENV_ROOT, MODEL_ROOT

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPO = "ShuaiYang03/instructvla_finetune_v2_xlora_freeze_head_instruction"
ASSET_REPO = "ShuaiYang03/InstructVLA_Assets"
MODEL_REVISION = "ae4db200ff03a99054fd71a5cbd555d3b8d8f84ef"
ASSET_REVISION = "1c43f75a778ff9f3f9cff1b4155e7dbfc1b0823c"
CHECKPOINT_NAME = "step-013500-epoch-01-loss=0.1093.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path(f"{MODEL_ROOT}/instructvla_stage2"))
    parser.add_argument("--assets-root", type=Path, default=Path(f"{MODEL_ROOT}/InstructVLA_Assets"))
    parser.add_argument("--metadata-only", action="store_true", help="Fetch small config files without large weight files")
    args = parser.parse_args()
    model_patterns = ["config.json", "dataset_statistics.json"]
    asset_patterns = ["Eagle2-2B/*.json", "Eagle2-2B/*.py", "Eagle2-2B/*.txt", "empty_language_adapter/*.json"]
    if not args.metadata_only:
        model_patterns.append(f"checkpoints/{CHECKPOINT_NAME}")
        asset_patterns.extend(["Eagle2-2B/model.safetensors", "empty_language_adapter/adapter_model.safetensors"])
    snapshot_download(MODEL_REPO, revision=MODEL_REVISION, allow_patterns=model_patterns, local_dir=args.model_root)
    snapshot_download(ASSET_REPO, revision=ASSET_REVISION, allow_patterns=asset_patterns, local_dir=args.assets_root)
    print(f"TRAP_INSTRUCTVLA_CKPT={args.model_root / 'checkpoints' / CHECKPOINT_NAME}")
    print(f"TRAP_INSTRUCTVLA_EAGLE2_DIR={args.assets_root / 'Eagle2-2B'}")
    print(f"TRAP_INSTRUCTVLA_EMPTY_ADAPTER={args.assets_root / 'empty_language_adapter'}")


if __name__ == "__main__":
    main()
