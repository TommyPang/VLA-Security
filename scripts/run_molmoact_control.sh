#!/usr/bin/env bash
# Train/evaluate one matched loss ablation; optionally run a random-patch control.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_SECURITY_ROOT="${VLA_SECURITY_ROOT:-$project_root}"
source "$VLA_SECURITY_ROOT/scripts/env.sh"
mode="${1:?Usage: run_molmoact_control.sh action_only|cot_only GPU [--then-random]}"
gpu="${2:?Specify the physical GPU index}"
case "$mode" in action_only|cot_only) ;; *) echo 'Invalid control mode' >&2; exit 2 ;; esac
export CUDA_VISIBLE_DEVICES="$gpu"
# These defaults also support the original GPU1 environment script.
export TRAP_ROOT="${TRAP_ROOT:-$VLA_SECURITY_ROOT/TRAP}"
export PAPER_DATA_ROOT="${PAPER_DATA_ROOT:-$VLA_SECURITY_ROOT/datasets/molmoact_gpu1_paper_direction_20261030}"
export PAPER_MASK_ROOT="${PAPER_MASK_ROOT:-${PAPER_DATA_ROOT}_table_masks}"
control_root="${CONTROL_ROOT:-$VLA_SECURITY_ROOT/outputs/molmoact_gpu1_8bit/controls_20261103}"
root="$control_root/$mode"
config="$VLA_SECURITY_ROOT/configs/molmoact_orange_to_coke_${mode}.yaml"
mkdir -p "$root"
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" > "$root/status.txt"; }
on_exit() { local code=$?; if (( code != 0 )); then status "FAILED exit=$code"; fi; }
trap on_exit EXIT
status WAITING_GPU
while true; do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( used < 2000 )); then break; fi
    sleep 30
done
if [[ ! -f "$root/smoke.done" ]]; then
    status RUNNING_SMOKE
    (cd "$TRAP_ROOT" && python -m trap.cli.attack --config "$config" --device cuda:0 \
        --smoke --output-dir "$root/smoke") > "$root/smoke.log" 2>&1
    touch "$root/smoke.done"
fi
if [[ ! -f "$root/train.done" ]]; then
    status RUNNING_OPT
    (cd "$TRAP_ROOT" && python -m trap.cli.attack --config "$config" --device cuda:0 \
        --output-dir "$root/opt") > "$root/opt.log" 2>&1
    touch "$root/train.done"
fi
patch=$(python -c 'from pathlib import Path; import sys; ps=list(Path(sys.argv[1]).glob("run_*/patch_best.png")); assert ps, "Missing validation-selected patch"; print(max(ps,key=lambda p:p.stat().st_mtime))' "$root/opt")
# Refuse stale cached episodes if a restarted training run selects another patch.
python - "$root" "$patch" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, patch = map(Path, sys.argv[1:])
signature = hashlib.sha256(patch.read_bytes()).hexdigest()
manifest = root / 'evaluation_patch.json'
if manifest.exists():
    assert json.loads(manifest.read_text())['sha256'] == signature, 'Patch changed: use a fresh evaluation directory'
else:
    manifest.write_text(json.dumps({'patch': str(patch), 'sha256': signature}, indent=2)+'\n')
PY
status "RUNNING_EVAL patch=$patch"
python "$VLA_SECURITY_ROOT/scripts/eval_molmoact_target_completion.py" \
    --condition patched --patch "$patch" --output-root "$root/eval" --device cuda:0 > "$root/eval.log" 2>&1
python - "$root/eval/eval_summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); assert d['completed_episodes'] == d['planned_episodes'] == 50
PY
status COMPLETE
if [[ "${3:-}" == --then-random ]]; then
    random_root="$control_root/random"
    mkdir -p "$random_root"
    # Redirect subsequent status/error reporting to the random control.
    root="$random_root"
    status PREPARING_RANDOM
    python - "$random_root" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
root=Path(sys.argv[1]); rng=np.random.default_rng(0)
patch=rng.integers(0,256,size=(180,180,3),dtype=np.uint8)
Image.fromarray(patch).save(root/'random_patch.png')
(root/'patch_metadata.json').write_text(json.dumps({'seed':0,'generator':'numpy.default_rng','distribution':'uniform integer RGB [0,255]','shape':[180,180,3],'physical_size_m':0.5},indent=2)+'\n')
PY
    status RUNNING_EVAL
    python "$VLA_SECURITY_ROOT/scripts/eval_molmoact_target_completion.py" \
        --condition patched --patch "$random_root/random_patch.png" \
        --output-root "$random_root/eval" --device cuda:0 > "$random_root/eval.log" 2>&1
    python - "$random_root/eval/eval_summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); assert d['completed_episodes'] == d['planned_episodes'] == 50
PY
    status COMPLETE
fi
