#!/usr/bin/env bash
# Matched orange-prompt / coke-target GPU1 experiment.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
stage="${1:-help}"
if (( $# )); then shift; fi
mode="${MODE:-teacher_force}"
device="${DEVICE:-cuda:0}"
case "$mode" in
    teacher_force) config_name=molmoact_orange_to_coke_teacher_force_20261101.yaml ;;
    predicted) config_name=molmoact_orange_to_coke_12000_validated_20261030.yaml ;;
    *) echo 'MODE must be teacher_force or predicted' >&2; exit 2 ;;
esac
config="${MOLMOACT_ATTACK_CONFIG:-$VLA_SECURITY_ROOT/configs/$config_name}"
data_root="${PAPER_DATA_ROOT:-$VLA_SECURITY_ROOT/datasets/molmoact_gpu1_paper_direction_20261030}"
mask_root="${PAPER_MASK_ROOT:-${data_root}_table_masks}"
run_root="${RUN_ROOT:-$VLA_SECURITY_ROOT/outputs/reproduction/$mode}"
clean_root="${CLEAN_EVAL_ROOT:-$VLA_SECURITY_ROOT/outputs/reproduction/clean}"
export PAPER_DATA_ROOT="$data_root" PAPER_MASK_ROOT="$mask_root"

collect() {
    python "$VLA_SECURITY_ROOT/scripts/collect_coke_orange_rollouts.py" \
        --output-root "$data_root" --teacher-target coke --num-layouts 25 \
        --steps-per-layout 24 --seed0 100 --closed-loop --match-eval-policy --device "$device" "$@"
}
masks() {
    python "$VLA_SECURITY_ROOT/scripts/replay_table_masks.py" \
        --data-root "$data_root" --mask-root "$mask_root" --teacher-target coke "$@"
}
train() {
    (cd "$TRAP_ROOT" && python -m trap.cli.attack --config "$config" \
        --device "$device" --output-dir "$run_root/opt" "$@")
}
evaluate() {
    local patch="${PATCH_PATH:-}"
    if [[ -z "$patch" ]]; then
        patch=$(python -c 'from pathlib import Path; import sys; ps=list(Path(sys.argv[1]).glob("run_*/patch_best.png")); assert ps, "No patch_best.png: train first or set PATCH_PATH"; print(max(ps,key=lambda p:p.stat().st_mtime))' "$run_root/opt")
    fi
    python "$VLA_SECURITY_ROOT/scripts/eval_molmoact_target_completion.py" \
        --condition patched --patch "$patch" --output-root "$run_root/eval" --device "$device" "$@"
}
clean() {
    python "$VLA_SECURITY_ROOT/scripts/eval_molmoact_target_completion.py" \
        --condition clean --output-root "$clean_root" --device "$device" "$@"
}
case "$stage" in
    collect) collect "$@" ;;
    masks) masks "$@" ;;
    smoke) train --smoke --output-dir "$run_root/smoke" "$@" ;;
    train) train "$@" ;;
    eval) evaluate "$@" ;;
    clean) clean "$@" ;;
    all) collect; masks; train --smoke --output-dir "$run_root/smoke"; train; clean; evaluate ;;
    *) echo 'Usage: MODE=teacher_force|predicted bash scripts/reproduce_molmoact.sh {collect|masks|smoke|train|clean|eval|all} [extra CLI args]' ;;
esac
