#!/usr/bin/env bash
export VLA_SECURITY_ROOT="${VLA_SECURITY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
# Core milestone: coke+orange scene → collect → TRAP opt → closed-loop eval.
set -euo pipefail
source ${VLA_SECURITY_ROOT}/scripts/env.sh

STAGE="${1:-all}"
shift || true
LOG_ROOT=${VLA_SECURITY_ROOT}/outputs
ATTACK_CONFIG="${MOLMOACT_ATTACK_CONFIG:-${VLA_SECURITY_ROOT}/configs/molmoact_coke_to_orange_gpu1_8bit.yaml}"
PATCH_ROOT="${MOLMOACT_PATCH_ROOT:-${VLA_SECURITY_ROOT}/outputs/molmoact_gpu1_8bit/GraspCokeOrangeScene-v0/coke_to_orange}"
mkdir -p "$LOG_ROOT"

collect() {
  echo "=== collect (coke→orange rollouts) ==="
  # Scale: ≥20 layouts × longer closed-loop orange rollouts (mapping polarity unchanged).
  python ${VLA_SECURITY_ROOT}/scripts/collect_coke_orange_rollouts.py \
    --num-layouts "${COLLECT_LAYOUTS:-20}" \
    --steps-per-layout "${COLLECT_STEPS:-12}" \
    --seed0 100 --closed-loop "$@"
}

optimize() {
  echo "=== optimize ==="
  cd ${TRAP_ROOT}
  python -m trap.cli.attack \
    --config "$ATTACK_CONFIG" \
    --device cuda:0 "$@"
}

evaluate() {
  echo "=== closed-loop eval (clean vs patched) ==="
  PATCH=$(ls -t "$PATCH_ROOT"/run_*/patch_best.png 2>/dev/null | head -1 || true)
  if [[ -z "${PATCH}" ]]; then
    echo "No patch_best.png under coke_to_orange/" >&2
    exit 1
  fi
  echo "Using patch: $PATCH"
  # shellcheck disable=SC2086
  python ${VLA_SECURITY_ROOT}/scripts/eval_coke_orange_closed_loop.py \
    --patch "$PATCH" --max-steps "${EVAL_STEPS:-40}" --seeds ${EVAL_SEEDS:-100 101} "$@"
}

proto() {
  echo "=== scene prototype screenshots ==="
  python ${VLA_SECURITY_ROOT}/scripts/prototype_coke_orange_scene.py --seed 42 "$@"
}

case "$STAGE" in
  proto) proto "$@" ;;
  collect) collect "$@" ;;
  opt|optimize) optimize "$@" ;;
  eval|evaluate) evaluate "$@" ;;
  all)
    proto
    collect
    optimize
    evaluate
    ;;
  *)
    echo "Usage: $0 {proto|collect|opt|eval|all}" >&2
    exit 1
    ;;
esac
