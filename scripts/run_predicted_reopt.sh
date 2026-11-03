#!/usr/bin/env bash
export VLA_SECURITY_ROOT="${VLA_SECURITY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
# Predicted-traj re-opt (base gpu config: action_traj_mode=predicted, ss=1.0, action_weight=2.0),
# then diag + offline gate, then closed-loop n=10 ASR (RUN_EVAL=1 default).
# Non-phased, constant weights -> patch_best (min-loss) is valid.
set -euo pipefail
source ${VLA_SECURITY_ROOT}/scripts/env.sh

python - <<'PY'
import numpy, torch
assert numpy.__version__.startswith("1.26"), numpy.__version__
assert torch.cuda.is_available(), "CUDA required for re-opt"
print("numpy", numpy.__version__, "cuda", torch.cuda.get_device_name(0))
PY

STEPS="${OPT_STEPS:-8000}"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${VLA_SECURITY_ROOT}/outputs/logs
mkdir -p "$LOG_DIR"
RUN_ROOT=${VLA_SECURITY_ROOT}/outputs/molmoact/GraspCokeOrangeScene-v0/coke_to_orange

echo "=== predicted opt steps=$STEPS (ss=1.0, action_weight=2.0) ==="
cd ${TRAP_ROOT}
python -m trap.cli.attack \
  --config ${VLA_SECURITY_ROOT}/configs/molmoact_coke_to_orange_gpu.yaml \
  --device cuda:0 --steps "$STEPS" \
  2>&1 | tee "$LOG_DIR/opt_predicted_${STAMP}.log"

RUN_DIR=$(ls -td "$RUN_ROOT"/run_* | head -1)
PATCH="$RUN_DIR/patch_best.png"
echo "RUN_DIR=$RUN_DIR"
echo "PATCH=$PATCH"

echo "=== diag teacher-force redirect ==="
python ${VLA_SECURITY_ROOT}/scripts/diag_teacher_force_redirect.py \
  --patch "$PATCH" \
  --out ${VLA_SECURITY_ROOT}/outputs/offline_redirect/diag_predicted_${STAMP}.json \
  2>&1 | tee "$LOG_DIR/diag_predicted_${STAMP}.log"

echo "=== offline redirect gate ==="
python ${VLA_SECURITY_ROOT}/scripts/offline_patch_redirect_check.py \
  --patch "$PATCH" --modes clean table_plane \
  --camera-params ${VLA_SECURITY_ROOT}/assets/molmoact/camera_params.npz \
  --out ${VLA_SECURITY_ROOT}/outputs/offline_redirect/gate_predicted_${STAMP}.json \
  2>&1 | tee "$LOG_DIR/gate_predicted_${STAMP}.log"

if [[ "${RUN_EVAL:-1}" == "1" ]]; then
  OUT=${VLA_SECURITY_ROOT}/outputs/simpler_eval/coke_to_orange_n10_predicted8k_${STAMP}
  echo "=== closed-loop n=10 -> $OUT ==="
  python ${VLA_SECURITY_ROOT}/scripts/eval_coke_orange_closed_loop.py \
    --patch "$PATCH" --patch-mode table_plane --patch-real-size 0.20 \
    --camera-params ${VLA_SECURITY_ROOT}/assets/molmoact/camera_params.npz \
    --max-steps 80 --max-new-tokens 320 \
    --seeds 100 101 102 103 104 105 106 107 108 109 \
    --output-root "$OUT" \
    2>&1 | tee "$LOG_DIR/eval_n10_predicted8k_${STAMP}.log"
fi

echo "DONE STAMP=$STAMP PATCH=$PATCH RUN_DIR=$RUN_DIR"
