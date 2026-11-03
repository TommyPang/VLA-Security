#!/usr/bin/env bash
export VLA_SECURITY_ROOT="${VLA_SECURITY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
# Diversity pipeline (avoid overfit): collect a LARGER teacher set on a seed range DISJOINT
# from the eval seeds, then 8k-step predicted opt on it, then closed-loop eval on HELD-OUT
# seeds 100-109. Existing 240-file set (seeds 100-119) is left untouched (new root).
set -euo pipefail
source ${VLA_SECURITY_ROOT}/scripts/env.sh

python - <<'PY'
import numpy, torch
assert numpy.__version__.startswith("1.26"), numpy.__version__
assert torch.cuda.is_available(), "CUDA required"
print("numpy", numpy.__version__, "cuda", torch.cuda.get_device_name(0))
PY

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${VLA_SECURITY_ROOT}/outputs/logs; mkdir -p "$LOG_DIR"
RUN_ROOT=${VLA_SECURITY_ROOT}/outputs/molmoact/GraspCokeOrangeScene-v0/coke_to_orange
DATA_ROOT=${VLA_SECURITY_ROOT}/datasets/molmoact_div60   # NEW root; keeps old 240 intact
NUM_LAYOUTS="${NUM_LAYOUTS:-60}"
SEED0="${SEED0:-200}"                                        # disjoint from eval 100-109
STEPS="${OPT_STEPS:-8000}"

echo "=== [1/4] collect $NUM_LAYOUTS layouts seeds ${SEED0}..$((SEED0+NUM_LAYOUTS-1)) -> $DATA_ROOT ==="
python ${VLA_SECURITY_ROOT}/scripts/collect_coke_orange_rollouts.py \
  --output-root "$DATA_ROOT" \
  --num-layouts "$NUM_LAYOUTS" --steps-per-layout 12 \
  --seed0 "$SEED0" --closed-loop --device cuda:0 \
  2>&1 | tee "$LOG_DIR/collect_div60_${STAMP}.log"
echo "collected .pt: $(find "$DATA_ROOT" -name '*.pt' | wc -l)"

echo "=== [2/4] predicted opt steps=$STEPS on diverse set ==="
export TRAP_MOLMOACT_DATA="$DATA_ROOT"
cd ${TRAP_ROOT}
python -m trap.cli.attack \
  --config ${VLA_SECURITY_ROOT}/configs/molmoact_coke_to_orange_gpu.yaml \
  --device cuda:0 --steps "$STEPS" \
  2>&1 | tee "$LOG_DIR/opt_div60_${STAMP}.log"

RUN_DIR=$(ls -td "$RUN_ROOT"/run_* | head -1)
PATCH="$RUN_DIR/patch_best.png"
echo "RUN_DIR=$RUN_DIR PATCH=$PATCH"

echo "=== [3/4] offline diag + gate ==="
python ${VLA_SECURITY_ROOT}/scripts/diag_teacher_force_redirect.py \
  --patch "$PATCH" \
  --out ${VLA_SECURITY_ROOT}/outputs/offline_redirect/diag_div60_${STAMP}.json \
  2>&1 | tee "$LOG_DIR/diag_div60_${STAMP}.log" || true
python ${VLA_SECURITY_ROOT}/scripts/offline_patch_redirect_check.py \
  --patch "$PATCH" --modes clean table_plane \
  --camera-params ${VLA_SECURITY_ROOT}/assets/molmoact/camera_params.npz \
  --out ${VLA_SECURITY_ROOT}/outputs/offline_redirect/gate_div60_${STAMP}.json \
  2>&1 | tee "$LOG_DIR/gate_div60_${STAMP}.log" || true

echo "=== [4/4] closed-loop n=10 on HELD-OUT seeds 100-109 ==="
OUT=${VLA_SECURITY_ROOT}/outputs/simpler_eval/coke_to_orange_n10_div60_8k_${STAMP}
python ${VLA_SECURITY_ROOT}/scripts/eval_coke_orange_closed_loop.py \
  --patch "$PATCH" --patch-mode table_plane --patch-real-size 0.20 \
  --camera-params ${VLA_SECURITY_ROOT}/assets/molmoact/camera_params.npz \
  --max-steps 80 --max-new-tokens 320 \
  --seeds 100 101 102 103 104 105 106 107 108 109 \
  --output-root "$OUT" \
  2>&1 | tee "$LOG_DIR/eval_n10_div60_8k_${STAMP}.log"

echo "DONE STAMP=$STAMP PATCH=$PATCH RUN_DIR=$RUN_DIR DATA_ROOT=$DATA_ROOT EVAL=$OUT"
