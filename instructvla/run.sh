#!/usr/bin/env bash
export VLA_SECURITY_ROOT="${VLA_SECURITY_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
set -euo pipefail
ROOT=${VLA_SECURITY_ROOT}
source "$ROOT/instructvla/env.sh"
MODE="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi

if [[ "$MODE" == download || "$MODE" == download-metadata ]]; then
  if [[ "$MODE" == download-metadata ]]; then
    exec "$INSTRUCTVLA_ENV/bin/python" "$ROOT/instructvla/download_weights.py" --metadata-only "$@"
  fi
  exec "$INSTRUCTVLA_ENV/bin/python" "$ROOT/instructvla/download_weights.py" "$@"
fi
if [[ "$MODE" == preflight ]]; then
  exec /usr/bin/python3 "$ROOT/instructvla/preflight.py" "$@"
fi

if [[ ! -x "$INSTRUCTVLA_ENV/bin/python" ]]; then
  echo "Missing InstructVLA environment; run bash $ROOT/instructvla/setup_env.sh" >&2
  exit 2
fi
case "$MODE" in
  collect) exec python -m instructvla.collect "$@" ;;
  opt) exec python -m instructvla.run_attack --config "$ROOT/configs/instructvla_coke_to_orange.yaml" "$@" ;;
  eval) exec python -m instructvla.evaluate "$@" ;;
  *) echo "Usage: bash $ROOT/instructvla/run.sh {download-metadata|download|preflight|collect|opt|eval} [args...]" >&2; exit 2 ;;
esac
