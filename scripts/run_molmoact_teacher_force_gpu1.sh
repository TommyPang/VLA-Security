#!/usr/bin/env bash
# Portable replacement for the original GPU1 job driver (no historical PID waits).
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MODE=teacher_force
bash "$root/scripts/reproduce_molmoact.sh" smoke
bash "$root/scripts/reproduce_molmoact.sh" train
bash "$root/scripts/reproduce_molmoact.sh" eval
