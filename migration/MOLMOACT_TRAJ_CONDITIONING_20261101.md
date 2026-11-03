# MolmoAct action-trajectory conditioning comparison — 2026-11-01

## Hypothesis

The GPU1 orange→coke patch used `action_traj_mode: predicted`, whereas the
released TRAP MolmoAct adapter teacher-forces the saved coke trajectory before
the saved coke action. A mismatch between a predicted trajectory and the coke
action target may make the optimized patch disrupt the policy rather than
produce a coherent target pick. This is a hypothesis, not an established cause.

The GPU1 implementation's `predicted` mode is more specific than the name
suggests. It runs a no-gradient forward over a teacher-forced sequence, replaces
the trajectory tokens with parallel greedy next-token predictions from that
forward, and then performs a differentiable forward over the modified sequence.
The resulting forward supplies **both** the CoT and action losses. It does not
generate a full autoregressive trajectory during training. Thus, switching the
mode tests the overall conditioning change, not an isolated action-loss term.

## Matched test

The local [reference config](../configs/molmoact_orange_to_coke_12000_validated_20261030.yaml)
is copied from GPU1's 12,000-step run. The
[teacher-forced config](../configs/molmoact_orange_to_coke_teacher_force_20261101.yaml) differs only
in `action_traj_mode: teacher_force` and its output directory. It retains the
same dataset, seed, checkpoint/load mode, effective batch of four, 8/255 PGD
step, 0.5 m/180×180 table-only patch, 256×256 optimization resize, CoT and
action weights, and validation selection.

The original queued driver waited for the clean
50-trial evaluation to free GPU 0, then ran a one-step smoke test, 12,000-step
optimization, and a 50-trial evaluation of `patch_best.png` on held-out layouts
200–209. It reuses the same target-completion evaluator and five policy RNG
trials per layout as the predicted-mode patch.

The published [portable driver](../scripts/run_molmoact_teacher_force_gpu1.sh)
uses the same stages without the original server PID waits. See the root
reproduction guide for current commands.

GPU1 driver PID at launch: `2091969`. Its status was
`WAITING_CLEAN_EVAL` at `2026-11-01T20:10:43Z`. Outputs and logs are under
`/home/tommy/vla_security/outputs/molmoact_gpu1_8bit/paper_direction_teacher_force_20261101/`.

## Interpretation

Compare target-task completion and benign completion by matched layout and
trial, plus CoT traces and TCP trajectories. A difference would implicate the
conditioning-mode change as a whole. It would not by itself identify which
part of the changed gradient (CoT versus action) caused the difference. The
quantized model, dataset coverage, scene, and preprocessing remain deviations
from the paper, but are held constant in this comparison.
