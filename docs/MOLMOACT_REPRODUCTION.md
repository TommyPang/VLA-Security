# Reproducing the GPU1 MolmoAct experiments

## Sources and environment

From the repository root, run the bootstrap and setup commands in the README.
`bootstrap_sources.py` clones the recorded TRAP, MolmoAct, SimplerEnv, and
ManiSkill2 commits. It applies `patches/trap-gpu1.patch` to TRAP. That patch
captures the GPU1 runner, adapter, patch application, and InstructVLA import
changes; it excludes the server-specific InstructVLA simulator symlink.
The camera calibration is committed under `assets/molmoact/`.

The setup script restores the observed Python package set with `--no-deps`,
then installs the four source packages as editable packages. This avoids
resolving the upstream projects' mutually incompatible broad dependency lists.
The freeze records the working GPU1 package set rather than a fresh solver
lock. Installation into a completely new machine has not been validated end
to end. Run the smoke stage before spending time on a full optimization.

The recorded runtime is PyTorch 2.6.0+cu124, Transformers 4.52.3,
bitsandbytes 0.50.2, Accelerate 1.15.0, NumPy 1.26.4, and SAPIEN 2.2.2.
Keep NumPy below 2: the earlier SAPIEN/IK stack crashed with NumPy 2.
Use a CUDA-compatible driver and working NVIDIA Vulkan rendering. The bundled
ICD descriptor names `libGLX_nvidia.so.0`; override `VK_ICD_FILENAMES` if your
installation uses another descriptor.

`download_molmoact.py` downloads the public checkpoint at its recorded revision.
Its model includes custom Python code loaded through `trust_remote_code=True`.
`--metadata-only` prints the revision without downloading weights.

Paths default to locations inside this clone, except Conda environments under
`$HOME/miniconda3/envs`. Set these variables **before** sourcing `scripts/env.sh`
when using other locations:

```bash
export CONDA_ROOT=/path/to/miniconda3
export MOLMOACT_ENV=/path/to/trap-molmoact
export TRAP_ROOT=/path/to/TRAP
export SIMPLER_ENV_ROOT=/path/to/SimplerEnv
export VLA_MODEL_ROOT=/path/to/models
# Or point directly at the downloaded checkpoint:
export TRAP_MOLMOACT_MODEL_PATH=/path/to/MolmoAct-7B-D-Pretrain-RT-1-0812
source scripts/env.sh
```

The bootstrap itself uses the relative paths in `upstreams.lock.json`; the
environment overrides support already provisioned source trees. Existing
checkouts with another HEAD or conflicting patch changes are refused.

## Teacher data and table masks

The current paper-direction configs use **all collected coke teacher samples**:
25 layouts, seeds 100–124, up to 24 actions each, 256×256 policy inputs, and
unmodified parsed actions. The recorded collection saved 510 samples and had
14 successful layouts. It was not filtered to those 14 layouts. This differs
from the earlier reverse-direction successful-orange-only experiment.

```bash
source scripts/env.sh
bash scripts/reproduce_molmoact.sh collect
bash scripts/reproduce_molmoact.sh masks
```

The collector stores the native RGB image, generated teacher text, action,
trace, scene positions, seed, and collection metadata. Mask replay replays each
teacher action and requires an exact RGB match before saving arena segmentation
masks. These masks preserve object and robot pixels during patch optimization.
Outputs are created under `datasets/`, which Git ignores.

Override `PAPER_DATA_ROOT` and `PAPER_MASK_ROOT` for another storage location.
Use one saved collection for both conditioning modes; collecting twice would
introduce another experimental variable. Recollection on a different runtime
may produce different samples and success counts.

For the earlier coke-prompt/orange-target direction:

```bash
python scripts/collect_coke_orange_rollouts.py \
  --teacher-target orange --closed-loop --match-eval-policy \
  --num-layouts 25 --seed0 100 --steps-per-layout 24 \
  --output-root "$VLA_SECURITY_ROOT/datasets/molmoact_gpu1_8bit_policy_aligned_20261027"
python scripts/replay_table_masks.py --teacher-target orange \
  --data-root "$VLA_SECURITY_ROOT/datasets/molmoact_gpu1_8bit_policy_aligned_20261027" \
  --mask-root "$VLA_SECURITY_ROOT/datasets/molmoact_gpu1_8bit_policy_aligned_20261027_table_masks"
python scripts/select_successful_layouts.py \
  --source "$VLA_SECURITY_ROOT/datasets/molmoact_gpu1_8bit_policy_aligned_20261027" \
  --destination "$VLA_SECURITY_ROOT/datasets/molmoact_gpu1_success_orange_20261029"
```

That earlier collection recorded 464 total samples across 25 layouts; the
successful subset contained 13 layouts and 176 samples. Its validated config
is `configs/molmoact_coke_to_orange_success_only_12000_validated_20261030.yaml`.
The generic closed-loop evaluator supports this direction; the new repeated
target-completion evaluator specifically measures orange-prompt/coke-target.

## Matched optimization

```bash
MODE=predicted bash scripts/reproduce_molmoact.sh smoke
MODE=predicted bash scripts/reproduce_molmoact.sh train
MODE=teacher_force bash scripts/reproduce_molmoact.sh smoke
MODE=teacher_force bash scripts/reproduce_molmoact.sh train
```

Both runs use 12,000 steps, seed 0, microbatch 1 with four gradient
accumulations, 8/255 PGD updates, a 180×180 RGB texture representing 0.5 m,
table-only masks, and action filtering at 0.035. Both use the same validation
split and select `patch_best.png` using validation trajectory accuracy.
Language weights load in int8; vision and output head remain BF16.

The two matched configs differ only in `attack.action_traj_mode` and
`output_dir`. In `teacher_force`, the saved trajectory conditions the saved
action. In GPU1's `predicted` implementation, a no-gradient forward produces
parallel greedy next-token trajectory predictions, which replace the saved
trajectory tokens before the differentiable forward. This is not full
autoregressive trajectory generation. Both CoT and action losses use that
modified forward, so this comparison changes the overall conditioning mode.

Set `DEVICE=cuda:1` or `CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0` to select a GPU.
Each mode writes under `outputs/reproduction/<mode>/`; `RUN_ROOT` overrides it.
Full training is not resumable from a pixel PNG with identical optimizer and
RNG state. Use a fresh run directory when restarting a training experiment.

## Held-out evaluation

```bash
bash scripts/reproduce_molmoact.sh clean
MODE=predicted bash scripts/reproduce_molmoact.sh eval
MODE=teacher_force bash scripts/reproduce_molmoact.sh eval
```

Each condition uses layouts 200–209, five policy RNG seeds per layout, an
80-action horizon, and a 320-token generation budget. Greedy generation and
fixed simulator layouts currently yield identical repeated trials within a
layout. Report layout outcomes alongside trial totals.

The environment designates orange and receives `pick orange`. An independent
coke predicate checks simultaneous grasp, no nonrobot support contact, and a
lift greater than 1 cm above the settled height. The evaluator also records
orange environment success, an independent orange predicate, maximum object
lifts, >2 cm lift counts, generated reasoning, action vectors, and TCP paths.
The clean/predicted runs had no disagreement between independent orange
completion and the environment's orange success.

`PATCH_PATH=/path/to/patch_best.png` selects an explicit patch. Otherwise the
launcher takes the newest validation-selected checkpoint under that mode's
optimization root. `CLEAN_EVAL_ROOT` changes the clean output location.
Evaluation resumes completed episodes when rerun against the same output
root. **Use a new output root after changing a patch, model, horizon, or other
evaluation setting**, because completed episode files are reused.

For a short check, pass `--layout-seeds 200 --trials 1 --max-steps 1` to `eval`
or `clean` and use a separate output root. Full evaluation can take many hours.

## Loss and random-patch controls

The controls keep the teacher-forced joint configuration and disable one loss
term each. Both retain the same TV regularization and validation checkpoint
selection. Run on two idle GPUs after collecting the shared data/masks:

```bash
bash scripts/run_molmoact_control.sh action_only 0 --then-random
# In a second terminal:
bash scripts/run_molmoact_control.sh cot_only 1
```

The physical GPU index is the second argument. The drivers wait for that GPU
to have less than 2 GB used before starting, write phase/error status files,
and run smoke, 12,000-step optimization, and 50-trial evaluation. The optional
random control uses a NumPy `default_rng(0)` uniform integer RGB texture of
180×180 pixels with the same 0.5 m table-only projection.

Completed training is marked by `train.done`. An evaluation patch hash guards
against reusing episodes after a different checkpoint is selected. Inspect any
failure before restarting; incomplete training restarts from scratch. Set
`CONTROL_ROOT` to a fresh directory for another independent experiment.

Analyze existing episode folders without loading a model:

```bash
python scripts/compare_molmoact_evaluations.py \
  --condition clean=/path/to/clean \
  --condition predicted=/path/to/predicted \
  --condition teacher_force=/path/to/teacher_force \
  --output-root "$VLA_SECURITY_ROOT/outputs/conditioning_comparison"
```

The JSON report includes completion outcomes, final-step retention,
deterministic repeat counts, and CoT parsing. The matched-trial CSV records
completion, initial image-space CoT trace shifts, and exploratory TCP DTW
distances. TCP is sampled after actions.
The DTW normalization is documented in the script and is not the paper's
trajectory-hijacking score. Add control conditions with more `--condition` flags.

## Remaining paper-fidelity gaps

The paper's full protocol includes additional task pairs, a distractor object,
and its own target-task and trajectory-hijacking scores. This repository
includes the executed two-object GPU1 workflow and completed random,
action-only, and CoT-only controls. It uses an independent simulator-style
grasp/contact/lift predicate for coke and exploratory matched-episode trajectory
distances; the exact paper scoring protocol remains to be reproduced.
Quantization and preprocessing also differ from the paper setup.
See the [TRAP source](https://github.com/Zhengxian-Huang/TRAP) and
[paper](https://arxiv.org/abs/2603.23117) for the authors' protocol.
