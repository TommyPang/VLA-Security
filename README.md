# VLA Security

Reproduction code for adversarial visual patches against vision-language-action
policies. The main experiment uses **MolmoAct**, TRAP's patch objective, and a
two-object coke/orange scene in SimplerEnv. The current matched comparison asks
MolmoAct to **pick orange** while optimizing a patch toward **picking coke**.

The repository contains the collection, optimization, evaluation, diagnostic,
and setup code. Model weights, rollout data, learned patches, videos, frames,
and generated outputs are excluded. Upstream repositories are fetched at fixed
commits and our adapter changes are applied from versioned patches.

## Reproduce the MolmoAct experiment

Use Linux with an NVIDIA GPU, a working Vulkan driver, Git, and Miniconda.
The GPU1 runs used Ubuntu 22.04, Python 3.11, and RTX 4090 GPUs with 24 GB VRAM.
One GPU can run the stages sequentially. Downloads require substantial disk space.

```bash
git clone https://github.com/TommyPang/VLA-Security.git
cd VLA-Security
python3 scripts/bootstrap_sources.py
bash scripts/setup_molmoact_env.sh
source scripts/env.sh
python scripts/download_molmoact.py

# Collect once and reuse these exact data/masks for both conditioning modes.
bash scripts/reproduce_molmoact.sh collect
bash scripts/reproduce_molmoact.sh masks
bash scripts/reproduce_molmoact.sh clean

MODE=predicted bash scripts/reproduce_molmoact.sh smoke
MODE=predicted bash scripts/reproduce_molmoact.sh train
MODE=predicted bash scripts/reproduce_molmoact.sh eval

MODE=teacher_force bash scripts/reproduce_molmoact.sh smoke
MODE=teacher_force bash scripts/reproduce_molmoact.sh train
MODE=teacher_force bash scripts/reproduce_molmoact.sh eval
```

The [full reproduction guide](docs/MOLMOACT_REPRODUCTION.md) explains storage
overrides, dataset construction, checkpoint selection, metrics, and earlier runs.
The package snapshot in [environments/](environments/molmoact-gpu1-pip-freeze.txt)
records the installed GPU1 environment. The portable entry points were added
when publishing this repository; they preserve the recorded experiment settings.

## What is included

| Path | Purpose |
| --- | --- |
| `scripts/` | MolmoAct scene, teacher collection, mask replay, optimization launchers, evaluation, diagnostics, and dependency setup |
| `configs/` | Recorded experiment variants; matched runs differ only in conditioning mode and output path |
| `patches/` | Local TRAP adapter/runner changes and the MolmoAct2 wrapper changes |
| `assets/` | Instruction mapping, 1.9 KB camera calibration, and a headless Vulkan ICD descriptor |
| `instructvla/` | Separate InstructVLA workflow and dated experiment notes |
| `diagnostics/molmoact2/` | Auxiliary printed-text and activation diagnostics for an existing MolmoAct2/LeRobot setup |
| `toy/` | CPU-only synthetic demonstration of the patch objective |
| `migration/`, `docs/history/` | Historical GPU migration and experiment notes |
| `upstreams.lock.json` | Exact source and MolmoAct checkpoint revisions |

## Results and limits

See the dated [experiment status](docs/EXPERIMENT_STATUS.md). The teacher-forced
patch completed coke picks during 50/50 trials across ten layouts; 45/50 still
met the pick criterion at the final step. The predicted-mode patch completed
0/50. With teacher-forced conditioning, CoT-only completed coke picks in
50/50 trials, while action-only completed 0/50. The random-patch evaluation is
still running. Repeated trials are deterministic,
so count distinct layouts when interpreting the evidence.
The coke completion metric mirrors the simulator's grasp/contact/lift predicate;
it is separate from the environment's designated orange success signal.

These experiments retain differences from TRAP's paper protocol: quantized
weights, a two-object scene, preprocessing details, and incomplete random-patch
and official trajectory scoring. Counts in this repository should not be treated as a
reproduction of the paper's reported aggregate attack success rate.

Upstream attribution and dependency details are in [UPSTREAMS.md](docs/UPSTREAMS.md).
Historical notes preserve the commands and interpretations at their original
dates; use this README and the reproduction guide for portable commands.
