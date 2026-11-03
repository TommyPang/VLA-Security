# InstructVLA coke-to-orange TRAP track

## Portable setup in this repository

Run these commands from the repository root on Linux. Keep this environment
separate from MolmoAct's environment.

```bash
python3 scripts/bootstrap_sources.py --track instructvla
bash instructvla/setup_env.sh
bash instructvla/run.sh download
bash instructvla/run.sh preflight
bash instructvla/run.sh collect --num-layouts 20 --seed0 100
bash instructvla/run.sh opt --smoke
bash instructvla/run.sh opt
bash instructvla/run.sh eval --patch /path/to/patch_best.png --seeds 200 201 202
```

`scripts/paths.sh` supplies clone-relative source/model/data locations.
Override `CONDA_ROOT`, `INSTRUCTVLA_ENV`, `TRAP_ROOT`, `SIMPLER_ENV_ROOT`, or
`VLA_MODEL_ROOT` before setup if needed. The following sections preserve
historical commands and observations from the original servers; their absolute
paths should be replaced with the current clone locations.

This track uses the existing `GraspSingleCustomInScene-v0` coke/orange scene. The benign instruction is **pick coke can**; clean **pick orange** rollouts provide the target text and continuous action chunks. TRAP then optimizes a table-plane patch under the coke instruction. It lives in `vla_security/instructvla` and does not change the MolmoAct scripts or their Python environment.

## Checkpoint choice

Use `ShuaiYang03/instructvla_finetune_v2_xlora_freeze_head_instruction`, specifically `checkpoints/step-013500-epoch-01-loss=0.1093.pt`. It is the Stage-2 generalist without robot state and has reported SimplerEnv-Instruct evaluation **with multimodal reasoning**. The linked `instructvla_pretraining_v2_query_64_lora_bs128` is a Stage-1 expert checkpoint; its model card describes SimplerEnv evaluation but not the Stage-2 CoT route needed for this attack. The loader also needs the `Eagle2-2B` and `empty_language_adapter` folders from `ShuaiYang03/InstructVLA_Assets`.

Sources: [TRAP README](https://github.com/Zhengxian-Huang/TRAP), [InstructVLA README](https://github.com/InternRobotics/InstructVLA), [Stage-2 model card](https://huggingface.co/ShuaiYang03/instructvla_finetune_v2_xlora_freeze_head_instruction), [assets model card](https://huggingface.co/ShuaiYang03/InstructVLA_Assets).

## Workflow

All commands below are on `liush-workstation`. Download and setup use CPU, disk, and network; `collect`, `opt`, and `eval` use the GPU. Do not start those three stages while another job owns the GPU.

```bash
bash /home/tommy/vla_security/instructvla/run.sh download-metadata
bash /home/tommy/vla_security/instructvla/run.sh download
bash /home/tommy/vla_security/instructvla/setup_env.sh
bash /home/tommy/vla_security/instructvla/run.sh preflight

# First check clean behavior on 3 seeds. This produces no attack data.
bash /home/tommy/vla_security/instructvla/run.sh eval --seeds 100 101 102

# Generate orange teacher data; only successful orange rollouts become .pt pairs.
bash /home/tommy/vla_security/instructvla/run.sh collect --num-layouts 20 --seed0 100

# Optimize a patch with batch size 1. Start with --smoke when GPU is free.
bash /home/tommy/vla_security/instructvla/run.sh opt --smoke
bash /home/tommy/vla_security/instructvla/run.sh opt

# Supply the chosen patch from the optimization output.
bash /home/tommy/vla_security/instructvla/run.sh eval --patch /data/tommy/vla_security/outputs/instructvla/GraspCokeOrangeScene-v0/coke_to_orange/run_YYYYMMDD_HHMMSS/patch_latest.png --seeds 100 101 102
```

The collector writes `sample_*_input.png` and `sample_*_output.pt` pairs with `response` and normalized `samples` fields expected by `trap.data.ImagePtDataset`. The `.pt` instruction is `pick orange`; the existing instruction map changes it to `pick coke can` inside TRAP. Failed orange teacher episodes are excluded. Evaluation records the model response, action, object grasp/lift state, and per-condition success rates under `outputs/instructvla_eval/`.

Every new evaluation episode also writes `episode.gif` beside `episode_meta.json` and the saved PNG frames. The GIF plays at 3 frames per second and captions the condition, seed, instruction, CoT, and object grasp/lift state. A `trajectory_index.md` in the run directory links every GIF and lists coke/orange lift outcomes, making successes and failures easy to find. Use `--gif-fps 5` to change playback speed or `--no-gif` to skip export. Existing runs can be converted without loading the model or using the GPU:

```bash
source /home/tommy/vla_security/instructvla/env.sh
python -m instructvla.trajectory_video /data/tommy/vla_security/outputs/instructvla_eval/run_YYYYMMDD_HHMMSS
# Or provide one episode folder, such as .../patched_coke_seed100.
```

The custom scene runs `google_robot_static`. The policy therefore uses InstructVLA's `fractal20220817_data` action statistics, two-chunk adaptive ensemble, and relative sticky gripper conversion. The first full clean control with the WidowX/Bridge conversion produced 0/1 grasps for both objects and should not be used as a model baseline.

The optimizer launcher specializes only the InstructVLA adapter in its process: it casts the VLM to bf16 as upstream inference does, enforces the existing camera calibration, and projects a 0.20 m patch onto the scene's `z=0.87` table plane. The 0.20 m size follows this workspace's MolmoAct occlusion preflight. TRAP's general files and current MolmoAct work remain untouched.

GPU experiments and measured outcomes are in [RESULTS.md](RESULTS.md). The 300-step patch did not redirect any of three coke episodes. After 2,300 total steps, including a higher CoT weight in the last 1,000, the patched coke policy lifted orange on 2/10 matched seeds; clean coke and clean orange each lifted their targets on 10/10. Generated reasoning still named coke on all ten patched episodes, so the CoT hijack is not reproduced. The evaluator resets model and simulator RNGs per seed for matched comparisons. To resume pixel-patch optimization, set `TRAP_INSTRUCTVLA_RESUME_PATCH` to a saved PNG. Set `TRAP_INSTRUCTVLA_COT_WEIGHT` to override the CoT loss weight for a follow-up run.

The expanded experiment is documented in [LARGE_RUN_20261026.md](LARGE_RUN_20261026.md): 100 successful clean-orange layouts, 415 pairs in a fresh data root, and a separate 6,000-step patch run. On ten held-out seeds that patch lifted orange 3/10 times; generated reasoning still named coke.

## Interpretation gates

1. Confirm clean coke and clean orange controls work with **this** Stage-2 checkpoint in the custom scene. The published scores are on the authors' task sets, so they do not establish that the coke/orange scene is solvable.
2. Inspect `collection_summary.json`: at least some successful orange teacher episodes and image/PT pairs are needed before optimizing.
3. For the smoke step, verify finite CoT and action losses and a nonzero patch gradient. The full checkpoint passed this check on the 16 GB GPU after casting action loss operands to fp32.
4. Compare patched coke against clean coke on identical seeds. Count orange grasp/lift separately from changes in generated reasoning.

The Hugging Face checkpoint is about 11.9 GB, plus Eagle2 weights. No quantization is configured for this path. A 300-step optimization used about 15.2 GB of GPU memory and completed on the RTX 4080.

The setup script installs the upstream FlashAttention 2.5.5 wheel. Although the outer Eagle2 loader selects eager attention, its nested vision and language configs request FlashAttention2. Forcing those nested modules to eager caused an attention-mask shape error on the first action, so the pipeline uses the upstream wheel.

The upstream `pip_requirements.txt` includes training and video packages that cannot be installed here (`av==14.3.0` is unavailable, and `ruckig==0.15.3` fails its build). `setup_env.sh` installs the inference and SimplerEnv subset and uses the available `ruckig==0.12.2` wheel. Two small changes in TRAP's vendored InstructVLA code defer RLDS/TensorFlow imports until the training dataset factory is called, so this image-only path can load without those packages. Clean coke and orange controls each succeeded on three matched seeds.
