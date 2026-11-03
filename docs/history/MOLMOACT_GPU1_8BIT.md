# MolmoAct 8-bit workflow on sysarch-gpu1

The MolmoAct language layers load through bitsandbytes int8; vision backbone and
`lm_head` stay BF16. The default is set in `scripts/env.sh` as
`TRAP_MOLMOACT_LOAD_MODE=8bit_skip_vision`. This setting now covers collection,
optimization, closed-loop evaluation, and the teacher-forcing diagnostic.
The public checkpoint is `allenai/MolmoAct-7B-D-Pretrain-RT-1-0812` at revision
`5eca3376a325b10ca879638e84f1141ef6bb241e`, downloaded to
`/home/tommy/models/MolmoAct-7B-D-Pretrain-RT-1-0812`.

The GPU1 config is `configs/molmoact_coke_to_orange_gpu1_8bit.yaml`. It uses a
physical microbatch of 1 with 4 sequential gradient accumulations, giving an
effective batch of 4. The runner now computes each microbatch gradient with
`torch.autograd.grad` before the next forward and backpropagates once through
the patch and regularizers. This avoids retaining four model activation graphs.
The config uses the paper's 8/255 update size, 180x180 pixel patch, action
weight 1, and the repository's 12,000 step default. The scene-specific
`predicted` action-trajectory mode, 0.20 m physical patch, and 256x256 policy
resize remain from the prior local experiment; those are deviations from the
repository baseline.

## GPU memory evidence

On GPU 0 (RTX 4090, 24 GB), a shape-matched synthetic input used 640x512 RGB
and about 208 generated-text tokens, matching the old dataset's median token
count without transferring its content. The actual model forward and patch
backward measured:

| Physical batch | Accumulations | Result | Peak reserved |
|---:|---:|---|---:|
| 1 | 1 | nonzero patch gradient; 112 int8, 0 int4 layers | 17.33 GiB |
| 2 | 1 | CUDA out of memory | 23.5 GiB used |
| 1 | 4 | nonzero patch gradient; effective batch 4 | 17.35 GiB |

A one-step `trap-attack` run on four fresh synthetic image/.pt pairs completed
and wrote a patch and loss log under
`outputs/molmoact_gpu1_8bit/smoke_synthetic/`. A coke/orange SimplerEnv scene
render completed under `outputs/molmoact_gpu1_8bit/scene_proto/`.

The 8-bit collector then saved two fresh clean-orange teacher pairs in
`datasets/molmoact_gpu1_8bit_pilot/`. One optimization step on those real
samples completed with the expected mapping `pick orange` -> `pick coke can`
under `outputs/molmoact_gpu1_8bit/pilot_real_opt/`. A one-step patched policy
evaluation also completed under `outputs/molmoact_gpu1_8bit/eval_pilot/`;
it parsed an action but reported no task success. These are code-path checks;
two samples and one optimization step do not measure attack effectiveness.
Probe, install, collection, optimization, and evaluation logs are under
`outputs/molmoact_gpu1_8bit/logs/`.

## Run with fresh rollouts

```bash
source /home/tommy/vla_security/scripts/env.sh
COLLECT_LAYOUTS=25 COLLECT_STEPS=12 bash /home/tommy/vla_security/scripts/run_coke_orange_pipeline.sh collect
bash /home/tommy/vla_security/scripts/run_coke_orange_pipeline.sh opt
bash /home/tommy/vla_security/scripts/run_coke_orange_pipeline.sh eval
```

The optimization config expects clean orange teacher samples under
`/home/tommy/vla_security/datasets/molmoact`. Existing old-server experiment
samples were not moved. A request to relay that private dataset through the
local Windows machine was rejected by automatic approval review, so this
workflow collects new samples on GPU1.

The environment is `/home/tommy/miniconda3/envs/trap-molmoact` with Python
3.11, PyTorch 2.6.0+cu124, Transformers 4.52.3, bitsandbytes 0.50.2,
Accelerate 1.15.0, and NumPy 1.26.4. The SimplerEnv/ManiSkill2 packages and
SAPIEN 2.2.2 are installed in that same environment.

[TRAP paper](https://arxiv.org/pdf/2603.23117) reports an H800 for patch
optimization, physical batch 4, 8/255 pixel updates, and 25 training layouts.
This GPU1 setup preserves the effective batch and update size while using int8
weights and sequential microbatches to fit the smaller GPU.
