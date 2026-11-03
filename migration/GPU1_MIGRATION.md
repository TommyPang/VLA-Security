# GPU1 code migration — 2026-10-26

Source: `liush-workstation.cs.uwaterloo.ca` (`tommy`).
Destination: `sysarch-gpu1.cs.uwaterloo.ca` (`tommy`).

## Copied code

- `/home/tommy/vla_security`: workflow scripts, InstructVLA track, configs,
  small assets, and documentation.
- `/home/tommy/TRAP`: repository checkout and its uncommitted working-tree
  changes. Source HEAD: `d52ed844eb74dcadc95b2228385866281ed0af40`.
- `/home/tommy/SimplerEnv`: repository checkout, simulator assets, and
  ManiSkill2 submodule. Source HEAD:
  `c701ea271b16b80616d6fed9713c4a50a0adea3b`; submodule:
  `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682`.

The destination-side migration rewrites `/data/tommy` defaults in operational
scripts and YAML configs to `/home/tommy`. Project symlinks point to the copied
TRAP and SimplerEnv trees. New `datasets` and `outputs` directories are empty.
`GPU1_MIGRATION_REPORT.json` lists every rewritten file.
The vendored InstructVLA `SimplerEnv/ManiSkill2_real2sim` symlink was also
repointed to `/home/tommy/SimplerEnv/ManiSkill2_real2sim`; this adds one
intentional local change inside that TRAP submodule. Its unrelated `LMUData`
symlink still points to an unavailable upstream path and is not used here.

## Not copied

Existing rollout datasets, results, evaluation frames, generated patches,
run outputs, Python environments, caches, and model weights remain on the old
server. The InstructVLA checkpoint and auxiliary assets alone occupy about
16.6 GB there; the InstructVLA Python environment occupies about 6.3 GB.
The copied code includes the pinned download script and environment setup
script so those runtime dependencies can be recreated on GPU1.
GPU1 did not have Miniconda at migration time. A code-only preflight finds the
copied camera parameters and reports the expected missing model assets and
zero attack pairs until setup and collection are complete.

## Resuming InstructVLA

1. Install Miniconda to `/home/tommy/miniconda3` if it is not already there.
2. Run `bash /home/tommy/vla_security/instructvla/setup_env.sh`.
3. Run `bash /home/tommy/vla_security/instructvla/run.sh download` to fetch the
   pinned Stage-2 checkpoint, Eagle2 model, and empty language adapter into
   `/home/tommy/models`.
4. Run `bash /home/tommy/vla_security/instructvla/run.sh preflight`.
5. Collect new orange teacher rollouts into a fresh root under
   `/home/tommy/vla_security/datasets`, then pass that root with `--data-path`
   to `run.sh opt`. The copied historical result documents refer to data on
   the old server and do not imply those samples are present here.

MolmoAct uses a separate `trap-molmoact` environment and checkpoint. Its
setup script is `/home/tommy/vla_security/scripts/setup_molmoact_env.sh`, and
its environment defaults are in `/home/tommy/vla_security/scripts/env.sh`.
Those paths have been relocated; install the environment and obtain the
checkpoint before running the MolmoAct pipeline.
