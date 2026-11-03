# Upstream sources and provenance

`upstreams.lock.json` records the source revisions observed in this workspace
and on GPU1. Bootstrap downloads source into ignored directories rather than
embedding other repositories' Git histories or uploading simulator assets.

| Component | Source | Local customization |
| --- | --- | --- |
| TRAP | [Zhengxian-Huang/TRAP](https://github.com/Zhengxian-Huang/TRAP) | `patches/trap-gpu1.patch`, captured from GPU1 on November 2, 2026 |
| MolmoAct | [allenai/molmoact](https://github.com/allenai/molmoact) | Pinned upstream source; custom runtime policy is in `scripts/` |
| SimplerEnv | [allenai/SimplerEnv](https://github.com/allenai/SimplerEnv) | Pinned checkout, unchanged on GPU1 |
| ManiSkill2 real2sim | [simpler-env/ManiSkill2_real2sim](https://github.com/simpler-env/ManiSkill2_real2sim) | Pinned checkout, unchanged on GPU1 |
| InstructVLA | [InternRobotics/InstructVLA](https://github.com/InternRobotics/InstructVLA) | Lazy imports in TRAP's bundled adapter; separate workflow in `instructvla/` |
| MolmoAct2 wrapper | [kaitlynchuchu/VLA](https://github.com/kaitlynchuchu/VLA) | `patches/molmoact2-diagnostics.patch` and `diagnostics/molmoact2/` |
| POEF | [ZJUshine/POEF](https://github.com/ZJUshine/POEF) | Pristine reference checkout; no local source changes to publish |

The TRAP patch includes int8 loading, trajectory conditioning modes, sequential
gradient accumulation, scheduled loss weights, optimization image resizing,
and table segmentation masks, plus the earlier lazy InstructVLA imports.
The original working tree and nested Git metadata are left in place locally;
they are ignored by this new parent repository.

To fetch another track:

```bash
python3 scripts/bootstrap_sources.py --track instructvla
python3 scripts/bootstrap_sources.py --track molmoact2
python3 scripts/bootstrap_sources.py --track poef
```

Upstream code, checkpoints, and assets remain subject to their respective
authors' license and usage terms. This repository does not assign a new license
to those dependencies. The synthetic CPU demo is independent of the VLA
checkpoints and is not evidence for the paper's reported attack success rates.
