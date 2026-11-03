# Experiment snapshot — November 2, 2026

The latest checked teacher-forced evaluation had completed **45 of 50 trials**.
These are interim results, recorded at approximately 8:50 p.m. Toronto time.
The remaining layout and the control experiments were still pending.

| MolmoAct condition: orange prompt | Finished trials | Coke pick completions | Orange completions |
| --- | ---: | ---: | ---: |
| Clean | 50/50 | 0 | 30 |
| Predicted-mode joint patch | 50/50 | 0 | 0 |
| Teacher-forced joint patch | 45/50 | 45 | 0 |

The teacher-forced patch succeeded on all nine evaluated layouts, seeds
200–208. Maximum coke lifts per layout were 2.82, 2.17, 1.45, 3.93, 4.24,
2.21, 2.40, 2.97, and 4.03 cm. Forty of the 45 trials exceeded 2 cm.
Completion uses the independent simulator-style grasp/contact/>1 cm predicate.
The environment continues to designate orange.

Repeated trials are deterministic: the inspected repeated episodes had identical
reasoning, action vectors, and TCP trajectories. Treat this as evidence across
nine layouts, rather than 45 independent samples. The comparison supports the
conditioning-mode hypothesis in this setup; it does not isolate the action-loss
term because both CoT and action losses use the altered forward context.

Both modes trained for 12,000 steps on the same 510-sample coke teacher dataset,
with 14 successful teacher layouts among the 25 collected layouts. The dataset
was not filtered to successful layouts in this paper-direction comparison.

Earlier reverse-direction coke-prompt/orange-target work reached one orange
grasp/lift in ten held-out layouts, with a maximum lift of 6.3 cm, while reducing
coke completion from 10/10 to 0/10. Those lift counts were proxies and did not
establish attacker-target task completion. Earlier orange-prompt/coke-target
pilots likewise did not establish target completion.

The InstructVLA track is documented separately in `instructvla/RESULTS.md` and
`instructvla/LARGE_RUN_20261026.md`. It uses another model, action interface,
checkpoint, and environment. Do not combine those counts with MolmoAct.

Raw rollouts, videos, trained patches, and logs remain outside Git. Running
the workflow regenerates them. The historical notes under `migration/` and
`docs/history/` describe earlier states and may contain superseded hypotheses.
