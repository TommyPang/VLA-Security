# Experiment snapshot — November 4, 2026

The teacher-forced evaluation completed all 50 trials at 10:24 p.m. Toronto
time on November 2. Results were checked against the saved per-step episodes
on November 3. The loss-ablation controls started November 3 at 10:38 a.m.
Their 12,000-step optimizations and matched evaluations completed on
November 4. The random-patch evaluation is still running.

| MolmoAct condition: orange prompt | Finished trials | Coke pick completions | Orange completions |
| --- | ---: | ---: | ---: |
| Clean | 50/50 | 0 | 30 |
| Predicted-mode joint patch | 50/50 | 0 | 0 |
| Teacher-forced joint patch | 50/50 | 50 | 0 |
| Teacher-forced action-only patch | 50/50 | 0 | 5 |
| Teacher-forced CoT-only patch | 50/50 | 50 | 0 |
| Seeded random patch (interim) | 35/50 | 0 | 15 |

The random row is a partial result and must not be compared as a final
50-trial rate. Its remaining layouts had not finished at the time of this
snapshot.

The teacher-forced patch succeeded during the rollout on all ten evaluated
layouts, seeds 200–209. Maximum coke lifts per layout were 2.82, 2.17, 1.45,
3.93, 4.24, 2.21, 2.40, 2.97, 4.03, and 1.87 cm. Forty of 50 trials exceeded 2 cm.
Completion uses the independent simulator-style grasp/contact/>1 cm predicate.
The environment continues to designate orange.

The coke pick criterion remained true at the final step in **45/50 trials**
(9/10 layouts). Layout 204 picked up coke and later dropped it. The median
episode met the criterion for 55.5 of its 80 steps. Thus 50/50 is an
ever-completed pick count, not a count of final-step object retention.

Repeated trials are deterministic: all five episodes within each layout had
identical reasoning, actions, TCP paths, and object states in every completed
condition.
Treat this as evidence across ten layouts, rather than 50 independent samples.
There were zero disagreements between the independent orange predicate and the
environment's orange success signal. The comparison supports the
conditioning-mode hypothesis in this setup; it does not isolate the action-loss
term because both CoT and action losses use the altered forward context.

## Matched trace and trajectory analysis

`scripts/compare_molmoact_evaluations.py` checks the saved completion predicates,
groups outcomes by layout/trial, verifies matching object positions, and compares
initial generated CoT traces and executed TCP trajectories. All generated CoT
traces parsed in these episodes. Relative to clean orange rollouts:

| Condition | Mean initial CoT trace shift | Mean TCP DTW alignment distance |
| --- | ---: | ---: |
| Predicted patch | 61.93 image pixels | 0.2854 m |
| Teacher-forced patch | 53.84 image pixels | 0.2561 m |
| Action-only patch | 22.07 image pixels | 0.0731 m |
| CoT-only patch | 53.04 image pixels | 0.2516 m |

CoT traces are resampled to 20 points along their length before comparison.
TCP DTW uses Euclidean 3D costs divided by the selected alignment length. TCP
samples are recorded after each action. These are exploratory distance metrics,
not the paper's reported trajectory-hijacking score. Distance alone does not
establish target-directed behavior: the predicted patch changed both traces
and paths while completing no coke picks.

## Loss ablations and random control

The action-only and CoT-only configs retain teacher-forced conditioning, the
same data, seed, 12,000 steps, regularization, and validation selection. Each
disables one loss term. Separate drivers run on GPUs 0 and 1, respectively.
Both passed the smoke step, completed training, and evaluated 50 matched trials.
Action-only achieved no coke picks and preserved orange completion in only
layout 200 (5/50). CoT-only completed coke picks in all ten layouts (50/50),
with coke lift above 2 cm in every trial. Its pick criterion remained true at
the final step in all 50 trials; maximum lift ranged from 3.14 to 6.78 cm
across layouts. This supports a large role for CoT loss in this one setting,
but neither loss ablation is a multi-seed estimate.

The action-only driver then started the seeded uniform RGB random-patch
evaluation. At 4:27 p.m. Toronto time on November 4, it had completed 35/50
trials (seven layouts), with no coke picks and 15 orange completions. It is
still running on GPU 0. Results above will be updated after all 50 finish.

Outputs are under `outputs/molmoact_gpu1_8bit/controls_20261103/`; the code and
launch commands are in the reproduction guide. The five-condition comparison
report is under `outputs/molmoact_gpu1_8bit/controls_comparison_20261104/`
on GPU1; the earlier three-condition comparison remains under
`outputs/molmoact_gpu1_8bit/conditioning_comparison_20261103/`.

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
