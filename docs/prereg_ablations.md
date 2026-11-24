# Pre-registration: ablations, robustness and external validity (Anirudh's workstream)

Registered 2026-11-24, before any ablation result exists. Every result here
is secondary to the registered primary endpoints (docs/prereg_E4.md, E6):
nothing in this file changes a primary test, its family or its data. A new
ablation is added only by a dated amendment at the end of this file,
committed before its first run. Plan and handoff: docs/HANDOFF_anirudh.md.

## Shared rules

* Seeds: at least 10 search or training seeds per GPU arm and 30 per CPU
  arm, fixed in the command; every record carries the git hash, packages,
  model revision, GPU and backend (leaguetst/architect/llm.py environment).
* Endpoint on ILI: held-out AUC regret over budgets 1 to 100 (as E3), in
  pooled seed-SD units and in raw MSE. ETTh1 has one seed per config, so
  ETTh1 replays are descriptive only (A18) unless fresh-seed picks exist.
* Tests: two-sample seed-level permutation test of each arm against its
  reference (exact up to 250k relabellings), Holm within each family below.
  Every test also reports the difference in means with a t interval and the
  smallest difference the design could detect at 80% power (simulated from
  the observed variances), so a null result is read with its resolution.
* A GPU arm and its reference run on the same backend. If Anirudh and
  Soumitra both use Modal (A10G, same code digests), the reference for AB-G
  is Soumitra's E4 J2 league arm; otherwise Anirudh runs it.
* Each GPU arm has its own Modal item (ABL_G_<arm>, ABL_P_<arm>), never
  reused for another configuration, so adapters, runs and launch locks
  cannot mix.

## AB-G: GRPO ablations (GPU, Phase 1 now, Phase 2 after E4)

Source ETTh1, target ILI (the direction whose target AUC is held out), 10
seeds per arm (100 to 109), league rivals, deployed at budget 100.
Reference: the E4 recipe (KL beta 0.04, 60 steps, learning rate 1e-4,
group 8, 4 rivals, snapshot every 5 steps, temperature 1.0, LoRA rank 16).

| Phase | Arm | Change | Runs with |
|---|---|---|---|
| 1 | beta0 | no KL term (beta 0) | scripts/modal_llm.py transfer --beta 0 |
| 1 | beta01 | beta 0.1 | --beta 0.1 |
| 1 | steps30 | 30 GRPO steps | --steps 30 |
| 1 | steps120 | 120 GRPO steps | --steps 120 |
| 2 | group4 | group 4 | new script (job field group) |
| 2 | rivals8 | 8 rivals | new script (job field n_rivals) |
| 2 | snap1 | snapshot every step | new script (job field snapshot_every) |
| 2 | temp07 | sampling temperature 0.7 in training and deployment | new script (job field temperature) |

Family AB-G1: the four Phase 1 arms against the reference, Holm over 4.
Family AB-G2: the four Phase 2 arms, Holm over 4. Phase 2 runs only after
its script passes review and its digest is added here by amendment.
LoRA rank and the rival-pool size (32) are constants in registered E4 code;
they may be ablated only after the E4 rerun is complete, by amendment.

## AB-P: prompted-architect ablations (GPU)

Prompted Qwen, 10 search seeds per table, Modal item ABL_P_<arm>, against
qwen_prompted (E3, A10G). Phase 1: anon (field names replaced by neutral
labels: does the LLM use forecasting knowledge?) and repair (unconstrained
decoding with nearest-feasible repair: what does constrained decoding buy?).
Phase 2: model scale, Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct, each
pinned to the revision resolved before launch and written here by amendment.
Family AB-P: ILI tests only (2 in Phase 1, 2 in Phase 2), Holm over 4; ETTh1
descriptive.

## AB-C: factorized-controller ablations (CPU)

leaguetst.search.replay.factorized_controller with league rivals, 30 search
seeds, ILI held-out AUC; reference: its defaults (group 8, 4 rivals,
learning rate 0.1, snapshot every 2, entropy 0.01). Arms: rivals 1, 2 and 8;
group 4 and 16; snapshot every 1 and 5; entropy 0 and 0.05; learning rate
0.03 and 0.3. Family AB-C: Holm over these 11 tests. The same arms on ETTh1
are reported descriptively.

## AB-R: referee ablations (CPU)

The E2 setup (scripts/fidelity.py draw: 300 training, 150 evaluation
configs). Arms: features (config fields only, curves only, both), model
(gradient-boosted trees, ridge regression, random forest), training size
(100, 300, 1,000), k in 1 to 3. Endpoint: Kendall tau against the
multi-seed mean, with a bootstrap interval over evaluation configs, on both
tables. Descriptive: no family, no claim.

## AB-N: how much does a one-seed table bias a replay? (CPU)

The ILI table restricted to seed 2021, with ETTh1's observation-noise rule,
replays every E3 method (30 seeds); its in-sample regret is compared with
the full table's held-out regret for the same runs. This measures the
in-sample bias that A18 guards against on ETTh1. Descriptive.

## ST: statistical robustness (CPU)

* The smallest detectable effect of every registered comparison (E3 claims,
  pick reseeds, E4 families) at 80% power, by simulation from the observed
  per-seed variances.
* Regret in raw MSE beside SD units; the ETTh1 replay summary recomputed
  with a robust (median absolute deviation) seed SD as a sensitivity check.
* Aggregate profiles across both tables: interquartile mean and probability
  of improvement over random search with stratified bootstrap intervals.
* The worst-case sensitivity analysis of A26, as a tested function that
  Diya's side calls on the E4 families.
* Equivalence margins, if reported, are labelled post hoc: the pick
  results were seen before this file.

## RQ3 extras (CPU)

* Pairwise interactions in the E8 ANOVA, and whether the ranking of design
  fields agrees between the two tables (Spearman over field shares).
* The RevIN study's validation part (A23): the 20 REVIN pairs at 100 epochs
  against their 30-epoch values, checked against results/revin_prediction.json.
  The test part stays with Diya's single evaluation on 30 Nov.

## Phase B, after the 30 Nov test (toward the paper)

Each item is registered here by amendment before it runs.

* External baselines on the same oracle: published LLM-based optimisers
  and multi-fidelity HPO methods that the lit-scout opens and cites.
* A third dataset table (for example ETTh2) under its own lock and
  test-split protocol, prepared by a person, never by Claude.
* Benchmark release: the tables with a datasheet, licence and loader.
* The wider design space of A16 (docs/wide_space.md).
