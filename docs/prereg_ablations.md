# Pre-registration: ablations, robustness and external validity (Anirudh's workstream)

Registered 2026-11-24 and revised the same day after the critic's and the
verifier's reviews, before any ablation result exists. Every result here is
secondary to the registered primary endpoints (docs/prereg_E4.md, E6):
nothing in this file changes a primary test, its family or its data. A new
ablation is added only by a dated amendment at the end of this file,
committed before its first run. Plan and handoff: docs/HANDOFF_anirudh.md.

## Shared rules

* Every record carries the git hash, packages, model revision, GPU and
  backend (leaguetst/architect/llm.py environment). Seeds are fixed in the
  command.
* Endpoint on ILI: held-out AUC regret over budgets 1 to 100 (as E3), in
  pooled seed-SD units and in raw MSE. ETTh1 has one seed per config, so
  ETTh1 replays are descriptive only (A18).
* Tests: seed-level permutation tests (exact up to 250k relabellings), Holm
  within each family below. Every test also reports the difference with a t
  interval and the smallest difference it could detect at 80% power.
  Planning estimate (critic, from E3): the per-seed SD of ILI AUC is about
  0.5 SD units, so 30 against 30 seeds with Holm over 3 detects about 0.42,
  and 100 against 100 about 0.23. Families are sized to that: few tests,
  more seeds, trends instead of pairwise sweeps.
* A GPU arm and its reference run on the same backend and code digests.
* Each GPU arm has its own Modal item (ABL_G_<arm>, ABL_P_<arm>), never
  reused for another configuration, so adapters, runs and launch locks
  cannot mix.
* Forking paths closed in advance: no AB-R variant replaces the E2 referee
  or feeds E5; anything run after the E4 code unfreezes (LoRA rank, rival
  pool size, sampling temperature, all constants in registered E4 code) is
  exploratory and gets no test; the A26 worst-case analysis is
  leaguetst/sensitivity.py, sha256
  e9ff5fe587b637cfc4d5bc7755e90be921d0a097347acbd9a88ed5ecc328c59b,
  committed (c3f69d1) before any E4 result.

## Tested families

### AB-L: did GRPO learn? (CPU, from the E4 training logs)

For every E4 policy, the Spearman slope of the group's mean reward over the
60 steps. Test per source table: the league policies' slopes against zero
(sign-flip permutation test over policies). Holm over 2 (ILI source, ETTh1
source). Descriptive beside it: the best-in-history of the last 10 steps
against prompted Qwen's best at the same epoch budget on the source table.
Without this, "RL did not help" cannot be told from "RL did not learn".

### AB-G: GRPO dose-response (GPU)

Source ETTh1, target ILI (held-out target AUC), league rivals, training
seeds 100 to 109, deployed at budget 100, E4 recipe otherwise.

| Trend | Levels (ordinal) | Arms to run |
|---|---|---|
| Training steps | 0 (ABL_P_ref on ILI, the untrained model), 30, 60 (reference), 120 | ABL_G_steps30, ABL_G_steps120 |
| KL weight beta | 0, 0.04 (reference), 0.1 | ABL_G_beta0, ABL_G_beta01 |

Test per trend: the seed-level Spearman correlation between level rank and
AUC, permutation test. Holm over 2. Reference: Soumitra's E4 J2 league arm
if it ran on the same backend and E4 is complete under A26; otherwise
ABL_G_ref, the same recipe, run by Anirudh.

### AB-P: prompted-architect ablations (GPU)

Prompted Qwen, 30 search seeds per table, rerun under the pinned A25 code
as its own reference (ABL_P_ref; the E3 runs predate the pins). Arms:
anon (field names replaced by neutral labels: does the LLM use forecasting
knowledge?), repair (unconstrained decoding with nearest-feasible repair:
what does constrained decoding buy?), and model scale (Qwen2.5-0.5B and
3B-Instruct around the 1.5B reference, revisions pinned by amendment
before launch; needs the WP4 script). ILI tests: anon against ref, repair
against ref, and the scale trend; Holm over 3. ETTh1 descriptive. This
supersedes the repair ablation of docs/prereg_E4.md (5 ILI seeds).

### AB-C: factorized-controller ablations (CPU)

leaguetst.search.replay.factorized_controller with league rivals, 100
search seeds per arm, ILI held-out AUC; reference: its defaults (group 8,
4 rivals, learning rate 0.1, snapshot every 2, entropy 0.01). Arms: 1
rival, 8 rivals, snapshot every step. Holm over 3. ETTh1 descriptive.

### AB-X: external HPO baselines (CPU, now)

DEHB, SMAC (random-forest Bayesian optimisation) and Hyperband on the same
oracle, budgets and charging rule as E3, 30 seeds, against E3's random
search on ILI held-out AUC; Holm over 3; ETTh1 descriptive. They read only
the validation tables, so nothing is gained by waiting for the test. Package
versions are pinned by amendment before the first run.

## Descriptive analyses (no tests)

* AB-R, referee: features (config only, curves only, both), model
  (gradient-boosted trees, ridge, random forest), training size (100, 300,
  1,000), k in 1 to 3; Kendall tau against the multi-seed mean with a
  bootstrap interval, both tables.
* AB-N, one-seed bias: the ILI table cut to seed 2021 with ETTh1's
  observation-noise rule; every E3 method (30 seeds); in-sample regret
  against the full table's held-out regret.
* AB-Q, noise-free lookups: every E3 method replayed on ILI with 3-seed
  means as lookups and regret on the same means. If methods separate here
  but not on fresh seeds, the plateau of near-equal configs explains the
  null.
* ST: the detectable-effect table for every registered comparison; regret
  in raw MSE; the ETTh1 replay with a robust (MAD) seed SD; interquartile
  mean and probability-of-improvement profiles with stratified bootstrap
  intervals. Any equivalence margin is labelled post hoc (the pick results
  were seen before this file).
* RQ3: ANOVA interactions, cross-table agreement of field shares, and the
  REVIN validation check against results/revin_prediction.json (the test
  part stays with Diya's single evaluation on 30 Nov).
* Exploratory after the E4 code unfreezes: LoRA rank, rival pool size,
  sampling temperature (a constant in training today).

## Phase B, after the 30 Nov test (toward the paper), in this order

Each item is registered by amendment before it runs.

1. ETTh1 seeds 2022 and 2023 for all 2,368 configs, so ETTh1 replays can
   carry claims (A18).
2. A third dataset from another domain (not ETTh2, which shares ETTh1's
   source), 3 seeds (about $42), chosen after checking its licence, under
   its own lock and test protocol, prepared by a person, never by Claude.
3. Benchmark release: the result tables, a loader that downloads and splits
   the original data itself (ETDataset is CC BY-ND 4.0, so the split CSVs
   are never redistributed), Croissant metadata, hosting on an ML dataset
   site and executable code, as the NeurIPS Datasets and Benchmarks call
   requires.
4. The wider design space of A16 (docs/wide_space.md), on a costed subset.
5. LLM-based optimisers from the literature (WP0) as further baselines.

## Amendment AB1 (2026-11-24 evening, before any ablation result exists)

Anirudh confirms in his first commit after this amendment that no ablation
result existed before it, or names any that did.

* AB-G reference. E4 completed under A26 on Soumitra's Modal workspace
  (PR #1, 70461a8). The reference of every Modal AB-G arm, the step trend
  included, is the J2 league arm; ABL_G_ref is not run on Modal.
* Retried runs. A29 and its addendum (docs/prereg_E4.md) apply to every
  ABL_* GPU item on Modal: retried runs are named with dashboard
  screenshots, the tests keep them, and the A26 worst case
  (leaguetst/sensitivity.py) treats them as failed. The hand-stop is 2.5
  hours for ABL_G_steps120 (its batches take about an hour) and one hour
  elsewhere.
* Who runs what. AB-L: Soumitra. AB-P ref, anon and repair and the two step
  arms: Anirudh on Modal. Measured cost ($22.17 for 45 policies, about
  twice the $11 estimate in docs/HANDOFF_soumitra.md) leaves no room on
  Anirudh's credit for AB-P scale and the KL arms (beta 0 and 0.1), so they
  move to Soumitra. He runs each of the two families before 2 Dec 12:00 ET
  on Modal if his credit shows its estimate plus $5 free, otherwise on his
  local GPU through scripts/ablate_llm.py (a new launcher on the unchanged
  runner that imports llm_local.py's executor). Amendment AB2 registers the
  launcher's sha256, the model revisions, and each family's backend, card
  and jobs per GPU before the first run. A family on the local GPU reruns
  its reference on the same card (ABL_G_ref_local, the E4 recipe;
  ABL_P_ref_local, Qwen2.5-1.5B), and local references serve only local
  arms. No arm waits for the December credit.
* Backend switching. An item moves to another backend only if it was never
  launched; a launched item is never rerun elsewhere.
* Hardware check, descriptive. ABL_G_ref_local against the J2 league arm
  and ABL_P_ref_local against ABL_P_ref, on the same seeds, are reported as
  a hardware check, with no test.
* Code. The ten files in docs/e4_rerun.sha256 stay unchanged until the last
  AB-G arm has run; an AB-G arm run under other digests is exploratory.
* GPU type. Modal serves both A10 and A10G for the A10G request (both occur
  in the reference arm). The GPU name is recorded per run and reported per
  arm; no test conditions on it.
* Missing arms. An arm with any job unfinished at 2 Dec 12:00 ET counts as
  not run and is reported so; its family keeps its registered Holm size,
  the missing test counted as not rejected, and the report's decisions are
  not revised if the arm lands later.
* AB-L, descriptive comparison. best_in_history resets every training
  episode, whose budget (3000) equals a prompted search's, so the
  comparison uses each policy's final episode against prompted Qwen at the
  same spend; null rewards are skipped. The tested slope is unchanged.
