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
  move to Soumitra. He runs each of the two families by 28 Nov 23:59 ET
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
* Missing arms. An arm with any job unfinished at 28 Nov 23:59 ET counts
  as not run and is reported so (the owner moved this deadline from 2 Dec
  on 24 Nov, before any result); its family keeps its registered Holm size,
  the missing test counted as not rejected, and the report's decisions are
  not revised if the arm lands later.
* AB-L, descriptive comparison. best_in_history resets every training
  episode, whose budget (3000) equals a prompted search's, so the
  comparison uses each policy's final episode against prompted Qwen at the
  same spend; null rewards are skipped. The tested slope is unchanged.

## Amendment AB2 (2026-11-25, before any run of the model-scale or KL arms)

Soumitra's side, under AB1 and docs/LEG2_soumitra.md L4. No result of any
item below exists when this is committed.

* Backend, chosen per family by the L4 rule and never revisited. J3 was not
  run (A30, commit 7808ea6). Soumitra's Modal workspace shows about $8.15 of
  free credit (metered $21.87 this month, billed $0.00). The model-scale
  family needs about $8 plus $5 free and the KL family about $10 plus $5
  free, so neither fits: both families run on the local GPU, an NVIDIA
  GeForce RTX 4080 Laptop GPU (12 GB, 12,282 MiB; driver 595.97; WSL2;
  compute capability 8.9; torch 2.15.0+cu132, CUDA runtime 13.2, cuDNN
  92600), one job at a time (--per-gpu 1). Nothing in AB2 runs on Modal.
* Packages: requirements-llm.txt as pinned by A25 (torch 2.15.0,
  transformers 5.24.0, peft 0.22.0, accelerate 1.16.0, huggingface_hub
  2.4.0); every record carries its env block (A25).
* Model revisions (Hugging Face commits, read on 2026-11-24, each the Hub
  head that day): Qwen/Qwen2.5-0.5B-Instruct
  7ae557604adf67be50417f59c2c2f167def9a775; Qwen/Qwen2.5-1.5B-Instruct
  989aa7980e4cf806f80c7fef2b1adb7bc71aa306 (the A25 pin);
  Qwen/Qwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1.
* Memory check, before this amendment and producing no search result: one
  prompted step of the 3B model under the registered settings (bf16, group
  8, constrained decoding, temperature 1.0, 112 new tokens, full 24-line
  history, about 1,970 prompt tokens) peaked at 7.62 GiB allocated and
  8.02 GiB reserved on the card, so it runs at one job with the registered
  settings unchanged. Output: results/logs/ab2_3b_memory_probe.txt.
* Code. The launcher is scripts/ablate_llm.py, sha256 3919d68337302590eda42f53ca10158ca27766bb488026b5a4fc8373c477b786 (commit 79977de); the
  queue that runs the items one after the other is
  scripts/ablations/run_l4_queue.sh, sha256 f2f71ecb28e664afdceb6914b1047a46f248b2fdf2bcd05b1c87bbfbee6d8d9b. The launcher builds
  plans with leaguetst/architect/runner.py and imports scripts/llm_local.py's
  executor and GPU check; the ten files of docs/e4_rerun.sha256 are
  unchanged, and the queue refuses to start if their check fails or if
  leaguetst/ or scripts/ has uncommitted changes. An item is DONE only if
  its launch exits 0 with no errors.jsonl row; otherwise it is PARTIAL or
  FAILED and the next relaunch reruns it, keeping finished seeds and
  adapters and never retraining an on-policy stop.
* Items, outputs under results/runs/llm_local/<item>/<name>/ and logs under
  results/logs/, in this order:

| Item | Command (scripts/ablate_llm.py ... --per-gpu 1) | Method |
|---|---|---|
| ABL_P_ref_local | prompted --dataset ILI (and later ETTh1) --seeds 30 --model Qwen/Qwen2.5-1.5B-Instruct | qwen_prompted_1.5b_local |
| ABL_P_05b_local | the same with Qwen/Qwen2.5-0.5B-Instruct | qwen_prompted_0.5b_local |
| ABL_P_3b_local | the same with Qwen/Qwen2.5-3B-Instruct | qwen_prompted_3b_local |
| ABL_G_ref_local | transfer --source ETTh1 --target ILI --arms league:10 (E4 recipe) | grpo_league_from_ETTh1 |
| ABL_G_beta0_local | the same with --beta 0 | grpo_league_from_ETTh1 |
| ABL_G_beta01_local | the same with --beta 0.1 | grpo_league_from_ETTh1 |

  The queue runs the three prompted items on ILI (ref, 0.5B, 3B), then the
  same three on ETTh1 (ILI is tested, ETTh1 descriptive), then the three KL
  items (ETTh1 to ILI). Search seeds 0 to 29;
  GRPO training seeds 100 to 109, deployed at seeds 0 to 9, budget 100.
* Tests, unchanged from AB-P and AB-G: on ILI held-out AUC, the scale trend
  (0.5B, 1.5B, 3B) uses ABL_P_ref_local as its 1.5B level and the KL trend
  (beta 0, 0.04, 0.1) uses ABL_G_ref_local as its 0.04 level; local
  references serve only local arms (AB1). ABL_G_ref_local against the J2
  league arm and ABL_P_ref_local against ABL_P_ref are the AB1 hardware
  check, descriptive.
* Hand-stop, on record progress (prompted searches print nothing between
  start and end, so a log line is not the signal). Prompted item: no new row
  in its *_runs.jsonl for one hour. Transfer item: no new GRPO step row in
  its log and no new training.jsonl or *_runs.jsonl row for 2.5 hours. A
  hand-stop kills the queue, is logged as a line in
  results/logs/l4_queue_status.txt, and is followed by a relaunch.
* Stale lock. A killed or suspended launch (hand-stop, sleep, WSL shutdown)
  can leave <item>/<name>/driver.lock behind, and the next launch of that
  item then fails. Procedure: confirm with ps that no ablate_llm.py
  process runs, delete that driver.lock, relaunch the queue, and log the
  deletion in the status file. This is the procedure registered here, not
  a workaround of the lock.
* Retried runs (A29 adapted to one local GPU). An on-policy stop is
  reported and never retrained. At --per-gpu 1 a run "started" in a killed
  launch is the first job of that launch's pending list with no training.jsonl,
  errors.jsonl or *_runs.jsonl row written by that launch; a GRPO policy that a later launch retrains after that, with
  no errors.jsonl line naming it written after that start, counts as
  retried: kept in the tests and
  treated as failed in the A26 worst case. A rerun search (prompted or
  deployment) is not retried: it has no on-policy stop to lose.
* Completion, by counts at bring-back (the queue's DONE, PARTIAL and FAIL
  status lines are operational only). A prompted item is complete on a
  table when its runs file holds 30 distinct seeds. A transfer item is
  complete when all 10 policies have an outcome (a saved adapter with its
  training record, or an on-policy stop in errors.jsonl) and every
  non-stopped policy is deployed. A job that failed without an on-policy
  stop (out of memory, a crash) is relaunched with the registered settings
  unchanged until 28 Nov 23:59 ET; nothing (history lines, group size, new
  tokens, model, card, backend) is ever changed to make a job fit or
  finish, and no item is analysed on fewer seeds as a fallback.
* Each prompted item's ILI table and ETTh1 table is its own arm under AB1:
  an arm with any job unfinished at 28 Nov 23:59 ET counts as not run. An
  unfinished ETTh1 (descriptive) table does not make the tested ILI table
  not run. A transfer item is one arm.
* Holm sizes are unchanged: AB-P stays Holm over 3 (anon and repair against
  Modal ABL_P_ref, run by Anirudh; the scale trend against ABL_P_ref_local)
  and AB-G stays Holm over 2 (the steps trend against the J2 league arm; the
  KL trend against ABL_G_ref_local). A trend with any of its arms not run
  (any one level missing) is not tested and counts as not rejected (AB1);
  it is never tested on fewer levels.
* Launch order: the queue is launched only with HEAD at the commit that
  adds AB2 or a descendant of it, so every record's git hash postdates this
  registration.
* Evidence: results/logs/ab2_modal_billing.txt (billing summary; the free
  credit was read from the dashboard, as the E4 screenshots show) and
  results/logs/ab2_3b_memory_probe.txt (the memory check).

## Amendment AB3 (2026-11-25, Anirudh, before any ablation result exists)

AB2 is reserved for Soumitra's launcher. Nothing below changes a primary
test, its family or its data.

What existed before this amendment. No ablation arm had been run. On
24 Nov, 14:02 to 14:06 ET (before AB1), Anirudh's session made three scratch
calculations on committed data, never committed: that AB-N's noise SD is
undefined as registered; that untrained prompted Qwen's reward rises within a
search (E3, ten searches per table); and what the AB-L slope looks like for a
model that does not learn (the same E3 searches chained). The scripts written
here were tested on synthetic tables only. When writing AB3 the author had read
the E3 and pick summaries, results/independent/E4_soumitra.json and AB-L
(3cb92a1).

* Where ablation outputs live. scripts/picks.py and scripts/e5_referee.py read
  every folder under results/search/, and the ablation items reuse E4's and
  E3's method names. With dummy files, merging the ABL_* folders named in the
  handoffs raised grpo_league_from_ETTh1 from 10 to 50 runs and qwen_prompted
  from 10 to 40 in the ILI picks, and would add methods to E5's Bonferroni
  count. Every output of this workstream therefore goes to results/ablations/,
  never to results/search/. Soumitra's ABL_* items meet the same problem; it is
  logged in docs/ISSUES.md for Diya and Soumitra.
* AB-C. No change. Reference and arms run on both tables; ETTh1 descriptive.
* AB-Q. The noise-free ILI table has one pseudo-seed per config: the best value
  is the mean of the three seeds' best values; the charge is the mean epochs
  run, rounded half up; the curves are seed means, a seed that stopped earlier
  carrying its last value forward. Regret is on the same means, in the real
  table's pooled seed SD. The ten CPU methods of E3, 30 seeds. Prompted Qwen
  needs a GPU: it is added only if the Mac item ABL_Q_prompted_mps runs.
* AB-N. A table cut to one seed has no seed SD, so the registered rule gives
  NaN. Two variants, both descriptive: etth1_like (seed 2021 for every config,
  seeds 2022 and 2023 for the 100 best by seed 2021, as the ETTh1 table was
  built; noise SD by the ETTh1 rule) and strict (seed 2021 only; noise SD = the
  full ILI table's pooled SD). Each run is scored as its cut table would report
  it and held out on the full table (the config's seeds the run did not see).
  The ten CPU methods of E3, 30 seeds.
* AB-X. smac 2.4.1, dehb 0.1.2 and ConfigSpace 1.2.2 in their own environment
  (requirements-ablations.txt; dehb needs numpy below 2). Definitions are in
  scripts/ablations/external_hpo.py: SMAC's random forest and EI with a random
  initial design of 10 and the acquisition maximised over every feasible config
  not yet evaluated; SMAC's Hyperband facade with random sampling; DEHB's
  ask/tell with TPE's nearest-unseen repair; rungs 1, 3, 10 and 30 epochs, eta
  3, through Oracle.partial, so a promoted config continues its run. SMAC's
  default acquisition search was replaced after it was seen, on a synthetic
  table, to fall back to random sampling in a space this small.
* AB-R. The E2 draw; training sets are the first 100, 300 and 1,000 of the
  configs after the 150 scored ones. Target: log of the seed-2021 final value.
  Ridge with a cross-validated penalty; random forest of 300 trees.
* ST. Detectable differences use the observed per-arm SDs, the registered Holm
  size and the normal approximation at 80% power. The robust ETTh1 SD is
  1.4826 x median|d| / sqrt(2), d the difference of a config's seed-2022 and
  seed-2023 values; it replaces the pooled SD as noise and as unit. E6 appears
  only as a planning figure from validation errors.
* RQ3. Interactions are the extra share of variance a field pair adds to the
  additive least-squares model of the config means.
* AB-L calibration, exploratory and post hoc (AB-L was already pushed): the
  raw and position-adjusted slopes of untrained prompted Qwen chained over 60
  steps with AB-L's episode resets. It changes no AB-L number.
* LLM arms on a Mac. Anirudh does not launch WP3 on Modal. The registered items
  (ABL_P_ref, ABL_P_anon, ABL_P_repair, ABL_G_steps30, ABL_G_steps120) are
  therefore not launched by Anirudh and stay free for Soumitra's RTX queue
  (docs/LEG2_anirudh.md); if nobody runs them they are reported as not run and
  their families keep their registered sizes. Separately, Anirudh runs the same
  questions on an Apple M4 Pro (torch MPS): scripts/ablations/llm_mps.py, items
  ABL_P_ref_mps, ABL_P_anon_mps, ABL_P_repair_mps (30 seeds, both tables) and
  ABL_G_steps_mps (10 policies, training seeds 100 to 109, ETTh1 source, ILI
  target). The owner's decision names Modal or a CUDA GPU, not a Mac, and the
  code is not the frozen E4 code, so every Mac result is exploratory, as AB1
  says of arms run under other digests. It replaces no registered item. Whether
  the team uses it, and how, is Diya's decision.
  * Every Mac arm and its reference run on the same Mac with the same code.
    Model, revision, prompt, trie, episode rules, recipe and seeds are E4's.
    Differences: the prompt is run once and its cache shared by the group (the
    same quantity; checked against the frozen formula), sampling uses a CPU
    generator, no gradient checkpointing.
  * Step trend: each policy is trained once to 120 steps and its adapter saved
    at 30 and 60, because a run with the same seed passes through those states;
    levels 0 (ABL_P_ref_mps on ILI), 30, 60 and 120. Levels of one policy are
    therefore dependent. Beside the trend: the same trend with 10 seeds per
    level, and over the trained levels only.
  * Tests as registered (anon and repair against ref, Holm size 3; step trend,
    Holm size 2), labelled exploratory. The on-policy check (0.05) is kept; a
    policy it stops is reported and not retrained.
  * Hardware check, no test: ABL_P_ref_mps against E3's ten A10 searches, and
    the 60-step Mac policies against the J2 league arm, on the same seeds.

### AB3 addendum 1 (2026-11-25, exploratory, after the CPU results were seen)

Added after the verifier's review of the first CPU results, before the new
outputs exist. All of it is exploratory and changes no registered test, family
size or endpoint.

* AB-Q hold-out: the noise-free replay with the lookups replaced by the mean of
  two seeds and the incumbent scored on the third, each ILI seed held out in
  turn, 30 search seeds, ten CPU methods; the unit is the search seed. (The
  verifier ran a scratch version first: TPE against random was -0.39 SD
  in-sample and -0.05 held out.)
* AB-N: the optimism of regret is also split into the optimism of the incumbent's
  value and the shift of the reference (the best value of each table).
* AB-R: every variant is also scored against the mean of seeds 2022 and 2023,
  which the referee never saw; the AB3 target (the multi-seed mean, which
  contains the seed-2021 value) stays the registered one.
* RQ3: conditional shares (what a field adds to a model that has the other
  seven), the seed-noise share of the config means on ILI, and interaction
  increments per single seed.
* AB-L calibration: the exact sign-flip p of the observed position-adjusted
  slopes is reported beside the null-based p, which rests on ten reused
  untrained searches.
* Records: every run record carries the git hash of the code that wrote it.

### AB3 addendum 2 (2026-11-25, exploratory, after the critic's review and before its outputs exist)

* AB-X DEHB, exploratory variant dehb_vector: a proposal that the repair rule
  replaces has its stored vector replaced by the evaluated config's, so DEHB's
  population holds what was trained, as TPE discards its raw proposal. The
  registered dehb arm is unchanged; AB-X's registered tests do not include the
  variant. Every AB-X run now counts its proposals and repairs.
* AB-Q hold-out: per-held-seed differences and the number of distinct final
  configs are reported beside the interval.
* AB-L calibration: the Mac reference (ABL_P_ref_mps) enters the null only when
  all 30 of its searches exist.
* ST: the E4 comparison that rejects (league worse than warm-started TPE) is
  labelled as such, and ETTh1 detectable differences are also given in robust
  seed SD. AB-N's split, AB-R's independent-seed target and RQ3's conditional
  shares are labelled exploratory in the report.

### AB3 addendum 3 (2026-11-25, exploratory, after the Mac LLM results were seen)

Written after the first run of scripts/ablations/llm_tests.py on the finished
Mac runs: its level means, p values and the step trend were seen. The two
analyses below are post hoc and exploratory; they change no registered test,
family size or endpoint, and their outputs do not exist yet.

* Why. The step trend's 60 values include three checkpoints of the same ten
  policies, and the untrained level (30 searches) differs from the trained
  levels by training at all, not only by more of it. The registered permutation
  p treats the values as exchangeable, so it is not valid across the dependent
  checkpoints, and a trend that includes level 0 cannot separate "trained at
  all" from "trained longer".
* Training at all: each policy's ILI held-out AUC averaged over its three
  checkpoints (10 values, one per policy) against the 30 untrained searches;
  permutation p, Welch interval and the detectable difference, as
  common.compare.
* Dose: per policy, the Spearman correlation of AUC with the checkpoint (30, 60,
  120); exact two-sided sign flip over the ten policies, uncorrected. Three
  ordered points carry little power, so a null here is weak.
* The report states both beside the registered trend. Wording: the Mac p values
  are written as "< 0.0001" when they sit at the Monte Carlo floor.

### AB3 addendum 4 (2026-11-25, exploratory, after the critic's review of the Mac LLM runs)

Post hoc: the critic's report gave the RevIN-off shares, the paired gap and the
ETTh1 differences below before this was written, and the code below was run once
before this commit to check that it works, so its outputs were seen. This
records them rather than predicts them. The committed outputs are written by the
committed code after this commit; the runs themselves are unchanged. Nothing
changes a registered test, family size or endpoint.

* RevIN preference (descriptive): the share of each arm's lookups on ILI with
  RevIN off, for the Mac levels (0, 30, 60, 120) and for the committed E3 and E4
  ILI runs; and, per table, the best config's regret among RevIN-on and among
  RevIN-off configs. If the trained policies' lookups are almost all RevIN-off,
  the transfer loss can come from a preference learned on ETTh1 (where RevIN off
  wins) rather than from GRPO or the league, which this data cannot separate.
* Paired gap (descriptive): each policy's AUC averaged over its three
  checkpoints minus the untrained search of the same deployment seed
  (seeds 0 to 9), and how many policies are worse.
* ETTh1 AB-P differences (descriptive, in-sample): anon and repair against ref,
  with Welch intervals and no p values.
* Diagnostics per checkpoint: lookups, repeat and invalid rates, and the number
  of logged updates (a step whose episode had no budget left logs nothing).
* Provenance: the digest of every Mac, E3 and E4 run file that the analyses read
  and the commits that ran each Mac item.

## Amendment AB4 (2026-11-26, before any run of the items below)

Soumitra's side, under AB1, AB3 and docs/LEG2_anirudh.md ("If you cannot run
WP3 on Modal at all ... the same items then join his RTX queue after the KL
arms, with the local references"), by the owner's instruction of 26 Nov.
AB3 (origin/anirudh-ablations, added in 31f9313 and read at dfb2b47)
records that Anirudh did not launch
ABL_P_anon, ABL_P_repair, ABL_G_steps30 or ABL_G_steps120 (nor ABL_P_ref) on
Modal, so they may move to another backend (AB1, "Backend switching"). No
result of any item below exists when this is committed. When writing AB4 the
author had read AB3 and its addenda, not the Mac (AB3) result files or
docs/ablations.md. AB3 addenda 3 and 4 describe the Mac step results
qualitatively (the level-0 confound, the RevIN-off hypothesis, how many
policies are worse), so the author knew the direction of the Mac
comparison. The author had also handled the AB2 run files of
ABL_P_ref_local and ABL_G_ref_local (counts, timings, on-policy gaps) and
computed no AUC or regret from them. Nothing below changes a primary test, its family or its
data.

* Backend: the local GPU of AB2 (the same card, driver, WSL2, torch, CUDA and
  cuDNN), one job at a time (--per-gpu 1), packages and model revision as
  AB2 (Qwen/Qwen2.5-1.5B-Instruct 989aa7980e4cf806f80c7fef2b1adb7bc71aa306).
  By the L4 rule the four items, about $20.5 on Modal (docs/LEG2_anirudh.md WP3: $3.5 + $4 + $3.5 +
  $9.5), do not fit Soumitra's about $8.15 of free credit
  (results/logs/ab2_modal_billing.txt; no Modal job has run since). Nothing
  in AB4 runs on Modal.
* Code. The launcher is unchanged: scripts/ablate_llm.py, sha256
  3919d68337302590eda42f53ca10158ca27766bb488026b5a4fc8373c477b786
  (79977de, AB2). Its flags --anonymous, --no-constrained and --steps pass to
  leaguetst/architect/runner.py unchanged. The queue is a new file,
  scripts/ablations/run_ab4_queue.sh, sha256
  2b9f3d98a0246fd93bf4cbce6561ebec198055fada156564e0413e1a29d34788, with the AB2 queue's guards
  (the same lock, dirty-tree refusal, the docs/e4_rerun.sha256 check) and its
  own status file, results/logs/ab4_queue_status.txt. The ten files of
  docs/e4_rerun.sha256 are unchanged.
* Items. Each is the registered item of the same name without "_local", run
  on the RTX; the Modal-named items stay not run. Outputs go to
  results/runs/llm_local/<item>/<name>/ and logs to results/logs/, in this
  order (short items first, ABL_G_steps120_local last):

| Item | Command (scripts/ablate_llm.py ... --per-gpu 1) | Method |
|---|---|---|
| ABL_P_anon_local | prompted --dataset ILI (and later ETTh1) --seeds 30 --model Qwen/Qwen2.5-1.5B-Instruct --anonymous | qwen_prompted_anon_1.5b_local |
| ABL_P_repair_local | the same with --no-constrained instead of --anonymous | qwen_prompted_repair_1.5b_local |
| ABL_G_steps30_local | transfer --source ETTh1 --target ILI --arms league:10 --steps 30 (E4 recipe otherwise) | grpo_league_from_ETTh1 |
| ABL_G_steps120_local | the same with --steps 120 | grpo_league_from_ETTh1 |

  The queue runs anon then repair on ILI, then the same two on ETTh1 (ILI is
  tested, ETTh1 descriptive), then steps30, then steps120. Search seeds 0 to
  29; GRPO training seeds 100 to 109, deployed at seeds 0 to 9, budget 100,
  KL weight 0.04, learning rate 1e-4.
* References and tests. Every reference is local (AB1: local references
  serve only local arms), which replaces AB2's naming of the Modal ABL_P_ref (never launched, AB3)
  and the J2 league arm (a Modal run, which under AB1 serves only Modal
  arms) as the references of these tests.
  On ILI held-out AUC: anon against ABL_P_ref_local and repair against
  ABL_P_ref_local (seed-level permutation tests, as AB-P); the steps trend
  with levels 0 (ABL_P_ref_local on ILI, the untrained model, 30 searches),
  30 (ABL_G_steps30_local), 60 (ABL_G_ref_local) and 120
  (ABL_G_steps120_local), as AB-G. Each trained level is its own 10 policies, trained from scratch (no
  level is a checkpoint of another, unlike AB3's Mac item),
  but all levels use training seeds 100 to 109 and deployment seeds 0 to 9;
  the learning rate is constant and the random streams seeded, so a 30-step
  policy follows roughly the first 30 steps of the 60-step policy with the
  same seed. Levels are therefore dependent, as in AB3's checkpoints, and the
  registered permutation p is reported with this caveat. Trend statistic:
  the Spearman correlation of level rank (0, 1, 2, 3) with ILI held-out AUC
  over 30 + 10 + 10 + 10 values, two-sided permutation p, computed by
  spearman_trend in scripts/ablations/llm_tests.py as committed at 5abbd20
  on origin/anirudh-ablations (sha256
  01b70835dd128001c78eceb040590b7e79ace2a17f260a3b98022cdbcd0ee0a6), with
  the same function, AUC and settings as the KL trend. Holm sizes are unchanged: AB-P over 3
  (anon, repair, the scale trend), AB-G over 2 (the steps trend, the KL
  trend). AB3's Mac items are exploratory and are not a level or a
  reference of any test here.
  The AB4 tests are the only registered results for these questions; the
  Mac results are reported beside them whether or not they agree. AB3
  addendum 3's "training at all" and "dose" analyses run on the RTX levels
  as exploratory only, specified exactly as in addendum 3. No other
  analysis is added after results are seen.
* Hardware check: AB1's ABL_P_ref_local against ABL_P_ref is not available
  (ABL_P_ref was not run); ABL_G_ref_local against the J2 league arm stays
  as AB2 says.
* Where outputs are committed. results/ablations/rtx/<item>/<table>/, never
  results/search/: scripts/picks.py and scripts/e5_referee.py read every
  folder under results/search/, and these items reuse grpo_league_from_ETTh1
  (AB3; docs/ISSUES.md on origin/anirudh-ablations, 25 Nov). The same rule applies to AB2's six items,
  now under results/search/ABL_*_local: before soumitra-leg2 is pushed, and
  before any picks.py, e5_referee.py, report or L5 run on it, they move
  unchanged to results/ablations/rtx/ in a separate commit with a sha256
  manifest. The move is logged in docs/ISSUES.md for Diya, and Anirudh is
  told that scripts/ablations/llm_tests.py reads results/search/ABL_*.
* Time, from the AB2 runs on this card (25 Nov). Prompted 1.5B searches took
  24.7 min for 30 ILI seeds and 18.2 min for 30 ETTh1 seeds, so the anon
  item takes about 45 min for both tables (the same model and prompt length). The repair item has not run on
  any backend; unconstrained decoding skips the trie and an episode stops
  after 20 steps with no new lookup, so it is estimated at 45 to 120 min
  for both tables. A 60-step policy trained in 17.1 to 20.9 min, and
  deploying 10 policies took 19 to 24 min (each 60-step item took 3.40 to
  3.59 h, results/logs/l4_queue_status.txt), so, with training time roughly
  proportional to steps, steps30 takes 10 x (8.6 to 10.5 min) plus 19 to 24
  min, 1.7 to 2.1 h, and steps120 10 x (34 to 42 min) plus 19 to 24 min, 6.0
  to 7.4 h (each policy below the 2.5 h hand-stop). Total about 9 to 12.5 h,
  against the 28 Nov 23:59 ET deadline.
* Memory: the 1.5B model at --per-gpu 1 ran every AB2 prompted search and
  all 30 GRPO policies on this card with no out-of-memory error; the per-step
  memory of GRPO does not depend on the number of steps, and the repair arm
  builds no trie. No new probe was run.
* As AB2, applied to AB4's items with results/logs/ab4_queue_status.txt in
  place of the AB2 status file: hand-stops on record progress (one hour
  prompted, 2.5 hours transfer), the stale driver.lock procedure, retried
  runs and their A26 treatment, completion by counts, each prompted item's
  ILI and ETTh1 table as its own arm, an arm unfinished at 28 Nov 23:59 ET
  counting as not run, a trend with any level missing not tested and
  counted as not rejected, and no setting changed to make a job fit or
  finish.
* Launch order: the queue is launched only with HEAD at the commit that
  adds AB4 or a descendant of it.

### AB4 addendum 1 (2026-11-26, the AB2 move; before any AB4 result exists)

AB4 asks that AB2's six items move from results/search/ABL_*_local to
results/ablations/rtx/ "before soumitra-leg2 is pushed". That could not be
met: the branch was already pushed on 25 Nov (13:51 ET, d8582c0), before
AB4 was written. From then until the move commit, origin/soumitra-leg2 held
the nine run files at the old path. Neither origin/main (575b4a4) nor
origin/anirudh-ablations contains them. The move commit renames them
unchanged (git reports every one as a 100% rename) and adds
results/ablations/rtx/AB2_move.sha256, whose digests were taken at the old
path before the move. History is not rewritten. The other conditions
stand: the move precedes any picks.py, e5_referee.py, report or L5 run on
this branch. The training records under results/runs/ABL_G_*_local/ stay
where they are, because neither picks.py nor e5_referee.py reads
results/runs/. Nothing in the move changes a test, a reference, a setting
or a run file.

## Amendment AB4 (2026-11-26, Diya's side, before any run of the items below)

Revised the same day after the critic's and the verifier's reviews, before
any run. Critic's verdict: docs/verdicts/AB4_registration.md.

What existed. No run of ABL_P_anon_local, ABL_P_repair_local,
ABL_G_steps30_local or ABL_G_steps120_local exists in git; Soumitra confirms
in his launch commit that results/runs/llm_local/ holds none of them, and
that his side computed no AUC, regret or test of any AB2 item (or names what
it computed). No registered AB-P or AB-G test has been computed on the RTX or
on Modal. The same tests were computed, exploratory, on the Mac (AB3), and
Anirudh saw their values. The author opened no Mac result file and no search
result of the AB2 items (of the AB2 records it read counts and settings
only), but read parts of AB3 addenda 3 and 4 and the Mac commit messages,
which describe the Mac step analyses qualitatively. The author has also seen
the E4 ILI results through the J3 re-score; the J2 league arm against E3
prompted Qwen is the 60-against-0 analogue on another card.

* Why, and why these references. AB1 gave ABL_P_ref, ABL_P_anon,
  ABL_P_repair, ABL_G_steps30 and ABL_G_steps120 to Anirudh on Modal. He did
  not launch them (AB3, 25 Nov 00:58 ET, before his Mac results), and they
  are never launched anywhere. docs/LEG2_anirudh.md and docs/LEG2_soumitra.md
  (committed 24 Nov in 7c2cb76, before any ablation result) send them to
  Soumitra's RTX queue with the local references, and the shared rule above
  (an arm and its reference run on the same backend and code digests) and
  AB1 ("local references serve only local arms") force those references. The
  J2 league arm ran on A10 and A10G cards (AB1), so it leaves the step trend
  and stays in the descriptive hardware check. Modal could not have run them
  anyway: about $8.15 of free credit (AB2) against about $11 and $13 plus $5.
* Items, on the AB2 card (NVIDIA GeForce RTX 4080 Laptop GPU), one job at a
  time, queue scripts/ablations/run_wp3_rtx_queue.sh, sha256 dc08a6404b4ffd9aad6af81bb206a40f2314d14b981677422dd9a25752b86e12,
  in this order:

| Item | Table | Command (scripts/ablate_llm.py ... --per-gpu 1) | Method |
|---|---|---|---|
| ABL_P_anon_local | ILI | prompted --dataset ILI --seeds 30 --model Qwen/Qwen2.5-1.5B-Instruct --anonymous | qwen_prompted_anon_1.5b_local |
| ABL_P_repair_local | ILI | the same with --no-constrained in place of --anonymous | qwen_prompted_repair_1.5b_local |
| ABL_G_steps30_local | ILI | transfer --source ETTh1 --target ILI --arms league:10 --steps 30 | grpo_league_from_ETTh1 |
| ABL_G_steps120_local | ILI | the same with --steps 120 | grpo_league_from_ETTh1 |
| ABL_P_anon_local | ETTh1 | as on ILI, --dataset ETTh1 | qwen_prompted_anon_1.5b_local |
| ABL_P_repair_local | ETTh1 | as on ILI, --dataset ETTh1 | qwen_prompted_repair_1.5b_local |

* Everything else as AB2: launcher scripts/ablate_llm.py unchanged (sha256
  3919d683..., commit 79977de), pins, packages, search seeds 0 to 29,
  training seeds 100 to 109 deployed at seeds 0 to 9, budget 100, the E4
  recipe apart from the steps, and the hand-stop, stale-lock, retried-run,
  completion, relaunch and per-table rules. The queue takes the AB2 queue's
  lock, refuses unless HEAD contains AB4 and the run-path code (leaguetst/,
  the launcher, llm_local.py, requirements-llm.txt) equals the AB2 commit
  c2dda8d, and logs the card and driver at start.
* Deadline for these arms: 28 Nov 18:00 ET. The queue starts no job after
  it and stops a running one at it; an arm unfinished then counts as not run
  (AB1). This leaves 18:00 to 23:59 ET for the analysis and the push, so the
  owner's 28 Nov 23:59 ET deadline for all teammate work stands.
* Where records live. results/ablations/rtx/<item>/<table>/, with any
  errors.jsonl and a retried.txt naming retried policies (one run name per
  line, written in the item's bring-back commit, never edited later). The
  queue copies the record files there after each item. In this commit the
  nine AB2 items move there with git mv, contents unchanged, from
  results/search/<item>/ and (training.jsonl) results/runs/<item>/ILI_T24/:
  AB2 registered results/runs/llm_local/, and docs/LEG2_soumitra.md L4 told
  Soumitra to copy records into results/search/, which scripts/picks.py and
  scripts/e5_referee.py read in full (docs/ISSUES.md, 25 Nov). No committed
  pick, K4 or E5 output was written while they were there. Their digests
  are frozen in results/ablations/rtx/AB2_records.sha256.
* References and tests, replacing AB2's Holm bullet for these arms. AB-P on
  ILI held-out AUC, Holm over 3: anon against ABL_P_ref_local, repair against
  ABL_P_ref_local, the scale trend (as AB2). AB-G on ILI held-out AUC, Holm
  over 2: the step trend with levels 0 (ABL_P_ref_local on ILI, 30
  searches), 30, 60 (ABL_G_ref_local) and 120, and the KL trend (as AB2).
  The tests are those defined above (AB-G, AB-P); trends are two-sided.
* Failures. Policies stopped by the on-policy check have no value and are
  left out of a trend (A26). Retried policies stay in it as they finished
  (AB1, AB2, A29). Beside each trend with a stopped or retried policy, every
  such value is set to the lowest and, separately, the highest value
  observed in that trend. These fills illustrate; they are not bounds.
  leaguetst/sensitivity.py is not used: it needs an arm under test, which a
  two-sided trend does not have. A trend with a level of fewer than two
  values is not run.
* Dependence (found in review, before any step arm runs). Training has no
  step-dependent schedule (constant-rate AdamW seeded by the training seed,
  leaguetst/architect/llm.py), and the levels share training and deployment
  seeds, so the 30-step policy of a seed is, up to GPU nondeterminism, a
  prefix of its 60- and 120-step policies. The registered step-trend p is
  then anti-conservative; it and its Holm decision are reported as
  registered, with this caveat. Registered secondary analyses, policy as
  unit, no Holm: (i) each policy's mean AUC over the trained levels it
  completed against the 30 untrained searches (common.compare); (ii) per
  policy with all three trained levels, the Spearman correlation of AUC
  with 30, 60 and 120 steps, exact two-sided sign flip over policies; (iii)
  diagnostic, the share of each 30- and 120-step policy's first 30
  training-log rows equal to the 60-step log of the same seed. The report
  calls the step result a dose-response only if (ii) has the same sign as
  the registered trend and its p is at most 0.05; otherwise, if (i) shows a
  difference, it says trained policies differ from the untrained model. The
  KL levels share seeds too (at step 1 the update does not depend on beta);
  with balanced levels this makes that test conservative, and the report
  says so.
* Code. scripts/ablations/llm_registered.py, sha256 0c101110afa8a46e0afd2fc831f6aec5822b257ee088c2cae1f819a4b70e583e (tests in
  tests/test_llm_registered.py), computes both families with common.compare
  (sha256 46aef3c196864a1f835505e58e2b1bef22940ca03219bb60887514039efa310b) for the pairs and, for the trends,
  llm_tests.spearman_trend (scripts/ablations/llm_tests.py,
  sha256 08c72a4c46dded669ebff913a45887ffe215caed24d10005fd27f808b34e9239; 100,000 permutations, seed 0, the observed
  labelling counted), the function of the Mac analysis. It checks every
  record's card, backend, pin, and commit (descending from AB2, and
  containing AB4 for the AB4 arms), and the frozen AB2 digests, per arm.
* When, enforced by the script. The families are computed once: when
  --check shows all 15 arms complete, or with --final only after 28 Nov
  18:00 ET (missing arms then count as not run, AB1; an unfinished ETTh1
  table does not make its ILI table not run, AB2). The script refuses
  --final before the deadline, refuses if results/ablations/LLM-registered.json
  exists, refuses on uncommitted records or code, refuses if common.py or
  llm_tests.py differ from the digests above, and stamps the time. No AB-P or
  AB-G value is computed before then, including the scale and KL trends,
  whose arms are already complete.
* Verification. A verifier re-scores both families with its own script,
  committed before llm_registered.py is run, and a critic attacks the
  result. If a bug is found after the run, the fix is a dated amendment, the
  families are recomputed once, both outputs are kept, and the report
  discloses it.
* Hardware check, descriptive. ABL_G_ref_local against the J2 league arm, as
  AB2. ABL_P_ref_local against Modal ABL_P_ref cannot be made (ABL_P_ref
  never ran); the E3 A10 prompted runs on the shared seeds are shown instead,
  labelled as not the registered check (E3 predates the pins).
* Anirudh's Mac items (AB3) stay exploratory. They are never pooled with
  these arms and never replace one.
* Report disclosure, fixed now: "The RTX AB-P and AB-G arms were registered
  on 26 Nov (AB4), after one author had analysed exploratory Mac runs of the
  same questions (AB3), and after the E4 results, which include an A10 and
  A10G analogue of the 60-step level, were known. Arms, references and tests
  follow rules written on 24 Nov before any ablation result; AB4 fixed the
  analysis code, the failure handling and the secondary step analyses. The
  Mac runs are exploratory and never pooled."
