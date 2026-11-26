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
