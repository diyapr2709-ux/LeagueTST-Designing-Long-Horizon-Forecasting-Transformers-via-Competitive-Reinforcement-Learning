# Plan and handoff: ablations, robustness and external validity (Anirudh)

For Anirudh and the Claude Code session on Anirudh's account. Written
2026-11-24. Read CLAUDE.md first: every hard rule there applies to you too.
The registered design is docs/prereg_ablations.md; this file says how to
run it. Soumitra's E4 rerun (docs/HANDOFF_soumitra.md) runs in parallel.

## Why this work

The primary results so far are null: on fresh seeds no search method beats
random search on either table (results/picks/). A null result is
publishable at NeurIPS level only if the reader can see its resolution and
its causes. This workstream supplies that: whether GRPO learned at all,
what each part of the method does (ablations as trends, sized for power),
how large an effect each comparison could detect, whether the null comes
from a plateau of near-equal configs, how much one-seed tables bias a
replay, how standard HPO baselines from outside this project compare,
which design choices matter across datasets (RQ3), and, after the 30 Nov
test, more seeds, a third dataset and a benchmark release.

## How this fits with Soumitra and Diya

| Who | Does | Branch | Writes |
|---|---|---|---|
| Soumitra | E4 primary rerun: 45 GRPO policies, J1 and J2 (and J3 if asked) | soumitra-e4 | results/search/E4/, results/runs/E4/ |
| Anirudh | Everything in this file | anirudh-ablations | results/search/ABL_*/, results/runs/ABL_*/, results/ablations/, scripts/ablations/, new tests, amendments at the end of docs/prereg_ablations.md, docs/ablations.md |
| Diya | Ledger, picks, E5, E6, test prereg, the single test, STATE, report, merges | main | everything else |

The GPU workstreams never share an output: Soumitra's items are E4,
Anirudh's are ABL_*. If both use Modal and E4 completes under A26, the AB-G
reference is Soumitra's J2 league arm (same seeds, recipe and digests);
otherwise Anirudh runs ABL_G_ref. AB-L (did GRPO learn?) reads Soumitra's
training logs once Diya has merged soumitra-e4.

## Authorisation

CLAUDE.md rule 10 names only Diya's two Modal workspaces. The owner decided
on 2026-11-24 that Anirudh's workstream runs on Anirudh's Modal workspace
(profile anirudh) or a local CUDA GPU (docs/ISSUES.md). Until Diya adds this
to rule 10 by hand, this file is the authority for the anirudh profile:

    Owner's decision, 2026-11-24: the ablation workstream
    (docs/prereg_ablations.md), which needs no data volume, runs on Anirudh's
    Modal workspace (MODAL_PROFILE=anirudh) or one local CUDA GPU
    (docs/HANDOFF_anirudh.md).

## Rules for Anirudh's Claude (in addition to CLAUDE.md)

1. Do not edit the ten files listed in docs/e4_rerun.sha256 until Diya says
   Soumitra's E4 rerun is complete (cut-off 26 Nov 18:00 ET). Add new files
   instead (scripts/ablations/, tests/test_ablations.py).
2. Never edit results/ledger.jsonl, docs/claims/, prereg/, docs/PLAN.md,
   docs/STATE.md, docs/prereg_E4.md, leaguetst/sensitivity.py or a locked
   file (leaguetst/eval/integrity.py lists them).
3. Anything new that tests something is first registered by a dated
   amendment at the end of docs/prereg_ablations.md, committed before its
   first run. Descriptive work is labelled descriptive.
4. Every number in docs/ablations.md is written by a script from a results
   file, never typed. Each result gets a verifier pass (an independent
   subagent re-computes it) and a critic pass before it is committed.
5. Cite only papers the lit-scout opened in that session, with arXiv id and
   URL (CLAUDE.md rule 9).
6. Never upload anything to Modal. The LLM jobs need no data; the CPU
   analyses read only the committed ledger and results files. Never touch
   data/ETT-small or data/illness (full files with test rows).
7. Commit on anirudh-ablations only; never push. Anirudh pushes the branch.

## Setup (Anirudh, by hand, once)

1. Clone main after Diya has pushed (commit a1751ae or later) and check the
   code: `shasum -a 256 -c docs/e4_rerun.sha256` must print OK ten times.
2. `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-modal.txt`,
   then `git switch -c anirudh-ablations` and `.venv/bin/python -m pytest -q`
   (must pass before anything else).
3. Modal (recommended): `.venv/bin/modal token new --profile anirudh` in your
   own workspace; set a $0 spend limit in the dashboard; check
   `MODAL_PROFILE=anirudh .venv/bin/modal run scripts/modal_llm.py::versions`
   prints transformers 5.24.0, peft 0.22.0, accelerate 1.16.0, torch 2.15.0.
4. A local card cannot run Phase 1 as written (scripts/llm_local.py has no
   --beta or --anonymous flags, and the E4 files are frozen); on a local card
   the GPU arms wait for WP5's script, and the CPU packages start at once.

## Work packages

### WP0: literature (lit-scout), today

A web-enabled subagent as lit-scout writes docs/related_work.md: LLM-driven
architecture and hyperparameter search, learned search policies and league
or self-play training, tabular NAS benchmarks and their validation-to-test
findings, multi-fidelity HPO (the methods of AB-X), and statistical practice
for many-seed comparisons. Each entry: arXiv id, URL, one line on what it
shows, and whether it can run on the oracle. Also, from pages actually
opened: the NeurIPS Datasets and Benchmarks requirements, and the licences
of candidate third datasets (Phase B).

### WP1: controller, plateau and one-seed bias, CPU (AB-C, AB-Q, AB-N), 24 to 26 Nov

* scripts/ablations/controller.py: AB-C, `factorized_controller(rivals="league", ...)`
  arms at 100 seeds, replayed with `leaguetst.search.replay.run_method`;
  wrap each Oracle into a run dict as scripts/replay.py does before calling
  `run_metric` (held-out AUC on ILI).
* scripts/ablations/plateau.py: AB-Q, every E3 method on ILI with 3-seed
  means as lookups.
* scripts/ablations/one_seed_bias.py: AB-N.
* Tests in tests/test_ablations.py on a small synthetic table (see
  tests/test_replay.py). Run detached; poll status files.

### WP2: external HPO baselines, CPU (AB-X), 24 to 27 Nov

scripts/ablations/external_hpo.py runs DEHB, SMAC (random-forest BO) and
Hyperband through the same Oracle (charges, budgets, seeds as E3; see how
scripts/replay.py wraps TPE). Pin the package versions in
requirements-ablations.txt and register them by amendment before the first
run. 30 seeds; tests against E3's random search on ILI.

### WP3: GPU Phase 1 on Modal (AB-P, AB-G), launch by 25 Nov

First warm the model cache with one launch, then the rest in parallel. Each
line is its own item and output folder. About 2 to 3 hours wall time and
about $18 (Modal A10G with 4 cores and 48 GB is about $1.67 per hour).

```
M=.venv/bin/modal; A=scripts/modal_llm.py; mkdir -p results/logs
MODAL_PROFILE=anirudh $M run $A::prompted --dataset ILI --seeds 30 --item ABL_P_ref --per-gpu 2 > results/logs/abl_p_ref_ILI.log 2>&1
MODAL_PROFILE=anirudh nohup $M run --detach $A::prompted --dataset ETTh1 --seeds 30 --item ABL_P_ref --per-gpu 2 > results/logs/abl_p_ref_ETTh1.log 2>&1 &
for d in ILI ETTh1; do
  MODAL_PROFILE=anirudh nohup $M run --detach $A::prompted --dataset $d --seeds 30 --anonymous --item ABL_P_anon --per-gpu 2 > results/logs/abl_p_anon_$d.log 2>&1 &
  MODAL_PROFILE=anirudh nohup $M run --detach $A::prompted --dataset $d --seeds 30 --no-constrained --item ABL_P_repair --per-gpu 2 > results/logs/abl_p_repair_$d.log 2>&1 &
done
MODAL_PROFILE=anirudh nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms league:10 --steps 30 --item ABL_G_steps30 --per-gpu 2 > results/logs/abl_g_steps30.log 2>&1 &
MODAL_PROFILE=anirudh nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms league:10 --steps 120 --item ABL_G_steps120 --per-gpu 2 > results/logs/abl_g_steps120.log 2>&1 &
MODAL_PROFILE=anirudh nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms league:10 --beta 0 --item ABL_G_beta0 --per-gpu 2 > results/logs/abl_g_beta0.log 2>&1 &
MODAL_PROFILE=anirudh nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms league:10 --beta 0.1 --item ABL_G_beta01 --per-gpu 2 > results/logs/abl_g_beta01.log 2>&1 &
```

The ETTh1 prompted runs share the ABL_P_ref item with the ILI run but write
to a different folder (ETTh1_T336), so the launch lock does not conflict.
Only if Soumitra is not on Modal, or E4 is declared incomplete under A26,
also launch `... transfer --source ETTh1 --target ILI --arms league:10 --item ABL_G_ref --per-gpu 2`.

Bring back each item:

```
for it in ABL_P_ref ABL_P_anon ABL_P_repair ABL_G_steps30 ABL_G_steps120 ABL_G_beta0 ABL_G_beta01; do
  for t in ILI_T24 ETTh1_T336; do
    S=$(mktemp -d); MODAL_PROFILE=anirudh $M volume get leaguetst-runs /$it/$t $S 2>/dev/null || continue
    mkdir -p results/search/$it/$t results/runs/$it/$t
    cp $S/$t/*_runs.jsonl results/search/$it/$t/
    cp $S/$t/training.jsonl $S/$t/errors.jsonl results/runs/$it/$t/ 2>/dev/null
  done
done
```

Checks: 30 searches per AB-P item and table, 10 training records and 10
deployments per AB-G item, env blocks matching docs/prereg_E4.md A25, no
out-of-memory errors. Then scripts/ablations/llm_tests.py computes the AB-P
and AB-G families into results/ablations/.

### WP4: did GRPO learn? CPU (AB-L), once soumitra-e4 is merged

scripts/ablations/learning.py reads results/runs/E4/soumitra_*/training.jsonl:
the reward slope per policy and the family test, plus the best-in-history
of the last 10 steps against prompted Qwen on the source table.

### WP5: referee, statistics, RQ3, and model scale (AB-R, ST, RQ3, AB-P scale)

* CPU, 24 to 28 Nov: scripts/ablations/referee.py (AB-R, reusing the draw
  in scripts/fidelity.py), power.py (the detectable-effect table for every
  registered comparison, adding the E4 families when they land), robust.py
  (raw-MSE regret, MAD-SD ETTh1 replay, IQM and probability-of-improvement
  profiles), rq3.py (ANOVA interactions in a new file modelled on
  scripts/anova.py, cross-table field ranking, the REVIN validation check).
  The A26 worst-case analysis already exists (leaguetst/sensitivity.py).
* GPU, after 26 Nov 18:00 ET: scripts/ablate_llm.py, a new file that builds
  plans with leaguetst.architect.runner and sets the job's model and
  revision (Qwen2.5-0.5B and 3B-Instruct), with executors for Modal (an app
  named leaguetst-ablate with the same pinned image) and a local GPU.
  Register its sha256 and the model revisions by amendment, get a verifier
  and a critic pass, then run AB-P scale (30 seeds per table, about $5).
  LoRA rank, rival pool size and temperature are exploratory only.

### WP6: write-up, 27 Nov to 2 Dec

scripts/ablations/report.py writes docs/ablations.md from results/ablations/
(regret curves with intervals, probability-of-improvement profiles, the
detectable-effect table, family tables with Holm-adjusted p and effect
sizes, the RQ3 field ranking), and docs/neurips_checklist.md drafts the
NeurIPS paper checklist with each answer pointing to a file.

### WP7: Phase B, after the 30 Nov test

In the order of docs/prereg_ablations.md: ETTh1 seeds 2022 and 2023 (Diya's
workspaces, which hold the data), a third dataset from another domain at 3
seeds (a person prepares it under its own lock), the benchmark release
(tables plus a loader that downloads and splits the original data, Croissant
metadata, hosting, executable code; never the split CSVs), the wider space,
and LLM-based optimisers from WP0.

## Credit (Anirudh's Modal, $30 free per month)

| Item | Est. cost |
|---|---|
| AB-P ref, anon, repair: 3 arms x 2 tables x 30 seeds | about $5 |
| AB-G: 4 arms x 10 policies | about $13 |
| AB-G reference (only if not shared with Soumitra) | about $3.5 |
| AB-P scale, 0.5B and 3B (WP5) | about $5 |

If credit runs short, cut in this order: AB-P scale 0.5B, ABL_G_beta01,
ABL_G_steps120. Check `modal billing summary` before each launch.

## Timeline

| Date | Anirudh | Soumitra | Diya |
|---|---|---|---|
| 24 Nov | setup, WP0, start WP1 and WP2 | setup, launch J1 and J2 | push main, picks batch 2 prep |
| 25 Nov | launch WP3 | J1, J2 finish | merge soumitra-e4, E4 summaries, E5 |
| 26 Nov | WP3 analysis, WP4 | J3 if asked; cut-off 18:00 | K4 inputs |
| 27 Nov | WP5 GPU after review | done | picks batch 2, K4 (28 Nov) |
| 28 Nov to 2 Dec | WP6, push branch | | prereg, test 30 Nov, report 3 Dec |
| December | WP7 | | paper |

## First message for Anirudh's Claude

> Read CLAUDE.md, docs/STATE.md, docs/prereg_ablations.md and
> docs/HANDOFF_anirudh.md. Check the code digests and run the tests. Work
> through WP0, WP1 and WP2 on CPU while WP3 runs on Modal profile anirudh,
> following the rules section exactly. Commit each verified result on branch
> anirudh-ablations. Do not push, and do not edit the files listed in
> docs/e4_rerun.sha256 until Diya says the E4 rerun is complete.
