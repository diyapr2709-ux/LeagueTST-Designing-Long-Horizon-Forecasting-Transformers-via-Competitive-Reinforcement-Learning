# Plan and handoff: ablations, robustness and external validity (Anirudh)

For Anirudh and the Claude Code session on Anirudh's account. Written
2026-11-24. Read CLAUDE.md first: every hard rule there applies to you too.
The registered design is docs/prereg_ablations.md; this file says how to
run it. Soumitra's E4 rerun (docs/HANDOFF_soumitra.md) runs in parallel.

## Why this work

The primary results so far are null: on fresh seeds no search method beats
random search on either table (results/picks/). A null result is
publishable at NeurIPS level only if the reader can see its resolution and
its causes. This workstream supplies that: what each part of the method
does (ablations), how large an effect the design could have detected
(power), how much one-seed tables bias a replay, which design choices
matter and whether that holds across datasets (RQ3), and, after the 30 Nov
test, external baselines, a third dataset and a benchmark release.

## How this fits with Soumitra and Diya

| Who | Does | Branch | Writes |
|---|---|---|---|
| Soumitra | E4 primary rerun: 45 GRPO policies, J1 and J2 (and J3 if asked) | soumitra-e4 | results/search/E4/, results/runs/E4/ |
| Anirudh | Everything in this file | anirudh-ablations | results/search/ABL_*/, results/ablations/, scripts/ablations/, new tests, appendable docs/prereg_ablations.md, docs/ablations.md |
| Diya | Ledger, picks, E5, E6, test prereg, the single test, STATE, report, merges | main | everything else |

The two GPU workstreams never share an output: Soumitra's items are E4,
Anirudh's are ABL_*. If both use Modal, the AB-G reference arm is
Soumitra's J2 league arm (same code digests, same A10G), which saves about
$4; Anirudh's AB-G analysis then waits for J2 (expected 25 Nov). If either
uses a local card, Anirudh runs the reference arm (ABL_G_ref) himself.

## Rules for Anirudh's Claude (in addition to CLAUDE.md)

1. Do not edit the ten files listed in docs/e4_rerun.sha256 until Diya says
   Soumitra's E4 rerun is complete (cut-off 26 Nov 18:00 ET). Add new files
   instead (scripts/ablations/, tests/test_ablations.py).
2. Never edit results/ledger.jsonl, docs/claims/, prereg/, docs/PLAN.md,
   docs/STATE.md, docs/prereg_E4.md or a locked file (leaguetst/eval/integrity.py).
3. A new ablation or analysis that tests something is first registered by a
   dated amendment at the end of docs/prereg_ablations.md, committed before
   its first run. Descriptive work needs no amendment but is labelled so.
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
   then `git switch -c anirudh-ablations`. Run `.venv/bin/python -m pytest -q`:
   it must pass before anything else.
3. Modal (recommended): `.venv/bin/modal token new --profile anirudh` in your
   own workspace; set a $0 spend limit in the dashboard; check
   `MODAL_PROFILE=anirudh .venv/bin/modal run scripts/modal_llm.py::versions`
   prints transformers 5.24.0, peft 0.22.0, accelerate 1.16.0, torch 2.15.0.
4. Local RTX instead: `pip install -r requirements.txt -r requirements-llm.txt` on Linux or WSL2
   with a 24 GB card. Phase 1 GPU commands below are Modal commands; on a
   local card, WP4's script must exist first (it covers both backends).

## Work packages

### WP0: literature (lit-scout), today

Run a web-enabled subagent as lit-scout. Deliver docs/related_work.md:
LLM-driven architecture and hyperparameter search, reinforcement-learned
search policies and self-play or league training, tabular NAS benchmarks
and their validation-to-test findings, multi-fidelity HPO baselines, and
statistical practice for reporting many-seed comparisons. Each entry: arXiv
id, URL, one line on what it shows, and whether it gives an external
baseline we can run on the oracle (WP6). Also list what a NeurIPS Datasets
and Benchmarks submission requires (datasheet, metadata format, licence),
from pages actually opened.

### WP1: controller and one-seed-table ablations, CPU (AB-C, AB-N), 24 to 26 Nov

Write scripts/ablations/controller.py: builds each AB-C arm with
`factorized_controller(rivals="league", ...)`, replays it with
`leaguetst.search.replay.run_method` on the ILI and ETTh1 tables (rows from
the ledger, item E1), 30 seeds, scores with `run_metric` (held-out AUC on
ILI), and writes results/ablations/AB-C/<setting>.json plus the Holm-family
tests. Write scripts/ablations/one_seed_bias.py for AB-N. Use scripts/replay.py
and scripts/picks.py as the model for loading tables and runs. Tests in
tests/test_ablations.py on a small synthetic table (see tests/test_replay.py).
Expected: a few CPU hours; run detached, poll status files.

### WP2: referee, statistics and RQ3, CPU (AB-R, ST, RQ3 extras), 24 to 27 Nov

* scripts/ablations/referee.py (AB-R), reusing the draw in scripts/fidelity.py.
* scripts/ablations/power.py: the smallest detectable effect of every
  registered comparison (docs/claims/E3.json, results/picks/*_analysis.json,
  and, when they land, the E4 families), by simulation at 80% power.
* scripts/ablations/robust.py: raw-MSE regret, the robust-SD replay summary
  for ETTh1, interquartile-mean and probability-of-improvement profiles with
  stratified bootstrap intervals.
* scripts/ablations/worst_case.py: the A26 worst-case sensitivity as a tested
  function (inputs: per-seed values per arm and the failed seeds), for
  Diya's side to call on E4.
* scripts/ablations/rq3.py: ANOVA interactions (extend the logic of
  scripts/anova.py in a new file), cross-table agreement of field shares,
  and the REVIN validation check against results/revin_prediction.json
  (REVIN rows are in the ledger; never look at test).

### WP3: GPU Phase 1 on Modal (AB-G1, AB-P1), launch by 25 Nov

Each line is a separate item, so they can all run at once (about 2 to 3
hours wall time, about $11 including AB-P).

```
M=.venv/bin/modal; A=scripts/modal_llm.py; mkdir -p results/logs
MODAL_PROFILE=anirudh nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms league:10 --beta 0 --item ABL_G_beta0 --per-gpu 2 > results/logs/abl_g_beta0.log 2>&1 &
MODAL_PROFILE=anirudh nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms league:10 --beta 0.1 --item ABL_G_beta01 --per-gpu 2 > results/logs/abl_g_beta01.log 2>&1 &
MODAL_PROFILE=anirudh nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms league:10 --steps 30 --item ABL_G_steps30 --per-gpu 2 > results/logs/abl_g_steps30.log 2>&1 &
MODAL_PROFILE=anirudh nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms league:10 --steps 120 --item ABL_G_steps120 --per-gpu 2 > results/logs/abl_g_steps120.log 2>&1 &
for d in ILI ETTh1; do
  MODAL_PROFILE=anirudh nohup $M run --detach $A::prompted --dataset $d --seeds 10 --anonymous --item ABL_P_anon --per-gpu 2 > results/logs/abl_p_anon_$d.log 2>&1 &
  MODAL_PROFILE=anirudh nohup $M run --detach $A::prompted --dataset $d --seeds 10 --no-constrained --item ABL_P_repair --per-gpu 2 > results/logs/abl_p_repair_$d.log 2>&1 &
done
```

Only if Soumitra is not on Modal: also launch the reference,
`... transfer --source ETTh1 --target ILI --arms league:10 --item ABL_G_ref --per-gpu 2`.

Bring back each item (folder ILI_T24 for AB-G, both tables for AB-P):

```
for it in ABL_G_beta0 ABL_G_beta01 ABL_G_steps30 ABL_G_steps120 ABL_P_anon ABL_P_repair; do
  for t in ILI_T24 ETTh1_T336; do
    S=$(mktemp -d); MODAL_PROFILE=anirudh $M volume get leaguetst-runs /$it/$t $S 2>/dev/null || continue
    mkdir -p results/search/$it/$t results/runs/$it/$t
    cp $S/$t/*_runs.jsonl results/search/$it/$t/
    cp $S/$t/training.jsonl $S/$t/errors.jsonl results/runs/$it/$t/ 2>/dev/null
  done
done
```

Checks: 10 training records and 10 deployments per AB-G item, 10 searches
per AB-P item and table, env blocks matching docs/prereg_E4.md A25, and no
out-of-memory errors. Then scripts/ablations/llm_tests.py computes AB-G1
and AB-P1 against their references (Holm per family) into
results/ablations/AB-G/ and results/ablations/AB-P/.

### WP4: GPU Phase 2 (AB-G2, AB-P2), after 26 Nov 18:00 ET

Write scripts/ablate_llm.py, a new file: plans built with
leaguetst.architect.runner (transfer_plan, prompted_plan), then the job
fields group, n_rivals, snapshot_every, temperature, model and revision set
per arm; executors for Modal (a separate app named leaguetst-ablate with
the same pinned image as scripts/modal_llm.py) and for a local GPU (as
scripts/llm_local.py). Resolve and pin the 0.5B and 3B model revisions
before launch. Tests for the plan builders on CPU. Then add an amendment
to docs/prereg_ablations.md with the script's sha256 and the revisions, get
a verifier and a critic pass, and only then launch. About $10.

### WP5: write-up, 27 Nov to 2 Dec

scripts/ablations/report.py writes docs/ablations.md (tables and figures
from results/ablations/, nothing typed): regret curves with intervals,
probability-of-improvement profiles, the detectable-effect table, the
ablation tables with Holm-adjusted p and effect sizes, the RQ3 field
ranking across tables. Also a draft of the NeurIPS paper checklist
(docs/neurips_checklist.md) with each answer pointing to a file. Diya's
side merges and links it from the report.

### WP6: Phase B, after the 30 Nov test (toward the paper)

Each item is registered by amendment before it runs.

* External baselines on the oracle: the LLM-based optimisers and
  multi-fidelity HPO methods from WP0 that can run on table lookups, with
  the same budgets and seeds.
* A third dataset (for example ETTh2): a person downloads it, splits it
  with the test protocol and vaults the test rows; a new E0-style lock;
  then its table on GPU (about $14 per seed on A10G). Claude never sees the
  full file.
* Benchmark release: tables, loader, datasheet, licence, metadata.
* The wider design space (A16, docs/wide_space.md), on a costed subset.

## Credit (Anirudh's Modal, $30 free per month)

| Item | Est. cost |
|---|---|
| AB-G1, 4 arms x 10 seeds | about $9 |
| AB-G reference (only if not shared with Soumitra) | about $4 |
| AB-P1, 2 arms x 2 tables x 10 seeds | about $1.5 |
| AB-G2 and AB-P2 (Phase 2) | about $10 |
| Reserve for reruns | the rest |

Check `modal billing summary` before each launch; stop if a launch would
pass the free credit.

## Timeline

| Date | Anirudh | Soumitra | Diya |
|---|---|---|---|
| 24 Nov | setup, WP0, start WP1 and WP2 | setup, launch J1 and J2 | push main, merge, picks batch 2 prep |
| 25 Nov | launch WP3 | J1, J2 finish | E4 summaries, E5 with GRPO |
| 26 Nov | WP3 analysis, WP1/WP2 results | J3 if asked; cut-off 18:00 | K4 inputs |
| 27 Nov | WP4 launch (after review) | done | picks batch 2, K4 (28 Nov) |
| 28 Nov to 2 Dec | WP5, push branch | | prereg, test 30 Nov, report 3 Dec |
| December | WP6 | | paper |

## First message for Anirudh's Claude

> Read CLAUDE.md, docs/STATE.md, docs/prereg_ablations.md and
> docs/HANDOFF_anirudh.md. Check the code digests and run the tests. Then
> work through WP0, WP1 and WP2 on CPU while WP3 runs on Modal profile
> anirudh, following the rules section exactly. Commit each verified result
> on branch anirudh-ablations. Do not push, and do not edit the files listed
> in docs/e4_rerun.sha256 until Diya says the E4 rerun is complete.
