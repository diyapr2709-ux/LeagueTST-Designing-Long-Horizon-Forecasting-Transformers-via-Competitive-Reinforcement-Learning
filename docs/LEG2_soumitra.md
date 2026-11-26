# Leg 2 for Soumitra: 25 to 28 Nov

For Soumitra and the Claude Code session on Soumitra's account. Written
2026-11-24 evening. The E4 rerun (J1 and J2) is merged into main (PR #1,
70461a8): 45 of 45 policies deployed, no on-policy stop, no error.

This leg runs end to end on Soumitra's side. Soumitra's Claude makes every
call in this file by the rules written here, including whether J3 runs and
whether each GPU family runs on Modal or on the local RTX, without waiting
on Diya. Everything is done and pushed by 28 Nov 23:59 ET, before Diya's
pre-registration (29 Nov) and single test evaluation (30 Nov).
Every rule of CLAUDE.md and docs/HANDOFF_soumitra.md still applies. Work on
a new branch, soumitra-leg2, from origin/main at the commit that adds this
file. Coordinate directly with Anirudh where noted.

## Answers to the open questions

* grpo_none_ILI_seed109 is not retried, by the registered rule (A29
  addendum): it trained once and its input succeeded. Its adapter and
  provenance.json also show that training finished without an on-policy
  stop: provenance.json is written only after train() returns, and a stop
  raises before the adapter is saved. The A26 worst case therefore does not
  count it as failed; adding it now would be a choice made after the runs.
  Its missing training record is disclosed and affects no test. The worst
  case treats grpo_league_ILI_seed109 and grpo_none_ILI_seed108 as failed.
* Launch logs: commit them; they are the A29 cross-check. First
  `grep -n -E "ak-|as-|token" results/logs/soumitra_e4_*.log` must print
  nothing, then `git add -f results/logs/soumitra_e4_*.log`.
* Git hash 894b65a: the 25 J1 deployment records and one training record
  carry it, but the rebase removed it from the shared history. Tag it from
  your reflog (`git tag e4-j1-launch 894b65a`; `git diff --stat e1a063c
  894b65a` must list only results files) and push the tag.
* Cost: $22.17 for 45 policies is about $0.49 per policy trained and
  deployed (about 25 minutes of training per policy with two sharing an
  A10G, so about 13 GPU-minutes, 2 to 3 minutes per deployment search, and
  container start-up on top). The estimates below use it.

## L0: close E4 (25 Nov, morning)

1. Commit the logs and tag 894b65a.
2. Commit the volume listing that dates the two adapters behind the answer
   above: `MODAL_PROFILE=soumitra .venv/bin/modal volume ls leaguetst-runs /E4/adapters/grpo_none_ILI_seed108`
   and the same for seed109, with their modification times, into
   results/runs/E4/soumitra_ETTh1_T336/adapters_ls.txt.
3. Push soumitra-leg2 and the tag (Soumitra pushes; Claude never does).

## L1: independent E4 analysis and the J3 decision (CPU, 25 Nov)

1. Write scripts/independent/e4_analysis.py from docs/prereg_E4.md alone
   (families 1 to 3, A17, A18, A25 to A30, with the pooled seed SD and
   held-out scoring as registered there and in docs/PLAN.md). It imports
   nothing from leaguetst/ or scripts/: it reads results/ledger.jsonl and
   the runs files directly. Commit it before opening scripts/replay.py,
   scripts/picks.py or any E4 summary file (results/search/E4/*/summary.*).
2. Run it. J3 is decided by A30: J3 runs only if, on ILI AUC, league is
   better than none with the two-sided permutation test at p <= 0.05/6.
   Write the decision and the p-value in the commit message.
3. Then run `scripts/replay.py --item E4 --summarize-only` in a separate
   git worktree (never committed) and diff every statistic and decision
   against yours. Any difference above 1e-9, or any differing decision,
   is written up with both values and a guess at the cause.
4. The ETTh1-target tests need pick batch 2, which trains forecasters on
   the data only Diya's workspace holds. Add them if it is on origin/main
   by 28 Nov 12:00 ET; otherwise say they are pending.

Outputs: results/independent/E4_soumitra.json and
docs/verdicts/E4_independent_soumitra.md, with a verifier and a critic
subagent before the commit. Never write results/ledger.jsonl.

## L2: J3, only if L1 says so (Modal, done by 26 Nov 18:00 ET)

J3 runs on Modal, the backend of J1 and J2 (A25), never on the RTX:
5 policies, about $2.5 and 40 minutes.

```
M=.venv/bin/modal; A=scripts/modal_llm.py; mkdir -p results/logs
MODAL_PROFILE=soumitra nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms uniform:5 --per-gpu 2 > results/logs/soumitra_e4_uniform.log 2>&1 &
```

Before launch, `modal billing summary` must show at least $5 free, and the
spend limit must sit above month-to-date spend plus the planned spend: a
workspace disabled mid-run kills the driver (what happened to J1). Same
checks, bring-back (folder ILI_T24) and A29 screenshots as J1 and J2. If
A30 says no, J3 is not run and nothing replaces it.

## L3: AB-L, did GRPO learn? (CPU, 25 to 26 Nov)

Moved to you from Anirudh's WP4 (docs/prereg_ablations.md, AB-L and AB1).

* For every E4 policy with a training record, the Spearman slope of
  reward_mean over its training steps.
* Per source table (ILI for J1, ETTh1 for J2), test the league policies'
  slopes against zero with an exact sign-flip test over policies (2^10
  sign patterns; leaguetst.eval.stats has none, so write and test it).
  Holm over the 2 source tables.
* Descriptive: best_in_history resets every episode, and an episode's
  budget (3000) equals a prompted search's, so use each policy's final
  episode only: its best_in_history at the episode's end against prompted
  Qwen's best at the same spend on the source table
  (results/search/E3/<table>/qwen_prompted_runs.jsonl; its trace holds
  [epochs, config id], so look the MSE up in the table). Skip null rewards
  (one exists: grpo_none_ETTh1_seed108, step 30).
* Exploratory: the reward at each episode's first step (the same prompt
  every time), last episode against first; the same slopes for the none
  and uniform arms; on-policy gaps; invalid, repeat and unique rates.

Write scripts/ablations/learning.py with tests/test_ablations_learning.py.
Outputs results/ablations/AB-L.json and AB-L.md, verifier and critic before
the commit, the critic's verdict in docs/verdicts/AB-L.md. Tell Anirudh when
it is pushed.

## L4: GPU ablation arms, Modal or RTX (25 to 28 Nov)

Two families moved to you: AB-P model scale (Qwen2.5-0.5B and 3B around
the 1.5B reference) and the AB-G KL trend (beta 0 and 0.1 around 0.04).
Scale first, then KL: model size is what reviewers ask about first when an
LLM method shows no effect, and it takes about half the GPU time.

Backend, chosen by your Claude per family before that family's first
launch, recorded in amendment AB2, never revisited:

* Modal, if `modal billing summary` (after J3) shows the family's estimate
  plus $5 free. KL arms: scripts/modal_llm.py transfer --beta, reference
  the J2 league arm, about $10. Scale: about $8, reference Anirudh's
  ABL_P_ref on Modal.
* Otherwise the local RTX (at least 12 GB), with the family's reference
  rerun on the same card: ABL_G_ref_local for KL, ABL_P_ref_local for
  scale. Local references serve only local arms.
* Neither: the family is not run and is reported as not run.
* An item launched on one backend is never rerun on the other.

At $8.15 less J3, Modal covers at most one family, so expect the RTX. If
Anirudh says on 25 Nov that he cannot run his WP3 on Modal, his WP3 items
(docs/LEG2_anirudh.md) join the RTX queue after the KL arms, with the
local references, and the same AB2 covers them.

Step 1, the launcher (25 to 26 Nov). scripts/llm_local.py is frozen (one of
the ten files) and has no --beta, --steps, --anonymous or model flags.
Write scripts/ablate_llm.py, a new file that builds plans with
leaguetst.architect.runner (unchanged) and imports llm_local.py's executor
and GPU check instead of copying them:

* `transfer` with --source, --target, --arms, --steps, --beta, --item,
  --per-gpu;
* `prompted` with --dataset, --seeds, --anonymous, --no-constrained,
  --model, --revision, --item, --per-gpu, and a method suffix naming the
  model and "_local";
* only if Modal is chosen for scale, a Modal entrypoint that reuses
  scripts/modal_llm.py's app and driver unchanged.

Tests in tests/test_ablate_llm.py: each flag reaches the jobs, defaults
equal the E4 recipe, item and method names. Pin the Hugging Face commits of
Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct. Register in amendment AB2 at
the end of docs/prereg_ablations.md, before any run: the script's sha256,
the revisions, the item names, each family's backend, and on the RTX the
card and --per-gpu. Verifier and critic before the commit.

Step 2, the runs (26 Nov evening to 28 Nov; overnight is fine). First
`shasum -a 256 -c docs/e4_rerun.sha256` must pass. On the RTX, one after
the other (24 GB at --per-gpu 2; 12 to 16 GB at 1, in about 1 to 1.5 times
the total time):

| Item | What | Time at 2 per GPU |
|---|---|---|
| ABL_P_ref_local | prompted Qwen2.5-1.5B, 30 seeds, ILI and ETTh1 | about 1 h |
| ABL_P_05b_local | same with 0.5B | under 1 h |
| ABL_P_3b_local | same with 3B | about 2 h |
| ABL_G_ref_local | ETTh1 to ILI, league:10, E4 recipe | about 2.5 h |
| ABL_G_beta0_local | same, --beta 0 | about 2.5 h |
| ABL_G_beta01_local | same, --beta 0.1 | about 2.5 h |

On Modal, drop the reference rows and use the commands of
docs/HANDOFF_anirudh.md WP3 with your profile; stop any llm_batch input
still running after an hour. Keep the machine awake; a relaunch keeps
finished work. Each item logs to its own file under results/logs/. Bring
back by copying each item's *_runs.jsonl, training.jsonl and errors.jsonl
from results/runs/llm_local/<item>/<name>/ (or the Modal volume) to
results/search/<item>/<name>/ and results/runs/<item>/<name>/ (<name> is
ILI_T24 or ETTh1_T336), with the J1 and J2 checks (counts, env blocks, no
out-of-memory, on-policy stops reported and never retrained). Commit, push,
and tell Anirudh, whose scripts/ablations/llm_tests.py computes the trends
from origin/soumitra-leg2.

(Superseded 26 Nov by amendment AB4: ablation records go to results/ablations/rtx/<item>/<name>/, never to results/search/ or results/runs/<item>/, which scripts/picks.py and scripts/e5_referee.py read. The nine AB2 items were moved there in the AB4 commit.)

## L5: fresh-clone reproduction audit (CPU, 27 to 28 Nov)

1. Clone origin/main at its newest commit on 27 Nov into a new folder. Set
   up from README.md and the requirements files only; note every step that
   is missing or wrong.
2. Run the test suite. Regenerate every CPU-derived output in that clone:
   the E3 and E4 replay summaries, the picks analysis, E5, AB-L and
   scripts/report.py. Then `git status` and `git diff --stat` in the clone
   list what changed; nothing in a regenerated file should.
3. Write docs/REPRODUCE.md on soumitra-leg2: the exact commands, run times,
   which outputs need a GPU, and every discrepancy. Commit nothing else
   from the audit clone.

The clone has no data files (only data/trainval/MANIFEST.json is tracked)
and needs none: the tables are in the ledger. Never fetch data.

## Dates

| Date | Soumitra |
|---|---|
| 25 Nov | L0; L1 and the J3 decision; L3; L4 step 1 |
| 26 Nov | L2 if A30 says so, done by 18:00 ET; L3 pushed; AB2 reviewed and committed; L4 step 2 starts in the evening (scale) |
| 27 Nov | L4 step 2 (KL arms); L1 step 3; L5 starts |
| 28 Nov | L4 bring-back; L5 and docs/REPRODUCE.md; push everything by 23:59 ET |

About 12 hours of RTX time (scale about 4, KL about 7.5) fit between
26 Nov evening and 28 Nov, with room for Anirudh's WP3 items if they come.
Anything pushed after 28 Nov 23:59 ET reaches the paper, not the report.

## Rules for this leg

* Never edit the ten files in docs/e4_rerun.sha256: every AB-G arm must run
  under the reference's digests (AB1). New code goes in new files.
* Never write results/ledger.jsonl, docs/claims/ or docs/STATE.md: results
  stay on soumitra-leg2 and are recorded when merged.
* Free credit only. No test rows anywhere. Claude never pushes.

## First message for Soumitra's Claude

> Read CLAUDE.md, docs/STATE.md, docs/HANDOFF_soumitra.md,
> docs/LEG2_soumitra.md, docs/prereg_E4.md (A25 to A30) and
> docs/prereg_ablations.md (AB-L, AB1). Create branch soumitra-leg2 from
> origin/main. Work through L0 to L5 as dated, making every decision by
> the rules in LEG2_soumitra.md (J3 by A30, Modal or RTX per family by L4)
> without waiting on Diya, each step with a verifier and a critic subagent
> before its commit. Tell Soumitra what to push and when to message
> Anirudh. Do not push.
