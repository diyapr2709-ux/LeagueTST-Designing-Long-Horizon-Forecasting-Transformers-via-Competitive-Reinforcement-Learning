# Leg 2 for Anirudh: 25 to 28 Nov

For Anirudh and the Claude Code session on Anirudh's account. Written
2026-11-24 evening. It changes docs/HANDOFF_anirudh.md where it says so;
everything else there, docs/prereg_ablations.md and CLAUDE.md still apply.
Amendment AB1 at the end of docs/prereg_ablations.md registers the changes.
Every deadline in docs/HANDOFF_anirudh.md moves to 28 Nov 23:59 ET at the
latest, before Diya's pre-registration (29 Nov) and test (30 Nov).

## What changed

1. The E4 rerun is complete and merged into main (PR #1, 70461a8; Soumitra's
   Modal workspace, 45 of 45 policies deployed, no on-policy stop). The
   AB-G reference is Soumitra's J2 league arm:
   results/search/E4/ILI_T24/grpo_league_from_ETTh1_runs.jsonl and
   results/runs/E4/soumitra_ILI_T24/training.jsonl. Do not run ABL_G_ref.
2. Measured cost is above the handoff's estimate: $0.49 per GRPO policy
   trained and deployed (1.5 times the handoff's $0.33), and an estimated
   $0.06 per prompted search (about twice; not yet metered: 2 minutes at 2
   per A10G plus container start-up). WP3 is cut to what
   fits $30 with a margin (below). The KL arms and model scale move to
   Soumitra, whose Claude runs each on Modal or his local RTX by the rule
   in AB1; nothing waits for December credit.
3. Soumitra's workspace hit its spend limit mid-run and Modal disabled it,
   which killed the driver and cost a retrain. Set your spend limit above
   month-to-date spend plus the planned spend, with $5 to spare, and check
   `modal billing summary` before every launch.
4. A29 and its addendum (docs/prereg_E4.md) apply to every ABL_* GPU item:
   after each item, screenshot the llm_batch inputs in the Modal dashboard,
   name every retried run in the commit, relaunch into a new log file, and
   stop any llm_batch input still running after an hour (2.5 hours for
   ABL_G_steps120, whose batches take about an hour).
5. AB-L (WP4) moves to Soumitra, who reads the training logs. Your
   power.py and report.py read results/ablations/AB-L.json from
   origin/soumitra-leg2; merge that branch into yours yourself.
6. The ten files in docs/e4_rerun.sha256 stay frozen until the last AB-G
   arm has run (28 Nov at the latest), because every AB-G arm runs under the
   reference's digests. New code goes in new files only. Soumitra writes
   scripts/ablate_llm.py (a local launcher with the flags llm_local.py
   lacks); you no longer write it.

## Step 0 (25 Nov, morning)

Nothing from you is on GitHub yet. Merge origin/main (the commit that adds
this file) into anirudh-ablations, run the tests, and commit what exists
with a one-line status per work package. The same commit message confirms
that no ablation result existed before amendment AB1, or names any that
did. Push the branch (Anirudh pushes; Claude never does).

## WP3, revised (launch 25 Nov)

| Item | Policies or searches | Est. cost |
|---|---|---|
| ABL_P_ref, both tables | 60 searches | about $3.5 |
| ABL_P_anon, both tables | 60 searches | about $3.5 |
| ABL_P_repair, both tables | 60 searches | about $4 |
| ABL_G_steps30 | 10 policies | about $3.5 |
| ABL_G_steps120 | 10 policies | about $9.5 |
| Total | | about $24 of $30 |

Launch in two waves. Wave 1 (about $18): ABL_P_ref on ILI first, detached;
once its log shows the first search finished (the model cache is then
warm), the step arms, ABL_P_ref on ETTh1 and ABL_P_anon on ILI:

```
M=.venv/bin/modal; A=scripts/modal_llm.py; mkdir -p results/logs
MODAL_PROFILE=anirudh nohup $M run --detach $A::prompted --dataset ILI --seeds 30 --item ABL_P_ref --per-gpu 2 > results/logs/abl_p_ref_ILI.log 2>&1 &
# wait for the first finished search in abl_p_ref_ILI.log, then:
MODAL_PROFILE=anirudh nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms league:10 --steps 120 --item ABL_G_steps120 --per-gpu 2 > results/logs/abl_g_steps120.log 2>&1 &
MODAL_PROFILE=anirudh nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms league:10 --steps 30 --item ABL_G_steps30 --per-gpu 2 > results/logs/abl_g_steps30.log 2>&1 &
MODAL_PROFILE=anirudh nohup $M run --detach $A::prompted --dataset ETTh1 --seeds 30 --item ABL_P_ref --per-gpu 2 > results/logs/abl_p_ref_ETTh1.log 2>&1 &
MODAL_PROFILE=anirudh nohup $M run --detach $A::prompted --dataset ILI --seeds 30 --anonymous --item ABL_P_anon --per-gpu 2 > results/logs/abl_p_anon_ILI.log 2>&1 &
```

Wave 2 (about $6), only after every wave-1 app has stopped and
`modal billing summary` shows the spend: launch ABL_P_anon on ETTh1, then
ABL_P_repair on ILI and ETTh1, each only if the billing summary shows its
estimate plus $5 still free (so the cut order is repair first, then anon on
ETTh1):

```
MODAL_PROFILE=anirudh nohup $M run --detach $A::prompted --dataset ETTh1 --seeds 30 --anonymous --item ABL_P_anon --per-gpu 2 > results/logs/abl_p_anon_ETTh1.log 2>&1 &
for d in ILI ETTh1; do
  MODAL_PROFILE=anirudh nohup $M run --detach $A::prompted --dataset $d --seeds 30 --no-constrained --item ABL_P_repair --per-gpu 2 > results/logs/abl_p_repair_$d.log 2>&1 &
done
```

The step trend this month has levels 0 (ABL_P_ref on ILI), 30, 60 (the
reference) and 120, as registered. Bring-back, checks and llm_tests.py as
in the handoff, plus the A29 screenshots under results/runs/ABL_*/.

## Moved to Soumitra (Modal or RTX, by 28 Nov)

* The KL trend (beta 0, 0.04, 0.1) and model scale (Qwen2.5-0.5B, 1.5B,
  3B) are Soumitra's (docs/LEG2_soumitra.md, L4). His Claude picks Modal or
  his RTX per family by AB1's rule; on the RTX the reference reruns on the
  same card (ABL_G_ref_local, ABL_P_ref_local). Your llm_tests.py computes
  both trends from origin/soumitra-leg2: local references serve local
  arms, Modal references (J2 league, ABL_P_ref) Modal arms.
* If you cannot run WP3 on Modal at all, tell Soumitra on 25 Nov: the same
  items then join his RTX queue after the KL arms, with the local
  references. A WP3 item you have launched is never rerun elsewhere.
* Anything not run by 28 Nov 23:59 ET is reported as not run and its family
  keeps its registered Holm size (AB1).

## CPU work, unchanged in content

* WP1 (AB-C, AB-Q, AB-N) and WP2 (AB-X: DEHB, SMAC, Hyperband) as in the
  handoff. These matter most for the report: with RQ1 null, the noise-free
  replay (AB-Q) and the external baselines (AB-X) are what a reviewer asks
  for first.
* WP5, CPU only now (AB-R, ST, RQ3): power.py adds the E4 families once
  Soumitra's E4 analysis (results/independent/E4_soumitra.json) is on
  origin/soumitra-leg2 (expected 25 to 26 Nov).
* WP6: docs/ablations.md and docs/neurips_checklist.md, with AB-L from
  Soumitra and the KL trend if it lands. Soumitra writes
  docs/REPRODUCE.md; the checklist's reproducibility answers point to it.

## Dates

| Date | Anirudh | Soumitra |
|---|---|---|
| 25 Nov | step 0; WP3 wave 1; WP1, WP2 | E4 analysis and J3 decision; AB-L; launcher |
| 26 Nov | WP3 wave 2 and bring-back; llm_tests.py on WP3; WP1, WP2 done | J3 if A30 says so; AB-L pushed; AB2; RTX runs start |
| 27 Nov | WP5 CPU with Soumitra's E4 analysis; WP6 starts | RTX runs (KL); reproduction audit |
| 28 Nov | llm_tests.py on Soumitra's scale and KL results; WP6 done; push by 23:59 ET | bring-back; push by 23:59 ET |

Anything pushed after 28 Nov 23:59 ET reaches the paper, not the report
(due 3 Dec).

WP7 (Phase B) is paper work after the report and is not part of this leg:
its first items need the data files, which only Diya's workspaces hold.

## First message for Anirudh's Claude

> Read CLAUDE.md, docs/STATE.md, docs/HANDOFF_anirudh.md,
> docs/LEG2_anirudh.md and docs/prereg_ablations.md (including amendment
> AB1). Merge origin/main into anirudh-ablations and do step 0. Launch the revised
> WP3 on Modal profile anirudh in the order given, checking the billing
> summary first, and continue WP1, WP2 and WP5 on CPU while it runs. Commit
> each verified result on anirudh-ablations. Do not push, and never edit
> the files in docs/e4_rerun.sha256.
