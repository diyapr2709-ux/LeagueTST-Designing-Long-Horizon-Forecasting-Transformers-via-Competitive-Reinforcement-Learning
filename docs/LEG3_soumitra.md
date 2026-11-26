# Leg 3 for Soumitra: the WP3 arms on the RTX, 26 to 28 Nov

For Soumitra and the Claude Code session on Soumitra's account. Written
2026-11-26, early morning ET, and revised the same day after the critic's and
the verifier's reviews. Amendment AB4 at the end of docs/prereg_ablations.md
registers everything here; docs/LEG2_soumitra.md, AB1 and AB2 still apply
where this file does not change them. Like leg 2, it runs end to end on your
side without waiting on Diya. Soumitra pushes everything by 28 Nov 23:59 ET;
Claude never pushes.

## What changed

1. J3 stays unrun. Diya's side re-scored your A30 test with scripts/replay.py
   on main in a throwaway worktree and got the same exact p-value. That
   re-score is unrecorded; your commit 7808ea6 is the record.
2. Anirudh did not launch WP3 on Modal (AB3). docs/LEG2_anirudh.md sends
   those items to your RTX queue in that case, so four arms are yours:
   ABL_P_anon_local and ABL_P_repair_local (ILI and ETTh1),
   ABL_G_steps30_local and ABL_G_steps120_local. They run against the local
   references you already have (ABL_P_ref_local, ABL_G_ref_local). The Modal
   items ABL_P_ref, ABL_P_anon, ABL_P_repair, ABL_G_steps30 and
   ABL_G_steps120 are never launched anywhere.
3. Your nine AB2 items moved from results/search/ and results/runs/ to
   results/ablations/rtx/ (git mv, contents unchanged; digests frozen in
   results/ablations/rtx/AB2_records.sha256). Leg 2 told you to copy them
   into results/search/, which was a mistake on Diya's side: scripts/picks.py
   and scripts/e5_referee.py read every folder there, and your GRPO arms
   reuse E4's method name, so pick batch 2 and E5 would have pooled them.
   Never copy ablation records into results/search/.
4. Nobody was computing the registered AB-P and AB-G tests: Anirudh's
   llm_tests.py only covers his Mac runs and left the registered items to
   "their owner", and your commits left them to llm_tests.py.
   scripts/ablations/llm_registered.py (new, with
   tests/test_llm_registered.py) computes them. You run it once at the end
   (step 4).
5. New queue: scripts/ablations/run_wp3_rtx_queue.sh. It runs the same
   launcher as AB2 with the same rules, takes the AB2 queue's lock, refuses
   unless HEAD contains AB4 and the run-path code equals the AB2 commit,
   stops every job at 28 Nov 18:00 ET, and copies each item's record files
   to results/ablations/rtx/<item>/<table>/ when the item ends.
6. Deadline for the GPU work: 28 Nov 18:00 ET (AB4), so the analysis and the
   push fit before 23:59 ET. An arm unfinished at 18:00 counts as not run.

## Rules for your Claude (AB4)

* Never edit or delete a row of any record file (*_runs.jsonl,
  training.jsonl, errors.jsonl, retried.txt). The runner writes a
  "no adapter to deploy" row for every stopped policy and leaves a crash
  row after a relaunch fixes the crash; llm_registered.py knows both. If a
  check misbehaves, log it in docs/ISSUES.md and tell Diya; a fix is a
  dated amendment committed before the families are computed.
* Never edit run_wp3_rtx_queue.sh, run_l4_queue.sh, llm_registered.py,
  llm_tests.py, common.py, ablate_llm.py or any file in
  docs/e4_rerun.sha256.
* Never compute an AUC, regret or test of any arm before step 4. `--check`
  prints counts only.

## Step 1 (26 Nov): sync, attest, check

```
git fetch origin && git checkout soumitra-leg2 && git merge --ff-only origin/main
```

If the fast-forward fails because soumitra-leg2 has commits that main does
not, run `git merge origin/main` instead (never rebase); if it conflicts,
stop and tell Diya. Then:

```
.venv/bin/python -m pytest -q
shasum -a 256 -c docs/e4_rerun.sha256
shasum -a 256 -c results/ablations/rtx/AB2_records.sha256
shasum -a 256 scripts/ablate_llm.py scripts/ablations/run_wp3_rtx_queue.sh scripts/ablations/llm_registered.py \
  scripts/ablations/llm_tests.py scripts/ablations/common.py
.venv/bin/python scripts/ablations/llm_registered.py --check
ls results/search results/runs/llm_local
```

The digests must equal the ones in AB4 and AB2. `--check` should list 9
arms complete and 6 incomplete (the new ones). `ls results/search` should
show only E3, E4 and E4smoke; delete any untracked ABL_* folder left there
by hand after checking that its files exist under results/ablations/rtx/.
results/runs/llm_local/ must hold none of the four new items.

Attestation (AB4): your launch commit, made before the queue starts, states
that results/runs/llm_local/ holds no ABL_P_anon_local, ABL_P_repair_local,
ABL_G_steps30_local or ABL_G_steps120_local output, and that your side
computed no AUC, regret or test of any AB2 item, or names what it computed.

## Step 2 (26 Nov): launch

```
setsid nohup bash scripts/ablations/run_wp3_rtx_queue.sh >/dev/null 2>&1 &
```

Launch with HEAD at the AB4 commit or a descendant and a clean tree under
leaguetst/ and scripts/ (otherwise the queue refuses). Order and expected
times, from your measured AB2 times (repair has not been timed):

| Item | Table | Expected |
|---|---|---|
| ABL_P_anon_local | ILI | about 30 min |
| ABL_P_repair_local | ILI | 30 to 60 min |
| ABL_G_steps30_local | ILI | about 2 h |
| ABL_G_steps120_local | ILI | 6.5 to 7 h |
| ABL_P_anon_local | ETTh1 | about 20 min |
| ABL_P_repair_local | ETTh1 | 20 to 45 min |

That is about 10 to 11.5 hours, so a launch on 26 Nov finishes on 27 Nov,
with a day to spare for relaunches. Poll
results/logs/wp3_rtx_queue_status.txt. AB2's rules apply as written:

* Hand-stop: one hour with no new runs row for a prompted item, or 2.5
  hours with no new GRPO step row (and no new record) for a transfer item.
* Stale lock: the AB2 procedure.
* Retried runs: the AB2 definition.
* Relaunch: the same command, settings unchanged, until 28 Nov 18:00 ET.
  Never change history lines, group size, tokens, model, card or backend to
  make a job fit.

## Step 3 (per item): bring back and commit

The queue copies the record files. For each finished item:

1. `llm_registered.py --check` shows it complete.
2. Run the AB2 checks: counts, env blocks, no out-of-memory or traceback in
   its log, and the key-pattern grep over the log.
3. If a policy was retried, write
   results/ablations/rtx/<item>/ILI_T24/retried.txt with one run name per
   line, in this commit, never edited afterwards. Write no file if none
   was.
4. Commit the item folder, results/logs/wp3_rtx_queue_status.txt and the
   queue's digest file. The commit message carries counts and checks only,
   no AUC. The .log files stay local (gitignored).

## Step 4 (27 or 28 Nov): the registered families, once

Before running anything, a verifier subagent writes and commits its own
re-scoring script (reading results/ablations/rtx/ and the registered test
definitions, importing nothing from llm_registered.py), without seeing any
output. Then, when `--check` shows all 15 arms complete:

```
.venv/bin/python scripts/ablations/llm_registered.py
```

It writes results/ablations/LLM-registered.json: AB-P (Holm over 3), AB-G
(Holm over 2) with the registered secondary step analyses, the hardware
check and the ETTh1 descriptives. If an arm is still unfinished at 28 Nov
18:00 ET, run it with `--final` after 18:00 instead (missing arms count as
not run; an unfinished ETTh1 table does not void its ILI arm). The script
refuses `--final` before 18:00, refuses a second run, and refuses
uncommitted records or code. Then:

* the verifier runs its script and compares;
* a critic subagent attacks the result;
* copy the critic's final verdict verbatim into docs/verdicts/AB4.md;
* commit, and Soumitra pushes by 23:59 ET.

If a bug is found after the run, do not fix it in place: log it, tell Diya,
and the fix is a dated amendment; the families are then recomputed once and
both outputs are kept (AB4).

## Dates

| Date | Soumitra |
|---|---|
| 26 Nov | step 1 and the attestation commit; step 2 launch; step 3 for the two ILI prompted items |
| 27 Nov | step 3 for the step arms and ETTh1; verifier script committed; step 4 once all 15 are complete; L5 continues |
| 28 Nov | relaunches until 18:00 ET; `--final` after 18:00 if needed; verifier, critic, verdict; Soumitra pushes by 23:59 ET |

## For Anirudh

* Do not read results/ablations/rtx/ and do not edit common.py or
  llm_tests.py until results/ablations/LLM-registered.json is committed:
  AB4 froze both files' digests, and the full run refuses if they change.
* scripts/ablations/llm_tests.py now points to results/ablations/rtx/ and to
  llm_registered.py for the registered families. Rerun it so LLM.json's
  registered note is current (no Mac number changes).
* docs/ablations.md and docs/neurips_checklist.md: add the registered AB-P
  and AB-G results from results/ablations/LLM-registered.json once Soumitra
  pushes it (27 or 28 Nov), with the AB4 report disclosure and the step
  caveat. Until then, report them as pending, not as not run. Push by
  28 Nov 23:59 ET.
* The Mac results stay exploratory. They are never pooled with the RTX arms
  and never replace one (AB4).

## First message for Soumitra's Claude

> Read docs/LEG3_soumitra.md and amendment AB4 at the end of
> docs/prereg_ablations.md, then docs/LEG2_soumitra.md and AB2 for the rules
> they keep. Do step 1, including the attestation commit. If every check
> passes, launch the WP3 queue (step 2), poll it, and bring back and commit
> each item as it finishes (step 3). Run step 4 once all 15 arms are
> complete, with a verifier whose script is committed first and a critic.
> Commit on soumitra-leg2; Soumitra pushes. Never copy records into
> results/search/, never edit or delete a row of a record file, and never
> edit run_wp3_rtx_queue.sh, run_l4_queue.sh, llm_registered.py,
> llm_tests.py, common.py, ablate_llm.py or the files in
> docs/e4_rerun.sha256. If something misbehaves, log it in docs/ISSUES.md
> and stop.
