# Handoff: GPU work on Soumitra's Modal workspace

For Soumitra and the Claude Code session on Soumitra's account. Written
2026-11-24. Read CLAUDE.md first: every hard rule there applies to you too.

## Why this exists

Diya's two Modal workspaces are nearly out of free credit (diyapr has about
$5 left, $3.5 of it held for E6 reruns; diya-027 is shared with another
project). The E4 GRPO training on 23 Nov lost 34 of 45 runs to CUDA out of
memory (three jobs on one A10G; docs/ISSUES.md). E4 is the pre-registered
primary endpoint (docs/prereg_E4.md), so it is rerun on Soumitra's
workspace, which has its own $30 monthly credit. The owner made this
decision on 2026-11-24 (docs/ISSUES.md). CLAUDE.md rule 10 still names only
Diya's two workspaces; until Diya adds the line below to rule 10 by hand,
this file is the authority for the soumitra profile:

    Owner's decision, 2026-11-24: the E4 GRPO reruns, which need no data
    volume, run on Soumitra's workspace with MODAL_PROFILE=soumitra
    (docs/HANDOFF_soumitra.md).

All 45 policies are retrained there with the same code, so every E4 result
comes from one workspace and one code version. The 11 policies that
finished on diyapr are superseded and are never reported on their own.

## What runs where

| Job | Workspace | When | Est. cost |
|---|---|---|---|
| J1: E4 GRPO, ILI source, deploy on ETTh1 (league 10, none 10, uniform 5) | Soumitra | today | about $6 |
| J2: E4 GRPO, ETTh1 source, deploy on ILI (league 10, none 10) | Soumitra | today, beside J1 | about $5 |
| J3: uniform-rival arm, ETTh1 source, 5 seeds (only if Diya's side says the A17 condition holds) | Soumitra | 26 Nov | about $1.5 |
| Pick reseeds of the E4 picks (batch 2), E6 reruns, test prereg, test evaluation | Diya | 25 to 30 Nov | Diya's credit |

The GRPO app (scripts/modal_llm.py) reads the table from the committed
results/ledger.jsonl and mounts only two volumes, leaguetst-hf (the Qwen
download cache) and leaguetst-runs (outputs). Both are created on first use.
It needs no data volume. Do not create leaguetst-data and do not upload
anything to Modal (CLAUDE.md rule 10); jobs that train forecasters stay on
Diya's side.

## Setup (Soumitra, by hand, once)

1. Diya pushes main to GitHub (owner step). Then clone it and build the
   environment: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
2. `.venv/bin/modal token new --profile soumitra` and log in to your own
   Modal workspace in the browser. Never paste the token into a chat.
3. `git switch -c soumitra-e4`. All commits go on this branch.
4. Check the credit: `MODAL_PROFILE=soumitra .venv/bin/modal billing summary`.
   Set a workspace spend limit of $0 in the Modal dashboard so nothing can
   run past the free credit (only a workspace owner can).

## J1 and J2 (Soumitra's Claude)

Launch both detached; a remote driver trains, then deploys each finished
policy, so a sleeping laptop loses nothing. Two jobs share each A10G.

```
M=.venv/bin/modal; A=scripts/modal_llm.py
mkdir -p results/logs
MODAL_PROFILE=soumitra nohup $M run --detach $A::transfer --source ILI --target ETTh1 --arms league:10,none:10,uniform:5 > results/logs/soumitra_e4_ili_to_etth1.log 2>&1 &
MODAL_PROFILE=soumitra nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms league:10,none:10 > results/logs/soumitra_e4_etth1_to_ili.log 2>&1 &
```

Poll the logs (never hold a shell open more than 9 minutes). Expect about
15 minutes per training run and 2 minutes per deployment, 3 to 4 hours of
wall time. If the local client loses the network, the job keeps running on
Modal; check `modal app list` and the volume instead of relaunching. A
relaunch of the same command keeps adapters that already finished.

When an app has completed, bring the outputs back:

```
for t in ETTh1_T336 ILI_T24; do
  S=$(mktemp -d); MODAL_PROFILE=soumitra $M volume get leaguetst-runs /E4/$t $S
  mkdir -p results/search/E4/$t results/runs/E4/soumitra_$t
  cp $S/$t/grpo_*_runs.jsonl results/search/E4/$t/
  cp $S/$t/training.jsonl $S/$t/errors.jsonl results/runs/E4/soumitra_$t/ 2>/dev/null
done
```

Check before committing:

* training.jsonl has 25 lines for the ILI-source job (target ETTh1_T336) and
  20 for the ETTh1-source job (target ILI_T24), or each missing run is in
  errors.jsonl with its reason.
* Each grpo_<arm>_from_<source>_runs.jsonl has one line per finished policy,
  and every line carries a seed, a trace and charges.
* errors.jsonl: an out-of-memory error means something is wrong (two jobs
  per GPU should fit); stop and report it. An on-policy stop ("log-
  probabilities differ") is a valid outcome: report it, do not loosen the
  check (amendment A22).

Then commit on soumitra-e4 with a message that gives the counts, and tell
Soumitra to push the branch. Do not run scripts/replay.py, gates.py,
picks.py or report.py, and do not touch results/ledger.jsonl or docs/claims/:
Diya's side summarizes, verifies and records, so the ledger has one writer.

## J3 (only on request)

Amendment A17 runs the ETTh1-source uniform arm only if, on ILI, the league
policies beat the no-league policies after Holm. Diya's side computes that
from J2 and tells Soumitra. If it holds:

```
MODAL_PROFILE=soumitra nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms uniform:5 > results/logs/soumitra_e4_uniform.log 2>&1 &
```

Bring it back the same way.

## Diya's side, meanwhile

1. Merge soumitra-e4, then `scripts/replay.py --item E4 --summarize-only` per
   target, the gates, and the verifier's re-score.
2. E5 (scripts/e5_referee.py) again with the GRPO proposals, under the
   margin fixed before they arrive.
3. From the critic (2026-11-24): the smallest effect the pick comparison can
   detect, regret in raw MSE beside the SD units, a robust (MAD) ETTh1 seed
   SD as a check, and cold random search and TPE on each target beside the
   transfer arms.
4. `scripts/picks.py collect` for the E4 picks, batch 2 on diyapr; `picks.py
   analyze` and K4 by 28 Nov.
5. Extended test prereg committed by 28 Nov (arena verdict); E6 verified and
   the primary prereg committed by 29 Nov; Diya runs scripts/eval_test.py
   once on 30 Nov.
6. Report regenerated from the ledger by scripts/report.py, due 3 Dec.

## Not before the report

* ETTh1 seeds 2022 and 2023 for all 2,368 configs (about $25), so the ETTh1
  replay can score held-out seeds as ILI does. It trains forecasters, so it
  needs the data volume in Diya's workspaces; next month's credit.
* The wider design space of amendment A16 (docs/wide_space.md, on the
  developer branch), costed there.
* The LeagueTST-winner E6 arm stays skipped (arena verdict): a late change
  to the test prereg is riskier than the arm is informative.

## First message for Soumitra's Claude

> Read CLAUDE.md, docs/STATE.md and docs/HANDOFF_soumitra.md. Run J1 and J2
> exactly as written there with MODAL_PROFILE=soumitra, poll until both
> complete, bring the outputs back, run the checks, and commit on branch
> soumitra-e4. Do not push. Do not run anything else on Modal.
