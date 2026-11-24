# Handoff: E4 GPU work on Soumitra's side (Modal or a local RTX)

For Soumitra and the Claude Code session on Soumitra's account. Written
2026-11-24. Read CLAUDE.md first: every hard rule there applies to you too.
The registered design is docs/prereg_E4.md, amendments A25 to A27.

## Why this exists

Diya's two Modal workspaces are nearly out of free credit (diyapr has about
$5 left, $3.5 of it held for E6 reruns; diya-027 is shared with another
project). The E4 GRPO training on 23 Nov lost 34 of 45 runs to CUDA out of
memory (three jobs on one A10G). E4 is the pre-registered primary endpoint,
so all 45 policies are retrained on Soumitra's side. The 11 that finished on
diyapr were never deployed and are not reported. The owner made this
decision on 2026-11-24 (docs/ISSUES.md). CLAUDE.md rule 10 still names only
Diya's two workspaces; until Diya adds the line below to rule 10 by hand,
this file is the authority for Soumitra's backend:

    Owner's decision, 2026-11-24: the E4 GRPO reruns, which need no data
    volume, run on Soumitra's side, on Soumitra's Modal workspace
    (MODAL_PROFILE=soumitra) or one local CUDA GPU (docs/HANDOFF_soumitra.md).

## Pick one backend before launch

Every policy of the rerun comes from one backend (A25). Do not split the
45 runs between Modal and a local card.

| | Modal (recommended) | Local RTX |
|---|---|---|
| Hardware | A10G, as every LLM run so far | 24 GB card (3090, 4090, A5000) at 2 jobs; 12 to 16 GB at 1 job; under 12 GB: use Modal |
| Wall time, both jobs | about 2 hours (up to 10 GPUs at once) | about 6 to 8 hours at 2 jobs; double at 1 |
| Cost | about $11 of the $30 monthly free credit | free |
| Runner | scripts/modal_llm.py | scripts/llm_local.py |

Same code, same plan, same pinned packages and Qwen revision either way;
only the executor differs (leaguetst/architect/runner.py). The GRPO jobs
read the table from the committed results/ledger.jsonl and need no data
files and no data volume. Never upload anything to Modal (rule 10).

## Setup (Soumitra, by hand, once)

1. Diya pushes main to GitHub (owner step). Clone it and check the code:
   `shasum -a 256 -c docs/e4_rerun.sha256` must print OK for every file.
   If any line fails, stop and tell Diya.
2. `git switch -c soumitra-e4`. All commits go on this branch.
3. Modal backend: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`,
   then `.venv/bin/modal token new --profile soumitra` (log in to your own
   workspace in the browser; never paste the token into a chat). Check the
   credit with `MODAL_PROFILE=soumitra .venv/bin/modal billing summary` and set a
   $0 spend limit in the dashboard so nothing runs past the free credit.
   Confirm the image: `MODAL_PROFILE=soumitra .venv/bin/modal run scripts/modal_llm.py::versions`
   must print transformers 5.24.0, peft 0.22.0, accelerate 1.16.0, torch 2.15.0.
4. Local backend: Linux or WSL2 with an NVIDIA driver, then
   `python3.12 -m venv .venv && .venv/bin/pip install -r requirements-llm.txt`
   (if `python -c "import torch; print(torch.cuda.is_available())"` prints
   False, install torch 2.15.0 from pytorch.org for your CUDA version). The
   first run downloads Qwen2.5-1.5B-Instruct (about 3 GB) at the pinned
   revision.

## J1 and J2 (Soumitra's Claude)

J1: policies trained on ILI (league 10, none 10, uniform 5), deployed on
ETTh1. J2: policies trained on ETTh1 (league 10, none 10), deployed on ILI.
Never hold a shell open more than 9 minutes; launch detached and poll the
logs.

Modal (both at once; a remote driver trains then deploys, so a sleeping
laptop loses nothing):

```
M=.venv/bin/modal; A=scripts/modal_llm.py; mkdir -p results/logs
MODAL_PROFILE=soumitra nohup $M run --detach $A::transfer --source ILI --target ETTh1 --arms league:10,none:10,uniform:5 --per-gpu 2 > results/logs/soumitra_e4_ili_to_etth1.log 2>&1 &
MODAL_PROFILE=soumitra nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms league:10,none:10 --per-gpu 2 > results/logs/soumitra_e4_etth1_to_ili.log 2>&1 &
```

If the local client loses the network, the job keeps running; check
`modal app list` and the volume before relaunching. A second launch on the
same output is refused while a driver.lock file exists; if no app is
running, remove it (`modal volume rm leaguetst-runs /E4/<target>/driver.lock`,
or delete the file locally) and relaunch. A relaunch of the same
command keeps finished adapters and deployed seeds.

Local (one after the other on the one GPU; `--per-gpu 1` on a card under 16 GB).
First J0, prompted Qwen on the same card, which K4 and family 3 then use as
the baseline (A26; about 20 minutes):

```
mkdir -p results/logs
for d in ILI ETTh1; do .venv/bin/python scripts/llm_local.py prompted --dataset $d --seeds 10 --per-gpu 2; done > results/logs/soumitra_e3_prompted_local.log 2>&1
nohup .venv/bin/python scripts/llm_local.py transfer --source ILI --target ETTh1 --arms league:10,none:10,uniform:5 --per-gpu 2 > results/logs/soumitra_e4_ili_to_etth1.log 2>&1 && \
nohup .venv/bin/python scripts/llm_local.py transfer --source ETTh1 --target ILI --arms league:10,none:10 --per-gpu 2 > results/logs/soumitra_e4_etth1_to_ili.log 2>&1 &
```

Keep the machine awake. If it stops, run the same command again: finished
adapters and deployed seeds are kept.

## Bring the outputs back

Modal:

```
for t in ETTh1_T336 ILI_T24; do
  S=$(mktemp -d); MODAL_PROFILE=soumitra $M volume get leaguetst-runs /E4/$t $S
  mkdir -p results/search/E4/$t results/runs/E4/soumitra_$t
  cp $S/$t/grpo_*_runs.jsonl results/search/E4/$t/
  cp $S/$t/training.jsonl $S/$t/errors.jsonl results/runs/E4/soumitra_$t/ 2>/dev/null
done
```

Local: the same copies from `results/runs/llm_local/E4/$t/` instead of `$S/$t/`,
plus J0: `cp results/runs/llm_local/E3/$t/qwen_prompted_local_runs.jsonl results/search/E3/$t/`.
Adapters stay where they are (git ignores them).

Cut-off (A26): if J1 and J2 are not complete (every job finished or failed)
by 26 Nov 18:00 ET, commit what exists, say so in the message, and stop.
E4 is then reported as incomplete; nothing is tested on a partial set.

Check before committing:

* training.jsonl has 25 lines for J1 (folder ETTh1_T336) and 20 for J2
  (folder ILI_T24), or each missing run is in errors.jsonl with its name.
* Each grpo_<arm>_from_<source>_runs.jsonl has one line per finished
  policy, and every line has a seed, a trace, charges and an env block whose
  backend, versions and model_revision match A25.
* errors.jsonl: an out-of-memory error means the card is too small for
  the chosen --per-gpu; stop and report it. An on-policy stop ("log-
  probabilities differ") is a valid outcome: report it; it is never
  retrained and the check is never loosened (A22).

Then commit on soumitra-e4 with a message that gives the counts and the
backend, and tell Soumitra to push the branch. Do not run scripts/replay.py,
gates.py, picks.py or report.py, and do not touch results/ledger.jsonl or
docs/claims/: Diya's side summarizes, verifies and records, so the ledger
has one writer.

## J3 (only on request)

A17 runs the ETTh1-source uniform arm only if, on ILI, the league policies
beat the no-league policies after Holm. Diya's side computes that from J2
and tells Soumitra. If it holds, on the same backend as J1 and J2:

```
MODAL_PROFILE=soumitra nohup $M run --detach $A::transfer --source ETTh1 --target ILI --arms uniform:5 --per-gpu 2 > results/logs/soumitra_e4_uniform.log 2>&1 &
# or: nohup .venv/bin/python scripts/llm_local.py transfer --source ETTh1 --target ILI --arms uniform:5 --per-gpu 2 > results/logs/soumitra_e4_uniform.log 2>&1 &
```

## Diya's side, meanwhile

1. Merge soumitra-e4; `scripts/replay.py --item E4 --summarize-only` per
   target; gates; the verifier's re-score.
2. `scripts/picks.py collect --budgets 25 50` for the E4 picks, batch 2 on
   diyapr; then `picks.py analyze`, the ETTh1-target family tests of A25,
   and K4 by 28 Nov.
3. E5 (scripts/e5_referee.py) with the GRPO proposals, under A24.
4. From the critic: the smallest effect the pick comparison can detect,
   regret in raw MSE beside the SD units, a robust ETTh1 seed SD as a check,
   cold random search and TPE beside the transfer arms.
5. Extended test prereg committed by 28 Nov; E6 verified and the primary
   prereg committed by 29 Nov; Diya runs scripts/eval_test.py once on 30 Nov.
6. Report regenerated from the ledger by scripts/report.py, due 3 Dec.

## Not before the report

* ETTh1 seeds 2022 and 2023 for all 2,368 configs (about $25), so the ETTh1
  replay can score held-out seeds as ILI does. It trains forecasters and
  needs the data files, so it stays in Diya's workspaces; next month.
* The wider design space of A16 (docs/wide_space.md).
* The LeagueTST-winner E6 arm stays skipped (arena verdict).

## First message for Soumitra's Claude

> Read CLAUDE.md, docs/STATE.md, docs/prereg_E4.md (A25 to A27) and
> docs/HANDOFF_soumitra.md. Check the code digests. Using the backend
> Soumitra chose (Modal profile soumitra, or the local GPU: J0 first), run J1 and J2
> exactly as written, poll until both finish, bring the outputs back, run
> the checks, and commit on branch soumitra-e4. Do not push. Run nothing
> else.
