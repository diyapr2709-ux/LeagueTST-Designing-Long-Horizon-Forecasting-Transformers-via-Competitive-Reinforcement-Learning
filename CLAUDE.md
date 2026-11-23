# LeagueTST: rules for every session

Read docs/STATE.md and docs/PLAN.md first. Work on the first unchecked item in docs/PLAN.md.

## Goal
RQ1: on the tabular benchmark, does a GRPO-trained LLM architect with a league beat random search, regularized
evolution, TPE, ASHA and the same LLM without RL, at equal total epochs?
RQ2: does the referee's ranking hold on the architect's own proposals?
RQ3: which design choices win?
A method "wins" only when gate G4 passes. A negative result is a valid result; report it.

## Hard rules
1. Never read, download or reconstruct test rows. Only data/trainval/ exists for you.
   The test split is evaluated by Diya, once, from a committed prereg file.
2. Select on exact, unshuffled validation MSE only. No drop_last anywhere in evaluation.
3. Every candidate uses the audited recipe: Adam beta1 0.95, epoch-1 rate eta/25,
   official schedules. Never tune the recipe per candidate.
4. Budgets are counted in training epochs, and every method gets the same total.
   Every distinct config a method looks up in the table is charged its epochs once.
5. Every number in docs/ or the report comes from results/ledger.jsonl through
   scripts/report.py. Never type a result by hand. Never delete or edit a ledger row.
6. No claim without: >= 5 seeds, a paired block-bootstrap interval, and the verifier's
   independent re-score.
7. Fix seeds, log the git hash and config in every record, commit after every loop.
   Never run git push; the owner pushes.
8. If a guard blocks you, do not work around it. Log it in docs/ISSUES.md and move on.
9. Papers: cite only what the lit-scout opened this session, with arXiv id and URL.
10. Never upload anything to Modal. data/trainval/ is already on the leaguetst-data Volume;
    Modal jobs read only that and write results to the leaguetst-runs Volume.
    ETTh1 jobs run with MODAL_PROFILE=diya-027, ILI jobs with MODAL_PROFILE=diyapr.

## Loop (one iteration)
plan -> experimenter builds and runs -> verifier re-scores and runs gates ->
critic attacks the claim -> record in ledger and docs/STATE.md -> commit.
Copy the critic's final verdict verbatim into docs/verdicts/<item>.md.
Call lit-scout every 5 iterations, and whenever a design choice is unclear.
When two designs are both defensible, run both as a small arena on the tabular
benchmark and keep the winner; record the loser too.

## Long jobs
Launch training detached (nohup ... &), write a status file, and poll it.
Never hold a shell open for more than 9 minutes.
While a detached Modal job runs, start the next item that does not depend on it.
The Modal CLI is .venv/bin/modal; the app is scripts/modal_app.py; records come
back with scripts/collect_runs.py.

## Kill checks and stopping
When a K-check in docs/PLAN.md fails, apply its "If it fails" action, log it in
docs/STATE.md and continue. Write docs/BLOCKED.md and stop only when K0 fails,
a fallback itself fails, a gate fails twice on the same item, or the GPU budget
in docs/PLAN.md is spent.
