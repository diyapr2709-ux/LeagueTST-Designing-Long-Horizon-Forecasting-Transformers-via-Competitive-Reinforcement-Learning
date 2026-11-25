# E4 independent analysis (docs/LEG2_soumitra.md L1): verdict

Script: scripts/independent/e4_analysis.py, written from docs/prereg_E4.md and
docs/PLAN.md alone and committed (c21d0f7, fix 68593a0) before it ran and
before scripts/replay.py, scripts/picks.py or any E4 summary file was opened.
Output: results/independent/E4_soumitra.json. Every number below is read from
that file.

## J3 decision (A30)

J3 is not run. On ILI AUC the league policies' mean (3.143 pooled seed SD) is
below the none policies' (3.233), but the registered two-sided permutation
test (exact, 184,756 splits) gives p = 0.8025, above 0.05/6 = 0.0083. Family 2
keeps size 6 with the ILI league-against-uniform test counted as not rejected.

## Verifier (independent re-score, own code, did not import the script)

PASS. Every deterministic value (pooled SD 0.009787, best config
p24s12l4h4d16f1r3n1i, every arm's mean AUC and regret at 25 and 50, the exact
p, the J3 flag) reproduces within 1e-9; Monte Carlo p-values agree within
Monte Carlo error. Incumbent paths match every record's trace. An
epoch-continuous AUC instead of the b = 1..100 grid gives league vs none
p = 0.81: the J3 decision does not depend on that choice. Noted: the Holm
rejection in family 1 is league worse than warm-started TPE; pending ETTh1
picks tests are entered at p = 1 provisionally and both families are
recomputed when pick batch 2 lands.

## Changes made after the critic, before this commit

Direction of every test (lower_regret), an explicit A26 statement for the ILI
tests (no failed or retried seed, worst case equals the primary test), the
A26 completeness statement, GPU name per arm (AB1), K3 from the training
logs, and a refusal on an empty arm. No number changed.

## Critic's verdict (verbatim)

The J3 decision stands: J3 is not run. On ILI AUC the league mean (3.143 SD) is below none (3.233 SD), but the exact two-sided permutation p = 0.80 is far above 0.05/6. No defensible statistic, sidedness or worst-case reading changes this. Nothing in E4_soumitra.json is a claim under CLAUDE.md rule 6: there is no paired block-bootstrap interval and no verifier re-score or replay.py diff yet. Until those exist the analysis may state only provisional descriptive findings on the ILI target:
- no detectable league effect against no-league;
- league worse than warm-started TPE (two-sided rejection, wrong direction for a win), so no ETTh1-to-ILI transfer win and no "competitive" claim are possible;
- family 3 p-values are unadjusted only.

The ETTh1-target tests, family 3's Holm composition, K3 and per-arm GPU-type reporting are still owed.

## Still owed

* L1 step 3: diff against `scripts/replay.py --item E4 --summarize-only` (27 Nov).
* ETTh1-target tests (A25) once pick batch 2 is on origin/main (by 28 Nov 12:00 ET).
* Family 3's Holm composition with the E3 CPU methods' comparisons.
* The prompted-Qwen records carry no env block or git hash (CLAUDE.md rule 7);
  all ten report "NVIDIA A10".

## L1 step 3: diff against scripts/replay.py (25 Nov)

`scripts/replay.py --item E4 --table E1 --dataset ILI --horizon 24
--summarize-only` was run in a scratch git worktree at c2dda8d (never
committed) and compared by scripts/independent/diff_replay.py (a730d1f);
output results/independent/E4_replay_diff.json, from which every value here
is read.

* 35 statistics compared (pooled SD, best config and mean, seed counts, AUC
  and regret means and medians at 25 and 50 for the four ILI-target arms,
  per-seed AUC in each claim, p-values). All agree within 1e-9 except one.
* The one difference: league vs portfolio, p = 0.639047 (independent) vs
  0.642064 (replay). Cause: both are Monte Carlo permutation tests with
  different random streams and split counts (200,000 vs 100,000); the gap,
  0.003, is about 1.6 combined standard errors (~1.9e-3). The exact league
  vs none test (the A30 J3 test) is identical in both, p = 0.802528740609236.
* Decisions agree on all three tests: neither analysis finds a league win.
  League vs warm-started TPE is rejected by the independent two-sided test
  with warm-started TPE lower; replay records claims_win false.
* Procedure difference, no effect on any decision: replay applies Holm over
  the claims in its file (2 in family 1, 1 in family 2); the independent
  analysis uses the registered family size of 6 with pending tests counted
  as not rejected (A25, A30).

The J3 decision stands under both analyses.
