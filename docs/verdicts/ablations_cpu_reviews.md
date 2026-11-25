# Ablations (CPU part): reviews, verbatim, and what changed

Reviews by AI subagents, not by people (docs/HANDOFF_anirudh.md requires a verifier and a critic before
a result is committed). AB-C, AB-Q, AB-N, AB-X, AB-R, AB-L calibration, ST, RQ3.

## Verifier 1 (results of 31f9313)

Verdict: every stored number checked reproduces. The problems are in interpretation and in two unfulfilled
text promises. (Charges and traces: 2,000 runs plus 620 E3 runs, 0 mismatches. AB-C, AB-Q, AB-N, AB-R (92
of 126 variants), RQ3, ST-robust, AB-L-null within Monte Carlo error. Independent code; leaguetst used only
to re-run traces.) Its ranked problems: (1) the AB-Q summary "seed noise hides differences" is
unsupported: a scratch hold-out of two-seed lookups scored on the third seed gave TPE -0.39 SD in-sample
and -0.05 held out; (2) the referee target contains seed 2021, which also trains it; (3) AB-N mixes two
biases; (4) the AB-L null rests on ten reused episodes, sign flip on the adjusted ETTh1 slopes p = 0.014;
(5) marginal shares on an unbalanced space; (6) RQ3 promised seed noise and did not report it; (7) AUC in
raw MSE missing; (8) ETTh1 is not purely in-sample; (9) run records carry no git hash, input digests
incomplete; (10) the 30 to 100 epoch comparison is confounded by patience. Not checked: ST-power, AB-X,
LLM code, AB-L itself, 34 forest variants.

## Critic (ablations.md at a7ca3dd)

Verdict: REVISE. Findings 1 and 2 change conclusions: the "noise-free lookups separate only in-sample"
claim contradicts E3's held-out gap of -0.38 SD; DEHB's repair rule is not TPE's; "0 of 3 differ" accepts
the null and the SMAC choices need listing; the AB-L table used an unfinished Mac source; "GRPO learned"
exceeds the AB-L verdict's wording; one of the three E4 "nulls" is a rejection (a loss); ETTh1's unit is
inflated; the order of runs and re-runs was not disclosed; exploratory labels and registered intervals
were missing; checklist rows 4, 7, 9, 12 and 16 needed correction; and it listed five open questions
(see docs/ablations.md, "Runs, re-runs and open questions").

## Verifier 2 (changed outputs at d632c41, report at a7ca3dd)

PASS on every item it could run: the AB-Q hold-out (900 runs re-run, identical), AB-N split, AB-R (57 of
63 variants per table), RQ3 arithmetic, AB-L p-values (14/1024 and 546/1024), records (3,080 lines), more
than 15 report numbers. FAIL on wording: "only in-sample" overstated, the referee's interval excludes zero
only across search seeds (over the three hold-outs it does not), an unfinished Mac file in the AB-L
null, missing raw-MSE columns, the seed-noise share included each seed's common shift (24.5% against
23.4%), and incomplete input digests. Not checked: AB-X, AB-C, AB-N and ST searches, the AB-L-null
simulation, E5 and E6 rows, addendum-2 outputs.

## What changed after the reviews

All numbers named above are regenerated from scripts at d8364cd. Addenda 1 and 2 of AB3 were committed
before their outputs existed (d632c41, 7a67954). The registered parts of AB-X, AB-C, AB-Q and AB-N were not
changed. Corrections to the write-up: the AB-Q section and "At a glance"; AB-X caveats, DEHB repair rates
and an exploratory DEHB variant; exploratory labels; robust-SD detectable effects; the E4 loss labelled;
checklist rows. The AB-X registered traces were identical in the two runs; a third run added only
repair counts and git hashes.
