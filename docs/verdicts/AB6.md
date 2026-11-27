# Critic's verdict: AB6 (reconciliation of the two AB4 registrations; the AB-P and AB-G result of record)

Tree: main at 3950c22b. Read-only; no result file edited; no test row opened.

## What is under review

Amendment AB6 (docs/prereg_ablations.md, commit e1406066) declares Soumitra's AB4 (15db1631) and Diya's AB4 (2ab29963) one registration for the four RTX arms: run-time rules from the first, analysis from the second. results/ablations/LLM-registered.json (ea0343c8) is the result of record, written once by scripts/ablations/llm_registered.py (sha256 0c101110, final_mode false, 15 of 15 arms complete, provenance git_hash e1406066, computed 21:56 UTC, which is the commit minute). results/ablations/LLM-registered_comparison.json (3950c22b): 132 fields, 0 disagreements against LLM-rtx.json (59 rows), verify_rtx/ab4/recompute_ab4.json (30) and rtx/RTX_tests.json (45). Commit 2aa3f889 makes rtx_tests.py read llm_tests.py from git at 5abbd20 and check 01b70835 on those bytes; G1 passes again (298 passed).

## Numbers checked (own script; scripts/ablations/common imported read-only, E1 ILI table)

* anon vs ref, ILI held-out AUC from the two RTX run files: 1.92763 against 1.56493, diff 0.36270, n 30 and 30; Welch p 0.0077 (record: permutation p 0.0073, Holm 0.0145). Matches LLM-registered.json to all printed digits.
* Step trend over levels 0 (30 untrained searches), 30, 60, 120 (10 policies each): level means 1.5649, 3.0322, 3.2600, 3.1662; Spearman rho 0.76223 on n 60. Matches.
* Per-policy rho of AUC with 30, 60, 120 steps: [0.5, 1.0, -1.0, 0.5, -0.5, 0.5, 1.0, -1.0, -1.0, 0.5], mean 0.05. Matches analysis (ii) (sign-flip p 1.0).

## Attack

1. Forking path? The two texts agree on arms, launcher digest, methods, seeds, references, statistics, Holm sizes and record paths. AB6 picks nothing that changes a test, level, reference or endpoint; llm_registered.py's bytes predate every value and ran unchanged. The only analysis-affecting difference is the status of policy-level analyses (i) to (iii) and the reading rule (registered secondary in 2ab29963, exploratory in 15db1631). 2ab29963 was committed at 17:21 ET on 26 Nov, after five of six AB4 jobs had DONE lines, by an author who had read no value. That is a post-run registration of a reading rule; ablations.md line 182 and the 27 Nov audit say so, AB6 does not say it in those words, and Diya's AB4 "What existed" paragraph (no run exists in git) stays uncorrected above it. The reading rule's outcome ("not a dose-response; trained policies differ from the untrained model, worse by +1.59 SD") goes against the method, which limits the incentive but not the disclosure duty.
2. Nested step levels: the registered trend p (1e-5) is anti-conservative; reported "as registered, with this caveat", and the headline reading comes from (ii), p 1.0. Honest. ablations.md operationalises "shows a difference" as p <= 0.05, which it labels as Anirudh's reading, not the amendment's; AB6 did not fix it.
3. Rule 6: Welch intervals and unpaired permutation p, no paired block bootstrap; ablations.md says "secondary tests, not claims under rule 6" throughout. Correct.
4. Independence of the re-scores: llm_rtx.py and rtx_tests.py were written after values were seen; verify_rtx/ was committed blind before the record run. Agreement to 1e-9 on exact fields and within 0.005 on MC p-values is strong; two comparison rows carry no `kind` key (cosmetic).
5. Scope wording: "larger model worse" is scoped to three checkpoints and this prompt; anon "does not isolate prior knowledge of the names"; repair "measures that combination". No overclaim found.
6. Stale statements now false: docs/ablations.md lines 12 and 140 say the computation of record "is not yet computed" and the arms are "not of record". ISSUES row of 27 Nov on the duplicate AB4 is "open, arena 27 Nov". STATE has no 27 Nov entry for AB6, the record run or the comparison.

## Findings

Blocking
* B1. Regenerate docs/ablations.md (scripts/ablations/report.py) so the RTX section names LLM-registered.json as the record, quotes the three-way comparison (132 fields, 0 disagreements), and drops "not yet computed".
* B2. Add, by dated note (AB6 addendum or ISSUES), that 2ab29963 postdates five of six AB4 DONE lines and that its "What existed" paragraph was wrong in fact; state that analyses (i) to (iii) and the reading rule are registered after the runs and before the author read any value.

Non-blocking
* N1. Fix the reading rule's "shows a difference" threshold by amendment (p <= 0.05 is what the report applied).
* N2. Close the 27 Nov ISSUES row; dated STATE entry; rtx_tests.py edit awaits Soumitra's confirmation as logged.
* N3. Two comparison rows lack `kind`.

Verdict: REVISE. Apply B1 and B2 (documentation only; no value, test or record changes); the result of record and its agreement with three re-scores then stand as a complete, honestly labelled secondary analysis.
