# AB4 addendum 2 (docs/prereg_ablations.md): the RTX analysis code, verdict

scripts/ablations/rtx_tests.py computes the registered AB-P (Holm 3) and AB-G
(Holm 2) families for the RTX arms of AB2 and AB4 with byte-identical copies
of Anirudh's scripts/ablations/common.py and llm_tests.py at 5abbd20. The code
was first committed in e6926d8 (26 Nov 18:44 ET, by Soumitra, with the RTX
run files and copies under results/runs/llm_local/; that commit carries no
verifier verdict). No AB-P or AB-G outcome had been computed.

## Review history

1. Verifier (fresh, first version): PASS. Copies byte-identical, families,
   levels, methods, endpoint and Holm sizes as registered, policy pairing
   correct, exploratory set limited to what AB4 allows.
2. Critic (NeurIPS-style review): may be committed and run once changes 1 to
   3 are made (verdict below). Applied: addendum 2; caveats and an
   approximate detectable rho on every trend; exact seed sets 0..n-1; a
   dirty-tree refusal; the scope wording.
3. Verifier (fresh, revised version): code correct; FAIL on two addendum
   wordings (the e6926d8 history; the re-score shares the AUC library).
   Applied.
4. Final check (fresh verifier): VERDICT: PASS (both fixes and the on-policy
   sentence present and accurate, rtx_tests.py diff a comment only,
   append-only, RTX_tests.json absent, full suite and 10 new tests pass,
   e4_rerun.sha256 OK).

Critic's item 5 (Diya's sign-off on addendum 2 and the framing) is open: the
analysis is not run until the owner decides.

## Critic's verdict (verbatim)

rtx_tests.py may be committed and run once these changes are made: items 1 to 3 below, before any outcome is computed. I found no implementation error. The pinned copies match 5abbd20 byte for byte. The arms, levels, method names, two-sided permutation tests, Holm sizes 3 and 2 with missing tests counted as p = 1, ILI held-out AUC endpoint and ordinal level ranks all match AB-P, AB-G, AB2 and AB4. All 15 run files hold exactly the registered seeds (I counted seeds, not outcomes). Deployment seed s maps to training seed 100+s at every level, so policy_level pairs the policies correctly. Two problems remain. AB-P never registered a statistic for the scale trend, so choosing one is the only open forking path. The dependence caveat is attached to the steps trend only.

REQUIRED
1. Commit an "AB4 addendum 2 (before any AB-P/AB-G outcome is computed)" stating: "The scale trend is spearman_trend (5abbd20) on level ranks 0.5B < 1.5B < 3B, n = 90, two-sided, 100,000 relabellings, seed 0. The KL levels share training seeds 100-109 and deployment seeds 0-9, and the scale levels share search seeds 0-29, so their permutation p values carry the same dependence caveat as the steps trend. No AUC from AB2 or AB4 was computed before this commit."
2. Code changes:
   - Attach the dependence caveat to the KL trend and the scale trend too.
   - Replace `len(vals) == n` with `sorted(seeds) == list(range(n))`.
   - Report the detectable |rho| for each trend as an approximation: `math.tanh((norm.ppf(1-0.05/fam/2)+norm.ppf(0.8))/math.sqrt(n-3))`.
   - Refuse to run if `git status --porcelain leaguetst scripts` is non-empty, so the recorded git hash covers this script.
   - Get a fresh verifier PASS before the commit (guardrail until 28 Nov).
3. Numbers reach docs only through results/ledger.jsonl and scripts/report.py (rule 5), not by hand from RTX_tests.json.
4. The write-up must say: "These ablations compare variants of the same architect with each other. No external baseline appears here, so none of this answers RQ1. RQ1's registered answer is E4: the league did not beat no-league on ILI and was worse than warm-started TPE. Lower AUC is better, so a negative rho favours the higher level. Level 0 of the steps trend is the prompted model on 30 seeds, so the trend mixes 'trained at all' with 'trained longer'. The addendum-3 analyses are exploratory. Under rule 6 these are secondary tests, not claims: they have permutation p values and Welch intervals, no paired block-bootstrap interval, and no independent verifier re-score yet. The author knew the direction of the Mac results (AB4)."
5. Diya must sign off on addendum 2 and on the framing in item 4. The owner's "beat baselines" goal must not shape what is reported.

Optional
- A verifier re-score with separate code: AUC recomputed from the traces through leaguetst.search.replay, scipy.stats.spearmanr, and its own permutation loop.
- A seed-paired block-bootstrap interval, labelled descriptive.
- Add the training.jsonl files under results/runs/ABL_G_*_local to the provenance inputs.
- Report the GPU per arm and the J2 hardware check as descriptive only (they differ by backend and code digest, not only by card).
- Make gpus() tolerate rows that have no env field.
