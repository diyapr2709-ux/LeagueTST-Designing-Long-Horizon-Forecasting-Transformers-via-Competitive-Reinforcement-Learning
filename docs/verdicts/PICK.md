# Critic's verdict: PICK (pick reseeds on fresh seeds 4001 and 4002)

Tree: main at 3950c22b. Read-only; no test row opened; nothing written under the repository.

## What is claimed

results/picks/ILI_T24_analysis.json and ETTh1_T336_analysis.json (regenerated 27 Nov, 9899bf27, by scripts/picks.py analyze): per method and budget (25, 50, 100), the mean over search seeds of the incumbent's fresh-seed validation MSE (mean of seeds 4001 and 4002), bootstrap interval over search seeds, seed-level permutation test against random search, Holm over the file. STATE (24 Nov): "on both tables no method beats random search after Holm, prompted Qwen included; TPE's in-table win on ILI does not replicate. The cross-dataset portfolio is worse than random on both." The six GRPO cells per table are left out (A27: picks not all reseeded; batch 2 ids collected, not trained).

## Numbers checked (own script, picks.py helpers and the ledger read-only)

* 197 ILI and 280 ETTh1 configs have both fresh seeds; ledger PICK rows 954 = (197 + 280) x 2. Batch 1 is complete.
* ILI tpe @25: mean 0.217993, n 30; random @25 0.221830; p 0.05545. tpe @50: 0.220360 vs 0.222988, p 0.26852. Both equal the file. The @25 gap is 0.39 pooled SD in TPE's favour.
* Holm family recomputed: 36 tests per table (12 methods x 3 budgets); the only rejections are portfolio_from_ETTh1 at 25, 50 and 100 (p 1e-5, portfolio worse). Equal to `vs_random`. Left-out cells equal `left_out` (league 7/6/5, none 6/6/6).

## Gates

`.venv/bin/python scripts/gates.py --item PICK`: G1 PASS, G2 PASS (123 rows re-scored, worst gap 3.3e-8), G3 PASS, G4 PASS ("item makes no claims"), G5 PASS (no search), G6 FAIL (no docs/verdicts/PICK.md). G2 passes only because the checkpoints were downloaded today into untracked results/runs/PICK/*/ckpt/; a clone cannot reproduce it.

## Attack

1. Selection: incumbents are fixed by the replay trace at each budget; fresh seeds never enter the search. Validation only, exact loader (G2 re-score agrees to 3e-8). Sound.
2. Fresh seed vs in-sample: this is the only fresh-seed evidence in the project and the ILI K7 fallback requires it. The in-table TPE win (E3, p 0.0011) becomes p 0.055 at 25 and 0.27 at 50. "Does not replicate" is fair at 50; at 25 it is a non-detection with the point estimate still in TPE's favour, so the honest wording is "not detected on fresh seeds", not "refuted".
3. Seeds and interval: 30 search seeds per method (10 for Qwen) are the unit; each value is a 2-seed mean, which adds equal noise to every arm and lowers power. No positive claim is made, so rule 6's five-seed and paired block-bootstrap requirements are not engaged; they would be if "portfolio worse than random" were promoted to a claim (it is a Holm rejection in the negative direction, stated descriptively in STATE).
4. Verifier re-score of the comparisons: none is recorded; G4 does not cover PICK (CLAIM_ITEMS is E3 and E4), so the 36 Holm tests are checked only by picks.py itself and, today, by this review.
5. Budget accounting: not engaged (no search); fresh-seed training cost is reported in the ledger table (REPORT_DRAFT lines 18 to 21).
6. K4 (PLAN, due 28 Nov): "after the pick reseeds, league GRPO beats prompted Qwen and RE on median regret". The GRPO cells are left out until batch 2, so K4 cannot be read from the picks yet; on the committed ILI AUC it fails (league median 3.28 against Qwen 1.65 and RE 1.47). No dated result or fallback is logged in PLAN, STATE or ISSUES.
7. Overclaim scan: STATE's "RQ1 for the CPU methods is a null result" and the audit's wording are supported. No sentence turns the portfolio result into a positive for any method.

## Findings

Blocking: none against the results as written.

Non-blocking
* N1. Reword "does not replicate" to "is not detected on fresh seeds (p 0.055 at 25, 0.27 at 50)" wherever it appears.
* N2. Record a verifier re-score of the two analysis files (the 36 tests each) in docs/verdicts/, or add PICK to a claims file so G4 recomputes them.
* N3. Commit or document the PICK checkpoints so G2 is reproducible from a clone (E3b and E6 checkpoints are committed; PICK's are untracked).
* N4. Record K4's status (not readable until batch 2; fails on in-table AUC) with a date in PLAN, STATE and ISSUES before 28 Nov; apply the "If it fails" action if batch 2 does not change it.
* N5. This verdict covers batch 1 cells only; the six GRPO cells per table are pending and must be re-analysed by picks.py analyze after batch 2 (A27), not by hand.

Verdict: ACCEPT
