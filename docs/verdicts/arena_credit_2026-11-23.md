# Arena: spending the remaining Modal credit (2026-11-23)

The owner asked that external decisions go to an arena of web-enabled proposers
and an independent judge. Three proposers (scientific value, deadline
pragmatist, cost and risk) ranked six items: 1 transfer deployment, 2 pick
reseeds, 3 uniform-rival arm, 4 LeagueTST-winner E6 arm, 5 extended test
prereg, 6 reserve.

| proposer | choice |
|---|---|
| cost and risk | 1, 2, 6, 3, 5, beyond-credit: no |
| scientific value | 2, 5, 3, 6, beyond-credit: no |
| deadline pragmatist | 2, 6, 3, beyond-credit: no |

All three: skip item 4, no new GPU work on diya-027 (its credit is shared with
another project), nothing beyond free credit.

## Judge's verdict (verbatim)

Judge's verdict: run items 1, 2, 6, 3 and 5 in that order. Item 3 stays conditional, the item 5 prereg is committed by 28 Nov, the diyapr reserve is $3.5, and nothing is spent beyond free credit.

1. **Item 3 stays conditional.** The rule that skips it unless league beats no-league was adopted in A17, before any LLM result existed, and credit is now short. "Competitive" needs league to beat both no-league and uniform, so if league fails against no-league, skipping uniform loses no claim. The CPU replays make that likely: league 1.41 vs no-league 1.32 on ETTh1, and 1.59 vs 1.69 on ILI. The report must say that the ETTh1-source uniform test was not run, and why.

2. **Item 5: commit the extended prereg by 28 Nov.** Commit it as a separate file after the pick reseeds, so their checkpoints can be listed. This week, download and hash-check the checkpoints and dry-run the scorer on validation. Run it only after the primary evaluation is scored and committed. It costs no GPU and moves all setup off test day.

3. **Order:** item 1 is already running. Launch the pick reseeds for the CPU replay methods (item 2) now beside it on diyapr: ILI first, as a $0.3 smoke test, then ETTh1. Add the GRPO picks once the item 1 deployments land. `picks.py collect` adds batches and skips configs already listed, so nothing is charged twice. CLAUDE.md says to start independent work while a Modal job runs, and K4 (due 28 Nov) needs both batches.

4. **Reserve: $3.5 on diyapr.** About $3.7 is left after items 1 and 2. E6 is never cut, and rerunning one E6 arm costs roughly $1.5 to $2. PLAN also stops the build above $28 in a workspace before E6 has run, and items 1 and 2 bring diyapr to about $25.8. If item 3 is triggered, pay for it from the reserve only after every E6 arm is in the ledger.

Notes for the owner:
- No new GPU work on diya-027. The $28 rule counts the whole workspace bill, so the other project's spending there could trip it.
- Modal's budgets page says only a workspace Owner or Manager can set a spend limit. Setting diyapr's limit to $0 makes "no beyond-credit spending" automatic.
- I could not confirm the NAS-Bench-201 claim about validation-test agreement among top configs, because the PDF would not open here. Nothing in this verdict depends on it.

VERDICT: 1, 2, 6, 3, 5; item3: conditional; item5 prereg: commit-by-28Nov; reserve: $3.5; beyond-credit: no

## Acted on
- 2026-11-23 19:3x: ILI pick reseeds launched on diyapr (197 configs, seeds 4001 and 4002). diyapr billing at launch: $17.68 compute used (lags running jobs).
