# Build state

Updated by every loop iteration. Newest first.

## 2026-11-23, late evening

* E1 ETTh1 T=336 seed 2021 complete (2,368 configs); top-100 reseeds and the
  E6 PatchTST/42 arm are running on diya-027.
* The two tables disagree: Spearman between the ETTh1 and ILI rankings is
  -0.21, their top-237 sets share no config, ILI's best config ranks 2,157
  on ETTh1 and ETTh1's best ranks 1,828 on ILI. A source-table portfolio
  should transfer badly; the transfer test therefore asks whether a learned
  policy carries search behaviour rather than config preferences.
* ETTh1 E2: epoch-k ranking is useless at k=1 and 2 (tau -0.06) and weak at
  k=3 (0.28); the referee reaches 0.85; k = 3.
* ETTh1 E8 (one seed): instance normalisation explains 83% of the table's
  variance; without RevIN validation MSE is 1.106, with it 1.166. Risk
  recorded before any test evaluation: DLinear (no RevIN) also had lower
  validation but higher test MSE than PatchTST/42, so validation selection
  on ETTh1 may favour configs that do worse on the test months. The
  registered validation-to-test rank check over the E3b configs measures
  this. 32% of ETTh1 runs had their best epoch at the 30-epoch cap.
* A20 fixed an off-policy sampler (Qwen's default repetition penalty) found
  by the verifier; the affected LLM runs were discarded before any E4
  endpoint and rerun.

## 2026-11-23, evening

* K5 passed. On Modal, PatchTST/42's validation MSE is within 1 seed SD of
  the laptop runs on A10G and T4 (ETTh1 T=336: 0.13 and 0.02 SD; ILI T=24:
  0.80 and 0.16 SD); A10G and T4 rank the same 32 configs with Spearman
  0.999 and 1.000. L4 had no capacity, so every job runs on A10G.
* E1 ILI T=24 complete: 2,368 configs at seeds 2021 to 2023 (7,104 runs, no
  failures, no divergence). ETTh1 T=336 seed 2021 running on diya-027.
* E2 ILI: plain epoch-k ranking is weak (Kendall tau 0.18 at k=1, 0.13 at
  k=2, 0.09 at k=3, because epoch 1 runs at rate/25); the referee reaches
  0.58 (ceiling for one full seed: 0.68). k = 1. K1 passes; the referee is
  kept (A9).
* E3b ILI: K7 fails as pre-registered (mean signed gap +0.0042, 95% interval
  [0.0013, 0.0070]; median |gap| 1.13 SD > 0.95). The 44 stratified configs
  are unbiased (+0.05 SD, median 0.89 SD); the top 20 are 1.6 SD optimistic,
  a winner's curse that an in-table leave-one-seed-out check reproduces.
  Fallback applied: RQ1 is stated for the table only; picks are scored on
  fresh seeds before any comparison.
* E3 ILI CPU replay (30 seeds): TPE has the lowest regret (AUC 0.91 SD) and
  is the only method that beats random search after Holm (budget 50). The
  factorized controller does no better than random; with league rivals it
  is worse (rivals spend budget). Results: results/search/E3/ILI_T24/.
* GRPO smoke test: base Qwen proposals were all invalid at step 1 and step 2
  ran out of memory. Over-cap proposals are now repaired to the nearest
  feasible config, as TPE's are; logits are kept only for completion tokens.
* History note: the owner rewrote local commits after ef2a80a to drop
  co-author trailers. Records written before that carry the old hashes:
  845e7ca is 9729789 and f0b5fcc is df76f4e (identical trees).

## 2026-11-23

* E0 built and tested (see docs/validation_log.md, pass 7). Train and
  validation rows are in data/trainval/ with MANIFEST.json and uploaded to
  the leaguetst-data volume in both Modal workspaces.
* K5 pilot launched on Modal (ETTh1 on diya-027, ILI on diyapr). A first
  launch that asked for 8 CPUs with an L4 never got a container in five
  minutes; it was stopped and relaunched with 4 CPUs.
* Reviews: the verifier passed all five baseline checks; the critic's
  review led to amendments A1 to A11 in docs/PLAN.md.
* K0 is still open: the vault hand step (move and encrypt the full CSVs) is
  the owner's.
