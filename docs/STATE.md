# Build state

Updated by every loop iteration. Newest first.

## 2026-11-24, evening

* E4 rerun complete on Soumitra's Modal workspace and merged (PR #1,
  70461a8): J2 (ETTh1 to ILI) 20 of 20 policies trained and deployed, J1
  (ILI to ETTh1) 25 of 25 deployed; no on-policy stop, no error. Modal
  disabled the workspace at its spend limit during J1, so the driver died
  and one run was retrained. Retried under A29: grpo_league_ILI_seed109 and
  grpo_none_ILI_seed108 (the A26 worst case treats them as failed);
  grpo_none_ILI_seed109 trained once and its training record was lost in
  the driver. Dashboard screenshots in results/runs/E4/soumitra_*/. No E4
  endpoint is computed yet.
* Cost: $22.17 metered for the 45 policies, about twice the estimate; all
  remaining GPU plans are re-costed from it.
* Next leg for both teammates: docs/LEG2_soumitra.md (J3 if A17 holds, AB-L,
  run end to end on Soumitra's side: an independent E4 analysis that also
  decides J3 under A30, J3 if it holds, AB-L, the model-scale and KL arms
  on Modal or his RTX by a registered rule, a fresh-clone reproduction
  audit) and docs/LEG2_anirudh.md (WP3 cut to fit $30, CPU ablations,
  write-up); amendments A30 (docs/prereg_E4.md) and AB1
  (docs/prereg_ablations.md), before any E4 endpoint or ablation result.
  Nothing waits for December credit.

## 2026-11-24

* Pick reseeds (fresh seeds 4001 and 4002, every E3 method's incumbents at
  budgets 25, 50 and 100): on both tables no method beats random search
  after Holm, prompted Qwen included; TPE's in-table win on ILI does not
  replicate. The cross-dataset portfolio is worse than random on both.
  results/picks/*_analysis.json. RQ1 for the CPU methods is a null result.
* E4 GRPO training (diyapr, 23 Nov): 34 of 45 runs ran out of GPU memory
  (three jobs per A10G). All 45 are rerun on Soumitra's workspace, two per
  GPU, by the owner's decision (docs/HANDOFF_soumitra.md, docs/ISSUES.md);
  the 11 that finished on diyapr are superseded.
* Modal outputs on our side are committed under results/runs/ (records,
  328 E3b and E6 checkpoints, per-window validation errors).
* The wider design space (A16) is merged as code and a costed page
  (docs/wide_space.md); it runs after the report.
* E4 rerun registered before any result (docs/prereg_E4.md A25, A26): one
  backend for all 45 policies, Soumitra's Modal workspace or a local RTX
  (scripts/llm_local.py on the same runner as Modal); pinned packages and
  Qwen revision; code digests in docs/e4_rerun.sha256; a local backend
  reruns prompted Qwen on the same card; failed-run rule; cut-off 26 Nov
  18:00 ET. Plan: docs/HANDOFF_soumitra.md.
* E5 under A24 (t interval, Bonferroni): the referee's tau on prompted-Qwen
  proposals against quality-matched random sets is ILI +0.009 [-0.029,
  +0.048], ETTh1 +0.020 [-0.009, +0.049], non-inferior at 0.05, which is
  not confirmatory (margin set after these were seen). Against plain random
  sets ETTh1 drops by 0.095 [-0.129, -0.062]: proposals sit in a narrow
  band of quality that is harder to rank.
* Stage-1 pipeline retired (tag stage1-baselines). A test had loaded the
  full CSVs, test rows included, for a shape check on every pytest run;
  removed and disclosed in docs/ISSUES.md.
* Critic (2026-11-24): workshop-ready no. Blocking: the GRPO rerun, a bound
  on the smallest detectable effect, E5's headline metric and margin, stale
  docs. Replay summaries now say when regret is partly in-sample.

## 2026-11-23, after midnight UTC (19:30 to 20:00 EST)

* E3b ETTh1 (64 live runs, seed 3001): K7 passes. Mean signed gap -0.0017,
  95% interval [-0.0042, 0.0009]; median |gap| 0.43 seed SD (threshold
  1.15). Spearman over all 64 is 0.93 but 0.08 within the top 20: the
  table cannot order its leaders, consistent with 50 configs within one SD
  of the best. Pending verifier re-score.
* E6 ETTh1 PatchTST/42 arm in the ledger (40 runs, 4 horizons x 10 seeds,
  checkpoints and per-window errors downloaded).
* E3 ETTh1 prompted Qwen (10 seeds, in-sample, no claims): AUC 1.30 against
  the controller's 1.32, a gap within in-sample noise; on ILI it was last.
  Corrected 2026-11-24 (critic): an earlier line here explained the
  difference by RevIN, which is untested.
* E5 (RQ2) audit built (scripts/e5_referee.py). Against the multi-seed mean,
  the referee's tau on the prompted architect's proposals differs from
  quality-matched random configs by ILI +0.014 [-0.017, +0.043] and ETTh1
  +0.020 [-0.003, +0.044] (10 runs each): no difference detected. No
  non-inferiority margin was fixed yet, so this is not a non-inferiority
  result. Corrected 2026-11-24 (critic): the first version quoted tau
  against the seed-2021 final value, the seed the referee's features use.
  PLAN specified plain random configs; quality matching is a deviation. GRPO proposals are added
  when the E4 deployments land. Pending verifier and critic.
* Credit arena (docs/verdicts/arena_credit_2026-11-23.md): transfer
  deployments, then pick reseeds (ILI launched as a smoke test, ETTh1
  next), a $3.5 reserve on diyapr for E6 reruns, the uniform-rival arm only
  under A17's condition, the extended test prereg committed by 28 Nov.
  Nothing beyond free credit; no new GPU work on diya-027.
* Dead code: regret_at and regret_curve removed. The stage-1 baseline
  pipeline (scripts that score the test split, unreachable from the E0 to
  E8 pipeline) is being retired; tag stage1-baselines keeps it.

## 2026-11-23, night

* ETTh1 top-100 reseeds done: pooled SD 0.0132 (seed 2021, used to select
  the top 100, left out); 50 configs within one SD of the best; the
  best config by 3-seed mean is p12s12l4h2d16f4r3n0i (RevIN off, 1.0687).
* ETTh1 replay (lookups with observation noise, in-sample, no claims): the
  factorized controller has the lowest AUC (1.32 SD), then TPE (1.37),
  random 1.61; screening methods do poorly. One dominant field (RevIN) is
  easy for a factorized policy. The ILI portfolio on ETTh1 scores 5.94.
* RevIN study pre-registered (A23): the critic found 64% of RevIN-on
  configs peak at the 30-epoch cap. A 100-epoch pair check (20 twin
  pairs), a RevIN-on twin arm in E6 and a month-by-month prediction (no
  reversal predicted: the off-advantage holds in all four validation
  months) were committed before any test evaluation.
* E6 reframed (A23): the space's best config against PatchTST/42, 10 seeds
  each, on both datasets.
* Credit: diya-027 is shared with another project's Modal apps
  (ocmdp-experiments, started 18:36); remaining LeagueTST GPU work runs on
  diyapr.

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
  [0.0013, 0.0070]; median |gap| 1.13 SD > 0.95 in the median-SD unit used
  then, 0.91 SD in the pooled unit of results/e3b/ILI_T24.json, so K7 fails
  on the bias criterion alone). The 44 stratified configs
  are unbiased (+0.05 SD, median 0.89 SD); the top 20 are 1.6 SD optimistic,
  a winner's curse that an in-table leave-one-seed-out check reproduces.
  Fallback applied: RQ1 is stated for the table only; picks are scored on
  fresh seeds before any comparison.
* E3 ILI CPU replay (30 seeds): TPE has the lowest regret (AUC 0.91 SD) and
  is the only method that beats random search after Holm (budget 50), in
  the table only: on fresh seeds it does not (2026-11-24). The
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
