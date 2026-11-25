# Ablations (CPU part): reviews in full, and what changed

Each report is the subagent's own text, quoted in full; only local machine paths are masked as [scratch]. The reviewers are AI subagents, not people. docs/HANDOFF_anirudh.md requires a verifier and a critic before a result is committed. Scope: AB-C, AB-Q,
AB-N, AB-X, AB-R, AB-L calibration, ST, RQ3. (The version in commit 0929d26 was condensed and wrongly headed
"verbatim"; this one replaces it.)

## Verifier 1 (results of 31f9313)

> **Verdict:** every stored number checked reproduces. The problems are in interpretation and in two unfulfilled text promises. My code is in scratchpad/verify_cpu/. The statistics import nothing from leaguetst. I imported leaguetst only to re-run traces, check `noise_offset`, and run one extra experiment.
>
> | # | Item | Result |
> |---|---|---|
> | 1 | Charges and traces | PASS. 2,000 runs plus 620 E3 runs, 0 mismatches (seed draw, extra epochs, spent, lookups, trace with noise). Spent 2971 to 3000. |
> | 2 | AB-C | PASS. Means, SDs, intervals and detectable_80 match to 4e-15. My p (400k permutations): 0.3272 / 0.7798 / 0.4725, stored 0.3289 / 0.7796 / 0.4752. Holm correct. Re-runs 28/28 identical (5 ILI and 2 ETTh1 seeds per arm). Ref seeds 0-29 equal E3 pg_league. |
> | 3 | AB-Q | PASS, 6e-15. Plateau 59/282/2368. E3 column matches E3 summary. |
> | 4 | AB-N | PASS, 4e-15. Noise SDs 0.0079614 / 0.0097869. |
> | 5 | AB-R | PASS. 92 of 126 variants refitted (all gbt and ridge, 20 forests), tau difference 0. No split overlap, no curve information in "config", training target is seed 2021 only. |
> | 6 | RQ3 | PASS, 6e-16, all 28 pairs. RevIN pairs match. 2545 windows alone proves nothing (a standard 12/4/4 split gives the same test count), but all 80 file means equal the ledger val MSE to 1.2e-7, so they are validation. |
> | 7 | ST-robust | PASS. Points 4e-15, MAD SD 0.0085479. Largest bootstrap-end gap: 0.066 (IQM), 0.0225 (PI). |
> | 8 | AB-L-null | PASS within MC error. ILI FPR at 0.05 is the loosest: stored 0.297, mine 0.270 (30k chains; script-size MC SD 0.017). ETTh1 0.486 vs 0.494. |
>
> Re-runs of AB-Q, AB-N and ST: 112/112 traces identical. Script sha256s match; listed input digests match. Family size (3) and endpoint agree with the prereg and E3.
>
> **Problems, ranked**
>
> 1. **report.py:312 (plateau.py:91-100).** "Seed noise hides differences" is unsupported. The AB-Q gap is in-sample. My check: lookups are 2-seed means, scored on the third seed, 30 seeds × 3 hold-outs. TPE vs random is −0.39 SD (P 0.84) in-sample but −0.05 (P 0.53) held out; RE −0.26 becomes +0.03. Caveat: only 3 hold-out seeds. The AB-Q section warns of this; the summary contradicts it.
>
> 2. **referee.py:81,101; report.py:136.** The tau target includes seed 2021, which also supplies curves and training targets. Field-explained effects account for 23-33% of seed-deviation variance (chance about 6%). ILI against the 2022/2023 mean: E2 referee 0.567 becomes 0.458, config-only 0.611 becomes 0.504, "one full seed" 0.675 becomes 0.471. ETTh1: 141 of 150 targets are the seed-2021 value. This follows AB3's definition but overstates skill.
>
> 3. **one_seed_bias.py:72,100.** "Optimism" for strict mixes two biases. The cut table's best is 0.995 SD below the full best and cancels the incumbents' bias. Random shows +0.04, but incumbent values are 1.04 SD too good (all methods 0.89 to 1.10). Report the decomposition.
>
> 4. **report.py:339, learning_null.py:86,95.** p = 0.001 is too strong. Observed between-policy slope SD is 0.21 (ETTh1) and 0.27 (ILI, adjusted) against about 0.15 for raw null chains, and the null rests on 10 reused episodes. Two-sided sign-flip on the adjusted ETTh1 slopes gives p = 0.014 (t 0.016). Raw ETTh1 p with episode resampling is 0.020, stored 0.0125. ILI conclusion unchanged.
>
> 5. **rq3.py:65,86,128.** Shares and level means are marginal on an unbalanced space. Drop-one shares give Spearman 0.26 (stored 0.14, n = 8), so "disagree" (report.py:347) is weak. Adjusted ETTh1 best levels flip: stride full to half, n_layers 1 to 2.
>
> 6. **rq3.py:7.** Promised seed noise is never output. On ILI it is 24% of variance of means. Cross-seed reproducible increments: patch×RevIN 0.0116 becomes 0.0058, patch×dropout 0.0144 becomes 0.0086; stride×RevIN holds (0.0301).
>
> 7. **common.py:193.** AUC in raw MSE, registered for the endpoint, is missing from AB-C.
>
> 8. **report.py:55,357.** ETTh1 is not purely in-sample: AB-C incumbents are held out 39% at budget 10, 21% at 100.
>
> 9. **common.py:73; plateau.py:83, one_seed_bias.py:88, learning_null.py:108, referee.py:76.** Run records carry no git hash or packages. Input digests omit the E3 run files; robust.py omits qwen runs and fidelity files.
>
> 10. E1 RevIN uses patience 10 (21-30 epochs run); REVIN uses patience 100. "30 vs 100" is confounded.
>
> **Not checked:** ST-power, AB-X, LLM/Mac code, AB-L itself, 34 forest variants, most AB-R bootstrap intervals, learning_null's `step_marks` branch, the test suite.
>
> **Scope note:** outside your list I read the E4 training.jsonl logs (step and episode fields), results/fidelity, results/revin_prediction.json and docs/ablations.md. No forbidden folder was opened.

## Critic (ablations.md at a7ca3dd)

> Verdict: REVISE
>
> Findings 1 and 2 change conclusions. Line numbers refer to a7ca3dd and the docs/ablations.md generated with it. I recomputed hold-out numbers from the committed-format traces and re-ran nothing heavy.
>
> 1. **docs/ablations.md:7** "Noise-free lookups separate the methods only in-sample… Most of the in-sample gap is the search fitting the seed noise". Contradicted by the file itself.
>    - E3's noisy, held-out replay gives TPE -0.38 SD (line 21: 1.14; random 1.52), larger than the in-sample -0.34. TPE @50 is a Holm win (docs/claims/E3.json; line 272).
>    - The hold-out scores one seed and cuts lookups to two. Its interval resamples search seeds only. TPE minus random per held seed: -0.16, +0.05, -0.05. At budget 100, TPE's 30 runs hold 9, 3 and 8 distinct configs.
>    - The registered reading (prereg:99-101, plateau explains the null) is silently replaced by a check whose key number was known before registration (prereg:273-275).
>    - Fix: drop "only" and the causal sentence. Show per-seed differences and E3's win. Say AB-Q neither supports nor refutes "noise explains the null".
>
> 2. **scripts/ablations/external_hpo.py:181-184** versus prereg:221 "DEHB's ask/tell with TPE's nearest-unseen repair". Not TPE's repair: TPE discards the raw proposal and learns from the evaluated config, while DEHB gets that fitness on the raw vector. Only 38.5% of the grid (2368/6144) is feasible. The repair rate is unreported.
>    - Fix: report the rate and label DEHB handicapped, or rerun by amendment.
>
> 3. **docs/ablations.md:9, 51** "0 of 3 differ from random search" and "SMAC3 2.4.1… through the E3 oracle". The first accepts the null: smac p 0.053, 0.24 SD against MDE 0.40. A SMAC author objects that the section omits:
>    - random design of 10 (default Sobol), retrain every trial (default 8), exhaustive acquisition, deterministic=True on a noisy oracle, ordinals as unordered categoricals;
>    - Hyperband and DEHB starting at 1 epoch, whose tau with the final value is 0.22 (line 168).
>    - Fix: "none of 3 rejected after Holm; effects below about 0.4 SD are not excluded". List the changes in the section.
>
> 4. **docs/ablations.md:109, 121** "Not run." while the AB-L table uses ABL_P_ref_mps: 19 of 30 unfinished Mac searches, not labelled Mac, absent from inputs_sha256. Line 11 quotes only the E3 rates (0.36, 0.18); the larger Mac ILI source gives 0.10.
>    - Fix: wait for 30 and hash the file, or drop the row. Give ILI as 0.10 to 0.18.
>
> 5. **docs/ablations.md:11** "GRPO learned on its ETTh1 source". Exceeds the AB-L critic's allowed wording ("raised the policies' mean reward"); the none arm's slope is larger (+0.247). Lines 382-383 compute MDE against zero although line 120 puts the null at +0.10.
>    - Fix: use the verdict's wording, add the none arm, flag the MDE.
>
> 6. **docs/ablations.md:8** "The nulls have limited resolution… E4's ILI comparisons… 1.02 to 1.25 SD". One of those three is not a null: league versus warm-start TPE is +1.50 (line 372), rejected, league worse (Holm p 6e-5). ETTh1's unit is inflated (line 259, ratio 0.65): 1.11 to 1.32 becomes about 1.7 to 2.0 robust SD.
>    - Fix: state the loss; split the tables.
>
> 7. **Order evidence.** 31f9313 is 00:58:13 and the first output 00:58:27, but the branch is unpushed, results are untracked, and first outputs were overwritten at 01:17-01:22. CPU records now carry d632c41, which postdates seen results. AB-X was launched three times (logs 01:00, 01:03, 01:20); docs are silent.
>    - Fix: disclose reruns and reasons, state whether numbers changed, push before the next registration.
>
> 8. **docs/ablations.md:12-13, 134, 151, 204, 219** lack "exploratory" for the independent-seed target, conditional shares, value optimism and per-seed increments (addendum: "All of it is exploratory"). AB-R tables omit registered bootstrap intervals (prereg:92-95). "The two datasets disagree" rests on Spearman over 8 fields.
>    - Fix: labels, intervals, "no detectable agreement".
>
> 9. **docs/neurips_checklist.md:12, 15, 17, 20, 24**
>    - Row 4 "Yes": results uncommitted, REPRODUCE.md missing. Row 7 "fixed before the results": family 3's Holm composition is still owed. Both Partly.
>    - Row 9 "weather": no such dataset.
>    - Row 12 omits SMAC3, DEHB, ConfigSpace, Optuna.
>    - Row 16 "with verifier and critic passes" hides that both are AI subagents. No verdict files exist for AB-C/N/Q/R/X, ST, RQ3.
>
> 10. **docs/ablations.md:3, 10, 17** "ILI regret is scored on seeds the search never saw" is false for AB-Q's main table and AB-N's reported column. "0 of 3… change ILI AUC" accepts the null. "59 of 2368… within one seed SD of the best mean" uses a selected minimum of means with SE 0.58 SD.
>     - Fix: qualify each.
>
> **Unanswered**
> 1. Why does TPE win E3's held-out replay but not the hold-out or picks?
> 2. What effect size matters, and where is an equivalence test (related_work item 5)?
> 3. Do rankings survive resampling the three table seeds?
> 4. Where are BOHB, one-epoch-then-top-K, and a common default first config?
> 5. Do AB-X results survive another fidelity ladder?

## Verifier 2 (changed outputs at d632c41, report at a7ca3dd)

> Reviewed: results written at d632c41 and the report from report.py at a7ca3dd (regenerating it in scratch gave a byte-identical docs/ablations.md). At 01:29 the authors began uncommitted edits to scripts, prereg (addendum 2) and report.py. Line numbers are a7ca3dd unless marked.
>
> 1. AB-Q hold-out: PASS. All 900 runs (10 methods x 30 seeds x 3 hold-outs) re-ran with identical traces and charges, through my own two-seed table. AUCs, paired differences, intervals and probabilities match (error 0). Scrambling the held seed changed 0 of 45 runs.
> 2. AB-N: PASS. Recomputed to 2e-15; the identity holds to 4e-16 for all 20 rows.
> 3. AB-R: PASS. 57 of 63 variants per table refit exactly. ILI has 150 configs and ETTh1 has 9, so None below 30 is sensible.
> 4. RQ3: PASS on arithmetic: shares, 7 pairs x 3 seeds, Spearman 0.262. Caveat in problem 6.
> 5. AB-L: PASS. 14/1024 = 0.0137; 546/1024 = 0.533.
> 6. Records: PASS for 3,080 lines. All hashes were clean and in `git log`; listed digests matched. Mac lines carry the hash only at env.git_hash. AB-L.json (Soumitra's) has no provenance block.
> 7. Numbers: PASS, more than 15 recomputed from raw runs or ledger. Wording: FAIL, below.
> 8. Mac wording: no registered claim found; weaknesses in problems 2 and 3.
>
> Problems, ranked:
>
> 1. ablations.md:7 (report.py:341): "only in-sample" and "most of the in-sample gap is the search fitting seed noise" overstate.
>    - It omits E3's held-out TPE gap of -0.38 [-0.57, -0.18], p about 0.0008 (referee -0.30).
>    - The hold-out runs' own in-sample gap is -0.39; -0.34 is a different search. The interval's lower end, -0.22, is over half of it.
>    - The interval resamples search seeds only. On held seed 2022, TPE's 30 runs end on 3 configs. Over the three hold-outs (t, 2 df), referee is -0.15 [-0.34, +0.04], so "1 of 9 below zero" disappears.
>    - The pending draft fixes most of this.
> 2. ablations.md:121 uses 19 of 30 searches from the unfinished ABL_P_ref_mps (25 at last check), unlabelled, while ablations.md:109 says AB-P "Not run." learning_null.py:118 omits that file from provenance, also in the pending diff. ablations.md:11 quotes ILI false-positive rate 0.18 and omits the 0.10 from the source with exact step boundaries.
> 3. report.py:37 would print "1 registered test(s) not run" inside the exploratory Mac AB-P table once LLM.json exists. "Not run." names no registered item or Holm size; AB1's deadline is 28 Nov.
> 4. ablations.md:128 (report.py:117): "searched with ETTh1's observation-noise rule" is false for strict, which per AB3 uses the full ILI pooled SD. Not fixed in the pending diff.
> 5. AB-X and AB-C print SD units only; the prereg shared rules require raw MSE too (auc_raw_mse is in the JSON).
> 6. rq3.py:73-74: the within-config variance includes the common seed shift, which adds nothing across configs; 23.4%, not 24.5%. ablations.md:215: 0.809 + 24% exceeds 1. Over additive, all-pairs gains 0.14 R2 in-sample but about 0.02 on a held seed, so the increments partly fit seed noise.
> 7. ablations.md:3: "ILI regret is scored on seeds the search never saw" is false for AB-Q noise-free and AB-N reported AUC.
> 8. ST-power inputs omit the E3/E4 run files, AB-C.json, AB-X.json and results/e5. RQ3 omits the per-window files.
> 9. Hold-out AUCs use the held seed's own minimum as reference (plateau.py:74, uncommitted line numbering may shift); the report does not say they are not comparable with 0.45 or 1.14.
> 10. ablations.md:11, "GRPO learned", rests on an uncorrected exploratory p 0.014 (Holm over 2: 0.027).
>
> Commit addendum 2 before dehb_vector runs; scripts are dirty now.
>
> Not checked: re-running AB-X, AB-C, AB-N or ST searches; the AB-L-null simulation; E5/E6 rows; addendum-2 outputs.

## What changed after the reviews

All numbers named above are regenerated from scripts at d8364cd. Addenda 1 and 2 of AB3 were committed
before their outputs existed (d632c41, 7a67954). The registered parts of AB-X, AB-C, AB-Q and AB-N were not
changed. Corrections to the write-up: the AB-Q section and "At a glance"; AB-X caveats, DEHB repair rates
and an exploratory DEHB variant; exploratory labels; robust-SD detectable effects; the E4 loss labelled;
checklist rows. The AB-X registered traces were identical in the two runs; a third run added only
repair counts and git hashes.
