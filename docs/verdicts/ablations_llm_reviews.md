# Ablations (Mac LLM part): reviews in full, and what changed

Each report is the subagent's own text, quoted in full; only local machine paths are masked as [scratch]. The reviewers are AI subagents, not people. Scope: AB-P, AB-G and AB-Q prompted on the Mac, the hardware check, and the AB-L null rows of the Mac
reference. Both reviewed the outputs written at f80d433, before any was committed. The critic's numbers on RevIN
shares, the paired gap and the ETTh1 differences are the ones addendum 4 records.

## Verifier

> All numbers in `LLM.json`, the LLM sections and the AB-L Mac rows reproduce. No sign, unit or rounding error found. I found wording and provenance issues only. I used my own code for draws, held-out regret, AUC, permutations and chaining. AUCs equal `run_metric` exactly. Nothing in the repo was modified.
>
> | Check | Result | Mine vs theirs |
> |---|---|---|
> | 1. Runs | PASS | 240 searches + 10 training records; seeds 0-29 or 0-9; no duplicate (method, seed); spent 2971-3000. Invalid only in repair: ILI 106/4524, ETTh1 67/3521. |
> | 1. Raw consistency | PASS | Charges, drawn seeds and traces rebuilt from the ledger: 0 mismatches (traces only rebuilt, not the model's sampling). |
> | 1. Stamps | PASS | All clean. ILI ref = 31f9313 (30 runs); everything else = 0929d26. `llm_mps.py` and `leaguetst` diff is empty; `scripts/replay.py` and the ledger are unchanged. `llm_mps.py` reaches `common.py` only through `plateau.noise_free_rows` (AB-Q, run only at 0929d26, function unchanged). |
> | 2. AB-P | PASS | ref 1.6535, anon 1.9593, repair 1.3209. Diffs +0.3058 [-0.0107, 0.6222] and -0.3326 [-0.6320, -0.0333], identical. p (1e6 Monte Carlo; C(60,30) rules out exact): 0.0573 / 0.0277 vs 0.0583 / 0.0268. Holm 0.115 / 0.083 vs 0.117 / 0.080; neither rejected. Detectable 0.511 / 0.483. |
> | 3. Trend | PASS | 1.654 / 2.891 / 2.938 / 2.842; rho 0.7059, p < 5e-7 (0 of 2e6). 10 per level: 0.552, p 0.00031 vs 0.00028. Trained only: -0.061, p 0.755 vs 0.754. |
> | 3. Policy level | PASS | +1.2371 [0.9306, 1.5435], p 2.8e-6 (Monte Carlo); dose mean rho -0.15, exact 714/1024 = 0.6973. |
> | 3. Seed order | PASS | Files are in seed order 0-9; 30 distinct adapters and 30 distinct charge sequences. Run records do not name the adapter, so policy-to-seed matching rests on code, logs and folder names. |
> | 4. AB-Q prompted | PASS | 0.947 [0.833, 1.061], charges = rounded mean epochs. |
> | 4. Hardware | PASS | Prompted 1.461 vs 1.809, diff -0.348 [-0.751, 0.055]; 60 steps 2.938 vs 3.143, diff -0.205 [-1.085, 0.676]. |
> | 4. ETTh1 and diagnostics tables | PASS | All cells match. |
> | 5. AB-L FPR 0.025 / 0.05 (10,000 fresh arms) | PASS | ETTh1 0.228 / 0.355 vs 0.237 / 0.339; ILI 0.106 / 0.182 vs 0.094 / 0.175. All within about 1.6 SE. |
> | 5. AB-L p values | PASS | Raw 0.0072 / 0.378 vs 0.0095 / 0.364; adjusted 0.0013 / 0.165 vs 0.0020 / 0.148; sign flips 0.0137 / 0.533 identical. |
> | 6. Text | PASS, with the notes below | |
>
> **Notes on `docs/ablations.md`**
> 1. The Mac trend is called "Registered step trend ... rejected" and, at a glance, "The registered trend". Only the test definition is registered; the registered arms were not run. Suggest "registered test applied to the exploratory Mac runs".
> 2. AB3 addendum 3 says the permutation p "is not valid across the dependent checkpoints". The report says only "dependent", and the glance bullet gives "Holm p < 0.0001" with no caveat.
> 3. Glance AB-P line: 95% intervals are not Holm-adjusted, so repair's [-0.63, -0.03] excludes 0 while Holm p is 0.080. Neither the interval level nor "not rejected" is stated. It also gives no per-arm direction (the header's "lower is better" covers it).
> 4. Glance AB-L line: quotes only the Mac ILI rate (0.09) and omits Mac ETTh1 (0.24), while both E3 rates appear. "Too liberal" still holds for both Mac rates.
> 5. Line 130 "every reference on the same machine" covers a section whose hardware check has A10 columns.
> 6. `LLM.json` provenance hashes only the ledger, not the Mac, E3 or E4 run files it reads. `AB-L-null.json` also omits the Mac files. The Mac run files are untracked, so nothing pins their content.
> 7. The step levels include unlogged steps (budget exhausted, no update): about 28-29, 57 and 113-114 updates at 30, 60 and 120. Undisclosed; I did not compare with E4's logs.
>
> **Not verified:** model sampling or training itself, and the E3_A10 null rows.
>
> Scratch code: [scratch]/verify_llm/

## Critic

> I read the code paths and re-scored the raw runs. No repo file changed. Nothing in the code makes a number wrong.
>
> **Blocking**
>
> 1. **(a) The AB-G result lacks scope.** This concerns the At-a-glance GRPO bullet (report.py:408) and "Trained at all" (report.py:250). Neither says what the result doesn't show. The policies' ILI lookups are 98% RevIN-off, against 55% untrained. That preference is right for ETTh1, where the best RevIN-on config has 6.5 SD regret, and costly on ILI, where the best RevIN-off config has 1.03 SD. It is already there at step 30, consistent with no dose effect. Committed E4 runs agree: ETTh1 portfolio without an LLM 3.28, no-league GRPO 3.23, league 3.14 (random 1.52, untrained Mac Qwen 1.65).
>    - Fix: say the result is consistent with a source-table preference that doesn't transfer. It isn't evidence about GRPO in general or the league (league rivals only, no "none" arm). There is no ILI-source arm or ETTh1 deployment on the Mac. Attribution is untested.
>    - State "league rivals" and "10 policies × 1 search". Cite the committed E4 values. Register the RevIN-off share as addendum 4 before outputting it.
>
> 2. **"Registered step trend … rejected" (At a glance; report.py:254).** Addendum 3 says this p "is not valid", but the report says only "dependent". It also isn't the registered test: nested checkpoints, Mac, different level-60 arm.
>    - Fix: "AB-G's registered statistic computed on Mac runs; p not valid (dependent checkpoints), exploratory", and drop "rejected". Apply this to the 10-per-level and trained-only p values.
>
> 3. **(b) The At-a-glance AB-P bullet never says "not rejected".** Repair's interval excludes 0 (unadjusted). Add "not rejected after Holm". Both effects change sign on ETTh1, visible only by comparing levels: repair is worse there (+0.28 [0.13, 0.44], my calculation), anon better (−0.13). State this and make no mechanism claim.
>
> 4. **(d) At a glance quotes only ILI's Mac rate (0.09; report.py:414).** The registered rejection is on ETTh1, where the Mac rate is 0.24. Quote both. Explain the difference from E3: 30 vs 10 searches, recorded vs evenly split steps, different hardware and code. The closing note covers only E3. The liberal-test conclusion holds.
>
> **Optional**
>
> 5. **(c) The hardware check isn't hardware-only.** E3 predates A25 pins and has no env record. Sampler and RNG streams differ, so equal seeds share only oracle draws. J2 is 5 A10 plus 5 A10G, so "A10" is wrong. Add a sentence. No test claim: ok. Nothing hinges on it: the trained-minus-untrained gap is +1.33 (A10) and +1.48 (Mac, same seeds).
> 6. **Repair is confounded:** 53% of its ILI proposals are replaced by the nearest *unseen* config (repeats 0.06 vs 0.18). Say so. Label the diagnostics table ILI and add ETTh1's rates (0.58 / 0.02).
> 7. **Provenance:** LLM.json's inputs hash only ledger.jsonl, and AB-L-null omits the Mac runs. The report cites the analysis commit, not the run commits (31f9313 for ILI ref, 0929d26 otherwise; llm_mps.py and leaguetst unchanged between).
> 8. **AB-G lacks diagnostics** (152 lookups, repeats 0.22–0.28). "Trained-levels row" doesn't exist.
> 9. **"Trained at all"** averages three deployments per policy against single searches, probably conservatively. Paired same-seed: +1.43 [1.12, 1.74], 10/10 policies worse.
>
> **Ok:** deployment across levels (prompt, trie, oracle, budget, seeds), 30 vs 10, checkpoint nesting, ETTh1 "in-sample" label, scale test counted.

## What changed (commits f80d433 and 5abbd20)

* AB3 addendum 3 (f80d433, committed before its outputs): policy-level comparison of trained and untrained, and
  the per-policy dose analysis; the trend is no longer reported as a rejection and its p is called not valid.
* AB3 addendum 4 (5abbd20, post hoc: the critic had reported the numbers and the code was run once before the
  commit): RevIN-off shares for the Mac and the committed E4 and E3 runs, best regret by RevIN setting,
  the paired gap, ETTh1 AB-P differences (descriptive), per-checkpoint diagnostics and the logged updates.
* Report wording: scope and no-causal-claim sentence for the trained policies; AB-P "none is rejected after
  Holm", unadjusted intervals, ETTh1 signs stated; AB-L rates from both Mac tables; hardware check called not a
  hardware test; provenance now digests all 16 input files.
* Not changed: the values of the tests AB3 names; the scripts were re-run from 5abbd20 and gave the same numbers.
