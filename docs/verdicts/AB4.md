# Amendment AB4 (docs/prereg_ablations.md): verdict

AB4 registers, before any run, Anirudh's WP3 items (never launched on Modal,
AB3) on the local RTX of AB2 with local references: ABL_P_anon_local and
ABL_P_repair_local (ILI and ETTh1, against ABL_P_ref_local) and
ABL_G_steps30_local and ABL_G_steps120_local (the steps trend, with
ABL_P_ref_local as level 0 and ABL_G_ref_local as level 60). The launcher is
unchanged (scripts/ablate_llm.py, AB2); the queue is the new
scripts/ablations/run_ab4_queue.sh with tests/test_ab4_queue.py.

## Review history

1. Verifier (fresh): queue correct (guards, order, flags, --per-gpu 1,
   errors.jsonl paths; plans built without a GPU give the stated methods,
   seeds, steps, beta and lr), hashes, times and costs accurate; FAIL on four
   wording fixes (the J2 league arm as a Modal run, AB3's commit, the ISSUES
   location, levels not checkpoints) plus one optional. All applied.
2. Critic: may be committed and launched once eight changes are added
   (verdict below). Changes 1 to 7 were added to AB4; change 8 is this
   commit's order. The results/search pooling of AB2's items is logged in
   docs/ISSUES.md.
3. Final check (fresh verifier): VERDICT: PASS (every fix and critic item
   present and accurate, both sha256 values match, e4_rerun.sha256 OK, full
   suite and the new tests pass, append-only).

## Critic's verdict (verbatim)

AB4 may be committed and launched once these changes are added. Moving the four items to the RTX with local references was authorised in advance by LEG2_anirudh.md ("join his RTX queue ... with the local references") and by AB1 (backend switching for items never launched; local references serve local arms). AB3 confirms the items were never launched, so Diya does not need to sign off before launch. The new _local item names follow the "never reused" rule. Measured times fit the deadline. Peak GRPO memory in AB2 was flat at about 7.67 GB over 60 steps, so OOM risk is low.

REQUIRED
1. Steps-trend dependence. Replace "Every policy level is trained separately from its own seed" with: "Each level is a separate training run, but all levels use training seeds 100 to 109 and deployment seeds 0 to 9. The learning rate is constant and the RNG seeded, so a 30-step policy follows roughly the first 30 steps of the 60-step policy with the same seed. Levels are therefore dependent, as in AB3's checkpoints. The registered permutation p is reported with this caveat."
2. Pin the test. Add: "Trend statistic: Spearman of level rank (0, 1, 2, 3) against ILI held-out AUC regret in pooled seed-SD units over 30+10+10+10 values, two-sided, computed by the same code and settings as the KL trend." Name the file and its sha256.
3. What the author knew. Add: "AB3 addenda 3 and 4 describe qualitative Mac step results (the level-0 confound, the RevIN-off hypothesis, 'how many policies are worse'). The author therefore knew the direction of the Mac comparison, and had seen the ABL_P_ref_local and ABL_G_ref_local results."
4. Close the post-hoc choice. Add: "The AB4 tests are the only registered results for these questions. Mac results are reported beside them whether or not they agree. Addendum 3's 'trained at all' and 'dose' analyses run on the RTX levels as exploratory only, specified exactly as in addendum 3. No other analysis is added after results are seen."
5. Output location for AB2 too. Add: "The same rule applies to AB2's six items now under results/search/ABL_*_local. Before soumitra-leg2 is pushed, and before any picks.py, e5_referee.py, report or L5 run, they move unchanged to results/ablations/rtx/ in a separate commit with a sha256 manifest. The move is logged in docs/ISSUES.md for Diya, and Anirudh is told that llm_tests.py line 5 reads results/search/ABL_*." Diya does not need to sign off: these are run files, not ledger rows. The move is required, because otherwise the 30 extra grpo_league_from_ETTh1 runs pool into the ILI picks and E5.
6. Time estimate. Replace the single figures with ranges taken from the measured run times. For steps120, 10 x (34 to 42 min) plus 19 to 24 min of deployment gives 6.0 to 7.4 h. For steps30, 1.5 to 1.9 h. Cross-check against the measured 3.40 to 3.59 h per 60-step item (results/logs/l4_queue_status.txt). Total 10 to 12.5 h.
7. Hardware check. Add: "AB1's ABL_P_ref_local against ABL_P_ref check is not available (ABL_P_ref was not run)."
8. Commit order. Commit the queue and tests/test_ab4_queue.py only after a fresh verifier PASS, and put the verdict in the commit message.

Optional
- Report a per-deployment-seed paired comparison beside the trend, as exploratory.
- Log in the status file the latest launch time by which steps120 can still finish.

## Notes on how the changes were applied

* Change 3: the author had handled the AB2 reference run files (counts,
  timings, on-policy gaps) but computed no AUC or regret from them; AB4
  says so instead of "had seen the results".
* Change 6: steps30 is 1.7 to 2.1 h by the same arithmetic (10 x 8.6 to 10.5
  min plus 19 to 24 min); total about 9 to 12.5 h.
* Change 2: the AUC is the one spearman_trend's caller computes for the KL
  trend (same function, AUC and settings), named with file, commit and sha256.

## Registered AB-P and AB-G result (27 Nov), under Diya's AB4 and AB5

results/ablations/LLM-registered.json, written once by scripts/ablations/llm_registered.py at 27 Nov 14:54 ET (HEAD 26abd39). It equals results/ablations/rtx/RTX_tests.json (rtx_tests.py, 26 Nov 19:05 ET) value for value. The verifier's script scripts/independent/ablations_rescore_B.py, committed before the run, rebuilt every trace with no mismatch and agrees to about 1e-15, with p values within Monte Carlo error and every Holm decision the same: results/independent/AB_rescore_B.json.

### Critic's verdict (verbatim)

LLM-registered.json equals RTX_tests.json value for value, on the same inputs. AB_rescore_B agrees to Monte Carlo error, with the same Holm decisions. AB-P: anonymising field names made ILI held-out AUC worse (+0.36 SD, Welch [0.10, 0.63], Holm p 0.015). Repair showed no detectable difference (+0.03, [-0.28, 0.33], detectable 0.49). The scale trend rejects, rho +0.43: regret rose from 0.5B to 3B. AB-G: the step trend rejects (rho +0.76) but by Diya's rule it is not a dose-response, because (ii) gives mean rho 0.05 with p 1.0. Instead, (i) shows trained policies worse than the untrained model (+1.59, [1.28, 1.90]), and their 30, 60 and 120-step means are flat. KL is not rejected (p 0.37). None of this answers RQ1; E4 does.

REQUIRED
1. Call all five "registered secondary tests", not claims. Rule 6 fails on the missing paired block-bootstrap interval. Numbers only via ledger.jsonl and report.py.
2. Steps: write "GRPO-trained policies (30 to 120 steps, ETTh1 source) were worse on ILI than the untrained model". Never write "dose-response" or "longer training hurts". Write the registered p as "< 0.0001" (Monte Carlo floor): it is anti-conservative and carried by level 0. Diagnostic (iii): the logs match only at step 1 (about 3.5% of rows), so nesting is weak; levels still share seeds. E4's no-league arm was equally poor, so do not blame the league.
3. Scale: limit the statement to Qwen2.5-Instruct 0.5/1.5/3B, with one prompt and trie, temperature 1.0, ILI and budget 100. Never write "larger LLMs are worse architects". The trend is carried mainly by 3B, and 0.5B vs 1.5B was not tested. ETTh1 (descriptive) reverses, with 3B lowest. The mechanism is untested: more random sampling, constrained decoding, or a prompt not adapted per size could each explain it.
4. Anon: say "removing field names and descriptions raised ILI regret". Consistent with use of field semantics, not proof of forecasting knowledge: prompt length and tokens also change, ETTh1 runs the other way; below the planned detectable 0.42, so likely inflated.
5. Repair: say "no detectable difference", not "constrained decoding does not matter".
6. KL: say not rejected, underpowered (detectable |rho| about 0.5) and conservative. Do not call its monotone means a trend.
7. Disclosure: include AB4's fixed paragraph, plus: the families were computed twice, rtx_tests.py (26 Nov 19:05 ET) then llm_registered.py (27 Nov 14:54 ET). The values are identical: Diya's "computed once" rule was broken, nothing changed.
8. Make no cross-backend comparison with E3 random or TPE (e.g. "0.5B beats random"). ETTh1 and the hardware check (RTX minus J2: +0.12, [-0.70, 0.93]) are descriptive only.

Optional
- Seed-paired block-bootstrap intervals, labelled descriptive.
- Exploratory diagnostics per arm: unique-config rate and RevIN-off share.
