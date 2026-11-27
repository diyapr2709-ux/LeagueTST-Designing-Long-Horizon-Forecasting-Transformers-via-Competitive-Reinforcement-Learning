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

2026-11-27: the line below is added by Diya for scripts/gates.py G6; the verdicts quoted above are unchanged. The critic's acceptance was conditional on changes 1 to 8, which this file records as applied, and the verifier's final check above is PASS; the arena judge of 27 Nov (docs/verdicts/arena_AB6_2026-11-27.md, D4) confirmed this file qualifies.
Verdict: ACCEPT
