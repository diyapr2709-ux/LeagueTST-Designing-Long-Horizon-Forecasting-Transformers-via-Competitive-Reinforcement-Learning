# Registered RTX families (AB2 and AB4) and the ablation report: reviews in full, and what changed

Each report below is the subagent's own text, quoted in full; only local machine paths are masked as [scratch], [repo] or [local path]. The reviewers are AI subagents, not people. Part 1 was written on 26 Nov in the afternoon for the AB2 values and kept on the local branch local/ab2-provisional-do-not-push (d577ef8) while AB4 ran (AB5, blinding); it is published here unchanged now that AB4 has run. Part 2 covers AB4 and the report on anirudh-ablations on 26 Nov in the evening. The verifier's scripts and outputs are in results/ablations/verify_rtx/ (AB2) and results/ablations/verify_rtx/ab4/ (AB4); in the committed copies of recompute_ab4.py and compare3.py only the scratch paths were replaced by paths inside the repository (rerun from the repository: identical output apart from the recorded code path).

## Part 1: AB2 (provisional, 26 Nov afternoon), as written on d577ef8

Each report is the subagent's own text, quoted in full; only local machine paths are masked as [scratch].
The reviewers are AI subagents, not people. Scope: the AB2 items computed by scripts/ablations/llm_tests.py
at 510633d from origin/soumitra-leg2 at 7fd5085, outputs first committed as 84a82f9 on the local branch
local/ab2-provisional-do-not-push. This file holds AB2 values, so it stays on that branch until AB4 has run
(AB5, blinding).

### Independent recomputation (fresh agent, blind to these outputs)

> I recomputed everything with my own code: my own table, held-out regret, AUC, Spearman permutation test and Holm. I used leaguetst's run_metric only as a cross-check, and it agrees exactly (largest per-seed difference 0.0). The charged seeds in every run also equal the oracle's draw, so "seeds the search did not see" is the same under either reading.
>
> **ILI held-out AUC (pooled SD 0.009787)**
>
> | Item | n | Mean [95% t] | RevIN-off share |
> |---|---|---|---|
> | ABL_P_05b_local | 30 | 1.391 [1.225, 1.557] | 0.413 |
> | ABL_P_ref_local | 30 | 1.565 [1.377, 1.753] | 0.551 |
> | ABL_P_3b_local | 30 | 2.056 [1.788, 2.324] | 0.524 |
> | ABL_G_beta0_local | 10 | 3.395 [2.918, 3.873] | 0.982 |
> | ABL_G_ref_local | 10 | 3.260 [2.856, 3.664] | 0.989 |
> | ABL_G_beta01_local | 10 | 3.095 [2.625, 3.564] | 0.981 |
>
> All nine tables are complete: seeds 0–29 for prompted, 0–9 for transfer. Each table has only its registered method, no duplicates, and no on-policy stops.
>
> **Tests**
> - **Scale trend:** rho = +0.4295 (bigger model, worse AUC). p = 3.10e-5 from 1,000,000 permutations (30 hits). A 10-million check gives 3.19e-5. Holm over 3 gives 9.3e-5: **reject**.
> - **KL trend:** rho = −0.1698, p = 0.3738 from 1,000,000 permutations. Holm over 2 gives 0.748: **not rejected**.
>
> **Descriptive (paired, shared seeds)**
> - ABL_G_ref_local − J2 league, seeds 0–9: +0.117 [−0.700, 0.935]
> - ABL_P_ref_local − Mac ref, 30 seeds: −0.089 [−0.351, 0.174]
> - ABL_P_ref_local − E3 prompted, seeds 0–9: −0.238 [−0.804, 0.327]
> - ETTh1 AUC, run_metric convention / every pick scored on its multi-seed mean: 0.5B 1.416 / 1.334, 1.5B 1.468 / 1.413, 3B 1.319 / 1.248
>
> **Checks (all pass)**
> - **Manifest:** all 9 sha256 match.
> - **Commits:** prompted ILI ran at c2dda8d, prompted ETTh1 and G_ref at 80a85b9, beta0 at 7bb46f4, beta01 at 79f1bf0. All are clean, descend from c2dda8d and hold launcher 3919d683. The ten frozen E4 files match a1751ae. These commits differ only by an unused script, scripts/independent/diff_replay.py.
> - **Revisions:** all as AB2 lists.
> - **Training:** each item has 10 records, seeds 100–109, steps 60 (last logged step 60, 57–59 log rows), its beta (0, 0.04, 0.1), lr 1e-4, league rivals, ETTh1 source.
>
> **Ambiguities**
> - "ETTh1 in-sample": run_metric holds a pick out when it has 3 seeds. That happens for only 20–35% of picks, so I give both versions.
> - RevIN share is pooled over charges. The mean of per-run shares differs by at most 0.002.
> - Transfer completeness can't be fully checked from git. Adapters aren't committed, and with no errors.jsonl at the pin, "no stops" looks the same as "stops not committed". Deployment runs carry no run name, so the policy-to-seed pairing can't be checked.
> - My p uses (hits+1)/(N+1); its Monte Carlo SE is about 5e-6.
> - The local HEAD moved to 3c359b8 during my run. I didn't make it, and it touched none of the files I read.
>
> The repo is unmodified. Files are in [scratch]/recompute/:
> - recompute.json
> - recompute.py
> - supplement.py

Programmatic comparison with results/ablations/LLM.json (results/ablations/verify_rtx/diff_vs_mine.txt):
32 of 32 values match (means and intervals to 1e-9; p values within Monte Carlo error; decisions
identical; RevIN shares exact).

### Critic (the standing direction reviewer, acting as critic)

> **Critic's verdict on the AB2 registered outputs (RTX, provisional; computed at 510633d from 7fd5085; local branch 84a82f9): GO WITH CHANGES**
>
> The numbers stand: the AB2 arms pass every check and the tests are the registered ones. The wording does not yet stand. Fix it in report.py only, in a commit logged as written after seeing the results. llm_tests.py's computation stays unchanged.
>
> 1. **Blocking. The summary bullet omits direction and scope.** Write: "a larger model gives a higher, that is worse, ILI held-out AUC: 0.5B 1.39, 1.5B 1.56, 3B 2.06 SD; rho +0.43, Holm p < 0.0001; ILI only." Never write "larger models are worse" without that scope.
>
> 2. **Blocking. The report contradicts itself.** It says "nothing is decided from these tables" and also prints "rejected: yes". Use AB5's actual rule instead. The scale-trend rejection cannot be undone by the pending tests, because Holm is monotone. The non-rejections are not final.
>
> 3. **Blocking. Each test must show what it could detect, as the registered rules require.**
>    - KL trend: not rejected. It could detect only |rho| of about 0.53, so it says nothing about smaller effects.
>    - Scale trend: it could detect about 0.33.
>    - Both figures are Fisher-z approximations, optimistic with tied levels. Put them in the family rows and in the bullet.
>
> 4. **Blocking. Add a scope paragraph under the scale trend.**
>    - The three levels are three Qwen2.5-Instruct checkpoints, run under one fixed protocol: the same prompt, temperature 1.0, constrained decoding and one card. They differ in more than size, so the test cannot isolate size.
>    - Three points allow no extrapolation.
>    - Most of the rise is between 1.5B and 3B (+0.49, against +0.17 from 0.5B to 1.5B). No pair was tested.
>    - The ETTh1 table (in-sample, descriptive) is not monotone: 3B 1.32, 0.5B 1.42, 1.5B 1.47, with overlapping intervals. Do not write "3B is best on ETTh1".
>
> 5. **Required process. Blinding is by agreement, not by access.** The run files have been on origin/soumitra-leg2 since 25 Nov. These outputs reveal levels 0 and 60 of AB4's steps trend (1.56 and 3.26). Keep the outputs, this verdict and the verifier's report on the local branch until AB4 lands. The only planned post-seeing change to the computing code is moving the pin.
>
> 6. **Optional.** The scale levels share search seeds 0 to 29. The unpaired permutation test ignores that pairing, which probably makes it conservative. Say so beside the AB-G caveat.
>
> 7. **Optional, labelled post hoc.** For context, E3 random search scores 1.52 [1.31, 1.72] on ILI. Against that, 1.5B is level with random, 3B is worse, and every KL policy (3.1 to 3.4) is worse still. Add no tests.
>
> 8. **Optional. The descriptive comparisons.**
>    - Rename them "AB1 hardware check (descriptive)".
>    - Intervals of about ±0.8 SD do not show the RTX matches Modal.
>    - The E3 searches predate the A25 pins.
>
> 9. **Optional. RevIN-off shares** (prompted 41%, 55%, 52%; trained policies 98 to 99%) are descriptive. They do not follow model size, so they do not explain the scale trend.
>
> 10. **Optional.**
>     - Label the levels "0.5B, 1.5B, 3B".
>     - Remove the nested parentheses in the status line.
>     - State that the 3B arm ran under the Qwen Research License (non-commercial), and check whether releasing its generated proposals needs that licence's notice.

### Standards reviewer (outputs of 84a82f9)

> **FIX FIRST.** Provenance and numbers meet standard; the report wording does not.
>
> **(1) Provenance: meets standard.**
> - git_hash is 510633d, clean.
> - leaguetst_path is inside the blind worktree.
> - The source is 7fd5085, equal to the tip.
> - fetch_time is recorded (26 Nov 16:44).
> - The manifest matches 9 of 9; all 14 files were read from git at 7fd5085.
> - All nine AB2 arms are DONE between 00:40 and 13:44 ET on 25 Nov, after c2dda8d (00:13), with no check failures and one GPU name.
> - 84a82f9 is on no remote-tracking branch.
>
> **(2) The diff: your account is incomplete.**
> - LLM.json: the mps and hardware_check blocks are identical; the provenance differs only in git_hash, script_sha256 and the new leaguetst_path.
> - ST-power.json matches what you said; its only input-digest change is LLM.json.
> - docs/ablations.md also changed in places you did not list: the header line, the ST source line, two new power-table rows, and a new history bullet ("Not done from WP6…"). Correct your account.
>
> **(3) Numbers: meet standard.** Every value I spot-checked matches LLM.json: scale and KL levels, p and Holm p, the three descriptive comparisons, the ETTh1 table, the RevIN shares and the training gaps. ref_local and beta01 report the same largest on-policy gap (0.006126) because it occurs at step 1, policy 102, before beta acts; this is not a data problem.
>
> **(4) Wording: fix these.**
> 1. **The rejected test's statistic comes from AB5, which the report does not cite** (report.py:230). AB5 (510633d) fixed the scale-trend statistic after the AB2 runs existed, though before Anirudh saw them. Cite AB5 and say exactly that.
> 2. **The direction is never said in words.** The summary bullet (report.py:502, ablations.md:11) gives only "rho +0.43, rejected". State plainly that larger prompted models had worse ILI AUC regret: 0.5B 1.39, 1.5B 1.56, 3B 2.06 SD.
> 3. **The status sentence contradicts itself** (report.py:228). "Nothing is decided from these tables" sits beside a "rejected: yes" that the next sentence says cannot be undone. The parentheses are also nested. Say that the scale-trend rejection holds whatever the pending tests show, and that only the pending tests are open.
> 4. **A p is printed as zero.** The power table shows the scale-trend p as "0.000" (report.py:390). Use pfmt, or the value with its Monte Carlo resolution, in all such places. p = 3.0e-5 is not at the floor of 1e-5.
> 5. **Level labels have no units.** "0.5 / 1.5 / 3.0" should be 0.5B / 1.5B / 3B, KL levels "0.0 / 0.04 / 0.1" should read as beta, and the ETTh1 row "ref" should read "1.5B (ref)".
> 6. **Unrun arms look like passed checks.** Arms with no runs show "none" in the checks column; write "not checked (no runs)".
> 7. **Smaller fixes.**
>    - Show the descriptive differences with their sign ("+0.12").
>    - Write "largest mean on-policy gap … (limit 0.05)".
>    - Put the KL trend's detectable rho (0.53) next to its null.
>
> Then regenerate the outputs in the worktree and send them again.

### What changed after the reviews

* Report wording only (3c359b8, written after the AB2 values were seen; no analysis changed), as the critic and
  the standards reviewer asked: the scale trend's direction in words and its scope (three checkpoints, size not
  isolated, most of the rise between 1.5B and 3B, ILI only, ETTh1 not monotone, no pair tested, shared search
  seeds), the detectable rho beside each trend, a status sentence that states AB5's rule (a rejection holds; only
  the pending tests and the non-rejections are open), the statistic's origin in AB5, units on levels, signed
  differences, p never printed as 0.000, the hardware check named and qualified, the 3B licence, random search
  as post hoc context, and that the RevIN shares do not follow model size.
* Outputs regenerated at 2597a23 (72fb603): every number unchanged; only provenance differs.
* Not changed: llm_tests.py's computation. The only planned change to it after seeing these values is moving
  the pin RTX_REF when AB4 lands.
* My earlier account of the output diff omitted the header line, the ST source line, two power rows and a history
  bullet in docs/ablations.md (the standards reviewer's correction).

## Part 2: AB4 and the report (26 Nov evening)

Scope: scripts/ablations/llm_tests.py (then llm_rtx.py) at a692f0e, f9c0d62 and later, reading origin/soumitra-leg2 at e6926d8. The report labels this an independent computation, not the result of record (Diya's AB4, 2ab2996 on main, names scripts/ablations/llm_registered.py, run once by Soumitra).

### Verifier: blind recomputation of every AB2 and AB4 arm, and Soumitra's rtx_tests.py

> All ten registered arms are complete and pass every check, and my numbers agree with rtx_tests.py to Monte Carlo noise; Holm rejects the same tests in both. I did not open LLM.json, ST-power.json or docs/ablations.md in the main tree.
>
> **A correction:** I did not write recompute.py. The blind recomputation agent did. I imported its table, AUC, t-interval, Spearman and Holm functions unchanged. The AB4 parts are my own: the two-sample permutation (10^6 draws), the Welch interval, the step analyses and the arm checks. The AUC matches leaguetst's `run_metric` exactly (maximum difference 0.0).
>
> **Checks.** The source is e6926d8, which is the local tip. AB2's nine files match the move manifest, and the copies under results/runs/llm_local/ are byte-identical. Each arm:
> - 30/30 searches or 10/10 deployments;
> - no duplicates, no errors.jsonl, so no on-policy stops;
> - registered model revision, local backend, one job per GPU;
> - every commit is clean, descends from c2dda8d (AB2) or 15db163 (AB4), and has the registered launcher and all ten frozen E4 digests;
> - training settings as registered: steps 30/60/120, beta 0/0.04/0.1, lr 1e-4, seeds 100–109.
>
> Every DONE line is on 25 or 26 Nov; the last is steps120 at 26 Nov 18:25 ET, well before the 28 Nov deadline.
>
> | ILI held-out AUC | mine | rtx_tests.py |
> |---|---|---|
> | anon vs ref: +0.3627 [+0.0999, +0.6255] (1.9276 vs 1.5649) | p 0.0075, Holm 0.0151, rejected | p 0.0073, Holm 0.0145, rejected |
> | repair vs ref: +0.0264 [−0.2778, +0.3307] | p 0.8624, Holm 0.8624 | p 0.8614, Holm 0.8614 |
> | scale trend: rho +0.4295 (1.3911 / 1.5649 / 2.0561) | p 0.0000, Holm 0.0001, rejected | p 0.0000, Holm 0.0001, rejected |
> | steps trend: rho +0.7622 (0: 1.5649, 30: 3.0322, 60: 3.2600, 120: 3.1662) | p 0.0000, Holm 0.0000, rejected | same |
> | KL trend: rho −0.1698 (3.3953 / 3.2600 / 3.0946) | p 0.3738, Holm 0.3738 | p 0.3745, Holm 0.3745 |
> | trained at all: +1.5879 [+1.2772, +1.8985] | p 0.0000 | p 0.0000 |
> | dose: mean per-policy rho +0.0500 | exact sign-flip p 1.0000 | 1.0000 |
>
> The p differences are about one Monte Carlo SE. On steps and "trained at all" both are at the floor of their permutation count (mine 10^6, theirs 10^5). ETTh1 descriptive means are identical in both: anon 1.3701, repair 1.4299.
>
> The steps-trend rejection comes from level 0 (untrained) against the trained levels. More steps change nothing: the dose analysis is flat. That is exactly AB4's caveat.
>
> **Issues for you:**
> 1. **Two AB4 amendments.** Diya's AB4 on main (2ab2996, committed 26 Nov 17:21 ET) says "before any run". It registers run_wp3_rtx_queue.sh and an 18:00 ET stop. The arms actually ran 01:32 to 18:25 ET on 26 Nov, through Soumitra's run_ab4_queue.sh under his AB4 (15db163), which is not on main. Diya or the owner must decide which registration governs before anything is called the result of record.
> 2. rtx_tests.py cites "AB4 addendum 2" for the scale statistic. That addendum is on neither e6926d8 nor origin/main as fetched.
> 3. rtx_tests.py records only the AB2 training files in its provenance, and its output shows no check of the AB4 steps arms' training settings. Mine checked them and they pass.
> 4. e6926d8 adds scripts/ablations/llm_tests.py (the 5abbd20 version). That will be an add/add conflict with your extended copy when merged. common.py is identical, so no conflict there.
>
> Throwaway worktree, left in place: [scratch]/direction/ab4/rtx_wt (remove later with `git worktree remove --force` on that path).
>
> Files are in [scratch]/direction/ab4/:
> - recompute_ab4.py
> - recompute_ab4.json
> - RTX_tests_e6926d8.json
>
> Ready for your output path.

### Verifier: three-way comparison, the f9c0d62 fix and the reading rule

> **VERDICT: PASS on the numbers, the fix and the reading rule. Do not push yet (point 4).**
>
> 1. **Three-way comparison: 113 of 113 values agree.** I compared your LLM.json "registered" block, my recompute_ab4.json and RTX_tests_e6926d8.json to 4 decimals. That covers:
>  - every difference, Welch interval, level mean, interval and n;
>  - every rho, p, Holm p and decision;
>  - (i): +1.5879 [+1.2772, +1.8985], p 0.0000;
>  - (ii): mean rho +0.0500, sign-flip p 1.0000, identical rhos;
>  - completeness of all ten arms;
>  - the hardware check (+0.1171 [−0.7005, +0.9347]), both AB5 comparisons, and the ETTh1 means.
>
>  Your p values equal rtx_tests.py's exactly, because both use the same functions and seeds. Mine differ only by Monte Carlo noise, at most 0.0010. Holm decisions: AB-P rejects anon (0.0145) and the scale trend (0.0001), and not repair (0.8614). AB-G rejects the steps trend (0.0000), and not KL (0.3745).
>
> 2. **The f9c0d62 fix is sound and changes only where the logs are read.**
>  - `item_rows` reads rtx/ or results/runs/, and stops if both copies exist and differ. That is a new guard.
>  - The rtx copies are byte-identical to results/runs/llm_local/ (I checked).
>  - The statistics computed in the first run are unchanged.
>
>  But "no statistic changed" is not quite true. The fix made the steps trend computable, and that moved the KL Holm p from 0.7490 to 0.3745. Correct the ISSUES line.
>
> 3. **The wording follows the reading rule in Diya's AB4.** (ii) has the trend's sign but p 1.0000, so the result is "not a dose-response"; (i) differs, so "trained policies differ from the untrained model". The caveats are present. Fixes:
>  - The summary bullet says "(i), p at most 0.05" and gives no direction. Write "trained policies are worse by +1.59 SD [+1.28, +1.90], p < 0.0001".
>  - "AB4" now means two amendments. Name 15db163 (soumitra-leg2, under which the runs were made) and 2ab2996 (main, which holds the reading rule and the record) wherever each applies.
>  - The figure "0.10 to 0.15" is typed into report.py. Cite its source, main's docs/ISSUES.md, with its condition (within-policy correlation 0.5 to 1.0).
>  - Optional: keep anon and repair out of the ETTh1 scale-order sentence.
>
> 4. **Process blocker for pushing.**
>  - LEG3 (main, 26 Nov 17:21 ET) asks Anirudh not to read results/ablations/rtx/ and not to edit common.py or llm_tests.py until LLM-registered.json is committed.
>  - Diya's AB4 says no AB-P or AB-G value is computed before then. That file is on no branch yet.
>  - Your llm_tests.py (466fb5ec) differs from main's frozen 08c72a4c, so merging this branch into main before Soumitra's run would make llm_registered.py refuse.
>  - Keep these commits unpushed until LLM-registered.json is committed, or until Diya agrees.
>  - The report discloses the computation; also tell Diya and Soumitra directly.
>
> 5. **Disclosure.** I computed the AB4 values at about 19:20 ET on 26 Nov, at your request. I first read LEG3 during this check.
>
> Comparison table: [scratch]/direction/ab4/compare3.txt

### Area-chair review of the ablations (NeurIPS/ICML/ICLR standard), round 1

> # Area-chair review: LeagueTST ablations (anirudh-ablations) and the AB4 governance
>
> **Snapshot.** This review covers anirudh-ablations at 943d893 (the llm_tests.py to llm_rtx.py move, committed during the review) and the working-tree docs/ablations.md regenerated at 19:44 ET on 26 Nov. It also covers origin/soumitra-leg2 at e6926d8 and origin/main at 2ab2996.
>
> **Scope.** After the coordinator narrowed the brief, Soumitra's work is reviewed only where the ablation report depends on it: the AB2 and AB4 RTX run files, rtx_tests.py and e6926d8.
>
> **How code is cited.** Code is cited by function name, with line numbers at 943d893.
>
> **What was not opened.** I did not open results/revalidation, results/baselines, data/ETT-small or data/illness, or anything else that touches the test split.
>
> ## A. Verdict
>
> As written, the ablation section would not survive main-track review, although most of what is needed is disclosure and wording rather than new experiments.
>
> **Strengths.** The CPU ablations are carefully pre-registered, sized by power, labelled exploratory where post hoc, and honest about nulls: AB-C, AB-X, AB-Q, AB-N, AB-R, RQ3 and ST.
>
> **Supported claims.**
> - On ILI, GRPO policies trained on the ETTh1 table are much worse than the untrained model:
>   - RTX: +1.59 SD [+1.28, +1.90];
>   - Mac: +1.24 SD;
>   - E4: +1.33 SD.
> - Those policies are also worse than warm-started TPE: +1.50 SD, Holm-rejected in E4 family 1.
> - Step count shows no dose-response: 3.03, 3.26 and 3.17 SD at 30, 60 and 120 steps; per-policy sign-flip p 1.000.
> - The KL weight shows no detectable effect, but the test is underpowered (detectable |rho| about 0.53).
> - No league effect is detectable at a resolution of about 1.1 to 1.25 SD.
> - No external HPO baseline beats random search at a resolution of about 0.40 to 0.49 SD.
>
> **Unsupported claims.**
> - **The registered step-trend "rejection".** Its p is anti-conservative, and it is almost entirely the contrast between level 0 and the trained levels.
> - **"A larger model gave a worse AUC" as a statement about scale.** The test uses three checkpoints and the order reverses on ETTh1.
> - **"The LLM uses forecasting knowledge" (the anonymised arm).** Anonymising also removes the field descriptions and changes the decoding trie. The sign reverses on ETTh1. The rejection does not survive a Holm correction over the whole ablation programme.
> - **"GRPO learned on ETTh1".** The registered AB-L test is shown to be too liberal, so only an exploratory test supports this.
>
> **Governance problems.** These are worse than the statistics:
> - two different amendments are both called "AB4";
> - the "computation of record" does not exist on any branch;
> - three computations pin mutually incompatible versions of llm_tests.py;
> - the report labels its own numbers "final" while saying they are not of record.
>
> The paper may claim a well-powered negative result for RL fine-tuning in the transfer setting on ILI, and nothing positive.
>
> ## B. Findings, most severe first
>
> ### B1. Two conflicting "AB4" amendments; Diya's is dated after the runs it says do not exist
>
> **Where.**
> - origin/soumitra-leg2 15db163: docs/prereg_ablations.md, "## Amendment AB4", 01:32 ET, 26 Nov.
> - origin/main 2ab2996: docs/prereg_ablations.md, "## Amendment AB4 (… Diya's side, before any run …)", 17:21 ET.
> - results/logs/ab4_queue_status.txt on leg2.
>
> **Problem.** Soumitra launched the four AB4 arms at 01:32 ET with his own queue, run_ab4_queue.sh (sha 2b9f3d98). The queue status shows:
>
> | Arm | DONE (26 Nov, ET) |
> |---|---|
> | anon, ILI | 02:09 |
> | repair, ILI | 04:01 |
> | anon, ETTh1 | 05:36 |
> | repair, ETTh1 | 07:19 |
> | steps30 | 11:29 |
> | steps120 | 18:25 |
>
> Diya's AB4 was committed at 17:21. It says "No run of ABL_P_anon_local, ABL_P_repair_local, ABL_G_steps30_local or ABL_G_steps120_local exists in git". That was false in fact, although the runs had not reached origin: this clone's 16:44 fetch shows leg2 at 7fd5085.
>
> Diya's AB4 also requires several things the actual runs did not meet:
> - a different queue, run_wp3_rtx_queue.sh (dc08a6…);
> - an 18:00 ET stop instead of 23:59;
> - a status file named wp3_rtx_queue_status.txt;
> - "Soumitra confirms in his launch commit that results/runs/llm_local/ holds none of them". The launch had already happened, and e6926d8 now commits all 20 records under results/runs/llm_local/.
>
> The two texts also disagree on substance:
>
> | Element | Soumitra's AB4 (15db163) | Diya's AB4 (2ab2996) |
> |---|---|---|
> | Policy-level analyses | "exploratory only … No other analysis is added after results are seen" | "Registered secondary analyses (i) to (iii)", plus a reading rule |
> | Computation | spearman_trend in llm_tests.py at 5abbd20 (01b70835) | llm_registered.py, pinning llm_tests.py 08c72a4c |
> | AB2 manifest | AB2_move.sha256 | AB2_records.sha256 |
>
> A merged docs/prereg_ablations.md will carry two sections with the same heading.
>
> **Why a reviewer objects.** Pre-registration is credible only if one dated, unambiguous protocol precedes the data. Two protocols under one name, one written after the data existed, is a forking path: either one can be cited after the fact. NeurIPS checklist item 7 (statistical significance) asks how the tests were chosen, and item 2 asks for limitations.
>
> **Fix.**
> - **(O) Diya.** Issue a dated amendment that:
>   - renames the two sections, for example AB4-S and AB4-D;
>   - states which one governs each element: queue, deadline, the status of analyses (i) to (iii), the reading rule, and the computation of record;
>   - corrects the "What existed" paragraph with the queue-status times above.
> - **(W) Anirudh.** Add this timeline to the RTX section. Stop citing "AB4" without saying which one: the report currently takes the arm registration and the 23:59 deadline from 15db163 (llm_rtx.py, `REGISTERED_AT`, `DEADLINE`) and the reading rule and "registered secondary" label from 2ab2996.
>
> **Class:** O, with W for the report.
>
> ### B2. The computation of record does not exist, and the three computations cannot coexist in one tree
>
> **Where.**
> - docs/ablations.md, the at-a-glance RTX bullet (line 11): "(final; the result of record is results/ablations/LLM-registered.json …)".
> - report.py, `RECORD_NOTE` (line 245).
> - llm_registered.py, `DIGESTS`, on origin/main.
> - rtx_tests.py, `PINNED`, on leg2.
>
> **Problem.**
> - llm_registered.py exists only on main. leg2 does not descend from 2ab2996, so Soumitra's branch cannot run it. It has never been run, and LLM-registered.json exists on no ref.
> - rtx_tests.py pins llm_tests.py at 01b70835. llm_registered.py pins it at 08c72a4c.
> - The merge base of main and leg2 (d8582c0) has no llm_tests.py, so a merge is an add/add conflict. Whichever version is kept, one of the two "registered" scripts refuses to run.
> - The 943d893 move fixes Anirudh's side: llm_tests.py now hashes to 08c72a4c, which I verified. It does not resolve the conflict between rtx_tests.py and llm_registered.py.
>
> **Why a reviewer objects.** The report points readers to a result file that does not exist, and it states rejections as settled. Checklist item 4 (reproducibility) asks whether the stated pipeline reproduces the numbers.
>
> **Fix.**
> - (W) Replace "final; the result of record is LLM-registered.json" with: "not yet computed: LLM-registered.json does not exist at 2ab2996 or e6926d8; the numbers here are an independent computation of the same tests, complete for 15 of 15 arms".
> - (O) Diya decides which script is of record and resolves the pins. One option is to amend rtx_tests.py's PINNED to 08c72a4c, since spearman_trend is identical in both files (AB5 records its source digest, e344236d).
>
> **Class:** W and O.
>
> ### B3. "Status: final" and the deadline line misstate the status
>
> **Where.**
> - report.py, `registered_llm`, line 273: "Status: final by AB5's rule, for this computation." Line 279: "Deadline for every arm: 28 Nov 23:59 ET (AB1)".
> - llm_rtx.py, `registered_source`, line 445: `final = pinned_is_tip and (…)`.
>
> **Problem.**
> - "Final" here means only that the pin equals the local tip as of the 19:04 fetch, two days before the deadline. Soumitra can still push.
> - docs/LEG3_soumitra.md ("For Anirudh") says to report these families "as pending, not as not run" until LLM-registered.json is pushed.
> - Diya's AB4 moves the AB4 arms' deadline to 18:00 ET.
>
> **Fix (W).**
> - Label the status "complete (15 of 15 arms), independent computation, not of record, not yet verified (B10)".
> - Give both deadlines, each with its source.
>
> No statistic changes.
>
> **Class:** W.
>
> ### B4. The step-trend headline invites a dose-response reading the data refute; analyses (i) and (ii) are mislabelled
>
> **Where.**
> - docs/ablations.md, line 11, and the RTX AB-G table (line 160).
> - report.py, `step_reading` (line 254), `glance` (lines 589 to 600), and `registered_llm` (line 334).
> - llm_rtx.py, `registered_families`, line 358 (key `exploratory_policy_level`).
>
> **Problem.**
> - The bullet reads: "steps trend … rho +0.76, Holm p < 0.0001, rejected … trained policies differ from the untrained model ((i), p at most 0.05) (trained minus untrained +1.59 SD)". It never says "worse", omits the actual p (below 0.0001), and does not say that the policies were trained on ETTh1 and deployed on ILI.
> - Level 0 (30 untrained searches, 1.56) against the trained levels (about 3.1) drives the rho. Across the trained levels the per-policy rho is +0.05, with sign-flip p 1.000.
> - Analyses (i) and (ii) are "registered secondary" in Diya's AB4 and "exploratory only" in Soumitra's. The JSON key calls them exploratory while the document calls them registered.
> - Diya's rule says only "if (i) shows a difference". The threshold p ≤ 0.05 in `step_reading` is Anirudh's operationalisation.
>
> **Why a reviewer objects.** Headline wording must match the valid inference. A rejected trend whose p is known to be anti-conservative should not lead.
>
> **Fix (W).** Lead with: "No dose-response across 30, 60 and 120 steps (3.03, 3.26, 3.17; per-policy sign-flip p 1.00). Policies trained on ETTh1 are +1.59 SD [+1.28, +1.90] worse on ILI than the untrained model (p < 0.0001). The registered step trend (rho +0.76) is reported as registered, with a p that is anti-conservative because the levels are nested." Label (i) and (ii) as registered by one AB4 and exploratory by the other, and disclose the p ≤ 0.05 operationalisation.
>
> **Class:** W.
>
> ### B5. The scale and anon results are specific to ILI and reverse on ETTh1; the summary omits this
>
> **Where.**
> - docs/ablations.md, line 11 (summary), line 162 (scope paragraph) and the RTX ETTh1 table (lines 182 to 188).
> - leaguetst/architect/prompt.py: `ANONYMOUS` (line 33) and `DESCRIPTIONS` (dropped at line 48).
> - leaguetst/architect/constrained.py, `canonical_text`.
>
> **Problem.**
>
> | Arm | ILI (registered) | ETTh1 (descriptive), RTX | ETTh1, Mac |
> |---|---|---|---|
> | 3B | worst, 2.06 | best, 1.32 (1.5B 1.47) | |
> | anon vs ref | +0.36 SD worse, rejected | better, 1.37 vs 1.47 | −0.13 |
>
> - The summary says only "ILI only". The Mac bullet does mention the ETTh1 reversal; the RTX bullet does not.
> - The anon arm removes the field names and the field descriptions. It also changes the canonical JSON keys to field_a … field_h, so the token paths of the constrained-decoding trie change. The test therefore confounds "prior knowledge" with prompt content and tokenisation.
>
> **Fix (W).**
> - Put the ETTh1 order in the summary sentence.
> - Define anon precisely.
> - Scope both results as "on ILI, for these three Qwen2.5 checkpoints and this prompt".
>
> No statistic changes.
>
> **Class:** W.
>
> ### B6. Descriptive diagnostics registered in AB5 are computed but not shown; the repair confound is stated only for the Mac
>
> **Where.**
> - results/ablations/LLM-rtx.json, `registered.diagnostics_ILI`. report.py never renders it.
> - AB5, bullet "Descriptive only … search diagnostics (lookups, repeat, repaired and invalid rates)".
>
> **Problem.** The JSON holds the following ILI search diagnostics, none of which reach docs/ablations.md:
>
> | Arm | Repeat rate | Other |
> |---|---|---|
> | 0.5B | 0.52 | |
> | 3B | 0.28 | |
> | ref | 0.175 | |
> | anon | 0.13 | |
> | repair | 0.065 | repaired 0.53, invalid 0.022 |
> | steps30 | 0.67 | |
>
> - On the RTX, 53% of the repair arm's proposals are replaced, exactly the confound the report spells out for the Mac (line 215) but not for the registered arm.
> - The best prompted arm, 0.5B, wastes half its proposals, which a reader needs to interpret the scale trend.
> - The registered endpoint includes raw MSE (Shared rules: "in pooled seed-SD units and in raw MSE"). No RTX or Mac LLM table gives it.
>
> **Fix (W).** Render `diagnostics_ILI` and a raw-MSE column, both from the JSON, and state the repair confound for the RTX arm.
>
> **Class:** W.
>
> ### B7. Diya's AB4 report disclosure sentence is missing
>
> **Where.** Diya's AB4 on main, bullet "Report disclosure, fixed now: 'The RTX AB-P and AB-G arms were registered on 26 Nov (AB4), after one author had analysed exploratory Mac runs …'". `grep -c "after one author had analysed" docs/ablations.md` returns 0.
>
> **Problem.** A registered disclosure is missing from the report it was written for.
>
> **Fix (W).** Quote it verbatim in the RTX section, beside the B1 timeline, which makes the sentence's "registered on 26 Nov" more precise.
>
> **Class:** W.
>
> ### B8. The record note contains a false sentence and an undisclosed deviation
>
> **Where.** report.py, `RECORD_NOTE` (lines 245 to 251).
>
> **Problem.**
> - "This branch does not change llm_registered.py or its inputs" is false for the branch history. llm_tests.py is a digest-pinned input of llm_registered.py (`DIGESTS`), and this branch changed it in 510633d, a692f0e and f9c0d62, until 943d893 restored the 08c72a4c bytes.
> - docs/LEG3_soumitra.md also says "do not edit common.py or llm_tests.py until … LLM-registered.json is committed". a692f0e (19:32) and f9c0d62 (19:35) edited llm_tests.py after this clone fetched LEG3 at 19:04. The note discloses reading results/ablations/rtx/ but not these edits.
> - "Before that file reached this clone" has an ambiguous antecedent: it could be LEG3 or LLM-registered.json.
>
> **Fix (W).** Replace the sentence with: "llm_tests.py, which llm_registered.py pins, was changed on this branch between 510633d and 523ae42, against LEG3's request, and restored byte-identical to main's 08c72a4c in 943d893". Name LEG3 explicitly.
>
> **Class:** W.
>
> ### B9. A rerun of the RTX computation is not disclosed in the report
>
> **Where.**
> - results/ablations/LLM-first-AB4-run.json.
> - The f9c0d62 fix in llm_rtx.py, `item_rows`.
> - docs/ablations.md, "Runs, re-runs and open questions" (lines 618 to 625).
>
> **Problem.** The first AB4 computation counted the step arms as missing: it read training logs only from results/runs/. In that output the KL trend's Holm p is 0.749; it is 0.3745 now, and no decision changed. The bug was found after the AB4 prompted values were seen. The commit message says this; the report does not. Diya's AB4 verification bullet requires that a post-run fix keeps both outputs and that the report discloses it.
>
> **Fix (W).** Add one line to "Runs, re-runs": both outputs, what changed, no decision changed.
>
> **Class:** W.
>
> ### B10. The registered RTX numbers are unverified on this branch
>
> **Where.**
> - docs/verdicts/ on anirudh-ablations holds no RTX review.
> - The AB2 verifier, critic and standards reviews (32 of 32 values match) exist only on local/ab2-provisional-do-not-push d577ef8 (docs/verdicts/ablations_rtx_reviews.md).
> - The AB4 values have no re-score at all: anon, repair, steps trend, (i), (ii).
>
> **Problem.** CLAUDE.md rule 6 ("the verifier's independent re-score") and the loop ("Copy the critic's final verdict verbatim") are not met for numbers the summary calls rejected and final.
>
> **Fix (W, now).** Label the RTX section "AB4 values not yet verified; AB2 verified on an unpushed branch (d577ef8)". The verifier and critic pass itself is a process step for Anirudh's agents: it is a re-score, not a new analysis.
>
> **Class:** W.
>
> ### B11. Rejections are presented without the "secondary, not a claim" scope
>
> **Where.** docs/ablations.md, line 3 says "secondary to the primary endpoints", but lines 11, 151, 153 and 160 say "rejected". Soumitra's rtx_tests.py has a `scope` string ("Secondary tests, not claims under CLAUDE.md rule 6: … no paired block-bootstrap interval"). Anirudh's report has no equivalent.
>
> **Problem.** CLAUDE.md rule 6 requires a paired block-bootstrap interval and a verifier re-score for any claim. G4 covers only E3 and E4. The RTX families use Welch t intervals and unpaired permutation p, so a reader may take "rejected" for a paper claim.
>
> **Fix (W).** Add the scope sentence to the RTX and Mac sections and to the summary.
>
> **Class:** W.
>
> ### B12. Multiplicity scope: the anon rejection holds only within its family
>
> **Where.** The registered ablation tests:
>
> | Family | Tests |
> |---|---|
> | AB-C | 3 |
> | AB-X | 3 |
> | AB-P | 3 |
> | AB-G | 2 |
> | AB-L | 2 |
>
> That is 13 tests in all.
>
> **Problem.** Holm within each family is registered and valid. Under Holm over all 13 registered ablation tests, the ranked p-values give:
> - the step trend survives: 13 × 1e-5;
> - the scale trend survives: 12 × 3e-5;
> - anon does not: 11 × 0.0073 = 0.080 > 0.05;
> - AB-L ETTh1 (raw p 0.016) is also not rejected.
>
> All the AB-P and AB-G tests reuse the same 30 ABL_P_ref_local values: anon, repair, the 1.5B scale level, and steps level 0. Their errors are therefore correlated across families.
>
> **Fix (X).** Report the programme-wide Holm as an exploratory sensitivity, labelled post hoc, and note the shared reference. The registered decisions stand.
>
> **Class:** X.
>
> ### B13. The AB-L summary asserts learning on the strength of an invalid registered test
>
> **Where.** docs/ablations.md, line 14: "On the ETTh1 source GRPO raised the league policies' mean reward …".
>
> **Problem.** The report's own calibration shows the registered test's false-positive rate is 0.24 to 0.36 at nominal 0.025. The only valid support is the post hoc position-adjusted sign flip, p 0.014 (Holm 0.027). The opening clause states the conclusion as fact.
>
> **Fix (W).** Start with "The registered AB-L rejection (ETTh1, Holm p 0.031) rests on a test that is too liberal; an exploratory position-adjusted test agrees (p 0.014)".
>
> **Class:** W.
>
> ### B14. A number typed by hand in docs/
>
> **Where.** report.py, `registered_llm`, line 324: "a false-positive rate of 0.10 to 0.15 at nominal 0.05", quoting main's docs/ISSUES.md.
>
> **Problem.** Rule 5 and the brief: every number in docs/ comes from scripts via JSON. This one is copied from someone else's critic, and no committed simulation backs it.
>
> **Fix.** Either (W) drop the figure and cite the source in words, or (X) compute it with a committed simulation, labelled exploratory.
>
> **Class:** W or X.
>
> ### B15. AB5's "What existed" is wrong in fact
>
> **Where.** docs/prereg_ablations.md, AB5, first paragraph: "AB4 registered (15db163), not yet run". Committed at 17:12 ET.
>
> **Problem.** By the queue status, anon, repair and steps30 had all finished by 11:29 ET. Anirudh's clone could not see this: its fetch at 16:44 had leg2 at 7fd5085.
>
> **Fix (W).** Add a dated correction in "Runs, re-runs". Do not edit AB5.
>
> **Class:** W.
>
> ### B16. An auto-generated RevIN sentence invites a wrong inference
>
> **Where.** report.py, `registered_llm`, line 356: "Among the prompted models the share does not follow model size, so it does not follow the scale trend's order."
>
> **Problem.** The RevIN-off shares are 41%, 55% and 52% for 0.5B, 1.5B and 3B. The best arm, 0.5B (AUC 1.39), has the lowest share, which is consistent with the RevIN account. The monotonicity test hides that.
>
> **Fix (W).** Print the shares beside the AUCs and drop the inference.
>
> **Class:** W.
>
> ### B17. The checklist is stale on the points a reviewer reads first
>
> **Where.** docs/neurips_checklist.md:
> - item 7 (line 15): "provisional until AB4 lands";
> - open items, line 32: "A1 to A30, AB1 to AB3";
> - open items, line 37: "AB-P and AB-G ran on an Apple M4 Pro … not on the registered … backend".
>
> **Problem.** The registered AB-P and AB-G now ran on the RTX. The deviation list lacks both AB4s, AB5, the AB3 addenda and the B9 rerun. Item 8 (total compute) and item 12 (software and data licences) are still open.
>
> **Fix (W).** Update all three from the current state. See also section D.
>
> **Class:** W.
>
> ### B18. The detectable effects are optimistic for 10-seed arms
>
> **Where.** common.py, `detectable` (normal quantiles). power.py, `e4_rows` and `trend_rows`. docs/ablations.md, line 8: "E4's ILI nulls … could detect 1.10 to 1.25 SD".
>
> **Problem.** At Holm size 6 with df about 10 to 12, the t-based multiplier is 4.03 to 4.16 against 3.48 for the normal: 16 to 19% larger. The resolution of E4's nulls is therefore nearer 1.3 to 1.5 SD. AB3 registered the normal approximation, so it stays. The Fisher-z rho for tied levels is already flagged.
>
> **Fix (W).** Add "optimistic by about 15 to 20% at n = 10" to the ST note.
>
> **Class:** W.
>
> ### B19. The run-path code of the prompted arms is not checked
>
> **Where.** llm_rtx.py: `registered_source` (line 418, `check_runs(..., frozen=False)`) and `check_hashes`.
>
> **Problem.** For prompted arms only ancestry and the launcher digest are checked, not that leaguetst/, scripts/llm_local.py and requirements-llm.txt equal those at c2dda8d. Diya's AB4 makes that a launch condition. I checked by hand: `git diff --stat c2dda8d 15db163` and `c2dda8d 7fd5085` over those paths are empty. So no result is affected. Two commits ran anon on ETTh1 because of the 05:12 laptop shutdown.
>
> **Fix.** Either (C) add the tree-hash check with a test, or (W) state the verified identity.
>
> **Class:** C, minor.
>
> ### B20. Soumitra's rtx_tests.py and e6926d8, as they bear on the report
>
> **Where.** origin/soumitra-leg2 e6926d8 ("Results"): scripts/ablations/rtx_tests.py and results/runs/llm_local/.
>
> **Problem.**
> - **It claims to be of record.** The docstring says it computes "the registered tests, computed by the items' owner", which contradicts Diya's AB4.
> - **It cites an amendment that does not exist.** `statistic_note` cites "AB4 addendum 2", which is not on leg2; there is only addendum 1. On leg2 the scale-trend statistic is unregistered; it is fixed in Anirudh's AB5 and in Diya's AB4.
> - **It checks less than the other two scripts.** It has none of llm_rtx.py's and llm_registered.py's checks: registration order, card, revision, frozen files, training settings.
> - **Its output is not committed.** RTX_tests.json is not in the repository.
> - **e6926d8 adds stray files.** It commits a stale results/logs/l4_queue.lock and second copies of all 20 run and training files under results/runs/llm_local/. I checked that they are byte-identical to results/ablations/rtx/ and results/runs/. This contradicts Diya's AB4 attestation and adds a path that scripts globbing results/runs/ could pool.
> - **Reported in the right place.** Anirudh's report correctly ignores the llm_local copies: `item_rows` reads results/runs/<item>/ and rtx/ only.
>
> **Fix (O, Soumitra).**
> - Retitle rtx_tests.py as a verifier, or drop it.
> - Correct the amendment citation.
> - Remove or explain the duplicates and the lock in a follow-up commit, without rewriting history.
>
> **Class:** O.
>
> ### B21. llm_registered.py's AB4 check cannot tell the two amendments apart
>
> **Where.** origin/main llm_registered.py, the commit check (lines 84 to 92): `if ab4 and "## Amendment AB4" not in … prereg_ablations.md`.
>
> **Problem.** Soumitra's 15db163 and 7fd5085 contain his own "## Amendment AB4" heading, so the check admits runs that were launched before Diya's AB4, by a different queue, under a different deadline.
>
> **Fix (O, Diya).** Check ancestry of a specific commit, or a digest of the amendment text, or state in the B1 amendment that 15db163 is accepted.
>
> **Class:** O.
>
> ### B22. The RTX section has no source line
>
> **Where.** docs/ablations.md, RTX section (lines 126 to 190).
>
> **Problem.** Every other section ends its intro with "Source: <script> at <hash> (script sha256 …)". The RTX section names LLM-rtx.json but not llm_rtx.py's hash or digest (b385101d at 943d893).
>
> **Fix (W).** Add the source line from the JSON provenance.
>
> **Class:** W.
>
> ### B23. GPU nondeterminism is not measured
>
> **Where.** The RTX section (line 164) relies on the steps levels being prefixes of each other "up to GPU nondeterminism". AB4's diagnostic (iii) is "not computed here".
>
> **Problem.** No same-seed rerun exists on the RTX. The hardware checks, with half-widths up to 0.82 SD, cannot bound nondeterminism.
>
> **Fix.** Either (O) diagnostic (iii) in llm_registered.py covers this, or (X) a same-seed rerun of one 30-step policy, labelled exploratory.
>
> **Class:** O or X.
>
> ## C. Headline claims check (RQ1, from what the ablations show)
>
> **Endpoint.** ILI held-out AUC in pooled seed-SD units; lower is better.
>
> **Sources.**
> - E3 arms: 30 seeds each; prompted Qwen has 10.
> - E4: 10 GRPO policies per arm, deployed ETTh1 to ILI (results/independent/E4_soumitra.json).
> - RTX: docs/ablations.md.
>
> | RQ1 baseline | Baseline AUC | GRPO + league, ETTh1 to ILI | Verdict |
> |---|---|---|---|
> | Random search | 1.52 [1.31, 1.72] | 3.14 [2.50, 3.79] (RTX 60-step 3.26) | Not beaten; about 1.6 SD worse (descriptive) |
> | Regularized evolution | 1.47 | 3.14 | Not beaten |
> | TPE | 1.14 | 3.14 | Not beaten |
> | TPE warm-started from ETTh1 (A13) | 1.65 | 3.14 | Worse, +1.50 SD, Holm p 6e-5 (E4 family 1, rejected) |
> | ASHA | 1.38 | 3.14 | Not beaten |
> | Same LLM without RL | E3 A10 1.81 (10); RTX ref 1.56 (30) | 3.14 / 3.15 | Worse: +1.33 SD (E4, descriptive); +1.59 [+1.28, +1.90], p < 1e-4 (RTX, registered by one AB4, exploratory by the other); +1.24 (Mac, exploratory) |
> | Portfolio from ETTh1 | 3.28 | 3.14 | −0.14, not rejected (detectable about 1.10 SD) |
> | No league | 3.23 | 3.14 | −0.09, not rejected (detectable about 1.25 SD); "competitive" cannot be claimed (uniform arm not run, A30) |
>
> Further results:
> - **Prompted Qwen against the CPU baselines.** Against random: +0.29 on AUC, and +0.67 at budget 25 (p 0.049, unadjusted). Against TPE: +0.67 on AUC. It beats neither.
> - **ILI to ETTh1.** Family 1 and family 2 at budgets 25 and 50 are pending (pick batch 2), with p = 1 placeholders.
> - **Gate G4.** docs/claims/E4.json does not exist, so G4 cannot pass. Under CLAUDE.md no method "wins".
> - **Equal epochs.** The comparison is equal-epoch on the target only. Source-table GRPO training is not charged (A12), so "equal compute" may not be claimed (A11).
>
> **What may honestly be claimed.**
> - On ILI, the GRPO-trained, league-trained LLM architect, transferred from ETTh1, does not beat random search, regularized evolution, TPE, ASHA or the untrained LLM at equal target epochs. It is significantly worse than the untrained LLM and than warm-started TPE.
> - No league effect, no step dose-response and no KL effect is detectable at the stated resolutions.
> - The ablations are consistent with a RevIN-off preference learned on ETTh1 that does not transfer: 96 to 100% of the trained policies' ILI lookups have RevIN off, against 55% for the untrained model. That mechanism is untested and must be worded as such.
>
> **What may not be claimed.**
> - that GRPO or the league is harmful in general (there is no within-table GRPO arm, and ILI to ETTh1 is pending);
> - that larger LLMs search worse;
> - that the LLM uses forecasting knowledge;
> - that GRPO "learned" (only an exploratory test supports it);
> - any ETTh1 result beyond description.
>
> ## D. Reproducibility gaps (NeurIPS checklist)
>
> **Code (item 4).**
> - At review time docs/ablations.md, results/ablations/LLM.json and ST-power.json were modified and uncommitted. LLM-rtx.json and results/logs/mps_queue.status were untracked. Nothing in docs/ reproduces from a commit until these are committed together.
> - The computation of record (llm_registered.py) has not been run, and three scripts compete (B2).
> - docs/REPRODUCE.md is pending.
>
> **Seeds (item 4).**
> - Fixed and recorded: search 0 to 29, training 100 to 109, deployment 0 to 9; permutations 100,000 with seed 0, plus exact sign flips.
> - Not reported:
>   - the Monte Carlo error of p near the 1e-5 floor;
>   - GPU nondeterminism (B23);
>   - the seed-to-policy mapping for deployment, which rests on runner.py (`aligned` checks only deployment seeds).
>
> **Compute (item 8).**
> - Seconds are recorded per search and per policy (RTX prompted: 49 s at 1.5B, 100 s at 3B, 199 s for repair; 60-step policies 17 to 21 min), but nothing is summed.
> - Missing totals:
>   - RTX GPU-hours;
>   - Mac hours;
>   - Modal dollars across E4, including the 34 out-of-memory policies of 23 Nov;
>   - the cost of the failed and rerun computations.
> - Fix (W): a script that sums the recorded seconds from the JSON.
>
> **Licences (item 12).**
> - Qwen2.5-0.5B and 1.5B are Apache 2.0.
> - 3B is under the Qwen Research License: non-commercial, and "Built with Qwen" attribution is required for derivatives. This is noted.
> - Still missing: ETDataset (recorded as CC BY-ND 4.0, not verified on its page), CDC ILI terms, PatchTST and DLinear code, SMAC3, DEHB, ConfigSpace, Optuna, scikit-learn, transformers and peft.
>
> **Statistical reporting (item 7).**
> - Intervals are Welch t at 95%, unadjusted, beside Holm-adjusted p. Say so in every table header; the Mac AB-P bullet does, the RTX tables do not.
> - The ablations use unpaired permutation tests despite shared seeds. That is conservative for the scale and KL trends and anti-conservative for the nested steps levels, and is disclosed.
> - No paired block-bootstrap is used: say "not claims under rule 6" (B11).
> - Detectable effects use the normal approximation and are optimistic at n = 10 (B18).
> - The multiplicity scope is per family (B12).
> - Raw MSE is missing for the LLM families (B6).
> - Verification: the AB4 values are unverified, and the AB2 reviews are on an unpushed branch (B10).

### Results review (Soumitra's results and the required baselines), round 1

> # LeagueTST results review: E4, the baselines and the RTX arms (NeurIPS/ICML main-track standard)
>
> Reviewer scope: the results, not the ablation report. Everything was read through git at origin/main 2ab2996 and origin/soumitra-leg2 e6926d8, as fetched on 26 Nov 19:04 ET. I did not open results/revalidation, results/baselines, data/ETT-small, data/illness or any test row. I used only the E1 rows of the ledger.
>
> How the numbers were made. The numbers in B, and every number marked "reviewer-computed", were recomputed in memory with `leaguetst.search.replay`, which is the same code as scripts/replay.py. Intervals are the replay.py bootstrap (4,000 resamples, seed 0) over search seeds. p-values come from two-sided two-sample permutation tests on the mean (100,000 splits, or exact). Only tests marked "registered" are registered. All other p-values are exploratory and unadjusted. My E4 and E3 means match results/independent/E4_soumitra.json and the E3 summary.md to three decimals.
>
> ## A. Verdict
>
> The E4 primary fails in the direction that has been tested. On ILI held-out AUC, the frozen ETTh1-trained league policy scores 3.14 [2.60, 3.64] pooled seed SD. That ties the ETTh1 portfolio (3.28; registered p 0.64) and is significantly worse than warm-started TPE (1.65; registered two-sided p about 1e-5, Holm 6e-5; the rejection is a loss). A transfer win is therefore impossible. The league shows no effect against no league (3.14 vs 3.23, exact p 0.80, detectable difference about 1.1 SD), so "competitive" is excluded by the rule fixed in advance.
>
> Descriptively, every GRPO arm on every card scores 3.0 to 3.4. That is 1.3 to 2.1 SD worse than random search, RE, TPE, ASHA, the referee, the PG controller and prompted Qwen. The failure has an obvious mechanism: about 99% of the GRPO policies' ILI lookups have instance norm off, and none of the 59 configs within 1 SD of ILI's best has it off. No LLM arm beats any required baseline after correction, except that the prompted arms beat the ETTh1 portfolio, which they never use.
>
> The independent re-score really agrees with scripts/replay.py: 35 statistics within 1e-9, and the one Monte Carlo p differs within MC error. But its coverage is narrow. It covers only full-run methods, shares every input with replay.py, and records the wrong git_head. The registered families are also incomplete. Four of six tests in each of families 1 and 2 wait on pick batch 2, family 3 was never computed as registered, and scripts/replay.py would apply the wrong Holm sizes. The governance of the RTX arms is split between two incompatible "AB4" amendments.
>
> As a positive-method paper this is a clear reject. As a carefully pre-registered negative result on a small, table-only benchmark it is defensible, but only after findings R3 to R6 and R9 to R11 are fixed. Workshop or negative-results track is the realistic venue, because the space is narrow (A10) and E3b K7 failed on ILI.
>
> ## B. Baselines table (ILI T=24, held-out regret AUC over budgets 1 to 100, pooled seed-SD units, lower is better)
>
> **Exposure notice.** This table contains per-arm means for the AB2 and AB4 RTX arms. These were computed before scripts/ablations/llm_registered.py has run, and they reveal the direction of the registered AB-P and AB-G tests. Whoever is bound by AB4's "computed once" rule must disclose that this review exists. The registered families must still be computed once, by the registered script. I did not compute any registered AB-P or AB-G statistic: no trend, no anon-vs-ref and no repair-vs-ref test.
>
> **Columns.**
> - "League − baseline": the E4 J2 arm `grpo_league_from_ETTh1` (n = 10, 3.14 [2.60, 3.64]), A10 and A10G cards, minus the baseline.
> - "Best LLM arm − baseline": the arm with the lowest mean of all 18 method arms, `qwen_prompted_0.5b_local` (RTX, n = 30, 1.39 [1.24, 1.55]), minus the baseline. It was selected post hoc as the best of 18 arms, so its p-values are optimistic.
> - Bracketed intervals are 95% bootstrap intervals of the difference.
>
> | Required baseline (n) | Baseline AUC [95% CI] | League − baseline, p | Best LLM arm − baseline, p | Beaten? | Source |
> |---|---|---|---|---|---|
> | Random search (30) | 1.52 [1.34, 1.72] | +1.63 [+1.04, +2.16], p 2e-5 | −0.12 [−0.37, +0.12], p 0.34 | No | results/search/E3/ILI_T24/cpu_runs.jsonl (summary.md 1.52, no CI) |
> | Regularized evolution (30) | 1.47 [1.32, 1.63] | +1.67 [+1.10, +2.20], p 1e-5 | −0.08 [−0.30, +0.13], p 0.47 | No | same |
> | TPE (30) | 1.14 [1.03, 1.24] | +2.01 [+1.44, +2.52], p 1e-5 | +0.25 [+0.07, +0.44], p 0.013 (worse) | No; every LLM arm is worse | same |
> | ASHA (30) | 1.38 [1.25, 1.53] | +1.76 [+1.19, +2.29], p 1e-5 | +0.01 [−0.20, +0.21], p 0.94 | No | same |
> | Random + referee (30) | 1.22 [1.07, 1.38] | +1.92 [+1.34, +2.46], p 1e-5 | +0.17 [−0.04, +0.39], p 0.14 | No | same |
> | PG controller, no rivals (30) | 1.69 [1.51, 1.89] | +1.45 [+0.87, +1.99], p 1e-5 | −0.30 [−0.56, −0.05], p 0.022 | No after any correction (one of about 180 post hoc pairs) | same |
> | PG controller, league rivals (30; the matched control, A4) | 1.59 [1.42, 1.76] | +1.55 [+0.97, +2.08], p 1e-5 | −0.20, p 0.10 | No | same |
> | Prompted Qwen 1.5B, E3 (10, A10; the registered K4 and F3 baseline, A26) | 1.81 [1.54, 2.15] | +1.33 [+0.69, +1.93], p 0.0013 | not compared (this is the scale ablation) | No; GRPO is worse | results/search/E3/ILI_T24/qwen_prompted_runs.jsonl |
> | Prompted Qwen 1.5B, RTX ref (30; code-pinned, other card) | 1.57 [1.39, 1.74] | (cross-card; see R6) | (AB-P reference; not computed) | — | origin/soumitra-leg2: results/ablations/rtx/ABL_P_ref_local/ILI_T24/ |
> | TPE warm-started from ETTh1 top 10 (30) | 1.65 [1.52, 1.78] | +1.50 [+0.93, +2.03]. **Registered F1:** p about 1e-5, Holm 6e-5. A loss | −0.26 [−0.46, −0.05], p 0.02 (the 0.5B arm uses no source data) | No | results/search/E4/ILI_T24/transfer_from_ETTh1_runs.jsonl; results/independent/E4_soumitra.json |
> | Portfolio of ETTh1's ranking (30) | 3.28 [3.02, 3.55] | −0.14 [−0.77, +0.44]. **Registered F1:** p 0.64, Holm 1.0 | −1.89 [−2.20, −1.58], p 1e-5 | Only by the prompted arms, never by a GRPO arm | same |
> | GRPO without league (10), for family 2 | 3.23 [2.83, 3.63] | −0.09. **Registered F2:** exact p 0.80 (J3 not run, A30) | — | No | results/search/E4/ILI_T24/grpo_none_from_ETTh1_runs.jsonl |
>
> All 18 method arms, by AUC [95% CI] (n), reviewer-computed:
> - **E4 on Modal.** league 3.14 [2.60, 3.64] (10); none 3.23 [2.83, 3.63] (10).
> - **RTX GRPO arms.** steps30 3.03 [2.58, 3.41] (10); steps60, the AB-G reference, 3.26 [2.91, 3.55] (10); steps120 3.17 [2.78, 3.61] (10); beta 0 3.40 [3.02, 3.79] (10); beta 0.1 3.10 [2.71, 3.47] (10).
> - **RTX prompted arms.** 0.5B 1.39 [1.24, 1.55] (30); 1.5B ref 1.57 [1.39, 1.74] (30); repair 1.59 [1.36, 1.82] (30); anon 1.93 [1.75, 2.11] (30); 3B 2.06 [1.80, 2.30] (30).
> - **E3 prompted.** 1.81 [1.54, 2.15] (10).
> - **The best GRPO arm.** steps30 is +1.34 to +1.90 worse than every non-portfolio baseline (all p 1e-5) and ties the portfolio (−0.25 [−0.76, +0.22], p 0.37).
> - **Effect sizes, league against the baselines.** Hedges g is +1.7 against prompted Qwen and +2.2 to +3.8 against the CPU baselines. The probability that a league run has lower AUC than a run of the baseline is 0.01 to 0.11 for every baseline except the portfolio (0.54).
>
> What may honestly be claimed from B is a negative result:
> - On this table, GRPO transfer from ETTh1 does not beat, and is clearly worse than, every within-table baseline and the untrained LLM. It only ties the source portfolio.
> - Prompted Qwen is not distinguishable from random search, RE, ASHA or the PG controllers, and is worse than TPE and the referee. The 10-seed arm can detect differences of about 0.56 SD against a 30-seed one.
> - "Competitive" may not be used, either for the league (no league effect, J3 not run) or for the LLM architect against HPO baselines (it ties the weak baselines and loses to TPE).
>
> ## C. Findings (most severe first)
>
> **R1. The headline comparison cannot answer RQ1 as stated.**
> - **Location.** docs/PLAN.md A12 and A18 (within-table GRPO dropped); results/independent/E4_soumitra.json `descriptive_ili`; results/search/E3/ILI_T24/summary.md.
> - **Problem.** RQ1 (CLAUDE.md) asks whether GRPO with a league beats random, RE, TPE, ASHA and the same LLM without RL at equal epochs. But the only GRPO evidence is a transfer from a table whose ranking is anti-correlated with the target (Spearman −0.21; top-237 sets disjoint; ILI's best config ranks 2,157 on ETTh1, STATE 23 Nov). The GRPO arm never saw a target reward. The baselines either search the target within the table (CPU methods) or are zero-shot (prompted).
> - **Reviewer's objection.** "This is a test of negative transfer between two tables with opposite dominant fields, not of RL for architecture or HPO search. The loss against every baseline was predictable from E1 and E8 (instance norm explains 83% of ETTh1 variance)."
> - **Fix.**
>   - State RQ1 in the paper in A15's transfer form: "does RL fine-tuning on a source table help an LLM tune a small PatchTST space on another table".
>   - Put the dropped within-table GRPO and the anti-correlation in Limitations.
>   - Present the CPU baselines as context, with their different information sets labelled. Only the portfolio and warm TPE are the registered controls.
> - **Owner.** Diya (framing decision); Anirudh (write-up). **Class:** (O), (W).
>
> **R2. The failure case is not reported in E4's own results: collapse onto the source table's instance-norm setting.**
> - **Location.** results/search/E4/ILI_T24/grpo_*_runs.jsonl; RTX arms on origin/soumitra-leg2 under results/ablations/rtx/ABL_G_*; docs/ablations.md (lines 12 and 190) shows it for the Mac and as 99% for E4.
> - **Problem.** Reviewer-computed:
>   - Share of instance-norm-off configs among charged lookups on ILI:
>
>     | Arm | Share |
>     |---|---|
>     | E4 league | 0.99 |
>     | E4 none | 1.00 |
>     | RTX steps30 | 0.96 |
>     | RTX steps60 | 0.99 |
>     | RTX steps120 | 1.00 |
>     | RTX beta 0 / beta 0.1 | 0.98 |
>     | Prompted, E3 / RTX ref | 0.54 / 0.55 |
>     | Random | 0.49 |
>     | Table | 0.50 |
>
>   - Not one of the 59 ILI configs within 1 SD of the best has instance norm off. The best such config, p24s12l2h4d16f1r1n0i, ranks 65th with regret 1.03 SD. So a policy that never proposes instance norm on is held to about 1 SD by construction.
>   - Per-seed league AUC is 1.60, 2.13, 4.39, 3.65, 3.27, 4.27, 3.12, 2.32, 3.28 and 3.39. The two good seeds (0 and 1) end on exactly that rank-65 config.
>   - The mirror image holds on ETTh1, descriptive and in-sample: ILI-trained policies propose instance norm on 80 to 88% of the time, while ETTh1's best has it off.
>   - K3 (unique configs ≥ 30%) passes at ≥ 0.91, so K3 cannot detect collapse onto a single field.
> - **Reviewer's objection.** "Show the reader why. Without this the 3.1 vs 1.5 gap reads as a broken pipeline."
> - **Fix.**
>   - Report the per-field proposal marginals (instance-norm share) per arm as the failure-case figure or table, labelled exploratory.
>   - Optionally, an exploratory regret conditional on instance norm on.
>   - Do not claim the cause: no intervention tested it.
>   - Cite AB-L (3cb92a1): GRPO did raise source reward on ETTh1 (slope +0.20, Holm p 0.031). The policies learned; what they learned does not transfer.
> - **Owner.** Soumitra (numbers); Anirudh (report). **Class:** (X).
>
> **R3. The registered E4 families are incomplete, and family 3 was never computed as registered.**
> - **Location.** results/independent/E4_soumitra.json (`pending_or_not_run`, `family_3_ili_unadjusted`); docs/claims/E3.json; results/picks/ (no batch 2).
> - **Problem.**
>   - Family 1 has 2 of 6 tests computed and family 2 has 1 of 6. The four ETTh1-pick tests in each wait on pick batch 2 (not on main as of 19:04 ET). So the "and back" direction (J1, ILI to ETTh1, 25 policies, with the three A29 runs) has no endpoint at all. K4 (due 28 Nov) also depends on the picks.
>   - Family 3 ("prompted Qwen against random and TPE at 25 and 50, Holm with the CPU methods' comparisons") does not exist in any committed output. docs/claims/E3.json has qwen vs random but no qwen vs TPE, and the independent file gives only unadjusted p-values (vs TPE: b25 0.0018, b50 0.0083).
>   - The composition is ambiguous: ILI only, or with the ETTh1 picks. Under the reading "the 20 ILI E3 claims plus the two qwen-vs-TPE tests", Holm rejects at b25 (0.0018 × 21 = 0.038) but not at b50 (0.0083 × 20 = 0.17). That is reviewer arithmetic, for illustration only.
>   - Diya's re-score of J3 with replay.py is unrecorded (STATE) and cannot be cited.
> - **Reviewer's objection.** "Half of the primary transfer test is missing, and a registered family was never computed."
> - **Fix.**
>   - Run pick batch 2 on diyapr (`scripts/picks.py collect --budgets 25 50`), then recompute both families with the independent script and with replay.py.
>   - Before computing family 3, fix its exact membership in a dated note.
>   - If batch 2 cannot run by the deadline, report family 1 and family 2 as incomplete under A26's spirit, and the J1 direction as not tested.
> - **Owner.** Diya (picks, family 3 definition); Soumitra (recompute). **Class:** (O), (C).
>
> **R4. The registered pipeline does not produce the registered E4 output.**
> - **Location.** scripts/replay.py lines 172 to 188 (claims and Holm); results/search/E4/{ILI_T24,ETTh1_T336}/summary.{md,json}; results/independent/E4_replay_diff.json (`holm_families`, `worktree_status`).
> - **Problem.**
>   - (a) replay.py takes the E4 Holm family to be whatever claims are in docs/claims/E4.json: 2 in family 1 and 1 in family 2 on ILI. A25 and A30 register 6 per family, with pending tests as not rejected. Gate G4 would therefore recompute the wrong family.
>   - (b) The E4 summary.md and summary.json on main were last written by 635042a (24 Nov 11:47), before e6ce92f and 22111da added the GRPO runs. They list only the two controls, so a reader of the canonical summary sees no GRPO arm.
>   - (c) docs/claims/E4.json and the regenerated summaries exist only in a scratch worktree that was discarded.
>   - (d) No committed output gives an interval for AUC, the primary endpoint: summary.md prints AUC bare, and the independent JSON gives means and medians only.
>   - (e) The CLAUDE.md rule-6 paired block-bootstrap interval (A1 windows) is absent for every E4 comparison. By the project's own rule, nothing in E4 is a claim yet, including the loss. The critic's verdict says so.
> - **Reviewer's objection.** "The primary endpoint has no interval in any artifact, and the gate cannot reproduce the registered decision."
> - **Fix.**
>   - Give replay.py a registered, fixed family list with p = 1 for missing tests.
>   - Add AUC bootstrap intervals to summarize().
>   - Regenerate and commit the E4 summaries and claims.
>   - Run G4.
>   - Add the window-block bootstrap for the E4 differences, or decide in writing that rule 6 does not apply to losses.
> - **Owner.** Diya. **Class:** (C), plus (O) for the rule-6 decision.
>
> **R5. Two incompatible amendments called "AB4" govern the same RTX arms.**
> - **Location.**
>   - Soumitra's 15db163, origin/soumitra-leg2, 26 Nov 01:32:44 ET: queue run_ab4_queue.sh (sha256 2b9f3d98...).
>   - Diya's 2ab2996, main, 26 Nov 17:21:57 ET: queue run_wp3_rtx_queue.sh (dc08a640...), scripts/ablations/llm_registered.py, docs/LEG3_soumitra.md.
>   - Runs: 185a0b1, c7e4bc9, e880b0a (queue 01:32 to 18:25 ET).
> - **Problem.**
>   - Diya's text says "before any run" and "No run of ABL_P_anon_local ... exists in git; Soumitra confirms in his launch commit ...". At 17:21 the queue had been running for 16 hours. The anon and repair arms were done (committed locally at 07:20) and steps30 was done (11:29). This clone's fetch reflog shows origin/soumitra-leg2 at 7fd5085 at 16:44, so Soumitra's AB4 was already on origin and the run data was not.
>   - LEG3 and the STATE section are headed "26 Nov, early morning" but were committed at 17:21.
>   - The two AB4s differ in:
>     - queue and sha;
>     - job order (prompted ETTh1 before the step arms, or after);
>     - stop time (28 Nov 18:00 ET or 23:59 ET);
>     - record layout (retried.txt);
>     - secondary step analyses and the dose-response decision rule (Diya's (i) to (iii) with an explicit rule, against Soumitra's exploratory AB3-addendum-3 analyses);
>     - the analysis script and its output: llm_registered.py writes results/ablations/LLM-registered.json; rtx_tests.py, added in e6926d8, writes results/ablations/rtx/RTX_tests.json.
>   - The scripts pin different versions of scripts/ablations/llm_tests.py: 08c72a4c on main, 01b70835 (5abbd20) on the branch. Each refuses to run on the other. A merge preview shows an add/add conflict on llm_tests.py, plus conflicts in docs/prereg_ablations.md (two "## Amendment AB4" sections) and docs/ISSUES.md. After a merge only one of the two scripts can run.
>   - The two llm_tests.py versions differ only in the docstring and registered_source(); spearman_trend is the same. So the statistic agrees and the forking is procedural.
>   - llm_registered.py's commit check (line 91) only greps for the string "## Amendment AB4". It accepts records stamped with either AB4, so it cannot tell which registration a run followed.
> - **Reviewer's objection.** "Which pre-registration governs? One postdates the runs and wrongly says it predates them. The other was committed seconds before launch, and nothing shows it was public first."
> - **Fix.**
>   - The owner should rule that Soumitra's 15db163 governs these runs. It was in force at launch, the records carry 15db163, 7fd5085 and 185a0b1, and the run-path code equals c2dda8d (checked).
>   - Re-label Diya's amendment in a dated amendment, for example "AB5, written 26 Nov 17:21 ET after the AB4 queue started; adds the analysis code and the secondary analyses; its 'before any run' sentence is wrong". Keep the analyses it adds as registered-after-launch but before any analysis, which needs Soumitra's attestation.
>   - Choose one analysis script and one llm_tests.py, and rename one of the duplicate section headers.
>   - Soumitra should state in writing whether rtx_tests.py was ever run and its output seen.
> - **Owner.** Diya (decision); Soumitra (attestation). **Class:** (O), plus (C) for the conflicting pins and the weak commit check.
>
> **R6. The "same LLM without RL" baseline is not matched in card, code or seeds.**
> - **Location.**
>   - results/search/E3/ILI_T24/qwen_prompted_runs.jsonl, added in b3e59e7 (23 Nov 18:51, A20 code): gpu "NVIDIA A10" in all 10, no env block, no git hash, no versions.
>   - E4 J2: env git e1a063c, pinned; deployments 5 on A10 and 5 on A10G per arm; training 12 A10 and 8 A10G.
>   - RTX arms: RTX 4080 Laptop GPU under WSL2. c7e4bc9 notes that from 05:18 on, the card was shared with other Windows load.
> - **Problem.**
>   - Comparing E4 GRPO with E3 prompted confounds card and code: llm.py was the A20 version (7a99cdef) or the A21 version for E3, which the records cannot say, and 7a983eff (the A27 addendum) for E4. A26 accepts the A10G prompted runs as the baseline, but the records say A10.
>   - Seeds do not replicate across cards. Reviewer-computed per-seed AUC correlations:
>     - J2 league against RTX ABL_G_ref_local (same training seeds 100 to 109 and deployment seeds 0 to 9): −0.17. Arm means 3.14 vs 3.26, p 0.73.
>     - E3 A10 prompted against RTX prompted on seeds 0 to 9: 0 of 10 identical, correlation −0.16. Means 1.81 vs 1.57, p 0.33.
>   - Within E4, A10 against A10G league deployments: 3.08 vs 3.21 (n = 5 each).
>   - The only comparison of RL and no RL on one card with one code version is the registered AB-G step trend, level 0 against the trained levels on the RTX, which has not been computed.
>   - With 10 seeds (E3 prompted, E4 arms) against 30, the detectable difference at 80% power is about 0.56 SD for prompted against random and about 0.84 SD for league against random. "Prompted ≈ random" means no difference was detected, not that they are equivalent.
> - **Reviewer's objection.** "Your RL and no-RL arms ran on different hardware with different code, and you show that hardware changes seed-level outcomes."
> - **Fix.**
>   - Report the card per arm (AB1) and the code commit per arm.
>   - Keep E3 prompted as the registered baseline, with its caveat. Show the 30-seed RTX prompted ref as a sensitivity check, and point to the AB-G step trend as the matched comparison.
>   - Never pair seeds across cards.
>   - State the detectable effects (docs/ablations.md already gives 1.10 to 1.25 SD for E4's ILI nulls).
> - **Owner.** Anirudh (report); Soumitra (cards and commits per arm). **Class:** (W).
>
> **R7. The GRPO levels are nested, and one set of policies serves three comparisons.**
> - **Location.** RTX ABL_G_steps30_local, ABL_G_ref_local and ABL_G_steps120_local (training seeds 100 to 109, deployment seeds 0 to 9, constant-rate AdamW). The Mac steps arms are literal checkpoints of one run. Both AB4s disclose this.
> - **Problem.**
>   - A 30-step policy is, up to GPU nondeterminism, a prefix of the 60- and 120-step policies with the same seed.
>   - ABL_G_ref_local is at the same time the 60-step level, the beta 0.04 KL level and the hardware twin of J2.
>   - Per-level intervals plotted side by side look like three independent replications. The registered trend p is anti-conservative. Diya's diagnostic (iii) will measure how nested the levels really are; the cross-card divergence in R6 suggests the nesting may be weaker than assumed, but that has to be measured.
> - **Reviewer's objection.** "n = 10 policies, not 30, for the step question."
> - **Fix.**
>   - Plot per-seed lines across levels.
>   - Lead with the policy-level analyses.
>   - Say in the caption that the levels share seeds and the 60-step policies.
> - **Owner.** Anirudh; Soumitra. **Class:** (W).
>
> **R8. A29 was registered after the E4 launch.**
> - **Location.**
>   - cff56ff, "Soumitra's pre-launch review ... A29 ... before any E4 rerun result", 24 Nov 14:01:34 ET.
>   - 8530eae, the A29 addendum, 14:06:23 ET.
>   - Launch at 13:09:06 ET, per results/logs/soumitra_e4_launch_time.txt (added in e6926d8) and e6ce92f.
>   - J2 and J1 attempt-1 records carry e1a063c, which predates A29.
> - **Problem.** The retry rule was committed 52 to 57 minutes into the runs, while J2 training records were coming back. It came before the 14:37 workspace stop it later governed, and it only affects the sensitivity analysis. But "pre-launch" and "before any E4 rerun result" are inaccurate.
> - **Reviewer's objection.** "Registered mid-run."
> - **Fix.** A dated disclosure in docs/prereg_E4.md and in the paper's pre-registration paragraph.
> - **Owner.** Diya. **Class:** (W).
>
> **R9. Git does not show that the registration came first, and the independent output records the wrong commit.**
> - **Location.** c21d0f7 23:41:47 → 68593a0 23:42:03 → 7808ea6 23:45:24; 56960ea → 3cb92a1 (16 s); a730d1f → 80a85b9 (11 s); results/independent/E4_soumitra.json.
> - **Problem.**
>   - In 3 min 37 s, the sequence "committed before it is run" → run → verifier re-score with its own code → critic review → 47-line script revision → regenerated output was all recorded. Commit order therefore does not show work order. The pre-commitment rests on attestation.
>   - The JSON says git_head 68593a0, but its script_sha256 a9c5fead is the 7808ea6 version. The run was on a dirty tree, after the critic's edits.
>   - Soumitra's AB4 (01:32:44) reached origin some time between 00:30 and 16:44 (this clone's fetch reflog), after the queue started at 01:32.
> - **Reviewer's objection.** "Your pre-registration claims are unverifiable from the artifacts."
> - **Fix.**
>   - From now on, push each registration and record its remote push time before running. GitHub's push log gives a third-party timestamp.
>   - Rerun scripts/independent/e4_analysis.py on a clean checkout of 7808ea6 and confirm the JSON is byte-identical apart from git_head.
>   - Record a dirty-tree flag in every output.
> - **Owner.** Soumitra (rerun); Diya (process). **Class:** (C), (W).
>
> **R10. Provenance gaps in the run records.**
> - **Location.**
>   - E3 cpu_runs.jsonl and E4 transfer_*_runs.jsonl: no git hash, env or method configuration in any record. qwen_prompted: gpu only.
>   - E4 J1 relaunch records (all 25 ETTh1 deployments and the retrained league109 training record): git 894b65a. That commit is not an ancestor of main or the branch and is reachable only through the remote tag e4-j1-launch. It is the pre-rebase twin of e6ce92f: its parent is e1a063c rather than 8530eae, and the trees differ only in docs.
>   - The training records of grpo_none_ILI_seed108 and seed109 are lost (678c439).
> - **Problem.** The baselines of RQ1 have no per-record code version, against CLAUDE.md rule 7. One E4 direction is stamped with a hash that a reader cannot find on any branch.
> - **Reviewer's objection.** "Which code produced the baselines?"
> - **Fix.**
>   - Add a provenance sidecar per runs file: the commit that wrote it (from `git log`), the code digests and the method settings.
>   - Record 894b65a ↔ e6ce92f in docs/ISSUES.md and results/README.md, and keep the tag forever.
>   - List the lost training records in the paper's failure table.
> - **Owner.** Diya (E3); Soumitra (E4). **Class:** (W).
>
> **R11. The "Results" commit has no description and adds a lock, back-filled logs, duplicates and a second analysis script.**
> - **Location.** e6926d8 (origin/soumitra-leg2, 26 Nov 18:44).
> - **Problem.**
>   - The message is the single word "Results", with no verifier or critic line.
>   - It commits an empty results/logs/l4_queue.lock. The AB2 queue and Diya's queue take this lock to refuse a second launch, so every fresh clone starts locked.
>   - It commits results/logs/soumitra_e4_launch_time.txt, a 24 Nov timestamp file written on 26 Nov.
>   - It adds copies of all 20 RTX run files under results/runs/llm_local/. I verified that they are byte-identical to results/ablations/rtx/, but on the branch the AB2 GRPO training.jsonl files exist only there. AB2_records.sha256 on main and AB2_move.sha256 on the branch also both describe the same move.
>   - It adds copies of common.py and llm_tests.py, plus rtx_tests.py and its tests, with no statement of whether rtx_tests.py was run. Diya's LEG3 forbids computing any AUC before step 4.
> - **Reviewer's objection.** "Undocumented data and code commit; duplicate records invite double counting."
> - **Fix.**
>   - A follow-up commit (no history rewrite) that describes e6926d8.
>   - `git rm --cached results/logs/l4_queue.lock`.
>   - Keep results/ablations/rtx/ as the only copy the analyses read, and mark results/runs/llm_local/ as raw launcher output with a manifest.
>   - Merge the two manifests.
>   - Attest whether rtx_tests.py was run.
> - **Owner.** Soumitra. **Class:** (W), (O).
>
> **R12. Test choice and reporting details.**
> - **Location.** scripts/independent/e4_analysis.py `permutation_p`, `holm`; results/independent/E4_soumitra.json.
> - **Problem.**
>   - The registered two-sample permutation test on the difference of means (A20) assumes exchangeability. With 10 against 30 seeds and unequal SDs (league 0.90, warm TPE about 0.4, portfolio about 0.7), it does not test equality of means alone. The decisions are unaffected because the effects are huge or null, but a reviewer will ask.
>   - The `holm_with_pending_as_not_rejected` block prints `"reject": true` for the warm-TPE loss with no direction. The direction appears only in the tests block.
>   - The A26 worst case is correctly trivial on ILI, with no failed or retried seed.
> - **Fix.**
>   - Report a Welch or studentized permutation interval beside each registered test, labelled as a sensitivity check.
>   - Add `league_wins: false` and `direction` fields to the Holm block.
> - **Owner.** Soumitra. **Class:** (X) for the sensitivity check; (W) for the fields.
>
> **R13. Equal epochs are not equal compute, and A6 cost reporting is absent.**
> - **Location.** PLAN A6 and A11; AB-L.json (final_episode about 4, so about 4 to 5 episodes of 3,000 source epochs per policy); 22111da ($22.17 metered for 45 Modal policies); RTX times in 79f1bf0 and e880b0a (17 to 21 min per 60-step policy, 36 to 43 min per 120-step policy, and slower after 05:18 because of shared load); deployment `seconds` 140 to 272 s per search in the E4 records.
> - **Problem.**
>   - Only target-table epochs are equal (3,000 per method). GRPO also spent source episodes and LLM GPU time. The portfolio and warm TPE consume the whole source table (2,368 × 30 epochs).
>   - No forecaster-epoch and GPU-second table exists, and there is no table-enumeration reference line.
> - **Reviewer's objection.** "Cost-normalized comparison missing."
> - **Fix.** A compute table per arm: target epochs, source epochs, GPU type, GPU-seconds for training and deployment, dollars. Use the A6 wording only.
> - **Owner.** Soumitra (numbers); Anirudh (table). **Class:** (C), since A6 registers it and no code produces it.
>
> **R14. ETTh1 numbers are in-sample, and the ETTh1 E4 summary is stale.**
> - **Location.** results/search/E4/ETTh1_T336/summary.md (controls only); J1 runs.
> - **Problem.**
>   - Reviewer-computed in-sample ETTh1 AUC, descriptive only (A18): league_from_ILI 3.51 (10), none 4.04 (10), uniform 2.22 (5), tpe_warm 2.11, portfolio 5.94, random 1.61, prompted 1.30.
>   - The ETTh1 table has one seed for 2,268 configs, and the pooled SD comes from a selected top 100.
> - **Reviewer's objection.** "Do not put these next to the ILI numbers."
> - **Fix.**
>   - Keep ETTh1 out of any results table until the pick reseeds exist.
>   - Label "in-sample, no claim" wherever ETTh1 appears.
> - **Owner.** Anirudh. **Class:** (W).
>
> **R15. Table validity has to sit next to every RQ1 number.**
> - **Location.** STATE 23 Nov (E3b ILI K7 fails: bias +0.0042 [0.0013, 0.0070]; top 20 about 1.6 SD optimistic); PLAN K7 fallback ("RQ1 for the table only"); A10 (74 architectures, a narrow space).
> - **Problem.** Every number in B describes the replay table, not live training on ILI.
> - **Fix.** Put the scope sentence in the caption of every results table and figure.
> - **Owner.** Anirudh. **Class:** (W).
>
> **R16. Results-section artifacts a main-track reviewer will require are missing.**
> - **Location.** No figure of search results exists in the repo; the only figures are the stage-1 report's (docs/report/overleaf/figures).
> - **Missing.**
>   - Anytime-regret curves (b = 1 to 100, mean and 95% band, every ILI method, with the A3 noise-floor band from the table's best config on fresh seeds).
>   - AUC with intervals.
>   - Effect sizes beside every comparison: difference with CI, Hedges g, probability of improvement.
>   - A per-seed strip plot for the 10-policy arms.
>   - The failure-case table from R2.
>   - A seeds table: 10 against 30 seeds; the training seed 100+i → deployment seed i mapping; the detectable effects.
>   - The compute table from R13.
>   - A hardware table per arm.
>   - Per-architecture regret (A15) for the GRPO arms.
>   - The pending ETTh1 direction, stated as not tested.
> - **Fix.** Generate all of these from results/ through scripts/report.py, under rule 5. The numbers currently typed into docs/verdicts/ and STATE are not rule-5 compliant.
> - **Owner.** Anirudh; Diya for report.py. **Class:** (X) for the descriptive figures; (C) for report.py.
>
> ## D. What the paper can claim
>
> 1. On the ILI replay table (held-out seeds; valid for the table only, E3b K7), Qwen2.5-1.5B policies trained with GRPO on the ETTh1 table and deployed frozen did not beat any required baseline. They were indistinguishable from replaying ETTh1's best configs (AUC 3.14 vs 3.28 pooled seed SD, p 0.64) and worse than warm-started TPE (1.65, registered Holm p 6e-5). The pre-registered transfer win therefore fails in the ETTh1-to-ILI direction, and the reverse direction is untested until the pick reseeds exist.
> 2. Descriptively (exploratory, unadjusted), every GRPO arm, on Modal A10 and A10G and on a local RTX 4080, was 1.3 to 2.1 SD worse than random search, regularized evolution, TPE, ASHA, the referee, the policy-gradient controller and the untrained prompted model. About 99% of its ILI lookups were instance-norm-off configs, none of which lies within one seed SD of ILI's best. This is consistent with a source-table preference that does not transfer, even though GRPO did raise its source-table reward (AB-L); the cause was not tested.
> 3. No league effect was detected (p 0.80; detectable difference about 1.1 SD), so, as fixed in advance, the paper does not call the league "competitive" and makes no claim for it.
> 4. Prompted Qwen without RL was not distinguishable from random search (10 seeds; detectable difference about 0.6 SD) and was worse than TPE on the table.
> 5. The paper may not claim that RL fine-tuning, the league or an LLM architect improves PatchTST search. Nor may it claim "equal compute": only target-table epochs were equal.
>
> ## Reviewer's notes
>
> - I wrote nothing to the working tree, index or refs. One command, a `git merge-tree --write-tree` preview of the main and branch merge, wrote unreferenced tree and blob objects into .git/objects. No ref, index or file changed, and `git gc` will prune them. I deleted a temporary pickle I had written to the scratchpad. This file is the only file I left.
> - The RTX arm means in B are an exposure under AB4 ("computed once"); see the notice at the top of B.

### Fixer, round 1

> I applied the area-chair review (every W finding, B19, the X item B12 and the two D items) and the items the results review gives Anirudh (R1, R2, R6, R7, R13, R14, R15 and the optional R16). The changes are in eight commits on anirudh-ablations, and nothing was pushed. The registered families in LLM-rtx.json are identical to the 40804c7 version (compared as dicts and as sorted JSON). Both `.venv/bin/python -m pytest -q tests/test_ablations.py` and `.venv/bin/python -m pytest -q` exit 0.
>
> ## Commits (author Anirudh Katupilla; none has a Co-Authored-By, "Claude", "Anthropic" or "Generated with" line)
> - **a516ec9** llm_rtx.py: adds the run-path check (B19) and descriptive blocks for the report: each arm's queue DONE time, run commits and cards, the timeline, Diya's disclosure read from 2ab2996, whether LLM-registered.json exists, common.summarise per arm and per-policy AUCs. The move test now reads llm_rtx.py at 943d893 and still checks llm_tests.py's sha256.
> - **84b49a5** power.py: adds a descriptive column with t quantiles beside the registered normal-approximation column (B18). The RTX rows now say "not of record" instead of "final".
> - **0cc7759** New `scripts/ablations/compute.py`, writing results/ablations/compute.json (D, R13).
> - **92006f4** New `scripts/ablations/programme_holm.py`: one Holm over all 13 registered ablation tests, exploratory and post hoc (B12).
> - **17c2940** New `scripts/ablations/anytime.py`: the anytime-regret figure, docs/figures/anytime_regret_ILI.svg (R16). Its test went into the next commit.
> - **8d40c88** report.py: all wording and rendering changes, with tests for the new code.
> - **2147eb3** docs/neurips_checklist.md (B17). Wording only.
> - **e62e88a** Outputs regenerated from a clean tree.
>
> Every message says the change was made after the results were seen and says whether it is wording only.
>
> ## Area-chair findings
> - **B1 (W):** The RTX section now has a timeline table built from git commit times and the queue lines in l4/ab4_queue_status.txt at the pin. The two AB4s are named by commit everywhere, with a paragraph saying which elements the report takes from each.
> - **B2 (W):** The report now says LLM-registered.json is not yet computed. That comes from git: it is absent at origin/main 2ab2996, at the pin e6926d8 and in the tree.
> - **B3:** The status is now "complete, 10 of 10 tested ILI arms, not the result of record", and the report explains what AB5's "final" means. Both deadlines are given with their sources, and every DONE time is shown to be before both.
> - **B4:**
> - The summary now leads with "no dose-response", then the trained-vs-untrained gap (+1.59 [+1.28, +1.90]), then the registered step trend with its anti-conservative caveat.
> - Analyses (i) and (ii) are labelled registered secondary in Diya's AB4 and exploratory in Soumitra's.
> - The p ≤ 0.05 reading of "shows a difference" is disclosed as Anirudh's operationalisation.
> - **B5:** The ETTh1 order is in the summary. The anon arm is defined from prompt.py and constrained.py. Scale and anon are scoped to ILI, these checkpoints and this prompt.
> - **B6:** diagnostics_ILI is rendered in a per-arm ILI table. That table has a raw-MSE column from common.summarise on the same runs, and the RTX repair confound is stated. Raw MSE is not shown for the Mac ILI arms, and the report says so: LLM.json does not hold it and llm_tests.py stays frozen.
> - **B7:** Diya's disclosure is quoted verbatim, read from 2ab2996, beside the timeline.
> - **B8:** The record note is rewritten as the review asked: the llm_tests.py history and docs/LEG3_soumitra.md named.
> - **B9:** The rerun is disclosed in "Runs, re-runs", built from LLM-first-AB4-run.json against LLM-rtx.json. The KL trend's Holm p went from 0.749 to 0.3745, and no decision changed.
> - **B10 (adapted):** The report says the values were independently recomputed and agree, pointing to docs/verdicts/ablations_rtx_reviews.md. It does not give a count. The sentence appears only while the pin and a digest of the families match the checked version; otherwise it says "Not verified". I did not write that file.
> - **B11:** The rule-6 scope sentence is in the RTX section, the Mac section and the summary.
> - **B12 (X):** There is a new exploratory section and a note in the anon bullet. Under one Holm over all 13 tests, anon gets p 0.080 and AB-L ETTh1 gets p 0.156; neither is rejected. The tests that share ABL_P_ref_local are listed.
> - **B13:** The AB-L summary bullet now starts with the registered rejection resting on a test that is too liberal, then the exploratory test that agrees.
> - **B14:** The figure is dropped and its source is cited in words.
> - **B15:** A dated correction in "Runs, re-runs" lists, from the queue lines, the AB4 jobs done before 510633d. AB5 itself is not edited.
> - **B16:** The RevIN shares are shown beside the AUCs, and the sentence drawing an inference from them is gone.
> - **B17:** Checklist item 7, item 8 (plus item 16) and the open items are updated (both AB4s, AB5, the AB3 addenda, the f9c0d62 rerun). Items 1 and 10 are unchanged.
> - **B18:** A t-quantile column was added. The report says those values are 4% to 25% larger overall and 11% to 19% larger for E4's 10-policy rows, all computed by the script. E4's nulls could detect 1.10 to 1.25 SD under the normal approximation, and 1.25 to 1.38 SD with t quantiles.
> - **B19 (C):** The run-path check (leaguetst/, scripts/llm_local.py and requirements-llm.txt equal to c2dda8d) now runs at every run and training commit. All arms pass and the families are unchanged; I confirmed this by running it.
> - **B22:** The RTX section has a source line.
> - **D:** Headers now say "Welch t 95%, unadjusted". The compute totals are rendered in a Compute section, and checklist item 8 points to that section and JSON instead of typing the numbers.
>
> ## Results-review items
> - **R1:** A transfer-scope sentence is in the baseline section and the summary bullet.
> - **R2:** There is a failure-case table: the RevIN-off share of ILI lookups beside the AUC for every arm, and best regret with RevIN off against on. It says no intervention tested the cause.
> - **R6:** The checks table shows the card and run commits per arm. The hardware checks, on the RTX and on the Mac, now show arm means and the difference of means with no paired interval.
> - **R7:** There is a per-policy table across 30, 60 and 120 steps and the two KL levels, with the per-policy rho and the caption you asked for. The policy-level analyses now come before the AB-G family table.
> - **R13:** The per-arm table gives the card, target epochs, source epochs and hours. Dollars come only from the committed results/logs/ab2_modal_billing.txt, a month-to-date summary of Soumitra's Modal profile that is not split per arm.
> - **R14:** ETTh1 is labelled "in-sample, no claim" in every section and summary bullet where it appears. The exception is the ST pick-reseed rows, which are on fresh seeds.
> - **R15:** The scope sentence (replay table, not live training; K7 cited in words) is in each results section's intro, the summary and several table captions. It is per section, not under every individual table.
> - **R16:** A plain SVG, two panels of four series each, mean only with no band. The palette passed the colour-vision validator in light and dark mode, I checked the render by eye, and the report has a table view of it.
>
> ## Notes
> - Untracked files I did not touch: results/logs/mps_queue.status and results/ablations/verify_rtx/, which another agent created during my run.
> - The regeneration order was the one in your brief, with compute.py, programme_holm.py and anytime.py run between power.py and report.py.
> - No protected file changed: llm_tests.py is still 08c72a4c, and common.py, the prereg files, PLAN, STATE, claims, the ledger, verdicts and the e4_rerun.sha256 files all have empty diffs.
>
> ## Items for the owners (O), not fixed
> - **B1 (O, Diya):** issue a dated amendment that renames the two AB4 sections, states which governs each element, and corrects the "What existed" paragraph.
> - **B2 (O, Diya):** decide which script is of record and resolve the conflicting llm_tests.py pins (rtx_tests.py 01b70835 against llm_registered.py 08c72a4c).
> - **B20 (O, Soumitra):** retitle rtx_tests.py as a verifier or drop it, correct the "AB4 addendum 2" citation, and explain or remove the duplicate llm_local copies and the stale l4_queue.lock.
> - **B21 (O, Diya):** make llm_registered.py's AB4 check tell the two amendments apart.
> - **B23 (O):** diagnostic (iii) in llm_registered.py for GPU nondeterminism, or an exploratory same-seed rerun.
> - **Results review, not mine:** R1's framing decision (Diya), R3, R4, R5, R8, R9, R10, R11 and R12.

### Area-chair re-review (critic's final verdict on the ablation section is its section 3)

> # Re-review: LeagueTST ablation report after the fixes (anirudh-ablations at e62e88a)
>
> I read the following:
> - the regenerated docs/ablations.md (903 lines) in full;
> - docs/neurips_checklist.md;
> - the commit messages 943d893..e62e88a;
> - the new code where a sentence depended on it: report.py `failure_case`, llm_rtx.py's timeline labels (line 106), and programme_holm.py's outputs.
>
> I spot-checked these numbers and found no wrong value:
> - the per-policy table: the 60-step column mean is 3.26, the per-policy rho values are right, and their mean is +0.05;
> - the Holm over all 13 tests: 13 × 1e-5, 12 × 3e-5, 11 × 0.0073 = 0.080, 10 × 0.0156 = 0.156;
> - the anytime table against the AB-X and E4 values;
> - the 93 baseline-favouring Holm rejections, counted by hand.
>
> I made no edits to the repository.
>
> ## 1. Status of B1 to B23
>
> | ID | Addressed | Is the fix correct? |
> |---|---|---|
> | B1 (two AB4s) | Yes for W; O open | Both amendments are named by commit; a timeline is built from git and the queue status files; the report states which elements it takes from which amendment. The timeline mislabels one commit (N2). Diya's corrective amendment is still owed. |
> | B2 (record absent, pins) | Yes for W; O open | The report now says "LLM-registered.json is not yet computed". The checklist's "three computations pin incompatible versions" is now stale (N7). |
> | B3 ("final", deadline) | Yes | The status is "complete … not the result of record", with both deadlines and their sources. |
> | B4 (step headline) | Yes | The summary leads with "no dose-response" and with the trained-minus-untrained gap, its direction, interval and p. It gives both labels for analyses (i) and (ii) and discloses the p ≤ 0.05 operationalisation. |
> | B5 (ETTh1 reversals, anon definition) | Yes | ETTh1 orders are given in the summary. The anon arm is defined precisely, including the trie change. |
> | B6 (diagnostics, raw MSE) | Yes | There is a per-arm ILI table with raw MSE, repeat, repaired and invalid rates and RevIN share. The repair confound is now stated for the RTX arm. For the Mac arms, the absence of raw MSE is explained. |
> | B7 (Diya's disclosure) | Yes | Quoted verbatim, with a correct gloss from the timeline. |
> | B8 (record note) | Yes | It is correct, with one nuance: "between 510633d and 523ae42, against LEG3's request" overstates for 510633d (17:12 ET), which predates LEG3 (17:21 ET). Only the edits after this clone fetched LEG3 at 19:04 went against a request Anirudh had seen. |
> | B9 (rerun) | Yes | Both outputs, the KL Holm p (0.749 then, 0.3745 now) and "no decision changed" are stated. |
> | B10 (verification) | **Partly** | The verification was done: the untracked results/ablations/verify_rtx/ab4/compare3.txt reads "BAD rows: 0 of 113". But the report and the checklist cite docs/verdicts/ablations_rtx_reviews.md, which does not exist on anirudh-ablations. It is only on local/ab2-provisional-do-not-push d577ef8, with AB2 scope ("32 of 32"). The verifier's scripts and outputs are untracked, and no critic verdict on the AB4 values is filed. See N1. |
> | B11 (scope) | Yes | The "not claims under CLAUDE.md rule 6" sentence is in the header, the summary and every LLM section. |
> | B12 (Holm over the whole programme) | Yes (X) | It is labelled post hoc, and the registered decisions are kept. The values are correct and the shared-reference note is given. |
> | B13 (AB-L) | Partly | The summary and the AB-L section are fixed. The new failure-case section reasserts the claim (N3). |
> | B14 (typed figure) | Yes | The 0.10 to 0.15 figure was removed and its source is cited in words. One older typed number remains (N10). |
> | B15 (AB5's "What existed") | Yes | A dated correction is in "Runs, re-runs"; AB5 itself is unedited. |
> | B16 (RevIN sentence) | Yes | No inference is drawn from the order. The added note on the 0.5B arm's 52% repeat rate is apt. |
> | B17 (checklist) | Mostly | Items 7, 8 and 16 and the deviation list are current. Remaining problems: items 7 and 16 cite the missing review file (N1); item 2 omits the K7 table-only limitation (N9); the "three incompatible pins" statement is stale (N7). |
> | B18 (t quantiles) | Yes | A t column is added beside the registered normal value. "1.25 to 1.38 SD" for E4 is consistent with the rows. |
> | B19 (run-path check) | Yes (C) | There is a check of leaguetst/, llm_local.py and requirements-llm.txt against c2dda8d at every run and training commit. All arms pass, which agrees with my diff. |
> | B20 (rtx_tests.py, e6926d8) | O, open | Not Anirudh's to fix. The report describes rtx_tests.py accurately. |
> | B21 (llm_registered.py check) | O, open | Not Anirudh's to fix. |
> | B22 (source line) | Yes | Source: llm_rtx.py at 2147eb3 (sha256 a0a5b32e…). |
> | B23 (GPU nondeterminism) | Partly | It is disclosed as "not measured" but not measured; that is left to Diya's diagnostic (iii) or an exploratory rerun. |
>
> ## 2. New problems the fixes introduced
>
> **N1 (blocking): the cited verification is not on the branch.**
> - docs/ablations.md says "A verifier recomputed these values independently and they agree (docs/verdicts/ablations_rtx_reviews.md)". It says so in the summary bullet (line 13), in the Verification paragraph (line 138) and in "Runs, re-runs".
> - Checklist items 7 and 16 cite the same file.
> - That file is absent on anirudh-ablations, and results/ablations/verify_rtx/ (recompute.py, ab4/recompute_ab4.py, ab4/compare3.txt) is untracked.
> - **Fix (W):** either commit the verifier's scripts, outputs and quoted reports, plus a critic verdict on the AB4 values, or say "verified locally, not yet committed".
>
> **N2: a timeline commit is mislabelled.**
> - llm_rtx.py, the timeline labels at line 106, labels the pinned commit e6926d8 "AB4 results committed".
> - The AB4 results were committed in 185a0b1, c7e4bc9 and e880b0a, which the timeline omits.
> - e6926d8 ("Results") holds the duplicate records under results/runs/llm_local/, rtx_tests.py and a lock file.
> - **Fix (W):** label e6926d8 for what it holds and add the three commits that carried the results.
>
> **N3: the failure-case section regresses B13.**
> - report.py, `failure_case`: "GRPO raised the policies' mean reward on the ETTh1 source (AB-L …), so the policies changed in training".
> - This re-asserts an exploratory, in-sample result as fact. It also explains RTX and Mac policies with AB-L slopes of the E4 Modal policies.
> - The RevIN shares already show that training changed the policies: 96 to 100% RevIN-off against 41 to 61%.
> - **Fix (W):** drop the AB-L clause.
>
> **N4: the hardware-check differences have no interval.**
> - The RTX and Mac hardware tables now print the "difference of means" with no interval, because the seeds are not paired.
> - An unpaired Welch interval is the honest uncertainty here (checklist item 7). Omitting it is a step back from round 1.
> - **Fix (W, from the JSON):** add the Welch interval.
>
> **N5: an ambiguous count.**
> - "93 of 162 favour the baseline" counts Holm rejections in the baseline's favour, not point estimates.
> - **Fix (W):** say "93 of 162 are Holm-rejected in the baseline's favour".
>
> **N6: a garbled sentence.**
> - report.py, around line 352, renders: "it is in none of Diya's AB4 = origin/main at 2ab2996; origin/soumitra-leg2 (pinned) at e6926d8, nor in this tree".
> - **Fix (W):** rewrite it as a list of the refs checked.
>
> **N7: the checklist's pin statement is stale.**
> - The checklist's open item still says "the three computations pin incompatible versions of llm_tests.py".
> - Since 943d893, llm_rtx.py and llm_registered.py share 08c72a4c; only rtx_tests.py (01b70835) differs.
> - **Fix (W):** state the current two-way conflict.
>
> **N8: presentation.**
> - The K7 "replay table only" sentence appears 22 times; state it once in the header and per section heading.
> - The exploratory 162-comparison bullet sits above the registered RTX bullet in the summary. Put the registered results first.
>
> **N9: checklist gaps.**
> - Item 2 (limitations) omits the qualifier the report now leans on: E3b's K7 failed on ILI, so every ILI result holds for the replay table only.
> - Item 8's hand-typed "$22.17" (from AB1) sits beside the Compute section's "$21.87 metered", read from the billing file, with no reconciliation.
>
> **N10: a residual hand-typed number.**
> - report.py, line 729 (Mac section): "about 29, 57 and 113 updates". It is derived by hand from the diagnostics row above it.
> - This predates the fixes; I missed it in round 1.
> - **Fix (W):** render it from the JSON.
>
> **N11: an unexplained timing, for description only.**
> - compute.json shows the 30-step RTX training taking 2.95 h for 10 policies, about the same as the 60-step reference (2.98 h), with half the source epochs (54,043 against 112,080).
> - One sentence (card state or throttling?) would pre-empt a reviewer's question. No statistic depends on it.
>
> **Labels of the two AB4s.** They are now consistent across the summary, the RTX section, the checklist and the ST rows. The one exception is the trend rows of the ST table, which say "(registered; not of record)" without naming the amendment.
>
> **Section C, briefly.** The new exploratory table strengthens the negative result:
> - no LLM arm beats random search, regularized evolution, TPE, ASHA or random + referee, even by an unadjusted interval;
> - every GRPO arm is Holm-worse than each of these baselines;
> - the only Holm-significant "wins" are untrained prompted arms over the E4 portfolio, which is a transfer control with a different information set.
>
> RQ1 is negative on the ILI replay table.
>
> ## 3. Final verdict (for docs/verdicts/)
>
> > Verdict on the ablation section (docs/ablations.md at e62e88a), area-chair re-review, 26 Nov 2026. As a negative-results ablation section, the substance now meets the standard of a NeurIPS/ICML/ICLR main-track paper. Registered and post hoc analyses are separated and labelled. Multiplicity is controlled per registered family, with a programme-wide Holm sensitivity in which the anonymised-prompt and AB-L rejections do not survive (exploratory). Detectable effects are reported, together with their t-quantile optimism. Each result is scoped to the replay table (E3b's K7 failed on ILI), to ILI, and to ETTh1-to-ILI transfer. The two "AB4" amendments, the run timeline and the registered disclosure are reported openly, and the trained-versus-untrained result is worded to the valid inference: no dose-response, and ETTh1-trained policies are 1.59 SD [1.28, 1.90] worse on ILI than the untrained model. Two things still block acceptance. (1) The section cites an independent verification, docs/verdicts/ablations_rtx_reviews.md, that is not committed on this branch, and the verifier's scripts and outputs (results/ablations/verify_rtx/) are untracked, with no critic verdict on the AB4 values. Until they are committed, the verification claim is unsupported by the repository. (2) The computation of record that Diya's AB4 names (LLM-registered.json) does not exist, and which of the two AB4 amendments governs each element is unresolved. The owner must settle this by a dated amendment and run the computation once, and the section must then show agreement, or the paper must state that the registered families are reported from an independent computation pending that record. Small wording regressions should be fixed before submission but do not block: the failure-case AB-L clause, the e6926d8 timeline label, the hardware-check differences without intervals, the "93 favour" phrasing, and a garbled sentence. The scope limits (a 74-architecture space, two tables, one LLM family, validity for the table only) bear on the paper's significance, not on the integrity of this section.

### Results re-review

> # Re-review of Anirudh's results items (anirudh-ablations at e62e88a)
>
> I read everything through git: `docs/ablations.md`, `docs/figures/anytime_regret_ILI.svg`, `results/ablations/{anytime-ILI,compute,LLM-vs-baselines}.json` and `docs/neurips_checklist.md`, all at e62e88a. I checked the numbers by recomputing in memory from origin/main and from the RTX files at origin/soumitra-leg2 e6926d8. Nothing in the repo was edited.
>
> ## What I verified as correct
>
> - **Anytime figure.** The 9 SVG paths decode to the 8 series plus the random reference. They equal anytime-ILI.json, and my recomputation of random and the E4 league agrees to 1e-15. The table under the figure matches the E3 and E4 summaries at budgets 10, 25, 50 and 100.
> - **Failure-case table.** The RevIN-off shares are right: E4 league 99%, none 100%, E3 prompted 55%, random 50%, 0.5B 41%, RTX GRPO 96 to 100%. So are the regret floors on ILI: the best RevIN-off config is 1.03 SD from the best and the best RevIN-on config is 0.00 SD. On ETTh1 the best RevIN-on config is 6.52 SD from the best.
> - **Per-policy table.** All 30 step-level values match my recomputation. Every per-policy rho is right, as is the mean of +0.05. The sign-flip p of 1.000 is exact: 0.5 is the smallest |sum| any sign assignment can give.
> - **Analysis (i).** +1.59 is correct: the mean of 3.03, 3.26 and 3.17, minus 1.565.
> - **Hardware rows.** +0.12 and −0.24 are correct.
> - **Compute table.** The per-card totals equal the sum of the per-arm rows. The E4 league deployment's 0.49 h equals its records' 1,761 s. Source epochs scale with steps: 54k, 112k and 222k.
>
> ## Status per item
>
> | Item | Addressed? | Correct? | Notes |
> |---|---|---|---|
> | R1 (write-up) | Partly | Yes, where present | Every relevant section states the transfer framing and the different information sets. Checklist item 1 still omits the E4 result: the league ties the portfolio, loses to warm TPE, and every GRPO arm is worse than every within-table baseline. Restating RQ1 in the transfer form is Diya's decision. |
> | R2 | Yes | Yes | Numbers verified. Labelled descriptive, and no causal claim is made. See N5. |
> | R6 | Mostly | Yes | The report gives cards per arm, never pairs seeds across cards, states that the E3 runs predate the pins, and has a hardware table. Still missing: the detectable effect for the 10-seed prompted arm against the 30-seed baselines (about 0.56 SD). |
> | R7 | Yes | Yes | The per-policy table leads, with the caption "10 policies, not 30 values". It says that `ABL_G_ref_local` serves three comparisons, and the step trend is reported as registered with its anti-conservative p. |
> | R13 | Yes | Numbers yes, labels no | See N1 to N3. |
> | R14 | Yes | Yes | ETTh1 is labelled in-sample, no claim, everywhere. |
> | R15 | Yes | Yes | The scope sentence appears in every section. |
> | R16 | Partly | Yes | Done: a means-only anytime figure, the per-policy table, the failure-case table, the compute table and Welch differences. Still missing: intervals on the curves, the A3 noise-floor band, per-architecture regret (A15), and the detectable effect from R6. See N6. |
>
> ## New problems: no wrong number and no overclaim, but these need fixing (all wording, Anirudh)
>
> - **N1. Superseded runs in the compute table.** The rows "E4 training (train_ETTh1)" and "(train_ILI)" are the 11 policies trained on diyapr on 23 Nov, which A25 superseded. That is 3.38 h, included in the A10 and A10G totals and in the at-a-glance bullet. Label them "superseded (A25), never deployed, compute only".
> - **N2. Card load inflates the RTX timings.** After the 05:18 reboot the RTX card carried other load (c7e4bc9). As a result:
>   - `steps30` shows 2.95 h of training, against 2.98 h for 60 steps.
>   - `steps30` shows 348 s per search, against 119 s for 60 steps.
>   - anon on ETTh1 shows 1.26 h, against 0.30 h for ref.
>
>   A reader would conclude that 30 steps costs as much as 60. Footnote the load in the per-arm ILI table and in the compute table.
> - **N3. Two dollar figures.** The Compute section cites a $21.87 month-to-date billing file read on 25 Nov at 00:10. Checklist item 8, commit 22111da and STATE say $22.17 for 24 Nov. A later month-to-date figure cannot be lower than an earlier one. Cite both and ask Soumitra to reconcile them.
> - **N4. The PG controller is left out of the baseline sentence.** "No arm's interval lies below random, RE, TPE, ASHA, random + referee (among the RQ1 and E3 baselines)" leaves out the PG controller, which is an RQ1 and E3 baseline (A4). Two prompted arms beat it on unadjusted intervals: Mac repair by −0.37 [−0.64, −0.10] and RTX 0.5B by −0.30 [−0.56, −0.04]. They also beat E3 prompted and warm TPE on unadjusted intervals. None of these survives Holm.
>   - Say this explicitly, and add `pg_league` (1.59), which is the control matched under A4.
>   - "93 comparisons favour the baseline" means 93 Holm rejections in the baseline's favour (I counted 78 for GRPO and 15 for prompted), so say so.
> - **N5. Wrong basis for the ETTh1 floor.** The 6.52 SD floor on ETTh1 comes from a one-seed config, p8s8l2h8d16f1r3n1i, not from "the table's 3-seed means".
> - **N6. Colours in the figure.** Blue is random in the left panel and prompted 1.5B in the right panel. Keep each series' colour fixed across panels.
>
> ## Final verdict (to be quoted verbatim in docs/verdicts/)
>
> > Results re-review of anirudh-ablations at e62e88a (26 Nov): R2, R7, R14 and R15 are addressed and correct; R1, R6, R13 and R16 are partly addressed. Every number checked in the new anytime figure, failure-case table, per-policy table and compute table reproduces from the run files (figure curves to 1e-15; RevIN shares, regret floors, per-policy rho and the 1.59 SD training gap exactly), and no new statement overclaims for any LLM arm. Six wording defects remain: superseded 23 Nov runs counted as E4 compute without a label (N1); RTX timings inflated by shared load after 05:18, presented without a caveat (N2); two irreconcilable dollar figures (N3); the PG controller left out of the "no arm beats" sentence, and "93" not said to be Holm rejections (N4); a one-seed ETTh1 floor described as a 3-seed mean (N5); and colours that change meaning between figure panels (N6). Curves with intervals, the A3 noise-floor band, per-architecture regret and the 10-seed detectable effect are still owed. The ablation report is fit to cite as exploratory, secondary evidence once N1 to N5 are fixed; it is not a claim under CLAUDE.md rule 6.
>
> ## Still open for Diya and Soumitra
>
> - **Diya.**
>   - **R1:** decide the paper's RQ1 framing (transfer form, A15).
>   - **R3:** run pick batch 2, and fix the membership of family 3 before it is computed.
>   - **R4:** fix `replay.py` so the Holm families use the registered sizes; add AUC intervals; regenerate and commit the E4 summaries and `docs/claims/E4.json`; run G4; decide whether rule 6 applies to losses, or add the window-block bootstrap.
>   - **R5:** rule which AB4 governs; relabel 2ab2996 and correct its "before any run" and "What existed" text; choose one analysis script and one `llm_tests.py`.
>   - **R8:** disclose that A29 was committed mid-run.
>   - **R9:** push each registration to the remote before running.
>   - **R10:** add provenance sidecars for the E3 records.
>   - **R16:** produce the numbers typed into verdicts and STATE through `report.py` (rule 5).
> - **Soumitra.**
>   - **R3:** recompute families 1 and 2 after the picks.
>   - **R5:** state in writing whether `rtx_tests.py` was run; run `llm_registered.py` once, as the computation of record.
>   - **R9:** rerun `e4_analysis.py` on a clean checkout of 7808ea6 and confirm the output is byte-identical.
>   - **R10:** document 894b65a ↔ e6ce92f and the tag `e4-j1-launch`; list the lost training records.
>   - **R11:** describe e6926d8 in a follow-up commit; untrack `l4_queue.lock`; keep a single copy of the RTX records and one manifest.
>   - **R12:** add a Welch or studentized sensitivity check beside each test; add a direction field to the Holm block.
>   - **N3:** reconcile the $22.17 and $21.87 figures.

### Verifier: check of the fixer's round 1

> **VERDICT: FAIL.** One blocking item, and it is quick to fix: the report cites a verdict file that is not on the branch. Everything else passes.
>
> **Blocking:**
> - docs/ablations.md (3 times) and docs/neurips_checklist.md (2 times) cite docs/verdicts/ablations_rtx_reviews.md. That file is not on anirudh-ablations; it exists only on local/ab2-provisional-do-not-push at d577ef8, and holds only the AB2 provisional reviews.
> - The fix is to commit it to anirudh-ablations before the push, with three things quoted verbatim: the AB2 reviews from d577ef8, the AB4 verification (113 of 113, and today's checks below), and my critic verdict. CLAUDE.md requires the critic's verdict to be copied verbatim into docs/verdicts/.
>
> **1. No registered statistic or decision changed: PASS.**
> - `registered.families` at HEAD equals 40804c7 and has the digest the report's gate expects (aee7655f).
> - My three-way comparison, rerun against HEAD's LLM-rtx.json, gives 113 of 113.
> - The status changes only add fields. All ten arms are complete with no problems, and final is True.
> - The B19 run-path check (leaguetst/, scripts/llm_local.py, requirements-llm.txt) is equal to c2dda8d at all seven run commits. It only fires after the existing descends and launcher checks, so it adds no new way for a current arm to fail.
> - llm_tests.py is 08c72a4c and common.py is 46aef3c, both equal to main.
> - In ST-power, no existing field changed in any row. The RTX rows were only relabelled, and the t-quantile column is additive and plausible (ratio about 1.04).
>
> **2. The new numbers are right: PASS.**
> - compute.json matches direct sums from the run files: seconds, target epochs and source epochs. The RTX totals of 10.1 h and 18.5 h check.
> - programme-holm.json: all 13 p values and family Holm values match their source files, and the ordering and adjustment recompute exactly. Only the steps and scale trends are rejected; anon is 0.0800.
> - anytime-ILI.json: each curve's mean over budgets equals the arm's AUC (1.5649, 3.2600, 3.1428, random 1.5151, TPE 1.1367).
> - Per-policy AUCs: 0 mismatches against my own code. The raw-MSE column is AUC × 0.0097869.
>
> **3. docs/ablations.md: PASS apart from the missing file.**
> - Regenerating it at e62e88a in a scratch worktree gives a byte-identical file.
> - The decimals in report.py's strings are thresholds and settings only. The typed "0.10 to 0.15" figure is gone.
> - The status wording is right.
> - The verification sentence is gated on the pin and the families digest.
> - The timeline matches both status files.
> - Diya's disclosure is verbatim from 2ab2996.
>
> Wording to fix (not blocking):
> - "Confounded by design" for repair mislabels the registered arm definition. Say what it measures instead.
> - "Which matters for reading the scale trend" is vague; give the reason or drop it.
> - "It is in none of Diya's AB4 = origin/main at 2ab2996; …" is garbled.
>
> **4. Authorship: PASS.** All 17 commits in 3aff768..HEAD are authored and committed by Anirudh Katupilla, with no Co-Authored-By, Claude or Anthropic line.
>
> **Before the push:**
> - results/ablations/verify_rtx/ in the main tree is untracked and holds copies of my scratch files. Either remove it or commit it deliberately; recompute_ab4.py has absolute scratchpad paths, so as copied it will not run from the repo.
> - The condition I raised before still applies. Pushing publishes RTX values before LLM-registered.json exists, against LEG3's request and Diya's AB4. That is the user's call, and Diya should be told first.
> - Observation only: the 30-step arm ran at about half speed (training as long as 60 steps; 348 s per deployment search against about 120 s).
>
> Scratch worktrees to remove later, each with `git worktree remove --force` on its path:
> - [scratch]/direction/ab4/head_wt
> - [scratch]/direction/ab4/rtx_wt

### Fixer, round 2

> Round 2 is applied: the area-chair re-review (N2 to N11, the B8 nuance, the ST trend labels), the results re-review (RN1 to RN6, R6, R16) and the verifier's three notes. That took seven new commits on anirudh-ablations, with nothing pushed. The registered families in LLM-rtx.json are still identical to 40804c7, and both `pytest -q tests/test_ablations.py` and `pytest -q` exit 0.
>
> ## Commits (author Anirudh Katupilla; no Co-Authored-By, "Claude", "Anthropic" or "Generated with" line)
> - **ca3a0b8** llm_rtx.py: the timeline fixes (N2) and the git and reflog times needed for the B8 wording.
> - **af2f4f1** compute.py, llm_vs_baselines.py, power.py and anytime.py.
> - **67c8edd** report.py, with tests.
> - **b82e2e4** docs/neurips_checklist.md. Wording only.
> - **1c90bed** report.py: a verb agreement fix. Wording only.
> - **f922d0a** Outputs regenerated from a clean tree in the usual order, with compute.py, programme_holm.py and anytime.py run before report.py.
>
> Each message says the change was made after the results were seen and whether it is wording only.
>
> ## Area-chair re-review
> - **N1:** Left to you. I did not create or stage docs/verdicts/ablations_rtx_reviews.md or results/ablations/verify_rtx/, and the citation is unchanged.
> - **N2:** e6926d8 is now labelled for what it holds:
>   - duplicate run copies under results/runs/llm_local/;
>   - rtx_tests.py and its tests;
>   - copies of common.py and llm_tests.py;
>   - l4_queue.lock.
>
>   The label also says no AB4 run file is new in it. 185a0b1, c7e4bc9 and e880b0a are added with their git times.
> - **N3:** The AB-L clause is gone. The failure case now rests only on the RevIN shares.
> - **N4:**
>   - Both hardware tables (RTX and Mac) show the unpaired Welch t 95% interval. It comes from the JSON's n, mean and sd, by a helper in report.py that a test shows matches common.compare.
>   - The RTX J2 check is now +0.12 [−0.60, +0.83].
> - **N5:** Now worded as "N of M comparisons are Holm-rejected in the baseline's favour", computed from the JSON.
> - **N6:** The record note now lists each place checked for LLM-registered.json: origin/main, Diya's AB4, the pin and the working tree, each marked absent.
> - **N7:** The checklist now gives the two-way pin conflict: llm_rtx.py and llm_registered.py share 08c72a4c, while rtx_tests.py pins 01b70835.
> - **N8:**
>   - The full K7 sentence now appears once, in the header; each section has one short "Replay table only" line.
>   - The registered RTX bullet now comes before the exploratory baseline bullet.
> - **N9:** Checklist item 2 gains the K7 table-only limitation. Item 8 no longer types $22.17; it names both dollar records and their sources.
> - **N10:** "about 29, 57, 113 updates" is now rendered from the JSON.
> - **N11:** The compute section has one sentence: 30-step training took 2.95 h for 54,043 source epochs against 2.98 h for 112,080 at 60 steps, and the 30-step job finished after the reboot. The cause is under RN2 below.
> - **B8:** The llm_tests.py edits are timed against git and the reflog:
>   - 510633d (17:12 ET) came before docs/LEG3_soumitra.md existed (17:21 ET);
>   - a692f0e and f9c0d62 came after this clone fetched LEG3 (19:04 ET), so they went against its request;
>   - 943d893 restored the file.
> - **ST trend rows:** each now names its amendment: AB2 with the statistic fixed in AB5, AB2, or Soumitra's AB4 (15db163).
>
> ## Results re-review and R items
> - **RN1:** The 23 Nov train_ETTh1 and train_ILI rows are labelled "superseded (A25), never deployed, compute only" and totalled separately (3.38 h), in both the compute table and the summary bullet.
> - **RN2:** A footnote (†) marks the steps30 and steps120 seconds in the per-arm ILI table, and the anon and repair ETTh1 rows and the steps rows in the compute table.
>   - The reboot time comes from the INTERRUPTED queue line (boot 05:18).
>   - The shared load is cited from c7e4bc9's message, in words.
>   - No end of the slow period is recorded, and the footnote says so.
> - **RN3:** Both dollar figures are rendered from their sources: the billing file says $21.87 metered (read 25 Nov 00:10), and 22111da's message line says $22.17. The report says they differ, that the later reading is the lower one, and that Soumitra is asked to reconcile them.
> - **RN4:**
>   - `pg_league` is added to llm_vs_baselines.py as the A4-matched control.
>   - **This enlarges the exploratory Holm set from 162 to 180 comparisons.** Holm losses are now 101: 87 with a GRPO arm and 14 with a prompted arm (the reviewer's 78/15 was for the old set).
>   - The baseline sentence now lists, on unadjusted intervals only, prompted 0.5B (RTX) and repair (Mac) below the PG controller, E3 prompted and warm TPE, and only Mac repair below pg_league. It says none of these survives Holm.
> - **RN5:** The report now says the ETTh1 floor comes from one-seed configs.
> - **RN6:**
>   - Each series in the figure keeps one colour across both panels, and random is a grey dashed reference in both.
>   - Each panel now has three coloured series, because no larger set passes the all-pairs colour-vision check in both light and dark mode. To get there, I dropped the E4 J2 league line from the figure; it is still in the other tables.
> - **R6 (owed):** Added. E3's 10-seed prompted arm could detect 0.51 to 0.56 SD against the 30-seed baselines (common.detectable, descriptive). It is shown in an ST group and a sentence in the baseline section.
> - **R16:** 95% t bands are now drawn from the per-seed values. The report says the A3 noise-floor band and per-architecture regret are not done.
>
> ## Verifier notes
> - The repair arm is now described as defined: no constrained decoding, and over-cap proposals replaced by the nearest feasible unseen config. The report adds that 53% of its proposals went through that replacement.
> - "Matters for reading the scale trend" is replaced by the computed fact: 0.5B repeats 52% of its proposals, the most of any prompted arm, and a repeat costs no budget.
> - The garbled sentence is the same one as N6, fixed above.
>
> ## Checks
> - llm_tests.py still hashes to 08c72a4c.
> - No protected file changed between 40804c7 and HEAD.
> - results/logs/mps_queue.status and results/ablations/verify_rtx/ are untouched and untracked.
>
> ## Still with the owners
> - The O items from round 1: B1 O, B2 O, B20, B21 and B23.
> - From the results re-review: Diya's R1, R3, R4, R5, R8, R9, R10 and R16, and Soumitra's R3, R5, R9, R10, R11, R12 and the dollar reconciliation.

## What changed in response (Anirudh's side, 26 Nov evening)

Commits on anirudh-ablations after 3aff768, all made after the AB2 values were seen, and from a692f0e on after the AB4 values were seen:

* b411bac: the RTX section is labelled an independent computation, not the result of record. Written before any AB4 value was computed.
* a692f0e: the pin moves to e6926d8. Only RTX_REF changes.
* f9c0d62: the transfer items' training and error logs are read from results/ablations/rtx/ as well as results/runs/. The first AB4 run (kept as results/ablations/LLM-first-AB4-run.json) had counted the step arms as missing. Logged in docs/ISSUES.md.
* 523ae42 and 90e1d60: report wording after the verifier's check.
* 943d893: the RTX code moves verbatim to scripts/ablations/llm_rtx.py, so that llm_tests.py is byte-identical to the version Diya's AB4 pins (08c72a4c). A test checks the move and the pin.
* 5fb23c6: an exploratory, post hoc comparison of every LLM arm with every required baseline (llm_vs_baselines.py).
* 40804c7: outputs.
* a516ec9 to e62e88a: the fixer's round 1, applying the area-chair review and the results review.
* ca3a0b8 to f922d0a: the fixer's round 2, applying the re-reviews and the verifier's notes.

Not done here, and left to their owners:

* **Diya:**
  * a dated amendment saying which of the two AB4s (15db163, 2ab2996) governs each element;
  * the computation of record (llm_registered.py, LLM-registered.json);
  * the llm_tests.py pin of rtx_tests.py;
  * llm_registered.py's AB4 check;
  * the RQ1 framing;
  * the pending E4 families and G4.
* **Soumitra:**
  * rtx_tests.py's status and its "AB4 addendum 2" citation;
  * the duplicate run copies and the lock file in e6926d8;
  * the dollar figures.

Not done at all:

* the A3 noise-floor band and per-architecture regret;
* diagnostic (iii) of Diya's AB4;
* a same-seed rerun to measure GPU nondeterminism.
