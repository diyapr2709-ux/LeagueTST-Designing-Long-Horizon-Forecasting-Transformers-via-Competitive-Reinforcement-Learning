# AB-L: did GRPO learn? Verdict

Script: scripts/ablations/learning.py (56960ea). Outputs: results/ablations/AB-L.json
and AB-L.md, generated at 56960ea. Registration: docs/prereg_ablations.md AB-L and
AB1; docs/LEG2_soumitra.md L3. Secondary to the registered primary endpoints.

## Registered test (from AB-L.json)

League policies' Spearman slopes of reward_mean over training steps against zero,
exact two-sided sign flip over policies, Holm over the two source tables:

* ETTh1 source (J2): mean slope +0.202, p 0.0156, Holm 0.031, rejected.
* ILI source (J1): mean slope +0.077, p 0.40, not rejected (without the A29-retried
  grpo_league_ILI_seed109: +0.093, p 0.36).

## Verifier

PASS. Independent code (did not import the script); every number in AB-L.json
reproduces with no difference above 1e-9; input and script sha256 match; the
episode-spend assumption holds for all 43 policies. Caveats it asked to disclose
(policies log 57 or 58 updates, short final episodes, an unlogged budget-crossing
step 60) are now in AB-L.md.

## Changes made after the critic and the verifier, before this commit

The logged-update count, the final-episode spend undercount (bias toward GRPO), the
short final episodes, and an exploratory, unregistered position-adjusted slope
(ETTh1 league +0.194, 8 of 10 positive; ILI league +0.059, 5 of 10). The registered
numbers did not change.

## Critic's verdict (verbatim)

The registered AB-L test is implemented faithfully. It uses 57 logged updates per policy, not 60, because the three budget-exhausting steps make no update. On the ETTh1 source the league policies' reward_mean slopes are positive (mean +0.202, exact p 0.0156, Holm 0.031, rejected). The slope survives removing the within-episode position effect (+0.188, 8 of 10 positive), so it is not an artifact of episode structure. On the ILI source it is not rejected (+0.077, p 0.40), and it shrinks to +0.03 once position is removed. Allowed wording: "on the ETTh1 source, GRPO training raised the policies' mean reward; on the ILI source no trend was detected." Not allowed: any claim that GRPO searched better than prompted Qwen, because that comparison is untested and biased toward GRPO (A18 noise on ETTh1, and final-episode spend undercounted by up to about 90 epochs). Also not allowed: any claim that the league helped learning, since the none arm's slopes are as large or larger. Before the commit, state the 57-step count and the spend bias in AB-L.md, and regenerate the output so its recorded git hash is the commit's.

## Note on the critic's numbers

The critic's position-adjusted slopes (+0.188 ETTh1, +0.032 ILI) came from its own
scratch computation; the script's exploratory field, which pools the position mean
over each arm, gives +0.194 and +0.059. Both agree that the ETTh1 slope is not a
position effect. Logged updates are 57 or 58 per policy (verifier).
