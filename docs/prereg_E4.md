# Pre-registration: LLM architect (E3 LLM arms and E4)

Written on 2026-11-23 before any LLM result existed (the first prompted run
and a GRPO smoke test were stopped before any episode finished, because
they used a repair rule that this design replaces). Amendment A18 in
docs/PLAN.md points here.

## Fixed settings

| Setting | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B-Instruct, bf16, A10G |
| Decoding | constrained to the canonical JSON of the 2,368 feasible configs (token trie), temperature 1.0 |
| Group | 8 proposals per step; a proposal already in the episode history is wasted (scores the group minimum minus one, costs nothing) |
| GRPO | LoRA r 16, alpha 32, dropout 0.05 on all attention and MLP projections; AdamW lr 1e-4; KL weight 0.04 (k3 estimator against the base model); one on-policy update per group; gradient clip 1.0 |
| Training | 60 steps on the source table, in search episodes of 100 full-run equivalents (3,000 epochs); snapshots every 5 steps |
| Rival arms | league: 4 free rivals from the episode's evaluated configs, weighted by a random past snapshot's sequence probability; uniform: 4 free rivals drawn uniformly from the same set; none |
| Seeds | league 10, none 10, uniform 5 (A17 cut order), per direction |
| Deployment | frozen policy, one search per seed on the target table, budget 100 full-run equivalents |
| Prompted Qwen | the base model with the same prompt and decoding, 10 search seeds per table |
| Lookups | through leaguetst.search.replay.Oracle: one seed per config drawn from (search seed, config id); on ETTh1 (one seed for most configs) a fixed Gaussian offset with the pooled seed SD is added to every lookup |

Code digests at registration:

```
leaguetst/architect/prompt.py: c4297c7aabd47db3379c18f91dd53cf84bcaae14e9a35f3032ea2f6c026beca3
leaguetst/architect/constrained.py: 37ddedaf91b98bf3d5aad3712e9df0d04a1a1eda1f2cd2b55bdeae3ae923cb9a
leaguetst/architect/episode.py: c387b058dfeb9f3238c2c3d96c06cc42cdeadc7960e8eed2c9558fb7798832b2
leaguetst/architect/llm.py: 824623a0ee340087bb82c225e83ae93f159df06310fd5d7ead5d2f3f52db51ed
leaguetst/search/replay.py: 7028405077c071a6d95fbdb02d49cb3769d7772c2b48f54e4b70361550e58bbe
```

## Endpoints and tests

Regret of the incumbent is scored on the seeds the search never saw, in
pooled seed-SD units, at budgets 25 and 50 (headline) and as the area under
the anytime-regret curve (AUC). Search seeds are independent; comparisons
are seed-level permutation tests.

* Family 1 (primary, transfer, A12 and A13): the frozen league policy
  against the portfolio of the source table's ranking and against TPE
  warm-started with the source's top 10, on AUC, per direction. Four tests,
  Holm.
* Family 2 (league, A5 as redesigned): league against none and league
  against uniform rivals, on the deployed policies' AUC, per direction.
  Four tests, Holm.
* Family 3 (secondary): prompted Qwen against random search and TPE on each
  table at budgets 25 and 50, Holm with the CPU methods' comparisons.

K3 is read from the training logs: after 10 steps, invalid proposals at
most 20% (zero by construction under constrained decoding) and unique
configs per group at least 30%.

Ablation (5 seeds, ILI): prompted Qwen without constrained decoding, with
over-cap proposals repaired to the nearest unseen feasible config.

## Interpretation fixed in advance

A transfer win requires the league policy to beat both the portfolio and
warm-started TPE after Holm. If the two tables' rankings correlate highly,
the portfolio is expected to be hard to beat; a loss is reported as such.
"Competitive" is claimed only if the league beats both none and uniform
rivals after Holm.

## Amendment A19 (2026-11-23 18:28, before any E4 or prompted result)

A GRPO smoke test with 40-equivalent episodes showed step time growing from
8 to 37 seconds as the league scored every evaluated config under a past
snapshot. Both free-rival arms (league and uniform), in the LLM and in the
CPU controller, now draw from the 32 configs evaluated most recently in the
episode. Nothing else changes.
