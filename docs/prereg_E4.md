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

## Amendment A20 (2026-11-23 18:45, before any E4 endpoint)

The verifier found that generation inherited Qwen2.5-Instruct's
generation_config.json (repetition_penalty 1.1, top_k 20), so the sampled
distribution differed from the one GRPO scored, and LoRA dropout made the
update score a different network. The first ILI training run (about five
steps per policy, no deployment) and the first prompted-Qwen ILI run were
made under that sampler; both are discarded and rerun.

* Sampling uses an explicit configuration: temperature 1.0, top_k 0, top_p
  1.0, repetition penalty 1.0, stopping at the end-of-turn token. Every
  GRPO step checks that the scored log-probabilities match the sampler's
  (largest gap logged; a gap above 0.25 stops the run).
* LoRA dropout is 0.
* The controller's free rivals come only from configs evaluated before the
  current step, as in the LLM.
* Family 1 and family 2 are tested on AUC with Holm within each family; G4
  recomputes them from the runs files. Tests stay two-sample (not paired),
  as registered.
* ETTh1's pooled seed SD leaves out seed 2021, on which its multi-seed
  configs were selected; ETTh1 replays write no claims (A18).

Code digests after A20:

```
leaguetst/architect/prompt.py: c4297c7aabd47db3379c18f91dd53cf84bcaae14e9a35f3032ea2f6c026beca3
leaguetst/architect/constrained.py: 37ddedaf91b98bf3d5aad3712e9df0d04a1a1eda1f2cd2b55bdeae3ae923cb9a
leaguetst/architect/episode.py: c387b058dfeb9f3238c2c3d96c06cc42cdeadc7960e8eed2c9558fb7798832b2
leaguetst/architect/llm.py: 7a99cdefa1806cdd32abddbe34bd62ccd40c404482ecef033b71effecd5af392
leaguetst/search/replay.py: fb121a89f6fcc9c378388aa0b71b7d4ae7e08e5c271756d15e8b0b28c73a522e
scripts/replay.py: 8be98e21f611520f74b26d332e736d89fd874edcda1d936917bf409c26537265
```

## Amendment A21 (2026-11-23 18:52, before any E4 endpoint)

The on-policy check of A20 returned NaN (padding after the end-of-turn
token has -inf scores, and -inf times a zero mask is NaN), so it could not
fire. It now ignores padded positions. Sampling and training are unchanged;
a separate three-step job measures the gap for the record.
