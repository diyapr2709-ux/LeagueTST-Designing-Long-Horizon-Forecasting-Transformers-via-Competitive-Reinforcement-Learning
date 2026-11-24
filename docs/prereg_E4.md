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

## Amendment A22 (2026-11-23 18:54, before any E4 endpoint)

The check job measured the largest per-token gap between the sampler's and
the scoring pass's log-probabilities at 0.13 to 0.24 nats, the usual bf16
difference between cached generation and a fresh forward pass. The check
now stops a run only when the mean gap per token exceeds 0.05 (a different
sampler, such as a repetition penalty, shifts many tokens); both the mean
and the largest gap are logged.

## Amendment A25 (2026-11-24, before any E4 rerun result exists)

Of 45 GRPO policies trained on diyapr on 23 Nov, 34 ran out of GPU memory
(three processes per A10G). None of the 11 that finished was deployed. All
45 are retrained, and no result from the 11 is reported.

* Backend: the owner chose Soumitra's side. Every policy of the rerun (and
  the conditional uniform arm) comes from one backend, chosen before launch
  and written in the records: Soumitra's Modal workspace (A10G) or one local
  CUDA GPU (scripts/llm_local.py). The two are never mixed within the rerun.
* GPU sharing: two jobs per GPU (one on a card under 16 GB). Jobs are
  ordered seed by seed across arms, so arms share GPUs evenly.
* Environment: the image and requirements-llm.txt pin torch 2.15.0,
  transformers 5.24.0, peft 0.22.0, accelerate 1.16.0 and huggingface_hub
  2.4.0, the versions behind every LLM run so far; Qwen2.5-1.5B-Instruct is
  pinned to revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306. Every record
  carries the git hash, package versions, model revision, GPU name, backend
  and jobs per GPU.
* Relaunches keep finished adapters and deployed (method, seed) pairs, and
  never retrain a run that the on-policy check stopped (A22); such a run is
  reported as failed. Batch errors name their runs.
* ETTh1-target tests: the ETTh1 table has one seed per config, so AUC
  there is in-sample (A18). Every test whose deployment target is ETTh1
  (family 1 for ILI to ETTh1, family 2 for the ILI-source arms) is scored on
  fresh-seed picks instead: per deployment, the mean validation MSE over
  seeds 4001 and 4002 of its incumbent at budgets 25 and 50, two-sample
  seed-level permutation test per budget. Family sizes grow accordingly;
  Holm within each family as before. ILI-target tests stay on AUC.
* Pick batch 2 lists only the E4 incumbents at budgets 25 and 50
  (scripts/picks.py collect --budgets 25 50) and runs on diyapr.
* Code digests (sha256), checked before launch with
  `shasum -a 256 -c docs/e4_rerun.sha256`:

```
leaguetst/architect/prompt.py: c4297c7aabd47db3379c18f91dd53cf84bcaae14e9a35f3032ea2f6c026beca3
leaguetst/architect/constrained.py: 37ddedaf91b98bf3d5aad3712e9df0d04a1a1eda1f2cd2b55bdeae3ae923cb9a
leaguetst/architect/episode.py: c387b058dfeb9f3238c2c3d96c06cc42cdeadc7960e8eed2c9558fb7798832b2
leaguetst/architect/llm.py: 8059b8ea751f4dc0de2de6668a5be9861fa37b3d92474ae071650a47c6e4870c
leaguetst/architect/runner.py: 3e2de41345ddadf02c07ab9b5b1633b4ad84251b2ba765ff496aab346cb79e7d
leaguetst/search/replay.py: 9601281c9a6b402ba1620a8cdd4976aeb4ba1112e4587288b869e6f528ec5440
scripts/replay.py: f4b5a16f9077a98e04729f73010e4a8f11da85ed97ff68bedd34f205380c2eaa
scripts/modal_llm.py: e8aa3043afe6688ad6f72911f7208909f3cc746a79edad8a216d856785343a6d
scripts/llm_local.py: 7adbbc68f6849997196dbd06eb39bdc52db7765b3ab3aa620900f6497e2cfdfe
scripts/picks.py: 8a77ec0194390944b5bfdd312972258bb0c11fcf2b1dcde43f3894eadc1e759a
```

## Amendment A26 (2026-11-24, before any E4 rerun result exists)

From the critic's review of A25.

* Backend and the prompted baseline. K4 and family 3 compare GRPO with
  prompted Qwen. If the rerun uses a local GPU, prompted Qwen is rerun on
  the same card first (10 seeds per table, method qwen_prompted_local,
  `scripts/llm_local.py prompted`), K4 and family 3 use those runs, and the
  A10G prompted runs serve as a reproducibility check. On Modal the A10G
  prompted runs stay the baseline. Families 1 and 2 need nothing extra:
  their controls run on CPU and every GRPO arm uses one backend.
* Failed runs. Tests use the seeds that completed; failures (on-policy
  stops, errors) are reported per arm. A worst-case sensitivity analysis
  goes beside each test: every failed seed of the arm under test takes the
  worst value observed in either arm of that test, every failed seed of the
  comparator the best, and the conclusion is reported with and without.
* Cut-off. If the rerun is not complete (every job finished or failed) by
  26 Nov 18:00 ET, E4 is reported as incomplete and is not tested on a
  partial set.
* Version evidence. The pinned versions were read on 24 Nov from diyapr's
  cached image of the unpinned definition, unchanged since the 23 Nov LLM
  runs; records from before A25 carry no versions, so this rests on the
  image cache, not on the records.
* Records also carry the CUDA runtime, cuDNN version and compute capability.
* Code digests replace those of A25 (docs/e4_rerun.sha256):

```
leaguetst/architect/prompt.py: c4297c7aabd47db3379c18f91dd53cf84bcaae14e9a35f3032ea2f6c026beca3
leaguetst/architect/constrained.py: 37ddedaf91b98bf3d5aad3712e9df0d04a1a1eda1f2cd2b55bdeae3ae923cb9a
leaguetst/architect/episode.py: c387b058dfeb9f3238c2c3d96c06cc42cdeadc7960e8eed2c9558fb7798832b2
leaguetst/architect/llm.py: 146a7e43cda82314fcd2a371a667aa366c72abb6a73d942912458d60babf28e2
leaguetst/architect/runner.py: 2dd62ad3eae4f45cd5d084a0de5468b019c5282f3d3169ccc8f19552bc324d45
leaguetst/search/replay.py: 9601281c9a6b402ba1620a8cdd4976aeb4ba1112e4587288b869e6f528ec5440
scripts/replay.py: f4b5a16f9077a98e04729f73010e4a8f11da85ed97ff68bedd34f205380c2eaa
scripts/modal_llm.py: e8aa3043afe6688ad6f72911f7208909f3cc746a79edad8a216d856785343a6d
scripts/llm_local.py: e7518498f9a6feac18841e1c2ed68a539ce052cbb891317d019bd8282c908170
scripts/picks.py: 8a77ec0194390944b5bfdd312972258bb0c11fcf2b1dcde43f3894eadc1e759a
```

## Amendment A27 (2026-11-24, before any E4 rerun result exists)

From the verifier's run_plan scenario tests. A record's environment block
is written inside the error guard, so a failure there cannot lose a
training record. An adapter is reused only if its provenance (backend and
model revision) matches the plan, so "all 45 retrained" (A25) holds even on
a volume that already holds adapters. A lock file refuses a second launch on
the same output. A deployment skipped for a missing adapter is logged.
scripts/picks.py analyze leaves out any (method, budget) cell whose picks
are not all reseeded (E4 at budget 100 after batch 2), so no selected subset
enters Holm. Code digests replace those of A26 (docs/e4_rerun.sha256):

```
leaguetst/architect/prompt.py: c4297c7aabd47db3379c18f91dd53cf84bcaae14e9a35f3032ea2f6c026beca3
leaguetst/architect/constrained.py: 37ddedaf91b98bf3d5aad3712e9df0d04a1a1eda1f2cd2b55bdeae3ae923cb9a
leaguetst/architect/episode.py: c387b058dfeb9f3238c2c3d96c06cc42cdeadc7960e8eed2c9558fb7798832b2
leaguetst/architect/llm.py: 51eca37a89a786180d29038739e970efb6004526f866e69289f2c653780d62ba
leaguetst/architect/runner.py: a5fb1a59bea7b68925e38f0af7655d75691ae417cdecd9daf661e9bd47093759
leaguetst/search/replay.py: 9601281c9a6b402ba1620a8cdd4976aeb4ba1112e4587288b869e6f528ec5440
scripts/replay.py: f4b5a16f9077a98e04729f73010e4a8f11da85ed97ff68bedd34f205380c2eaa
scripts/modal_llm.py: e8aa3043afe6688ad6f72911f7208909f3cc746a79edad8a216d856785343a6d
scripts/llm_local.py: e7518498f9a6feac18841e1c2ed68a539ce052cbb891317d019bd8282c908170
scripts/picks.py: f2145f0c6fcb0331954d9e5a9c5c084ae3df7a0cce4353112bb662d98e484f46
```
