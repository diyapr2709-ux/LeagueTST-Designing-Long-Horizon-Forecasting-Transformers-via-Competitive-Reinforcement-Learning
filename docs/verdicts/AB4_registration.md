# Critic's verdict on the AB4 registration (2026-11-26, before any run of the AB4 arms)

Reviewed: amendment AB4 (docs/prereg_ablations.md), docs/LEG3_soumitra.md,
scripts/ablations/llm_registered.py, scripts/ablations/run_wp3_rtx_queue.sh,
in the working tree before the AB4 commit. The critic computed no value of
any run under results/ablations/rtx/ and ran llm_registered.py with --check
only. A separate verifier found the same two blocking defects (findings 1
and 2 below) independently.

## Blocking items (all applied before the commit)

1. llm_registered.py voided any GRPO arm with an on-policy stop or a
   relaunched crash (the runner's "no adapter to deploy" and crash rows).
   Fixed: an errors.jsonl row is a problem only if no stop and no later
   finish resolves it; tests build the folder with the real runner.
2. Retried runs were dropped from the trends, against AB1, AB2 and A29.
   Fixed: they stay as they finished; fills at the lowest and highest
   observed value are shown beside the trend, labelled as illustrative.
3. "Computed once" and the --final date were not enforced. Fixed: clock
   gate, refusal if the output exists, refusal on uncommitted records or
   code, digest check of common.py and llm_tests.py, time stamp; the queue
   stops every job at the deadline.
4. The --final path could not reach the report. Fixed: the GPU deadline is
   28 Nov 18:00 ET, leaving 18:00 to 23:59 ET for analysis and push; an
   unfinished ETTh1 table does not void its ILI arm.
5. The step levels are nested (shared seeds, no step-dependent schedule),
   so the registered trend's p is anti-conservative. Disclosed in AB4 with
   registered secondary policy-level analyses and a nesting diagnostic.
6. The exposure statement was inaccurate. Replaced as proposed, with
   Soumitra's attestation in his launch commit.
7. LEG3 said "Commit and push". Fixed: Soumitra pushes.

## Final verdict (verbatim)

AB4 is registrable after the blocking fixes. The reference switch is not a
forking path: the rule that an arm and its reference share a backend, AB1
and both LEG2 fallbacks, all committed before any ablation result, force
the local references, and the switch replaces a reference the author had
seen with one nobody has scored. As written, though, llm_registered.py
would void any GRPO arm with an on-policy stop or a relaunched crash. It
also drops retried runs that AB1, AB2 and A29 keep, enforces neither "once"
nor the --final date, and its --final output cannot reach the report. The
step levels are nested, so the trend's p is anti-conservative; disclose it.
Correct the exposure statement, then commit.
