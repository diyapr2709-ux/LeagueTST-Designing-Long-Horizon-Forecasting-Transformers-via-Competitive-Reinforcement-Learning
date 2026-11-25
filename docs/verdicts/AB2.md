# Amendment AB2 (docs/prereg_ablations.md): verdict

AB2 registers, before any run, the local-RTX backend for the model-scale (AB-P)
and KL-weight (AB-G) families, the model pins, the launcher
scripts/ablate_llm.py and the queue scripts/ablations/run_l4_queue.sh (both
committed in 79977de after two code-verifier passes), the six items and the
rules for hand-stops, stale locks, retried runs, completion and missing arms.

## Review history

1. Verifier: FAIL on one wording error (the queue-order sentence); every
   hash, pin, card and software fact, item, seed and rule checked out. Fixed.
   It could not verify the 3B memory figure from a committed file; the probe
   output is now results/logs/ab2_3b_memory_probe.txt.
2. Critic: may be committed and launched once seven changes are added
   (verdict below). All seven were added; the billing evidence is
   results/logs/ab2_modal_billing.txt.
3. Re-verifier (fresh): every required item present and every fact accurate;
   FAIL on three wording corrections (arm as AB1's unit, partial trends not
   tested, the started-run rule). Applied verbatim.
4. Final check (fresh): VERDICT: PASS (all three corrections present, no contradicting sentence, append-only, no result exists, launcher and queue hashes match).

## Critic's verdict (verbatim)

AB2 may be committed and the queue launched tonight once these changes are added to the amendment before the commit:
- (1) Define hand-stops on record-file progress (prompted: no new `*_runs.jsonl` row for 1 h; GRPO: no step row or record for 2.5 h), each logged in the status file.
- (2) Define item completion by counts. A job with a non-stop error is relaunched with unchanged settings until 28 Nov 23:59 ET; if it is still unfinished, the item counts as not run (AB1), with no fallback settings.
- (3) Register the stale driver.lock procedure.
- (4) State that rerun searches are not "retried", and how a started-but-unrecorded run is identified.
- (5) Commit the billing and 3B memory-check evidence.
- (6) State that the Holm sizes stay at 3 (AB-P) and 2 (AB-G).
- (7) Launch only with HEAD at or after the AB2 commit.

No sign-off from Diya and no push is needed before launch. The backend choice (RTX for both families) follows the L4 rule.
