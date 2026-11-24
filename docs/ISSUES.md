# Open issues

Problems found during the build that are logged instead of worked around.

| Date | Issue | Status |
|---|---|---|
| 2026-11-23 | The full CSVs, test rows included, are still in data/ETT-small and data/illness. Moving them to ~/leaguetst_vault and encrypting it is the owner's hand step (K0). results/revalidation/per_window_test_mse_*.npz and checkpoints/ also hold test-derived data from the October reproduction. | open, owner |
| 2026-11-23 | `run_experiment` (leaguetst/experiment.py) always scores the test split. It is kept only for the October reproduction records; the search path uses `leaguetst.table` on data/trainval, which cannot load test rows. | by design |
| 2026-11-23 | The 24 DLinear records from October lack `checkpoint_sha256`, so their checkpoints cannot be hash-checked. They are not used by the search. | open, minor |
| 2026-11-23 | The plan doc keeps the E0 prompt in a claude.ai chat that Claude Code cannot open. E0 was built from the doc's rescope and setup sections instead. | noted |
| 2026-11-23 | Team question: the progress report lists three authors while the plan doc treats the work as individual. The owner settles this with teammates and the instructor. | open, owner |
| 2026-11-23 | `pytest -q` together with `addopts = "-q"` hides the "N passed" line. gates.py reads the exit code, not the text. | noted |

## Contacts with test rows during the 23 Nov build session

The vault step (K0) was not done yet, so the full CSVs were readable. Every
known contact is listed here. None reached a table, a search, a model
selection or any score on test.

| Who | What | What was seen |
|---|---|---|
| Builder (main session) | `build_splits` and `build_dataloaders` on data/ while checking that data/trainval and the GPU loader reproduce the full pipeline | Test datasets were built in memory; only train and validation were compared or iterated. Nothing from test was printed. |
| Verifier agent, baseline audit | `build_splits` and `build_dataloaders` on data/ to check split borders and loader flags | Test window counts only (ETTh1 2785 / 2689 / 2545 / 2161, ILI 170 / 158 / 146 / 134), which the committed records already hold. |
| Verifier agent, milestone 1 | Printed the first 40 bytes of the first test-only row of each full CSV while checking the byte prefix | Part of one test row per dataset: ETTh1 "2017-10-24 00:00:00,9.979999542236328,3." and ILI "2016-10-25 00:00:00,0.733295,0.818527,15". |
| Verifier agent, milestone 2 | `load_test_split` for ILI T=24 with a scratch prereg file and log, to show the door could be bypassed | Window count only (170). The access is not in results/test_access.log because it went to a scratch log. |

Since then the agents are told never to load, slice, print or count test
rows, and to use the synthetic fixtures to demonstrate bypasses. The test
door now requires a committed prereg file and always logs to
results/test_access.log. The remaining protection is the owner's vault
step: moving and encrypting the full CSVs.
| 2026-11-24 | E4 GRPO training (diyapr, 23 Nov): 34 of 45 runs failed with CUDA out of memory, because three processes of about 7.6 GB shared a 22 GB A10G. Finished: 6 of 25 ILI-source runs (league 102, 105; none 101; uniform 101, 103, 104) and 5 of 20 ETTh1-source runs (league 100, 104, 108; none 108, 109). scripts/modal_llm.py now puts two jobs on a GPU and skips adapters that already exist. The reruns go to a teammate's Modal workspace (docs/HANDOFF_soumitra.md) because diyapr has about $5 left, most of it the E6 reserve. | open, rerun planned |
| 2026-11-24 | One ILI-source GRPO run stopped on the on-policy check (sampled and scored log-probabilities differ by 0.250, above A22's 0.05 mean-gap limit). The check is kept as is; a run that fails it again is reported as failed, not rerun with a looser limit. | noted |
| 2026-11-24 | Dead code in locked files stays (editing would break the E0 lock): trainval.scaled_values, datasets.SPLITS, test_split.PREREG_DIR. | by design |
| 2026-11-24 | CLAUDE.md rule 10 names only Diya's two workspaces. The owner decided that the E4 GRPO reruns run on Soumitra's workspace (profile soumitra); the GRPO app mounts no data volume, so nothing is uploaded. All 45 policies are retrained there; the 11 diyapr survivors are superseded. | decided, owner |
| 2026-11-24 | A guard blocked Claude from adding that decision to CLAUDE.md rule 10. Not worked around; the line to add is in docs/HANDOFF_soumitra.md. | open, owner |
| 2026-11-24 | Wider space (A16) cost model: Modal "A10G" requests also land on A10 hardware, and the A10G fit misses A10 rows by about 20% (median); any cost claim needs the record's gpu field. scripts/report.py does not yet regenerate docs/wide_space.md (run scripts/wide_space.py). The locked NumPy re-scorer knows only the flatten head, so a wide-space table would need it extended first. | open, after the report |
