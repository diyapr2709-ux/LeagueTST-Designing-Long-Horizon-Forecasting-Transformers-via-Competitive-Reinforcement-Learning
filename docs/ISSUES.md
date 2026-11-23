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
