# Reproducing LeagueTST from a fresh clone (L5 audit)

Audit of docs/LEG2_soumitra.md L5. It was run on 27 Nov 2026, 14:52 to about 15:30 ET, by Soumitra's
Claude session. The clone was of origin/main at its newest commit that day, 2ab2996, in a new folder outside
the working repository, on WSL2 Ubuntu (Linux 6.6) with Python 3.12.3. Nothing from the audit clone was
committed. Every number below is a wall-clock time measured in that clone.

## 1. Setup (README.md "Setup", followed verbatim)

```
git clone <origin> LeagueTST && cd LeagueTST          # origin/main, 2ab2996
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt        # 51 s (pip's download cache was warm; a cold cache downloads torch 2.15.0)
.venv/bin/python -m pytest                           # 42.6 s, exit 0
```

The steps worked as written. The README is missing two things:

* The ablation tests (tests/test_ablations.py) need `requirements-ablations.txt` (smac 2.4.1, dehb 0.1.2,
  ConfigSpace 1.2.2, which needs numpy below 2), and the README never mentions that file. Without it, 4 tests
  are skipped (ConfigSpace 1, smac 3), and so is the MPS test on any machine without an Apple GPU. AB-X
  needs its own environment, as amendment AB3 says.
* The LLM extras (`requirements-llm.txt`, `requirements-modal.txt`) are not in the Setup section. They are
  needed only for the GPU steps in section 3.

No data file is needed or fetched. Only data/trainval/MANIFEST.json is tracked, and every table comes from
results/ledger.jsonl.

## 2. Regenerating the CPU-derived outputs

Run in the clone, in this order. Every command exited 0.

| Output | Command | Time |
|---|---|---|
| E3 replay summary, ILI | `.venv/bin/python scripts/replay.py --item E3 --table E1 --dataset ILI --horizon 24 --summarize-only` | 13.1 s |
| E3 replay summary, ETTh1 | `... --item E3 --table E1 --dataset ETTh1 --horizon 336 --summarize-only` | 2.2 s |
| E4 replay summary, ILI | `... --item E4 --table E1 --dataset ILI --horizon 24 --summarize-only` | 3.6 s |
| E4 replay summary, ETTh1 | `... --item E4 --table E1 --dataset ETTh1 --horizon 336 --summarize-only` | 1.9 s |
| Picks analysis | `.venv/bin/python scripts/picks.py analyze --dataset ILI --horizon 24` (and ETTh1 336) | 16.1 s, 17.3 s |
| E5 | `.venv/bin/python scripts/e5_referee.py --dataset ILI --horizon 24` (and ETTh1 336) | 11.2 s, 11.0 s |
| AB-L | `.venv/bin/python scripts/ablations/learning.py` | 1.0 s |
| Report draft | `.venv/bin/python scripts/report.py` | 0.4 s |
| Gates | `.venv/bin/python scripts/gates.py --all` (it refuses to run without `--item` or `--all`) | 5 min 6 s (exit nonzero, see below) |

The CPU searches reproduce exactly. A full E3 ILI replay
(`scripts/replay.py --item E3 --table E1 --dataset ILI --horizon 24 --seeds 30`, 52 s) wrote a cpu_runs.jsonl
byte-identical to the committed one. With `--out` set to a folder outside the repository, the script then
fails in its summary step (`out_dir.relative_to(ROOT)` in summarize). That is a usability bug only: the runs
file is written first, and the default `--out` is unaffected.

Gates: `gates.py --all` checks 7 items: AB-L, AB2, AB4_registration, E4_independent_soumitra,
ablations_cpu_reviews, ablations_llm_reviews and arena_credit_2026-11-23. G1 to G5 pass for all 7; G1 runs
the suite (242 passed, 5 skipped). G6 fails for all 7 with "last verdict line: none". G6 requires
docs/verdicts/<item>.md to contain a line that is exactly "Verdict: ACCEPT", and none of these leg-2 verdict
files, from any of the three authors, uses that line. This is a format gap, not a failed review. The audit
did not stamp any verdict file; see docs/ISSUES.md.

## 3. Outputs that need a GPU (not reproducible on CPU)

* The tables E1, E3b and E6: `scripts/modal_app.py` on Modal (profiles diya-027 for ETTh1, diyapr for
  ILI), collected with `scripts/collect_runs.py`. Their rows are in results/ledger.jsonl.
* The prompted and GRPO LLM runs of E3 and E4: `scripts/modal_llm.py` (Modal A10G).
* The RTX ablation arms of AB2 and AB4: `scripts/ablate_llm.py`, through `scripts/ablations/run_l4_queue.sh`
  and `run_ab4_queue.sh`, on one RTX 4080 Laptop GPU. Their records are in results/ablations/rtx/.
  Measured times: a 30-seed prompted search takes 18 to 100 min, and a GRPO policy takes 15 to 43 min,
  depending on steps and on other load on the card.
* The fresh-seed pick trainings that `scripts/picks.py collect` lists (owner's Modal workspace).
* The test evaluation, once, by the owner: `scripts/prereg.py`, then `scripts/eval_test.py`.

Every analysis of these records runs on CPU from the committed files, as in section 2.

## 4. Discrepancies (what changed in the clone after section 2)

`git status` in the clone listed 10 modified files and 1 new one. None of the changes comes from
nondeterminism. Each comes from a committed derived file that was generated before records it depends on
were committed, so regenerating it adds rows. Every existing number was unchanged, except where noted.

| File | Change | Cause |
|---|---|---|
| results/search/E4/{ILI_T24,ETTh1_T336}/summary.json and .md | rows added for the GRPO arms; 0 lines deleted | the summaries predate the E4 rerun's GRPO runs (24 Nov) |
| docs/claims/E4.json | new file | `replay.py` writes a claims file for E4, and none is committed on main |
| results/picks/{ILI_T24,ETTh1_T336}_analysis.json | a "left_out" block for the GRPO arms; 0 lines deleted | GRPO picks without fresh-seed runs yet (pick batch 2) |
| results/e5/{ILI_T24,ETTh1_T336}.json | GRPO methods added; the interval level moves from 0.95 to 0.983 (k = 3 methods), so the interval bounds of the existing methods widen | E5 predates the E4 rerun's LLM runs |
| results/ablations/AB-L.json | git_head only (56960ea becomes 2ab2996) | provenance field |
| docs/REPORT_DRAFT.md | E3 ETTh1 and E4 sections added; E3 ILI pg_league and pg_uniform_rivals rows updated; E6 and baseline rows updated | the draft (1b6bd52, 23 Nov 18:25 ET) predates the A20 replay (f0a4808, 18:46 ET), whose numbers the committed summary.json holds, and the later ledger rows (24 Nov) |

To bring main up to date, rerun section 2 on main and commit the regenerated files. docs/claims/E4.json and
the report are the owner's (CLAUDE.md rule 5). Nothing was committed from the audit clone.
