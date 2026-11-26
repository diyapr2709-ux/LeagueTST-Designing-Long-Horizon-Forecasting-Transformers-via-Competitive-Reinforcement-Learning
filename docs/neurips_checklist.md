# NeurIPS paper checklist (draft)

Drafted 2026-11-25 from the repository. Question wording is shortened; check it
against the call the paper is submitted to. Each answer points to the file that
supports it. "Open" marks what the authors must still do or decide.

| # | Question | Answer | Where |
|---|---|---|---|
| 1 | Do the claims in the abstract and introduction match the contributions and scope? | Open (left to the authors after the 30 Nov test). The supported claims are null or negative: no search method beats random search on fresh seeds (RQ1 for the CPU methods), no league effect is detected, and the space is a small PatchTST family (A10, A11, A15). The abstract must not claim a win, "competitive" learning, or transformer design. | docs/STATE.md; docs/PLAN.md A10, A11, A15; results/picks/*_analysis.json; results/independent/E4_soumitra.json; docs/ablations.md |
| 2 | Are limitations discussed? | Yes, to be gathered in one section: 74 architectures crossed with 32 training settings; ETTh1 has one seed per config, so its replays are in-sample; tables give validation error only; one LLM family; results of the Mac arms are exploratory; AB-L's registered test is too liberal. | docs/feasible_set.md; docs/prereg_E4.md A18; docs/prereg_ablations.md AB1, AB3; docs/ablations.md |
| 3 | Theory: assumptions and proofs? | Not applicable; there are no theoretical results. | |
| 4 | Can the main results be reproduced from the paper and its supplement? | Partly. CPU analyses re-run from the committed ledger and scripts; the ablation outputs are committed with them. GPU runs need the listed hardware and their own rerun. docs/REPRODUCE.md (Soumitra's audit) is still to come. | results/ledger.jsonl; scripts/; docs/HANDOFF_soumitra.md; docs/REPRODUCE.md (pending) |
| 5 | Open access to data and code? | Partly. Code and result tables are in the repository. The original benchmark files must be downloaded by the user: ETDataset's licence (CC BY-ND 4.0 as recorded in the prereg; to be checked on its page) does not allow redistributing the split files. | scripts/download_data.py; scripts/make_trainval.py; docs/prereg_ablations.md Phase B item 3 |
| 6 | Are the experimental settings specified? | Yes. | configs/; docs/PLAN.md; docs/prereg_E4.md; docs/prereg_ablations.md |
| 7 | Are error bars or significance tests reported and defined? | Partly. Seeds per arm, intervals and what they resample are stated, and each family's Holm size is registered; but family 3's composition is still owed (docs/verdicts/E4_independent_soumitra.md), and AB-P and AB-G have registered families on Soumitra's RTX (AB2, AB4; provisional until AB4 lands or 28 Nov 23:59 ET), beside exploratory Mac runs. | leaguetst/eval/stats.py; docs/ablations.md (ST sections); docs/prereg_ablations.md |
| 8 | Are the compute resources reported? | Partly. Epoch charges and GPU time per record are in the ledger and run files; the E4 rerun cost $22.17. The Mac arms and the RTX arms (AB2, AB4) record seconds per search and per policy. Total compute, including failed runs (34 of 45 policies on 23 Nov), still has to be summed. | results/ledger.jsonl; docs/STATE.md (24 Nov); results/ablations/mps/; docs/LEG2_anirudh.md |
| 9 | Does the work follow the NeurIPS code of ethics? | Yes to the authors' knowledge: public electricity-transformer and influenza-surveillance series, no personal data. | data/README.md |
| 10 | Broader impacts? | Open (left to the authors after the 30 Nov test): one short paragraph. Likely points: forecasting errors in health surveillance, compute cost of search, and that a null result discourages spending on LLM-driven search in small spaces. | |
| 11 | Safeguards for high-risk data or models? | Not applicable; LoRA adapters of an existing open model trained to write forecaster configs. | |
| 12 | Are existing assets credited and their licences respected? | Partly. Model licences, read on 2026-11-26 from the LICENSE files at the pinned revisions and the licence field of each model card (the fetch report is in docs/verdicts/model_licences.md): Qwen2.5-0.5B-Instruct (7ae5576) and Qwen2.5-1.5B-Instruct (989aa79) are Apache 2.0; Qwen2.5-3B-Instruct (aa8e725, used only in AB-P scale) is under the Qwen Research License (non-commercial use only, s.2a; a model trained on its outputs and distributed must say "Built with Qwen" or "Improved using Qwen", s.4b). Still open: PatchTST, DLinear, RevIN, ETDataset, the CDC ILI data and the software baselines (SMAC3, DEHB, ConfigSpace, Optuna, scikit-learn, transformers, peft) must be cited with licences. | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE ; https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE ; https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE ; model cards (current page, not pinned): https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct , https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct , https://huggingface.co/Qwen/Qwen2.5-3B-Instruct ; docs/related_work.md; requirements-*.txt |
| 13 | Are new assets documented? | Planned: the two result tables as a benchmark with a loader and Croissant metadata (Phase B item 3). Not part of this submission unless finished. | docs/prereg_ablations.md Phase B |
| 14 | Crowdsourcing and human subjects: instructions and pay? | Not applicable. | |
| 15 | IRB approval? | Not applicable. | |
| 16 | Declaration of LLM usage? | Yes: Qwen2.5-1.5B-Instruct, prompted and GRPO-trained, is the method under study. Separately, AI coding assistants wrote and ran much of the code and drafted documents under the authors' direction; the verifier and critic passes were AI subagents too, not independent people. The authors decide how to state this under the call's policy. Verdict files: AB-L and E4 (docs/verdicts), and the ablations' CPU and Mac LLM parts (docs/verdicts/ablations_cpu_reviews.md, docs/verdicts/ablations_llm_reviews.md). | docs/prereg_E4.md; leaguetst/architect/; CLAUDE.md; docs/HANDOFF_*.md |

## Open items for the authors

* Authorship: docs/ISSUES.md (23 Nov) leaves the author list open and README.md
  still calls this a solo project.
* Test-row contacts are disclosed in docs/ISSUES.md and should be in the paper.
* The vault step (K0) was open on 23 Nov; say when it was completed.
* Deviations from registration to list in an appendix: A1 to A30, AB1 to AB3,
  including the E4 rerun after out-of-memory failures, the two retried policies
  (A29), the A24 margin set after prompted Qwen was seen, and the Mac arms.
* docs/STATE.md quotes a cross-table rank (1,828) that results/table_stats.json
  now gives as 1,341.
* AB-P and AB-G ran on an Apple M4 Pro (torch MPS), not on the registered Modal
  or CUDA backend; they are exploratory (docs/prereg_ablations.md, AB3).
