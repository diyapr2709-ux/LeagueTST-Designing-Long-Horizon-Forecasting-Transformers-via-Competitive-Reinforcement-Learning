# External comparison: baselines and results against published work (lit-scout, 26 Nov 2026)

The lit-scout subagent's report, quoted in full; only local paths are masked. Every paper it cites was opened on 2026-11-26 (CLAUDE.md rule 9); its "Not opened" list must not be cited. The project's own numbers it quotes were read from docs/ablations.md and the E3 and E4 summaries at 40804c7; they are context, not a source (rule 5: the numbers come from scripts/ablations/report.py and scripts/report.py).

> # LeagueTST against published work: baselines, results, protocol (lit-scout, 2026-11-26)
>
> ## How this was read
>
> * Every paper cited here was opened in this session, on **2026-11-26**. There were 85 arXiv records in all. For each, the title, authors, date and abstract were pulled from the arXiv API (export.arxiv.org), which also confirms the id.
> * **[F]** means I also downloaded the full text and searched it for the passages quoted. The full text came from arxiv.org/html/<id> or, for older papers, ar5iv.labs.arxiv.org/html/<id>. Every number quoted from an [F] paper was read in that text.
> * **[A]** means the abstract only. Claims from [A] papers stay within what the abstract says.
> * AgentHPOBench has no HTML version. Its PDF text was pulled out with a small parser, so the layout is lost; its numbers come from the extracted table text.
> * Three works are not on arXiv: JAHS-Bench-201, Bergstra and Bengio (2012), and Demšar (2006). I opened their abstract pages, or for Demšar the first page of the PDF. Only that page could be extracted.
> * WP0 (commit 8b01d22, docs/related_work.md) was the starting point. Its entries that I did not re-open are under "Not opened" and must not be cited from this report.
> * The project's numbers come from three places: docs/ablations.md (working tree), results/search/E3/ILI_T24/summary.md and results/search/E4/ILI_T24/summary.md. I opened nothing under results/revalidation, results/baselines, data/ETT-small or data/illness.
>
> ## Our numbers being compared (ILI T = 24, held-out-seed AUC of regret, budgets 1 to 100, pooled seed-SD units, lower is better)
>
> | arm | AUC [95% CI] | seeds | source |
> |---|---|---|---|
> | TPE | 1.14 | 30 | E3 summary |
> | SMAC3 | 1.27 [1.12, 1.42] | 30 | AB-X |
> | ASHA | 1.38 | 30 | E3 |
> | regularized evolution | 1.47 | 30 | E3 |
> | epoch-k screening (groups of 8, top 2 promoted) | 1.50 | 30 | E3 |
> | Hyperband | 1.51 [1.28, 1.74] | 30 | AB-X |
> | **random search** | **1.52 [1.31, 1.72]** | 30 | E3 |
> | DEHB (69% of proposals repaired) | 1.53 [1.35, 1.72] | 30 | AB-X |
> | prompted Qwen2.5-0.5B (RTX) | 1.39 [1.23, 1.56] | 30 | AB-P |
> | prompted Qwen2.5-1.5B (RTX / Mac / A10) | 1.56 [1.38, 1.75] / 1.65 [1.41, 1.89] / 1.81 [1.42, 2.20] | 30 / 30 / 10 | AB-P, E3 |
> | prompted Qwen2.5-3B (RTX) | 2.06 [1.79, 2.32] | 30 | AB-P |
> | GRPO league, ETTh1 to ILI | 3.14 [2.50, 3.79] | 10 | E4 |
> | GRPO no league, ETTh1 to ILI | 3.23 [2.74, 3.73] | 10 | E4 |
> | portfolio of ETTh1's top configs | 3.28 | 30 | E4 summary |
> | TPE warm-started from ETTh1 | 1.65 | 30 | E4 summary |
>
> Other facts used below:
>
> * At budget 10 (E3), prompted Qwen-1.5B has regret 2.89 against 2.07 for random search.
> * On the pick reseeds, no method beats random search on either table after Holm.
> * 59 ILI configs lie within 1 SD of the best config.
> * Trained policies look up RevIN-off configs 96 to 100% of the time; the untrained model does so 55% of the time.
> * E4 league vs none: -0.09 SD, with a detectable difference of 1.25 SD.
> * League vs portfolio: -0.14 SD, detectable 1.10 SD.
> * RTX trained minus untrained: +1.59 SD [+1.28, +1.90].
>
> **Discrepancy for the authors.** docs/REPORT_DRAFT.md lists pg_league at 1.57 and pg_uniform_rivals at 1.57 in its E3 table. results/search/E3/ILI_T24/summary.md and the AB-Q table both give 1.59 and 1.61. The draft looks stale; regenerate it.
>
> ---
>
> ## Table of external methods and benchmarks
>
> All papers below were opened on 2026-11-26. "In LeagueTST?" means: is the method run, or the benchmark used, in this project?
>
> ### LLM-based HPO and NAS
>
> | Work | arXiv / URL | What it reports (as opened) | Relevance | In LeagueTST? |
> |---|---|---|---|---|
> | LLAMBO, Liu et al. 2024 [F] | 2402.03921, https://arxiv.org/abs/2402.03921 | GPT-3.5 inside BO, 25 trials, 5 seeded searches per task. Tasks are from Bayesmark, plus private and synthetic datasets "to address any memorization concerns". The baselines are GP-DKL, SKOpt, Optuna-TPE and SMAC3; there is no random-search arm in the end-to-end comparison. It is best overall and "excels in earlier stages". | The most-cited LLM-BO method; reviewers will expect it. | No |
> | Zhang et al. 2023, Using LLMs for HPO [F] | 2312.04528, https://arxiv.org/abs/2312.04528 | HPOBench, 32 tasks, 10 evaluations, temperature 0. Share of tasks that beat random search: GPT-4 Turbo 81.25%, GPT-4 68.75%, GPT-3.5 Turbo 43.75%, BO-RF 56.25%, BO-GP 50.00%. Random search is bootstrapped from 500 configs. Llama3-8B "could pick initial hyperparameters but was less effective during the search phase". | The main positive claim our null answers; it also shows a strong dependence on model scale. | No (prompted Qwen is the analogue) |
> | OPRO, Yang et al. 2023 [F] | 2309.03409, https://arxiv.org/abs/2309.03409 | Meta-prompt of past solutions sorted by score. Temperature 1.0 was best; temperatures of 0 to 0.5 "lack exploration". gpt-4 beat gpt-3.5-turbo and text-bison on TSP. | **Our prompted arm is an OPRO-style loop.** leaguetst/architect/prompt.py shows up to 24 past configs, best first, with their validation MSE, at temperature 1.0. Name it that way. | Partly (the design, not the code) |
> | Revisiting OPRO, Zhang, Yuan and Avestimehr 2024 [F, partial] | 2405.10276, https://arxiv.org/abs/2405.10276 | Small-scale LLMs (LLaMa-2 7B/13B/70B, Mistral 7B) are limited as self-optimisers for prompt optimisation. | Small-model caveat, from a different domain. | No |
> | AgentHPO, Liu et al. 2024 [F] | 2402.01881, https://arxiv.org/abs/2402.01881 | A GPT-3.5/GPT-4 agent on 12 tasks, compared with random search and BO, 10 runs each. | Agent-style LLM HPO. | No |
> | SLLMBO, Mahammadli and Ertekin 2024 [A] | 2410.20302, https://arxiv.org/abs/2410.20302 | An LLM-TPE hybrid sampler beats fully-LLM methods, and beats BO on 9 of 14 tabular tasks. | Hybrid LLM + TPE. | No |
> | Small LLMs with Expert Blocks, Naphade et al. 2025 [A] | 2509.15561, https://arxiv.org/abs/2509.15561 | With a trajectory summariser, 14B and 32B local models come within about 0.9 pp of GPT-4, with 10 trials. | Small-model HPO. | No |
> | GENIUS, Zheng et al. 2023 [F] | 2304.10970, https://arxiv.org/abs/2304.10970 | GPT-4 on NAS-Bench-Macro (6,561 archs), 10 iterations, 3 repeats. Its best pick ranks 8/6561 and its worst 61/6561. The random baseline is the best of 10 samples, averaged over 10,000 repeats. On NAS-Bench-201 it ran 5 trials at temperature 0 against one-shot NAS. | Prompted NAS on tables. | No |
> | EvoPrompting, Chen et al. 2023 [F] | 2302.14838, https://arxiv.org/abs/2302.14838 | 62B PaLM. "LMs perform poorly at designing novel and effective neural architectures via naive few-shot prompting"; evolution plus prompt-tuning is needed. | Says prompting alone is weak. | No |
> | LLMatic, Nasir et al. 2023 [F] | 2306.01102, https://arxiv.org/abs/2306.01102 | NAS-Bench-201 CIFAR-10, 2,000 evaluations, 10 runs. Its table: LLMatic 94.26±0.13, random search 93.70±0.36, GENIUS 93.79±0.09. | An LLM and random search on the same table. | No |
> | Rodrigues et al. 2026 [F] | 2606.21641, https://arxiv.org/abs/2606.21641 | Budget-matched study: 8 PMLB tasks × 5 seeds, 7 advisor models "nano-to-frontier". The advisor's strong first point is a fixed default config. Seeded random search ties it by 5 evaluations and leads by 12. The LLM adds -0.01 pp on test (p = 0.92). The effect of model size is "non-monotone". "We run many paired tests without a multiple-comparison correction". | **The closest published null.** | No (its default-seed control is not run) |
> | Ferreira et al. 2026 [F] | 2603.24647, https://arxiv.org/abs/2603.24647 | Fixed 14-HP space, 24 h budget, 3 seeds. TPE 0.9768, SMAC 0.9778, random 0.9873, LLAMBO-Optuna with Qwen 27B 0.9882. "Several pure LLM methods performed worse than random search". In the hybrid, Centaur with 0.8B (0.9763) beat Centaur with 27B (0.9766). | A null in a fixed space, with a scale effect that is not monotone. | No |
> | AgentHPOBench, Huai et al. 2026 [F, PDF text] | 2607.29626, https://arxiv.org/abs/2607.29626 | 30 executable tasks, 5 interventions. Open-weight agents run on 3 seeds, API agents once; baselines are random, TPE and BOHB. **On the 7 time-series tasks (iTransformer, TimeMixer, SparseTSF and TimeXer among them), random search scores 0.299 MBNS; the open-weight agents score 0.233 to 0.316, and only DeepSeek-R1-Qwen-14B is above random.** API agents score 0.266 to 0.403. Results carry task-bootstrap intervals and no tests. | Closest to ours in domain and model size. | No |
> | AgenticBBO-Bench, Chen et al. 2026 [F] | 2610.12183, https://arxiv.org/abs/2610.12183 | On the HPO family, a "Direct" LLM (DeepSeek V4.1 Flash) scores 0.519, random 0.273 and TPE 0.344. HPO runs use 5 + 25 evaluations, with held-out deployment seeds 2 to 5. The score is 70% the trajectory mean of normalised best-so-far plus 30% the final value. | Large-model positive; its score is close to our AUC. | No |
> | Huang et al. 2024, LLM black-box optimisation [A] | 2404.06290, https://arxiv.org/abs/2404.06290 | LLMs "exhibit subpar performance in pure numerical tasks". | General caveat. | No |
> | Greedy Is a Strong Default, Li 2026 [A] | 2603.27415, https://arxiv.org/abs/2603.27415 | An LLM proposer with greedy hill climbing, 3 runs; simulated annealing and a second LLM add nothing. | Hybrid baseline idea. | No |
> | SemanticOpt, Meindl et al. 2025 (v3) [F] | 2510.25404, https://arxiv.org/abs/2510.25404 | SFT of Qwen 3.5 27B on BO trajectories. After fine-tuning, the 9B model loses initialisation quality ("capacity-dependent forgetting ... can overwrite the pretrained priors"). | Fine-tuning can erase a small model's prior. | No |
> | Universal Searcher, Hu et al. 2026 [F] | 2611.12791, https://arxiv.org/abs/2611.12791 | An LLM trained on 228,000 trajectories ranks candidates from an evolutionary + random pool. GRPO (Qwen3-8B, 2,000 instances) is used only to add chain-of-thought. NAS-Bench-201: 5 runs × 200 steps; 46.51% on ImageNet16-120 against 46.79% for CoLLM-NAS. | Trained LLM searchers transfer, given very large corpora. | No |
> | NNGPT, Kochnev et al. 2025 [A] | 2511.20333, https://arxiv.org/abs/2511.20333 | An LLM AutoML engine with an RL pipeline among five. | Precedent for RL on an LLM architect. | No |
>
> ### RL-trained and learned optimisers, and RL on LLMs
>
> | Work | arXiv / URL | What it reports (as opened) | Relevance | In LeagueTST? |
> |---|---|---|---|---|
> | Zoph and Le 2016 [A] | 1611.01578, https://arxiv.org/abs/1611.01578 | An RNN controller trained by RL on validation accuracy. | Origin of RL-NAS. | Partly (factorized controller) |
> | Real et al. 2018, regularized evolution [F] | 1802.01548, https://arxiv.org/abs/1802.01548 | "Evolution searched faster than RL and random search, especially at the earlier stages". | RE baseline. | Yes |
> | Klein and Hutter 2019 [F] | 1905.04970, https://arxiv.org/abs/1905.04970 | RL "seems to be too sample inefficient for these tasks". | Prior null for RL on tables. | Protocol only |
> | Chen et al. 2016, learned black-box optimisers [A] | 1611.03824, https://arxiv.org/abs/1611.03824 | RNN optimisers trained on synthetic functions transfer to HPO and compare favourably with BO packages. | Learned-optimiser precedent. | No |
> | MetaBO, Volpp et al. 2019 [A] | 1904.02642, https://arxiv.org/abs/1904.02642 | An acquisition function meta-trained by RL on related tasks; it "falls back to the performance level of general AFs if no particular structure is present". | RL-trained transfer, with a fallback we lack. | No |
> | Hyp-RL, Jomaa et al. 2019 [A] | 1906.11527, https://arxiv.org/abs/1906.11527 | HPO as sequential decisions, trained by RL on 50 datasets. | RL-for-HPO precedent. | No |
> | OptFormer, Chen et al. 2022 [A] | 2205.13320, https://arxiv.org/abs/2205.13320 | A text transformer trained on "vast tuning data" (Vizier) imitates at least 7 HPO algorithms. | Learned universal optimiser. | No |
> | GRPOformer, Guo et al. 2025 [F] | 2509.17105, https://arxiv.org/abs/2509.17105 | GRPO for HPO on OpenML. The word "seed" does not appear in the HTML full text (0 hits). | Prior art for "GRPO for HPO"; weak evaluation. | No |
> | VeLO, Metz et al. 2022 [A] | 2211.09760, https://arxiv.org/abs/2211.09760 | A learned weight optimiser, meta-trained with about 4,000 TPU-months. | A different problem (weights, not configs); cite only to separate the two. | No |
> | DeepSeekMath / GRPO, Shao et al. 2024 [F] | 2402.03300, https://arxiv.org/abs/2402.03300 | Advantage = (r - mean(r)) / std(r) over the group. | Our optimiser. | Yes |
> | Gao et al. 2026, GRPO tournament [F] | 2611.02404, https://arxiv.org/abs/2611.02404 | "When the reward of a response depends only on that response ... subtracting their mean lowers variance without changing the expected update" (classical; it cites Williams 1992). GRPO "collapses onto the majority answer and ends further from human opinion than the untrained model". | Predicts the league null; GRPO collapse. | Ablation idea |
> | Yue et al. 2025 [F] | 2504.13837, https://arxiv.org/abs/2504.13837 | "After RLVR training, the model often exhibits narrower reasoning coverage compared to its base model"; base models win at large k. | RL narrows the proposal distribution. | No |
> | Kirk et al. 2023 [A] | 2310.06452, https://arxiv.org/abs/2310.06452 | RLHF "significantly reduces output diversity" compared with SFT. | Same direction. | No |
> | Chu et al. 2025 [F] | 2501.17161, https://arxiv.org/abs/2501.17161 | RL generalises out of distribution better than SFT on rule and visual variants. | **Counter-evidence; cite it honestly.** | No |
> | Krishnamurthy et al. 2024 [F] | 2403.15371, https://arxiv.org/abs/2403.15371 | GPT-3.5, GPT-4 and Llama2 fail to explore in bandits through "suffix failures" and "uniform-like failures". Only GPT-4 with chain-of-thought and a summarised history succeeds. | Explains a prompted search that does not beat random. | No |
> | McKenzie et al. 2023, Inverse Scaling [A] | 2306.09479, https://arxiv.org/abs/2306.09479 | Tasks on which larger LMs do worse, and four possible causes. | General precedent for larger-is-worse. | No |
>
> ### BO, multi-fidelity and transfer-HPO baselines
>
> | Work | arXiv / URL | What it reports (as opened) | Relevance | In LeagueTST? |
> |---|---|---|---|---|
> | Random search, Bergstra and Bengio 2012 (JMLR 13:281-305; no arXiv) | https://www.jmlr.org/papers/v13/bergstra12a.html | Random trials beat grid; proposed as the baseline. | Baseline. | Yes |
> | TPE, Watanabe 2023 [A]; Optuna, Akiba et al. 2019 [A] | 2304.11127, https://arxiv.org/abs/2304.11127; 1907.10902, https://arxiv.org/abs/1907.10902 | TPE components and the Optuna framework. | Baseline. | Yes |
> | SMAC3, Lindauer et al. 2021 [A] | 2109.09831, https://arxiv.org/abs/2109.09831 | RF-based BO package (BSD). | Baseline. | Yes (AB-X) |
> | HEBO, Cowen-Rivers et al. 2020 [A] | 2012.03826, https://arxiv.org/abs/2012.03826 | Won the NeurIPS 2020 BBO challenge; beats others on 108 Bayesmark tasks. | Strong BO. | **No** |
> | Hyperband, Li et al. 2016 [A] | 1603.06560, https://arxiv.org/abs/1603.06560 | Bandit-based early stopping. | Baseline. | Yes (AB-X) |
> | ASHA, Li et al. 2018 [A] | 1810.05934, https://arxiv.org/abs/1810.05934 | Asynchronous successive halving. | Baseline. | Yes |
> | BOHB, Falkner et al. 2018 [A] | 1807.01774, https://arxiv.org/abs/1807.01774 | Hyperband with a model-based sampler. | The standard MF + model baseline. | **No** |
> | DEHB, Awad et al. 2021 [F] | 2105.09821, https://arxiv.org/abs/2105.09821 | DE inside Hyperband; 50 runs per algorithm by default; strong on discrete tabular NAS benchmarks. | Baseline. | Yes (AB-X; handicapped by repair) |
> | PriorBand, Mallik et al. 2023 [F] | 2306.12370, https://arxiv.org/abs/2306.12370 | Hyperband with expert priors; 50 seeds for single workers; average ranks and normalised regret. | The LLM as a prior. | No |
> | ifBO, Rakotoarison et al. 2024 [F] | 2404.16795, https://arxiv.org/abs/2404.16795 | Freeze-thaw BO, one budget step at a time; regret averaged over 10 seeds. | Epoch-level allocation. | No |
> | DyHPO, Wistuba et al. 2022 [A] | 2202.09774, https://arxiv.org/abs/2202.09774 | Multi-fidelity race with a deep-kernel GP. | Epoch-level allocation. | No |
> | One epoch, Egele et al. 2023 [F]; i-Epoch, Egele et al. 2024 [A] | 2307.15422, https://arxiv.org/abs/2307.15422; 2404.04111, https://arxiv.org/abs/2404.04111 | "Discard all but the top-K after one epoch" matches SHA, HB and learning-curve extrapolation at about a tenth of the cost; researchers "should always use the suggested baseline". | Required baseline. | **Partly** (our epochk uses groups of 8 with 2 promoted; it does not sweep K) |
> | PFNs4BO, Müller et al. 2023 [A] | 2305.17535, https://arxiv.org/abs/2305.17535 | PFN surrogates for BO, tested on HPO-B, Bayesmark and PD1. | Modern BO. | No |
> | RGPE, Feurer et al. 2018 [F] | 1802.02219, https://arxiv.org/abs/1802.02219 | Ensemble transfer BO that "guarantees to not perform substantially worse than standard BO". | Transfer baseline with a safeguard. | **No** |
> | FSBO, Wistuba and Grabocka 2021 [A] | 2101.07667, https://arxiv.org/abs/2101.07667 | Meta-learned deep-kernel surrogate. | Transfer baseline. | No |
> | HyperBO, Wang et al. 2021 [A] | 2109.08215, https://arxiv.org/abs/2109.08215 | Pre-trained GP priors. | Transfer baseline. | No |
> | Auto-sklearn 2.0, Feurer et al. 2020 [A] | 2007.04074, https://arxiv.org/abs/2007.04074 | Portfolio-based meta-learning. | Source of the "portfolio" control. | Yes (portfolio_from_ETTh1) |
>
> ### Tabular and surrogate benchmarks
>
> | Work | arXiv / URL | What it reports (as opened) | Relevance | In LeagueTST? |
> |---|---|---|---|---|
> | NAS-Bench-101, Ying et al. 2019 [F] | 1902.09635, https://arxiv.org/abs/1902.09635 | Each query uses "a random trial index, drawn uniformly at random from {1,2,3}". Regret is measured against the best *mean test* accuracy; 500 runs per method. | **The precedent for our seed-drawing lookups.** | Protocol only |
> | NAS-Bench-201, Dong and Yang 2020 [F] | 2001.00326, https://arxiv.org/abs/2001.00326 | CIFAR-10 test accuracy over 500 runs: REA 93.92±0.30, REINFORCE 93.85±0.37, RS 93.70±0.36. | The same near-null among RS, RE and RL. | No |
> | NAS-Bench-301, Zela et al. 2020 [F] | 2008.09777, https://arxiv.org/abs/2008.09777 | A one-seed table predicts the other seeds worse than a surrogate does: MAE 4.534e-3 for the table against 3.446e-3 for the surrogate. | One-seed noise (our ETTh1). | No |
> | Klein and Hutter 2019 [F] | 1905.04970, https://arxiv.org/abs/1905.04970 | 4 seeds per config; each evaluation "randomly sample[s] one of the four performance values"; 500 runs; test regret. | Template for our replay. | Protocol only |
> | HPOBench, Eggensperger et al. 2021 [F] | 2109.06716, https://arxiv.org/abs/2109.06716 | 32 repetitions, regret and ranks over time. A sign test across benchmarks, with Bonferroni-style division of α. | The standard suite. | **No** |
> | YAHPO Gym, Pfisterer et al. 2021 [F] | 2109.03670, https://arxiv.org/abs/2109.03670 | 30 replications; mean normalised regret and ranks for anytime performance; Friedman tests with critical-difference plots at 25% and 100% of the budget. "Hyperband appears to consistently perform better on tabular benchmarks". | Tabular bias caveat. | **No** |
> | HPO-B, Pineda Arango et al. 2021 [F] | 2106.06257, https://arxiv.org/abs/2106.06257 | Recommends "normalized regret and the mean ranks for all trials from 1 to 100" and CD diagrams "@25, @50, and @100 trials". Transfer HPO uses fixed meta-train, meta-validation and meta-test splits. | **Norm for the transfer test.** | **No** |
> | LCBench / Auto-PyTorch, Zimmer et al. 2020 [A] | 2006.13799, https://arxiv.org/abs/2006.13799 | Learning-curve benchmark. | Epoch-level table. | No |
> | JAHS-Bench-201, Bansal et al. 2022 (NeurIPS D&B; no arXiv found) | https://proceedings.neurips.cc/paper_files/paper/2022/hash/fd78f2f65881c1c7ce47e26b040cf48f-Abstract-Datasets_and_Benchmarks.html | Surrogate benchmark for joint architecture and HP search, with a 14-D search and fidelity space. | Joint architecture-HP table. | No |
> | NAS-Bench-Suite, Mehta et al. 2022 [A] | 2201.13396, https://arxiv.org/abs/2201.13396 | "Many conclusions drawn from a few NAS benchmarks do not generalize to other benchmarks". | We have two tables. | No |
> | TSBench, Madhusudhanan et al. 2024 [F] | 2403.04477, https://arxiv.org/abs/2403.04477 | 97,200 forecasting evaluations (MLPs). With 50 trials, SMAC, Hyperband and BOHB beat random on a CD diagram over 20 datasets. | The only public forecasting HPO table found. | **No** |
> | COCO, Hansen et al. 2016 [F] | 1603.08785, https://arxiv.org/abs/1603.08785 | Anytime assessment; the "area above the curve" of the ECDF is meaningful. | Precedent for area metrics. | Metric precedent |
>
> ### Time-series forecasting context
>
> | Work | arXiv / URL | What it reports (as opened) | Relevance | In LeagueTST? |
> |---|---|---|---|---|
> | PatchTST, Nie et al. 2022 [F] | 2211.14730, https://arxiv.org/abs/2211.14730 | ILI horizons {24, 36, 48, 60}. For ILI, ETTh1 and ETTh2 it uses "a reduced size of parameters ... (H=4, D=16 and F=128)", with dropout 0.2. PatchTST/42 and /64 variants. | The model being tuned. | Yes |
> | iTransformer [A], TimesNet [A], DLinear [A], SparseTSF [A] | 2310.06625, 2210.02186, 2205.13504, 2405.00946 (https://arxiv.org/abs/<id>) | Successor and linear forecasters. | Reviewers ask "why only PatchTST?". | No |
> | No Champions, Brigato et al. 2025 [F] | 2502.14045, https://arxiv.org/abs/2502.14045 | HP search with Optuna (TPE sampler and SuccessiveHalvingPruner); final models on 3 seeds; Friedman test finds no model that differs. | Forecasting HPO protocol. | Protocol only |
> | Xu et al. 2025, HPO pipeline for TSF transformers [F] | 2501.01394, https://arxiv.org/abs/2501.01394 | OptunaSearch (TPE) with 20 trials per model and dataset, PatchTST among the models. | The usual depth of forecasting HPO. | No |
> | Auto-PyTorch-TS [A], hierarchical NAS for TSF [A], TimeRecipe [A] | 2205.05511, 2406.05088, 2506.06482 | BO + multi-fidelity and NAS for forecasting; TimeRecipe is a module-level benchmark of 10,000+ experiments. | Forecasting AutoML context. | No |
> | TFB [A], BasicTS+ [A], simple feed-forward networks [A] | 2403.20150, 2310.06119, 2503.23621 | Fair forecasting benchmarks; strong simple baselines. | Context. | No |
>
> ### Statistics and protocol
>
> | Work | arXiv / URL | What it reports (as opened) | In LeagueTST? |
> |---|---|---|---|
> | Lindauer and Hutter 2019, NAS best practices [F] | 1909.02453, https://arxiv.org/abs/1909.02453 | BP7: report performance against compute, which may be measured "in epochs (if these are roughly similarly expensive ...)". BP8: compare against random search, run "for the same amount of time". BP9: multiple seeds, report mean and SD. BP11: control hardware and library confounds. | Met (see Q3) |
> | Bischl et al. 2021, HPO review [F] | 2107.05847, https://arxiv.org/abs/2107.05847 | "It is mandatory that all tuners are granted the same budget". Compare by wall-clock time and "explicitly report overhead for model inference and point proposal"; "compare the whole optimization traces". | Met (A6 reports L4-seconds) |
> | rliable, Agarwal et al. 2021 [F] | 2108.13264, https://arxiv.org/abs/2108.13264 | Stratified bootstrap CIs, IQM, probability of improvement. | Met (ST) |
> | Bouthillier et al. 2021 [F] | 2103.03098, https://arxiv.org/abs/2103.03098 | P(A>B) ≥ γ = 0.75; "29 trainings" for β = 0.05; mentions Bonferroni for multiple comparisons. | Partly |
> | Colas et al. 2018 [A]; Henderson et al. 2017 [A] | 1806.08295, 1709.06560 | Power analysis for seed counts; significance reporting. | Met |
> | Demšar 2006 (JMLR 7:1-30; no arXiv; only page 1 extractable) | https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf | Wilcoxon signed-ranks for two classifiers; Friedman plus post-hoc tests for several, across data sets. | Not applicable (2 tasks) |
> | García and Herrera 2008 (JMLR 9:2677-2694; abstract page only) | https://www.jmlr.org/papers/v9/garcia08a.html | More powerful all-pairs procedures. The abstract does **not** name Holm. | n/a |
> | Benavoli et al. 2015 [A] | 1505.02288, https://arxiv.org/abs/1505.02288 | Do not use mean-rank post-hoc tests; use sign or Wilcoxon tests. | Met (pairwise tests) |
> | Knopov and Shvai 2026 [A] | 2610.23700, https://arxiv.org/abs/2610.23700 | A non-significant difference is not equivalence: use TOST with a pre-set margin, and at least 5 runs. | **Not met** |
> | Li and Talwalkar 2019 [A]; Yang et al. 2019 [A] | 1902.07638, 1912.12522 | Random search is a strong NAS baseline; many methods barely beat the average architecture. | Met |
> | Eimer et al. 2023 [A]; Schneider et al. 2025, overtuning [A] | 2306.01324, 2506.19540 | Separate tuning and test seeds; overtuning in about 10% of cases. | Met (held-out seeds); overtuning not computed |
>
> ### Model and method citations needed for the setup (not comparisons)
>
> | Work | arXiv / URL | In LeagueTST? |
> |---|---|---|
> | Qwen2.5 Technical Report [A] | 2412.15115, https://arxiv.org/abs/2412.15115 | Yes |
> | LoRA, Hu et al. 2021 [A] | 2106.09685, https://arxiv.org/abs/2106.09685 | Yes |
>
> ---
>
> ## Q1. Missing external baselines and benchmarks, and what reviewers will flag
>
> What the project already has is a strong classical set. It covers random search, RE, TPE, ASHA, Hyperband, DEHB, SMAC3, a factorized policy-gradient controller (the descendant of Zoph and Le), a grouped epoch-k screener, a referee, and two transfer controls (portfolio and warm-started TPE). That is more classical coverage than any LLM-HPO paper I opened: LLAMBO has no random arm end to end, and AgentHPOBench has three classical baselines.
>
> What is missing, in the order a NeurIPS/ICML reviewer is most likely to raise it:
>
> 1. **An LLM arm beyond 0.5 to 3B, or an established LLM-HPO method.** Every positive LLM-HPO result I opened used a large model:
>    * GPT-4 Turbo in Zhang et al.;
>    * GPT-3.5 in LLAMBO;
>    * GPT-4 in GENIUS;
>    * DeepSeek V4.1 Flash in AgenticBBO;
>    * a 62B PaLM in EvoPrompting.
>
>    A reviewer will say the paper tests "small prompted Qwen", not "LLM architects". Two minimum fixes:
>    * Call the prompted arm an OPRO-style loop (it is one: sorted top-24 history with scores, temperature 1.0), cite OPRO, and limit every LLM claim to Qwen2.5 at 0.5 to 3B.
>    * Or add one 7B to 14B open model on the RTX, or one LLAMBO-style arm.
>
>    Zhang et al.'s own table shows why scale matters: GPT-3.5 Turbo beats random on only 43.75% of tasks, GPT-4 Turbo on 81.25%.
>
> 2. **No public benchmark to check the harness.** The paper's central finding is that nothing beats random search on fresh seeds. A reviewer cannot tell whether that is a property of these tables or of the replay harness:
>    * the space is narrow (59 configs within 1 SD; 74 architectures);
>    * NAS-Bench-Suite says conclusions from a few benchmarks "do not generalize";
>    * YAHPO says tabular benchmarks "may produce unfaithful results".
>
>    Running the same CPU methods through the same harness on one public table settles this. TSBench is a forecasting table. NAS-Bench-201 or LCBench (via YAHPO or HPOBench) are standard. The check passes if the published orderings come back: BO above RS by HPOBench's sign test, and SMAC, HB and BOHB above random on TSBench.
>
> 3. **BOHB, and the exact one-epoch / i-Epoch baseline.** BOHB is the standard combined multi-fidelity and model-based baseline (HPOBench, YAHPO, DEHB and TSBench all run it). DEHB is not a substitute here, because 69% of its proposals were repaired. Egele et al. ask that "discard all but top-K after one epoch" always be included, with i-Epoch judged on its Pareto front. Our epochk arm is a grouped variant (8 per group, 2 promoted) and does not sweep K. docs/ablations.md already lists both as not done.
>
> 4. **Transfer-HPO baselines and protocol for the RL transfer claim.** The portfolio and warm-started TPE are good, cheap controls. An HPO reviewer will still ask for:
>    * one transfer-BO method with a safeguard: RGPE "guarantees to not perform substantially worse than standard BO", and MetaBO "falls back" to general acquisition functions;
>    * and will note that HPO-B's transfer protocol needs meta-train, meta-validation and meta-test tasks.
>
>    We have one source and one target per direction. Without these, "RL hurts transfer" cannot be separated from "these two tables disagree" (see Q2c).
>
> 5. **The default-configuration warm-start control.** Rodrigues et al. show that an LLM advisor's early lead was the fixed default seed. Our LLM arms start from the model's first proposal and the classical arms from random draws. Give every method PatchTST/42's config as its first point, or show that no arm gets one. This is a one-line replay flag.
>
> 6. **LLM plus classical hybrids.** Ferreira et al. (Centaur) and SLLMBO find the LLM most useful as a complement to a classical optimiser. PLAN's K4 fallback ("run Qwen as the mutation step in RE") is exactly this arm. If it was not run, say so.
>
> 7. **Lower priority:** HEBO (strong BO, but tuned for continuous Bayesmark); ifBO or DyHPO (freeze-thaw, epoch-level); PriorBand with the untrained LLM's proposal distribution as the prior (a clean test of "does the LLM know anything"); PFNs4BO.
>
> 8. **Learned optimisers (OptFormer, Chen et al. 2016, VeLO).** Cite them, do not run them:
>    * OptFormer and the Universal Searcher need very large corpora (Vizier; 228,000 trajectories).
>    * VeLO optimises weights, not configurations; one sentence separating the two avoids confusion.
>    * GRPOformer is prior art for "GRPO for HPO". It has no seed reporting (0 hits for "seed" in its full text), so the claim to make against it is better evaluation, not novelty.
>
> 9. **Forecasting scope.** Only PatchTST-style models on two datasets. Expect "why not iTransformer or DLinear?" A11 already disclaims SOTA. Cite AgentHPOBench's time-series tasks and TSBench as the only HPO evidence in forecasting that I found, and use Xu et al. (20 TPE trials) and No Champions (Optuna TPE with successive halving) for typical forecasting HPO depth.
>
> **Most likely to be flagged:** items 1, 2 and 4, then 3 and 5.
>
> ---
>
> ## Q2. Our findings against published results
>
> **Comparability warning, which applies to everything below.** None of the published numbers are on our tables or on our metric. Here is how they differ:
>
> * **Budgets.** Published budgets are 5 interventions (AgentHPOBench), 10 evaluations (Zhang; GENIUS), 25 trials (LLAMBO), 5 + 25 (AgenticBBO), 12 to 40 evaluations (Rodrigues) and 24 GPU-hours (Ferreira). Ours are 1 to 100 full-run equivalents of 30 epochs, charged per distinct config.
> * **Metrics.** Published metrics are win rate against random, normalised regret or rank, normalised scores (MBNS, AgenticBBO), accuracy, percentage points of CV accuracy and val_bpb. Ours is the AUC of held-out-seed regret in pooled seed-SD units.
> * **Scoring.** Almost all published LLM-HPO studies score the incumbent in-sample, on the same evaluation it was chosen by. The exceptions are Rodrigues (held-out test) and AgenticBBO (held-out deployment seeds). We score on seeds the search never saw, which removes the reward for seed luck.
>
> So the right comparison is the *direction and qualitative pattern* of each finding, never the numbers.
>
> ### (a) The LLM does not beat random search
>
> **Ours.** Prompted Qwen2.5-1.5B has ILI AUC 1.56 to 1.81 on three machines, against 1.52 for random. The E4 family-3 difference is +0.29 SD, with 0.56 detectable. It is worse early: regret 2.89 against 2.07 at budget 10. On the pick reseeds, nothing is rejected. 0.5B is numerically below random (1.39), but that comparison is untested and on another machine. docs/ablations.md was committed during this session (40804c7) with a new exploratory, post hoc section: no LLM arm beats random, RE, TPE, ASHA or random + referee on ILI held-out AUC, "even by an unadjusted interval", across 162 comparisons.
>
> **Consistent published results:**
> * **Rodrigues et al.** (8 tabular tasks × 5 seeds, 7 models): with the same default seed, random search ties the advisor by 5 evaluations and leads by 12. The LLM adds nothing on held-out test.
> * **Ferreira et al.** (fixed 14-HP space, Qwen3.5 0.8B and 27B): TPE and SMAC beat the pure LLM methods, and "several pure LLM methods performed worse than random search". LLAMBO-Optuna with Qwen 27B scores 0.9882 against 0.9873 for random.
> * **AgentHPOBench, closest in domain:** on 7 time-series tasks, random search (0.299) is at or above five of six open-weight agents of 2B to 32B (0.233 to 0.285). Only DeepSeek-R1-Qwen-14B (0.316) and the API models exceed it.
> * **EvoPrompting and LLMatic:** naive prompting is poor for NAS.
> * **Krishnamurthy et al.:** LLMs fail to explore in-context through suffix failures, unless the history is summarised. Our prompt shows a sorted top-24 list, which is a partial summary.
>
> **Inconsistent published results**, all with large models:
> * **Zhang et al.:** GPT-4 Turbo beats random on 81% of 32 HPOBench tasks at 10 evaluations. But GPT-3.5 Turbo manages only 43.75%, and Llama3-8B was "less effective during the search phase". At the smaller end, their result agrees with ours.
> * **LLAMBO:** strongest early in the search, but against BO baselines; there is no random-search arm end to end.
> * **AgenticBBO:** Direct LLM 0.519 against random 0.273.
> * **GENIUS:** top 0.12% to 0.93% of 6,561 configs in 10 iterations.
>
> **Reading.** Our null matches every published result that uses open models under about 32B in a fixed, discrete space, and it is stricter than most because it scores on held-out seeds. It contradicts only results with frontier-class models. Limit the claim to that scope. The early-budget advantage LLAMBO and Zhang et al. report for large models does not appear for 1.5B Qwen: it is behind at budget 10.
>
> **One confound to rule out before publishing.** The anonymised prompt was worse (+0.36 SD, Holm p 0.0145), so field names carry information. LLAMBO and GENIUS both raise memorisation, and ETTh1, ILI and PatchTST are public. Our tables are new, but the model may still know PatchTST's ILI defaults.
>
> ### (b) Larger prompted models searched worse
>
> **Ours.** 0.5B 1.39, 1.5B 1.56, 3B 2.06 (rho +0.43, Holm p < 0.0001). This is ILI only, with three checkpoints of one family. On ETTh1 the order is different (3B best, descriptive), and RevIN-off shares do not follow size (41%, 55%, 52%).
>
> **Published.** I found no published model-size sweep at 0.5 to 3B on a tabular HPO or NAS benchmark. The evidence that does exist is mixed:
> * **Larger is better:** Zhang et al. (GPT-4 Turbo > GPT-4 > GPT-3.5); OPRO (gpt-4 beats gpt-3.5 and text-bison on TSP); SemanticOpt (27B keeps its prior after fine-tuning, 9B does not); AgentHPOBench overall (open-weight MBNS rises from -0.001 for Gemma2-2B to 0.148 for Qwen3-32B).
> * **Not monotone, or smaller better:** Ferreira et al. (Centaur with 0.8B beat Centaur with 27B; in the fixed space, Gemini Flash-Lite was "comparable to Qwen3.5-0.8B"); Rodrigues et al. ("non-monotone", and all seven models tie once one task is removed); AgentHPOBench's time-series column is not ordered by size (the 14B model leads, Qwen3-8B and Gemma2-2B tie).
> * **Inverse scaling** (McKenzie et al.) gives general mechanisms: preferring memorised sequences over in-context instructions, and distractor tasks.
>
> **Reading.** "Larger is worse" goes against the dominant published trend (larger or frontier is better). It has precedent only as non-monotone behaviour in a narrow band. A plausible explanation, which is untested and should be called a hypothesis, is that smaller instruct models sample closer to uniform. The 0.5B arm (1.39) is near random (1.52), which is what you would expect if randomness is what helps in a plateau-shaped space. That fits the exploration findings of OPRO's temperature ablation and of Krishnamurthy et al.
>
> **What would make the claim stronger:** report the unique-config rate and proposal entropy by model size. The claim to state is "within Qwen2.5-Instruct 0.5 to 3B on ILI", not "scale". AB5's scope text already says this; keep it.
>
> ### (c) RL fine-tuning hurt transfer
>
> **Ours.**
> * GRPO policies trained on ETTh1 and deployed on ILI score 3.0 to 3.3, against 1.56 untrained (+1.59 SD [+1.28, +1.90]).
> * There is no dose effect over 30, 60 or 120 steps, and no KL trend.
> * The trained policies collapse onto RevIN-off configs (96 to 100%, against 55% untrained). The best RevIN-off ILI config has regret 1.03 SD.
> * The league policy is indistinguishable from the ETTh1 top-config portfolio (-0.14 SD; detectable 1.10) and much worse than warm-started TPE (+1.50 SD, rejected).
>
> **Published:**
> * **Trained optimisers transfer only with large, diverse source data.** OptFormer is trained on the Vizier database; the Universal Searcher on 228,000 trajectories, with an evolutionary/random pool to choose from; Chen et al. 2016 on families of synthetic functions; MetaBO on a set of related tasks, and it falls back when structure is absent. Our policies are trained on one table. Overfitting to that table's dominant field (RevIN: 84% of ETTh1's variance, 5% of ILI's; field-share Spearman 0.14 between the tables) is what this literature would predict. The portfolio control shows the same loss with no RL at all.
> * **RL narrows the proposal distribution.**
>   * Yue et al.: after RLVR the model has "narrower reasoning coverage"; the base model wins at large k.
>   * Kirk et al.: RLHF "significantly reduces output diversity".
>   * Gao et al.: GRPO "collapses onto the majority answer and ends further from human opinion than the untrained model".
>   * SemanticOpt's 9B fine-tune overwrote its prior (that one is SFT, not RL).
>   * Our 98% RevIN-off collapse is this pattern.
> * **Counter-evidence:** Chu et al. find that RL generalises out of distribution better than SFT, on rule and visual variants of a single task family. That is a different kind of shift from two unrelated tables. Cite it so the paper does not read as one-sided.
> * **The league null is expected.** Gao et al. state the classical result that subtracting groupmates' mean "lowers variance without changing the expected update" when a reward depends only on its own response. Our MSE reward does not depend on rivals, and rivals enter only the mean and standard deviation of the GRPO advantage (DeepSeekMath: (r - mean)/std). Klein and Hutter ("RL ... too sample inefficient") and Real et al. (evolution faster than RL early) are the tabular precedents for RL controllers not beating simple search.
>
> **Reading.** The honest claim is: "GRPO on one source table learns that table's preferences; deployed on a dissimilar table it does no better than replaying the source's top configs, and worse than the untrained model."
>
> Do not claim "RL fine-tuning hurts transfer" in general. There is one task pair per direction (HPO-B's transfer protocol uses many tasks with fixed splits), no same-table deployment, and no check that removes the RevIN preference while keeping the training.
>
> ---
>
> ## Q3. Protocol norms: how published work reports, and where LeagueTST stands
>
> | Norm | Published practice (opened) | LeagueTST | Verdict |
> |---|---|---|---|
> | **Budget unit** | Lindauer and Hutter BP7: compute against performance, which may be measured "in epochs". HPOBench, NAS-Bench-101 and Klein and Hutter use simulated training time. YAHPO counts full-fidelity evaluations. ifBO: one step = one epoch. Egele: epochs on the i-Epoch Pareto front. Bischl: "the same budget" for all tuners, with inference overhead reported. LLM-HPO papers count evaluations only, ignoring how long each config trains (Zhang 10, LLAMBO 25, AgentHPOBench 5; AgenticBBO counts objective evaluations, "not model requests"). | Equal total epochs, each distinct config charged once, with repeats free but wasted (A18). L4-seconds including LLM sampling and updates are reported separately (A6). | **Exceeds the LLM-HPO norm; meets the multi-fidelity norm.** Disclose that repeats are free; HPOBench charges stateless re-runs in full. |
> | **Seeds or runs** | Tabular NAS: 500 runs (NAS-Bench-101, NAS-Bench-201, Klein and Hutter). HPOBench 32, YAHPO 30, DEHB and PriorBand 50, ifBO 10, HPO-B 5 initial-design seeds × tasks. LLM-HPO: LLAMBO 5, Ferreira 3, AgentHPOBench 3 (API agents 1), Rodrigues 5 × 8 tasks, GENIUS 1 to 5, Universal Searcher 5. Bouthillier: 29 paired trainings for P(A>B) at γ = 0.75. | CPU methods 30 (100 in AB-C). LLM arms 10 to 30. GRPO 10 policies. | **Above the LLM-HPO norm; below the tabular-NAS norm for CPU methods, where seeds are nearly free.** Raising CPU methods to 100 or more would remove an easy objection. |
> | **Seed noise in lookups** | NAS-Bench-101: a random trial index per query. Klein and Hutter: one of 4 values per evaluation. Both score regret against the mean of the same seeds (in-sample). NAS-Bench-301: a one-seed table is a noisy estimate. | One seed drawn per config per search run (A2); ETTh1 uses one seed plus a Gaussian offset. | **Matches the standard.** |
> | **Held-out scoring** | Most score in-sample against the best mean or the best observed validation. Exceptions: Rodrigues (held-out test); AgenticBBO (development seeds 0 to 1, deployment seeds 2 to 5); Eimer et al. (separate tuning and test seeds). | ILI regret on seeds the search never saw; ETTh1 claims only from pick reseeds. | **Exceeds the norm.** This is the strongest protocol point; say so in the paper. |
> | **Anytime metric** | HPO-B: normalised regret and mean ranks for trials 1 to 100, CD @25/@50/@100. YAHPO: mean normalised regret and ranks. Bischl: compare whole traces. AgenticBBO: 70% trajectory mean of normalised best-so-far plus 30% final. COCO: ECDF area. | AUC over budgets 1 to 100, plus @10/@25/@50/@100 and raw-MSE AUC. | **Meets the norm.** Seed-SD units are not standard: add normalised regret, (y - y*)/(y_random-median - y*) or min-max, in an appendix for comparability. |
> | **Tests and multiple comparisons** | HPOBench: sign test across benchmarks with α divided by the number of comparisons. YAHPO, HPO-B, TSBench, No Champions: Friedman tests and CD diagrams across tasks. Demšar: Wilcoxon or Friedman across data sets. Benavoli: avoid mean-rank post-hoc tests. Rodrigues: paired t, Wilcoxon and bootstrap, "without a multiple-comparison correction". AgentHPOBench: task-bootstrap intervals, no tests. LLAMBO, Zhang and GENIUS: no tests (win rates and ranks). rliable: stratified bootstrap, IQM, P(improvement). | Pre-registered families, Holm within families, paired or permutation tests, detectable-effect (power) tables, IQM and P(improvement). | **Clearly exceeds the norm.** Among the 61 full texts I searched, only Knopov and Shvai (the equivalence-testing paper) mention pre-registration or TOST; none of the HPO, NAS or LLM-HPO papers do. |
> | **Equivalence for nulls** | Knopov and Shvai: TOST with a pre-set margin; n ≥ 5. Bouthillier: P(A>B) ≥ 0.75. | Detectable effects are reported, but there is no TOST. | **Gap.** "No significant difference" for league vs none (detectable 1.25 SD) is weak. Pre-register a margin and run TOST, or state the claim as "differences under about 1.2 SD are not excluded". |
> | **Generality across tasks** | NAS-Bench-Suite: few benchmarks do not generalise. Demšar, Friedman and CD tests need many data sets. | Two tables; ILI counts as a robustness check (A11). | **Below the norm by design.** No across-task claim is possible. The public-table harness check (Q1, item 2) is the cheapest partial fix. |
> | **Confounds** | Lindauer and Hutter BP11: control hardware and library versions. | Prompted 1.5B gives 1.56 (RTX), 1.65 (Mac) and 1.81 (A10); each comparison uses a reference on the same card; hardware checks are reported. | **Meets the norm by disclosure.** Expect reviewers to ask about it. Every LLM comparison must be within one card. |
>
> ---
>
> ## Q4. The citations the paper must have for the claims it can honestly make
>
> The claims the evidence supports:
>
> * On two new PatchTST tables, no search method, the prompted small LLM included, reliably beats random search on fresh seeds.
> * Within Qwen2.5-Instruct 0.5 to 3B, larger checkpoints searched worse on ILI.
> * GRPO policies trained on one table and transferred to the other do worse than the untrained model, about as badly as replaying the source table's top configs.
> * League rivals in the baseline make no detectable difference, which is what theory predicts.
>
> Must-have citations:
>
> 1. **Rodrigues et al. 2026, arXiv 2606.21641**, https://arxiv.org/abs/2606.21641. The closest published budget-matched null for LLM-HPO: seeded classical search overtakes the advisor, and the model effect is not monotone. It anchors the claim that prompted LLMs match random search. It also motivates the default-seed control.
> 2. **Zhang et al. 2023, arXiv 2312.04528**, https://arxiv.org/abs/2312.04528. The positive claim that LLMs match or beat BO at small budgets. Its own table (GPT-3.5 Turbo beats random on 43.75% of tasks, GPT-4 Turbo on 81.25%) is the best published evidence that the result depends on model capability. That is how to place our 0.5 to 3B result without overclaiming.
> 3. **Lindauer and Hutter 2019, arXiv 1909.02453**, https://arxiv.org/abs/1909.02453, together with **Ying et al. 2019 (NAS-Bench-101), arXiv 1902.09635**, https://arxiv.org/abs/1902.09635. The protocol basis: random-search baseline at equal compute, epochs allowed as the compute unit, multiple seeds, tabular benchmarks; and the random-trial-per-query lookup that A2 reproduces.
> 4. **Gao et al. 2026, arXiv 2611.02404**, https://arxiv.org/abs/2611.02404, together with **Shao et al. 2024 (GRPO), arXiv 2402.03300**, https://arxiv.org/abs/2402.03300, and **Yue et al. 2025, arXiv 2504.13837**, https://arxiv.org/abs/2504.13837:
>    * GRPO's advantage formula;
>    * the classical result that a response-independent baseline does not change the expected update, which predicts the league null;
>    * GRPO and RLVR narrowing the proposal distribution, which fits the 98% RevIN-off collapse.
>
>    Gao et al. is a v1 preprint from November 2026. Pair it with Yue et al., which is well established, rather than relying on it alone.
>
> Next tier, strongly recommended:
> * AgentHPOBench (2607.29626): closest domain, with open models on time-series tasks.
> * Ferreira et al. (2603.24647).
> * LLAMBO (2402.03921) and OPRO (2309.03409): reviewers expect them, and our prompt is OPRO-style.
> * Krishnamurthy et al. (2403.15371): exploration failure.
> * Klein and Hutter (1905.04970): tabular replay, and RL found too sample-inefficient.
> * NAS-Bench-201 (2001.00326): RS, RE and REINFORCE within about 1 SD.
> * YAHPO (2109.03670): bias of tabular benchmarks.
> * HPO-B (2106.06257): the transfer protocol.
> * Egele et al. (2307.15422): the required baseline.
> * Chu et al. (2501.17161): counter-evidence on RL.
> * Knopov and Shvai (2610.23700): for any equivalence claim.
> * PatchTST (2211.14730), Qwen2.5 (2412.15115) and LoRA (2106.09685): method citations.
>
> ---
>
> ## Corrections to WP0 found while re-reading
>
> * WP0 [1.3] says SemanticOpt is weaker "on purely categorical spaces such as NAS-Bench-201". The v3 full text has no NAS-Bench content (0 hits). Drop that sentence unless v1 or v2 is re-opened.
> * WP0 [1.7] (NNGPT) quotes a "GRPO-style policy-gradient update ... applied to LoRA adapters". I opened only the abstract this session, which mentions an RL pipeline but not GRPO. Re-open the full text before quoting it.
> * WP0 [5.2] (Bouthillier: γ = 0.75, 29 trainings) is confirmed in the full text.
> * WP0 [2.4] (Gao et al.) is confirmed, as is WP0 [3.2]'s description of NAS-Bench-101's random-trial lookup.
>
> ## Not opened (do not cite from this report)
>
> * **Holm 1979**, "A simple sequentially rejective multiple test procedure" (Scand. J. Statist.). Not opened this session, and WP0 could not open it either. Neither the Demšar first page I could extract nor García and Herrera's abstract describes Holm. Find an open source before naming it as a reference; the procedure itself can be described in the methods text.
> * **Bergstra et al. 2011**, "Algorithms for Hyper-Parameter Optimization" (the original TPE paper), and **Hutter et al. 2011** (the original SMAC paper). Not opened; Watanabe 2304.11127, Optuna 1907.10902 and SMAC3 2109.09831 are used instead.
> * **Demšar 2006 beyond page 1.** The PDF's other pages could not be decoded; only the abstract recommendations are used.
> * **Srinivasan and Menzies 2026**, arXiv 2602.02752 (high-dimensional LLM optimisation). Seen only as a reference in Rodrigues et al. and as an arXiv listing title.
> * **TimeMixer, TimeXer, STrans (AAAI), and Learning Novel Transformer Architecture for TSF (2502.13721).** Search snippets only.
> * **WP0 entries not re-opened this session:**
>   * TStarBot-X (2011.13729);
>   * AlphaStar (Nature 575:350);
>   * Dr. GRPO (2503.20783);
>   * Bay and Yearick (2607.00152);
>   * Elo-Evolve (2602.13575);
>   * White et al. (2005.02960);
>   * Colas et al. 2019 (1904.06979);
>   * Benavoli et al. 2016 (1606.04316);
>   * RevIN (ICLR 2022);
>   * Berthelier et al. (2603.11869);
>   * Saqur et al. (2603.15506);
>   * Silvestri et al. (2510.20351, LLM contamination of tabular data; relevant to the anonymised-prompt result, so re-open it if used);
>   * LEMUR 2 (2607.06839);
>   * the NeurIPS E&D call pages and all dataset licence pages.
> * Opened but out of scope, so not tabled: LLMSYS-HPOBench (2605.08305), LM-Searcher (2509.05657), CoLLM-NAS (2509.26037), LLMO (2406.05433) and Guo et al. 2023 (2310.05204). All were opened at abstract level only.
