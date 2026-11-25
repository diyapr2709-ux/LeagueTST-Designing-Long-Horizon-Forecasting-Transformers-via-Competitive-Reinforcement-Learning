## What this means for us

1. **Novelty:** "first GRPO-trained optimizer" is not available. GRPO already trains an HPO transformer [2.5] and, in "GRPO-style" form, the LoRA adapters of an LLM that writes networks [1.7]; an LLM trained on NAS-table trajectories already searches unseen spaces [1.6]. I found no work with league rivals in the baseline plus epoch-charged replay (a handful of searches, so not proof).
2. **League in the baseline:** for a reward that depends only on a response itself, subtracting other responses' rewards "lowers variance without changing the expected update" [2.4]. If our rivals enter only the mean, a null is the expected result; ablate rivals in the mean, in the group standard deviation [2.3], and in the reward.
3. **Warm start:** an LLM advisor's lead vanished once classical search got the same default first point, and it added nothing on held-out test [1.5]. Give every baseline the same first configuration and report test as well as validation.
4. **Missing baseline:** "train all for one epoch, keep the top K" matched multi-fidelity methods with an order of magnitude less computation [4.6]. Since we charge in epochs, it has to be in the table.
5. **The null needs a test of its own:** a non-significant difference is not equivalence; fix a margin, use TOST, and take at least 5 runs per method [5.6]; a paired P(A>B) test at gamma = 0.75 needs 29 pairs [5.2].
6. **Tables and data:** one stored seed is a noisy estimate [3.4], tabular replay can favour Hyperband-type methods [3.5], ILI is called unreliable [6.6], and ETDataset is CC BY-ND (no sharing of adapted files) [8.1]; Jena weather from the institute's own page is CC BY 4.0 [8.2].

---

**How this was read (all pages opened 2026-11-25).** Pages were opened with WebFetch, which returns a model-written reading of the page. For every entry marked **[R]** I also pulled the raw page and checked the main quoted sentence or number in the text (not every number in the entry). Titles, first authors and years of all arXiv entries were also matched against the arXiv metadata API. Entries without [R] rest on the abstract or on the WebFetch reading only. Year is the first arXiv version unless a venue is given.

**"Tabular oracle?"** means: could it run on our oracle, which returns validation MSE at epoch k for one of 2,368 configurations per task. This column is **my judgement**, not a statement from the paper.

---

## 1. LLM-driven architecture and hyperparameter search

**[1.1] Large Language Models to Enhance Bayesian Optimization (LLAMBO).** Liu et al., 2024 (ICLR 2024). arXiv 2402.03921. Opened: https://arxiv.org/abs/2402.03921 and https://arxiv.org/html/2402.03921v2 (2026-11-25).
- Shows: in-context LLM warm-start, surrogate and candidate sampler, strongest early in the search (25 trials, 5 seeds); adds private and synthetic datasets because public ones may be memorised.
- Tabular oracle? Partly: proposals must be snapped to table rows, and it has no fidelity axis, so each proposal costs full epochs.

**[1.2] Can GPT-4 Perform Neural Architecture Search? (GENIUS).** Zheng et al., 2023. arXiv 2304.10970. Opened: https://arxiv.org/abs/2304.10970 and https://arxiv.org/html/2304.10970v4 (2026-11-25).
- Shows: prompting-only NAS on public tables, with the authors' own warning that "we cannot rule out contamination" and that runs vary at temperature 0. Our tables are new, which is a point in our favour, but ETTh1 and ILI themselves are not.
- Tabular oracle? Partly: same prompt loop works on our table, but it needs a closed model; the untrained Qwen arm already covers the idea.

**[1.3] SemanticOpt: Towards LLM-Based Semantic Black-Box Optimization** (v1 was titled GPTOpt). Meindl et al., 2025, v3 May 2026. arXiv 2510.25404. Opened: https://arxiv.org/abs/2510.25404 and https://arxiv.org/html/2510.25404v3 (2026-11-25).
- Shows: LoRA supervised fine-tuning of a 27B LLM on about 130k Bayesian-optimisation trajectories beats GP-BO on average, but is weaker on purely categorical spaces such as NAS-Bench-201; a smaller 9B variant loses much of its initialisation quality. Our 1.5B policy is far below the sizes tested there (my note).
- Tabular oracle? Partly: proposals can be snapped to rows, but no fidelity axis, and I did not check whether weights are released.

**[1.4] Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch.** Ferreira et al., 2026 (v5). arXiv 2603.24647. Opened: https://arxiv.org/abs/2603.24647 and https://arxiv.org/html/2603.24647v5 (2026-11-25). **[R]**
- Shows: in a fixed search space, TPE (0.9768) and SMAC (0.9778) beat pure LLM methods, some of which lose to random search; a CMA-ES plus LLM hybrid is best (0.9763), and a 0.8B LLM is enough for it. One task, 3 seeds.
- Tabular oracle? Partly: the hybrid's CMA-ES state is continuous, so samples must be rounded to rows; no fidelity axis.

**[1.5] When Is an LLM Worth It for Hyperparameter Optimization? A Budget-Matched Study on Tabular Data Finds the Warm-Start Is a Default Configuration, Not the Model.** Rodrigues et al., 2026 (v2). arXiv 2606.21641. Opened: https://arxiv.org/abs/2606.21641 and https://arxiv.org/html/2606.21641 (2026-11-25). **[R]**
- Shows: 8 tasks x 5 seeds; LLM proposals add +0.40 pp on CV and -0.01 pp on held-out test (p = 0.92); with the same default seed, random search is ahead by 12 evaluations. The closest published null to ours.
- Tabular oracle? Yes: the control (same first configuration for every method) is one line in the replay harness.

**[1.6] Large Language Models as a Universal Searcher for Open-World Automated Machine Learning.** Hu et al., 2026. arXiv 2611.12791. Opened: https://arxiv.org/abs/2611.12791 and https://arxiv.org/html/2611.12791 (2026-11-25). **[R]**
- Shows: an 8B LLM, supervised on about 228k trajectories drawn from the NAS-Bench-301 surrogate, ranks candidates from an evolutionary/random pool and transfers to new spaces; GRPO (with LoRA on an 8B Qwen3) is used only to add explanations. On NAS-Bench-201 it does not beat the strongest LLM baseline. Very recent and close to our setting.
- Tabular oracle? Partly: "rank a candidate pool" fits a finite table well, but it needs the trained 8B model and has no epoch-level fidelity.

**[1.7] NNGPT: Rethinking AutoML with Large Language Models.** Kochnev et al., 2025 (CVPRW 2026). arXiv 2511.20333. Opened: https://arxiv.org/abs/2511.20333 and https://arxiv.org/html/2511.20333v1 (2026-11-25). **[R]**
- Shows: "a GRPO-style policy-gradient update ... is applied to LoRA adapters" of an LLM that writes architecture code, rewarded by validity plus validation accuracy; no seed counts, no untrained-LLM or random-search arm, and the authors call the RL study diagnostic. Precedent for our training recipe, weak precedent for evaluation.
- Tabular oracle? No: it generates code, not rows of a fixed space.

## 2. Learned search policies, RL-based NAS, league and self-play training

**[2.1] Neural Architecture Search with Reinforcement Learning.** Zoph and Le, 2016 (v2 2017). arXiv 1611.01578. Opened: https://arxiv.org/abs/1611.01578 (2026-11-25).
- Shows: the original recurrent controller trained by policy gradient on validation accuracy; our factorized controller is the simplified descendant.
- Tabular oracle? Yes: a controller over our factors, reward = minus validation MSE (already in our baseline set in factorized form).

**[2.2] Towards Learning Universal Hyperparameter Optimizers with Transformers (OptFormer).** Chen et al., 2022 (NeurIPS 2022). arXiv 2205.13320. Opened: https://arxiv.org/abs/2205.13320 and https://ar5iv.labs.arxiv.org/html/2205.13320 (2026-11-25).
- Shows: a 250M text transformer trained by supervised imitation on hundreds of thousands of real tuning studies; learned text optimizers needed that much data, we have two tables.
- Tabular oracle? No in practice: needs a large offline corpus (partly proprietary Vizier data); no fidelity axis.

**[2.3] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO).** Shao et al., 2024. arXiv 2402.03300. Opened: https://arxiv.org/abs/2402.03300 and https://ar5iv.labs.arxiv.org/html/2402.03300 (2026-11-25).
- Shows: GRPO's advantage is reward minus the group mean, divided by the group standard deviation, with no critic. The baseline is a variance-reduction device, and the division by the standard deviation is the part rivals could change.
- Tabular oracle? Yes: it is the optimizer we already use.

**[2.4] When Rewards Depend on the Group, GRPO Plays a Tournament.** Gao et al., 2026 (v1, preprint). arXiv 2611.02404. Opened: https://arxiv.org/abs/2611.02404 and https://arxiv.org/html/2611.02404 (2026-11-25). **[R]**
- Shows: if a reward depends only on its own response, the mean baseline leaves the expected update unchanged; if the reward depends on groupmates, GRPO weights the effect on groupmates by -1/(K-1) and becomes a zero-sum tournament; "any baseline b_k independent of y_k keeps the estimator unbiased"; standard-deviation normalisation does not repair it. Directly about what rivals in a baseline can and cannot do.
- Tabular oracle? Yes: Team-GRPO is an advantage formula and can be an ablation arm if we put rivals in the reward.

**[2.5] GRPOformer: Advancing Hyperparameter Optimization via Group Relative Policy Optimization.** Guo et al., 2025. arXiv 2509.17105. Opened: https://arxiv.org/abs/2509.17105 and https://arxiv.org/html/2509.17105v1 (2026-11-25). **[R]**
- Shows: GRPO applied to HPO on 36 OpenML scenarios with a "policy churn" regulariser; the word "seed" does not appear in the paper, no tests are reported, and the same method has mean rank 1.81 in one table and 1.31 in another. Prior art for "GRPO for HPO", and a fair target for better evaluation. I could not tell whether the policy is a pretrained LLM; the abstract says it learns from scratch.
- Tabular oracle? Partly: the method would fit, but I found no code on the pages opened and no fidelity axis.

**[2.6] Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar).** Vinyals et al., 2019, Nature 575:350-354, DOI 10.1038/s41586-019-1724-z. No arXiv id. Opened: https://www.nature.com/articles/s41586-019-1724-z (abstract page, after its cookie redirect) and https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/ (2026-11-25).
- Shows: the league exists to fix forgetting and rock-paper-scissors cycles in games where the opponent changes the payoff. In our task a configuration's MSE does not depend on any rival, so that motivation does not carry over automatically (my inference).
- Tabular oracle? No: it is a game-training scheme; only the idea of frozen past opponents transfers.

**[2.7] TStarBot-X: An Open-Sourced and Comprehensive Study for Efficient League Training in StarCraft II Full Game.** Han et al., 2020 (v2 2021). arXiv 2011.13729. Opened: https://arxiv.org/abs/2011.13729 and https://ar5iv.labs.arxiv.org/html/2011.13729 (2026-11-25). **[R]**
- Shows: with orders of magnitude less compute "a faithful reimplementation of AlphaStar's methods can not succeed"; league diversity was "significantly limited" and the main agent collapsed onto two strategies. Independent evidence that small leagues fail by collapse, so log proposal diversity.
- Tabular oracle? No: method for games.

## 3. Tabular NAS/HPO benchmarks, validation-to-test and seed noise

**[3.1] Tabular Benchmarks for Joint Architecture and Hyperparameter Optimization.** Klein and Hutter, 2019. arXiv 1905.04970. Opened: https://arxiv.org/abs/1905.04970 and https://ar5iv.labs.arxiv.org/html/1905.04970 (2026-11-25). **[R]**
- Shows: the closest template for our oracle (MSE regression, 62,208 configurations, 4 seeds each, full validation curves, 500 runs per optimizer); RL "seems to be too sample inefficient for these tasks", regularized evolution ended best, and rank correlation between budgets drops among the top configurations.
- Tabular oracle? Yes as a protocol (it is a benchmark, not a method); its curves could also serve as an external third table.

**[3.2] NAS-Bench-101: Towards Reproducible Neural Architecture Search.** Ying et al., 2019. arXiv 1902.09635. Opened: https://arxiv.org/abs/1902.09635 and https://ar5iv.labs.arxiv.org/html/1902.09635 (2026-11-25). **[R]**
- Shows: 3 seeds per cell; validation and test correlate at r = 0.999 at full budget, but top-10% rank correlation between 36 and 108 epochs is only 0.365, which is why Hyperband/BOHB gained little; a PPO LSTM controller "seems to fail to learn to traverse the space", and only a plain per-edge multinomial REINFORCE worked. Report our own top-decile rank correlation by epoch.
- Tabular oracle? Yes as a protocol; its multinomial REINFORCE is our factorized controller.

**[3.3] NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search.** Dong and Yang, 2020. arXiv 2001.00326. Opened: https://arxiv.org/abs/2001.00326 and https://ar5iv.labs.arxiv.org/html/2001.00326 (2026-11-25). **[R]**
- Shows: 15,625 cells with 1 to 3 seeds (6,219 / 1,621 / 7,785), per-epoch logs; over 500 runs regularized evolution (93.92 +- 0.30), REINFORCE (93.85 +- 0.37) and random search (93.70 +- 0.36) are within about one standard deviation. Same seed structure as ours and the same kind of near-null.
- Tabular oracle? Yes as a protocol.

**[3.4] Surrogate NAS Benchmarks: Going Beyond the Limited Search Spaces of Tabular NAS Benchmarks (NAS-Bench-301).** Zela et al., 2020 (v4 2022). arXiv 2008.09777. Opened: https://arxiv.org/abs/2008.09777 and https://ar5iv.labs.arxiv.org/html/2008.09777 (2026-11-25). **[R]**
- Shows: a single stored seed is a worse estimate of true mean performance than a surrogate fitted to the table (MAE 4.53e-3 vs 3.45e-3 on NAS-Bench-101; 1.38e-3 vs 1.13e-3 for the best surrogate on the DARTS space; XGBoost did not beat the table); the table's error shrinks with k seeds. Our 1-seed rows need a noise statement.
- Tabular oracle? Partly: a surrogate smoothed over our table could be a robustness check on the rankings.

**[3.5] YAHPO Gym: An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization.** Pfisterer et al., 2021 (v4 2022). arXiv 2109.03670. Opened: https://arxiv.org/abs/2109.03670 and https://ar5iv.labs.arxiv.org/html/2109.03670 (2026-11-25). **[R]**
- Shows: tabular versions "may produce unfaithful results regarding the performance ranking of HPO methods"; on synthetic functions GP-based BO converges faster and "Hyperband appears to consistently perform better on tabular benchmarks"; recommends 30 replications. A caveat for any ASHA win on our tables.
- Tabular oracle? Yes as a protocol (its replication and regret rules), no as a method.

**[3.6] Hyperparameter Tuning MLPs for Probabilistic Time Series Forecasting (TSBench).** Madhusudhanan et al., 2024. arXiv 2403.04477. Opened: https://arxiv.org/html/2403.04477v1 (2026-11-25). **[R]**
- Shows: the closest forecasting table: 97,200 evaluations of NLinear variants on 20 Monash datasets, 3 seeds, per-epoch logs; SMAC, Hyperband and BOHB beat random at 50 trials; some datasets have very large seed spread (Bitcoin 7.37 +- 2.21). Our PatchTST table is smaller but a different model family; this must be cited.
- Tabular oracle? Yes as a benchmark: it is itself a tabular oracle and could be a third table without new training.

**[3.7] Overtuning in Hyperparameter Optimization.** Schneider et al., 2025 (AutoML 2025). arXiv 2506.19540. Opened: https://arxiv.org/html/2506.19540v1 (2026-11-25). **[R]**
- Shows: in about 10% of cases the validation-best configuration generalises worse than the default or first one tried; FCNet shows almost none, small-data holdout studies show over 50% with any overtuning. ILI (under 1,000 points, single holdout) is in the risky regime.
- Tabular oracle? Yes: overtuning is a post-hoc statistic, computable if the table also stores test MSE.

## 4. Multi-fidelity HPO

**[4.1] A System for Massively Parallel Hyperparameter Tuning (ASHA).** Li et al., 2018 (MLSys 2020). arXiv 1810.05934. Opened: https://arxiv.org/abs/1810.05934 (2026-11-25).
- Shows: asynchronous early stopping, robust at 500 workers; our serial replay uses none of its asynchrony, so compare it as plain successive halving.
- Tabular oracle? Yes (already in our baseline set).

**[4.2] BOHB: Robust and Efficient Hyperparameter Optimization at Scale.** Falkner et al., 2018 (ICML 2018). arXiv 1807.01774. Opened: https://arxiv.org/abs/1807.01774 (2026-11-25).
- Shows: Hyperband brackets with a model-based sampler; combines our TPE and ASHA arms, so its absence is a visible gap.
- Tabular oracle? Yes: restrict sampling to table rows.

**[4.3] DEHB: Evolutionary Hyperband for Scalable, Robust and Efficient Hyperparameter Optimization.** Awad et al., 2021. arXiv 2105.09821. Opened: https://arxiv.org/abs/2105.09821 and https://ar5iv.labs.arxiv.org/html/2105.09821 (2026-11-25).
- Shows: differential evolution inside Hyperband; evaluated on NAS-Bench-101, -1shot1, -201 and NAS-HPO-Bench, with categorical values mapped to bins of [0,1]; strong on discrete spaces.
- Tabular oracle? Yes: it has already been run on tabular lookups with discrete spaces.

**[4.4] PriorBand: Practical Hyperparameter Optimization in the Age of Deep Learning.** Mallik et al., 2023. arXiv 2306.12370. Opened: https://arxiv.org/abs/2306.12370 and https://arxiv.org/html/2306.12370 (2026-11-25).
- Shows: mixes uniform, prior and incumbent sampling inside Hyperband, and stays robust when the prior is bad; 50 seeds on surrogate benchmarks. Using the untrained LLM's proposal distribution as the prior is the clean test of whether the LLM knows anything useful.
- Tabular oracle? Yes: discrete prior over rows; needs a prior we supply (the LLM or the default).

**[4.5] In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization (ifBO).** Rakotoarison et al., 2024 (ICML 2024). arXiv 2404.16795. Opened: https://arxiv.org/abs/2404.16795 and https://arxiv.org/html/2404.16795v3 (2026-11-25). **[R]**
- Shows: epoch-by-epoch allocation with a pretrained learning-curve network; LCBench, Taskset and PD1 "are tabular, meaning that the set of possible configurations to sample from is finite"; 1,000-step budget, 10 seeds. The strongest baseline that matches our accounting exactly.
- Tabular oracle? Yes: finite pool, one step = one epoch; metric scaling of raw MSE needs a check.

**[4.6] Is One Epoch All You Need For Multi-Fidelity Hyperparameter Optimization?** Egele et al., 2023. arXiv 2307.15422. Opened: https://arxiv.org/abs/2307.15422 and https://ar5iv.labs.arxiv.org/html/2307.15422 (2026-11-25). **[R]**
- Shows: one epoch for all, then train the top K, matches SHA, Hyperband and curve extrapolation with about ten times less computation on HPOBench, LCBench and YAHPO; authors ask that it be always included.
- Tabular oracle? Yes: trivially.

## 5. Statistical practice for many-seed comparisons

**[5.1] Deep Reinforcement Learning at the Edge of the Statistical Precipice (rliable).** Agarwal et al., 2021 (NeurIPS 2021). arXiv 2108.13264. Opened: https://arxiv.org/abs/2108.13264 and https://ar5iv.labs.arxiv.org/html/2108.13264 (2026-11-25). **[R]**
- Shows: report IQM, score distributions and stratified bootstrap intervals; with overlapping intervals, compute the interval of the **difference**; intervals with 3 runs are too narrow, and few tasks give wide ones. We have two tasks, so aggregate claims are weak whatever the seed count.
- Tabular oracle? No (analysis tool); applies directly to the replay output.

**[5.2] Accounting for Variance in Machine Learning Benchmarks.** Bouthillier et al., 2021. arXiv 2103.03098. Opened: https://arxiv.org/abs/2103.03098 and https://ar5iv.labs.arxiv.org/html/2103.03098 (2026-11-25). **[R]**
- Shows: randomise as many sources as possible; use paired P(A>B) with gamma = 0.75, for which 29 trainings suffice at 5% errors; data-split variance exceeds initialisation variance. Table replay randomises the search only, not data splits, so state that.
- Tabular oracle? No (analysis); 29 paired search seeds is cheap on a table.

**[5.3] How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments.** Colas et al., 2018. arXiv 1806.08295. Opened: https://arxiv.org/abs/1806.08295 and https://ar5iv.labs.arxiv.org/html/1806.08295 (2026-11-25). **[R]**
- Shows: choose N from effect size, pilot standard deviation and power; prefer Welch's t-test; "a bootstrap test should not be used with less than 20 samples"; small pilots underestimate N; correct for multiple comparisons.
- Tabular oracle? No (analysis).

**[5.4] Statistical Comparisons of Classifiers over Multiple Data Sets.** Demsar, 2006, JMLR 7:1-30. No arXiv id. Opened: https://www.jmlr.org/papers/v7/demsar06a.html (2026-11-25).
- Shows: Wilcoxon signed-rank for two methods, Friedman plus post-hoc for many, across data sets. Designed for many data sets; with two tasks our unit of replication is seeds, not data sets, so this does not license a claim across tasks.
- Tabular oracle? No (analysis).

**[5.5] Should we really use post-hoc tests based on mean-ranks?** Benavoli et al., 2015. arXiv 1505.02288. Opened: https://arxiv.org/abs/1505.02288 (2026-11-25).
- Shows: whether A beats B by mean-rank post-hoc can flip depending on which other methods are in the pool; use sign or Wilcoxon pairwise tests (then correct for multiplicity, see Could not open on Holm).
- Tabular oracle? No (analysis).

**[5.6] Equivalence testing and run-count design for stochastic machine-learning experiments.** Knopov and Shvai, 2026. arXiv 2610.23700. Opened: https://arxiv.org/abs/2610.23700 and https://arxiv.org/html/2610.23700 (2026-11-25). **[R]**
- Shows: "no significant difference" from 2 or 3 runs does not show equivalence; pre-set margin plus TOST or a confidence interval inside it, closed-form run counts, "below n = 5 we do not rely on the formula at all", upper confidence bound on pilot variance. The method to turn our null into a claim. Its case study is AUROC, not search regret, so recompute the variance.
- Tabular oracle? No (analysis); search seeds are cheap to add on a table.

## 6. Time-series forecasting context

**[6.1] A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST).** Nie et al., 2022 (ICLR 2023). arXiv 2211.14730. Opened: https://arxiv.org/abs/2211.14730 (2026-11-25).
- Shows: the model family of our oracle (patching, channel independence).
- Tabular oracle? No: it is the object being tuned.

**[6.2] Are Transformers Effective for Time Series Forecasting? (DLinear).** Zeng et al., 2022. arXiv 2205.13504. Opened: https://arxiv.org/abs/2205.13504 (2026-11-25).
- Shows: one-layer linear models beat earlier transformers on nine datasets; give the best table configuration's test MSE next to a linear reference so "good configuration" means something.
- Tabular oracle? No (a forecaster); useful as a reference row.

**[6.3] TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods.** Qiu et al., 2024. arXiv 2403.20150. Opened: https://arxiv.org/abs/2403.20150 and https://arxiv.org/html/2403.20150v4 (2026-11-25). **[R]**
- Shows: "drop last" in testing changes PatchTST's error from 0.4203 (batch size 1) to 0.3483 (batch size 512). If batch size is a searched hyperparameter and the test loader drops the last batch, the table is biased: check our oracle code.
- Tabular oracle? No (benchmark).

**[6.4] SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters.** Lin et al., 2024. arXiv 2405.00946. Opened: https://arxiv.org/html/2405.00946 (2026-11-25). **[R]**
- Shows: names the "long-standing bug" that discards the incomplete final test batch; largest effect on ETTh1 and ETTh2 with large batches. After the fix PatchTST's ETTh1 row reads 0.385/0.413/0.440/0.456 versus the original 0.370/0.413/0.422/0.447 (different sources, not like for like).
- Tabular oracle? No (a forecaster).

**[6.5] There are no Champions in Supervised Long-Term Time Series Forecasting** (v1: Position: There are no Champions in Long-Term Time Series Forecasting). Brigato et al., 2025 (TMLR, v2 January 2026). arXiv 2502.14045. Opened: https://arxiv.org/abs/2502.14045 and https://arxiv.org/html/2502.14045v1 (2026-11-25).
- Shows: in about 5,000 trained networks (v1: 40 Optuna trials per search, 3 seeds, Friedman and sign tests) small setup changes reverse "state of the art"; v1 notes subsets may drop small sets "like ILI". Supports publishing a tuned, tested, null comparison.
- Tabular oracle? No (evaluation study), but its TPE-plus-successive-halving setup is a reference protocol.

**[6.6] Simple Feedforward Neural Networks are Almost All You Need for Time Series Forecasting** (title as printed: "Feedfoward"). Sun et al., 2025. arXiv 2503.23621. Opened: https://arxiv.org/abs/2503.23621 and https://arxiv.org/html/2503.23621 (2026-11-25). **[R]**
- Shows: ILI, Weather and Exchange "are too small and idiosyncratic to serve as reliable benchmarks"; ILI has under 1,000 steps and strong train-test drift (COVID). Weakens ILI as one of our two tasks, and Weather as a candidate third.
- Tabular oracle? No (a forecaster).

## Additional pages opened (one line each, same rules)

| # | Work | Page opened (2026-11-25) | Why it matters | Tabular oracle? |
|---|---|---|---|---|
| A1 | Large Language Models as Optimizers (OPRO), Yang et al., 2023, 2309.03409 | https://arxiv.org/abs/2309.03409 | Prompt-history loop, the parent of LLM-as-optimizer work; its gains are in prompt optimisation, not HPO | Partly: prompt loop, no fidelity |
| A2 | Using Large Language Models for Hyperparameter Optimization, Zhang et al., 2023, 2312.04528 | https://arxiv.org/abs/2312.04528 | LLMs match BO only under tight budgets | Partly: no fidelity |
| A3 | Large Language Model Agent for Hyper-Parameter Optimization (AgentHPO), Liu et al., 2024, 2402.01881 | https://arxiv.org/abs/2402.01881 | 12 tasks versus human trials, not versus tuned classical search | Partly: no fidelity |
| A4 | EvoPrompting, Chen et al., 2023, 2302.14838 | https://arxiv.org/abs/2302.14838 | LLMs alone fail at NAS by prompting; evolution plus soft-prompt tuning needed | No: writes code |
| A5 | LLMatic, Nasir et al., 2023, 2306.01102 | https://arxiv.org/abs/2306.01102 | LLM plus quality-diversity on NAS-Bench-201, about 2,000 evaluations | No: writes code |
| A6 | Greedy Is a Strong Default: Agents as Iterative Optimizers, Li, 2026, 2603.27415 | https://arxiv.org/abs/2603.27415 | Parallel investigators and a second LLM added nothing over greedy hill climbing (3 runs) | Partly: LLM proposals plus greedy acceptance |
| A7 | LEMUR 2, Uzun et al., 2026, 2607.06839 | https://arxiv.org/html/2607.06839 | GRPO loop for an LLM writing networks, rewarded by a mini-epoch; no untrained-LLM or random arm | No: writes code |
| A8 | Understanding R1-Zero-Like Training (Dr. GRPO), Liu et al., 2025, 2503.20783 | https://arxiv.org/abs/2503.20783 | Finds an optimisation bias in GRPO and proposes an unbiased variant; an ablation arm | Yes: advantage formula |
| A9 | GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number, Bay and Yearick, 2026, 2607.00152 | https://arxiv.org/abs/2607.00152 | For right/wrong rewards, the group standard deviation sets the update size; where rivals could matter | Yes: advantage formula |
| A10 | Elo-Evolve, Zhao et al., 2026, 2602.13575 | https://arxiv.org/abs/2602.13575 | Opponent pool with Elo-based sampling for LLM alignment: rivals are in the **reward**, not only in the baseline | No: preference task |
| A11 | Regularized Evolution for Image Classifier Architecture Search, Real et al., 2018, 1802.01548 | https://arxiv.org/abs/1802.01548 | Evolution reaches results faster than RL on equal hardware, especially early | Yes (already a baseline) |
| A12 | Random Search and Reproducibility for NAS, Li and Talwalkar, 2019, 1902.07638 **[R]** | https://ar5iv.labs.arxiv.org/html/1902.07638 | Report a "multiple of random search" and several independent runs | Yes: random search with early stopping |
| A13 | NAS evaluation is frustratingly hard, Yang et al., 2019, 1912.12522 **[R]** | https://ar5iv.labs.arxiv.org/html/1912.12522 | Many NAS methods barely beat the average architecture; two seeds of 32 architectures correlate at Kendall tau 0.48 | Yes as protocol (random-architecture baseline) |
| A14 | HPOBench, Eggensperger et al., 2021, 2109.06716 | https://arxiv.org/abs/2109.06716 | Containerised multi-fidelity tables; 13 optimizers compared | Yes as protocol |
| A15 | Auto-PyTorch Tabular (LCBench), Zimmer et al., 2020, 2006.13799 | https://arxiv.org/abs/2006.13799 | Source of LCBench learning curves used by [4.5] | Yes as protocol |
| A16 | JAHS-Bench-201, Bansal et al., 2022, NeurIPS Datasets and Benchmarks, no arXiv id found | https://proceedings.neurips.cc/paper_files/paper/2022/hash/fd78f2f65881c1c7ce47e26b040cf48f-Abstract.html | Surrogate joint architecture-HPO benchmark with fidelities | Yes as protocol |
| A17 | NAS-Bench-Suite, Mehta et al., 2022, 2201.13396 | https://arxiv.org/abs/2201.13396 | Conclusions from a few benchmarks often do not generalise (25 combinations); we have two | Yes as protocol |
| A18 | Exploring the Loss Landscape in NAS, White et al., 2020, 2005.02960 | https://arxiv.org/abs/2005.02960 | Hill climbing beats many methods when noise is low | Yes: one-edit neighbourhood on rows |
| A19 | Hyperband, Li et al., 2016, 1603.06560 | https://arxiv.org/abs/1603.06560 | Bracketed successive halving | Yes |
| A20 | SMAC3, Lindauer et al., 2021, 2109.09831 | https://arxiv.org/abs/2109.09831 | Maintained BO package (BSD) | Yes: pool-restricted |
| A21 | DyHPO: Supervising the Multi-Fidelity Race of Hyperparameter Configurations, Wistuba et al., 2022, 2202.09774 | https://ar5iv.labs.arxiv.org/html/2202.09774 | Epoch-level race with deep-kernel GP on LCBench, TaskSet, NAS-Bench-201 | Yes |
| A22 | The Unreasonable Effectiveness Of Early Discarding After One Epoch, Egele et al., 2024, 2404.04111 | https://arxiv.org/abs/2404.04111 | Proposes the "i-Epoch" baseline and judging against its epochs-versus-loss Pareto front | Yes |
| A23 | Deep Reinforcement Learning that Matters, Henderson et al., 2017, 1709.06560 | https://arxiv.org/abs/1709.06560 | Reporting without significance metrics makes gains uninterpretable | No (analysis) |
| A24 | A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms, Colas et al., 2019, 1904.06979 | https://arxiv.org/abs/1904.06979 | Compares tests on false-positive rate and power by seed count | No (analysis) |
| A25 | Hyperparameters in Reinforcement Learning and How To Tune Them, Eimer et al., 2023, 2306.01324 | https://arxiv.org/abs/2306.01324 | Use separate tuning and testing seeds; applies to the architect's training | No (analysis) |
| A26 | Time for a change: ... Bayesian analysis, Benavoli et al., 2016, 1606.04316 | https://arxiv.org/abs/1606.04316 | Bayesian alternative to NHST; the abstract does not mention equivalence regions, so check the paper before using them | No (analysis) |
| A27 | Reversible Instance Normalization ... (RevIN), Kim et al., 2022 (ICLR) | https://iclr.cc/virtual/2022/poster/6034 | Removes and restores per-instance statistics to handle distribution shift; context if our search space has a RevIN axis | No |
| A28 | On the Role of Reversible Instance Normalization, Berthelier et al., 2026, 2603.11869 | https://arxiv.org/abs/2603.11869 | Several RevIN components are "redundant or even detrimental"; a RevIN on/off axis would need this | No |
| A29 | Seeking SOTA: ... Taxonomy-Specific Evaluation, Saqur et al., 2026, 2603.15506 | https://arxiv.org/abs/2603.15506 | Gains often come from strongly seasonal benchmarks; argues for structural breaks | No |
| A30 | A Unified Hyperparameter Optimization Pipeline for Transformer-Based TSF Models, Xu et al., 2025, 2501.01394 | https://arxiv.org/abs/2501.01394 | HPO pipeline for transformer forecasters; no table promised in the abstract | Partly: not a method |
| A31 | Auto-PyTorch-TS: Efficient Automated Deep Learning for Time Series Forecasting, Deng et al., 2022, 2205.05511 | https://arxiv.org/abs/2205.05511 | BO plus multi-fidelity for forecasting, with a study of budget types | Partly: full system |
| A32 | Optimizing Time Series Forecasting Architectures: A Hierarchical NAS, Deng and Lindauer, 2024, 2406.05088 | https://arxiv.org/abs/2406.05088 | NAS for forecasting; the abstract does not describe the search method | Partly |
| A33 | When Large Language Models Know the Table, Silvestri et al., 2025, 2510.20351 | https://arxiv.org/abs/2510.20351 | Contamination clear in 4 of 8 public tabular datasets; a way to test whether the LLM knows ETTh1 or ILI | No (audit) |

Excluded on purpose: AutoResearch-RL (arXiv 2603.07300, PPO-trained LLM editing a training script). The page opened, but both versions are marked withdrawn by arXiv for violating policy, so it should not be cited as evidence.

## 7. NeurIPS Datasets and Benchmarks requirements (most recent call I could open: 2026)

The 2026 call is the **Evaluations & Datasets (E&D) Track**: "Formerly the Datasets & Benchmarks Track". Call: https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets . Hosting: https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting . FAQ: https://neurips.cc/Conferences/2026/EvaluationsDatasetsFAQ . RAI post: https://blog.neurips.cc/2026/05/04/responsible-ai-metadata-requirements-for-the-evaluations-and-datasets-track-neurips-2026/ . All opened 2026-11-25; quotes marked [R] checked in raw text. No 2027 call exists at https://neurips.cc/Conferences/2027 (404), so these rules are for a closed cycle (full papers were due 6 May 2026) and must be rechecked.

- **Null results are in scope [R].** The call asks for papers that "present negative results, critical analyses". This is the best positioning for our result.
- **Hosting [R].** Data must be "properly hosted, accessible, and clearly documented upon submission", preferably on Dataverse, Kaggle, Hugging Face or OpenML; "new datasets should remain available long-term". A GitHub repo or Zenodo is allowed as "other platforms" if you build the Croissant file yourself and reviewers can reach the URL. Over 4 GB needs a small sample.
- **Croissant [R].** Required for submissions that introduce datasets, with core and Responsible AI fields (limitations, biases, sensitive information, use cases, social impact, synthetic-data flag, source datasets, provenance), one validated file per dataset. The host fills the core fields only. Missing RAI fields are flagged at review and "may constitute grounds for rejection" (hosting page); invalid files risk desk rejection.
- **Code [R].** "Code should be documented and executable. Non-compliance justifies the desk rejection of the paper." Release is required at submission when the contribution is a reusable executable artifact such as a benchmark suite; our tabular oracle would be exactly that. Anonymised hosting is expected.
- **Licence.** The hosting page requires a clear licence in the Croissant file and says that if datasets are processed versions of existing ones, "and this is allowed under their license and intended use", the same rules apply [R]. This is where [8.1] bites.
- **Review.** Double-blind by default; single-blind only if anonymisation is impossible. Anonymous hosting accounts; no identifying text inside Croissant files.
- **If the source data is only slightly processed [R]:** the FAQ says you can cite the original source URLs and provide preprocessing code, with no hosting and no Croissant for that data. If processing is a key part of the contribution, hosting rules apply. Shipping a download-and-build script plus our own per-epoch metrics tables may fit this route.

## 8. Licences of candidate datasets

Not legal advice; this is what the opened pages say. "Derived files" means re-scaled, split, windowed or merged copies of the values. Our per-epoch MSE tables are a different thing: I found no page that addresses them.

| # | Dataset | Page opened (2026-11-25) | Licence text | Redistributing derived files? |
|---|---|---|---|---|
| 8.1 | **ETDataset** (ETTh1) | https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/LICENSE **[R]**; README: https://github.com/zhouhaoyi/ETDataset/blob/main/README.md | Creative Commons **Attribution-NoDerivatives 4.0**: "You do not have permission under this Public License to Share Adapted Material." README has only a citation request. | **No** for adapted CSVs. Unmodified copies with attribution: yes. Ship a script, not the data. |
| 8.1b | The same files in https://huggingface.co/datasets/thuml/Time-Series-Library (raw README opened with curl) | same page | Whole repository marked `cc-by-4.0`, ETTh1 included | Conflicts with the upstream CC BY-ND file above. Do not rely on it. |
| 8.2 | **Weather** (Jena, MPI-BGC) | https://www.bgc-jena.mpg.de/wetter/weather_data.html **[R]** (the landing page https://www.bgc-jena.mpg.de/wetter/ has no terms) | "Terms of Use (as per Creative Commons CC-BY-4.0)" on the download page, with ZIP files by period | **Yes, with attribution**, if built from the institute's ZIPs. The usual `weather.csv` mirror (Autoformer README, https://github.com/thuml/Autoformer) states no dataset licence, so do not copy that file. |
| 8.3 | **Electricity** (UCI ElectricityLoadDiagrams20112014) | https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 **[R]** | "licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license" | **Yes, with attribution.** Domain is still electricity load, so it may not satisfy the "other domain" rule. The 321-client file people use comes from the next row. |
| 8.4 | **Traffic** (Caltrans PeMS, via LSTNet) | https://pems.dot.ca.gov/ **[R]**; https://zenodo.org/records/4656132; https://archive.ics.uci.edu/dataset/204/pems+sf | PeMS site: "In general, information presented on this web site, unless otherwise indicated, is considered in the public domain. It may be distributed or copied as permitted by law." Zenodo (Monash, hourly, 862 series) and UCI PEMS-SF (different file): CC BY 4.0. | Looks allowed through Zenodo or UCI with attribution. The PeMS text is a website policy, not a data licence, and says other owners' material needs separate permission. |
| 8.5 | **Exchange-rate** (LSTNet) | https://github.com/laiguokun/multivariate-time-series-data; https://huggingface.co/datasets/THUgewu/exchange_rate | GitHub repo: **no licence stated**, and no source named for this series. The HF copy says CC BY 4.0 but derives from another HF repo (8.1b), not from an identified owner. | **Not cleared.** Provenance unknown; also criticised in [6.6]. |
| 8.6 | **Solar** (NREL, now NLR) | https://www.nlr.gov/grid/solar-power-data.html (the old nrel.gov host did not resolve); https://zenodo.org/records/4656144 **[R]** | NLR page: **no licence stated**; says data are synthetic and "should not generally be used for site-specific project development work". Zenodo (Monash, 10-minute, 137 series): CC BY 4.0. | Via Zenodo: looks allowed. Upstream terms unverified: the NREL dataset-licence page would not open (below). |
| 8.7 | **Extra candidate: Beijing Multi-Site Air Quality** | https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data | CC BY 4.0 | **Yes, with attribution.** Environment domain, hourly, has missing values ("NA"). [6.5] lists a "BeijingAir" set; I did not check it is the same one. |
| 8.8 | **Extra candidate: Melbourne Pedestrian Counts** (Monash) | https://zenodo.org/records/4656626 | CC BY 4.0 | **Yes, with attribution.** Transport domain, hourly, 66 sensors, 4.6 MB. [6.5] lists a "PedestrianCounts" set (not checked to be the same file). |

Reading: cleanest third datasets are Jena weather rebuilt from the institute's files (8.2) or Beijing air quality (8.7) or pedestrian counts (8.8). The Monash archive's own site (https://forecastingdata.org/, opened) says its datasets are "intended to use only for research purpose" and lists no licence, while the Zenodo records say CC BY 4.0: cite the Zenodo record. I did not check the ILI (CDC) terms.

## Could not open

- **OpenML 45981** (Jena weather copy): api.openml.org redirected to www.openml.org/d/45981, which returned no content. Not used.
- **www.nrel.gov/grid/solar-power-data.html** and **data.nrel.gov/node/1/license**: host did not resolve; **data.nlr.gov/node/1/license** returned 503. So the NREL/NLR dataset licence is unread.
- **Holm 1979**: the JSTOR page (stable/4615733) failed to load. Holm is not cited; use a page you can open, or a secondary source, before naming it.
- **neurips.cc/Conferences/2027**: 404.
- **arXiv PDFs of 2606.21641 and 2510.20351** came back as undecodable PDF; the HTML/abstract pages were used instead. **Jena station PDF** (Weatherstation.pdf) opened, but its text could not be extracted.
- **arXiv HTML of 2502.14045v2** served only part of the paper; the v1 HTML was used for statistical-test and trial-count details, and the v2 numbers (about 5,000 networks, eight models) come from the abstract page.
- **2603.07300** opened but withdrawn (see above).

## Candidates for Phase B baselines that could run on the oracle

Ordered by value per hour of work. "Cost" is in our own accounting: one oracle epoch = one charge.

| Priority | Baseline | Source | Why | What is needed |
|---|---|---|---|---|
| 1 | One-epoch then top-K, plus the i-Epoch Pareto front | [4.6], A22 | Explicitly requested as a default baseline; the strongest cheap test of whether epochs are being spent well | ~30 lines; sweep K and i |
| 2 | Default-seeded and LLM-seeded random search, TPE and ASHA | [1.5] | The control that removes the warm-start confound: every method gets the same first configuration | Replay harness flag |
| 3 | ifBO | [4.5] | Finite-pool, epoch-level, uses tables; same accounting as ours; code linked from the paper (https://github.com/automl/ifBO, not opened by me) | Metric scaling for MSE; check the licence |
| 4 | PriorBand with the untrained LLM's proposal distribution as prior | [4.4] | Separates "LLM knows something" from "training helped" | A discrete prior over rows |
| 5 | Local search / hill climbing on one-edit neighbours | A18 | Known to be strong on tables with low noise; cheapest strong baseline | Neighbour function |
| 6 | BOHB and DEHB | [4.2], [4.3] | Standard model-based and evolutionary multi-fidelity; DEHB already handles discrete tables | Pool restriction; bin mapping as in the DEHB paper |
| 7 | Hyperband | A19 | Needed to interpret ASHA; [3.5] warns it may look better on tables than in real runs | Bracket schedule |
| 8 | DyHPO, SMAC3 (pool-restricted) | A21, A20 | Second freeze-thaw/BO family | Heavier dependencies |
| 9 | Dr. GRPO / Team-GRPO / rival-in-reward variants of our own architect | A8, [2.4] | Tests where the league can matter; these are ablations of our method, not external baselines | Change the advantage formula only |
| Not recommended | OptFormer [2.2], SemanticOpt [1.3], Universal Searcher [1.6], GRPOformer [2.5], NNGPT [1.7], LEMUR 2 (A7) | | Need large offline corpora, large models or code generation; no epoch fidelity | Cite, do not run |

Also reportable on the existing oracle without new methods: overtuning rate [3.7], top-decile rank correlation by epoch [3.2], a seed-noise surrogate check [3.4], paired P(A>B) [5.2], and TOST with a pre-set margin [5.6].

## Reading notes by the authors (not findings of any paper)

* The league, as implemented, adds rivals only to the group's mean and standard deviation; rivals
  are drawn from configs evaluated before the step, so they do not depend on the current proposals.
  Shifting the baseline by something independent of the proposals leaves the expected gradient
  unchanged; the rival term can act only through the standard deviation, a common scale on the
  step. This is our own argument about our code, consistent with [2.4] and [2.3], not a test.
  It predicts what the CPU controller (AB-C) and E4's league against none show: no detectable difference.
* Baselines suggested here that could run on the oracle and are not in the paper yet: one epoch for
  every config followed by the top K [4.6], the same first configuration for every method [1.5], and
  equivalence tests with a pre-set margin [5.6]. Any such run needs an amendment first.
* Entries without [R] rest on abstracts or summaries. Claims quoted in the paper should be re-read
  in the full text first.
