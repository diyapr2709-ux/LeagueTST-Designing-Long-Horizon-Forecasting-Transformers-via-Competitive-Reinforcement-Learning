# LeagueTST: related work and where the project stands (lit-scout, 2026-11-26)

Scope: every paper below was opened in this session (arXiv abstract page, arXiv/ar5iv HTML, or the named
venue page) on 2026-11-26. Rule 9: only these may be cited. Entries marked [HTML] had their full text
opened and the quoted number or sentence read there; entries without it rest on the abstract page.
Pages that would not open are listed at the end. Project numbers are copied from
results/search/E3/ILI_T24/summary.md, results/search/E3/ETTh1_T336/summary.md,
results/search/E4/ILI_T24/summary.md, results/independent/E4_soumitra.json and docs/STATE.md; nothing
was recomputed, and nothing under results/ablations/rtx/ was read.

## 0. What the project does, in one paragraph

A Qwen2.5-1.5B-Instruct architect, LoRA fine-tuned with GRPO (group 8, KL 0.04, constrained decoding over
a trie of 2,368 feasible PatchTST-style configs), proposes configs that are scored by lookup in two
pre-computed validation-MSE tables (ETTh1 T=336, one seed for most configs; ILI T=24, three seeds). A
"league" puts 4 free rivals (past-snapshot-weighted configs from the episode) into the GRPO group baseline.
Methods are compared at equal training-epoch budgets (10, 25, 50, 100 full-run equivalents of 30 epochs)
against random search, regularized evolution, Optuna multivariate TPE, ASHA, a random+referee screen, a
factorized policy-gradient controller and the same LLM without RL; the primary RL test is transfer
(train on one table, deploy frozen on the other) against a portfolio and warm-started TPE. A learned
referee (early-epoch features) is tested as a ranker of the architect's own proposals (RQ2). Everything
is pre-registered (docs/PLAN.md, docs/prereg_E4.md), seeded, Holm-corrected, and independently re-scored.

Committed state as of 26 Nov (from the files named above, not recomputed):

* E3 ILI AUC (held-out-seed regret, pooled-SD units, 30 search seeds for CPU methods, 10 for Qwen):
  TPE 1.14, referee 1.22, ASHA 1.38, RE 1.47, epoch-k 1.50, random 1.52, PG league 1.59, PG uniform rivals
  1.61, PG 1.69, PG league charged 1.77, prompted Qwen 1.81 (last).
* E3 ETTh1 AUC (in-sample, "no claims"): prompted Qwen 1.30, PG 1.32, TPE 1.37, ..., random 1.61.
* Pick reseeds (fresh seeds 4001, 4002): on both tables no method beats random search after Holm,
  prompted Qwen included; TPE's in-table ILI win does not replicate.
* E4 transfer, ILI target (independent re-score): GRPO league from ETTh1 AUC mean 3.14, GRPO none 3.23,
  portfolio 3.28, warm-started TPE 1.65. League vs warm TPE p = 1e-5 with TPE better; league vs portfolio
  p = 0.64; league vs none p = 0.80; J3 (uniform rivals) not run under A30. Within-table GRPO was dropped
  by amendment (A12/A18) because 25 to 50 equivalents allow only 2 to 8 updates.
* Family 3, ILI, unadjusted: prompted Qwen worse than random at budget 25 (p = 0.048) and than TPE
  (p = 0.0018); at budget 50 vs random p = 0.54, vs TPE p = 0.0083.
* K3 passes (invalid 0% under constrained decoding; mean unique rate per group >= 0.91).
* E5 (referee on architect proposals): tau difference vs quality-matched random sets ILI +0.009
  [-0.029, +0.048], ETTh1 +0.020 [-0.009, +0.049]; margin set after seeing prompted Qwen, so not
  confirmatory.

So the two LLM arms (prompted, GRPO-transferred) trail TPE and random on ILI AUC, the league is
indistinguishable from no league, and on fresh seeds nothing beats random.

## 1. Papers opened, by theme

Format: id, URL, what it does, how it relates, numbers (only where read on the page).

### 1a. LLM-driven NAS and HPO

**GENIUS, Zheng et al. 2023.** arXiv 2304.10970, https://arxiv.org/abs/2304.10970 and
https://arxiv.org/html/2304.10970v4 [HTML]. GPT-4 as a prompted black-box NAS optimiser, 10 iterations
per run. Relates: our prompted-Qwen arm is the same idea with an open 1.5B model and a trie over feasible
rows. Numbers: NAS-Bench-201 at temperature 0, 5 trials: CIFAR-10 test 93.79 +- 0.09, CIFAR-100
70.91 +- 0.72, ImageNet16-120 44.96 +- 1.02, "slightly behind" the leaders (94.36 / 73.51 / 46.34);
NAS-Bench-Macro best model rank 8/6561, worst 61/6561 over 3 trials. The authors write that "we cannot
rule out contamination represents a significant caveat to our findings" and "even with the temperature
set to 0, we observe some variation in GPT-4 responses." No RE or BO baseline, no seeds reported.

**EvoPrompting, Chen, Dohan and So 2023.** arXiv 2302.14838, https://arxiv.org/abs/2302.14838. LM as
mutation/crossover operator in evolutionary code-level NAS, with soft-prompt tuning; beats naive few-shot
prompting on MNIST-1D and 21 of 30 CLRS tasks. Relates: an LLM-in-the-loop design method that writes code,
not rows; its claim that prompting alone is weak matches our prompted arm's ILI result. No seeds on the page.

**LLMatic, Nasir et al. 2023 (GECCO 2024).** arXiv 2306.01102, https://arxiv.org/abs/2306.01102. LLM code
generation plus quality-diversity search; "competitive" networks on CIFAR-10 and NAS-Bench-201 with about
2,000 candidate evaluations. Relates: 2,000 evaluations is close to enumerating our whole 2,368-row table,
so its budget regime is the one where our methods cannot separate (A3). No numbers on the page.

**CoLLM-NAS, Li, Lin and Wang 2025 (CVPRW 2026).** arXiv 2509.26037, https://arxiv.org/abs/2509.26037.
Two LLMs (Navigator and Generator) over a supernet; claims SOTA on NAS-Bench-201 with 4 to 10x lower
search cost. Relates: a strong recent prompted-LLM NAS baseline; its NAS-Bench-201 numbers (94.37 / 73.44
/ 46.79 test) are quoted in Hu et al. below, not on its own abstract page.

**OptFormer, Chen et al. 2022 (NeurIPS 2022).** arXiv 2205.13320, https://arxiv.org/abs/2205.13320.
Text transformer trained by imitation on Google Vizier tuning data; imitates at least 7 HPO algorithms
and gives better-calibrated predictions than a GP. Relates: the learned-optimizer route needs a large
offline corpus; we have two tables, which is why we use RL on lookups instead.

**LLAMBO, Liu et al. 2024 (ICLR 2024).** arXiv 2402.03921, https://arxiv.org/abs/2402.03921 and
https://arxiv.org/html/2402.03921v2 [HTML]. GPT-3.5 as warm-starter, surrogate and candidate sampler
inside BO. Protocol: Bayesmark (25 public tasks plus 30 private/synthetic) and 24 HPOBench tabular tasks,
25 trials, 5 initial points, 5 seeds (one section says ten), baselines GP, TPE (Optuna), SMAC3, GP-DKL.
Finding: "LLAMBO excels in earlier stages of the search, when fewer observations are available." Relates:
the positive LLM-HPO result closest to our budget regime; it needs a closed 175B-class model and
continuous spaces, whereas our space is categorical and our model is 1.5B.

**OPRO, Yang et al. 2023 (ICLR 2024).** arXiv 2309.03409, https://arxiv.org/abs/2309.03409. LLM as
optimizer with a scored-history prompt; gains are in prompt optimisation (up to 8% GSM8K, up to 50% BBH).
Relates: parent of the prompt-history loop our episode prompt uses.

**Using LLMs for HPO, Zhang et al. 2023 (v2 2024).** arXiv 2312.04528, https://arxiv.org/abs/2312.04528
and https://arxiv.org/html/2312.04528 [HTML]. GPT-4 Turbo prompted on HPOBench sklearn tasks. Numbers:
at 10 evaluations "GPT-4 Turbo beats random search 81.25% of the time" with best mean rank 2.42 (BO-RF
3.45, BO-GP 3.80); at 30 evaluations beats random on 90.62% of tasks; at 100 evaluations ranks 2.08 behind
BO-GP 1.88. HPOBench runs use a single seed at temperature 0. Relates: the strongest published claim that a
prompted LLM beats random and BO at small budgets; our prompted arm is behind random and TPE on ILI at 25
and 50 equivalents with 10 seeds. Differences: closed frontier model, continuous sklearn spaces, one seed,
possible contamination of public HPOBench tasks.

**AutoML-Agent, Trirat, Jeong and Hwang 2024 (ICML 2025).** arXiv 2410.02958,
https://arxiv.org/abs/2410.02958. Multi-agent LLM pipeline (planning, sub-task agents, verification),
seven tasks, fourteen datasets, "higher success rate". Relates: full-pipeline agent system; the abstract
names no matched-budget HPO baseline, so it does not bear on RQ1.

**SemanticOpt (was GPTOpt), Meindl et al. 2025 (v3 2026).** arXiv 2510.25404,
https://arxiv.org/abs/2510.25404. LLM fine-tuned on BO trajectories with natural-language context;
outperforms classical optimizers "on average when relevant semantic information is provided". Relates:
supervised rather than RL fine-tuning of an LLM optimizer.

**Can LLMs beat classical HPO? (autoresearch), Ferreira et al. 2026.** arXiv 2603.24647,
https://arxiv.org/abs/2603.24647 and https://arxiv.org/html/2603.24647v5 [HTML]. 14 hyperparameters of
a ~50M nanochat model, 24 GPU-hours per method, 3 seeds, val_bpb. Numbers: TPE 0.9768 +- 0.0019, SMAC
0.9778 +- 0.0020, CMA-ES 0.9785 +- 0.0036, Random 0.9873 +- 0.0021, LLAMBO (Optuna) 0.9882 +- 0.0012,
Karpathy agent (14 HPs) 0.9904 +- 0.0002; Centaur hybrid 0.9763 +- 0.0005 (Qwen3.5-27B) and 0.9766 with a
0.8B model. "Several pure LLM methods performed worse than random search." Relates: live (not tabular),
one task, but the ordering TPE > random > pure LLM is ours.

**When is an LLM worth it for HPO? Rodrigues et al. 2026.** arXiv 2606.21641,
https://arxiv.org/abs/2606.21641. Budget-matched LLM advisor vs random, Optuna-TPE, GP-BO, successive
halving on 8 PMLB tabular datasets, 40 task-seed units, paired tests and bootstrap CIs. Numbers: a fixed
default reaches 88.7% mean best-CV alone; LLM adds +0.40 pp CV and -0.01 pp test (p = 0.92); with the
same default given to random search the LLM lead is +0.20 pp at 2 evaluations, ties at 5, is behind at 12
(-0.37 pp); without the seed classical search ties the advisor by 12 and beats it by 40 evaluations
(+0.6 to +0.8 pp, p <= 1e-4). Relates: the closest published null to ours, and the source of the
"same first configuration for every method" control we do not yet run.

**Greedy is a strong default, Li 2026.** arXiv 2603.27415, https://arxiv.org/abs/2603.27415. LLM agent
as the proposal step of hill climbing; matches Optuna TPE with 2x efficiency on one LoRA task; parallel
investigators and a second LLM gave no benefit. Relates: another small-scale report where the LLM's value
is equal to, not above, TPE.

**Agentic BO (Sara), Brunzema et al. 2026.** arXiv 2608.00316, https://arxiv.org/abs/2608.00316. LLM
agent steering a BoTorch backend; "preserves the reliability of state-of-the-art BO without prior
knowledge" and beats LLM-based baselines. Relates: the 2026 consensus design is LLM-around-BO, not
LLM-instead-of-BO.

**LM-Searcher (Universal Searcher), Hu et al. 2026.** arXiv 2611.12791, https://arxiv.org/abs/2611.12791
and https://arxiv.org/html/2611.12791 [HTML]. LLaMA-3.1-8B fully fine-tuned by supervised learning on
about 228,000 trajectories sampled from NAS-Bench-301 sub-spaces; ranks candidates encoded as numeric
strings; GRPO is used only to post-train a Qwen3-8B variant to emit chain-of-thought (reward = accuracy of
picking the optimal candidate plus format), 2,000 trajectories, 8 responses per prompt. NAS-Bench-201
test accuracy: Random Search 93.89 / 72.07 / 45.98; GENIUS 93.79 / 70.91 / 44.96; LLMatic 94.26 / 71.62
/ 45.87; CoLLM-NAS 94.37 / 73.44 / 46.79; theirs 94.20 / 72.96 / 46.51; the table has no RE or BO row.
Relates: the nearest prior "LLM trained on a NAS table" and the only one combining Qwen, GRPO and a
tabular NAS benchmark; but the searcher itself is SFT, GRPO is interpretive, and the 8B model and 228k
trajectories are far beyond our 1.5B model and 60 GRPO steps.

### 1b. RL-trained proposers, GRPO and its baselines, RL for design generation

**NAS with RL, Zoph and Le 2016.** arXiv 1611.01578, https://arxiv.org/abs/1611.01578. Recurrent
controller trained by policy gradient on validation accuracy (CIFAR-10 3.65% error). Relates: our
factorized categorical controller (A4) is the lineage; NAS-Bench-101 and Klein and Hutter below report
how that lineage does on tables.

**Hyp-RL, Jomaa, Grabocka and Schmidt-Thieme 2019.** arXiv 1906.11527, https://arxiv.org/abs/1906.11527.
An RL policy replaces the acquisition function; 50 datasets. Relates: RL-as-HPO-proposer predates LLMs.

**DeepSeekMath (GRPO), Shao et al. 2024.** arXiv 2402.03300, https://arxiv.org/abs/2402.03300 and
https://ar5iv.labs.arxiv.org/html/2402.03300 [HTML]. Defines GRPO: rewards "normalized by subtracting the
group average and dividing by the group standard deviation", KL added to the loss with the k3-type
estimator, KL coefficient 0.04 in their run. Relates: our optimizer, including the 0.04 KL weight.

**DeepSeek-R1, DeepSeek-AI 2025 (Nature 645).** arXiv 2501.12948, https://arxiv.org/abs/2501.12948.
Pure RL on verifiable tasks induces reasoning. Relates: the result that made "GRPO on a verifiable
reward" the default; table lookups are a verifiable reward.

**Dr. GRPO, Liu et al. 2025.** arXiv 2503.20783, https://arxiv.org/abs/2503.20783. Identifies an
optimisation bias in GRPO (response length inflation) and gives an unbiased variant. Relates: an
alternative advantage formula for an ablation arm.

**When rewards depend on the group, GRPO plays a tournament, Gao et al. 2026.** arXiv 2611.02404,
https://arxiv.org/abs/2611.02404. If rewards depend on groupmates, the group-relative baseline "changes
the objective" and GRPO becomes a zero-sum tournament weighted -1/(K-1); Team-GRPO corrects it. Relates:
directly about what rivals in a GRPO baseline can do. Our rivals are drawn from configs evaluated before
the step, so they are independent of the current proposals; a baseline independent of the response
leaves the expected gradient unchanged and can act only through the standard-deviation scale. That is
the a-priori reason to expect league = none, which is what E4 shows (p = 0.80).

**GRPOformer, Guo, Pan and Zhai 2025.** arXiv 2509.17105, https://arxiv.org/abs/2509.17105 and
https://arxiv.org/html/2509.17105v1 [HTML]. A transformer trained from scratch with GRPO (plus a
policy-churn regulariser) to generate hyperparameter configurations from trajectories; 36 OpenML
scenarios; baselines BOformer, LLM-HPO, OPTformer, SBOA; mean rank 1.81 in Table 1 and 1.31 in the
ablation table; seeds and statistical tests not reported. Relates: prior art for "GRPO for HPO"; not an
LLM, no random-search or TPE row, no seeds.

**NNGPT, Kochnev et al. 2025 (CVPRW 2026).** arXiv 2511.20333, https://arxiv.org/abs/2511.20333. LLM as a
self-improving AutoML engine over the LEMUR dataset; RL is one of five pipelines; HPO vs Optuna RMSE 0.60
vs 0.64. Relates: an RL-fine-tuned code-writing architect; no random-search or untrained-LLM control on
the page, no seeds.

**LEMUR 2, Uzun et al. 2026.** arXiv 2607.06839, https://arxiv.org/abs/2607.06839. 14,000+ architectures
and 750,000+ training records, generated partly by "reinforcement-learning evolution" and LLM-guided
synthesis. Relates: a code-level architecture corpus; not a lookup table with a fixed space.

**From memorization to creativity, Khalid, Ignatov and Timofte 2026 (CVPRW 2026).** arXiv 2601.02997,
https://arxiv.org/abs/2601.02997. 22 cycles of SFT (LoRA) on self-generated, novelty-filtered CNN code;
mean first-epoch accuracy 28.1% to 51.0%; no RL, no external baseline. Relates: shows the self-improvement
loop without RL; our GRPO is the RL analogue.

**Delta-based NAS, Adhikari, Timofte and Ignatov 2026.** arXiv 2605.04903,
https://arxiv.org/abs/2605.04903. LoRA fine-tuning of 7B coders to emit diffs; valid rate 75.3% / mean
first-epoch accuracy 65.8% (DeepSeek-Coder-7B) vs 50.6% / 42.3% for full generation; 1-epoch proxy
preserves ranking (Spearman 0.926 for Mistral). Relates: a fidelity-1 proxy claim that is the opposite of
our E2 finding (epoch-1 tau 0.18 on ILI, -0.06 on ETTh1), because our recipe runs epoch 1 at rate/25.

**RAAS, Yang et al. CVPR 2026.** https://cvpr.thecvf.com/virtual/2026/poster/38932 (the open-access
HTML returned 403). GRPO over a probabilistic space of LLM agentic-system architectures; averages several
independent runs per candidate to remove execution noise. Relates: GRPO for architecture search in a
different domain; its multi-trial reward is the analogue of our multi-seed lookups (A2).

**AlphaEvolve, Novikov et al. 2025.** arXiv 2506.13131, https://arxiv.org/abs/2506.13131. LLM-driven
evolutionary code modification with evaluator feedback; 4x4 complex matrix multiplication in 48 scalar
multiplications. Relates: the strongest evidence that LLM-plus-evolution designs things; it is a code
evolver with enormous evaluation budgets, not a budget-matched HPO comparison.

### 1c. Self-play, leagues and populations

**AlphaStar, Vinyals et al. 2019, Nature 575, DOI 10.1038/s41586-019-1724-z.** Opened via
https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/
(the nature.com article page would not open; see "Could not open"). League of main agents and
exploiters exists because self-play forgets how to beat earlier versions ("chasing its tail") and to keep
strategy diversity. Relates: the word "league" in our method. In our task a config's MSE does not depend
on any rival, so the forgetting/cycling motivation does not transfer; A5 and A23 already say
"competitive" is claimed only if league beats both none and uniform rivals, which it does not.

**TStarBot-X, Han et al. 2020.** arXiv 2011.13729, https://arxiv.org/abs/2011.13729 and
https://ar5iv.labs.arxiv.org/html/2011.13729 [HTML]. "a faithful reimplementation of AlphaStar's methods
can not succeed" at lower compute; "the strategic diversity and strength in the entire league is
significantly limited"; the main agent "converges to either Zergingling Rush or Roach Push". Relates:
small leagues collapse; our K3 unique-rate check (>= 0.91 mean) is the diversity log.

**Population Based Training, Jaderberg et al. 2017.** arXiv 1711.09846, https://arxiv.org/abs/1711.09846.
Asynchronous population jointly tunes weights and hyperparameters; the page does not mention leagues.
Relates: a population method for HPO that is not competitive; a natural extra baseline if live training
were affordable (it is not on a lookup table).

**SPIRAL, Liu et al. 2025 (ICLR 2026).** arXiv 2506.24119, https://arxiv.org/abs/2506.24119. Self-play on
zero-sum games against improving copies; up to 10% reasoning gains on eight benchmarks. **Absolute Zero,
Zhao et al. 2025.** arXiv 2505.03335, https://arxiv.org/abs/2505.03335. Single model proposes and solves
its own tasks with a code executor as verifier. Relates: in both, the opponent or proposer changes the
reward; in ours it does not, which is the structural reason our "league" is a baseline shift.

**Elo-Evolve, Zhao et al. 2026.** arXiv 2602.13575, https://arxiv.org/abs/2602.13575. Opponent pool with
Elo-based sampling; learning from pairwise win/loss outcomes. Relates: rivals in the reward rather than
in the baseline; a design we could ablate.

### 1d. Tabular NAS/HPO benchmarks and their protocols

**Klein and Hutter 2019, tabular benchmarks for joint architecture and HPO.** arXiv 1905.04970,
https://arxiv.org/abs/1905.04970 and https://ar5iv.labs.arxiv.org/html/1905.04970 [HTML]. 62,208
feed-forward configs, 4 seeds each, four UCI regression datasets (MSE), 500 runs per optimizer; random
search, SMAC, TPE, BOHB, Hyperband, RE, RL, Bohamiann. RE "achieved the best final performance" on
Protein but was less robust than TPE; RL "seems to be too sample inefficient for these tasks". Relates:
the closest protocol template (regression, MSE, seeds, full curves), and a prior finding that RL
controllers lose on tables.

**NAS-Bench-101, Ying et al. 2019.** arXiv 1902.09635, https://arxiv.org/abs/1902.09635 and
https://ar5iv.labs.arxiv.org/html/1902.09635 [HTML]. 423k architectures, 3 seeds, budgets 4/12/36/108
epochs; RS, RE, SMAC, TPE, HB, BOHB, REINFORCE over 500 runs: "RE, BOHB, and SMAC perform best" and
overtake RS only after about 50,000 TPU-seconds; the PPO/LSTM controller "seems to fail to learn to
traverse the space"; top-10% rank correlation between 36 and 108 epochs is 0.365. Relates: random search
is hard to beat early; low-fidelity rankings are unreliable among the leaders, which matches our
E2 and E3b findings (Spearman 0.08 within the ETTh1 top 20).

**NAS-Bench-201, Dong and Yang 2020 (ICLR 2020).** arXiv 2001.00326, https://arxiv.org/abs/2001.00326
and https://ar5iv.labs.arxiv.org/html/2001.00326 [HTML]. 15,625 cells; seeds: 6,219 with 1, 1,621 with
2, 7,785 with 3; CIFAR-10 test accuracy over 500 runs: RS 93.70 +- 0.36, REA 93.92 +- 0.30, REINFORCE
93.85 +- 0.37, BOHB 93.61 +- 0.52. Relates: the same mixed seed structure as ours and a near-null between
random, evolution and REINFORCE within one SD; our E3 table is the same picture.

**NAS-Bench-301, Zela et al. 2020 (v4 2022).** arXiv 2008.09777, https://arxiv.org/abs/2008.09777 and
https://ar5iv.labs.arxiv.org/html/2008.09777 [HTML]. "In contrast to a single tabular entry, the
surrogate model learns to smooth out the noise"; on NAS-Bench-101 the surrogate's MAE is about 3.45e-3
vs about 4.54e-3 for a tabular entry; about 60k DARTS architectures trained. Relates: our ETTh1 table has
one seed for most rows, so A18's Gaussian offset and the fresh-seed picks are the mitigation.

**NAS-Bench-x11, Yan et al. 2021 (NeurIPS 2021).** arXiv 2111.03602, https://arxiv.org/abs/2111.03602.
Surrogates that return full learning curves, because two popular tabular benchmarks lack them, enabling
learning-curve extrapolation. Relates: our tables store per-epoch validation MSE, which is what makes
ASHA, epoch-k and the referee chargeable in epochs.

**HPO-B, Pineda Arango et al. 2021.** arXiv 2106.06257, https://arxiv.org/abs/2106.06257. 176 search
spaces, 196 datasets, about 6.4M evaluations, fixed protocols and splits. Relates: a protocol precedent
for explicit splits; black-box only (no fidelity).

**YAHPO Gym, Pfisterer et al. 2021 (v4 2022).** arXiv 2109.03670, https://arxiv.org/abs/2109.03670 and
https://ar5iv.labs.arxiv.org/html/2109.03670 [HTML]. Tabular benchmarks "may produce unfaithful results
regarding the performance ranking of HPO methods"; "Hyperband appears to consistently perform better on
tabular benchmarks"; 30 replications per run, with averages and standard errors recommended. Relates: a
caveat on our ASHA and screening rows, and the source of our 30-seed convention.

**JAHS-Bench-201, Bansal et al. 2022 (NeurIPS 2022 D&B).**
https://proceedings.neurips.cc/paper_files/paper/2022/hash/fd78f2f65881c1c7ce47e26b040cf48f-Abstract-Datasets_and_Benchmarks.html
(no arXiv id found). 14-dimensional joint space with epochs as fidelity, about 161M data points, three
tasks. Relates: joint arch+HP with an epoch fidelity, as ours.

**HPOBench, Eggensperger et al. 2021 (NeurIPS 2021 D&B).** arXiv 2109.06716,
https://arxiv.org/abs/2109.06716 and https://ar5iv.labs.arxiv.org/html/2109.06716 [HTML]. 12 families,
13 optimizers, 32 seeds; "advanced methods consistently outperform random search"; "multi-fidelity
optimizers outperform black-box optimizers for relatively small compute budgets" (at 1% of budget HB,
DEHB, SMAC-HB beat their black-box counterparts), with no significant advantage at full budget; "DEHB
and SMAC-HB are reliable multi-fidelity optimizers". Relates: at our budgets (25 to 50 full runs out of
2,368) the literature expects multi-fidelity methods to lead; on our tables ASHA does not (AUC 1.38 vs
TPE 1.14 on ILI), plausibly because epoch 1 runs at rate/25 (E2).

**LCBench / Auto-PyTorch Tabular, Zimmer, Lindauer and Hutter 2020.** arXiv 2006.13799,
https://arxiv.org/abs/2006.13799. Introduces the LCBench learning-curve benchmark. Relates: the
learning-curve table ifBO and DyHPO are evaluated on.

**NAS-Bench-Suite, Mehta et al. 2022.** arXiv 2201.13396, https://arxiv.org/abs/2201.13396. Across 25
space-dataset combinations "many conclusions drawn from a few NAS benchmarks do not generalize".
Relates: we have two tables that anti-correlate (Spearman -0.21); transfer of config preferences from
ETTh1 to ILI (portfolio AUC 3.28) is a worked example.

**TSBench, Madhusudhanan, Jawed and Schmidt-Thieme 2024 (PAKDD 2024).** arXiv 2403.04477,
https://arxiv.org/abs/2403.04477 and https://arxiv.org/html/2403.04477v1 [HTML]. NLinear-family MLPs on
20 Monash datasets, 97,200 evaluations, seeds 100/101/102, 50 epochs with per-epoch train/val/test
metrics stored; HPO with 50 trials: "SMAC, HyperBand and BOHB outperform the Random strategy". Relates:
the only forecasting-specific tabular HPO benchmark I found; different model family (MLP, probabilistic
head) and datasets, but it is the one to cite as precedent for a forecasting table.

**pTNAS, Xing et al. 2024 (v3 2026).** arXiv 2403.10318, https://arxiv.org/abs/2403.10318. Zero-cost
proxy filter plus fixed-budget refinement for tabular-data networks. Relates: tangential (tabular data,
not tabular benchmark); the abstract page does not name NAS-Bench-Tabular.

No tabular NAS benchmark of forecasting transformers with per-epoch curves was found in two extended
searches; EvoTS (below) and the hierarchical TS-NAS paper search live.

### 1e. HPO baselines

**Random search, Bergstra and Bengio 2012, JMLR 13.** https://www.jmlr.org/papers/v13/bergstra12a.html.
Random beats grid because few hyperparameters matter. **Algorithms for HPO, Bergstra et al. 2011, NIPS
24.** https://proceedings.neurips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html.
"two new greedy sequential methods based on the expected improvement criterion"; the page does not name
TPE. **TPE components, Watanabe 2023 (v5 2026).** arXiv 2304.11127, https://arxiv.org/abs/2304.11127.
Names TPE as the BO method in Hyperopt and Optuna and ablates its control parameters. **Optuna, Akiba et
al. 2019.** arXiv 1907.10902, https://arxiv.org/abs/1907.10902. Our TPE is
optuna.samplers.TPESampler(multivariate=True) over categorical fields (leaguetst/search/replay.py),
so cite Optuna plus Watanabe for the algorithm and Bergstra 2011 for the origin.

**Hyperband, Li et al. 2016 (JMLR 2018).** arXiv 1603.06560, https://arxiv.org/abs/1603.06560. "over an
order-of-magnitude speedup" over BO competitors. **ASHA, Li et al. 2018 (MLSys 2020).** arXiv 1810.05934,
https://arxiv.org/abs/1810.05934. Asynchronous successive halving; our serial replay uses none of the
asynchrony. **BOHB, Falkner, Klein and Hutter 2018 (ICML 2018).** arXiv 1807.01774,
https://arxiv.org/abs/1807.01774. "consistently outperforms both Bayesian optimization and Hyperband";
absent from our set. **DEHB, Awad, Mallik and Hutter 2021.** arXiv 2105.09821,
https://arxiv.org/abs/2105.09821. "up to 1000x faster than random search"; strong on discrete spaces;
absent from our set. **SMAC3, Lindauer et al. 2021 (JMLR).** arXiv 2109.09831,
https://arxiv.org/abs/2109.09831. **Regularized evolution, Real et al. 2018 (AAAI 2019).** arXiv
1802.01548, https://arxiv.org/abs/1802.01548. Evolution "can obtain results faster with the same
hardware, especially at the earlier stages of the search" than RL. On our ILI table RE (AUC 1.47) is
behind TPE and ahead of random, on ETTh1 it is behind both (in-sample).

**PriorBand, Mallik et al. 2023.** arXiv 2306.12370, https://arxiv.org/abs/2306.12370. Hyperband with an
expert prior, robust to bad priors. Relates: the clean test of "does the LLM know anything": use the
prompted LLM's proposal distribution as the prior. **ifBO, Rakotoarison et al. 2024 (ICML 2024).** arXiv
2404.16795, https://arxiv.org/abs/2404.16795. In-context freeze-thaw BO with a PFN surrogate, 10 to 100x
faster predictions. Relates: epoch-level allocation that matches our accounting. **Is one epoch all you
need? Egele et al. 2023.** arXiv 2307.15422, https://arxiv.org/abs/2307.15422. Train all for one epoch,
keep top-K; matches MF-HPO methods with "an order of magnitude less computation". Relates: our epoch-k
row is this baseline at k = 1 (ILI) and k = 3 (ETTh1); it is the second-worst method on ILI (AUC 1.50),
again because of the rate/25 first epoch.

**HPO survey, Bischl et al. 2021.** arXiv 2107.05847, https://arxiv.org/abs/2107.05847. General reference
for the methods above.

### 1f. Learned performance predictors and zero-cost proxies (the referee analogue)

**Zero-cost proxies, Abdelfattah et al. 2021 (ICLR 2021).** arXiv 2101.08134,
https://arxiv.org/abs/2101.08134. One-minibatch proxies; best Spearman 0.82 with final accuracy on
NAS-Bench-201 vs 0.61 for EcoNAS; 4x faster to the same accuracy on NAS-Bench-101. **How powerful are
performance predictors? White et al. 2021 (NeurIPS 2021).** arXiv 2104.01177,
https://arxiv.org/abs/2104.01177. 31 predictors across learning-curve extrapolation, weight sharing,
supervised and zero-cost families; combining families helps. **NAS-Bench-Suite-Zero, Krishnakumar et
al. 2022 (NeurIPS 2022 D&B).** arXiv 2210.03230, https://arxiv.org/abs/2210.03230. 13 proxies, 28 tasks;
proxies carry complementary information; adding them to surrogates improves prediction by up to 42%.
**LLM performance predictors, Jawahar et al. 2023 (ACL Findings 2024).** arXiv 2310.16712,
https://arxiv.org/abs/2310.16712. GPT-4 as a prompted predictor for MT architectures; slight drop in rank
correlation vs baselines; HS-NAS halves search time. Relates: our referee is a supervised predictor on
early-epoch features (tau 0.58 ILI, 0.85 ETTh1 against the table); RQ2 asks the question these papers do
not: does the predictor still rank when the candidates come from a narrow, LLM-chosen band. E5's
quality-matched control is the right design for that, and the ETTh1 drop of 0.095 against plain random
sets is the expected "narrow band is harder to rank" effect.

### 1g. Long-horizon forecasting transformers and their seed variance

**PatchTST, Nie et al. 2022 (ICLR 2023).** arXiv 2211.14730, https://arxiv.org/abs/2211.14730 and
https://ar5iv.labs.arxiv.org/html/2211.14730 [HTML]. Table 14: supervised PatchTST/42 with seeds 2019 to
2023, ETTh1 MSE 0.3752 +- 0.0008 (96), 0.4127 +- 0.0012 (192), 0.4278 +- 0.0033 (336), 0.4462 +- 0.0035
(720); "higher variance can be seen on smaller datasets"; ILI is not in Table 14, and A.6.2 says "Except
ILI dataset reveals high variance with different hyper-parameter settings" (hyperparameter, not seed,
sensitivity). Relates: the model family of our tables and the training recipe we fix. Note the scale: the
published ETTh1 T=336 seed SD is 0.0033 at the paper's own settings, while our pooled seed SD over the
top-100 configs is 0.0132 and the ILI pooled SD is 0.0098; our tables are noisier than the headline
model, which is why regret is reported in SD units.

**MTST, Zhang et al. 2023 (AISTATS 2024).** arXiv 2311.04147, https://arxiv.org/abs/2311.04147 and
https://arxiv.org/html/2311.04147v2 [HTML]. Table 7 reruns PatchTST at L=336 with seeds 2021 to 2023:
ETTh1 MSE 0.375 +- 0.0008, 0.413 +- 0.0012, 0.428 +- 0.0033, 0.446 +- 0.0035; "using a different random
seed has minimal impact on the forecast". Relates: an independent reproduction of the PatchTST seed SD.

**DLinear, Zeng et al. 2022.** arXiv 2205.13504, https://arxiv.org/abs/2205.13504. One-layer linear
models beat earlier transformers on nine datasets. Relates: the reference row for "is the best table
config good at all" and the A7 caution that validation and test can disagree.

**RevIN, Kim et al. ICLR 2022.** https://iclr.cc/virtual/2022/poster/6034 (OpenReview page would not
open). Instance normalisation and denormalisation with learnable affine. Relates: the field that explains
83% of the ETTh1 table's variance (E8) and the A23 RevIN study.

**TFB, Qiu et al. 2024 (PVLDB 2024).** arXiv 2403.20150, https://arxiv.org/abs/2403.20150 and
https://arxiv.org/html/2403.20150v4 [HTML]. "implementations of existing methods often employ a 'Drop
Last' trick in the testing phase"; PatchTST on ETTh2 with drop-last at batch size 1 / 32 / 64 / 128 /
256 / 512 reads 0.4203 / 0.4138 / 0.3999 / 0.3750 / 0.3561 / 0.3483; 8 hyperparameter sets per method.
Relates: the literature basis for our rule 2 (no drop_last anywhere in evaluation).

**There are no champions in supervised LTSF, Brigato et al. 2025 (TMLR, v2 2026).** arXiv 2502.14045,
https://arxiv.org/abs/2502.14045 and https://arxiv.org/html/2502.14045v1 [HTML]. Eight models (v2), 14
datasets, about 5,000 networks; v1: Optuna "with a budget of 40 trials", three seeds for the final model,
Friedman test (statistic 1.14, p = 0.95) and pairwise sign tests; dropping one horizon gives "three
different champions in terms of MSE ... out of four different cases"; ILI is not among their datasets and
is mentioned as a "too small" set other papers drop. Relates: tuned, seeded, tested comparisons find no
champion; our E6 framing ("the space's best config against PatchTST/42", A23) sits inside that finding.

**Simple feedforward networks, Sun, Wu and Boning 2025.** arXiv 2503.23621,
https://arxiv.org/abs/2503.23621 and https://arxiv.org/html/2503.23621 [HTML]. ILI, Weather and
Exchange "are too small and idiosyncratic to serve as reliable benchmarks for model performance"; "The
ILI dataset is particularly small, containing fewer than 1,000 time steps across seven series" and "shows
significant distribution drift between the training and testing sets". Relates: A11 already treats ILI as
a robustness check, not primary evidence; cite this when saying so.

**Hierarchical NAS for forecasting, Deng and Lindauer 2024 (v2 2025).** arXiv 2406.05088,
https://arxiv.org/abs/2406.05088 and https://arxiv.org/html/2406.05088 [HTML]. DARTS-PT over a
hierarchical forecasting space, 5 seeds; ETTh1 average MSE 0.413 (DARTS-TS) vs 0.415 (PatchTST); no ILI.
**EvoTS, ElSaid and Pulatov 2026.** arXiv 2607.00154, https://arxiv.org/abs/2607.00154. Evolutionary
search over transformer-like genomes on the four ETT sets, "competitive and, in several cases, improved"
MSE vs a transformer baseline; no numbers on the page. **EvoCast, Xu et al. 2026.** arXiv 2610.04517,
https://arxiv.org/abs/2610.04517. LLM research agents evolve forecasting architectures under
deterministic promotion rules on "three real-world forecasting cases"; datasets not named on the page.
Relates: live NAS for forecasting exists (gradient, evolutionary, LLM-agent), none on a lookup table, none
with budget-matched HPO baselines and seeds.

**Overtuning, Schneider, Bischl and Feurer 2025 (AutoML 2025).** arXiv 2506.19540,
https://arxiv.org/abs/2506.19540. In roughly 10% of cases the validation-best configuration generalises
worse than the default or the first one tried. Relates: the A7/A8 validation-to-test checks.

**Table contamination, Silvestri et al. 2025.** arXiv 2510.20351, https://arxiv.org/abs/2510.20351.
Contamination in 4 of 8 public tabular datasets. Relates: ETTh1 and ILI are public; our tables are new,
but a Qwen memorisation check of the raw series would be the audit.

### 1h. Statistical practice

**rliable, Agarwal et al. 2021 (NeurIPS 2021).** arXiv 2108.13264, https://arxiv.org/abs/2108.13264.
Interval estimates, performance profiles, IQM. **Accounting for variance, Bouthillier et al. 2021.**
arXiv 2103.03098, https://arxiv.org/abs/2103.03098. Randomise data sampling, initialisation and
hyperparameter choice; our replay randomises the search seed only. **How many random seeds? Colas,
Sigaud and Oudeyer 2018.** arXiv 1806.08295, https://arxiv.org/abs/1806.08295. Power analysis for t-test
and bootstrap tests. **Deep RL that matters, Henderson et al. 2017 (AAAI 2018).** arXiv 1709.06560,
https://arxiv.org/abs/1709.06560. **Demsar 2006, JMLR 7.** https://www.jmlr.org/papers/v7/demsar06a.html.
Wilcoxon for two methods, Friedman plus post-hoc for many, across data sets (we have two, so seeds are our
replication unit). **Equivalence testing and run-count design, Knopov and Shvai 2026.** arXiv 2610.23700,
https://arxiv.org/abs/2610.23700. A non-significant difference is not equivalence; pre-specify a margin,
use TOST or a CI inside the margin, size runs from the margin and pilot variance (an upper confidence
bound on the pilot variance); certifying a 0.01 AUROC margin at 80% power "needs twelve even under exact
equality". Relates: our RQ1 null (league = none, LLM <= random) is currently "not rejected", not
"equivalent"; this is the protocol to upgrade it. **NAS evaluation is frustratingly hard, Yang,
Esperanca and Carlucci 2019 (ICLR 2020).** arXiv 1912.12522, https://arxiv.org/abs/1912.12522. "many
NAS techniques struggle to significantly beat the average architecture baseline"; "the seed has a
considerable impact on architecture rankings". **Random search and reproducibility for NAS, Li and
Talwalkar 2019 (UAI 2019).** arXiv 1902.07638, https://arxiv.org/abs/1902.07638. Random search with early
stopping is "a competitive NAS baseline".

## 2. Comparison on protocol dimensions

| Work | Search space | Budget accounting | Seeds / repeats | Tabular or live | Baselines | Statistical tests | Pre-registered |
|---|---|---|---|---|---|---|---|
| **LeagueTST (ours)** | 2,368 PatchTST-style configs (74 architectures x 32 training settings), fixed recipe | training epochs actually run per looked-up config; equal totals; budgets fixed before replay (A3) | 30 search seeds CPU methods, 10 LLM; 3 table seeds ILI, 1 (+3 for top 100) ETTh1; regret on held-out seeds | tabular, validated live (E3b: 64 retrained configs, K7 pass on ETTh1, fail on ILI, fallback applied) | random, RE, TPE, ASHA, epoch-k, referee, factorized PG (+league, +uniform rivals), prompted LLM, portfolio, warm TPE | seed-level permutation, Holm within families, bootstrap intervals, independent re-score | yes (PLAN amendments, prereg_E4, digests, cut-offs) |
| GENIUS 2304.10970 | NAS-Bench-201, -Macro, Channel-Bench | GPT-4 iterations (10); no epoch accounting | 5 / 3 trials, no seeds | tabular | none in the tables I read (RS only for Macro, unreported values) | none | no |
| LLAMBO 2402.03921 | Bayesmark continuous HPO + HPOBench tabular | 25 trials | 5 (or 10) seeds | mixed | GP, TPE, SMAC3, GP-DKL | none stated on pages read | no |
| Zhang et al. 2312.04528 | HPOBench sklearn, CIFAR-10 | 10 / 30 / 60 / 100 evaluations | 1 seed (HPOBench), 3 (CIFAR) | tabular + live | random, BO-RF, BO-GP | ranks, win rates; no tests | no |
| Rodrigues 2026 2606.21641 | sklearn HPO on 8 PMLB tasks | matched evaluation counts (2, 5, 12, 40) | 40 task-seed units | live | random, Optuna-TPE, GP-BO, successive halving | paired tests, bootstrap 95% CIs, p-values | not stated |
| Ferreira 2026 2603.24647 | 14 HPs of a 50M LM | 24 GPU-hours per method (LLM time excluded) | 3 seeds | live | TPE, SMAC, CMA-ES, random | mean +- std only | no |
| GRPOformer 2509.17105 | 36 OpenML HPO scenarios | not stated on pages read | not reported | OpenML tabular | BOformer, LLM-HPO, OPTformer, SBOA (no random/TPE) | none | no |
| LM-Searcher 2611.12791 | NAS-Bench-201 search, NAS-Bench-301 training | not stated in the table read | not stated | tabular + live downstream | RS, DARTS family, GENIUS, LLMatic, CoLLM-NAS (no RE/BO) | none | no |
| NNGPT 2511.20333 | code-level CNNs (LEMUR) | not stated | not reported | live | Optuna (HPO only) | none | no |
| Klein & Hutter 1905.04970 | 62,208 MLP configs, 4 UCI regression sets | wall-clock/epochs of looked-up configs | 4 table seeds, 500 runs | tabular | RS, SMAC, TPE, BOHB, HB, RE, RL, Bohamiann | regret curves, no formal tests on pages read | no (benchmark) |
| NAS-Bench-201 2001.00326 | 15,625 cells | simulated train time | 1 to 3 table seeds, 500 runs | tabular | RS, REA, REINFORCE, BOHB + weight-sharing methods | mean +- std | no (benchmark) |
| HPOBench 2109.06716 | 12 families | fidelity-weighted budget fractions | 32 seeds | tabular/surrogate/raw | 13 optimizers | significance tests across benchmarks | no (benchmark) |
| YAHPO 2109.03670 | 14 surrogate scenarios | fidelity budget | 30 replications | surrogate (warns about tabular) | 7 + 7 optimizers | ranks with standard errors | no (benchmark) |
| TSBench 2403.04477 | 1,620 NLinear-MLP configs x 20 Monash sets | 50 trials | 3 table seeds | tabular (forecasting) | random, HB, SMAC, BOHB | critical-difference diagram | no (benchmark) |
| Brigato 2502.14045 | 5 to 8 forecasting models, 14 datasets | 40 Optuna trials per model | 3 final seeds | live | models vs models | Friedman (p = 0.95), sign tests | no |

Reading of the table: no prior LLM-search paper combines epoch-charged budgets, >= 10 search seeds,
held-out-seed scoring, a pre-registered analysis and an independent re-score. The benchmark papers have
the seeds and replications but no LLM arm; the LLM papers have the LLM arm but 1 to 5 repeats and no
tests. Rodrigues 2026 is the only LLM-HPO paper with paired tests and intervals, and it is live sklearn HPO.

## 3. Where our committed results sit

* **Prompted LLM vs random and TPE.** Our prompted Qwen is last on ILI AUC (1.81 vs random 1.52, TPE
  1.14) and behind random and TPE at budget 25 (unadjusted p = 0.048 and 0.0018). This **matches**
  Rodrigues 2026 (random ahead by 12 evaluations once given the same default), Ferreira 2026 (TPE
  0.9768 < random 0.9873 < LLAMBO-Optuna 0.9882 < agent 0.9904 in val_bpb), Li 2026 (LLM agent at most
  equal to TPE) and the benchmark-era finding that random search is hard to beat early (NAS-Bench-101:
  RE/BOHB/SMAC overtake RS only after about 50,000 TPU-seconds; Li and Talwalkar; Yang et al.). It
  **contradicts** Zhang et al. 2023 (GPT-4 Turbo beats random on 81.25% of HPOBench tasks at 10
  evaluations, single seed), LLAMBO (early-stage gains with GPT-3.5) and GENIUS (top 0.12% of
  NAS-Bench-Macro). The disagreements line up with three differences, not one: model scale (1.5B vs
  frontier closed models; Ferreira's hybrid needed only 0.8B but only because CMA-ES carried the state),
  space type (categorical rows vs continuous sklearn ranges), and evaluation (10 seeds with held-out
  table seeds vs 1 to 5 runs on public tasks the LLM may have seen; GENIUS's authors say so themselves).
  On ETTh1 (in-sample, no claims) prompted Qwen has the best AUC (1.30) and PG next (1.32), which is the
  pattern expected when one field (RevIN, 83% of variance) dominates; the pick reseeds remove it.

* **GRPO (transfer) vs controls.** The frozen league policy from ETTh1 has AUC 3.14 on ILI, far behind
  warm-started TPE (1.65, p = 1e-5) and indistinguishable from the portfolio (3.28, p = 0.64): the policy
  carried ETTh1's config preferences to a table whose ranking anti-correlates (Spearman -0.21). This is
  NAS-Bench-Suite's "conclusions do not generalize" in method form, and it is the opposite of what
  LM-Searcher claims for its 8B SFT searcher trained on 228k trajectories. The honest reading is that a
  1.5B policy trained for 60 GRPO steps on one table learns the table, not search behaviour; GRPOformer
  and LM-Searcher never ran this control (no warm TPE, no portfolio), so the literature has no number to
  compare against.

* **League vs none.** p = 0.80, means 3.14 vs 3.23. This is the predicted outcome from the GRPO
  definition (Shao et al.) and the tournament analysis (Gao et al.): rivals drawn from configs evaluated
  before the step are independent of the current group, so they shift the baseline without changing the
  expected gradient and can act only through the std scale. TStarBot-X shows that small leagues collapse
  even where the league does change the payoff. "Competitive" cannot be claimed (A5, A23), and the report
  should say the design was structurally inert, not that it was tested and lost.

* **Multi-fidelity and screening.** HPOBench expects multi-fidelity methods to win at small budget
  fractions; Egele et al. expect one-epoch screening to match them. On our tables ASHA (1.38) and epoch-k
  (1.50) are behind TPE and the referee because epoch-1 ranking is near useless under the audited recipe
  (rate/25 warm-up: tau 0.18 on ILI, -0.06 on ETTh1). That is a recipe-specific finding worth stating,
  with NAS-Bench-101's 0.365 top-10% correlation between 36 and 108 epochs as the general version.

* **Referee (RQ2).** The referee's tau on architect proposals is within [-0.03, +0.05] of
  quality-matched random sets; the drop against plain random sets on ETTh1 (-0.095) is what the predictor
  literature would predict for a narrow band (White et al.; Abdelfattah's 0.82 is over the whole
  NAS-Bench-201 range). No prior paper tests a predictor on LLM-chosen candidates specifically.

* **Seed noise.** Published PatchTST seed SDs on ETTh1 are 0.0008 to 0.0035 (five seeds, Table 14;
  three seeds, MTST Table 7). Our pooled table SDs are 0.0132 (ETTh1 top-100) and 0.0098 (ILI), and 50 to
  59 configs sit within one SD of the best. The tables are therefore flatter and noisier at the top than
  the headline model, which is the NAS-Bench-201 situation (RS, REA, REINFORCE within one SD) and the
  reason a null is the expected result at 25 to 50 full runs.

## 4. Which prior claims the negative result challenges or confirms

Confirms:
1. Random search with early stopping is a competitive baseline (Li and Talwalkar 1902.07638; Yang et al.
   1912.12522); RE/BOHB/SMAC need large budgets to separate from RS (NAS-Bench-101 1902.09635).
2. RL controllers are sample-inefficient on tables (Klein and Hutter 1905.04970; NAS-Bench-101's PPO
   controller); REINFORCE, REA and RS are within one SD on NAS-Bench-201 (2001.00326).
3. Budget-matched LLM advisors add nothing over seeded classical search (Rodrigues 2606.21641); pure LLM
   methods can lose to random search while TPE wins (Ferreira 2603.24647).
4. A response-independent GRPO baseline does not change the expected update (Shao et al. 2402.03300 as
   definition; Gao et al. 2611.02404 as analysis).
5. Tabular replay can misorder HPO methods, and 30 replications are needed (YAHPO 2109.03670); a single
   stored seed is a noisy estimate (NAS-Bench-301 2008.09777); conclusions from one benchmark do not
   transfer (NAS-Bench-Suite 2201.13396).

Challenges (with the caveats above about scale and space):
6. "LLMs match or beat BO under constrained budgets" (Zhang et al. 2312.04528, 81.25% win rate over
   random at 10 evaluations); "LLAMBO excels in earlier stages" (2402.03921); GENIUS reaching the top
   0.12% by prompting (2304.10970). Our result says: not with a 1.5B open model on a categorical
   forecasting space at 10 seeds, even after RL.
7. "GRPO-trained HPO consistently outperforms baselines" (GRPOformer 2509.17105), which reports no seeds,
   no random/TPE row and no tests.
8. Multi-fidelity methods lead at small budget fractions (HPOBench 2109.06716) and one-epoch screening
   matches them (Egele 2307.15422): not under a recipe whose first epoch runs at rate/25.
9. The LM-Searcher transfer claim (2611.12791): a policy trained on one table carried the wrong
   preferences to an anti-correlated table; the literature should not assume search behaviour transfers.

## 5. Ranked citations for the related-work section (12)

1. Rodrigues et al. 2026, arXiv 2606.21641 (budget-matched LLM vs classical HPO; the closest null).
2. Ferreira et al. 2026, arXiv 2603.24647 (TPE and SMAC beat pure LLM methods; some LLM methods lose to
   random; small-LLM hybrid).
3. Zhang et al. 2023, arXiv 2312.04528 (the positive small-budget LLM-HPO claim we contradict).
4. Liu et al. 2024 LLAMBO, arXiv 2402.03921 (LLM-in-BO, early-stage gains, 25 trials, 5 seeds).
5. Shao et al. 2024 DeepSeekMath, arXiv 2402.03300 together with Gao et al. 2026, arXiv 2611.02404
   (GRPO definition and the group-dependent-reward analysis that predicts league = none).
6. Guo et al. 2025 GRPOformer, arXiv 2509.17105 (prior GRPO-for-HPO; evaluation gaps our protocol fills).
7. Hu et al. 2026 LM-Searcher, arXiv 2611.12791 (LLM searcher trained on a NAS table; GRPO for CoT only;
   no RE/BO or transfer controls).
8. Klein and Hutter 2019, arXiv 1905.04970 (tabular joint arch+HP regression benchmark; RL too sample
   inefficient; protocol template).
9. Dong and Yang 2020 NAS-Bench-201, arXiv 2001.00326, with Ying et al. 2019 NAS-Bench-101, arXiv
   1902.09635 (mixed seeds per entry; RS/REA/REINFORCE within one SD; early-budget rank correlation 0.365).
10. Pfisterer et al. 2021 YAHPO, arXiv 2109.03670, with Eggensperger et al. 2021 HPOBench, arXiv
    2109.06716 (tabular caveats, 30 to 32 replications, multi-fidelity expectations at small budgets).
11. Madhusudhanan et al. 2024 TSBench, arXiv 2403.04477, with Nie et al. 2022 PatchTST, arXiv 2211.14730
    and Brigato et al. 2025, arXiv 2502.14045 (the only forecasting HPO table; our model family and its
    published seed SD; "no champions" under tuned, seeded, tested comparison).
12. Vinyals et al. 2019 AlphaStar (DOI 10.1038/s41586-019-1724-z, opened via the DeepMind blog) with Han
    et al. 2020 TStarBot-X, arXiv 2011.13729 (what a league is for, and that small leagues collapse);
    Knopov and Shvai 2026, arXiv 2610.23700 for turning the null into an equivalence claim.

Also cite in passing: GENIUS 2304.10970 (contamination caveat), EvoPrompting 2302.14838, LLMatic
2306.01102, OptFormer 2205.13320, Sun et al. 2503.23621 (ILI unreliability), TFB 2403.20150 (drop-last),
Egele et al. 2307.15422 (one-epoch baseline), Abdelfattah et al. 2101.08134 and White et al. 2104.01177
(predictors), NAS-Bench-Suite 2201.13396 (non-transfer across benchmarks).

## 6. Has anyone already done essentially this?

No. Among everything opened, no paper trains an LLM architect with RL on lookups in a tabular benchmark
of forecasting architectures and compares it at equal training-epoch budgets with random search,
regularized evolution, TPE, ASHA and the same LLM without RL, with seeds, intervals and pre-registration.
The nearest pieces, each covering one part:

* GRPO for HPO: GRPOformer (2509.17105), a from-scratch transformer on OpenML, no seeds, no random/TPE row.
* LLM trained on a NAS table and deployed as a searcher: LM-Searcher (2611.12791), 8B SFT on 228k
  NAS-Bench-301 trajectories, GRPO only for chain-of-thought, vision only, no RE/BO/transfer controls.
* RL/"GRPO-style" fine-tuning of a code-writing architect: NNGPT (2511.20333) and LEMUR 2 (2607.06839),
  live, no untrained-LLM or random arm reported on the pages read.
* Budget-matched LLM vs classical HPO with tests: Rodrigues 2026 (2606.21641), live sklearn HPO, no RL,
  no forecasting.
* Forecasting HPO table: TSBench (2403.04477), MLPs on Monash data, classical optimizers only.
* League/rival baselines in GRPO: none found; the only analysis of group-dependent baselines is Gao et
  al. (2611.02404), which predicts our null.
* LLM-agent design of forecasting architectures: EvoCast (2610.04517), live, no HPO baselines named.

Two extended searches for a tabular forecasting-transformer NAS benchmark and for an RL-trained LLM
architect against HPO baselines returned nothing closer. "First" claims should still be worded as "we
found no prior work that ..." with this list, per A11.

## 7. Suggestions that follow from the reading (for the owner; nothing here is a result)

* Add the Rodrigues control: the same first configuration for every method in the replay harness.
* Add Egele's one-epoch-then-top-K row at a k where the recipe's ranking is informative (k = 3 on ETTh1).
* State the RQ1 null as an equivalence test with a pre-set margin (Knopov and Shvai), not as "not
  rejected"; 10 LLM seeds may be too few for the margin the project would want.
* Write the league result as predicted inert (independent baseline) rather than as a defeated design,
  and cite Gao et al. for why rivals would have to enter the reward to matter.
* Say in the report that our top-of-table seed SD (0.0132, 0.0098) is 3 to 10x the published PatchTST
  ETTh1 SD at fixed settings, so the tables are noisier than the headline model.

## Could not open (not cited as evidence)

* https://www.nature.com/articles/s41586-019-1724-z (cookie redirect loop twice); AlphaStar is cited via
  the DeepMind blog page, which gives the Nature DOI.
* https://openreview.net/forum?id=cGDAkQo1C0p (verification wall); RevIN cited via the ICLR 2022
  virtual page.
* https://openaccess.thecvf.com/content/CVPR2026/html/Yang_RAAS_... (HTTP 403); RAAS cited via the
  CVPR virtual poster page.
* arXiv PDFs of 2211.14730 and the PMLR PDF of MTST came back as binary; the ar5iv and arXiv HTML
  versions were used instead, and the seed tables were read there.
* The NIPS 2011 Bergstra et al. page opened but does not name TPE; the name is taken from Watanabe
  2023 (2304.11127) and Optuna (1907.10902).
* NAS-Bench-Tabular (named in a search snippet about pTNAS) was not found on the pTNAS abstract page and
  is not cited.
* Holm 1979 was not opened; "Holm" in this file refers to the project's registered procedure, not to a
  citation.
