"""Independent re-score of the registered AB-P and AB-G families for the RTX arms (AB2, Soumitra's AB4, Diya's AB4, AB5).

Written by a verifier subagent on 26 Nov 2026 before scripts/ablations/llm_registered.py was ever run, importing
nothing from it or from scripts/ablations/ (only leaguetst.architect.runner.table_rows, the ledger loader). It
rebuilds each run's trace from its charges and the ledger, computes ILI held-out AUC regret over budgets 1..100 in
pooled seed-SD units itself, and its own permutation (200,000, seed 20261126), Spearman, Holm and sign-flip code.

    .venv/bin/python scripts/independent/ablations_rescore_B.py   # writes results/independent/AB_rescore_B.json
"""
import json, math, hashlib, sys
from pathlib import Path
import numpy as np
from scipy import stats as sst

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from leaguetst.architect.runner import table_rows

OUT = ROOT / "results" / "independent"
rows = table_rows("ILI")
# ---- table ----
fin = [float(r["best_val_mse"]) for r in rows if math.isfinite(float(r["best_val_mse"]))]
pen = 10 * max(fin)
vals = {}
dups = 0
for r in rows:
    b = float(r["best_val_mse"]); b = b if math.isfinite(b) else pen
    d = vals.setdefault(r["config_id"], {})
    if int(r["seed"]) in d: dups += 1
    d[int(r["seed"])] = b
n_nonfinite = len(rows) - len(fin)
true = {c: sum(d.values()) / len(d) for c, d in vals.items()}
most = max(len(d) for d in vals.values())
ref = {c: v for c, v in true.items() if len(vals[c]) == most}
best_true = min(ref.values())
multi = [d for d in vals.values() if len(d) >= 2]
assert len(multi) >= len(vals) / 2  # ILI: no seed exclusion
var = []
for d in multi:
    x = list(d.values()); m = sum(x) / len(x)
    var.append(sum((v - m) ** 2 for v in x) / (len(x) - 1))
sigma = math.sqrt(sum(var) / len(var))
fallback = sum(true.values()) / len(true)
seedcounts = {}
for d in vals.values(): seedcounts[len(d)] = seedcounts.get(len(d), 0) + 1
info = dict(rows=len(rows), configs=len(vals), dup_rows=dups, nonfinite=n_nonfinite, seedcounts=seedcounts,
            best_true=best_true, sigma=sigma, fallback=fallback)

def hseed(run_seed, cid):
    s = sorted(vals[cid]); h = hashlib.sha256(f"{run_seed}:{cid}".encode()).hexdigest()
    return s[int(h[:12], 16) % len(s)]

checks = {"trace_mismatch": 0, "seed_mismatch": 0, "budget_over": 0, "unseen_trace": 0}
def auc(run):
    seed = int(run["seed"])
    seen = {}
    spent = 0; full = {}; my_trace = []
    for cid, oseed, extra, kind, read in run["charges"]:
        if cid not in seen: seen[cid] = oseed
        if oseed != hseed(seed, cid): checks["seed_mismatch"] += 1
        spent += extra
        if kind == "full" and cid not in full:
            full[cid] = vals[cid][oseed]
            inc = min(full, key=full.get)
            my_trace.append((spent, inc))
    if spent > 3000: checks["budget_over"] += 1
    tr = [tuple(t) for t in run["trace"]]
    if tr != my_trace: checks["trace_mismatch"] += 1
    reg = []
    for g in range(30, 3001, 30):
        inc = None
        for s, c in my_trace:
            if s <= g: inc = c
            else: break
        if inc is None:
            v = fallback
        else:
            if inc not in seen: checks["unseen_trace"] += 1
            o = [b for s_, b in vals[inc].items() if s_ != seen[inc]]
            v = sum(o) / len(o) if o else true[inc]
        reg.append(v - best_true)
    return float(np.mean(reg) / sigma)

def load(p):
    rs = [json.loads(l) for l in open(p) if l.strip()]
    return {int(r["seed"]): auc(r) for r in rs}

A = {}
base = ROOT / "results/ablations/rtx"
for item in ["ABL_P_ref_local","ABL_P_anon_local","ABL_P_repair_local","ABL_P_05b_local","ABL_P_3b_local",
             "ABL_G_ref_local","ABL_G_beta0_local","ABL_G_beta01_local","ABL_G_steps30_local","ABL_G_steps120_local"]:
    (f,) = list((base / item / "ILI_T24").glob("*_runs.jsonl"))
    A[item] = load(f)
A["J2"] = load(ROOT / "results/search/E4/ILI_T24/grpo_league_from_ETTh1_runs.jsonl")

rng = np.random.default_rng(20261126)
NP = 200_000
def perm2(a, b):
    a = np.asarray(a); b = np.asarray(b); x = np.concatenate([a, b]); n = len(a)
    obs = a.mean() - b.mean(); cnt = 0; done = 0
    while done < NP:
        k = min(20000, NP - done)
        P = np.argsort(rng.random((k, len(x))), axis=1)
        y = x[P]; d = y[:, :n].mean(1) - y[:, n:].mean(1)
        cnt += int((np.abs(d) >= abs(obs) - 1e-12).sum()); done += k
    return (cnt + 1) / (NP + 1)
def welch(a, b):
    a = np.asarray(a); b = np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb); df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = sst.t.ppf(0.975, df); d = a.mean() - b.mean()
    return d, (d - t * se, d + t * se), se, df
def rank(x):
    return sst.rankdata(x)
def spear(x, y):
    rx, ry = rank(x), rank(y); rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / math.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
def trend(levels):
    x = np.concatenate([[i] * len(v) for i, v in enumerate(levels)]).astype(float)
    y = np.concatenate([np.asarray(v) for v in levels])
    rho = spear(x, y)
    rx = rank(x); rx = rx - rx.mean(); ry = rank(y); ry = ry - ry.mean()
    den = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    cnt = 0; done = 0
    while done < NP:
        k = min(20000, NP - done)
        P = np.argsort(rng.random((k, len(y))), axis=1)
        r = (ry[P] * rx).sum(1) / den
        cnt += int((np.abs(r) >= abs(rho) - 1e-12).sum()); done += k
    n = len(y); det = math.tanh((sst.norm.ppf(0.975) + sst.norm.ppf(0.8)) / math.sqrt(n - 3))
    return rho, (cnt + 1) / (NP + 1), n, det
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); m = len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj

v = lambda k: [A[k][s] for s in sorted(A[k])]
res = {"table": info, "checks": checks}
res["arm_means"] = {k: dict(n=len(A[k]), mean=float(np.mean(v(k))), sd=float(np.std(v(k), ddof=1))) for k in A}
res["per_run"] = {k: {str(s): A[k][s] for s in sorted(A[k])} for k in A}
# AB-P
abp = {}
for arm in ["ABL_P_anon_local", "ABL_P_repair_local"]:
    d, ci, se, df = welch(v(arm), v("ABL_P_ref_local"))
    abp[arm + "_vs_ref"] = dict(diff=d, welch95=ci, p=perm2(v(arm), v("ABL_P_ref_local")),
                                 detectable80_holm3=(sst.t.ppf(1 - 0.05 / 6, df) + sst.t.ppf(0.8, df)) * se,
                                 detectable80_unadj=(sst.t.ppf(0.975, df) + sst.t.ppf(0.8, df)) * se)
rho, p, n, det = trend([v("ABL_P_05b_local"), v("ABL_P_ref_local"), v("ABL_P_3b_local")])
abp["scale_trend"] = dict(rho=rho, p=p, n=n, detectable_rho80=det)
keys = list(abp); adj = holm([abp[k]["p"] for k in keys])
for k, a in zip(keys, adj): abp[k]["holm_p"] = a; abp[k]["reject_0.05"] = a <= 0.05
res["AB_P"] = abp
# AB-G
abg = {}
rho, p, n, det = trend([v("ABL_P_ref_local"), v("ABL_G_steps30_local"), v("ABL_G_ref_local"), v("ABL_G_steps120_local")])
abg["steps_trend"] = dict(rho=rho, p=p, n=n, detectable_rho80=det)
rho, p, n, det = trend([v("ABL_G_beta0_local"), v("ABL_G_ref_local"), v("ABL_G_beta01_local")])
abg["kl_trend"] = dict(rho=rho, p=p, n=n, detectable_rho80=det)
keys = list(abg); adj = holm([abg[k]["p"] for k in keys])
for k, a in zip(keys, adj): abg[k]["holm_p"] = a; abg[k]["reject_0.05"] = a <= 0.05
res["AB_G"] = abg
# Exploratory
trained = ["ABL_G_steps30_local", "ABL_G_ref_local", "ABL_G_steps120_local"]
pol = [float(np.mean([A[t][s] for t in trained])) for s in range(10)]
d, ci, se, df = welch(pol, v("ABL_P_ref_local"))
ex = {"policy_means": pol, "training_at_all": dict(diff=d, welch95=ci, p=perm2(pol, v("ABL_P_ref_local")),
      detectable80=(sst.t.ppf(0.975, df) + sst.t.ppf(0.8, df)) * se)}
gap = [pol[s] - A["ABL_P_ref_local"][s] for s in range(10)]
ex["paired_gap"] = dict(gaps=gap, mean=float(np.mean(gap)), n_worse=int(sum(g > 0 for g in gap)))
rhos = [spear(np.array([0., 1, 2]), np.array([A[t][s] for t in trained])) for s in range(10)]
r = np.array([0 if math.isnan(x) else x for x in rhos]); obs = r.mean(); cnt = 0
for m in range(1024):
    sg = np.array([1 if (m >> i) & 1 else -1 for i in range(10)])
    cnt += abs((sg * r).mean()) >= abs(obs) - 1e-12
ex["dose"] = dict(rhos=rhos, mean_rho=float(obs), p_exact=cnt / 1024)
res["exploratory"] = ex
sh = sorted(set(A["ABL_G_ref_local"]) & set(A["J2"]))
dd = [A["ABL_G_ref_local"][s] - A["J2"][s] for s in sh]
res["hardware_check"] = dict(seeds=sh, diffs=dd, mean=float(np.mean(dd)), sd=float(np.std(dd, ddof=1)),
                             ref_local_mean=float(np.mean([A["ABL_G_ref_local"][s] for s in sh])),
                             J2_mean=float(np.mean([A["J2"][s] for s in sh])))
json.dump(res, open(OUT / "AB_rescore_B.json", "w"), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
print(json.dumps({k: res[k] for k in res if k != "per_run"}, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
