"""Independent recomputation of the AB2 (RTX) results: ILI held-out AUC, trends, Holm, descriptives, checks.

Own implementation of the table, held-out regret and AUC; leaguetst's run_metric is used only as a cross-check.
Usage: .venv/bin/python recompute.py <repo> <out.json>
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t as student_t

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PIN = "7fd5085f9ab9b3d106db37eec917ac3ab566e4fc"
AB2_REG = "c2dda8d"
LAUNCHER = "3919d68337302590eda42f53ca10158ca27766bb488026b5a4fc8373c477b786"
REV = {"0.5B": "7ae557604adf67be50417f59c2c2f167def9a775", "1.5B": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
       "3B": "aa8e72537993ba99e69dfaafa59ed015b17504d1"}
HORIZON = {"ILI": 24, "ETTh1": 336}
PROMPTED = {"ABL_P_ref_local": ("qwen_prompted_1.5b_local", "1.5B"),
            "ABL_P_05b_local": ("qwen_prompted_0.5b_local", "0.5B"),
            "ABL_P_3b_local": ("qwen_prompted_3b_local", "3B")}
TRANSFER = {"ABL_G_ref_local": 0.04, "ABL_G_beta0_local": 0.0, "ABL_G_beta01_local": 0.1}
TMETHOD = "grpo_league_from_ETTh1"
N_PERM = 1_000_000
PERM_SEED = 20261126


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def git_ok(*args: str) -> bool:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True).returncode == 0


def show(path: str, ref: str = PIN) -> bytes | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def jsonl(data: bytes) -> list[dict]:
    return [json.loads(l) for l in data.decode().splitlines() if l.strip()]


# ------------------------------------------------------------------ own table
class MyTable:
    def __init__(self, rows: list[dict]):
        finite = [float(r["best_val_mse"]) for r in rows if math.isfinite(float(r["best_val_mse"]))]
        penalty = 10.0 * max(finite)
        self.diverged = 0
        self.best: dict[str, dict[int, float]] = defaultdict(dict)
        for r in rows:
            b = float(r["best_val_mse"])
            if not math.isfinite(b):
                b = penalty
                self.diverged += 1
            self.best[r["config_id"]][int(r["seed"])] = b
        self.ids = sorted(self.best)
        self.mean = {c: float(np.mean(list(s.values()))) for c, s in self.best.items()}
        most = max(len(s) for s in self.best.values())
        self.most = most
        ref = {c: v for c, v in self.mean.items() if len(self.best[c]) == most}
        self.best_id = min(ref, key=ref.get)
        self.best_true = ref[self.best_id]
        self.fallback = float(np.mean([self.mean[c] for c in self.ids]))
        multi = {c: s for c, s in self.best.items() if len(s) >= 2}
        exclude = None
        if len(multi) < len(self.best) / 2:  # one-seed table (ETTh1): leave out the shared selection seed
            shared = set.intersection(*(set(s) for s in self.best.values()))
            exclude = min(shared) if shared else None
        var = []
        for c, s in multi.items():
            vals = [v for k, v in s.items() if k != exclude]
            if len(vals) >= 2:
                var.append(np.var(vals, ddof=1))
        self.sigma = float(np.sqrt(np.mean(var)))
        self.sd_excluded_seed = exclude
        self.n_multi = len(multi)

    def seen_seed(self, cid: str, run_seed: int) -> int:
        seeds = sorted(self.best[cid])
        h = hashlib.sha256(f"{run_seed}:{cid}".encode()).hexdigest()
        return seeds[int(h[:12], 16) % len(seeds)]

    def held(self, cid: str, run_seed: int) -> tuple[float, bool]:
        seen = self.seen_seed(cid, run_seed)
        others = [v for k, v in self.best[cid].items() if k != seen]
        return (float(np.mean(others)), True) if others else (self.mean[cid], False)


def incumbent_at(trace: list, budget: int) -> str | None:
    inc = None
    for spent, cid in trace:
        if spent <= budget:
            inc = cid
        else:
            break
    return inc


def auc(run: dict, tab: MyTable, mode: str = "heldout") -> float:
    """Mean regret over budgets 1..100 full runs (30..3000 epochs), in pooled seed-SD units."""
    seed = int(run["seed"])
    regrets = []
    for g in range(30, 3001, 30):
        cid = incumbent_at(run["trace"], g)
        if cid is None:
            v = tab.fallback
        elif mode == "heldout":
            v = tab.held(cid, seed)[0]
        else:
            v = tab.mean[cid]
        regrets.append(v - tab.best_true)
    return float(np.mean(regrets) / tab.sigma)


def heldout_share(runs: list[dict], tab: MyTable) -> float:
    flags = []
    for r in runs:
        for g in range(30, 3001, 30):
            cid = incumbent_at(r["trace"], g)
            flags.append(cid is not None and tab.held(cid, int(r["seed"]))[1])
    return float(np.mean(flags))


def tci(x) -> dict:
    x = np.asarray(list(x), dtype=float)
    n = len(x)
    m = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    half = float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": m, "sd": sd, "t_low": m - half, "t_high": m + half, "median": float(np.median(x))}


def spearman_perm(groups: list[list[float]], n_perm: int = N_PERM, seed: int = PERM_SEED) -> dict:
    """Spearman of ordinal level (group index) with values; two-sided permutation p over relabellings of values."""
    lev = np.concatenate([[i] * len(g) for i, g in enumerate(groups)]).astype(float)
    val = np.concatenate([np.asarray(g, float) for g in groups])
    rx, ry = rankdata(lev), rankdata(val)
    zx = (rx - rx.mean()) / rx.std()
    zy = (ry - ry.mean()) / ry.std()
    n = len(val)
    rho = float(zx @ zy / n)
    rng = np.random.default_rng(seed)
    hits, done, batch = 0, 0, 20_000
    while done < n_perm:
        b = min(batch, n_perm - done)
        idx = np.argsort(rng.random((b, n)), axis=1)
        r = (zy[idx] @ zx) / n
        hits += int(np.sum(np.abs(r) >= abs(rho) - 1e-12))
        done += b
    # scipy cross-check of rho
    from scipy.stats import spearmanr
    return {"rho": rho, "rho_scipy": float(spearmanr(lev, val).statistic), "n": n, "n_perm": n_perm, "perm_seed": seed,
            "hits_abs_ge_obs": hits, "p_plus1": (hits + 1) / (n_perm + 1), "p_raw": hits / n_perm,
            "mc_se": math.sqrt(max(hits, 1) / n_perm * (1 - hits / n_perm) / n_perm)}


def holm(ps: dict[str, float], alpha: float = 0.05) -> dict:
    names = sorted(ps, key=ps.get)
    m = len(names)
    adj, running, out = {}, 0.0, {}
    for i, nm in enumerate(names):
        running = max(running, min(1.0, (m - i) * ps[nm]))
        adj[nm] = running
    for nm in ps:
        out[nm] = {"p": ps[nm], "p_holm": adj[nm], "reject_0.05": adj[nm] <= alpha}
    return out


def revin_off(cid: str) -> bool:
    m = re.search(r"n([01])i$", cid)
    if not m:
        raise ValueError(cid)
    return m.group(1) == "0"


def revin_block(runs: list[dict]) -> dict:
    flags = [revin_off(c[0]) for r in runs for c in r["charges"]]
    per_run = [np.mean([revin_off(c[0]) for c in r["charges"]]) for r in runs if r["charges"]]
    dup_within = sum(len(r["charges"]) - len({c[0] for c in r["charges"]}) for r in runs)
    kinds = Counter(c[3] for r in runs for c in r["charges"])
    return {"pooled_share_off": float(np.mean(flags)), "n_charges": len(flags), "n_off": int(sum(flags)),
            "mean_of_per_run_shares": float(np.mean(per_run)), "repeat_cids_within_run_charges": dup_within,
            "charge_kinds": dict(kinds)}


def paired(a: dict[int, float], b: dict[int, float]) -> dict:
    both = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in both]
    return {"shared_seeds": both, "a": tci([a[s] for s in both]), "b": tci([b[s] for s in both]), "a_minus_b": tci(d),
            "per_seed_diff": {str(s): a[s] - b[s] for s in both}}


def run_consistency(run: dict, tab: MyTable) -> dict:
    """Trace spent equals cumulative charges; charged seed equals the oracle's draw; nothing above budget 3000."""
    seed = int(run["seed"])
    cum, spent_after = 0, []
    seed_mismatch = 0
    for cid, s, extra, kind, read in run["charges"]:
        cum += extra
        spent_after.append(cum)
        if int(s) != tab.seen_seed(cid, seed):
            seed_mismatch += 1
    trace_spents = [t[0] for t in run["trace"]]
    return {"spent_eq_sum_charges": cum == run["spent"], "spent_le_3000": run["spent"] <= 3000,
            "trace_monotone": all(x <= y for x, y in zip(trace_spents, trace_spents[1:])),
            "trace_spents_subset_cum": set(trace_spents) <= set(spent_after),
            "charge_seed_mismatch": seed_mismatch, "lookups_eq_len_charges": run.get("lookups") == len({c[0] for c in run["charges"]})}


def main() -> None:
    sys.path.insert(0, str(REPO))
    ledger = [json.loads(l) for l in (REPO / "results/ledger.jsonl").read_text().splitlines() if l.strip()]
    tabs = {}
    for ds in ("ILI", "ETTh1"):
        rows = [r for r in ledger if r["item"] == "E1" and r["dataset"] == ds and int(r["horizon"]) == HORIZON[ds]]
        tabs[ds] = MyTable(rows)
    out: dict = {"pin": PIN, "n_perm": N_PERM, "perm_seed": PERM_SEED,
                 "ledger_sha256": hashlib.sha256((REPO / "results/ledger.jsonl").read_bytes()).hexdigest(),
                 "tables": {ds: {"configs": len(t.ids), "seed_sd_pooled": t.sigma, "best_true": t.best_true, "best_id": t.best_id,
                                 "reference_seeds": t.most, "fallback": t.fallback, "diverged_rows": t.diverged,
                                 "multi_seed_configs": t.n_multi, "sd_excluded_seed": t.sd_excluded_seed}
                            for ds, t in tabs.items()}}

    # cross-check implementation: leaguetst
    from leaguetst.search.replay import Table, run_metric, no_incumbent_value
    lt = {}
    for ds in ("ILI", "ETTh1"):
        rows = [{k: r[k] for k in ("config_id", "seed", "val_curve", "train_curve", "best_val_mse", "epochs_run", "model_config")}
                for r in ledger if r["item"] == "E1" and r["dataset"] == ds and int(r["horizon"]) == HORIZON[ds]]
        T = Table(rows)
        lt[ds] = (T, no_incumbent_value(T), T.seed_sd())

    # ---------------------------------------------------------------- files
    manifest_raw = show("results/ablations/rtx/AB2_move.sha256")
    manifest = {p: d for d, p in (l.split() for l in manifest_raw.decode().splitlines() if l.strip())}
    files, rawbytes = {}, {}
    for item in list(PROMPTED) + list(TRANSFER):
        listing = git("ls-tree", "-r", "--name-only", PIN, f"results/ablations/rtx/{item}").decode().split()
        for p in listing:
            if p.endswith("_runs.jsonl"):
                b = show(p)
                rawbytes[p] = b
                files[p] = hashlib.sha256(b).hexdigest()
    out["check_manifest"] = {"n_manifest": len(manifest), "n_files_found": len(files),
                             "per_file": {p: {"manifest": manifest.get(p), "actual": files.get(p), "match": manifest.get(p) == files.get(p)}
                                          for p in sorted(set(manifest) | set(files))}}
    out["check_manifest"]["all_match"] = (len(manifest) == 9 and set(manifest) == set(files)
                                          and all(v["match"] for v in out["check_manifest"]["per_file"].values()))

    # ---------------------------------------------------------------- items
    items: dict = {}
    ili_auc: dict[str, dict[int, float]] = {}
    all_hashes: dict[str, set] = defaultdict(set)
    revisions_seen: dict[str, dict] = {}
    xcheck_maxdiff = 0.0
    for item in list(PROMPTED) + list(TRANSFER):
        entry: dict = {}
        is_p = item in PROMPTED
        settings = ("ILI_T24", "ETTh1_T336") if is_p else ("ILI_T24",)
        for setting in settings:
            ds = setting.split("_")[0]
            tab = tabs[ds]
            paths = [p for p in rawbytes if p.startswith(f"results/ablations/rtx/{item}/{setting}/")]
            runs = [r for p in sorted(paths) for r in jsonl(rawbytes[p])]
            methods = Counter(r["method"] for r in runs)
            keys = Counter((r["method"], int(r["seed"])) for r in runs)
            dups = sorted([list(k) for k, n in keys.items() if n > 1])
            seeds = sorted({int(r["seed"]) for r in runs})
            want_method = PROMPTED[item][0] if is_p else TMETHOD
            e: dict = {"files": paths, "rows": len(runs), "methods": dict(methods), "distinct_seeds": len(seeds), "seeds": seeds,
                       "duplicates": dups, "method_is_registered": set(methods) == {want_method}}
            if is_p:
                e["complete_AB2"] = seeds == list(range(30)) and not dups and set(methods) == {want_method}
            else:
                stops_paths = git("ls-tree", "-r", "--name-only", PIN, f"results/runs/{item}").decode().split()
                errs = [p for p in stops_paths if p.endswith("errors.jsonl")]
                e["errors_files"] = errs
                e["on_policy_stops"] = []
                e["complete_AB2"] = seeds == list(range(10)) and not dups and set(methods) == {want_method} and not errs
            # AUC
            mode = "heldout"
            per_seed = {int(r["seed"]): auc(r, tab, mode) for r in runs if r["method"] == want_method}
            e["auc_heldout_per_seed"] = {str(k): v for k, v in sorted(per_seed.items())}
            e["auc_heldout"] = tci(per_seed.values())
            e["heldout_share_over_budgets"] = heldout_share(runs, tab)
            if ds == "ETTh1":
                ins = {int(r["seed"]): auc(r, tab, "insample") for r in runs if r["method"] == want_method}
                e["auc_insample_multiseed_mean_per_seed"] = {str(k): v for k, v in sorted(ins.items())}
                e["auc_insample_multiseed_mean"] = tci(ins.values())
                e["note"] = ("auc_heldout = run_metric convention (held out where the incumbent has unseen seeds, else in-sample); "
                             "auc_insample_multiseed_mean = every incumbent scored on its multi-seed mean")
            # cross-check with leaguetst run_metric
            T, fb, sg = lt[ds]
            for r in runs:
                if r["method"] == want_method:
                    xcheck_maxdiff = max(xcheck_maxdiff, abs(run_metric(r, T, "auc", fb, sg) - per_seed[int(r["seed"])]))
            cons = [run_consistency(r, tab) for r in runs]
            e["consistency"] = {k: (all(c[k] for c in cons) if isinstance(cons[0][k], bool) else int(sum(c[k] for c in cons)))
                                for k in cons[0]}
            e["revin"] = revin_block(runs)
            envs = [r.get("env", {}) for r in runs]
            e["env"] = {"backend": sorted({str(v.get("backend")) for v in envs}), "per_gpu": sorted({str(v.get("per_gpu")) for v in envs}),
                        "gpu_name": sorted({str(v.get("gpu_name")) for v in envs}),
                        "model_revision": sorted({str(v.get("model_revision")) for v in envs}),
                        "git_hash": sorted({str(v.get("git_hash")) for v in envs})}
            want_rev = REV[PROMPTED[item][1]] if is_p else REV["1.5B"]
            e["revision_ok"] = e["env"]["model_revision"] == [want_rev]
            for h in e["env"]["git_hash"]:
                all_hashes[h].add(f"{item}/{setting} runs")
            e["lookups"] = tci([r["lookups"] for r in runs])
            e["proposals_total"] = int(sum(r.get("proposals", 0) for r in runs))
            tot = sum(r.get("proposals", 0) for r in runs) or 1
            e["rates"] = {k: sum(r.get(k, 0) for r in runs) / tot for k in ("invalid", "repaired", "repeats")}
            entry[setting] = e
            if ds == "ILI":
                ili_auc[item] = per_seed
        if not is_p:
            raw = show(f"results/runs/{item}/ILI_T24/training.jsonl")
            recs = jsonl(raw)
            jobs = [r["job"] for r in recs]
            want_beta = TRANSFER[item]
            tr = {"records": len(recs), "seeds": sorted(int(j["seed"]) for j in jobs),
                  "steps": sorted({j["steps"] for j in jobs}), "beta": sorted({j["beta"] for j in jobs}),
                  "lr": sorted({j["lr"] for j in jobs}), "rivals": sorted({j.get("rivals") for j in jobs}),
                  "dataset": sorted({j.get("dataset") for j in jobs}), "revision": sorted({j.get("revision") for j in jobs}),
                  "model": sorted({j.get("model") for j in jobs}), "backend": sorted({str(j.get("backend")) for j in jobs}),
                  "per_gpu": sorted({str(j.get("per_gpu")) for j in jobs}),
                  "run_names": sorted(j.get("run_name") for j in jobs),
                  "max_logged_step": sorted({max(row["step"] for row in r["log"]) for r in recs}),
                  "log_rows": sorted({len(r["log"]) for r in recs}),
                  "env_git_hash": sorted({str(r.get("env", {}).get("git_hash")) for r in recs}),
                  "job_git_hash": sorted({str(j.get("git_hash")) for j in jobs}),
                  "env_model_revision": sorted({str(r.get("env", {}).get("model_revision")) for r in recs}),
                  "minutes_per_policy": tci([r["seconds"] / 60 for r in recs]),
                  "onpolicy_mean_max": max(row["onpolicy_mean"] for r in recs for row in r["log"])}
            tr["ok"] = (tr["seeds"] == list(range(100, 110)) and tr["steps"] == [60]
                        and len(tr["beta"]) == 1 and abs(tr["beta"][0] - want_beta) < 1e-12
                        and len(tr["lr"]) == 1 and abs(tr["lr"][0] - 1e-4) < 1e-15 and tr["rivals"] == ["league"]
                        and tr["dataset"] == ["ETTh1"] and tr["revision"] == [REV["1.5B"]] and tr["max_logged_step"] == [60]
                        and tr["backend"] == ["local"] and tr["per_gpu"] == ["1"])
            tr["sha256"] = hashlib.sha256(raw).hexdigest()
            for h in tr["env_git_hash"] + tr["job_git_hash"]:
                all_hashes[h].add(f"{item} training")
            entry["training"] = tr
        items[item] = entry
    out["items"] = items
    out["crosscheck_leaguetst_run_metric_max_abs_diff"] = xcheck_maxdiff

    # ---------------------------------------------------------------- tests
    s05, s15, s3 = (list(dict(sorted(ili_auc[i].items())).values()) for i in ("ABL_P_05b_local", "ABL_P_ref_local", "ABL_P_3b_local"))
    scale = spearman_perm([s05, s15, s3])
    scale["levels"] = {"0.5B": tci(s05), "1.5B": tci(s15), "3B": tci(s3)}
    b0, b04, b1 = (list(dict(sorted(ili_auc[i].items())).values()) for i in ("ABL_G_beta0_local", "ABL_G_ref_local", "ABL_G_beta01_local"))
    kl = spearman_perm([b0, b04, b1], seed=PERM_SEED + 1)
    kl["levels"] = {"0": tci(b0), "0.04": tci(b04), "0.1": tci(b1)}
    out["scale_trend"] = scale
    out["kl_trend"] = kl
    out["holm"] = {"AB-P": holm({"scale trend": scale["p_plus1"], "anon vs ref_local (AB4, not run)": 1.0,
                                 "repair vs ref_local (AB4, not run)": 1.0}),
                   "AB-G": holm({"KL trend": kl["p_plus1"], "steps trend (AB4, not run)": 1.0})}

    # ---------------------------------------------------------------- descriptive comparisons
    tab = tabs["ILI"]

    def wt(path: str, method: str) -> dict[int, float]:
        rows = [json.loads(l) for l in (REPO / path).read_text().splitlines() if l.strip()]
        return {int(r["seed"]): auc(r, tab) for r in rows if r["method"] == method}

    j2 = wt("results/search/E4/ILI_T24/grpo_league_from_ETTh1_runs.jsonl", TMETHOD)
    mac = wt("results/ablations/mps/ABL_P_ref_mps/ILI_T24/qwen_prompted_mps_runs.jsonl", "qwen_prompted_mps")
    e3 = wt("results/search/E3/ILI_T24/qwen_prompted_runs.jsonl", "qwen_prompted")
    out["descriptive"] = {
        "G_ref_local_minus_E4_J2_league": paired(ili_auc["ABL_G_ref_local"], j2),
        "P_ref_local_minus_mps_ref": paired(ili_auc["ABL_P_ref_local"], mac),
        "P_ref_local_minus_E3_prompted": paired(ili_auc["ABL_P_ref_local"], e3),
        "revin_off_share_ILI": {i: items[i]["ILI_T24"]["revin"] for i in items},
        "revin_off_share_ETTh1": {i: items[i]["ETTh1_T336"]["revin"] for i in PROMPTED},
        "ETTh1_auc_per_prompted_item": {i: {"run_metric_convention": items[i]["ETTh1_T336"]["auc_heldout"],
                                            "multiseed_mean_insample": items[i]["ETTh1_T336"]["auc_insample_multiseed_mean"],
                                            "heldout_share": items[i]["ETTh1_T336"]["heldout_share_over_budgets"]} for i in PROMPTED},
        "input_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in (
            "results/search/E4/ILI_T24/grpo_league_from_ETTh1_runs.jsonl",
            "results/ablations/mps/ABL_P_ref_mps/ILI_T24/qwen_prompted_mps_runs.jsonl",
            "results/search/E3/ILI_T24/qwen_prompted_runs.jsonl")},
    }

    # ---------------------------------------------------------------- commit checks
    frozen_list = show("docs/e4_rerun.sha256", "a1751ae2a9dd4323bab484a42625fc835e76f989")
    frozen = {p: d for d, p in (l.split() for l in frozen_list.decode().splitlines() if l.strip())} if frozen_list else {}
    hc = {}
    for h, used in sorted(all_hashes.items()):
        dirty = h.endswith("-dirty")
        base = h.replace("-dirty", "")
        exists = git_ok("cat-file", "-e", f"{base}^{{commit}}")
        desc = exists and git_ok("merge-base", "--is-ancestor", AB2_REG, base)
        lb = show("scripts/ablate_llm.py", base) if exists else None
        lsha = hashlib.sha256(lb).hexdigest() if lb is not None else None
        fz = None
        if exists and frozen:
            fz = all((show(p, base) is not None and hashlib.sha256(show(p, base)).hexdigest() == d) for p, d in frozen.items())
        hc[h] = {"used_by": sorted(used), "dirty": dirty, "exists": exists, "descends_from_c2dda8d": desc,
                 "launcher_sha256": lsha, "launcher_ok": lsha == LAUNCHER, "e4_frozen_10_files_ok": fz,
                 "commit_time": git("log", "-1", "--format=%cI", base).decode().strip() if exists else None}
    out["check_git_hashes"] = hc
    out["check_git_hashes_all_ok"] = all(v["exists"] and not v["dirty"] and v["descends_from_c2dda8d"] and v["launcher_ok"] for v in hc.values())
    out["check_frozen_files_n"] = len(frozen)
    out["check_revisions_all_ok"] = all(items[i][s]["revision_ok"] for i in items for s in items[i] if s != "training")
    out["check_training_all_ok"] = all(items[i]["training"]["ok"] for i in TRANSFER)
    out["completeness_all"] = {f"{i}/{s}": items[i][s]["complete_AB2"] for i in items for s in items[i] if s != "training"}

    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    print("written", OUT)


if __name__ == "__main__":
    main()
