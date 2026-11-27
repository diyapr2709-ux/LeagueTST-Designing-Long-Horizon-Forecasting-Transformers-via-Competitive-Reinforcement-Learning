"""Independent recomputation of every registered RTX arm (AB2 and AB4) at a pinned commit of origin/soumitra-leg2.

Table, held-out AUC, t intervals, Spearman permutation and Holm come from the blind recomputation agent's
recompute.py (results/ablations/verify_rtx/recompute.py, written blind and already matched against llm_tests on AB2), imported
unchanged. Everything AB4-specific (two-sample permutation, Welch, the step analyses, the arm checks) is
written here. leaguetst's run_metric is used only as a cross-check of the AUC.

Usage: python recompute_ab4.py <repo> <out.json>
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t as student_t

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PIN = "e6926d8664b8493442f4e372a4cf0349a34d06e0"
REG = {"AB2": "c2dda8dfa49c6f090816f516285792a00585a79d", "AB4": "15db16314f1303854d4540894d7aaed6fa1a3270"}
FROZEN_FROM = "a1751ae2a9dd4323bab484a42625fc835e76f989"  # docs/e4_rerun.sha256 as committed on main before E4
LAUNCHER = "3919d68337302590eda42f53ca10158ca27766bb488026b5a4fc8373c477b786"
REV = {"0.5B": "7ae557604adf67be50417f59c2c2f167def9a775", "1.5B": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
       "3B": "aa8e72537993ba99e69dfaafa59ed015b17504d1"}
DEADLINE = datetime.fromisoformat("2026-11-28T23:59:00-05:00")
RTX = "results/ablations/rtx"
TMETHOD = "grpo_league_from_ETTh1"
N_PERM = 1_000_000
SEED0 = 20261126

# arm -> (item, method, revision key, amendment, queue name of the ILI job)
PROMPTED = {"ref": ("ABL_P_ref_local", "qwen_prompted_1.5b_local", "1.5B", "AB2", "abl_p_ref_local_ILI"),
            "0.5B": ("ABL_P_05b_local", "qwen_prompted_0.5b_local", "0.5B", "AB2", "abl_p_05b_local_ILI"),
            "3B": ("ABL_P_3b_local", "qwen_prompted_3b_local", "3B", "AB2", "abl_p_3b_local_ILI"),
            "anon": ("ABL_P_anon_local", "qwen_prompted_anon_1.5b_local", "1.5B", "AB4", "abl_p_anon_local_ILI"),
            "repair": ("ABL_P_repair_local", "qwen_prompted_repair_1.5b_local", "1.5B", "AB4", "abl_p_repair_local_ILI")}
# arm -> (item, steps, beta, amendment, training.jsonl path)
TRANSFER = {"ref": ("ABL_G_ref_local", 60, 0.04, "AB2", "results/runs/ABL_G_ref_local/ILI_T24/training.jsonl"),
            "beta0": ("ABL_G_beta0_local", 60, 0.0, "AB2", "results/runs/ABL_G_beta0_local/ILI_T24/training.jsonl"),
            "beta01": ("ABL_G_beta01_local", 60, 0.1, "AB2", "results/runs/ABL_G_beta01_local/ILI_T24/training.jsonl"),
            "steps30": ("ABL_G_steps30_local", 30, 0.04, "AB4", f"{RTX}/ABL_G_steps30_local/ILI_T24/training.jsonl"),
            "steps120": ("ABL_G_steps120_local", 120, 0.04, "AB4", f"{RTX}/ABL_G_steps120_local/ILI_T24/training.jsonl")}


def load_blind_module():
    path = REPO.parent / "dummy"  # recompute.py reads sys.argv at import; give it harmless values
    saved = sys.argv
    sys.argv = [saved[0], str(REPO), str(path)]
    spec = importlib.util.spec_from_file_location(
        "blind_recompute", str(Path(__file__).resolve().parent.parent / "recompute.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = saved
    return mod


B = load_blind_module()


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def git_ok(*args: str) -> bool:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True).returncode == 0


def show(path: str, ref: str = PIN) -> bytes | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def ls(path: str) -> list[str]:
    r = subprocess.run(["git", "ls-tree", "-r", "--name-only", PIN, path], cwd=REPO, capture_output=True)
    return r.stdout.decode().split() if r.returncode == 0 else []


def jsonl(data: bytes | None) -> list[dict]:
    return [json.loads(l) for l in data.decode().splitlines() if l.strip()] if data else []


# ------------------------------------------------------------------ statistics written here
def welch(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = float(a.mean() - b.mean())
    h = float(student_t.ppf(0.975, df) * se)
    return {"diff": d, "low": d - h, "high": d + h, "df": df, "n": [len(a), len(b)], "mean": [float(a.mean()), float(b.mean())]}


def perm_two_sample(a, b, n_perm: int = N_PERM, seed: int = SEED0) -> dict:
    """Two-sided Monte Carlo permutation test of the difference in means (observed counted)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n, k = len(pooled), len(a)
    obs = a.mean() - b.mean()
    tol = 1e-12 * max(1.0, abs(obs))
    rng = np.random.default_rng(seed)
    hits = done = 0
    total = pooled.sum()
    while done < n_perm:
        m = min(20_000, n_perm - done)
        idx = np.argsort(rng.random((m, n)), axis=1)[:, :k]
        sa = pooled[idx].sum(axis=1)
        diff = sa / k - (total - sa) / (n - k)
        hits += int(np.sum(np.abs(diff) >= abs(obs) - tol))
        done += m
    return {"diff": float(obs), "hits": hits, "n_perm": n_perm, "seed": seed, "p": (hits + 1) / (n_perm + 1)}


def compare(a, b, seed: int) -> dict:
    return {**welch(a, b), "perm": perm_two_sample(a, b, seed=seed)}


def sign_flip_exact(values) -> dict:
    v = np.asarray(values, float)
    obs = abs(v.mean())
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(v))))
    stats = np.abs(signs @ v / len(v))
    return {"mean": float(v.mean()), "patterns": len(signs), "p": float(np.mean(stats >= obs - 1e-12))}


# ------------------------------------------------------------------ checks
def frozen_map() -> dict[str, str]:
    raw = show("docs/e4_rerun.sha256", FROZEN_FROM)
    return {p: d for d, p in (l.split() for l in raw.decode().splitlines() if l.strip())}


FROZEN = frozen_map()
_hash_cache: dict[str, dict] = {}


def hash_check(h: str, amendment: str) -> dict:
    key = f"{h}|{amendment}"
    if key in _hash_cache:
        return _hash_cache[key]
    base = h.replace("-dirty", "")
    exists = git_ok("cat-file", "-e", f"{base}^{{commit}}")
    out = {"dirty": h.endswith("-dirty"), "exists": exists,
           "descends": exists and git_ok("merge-base", "--is-ancestor", REG[amendment], base)}
    lb = show("scripts/ablate_llm.py", base) if exists else None
    out["launcher_ok"] = lb is not None and hashlib.sha256(lb).hexdigest() == LAUNCHER
    out["frozen_ok"] = exists and all((show(p, base) is not None and hashlib.sha256(show(p, base)).hexdigest() == d)
                                      for p, d in FROZEN.items()) and len(FROZEN) == 10
    out["ok"] = (not out["dirty"]) and all(out[k] for k in ("exists", "descends", "launcher_ok", "frozen_ok"))
    _hash_cache[key] = out
    return out


def done_times() -> dict[str, str]:
    out = {}
    for rel in ("results/logs/l4_queue_status.txt", "results/logs/ab4_queue_status.txt"):
        for line in (show(rel) or b"").decode().splitlines():
            p = line.split()
            if len(p) >= 4 and p[1] == "DONE" and p[3] == "rc=0":
                out[p[2]] = p[0]
    return out


def main() -> None:
    ledger = [json.loads(l) for l in (REPO / "results/ledger.jsonl").read_text().splitlines() if l.strip()]
    tabs = {}
    for ds, hz in (("ILI", 24), ("ETTh1", 336)):
        tabs[ds] = B.MyTable([r for r in ledger if r["item"] == "E1" and r["dataset"] == ds and int(r["horizon"]) == hz])
    sys.path.insert(0, str(REPO))
    from leaguetst.search.replay import Table, no_incumbent_value, run_metric
    lt = {}
    for ds, hz in (("ILI", 24), ("ETTh1", 336)):
        T = Table([{k: r[k] for k in ("config_id", "seed", "val_curve", "train_curve", "best_val_mse", "epochs_run", "model_config")}
                   for r in ledger if r["item"] == "E1" and r["dataset"] == ds and int(r["horizon"]) == hz])
        lt[ds] = (T, no_incumbent_value(T), T.seed_sd())

    out: dict = {"pin": PIN, "tip_local": git("rev-parse", "origin/soumitra-leg2").decode().strip(), "n_perm": N_PERM,
                 "ledger_sha256": hashlib.sha256((REPO / "results/ledger.jsonl").read_bytes()).hexdigest(),
                 "auc_code": "results/ablations/verify_rtx/recompute.py (blind agent), sha256 " + hashlib.sha256(
                     (Path(__file__).resolve().parent.parent / "recompute.py").read_bytes()).hexdigest()}
    # AB2 manifest at the new pin
    man = {p: d for d, p in (l.split() for l in show(f"{RTX}/AB2_move.sha256").decode().splitlines() if l.strip())}
    out["ab2_manifest"] = {p: hashlib.sha256(show(p) or b"").hexdigest() == d for p, d in man.items()}
    done = done_times()
    out["queue_done_times"] = done
    xmax = 0.0
    files: dict[str, str] = {}
    arms: dict = {}
    ili: dict[str, dict[int, float]] = {}

    def arm_runs(item: str, setting: str) -> list[dict]:
        rows = []
        for p in ls(f"{RTX}/{item}/{setting}"):
            if p.endswith("_runs.jsonl"):
                data = show(p)
                files[p] = hashlib.sha256(data).hexdigest()
                copy = show(p.replace(RTX, "results/runs/llm_local"))
                if copy is not None and copy != data:
                    raise SystemExit(f"copy under results/runs/llm_local differs: {p}")
                rows += jsonl(data)
        return rows

    def score(runs, method, ds):
        nonlocal xmax
        per = {int(r["seed"]): B.auc(r, tabs[ds]) for r in runs if r["method"] == method}
        T, fb, sg = lt[ds]
        for r in runs:
            if r["method"] == method:
                xmax = max(xmax, abs(run_metric(r, T, "auc", fb, sg) - per[int(r["seed"])]))
        return per

    def env_check(runs, rev_key, amendment):
        envs = [r.get("env", {}) for r in runs]
        hashes = sorted({str(e.get("git_hash")) for e in envs})
        return {"backend": sorted({str(e.get("backend")) for e in envs}), "per_gpu": sorted({str(e.get("per_gpu")) for e in envs}),
                "gpu": sorted({str(e.get("gpu_name")) for e in envs}),
                "revision_ok": {e.get("model_revision") for e in envs} == {REV[rev_key]},
                "git_hashes": {h: hash_check(h, amendment) for h in hashes}}

    for arm, (item, method, rev, amd, qname) in PROMPTED.items():
        entry = {}
        for setting, ds in (("ILI_T24", "ILI"), ("ETTh1_T336", "ETTh1")):
            runs = arm_runs(item, setting)
            keys = Counter((r["method"], int(r["seed"])) for r in runs)
            e = {"rows": len(runs), "methods": sorted({r["method"] for r in runs}),
                 "seeds_complete": {int(r["seed"]) for r in runs if r["method"] == method} == set(range(30)),
                 "duplicates": sorted(str(k) for k, n in keys.items() if n > 1), **env_check(runs, rev, amd)}
            per = score(runs, method, ds)
            e["auc"] = B.tci(per.values())
            tot = sum(r.get("proposals", 0) for r in runs) or 1
            e["rates"] = {k: sum(r.get(k, 0) for r in runs) / tot for k in ("invalid", "repaired", "repeats")}
            q = qname if ds == "ILI" else qname.replace("_ILI", "_ETTh1")
            e["done"] = done.get(q)
            e["done_before_deadline"] = e["done"] is not None and datetime.fromisoformat(e["done"]) <= DEADLINE
            e["registered_ok"] = (e["seeds_complete"] and not e["duplicates"] and e["methods"] == [method] and e["revision_ok"]
                                  and e["backend"] == ["local"] and e["per_gpu"] == ["1"] and all(v["ok"] for v in e["git_hashes"].values())
                                  and e["done_before_deadline"])
            entry[setting] = e
            if ds == "ILI":
                ili[f"P:{arm}"] = per
        arms[f"P:{arm}"] = entry

    for arm, (item, steps, beta, amd, tpath) in TRANSFER.items():
        runs = arm_runs(item, "ILI_T24")
        keys = Counter((r["method"], int(r["seed"])) for r in runs)
        recs = jsonl(show(tpath))
        if tpath.startswith(RTX):
            copy = show(tpath.replace(RTX, "results/runs/llm_local"))
            if copy is not None and copy != show(tpath):
                raise SystemExit(f"training copy differs: {tpath}")
        errs = [p for p in ls("results") if p.endswith("errors.jsonl") and item in p]
        jobs = [r["job"] for r in recs]
        e = {"rows": len(runs), "methods": sorted({r["method"] for r in runs}),
             "deployed_seeds": sorted({int(r["seed"]) for r in runs}), "duplicates": sorted(str(k) for k, n in keys.items() if n > 1),
             "errors_files": errs, "on_policy_stops": [],
             "training": {"records": len(recs), "seeds": sorted(int(j["seed"]) for j in jobs), "steps": sorted({j["steps"] for j in jobs}),
                          "beta": sorted({j["beta"] for j in jobs}), "lr": sorted({j["lr"] for j in jobs}),
                          "rivals": sorted({j["rivals"] for j in jobs}), "dataset": sorted({j["dataset"] for j in jobs}),
                          "revision": sorted({j["revision"] for j in jobs}), "backend": sorted({str(j["backend"]) for j in jobs}),
                          "per_gpu": sorted({str(j["per_gpu"]) for j in jobs}),
                          "max_logged_step": sorted({max(row["step"] for row in r["log"]) for r in recs}),
                          "onpolicy_mean_max": max(row["onpolicy_mean"] for r in recs for row in r["log"]),
                          "minutes_per_policy": B.tci([r["seconds"] / 60 for r in recs]),
                          "git_hashes": {h: hash_check(h, amd) for h in sorted({str(r["env"]["git_hash"]) for r in recs} | {str(j["git_hash"]) for j in jobs})}},
             **env_check(runs, "1.5B", amd)}
        tr = e["training"]
        tr["ok"] = (tr["seeds"] == list(range(100, 110)) and tr["steps"] == [steps] and len(tr["beta"]) == 1
                    and abs(tr["beta"][0] - beta) < 1e-12 and tr["lr"] == [1e-4] and tr["rivals"] == ["league"]
                    and tr["dataset"] == ["ETTh1"] and tr["revision"] == [REV["1.5B"]] and tr["backend"] == ["local"]
                    and tr["per_gpu"] == ["1"] and tr["max_logged_step"][-1] <= steps and all(v["ok"] for v in tr["git_hashes"].values()))
        per = score(runs, TMETHOD, "ILI")
        e["auc"] = B.tci(per.values())
        tot = sum(r.get("proposals", 0) for r in runs) or 1
        e["rates"] = {k: sum(r.get(k, 0) for r in runs) / tot for k in ("invalid", "repaired", "repeats")}
        e["done"] = done.get(item.lower())
        e["done_before_deadline"] = e["done"] is not None and datetime.fromisoformat(e["done"]) <= DEADLINE
        e["registered_ok"] = (e["deployed_seeds"] == list(range(10)) and not e["duplicates"] and e["methods"] == [TMETHOD]
                              and not errs and e["revision_ok"] and e["backend"] == ["local"] and e["per_gpu"] == ["1"]
                              and all(v["ok"] for v in e["git_hashes"].values()) and tr["ok"] and e["done_before_deadline"])
        arms[f"G:{arm}"] = e
        ili[f"G:{arm}"] = per

    out["arms"] = arms
    out["files_sha256"] = files
    out["crosscheck_run_metric_max_abs_diff"] = xmax

    def vals(key):
        return [v for _, v in sorted(ili[key].items())]

    ref = vals("P:ref")
    t = {}
    t["anon vs ref_local"] = compare(vals("P:anon"), ref, SEED0 + 10)
    t["repair vs ref_local"] = compare(vals("P:repair"), ref, SEED0 + 11)
    t["scale trend"] = B.spearman_perm([vals("P:0.5B"), ref, vals("P:3B")], seed=SEED0)
    t["steps trend"] = B.spearman_perm([ref, vals("G:steps30"), vals("G:ref"), vals("G:steps120")], seed=SEED0 + 2)
    t["KL trend"] = B.spearman_perm([vals("G:beta0"), vals("G:ref"), vals("G:beta01")], seed=SEED0 + 1)
    for name, levels in (("scale trend", {"0.5B": vals("P:0.5B"), "1.5B": ref, "3B": vals("P:3B")}),
                         ("steps trend", {"0": ref, "30": vals("G:steps30"), "60": vals("G:ref"), "120": vals("G:steps120")}),
                         ("KL trend", {"0": vals("G:beta0"), "0.04": vals("G:ref"), "0.1": vals("G:beta01")})):
        t[name]["levels"] = {k: B.tci(v) for k, v in levels.items()}
    p = {k: (v["perm"]["p"] if "perm" in v else v["p_plus1"]) for k, v in t.items()}
    out["tests"] = t
    out["holm"] = {"AB-P": B.holm({k: p[k] for k in ("anon vs ref_local", "repair vs ref_local", "scale trend")}),
                   "AB-G": B.holm({k: p[k] for k in ("steps trend", "KL trend")})}
    # secondary step analyses (AB4: AB3 addendum 3, exploratory); policy s = training seed 100+s deployed at seed s
    seeds = sorted(ili["G:steps30"])
    assert seeds == sorted(ili["G:ref"]) == sorted(ili["G:steps120"]) == list(range(10))
    M = np.array([[ili[f"G:{a}"][s] for s in seeds] for a in ("steps30", "ref", "steps120")])  # 3 x 10
    pm = M.mean(axis=0)
    rhos = []
    for col in M.T:
        rhos.append(float(np.corrcoef(rankdata([30, 60, 120]), rankdata(col))[0, 1]) if len(set(col)) > 1 else 0.0)
    out["secondary"] = {"policy_mean_vs_untrained": {**compare(pm, ref, SEED0 + 20), "policy_means": pm.tolist()},
                        "dose_per_policy_spearman": {"rhos": rhos, **sign_flip_exact(rhos)},
                        "paired_gap_same_seed_descriptive": B.tci(pm - np.array(ref[:10]))}
    out["all_registered_ok"] = {k: (v["ILI_T24"]["registered_ok"] if "ILI_T24" in v else v["registered_ok"]) for k, v in arms.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    print("written", OUT)


if __name__ == "__main__":
    main()
