"""Supplement: the scale trend with 10,000,000 permutations (another seed), added to recompute.json."""
import json, math, sys
import numpy as np
from scipy.stats import rankdata
path = sys.argv[1]
d = json.load(open(path))
it = d["items"]
groups = [list(dict(sorted(((int(k), v) for k, v in it[i]["ILI_T24"]["auc_heldout_per_seed"].items()))).values())
          for i in ("ABL_P_05b_local", "ABL_P_ref_local", "ABL_P_3b_local")]
lev = np.concatenate([[i] * len(g) for i, g in enumerate(groups)]).astype(float)
val = np.concatenate([np.asarray(g) for g in groups])
zx = rankdata(lev); zx = (zx - zx.mean()) / zx.std()
zy = rankdata(val); zy = (zy - zy.mean()) / zy.std()
n = len(val); rho = float(zx @ zy / n)
rng = np.random.default_rng(7); N = 10_000_000; hits = 0; done = 0
while done < N:
    b = min(50_000, N - done)
    idx = np.argsort(rng.random((b, n)), axis=1)
    hits += int(np.sum(np.abs((zy[idx] @ zx) / n) >= abs(rho) - 1e-12)); done += b
t = rho * math.sqrt(n - 2) / math.sqrt(1 - rho**2)
from scipy.stats import t as st
d["scale_trend_supplement_1e7"] = {"rho": rho, "n_perm": N, "perm_seed": 7, "hits": hits, "p_plus1": (hits + 1) / (N + 1),
                                   "p_holm_x3": min(1.0, 3 * (hits + 1) / (N + 1)), "t_approx_two_sided_p": float(2 * st.sf(abs(t), n - 2))}
json.dump(d, open(path, "w"), indent=1, sort_keys=True); open(path, "a").write("\n")
print(d["scale_trend_supplement_1e7"])
