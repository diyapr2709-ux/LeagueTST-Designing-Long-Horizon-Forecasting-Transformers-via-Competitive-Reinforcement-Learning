import json, sys
from pathlib import Path
L = json.load(open(Path(__file__).resolve().parents[4] / "results/ablations/LLM-rtx.json"))["registered"]  # LLM.json["registered"] before the 943d893 move; identical
M = json.load(open(Path(__file__).resolve().parent / "recompute_ab4.json"))
R = json.load(open(Path(__file__).resolve().parent / "RTX_tests_e6926d8.json"))
rows = []
def add(name, l, m, r, tol=1e-4, mc=False):
    vals = [v for v in (l, m, r) if v is not None]
    if isinstance(vals[0], bool):
        ok = len(set(vals)) == 1
    else:
        spread = max(vals) - min(vals)
        ok = spread < tol or (mc and spread < 0.003)
    fmt = lambda v: "-" if v is None else (str(v) if isinstance(v, bool) else f"{v:.4f}")
    rows.append((("OK " if ok else "BAD") + (" (MC)" if mc and not (max(vals)-min(vals) < tol if not isinstance(vals[0], bool) else True) else ""), name, fmt(l), fmt(m), fmt(r)))
LP, LG = L["families"]["AB-P"]["tests"], L["families"]["AB-G"]["tests"]
RP, RG = R["AB-P"]["tests"], R["AB-G"]["tests"]
MT, MH = M["tests"], M["holm"]
for lk, rk in (("anon vs ref_local", "anon vs ref"), ("repair vs ref_local", "repair vs ref")):
    for f in ("diff", "low", "high"):
        add(f"{lk} {f}", LP[lk][f], MT[lk][f], RP[rk][f])
    add(f"{lk} mean arm", LP[lk]["mean"][0], MT[lk]["mean"][0], RP[rk]["mean"][0])
    add(f"{lk} mean ref", LP[lk]["mean"][1], MT[lk]["mean"][1], RP[rk]["mean"][1])
    add(f"{lk} detectable", LP[lk]["detectable_80"], None, RP[rk]["detectable_80"])
    add(f"{lk} p", LP[lk]["p"], MT[lk]["perm"]["p"], RP[rk]["p"], mc=True)
    add(f"{lk} p_holm", LP[lk]["p_holm"], MH["AB-P"][lk]["p_holm"], RP[rk]["p_holm"], mc=True)
    add(f"{lk} reject", LP[lk]["reject"], MH["AB-P"][lk]["reject_0.05"], RP[rk]["reject"])
for k, fam, mf in (("scale trend", (LP, RP), "AB-P"), ("steps trend", (LG, RG), "AB-G"), ("KL trend", (LG, RG), "AB-G")):
    l, r = fam[0][k], fam[1][k]
    add(f"{k} rho", l["rho"], MT[k]["rho"], r["rho"])
    add(f"{k} n", l["n"], MT[k]["n"], r["n"])
    add(f"{k} p", l["p"], MT[k]["p_plus1"], r["p"], mc=True)
    add(f"{k} p_holm", l["p_holm"], MH[mf][k]["p_holm"], r["p_holm"], mc=True)
    add(f"{k} reject", l["reject"], MH[mf][k]["reject_0.05"], r["reject"])
    ml = [v for _, v in sorted(MT[k]["levels"].items(), key=lambda kv: float(kv[0].rstrip("B")))]; ll = sorted(l["levels"].items(), key=lambda kv: float(kv[0])); rl = sorted(r["levels"].items(), key=lambda kv: float(kv[0]))
    for (lk_, lv), mv, (rk_, rv) in zip(ll, ml, rl):
        add(f"{k} level {lk_} mean", lv["mean"], mv["mean"], rv["mean"])
        add(f"{k} level {lk_} low", lv["low"], mv["t_low"], rv["low"])
        add(f"{k} level {lk_} high", lv["high"], mv["t_high"], rv["high"])
        add(f"{k} level {lk_} n", lv["n"], mv["n"], rv["n"])
pl, rpl, ms = L["families"]["AB-G"]["exploratory_policy_level"], R["AB-G"]["exploratory_addendum3"], M["secondary"]
u, ru, mu = pl["trained_vs_untrained"], rpl["trained_vs_untrained"], ms["policy_mean_vs_untrained"]
for f in ("diff", "low", "high"):
    add(f"(i) trained vs untrained {f}", u[f], mu[f], ru[f])
add("(i) mean trained", u["mean"][0], mu["mean"][0], ru["mean"][0])
add("(i) p", u["p"], mu["perm"]["p"], ru["p"], mc=True)
add("(ii) dose mean rho", pl["dose"]["mean_rho"], ms["dose_per_policy_spearman"]["mean"], rpl["dose"]["mean_rho"])
add("(ii) dose p", pl["dose"]["p_sign_flip"], ms["dose_per_policy_spearman"]["p"], rpl["dose"]["p_sign_flip"])
add("(ii) rhos equal", pl["dose"]["rhos"] == ms["dose_per_policy_spearman"]["rhos"], True, rpl["dose"]["rhos"] == pl["dose"]["rhos"])
# completeness
for kind, mk in (("prompted", "P"), ("transfer", "G")):
    for arm, s in L["status"][kind].items():
        m = M["all_registered_ok"][f"{mk}:{arm}"]
        add(f"complete {kind} {arm}", s["complete"] and not s["problems"], m, None)
add("AB-P not run", L["families"]["AB-P"]["missing"] == [], True, R["AB-P"]["not_run"] == [])
add("AB-G not run", L["families"]["AB-G"]["missing"] == [], True, R["AB-G"]["not_run"] == [])
# hardware check (AB2) and AB5 descriptives; recompute.json (blind agent, AB2 run) for the latter
B = json.load(open(Path(__file__).resolve().parent.parent / "recompute.json"))["descriptive"]
h = L["descriptive"]["hardware_G_ref_local_vs_J2"]; rh = R["hardware_check"]
for x, rx, bx in (("a", "rtx_local", "a"), ("b", "j2_modal", "b"), ("diff", "diff_rtx_minus_j2", "a_minus_b")):
    for f, bf in (("mean", "mean"), ("low", "t_low"), ("high", "t_high")):
        add(f"hardware G_ref vs J2 {x} {f}", h[x][f], B["G_ref_local_minus_E4_J2_league"][bx][bf], rh[rx][f])
for key, bkey in (("prompted_ref_local_vs_mac", "P_ref_local_minus_mps_ref"), ("prompted_ref_local_vs_E3_A10", "P_ref_local_minus_E3_prompted")):
    for f, bf in (("mean", "mean"), ("low", "t_low"), ("high", "t_high")):
        add(f"{key} diff {f}", L["descriptive"][key]["diff"][f], B[bkey]["a_minus_b"][bf], None)
# ETTh1 descriptive
for arm, mk in (("anon", "P:anon"), ("repair", "P:repair"), ("ref", "P:ref"), ("0.5B", "P:0.5B"), ("3B", "P:3B")):
    le = L["ETTh1_descriptive"][arm]["summary"]; lv = next(iter(le.values()))["auc"]
    rv = R["ETTh1"][arm.lower().replace(".", "")] if arm.lower().replace(".", "") in R["ETTh1"] else R["ETTh1"].get(arm.lower())
    add(f"ETTh1 {arm} AUC mean", lv["mean"], M["arms"][mk]["ETTh1_T336"]["auc"]["mean"], rv["mean"] if rv else None)
for r in rows:
    print(f"{r[0]:8s} {r[1]:42s} LLM {r[2]:>8s} | mine {r[3]:>8s} | rtx {r[4]:>8s}")
print("BAD rows:", sum(1 for r in rows if r[0].startswith("BAD")), "of", len(rows))
