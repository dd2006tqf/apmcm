"""E15: CVaR evaluation of the 16 July relief plan under demand and network uncertainty.

The plan (six depots) is frozen; only the shipments are re-optimised in each scenario,
which is the recourse a real operation would run. Two sources of uncertainty are used:

  * demand: lognormal multiplicative noise, sigma = 0.2, truncated at +-3 sigma
  * network: the empirical difference between the 4 July and 16 July cut networks.
    A pair that is admissible on both dates is held fixed. A pair that is admissible on
    16 July but not on 4 July -- a road that reopened as the water receded -- is cut
    again with probability 0.5. The depot sets of the two dates share half their members
    (Jaccard 0.50) and their admissible-pair sets only a third (0.343), so 0.5 sits in
    the middle of that measured range. Reopening is never sampled: the evaluation prices
    further deterioration only, never an unobserved improvement.

Output: experiments/results/e15_q5_cvar.json
"""
import os
import sys
import json
import time
import copy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
os.chdir(os.path.join(ROOT, "code"))

import numpy as np
import rasterio
import config as C
from network import RoadNet
from problem4 import facility_proxies
import problem5 as P5

N_SCEN = int(os.environ.get("E15_N", "120"))
SIGMA = 0.2
P_BREAK = 0.5
SEED = 2026


def summarise(v):
    v = np.asarray(v, dtype=float)
    k = max(1, int(0.05 * len(v)))
    worst = np.sort(v)[-k:]
    return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.quantile(v, 0.95)),
                worst=float(v.max()), cvar95=float(worst.mean()),
                p05=float(np.quantile(v, 0.05)))


def main():
    t0 = time.time()
    with rasterio.open(C.OUT / "q3_newflood_0716.tif") as d:
        crs = d.crs
    net = RoadNet(crs)
    fac = facility_proxies(crs)
    p0 = P5.base_params()
    tmax = p0["t_max"]

    inp16 = P5.build_inputs(C.MAIN_DATE, p0, net, fac)
    sol0 = P5.solve(inp16, stage="lex")
    eta0 = sol0["eta"]
    print(f"[e15] base f1={sol0['f1']:.4f} f2={sol0['f2']:.4f} served={sol0['served_pop']:.1f} "
          f"open={sol0['open_depots']}")

    inp04 = P5.build_inputs("20260704", p0, net, fac)
    tt16 = inp16["tt"]
    name16 = inp16["cand"]["name"].values
    name04 = inp04["cand"]["name"].values
    m04i = np.array([{n: i for i, n in enumerate(name04)}[n] for n in name16])
    uid16 = inp16["dem"]["uid"].values
    uid04 = inp04["dem"]["uid"].values
    m04j = np.array([{int(u): j for j, u in enumerate(uid04)}.get(int(u), -1) for u in uid16])
    tt04 = np.full_like(tt16, np.inf)
    jj = m04j >= 0
    tt04[:, jj] = inp04["tt"][np.ix_(m04i, m04j[jj])]
    A16, A04 = tt16 <= tmax, tt04 <= tmax
    only16, only04 = A16 & ~A04, ~A16 & A04
    print(f"[e15] pairs 16Jul={int(A16.sum())} 04Jul={int(A04.sum())} "
          f"intersection={int((A16 & A04).sum())} only16={int(only16.sum())} only04={int(only04.sum())}")

    rng = np.random.default_rng(SEED)
    d0 = inp16["dem"]["d"].values.astype(float)
    nJ = len(d0)

    rows = []
    for k in range(N_SCEN):
        kind = ("demand", "network", "both")[k % 3]
        dem = inp16["dem"].copy()
        tt = tt16.copy()
        if kind in ("demand", "both"):
            z = np.clip(rng.standard_normal(nJ), -3, 3)
            dem["d"] = d0 * np.exp(SIGMA * z)
        if kind in ("network", "both"):
            r = rng.random(tt.shape)
            cut = only16 & (r < P_BREAK)
            tt[cut] = np.inf
        inp_s = dict(inp16, dem=dem, tt=tt)
        s = P5.solve(inp_s, stage="lex", fix_eta=eta0)
        rows.append(dict(k=k, kind=kind, f1=s["f1"], f2=s["f2"], served_pop=s["served_pop"],
                         n_pairs=s["n_pairs"], optimal=s["optimal"]))
        if (k + 1) % 10 == 0:
            print(f"[e15] {k+1}/{N_SCEN} f1={s['f1']:.4f} elapsed={time.time()-t0:.0f}s", flush=True)

    res = dict(meta=dict(n_scen=N_SCEN, sigma=SIGMA, p_break=P_BREAK, seed=SEED,
                         pairs_16=int(A16.sum()), pairs_04=int(A04.sum()),
                         pairs_inter=int((A16 & A04).sum()),
                         jaccard=float((A16 & A04).sum() / (A16 | A04).sum()),
                         solver=P5.SOLVER, seconds=time.time() - t0),
               base=dict(f1=sol0["f1"], f2=sol0["f2"], served_pop=sol0["served_pop"],
                         open_depots=sol0["open_depots"]))
    for kind in ("demand", "network", "both"):
        sub = [r for r in rows if r["kind"] == kind]
        res[kind] = dict(n=len(sub),
                         f1=summarise([r["f1"] for r in sub]),
                         f2=summarise([r["f2"] for r in sub]),
                         served_pop=summarise([r["served_pop"] for r in sub]))
    allr = rows
    res["all"] = dict(n=len(allr), f1=summarise([r["f1"] for r in allr]),
                      f2=summarise([r["f2"] for r in allr]),
                      served_pop=summarise([r["served_pop"] for r in allr]))
    # worst interpretable case: every pair that reopened after 4 July is cut again
    tt_s = tt16.copy()
    tt_s[only16] = np.inf
    ss = P5.solve(dict(inp16, tt=tt_s), stage="lex", fix_eta=eta0)
    res["stress"] = dict(f1=ss["f1"], f2=ss["f2"], served_pop=ss["served_pop"], n_pairs=ss["n_pairs"])

    # tail: how often does the frozen plan lose more than one extra point of coverage?
    f1s = np.array([r["f1"] for r in allr])
    res["tail"] = dict(share_f1_above_060=float((f1s > 0.60).mean()),
                       share_f1_above_062=float((f1s > 0.62).mean()),
                       worst_scenario_f1=float(f1s.max()),
                       worst_scenario_kind=[r["kind"] for r in allr][int(np.argmax(f1s))])

    os.makedirs(os.path.join(ROOT, "experiments", "results"), exist_ok=True)
    outp = os.path.join(ROOT, "experiments", "results", "e15_q5_cvar.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(dict(res, scenarios=rows), f, ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ("meta", "base", "demand", "network", "both", "all", "tail")},
                     ensure_ascii=False, indent=1))
    print(f"[e15] written {outp}")


if __name__ == "__main__":
    main()
