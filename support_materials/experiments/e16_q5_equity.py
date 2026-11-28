"""E16: minimum service floor on the 16 July relief plan.

For every demand unit that has at least one admissible pair, add
    sum_i f_ij >= q * d_j
and scan q. Two depot budgets are used: the base K=6 and a stressed K=3.
Output: experiments/results/e16_q5_equity.json
"""
import os
import sys
import json
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
os.chdir(os.path.join(ROOT, "code"))

import numpy as np
import rasterio
import pulp
import config as C
from network import RoadNet
from problem4 import facility_proxies
import problem5 as P5

TL = 300


def solve_floor(inp, q, K=None, tl=TL):
    p = dict(inp["p"])
    if K is not None:
        p["K"] = K
    dem, cand, tt, cap = inp["dem"], inp["cand"], inp["tt"], inp["cap"]
    nI, nJ = tt.shape
    d, w = dem["d"].values.astype(float), dem["w"].values.astype(float)
    P = dem["P"].values.astype(float)
    Wd = float((w * d).sum())
    A = [(i, j) for i in range(nI) for j in range(nJ) if tt[i, j] <= p["t_max"]]

    def build():
        m = pulp.LpProblem("eq", pulp.LpMinimize)
        eta = {i: pulp.LpVariable(f"e{i}", cat=pulp.LpBinary) for i in range(nI)}
        f = {(i, j): pulp.LpVariable(f"f{i}_{j}", lowBound=0) for (i, j) in A}
        u = {j: pulp.LpVariable(f"u{j}", lowBound=0) for j in range(nJ)}
        f1 = pulp.lpSum(w[j] * u[j] for j in range(nJ)) * (1.0 / Wd)
        f2 = pulp.lpSum(w[j] * tt[i, j] * f[i, j] for (i, j) in A) * (1.0 / Wd)
        byj, byi = {}, {}
        for (i, j) in A:
            byj.setdefault(j, []).append(f[i, j])
            byi.setdefault(i, []).append(f[i, j])
        for j in range(nJ):
            m += pulp.lpSum(byj.get(j, [])) + u[j] == d[j], f"b{j}"
        for i in range(nI):
            m += pulp.lpSum(byi.get(i, [])) <= cap[i] * eta[i], f"c{i}"
            if not cand["admissible"].iloc[i]:
                m += eta[i] == 0, f"x{i}"
        for (i, j) in A:
            m += f[i, j] <= d[j] * eta[i], f"t{i}_{j}"
        m += pulp.lpSum(eta.values()) <= p["K"], "K"
        m += pulp.lpSum((2 * tt[i, j] + p["t_load"]) / p["q_veh"] * f[i, j]
                        for (i, j) in A) <= 60 * p["n_veh"] * p["shift"] * p["H"], "fleet"
        for j in range(nJ):
            if j in byj:
                m += pulp.lpSum(byj[j]) >= q * d[j], f"q{j}"
        return m, eta, f, u, f1, f2

    m, eta, f, u, f1, f2 = build()
    m.setObjective(f1)
    m.solve(pulp.HiGHS(msg=False, timeLimit=tl, gapRel=0, threads=1))
    s1 = pulp.LpStatus[m.status]
    if s1 != "Optimal":
        return dict(q=q, K=p["K"], status=s1, feasible=False)
    f1s = float(pulp.value(f1))

    m, eta, f, u, f1, f2 = build()
    m += f1 <= f1s + 1e-6, "lex"
    m.setObjective(f2)
    m.solve(pulp.HiGHS(msg=False, timeLimit=tl, gapRel=0, threads=1))
    s2 = pulp.LpStatus[m.status]
    if s2 != "Optimal":
        return dict(q=q, K=p["K"], status=s2, feasible=False)

    served = np.zeros(nJ)
    reach = sorted({j for (i, j) in A})
    for (i, j) in A:
        v = f[i, j].value() or 0
        if v > 1e-7:
            served[j] += v
    reach_unmet = float(sum(w[j] * (u[j].value() or 0) for j in reach)) / Wd
    return dict(q=q, K=p["K"], status="Optimal", feasible=True,
                f1=f1s, f2=float(pulp.value(f2)),
                reach_unmet=reach_unmet, n_reach=len(reach),
                served_pop=float((np.minimum(served / np.maximum(d, 1e-12), 1) * P).sum()),
                n_open=int(sum(round(eta[i].value() or 0) for i in range(nI))),
                open=sorted(cand["name"].iloc[i] for i in range(nI) if round(eta[i].value() or 0)))


def main():
    t0 = time.time()
    with rasterio.open(C.OUT / "q3_newflood_0716.tif") as dd:
        crs = dd.crs
    net = RoadNet(crs)
    fac = facility_proxies(crs)
    p0 = P5.base_params()
    inp = P5.build_inputs(C.MAIN_DATE, p0, net, fac)

    rows = []
    for K in (p0["K"], 3):
        for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            r = solve_floor(inp, q, K=K)
            print(f"[e16] K={K} q={q:.1f} status={r['status']} "
                  + (f"f1={r['f1']:.4f} f2={r['f2']:.2f} reach_unmet={r['reach_unmet']:.4f} "
                     f"served={r['served_pop']:.1f} n_open={r['n_open']}" if r.get("feasible") else ""),
                  flush=True)
            rows.append(r)

    out = dict(rows=rows, seconds=time.time() - t0)
    with open(os.path.join(ROOT, "experiments", "results", "e16_q5_equity.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[e16] written, {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
