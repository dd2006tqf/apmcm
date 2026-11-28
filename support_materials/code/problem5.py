"""Problem 5 — emergency relief allocation with depot activation on the flood-cut network
(MODELING_REPORT.md §8). Bi-objective MILP (8.2)-(8.3) solved exactly (HiGHS by default, CBC selectable):
lexicographic (min weighted unmet share f1, then min weighted delivery time f2) + epsilon-constraint
Pareto front, information-update value 07-04 -> 07-16, and 30 one-at-a-time sensitivity solves
(the 28 of the plan + the facility-proxy alternative A12).
Demand d_j = q*H*P_j/1000 and priority w_j = S_j come from Q4; road cuts and depot exclusion from Q3 F.
"""
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import pulp
import config as C
import common as U
from network import RoadNet
from problem4 import read_F, near_F, facility_proxies

SOLVER = os.environ.get("Q5_SOLVER", "highs")   # "highs" (default) or "cbc"
W_FLOOR = 1e-3     # priority floor so that an affected unit with S_j = 0 (TOPSIS minimum) still counts


def base_params():
    return dict(K=C.K_MAX, cap_town=C.CAP_TOWN_T, cap_city=C.CAP_CITY_T, t_max=C.T_MAX_MIN, q=C.Q_KG, H=C.HORIZON_D,
                speed=C.SPEED_FACTOR, buffer=C.ROAD_BUFFER_M, n_veh=C.N_VEH, q_veh=C.Q_VEH_T, shift=C.SHIFT_H,
                t_load=C.T_LOAD_MIN, cap_scale=1.0, facility=C.FACILITY_PROXY)


def demand_table(date):
    """Q4 units of the decision date: affected units with flooded population >= 1."""
    if date == C.MAIN_DATE:
        g = gpd.read_file(C.OUT / "q4_units.geojson")
        df = pd.DataFrame(dict(uid=g["uid"], P=g["E1"], S=g["S"], cx=g["cx"], cy=g["cy"], affected=g["affected"].astype(bool)))
    else:
        df = pd.read_csv(C.OUT / "q4_units_0704.csv").rename(columns={"E1": "P"})
        df["affected"] = df["affected"].astype(bool)
    return df[(df.affected) & (df.P >= 1)].reset_index(drop=True)


def build_inputs(date, p, net, fac, cache={}):
    """Cut graph, candidate depots (C14), demand (C15) and travel times t_ij for one decision date."""
    key = (date, p["buffer"], p["speed"], p.get("facility", C.FACILITY_PROXY))
    if key in cache:
        base = cache[key]
    else:
        F, T, crs = read_F(f"q3_newflood_{date[4:]}.tif")
        cut = net.cut_mask(F, T, buffer_m=p["buffer"])
        G, active = net.graph(cut, speed_factor=p["speed"])
        # candidate depots: town/city nodes + schools merged within 300 m (city first)
        if p.get("facility", C.FACILITY_PROXY) == "schools_only":     # A12 alternative
            fac = fac[fac["kind"] == "school"]
        fc = fac.sort_values("kind", key=lambda s: s.map({"city": 0, "town": 1, "school": 2})).reset_index(drop=True)
        keep = []
        for i, r in fc.iterrows():
            if all(np.hypot(r.x - fc.x[j], r.y - fc.y[j]) > C.DEPOT_MERGE_M for j in keep):
                keep.append(i)
        cand = fc.loc[keep].reset_index(drop=True)
        cand["near_F100"] = [near_F(F, T, x, y, C.DEPOT_FLOOD_CLEAR_M) for x, y in zip(cand.x, cand.y)]
        node, dist = net.snap(np.c_[cand.x, cand.y], active)
        cand["node"], cand["access_m"] = node, dist
        cand["admissible"] = (~cand.near_F100) & (cand.access_m <= C.DEPOT_ACCESS_MAX_KM * 1000)
        dem = demand_table(date)
        dnode, ddist = net.snap(np.c_[dem.cx, dem.cy], active)
        dem["node"], dem["access_m"] = dnode, ddist
        dem["reach"] = ddist <= C.ACCESS_MAX_KM * 1000
        acc = C.ACCESS_KMH * 1000 / 60
        D = net.times_from(G, cand["node"].values)                    # (n_cand, n_nodes)
        tt = cand["access_m"].values[:, None] / acc + D[:, dem["node"].values] + dem["access_m"].values[None, :] / acc
        tt[:, ~dem["reach"].values] = np.inf
        tt[~cand["admissible"].values, :] = np.inf
        base = dict(T=T, cut=cut, G=G, active=active, cand=cand, dem=dem, tt=tt)
        del F
        cache[key] = base
    dem = base["dem"].copy()
    dem["d"] = p["q"] * p["H"] * dem["P"] / 1000.0
    dem["w"] = np.maximum(dem["S"], W_FLOOR)
    cap = np.where(base["cand"]["kind"].values == "city", p["cap_city"], p["cap_town"]) * p["cap_scale"]
    return dict(base, dem=dem, cap=cap, p=p, date=date)
# ============================================================ MILP (8.2)-(8.3)
def solve(inp, stage="lex", eps=None, fix_eta=None, time_limit=300):
    p, dem, cand, tt, cap = inp["p"], inp["dem"], inp["cand"], inp["tt"], inp["cap"]
    nI, nJ = tt.shape
    d, w = dem["d"].values, dem["w"].values
    Wd = float((w * d).sum())
    A = [(i, j) for i in range(nI) for j in range(nJ) if tt[i, j] <= p["t_max"]]
    t0 = time.time()

    def build(obj):
        m = pulp.LpProblem("relief", pulp.LpMinimize)
        eta = {i: pulp.LpVariable(f"eta_{i}", cat=pulp.LpBinary) for i in range(nI)}
        f = {(i, j): pulp.LpVariable(f"f_{i}_{j}", lowBound=0) for (i, j) in A}
        u = {j: pulp.LpVariable(f"u_{j}", lowBound=0) for j in range(nJ)}
        f1 = pulp.lpSum(w[j] * u[j] for j in range(nJ)) * (1.0 / Wd)
        f2 = pulp.lpSum(w[j] * tt[i, j] * f[i, j] for (i, j) in A) * (1.0 / Wd)
        byj, byi = {}, {}
        for (i, j) in A:
            byj.setdefault(j, []).append(f[i, j]); byi.setdefault(i, []).append(f[i, j])
        for j in range(nJ):                                              # C10 demand balance
            m += pulp.lpSum(byj.get(j, [])) + u[j] == d[j], f"bal_{j}"
        for i in range(nI):                                              # C11 capacity / open depots only
            m += pulp.lpSum(byi.get(i, [])) <= cap[i] * eta[i], f"cap_{i}"
            if not cand["admissible"].iloc[i]:                          # C14 excluded candidates stay closed
                m += eta[i] == 0, f"excl_{i}"
            if fix_eta is not None:
                m += eta[i] == int(fix_eta[i] and cand["admissible"].iloc[i]), f"fix_{i}"
        for (i, j) in A:                                                 # valid tightening
            m += f[i, j] <= d[j] * eta[i], f"tight_{i}_{j}"
        m += pulp.lpSum(eta.values()) <= p["K"], "K"                     # C12
        m += pulp.lpSum((2 * tt[i, j] + p["t_load"]) / p["q_veh"] * f[i, j] for (i, j) in A) \
            <= 60 * p["n_veh"] * p["shift"] * p["H"], "fleet"
        return m, eta, f, u, f1, f2

    def run(m):
        # HiGHS is the default: CBC 2.10 (PULP_CBC_CMD) stalled without CPU progress for 20 min on one
        # sensitivity case on Windows. Both are exact branch-and-bound MILP solvers with gapRel = 0.
        if SOLVER == "cbc":
            m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=0, threads=1, options=["twoMirCuts off"]))
        else:
            m.solve(pulp.HiGHS(msg=False, timeLimit=time_limit, gapRel=0, threads=1))
        return pulp.LpStatus[m.status]

    statuses = []
    if stage == "lex":
        m, eta, f, u, f1, f2 = build(None)
        m.setObjective(f1)
        statuses.append(run(m))
        f1s = pulp.value(f1)
        m, eta, f, u, f1, f2 = build(None)
        m += f1 <= f1s + 1e-6, "lex"
        m.setObjective(f2)
        statuses.append(run(m))
    else:
        m, eta, f, u, f1, f2 = build(None)
        m += f1 <= eps, "eps"
        m.setObjective(f2)
        statuses.append(run(m))
    sol = dict(status=statuses, optimal=all(s == "Optimal" for s in statuses), f1=pulp.value(f1), f2=pulp.value(f2),
               eta=[int(round(eta[i].value() or 0)) for i in range(nI)],
               flows=[dict(i=i, j=j, depot=cand["name"].iloc[i], uid=int(dem["uid"].iloc[j]), t=float(tt[i, j]),
                           tons=float(f[i, j].value())) for (i, j) in A if (f[i, j].value() or 0) > 1e-7],
               u=[float(u[j].value() or 0) for j in range(nJ)], n_pairs=len(A), seconds=time.time() - t0,
               gap=0.0 if all(s == "Optimal" for s in statuses) else None)
    served = np.zeros(nJ)
    for fl in sol["flows"]:
        served[fl["j"]] += fl["tons"]
    P = dem["P"].values
    sol["served_pop"] = float((np.minimum(served / np.maximum(d, 1e-12), 1) * P).sum())
    sol["total_demand_t"] = float(d.sum())
    sol["total_shipped_t"] = float(served.sum())
    sol["open_depots"] = [cand["name"].iloc[i] for i in range(nI) if sol["eta"][i]]
    use = np.zeros(nI)
    for fl in sol["flows"]:
        use[fl["i"]] += fl["tons"]
    sol["depot_use_t"] = {cand["name"].iloc[i]: float(use[i]) for i in range(nI) if sol["eta"][i]}
    sol["fleet_used_vehmin"] = float(sum((2 * fl["t"] + p["t_load"]) / p["q_veh"] * fl["tons"] for fl in sol["flows"]))
    sol["fleet_budget_vehmin"] = 60 * p["n_veh"] * p["shift"] * p["H"]
    return sol


def pareto(inp, f1s):
    hi = min(1.0, f1s + 0.5)
    pts = []
    for k in range(15):
        e = f1s + k / 14 * (hi - f1s)
        s = solve(inp, stage="eps", eps=e + (1e-6 if k == 0 else 0))
        pts.append(dict(k=k, eps=e, f1=s["f1"], f2=s["f2"], n_open=sum(s["eta"]), served_pop=s["served_pop"],
                        optimal=s["optimal"], sol=s))
    x = np.array([q["f1"] for q in pts]); y = np.array([q["f2"] for q in pts])
    xn = (x - x.min()) / max(x.max() - x.min(), 1e-12); yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    a, b = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    ab = b - a
    dist = np.abs(ab[0] * (yn - a[1]) - ab[1] * (xn - a[0])) / max(np.linalg.norm(ab), 1e-12)
    return pts, int(np.argmax(dist))
# ============================================================ reporting helpers
def plan_record(inp, sol, label):
    """Everything the independent audit needs to recompute C10-C15 from scratch."""
    c, dm = inp["cand"], inp["dem"]
    return dict(label=label, date=inp["date"], params=inp["p"], f1=sol["f1"], f2=sol["f2"], status=sol["status"],
                optimal=sol["optimal"], gap=sol["gap"], open_depots=sol["open_depots"], eta=sol["eta"],
                depots=[dict(name=c["name"].iloc[i], kind=c["kind"].iloc[i], x=float(c.x.iloc[i]), y=float(c.y.iloc[i]),
                             cap=float(inp["cap"][i]), admissible=bool(c["admissible"].iloc[i]),
                             near_F100=bool(c["near_F100"].iloc[i]), access_m=float(c["access_m"].iloc[i])) for i in range(len(c))],
                demand=[dict(uid=int(dm.uid.iloc[j]), P=float(dm.P.iloc[j]), d=float(dm.d.iloc[j]), w=float(dm.w.iloc[j]),
                             S=float(dm.S.iloc[j]), cx=float(dm.cx.iloc[j]), cy=float(dm.cy.iloc[j]),
                             access_m=float(dm.access_m.iloc[j]), u=sol["u"][j]) for j in range(len(dm))],
                flows=sol["flows"], served_pop=sol["served_pop"], total_demand_t=sol["total_demand_t"],
                total_shipped_t=sol["total_shipped_t"], depot_use_t=sol["depot_use_t"],
                fleet_used_vehmin=sol["fleet_used_vehmin"], fleet_budget_vehmin=sol["fleet_budget_vehmin"],
                n_pairs=sol["n_pairs"], seconds=sol["seconds"])


def flow_paths(inp, sol, net):
    """Node paths of every positive flow (for the allocation map)."""
    from scipy.sparse.csgraph import dijkstra
    out = []
    srcs = sorted({fl["i"] for fl in sol["flows"]})
    if not srcs:
        return out
    nodes = inp["cand"]["node"].values[srcs]
    _, pred = dijkstra(inp["G"], directed=True, indices=nodes, return_predecessors=True)
    for fl in sol["flows"]:
        k = srcs.index(fl["i"])
        tgt = int(inp["dem"]["node"].iloc[fl["j"]])
        path, v, guard = [], tgt, 0
        while v != -9999 and v >= 0 and guard < 100000:
            path.append(v)
            if v == nodes[k]:
                break
            v = pred[k, v]; guard += 1
        xy = net.xy[path[::-1]]
        step = max(1, len(xy) // 200)
        out.append(dict(depot=fl["depot"], uid=fl["uid"], tons=fl["tons"], t=fl["t"], xy=xy[::step].tolist() + [xy[-1].tolist()]))
    return out


def cut_roads_geojson(inp, net, name):
    import shapely.geometry as sg
    g = gpd.GeoDataFrame(dict(cut=inp["cut"].astype(int), speed=net.speed),
                         geometry=[sg.LineString([(a, b), (c, d)]) for a, b, c, d in net.segs], crs="EPSG:32649")
    g.to_crs(4326).to_file(C.OUT / name, driver="GeoJSON")


def validate_capability(res):
    for pl in [res["plan_0716"], res["plan_0704"], res["recourse_0704_to_0716"]]:
        assert pl["optimal"], f"[P5-C2] {pl['label']} not optimal"
        assert sum(pl["eta"]) <= pl["params"]["K"]
    assert res["audit"]["PASS"], "[P5-C2] audit failed"
    assert res["chain"]["demand_from_q4"] and res["chain"]["cuts_from_q3"], "[P5-C3]"
    assert res["plan_0704"]["total_demand_t"] != res["plan_0716"]["total_demand_t"] or \
        res["voi"]["pairs_lost"] + res["voi"]["pairs_gained"] > 0, "[P5-C3] 07-04 and 07-16 inputs identical"
    print("[Q5] validate_capability PASS")
# ============================================================ main
def main():
    U.set_all_seeds()
    t0 = time.time()
    with rasterio.open(C.OUT / "q3_newflood_0716.tif") as d:
        crs = d.crs
    net = RoadNet(crs)
    fac = facility_proxies(crs)
    p0 = base_params()
    md, ed = C.MAIN_DATE, C.FLOOD_DATES[0]
    inp16, inp04 = build_inputs(md, p0, net, fac), build_inputs(ed, p0, net, fac)
    s16 = solve(inp16)
    print(f"[Q5] 07-16 plan: f1*={s16['f1']:.4f} f2={s16['f2']:.2f} min open={s16['open_depots']} {s16['status']}", flush=True)
    pts, knee = pareto(inp16, s16["f1"])
    s04 = solve(inp04)
    # recourse: keep the 07-04 depot set (by name), re-optimize flows under 07-16 inputs
    names04 = {inp04["cand"]["name"].iloc[i] for i in range(len(inp04["cand"])) if s04["eta"][i]}
    fix = [int(n in names04) for n in inp16["cand"]["name"]]
    rec = solve(inp16, fix_eta=fix)
    ref = solve(inp16, stage="eps", eps=rec["f1"] + 1e-6)
    pairs = lambda inp: {(inp["cand"]["name"].iloc[i], int(inp["dem"]["uid"].iloc[j]))
                         for i in range(inp["tt"].shape[0]) for j in range(inp["tt"].shape[1]) if inp["tt"][i, j] <= inp["p"]["t_max"]}
    P16, P04 = pairs(inp16), pairs(inp04)
    o16, o04 = set(s16["open_depots"]), set(s04["open_depots"])
    voi = dict(VoI1=rec["f1"] - s16["f1"], VoI2=rec["f2"] - ref["f2"], f1_recourse=rec["f1"], f2_recourse=rec["f2"],
               f2_ref_at_equal_f1=ref["f2"], jaccard_open=len(o16 & o04) / max(len(o16 | o04), 1),
               demand_change_t=inp16["dem"]["d"].sum() - inp04["dem"]["d"].sum(),
               pairs_lost=len(P04 - P16), pairs_gained=len(P16 - P04),
               depots_04_excluded_on_16=[n for n in names04 if not inp16["cand"].set_index("name").loc[n, "admissible"]],
               served_pop_recourse=rec["served_pop"], served_pop_opt=s16["served_pop"])
    print(f"[Q5] VoI1={voi['VoI1']:.4f} VoI2={voi['VoI2']:.3f} jaccard={voi['jaccard_open']:.2f}", flush=True)
    # ---------------- sensitivity (8.6), exact re-solves
    sens = []
    grid = [("K", v) for v in (3, 4, 5, 6, 7, 8)] + [("cap_scale", v) for v in (0.5, 0.75, 1.0, 1.5)] + \
           [("t_max", v) for v in (60, 90, 120, 180, 240)] + [("q", v) for v in (2.5, 3.5, 7.0, 15.6)] + \
           [("speed", v) for v in (0.6, 0.8, 1.0)] + [("buffer", v) for v in (0, 20, 50)] + [("n_veh", v) for v in (20, 40, 60)] + \
           [("facility", v) for v in ("town_nodes", "schools_only")]
    for name, v in grid:
        p = dict(p0); p[name] = v
        s = solve(build_inputs(md, p, net, fac), stage="lex")
        sens.append(dict(param=name, value=v, f1=s["f1"], f2=s["f2"], open=s["open_depots"], n_open=sum(s["eta"]),
                         served_pop=s["served_pop"], optimal=s["optimal"], total_demand_t=s["total_demand_t"],
                         fleet_used_share=s["fleet_used_vehmin"] / s["fleet_budget_vehmin"]))
        print(f"[Q5] sens {name}={v}: f1={s['f1']:.4f} f2={s['f2']:.2f} open={sum(s['eta'])}", flush=True)
    # monotonic probes (logic contract)
    def probe(par):
        rows = [r for r in sens if r["param"] == par]
        f = [r["f1"] for r in rows]
        diffs = np.diff(f)
        return int(np.sign(diffs[np.abs(diffs) > 1e-9][0])) if np.any(np.abs(diffs) > 1e-9) else 0, \
            bool(np.all(diffs <= 1e-7))
    k_sign, k_ok = probe("K")
    c_sign, c_ok = probe("cap_scale")
    res = dict(meta=U.run_meta(), solver=SOLVER, decision_time=md, update_time=ed, params=p0,
               task="depot activation + relief allocation (location-allocation MILP) on the Q3 flood-cut network",
               demand_formula="d_j = q*H*P_j/1000 (t), P_j = Q4 flooded population, w_j = max(S_j, 1e-3)",
               n_candidates=len(inp16["cand"]), n_admissible=int(inp16["cand"]["admissible"].sum()),
               n_demand_units=len(inp16["dem"]), unreachable_units=int((~np.isfinite(inp16["tt"]).any(0) |
                                                                       ~(inp16["tt"] <= p0["t_max"]).any(0)).sum()),
               cut_road_km=dict(d0716=float(net.len_m[inp16["cut"]].sum() / 1000), d0704=float(net.len_m[inp04["cut"]].sum() / 1000)),
               plan_0716=plan_record(inp16, s16, "plan_0716"), plan_0704=plan_record(inp04, s04, "plan_0704"),
               recourse_0704_to_0716=plan_record(inp16, rec, "recourse_0704_to_0716"),
               pareto=[dict({k: v for k, v in q.items() if k != "sol"}) for q in pts], knee_index=knee,
               knee_plan=plan_record(inp16, pts[knee]["sol"], "knee"),
               pareto_plans=[plan_record(inp16, q["sol"], f"pareto_{q['k']}") for q in pts],
               voi=voi, sensitivity=sens, paths_0716=flow_paths(inp16, s16, net),
               monotonic_probes=dict(K=dict(sign=k_sign, nonincreasing=k_ok), cap=dict(sign=c_sign, nonincreasing=c_ok)),
               chain=dict(demand_from_q4="output/q4_units.geojson", demand_0704_from_q4="output/q4_units_0704.csv",
                          cuts_from_q3=["output/q3_newflood_0716.tif", "output/q3_newflood_0704.tif"],
                          sha256={f: U.file_sha(C.ROOT / f) for f in ["output/q4_units.geojson", "output/q4_units_0704.csv",
                                                                     "output/q3_newflood_0716.tif", "output/q3_newflood_0704.tif"]}),
               seconds=time.time() - t0)
    cut_roads_geojson(inp16, net, "q5_roads_cut_0716.geojson")
    U.save_json(res, C.OUT / "q5_results.json")
    import q5_audit
    audit = q5_audit.audit(C.OUT / "q5_results.json")
    res["audit"] = audit
    validate_capability(res)
    ok = U.validate_constraints({"f1": [s16["f1"], s04["f1"], rec["f1"]], "n_open": [sum(s16["eta"]), sum(s04["eta"])]},
                                {"f1": (0, 1, "f1 in [0,1]"), "n_open": (0, C.K_MAX, "open depots <= K")}, "Q5")
    if not ok:
        raise ValueError("Q5 constraint validation failed")
    U.save_json(res, C.OUT / "q5_results.json")
    U.save_json(res, C.FIG / "problem_5_results.json")
    print(f"[Q5] done in {time.time() - t0:.0f}s")
    return res


if __name__ == "__main__":
    main()
