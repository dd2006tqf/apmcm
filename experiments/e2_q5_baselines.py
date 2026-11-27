"""E2 — Q5 baselines and the no-flood-map counterfactual.

16 Jul decision input:
  * lexicographic MILP optimum (reference);
  * "no flood map" counterfactual: all roads passable, no depot excluded by inundation;
  * priority-greedy constructive heuristic (nearest admissible depot, highest priority first);
  * capacity-proportional allocation on the optimal depot set.
"""
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "code"))

import numpy as np
import rasterio
import config as C
import common as U
import types, importlib.util
if importlib.util.find_spec("lightgbm") is None:      # problem5 -> problem4 -> problem2 -> problem1 chain
    sys.modules["lightgbm"] = types.ModuleType("lightgbm")
import problem5 as P5
from network import RoadNet
from problem4 import facility_proxies

RES = os.path.join(_HERE, "results")
os.makedirs(RES, exist_ok=True)


def rebuild_without_map(inp, net):
    p = inp["p"]
    G0, active0 = net.graph(None, speed_factor=p["speed"])
    cand, dem = inp["cand"].copy(), inp["dem"].copy()
    node, dist = net.snap(np.c_[cand.x, cand.y], active0)
    cand["node"], cand["access_m"] = node, dist
    cand["admissible"] = cand["access_m"] <= C.DEPOT_ACCESS_MAX_KM * 1000
    dnode, ddist = net.snap(np.c_[dem.cx, dem.cy], active0)
    dem["node"], dem["access_m"] = dnode, ddist
    dem["reach"] = ddist <= C.ACCESS_MAX_KM * 1000
    acc = C.ACCESS_KMH * 1000 / 60
    D = net.times_from(G0, cand["node"].values)
    tt = cand["access_m"].values[:, None] / acc + D[:, dem["node"].values] + dem["access_m"].values[None, :] / acc
    tt[:, ~dem["reach"].values] = np.inf
    tt[~cand["admissible"].values, :] = np.inf
    cap = np.where(cand["kind"].values == "city", p["cap_city"], p["cap_town"]) * p["cap_scale"]
    return dict(inp, G=G0, cand=cand, dem=dem, tt=tt, cap=cap)


def summarize(inp, served, fleet, opened, assign):
    dem = inp["dem"]
    d, w, P = dem["d"].values, dem["w"].values, dem["P"].values
    Wd = float((w * d).sum())
    f1 = float((w * np.maximum(d - served, 0)).sum() / Wd)
    f2 = float(sum(w[j] * inp["tt"][i, j] * t for j, (i, t) in assign.items()) / max(Wd, 1e-9))
    return dict(f1=f1, f2=f2, served_pop=float((np.minimum(served / np.maximum(d, 1e-12), 1) * P).sum()),
                shipped_t=float(served.sum()), demand_t=float(d.sum()),
                n_open=int(opened.sum()), open_depots=[inp["cand"]["name"].iloc[i] for i in np.where(opened)[0]],
                fleet_used_vehmin=float(fleet),
                fleet_budget_vehmin=float(60 * inp["p"]["n_veh"] * inp["p"]["shift"] * inp["p"]["H"]))


def greedy(inp):
    p, cand, dem, tt, cap = inp["p"], inp["cand"], inp["dem"], inp["tt"], inp["cap"]
    d, w = dem["d"].values, dem["w"].values
    rem, opened = cap.copy(), np.zeros(len(cand), bool)
    served, fleet, assign = np.zeros(len(dem)), 0.0, {}
    budget = 60 * p["n_veh"] * p["shift"] * p["H"]
    order = np.lexsort((-d, -w))
    for j in order:
        ci = np.where((tt[:, j] <= p["t_max"]) & (rem > 1e-9) & (opened | (opened.sum() < p["K"])))[0]
        if not len(ci):
            continue
        i = ci[np.argmin(tt[ci, j])]
        x = min(d[j] - served[j], rem[i])
        vpm = (2 * tt[i, j] + p["t_load"]) / p["q_veh"]
        if fleet + vpm * x > budget:
            x = max(0.0, (budget - fleet) / max(vpm, 1e-9))
        if x <= 1e-9:
            continue
        opened[i] = True; rem[i] -= x; served[j] += x; fleet += vpm * x
        assign[j] = (i, x)
    return summarize(inp, served, fleet, opened, assign)


def proportional(inp, eta):
    p, cand, dem, tt, cap = inp["p"], inp["cand"], inp["dem"], inp["tt"], inp["cap"]
    d, w = dem["d"].values, dem["w"].values
    served, fleet, assign = np.zeros(len(dem)), 0.0, {}
    budget = 60 * p["n_veh"] * p["shift"] * p["H"]
    for i in np.where(np.asarray(eta) > 0)[0]:
        js = np.where(tt[i] <= p["t_max"])[0]
        if not len(js):
            continue
        rem_cap = cap[i]
        for _ in range(30):                       # water-filling within the depot capacity
            if rem_cap <= 1e-9:
                break
            need = np.maximum(d[js] - served[js], 0)
            key = w[js] * d[js]
            wsum = key[need > 1e-9].sum()
            if wsum <= 0:
                break
            shipped = np.minimum(rem_cap * key / wsum, need)
            vpm = ((2 * tt[i, js] + p["t_load"]) / p["q_veh"] * shipped).sum()
            if fleet + vpm > budget:
                shipped = shipped * max(0.0, (budget - fleet) / max(vpm, 1e-9))
                vpm = ((2 * tt[i, js] + p["t_load"]) / p["q_veh"] * shipped).sum()
            if shipped.sum() <= 1e-9:
                break
            served[js] += shipped
            rem_cap -= shipped.sum()
            fleet += vpm
            for j, t in zip(js, shipped):
                if t > 1e-9:
                    assign[j] = (i, assign.get(j, (i, 0.0))[1] + float(t))
            if (need - shipped).max() <= 1e-9:
                break
    return summarize(inp, served, fleet, np.asarray(eta) > 0, assign)


def main():
    U.set_all_seeds()
    t0 = time.time()
    with rasterio.open(C.OUT / "q3_newflood_0716.tif") as d:
        crs = d.crs
    net = RoadNet(crs)
    fac = facility_proxies(crs)
    p0 = P5.base_params()
    inp = P5.build_inputs(C.MAIN_DATE, p0, net, fac)
    print(f"[E2] inputs {time.time()-t0:.0f}s cand={len(inp['cand'])} dem={len(inp['dem'])}", flush=True)

    s = P5.solve(inp, stage="lex")
    ref = dict(f1=float(s["f1"]), f2=float(s["f2"]), served_pop=s["served_pop"], shipped_t=s["total_shipped_t"],
               demand_t=s["total_demand_t"], n_open=sum(s["eta"]), open_depots=s["open_depots"],
               fleet_used_vehmin=s["fleet_used_vehmin"], fleet_budget_vehmin=s["fleet_budget_vehmin"],
               seconds=s["seconds"], status=s["status"], optimal=s["optimal"])
    print(f"[E2] MILP f1={s['f1']:.4f} f2={s['f2']:.2f} {s['seconds']:.0f}s", flush=True)

    t = time.time(); inp_nm = rebuild_without_map(inp, net); print(f"[E2] no-map rebuild {time.time()-t:.0f}s", flush=True)
    s_nm = P5.solve(inp_nm, stage="lex")
    nm = dict(f1=float(s_nm["f1"]), f2=float(s_nm["f2"]), served_pop=s_nm["served_pop"], shipped_t=s_nm["total_shipped_t"],
              demand_t=s_nm["total_demand_t"], n_open=sum(s_nm["eta"]), open_depots=s_nm["open_depots"],
              fleet_used_vehmin=s_nm["fleet_used_vehmin"], fleet_budget_vehmin=s_nm["fleet_budget_vehmin"],
              seconds=s_nm["seconds"], status=s_nm["status"], optimal=s_nm["optimal"],
              n_candidates=len(inp_nm["cand"]), n_admissible=int(inp_nm["cand"]["admissible"].sum()))
    print(f"[E2] no-map f1={s_nm['f1']:.4f} f2={s_nm['f2']:.2f}", flush=True)

    # replay the no-map depot set on the real (cut) network: what a plan built without the map actually delivers
    names_nm = {inp_nm["cand"]["name"].iloc[i] for i in range(len(inp_nm["cand"])) if s_nm["eta"][i]}
    fix_nm = [int(n in names_nm) for n in inp["cand"]["name"]]
    s_replay = P5.solve(inp, fix_eta=fix_nm)
    replay = dict(f1=float(s_replay["f1"]), f2=float(s_replay["f2"]), served_pop=s_replay["served_pop"],
                  shipped_t=s_replay["total_shipped_t"], demand_t=s_replay["total_demand_t"],
                  n_open=sum(s_replay["eta"]), open_depots=s_replay["open_depots"],
                  seconds=s_replay["seconds"], status=s_replay["status"], optimal=s_replay["optimal"],
                  note="depot set chosen without the flood map, flows re-optimised on the real cut network")
    print(f"[E2] no-map replay on real cuts f1={s_replay['f1']:.4f} served_pop={s_replay['served_pop']:.0f}", flush=True)

    g = greedy(inp)
    pr = proportional(inp, s["eta"])
    out = dict(meta=dict(date=C.MAIN_DATE, seed=C.SEED, runtime_s=time.time() - t0,
                         n_candidates=len(inp["cand"]), n_admissible=int(inp["cand"]["admissible"].sum()),
                         n_demand=len(inp["dem"])),
               milp=ref, no_map=nm, no_map_replay=replay, greedy=g, proportional=pr,
               gaps=dict(f1_greedy_vs_opt=g["f1"] - ref["f1"], f2_greedy_vs_opt=g["f2"] - ref["f2"],
                         f1_nomap_vs_opt=nm["f1"] - ref["f1"], f2_nomap_vs_opt=nm["f2"] - ref["f2"],
                         f1_prop_vs_opt=pr["f1"] - ref["f1"],
                         served_pop_nomap_share=nm["served_pop"] / max(ref["served_pop"], 1e-9)))
    json.dump(out, open(os.path.join(RES, "e2_q5_baselines.json"), "w"), indent=1)
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in out.items() if k != "meta"}, indent=1))


if __name__ == "__main__":
    main()
