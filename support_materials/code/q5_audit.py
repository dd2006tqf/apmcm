"""Independent constraint audit of the saved Q5 plans (MODELING_REPORT.md §8.5).

Reloads output/q5_results.json and recomputes from the raw inputs (Q3 F rasters, Q4 unit files, OSM roads):
C10 demand balance, C11 capacity, C12 open-depot count, C13 reachability within T_max (fresh Dijkstra on a
re-built cut graph), C14 open depots clear of F (100 m), fleet budget, C15 demand = q*H*P/1000 from Q4.
Prints conclusions only.
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import config as C
from network import RoadNet
from problem4 import near_F

TOL = 1e-6


def audit(path=C.OUT / "q5_results.json"):
    res = json.load(open(path, encoding="utf-8"))
    plans = [res["plan_0716"], res["plan_0704"], res["recourse_0704_to_0716"], res["knee_plan"]] + res["pareto_plans"]
    with rasterio.open(C.OUT / "q3_newflood_0716.tif") as d:
        crs = d.crs
    net = RoadNet(crs)
    Fc, gc = {}, {}
    q4 = gpd.read_file(C.OUT / "q4_units.geojson")
    P16 = dict(zip(q4["uid"].astype(int), q4["E1"]))
    q404 = pd.read_csv(C.OUT / "q4_units_0704.csv")
    P04 = dict(zip(q404["uid"].astype(int), q404["E1"]))
    report, allok = [], True
    for pl in plans:
        p, date = pl["params"], pl["date"]
        if date not in Fc:
            with rasterio.open(C.OUT / f"q3_newflood_{date[4:]}.tif") as d:
                Fc[date] = (d.read(1).astype(bool), d.transform)
        F, T = Fc[date]
        key = (date, p["buffer"], p["speed"])     # the cut graph does not depend on the facility set
        if key not in gc:
            cut = net.cut_mask(F, T, buffer_m=p["buffer"])
            gc[key] = net.graph(cut, speed_factor=p["speed"])
        G, active = gc[key]
        dep, dem = pl["depots"], pl["demand"]
        nI, nJ = len(dep), len(dem)
        jidx = {dm["uid"]: j for j, dm in enumerate(dem)}
        iidx = {dp["name"]: i for i, dp in enumerate(dep)}
        inflow, outflow = np.zeros(nJ), np.zeros(nI)
        for fl in pl["flows"]:
            inflow[jidx[fl["uid"]]] += fl["tons"]; outflow[iidx[fl["depot"]]] += fl["tons"]
        d = np.array([dm["d"] for dm in dem]); u = np.array([dm["u"] for dm in dem])
        v = {}
        v["C10_balance"] = int((np.abs(inflow + u - d) > TOL).sum())
        cap = np.array([dp["cap"] for dp in dep]); eta = np.array(pl["eta"])
        v["C11_capacity"] = int((outflow > cap * eta + TOL).sum())
        v["C12_K"] = int(eta.sum() > p["K"])
        # C13: fresh Dijkstra
        acc = C.ACCESS_KMH * 1000 / 60
        bad13 = 0
        if pl["flows"]:
            src = sorted({iidx[fl["depot"]] for fl in pl["flows"]})
            sxy = np.array([[dep[i]["x"], dep[i]["y"]] for i in src])
            snode, sdist = net.snap(sxy, active)
            dist = net.times_from(G, snode)
            for fl in pl["flows"]:
                k = src.index(iidx[fl["depot"]])
                dm = dem[jidx[fl["uid"]]]
                dn, dd = net.snap(np.array([[dm["cx"], dm["cy"]]]), active)
                t = sdist[k] / acc + dist[k, dn[0]] + dd[0] / acc
                if not (np.isfinite(t) and t <= p["t_max"] + 1e-6 and dd[0] <= C.ACCESS_MAX_KM * 1000 + 1e-6):
                    bad13 += 1
        v["C13_reachability"] = bad13
        v["C14_open_in_F100"] = int(sum(near_F(F, T, dep[i]["x"], dep[i]["y"], C.DEPOT_FLOOD_CLEAR_M) for i in range(nI) if eta[i]))
        fleet = sum((2 * fl["t"] + p["t_load"]) / p["q_veh"] * fl["tons"] for fl in pl["flows"])
        v["fleet_budget"] = int(fleet > 60 * p["n_veh"] * p["shift"] * p["H"] + TOL)
        Pq = P16 if date == C.MAIN_DATE else P04
        d_re = np.array([p["q"] * p["H"] * Pq[dm["uid"]] / 1000 for dm in dem])
        v["C15_demand"] = int((np.abs(d_re - d) > 1e-6 * np.maximum(1, d)).sum())
        v["status_not_optimal"] = int(not pl["optimal"])
        n = sum(v.values())
        allok &= n == 0
        report.append(dict(plan=pl["label"], PASS=n == 0, n_violations=n, **v))
        print(f"[{pl['label']}] PASS={n == 0} n_violations={n}" + ("" if n == 0 else f" {v}"))
    print(f"[Q5 audit] plans audited={len(report)} PASS={allok}")
    return dict(PASS=allok, plans=report)


if __name__ == "__main__":
    out = audit()
    raise SystemExit(0 if out["PASS"] else 1)
