"""E15 - Q5 emergency road-repair analysis (numbers only; no paper/figure side effects).

Reads the published Q5 inputs for 2026-07-16 (base params), then:
  1) builds the flood-cut graph, weak components and the component graph H;
  2) BFS from the depot components -> minimum number of cut segments k per component;
  3) lists the 57 unreachable demand units with k, weight and coordinates;
  4) takes the *directed* reachable node set from the depots and restores every boundary cut
     segment once, recomputing Dijkstra travel times and the realized gain;
  5) greedy budget curve (max realized dF1 per km) with an exact lexicographic MILP re-solve at
     every step;
  6) scenario MILPs: <=300 m short repairs, all boundary segments.

Outputs: output/q5_repair.json, output/q5_repair_edges.geojson
"""
import os, sys, json, time, types
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.modules.setdefault("lightgbm", types.ModuleType("lightgbm"))

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import shapely.geometry as sg
from scipy.sparse.csgraph import connected_components
import config as C
import common as U
import problem5 as P5
from network import RoadNet
from problem4 import facility_proxies

T0 = time.time()
def log(msg):
    print(f"[e15 {time.time() - T0:7.1f}s] {msg}", flush=True)

# ---------------------------------------------------------------- inputs
with rasterio.open(C.OUT / "q3_newflood_0716.tif") as d:
    crs = d.crs
net = RoadNet(crs)
fac = facility_proxies(crs)
p = P5.base_params()
inp = P5.build_inputs(C.MAIN_DATE, p, net, fac)
log(f"network {net.n} nodes / {len(net.eu)} edges; cut {int(inp['cut'].sum())}; "
    f"candidates {len(inp['cand'])} ({int(inp['cand']['admissible'].sum())} admissible); demand {len(inp['dem'])}")

dem, cand = inp["dem"], inp["cand"]
eu, ev, L, SPD = net.eu, net.ev, net.len_m, net.speed
nE = len(eu)
acc = C.ACCESS_KMH * 1000 / 60
cn = cand["node"].values.astype(int)
dn = dem["node"].values.astype(int)
ca = cand["access_m"].values / acc
da = dem["access_m"].values / acc
adm = cand["admissible"].values.astype(bool)
dr = dem["reach"].values.astype(bool)
w = dem["w"].values
dd = dem["d"].values
wd = w * dd
Wd = float(wd.sum())
tmax = p["t_max"]

base = P5.solve(inp)
f1_base = base["f1"]
reach0 = (inp["tt"] <= tmax).any(0)
log(f"baseline MILP f1={f1_base:.6f} f2={base['f2']:.3f} optimal={base['optimal']} "
    f"served_pop={base['served_pop']:.1f}; unreachable units {int((~reach0).sum())}")
log(f"baseline f1 recomputed from u: {float((w * np.asarray(base['u'])).sum() / Wd):.6f}")

# ---------------------------------------------------------------- components and k
G0, _ = net.graph(inp["cut"])
lab0 = connected_components(G0 + G0.T, directed=False)[1]
depot_comps = sorted({int(lab0[cn[i]]) for i in range(len(cn)) if adm[i]})
log(f"components {len(set(lab0.tolist()))}; depot components {len(depot_comps)}")

def comps_for(cut2):
    G2, _ = net.graph(cut2)
    lab = connected_components(G2 + G2.T, directed=False)[1]
    adj = {}
    for e in np.where(cut2)[0]:
        cu, cv = int(lab[eu[e]]), int(lab[ev[e]])
        if cu != cv:
            adj.setdefault(cu, []).append((int(e), cv))
            adj.setdefault(cv, []).append((int(e), cu))
    kc, dq = {}, deque()
    for c0 in depot_comps:
        kc[c0] = 0
        dq.append(c0)
    while dq:
        c = dq.popleft()
        for e, nb in adj.get(c, ()):
            if nb not in kc:
                kc[nb] = kc[c] + 1
                dq.append(nb)
    return lab, kc, adj

lab, kc, adj = comps_for(inp["cut"])

# ---------------------------------------------------------------- 57 units table
q4 = gpd.read_file(C.OUT / "q4_units.geojson")[["uid", "name", "place", "zone"]]
q4 = q4.set_index("uid")

def unit_rows():
    rows = []
    tt = inp["tt"]
    for j in range(len(dem)):
        if reach0[j]:
            continue
        c = int(lab[dn[j]])
        k = kc.get(c)
        rows.append(dict(uid=int(dem["uid"].iloc[j]), name=str(q4.loc[dem["uid"].iloc[j], "name"]),
                         place=str(q4.loc[dem["uid"].iloc[j], "place"]),
                         zone=str(q4.loc[dem["uid"].iloc[j], "zone"]),
                         P=float(dem["P"].iloc[j]), d=float(dd[j]), w=float(w[j]),
                         S=float(dem["S"].iloc[j]), wd_share=float(wd[j] / Wd * 100),
                         cx=float(dem["cx"].iloc[j]), cy=float(dem["cy"].iloc[j]),
                         access_m=float(dem["access_m"].iloc[j]), k=k,
                         comp=c, min_tt=float(np.min(tt[:, j])),
                         status=("depot_component_but_directed_blocked" if k == 0 else "cut_off")))
    return sorted(rows, key=lambda r: -r["wd_share"])

units = unit_rows()
log(f"unreachable units {len(units)}; top10 share {sum(r['wd_share'] for r in units[:10]):.2f}%; "
    f"k>=4 {sum(1 for r in units if r['k'] is not None and r['k'] >= 4)}")

# ---------------------------------------------------------------- directed boundary segments
D0 = net.times_from(G0, cn)
nr0 = np.isfinite(D0).any(0)
front0 = [int(e) for e in np.where(inp["cut"])[0] if bool(nr0[eu[e]]) != bool(nr0[ev[e]])]
log(f"directed boundary segments {len(front0)}, total {sum(L[e] for e in front0) / 1000:.2f} km")

def state_for(cut2):
    G2, _ = net.graph(cut2)
    D = net.times_from(G2, cn)
    tt = ca[:, None] + D[:, dn] + da[None, :]
    tt[:, ~dr] = np.inf
    tt[~adm, :] = np.inf
    return tt, np.isfinite(D).any(0)

single = []
for i, e in enumerate(front0):
    cut2 = inp["cut"].copy()
    cut2[e] = False
    tt2, nr2 = state_for(cut2)
    r2 = (tt2 <= tmax).any(0)
    new = r2 & ~reach0
    single.append(dict(edge=e, len_m=float(L[e]), speed_kmh=float(SPD[e]),
                       comp_u=int(lab[eu[e]]), comp_v=int(lab[ev[e]]),
                       k_u=kc.get(int(lab[eu[e]])), k_v=kc.get(int(lab[ev[e]])),
                       gain_conn_pp=float(100 * wd[new].sum() / Wd), n_new=int(new.sum()),
                       new_pop=float(dem["P"].values[new].sum()),
                       uids=[int(dem["uid"].iloc[j]) for j in np.where(new)[0]]))
    if (i + 1) % 50 == 0:
        log(f"  single-edge {i + 1}/{len(front0)}")
single.sort(key=lambda r: -r["gain_conn_pp"])
log("top single segments: " + "; ".join(f"e{r['edge']} {r['len_m']:.0f}m -> {r['gain_conn_pp']:.2f}pp/{r['n_new']}u"
                                        for r in single[:5]))

# ---------------------------------------------------------------- MILP scenarios
def scenario(name, edges):
    cut2 = inp["cut"].copy()
    if len(edges):
        cut2[np.asarray(edges, dtype=int)] = False
    tt2, _ = state_for(cut2)
    sol = P5.solve(dict(inp, tt=tt2))
    r2 = (tt2 <= tmax).any(0)
    new = r2 & ~reach0
    d = dict(name=name, n_edges=int(len(edges)), km=float(sum(L[e] for e in edges) / 1000),
             f1=float(sol["f1"]), f2=float(sol["f2"]), optimal=bool(sol["optimal"]),
             served_pop=float(sol["served_pop"]), total_shipped_t=float(sol["total_shipped_t"]),
             n_new=int(new.sum()), new_pop=float(dem["P"].values[new].sum()),
             gain_conn_pp=float(100 * wd[new].sum() / Wd), open_depots=sol["open_depots"],
             uids=[int(dem["uid"].iloc[j]) for j in np.where(new)[0]], seconds=float(sol["seconds"]))
    log(f"scenario {name}: edges {d['n_edges']} km {d['km']:.2f} f1 {d['f1']:.4f} f2 {d['f2']:.2f} "
        f"new units {d['n_new']} conn -{d['gain_conn_pp']:.2f}pp")
    return d

scen = {}
scen["baseline"] = dict(name="baseline", n_edges=0, km=0.0, f1=f1_base, f2=base["f2"], served_pop=base["served_pop"],
                        n_new=0, gain_conn_pp=0.0, open_depots=base["open_depots"], optimal=base["optimal"])
short = [e for e in front0 if L[e] <= 300]
scen["short_300m"] = scenario("short_300m", short)
scen["all_frontier"] = scenario("all_frontier", front0)

# ---------------------------------------------------------------- greedy budget curve
restored = np.zeros(nE, bool)
reach_cur = reach0.copy()
nr_cur = nr0.copy()
steps = []
cum = 0.0
for step in range(1, 21):
    cut2 = inp["cut"] & ~restored
    fr = [int(e) for e in np.where(cut2)[0] if bool(nr_cur[eu[e]]) != bool(nr_cur[ev[e]])]
    log(f"greedy step {step}: candidates {len(fr)}")
    best = None
    for e in fr:
        cut3 = cut2.copy()
        cut3[e] = False
        tt3, nr3 = state_for(cut3)
        r3 = (tt3 <= tmax).any(0)
        new = r3 & ~reach_cur
        g = float(wd[new].sum() / Wd)
        if g <= 1e-10:
            continue
        km = float(L[e] / 1000)
        score = g / max(km, 1e-9)
        if best is None or score > best[0]:
            best = (score, e, g, new, tt3, nr3, km)
    if best is None:
        log("greedy: no remaining segment adds reachable demand")
        break
    score, e, g, new, tt3, nr3, km = best
    restored[e] = True
    cum += km
    reach_cur = (tt3 <= tmax).any(0)
    nr_cur = nr3
    sol = P5.solve(dict(inp, tt=tt3))
    steps.append(dict(step=step, edge=int(e), len_m=float(L[e]), speed_kmh=float(SPD[e]),
                      km_cum=float(cum), gain_conn_pp=float(100 * g), n_new=int(new.sum()),
                      new_pop=float(dem["P"].values[new].sum()),
                      uids=[int(dem["uid"].iloc[j]) for j in np.where(new)[0]],
                      f1=float(sol["f1"]), f2=float(sol["f2"]), served_pop=float(sol["served_pop"]),
                      open_depots=sol["open_depots"], optimal=bool(sol["optimal"]),
                      seconds=float(sol["seconds"])))
    log(f"  picked e{e} {L[e]:.0f}m cum {cum:.2f}km conn -{100 * g:.2f}pp | MILP f1 {sol['f1']:.4f} "
        f"f2 {sol['f2']:.2f} new {int(new.sum())}u")
    if cum > 50:
        break

scen["greedy_5km"] = next((dict(name="greedy_5km", n_edges=s["step"], km=s["km_cum"], f1=s["f1"], f2=s["f2"],
                                served_pop=s["served_pop"], optimal=s["optimal"], n_new=None, gain_conn_pp=None,
                                open_depots=s["open_depots"]) for s in reversed(steps) if s["km_cum"] <= 5.0), None)

rem_uids = {int(dem["uid"].iloc[j]) for j in np.where(~reach_cur)[0]}
rem = [u for u in units if u["uid"] in rem_uids]
rem_share = float(sum(u["wd_share"] for u in rem))
log(f"after greedy {cum:.2f} km: remaining {len(rem)} units, weighted share {rem_share:.2f}%; "
    f"largest remaining " + ", ".join(f"{u['name']}({u['wd_share']:.2f}%,k={u['k']})" for u in rem[:3]))

# ---------------------------------------------------------------- write
top10 = sum(r["wd_share"] for r in units[:10])
top20 = sum(r["wd_share"] for r in units[:20])
kdist = {}
for r in units:
    key = "k>=4" if (r["k"] is not None and r["k"] >= 4) else ("blocked_k0" if r["k"] == 0 else f"k={r['k']}")
    kdist.setdefault(key, dict(n=0, share=0.0, pop=0.0))
    kdist[key]["n"] += 1
    kdist[key]["share"] += r["wd_share"]
    kdist[key]["pop"] += r["P"]

res = dict(
    meta=U.run_meta(), purpose="Q5 emergency road repair: reachability and MILP response",
    params=p, date=C.MAIN_DATE,
    baseline=dict(f1=float(f1_base), f2=float(base["f2"]), served_pop=float(base["served_pop"]),
                  open_depots=base["open_depots"], optimal=bool(base["optimal"]),
                  unreachable_units=int((~reach0).sum()), reachable_units=int(reach0.sum())),
    cut=dict(n_edges=int(inp["cut"].sum()), km=float(L[inp["cut"]].sum() / 1000),
             n_components=int(len(set(lab.tolist()))), n_depot_components=len(depot_comps),
             frontier_edges=len(front0), frontier_km=float(sum(L[e] for e in front0) / 1000)),
    k_distribution=kdist, top10_share=float(top10), top20_share=float(top20),
    units=units, single_edge=single, scenarios=scen, greedy_steps=steps,
    remaining_after_greedy=dict(km=float(cum), n=len(rem), share=float(rem_share), units=rem[:20],
                                f1=float(steps[-1]["f1"]) if steps else None),
)

single_map = {r["edge"]: r for r in single}
step_map = {s["edge"]: s["step"] for s in steps}
recs = []
geom = []
for e in np.where(inp["cut"])[0]:
    e = int(e)
    recs.append(dict(edge=e, length_m=float(L[e]), speed_kmh=float(SPD[e]), oneway=bool(net.oneway[e]),
                     comp_u=int(lab[eu[e]]), comp_v=int(lab[ev[e]]),
                     k_u=kc.get(int(lab[eu[e]])), k_v=kc.get(int(lab[ev[e]])),
                     frontier=int(e in single_map),
                     single_gain_pp=float(single_map.get(e, {}).get("gain_conn_pp", 0.0)),
                     single_new_units=int(single_map.get(e, {}).get("n_new", 0)),
                     greedy_step=int(step_map.get(e, 0))))
    a, b, c_, d_ = net.segs[e]
    geom.append(sg.LineString([(a, b), (c_, d_)]))
gdf = gpd.GeoDataFrame(recs, geometry=geom, crs=crs)
gdf.to_crs(4326).to_file(C.OUT / "q5_repair_edges.geojson", driver="GeoJSON")
U.save_json(res, C.OUT / "q5_repair.json")

log("---- summary ----")
log(f"cut {inp['cut'].sum()} edges / {L[inp['cut']].sum() / 1000:.1f} km; components {len(set(lab.tolist()))}; "
    f"depot comps {len(depot_comps)}")
log("k: " + ", ".join(f"{k}:{v['n']}u/{v['share']:.2f}%" for k, v in sorted(kdist.items())))
for s in steps:
    log(f"  step {s['step']}: e{s['edge']} {s['len_m']:.0f}m cum {s['km_cum']:.2f}km -> f1 {s['f1']:.4f} "
        f"f2 {s['f2']:.2f} (conn -{s['gain_conn_pp']:.2f}pp)")
log(f"short_300m: {scen['short_300m']['n_edges']} edges {scen['short_300m']['km']:.2f}km -> f1 {scen['short_300m']['f1']:.4f}")
log(f"all_frontier: {scen['all_frontier']['n_edges']} edges {scen['all_frontier']['km']:.2f}km -> f1 {scen['all_frontier']['f1']:.4f}")
log(f"wrote {C.OUT / 'q5_repair.json'} and {C.OUT / 'q5_repair_edges.geojson'}")
