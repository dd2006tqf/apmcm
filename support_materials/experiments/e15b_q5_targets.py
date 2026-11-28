"""E15b - targeted repair paths to the three heaviest units left by the greedy curve.

For each target unit, find the minimum-length path in the component graph from any depot
component to its component, restore exactly those cut segments, recompute Dijkstra times and
re-solve the same lexicographic MILP. Also solve the union of the three paths.
Writes the `targeted` block into output/q5_repair.json (keeps every existing key).
"""
import os, sys, json, time, types, heapq
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.modules.setdefault("lightgbm", types.ModuleType("lightgbm"))

import numpy as np
import geopandas as gpd
import rasterio
from scipy.sparse.csgraph import connected_components
import config as C
import common as U
import problem5 as P5
from network import RoadNet
from problem4 import facility_proxies

T0 = time.time()
def log(m):
    print(f"[e15b {time.time() - T0:6.1f}s] {m}", flush=True)

with rasterio.open(C.OUT / "q3_newflood_0716.tif") as d:
    crs = d.crs
net = RoadNet(crs)
fac = facility_proxies(crs)
p = P5.base_params()
inp = P5.build_inputs(C.MAIN_DATE, p, net, fac)
dem, cand = inp["dem"], inp["cand"]
eu, ev, L = net.eu, net.ev, net.len_m
cn = cand["node"].values.astype(int)
dn = dem["node"].values.astype(int)
acc = C.ACCESS_KMH * 1000 / 60
ca = cand["access_m"].values / acc
da = dem["access_m"].values / acc
adm = cand["admissible"].values.astype(bool)
dr = dem["reach"].values.astype(bool)
w, dd = dem["w"].values, dem["d"].values
wd, Wd = w * dd, float((w * dd).sum())
tmax = p["t_max"]
reach0 = (inp["tt"] <= tmax).any(0)

G0, _ = net.graph(inp["cut"])
lab = connected_components(G0 + G0.T, directed=False)[1]
depot_comps = sorted({int(lab[cn[i]]) for i in range(len(cn)) if adm[i]})

# component graph with lengths
adj = {}
for e in np.where(inp["cut"])[0]:
    cu, cv = int(lab[eu[e]]), int(lab[ev[e]])
    if cu != cv:
        adj.setdefault(cu, []).append((cv, int(e), float(L[e])))
        adj.setdefault(cv, []).append((cu, int(e), float(L[e])))

def path_to(comp):
    dist = {c: np.inf for c in adj}
    prev = {}
    pq = []
    for c in depot_comps:
        dist[c] = 0.0
        heapq.heappush(pq, (0.0, c))
    while pq:
        d, c = heapq.heappop(pq)
        if d > dist.get(c, np.inf):
            continue
        for nb, e, lenm in adj.get(c, ()):
            nd = d + lenm
            if nd < dist.get(nb, np.inf):
                dist[nb] = nd
                prev[nb] = (c, e)
                heapq.heappush(pq, (nd, nb))
    if comp not in prev:
        return None, np.inf
    edges, c = [], comp
    while c in prev:
        c0, e = prev[c]
        edges.append(e)
        c = c0
    return edges[::-1], dist[comp]

def tt_of(edges):
    cut2 = inp["cut"].copy()
    if edges:
        cut2[np.asarray(edges, dtype=int)] = False
    G2, _ = net.graph(cut2)
    D = net.times_from(G2, cn)
    tt = ca[:, None] + D[:, dn] + da[None, :]
    tt[:, ~dr] = np.inf
    tt[~adm, :] = np.inf
    return tt

def solve_scn(name, edges):
    tt = tt_of(edges)
    sol = P5.solve(dict(inp, tt=tt))
    r = (tt <= tmax).any(0)
    new = r & ~reach0
    d = dict(name=name, n_edges=len(edges), km=float(sum(L[e] for e in edges) / 1000),
             edges=[int(e) for e in edges],
             f1=float(sol["f1"]), f2=float(sol["f2"]), optimal=bool(sol["optimal"]),
             served_pop=float(sol["served_pop"]), n_new=int(new.sum()),
             new_pop=float(dem["P"].values[new].sum()),
             gain_conn_pp=float(100 * wd[new].sum() / Wd),
             uids=[int(dem["uid"].iloc[j]) for j in np.where(new)[0]],
             open_depots=sol["open_depots"])
    log(f"{name}: {len(edges)} edges {d['km']:.2f} km -> f1 {d['f1']:.4f} f2 {d['f2']:.2f} new {d['n_new']}u "
        f"(-{d['gain_conn_pp']:.2f}pp)")
    return d

q4 = gpd.read_file(C.OUT / "q4_units.geojson")[["uid", "name"]].set_index("uid")
targets = ["马岭镇", "新兴", "杨江"]
out, all_edges = {}, set()
for nm in targets:
    uid = next(int(u) for u in q4.index if str(q4.loc[u, "name"]) == nm)
    j = int(np.where(dem["uid"].values == uid)[0][0])
    comp = int(lab[dn[j]])
    edges, dist = path_to(comp)
    log(f"target {nm} uid {uid} comp {comp} k={len(edges) if edges else None} path_len={dist:.0f} m")
    if not edges:
        out[nm] = dict(name=nm, uid=uid, reachable=False, k=None, km=None)
        continue
    out[nm] = solve_scn(nm, edges)
    out[nm].update(uid=uid, k=len(edges), comp=comp, reachable=True)
    all_edges |= set(edges)

out["combined"] = solve_scn("combined", sorted(all_edges))
out["combined"].update(n_targets=len(targets), reachable=True)

res = json.load(open(C.OUT / "q5_repair.json", encoding="utf-8"))
res["targeted"] = out
U.save_json(res, C.OUT / "q5_repair.json")
log("targeted block written")
