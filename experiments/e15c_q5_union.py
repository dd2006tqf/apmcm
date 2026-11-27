"""E15c - union of the 18-step greedy program and the targeted paths (heaviest 3 units).

Reads output/q5_repair.json (greedy_steps edges + targeted.* edges), restores the union,
recomputes Dijkstra times and re-solves the same lexicographic MILP. Adds
targeted.greedy_plus_targets to output/q5_repair.json.
"""
import os, sys, json, time, types
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.modules.setdefault("lightgbm", types.ModuleType("lightgbm"))

import numpy as np
import rasterio
import config as C
import common as U
import problem5 as P5
from network import RoadNet
from problem4 import facility_proxies

T0 = time.time()
def log(m):
    print(f"[e15c {time.time() - T0:6.1f}s] {m}", flush=True)

res = json.load(open(C.OUT / "q5_repair.json", encoding="utf-8"))
greedy_edges = [int(s["edge"]) for s in res["greedy_steps"]]
target_edges = []
for k, v in res["targeted"].items():
    if isinstance(v, dict) and v.get("edges"):
        target_edges += [int(e) for e in v["edges"]]
union = sorted(set(greedy_edges) | set(target_edges))
log(f"greedy {len(greedy_edges)} edges; targeted {len(set(target_edges))} edges; union {len(union)} edges")

with rasterio.open(C.OUT / "q3_newflood_0716.tif") as d:
    crs = d.crs
net = RoadNet(crs)
fac = facility_proxies(crs)
p = P5.base_params()
inp = P5.build_inputs(C.MAIN_DATE, p, net, fac)
dem, cand = inp["dem"], inp["cand"]
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

cut2 = inp["cut"].copy()
cut2[np.asarray(union, dtype=int)] = False
G2, _ = net.graph(cut2)
D = net.times_from(G2, cn)
tt = ca[:, None] + D[:, dn] + da[None, :]
tt[:, ~dr] = np.inf
tt[~adm, :] = np.inf
sol = P5.solve(dict(inp, tt=tt))
r = (tt <= tmax).any(0)
new = r & ~reach0
n_rest = int(len(dem) - r.sum())

d = dict(name="greedy_plus_targets", n_edges=len(union),
         km=float(sum(net.len_m[e] for e in union) / 1000), edges=union,
         f1=float(sol["f1"]), f2=float(sol["f2"]), optimal=bool(sol["optimal"]),
         served_pop=float(sol["served_pop"]), n_new=int(new.sum()),
         new_pop=float(dem["P"].values[new].sum()),
         gain_conn_pp=float(100 * wd[new].sum() / Wd),
         uids=[int(dem["uid"].iloc[j]) for j in np.where(new)[0]],
         n_remaining=n_rest,
         remaining_share=float(100 * wd[~r].sum() / Wd),
         open_depots=sol["open_depots"])
log(f"union: {d['n_edges']} edges {d['km']:.2f} km -> f1 {d['f1']:.4f} f2 {d['f2']:.2f} "
    f"new {d['n_new']}u remaining {n_rest}u/{d['remaining_share']:.2f}%")

res["targeted"]["greedy_plus_targets"] = d
U.save_json(res, C.OUT / "q5_repair.json")
log("written")
