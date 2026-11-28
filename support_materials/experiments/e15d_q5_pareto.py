"""E15d - Pareto scan of the repaired network (greedy + targeted union).

For eps on a grid from the union f1* to the published baseline f1, solve
min f2 s.t. f1 <= eps on the repaired network. Shows whether repair improves the
bi-objective trade-off (lower f1 at the same f2), not just the lexicographic endpoint.
Adds targeted.greedy_plus_targets.pareto to output/q5_repair.json.
"""
import os, sys, json, time, types

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
    print(f"[e15d {time.time() - T0:6.1f}s] {m}", flush=True)

res = json.load(open(C.OUT / "q5_repair.json", encoding="utf-8"))
union = [int(e) for e in res["targeted"]["greedy_plus_targets"]["edges"]]
f1u = float(res["targeted"]["greedy_plus_targets"]["f1"])
f1b = float(res["baseline"]["f1"])

with rasterio.open(C.OUT / "q3_newflood_0716.tif") as d:
    crs = d.crs
net = RoadNet(crs)
fac = facility_proxies(crs)
p = P5.base_params()
inp = P5.build_inputs(C.MAIN_DATE, p, net, fac)
cn = inp["cand"]["node"].values.astype(int)
dn = inp["dem"]["node"].values.astype(int)
acc = C.ACCESS_KMH * 1000 / 60
ca = inp["cand"]["access_m"].values / acc
da = inp["dem"]["access_m"].values / acc
adm = inp["cand"]["admissible"].values.astype(bool)
dr = inp["dem"]["reach"].values.astype(bool)

cut2 = inp["cut"].copy()
cut2[np.asarray(union, dtype=int)] = False
G2, _ = net.graph(cut2)
D = net.times_from(G2, cn)
tt = ca[:, None] + D[:, dn] + da[None, :]
tt[:, ~dr] = np.inf
tt[~adm, :] = np.inf

pts = []
for k, eps in enumerate(np.linspace(f1u, f1b, 10)):
    s = P5.solve(dict(inp, tt=tt), stage="eps", eps=float(eps) + (1e-7 if k == 0 else 0))
    pts.append(dict(eps=float(eps), f1=float(s["f1"]), f2=float(s["f2"]), optimal=bool(s["optimal"]),
                    n_open=int(sum(s["eta"])), served_pop=float(s["served_pop"])))
    log(f"eps {eps:.4f} -> f1 {s['f1']:.4f} f2 {s['f2']:.2f} open {sum(s['eta'])}")

res["targeted"]["greedy_plus_targets"]["pareto"] = pts
U.save_json(res, C.OUT / "q5_repair.json")
log("written")
