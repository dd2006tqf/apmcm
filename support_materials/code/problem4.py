"""Problem 4 — regional flood disaster assessment of Hengzhou (MODELING_REPORT.md §7).

Units = Voronoi cells of the 393 OSM places clipped to the boundary (aggregated to 17 town/city zones).
Indicators are overlays of the Q3 new-inundation mask F with public layers:
 hazard H1 A_new, H2 rho, H3 expected grade (Q2 thresholds + MC (5.9));
 exposure E1 flooded population (WorldPop), E2 flooded built-up, E3 flooded cropland (WorldCover),
 E4 exposed facility proxies; response R1 access time on the flood-cut network, R2 road-cut share.
Weights: entropy + CRITIC game-theory combination inside criteria, AHP between criteria; TOPSIS index;
Jenks levels; Dirichlet weight robustness, hazard-only ranking, alternative F inputs.
"""
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry import MultiPoint, Point, box
from shapely.ops import voronoi_diagram, unary_union
from pyproj import Geod, Transformer
from scipy.stats import spearmanr
import config as C
import common as U
import problem2 as P2
from network import RoadNet

CQ4 = C.CACHE / "q4"
CQ4.mkdir(parents=True, exist_ok=True)
GEOD = Geod(ellps="WGS84")
HAZ, EXP, RES = ["H1", "H2", "H3"], ["E1", "E2", "E3", "E4"], ["R1", "R2"]
IND = HAZ + EXP + RES


def read_F(name):
    with rasterio.open(C.OUT / name) as d:
        return d.read(1).astype(bool), d.transform, d.crs


# ============================================================ units (7.1)
def build_units(crs):
    bnd = gpd.read_file(C.PUB / "Boundary" / "Hengzhou_boundary.geojson")
    bu = bnd.to_crs(crs).geometry.union_all() if hasattr(bnd.geometry, "union_all") else unary_union(bnd.to_crs(crs).geometry)
    pl = gpd.read_file(C.PUB / "OSM" / "places.geojson").to_crs(crs)
    pl = pl[pl.geometry.within(bu.buffer(1))].reset_index(drop=True)
    keep, pts = [], []
    for i, g in enumerate(pl.geometry):            # merge duplicates within 50 m
        if all(g.distance(p) > 50 for p in pts):
            keep.append(i); pts.append(g)
    pl = pl.iloc[keep].reset_index(drop=True)
    vor = voronoi_diagram(MultiPoint(list(pl.geometry)), envelope=box(*bu.buffer(5000).bounds))
    cells = []
    for g in pl.geometry:
        cell = [c for c in vor.geoms if c.contains(g)][0]
        cells.append(cell.intersection(bu))
    units = gpd.GeoDataFrame(dict(uid=np.arange(1, len(pl) + 1), name=pl["name"].fillna("").values,
                                  place=pl["place"].values), geometry=cells, crs=crs)
    units["px"] = [g.x for g in pl.geometry]
    units["py"] = [g.y for g in pl.geometry]
    g4 = units.to_crs(4326)
    units["A_reg_km2"] = [abs(GEOD.geometry_area_perimeter(g)[0]) * 1e-6 for g in g4.geometry]
    # zone = nearest town/city node
    tc = pl[pl["place"].isin(["town", "city"])].reset_index(drop=True)
    tcx = np.c_[[g.x for g in tc.geometry], [g.y for g in tc.geometry]]
    d = np.hypot(units["px"].values[:, None] - tcx[None, :, 0], units["py"].values[:, None] - tcx[None, :, 1])
    units["zone"] = [str(tc["name"].iloc[j]) for j in d.argmin(1)]
    a_bnd = abs(sum(GEOD.geometry_area_perimeter(g)[0] for g in bnd.geometry)) * 1e-6
    return units, tc, bu, a_bnd


def facility_proxies(crs):
    """A12: town/city nodes (township health centres) + OSM schools (5 POIs)."""
    pl = gpd.read_file(C.PUB / "OSM" / "places.geojson").to_crs(crs)
    tc = pl[pl["place"].isin(["town", "city"])]
    am = gpd.read_file(C.PUB / "OSM" / "amenities.geojson").to_crs(crs)
    rows = [dict(name=str(n), kind=str(k), x=g.x, y=g.y) for n, k, g in zip(tc["name"], tc["place"], tc.geometry)]
    rows += [dict(name=str(n) if isinstance(n, str) and n else f"school_{i + 1}", kind="school",
                  x=g.centroid.x, y=g.centroid.y) for i, (n, g) in enumerate(zip(am["name"], am.geometry))]
    df = pd.DataFrame(rows)
    assert df["name"].is_unique
    return df
# ============================================================ overlay indicators (7.2)
def unit_raster(units, shape, T):
    f = CQ4 / "units.i2"
    if f.exists():
        return np.memmap(f, dtype=np.int16, mode="r", shape=shape)
    a = features.rasterize([(g, int(u)) for g, u in zip(units.geometry, units["uid"])], out_shape=shape,
                           transform=T, dtype=np.int16)
    m = np.memmap(f, dtype=np.int16, mode="w+", shape=shape)
    m[:] = a
    m.flush()
    return np.memmap(f, dtype=np.int16, mode="r", shape=shape)


def near_F(F, T, x, y, radius_m):
    """True if any F pixel lies within radius_m of point (x, y)."""
    res = T.a
    rad = int(np.ceil(radius_m / res))
    c, r = int((x - T.c) / T.a), int((y - T.f) / T.e)
    r0, r1, c0, c1 = max(0, r - rad), min(F.shape[0], r + rad + 1), max(0, c - rad), min(F.shape[1], c + rad + 1)
    if r0 >= r1 or c0 >= c1:
        return False
    win = F[r0:r1, c0:c1]
    if not win.any():
        return False
    rr, cc = np.nonzero(win)
    d = np.hypot((rr + r0 - r) * res, (cc + c0 - c) * res)
    return bool((d <= radius_m).any())


def indicators(F, T, crs, units, U_r, q3_tiles, net, fac, theta, boot, resid, tag, road_buffer=C.ROAD_BUFFER_M):
    t0 = time.time()
    nU = len(units) + 1
    q3 = json.load(open(C.OUT / "q3_results.json", encoding="utf-8"))
    pix_area = json.load(open(C.CACHE / "q3" / "layers.json"))["pixel_area_m2"]
    rows, cols = np.nonzero(F)
    uid = np.asarray(U_r[rows, cols]).astype(int)
    inb = uid > 0
    rows, cols, uid = rows[inb], cols[inb], uid[inb]
    tr0, tc0 = (rows // 256) * 256, (cols // 256) * 256
    a = np.array([pix_area[f"{r}_{c}"] for r, c in zip(tr0, tc0)]) * 1e-6 if len(rows) else np.zeros(0)
    wc = np.memmap(C.CACHE / "q3" / "wc.u1", dtype=np.uint8, mode="r", shape=F.shape)
    cls = np.asarray(wc[rows, cols])
    H1 = np.bincount(uid, weights=a, minlength=nU)
    E2 = np.bincount(uid, weights=a * (cls == 50), minlength=nU)
    E3 = np.bincount(uid, weights=a * (cls == 40), minlength=nU)
    # flooded population: pop_c * (flooded area in c / cell area)
    xs, ys = T.c + (cols + 0.5) * T.a, T.f + (rows + 0.5) * T.e
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(xs, ys)
    with rasterio.open(C.PUB / "WorldPop" / "WorldPop_Hengzhou_2020_100m.tif") as wp:
        pop = wp.read(1).astype(np.float64)
        pop[(pop < 0) | ~np.isfinite(pop)] = 0
        WT = wp.transform
    pr = np.floor((lat - WT.f) / WT.e).astype(int); pc = np.floor((lon - WT.c) / WT.a).astype(int)
    ok = (pr >= 0) & (pr < pop.shape[0]) & (pc >= 0) & (pc < pop.shape[1])
    lat_r = WT.f + (np.arange(pop.shape[0]) + 0.5) * WT.e
    cell_area = np.array([abs(GEOD.polygon_area_perimeter([0, WT.a, WT.a, 0], [la - WT.e / 2, la - WT.e / 2, la + WT.e / 2, la + WT.e / 2])[0])
                          for la in lat_r]) * 1e-6
    contrib = np.zeros(len(rows))
    contrib[ok] = pop[pr[ok], pc[ok]] * a[ok] / cell_area[pr[ok]] * C.POP_SCALE
    E1 = np.bincount(uid, weights=contrib, minlength=nU)
    # flooded-population centroid (area centroid if no population; place node if no flood)
    wsum = np.bincount(uid, weights=contrib, minlength=nU)
    cx = np.bincount(uid, weights=contrib * xs, minlength=nU) / np.maximum(wsum, 1e-12)
    cy = np.bincount(uid, weights=contrib * ys, minlength=nU) / np.maximum(wsum, 1e-12)
    ax = np.bincount(uid, weights=a * xs, minlength=nU) / np.maximum(H1, 1e-12)
    ay = np.bincount(uid, weights=a * ys, minlength=nU) / np.maximum(H1, 1e-12)
    cx = np.where(wsum > 0, cx, np.where(H1 > 0, ax, 0)); cy = np.where(wsum > 0, cy, np.where(H1 > 0, ay, 0))
    cx[1:] = np.where(cx[1:] == 0, units["px"].values, cx[1:]); cy[1:] = np.where(cy[1:] == 0, units["py"].values, cy[1:])
    # H3: sigma'_j allocated from Q3 tiles by the tile's new-area share inside the unit
    tmap = {(t["r"], t["c"]): t for t in q3_tiles}
    ratio = np.array([tmap[(r, c)]["sigma_prime"] / tmap[(r, c)]["A_hat"] if tmap.get((r, c), {}).get("A_hat", 0) > 0 else 0
                      for r, c in zip(tr0, tc0)]) if len(rows) else np.zeros(0)
    sigp = np.bincount(uid, weights=a * ratio, minlength=nU)
    Areg = np.r_[1.0, units["A_reg_km2"].values]
    affected = H1 >= C.AFFECTED_MIN_KM2
    pi = np.zeros((nU, 4)); pi[:, 0] = 1
    jj = np.where(affected)[0]
    if len(jj):
        pi[jj] = P2.mc_grades(H1[jj], np.maximum(sigp[jj], 1e-9), Areg[jj], resid, boot)
    H3 = pi @ np.array([1, 2, 3, 4])
    # E4 facility proxies within 500 m of F
    E4 = np.zeros(nU)
    fac = fac.copy()
    fac["uid"] = [int(U_r[int((y - T.f) / T.e), int((x - T.c) / T.a)]) for x, y in zip(fac.x, fac.y)]
    fac["exposed"] = [near_F(F, T, x, y, C.FACILITY_EXPOSE_M) for x, y in zip(fac.x, fac.y)]
    fac["flooded100"] = [near_F(F, T, x, y, C.DEPOT_FLOOD_CLEAR_M) for x, y in zip(fac.x, fac.y)]
    for u_, e in zip(fac["uid"], fac["exposed"]):
        if u_ > 0 and e:
            E4[u_] += 1
    # response: flood-cut network
    cut = net.cut_mask(F, T, buffer_m=road_buffer)
    G, active = net.graph(cut)
    mid = (net.segs[:, :2] + net.segs[:, 2:]) / 2
    er = np.clip(((mid[:, 1] - T.f) / T.e).astype(int), 0, F.shape[0] - 1)
    ec = np.clip(((mid[:, 0] - T.c) / T.a).astype(int), 0, F.shape[1] - 1)
    eu = np.asarray(U_r[er, ec]).astype(int)
    Ltot = np.bincount(eu, weights=net.len_m, minlength=nU)
    Lcut = np.bincount(eu, weights=net.len_m * cut, minlength=nU)
    R2 = np.where(Ltot > 0, Lcut / np.maximum(Ltot, 1e-9), 0.0)
    tg = fac[~fac["flooded100"]]
    tnode, tdist = net.snap(np.c_[tg.x, tg.y], active)
    dist = net.times_from(G, tnode, reverse=True)                      # time from any node to target k
    acc_t = tdist / (C.ACCESS_KMH * 1000 / 60)
    best = (dist + acc_t[:, None]).min(0)
    dnode, ddist = net.snap(np.c_[cx[1:], cy[1:]], active)
    R1 = np.r_[0.0, np.where(ddist <= C.ACCESS_MAX_KM * 1000, best[dnode] + ddist / (C.ACCESS_KMH * 1000 / 60), np.inf)]
    fin = np.isfinite(R1[1:][affected[1:]])
    mx = R1[1:][affected[1:]][fin].max() if fin.any() else 0
    R1 = np.where(np.isfinite(R1), R1, 2 * mx)
    df = pd.DataFrame(dict(uid=np.arange(nU), H1=H1, H2=H1 / Areg, H3=H3, E1=E1, E2=E2, E3=E3, E4=E4, R1=R1, R2=R2,
                           sigma_prime=sigp, cx=cx, cy=cy, affected=affected, demand_node=np.r_[-1, dnode],
                           access_m=np.r_[0, ddist])).iloc[1:].reset_index(drop=True)
    for g in range(4):
        df[f"pi{g + 1}"] = pi[1:, g]
    print(f"[Q4] indicators {tag}: affected={int(affected[1:].sum())} flooded pop={E1.sum():.0f} "
          f"cut roads {net.len_m[cut].sum() / 1000:.1f} km ({time.time() - t0:.0f}s)", flush=True)
    extra = dict(H1_total=float(H1[1:].sum()), H1_outside=float(H1[0]), cut_km=float(net.len_m[cut].sum() / 1000),
                 facilities=fac.to_dict("records"), n_targets=int(len(tg)))
    return df, extra
# ============================================================ weights (7.1)-(7.4), TOPSIS (7.5)
def minmax(X):
    lo, hi = X.min(0), X.max(0)
    return np.where(hi > lo, (X - lo) / np.where(hi > lo, hi - lo, 1), 0.0)


def entropy_w(R):
    n = R.shape[0]
    P = R / np.where(R.sum(0) > 0, R.sum(0), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -np.nansum(np.where(P > 0, P * np.log(P), 0), 0) / np.log(n)
    d = 1 - e
    return d / d.sum()


def critic_w(R):
    s = R.std(0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cr = np.corrcoef(R, rowvar=False)
    cr = np.nan_to_num(np.atleast_2d(cr), nan=0.0)
    c = s * (1 - cr).sum(1)
    return c / c.sum() if c.sum() > 0 else np.full(R.shape[1], 1 / R.shape[1])


def game_combine(we, wc):
    A = np.array([[we @ we, we @ wc], [wc @ we, wc @ wc]])
    b = np.array([we @ we, wc @ wc])
    try:
        al = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        al = np.array([0.5, 0.5])
    w = (abs(al[0]) * we + abs(al[1]) * wc) / (abs(al[0]) + abs(al[1]))
    return w / w.sum(), al


def ahp():
    J = np.array([[1, 2, 3], [1 / 2, 1, 2], [1 / 3, 1 / 2, 1]])
    vals, vecs = np.linalg.eig(J)
    k = np.argmax(vals.real)
    w = np.abs(vecs[:, k].real); w /= w.sum()
    lam = vals[k].real
    CI = (lam - 3) / 2
    return w, lam, CI / 0.58


def topsis(R, w):
    V = R * w
    vp, vm = V.max(0), V.min(0)
    dp = np.sqrt(((V - vp) ** 2).sum(1)); dm = np.sqrt(((V - vm) ** 2).sum(1))
    return np.where(dp + dm > 0, dm / (dp + dm), 0.0)


def score(df, crit_sets=((HAZ, 0), (EXP, 1), (RES, 2)), Omega=None, wcrit=None):
    """Returns S for all units (0 for non-affected), weights and diagnostics."""
    aff = df["affected"].values
    X = df.loc[aff, IND].values.astype(float)
    R = minmax(X)
    Om, lam, CR = ahp()
    if Omega is not None:
        Om = Omega
    w = np.zeros(len(IND)); diag = {}
    for names, ci in crit_sets:
        if not names:
            continue
        j = [IND.index(n) for n in names]
        if wcrit is not None:
            wc_ = wcrit[ci]
        else:
            we, wcr = entropy_w(R[:, j]), critic_w(R[:, j])
            wc_, al = game_combine(we, wcr)
            diag[ci] = dict(entropy=we, critic=wcr, game=wc_, alpha=al)
        w[j] = Om[ci] * wc_
    S = np.zeros(len(df))
    S[aff] = topsis(R, w)
    return S, w, dict(Omega=Om, lambda_max=lam, CR=CR, criteria=diag), R


def ranks(S):
    order = np.argsort(-S, kind="stable")
    r = np.empty(len(S), int); r[order] = np.arange(1, len(S) + 1)
    return r


def zone_table(df, units, w):
    rows = []
    for z, g in df.groupby(units["zone"].values):
        H1 = g.H1.sum(); A = units.loc[g.index, "A_reg_km2"].sum()
        rows.append(dict(zone=z, n_units=len(g), A_reg=A, H1=H1, H2=H1 / A, H3=(g.H3 * g.H1).sum() / H1 if H1 > 0 else 1.0,
                         E1=g.E1.sum(), E2=g.E2.sum(), E3=g.E3.sum(), E4=g.E4.sum(),
                         R1=(g.R1 * g.E1).sum() / g.E1.sum() if g.E1.sum() > 0 else g.R1.mean(),
                         R2=g.R2.mean(), n_affected=int(g.affected.sum())))
    zt = pd.DataFrame(rows)
    zt["affected"] = zt.H1 >= C.AFFECTED_MIN_KM2
    R = minmax(zt.loc[zt.affected, IND].values.astype(float))
    zt["S"] = 0.0
    zt.loc[zt.affected, "S"] = topsis(R, w)
    zt["rank"] = ranks(zt.S.values)
    return zt.sort_values("rank")
# ============================================================ main
def levels_jenks(S, aff):
    lv = np.zeros(len(S), int)       # 0 = not affected; 1 = Level I (most severe) .. 4 = Level IV
    s = S[aff]
    if len(np.unique(s)) >= 4:
        cuts = P2.jenks(s, 4)
        cls = np.searchsorted(cuts, s, side="left")       # 0..3 low->high
        lv[aff] = 4 - cls
    return lv


def run_alt(Fname, T, crs, units, U_r, tiles, net, fac, theta, boot, resid, tag):
    F, _, _ = read_F(Fname)
    df, ex = indicators(F, T, crs, units, U_r, tiles, net, fac, theta, boot, resid, tag)
    del F
    S, w, d, _ = score(df)
    return df, S, ex


def validate_capability(res, units, df, w, a_q3):
    assert res["partition_rel_error"] < 0.01, "[P4-C2] units do not partition the boundary"
    assert res["A_new_sum_rel_error"] <= 0.005, "[Q3->Q4] unit new areas do not sum to Q3 A_new"
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-9, "[P4-C3] weights"
    assert res["ahp"]["Omega"][0] >= res["ahp"]["Omega"][1] and res["ahp"]["CR"] < 0.1, "[P4-C3] AHP"
    assert df["S"].between(0, 1).all()
    assert len(res["exposure_layers"]) >= 4, "[P4-C1]"
    print("[Q4] validate_capability PASS")


def main():
    U.set_all_seeds()
    t0 = time.time()
    q2 = json.load(open(C.OUT / "q2_results.json", encoding="utf-8"))
    q3 = json.load(open(C.OUT / "q3_results.json", encoding="utf-8"))
    theta = np.array(q2["grading"]["theta"]); boot = np.array(q2["grading"]["boot_theta"])
    resid = np.array(q2["conformal"]["resid_pool"])
    F, T, crs = read_F("q3_newflood_0716.tif")
    units, tc, bu, a_bnd = build_units(crs)
    U_r = unit_raster(units, F.shape, T)
    net = RoadNet(crs)
    fac = facility_proxies(crs)
    tiles = q3["dates"][C.MAIN_DATE]["tiles"]
    df, ex = indicators(F, T, crs, units, U_r, tiles, net, fac, theta, boot, resid, "main")
    del F
    S, w, diag, R = score(df)
    df["S"] = S
    df["rank"] = ranks(S)
    df["level"] = levels_jenks(S, df["affected"].values)
    aff = df["affected"].values
    # ---------------- robustness: Dirichlet weights
    rng = np.random.default_rng(C.SEED)
    Om0 = diag["Omega"]; wc0 = [diag["criteria"][c]["game"] for c in range(3)]
    RK = np.zeros((C.MC_GRADE, aff.sum()), int)
    sp = []
    base_r = ranks(S[aff])
    for m in range(C.MC_GRADE):
        Om = rng.dirichlet(50 * Om0)
        wcs = [rng.dirichlet(50 * np.maximum(wc, 1e-6)) for wc in wc0]
        Sm, _, _, _ = score(df, Omega=Om, wcrit=wcs)
        rk = ranks(Sm[aff])
        RK[m] = rk
        sp.append(spearmanr(rk, base_r).statistic)
    top10 = (RK <= 10).mean(0)
    df.loc[aff, "rank_med"] = np.median(RK, 0)
    df.loc[aff, "rank_lo"] = np.percentile(RK, 5, 0)
    df.loc[aff, "rank_hi"] = np.percentile(RK, 95, 0)
    df.loc[aff, "top10_freq"] = top10
    # hazard-only
    Sh, wh, _, _ = score(df, crit_sets=((HAZ, 0), ([], 1), ([], 2)), Omega=np.array([1.0, 0, 0]))
    rh = ranks(Sh[aff])
    top_full = set(np.where(base_r <= 10)[0]); top_h = set(np.where(rh <= 10)[0])
    # alternative inputs
    alts = {}
    for nm, fn, tl in (("F_0704", "q3_newflood_0704.tif", q3["dates"]["20260704"]["tiles"]),
                       ("W0_sar_only", "q3_newflood_0716_sar_only.tif", tiles),
                       ("hstar_30", "q3_newflood_0716_h30.tif", tiles)):
        dfa, Sa, exa = run_alt(fn, T, crs, units, U_r, tl, net, fac, theta, boot, resid, nm)
        if nm == "F_0704":       # Q5 information-update plan uses the 07-04 demand and priority
            dfa.assign(S=Sa)[["uid", "E1", "S", "cx", "cy", "affected", "H1"]].to_csv(C.OUT / "q4_units_0704.csv", index=False)
        both = aff | dfa["affected"].values
        alts[nm] = dict(spearman_all_units=float(spearmanr(S[both], Sa[both]).statistic),
                        n_affected=int(dfa["affected"].sum()), flooded_pop=float(dfa.E1.sum()),
                        top10_overlap=len(set(np.argsort(-S)[:10]) & set(np.argsort(-Sa)[:10])),
                        H1_total=exa["H1_total"])
    zt = zone_table(df, units, w)
    # outputs
    units_out = units.copy()
    for c in df.columns:
        if c != "uid":
            units_out[c] = df[c].values
    units_out.to_crs(4326).to_file(C.OUT / "q4_units.geojson", driver="GeoJSON")
    a_q3 = q3["dates"][C.MAIN_DATE]["A_new"]
    part_err = abs(units["A_reg_km2"].sum() - a_bnd) / a_bnd
    res = dict(meta=U.run_meta(), theta_used=theta.tolist(), n_units=len(units), n_affected=int(aff.sum()), boundary_area_km2=a_bnd,
               units_area_sum_km2=float(units["A_reg_km2"].sum()), partition_rel_error=part_err,
               A_new_units_sum=float(df.H1.sum()), A_new_q3=a_q3,
               A_new_sum_rel_error=abs(df.H1.sum() - a_q3) / a_q3 if a_q3 > 0 else 0.0,
               exposure_layers=[dict(layer="WorldPop 2020 constrained 100 m", indicator="E1"),
                                dict(layer="ESA WorldCover 2021 built-up (50)", indicator="E2"),
                                dict(layer="ESA WorldCover 2021 cropland (40)", indicator="E3"),
                                dict(layer="OSM places (town/city) + schools, A12", indicator="E4"),
                                dict(layer="OSM roads (Geofabrik) cut by F", indicator="R1,R2"),
                                dict(layer="Copernicus DEM via REM guard in F", indicator="hazard")],
               weights=dict(zip(IND, w)), ahp=dict(Omega=diag["Omega"], lambda_max=diag["lambda_max"], CR=diag["CR"]),
               criteria_weights={str(k): v for k, v in diag["criteria"].items()},
               totals=dict(flooded_pop=float(df.E1.sum()), flooded_built_km2=float(df.E2.sum()),
                           flooded_crop_km2=float(df.E3.sum()), exposed_facilities=int(df.E4.sum()),
                           cut_road_km=ex["cut_km"], total_road_km=float(net.len_m.sum() / 1000)),
               level_counts={f"Level {['I', 'II', 'III', 'IV'][l - 1]}": int((df.level == l).sum()) for l in (1, 2, 3, 4)},
               top_units=df[aff].sort_values("rank").head(15)[["uid", "H1", "H2", "H3", "E1", "E2", "E3", "E4", "R1", "R2", "S", "rank", "level"]]
               .assign(name=lambda x: units.loc[x.index, "name"].values, zone=lambda x: units.loc[x.index, "zone"].values).to_dict("records"),
               units=df.assign(name=units["name"].values, zone=units["zone"].values, A_reg=units["A_reg_km2"].values).to_dict("records"),
               zones=zt.to_dict("records"), facilities=ex["facilities"],
               robustness=dict(dirichlet_M=C.MC_GRADE, spearman_median=float(np.median(sp)), spearman_p05=float(np.percentile(sp, 5)),
                               hazard_only_spearman=float(spearmanr(rh, base_r).statistic), hazard_only_top10_overlap=len(top_full & top_h),
                               alternative_inputs=alts),
               seconds=time.time() - t0)
    validate_capability(res, units, df, w, a_q3)
    ok = U.validate_constraints({"S": df.S.values, "w": w, "CR": diag["CR"], "part": part_err},
                                {"S": (0, 1, "S_j"), "w": (0, 1, "weights"), "CR": (0, 0.1, "AHP CR"),
                                 "part": (0, 0.01, "partition error")}, "Q4")
    if not ok:
        raise ValueError("Q4 constraint validation failed")
    U.save_json(res, C.OUT / "q4_results.json")
    U.save_json(res, C.FIG / "problem_4_results.json")
    print(f"[Q4] done in {time.time() - t0:.0f}s; affected units {aff.sum()}, flooded pop {df.E1.sum():.0f}")
    return res


if __name__ == "__main__":
    main()
