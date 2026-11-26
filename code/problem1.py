"""Problem 1 — preprocessing, new-inundation recognition (physics baseline A + LightGBM baseline B),
spatially blocked validation, pixel/area metrics, FP/FN by land cover, error drivers (TreeSHAP),
and the label-semantics test P1-C7.  MODELING_REPORT.md §3-§4.
"""
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import config as C
import common as U

CQ = C.CACHE / "q1"
CQ.mkdir(parents=True, exist_ok=True)
PRED = C.OUT / "q1_pred"
PRED.mkdir(parents=True, exist_ok=True)
NT, H = 1000, C.TILE
NF = len(U.FEATURES)


def mm(name, dtype, shape, mode="r"):
    return np.memmap(CQ / name, dtype=dtype, mode=mode, shape=shape)


# ============================================================ tiles and folds
def tile_table():
    ids, L, S, O = U.official_index()
    import rasterio
    rows = []
    for t in ids:
        with rasterio.open(S[t]) as d:
            T = d.transform
        rows.append((t, T.c, T.f))
    a = np.array(rows)
    S_ = C.SCALE_DEG * C.TILE
    col = np.round((a[:, 1] - a[:, 1].min()) / S_).astype(int)
    row = np.round((a[:, 2].max() - a[:, 2]) / S_).astype(int)
    return ids, L, S, O, col, row, a[:, 1], a[:, 2]


def clusters_from_grid(col, row, gap=3):
    """Spatial clusters = single-linkage components of tile positions (gap in tile units)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    lab = fcluster(linkage(np.c_[col, row].astype(float), "single"), gap, "distance")
    order = np.argsort(-np.bincount(lab)[1:])           # cluster 0 = largest
    remap = {old + 1: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in lab])


def build_folds(col, row, flood_share, seed=C.FOLD_SEED):
    """Greedy stratified assignment of 5x5-tile blocks to 5 folds, balancing tile count and flood area."""
    B, K = C.BLOCK, C.N_FOLDS
    bid = (col // B) * 10000 + (row // B)
    ub, inv = np.unique(bid, return_inverse=True)
    barea = np.bincount(inv, weights=flood_share)
    bn = np.bincount(inv)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ub))
    order = perm[np.argsort(-barea[perm], kind="stable")]
    fa, fn = np.zeros(K), np.zeros(K)
    fob = np.zeros(len(ub), int)
    for b in order:
        k = min(range(K), key=lambda k: (fn[k] / 200 + fa[k] / flood_share.sum() * 5))
        fob[b] = k
        fa[k] += barea[b]
        fn[k] += bn[b]
    return fob[inv], inv, len(ub)


def fold_sets(fold, col, row):
    """test = k, cal = k+1, train = rest minus Chebyshev-1 buffer around test and cal tiles."""
    out = []
    cells = {}
    for i, (c, r) in enumerate(zip(col, row)):
        cells[(c, r)] = i
    for k in range(C.N_FOLDS):
        te = np.where(fold == k)[0]
        ca = np.where(fold == (k + 1) % C.N_FOLDS)[0]
        held = set(zip(col[te], row[te])) | set(zip(col[ca], row[ca]))
        tr, buf = [], []
        for i in np.where((fold != k) & (fold != (k + 1) % C.N_FOLDS))[0]:
            near = any((col[i] + dc, row[i] + dr) in held for dc in (-1, 0, 1) for dr in (-1, 0, 1))
            (buf if near else tr).append(i)
        out.append(dict(test=te.tolist(), cal=ca.tolist(), train=tr, buffer_removed=buf))
    return out


def check_folds(fs, col, row):
    allt = set()
    for k, f in enumerate(fs):
        te, ca, tr = set(f["test"]), set(f["cal"]), set(f["train"])
        assert te.isdisjoint(tr) and ca.isdisjoint(tr) and te.isdisjoint(ca), k
        held = te | ca
        hc = {(col[i], row[i]) for i in held}
        for i in tr:
            assert not any((col[i] + dc, row[i] + dr) in hc for dc in (-1, 0, 1) for dr in (-1, 0, 1)), \
                f"buffer violated fold {k} tile {i}"
        allt |= te
    assert len(allt) == NT
    return True


# ============================================================ preprocessing (P1-C1, P1-C7)
def preprocess(ids, L, S, O):
    meta_f = CQ / "pre_meta.json"
    if meta_f.exists():
        meta = json.load(open(meta_f))
        if meta.get("n") == NT and meta.get("common_sha") == U.file_sha(os.path.join(_HERE, "common.py")):
            print("[Q1] preprocessing cache valid, reuse")
            return meta
    feats = mm("feats.f16", np.float16, (NT, NF, H, H), "w+")
    Y = mm("y.u8", np.uint8, (NT, H, H), "w+")
    V = mm("valid.u8", np.uint8, (NT, H, H), "w+")
    area_rows = np.zeros((NT, H))
    tiles = []
    # P1-C7 contingency: label x W0-definition x SAR-dark on valid pixels
    defs = ["ndwi>0", "ndwi>0.1", "ndwi>0&ndvi<0"]
    cont = {d: np.zeros((2, 2, 2)) for d in defs}   # [y, w0, dark]
    t0 = time.time()
    for i, t in enumerate(ids):
        r = U.read_official_tile(t, L, S, O)
        y, sar, opt, valid = r["y"], r["sar"], r["opt"], r["valid"]
        m = opt.max(0) > 0
        z = U.rank_normalize(sar, valid)
        zt, enl = U.lee_filter(z, valid)
        f = U.feature_stack(z, zt, opt, m)
        feats[i] = f.astype(np.float16)
        Y[i], V[i] = y, valid
        area_rows[i] = U.pixel_area_rows(r["T"], H)
        tau, ash = U.split_otsu(zt, valid)
        ndwi, ndvi = f[U.FI["ndwi"]], f[U.FI["ndvi"]]
        dark = zt < 0.2
        for d, w0 in zip(defs, [(ndwi > 0) & m, (ndwi > 0.1) & m, (ndwi > 0) & (ndvi < 0) & m]):
            for a in (0, 1):
                for b in (0, 1):
                    for c in (0, 1):
                        cont[d][a, b, c] += np.count_nonzero(valid & ((y == 1) == a) & (w0 == b) & (dark == c))
        yb = y.astype(bool) & valid
        er = ndi.binary_erosion(yb, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool), border_value=0)
        bden = float((yb & ~er).sum() / yb.sum()) if yb.sum() else 0.0
        tiles.append(dict(id=t, enl=enl, otsu_tau=tau, ashman_share=ash, sar_mean=float(sar[valid].mean()),
                          sar_std=float(sar[valid].std()), opt_invalid=float(1 - m.mean()),
                          ndvi_mean=float(ndvi[m].mean()) if m.any() else 0.0,
                          prewater=float(((ndwi > 0) & m).mean()), boundary_density=bden,
                          resampled=r["resampled"], n_valid=int(valid.sum()),
                          flood_share=float(yb.sum() / max(valid.sum(), 1)),
                          area_km2=float((area_rows[i][:, None] * valid).sum() * 1e-6)))
        if i % 100 == 0:
            print(f"[Q1] preprocess {i}/{NT} ({time.time() - t0:.0f}s)", flush=True)
    feats.flush(); Y.flush(); V.flush()
    np.save(CQ / "area_rows.npy", area_rows)
    meta = dict(n=NT, ids=ids, tiles=tiles, contingency={d: cont[d].tolist() for d in defs},
                common_sha=U.file_sha(os.path.join(_HERE, "common.py")), seconds=time.time() - t0)
    json.dump(U.to_jsonable(meta), open(meta_f, "w"))
    return meta


def label_semantics(meta):
    out = {}
    ok = True
    for d, c in meta["contingency"].items():
        c = np.array(c)   # [y, w0, dark]
        lab1 = c[1].sum()
        lab1_in_w0 = c[1, 1].sum() / lab1
        w0_lab0 = c[0, 1].sum()
        dark_share = c[0, 1, 1] / w0_lab0 if w0_lab0 else None
        w0_lab1 = c[1, 1].sum() / c[:, 1].sum()
        passed = (lab1_in_w0 < 0.05) and (dark_share is not None and dark_share > 0.5)
        ok &= passed
        out[d] = dict(label1_share_in_W0=lab1_in_w0, W0_label0_sar_dark_share=dark_share,
                      W0_share_labelled1=w0_lab1, n_valid_px=c.sum(), W0_px=c[:, 1].sum(),
                      table_y_w0_dark=c.tolist(), passed=passed)
    out["accepted_A1"] = ok
    out["n_pixels_total"] = float(np.array(meta["contingency"]["ndwi>0"]).sum())
    return out
# ============================================================ models
def save_mask(model, tid, mask):
    # store 0/255 so the PNG is directly viewable; load_mask() still thresholds at >0
    Image.fromarray((mask.astype(np.uint8)) * 255).save(PRED / f"{model}_{tid}.png")


def load_mask(model, tid):
    return np.array(Image.open(PRED / f"{model}_{tid}.png")) > 0


def baseline_A(ids, meta, fs):
    """Split-based Otsu W1 minus NDWI>0 W0, MMU 10 px. tau_g fallback = median of accepted
    training-tile thresholds of the outer fold (label-free)."""
    feats = mm("feats.f16", np.float16, (NT, NF, H, H))
    V = mm("valid.u8", np.uint8, (NT, H, H))
    taus = np.array([np.nan if t["otsu_tau"] is None else t["otsu_tau"] for t in meta["tiles"]])
    info = []
    for k, f in enumerate(fs):
        tr = np.array(f["train"])
        tau_g = float(np.nanmedian(taus[tr]))
        nfb = 0
        for i in f["test"]:
            tau = taus[i] if np.isfinite(taus[i]) else tau_g
            nfb += not np.isfinite(taus[i])
            zt = feats[i, U.FI["zt"]].astype(np.float32)
            valid = V[i] > 0
            m = feats[i, U.FI["m"]] > 0.5
            w1 = (zt < tau) & valid
            w0 = (feats[i, U.FI["ndwi"]] > 0) & m
            fA = U.mmu_filter(w1 & ~w0, C.MMU_PX_TILE)
            save_mask("A", ids[i], fA)
        info.append(dict(fold=k, tau_g=tau_g, n_fallback=nfb))
    print("[Q1] baseline A done", info)
    return info


def sample_pixels(idx_tiles, per_tile, stratified, rng):
    feats = mm("feats.f16", np.float16, (NT, NF, H, H))
    Y = mm("y.u8", np.uint8, (NT, H, H))
    V = mm("valid.u8", np.uint8, (NT, H, H))
    Xs, ys = [], []
    for i in idx_tiles:
        v = (V[i] > 0).ravel()
        y = Y[i].ravel()
        pos, neg = np.where(v & (y == 1))[0], np.where(v & (y == 0))[0]
        if stratified:
            npos = min(len(pos), per_tile // 2)
            nneg = min(len(neg), per_tile - npos)
            sel = np.concatenate([rng.choice(pos, npos, replace=False) if npos else np.array([], int),
                                  rng.choice(neg, nneg, replace=False) if nneg else np.array([], int)])
        else:
            allv = np.where(v)[0]
            sel = rng.choice(allv, min(per_tile, len(allv)), replace=False)
        X = feats[i].reshape(NF, -1)[:, sel].T.astype(np.float32)
        Xs.append(X)
        ys.append(y[sel])
    return np.concatenate(Xs), np.concatenate(ys)


def f1_threshold(p, y):
    best = (0.5, -1)
    for t in np.arange(0.05, 0.951, 0.01):
        yh = p >= t
        tp = np.sum(yh & (y == 1)); fp = np.sum(yh & (y == 0)); fn = np.sum(~yh & (y == 1))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
        if f1 > best[1]:
            best = (float(round(t, 2)), float(f1))
    return best


def train_lgb(tr_idx, cal_idx, seed):
    rng = np.random.default_rng(seed)
    Xtr, ytr = sample_pixels(tr_idx, C.LGB_PIX_PER_TILE, True, rng)
    Xca, yca = sample_pixels(cal_idx, C.LGB_CAL_PIX_PER_TILE, False, rng)
    clf = lgb.LGBMClassifier(objective="binary", n_jobs=8, verbose=-1, **C.LGB_PARAMS)
    clf.fit(Xtr, ytr, eval_set=[(Xca, yca)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    pca = clf.predict_proba(Xca)[:, 1]
    tau, f1 = f1_threshold(pca, yca)
    return clf, tau, f1, len(ytr), len(yca)


def predict_tiles(clf, idx, tau, ids, model_name, prob_mm=None):
    feats = mm("feats.f16", np.float16, (NT, NF, H, H))
    V = mm("valid.u8", np.uint8, (NT, H, H))
    for i in idx:
        X = feats[i].reshape(NF, -1).T.astype(np.float32)
        p = clf.predict_proba(X)[:, 1].reshape(H, H)
        if prob_mm is not None:
            prob_mm[i] = np.round(p * 255).astype(np.uint8)
        save_mask(model_name, ids[i], (p >= tau) & (V[i] > 0))


def baseline_B(ids, fs):
    sig = dict(params=C.LGB_PARAMS, n=C.LGB_PIX_PER_TILE, ncal=C.LGB_CAL_PIX_PER_TILE,
               feats=json.load(open(CQ / "pre_meta.json"))["common_sha"],
               folds=[sorted(f["test"]) for f in fs])
    bf = CQ / "b_info.json"
    if bf.exists() and all((PRED / f"B_{t}.png").exists() for t in ids):
        cached = json.load(open(bf))
        if cached.get("sig") == U.to_jsonable(sig):
            print("[Q1] LightGBM fold models and masks reused (same features, folds, params)")
            return cached["info"]
    P = mm("prob_m0.u8", np.uint8, (NT, H, H), "w+")
    info = []
    for k, f in enumerate(fs):
        t0 = time.time()
        clf, tau, f1, ntr, nca = train_lgb(f["train"], f["cal"], C.SEED + k)
        predict_tiles(clf, f["test"], tau, ids, "B", P)
        (CQ / f"lgb_fold{k}.txt").write_text(clf.booster_.model_to_string(), encoding="utf-8")
        info.append(dict(fold=k, tau=tau, cal_F1=f1, best_iter=int(clf.best_iteration_ or clf.n_estimators),
                         n_train_px=ntr, n_cal_px=nca, seconds=time.time() - t0))
        print(f"[Q1] LightGBM fold {k}: tau={tau} calF1={f1:.3f} iters={info[-1]['best_iter']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    P.flush()
    json.dump(U.to_jsonable(dict(sig=sig, info=info)), open(bf, "w"))
    return info
# ============================================================ land cover strata (A4)
WC_STRATA = {"Tree": [10], "Shrub/grass": [20, 30], "Cropland": [40], "Built-up": [50],
             "Bare": [60], "Perm. water": [80], "Wetland": [90]}
IDX_STRATA = ["Water (NDWI>0)", "Dense veg.", "Cropland/sparse veg.", "Built/bare", "No optical"]


def load_worldcover():
    f = C.OUT / "q1_worldcover_tiles.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z["wc"]


def strata_maps(i, wc):
    """Returns (wc_strata int map or None, index-rule strata int map) for tile i."""
    feats = mm("feats.f16", np.float16, (NT, NF, H, H))
    ndwi = feats[i, U.FI["ndwi"]].astype(np.float32)
    ndvi = feats[i, U.FI["ndvi"]].astype(np.float32)
    m = feats[i, U.FI["m"]] > 0.5
    idx = np.full((H, H), 4, np.int8)
    idx[m & (ndvi <= 0.15)] = 3
    idx[m & (ndvi > 0.15) & (ndvi <= 0.45)] = 2
    idx[m & (ndvi > 0.45)] = 1
    idx[m & (ndwi > 0)] = 0
    wcs = None
    if wc is not None:
        wcs = np.full((H, H), len(WC_STRATA), np.int8)   # last = other
        for s, (name, codes) in enumerate(WC_STRATA.items()):
            wcs[np.isin(wc[i], codes)] = s
    return wcs, idx


def evaluate(get_mask, ids, fs, wc, name):
    """Pixel + area metrics of a model on held-out tiles, computed from its saved masks."""
    Y = mm("y.u8", np.uint8, (NT, H, H))
    V = mm("valid.u8", np.uint8, (NT, H, H))
    area_rows = np.load(CQ / "area_rows.npy")
    fold_of = np.zeros(NT, int)
    for k, f in enumerate(fs):
        fold_of[f["test"]] = k
    n_wc = len(WC_STRATA) + 1
    lc_wc = np.zeros((n_wc, 3))    # ref area, FP, FN (km2)
    lc_idx = np.zeros((len(IDX_STRATA), 3))
    cms = [dict(tp=0, fp=0, fn=0, tn=0) for _ in range(len(fs))]
    A_ref, A_hat = np.zeros(NT), np.zeros(NT)
    tile_cm = np.zeros((NT, 4))
    for i in range(NT):
        yh = get_mask(i)
        y, v = Y[i] > 0, V[i] > 0
        a = np.broadcast_to(area_rows[i][:, None], (H, H)) * 1e-6
        cm = U.confusion(yh, y, v)
        for key in cm:
            cms[fold_of[i]][key] += cm[key]
        tile_cm[i] = [cm["tp"], cm["fp"], cm["fn"], cm["tn"]]
        A_ref[i] = (a * (y & v)).sum()
        A_hat[i] = (a * (yh & v)).sum()
        fp = a * (yh & ~y & v)
        fn = a * (~yh & y & v)
        wcs, idx = strata_maps(i, wc)
        for arr, table, ns in ((wcs, lc_wc, n_wc), (idx, lc_idx, len(IDX_STRATA))):
            if arr is None:
                continue
            flat = arr.ravel()
            table[:, 0] += np.bincount(flat, weights=(a * (y & v)).ravel(), minlength=ns)
            table[:, 1] += np.bincount(flat, weights=fp.ravel(), minlength=ns)
            table[:, 2] += np.bincount(flat, weights=fn.ravel(), minlength=ns)
    pooled = {k: sum(c[k] for c in cms) for k in cms[0]}
    res = dict(model=name, pooled_pixel=U.pixel_metrics(pooled), pooled_cm=pooled,
               pooled_area=U.area_metrics(A_ref, A_hat),
               per_fold=[dict(fold=k, **U.pixel_metrics(cms[k]),
                              **{f"area_{kk}": vv for kk, vv in
                                 U.area_metrics(A_ref[fold_of == k], A_hat[fold_of == k]).items()})
                         for k in range(len(fs)) if (fold_of == k).any() and len(fs[k]["test"])],
               A_ref_t=A_ref, A_hat_t=A_hat, tile_cm=tile_cm)
    dA = res["pooled_area"]["DeltaA"]

    def table(tab, names):
        rows = []
        for s, nm in enumerate(names):
            ref, fp, fn = tab[s]
            rows.append(dict(stratum=nm, ref_km2=ref, FP_km2=fp, FN_km2=fn, net_km2=fp - fn,
                             RAE=(fp - fn) / ref if ref > 0 else None))
        ident = abs(sum(r["net_km2"] for r in rows) - dA)
        return dict(rows=rows, identity_abs_error=ident,
                    identity_ok=ident <= 1e-9 * max(1, abs(dA)) + 1e-9)
    res["fpfn_index_rules"] = table(lc_idx, IDX_STRATA)
    if wc is not None:
        res["fpfn_worldcover"] = table(lc_wc, list(WC_STRATA) + ["Other"])
    return res
# ============================================================ key factors (P1-C6)
DESC = ["rho_ref", "sar_mean", "sar_std", "ashman_share", "enl", "ndvi_mean", "share_built",
        "share_cropland", "share_tree", "share_prewater", "opt_invalid", "boundary_density"]


def descriptors(meta, wc):
    feats = mm("feats.f16", np.float16, (NT, NF, H, H))
    X = np.zeros((NT, len(DESC)))
    src = "worldcover" if wc is not None else "index_rules"
    for i, t in enumerate(meta["tiles"]):
        if wc is not None:
            w = wc[i]
            sb, sc, st, sw = [(np.isin(w, c)).mean() for c in ([50], [40], [10], [80])]
        else:
            ndvi = feats[i, U.FI["ndvi"]].astype(np.float32)
            m = feats[i, U.FI["m"]] > 0.5
            sb, sc, st = (m & (ndvi <= .15)).mean(), (m & (ndvi > .15) & (ndvi <= .45)).mean(), (m & (ndvi > .45)).mean()
            sw = t["prewater"]
        X[i] = [t["flood_share"], t["sar_mean"], t["sar_std"], t["ashman_share"], t["enl"], t["ndvi_mean"],
                sb, sc, st, sw, t["opt_invalid"], t["boundary_density"]]
    return X, src


def partial_spearman(X, e):
    from scipy.stats import rankdata
    R = np.column_stack([rankdata(c) for c in X.T])
    re = rankdata(e)
    out = []
    for j in range(X.shape[1]):
        Z = np.column_stack([np.ones(len(e)), np.delete(R, j, axis=1)])
        rx = R[:, j] - Z @ np.linalg.lstsq(Z, R[:, j], rcond=None)[0]
        ry = re - Z @ np.linalg.lstsq(Z, re, rcond=None)[0]
        out.append(float(np.corrcoef(rx, ry)[0, 1]))
    return out


def error_drivers(X, e, fold_of):
    """OOF LightGBM regressors (300 trees, depth 3) on signed and absolute tile errors; mean |TreeSHAP|."""
    res = {}
    for target, yv in (("signed", e), ("abs", np.abs(e))):
        shap_vals = np.zeros_like(X)
        pred = np.zeros(len(yv))
        for k in range(C.N_FOLDS):
            te, tr = fold_of == k, fold_of != k
            reg = lgb.LGBMRegressor(n_estimators=300, max_depth=3, num_leaves=8, learning_rate=0.05,
                                    min_child_samples=10, random_state=C.SEED, verbose=-1, n_jobs=4)
            reg.fit(X[tr], yv[tr])
            pred[te] = reg.predict(X[te])
            shap_vals[te] = reg.booster_.predict(X[te], pred_contrib=True)[:, :-1]   # exact TreeSHAP
        imp = np.abs(shap_vals).mean(0)
        order = np.argsort(-imp)
        r2 = 1 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2)
        res[target] = dict(mean_abs_shap={DESC[j]: imp[j] for j in order}, ranking=[DESC[j] for j in order],
                           oof_R2=r2, shap_values=shap_vals, partial_spearman=dict(zip(DESC, partial_spearman(X, yv))))
    res["descriptors"] = DESC
    res["X"] = X
    return res


def loco_stress(ids, cl, b_info, wc):
    """Leave-one-cluster-out stress test of LightGBM (tau = median fold tau, n_iter = median best_iter)."""
    tau = float(np.median([b["tau"] for b in b_info]))
    n_it = int(np.median([b["best_iter"] for b in b_info]))
    feats = mm("feats.f16", np.float16, (NT, NF, H, H))
    V = mm("valid.u8", np.uint8, (NT, H, H))
    masks = {}
    rows = []
    for c in range(4):
        te = np.where(cl == c)[0]
        tr = np.where(cl != c)[0]
        rng = np.random.default_rng(C.SEED + 100 + c)
        Xtr, ytr = sample_pixels(tr, C.LGB_PIX_PER_TILE, True, rng)
        p = dict(C.LGB_PARAMS); p["n_estimators"] = n_it
        clf = lgb.LGBMClassifier(objective="binary", n_jobs=8, verbose=-1, **p).fit(Xtr, ytr)
        for i in te:
            pr = clf.predict_proba(feats[i].reshape(NF, -1).T.astype(np.float32))[:, 1].reshape(H, H)
            masks[i] = (pr >= tau) & (V[i] > 0)
        print(f"[Q1] LOCO cluster {c}: {len(te)} test tiles", flush=True)
    fs_dummy = [dict(test=np.where(cl == c)[0].tolist()) for c in range(4)]
    ev = evaluate(lambda i: masks[i], ids, fs_dummy, wc, "B_LOCO")
    for c in range(4):
        rows.append(dict(cluster=c, n_tiles=int((cl == c).sum()), **ev["per_fold"][c]))
    return dict(tau=tau, n_estimators=n_it, per_cluster=rows, pooled_pixel=ev["pooled_pixel"],
                pooled_area=ev["pooled_area"])
# ============================================================ main
def validate_capability(res, meta, fs):
    assert len(meta["ids"]) == 1000 and meta["ids"][0] == 1001 and meta["ids"][-1] == 2000, "[P1-C1] not all tiles"
    ar = np.load(CQ / "area_rows.npy")
    assert ar.std() > 0.01, "[P1-C1] pixel area is constant (no latitude dependence)"
    for k, f in enumerate(fs):
        assert set(f["train"]).isdisjoint(f["test"]), "[P1-C3] train/test overlap"
    for m in ("A", "B"):
        r = res["models"][m]
        assert np.isfinite(r["pooled_pixel"]["IoU"]) and np.isfinite(r["pooled_area"]["RAE"]), "[P1-C4]"
        assert r["fpfn_index_rules"]["identity_ok"], "[P1-C5] identity (4.4) failed"
        if "fpfn_worldcover" in r:
            assert len(r["fpfn_worldcover"]["rows"]) >= 3 and r["fpfn_worldcover"]["identity_ok"]
    assert len(res["drivers"]["signed"]["ranking"]) == len(DESC), "[P1-C6]"
    assert res["label_semantics"]["n_pixels_total"] > 6.0e7, "[P1-C7] not computed on all pixels"
    print("[Q1] validate_capability PASS")


def main():
    U.set_all_seeds()
    t0 = time.time()
    ids, L, S, O, col, row, lon, lat = tile_table()
    meta = preprocess(ids, L, S, O)
    flood = np.array([t["flood_share"] for t in meta["tiles"]])
    fold, block_of, nblocks = build_folds(col, row, flood)
    fs = fold_sets(fold, col, row)
    check_folds(fs, col, row)
    shares = [float(flood[fold == k].mean()) for k in range(C.N_FOLDS)]
    assert max(shares) - min(shares) <= 0.05, shares
    cl = clusters_from_grid(col, row)
    U.save_json(dict(ids=ids, fold=fold, block=block_of, col=col, row=row, cluster=cl,
                     folds=[dict(k=k, test=[ids[i] for i in f["test"]], cal=[ids[i] for i in f["cal"]],
                                 train=[ids[i] for i in f["train"]],
                                 buffer_removed=[ids[i] for i in f["buffer_removed"]]) for k, f in enumerate(fs)]),
                C.OUT / "folds.json")
    sem = label_semantics(meta)
    print("[Q1] P1-C7 label semantics accepted:", sem["accepted_A1"])
    a_info = baseline_A(ids, meta, fs)
    b_info = baseline_B(ids, fs)
    wc = load_worldcover()
    lc_src = "worldcover" if wc is not None else "index_rules"
    models = {}
    for m in ("A", "B"):
        models[m] = evaluate(lambda i, m=m: load_mask(m, ids[i]), ids, fs, wc, m)
        pp, pa = models[m]["pooled_pixel"], models[m]["pooled_area"]
        print(f"[Q1] model {m}: IoU={pp['IoU']:.4f} F1={pp['F1']:.4f} kappa={pp['kappa']:.4f} "
              f"RAE={pa['RAE']:.4f} MAE_A={pa['MAE_A']:.4f}")
    e = models["B"]["A_hat_t"] - models["B"]["A_ref_t"]
    X, dsrc = descriptors(meta, wc)
    drivers = error_drivers(X, e, fold)
    drivers["descriptor_landcover_source"] = dsrc
    loco = loco_stress(ids, cl, b_info, wc)
    # hard cases passed to Q2
    A_ref = models["B"]["A_ref_t"]
    rae_t = np.where(A_ref >= 0.05, e / np.maximum(A_ref, 1e-12), np.nan)
    hard = [ids[i] for i in range(NT) if A_ref[i] >= 0.05 and abs(rae_t[i]) > 0.5]
    tab = models["B"].get("fpfn_worldcover", models["B"]["fpfn_index_rules"])["rows"]
    worst = sorted(tab, key=lambda r: -abs(r["net_km2"]))[:3]
    # example tiles for the visual comparison figure (deterministic choice by flood-share quantiles)
    order = np.argsort(flood)
    examples = [ids[order[int(q * (NT - 1))]] for q in (0.35, 0.6, 0.8, 0.95)]
    res = dict(meta=U.run_meta(), n_tiles=NT, tile_ids=ids, landcover_source=lc_src,
               preprocessing=dict(n_processed=len(meta["ids"]), processed_ids_first_last=[ids[0], ids[-1]],
                                  n_resampled_labels=sum(t["resampled"] for t in meta["tiles"]),
                                  enl_median=float(np.median([t["enl"] for t in meta["tiles"]])),
                                  ashman_share_mean=float(np.mean([t["ashman_share"] for t in meta["tiles"]])),
                                  pixel_area_m2_range=[float(np.load(CQ / "area_rows.npy").min()),
                                                       float(np.load(CQ / "area_rows.npy").max())],
                                  tile_area_km2_mean=float(np.mean([t["area_km2"] for t in meta["tiles"]]))),
               folds=dict(n_blocks=nblocks, fold_sizes=[len(f["test"]) for f in fs], fold_flood_share=shares,
                          train_sizes=[len(f["train"]) for f in fs],
                          buffer_removed=[len(f["buffer_removed"]) for f in fs], disjoint_and_buffer_ok=True),
               label_semantics=sem, baseline_A=a_info, baseline_B=b_info, models=models, drivers=drivers,
               loco=loco, hard_tiles=hard, worst_strata=[w["stratum"] for w in worst],
               example_tiles=examples, tile_flood_share=flood, tile_fold=fold, tile_cluster=cl,
               tile_col=col, tile_row=row, seconds=time.time() - t0)
    validate_capability(res, meta, fs)
    ok = U.validate_constraints(
        {"IoU": [models[m]["pooled_pixel"]["IoU"] for m in models] + [f["IoU"] for m in models for f in models[m]["per_fold"]],
         "F1": [models[m]["pooled_pixel"]["F1"] for m in models],
         "kappa": [models[m]["pooled_pixel"]["kappa"] for m in models],
         "OA": [models[m]["pooled_pixel"]["OA"] for m in models],
         "RAE": [models[m]["pooled_area"]["RAE"] for m in models],
         "Ahat_le_tile": [float(np.max(models[m]["A_hat_t"] - np.array([t["area_km2"] for t in meta["tiles"]]))) for m in models]},
        {"IoU": (0, 1, "IoU"), "F1": (0, 1, "F1"), "kappa": (-1, 1, "kappa"), "OA": (0, 1, "OA"),
         "RAE": (-1, np.inf, "RAE"), "Ahat_le_tile": (-np.inf, 1e-9, "A_hat_t <= tile area")}, "Q1")
    if not ok:
        raise ValueError("Q1 constraint validation failed")
    res["constraints_ok"] = ok
    U.save_json(res, C.OUT / "q1_results.json")
    U.save_json(res, C.FIG / "problem_1_results.json")
    print(f"[Q1] done in {time.time() - t0:.0f}s")
    return res


if __name__ == "__main__":
    main()
