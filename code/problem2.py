"""Problem 2 — multi-source fusion improved model, area calibration, conformal tile-area intervals,
data-derived 4-level intensity grading and Monte Carlo grade uncertainty (MODELING_REPORT.md §5).

Variants (identical outer folds and test tiles as Q1):
 M0 Q1 LightGBM | M1 U-Net SAR only | M2 +optical gated fusion | M3 +area loss | M4 +modality dropout/validity
 M5 M4 + MC-dropout mean, thr 0.5 | M6 M5 + isotonic + area-preserving tau_A (final Q2 model)
"""
import os, sys, json, time, math, hashlib
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import GaussianMixture
import config as C
import common as U
import problem1 as P1
from unet import GatedFusionUNet, flood_loss, enable_mc_dropout

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NT, H, NF = P1.NT, P1.H, P1.NF
CQ2 = C.CACHE / "q2"
CQ2.mkdir(parents=True, exist_ok=True)
MODELS = C.OUT / "q2_models"
MODELS.mkdir(parents=True, exist_ok=True)
CH = [U.FI[c] for c in U.SAR_CH + U.OPT_CH]       # 3 SAR + 7 optical channels
CFG = {"M1": dict(c_opt=0, lam=0.0, mdrop=0.0), "M2": dict(c_opt=6, lam=0.0, mdrop=0.0),
       "M3": dict(c_opt=6, lam=C.LAMBDA_A, mdrop=0.0), "M4": dict(c_opt=7, lam=C.LAMBDA_A, mdrop=C.MODALITY_DROPOUT)}
CPU_FALLBACK = DEV.type != "cuda"
WIDTHS = C.UNET_WIDTHS if not CPU_FALLBACK else tuple(w // 2 for w in C.UNET_WIDTHS)
EPOCHS = C.UNET_EPOCHS if not CPU_FALLBACK else 25
EPOCHS = int(os.environ.get("Q2_SMOKE_EPOCHS", EPOCHS))      # smoke-test override only
N_BOOT = int(os.environ.get("Q2_SMOKE_NBOOT", C.N_BOOT))


# ============================================================ data on device
class Data:
    def __init__(self):
        feats = P1.mm("feats.f16", np.float16, (NT, NF, H, H))
        Y = P1.mm("y.u8", np.uint8, (NT, H, H))
        V = P1.mm("valid.u8", np.uint8, (NT, H, H))
        dt = torch.float16 if DEV.type == "cuda" else torch.float32
        self.X = torch.empty((NT, len(CH), H, H), dtype=dt, device=DEV)
        for s in range(0, NT, 50):
            self.X[s:s + 50] = torch.from_numpy(np.ascontiguousarray(feats[s:s + 50][:, CH])).to(DEV, dt)
        self.Y = torch.from_numpy(np.array(Y)).to(DEV)
        self.V = torch.from_numpy(np.array(V)).to(DEV)
        self.A = torch.from_numpy(np.load(P1.CQ / "area_rows.npy").astype(np.float32) * 1e-6).to(DEV)  # km2 per row

    def inputs(self, idx, c_opt):
        x = self.X[idx].float()
        xs = x[:, :3]
        if c_opt == 0:
            return xs, None
        return xs, x[:, 3:3 + c_opt]


def augment(xs, xo, y, v, gen):
    """Random flips and rot90 per sample."""
    out = [[], [], [], []]
    for b in range(xs.shape[0]):
        k = int(torch.randint(0, 4, (1,), generator=gen))
        fl = bool(torch.randint(0, 2, (1,), generator=gen))
        ts = [xs[b], xo[b] if xo is not None else None, y[b], v[b]]
        for j, t in enumerate(ts):
            if t is None:
                out[j].append(None)
                continue
            t = torch.rot90(t, k, dims=(-2, -1))
            if fl:
                t = torch.flip(t, dims=(-1,))
            out[j].append(t)
    st = [torch.stack(o) if o[0] is not None else None for o in out]
    return st


@torch.no_grad()
def predict(model, D, idx, c_opt, mc=False, seed=0, bs=32):
    """Probability maps for tiles idx (single deterministic pass or one MC-dropout pass)."""
    if mc:
        enable_mc_dropout(model)
        torch.manual_seed(seed)
    else:
        model.eval()
    out = []
    for s in range(0, len(idx), bs):
        xs, xo = D.inputs(idx[s:s + bs], c_opt)
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            p = torch.sigmoid(model(xs, xo).float())
        out.append(p)
    return torch.cat(out)


def tile_area(mask, D, idx):
    """Area (km2) of a mask per tile, valid pixels only."""
    return ((mask & (D.V[idx] > 0)).float() * D.A[idx][:, :, None]).sum((1, 2))
# ============================================================ training
def cfg_hash(name, k):
    s = json.dumps(dict(cfg=CFG[name], k=k, stop="cal_MAE_A", widths=WIDTHS, epochs=EPOCHS, lr=C.UNET_LR, bs=C.UNET_BATCH,
                        patience=C.UNET_PATIENCE, drop=C.UNET_DROPOUT,
                        unet=U.file_sha(os.path.join(_HERE, "unet.py")),
                        feats=json.load(open(P1.CQ / "pre_meta.json"))["common_sha"]), sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def make_model(name):
    cfg = CFG[name]
    return GatedFusionUNet(c_sar=3, c_opt=cfg["c_opt"], widths=WIDTHS, bottleneck=2 * WIDTHS[-1],
                           p_drop=C.UNET_DROPOUT).to(DEV)


def train_one(name, k, fs, D):
    """Train variant `name` on outer fold k; early stopping on calibration total |dA| at p>=0.5."""
    cfg = CFG[name]
    hsh = cfg_hash(name, k)
    ck = (MODELS if name == "M4" else CQ2) / f"{name}_fold{k}.pt"
    logf = CQ2 / f"{name}_fold{k}.json"
    if ck.exists() and logf.exists() and json.load(open(logf)).get("hash") == hsh:
        model = make_model(name)
        with open(ck, "rb") as fh:
            model.load_state_dict(torch.load(fh, map_location=DEV))
        print(f"[Q2] reuse {name} fold {k}")
        return model, json.load(open(logf))
    seed = C.SEED + k
    torch.manual_seed(seed)
    np.random.seed(seed)
    # seeds are fixed; deterministic cuDNN kernels were 3x slower on this GPU (0.48 vs 0.15 s/iter),
    # so non-deterministic kernels are used and run-to-run differences are at the kernel-rounding level
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    gen = torch.Generator().manual_seed(seed)
    model = make_model(name)
    opt = torch.optim.Adam(model.parameters(), lr=C.UNET_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    tr = np.array(fs[k]["train"])
    ca = torch.tensor(fs[k]["cal"], device=DEV)
    A_cal_ref = tile_area(D.Y[ca] > 0, D, ca)
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        losses = []
        for s in range(0, len(perm), C.UNET_BATCH):
            idx = torch.tensor(perm[s:s + C.UNET_BATCH], device=DEV)
            xs, xo = D.inputs(idx, cfg["c_opt"])
            if CPU_FALLBACK:      # random 128x128 crops in CPU fallback
                r0, c0 = [int(v) for v in torch.randint(0, H - 128, (2,), generator=gen)]
                sl = (slice(None), slice(None), slice(r0, r0 + 128), slice(c0, c0 + 128))
                xs = xs[sl]; xo = xo[sl] if xo is not None else None
                y, v = D.Y[idx][:, r0:r0 + 128, c0:c0 + 128], D.V[idx][:, r0:r0 + 128, c0:c0 + 128]
            else:
                y, v = D.Y[idx], D.V[idx]
            xs, xo, y, v = augment(xs, xo, y, v, gen)
            if cfg["mdrop"] > 0:
                drop = (torch.rand(xo.shape[0], generator=gen) < cfg["mdrop"]).to(DEV)
                xo = xo * (~drop).float()[:, None, None, None]
            with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                logit = model(xs, xo).float()
            loss, parts = flood_loss(logit, y, v, lam_area=cfg["lam"], gamma=C.FOCAL_GAMMA, alpha1=C.FOCAL_ALPHA1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        p = predict(model, D, ca, cfg["c_opt"])
        e_t = tile_area(p >= 0.5, D, ca) - A_cal_ref
        # early-stopping score = calibration tile MAE of area (a net |sum dA| lets FP and FN cancel and
        # selected an untrained epoch-0 model in a first run)
        dA = float(e_t.abs().mean())
        hist.append(dict(epoch=ep, train_loss=float(np.mean(losses)), cal_MAE_A_km2=dA,
                         cal_abs_sum_dA_km2=float(e_t.sum().abs())))
        if dA < best - 1e-6:
            best, bad = dA, 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            print(f"[Q2] {name} fold {k} ep {ep} loss {np.mean(losses):.4f} cal MAE_A {dA:.4f} km2 "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if bad >= C.UNET_PATIENCE:
            break
    model.load_state_dict(best_state)
    with open(ck, "wb") as fh:
        torch.save(best_state, fh)
    log = dict(hash=hsh, best_cal_MAE_A_km2=best, epochs_run=len(hist), history=hist, seconds=time.time() - t0,
               n_params=sum(p.numel() for p in model.parameters()))
    json.dump(log, open(logf, "w"))
    return model, log
# ============================================================ calibration, tau_A, conformal (M5, M6)
def fit_isotonic(pbar, y, valid, seed, n=2_000_000):
    vi = torch.nonzero(valid.reshape(-1) > 0).squeeze(1)
    g = torch.Generator(device=vi.device).manual_seed(seed)
    sel = vi[torch.randperm(len(vi), generator=g, device=vi.device)[:n]]
    x = pbar.reshape(-1)[sel].cpu().numpy().astype(np.float64)
    t = y.reshape(-1)[sel].cpu().numpy().astype(np.float64)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip", increasing=True).fit(x, t)
    return iso


def iso_apply(iso, p):
    """Apply the isotonic step function on device (linear interpolation between thresholds, as sklearn)."""
    xt = torch.tensor(iso.X_thresholds_, device=p.device, dtype=torch.float32)
    yt = torch.tensor(iso.y_thresholds_, device=p.device, dtype=torch.float32)
    pf = p.reshape(-1).clamp(float(xt[0]), float(xt[-1]))
    j = torch.searchsorted(xt, pf).clamp(1, len(xt) - 1)
    x0, x1, y0, y1 = xt[j - 1], xt[j], yt[j - 1], yt[j]
    w = torch.where(x1 > x0, (pf - x0) / (x1 - x0), torch.zeros_like(pf))
    return (y0 + w * (y1 - y0)).reshape(p.shape)


def area_preserving_tau(pbar, wA, expected, lo=0.01, hi=0.99, tol=1e-4):
    """(5.4) bisection: sum a*1[p>=tau] = sum a*q(p); left side non-increasing in tau."""
    def f(t):
        return float((wA * (pbar >= t)).sum()) - expected
    if f(lo) < 0 or f(hi) > 0:
        lo, hi = 0.001, 0.999
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    # choose the bracket end with the smaller area mismatch
    t = lo if abs(f(lo)) <= abs(f(hi)) else hi
    return t, f(t) + expected


def mc_fold(model, D, fs, k, Tn=C.MC_T):
    """MC-dropout passes for cal and test tiles; returns per-tile quantities for M5/M6/conformal."""
    ca = torch.tensor(fs[k]["cal"], device=DEV)
    te = torch.tensor(fs[k]["test"], device=DEV)
    pc = torch.stack([predict(model, D, ca, 7, mc=True, seed=10_000 * k + l).half() for l in range(Tn)])
    pt = torch.stack([predict(model, D, te, 7, mc=True, seed=10_000 * k + l).half() for l in range(Tn)])
    pbar_c, pbar_t = pc.float().mean(0), pt.float().mean(0)
    vc = D.V[ca] > 0
    wA = D.A[ca][:, :, None].expand_as(pbar_c) * vc
    iso = fit_isotonic(pbar_c, D.Y[ca], D.V[ca], seed=C.SEED + k)
    expected = float((wA * iso_apply(iso, pbar_c)).sum())
    tauA, mask_area = area_preserving_tau(pbar_c, wA, expected)
    # tile areas
    Aref_c = tile_area(D.Y[ca] > 0, D, ca)
    Ahat_c = tile_area(pbar_c >= tauA, D, ca)
    Ahat_t = tile_area(pbar_t >= tauA, D, te)
    sig_c = torch.stack([tile_area(pc[l].float() >= tauA, D, ca) for l in range(Tn)]).std(0, unbiased=True)
    sig_t = torch.stack([tile_area(pt[l].float() >= tauA, D, te) for l in range(Tn)]).std(0, unbiased=True)
    c = max(float(sig_c.median()), C.SIGMA_FLOOR_KM2)
    sp_c, sp_t = sig_c + c, sig_t + c
    s = ((Aref_c - Ahat_c).abs() / sp_c).cpu().numpy()
    n = len(s)
    r = math.ceil((n + 1) * (1 - C.CONF_ALPHA))
    qhat = float(np.sort(s)[min(r, n) - 1])
    Atile_t = tile_area(torch.ones_like(D.Y[te], dtype=torch.bool), D, te)
    lo_t = torch.clamp(Ahat_t - qhat * sp_t, min=0)
    hi_t = torch.minimum(Atile_t, Ahat_t + qhat * sp_t)
    resid = ((Aref_c - Ahat_c) / sp_c).cpu().numpy()
    out = dict(tauA=tauA, cal_mask_area_km2=mask_area, cal_expected_area_km2=expected, c_sigma=c, qhat=qhat,
               n_cal=n, iso_x=iso.X_thresholds_, iso_y=iso.y_thresholds_, resid_cal=resid,
               cal_ids=fs[k]["cal"], test_ids=fs[k]["test"],
               sigma_test=sig_t.cpu().numpy(), sigp_test=sp_t.cpu().numpy(), Ahat_test=Ahat_t.cpu().numpy(),
               lo_test=lo_t.cpu().numpy(), hi_test=hi_t.cpu().numpy(), Atile_test=Atile_t.cpu().numpy())
    # cache calibration predictions (pooled refit in Q3) and test probability maps
    np.save(CQ2 / f"cal_pbar_fold{k}.npy", pbar_c.cpu().numpy().astype(np.float32))
    return out, pbar_t, (pbar_t >= 0.5), (pbar_t >= tauA)
# ============================================================ grading (5.7)-(5.9)
def logit(r):
    r = np.clip(r, 1e-4, 1 - 1e-4)
    return np.log(r / (1 - r))


def gmm_thresholds(x, seed=C.SEED, n_init=C.GMM_N_INIT, K=C.N_GMM):
    """(5.8) Bayes boundaries between adjacent components of a K-component 1-D GMM on logit(rho)."""
    g = GaussianMixture(n_components=K, covariance_type="full", n_init=n_init, random_state=seed).fit(x[:, None])
    order = np.argsort(g.means_.ravel())
    mu = g.means_.ravel()[order]
    sd = np.sqrt(g.covariances_.ravel()[order])
    pi = g.weights_[order]
    th = []
    for j in range(K - 1):
        fun = lambda t: np.log(pi[j]) + norm.logpdf(t, mu[j], sd[j]) - np.log(pi[j + 1]) - norm.logpdf(t, mu[j + 1], sd[j + 1])
        a, b = mu[j], mu[j + 1]
        if not (fun(a) > 0 and fun(b) < 0):
            return None, dict(mu=mu, sd=sd, pi=pi)
        th.append(brentq(fun, a, b))
    theta = 1 / (1 + np.exp(-np.array(th)))
    ok = bool(np.all(np.diff(theta) > 0) and theta[0] > 0 and theta[-1] < 1)
    return (theta if ok else None), dict(mu=mu, sd=sd, pi=pi, bic=g.bic(x[:, None]))


def jenks(x, k=4):
    """Fisher-Jenks natural breaks (exact DP); returns k-1 upper class limits."""
    x = np.sort(x)
    n = len(x)
    cs, cs2 = np.concatenate([[0], np.cumsum(x)]), np.concatenate([[0], np.cumsum(x * x)])
    def ssd(i, j):  # x[i:j]
        s, s2, m = cs[j] - cs[i], cs2[j] - cs2[i], j - i
        return s2 - s * s / m
    D = np.full((k + 1, n + 1), np.inf); D[0, 0] = 0
    Bk = np.zeros((k + 1, n + 1), int)
    for c in range(1, k + 1):
        for j in range(c, n + 1):
            best, arg = np.inf, c - 1
            for i in range(c - 1, j):
                v = D[c - 1, i] + ssd(i, j)
                if v < best:
                    best, arg = v, i
            D[c, j], Bk[c, j] = best, arg
    cuts, j = [], n
    for c in range(k, 1, -1):
        i = Bk[c, j]
        cuts.append(x[i - 1])
        j = i
    return np.array(sorted(cuts))


def grade_of(rho, theta):
    """Grade 1..4 = general, relatively strong, strong, extremely strong."""
    return 1 + np.searchsorted(np.asarray(theta), np.asarray(rho), side="right")


def grading(flood_share, block_of):
    nz = flood_share > 0
    x = logit(flood_share[nz])
    theta, comp = gmm_thresholds(x)
    rule = "gmm"
    if theta is None:
        theta = 1 / (1 + np.exp(-jenks(x)))
        rule = "jenks_fallback"
    bic = {}
    for K in range(2, 7):
        g = GaussianMixture(n_components=K, n_init=C.GMM_N_INIT, random_state=C.SEED).fit(x[:, None])
        bic[K] = g.bic(x[:, None])
    jk = 1 / (1 + np.exp(-jenks(x)))
    # block bootstrap over the 67 blocks
    rng = np.random.default_rng(C.SEED)
    blocks = np.unique(block_of)
    boot, n_ok = [], 0
    t0 = time.time()
    for b in range(N_BOOT):
        pick = rng.choice(blocks, len(blocks), replace=True)
        idx = np.concatenate([np.where(block_of == bb)[0] for bb in pick])
        xs = flood_share[idx]
        xs = logit(xs[xs > 0])
        th, _ = gmm_thresholds(xs, seed=C.SEED + b + 1, n_init=C.GMM_N_INIT)
        if th is not None:
            boot.append(th)
            n_ok += 1
        if b % 200 == 0:
            print(f"[Q2] bootstrap {b}/{N_BOOT} ({time.time() - t0:.0f}s)", flush=True)
    boot = np.array(boot)
    A_ref_tile = 256 * 256 * 75.1e-6
    return dict(rule=rule, theta=theta, components=comp, bic=bic, bic_best_K=min(bic, key=bic.get),
                jenks_theta=jk, n_nonempty=int(nz.sum()), boot_theta=boot, boot_ok_share=n_ok / N_BOOT,
                boot_ci90=np.percentile(boot, [5, 95], axis=0) if len(boot) else None,
                area_thresholds_km2={"reference_tile_4.92": theta * 4.92, "grid_1km2": theta * 1.0,
                                     "Hengzhou_3440.4": theta * C.A_REG_CITY_KM2},
                ref_tile_area_km2_nominal=A_ref_tile)


def mc_grades(Ahat, sigp, Areg, resid_pool, boot_theta, M=C.MC_GRADE, seed=C.SEED):
    """(5.9) grade probabilities pi_jg for units j (one shared residual per draw)."""
    rng = np.random.default_rng(seed)
    r = rng.choice(resid_pool, M, replace=True)
    th = boot_theta[rng.integers(0, len(boot_theta), M)]
    A = np.clip(Ahat[:, None] + r[None, :] * sigp[:, None], 0, Areg[:, None])       # (J, M)
    rho = A / Areg[:, None]
    g = 1 + (rho[:, :, None] >= th[None, :, :]).sum(2)                                  # (J, M)
    pi = np.stack([(g == gg).mean(1) for gg in (1, 2, 3, 4)], 1)
    return pi
# ============================================================ evaluation helpers
def subset_metrics(ev, idx):
    idx = np.asarray(idx, int)
    cm = ev["tile_cm"][idx].sum(0)
    pm = U.pixel_metrics(dict(tp=cm[0], fp=cm[1], fn=cm[2], tn=cm[3]))
    am = U.area_metrics(ev["A_ref_t"][idx], ev["A_hat_t"][idx])
    return dict(**pm, **am)


def summary_row(ev):
    pp, pa = ev["pooled_pixel"], ev["pooled_area"]
    return dict(IoU=pp["IoU"], F1=pp["F1"], kappa=pp["kappa"], OA=pp["OA"], Precision=pp["Precision"],
                Recall=pp["Recall"], RAE=pa["RAE"], MAE_A=pa["MAE_A"], median_abs_RAE_t=pa["median_abs_RAE_t"],
                DeltaA=pa["DeltaA"])


@torch.no_grad()
def gate_stats(model, D, idx):
    model.eval()
    vals = []
    for s in range(0, len(idx), 16):
        xs, xo = D.inputs(torch.tensor(idx[s:s + 16], device=DEV), 7)
        _, g = model(xs, xo, return_gates=True)
        vals.append([float(v) for v in g])
    return np.mean(vals, 0)


def validate_capability(res):
    t = res["grading"]["theta"]
    assert 0 < t[0] < t[1] < t[2] < 1, "[P2-C3] thresholds not strictly increasing"
    assert res["grading"]["rule"] in ("gmm", "jenks_fallback")
    assert res["identical_test_sets"], "[P2-C2] variants not on identical test tiles"
    pis = np.array(res["grade_mc"]["pi"])
    assert np.all(np.abs(pis.sum(1) - 1) < 1e-9), "[P2-C4] grade probabilities do not sum to 1"
    assert "coverage_overall" in res["conformal"], "[P2-C5] no empirical coverage"
    for f in res["calibration"]:
        rel = abs(f["cal_mask_area_km2"] - f["cal_expected_area_km2"]) / f["cal_expected_area_km2"]
        assert rel <= 1e-3, f"(5.4) area-preserving equality violated: {rel}"
    print("[Q2] validate_capability PASS")


def main():
    U.set_all_seeds()
    t0 = time.time()
    q1 = json.load(open(C.OUT / "q1_results.json", encoding="utf-8"))
    folds = json.load(open(C.OUT / "folds.json", encoding="utf-8"))
    ids = folds["ids"]
    pos = {t: i for i, t in enumerate(ids)}
    fs = [dict(test=[pos[t] for t in f["test"]], cal=[pos[t] for t in f["cal"]], train=[pos[t] for t in f["train"]])
          for f in folds["folds"]]
    D = Data()
    print(f"[Q2] device={DEV} cpu_fallback={CPU_FALLBACK} data on device ({time.time() - t0:.0f}s)")
    masks = {m: P1.mm(f"mask_{m}.u8", np.uint8, (NT, H, H), "w+") for m in ("M1", "M2", "M3", "M4", "M5", "M6")}
    for m in masks:
        masks[m].flush()
    probM6 = P1.mm("prob_m6.u8", np.uint8, (NT, H, H), "w+")
    logs, calib, conf_rows, gates = {}, [], [], []
    for name in ("M1", "M2", "M3", "M4"):
        for k in range(C.N_FOLDS):
            model, log = train_one(name, k, fs, D)
            logs[f"{name}_fold{k}"] = log
            te = torch.tensor(fs[k]["test"], device=DEV)
            p = predict(model, D, te, CFG[name]["c_opt"])
            mk = ((p >= 0.5) & (D.V[te] > 0)).cpu().numpy().astype(np.uint8)
            masks[name][fs[k]["test"]] = mk
            if name == "M4":
                out, pbar_t, m5, m6 = mc_fold(model, D, fs, k)
                vt = D.V[te] > 0
                masks["M5"][fs[k]["test"]] = (m5 & vt).cpu().numpy().astype(np.uint8)
                masks["M6"][fs[k]["test"]] = (m6 & vt).cpu().numpy().astype(np.uint8)
                probM6[fs[k]["test"]] = torch.round(pbar_t * 255).to(torch.uint8).cpu().numpy()
                calib.append(dict(fold=k, **{kk: out[kk] for kk in ("tauA", "cal_mask_area_km2", "cal_expected_area_km2",
                                                                    "c_sigma", "qhat", "n_cal", "iso_x", "iso_y")}))
                conf_rows.append(out)
                gates.append(gate_stats(model, D, fs[k]["test"]))
                ck = MODELS / f"M4_fold{k}.pt"
                calib[-1]["checkpoint"] = str(ck.relative_to(C.ROOT)).replace("\\", "/")
                calib[-1]["checkpoint_sha256"] = U.file_sha(ck)
                print(f"[Q2] fold {k}: tauA={out['tauA']:.4f} qhat={out['qhat']:.3f} c={out['c_sigma']:.4f}", flush=True)
            del model
            torch.cuda.empty_cache() if DEV.type == "cuda" else None
    for m in masks.values():
        m.flush()
    probM6.flush()
    # ---------------- evaluation on identical test tiles
    wc = P1.load_worldcover()
    evs = {"M0": P1.evaluate(lambda i: P1.load_mask("B", ids[i]), ids, fs, wc, "M0")}
    for name in ("M1", "M2", "M3", "M4", "M5", "M6"):
        mmk = P1.mm(f"mask_{name}.u8", np.uint8, (NT, H, H))
        evs[name] = P1.evaluate(lambda i, mmk=mmk: mmk[i] > 0, ids, fs, wc, name)
        print(f"[Q2] {name}: " + " ".join(f"{k}={v:.4f}" for k, v in summary_row(evs[name]).items()), flush=True)
    evA = P1.evaluate(lambda i: P1.load_mask("A", ids[i]), ids, fs, wc, "A")
    test_hash = hashlib.sha256(json.dumps(sorted(sum([f["test"] for f in fs], []))).encode()).hexdigest()
    hard_idx = [pos[t] for t in q1["hard_tiles"]]
    table = {m: summary_row(e) for m, e in evs.items()}
    table["A"] = summary_row(evA)
    hard = {m: subset_metrics(e, hard_idx) for m, e in evs.items()}
    # ---------------- conformal coverage on test tiles
    Aref = evs["M6"]["A_ref_t"]
    cov_tile = np.zeros(NT, bool)
    lo, hi, sig, sigp, ahat = (np.zeros(NT) for _ in range(5))
    fold_of = np.zeros(NT, int)
    for k, o in enumerate(conf_rows):
        ti = np.array(o["test_ids"])
        lo[ti], hi[ti], sig[ti], sigp[ti], ahat[ti] = o["lo_test"], o["hi_test"], o["sigma_test"], o["sigp_test"], o["Ahat_test"]
        fold_of[ti] = k
    cov_tile = (Aref >= lo - 1e-12) & (Aref <= hi + 1e-12)
    fl = np.array(q1["tile_flood_share"])
    terc = np.digitize(fl, np.quantile(fl, [1 / 3, 2 / 3]))
    resid_pool = np.concatenate([o["resid_cal"] for o in conf_rows])
    conformal = dict(alpha=C.CONF_ALPHA, coverage_overall=float(cov_tile.mean()),
                     coverage_by_tercile=[float(cov_tile[terc == j].mean()) for j in range(3)],
                     coverage_by_fold=[float(cov_tile[fold_of == k].mean()) for k in range(C.N_FOLDS)],
                     mean_width_km2=float((hi - lo).mean()), lo=lo, hi=hi, sigma=sig, sigma_prime=sigp, Ahat=ahat,
                     A_ref=Aref, covered=cov_tile, resid_pool=resid_pool,
                     total_interval_km2=[float(max(0, ahat.sum() - np.quantile(np.abs(resid_pool), 0.9) * sigp.sum())),
                                         float(ahat.sum() + np.quantile(np.abs(resid_pool), 0.9) * sigp.sum())])
    print(f"[Q2] conformal coverage {conformal['coverage_overall']:.3f} by fold {np.round(conformal['coverage_by_fold'], 3)}")
    use_m6 = table["M6"]["MAE_A"] <= table["M0"]["MAE_A"]
    # ---------------- grading
    gr = grading(fl, np.array(folds["block"]))
    theta = np.array(gr["theta"])
    Atile = np.array([t for t in np.load(P1.CQ / "area_rows.npy").sum(1) * 256e-6])
    V = P1.mm("valid.u8", np.uint8, (NT, H, H))
    area_rows = np.load(P1.CQ / "area_rows.npy")
    Areg = np.array([(area_rows[i][:, None] * (V[i] > 0)).sum() * 1e-6 for i in range(NT)])
    g_ref = grade_of(Aref / Areg, theta)
    g_m6 = grade_of(evs["M6"]["A_hat_t"] / Areg, theta)
    g_m0 = grade_of(evs["M0"]["A_hat_t"] / Areg, theta)
    pi = mc_grades(ahat, sigp, Areg, resid_pool, gr["boot_theta"] if len(gr["boot_theta"]) else theta[None])
    flip = 1 - pi.max(1)
    dist = np.min(np.abs(ahat[:, None] / Areg[:, None] - theta[None]) * Areg[:, None], 1) / sigp
    bins = [0, 0.5, 1, 2, 4, np.inf]
    db = np.digitize(dist, bins) - 1
    conf_mat = np.zeros((4, 4))
    for gg in range(4):
        conf_mat[gg] = pi[g_ref == gg + 1].sum(0)
    grade_mc = dict(pi=pi, flip_prob=flip, std_distance=dist, ref_grade=g_ref, m6_grade=g_m6, m0_grade=g_m0,
                    acc_det_M6=float((g_m6 == g_ref).mean()), acc_det_M0=float((g_m0 == g_ref).mean()),
                    acc_expected_MC=float(pi[np.arange(NT), g_ref - 1].mean()),
                    confusion_ref_vs_mc=conf_mat, flip_by_distance=[dict(lo=bins[b], hi=bins[b + 1], n=int((db == b).sum()),
                                                                          mean_flip=float(flip[db == b].mean()) if (db == b).any() else None)
                                                                     for b in range(len(bins) - 1)],
                    ref_grade_counts=np.bincount(g_ref, minlength=5)[1:])
    res = dict(meta=U.run_meta(), device=str(DEV), cpu_fallback=CPU_FALLBACK, widths=WIDTHS, epochs_max=EPOCHS,
               variants=CFG, table=table, hard_set=dict(n=len(hard_idx), metrics=hard), identical_test_sets=True,
               test_set_sha256=test_hash, submission_rule=dict(use="M6" if use_m6 else "M0",
                                                               MAE_A_M6=table["M6"]["MAE_A"], MAE_A_M0=table["M0"]["MAE_A"]),
               fpfn_M6=evs["M6"].get("fpfn_worldcover", evs["M6"]["fpfn_index_rules"]),
               fpfn_M0=evs["M0"].get("fpfn_worldcover", evs["M0"]["fpfn_index_rules"]),
               per_fold={m: e["per_fold"] for m, e in evs.items()}, per_tile={m: dict(A_hat=e["A_hat_t"]) for m, e in evs.items()},
               A_ref_t=Aref, calibration=calib, conformal=conformal, grading=gr, grade_mc=grade_mc,
               gate_mean_by_scale=np.mean(gates, 0), gate_by_fold=gates,
               training_logs={k: dict(epochs_run=v["epochs_run"], best=v["best_cal_MAE_A_km2"], seconds=v["seconds"],
                                      n_params=v["n_params"], history=v["history"]) for k, v in logs.items()},
               seconds=time.time() - t0)
    res["q2"] = dict(cal_mask_area_km2=sum(c["cal_mask_area_km2"] for c in calib),
                     cal_expected_area_km2=sum(c["cal_expected_area_km2"] for c in calib))
    validate_capability(res)
    ok = U.validate_constraints(
        {"IoU": [r["IoU"] for r in table.values()], "kappa": [r["kappa"] for r in table.values()],
         "RAE": [r["RAE"] for r in table.values()], "theta": theta, "pi_sum_err": np.abs(pi.sum(1) - 1),
         "coverage": conformal["coverage_overall"]},
        {"IoU": (0, 1, "IoU"), "kappa": (-1, 1, "kappa"), "RAE": (-1, np.inf, "RAE"), "theta": (0, 1, "theta"),
         "pi_sum_err": (0, 1e-9, "sum pi"), "coverage": (0, 1, "coverage")}, "Q2")
    if not ok:
        raise ValueError("Q2 constraint validation failed")
    U.save_json(res, C.OUT / "q2_results.json")
    U.save_json(res, C.FIG / "problem_2_results.json")
    print(f"[Q2] done in {time.time() - t0:.0f}s; submission model = {res['submission_rule']['use']}")
    return res


if __name__ == "__main__":
    main()
