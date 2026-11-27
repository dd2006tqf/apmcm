"""E4 — dual-scale grading (share vs absolute area) on the 393 Q4 units.

The paper grades exposure with a 4-component GMM on the flooded share only. Modex-style dual-scale
grading takes L = max(L_share, L_area), with L_area from the same 4-component GMM on log10(A_reg).
We report how many units change level and how the priority weights move.
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "code"))

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture
import config as C

RES = os.path.join(_HERE, "results")
os.makedirs(RES, exist_ok=True)


def main():
    g = gpd.read_file(C.OUT / "q4_units.geojson")
    df = pd.DataFrame(dict(uid=g.uid, A=g.A_reg_km2, share=g.H1 / g.A_reg_km2, level=g.level.astype(int),
                           E1=g.E1, S=g.S, affected=g.affected.astype(bool)))
    x = np.log10(df.A.clip(lower=1e-3))
    gm = GaussianMixture(n_components=4, covariance_type="full", n_init=20, random_state=C.SEED).fit(x.values[:, None])
    mu = np.sort(gm.means_.ravel())
    bounds = (mu[:-1] + mu[1:]) / 2
    la = np.digitize(x.values, bounds) + 1
    la = np.where(df.affected.values, la, 0)          # unaffected units keep level 0
    dual = np.maximum(df.level.values, la)
    up = dual > df.level.values
    down = dual < df.level.values

    out = dict(meta=dict(n_units=int(len(df)), n_affected=int(df.affected.sum()), gmm_means_log10=[float(v) for v in mu],
                         area_bounds_km2=[float(10 ** b) for b in bounds],
                         threshold_area_note="boundaries are Bayes midpoints of a 4-component GMM on log10(A_reg)"),
               levels_share=pd.Series(df.level).value_counts().sort_index().to_dict(),
               levels_area=pd.Series(la).value_counts().sort_index().to_dict(),
               levels_dual=pd.Series(dual).value_counts().sort_index().to_dict(),
               n_upgraded=int(up.sum()), n_downgraded=int(down.sum()),
               frac_upgraded=float(up.mean()),
               area_upgraded_km2=float(df.A[up].sum()), area_total_km2=float(df.A.sum()),
               spearman_dual_vs_share=float(spearmanr(dual, df.level).statistic),
               spearman_area_vs_share=float(spearmanr(la, df.level).statistic),
               affected_only=dict(n=int(df.affected.sum()), n_upgraded=int((up & df.affected.values).sum()),
                                  frac=float((up & df.affected.values).sum() / max(df.affected.sum(), 1)),
                                  share_level_counts=pd.Series(df.level[df.affected.values]).value_counts().sort_index().to_dict(),
                                  dual_level_counts=pd.Series(dual[df.affected.values]).value_counts().sort_index().to_dict()))
    g2 = g.copy()
    g2["level_share"] = df.level.values; g2["level_area"] = la; g2["level_dual"] = dual
    g2[["uid", "name", "A_reg_km2", "H1", "level_share", "level_area", "level_dual", "E1", "S"]].to_csv(
        os.path.join(RES, "e4_dual_grades.csv"), index=False, encoding="utf-8-sig")
    json.dump(out, open(os.path.join(RES, "e4_dual_grading.json"), "w"), indent=1)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
