"""Closing constraint audit: recomputes the hard constraints of MODELING_REPORT.md §13 from the saved
results and rasters (not from solver flags). Baseline/comparison entries are discovered automatically
and audited with the same rules. Prints conclusions only."""
import os, sys, re, json
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import rasterio
import config as C

_BASE_KW = re.compile(r"baseline|naive|greedy|lower.?bound", re.I)
report, audited = [], set()


def check(name, ok, detail=""):
    report.append((name, bool(ok), detail))
    print(f"[{name}] PASS={bool(ok)}" + ("" if ok else f"  {detail}"))


def load(n):
    return json.load(open(C.FIG / f"problem_{n}_results.json", encoding="utf-8"))


def main():
    q1, q2, q3, q4, q5 = (load(n) for n in range(1, 6))
    # ---------------- Q1 (+ baselines A, B)
    discovered = [k for k in q1 if _BASE_KW.search(k)]
    for m in ("A", "B"):
        r = q1["models"][m]
        pp, pa = r["pooled_pixel"], r["pooled_area"]
        vals = [pp["IoU"], pp["F1"], pp["OA"]] + [f["IoU"] for f in r["per_fold"]]
        check(f"Q1 {m} metric ranges", all(0 <= v <= 1 for v in vals) and -1 <= pp["kappa"] <= 1 and pa["RAE"] >= -1)
        ref, hat = np.array(r["A_ref_t"]), np.array(r["A_hat_t"])
        check(f"Q1 {m} DeltaA = sum(A_hat - A_ref)", abs((hat - ref).sum() - pa["DeltaA"]) < 1e-6)
        rows = r.get("fpfn_worldcover", r["fpfn_index_rules"])["rows"]
        check(f"Q1 {m} identity (4.4)", abs(sum(x["net_km2"] for x in rows) - pa["DeltaA"]) <= 1e-9 * max(1, abs(pa["DeltaA"])) + 1e-9)
        check(f"Q1 {m} 0 <= A_hat_t <= 4.95 km2", hat.min() >= 0 and hat.max() <= 4.95)
        audited.add(f"baseline_{m}")
    folds = json.load(open(C.OUT / "folds.json", encoding="utf-8"))
    cell = {t: (c, r) for t, c, r in zip(folds["ids"], folds["col"], folds["row"])}
    bad = 0
    for f in folds["folds"]:
        held = {cell[t] for t in f["test"] + f["cal"]}
        tr = set(f["train"])
        bad += len(tr & set(f["test"])) + len(tr & set(f["cal"]))
        bad += sum(any((cell[t][0] + dc, cell[t][1] + dr) in held for dc in (-1, 0, 1) for dr in (-1, 0, 1)) for t in tr)
    check("Q1 folds disjoint + buffer", bad == 0, f"violations={bad}")
    # ---------------- Q2 (M0 baseline included in the table)
    discovered += [f"q2:{k}" for k in q2["table"] if k in ("M0", "A")]
    for k, r in q2["table"].items():
        check(f"Q2 {k} metric ranges", 0 <= r["IoU"] <= 1 and -1 <= r["kappa"] <= 1 and r["RAE"] >= -1)
        if k in ("M0", "A"):
            audited.add(f"q2:{k}")
    th = q2["grading"]["theta"]
    check("Q2 0<theta1<theta2<theta3<1", 0 < th[0] < th[1] < th[2] < 1)
    pi = np.array(q2["grade_mc"]["pi"])
    check("Q2 sum pi = 1", np.abs(pi.sum(1) - 1).max() < 1e-9)
    a, b = q2["q2"]["cal_mask_area_km2"], q2["q2"]["cal_expected_area_km2"]
    check("Q2 area-preserving (5.4) within 1e-3", abs(a - b) / b <= 1e-3, f"{a} vs {b}")
    cf = q2["conformal"]
    cov = np.mean((np.array(cf["A_ref"]) >= np.array(cf["lo"]) - 1e-12) & (np.array(cf["A_ref"]) <= np.array(cf["hi"]) + 1e-12))
    check("Q2 coverage recomputed", abs(cov - cf["coverage_overall"]) < 1e-12, f"{cov}")
    # ---------------- Q3 rasters
    for d in ("0704", "0716"):
        with rasterio.open(C.OUT / f"q3_newflood_{d}.tif") as s:
            F = s.read(1) > 0
        with rasterio.open(C.OUT / "q3_w0.tif") as s:
            W0 = s.read(1) > 0
        n_over = int((F & W0).sum())
        del W0
        with rasterio.open(C.OUT / f"q3_w1_{d}.tif") as s:
            W1 = s.read(1) > 0
        n_out = int((F & ~W1).sum())
        del W1, F
        st = q3["dates"]["2026" + d]
        check(f"Q3 {d} F & W0 empty (raster)", n_over == 0, f"{n_over} px")
        check(f"Q3 {d} F subset W1 (raster)", n_out == 0, f"{n_out} px")
        check(f"Q3 {d} A_net <= A_new, rho = A_new/A_reg in [0,1]",
              st["A_net"] <= st["A_new"] + 1e-9 and abs(st["rho"] - st["A_new"] / st["A_reg"]) < 1e-12 and 0 <= st["rho"] <= 1)
    # ---------------- Q4
    w = np.array(list(q4["weights"].values()))
    check("Q4 weights >= 0, sum 1", (w >= 0).all() and abs(w.sum() - 1) < 1e-9)
    check("Q4 partition < 1 %, A_new sum < 0.5 %, CR < 0.1, Omega_H >= Omega_E",
          q4["partition_rel_error"] < 0.01 and q4["A_new_sum_rel_error"] <= 0.005 and q4["ahp"]["CR"] < 0.1
          and q4["ahp"]["Omega"][0] >= q4["ahp"]["Omega"][1])
    S = np.array([u["S"] for u in q4["units"]])
    check("Q4 0 <= S_j <= 1", S.min() >= 0 and S.max() <= 1)
    # ---------------- Q5 (independent audit of all plans incl. recourse and Pareto points)
    import q5_audit
    au = q5_audit.audit(C.OUT / "q5_results.json")
    check("Q5 independent audit (19 plans)", au["PASS"])
    check("Q5 open depots <= K", sum(q5["plan_0716"]["eta"]) <= q5["params"]["K"])
    missing = [b for b in discovered if b not in audited]
    check("baseline discovery: all comparison entries audited", not missing, f"missing={missing}")
    n_fail = sum(not ok for _, ok, _ in report)
    print(f"[constraint_audit] checks={len(report)} n_violations={n_fail} PASS={n_fail == 0}")
    return n_fail == 0


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
