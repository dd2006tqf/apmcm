"""Collect key results of Q1-Q5 into figures/all_results.json (+ logic probes) and
figures/sensitivity_results.json. Reads only the saved per-problem result files."""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import config as C
import common as U


def load(n):
    f = C.FIG / f"problem_{n}_results.json"
    return json.load(open(f, encoding="utf-8")) if f.exists() else None


def backfill_ledger(q2, q3, q4, q5):
    """Write the observed cross-problem quantities into CROSS_PROBLEM_LEDGER.json (computed from results)."""
    f = C.ROOT / "CROSS_PROBLEM_LEDGER.json"
    if not f.exists() or not (q2 and q3 and q4 and q5):
        return
    led = json.load(open(f, encoding="utf-8"))
    th = list(q2["grading"]["theta"])
    obs = {"Q3": dict(threshold_mismatch_vs_Q2=sum(a != b for a, b in zip(th, q3["grade"]["theta"])),
                      model_checkpoint_mismatch_vs_Q2=q3["model"]["checkpoint_hash_mismatches"]),
           "Q4": dict(threshold_mismatch_vs_Q2=sum(a != b for a, b in zip(th, q4["theta_used"])),
                      A_new_sum_rel_error_vs_Q3=q4["A_new_sum_rel_error"], partition_area_rel_error=q4["partition_rel_error"])}
    plans = q5["audit"]["plans"]
    P = {u["uid"]: u["E1"] for u in q4["units"]}
    p = q5["plan_0716"]
    rel = max(abs(d["d"] - p["params"]["q"] * p["params"]["H"] * P[d["uid"]] / 1000) / max(d["d"], 1e-12) for d in p["demand"])
    obs["Q5"] = dict(open_depots_inside_F=sum(x["C14_open_in_F100"] for x in plans),
                     flow_on_cut_or_unreachable_pairs=sum(x["C13_reachability"] for x in plans),
                     demand_rel_error_vs_Q4=rel)
    for pr in led["problems"]:
        if pr["id"] in obs:
            pr["observed"] = obs[pr["id"]]
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(U.to_jsonable(led), fh, ensure_ascii=False, indent=1)


def main():
    q1, q2, q3, q4, q5 = (load(n) for n in range(1, 6))
    out = dict(meta=U.run_meta())
    if q1:
        out["q1"] = dict(label_semantics_accepted=q1["label_semantics"]["accepted_A1"],
                         label1_share_in_W0=q1["label_semantics"]["ndwi>0"]["label1_share_in_W0"],
                         A=dict(**{k: q1["models"]["A"]["pooled_pixel"][k] for k in ("IoU", "F1", "kappa", "OA")},
                                **{k: q1["models"]["A"]["pooled_area"][k] for k in ("RAE", "MAE_A", "DeltaA")}),
                         B=dict(**{k: q1["models"]["B"]["pooled_pixel"][k] for k in ("IoU", "F1", "kappa", "OA", "Precision", "Recall")},
                                **{k: q1["models"]["B"]["pooled_area"][k] for k in ("RAE", "MAE_A", "DeltaA", "median_abs_RAE_t")}),
                         top_driver_abs=q1["drivers"]["abs"]["ranking"][:3], n_hard_tiles=len(q1["hard_tiles"]))
    if q2:
        out["q2"] = dict(table=q2["table"], submission=q2["submission_rule"]["use"],
                         theta=q2["grading"]["theta"], theta_ci90=q2["grading"]["boot_ci90"],
                         coverage=q2["conformal"]["coverage_overall"], grade_acc_M6=q2["grade_mc"]["acc_det_M6"],
                         grade_acc_M0=q2["grade_mc"]["acc_det_M0"],
                         cal_mask_area_km2=q2["q2"]["cal_mask_area_km2"], cal_expected_area_km2=q2["q2"]["cal_expected_area_km2"])
    if q3:
        out["q3"] = {d[4:]: dict(A_new_km2=s["A_new"], A_net_km2=s["A_net"], rho=s["rho"], grade=s["grade_det"],
                                 interval_km2=s["interval_km2"], guard_removed_share=s["guard_removed_share"])
                     for d, s in q3["dates"].items()}
        out["q3"]["A_new_0716_km2"] = q3["dates"][C.MAIN_DATE]["A_new"]
        out["q3"]["s1_scale"] = q3["s1_scale"]
    if q4:
        out["q4"] = dict(n_units=q4["n_units"], n_affected=q4["n_affected"], totals=q4["totals"], weights=q4["weights"],
                         CR=q4["ahp"]["CR"], spearman_dirichlet_median=q4["robustness"]["spearman_median"])
    if q5:
        p = q5["plan_0716"]
        out["q5"] = dict(f1_weighted_unmet=p["f1"], f2_weighted_time_min=p["f2"], n_open_depots=sum(p["eta"]),
                         open_depots=p["open_depots"], served_pop=p["served_pop"], total_demand_t=p["total_demand_t"],
                         voi=q5["voi"], audit_pass=q5["audit"]["PASS"])
    # logic probes (logic contract of MODELING_REPORT.md §20)
    probes = dict(bounds=[], monotonic=[])
    if q3:
        probes["bounds"].append(dict(quantity="q3.A_new_0716_km2", claim="lower",
                                     probe="W0 union -> sar_only (shrinks W0)",
                                     probe_delta_sign=q3["logic_probe"]["probe_delta_sign"]))
    if q5:
        for par, name in (("K", "K_MAX"), ("cap", "CAP_TOWN_T")):
            probes["monotonic"].append(dict(more=name, then="q5.f1_weighted_unmet", observed_sign=q5["monotonic_probes"][par]["sign"],
                                            expect_sign=-1))
    out["logic_probes"] = probes
    # flat keys named as in the logic contract of MODELING_REPORT.md §20
    if q2:
        out["q2.cal_mask_area_km2"] = q2["q2"]["cal_mask_area_km2"]
        out["q2.cal_expected_area_km2"] = q2["q2"]["cal_expected_area_km2"]
        out["observed_model_features"] = ["z", "z_lee", "z_lee_std7", "B", "G", "R", "N", "ndwi", "ndvi", "m"]
    if q3:
        out["q3.A_new_0716_km2"] = q3["dates"][C.MAIN_DATE]["A_new"]
    if q5:
        out["q5.n_open_depots"] = sum(q5["plan_0716"]["eta"])
        out["q5.f1_weighted_unmet"] = q5["plan_0716"]["f1"]
    U.save_json(out, C.FIG / "all_results.json")
    backfill_ledger(q2, q3, q4, q5)
    sens = {}
    if q3:
        sens["q3_grid_dtau_hstar"] = q3["sensitivity_grid"]
        sens["q3_one_at_a_time"] = q3["sensitivity_oat"]
    if q5:
        for r in q5["sensitivity"]:
            sens.setdefault(f"q5_{r['param']}", dict(values=[], f1=[], f2=[], n_open=[], served_pop=[]))
            s = sens[f"q5_{r['param']}"]
            s["values"].append(r["value"]); s["f1"].append(r["f1"]); s["f2"].append(r["f2"])
            s["n_open"].append(r["n_open"]); s["served_pop"].append(r["served_pop"])
    if q4:
        sens["q4_robustness"] = q4["robustness"]
    U.save_json(sens, C.FIG / "sensitivity_results.json")
    print("[summary] figures/all_results.json and figures/sensitivity_results.json written")


if __name__ == "__main__":
    main()
