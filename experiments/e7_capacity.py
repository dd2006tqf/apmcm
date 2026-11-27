"""E7 — capacity audit from the stored Q2 training logs (no retraining).

Reports parameter counts, epochs actually run against the 40-epoch budget, best-epoch position and
the fraction of folds that stopped early. This is the evidence used for the '1,000 tiles is small'
discussion: whether the limit is data volume or model/optimisation capacity.
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "code"))
import config as C

RES = os.path.join(_HERE, "results")
os.makedirs(RES, exist_ok=True)


def main():
    q = json.load(open(C.OUT / "q2_results.json"))
    logs = q["training_logs"]
    rows = []
    for k, v in logs.items():
        h = v.get("history", [])
        best_ep = min(h, key=lambda e: e["cal_MAE_A_km2"])["epoch"] if h else None
        rows.append(dict(run=k, n_params=v.get("n_params"), epochs_run=v.get("epochs_run"),
                         epochs_max=q["epochs_max"], best_epoch=best_ep, best_cal_MAE=v.get("best"),
                         seconds=v.get("seconds")))
    df = {r["run"]: r for r in rows}
    params = sorted({r["n_params"] for r in rows if r["n_params"]})
    early = [r for r in rows if r["epochs_run"] < q["epochs_max"]]
    out = dict(epochs_max=q["epochs_max"], n_runs=len(rows), n_params=params,
               mean_epochs_run=sum(r["epochs_run"] for r in rows) / len(rows),
               frac_stopped_early=len(early) / len(rows),
               frac_budget_used=sum(r["epochs_run"] for r in rows) / (len(rows) * q["epochs_max"]),
               runs=rows,
               note="All variants share the same 4-level U-Net (widths 16/32/64/128, bottleneck 256); "
                    "early stopping is patience 8 on calibration MAE.")
    json.dump(out, open(os.path.join(RES, "e7_capacity.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
