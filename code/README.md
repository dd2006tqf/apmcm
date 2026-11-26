# Code — 2026 APMCM Problem A (flood recognition → assessment → relief allocation)

Python 3.11.9, CPU works for everything; an NVIDIA GPU (CUDA build of torch) makes Q2 training ~3x faster.

```
pip install -r requirements.txt
python code/main.py --data-dir user_data/a题数据汇总        # all problems
python code/main.py --only 3 4 5                            # reuse Q1/Q2 outputs
```

Data layout expected under `--data-dir`: `Problem A Dataset 1/Dataset/{label information, post-disaster SAR images}`,
`Problem A Dataset 2/Dataset/pre-disaster optical images` (official tiles, Q1/Q2 only) and `公开数据/` (Hengzhou
layers, Q3–Q5 only). The two datasets are never mixed.

| File | Role | Typical time (RTX 4050 / CPU) |
|---|---|---|
| `config.py` | paths, seeds, every assumption switch A1–A16 | – |
| `common.py` | preprocessing (pixel area, rank normalization, Lee filter, indices, split Otsu, metrics) | – |
| `fetch_worldcover.py` | WorldCover 2021 windows for the 1000 official tiles (HTTP COG) | 15 min (network) |
| `problem1.py` | folds, preprocessing, Otsu baseline A, LightGBM baseline B, metrics, FP/FN by land cover, TreeSHAP drivers, LOCO | 20 min |
| `unet.py` | dual-branch gated-fusion U-Net and loss (5.1)–(5.2) | – |
| `problem2.py` | ablation M1–M6, isotonic + area-preserving τ_A, conformal intervals, GMM grading, MC grade probabilities | 60 min GPU |
| `coreg_official.py` | SAR–optical co-registration check of the official tiles | 1 min |
| `problem3.py` | Sentinel-1 co-registration, Sentinel-2 alignment check, Hengzhou transfer, W0/F/W1, areas, guards, transfer checks, sensitivity grid | 30 min |
| `network.py` | flood-cut OSM road graph and Dijkstra | – |
| `problem4.py` | units, overlay indicators, entropy–CRITIC–AHP–TOPSIS, robustness | 10 min |
| `problem5.py` | location–allocation MILP (HiGHS, CBC selectable), Pareto front, value of information, 30 sensitivity solves | 10 min |
| `q5_audit.py` | independent constraint audit of all saved Q5 plans | 2 min |
| `summarize.py` | writes `figures/all_results.json`, `figures/sensitivity_results.json` | – |

Results are written to (结果输出 / generated files): `output/` (results JSON, masks, GeoTIFFs, model checkpoints `output/q2_models/M4_fold*.pt`), `figures/problem_*_results.json`.
Large regenerable caches go to `_tmp/cache/`. Seeds: 2026 (+fold index). If the WorldCover download fails, Q1 uses
the optical index-rule land cover (recorded as `landcover_source`).
