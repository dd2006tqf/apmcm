# RESULTS — 2026 APMCM Problem A (comp-code step)

All numbers below come from the saved result files (`figures/problem_{1..5}_results.json`, identical copies in `output/q{1..5}_results.json`); the RESULTS_KEYS block at the end lists the exact values. Code: `code/` (entry `code/main.py`). Hardware: RTX 4050 (CUDA torch 2.13), CPU for everything else.

## Implementation checklist

| Problem | Method actually run (code) | Status |
|---|---|---|
| Shared | WGS-84 pixel area (3.1); per-tile 2–98 % rank normalization (3.2); Lee 5×5 with ENL estimate (3.3); NDWI/NDVI/BRI; 19 pixel features (`common.py`) | done, 1000/1000 tiles |
| Q1 | blocked 5-fold CV (67 blocks of 5×5 tiles, 1-tile buffer); baseline A split-Otsu − NDWI; baseline B LightGBM; pixel + area metrics; FP/FN by WorldCover stratum; TreeSHAP drivers; leave-one-cluster-out (`problem1.py`) | done |
| Q1 | label-semantics test P1-C7 on all 65.5 M valid pixels; SAR–optical co-registration check (`coreg_official.py`) | done |
| Q2 | dual-branch gated-fusion U-Net (5.1) with Focal + Dice + area loss (5.2); ablation M1–M6; MC dropout (T=10); isotonic + area-preserving τ_A (5.4); split-conformal tile intervals; 4-component GMM grading with 1000 block-bootstrap resamples; MC grade probabilities M=2000 (`problem2.py`, `unet.py`) | done (GPU) |
| Q3 | S1 co-registration to the water reference; physics water detector; W0 union; transfer of the 5 M4 checkpoints (2 MC passes each); REM / ΔVV / MMU guards; A_new, A_net, ρ for 07-04 and 07-16; transfer checks; 30-cell Δτ×h* grid + one-at-a-time cases (`problem3.py`) | done |
| Q4 | 393 place-Voronoi units → 17 zones; 9 overlay indicators; entropy + CRITIC game-theory weights + AHP; TOPSIS; Jenks levels; Dirichlet (M=2000), hazard-only, 3 alternative-F robustness runs (`problem4.py`, `network.py`) | done |
| Q5 | depot-activation + allocation MILP (8.3), lexicographic then ε-constraint (15 points), knee, 07-04→07-16 recourse / value of information, 30 one-at-a-time exact re-solves (28 planned + facility-proxy alternative), independent audit of 19 plans (`problem5.py`, `q5_audit.py`) | done |

## Problem 1 — recognition model and area-oriented evaluation

**Label semantics (P1-C7).** On all 65,491,966 valid pixels, only 2.37 % of label-1 pixels were pre-flood water (NDWI > 0); 2.06 % under the stricter NDWI > 0 ∧ NDVI < 0. Of pre-flood water pixels labelled 0, 70.1 % are SAR-dark after the event. The labels therefore mark *new* inundation (permanent water = 0); A1 accepted for all three W0 definitions.

**Preprocessing.** 1000 tiles (IDs 1001–2000); 9 anisotropic label tiles resampled to the SAR grid; pixel area 74.78–75.41 m² (latitude dependent), mean tile area 4.913 km²; median ENL 31.0. Co-registration check SAR vs optical (491 tiles with 3–60 % optical water): median shift (0, 0) px, 63.3 % within 2 px — no re-registration needed for the official data.

**Folds.** 67 blocks → test folds of 194/200/202/202/202 tiles, flood shares 0.184–0.202; 170–196 buffer tiles removed per fold; disjointness and buffer asserted.

**Held-out metrics (pooled over the 5 test folds, each tile tested once).**

| Model | OA | Precision | Recall | F1 | IoU | κ | RAE | MAE_A (km²/tile) | median \|RAE_t\| |
|---|---|---|---|---|---|---|---|---|---|
| A: split-Otsu − NDWI | 0.634 | 0.308 | 0.740 | 0.435 | 0.278 | 0.227 | +1.406 | 1.756 | 2.84 |
| B: LightGBM (M0) | 0.900 | 0.717 | 0.783 | 0.749 | 0.598 | 0.686 | +0.092 | 0.430 | 0.52 |

Total reference new area of the 1000 tiles: 933.41 km²; LightGBM estimates 1019.12 km² (ΔA = +85.71 km²; Bland–Altman bias +0.086 km², limits −1.42 / +1.60 km²). Baseline A splits land-cover contrasts on tiles without water (Ashman test passes on 92 % of sub-blocks), which explains its large FP; this is reported as a baseline finding, not tuned.

**FP/FN by pre-flood land cover (WorldCover 2021, LightGBM), km².** Tree: ref 33.46, FP 79.80, FN 4.93, net +74.87; Shrub/grass: 53.92 / 23.43 / 7.65 / +15.77; Cropland: 379.62 / 105.59 / 59.09 / +46.50; Built-up: 8.79 / 19.38 / 1.81 / +17.57; Bare: 7.97 / 7.70 / 1.15 / +6.54; Permanent water: 156.06 / 26.80 / 50.91 / −24.11; Wetland: 293.59 / 25.46 / 76.89 / −51.43. The identity Σ(FP−FN) = ΔA holds to rounding. Worst strata by |net|: Tree, Wetland, Cropland. The index-rule land cover is also stored (`fpfn_index_rules`).

**Key factors (TreeSHAP of out-of-fold LightGBM regressors on tile |e_t|, OOF R² = 0.417).** Ranking by mean |SHAP|: raw SAR std (0.196), tree share (0.077), label boundary density (0.074), ENL (0.073), NDVI mean (0.057), bimodality share (0.050), pre-water share (0.045). Signed error is driven first by the reference flood share, then SAR std and boundary density. 397 "hard" tiles (A_t ≥ 0.05 km², |RAE_t| > 0.5) are passed to Q2.

**Leave-one-cluster-out stress test (4 spatial clusters of 512/450/22/16 tiles).** Pooled IoU 0.508, F1 0.674 (vs 0.598 / 0.749 under blocked CV); RAE per cluster +0.26 (512 tiles) and +0.15 (450 tiles). Transfer to an unseen region costs about 0.09 IoU.
## Problem 2 — fused, area-calibrated model and intensity grading

Same 1000 held-out tiles and folds as Q1 (test-set ID hash stored as `test_set_sha256`). U-Net M4: 2.30 M parameters, 40 epochs max, early stopping on the calibration fold.

| Variant | IoU | F1 | κ | RAE | MAE_A | median \|RAE_t\| |
|---|---|---|---|---|---|---|
| A physics | 0.278 | 0.435 | 0.227 | +1.406 | 1.756 | 2.84 |
| M0 LightGBM | 0.598 | 0.749 | 0.686 | +0.092 | 0.430 | 0.52 |
| M1 U-Net SAR only | 0.549 | 0.709 | 0.628 | +0.349 | 0.509 | 0.50 |
| M2 + optical gated fusion | 0.798 | 0.888 | 0.861 | +0.034 | 0.149 | 0.17 |
| M3 + area loss | 0.793 | 0.885 | 0.857 | +0.034 | 0.160 | 0.20 |
| M4 + modality dropout / validity | 0.737 | 0.849 | 0.812 | +0.041 | 0.227 | 0.30 |
| M5 + MC-dropout mean | 0.737 | 0.849 | 0.812 | +0.039 | 0.227 | 0.31 |
| **M6** + isotonic + τ_A (final) | 0.732 | 0.845 | 0.808 | **+0.0145** | 0.236 | 0.31 |

Reading: optical fusion is the decisive step (M1→M2: IoU +0.25, MAE_A −71 %). The area loss does not improve in-domain accuracy (M2≈M3). Modality dropout costs 0.06 IoU in-domain (M3→M4) and is kept because Hengzhou has no usable optics (Q3). Calibration (M6) removes most of the area bias: RAE +4.1 % → +1.45 %, total ΔA 13.52 km² on 933.41 km². On the 397 Q1-hard tiles: IoU 0.266 → 0.562, MAE_A 0.620 → 0.230 km², RAE +0.208 → −0.052. Submission rule (fixed in advance): MAE_A(M6) = 0.236 ≤ MAE_A(M0) = 0.430 → **Q3 uses M6**. Mean fusion gates by scale (weight on SAR features): 0.52 / 0.57 / 0.56 / 0.53.

**Area-preserving threshold.** τ_A per fold 0.708 / 0.515 / 0.388 / 0.690 / 0.574; on the calibration folds Σ mask area = 933.111 km² vs Σ calibrated expected area = 933.112 km² (relative gap 6e-7 < 1e-3).

**Conformal tile intervals (α = 0.1).** Empirical coverage on test tiles 0.878 (folds 0.897 / 0.905 / 0.827 / 0.837 / 0.926; flood-share terciles 0.928 / 0.853 / 0.853); mean width 0.54 km². Coverage is slightly below the nominal 0.90 in two folds (exchangeability between neighbouring folds is imperfect); reported as is.

**Grading thresholds (GMM K = 4 on logit ρ, 985 non-empty tiles).** θ = (0.0593, 0.530, 0.969): general ρ < 5.9 %, relatively strong 5.9–53 %, strong 53–96.9 %, extremely strong ≥ 96.9 %. Area equivalents: reference tile 4.92 km² → 0.29 / 2.61 / 4.77 km²; 1 km² cell → 0.059 / 0.530 / 0.969 km²; Hengzhou 3440.4 km² → 203.9 / 1822.5 / 3333.5 km². Bootstrap 90 % intervals: θ1 [0.0075, 0.079], θ2 [0.067, 0.655], θ3 [0.522, 0.998]; 71 % of resamples give a valid ordered 4-component solution. BIC is lowest at K = 2 (4604.8 vs 4629.5 at K = 4), so the data alone do not prefer four modes; four grades are required by the problem and the Jenks cross-check gives θ = (0.035, 0.371, 0.967), close for θ1 and θ3. Reference grade counts of the 1000 tiles: 575 / 283 / 89 / 53.

**Grade uncertainty (MC, M = 2000 residual × threshold draws).** Deterministic grade accuracy M6 0.862 vs M0 0.543; expected MC accuracy 0.670. Flip probability falls from 0.33–0.37 for tiles within 4 σ′ of a threshold to 0.22 for tiles farther away. Σ_g π_jg = 1 for every tile.
## Problem 3 — Hengzhou (Guangxi), July 2026

**Study area and imagery.** GADM 4.1 boundary of Hengzhou City, geodesic area 3440.09 km² (nominal 3440.4). Sentinel-1 RTC VV/VH: 2026-06-27 (pre-flood baseline), 2026-07-04 and 2026-07-16 (flood time; no pass exists on 6 July). Ancillary: WorldCover 2021, OSM water/waterways/roads/places, Copernicus DEM 30 m, WorldPop 2020.

**Data verification findings (self-collected data).**
1. *Sentinel-1 georeferencing offset.* Against the permanent-water reference (WorldCover 80 ∩ OSM water, two independent sources that agree with each other), the S1 rasters are displaced north–south. Estimated translations (corrected[r,c] = raw[r+dr, c+dc]): 06-27 dr = −623, dc = −1 px; 07-04 dr = −544, dc = 0; 07-16 dr = −539, dc = −1 (≈ 5.4–6.2 km, and 0.8 km between dates). Share of reference water that is SAR-dark (< −15 dB) rises from 0.034 / 0.044 / 0.049 to 0.346 / 0.356 / 0.378 after correction. All Q3–Q5 numbers use the co-registered SAR. Without this step the change guard ΔVV would compare different ground locations.
2. *Sentinel-2 2026-05-31 unusable.* Its NDWI > 0 share on reference water is 0.026 at zero shift and at most 0.041 for any translation within ±800 m × ±320 m (valid share 0.35). The scene cannot be co-registered by a translation, so optics are switched off in Q3 (m = 0 everywhere; the M4/M6 model was trained with modality dropout for exactly this case) and S2 NDWI is dropped from W0.

**Pre-flood water and physics detector.** Split-based Otsu on Lee-filtered VV dB (128-px sub-blocks, Ashman D > 2, minor class ≥ 10 %, dark-class mean ≤ −15 dB); thresholds −12.23 / −12.02 / −12.80 dB (pooled sub-block medians; 245 / 245 / 272 sub-blocks accepted). Pre-flood SAR water 182.8 km²; W0 = SAR ∪ WorldCover 80 ∪ OSM water = 292.57 km². SAR scale choice by KS distance: dB 0.106 vs amplitude 0.241 → dB. Pooled calibration of the 5 members: τ_A = 0.557.

**Main results (inside the boundary).**

| Date | A_new (km²) | A_net (km²) | ρ = A_new/A_reg | Grade (deterministic) | π (general / rel. strong / strong / extreme) |
|---|---|---|---|---|---|
| 2026-07-04 | 24.53 | −54.70 | 0.713 % | general | – |
| **2026-07-16** | **26.90** | **−52.53** | **0.782 %** | general | 0.72 / 0.26 / 0.016 / 0 |

A_net ≤ A_new holds; F ∩ W0 = ∅ and F ⊆ W1 checked on the saved rasters (0 violating pixels). A_net is negative because 79.4 km² of pre-flood SAR water (06-27) is no longer SAR water on 07-16 (72.4 km² of it is not in any mapped water source: wet fields / ponds dark in late June), i.e. the 06-27 baseline itself was a wet-season state. The literal (6.4) definition, which also counts mapped-but-never-SAR-visible channels as receded, gives A_net = −153.19 km² (07-16) and −153.98 km² (07-04); the reported A_net uses observed recession only (W0 pixel dark on 06-27, not dark on the flood date). F_0704 vs F_0716 Jaccard 0.391.

New inundation by pre-flood land cover (07-16): cropland 13.60, tree 9.45, built-up 1.88, grassland 1.53, bare 0.43 km². All of F lies in the REM ≤ 15 m floodplain zone (2079.5 km²; ρ_floodplain 1.29 %).

**Guards and transfer checks (07-16).** Model mask 154.43 km² → after W0 68.85 → after REM 44.74 → after ΔVV 36.98 → after MMU 26.90 km²: the guards remove 82.6 % (07-04: 83.3 %), more than the 70 % flag of the modeling plan, so the Hengzhou estimate relies heavily on the physics guards (stated as a limitation). Mean model probability on permanent water 0.285 (no optics: the model cannot separate permanent water by itself, the W0 guard does). IoU of the unguarded model mask with the physics change mask 0.207 (area ratio 2.60). Ashman D of floodplain VV 1.92 (07-04) and 3.53 (07-16).

**Uncertainty and sensitivity (07-16).** Model interval: comonotone aggregation of the 449 active tile intervals [0, 243.7] km² (uninformative); independent-error aggregation [13.48, 40.33] km² (secondary, not calibrated). Scenario range over Δτ ∈ {−0.2..0.2} × h* ∈ {5..∞}: A_new 1.57–79.52 km²; at Δτ = 0, h* 5/10/15/20/30/∞ → 17.04 / 23.37 / 26.90 / 28.93 / 31.45 / 40.31 km². δ = 0/1.5/3/4.5 dB → 32.40 / 30.05 / 26.90 / 23.59. W0 = SAR only → 30.47 (A_net −48.97); amplitude scale → 49.39; MMU 10/50/100 px → 34.89 / 26.90 / 22.43. The grade stays "general" in every scenario (all ρ < θ1 = 5.9 %). Logic probe: shrinking W0 to SAR-only raises A_new (sign +1), consistent with A_new(union) being a lower bound.

Rasters: `output/q3_newflood_0716.tif`, `_0704.tif`, `q3_w0.tif`, `q3_w1_0716.tif`, `q3_receded_0716.tif`, `q3_prob_0716.tif` (+ 07-04 versions), sensitivity masks `q3_newflood_0716_sar_only.tif`, `q3_newflood_0716_h30.tif` (UTM 49N, 10 m, co-registered grid).
## Problem 4 — regional disaster assessment

**Units.** 393 Voronoi cells of OSM places clipped to the boundary: Σ A_reg = 3440.0919 km² vs boundary 3440.0920 km² (relative error 1.7e-8); 17 town/city zones. Σ unit A_new = 26.9025 km² = Q3 A_new (relative error 4e-9). Affected units (A_new ≥ 0.005 km²): 259.

**Exposure totals (07-16 F).** Flooded population 13,954 (WorldPop 2020, area-weighted cell fractions); flooded built-up 1.88 km²; flooded cropland 13.60 km²; 8 of 22 facility proxies within 500 m of F; 117.4 km of 2542.3 km of road impassable (20 m buffer); 75 affected units have no road path to an unflooded facility proxy (R1 set to 2 × max finite = 165.6 min).

**Weights.** AHP Ω = (0.540, 0.297, 0.163) for hazard / exposure / response, λ_max = 3.0092, CR = 0.0079. Final indicator weights: H1 0.233, H2 0.198, H3 0.108, E1 0.055, E2 0.055, E3 0.047, E4 0.140, R1 0.080, R2 0.084 (Σ = 1). Hazard entropy/CRITIC/game weights: (0.477, 0.404, 0.119) / (0.320, 0.273, 0.407) / (0.432, 0.367, 0.201), α = (0.767, 0.303).

**Index and levels.** Jenks levels of S_j over affected units: Level I 3, II 28, III 96, IV 132. Top units: 1 马岭镇 Maling (S 0.643; A_new 1.46 km², 836 people), 2 旺宅 Wangzhai (Shitang zone, 0.579), 3 新兴 Xinxing (Luancheng zone, 0.515), 4 杨江 Yangjiang (0.446), 5 新张 Xinzhang (Liantang zone, 0.425). Top zones: 石塘镇 Shitang (S 0.787; 4.91 km², 1189 people), 马岭镇 Maling (0.760; 2.96 km², 2305), 峦城镇 Luancheng (0.740; 3.58 km², 2144), 云表镇 Yunbiao (0.648), 校椅镇 Xiaoyi (0.558), 那阳镇 Nayang (0.501).

**Robustness.** Dirichlet weight perturbation (M = 2000): median Spearman to the baseline ranking 0.987 (5 % quantile 0.934). Hazard-only ranking: Spearman 0.792, top-10 overlap 8/10. Alternative inputs: F_0704 Spearman 0.662 (top-10 overlap 6; 243 affected; 14,301 flooded people); W0 = SAR only 0.932 (8); h* = 30 m 0.889 (9). Outputs: `output/q4_units.geojson` (indicators, S_j, level, rank 5–95 % interval, zone), `output/q4_units_0704.csv`.

## Problem 5 — relief depot activation and allocation (decision time 2026-07-16)

**Inputs from Q3/Q4.** Demand d_j = q·H·P_j/1000 with q = 3.5 kg/person/day, H = 3 days; 164 demand units (affected and P_j ≥ 1), total 146.46 t; priority w_j = max(S_j, 0.001). 21 candidate depots (16 towns, 1 city, 4 schools after 300 m merging), 20 admissible (one within 100 m of F); 1189 admissible pairs (t_ij ≤ 120 min) on the flood-cut network; 57 demand units have no admissible depot. Resources: K = 6, 60 t per town/school depot, 200 t city depot, 40 trucks × 5 t, 12 h shifts.

**Recommended plan (lexicographic, HiGHS, status Optimal at both stages, gap 0).** Open 云表镇 Yunbiao (28.82 t), 陶圩镇 Taoxu (39.59 t), 石塘镇 Shitang (5.65 t), 高村小学 Gaocun school (1.97 t), 镇龙乡 Zhenlong (0.66 t), 六景镇 Liujing (0.13 t). f1* = 0.570 (priority-weighted unmet share), f2 = 13.05 min (priority-weighted delivery time); 76.82 t shipped of 146.46 t; 7,316 of the 13,948 people in demand units served (Q4 total 13,954 incl. units below 1 person or not affected). Fleet use 1,497 of 86,400 vehicle-minutes (inactive). The unmet share is caused by reachability, not supply: depot capacity (≥ 360 t) exceeds demand and f1* barely moves with K, capacity, fleet or T_max; it moves with the road buffer (0 m: 0.564; 50 m: 0.650).

**Pareto front (ε-constraint, 15 points).** From (f1, f2) = (0.570, 13.05 min, 7316 served) to (1, 0); knee at k = 3: f1 0.662, f2 4.79 min, 5195 served.

**Information update 07-04 → 07-16.** The 07-04 plan (open 校椅镇, 云表镇, 六景镇, 镇龙乡, 横州镇, 高村小学; f1 0.477 under 07-04 inputs) re-optimized under 07-16 inputs with its depots fixed gives f1 = 0.609, f2 = 12.43 min, 6,775 served. Value of information VoI1 = +0.0387 (unmet share), VoI2 = +5.54 min at equal f1; open-set Jaccard 0.50; demand change −3.57 t; 677 admissible pairs lost and 625 gained.

**Sensitivity (30 exact re-solves, all Optimal).** K = 3..8: f1 0.5728 → 0.5699 (non-increasing), f2 22.24 → 9.32 min. Capacity × 0.5..1.5: f1 0.5699 constant. T_max 60/90/120/180/240: f1 0.5768 / 0.5701 / 0.5699 / 0.5699 / 0.5699. q 2.5/3.5/7/15.6 kg: f1 0.5699 / 0.5699 / 0.5699 / 0.5701. Speed factor 0.6/0.8/1.0: f1 0.5705 / 0.5701 / 0.5699. Fleet 20/40/60: unchanged. Facility proxies = schools only (A12 alternative, 4 candidates after merging): f1 0.611, f2 21.08 min, 4 depots open.

**Independent audit (`code/q5_audit.py`).** 19 plans (recommended, 07-04, recourse, knee, 15 Pareto points) re-checked from raw inputs: C10 balance, C11 capacity, C12 K, C13 reachability by fresh Dijkstra, C14 depot clearance, fleet budget, C15 demand recomputed from `q4_units.geojson` — 0 violations in every plan.
## Differences between the code and MODELING_REPORT.md (also in UPSTREAM_ISSUES.md)

1. **U-Net early stopping** uses the calibration tile MAE of area instead of |Σ ΔA|: in a first run the net sum let FP and FN cancel and kept an untrained epoch-0 model. Calibration data only; test folds untouched.
2. **cuDNN determinism off** (seeds fixed): deterministic kernels were 3× slower (0.48 vs 0.15 s/iteration). Re-runs can differ at kernel-rounding level.
3. **Sentinel-1 co-registration** (translation 5.4–6.2 km) added before every Hengzhou computation; **Sentinel-2 dropped** (cannot be co-registered). Optics in Q3 are therefore missing everywhere and W0 has no S2 term.
4. **S1 physics detector**: 128-px sub-blocks and an extra acceptance rule (dark-class mean ≤ −15 dB). With the literal rule (64 px, no physical check) 52 % of the city was labelled water at −6.9 dB.
5. **A_net** uses observed recession (W0 pixel SAR-dark before, not after); the literal (6.4) value is reported next to it.
6. **MILP solver**: HiGHS (exact B&B, gapRel = 0). CBC 2.10 stalled for 20 min without CPU progress on one sensitivity case. CBC remains selectable (`Q5_SOLVER=cbc`).
7. **Q5 demand units** = affected units with P_j ≥ 1; priority floor w_j ≥ 0.001 so that the TOPSIS-minimum unit is not ignored.
8. **Q1 LOCO** uses the median fold threshold and median fold iteration count (no calibration fold exists for a whole held-out cluster).
9. **Q3 interval**: the comonotone aggregation prescribed by the plan is uninformative over 449 tiles; an independent-error aggregation is added as a labelled secondary interval.
10. Unit-level σ′_j in Q4 is allocated from the Q3 tile σ′_t in proportion to the tile's new area inside the unit (rule not specified in the plan).

## Validation evidence

- Q1–Q5 each end with `validate_capability()` (raises on failure) and `validate_constraints()`; all passed in the final runs (logs `_tmp/logs/q{1..5}.log`).
- Q2: identical test sets for M0–M6; (5.4) equality within 6e-7; Σπ = 1.
- Q3: F ∩ W0 = ∅, F ⊆ W1, A_net ≤ A_new, ρ = A_new / A_reg on saved rasters; checkpoint hashes equal to Q2 (0 mismatches); thresholds equal to Q2.
- Q4: partition and A_new sum checks; weights ≥ 0, Σ = 1; CR < 0.1; 0 ≤ S_j ≤ 1.
- Q5: 19-plan independent audit, 0 violations; every recorded MILP solve (recommended, 07-04, recourse, 15 Pareto points, 30 sensitivity cases) returned Optimal.
- Cross-problem ledger (`CROSS_PROBLEM_LEDGER.json`, observed values back-filled from results): 8/8 downstream conditions satisfied (`_utils/cross_problem_check.py` exit 0). Logic audit exit 0 (bound probe sign +1; K and capacity monotone probes non-increasing).
- Evaluation independence (leakage check; hold-out / out-of-sample evaluation against the official ground-truth labels; 标签源特征未进入评估模型): every Q1/Q2 classification metric is computed on the held-out test fold of a spatially blocked split (independent test set with real labels, 1-tile buffer, thresholds/early stopping/isotonic/conformal fitted on the separate calibration fold, per-tile normalization uses only the tile itself). No weak or pseudo labels are generated anywhere: the Hengzhou guards filter the transferred model output and never train a model, and the generating features of no label enter an evaluated model. The values f1 = 1 in `problem_5_results.json::pareto[14]` are the Q5 weighted unmet share at the ε = 1 end of the Pareto front (nothing delivered), not a classification score.
- Static method scan (`claim_code_check.py`): 1 WARN on M7 (the forbidden-pattern fragment `0.3` matches unrelated constants such as the S2 usability threshold 0.3; GMM thresholds are data-derived, `grading.rule = gmm`).

## Files

Code: `code/config.py, common.py, fetch_worldcover.py, problem1.py, coreg_official.py, unet.py, problem2.py, problem3.py, network.py, problem4.py, problem5.py, q5_audit.py, summarize.py, main.py`.
Results: `figures/problem_{1..5}_results.json`, `figures/all_results.json`, `figures/sensitivity_results.json`, `output/` (see `DELIVERABLES.json`).
Figure data for the planned figures: per-tile areas and errors (`problem_1/2 … A_ref_t, per_tile`), SHAP values (`drivers`), training histories (`training_logs`), grade probabilities (`grade_mc`), Q3 rasters and sensitivity grid, Q4 unit GeoJSON and Dirichlet ranks, Q5 flow paths (`paths_0716`), cut roads GeoJSON, Pareto points.

<!-- RESULTS_KEYS（由 _utils/results_consistency_check.py --emit 生成；数值变了要重新生成，勿手改）
problem_1_results.json::models.A.pooled_pixel.OA = 0.63398877047
problem_1_results.json::models.A.pooled_pixel.Precision = 0.307582019021
problem_1_results.json::models.A.pooled_pixel.Recall = 0.74003032168
problem_1_results.json::models.A.pooled_pixel.F1 = 0.434550093839
problem_1_results.json::models.A.pooled_pixel.IoU = 0.277587990602
problem_1_results.json::models.A.pooled_pixel.kappa = 0.227002563613
problem_1_results.json::models.A.pooled_area.A_ref_total = 933.408084246
problem_1_results.json::models.A.pooled_area.A_hat_total = 2245.65886157
problem_1_results.json::models.A.pooled_area.DeltaA = 1312.25077733
problem_1_results.json::models.A.pooled_area.RAE = 1.40587037918
problem_1_results.json::models.A.pooled_area.MAE_A = 1.75614992624
problem_1_results.json::models.A.pooled_area.median_abs_RAE_t = 2.83915619685
problem_1_results.json::models.A.pooled_area.BA_bias = 1.31225077733
problem_1_results.json::models.A.pooled_area.BA_loa_low = -1.75547915744
problem_1_results.json::models.A.pooled_area.BA_loa_high = 4.3799807121
problem_1_results.json::models.A.pooled_area.n_tiles = 1000
problem_1_results.json::models.B.pooled_pixel.OA = 0.900077285815
problem_1_results.json::models.B.pooled_pixel.Precision = 0.71719305445
problem_1_results.json::models.B.pooled_pixel.Recall = 0.782961690558
problem_1_results.json::models.B.pooled_pixel.F1 = 0.748635683402
problem_1_results.json::models.B.pooled_pixel.IoU = 0.598255578708
problem_1_results.json::models.B.pooled_pixel.kappa = 0.686430089254
problem_1_results.json::models.B.pooled_area.A_ref_total = 933.408084246
problem_1_results.json::models.B.pooled_area.A_hat_total = 1019.12006723
problem_1_results.json::models.B.pooled_area.DeltaA = 85.7119829887
problem_1_results.json::models.B.pooled_area.RAE = 0.0918269130462
problem_1_results.json::models.B.pooled_area.MAE_A = 0.430139517697
problem_1_results.json::models.B.pooled_area.median_abs_RAE_t = 0.523112253933
problem_1_results.json::models.B.pooled_area.BA_bias = 0.0857119829887
problem_1_results.json::models.B.pooled_area.BA_loa_low = -1.4242145414
problem_1_results.json::models.B.pooled_area.BA_loa_high = 1.59563850738
problem_1_results.json::models.B.pooled_area.n_tiles = 1000
problem_1_results.json::label_semantics.accepted_A1 = true
problem_1_results.json::folds.fold_sizes[0] = 194
problem_1_results.json::folds.fold_sizes[1] = 200
problem_1_results.json::folds.fold_sizes[2] = 202
problem_1_results.json::folds.fold_sizes[3] = 202
problem_1_results.json::folds.fold_sizes[4] = 202
problem_1_results.json::loco.pooled_pixel.OA = 0.862052591306
problem_1_results.json::loco.pooled_pixel.Precision = 0.611963006488
problem_1_results.json::loco.pooled_pixel.Recall = 0.749191502217
problem_1_results.json::loco.pooled_pixel.F1 = 0.673659722243
problem_1_results.json::loco.pooled_pixel.IoU = 0.507908666833
problem_1_results.json::loco.pooled_pixel.kappa = 0.587325329915
problem_2_results.json::table.M6.IoU = 0.731560874586
problem_2_results.json::table.M6.F1 = 0.844972747217
problem_2_results.json::table.M6.kappa = 0.808269804738
problem_2_results.json::table.M6.OA = 0.940643070633
problem_2_results.json::table.M6.Precision = 0.838866680856
problem_2_results.json::table.M6.Recall = 0.851168356828
problem_2_results.json::table.M6.RAE = 0.0144890326387
problem_2_results.json::table.M6.MAE_A = 0.236304830307
problem_2_results.json::table.M6.median_abs_RAE_t = 0.313066118734
problem_2_results.json::table.M6.DeltaA = 13.5241801979
problem_2_results.json::table.M2.IoU = 0.79787006316
problem_2_results.json::table.M2.F1 = 0.887572555447
problem_2_results.json::table.M2.kappa = 0.860638319008
problem_2_results.json::table.M2.OA = 0.956530118519
problem_2_results.json::table.M2.Precision = 0.872776782027
problem_2_results.json::table.M2.Recall = 0.90287863133
problem_2_results.json::table.M2.RAE = 0.0344291015431
problem_2_results.json::table.M2.MAE_A = 0.148725741358
problem_2_results.json::table.M2.median_abs_RAE_t = 0.169487153326
problem_2_results.json::table.M2.DeltaA = 32.1364017136
problem_2_results.json::table.M4.IoU = 0.736947784301
problem_2_results.json::table.M4.F1 = 0.848554908745
problem_2_results.json::table.M4.kappa = 0.812125830571
problem_2_results.json::table.M4.OA = 0.941244976521
problem_2_results.json::table.M4.Precision = 0.831686119848
problem_2_results.json::table.M4.Recall = 0.866122151216
problem_2_results.json::table.M4.RAE = 0.0412575578158
problem_2_results.json::table.M4.MAE_A = 0.227270907225
problem_2_results.json::table.M4.median_abs_RAE_t = 0.303688630278
problem_2_results.json::table.M4.DeltaA = 38.5101380016
problem_2_results.json::table.M1.IoU = 0.549255715107
problem_2_results.json::table.M1.F1 = 0.709057529692
problem_2_results.json::table.M1.kappa = 0.62781204218
problem_2_results.json::table.M1.OA = 0.870110449883
problem_2_results.json::table.M1.Precision = 0.617312616596
problem_2_results.json::table.M1.Recall = 0.832833060433
problem_2_results.json::table.M1.RAE = 0.349057792144
problem_2_results.json::table.M1.MAE_A = 0.509144237368
problem_2_results.json::table.M1.median_abs_RAE_t = 0.500455236692
problem_2_results.json::table.M1.DeltaA = 325.813365056
problem_2_results.json::grading.theta[0] = 0.0592651577161
problem_2_results.json::grading.theta[1] = 0.529753493914
problem_2_results.json::grading.theta[2] = 0.96893016657
problem_2_results.json::conformal.coverage_overall = 0.878
problem_2_results.json::grade_mc.acc_det_M6 = 0.862
problem_2_results.json::grade_mc.acc_det_M0 = 0.543
problem_2_results.json::submission_rule.MAE_A_M6 = 0.236304830307
problem_2_results.json::submission_rule.MAE_A_M0 = 0.430139517697
problem_2_results.json::submission_rule.use = M6
problem_2_results.json::q2.cal_mask_area_km2 = 933.111053467
problem_2_results.json::q2.cal_expected_area_km2 = 933.111587524
problem_3_results.json::dates.20260716.A_new = 26.9025316667
problem_3_results.json::dates.20260716.A_net = -52.5341092017
problem_3_results.json::dates.20260716.rho = 0.00782029433234
problem_3_results.json::dates.20260704.A_new = 24.5302441418
problem_3_results.json::dates.20260704.A_net = -54.697048333
problem_3_results.json::dates.20260704.rho = 0.00713069430082
problem_3_results.json::dates.20260716.interval_independent_km2[0] = 13.4792701532
problem_3_results.json::dates.20260716.interval_independent_km2[1] = 40.3257931801
problem_3_results.json::dates.20260716.guard_removed_share = 0.825798149357
problem_3_results.json::coregistration.20260627.dr = -623
problem_3_results.json::coregistration.20260627.dc = -1
problem_3_results.json::coregistration.20260627.score_before = 0.0344416666667
problem_3_results.json::coregistration.20260627.score_after = 0.34605
problem_3_results.json::coregistration.20260627.shift_north_m = 6230
problem_3_results.json::coregistration.20260627.shift_east_m = -10
problem_3_results.json::coregistration.20260704.dr = -544
problem_3_results.json::coregistration.20260704.dc = 0
problem_3_results.json::coregistration.20260704.score_before = 0.04395
problem_3_results.json::coregistration.20260704.score_after = 0.355625
problem_3_results.json::coregistration.20260704.shift_north_m = 5440
problem_3_results.json::coregistration.20260704.shift_east_m = 0
problem_3_results.json::coregistration.20260716.dr = -539
problem_3_results.json::coregistration.20260716.dc = -1
problem_3_results.json::coregistration.20260716.score_before = 0.048525
problem_3_results.json::coregistration.20260716.score_after = 0.377783333333
problem_3_results.json::coregistration.20260716.shift_north_m = 5390
problem_3_results.json::coregistration.20260716.shift_east_m = -10
problem_3_results.json::model.tauA = 0.557362060547
problem_4_results.json::totals.flooded_pop = 13954.1799918
problem_4_results.json::totals.flooded_built_km2 = 1.87638583867
problem_4_results.json::totals.flooded_crop_km2 = 13.6026454105
problem_4_results.json::totals.exposed_facilities = 8
problem_4_results.json::totals.cut_road_km = 117.434419446
problem_4_results.json::totals.total_road_km = 2542.32833007
problem_4_results.json::n_affected = 259
problem_4_results.json::ahp.CR = 0.00793337302955
problem_4_results.json::robustness.spearman_median = 0.986605631954
problem_5_results.json::plan_0716.f1 = 0.569927905096
problem_5_results.json::plan_0716.f2 = 13.0546653947
problem_5_results.json::plan_0716.served_pop = 7315.81048159
problem_5_results.json::plan_0716.total_demand_t = 146.45674371
problem_5_results.json::plan_0716.total_shipped_t = 76.8160100567
problem_5_results.json::unreachable_units = 57
problem_5_results.json::voi.VoI1 = 0.0387108392923
problem_5_results.json::voi.VoI2 = 5.54050442997
problem_5_results.json::plan_0704.f1 = 0.477359194007
-->

Remaining open capability item: P5-C4 (recommendation letter) is written in the paper step from `problem_5_results.json` and `problem_3_results.json`; it is not claimed as done here.

<!-- METHOD_CHECK static_cc=0 n_warn=1 n_claims=11 n_implemented=11 fast_mode=0 note=M7_warn_is_unrelated_constant_0.3 -->
<!-- AUDIT_OK source=results.json rechecked_at=2026-11-26T18:18:50+08:00 checks=33 n_violations=0 script=code/constraint_audit.py baselines=A,B,M0 q5_plans=19 -->
