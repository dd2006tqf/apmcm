# Modeling Report — 2026 APMCM Problem A

Flood disaster assessment and emergency rescue optimization based on image recognition. Five chained problems: Q1 recognizer and metrics → Q2 fused, area-calibrated model and grading → Q3 Hengzhou transfer → Q4 unit-level assessment → Q5 supply allocation MILP and letter.

Inputs read in full: `PROBLEM_ANALYSIS.md`, `DATA_FACTS.json`, `DATA_PROFILE.json`, `CAPABILITY_CHECKLIST.json`, and the problem text. Three facts were re-checked during this step with one-off scripts in `_tmp/` (`fold_probe.py`, `road_probe.py`):

- The 1000 official tiles lie on a regular grid of 199 × 42 tile positions (each 256 px ≈ 2.08 km E–W × 2.37 km N–S ≈ 4.92 km², at a pixel scale of 8.333e-5°). Grouping them into 5 × 5-tile blocks gives 67 blocks. A greedy stratified assignment of the blocks to 5 folds gives fold sizes 194–210 tiles and fold flood shares 0.173–0.202. A 1-tile (Chebyshev) buffer removes 134–155 tiles from each training set.
- The Hengzhou DEM mosaic (108.6–109.8°E, 22.3–23.3°N) has no zero or no-data pixels inside the boundary. The Sentinel-1 grid (UTM 49N, x 272–359 km, y 2486–2561 km, no-data −999.9) encloses the boundary.
- The OSM road graph built by vertex snapping (1 m) has 11 components. The largest holds 2532.8 of 2542 km. Of the 393 places, 90% lie within 1.44 km of a road node and the maximum distance is 6.46 km.

## 0. Review of the analysis-stage upgrade conclusions

Every upgrade in §5.7 of the analysis is adopted. The table lists where each one is implemented.

| Analysis upgrade | Trigger in problem text | Decision | Where |
|---|---|---|---|
| water segmentation → change segmentation $W_1\setminus W_0$ | "newly inundated" (S8, S18) | adopted. Labels are new inundation (P1-C7 re-test), so Q1/Q2 learn $F$ directly. Q3 additionally enforces $F\cap W_0=\emptyset$ with independent $W_0$ | §3, §4, §6 |
| pixel metrics → area-oriented loss and estimator | "area error / relative area error" (S8, S12) | adopted: area-consistency loss, calibrated probability-sum area, area-preserving threshold | §5.2–5.4 |
| deterministic → probabilistic grading | "effects of errors and uncertainties" (S15) | adopted: conformal residual Monte Carlo gives grade probabilities $\pi_{jg}$ | §5.6–5.7 |
| in-domain → domain adaptation with physical guards | different sensor in Q3 (S16) | adopted: the same per-tile rank normalization, a log/linear choice by KS distance, optical quantile mapping, modality dropout, REM/change/MMU guards | §6 |
| static TOPSIS → hazard–exposure–response with overlay | "exposure of populations and critical facilities" (S23) | adopted: every exposure indicator is an overlay with the Q3 $F$ | §7 |
| plain LP → flood-cut network with Q4 demand | "directly incorporated" (S26) | adopted: edges and depots are removed by $F$, and demand and priority come from Q4 | §8 |
| single → bi-objective (ε-constraint) | efficiency vs coverage | adopted (suggested level) | §8.4 |

Departures from the analysis recommendations, with reasons:

1. **HAND.** The analysis proposed HAND from flow routing. No hydrological routing library is bundled, and the DEM is a 30 m DSM (it includes canopy and buildings). We use the **relative elevation above the nearest mapped drainage (REM)**, which is a Euclidean-nearest variant of HAND. The drainage is OSM waterways ∪ OSM water polygons ∪ WorldCover water. It is a guard, not a predictor, and its threshold is scanned (§6.6). The paper calls it REM, not HAND.
2. **Units for Q4.** The analysis proposed 16 town-Voronoi units. Seventeen units are too few for objective weighting and for a meaningful MILP. We use **place-catchment units**, the Voronoi cells of all 393 OSM places clipped to the boundary. They partition the boundary exactly and are aggregated to the 17 town/city zones for reporting.
3. **Jenks natural breaks.** These are kept only as a one-sentence cross-check of the GMM thresholds. They are not a second claimed method and get no figure of their own.
4. **"Two-stage vs end-to-end" comparison.** The two-stage method is the physics baseline A (Otsu-SAR minus NDWI). We do not train a learned $W_1$ model, because no $W_1$ labels exist.

## 1. Model assumptions (parameterized)

Each assumption is a switch or constant in `code/config.py` (to be created by comp-code). "Observed" means it comes from data and "assumed" means it is a parameter that gets a sensitivity analysis (S4).

**A1. The official labels encode new inundation (permanent water = 0).**
- Reason: of label-1 pixels, only 2.37% were pre-water (NDWI > 0). Of pre-water pixels, 71.2% are SAR-dark after the flood, yet only 4.87% are labelled 1 (full 65.5 M px).
- Parameterized: `LABEL_SEMANTICS = "new"`. P1-C7 re-tests it with three stricter $W_0$ definitions.
- Alternative: `"post_water"` would train on labels as $W_1$ and subtract $\hat W_0$. It is kept as physics baseline A only.

**A2. Tiles within one 5 × 5-tile block (≈ 10.4 × 11.9 km) are dependent, and blocks are exchangeable.**
- Reason: floods and land cover are spatially autocorrelated. A buffer of 1 tile removes edge leakage.
- Parameterized: `BLOCK = 5`, `N_FOLDS = 5`, `BUFFER_TILES = 1`, `FOLD_SEED = 2026`.
- Alternative: leave-one-cluster-out (4 clusters of 512/450/22/16 tiles) is too unbalanced to estimate errors. It is reported as a stress test only (§9).

**A3. Pixel ground area follows the WGS-84 ellipsoid for EPSG:4326 tiles and the UTM scale factor for Hengzhou.** This is exact geometry, not an assumption, and is listed for traceability.
- Parameterized: `AREA_MODE = "ellipsoid"`.
- Alternative: a constant 75 m² gives a ≤ 1% error over 28.4–29.3°N. It is not used.

**A4. Land cover of the official tiles is unknown (assumed role).**
- Primary source: ESA WorldCover 2021 v200, read windowed over HTTP (COG `/vsicurl/` from the public `esa-worldcover` S3 bucket, tiles N27E111 and N27E114) and resampled by nearest neighbour to each tile grid. The acquisition year of the official imagery is not given, so a possible date mismatch with WorldCover 2021 is a stated assumption.
- Fallback (if the download fails, recorded in results): index rules on the pre-flood optical image, with water NDWI > 0, dense vegetation NDVI > 0.45, cropland/sparse vegetation 0.15 < NDVI ≤ 0.45, built/bare NDVI ≤ 0.15.
- Parameterized: `LANDCOVER_SOURCE = "worldcover" | "index_rules"`.
- Alternative: the other source, run as a sensitivity check of the FP/FN table.

**A5. The GF-3 uint8 DN is a monotone function of backscatter; its radiometric scale (linear or dB) is unknown.**
- Reason: no calibration metadata is supplied.
- Consequence: the model sees only per-tile rank-normalized SAR. For Sentinel-1 the domain choice (dB vs linear amplitude before normalization) is made by the KS distance to the training distribution (§6.2).
- Parameterized: `S1_SCALE = "auto" | "db" | "amplitude"`.

**A6. The Hengzhou flood is observed by the two available Sentinel-1 passes, 2026-07-04 (2 days before the 6 July reference date) and 2026-07-16 (10 days after). The 06-27 pass is pre-flood.**
- Reason: no pass exists on the peak date.
- Parameterized: `FLOOD_DATES = ["20260704", "20260716"]`, `PRE_DATE = "20260627"`.
- Alternative: if the user later supplies a pass between 07-05 and 07-12, it is added to the list with no code change.

**A7. Pre-existing water is the union of independent sources.** These are the 06-27 SAR water (physics Otsu), WorldCover class 80, OSM water polygons and S2 05-31 NDWI > 0 on valid pixels.
- Reason: the union is conservative, because it never counts a permanent water pixel as new.
- Parameterized: `W0_MODE = "union" | "sar_only"`.
- Alternative: `"sar_only"` is a sensitivity case.

**A8. REM guard: new inundation requires REM ≤ h* (default h* = 15 m).**
- Reason: floodplain water cannot persist far above the drainage. A 30 m DSM has metre-level vertical noise and canopy bias, so the guard is loose.
- Parameterized: `REM_MAX = 15` m, scanned {5, 10, 15, 20, 30, ∞}.

**A9. Change guard: a new-inundation pixel must darken relative to the pre-flood pass, $\Delta VV = VV_d - VV_{0627} \le -\delta$, with default δ = 3 dB.**
- Reason: δ = 3 dB is the common change-detection margin for open-water appearance, and it removes permanently dark surfaces (shadow, smooth tarmac) that are the main Q1 FP mechanism.
- Parameterized: `DELTA_DB = 3.0`, scanned {0, 1.5, 3, 4.5}.

**A10. Minimum mapping unit (MMU): connected components (8-neighbour) smaller than 50 px (0.5 ha at 10 m) are removed in Q3.** The official tiles use 10 px (≈ 0.075 ha) in the physics baseline only.
- Reason: this removes speckle-scale false detections and matches the 20 m effective S1 resolution (≥ 12 independent looks).
- Parameterized: `MMU_PX_S1 = 50`, `MMU_PX_TILE = 10`.

**A11. Population is static at WorldPop 2020 levels and spatially uniform inside each 100 m cell.**
- Reason: no 2026 census grid exists.
- Parameterized: `POP_SCALE = 1.0` (sensitivity 0.9–1.1).

**A12. Each town and city place node hosts a township health centre and can serve as a relief depot or shelter.**
- Reason: OSM lists only 5 schools. Chinese policy places one township health centre per town (assumption, stated).
- Parameterized: `FACILITY_PROXY = "town_nodes"`.
- Alternative: schools only (5 nodes), a sensitivity case.

**A13. Road speeds by OSM class.** motorway 80, trunk 60, primary 50, secondary 40, tertiary 30, unclassified/residential/service/living_street 20, track 15 km/h; links at the parent class speed minus 20 km/h (floor 20 km/h).
- Reason: design speeds of JTG B01-2014 reduced for emergency driving.
- Parameterized: `SPEED_FACTOR = 1.0`, scanned 0.6–1.0.

**A14. A road edge is impassable if any of its sample points (one every 20 m) lies within a 20 m buffer of $F$.**
- Reason: the flood mask is at 10–20 m resolution, and partial inundation of a carriageway blocks trucks.
- Parameterized: `ROAD_BUFFER_M = 20`.
- Alternative: 0 m buffer, a sensitivity case.

**A15. Relief need per affected person q = 3.5 kg/day** (3 L survival drinking water + about 0.5 kg ration for 2100 kcal, Sphere Handbook 2018 minimums), over horizon H = 3 days.
- Parameterized: `Q_KG = 3.5` (range 2.5–15.6, where 15.6 kg is the full 15 L Sphere water standard plus food), `HORIZON_D = 3`.

**A16. Depot resources (all assumed).** Capacity C = 60 t for a town depot and 200 t for the city depot; at most K = 6 open depots; fleet N_v = 40 trucks × 5 t; shift length 12 h; loading/unloading time 30 min per trip; delivery time limit T_max = 120 min; off-road access leg ≤ 5 km at 15 km/h.
- Parameterized: `CAP_TOWN_T`, `CAP_CITY_T`, `K_MAX`, `N_VEH`, `Q_VEH_T`, `SHIFT_H`, `T_LOAD_MIN`, `T_MAX_MIN`, `ACCESS_MAX_KM`, `ACCESS_KMH`.
- Each one is scanned in §8.6.

## 2. Symbols and units

Symbol description table (符号说明):

| Symbol | Meaning | Unit | Range | First used |
|---|---|---|---|---|
| $t$, $p$ | tile index, pixel index | – | $t\in\{1001..2000\}$ | §3 |
| $a_p$ | true ground area of pixel $p$ | m² | ≈ 74.6–75.6 (tiles); ≈ 100 (S1) | (3.1) |
| $I_p$ | raw SAR value (GF-3 DN or S1 dB) | DN / dB | [0,255] / [−30,10] | §3 |
| $z_p$ | per-tile rank-normalized SAR | – | [0,1] | (3.2) |
| $\tilde z_p$ | Lee-filtered $z_p$ | – | [0,1] | (3.3) |
| $B,G,R,N$ | pre-flood optical bands | DN | [0,255] | §3 |
| $\mathrm{NDWI},\mathrm{NDVI}$ | water / vegetation index | – | [−1,1] | (3.4) |
| $m_p$ | optical validity (1 = valid) | – | {0,1} | §3 |
| $y_p$ | reference label (new inundation) | – | {0,1} | §4 |
| $W_0,W_1,F$ | pre-flood water, flood-time water, new inundation | pixel set | – | (4.1) |
| $\hat P_p$ | model probability of new inundation | – | [0,1] | §4 |
| $\tau$, $\tau_A$ | F1-optimal threshold (Q1); area-preserving threshold (Q2) | – | (0,1) | §4.2, (5.4) |
| $A_t,\hat A_t$ | reference and estimated new area of tile $t$ | km² | [0, 4.93] | (4.2) |
| $A_{new},A_{net}$ | new and net inundated area | km² | $A_{net}\le A_{new}$ | (4.1) |
| $A_{reg}$ | total area of study region / unit | km² | >0 | (4.1) |
| $\rho$ | new-inundation proportion $A_{new}/A_{reg}$ | – | [0,1] | (4.1) |
| RAE | relative area error | – | ≥ −1 | (4.3) |
| $q(\cdot)$ | isotonic calibration map | – | [0,1] | (5.3) |
| $q$, $H$ | relief need per person-day; horizon | kg, day | 3.5, 3 | (8.1) |
| $f_1,f_2$ | weighted unmet share; weighted delivery time | –, min | [0,1], ≥0 | (8.2) |
| $\sigma_t$, $\sigma'_t$ | MC-dropout std of tile area; stabilized std | km² | ≥ 0 | (5.5), (5.6) |
| $s_t$, $\hat q_{1-\alpha}$ | conformal score and its quantile | – | ≥ 0 | (5.6) |
| $\theta_1<\theta_2<\theta_3$ | proportion thresholds of the 4 grades | – | (0,1) | (5.8) |
| $G$, $\pi_{jg}$ | grade (1–4, 0 = none); probability unit $j$ is in grade $g$ | – | $\sum_g\pi_{jg}=1$ | (5.9) |
| $VV_d$, $\Delta VV$ | S1 VV backscatter on date $d$; change vs 06-27 | dB | – | §6 |
| $\mathrm{REM}_p$ | relative elevation above nearest drainage | m | ≥ 0 | (6.1) |
| $h^*,\delta$ | REM and change guard thresholds | m, dB | 15, 3 | A8, A9 |
| $j$ | Q4 assessment unit (place catchment) | – | 1..393 | §7 |
| $x_{jk}$, $r_{jk}$ | raw and normalized indicator $k$ of unit $j$ | mixed / – | – | (7.1) |
| $w_k$, $\Omega_c$ | indicator weight; criterion weight | – | $\sum=1$ | (7.4) |
| $S_j$ | TOPSIS closeness (disaster index) | – | [0,1] | (7.5) |
| $P_j$ | flooded population | persons | ≥ 0 | (7.1) |
| $d_j$ | relief demand of unit $j$ | t | ≥ 0 | (8.1) |
| $w_j$ | priority weight $=S_j$ | – | [0,1] | §8.1 |
| $i$ | candidate depot | – | ~22 | §8 |
| $\eta_i$ | depot $i$ open | – | {0,1} | (8.3) |
| $f_{ij}$, $u_j$ | delivered and unmet relief | t | ≥ 0 | (8.3) |
| $t_{ij}$ | one-way travel time on the flood-cut network | min | ≥ 0 or ∞ | (8.2) |
| $C_i,K,T_{max}$ | depot capacity, max open depots, time limit | t, –, min | A16 | (8.3) |
| $N_v,Q_v,T_s,t_l$ | trucks, truck payload, shift length, load time | –, t, min, min | A16 | (8.3) |

## 3. Shared preprocessing and area definition (all problems)

**P1-C1 image preprocessing.** It runs on all 1000 tiles and logs every processed ID. The same functions are reused unchanged in Q3.

**Pixel area (C8).** For EPSG:4326 pixels with centre latitude $\varphi_p$ and angular sizes $\Delta\lambda,\Delta\varphi$ (radians), with WGS-84 semi-major axis $a=6378137$ m and eccentricity $e^2=0.00669438$:

$$
M(\varphi)=\frac{a(1-e^2)}{(1-e^2\sin^2\varphi)^{3/2}},\qquad
N(\varphi)=\frac{a}{(1-e^2\sin^2\varphi)^{1/2}},\qquad
a_p=M(\varphi_p)\,\Delta\varphi\;\cdot\;N(\varphi_p)\cos\varphi_p\,\Delta\lambda. \tag{3.1}
$$

For UTM pixels (10 m grid), $a_p = 100/k_p^2$ m², where $k_p$ is the UTM point scale factor from `pyproj.Proj(...).get_factors`. Every area in the paper is $\sum a_p$ over a mask, converted to km² ($10^{-6}$).

**Geometry check.** Label, SAR and optical tie points differ by ≤ 0.47 px, so the 991 isotropic tiles are used on their native grid. The 9 anisotropic label tiles (7.9119e-5° × 8.3349e-5°) are resampled by nearest neighbour onto the SAR grid using both geotransforms. Pixels outside the label footprint are marked invalid and excluded from all metrics.

**SAR.** The no-data mask is $I_p=0$. Rank normalization uses the per-tile 2% and 98% quantiles $Q_{2},Q_{98}$ of valid pixels:

$$
z_p=\mathrm{clip}\!\left(\frac{I_p-Q_2}{Q_{98}-Q_2+\epsilon},\,0,\,1\right),\qquad \epsilon=10^{-6}. \tag{3.2}
$$

Speckle filtering uses the Lee filter with a 5 × 5 window $\Omega_p$, local mean $\bar z$ and variance $\mathrm{Var}(z)$, and noise coefficient of variation $C_u=1/\sqrt{L}$:

$$
\tilde z_p=\bar z_p+k_p\,(z_p-\bar z_p),\qquad
k_p=\max\!\left(0,\;1-\frac{C_u^2}{C_{z,p}^2}\right),\qquad C_{z,p}^2=\frac{\mathrm{Var}_{\Omega_p}(z)}{\bar z_p^2+\epsilon}. \tag{3.3}
$$

The equivalent number of looks $L$ is estimated per tile as the median of $\bar z^2/\mathrm{Var}(z)$ over the 10% of 7 × 7 windows with the lowest coefficient of variation. This captures the most homogeneous areas.

**Optical.** The validity mask is $m_p = \mathbb 1[\max(B,G,R,N)>0]$. The indices are

$$
\mathrm{NDWI}=\frac{G-N}{G+N+\epsilon},\qquad \mathrm{NDVI}=\frac{N-R}{N+R+\epsilon},\qquad \mathrm{BRI}=\tfrac14(B+G+R+N)/255. \tag{3.4}
$$

**Pixel features for learned models.** These are $z,\tilde z$; the local mean and std of $\tilde z$ in 3/7/15 windows; the local min of $\tilde z$ in a 7 window; NDWI, NDVI, BRI, and the 7-window mean NDWI; the four band values divided by 255; $m_p$; and $\tilde z\times(1-\mathrm{NDWI}^+)$ (an interaction for "dark now and not water before"). Each feature uses only the pixel's own tile, so no information crosses folds.

## Problem 1 — Preprocessing, identification model, area-oriented evaluation

### 4.1 Definitions (problem text S8, S18, S19)

$$
F=W_1\setminus W_0,\qquad A_{new}=\sum_{p\in F}a_p,\qquad A_{net}=\sum_{p\in W_1}a_p-\sum_{p\in W_0}a_p,\qquad \rho=\frac{A_{new}}{A_{reg}}. \tag{4.1}
$$

Since $A_{net}=A_{new}-|W_0\setminus W_1|_a$, the bound $A_{net}\le A_{new}$ holds identically. On the official tiles the reference is $F^{ref}=\{y_p=1\}$ (A1). The reference and estimated tile areas are

$$
A_t=\sum_{p\in t} a_p\,y_p,\qquad \hat A_t=\sum_{p\in t}a_p\,\hat y_p . \tag{4.2}
$$

**P1-C7 label-semantics verification.** We re-compute the contingency label × pre-water × SAR-dark on all 65.5 M valid pixels with three $W_0$ definitions: (i) NDWI > 0; (ii) NDWI > 0.1; (iii) NDWI > 0 ∧ NDVI < 0. SAR-dark is $\tilde z<0.2$. A1 is accepted if, for every definition, the share of label-1 pixels inside $W_0$ is < 5% while the SAR-dark share of $W_0\cap\{y=0\}$ is > 50%.

### 4.2 Models

**Baseline A, physics two-stage (P1-C2).**
- $\hat W_1$: split-based Otsu. Each tile is divided into 16 sub-blocks of 64 × 64 px. For a sub-block, the Otsu threshold $\tau_b$ on $\tilde z$ is accepted if the two classes have Ashman's $D=\sqrt2|\mu_1-\mu_2|/\sqrt{s_1^2+s_2^2}>2$ and the smaller class has ≥ 10% of the pixels. The tile threshold is the median of the accepted $\tau_b$. If no sub-block passes, the fold-level fallback $\tau_g$ (median of all accepted tile thresholds in the training fold) is used.
- $\hat W_0=\{\mathrm{NDWI}>0\}\cap\{m_p=1\}$.
- $\hat F_A=(\hat W_1\setminus\hat W_0)$, after removing components smaller than `MMU_PX_TILE` = 10 px.
- Labels are used only for evaluation and for $\tau_g$ pooling. $\tau_g$ uses no labels at all.

**Baseline B, LightGBM pixel classifier (P1-C2).** It learns $\Pr(y_p=1\mid\text{features})$ directly (A1).
- Training sample per fold: from each training tile, up to 2000 pixels, stratified as half positive and half negative where possible. That is about 0.9 M rows per fold.
- Class imbalance is handled by stratified sampling. The resulting prior shift is corrected by the threshold, which is selected on the calibration fold.
- Hyperparameters: `num_leaves=63, learning_rate=0.05, n_estimators≤1500, early_stopping=50` on calibration-fold pixels, `feature_fraction=0.8, bagging_fraction=0.8, min_child_samples=100, seed=2026`.
- Threshold: $\tau=\arg\max_\tau \mathrm{F1}$ on the calibration fold, searched over a grid from 0.05 to 0.95 in steps of 0.01.

### 4.3 Validation design (P1-C3, S7)

The 67 blocks of 5 × 5 tiles are assigned to folds $k=0..4$ by greedy stratified assignment that balances tile count and flood area (`FOLD_SEED = 2026`). For outer fold $k$:
- **test** = fold $k$;
- **calibration** = fold $(k+1)\bmod 5$, used for the threshold, early stopping, isotonic map and conformal quantile;
- **train** = the other 3 folds minus every tile within Chebyshev distance 1 (tile-grid units) of a test or calibration tile.

The same fold file `output/folds.json` is used by Q1 and Q2 (P2-C2). Every reported metric is computed on test tiles only and pooled over the 5 outer folds (each tile is tested exactly once).

### 4.4 Metrics (P1-C4, S8)

Pixel metrics: OA, Precision, Recall, F1, IoU and Cohen's κ, from the pooled confusion matrix and also per fold. Area metrics use the per-tile errors $e_t=\hat A_t-A_t$:

$$
\Delta A=\sum_t e_t,\qquad \mathrm{RAE}=\frac{\sum_t \hat A_t-\sum_t A_t}{\sum_t A_t},\qquad \mathrm{MAE}_A=\frac1T\sum_t|e_t|,\qquad \mathrm{RAE}_t=\frac{e_t}{A_t}\ (A_t\ge 0.05\ \text{km}^2). \tag{4.3}
$$

The Bland–Altman bias is $\bar e$ with limits of agreement $\bar e\pm1.96\,s_e$. We also report median $|\mathrm{RAE}_t|$ over tiles with $A_t\ge0.05$ km². This floor (≈ 1% of a tile) avoids division by near-zero reference areas.

### 4.5 FP/FN decomposition by land cover (P1-C5, S9)

For land-cover stratum $c$ (A4):

$$
\mathrm{FP}_c=\sum_{p\in c}a_p\,\hat y_p(1-y_p),\qquad \mathrm{FN}_c=\sum_{p\in c}a_p(1-\hat y_p)y_p,\qquad \Delta A=\sum_c(\mathrm{FP}_c-\mathrm{FN}_c). \tag{4.4}
$$

The identity in (4.4) holds exactly and is checked in validation. The table reports, per stratum, the reference area, $\mathrm{FP}_c$, $\mathrm{FN}_c$, the net effect, and the stratum RAE. Land cover is taken from the pre-flood state, so "flooded cropland" means cropland before the flood.

### 4.6 Key factors of area accuracy (P1-C6, S10)

Each held-out tile gets descriptors that are computable without the label, plus the reference share:
- reference flood share $\rho_t^{ref}$;
- raw SAR mean and std;
- the share of sub-blocks passing the Ashman test (bimodality);
- the median ENL $L$;
- NDVI mean;
- WorldCover (or index) shares of built-up, cropland, tree and pre-water;
- the optical invalid share;
- label boundary density (perimeter of $F^{ref}$ divided by its area, in px⁻¹).

A gradient-boosted regressor (LightGBM, 300 trees, depth 3) is fitted to the signed tile error $e_t$ of baseline B, and a second one to $|e_t|$. The regressors use the same spatial folds. Drivers are ranked by mean |TreeSHAP| on out-of-fold predictions, and a Spearman partial correlation is a one-line cross-check. The Q1 "hard cases" passed to Q2 are the strata with the largest $|\mathrm{FP}_c-\mathrm{FN}_c|$, plus the tile set with $|\mathrm{RAE}_t|>0.5$.

### 4.7 Solution procedure (Q1)

1. Build `folds.json`, then assert that the sets are disjoint and the buffer rule holds.
2. Preprocess the 1000 tiles and cache $z,\tilde z$, indices and land cover as `.npz`.
3. Run baseline A for all tiles.
4. For each fold, train baseline B on the training pixels, select $\tau$ on calibration, and predict the test tiles. Save the masks as `output/q1_pred/{model}_{id}.png` (uint8).
5. Compute (4.3)–(4.4) from the saved masks.
6. Fit the driver models with SHAP.
7. Write `output/q1_results.json` with seed, versions, fold ids, metrics and tables.

Cost: preprocessing is about 5 min, and LightGBM is about 3 min per fold on CPU.

## Problem 2 — Multi-source fusion, area-calibrated model, intensity grading

### 5.1 What is kept, what is new

Q2 keeps the Q1 preprocessing, folds, metrics and land-cover strata. It adds four things:
- (i) spatial context by convolution, aimed at the FN of fragmented or vegetated flooding and the FP of dark built or bare surfaces found in Q1;
- (ii) feature-level SAR–optical fusion with learned gates;
- (iii) an area-aware loss and area calibration;
- (iv) uncertainty, which is carried into grading.

This answers S11 and S12. **P2-C1 multi-source fusion improved model.**

### 5.2 Dual-branch gated-fusion U-Net

- **Inputs.** SAR branch: $[z,\tilde z,\mathrm{std}_7(\tilde z)]$. Optical branch: $[B,G,R,N]/255$, NDWI, NDVI, $m_p$.
- **Encoder.** Each branch has 4 stages of channel width 16-32-64-128. A stage is two 3 × 3 conv + BatchNorm + ReLU, followed by a 2 × 2 max-pool.
- **Gated fusion at each scale $s$.**

$$
g^{(s)}=\sigma\!\left(W_g^{(s)}*[h_{S}^{(s)};h_{O}^{(s)}]\right),\qquad h^{(s)}=g^{(s)}\odot h_{S}^{(s)}+(1-g^{(s)})\odot h_{O}^{(s)}, \tag{5.1}
$$

  where $W_g$ is a 1 × 1 convolution. The fused $h^{(s)}$ feeds the skip connection.
- **Bottleneck and decoder.** The bottleneck has 256 channels. The decoder is a U-Net with bilinear upsampling. Dropout ($p=0.2$) sits in the bottleneck and in each decoder stage. The output is a 1 × 1 conv to a logit $o_p$. The network has about 1.9 M parameters.
- **Modality dropout.** With probability 0.25 per sample during training, the optical input is set to 0 together with $m_p=0$. Optical no-data pixels are always zeroed with $m_p=0$. This makes the model valid where optics are cloudy or missing, which happens in 60 official tiles with > 5% zeros and in about 52% of the Hengzhou S2 pixels.

**Loss.** With $\hat p_p=\sigma(o_p)$, batch $\mathcal B$ and $N_b$ valid pixels in sample $b$:

$$
\mathcal L=\underbrace{-\frac1{|\mathcal V|}\sum_{p}\alpha_y(1-\hat p_{p,y})^{\gamma}\log\hat p_{p,y}}_{\text{Focal},\ \gamma=2,\ \alpha_1=0.75}
+\underbrace{1-\frac{2\sum_p\hat p_py_p+1}{\sum_p\hat p_p+\sum_py_p+1}}_{\text{Dice}}
+\lambda_A\underbrace{\frac1{|\mathcal B|}\sum_{b}\frac{\left|\sum_{p\in b}(\hat p_p-y_p)\right|}{N_b}}_{\text{area consistency}}. \tag{5.2}
$$

Here $\hat p_{p,y}$ is the probability of the true class, and $\lambda_A=1$ (ablated at 0).

**Training.**
- Same outer folds as Q1. The calibration fold provides early stopping, using the calibration total |ΔA| at $\hat p\ge0.5$ with patience 8.
- Adam, learning rate $10^{-3}$, cosine decay, at most 40 epochs, batch 16. Augmentation: flips and rot90.
- Seeds: `torch.manual_seed(2026+k)`, deterministic algorithms enabled.
- Tiles with 0 label pixels stay in training. They teach the negatives.

### 5.3 Area calibration and the area-preserving threshold

**MC-dropout mean.** $\bar p_p=\frac1T\sum_{\ell=1}^{T}\hat p_p^{(\ell)}$, with $T=10$ forward passes and dropout active.

**Isotonic calibration** on calibration-fold pixels (2 M-pixel stratified subsample). The map $q$ is monotone non-decreasing and estimates $\Pr(y=1\mid\bar p)$:

$$
q=\arg\min_{q\ \uparrow}\sum_{p\in\text{cal}}\big(y_p-q(\bar p_p)\big)^2 . \tag{5.3}
$$

**Area-preserving threshold.** $\tau_A$ makes the mask area equal the calibrated expected area on the calibration fold:

$$
\sum_{p\in\text{cal}}a_p\,\mathbb 1[\bar p_p\ge\tau_A]=\sum_{p\in\text{cal}}a_p\,q(\bar p_p). \tag{5.4}
$$

The left side is non-increasing in $\tau_A$, so $\tau_A$ is found by bisection on [0.01, 0.99] to a tolerance of $10^{-4}$. The final mask is $\hat F=\{\bar p\ge\tau_A\}$, and the reported area is the mask area $\hat A_t$. Maps and areas therefore always agree, and C2 holds in Q3. Equation (5.4) makes the estimator unbiased on calibration data. Its bias on test data is an empirical result, reported as RAE.

### 5.4 Tile area uncertainty and conformal intervals (P2-C5)

The MC spread of the tile area is

$$
\sigma_t=\mathrm{std}_{\ell}\Big(\sum_{p\in t}a_p\,\mathbb 1[\hat p_p^{(\ell)}\ge\tau_A]\Big). \tag{5.5}
$$

With $\sigma'_t=\sigma_t+c$ and $c=\max(\mathrm{median}_{cal}\,\sigma_t,\,0.005\ \mathrm{km}^2)$, the split-conformal score and interval are

$$
s_t=\frac{|A_t-\hat A_t|}{\sigma'_t},\quad \hat q_{1-\alpha}=s_{(\lceil (n+1)(1-\alpha)\rceil)},\quad
\mathcal I_t=\big[\max(0,\hat A_t-\hat q\,\sigma'_t),\ \min(A^{tile}_t,\hat A_t+\hat q\,\sigma'_t)\big]. \tag{5.6}
$$

Here $\alpha=0.1$, and $n$ is the number of calibration tiles in the fold (about 200). Empirical coverage is $\frac1T\sum_t\mathbb 1[A_t\in\mathcal I_t]$ on test tiles, reported overall and by flood-share tercile. The signed standardized residuals $r_t=(A_t-\hat A_t)/\sigma'_t$ from all calibration folds form the empirical residual pool $\mathcal R$ used in §5.6.

### 5.5 Ablation and hard-case evaluation (P2-C2)

All variants are evaluated on the identical test tiles of the 5 outer folds:

| ID | Variant | Trained? |
|---|---|---|
| M0 | Q1 LightGBM, τ (F1) | Q1 |
| M1 | U-Net, SAR branch only, Focal + Dice | yes |
| M2 | M1 + optical branch with gated fusion (5.1) | yes |
| M3 | M2 + area-consistency loss ($\lambda_A=1$) | yes |
| M4 | M3 + modality dropout and validity channel | yes |
| M5 | M4 + MC-dropout mean, threshold 0.5 | inference |
| M6 (final Q2) | M5 + isotonic $q$ + area-preserving $\tau_A$ | inference |

Reported per variant: IoU, F1, κ, RAE, MAE$_A$, median $|\mathrm{RAE}_t|$ and, for M6, interval coverage. The same metrics are also reported on the **Q1-hard set**, defined as the test tiles with $|\mathrm{RAE}_t|>0.5$ under M0 and $A_t\ge0.05$ km². The per-land-cover FP/FN table (4.4) is repeated for M6. M4 is needed for transfer even if it costs some in-domain accuracy against M3. That cost is reported.

**Submission rule.** Q3 uses M6 if its pooled MAE$_A$ ≤ that of M0. Otherwise Q3 uses M0 with the same calibration (5.3)–(5.4), and the reason is reported. The rule is fixed here so that the outcome cannot be cherry-picked.

### 5.6 Intensity grading (P2-C3, S13, S14)

**Indicator pair.** The core indicators are $A_{new}$ and $\rho=A_{new}/A_{reg}$ (S13). The thresholds are set on $\rho$, which is scale-free. The equivalent area thresholds for a unit of land area $A_{reg}$ are

$$
A^{*}_g(A_{reg})=\theta_g\,A_{reg},\qquad g=1,2,3, \tag{5.7}
$$

so the rule reads equally as "area or proportion thresholds" (S14). Tables give $A^*_g$ for the reference tile (4.92 km²), a 1 km² grid cell, and the Hengzhou boundary (3440.4 km²).

**Threshold derivation.** A 4-component 1-D Gaussian mixture is fitted on $x_t=\mathrm{logit}(\rho^{ref}_t)$ of the 985 non-empty official tiles, with ρ clipped to $[10^{-4},1-10^{-4}]$. Fitting uses EM with `n_init=20`, `seed=2026`, full variance, and components ordered by mean. Each threshold is the Bayes boundary between adjacent components, i.e. the root in $(\mu_g,\mu_{g+1})$ of

$$
\pi_g\,\phi(x;\mu_g,s_g^2)=\pi_{g+1}\,\phi(x;\mu_{g+1},s_{g+1}^2),\qquad \theta_g=\frac{1}{1+e^{-x_g^*}}. \tag{5.8}
$$

The grades are general $[0,\theta_1)$, relatively strong $[\theta_1,\theta_2)$, strong $[\theta_2,\theta_3)$ and extremely strong $[\theta_3,1]$. A unit with $A_{new}$ smaller than one MMU is flagged "not affected" for Q4/Q5 but keeps grade "general". BIC for $K=2..6$ is reported to show whether 4 components are supported. Jenks natural breaks (4 classes on $x_t$) are a one-line cross-check.

**Stability.** B = 1000 bootstrap resamples of tiles (block bootstrap, resampling the 67 blocks) are each refit with (5.8). This gives 90% intervals for $\theta_g$ and the share of resamples in which the order $\theta_1<\theta_2<\theta_3$ holds.

### 5.7 Effect of area error on grades (P2-C4, S15)

For unit $j$ (a test tile in validation, or a Q3/Q4 unit later) we draw $m=1..M$ with $M=2000$: a residual $r^{(m)}$ from the pool $\mathcal R$ and a threshold vector $\theta^{(m)}$ from the bootstrap set. Then

$$
A_j^{(m)}=\mathrm{clip}\big(\hat A_j+r^{(m)}\sigma'_j,\ 0,\ A_{reg,j}\big),\qquad
\pi_{jg}=\frac1M\sum_m\mathbb 1\big[G(A_j^{(m)}/A_{reg,j};\theta^{(m)})=g\big],\qquad \sum_g\pi_{jg}=1. \tag{5.9}
$$

For aggregated units made of several tiles, $\hat A_j=\sum_t\hat A_t$ and $\sigma'_j=\sum_t\sigma'_t$, with one shared residual per draw. This comonotone choice is conservative.

Outputs:
- the deterministic grade accuracy of M0 and M6 against the reference grade;
- the grade confusion matrix (reference vs Monte Carlo grade);
- the flip probability $1-\max_g\pi_{jg}$;
- the flip probability as a function of the standardized distance to the nearest threshold, $\min_g|\hat\rho_j-\theta_g|A_{reg,j}/\sigma'_j$.

### 5.8 Compute budget (Q2)

The U-Net configs M1–M4 need 4 configs × 5 folds = 20 trainings.
- **GPU:** the RTX 4050 is present, so install the CUDA torch build via `_utils/dl_env.py install` (≈ 2 GB). Expect about 4 min per training at 40 epochs, ≈ 1.5 h in total.
- **CPU fallback** (if the install fails): use random 128 × 128 crops in training (full tiles at inference), halve the widths to 8-16-32-64 (≈ 0.5 M params), and cap at 25 epochs. Expect about 5 min per training, ≈ 1.7 h in total. The fallback is recorded in the results.

Each run stays under 60 min.

## Problem 3 — New inundation in Hengzhou (Guangxi), July 2026

### 6.1 Study area and imagery (P3-C2, S17)

**Study area.**
- Primary: the GADM 4.1 boundary of Hengzhou City, total area $A_{reg}=3440.4$ km². It is used for totals and ρ.
- Secondary: the **Yu River floodplain zone**, $\{\mathrm{REM}\le h^*\}$ inside the boundary, for detailed maps and as a second ρ denominator. Its area is reported.
- The proportion uses the *total* area of the study region (S13 wording "total area").

**Imagery.** All rasters are resampled to the S1 UTM 49N 10 m grid, nearest neighbour for classes and bilinear for continuous values.
- Pre-flood: Sentinel-1 RTC 2026-06-27 (VV, VH dB) and Sentinel-2 L2A 2026-05-31. S2 valid pixels are those with SCL ∈ {4, 5, 6, 7, 11}. SCL 0, 1, 3, 8, 9 and 10 are invalid.
- Flood-time: Sentinel-1 2026-07-04 and 2026-07-16 (A6).
- Ancillary layers: WorldCover 2021, OSM water and waterways, Copernicus DEM 30 m, WorldPop 2020.

A provenance table (source, date, resolution, coverage inside the boundary, processing) is saved in `q3_results.json` (S3).

### 6.2 Transfer of the Q2 model (P3-C1, S16)

The model weights are unchanged: the ensemble of the 5 fold models M6, with $T=2$ MC passes each, i.e. 10 passes in total. The isotonic map $q$ and $\tau_A$ are refit on the pooled calibration-fold predictions of the five members. Only the input representation is mapped.

1. **Tiling.** 256 × 256 tiles at 10 m (2.56 km), the same pixel count as training.
2. **SAR scale choice (A5).** For $T_1$ = dB and $T_2$ = amplitude $10^{VV/20}$, apply (3.2)–(3.3) per tile. Then compute the two-sample KS distance between pooled $\tilde z$ on Hengzhou land pixels (WorldCover ≠ 80) and on official non-flood, non-water pixels ($y=0$, NDWI ≤ 0). The transform with the smaller KS distance is used, and both distances are reported. The co-pol VV is used as the analogue of GF-3 HH.
3. **Optical mapping.** Each S2 band is quantile-mapped (1000 quantiles, valid pixels of the whole scene) onto the pooled distribution of the corresponding official uint8 band. NDWI, NDVI and BRI are then recomputed from the mapped bands, and $m_p$ = S2 validity. Invalid pixels get zero optics, as in modality-dropout training.
4. **Model probability.** $\bar p_p$ is computed for every pixel on each flood date $d$.

### 6.3 Physical guards and the definition of W0, W1, F (P3-C3, S18)

The relative elevation above the nearest drainage is

$$
\mathrm{REM}_p=\max\big(0,\ z^{DEM}_p-z^{DEM}_{n(p)}\big),\qquad n(p)=\arg\min_{q\in\mathcal D}\lVert x_p-x_q\rVert, \tag{6.1}
$$

where $\mathcal D$ is the drainage raster (OSM waterways river/canal/stream rasterized at 10 m, plus OSM water polygons, plus WorldCover class 80), and $n(p)$ comes from a Euclidean distance transform with index return.

The physics water detector $D_d$ on date $d$ is the split-based Otsu of §4.2, applied to Lee-filtered VV in dB on 256-px tiles with the same Ashman rule and median fallback.

Pre-flood water (A7):

$$
W_0=D_{0627}\ \cup\ \{\mathrm{WC}=80\}\ \cup\ \mathrm{OSM}_{water}\ \cup\ \big(\{\mathrm{NDWI}_{S2}>0\}\cap\{m=1\}\big). \tag{6.2}
$$

New inundation on date $d$ (model, then guards, then MMU):

$$
F_d=\mathrm{MMU}_{50}\Big(\{\bar p\ge\tau_A\}\ \cap\ \overline{W_0}\ \cap\ \{\mathrm{REM}\le h^*\}\ \cap\ \{VV_d-VV_{0627}\le-\delta\}\Big). \tag{6.3}
$$

Flood-time water and receded water:

$$
W_{1,d}=F_d\ \cup\ (W_0\cap D_d),\qquad R_d=W_0\setminus W_{1,d}. \tag{6.4}
$$

By construction $F_d\cap W_0=\emptyset$ (C1) and $F_d\subseteq W_{1,d}$ (C2).

### 6.4 Areas (P3-C4, S19)

$$
A_{new,d}=\sum_{p\in F_d}a_p,\qquad A_{net,d}=\sum_{p\in W_{1,d}}a_p-\sum_{p\in W_0}a_p=A_{new,d}-\sum_{p\in R_d}a_p,\qquad \rho_d=\frac{A_{new,d}}{A_{reg}} . \tag{6.5}
$$

These are computed inside the boundary (and in the floodplain zone) for both dates, with $a_p=100/k_p^2$ m². The breakdown is by WorldCover class (pre-flood land cover) and by Q4 unit.
- **Interval.** Per-tile conformal intervals (5.6) use $\sigma_t$ from the 10 passes, aggregated comonotonically. This is the model interval.
- **Grade.** $G$ and $\pi_g$ for Hengzhou and for each unit come from (5.9).

**P3-C5 map.** `output/q3_newflood_0716.tif` and `_0704.tif` (uint8, 1 = new), plus `q3_w0.tif`, `q3_w1_{d}.tif`, `q3_receded_{d}.tif` and `q3_prob_{d}.tif` (uint8 = round(255·p̄)), all GeoTIFF on the S1 grid.

### 6.5 Transfer checks without labels

These checks are reported, not tuned against:
- (a) mean $\bar p$ on permanent water (WorldCover 80 ∩ OSM water) where optics are valid. It should be low, because the model must not call permanent water "new".
- (b) IoU and area ratio between the unguarded model mask and the physics change mask $\{\Delta VV\le-3\}\cap D_d\setminus W_0$.
- (c) the share of the raw model mask removed by each guard.
- (d) the Jaccard index of $F_{0704}$ and $F_{0716}$.
- (e) the bimodality (Ashman D) of VV in the floodplain on each date.

If (c) shows that the guards remove more than 70% of the model mask, the Q3 answer is still the guarded $F$, and the paper states that transfer relies heavily on the physics guards.

### 6.6 Sensitivity (P0-C1)

One scan grid, each cell computed once from cached $\bar p$ and rasters:
- probability-threshold offset $\Delta\tau\in\{-0.2,-0.1,0,0.1,0.2\}$ × $h^*\in\{5,10,15,20,30,\infty\}$ m (the response surface figure);
- one at a time: $\delta\in\{0,1.5,3,4.5\}$ dB, `W0_MODE` ∈ {union, sar_only}, `S1_SCALE` ∈ {db, amplitude}, MMU ∈ {10, 50, 100} px.

Each case reports $A_{new}$, $A_{net}$, ρ and the grade. The scenario range is reported separately from the model interval.

### 6.7 Solution procedure (Q3)

1. Reproject and stack all layers onto the S1 grid, and rasterize the boundary.
2. Compute REM.
3. Compute $D_d$ for the three dates.
4. Build $W_0$.
5. Tile, map, and run the ensemble on 07-04 and 07-16 (≈ 1020 tiles × 10 passes per date).
6. Apply (6.3)–(6.5).
7. Run the sensitivity grid.
8. Write the rasters and `output/q3_results.json`.

Cost: the GPU run takes minutes. On CPU it is about 0.08 s per tile pass, ≈ 30 min.

## Problem 4 — Regional disaster assessment

### 7.1 Units (P4-C2, S22)

Assessment units are the Voronoi cells of the 393 OSM place nodes (370 villages, 16 towns, 5 hamlets, 1 city, 1 suburb), clipped to the boundary. Duplicate points within 50 m are merged. The cells partition the boundary. The check is $|\sum_j A_{reg,j}-3440.4|/3440.4<1\%$; with exact clipping the error is only floating-point.

Each unit inherits the town or city zone of its nearest town or city node. This gives 17 zones for reporting and for the letter.

**Why these units.** A catchment around each settlement is the natural delivery target for relief (Q5). These units are finer than townships, for which no boundary is available, and coarser than pixels.

### 7.2 Indicator system (P4-C1, P4-C3, S21, S23)

All indicators are overlays of Q3 $F_{0716}$ with public layers. Each is positive in the sense that larger means worse.

| Criterion | Indicator | Definition | Source |
|---|---|---|---|
| Hazard (core) | H1 new area $A_{new,j}$ (km²) | $\sum_{p\in F\cap j}a_p$ | Q3 |
| Hazard (core) | H2 proportion $\rho_j$ | $A_{new,j}/A_{reg,j}$ | Q3 |
| Hazard (core) | H3 intensity $I_j=\sum_g g\,\pi_{jg}$ | expected grade from (5.9) | Q2 + Q3 |
| Exposure | E1 flooded population $P_j$ | $\sum_{c\in j}\mathrm{pop}_c\cdot\phi_c$, where $\phi_c$ is the flooded fraction of 100 m cell $c$ | WorldPop 2020 |
| Exposure | E2 flooded built-up (km²) | $F\cap\{\mathrm{WC}=50\}$ | WorldCover 2021 |
| Exposure | E3 flooded cropland (km²) | $F\cap\{\mathrm{WC}=40\}$ | WorldCover 2021 |
| Exposure | E4 exposed critical facilities | number of facility proxies (A12: town/city nodes + schools) within 500 m of $F$ | OSM |
| Response | R1 access time (min) | shortest flood-cut network time from the unit's flooded-population centroid to the nearest unflooded facility proxy; unreachable = 2 × the maximum finite value | OSM roads + Q3 |
| Response | R2 road-cut share | impassable road length / unit road length (A14); 0 if the unit has no road | OSM roads + Q3 |

Terrain enters through REM in $F$ (Q3 guard). Buildings enter through WorldCover built-up, because OSM has only 609 building footprints. Hospitals and shelters enter through A12.

### 7.3 Weights

Only *affected* units are scored ($A_{new,j}\ge$ one MMU, 0.005 km²). Unaffected units get $S_j=0$.

Min–max normalization over the affected units:

$$
r_{jk}=\frac{x_{jk}-\min_j x_{jk}}{\max_j x_{jk}-\min_j x_{jk}}. \tag{7.1}
$$

Inside each criterion $c$, we combine entropy weights $w^{e}$ (with $e_k=-\frac{1}{\ln n}\sum_j\tilde r_{jk}\ln\tilde r_{jk}$, $\tilde r_{jk}=r_{jk}/\sum_j r_{jk}$, $w^e_k\propto 1-e_k$) and CRITIC weights $w^{c}$ (with $w^c_k\propto s_k\sum_l(1-\mathrm{corr}_{kl})$). The game-theoretic combination is

$$
\begin{bmatrix} w^e\cdot w^e & w^e\cdot w^c\\ w^c\cdot w^e & w^c\cdot w^c\end{bmatrix}\begin{bmatrix}\alpha_1\\ \alpha_2\end{bmatrix}=\begin{bmatrix} w^e\cdot w^e\\ w^c\cdot w^c\end{bmatrix},\qquad
w^{(c)}=\frac{|\alpha_1|w^e+|\alpha_2|w^c}{|\alpha_1|+|\alpha_2|}. \tag{7.2}
$$

Criterion weights $\Omega=(\Omega_H,\Omega_E,\Omega_R)$ come from the AHP judgment matrix below (hazard is core per S23, exposure second, response third):

$$
J=\begin{bmatrix}1&2&3\\ 1/2&1&2\\ 1/3&1/2&1\end{bmatrix},\qquad \Omega=\text{principal eigenvector}(J),\qquad \mathrm{CR}=\frac{(\lambda_{max}-3)/2}{0.58}<0.1 . \tag{7.3}
$$

This gives $\Omega\approx(0.54,0.30,0.16)$ and satisfies $\Omega_H\ge\Omega_E$ (C9). The final indicator weights are

$$
w_k=\Omega_{c(k)}\,w^{(c(k))}_k,\qquad \sum_k w_k=1. \tag{7.4}
$$

### 7.4 Index and grade

TOPSIS on $v_{jk}=w_kr_{jk}$:

$$
D_j^{+}=\lVert v_j-v^{+}\rVert_2,\quad D_j^{-}=\lVert v_j-v^{-}\rVert_2,\quad v^{+}_k=\max_jv_{jk},\ v^{-}_k=\min_jv_{jk},\qquad S_j=\frac{D_j^-}{D_j^++D_j^-}\in[0,1]. \tag{7.5}
$$

Disaster levels I–IV for the affected units come from Jenks natural breaks (4 classes) on $S_j$. The 17 zones are scored the same way from their aggregated extensive indicators ($\rho$ recomputed as area ratio, $I$ as area-weighted mean, R1 as population-weighted mean).

### 7.5 Robustness

- **Weight perturbation.** $M=2000$ draws, $\Omega\sim\mathrm{Dir}(50\,\Omega_0)$ and $w^{(c)}\sim\mathrm{Dir}(50\,w^{(c)}_0)$. Outputs: rank median and 5–95% interval per unit, top-10 membership frequency, and Spearman ρ to the baseline ranking.
- **Hazard-only ranking.** TOPSIS on H1–H3 only, compared with the full ranking (Spearman, top-10 overlap).
- **Alternative inputs.** Q4 is re-run with $F_{0704}$ and with the Q3 sensitivity cases `W0_MODE = sar_only` and $h^*=30$. The result is the rank correlation.

Output: `output/q4_results.json` and `output/q4_units.geojson` (indicators, $S_j$, level, rank CI, zone).

## Problem 5 — Emergency supply allocation with depot activation; letter

### 8.1 Task, decision time, demand (P5-C1, S24, S25)

**Task.** Choose which relief depots to open and how many tonnes of daily relief kits to send from each open depot to each affected unit, over the flood-cut road network.

**Decision times.**
- Main plan: $T_0$ = 2026-07-16, when the S1 pass of that day has been processed.
- Information-update comparison: $T_0'$ = 2026-07-04, using the earlier pass.

**Demand** (C15). From the Q4 flooded population, A11 and A15:

$$
d_j=\frac{q\,H\,P_j}{1000}\ \ [\mathrm{t}],\qquad q=3.5\ \mathrm{kg\,person^{-1}\,day^{-1}},\ H=3\ \mathrm{d}. \tag{8.1}
$$

Units with $P_j<1$ have no demand.

**Priority.** $w_j=S_j$, the Q4 TOPSIS index.

### 8.2 Network (P5-C3, S26)

- **Graph.** OSM roads are snapped at 1 m to a directed graph (`oneway=yes` is respected; other roads are two-way). Edge time = length / (A13 speed × `SPEED_FACTOR`).
- **Flood cut.** Edges are removed if they touch $F_d$ under A14. The cut set is recomputed for each date.
- **Demand node.** The flooded-population centroid of unit $j$ is snapped to the nearest node of the cut graph. Access time is $60\,\ell_j/15$ min, with access distance $\ell_j\le5$ km. A larger distance makes the unit unreachable.
- **Candidate depots** $\mathcal I$. These are the 16 town nodes, 1 city node and 5 schools, merged within 300 m. A candidate is excluded (C14) if it lies within 100 m of $F_d$ or its access leg to the cut graph exceeds 1 km.
- **Travel time.** $t_{ij}$ = depot access + Dijkstra time + demand access, in minutes, on the cut graph. It is ∞ if unreachable.

The admissible pairs are $\mathcal A=\{(i,j):t_{ij}\le T_{max}\}$ (C13).

### 8.3 MILP

Decision variables: $\eta_i\in\{0,1\}$, and $f_{ij}\ge0$ for $(i,j)\in\mathcal A$ only, and $u_j\ge0$. The two objectives (both minimized) are

$$
f_1=\frac{\sum_j w_ju_j}{\sum_j w_jd_j}\quad(\text{priority-weighted unmet share}),\qquad
f_2=\frac{\sum_{(i,j)\in\mathcal A}w_jt_{ij}f_{ij}}{\sum_jw_jd_j}\quad(\text{priority-weighted delivery time, min}). \tag{8.2}
$$

Constraints:

$$
\begin{aligned}
&\textstyle\sum_{i:(i,j)\in\mathcal A}f_{ij}+u_j=d_j &&\forall j &&\text{(C10 demand balance)}\\
&\textstyle\sum_{j:(i,j)\in\mathcal A}f_{ij}\le C_i\eta_i &&\forall i &&\text{(C11 capacity, only open depots ship)}\\
&f_{ij}\le d_j\eta_i &&\forall (i,j)\in\mathcal A &&\text{(valid tightening)}\\
&\textstyle\sum_i \eta_i\le K &&&&\text{(C12)}\\
&\textstyle\sum_{(i,j)\in\mathcal A}\frac{2t_{ij}+t_l}{Q_v}\,f_{ij}\le 60\,N_vT_sH &&&&\text{(fleet vehicle-minutes over horizon }H)\\
&f_{ij}\ge0,\ u_j\ge0,\ \eta_i\in\{0,1\}.
\end{aligned} \tag{8.3}
$$

The fleet constraint counts fractional trips ($f_{ij}/Q_v$). It is an aggregate vehicle-time budget, not a vehicle routing model; Q5 chooses allocation, not routes. The time limit $T_{max}$ enters through $\mathcal A$.

**Solution (lexicographic + ε-constraint).** Solved with PuLP and CBC, `msg` logged, `gapRel=0`, time limit 300 s.
1. Stage 1: $f_1^*=\min f_1$ s.t. (8.3).
2. Stage 2: $\min f_2$ s.t. (8.3) and $f_1\le f_1^*+10^{-6}$. This is the **recommended plan**.
3. Pareto front: $\min f_2$ s.t. $f_1\le\varepsilon_k$, with $\varepsilon_k=f_1^*+\frac{k}{14}\big(\min(1,f_1^*+0.5)-f_1^*\big)$ for $k=0..14$. The knee point is the one with the largest distance to the chord between the end points of the front (normalized axes).

The size is about 22 binaries and about 22 × (number of affected units) continuous variables, at most about 9000. CBC is expected to solve it to optimality in seconds. Status, objective and gap are saved for every solve.

### 8.4 Information update and value of information

1. Solve the plan $(\eta^{04},f^{04})$ with all inputs from 07-04 (cut graph, demand, priority, candidates).
2. Fix $\eta=\eta^{04}$ and re-optimize only $f,u$ (recourse) under the 07-16 inputs. This gives $(f_1^{04\to16},f_2^{04\to16})$.
3. Compute

$$
\mathrm{VoI}_1=f_1^{04\to16}-f_1^{*,16},\qquad \mathrm{VoI}_2=f_2^{04\to16}-f_2^{*,16}\ \ (\text{the latter at equal }f_1). \tag{8.4}
$$

We also report the Jaccard index of the open-depot sets, the change in total demand, and the number of admissible pairs lost or gained. If the recourse is infeasible, which is impossible because $u_j$ absorbs every shortfall, the result would be reported as is.

### 8.5 Constraint audit (P5-C2)

`code/q5_audit.py` reloads the saved solution and recomputes the following. It runs on the recommended plan, on each Pareto point and on the 07-04 recourse plan:
- (C10) $|\sum_if_{ij}+u_j-d_j|\le10^{-6}$;
- (C11) $\sum_jf_{ij}\le C_i\eta_i+10^{-6}$;
- (C12) $\sum \eta_i\le K$;
- (C13) $f_{ij}>0\Rightarrow t_{ij}\le T_{max}$, using a fresh Dijkstra on the saved cut graph;
- (C14) no open depot is within 100 m of $F_d$;
- the fleet budget;
- (C15) $d_j$ equals the recomputation from `q4_units.geojson`.

### 8.6 Sensitivity (P0-C1)

One-at-a-time exact re-solves:
- $K\in\{3,4,5,6,7,8\}$;
- $C$ × {0.5, 0.75, 1, 1.5};
- $T_{max}\in\{60,90,120,180,240\}$ min;
- $q\in\{2.5,3.5,7,15.6\}$;
- `SPEED_FACTOR` ∈ {0.6, 0.8, 1.0};
- `ROAD_BUFFER_M` ∈ {0, 20, 50};
- $N_v\in\{20,40,60\}$.

That is 28 solves in total. Each reports $f_1^*$, $f_2$, the open depots and the served population.

### 8.7 Recommendation letter (P5-C4, S27, S28)

The letter is written in the paper step. Its content is fixed here and every number must be filled from the results files.
1. **Remote sensing monitoring:** SAR-first acquisition with a fixed pre-flood baseline, using the Q3 timing gap (no pass near 6 July) as evidence.
2. **Information updates:** the 07-04 → 07-16 VoI and plan change from §8.4, plus a proposed update cadence.
3. **Resource allocation:** open depots, served share, the priority units and the binding resources from §8.6.
4. **Implementation conditions:** data latency, the need for a local calibration sample, and a facility census.
5. **Scope:** river-flood plains with SAR coverage, not flash floods in steep terrain, and not under dense urban canopies. Model limits come from Q1/Q2 (RAE, coverage) and Q3 (guard dependence).

**P0-C2 overall chain.** Every Q5 input file traces to `q3_*` and `q4_*` outputs. The chain is checked by the C15 recomputation and by file hashes recorded in `q5_results.json`.

## 9. Validation and sensitivity plan (summary)

| Problem | Model check | Sensitivity / robustness | Expected evidence |
|---|---|---|---|
| Q1 | spatial 5-fold with buffer; pooled test metrics; exact identity (4.4) | land-cover source switch; leave-one-cluster-out stress test | `q1_results.json` |
| Q2 | identical folds; ablation M0–M6; conformal coverage | $\lambda_A\in\{0,1\}$ (M2 vs M3); GMM $K$ by BIC; bootstrap thresholds | `q2_results.json` |
| Q3 | C1/C2 raster checks; transfer checks (a)–(e) | $\Delta\tau\times h^*$ grid; δ, W0 mode, S1 scale, MMU | `q3_results.json` |
| Q4 | partition check; AHP CR; weight sums | Dirichlet weights; hazard-only; alternative $F$ | `q4_results.json` |
| Q5 | CBC status/gap; independent audit | 28 one-at-a-time solves; VoI | `q5_results.json` |

Leakage risks and protections. The paper uses a spatial split, so these matter for credibility even though this is not a big-data contest.
1. Neighbouring tiles in train and test → block folds plus a 1-tile buffer.
2. Threshold or early stopping tuned on test data → all tuning on the calibration fold only.
3. Normalization fitted on pooled data → per-tile normalization uses only the tile itself, and the GMM thresholds are definitions fitted on reference labels, not model predictions.
4. Ablation or model selection by test score → the submission rule in §5.5 is fixed in advance.
5. The hardness definition of Q2 uses M0 *test* errors for evaluation only, never for training weights.

## 10. Assumption accountability table

Results in column (c) are filled by comp-code; the directions are expected values.

| Assumed parameter | (a) Why no data | (b) Basis of value | (c) Sensitivity planned | (d) Conclusion if wrong |
|---|---|---|---|---|
| Official land cover (A4) | not supplied | WorldCover 2021; index fallback | FP/FN table under both sources | stratum shares shift; the total ΔA is unchanged (it does not depend on strata) |
| REM threshold $h^*$ (A8) | no flood depth data | floodplain relief; 30 m DSM noise | $h^*$ 5–∞ | **soft spot**: $A_{new}$ may change by tens of %. The paper must say the Hengzhou area depends on $h^*$ |
| Change margin δ (A9) | no labels in Hengzhou | 3 dB change-detection convention | δ 0–4.5 dB | **soft spot**, same handling |
| W0 union (A7) | no single reference | conservative union | union vs sar_only | $A_{net}$ can flip sign; report both, never claim the sign without the scenario |
| S1 scale (A5) | GF-3 calibration unknown | KS choice | dB vs amplitude | if the two differ strongly, transfer is fragile, and this is stated |
| Population 2020 (A11) | no 2026 grid | WorldPop | ±10% | $d_j$ scales linearly; ranks are unchanged |
| Facility proxies (A12) | OSM has 5 POIs | township health-centre policy | schools-only | R1/E4 and depot set change; reported |
| Speeds (A13) | no traffic data | JTG B01 | 0.6–1.0 | $f_2$ scales; $f_1$ changes only via $T_{max}$ |
| Road buffer (A14) | no road-level flood data | mask resolution | 0/20/50 m | admissible pairs change |
| q, H (A15) | no local norm | Sphere 2018 | 2.5–15.6 kg | the supply-shortage regime may switch: when $\sum d_j>\sum C_i\eta_i$, $f_1^*>0$ |
| C, K, $N_v$, $T_s$, $T_{max}$ (A16) | not public | order-of-magnitude county logistics | §8.6 | the plan is conditional on the resource scenario, and the paper calls it a scenario |

## 11. Objective/constraint traceability

| Model object | Mathematical form | Problem sentence |
|---|---|---|
| definition of new area | (4.1) $F=W_1\setminus W_0$ | "newly inundated area" S8/S19, "identify pre-existing water bodies and newly inundated areas" S18 |
| Q1 evaluation | (4.3), pixel metrics | "pixel-level accuracy … estimation error of newly inundated area, relative area error … spatial overlap" S8 |
| independent samples | §4.3 blocked folds | "using independent samples" S7 |
| FP/FN by land cover | (4.4) | S9 |
| Q2 objective: minimize area error | loss term $\lambda_A$ (5.2), area-preserving (5.4) | "primary objectives of improving the accuracy of newly inundated area estimation and reducing area estimation errors" S12 |
| grading thresholds | (5.7)–(5.8) | "area or proportion thresholds … general, relatively strong, strong, extremely strong" S14 |
| grade uncertainty | (5.9) | S15 |
| Q3 outputs | (6.5) | "newly inundated area, the net increase …, and the proportion" S19 |
| Q4 hazard core | H1–H3, $\Omega_H\ge\Omega_E$ | "takes newly inundated area, inundation proportion, and flood inundation intensity as the core indicators" S23 |
| Q5 objectives min $f_1$, min $f_2$ | (8.2) | "enabling limited rescue resources to serve areas and populations in urgent need … more timely and targeted" (Q5 preamble) |
| Q5 constraints | (8.3) | "available resources, and constraints" S25 |
| Q5 link | $\mathcal A$, $\mathcal I$, $d_j$, $w_j$ from Q3/Q4 | "results of image recognition and disaster assessment are directly incorporated" S26 |

All objectives are minimizations, and no maximization is reversed.

## 12. Parameter caliber table (参数口径表)

| Quantity | Value | Unit | Semantic tag | Used in |
|---|---|---|---|---|
| pixel area, official | (3.1), ≈ 75.1 | m² | pixel_area_tile | Q1, Q2 |
| pixel area, Hengzhou | $100/k^2$ ≈ 100 | m² | pixel_area_s1 | Q3, Q4 |
| tile size | 256 × 256 | px | tile_px | Q1–Q3 |
| reference tile area | ≈ 4.92 | km² | unit_area_ref | Q2 thresholds table |
| Hengzhou total area | 3440.4 | km² | A_reg_city | Q3, Q4 |
| MMU (tiles / S1) | 10 / 50 | px | mmu | Q1 baseline A / Q3 |
| not-affected cut-off | 0.005 | km² | affected_min_area | Q4, Q5 (= 50 px × 100 m²) |
| REM threshold | 15 | m | rem_max | Q3, Q4 (via F), Q5 (via F) |
| change margin | 3 | dB | delta_db | Q3 |
| conformal level | α = 0.1 | – | alpha | Q2, Q3 |
| MC draws, grading | 2000 | – | mc_grade | Q2, Q3, Q4 |
| relief need | 3.5 | kg person⁻¹ day⁻¹ | q_need | Q5 |
| horizon | 3 | day | horizon | Q5 |
| town depot capacity | 60 | t (over horizon H) | cap_town | Q5 |
| city depot capacity | 200 | t (over horizon H) | cap_city | Q5 |
| truck payload | 5 | t/trip | q_veh | Q5 |
| fleet | 40 | trucks | n_veh | Q5 |
| shift | 12 | h/day | shift | Q5 |
| load/unload time | 30 | min/trip | t_load | Q5 |
| time limit | 120 | min (one way) | t_max | Q5 |

**Caliber check.** Capacity and demand are both stated over the horizon $H$ (3 days). The fleet budget in (8.3) is $60N_vT_sH$ vehicle-minutes, so it covers the same 3-day horizon ($T_s$ in h/day). Theoretical fleet capacity at $t_{ij}=60$ min is $40\times5\times(720\cdot3)/(150)\approx2880$ t, which is larger than the total depot capacity of 6 × 60 = 360 t. The fleet budget is therefore expected to be inactive in the base case. That is a valid finding, not an error.

## 13. Result constraint list

### Hard bounds (problem / identity)
- `0 <= IoU <= 1`, `0 <= F1 <= 1`, `-1 <= kappa <= 1`, `0 <= OA <= 1` for every model and fold.
- `abs(sum_c(FP_c - FN_c) - DeltaA) <= 1e-9 * max(1, abs(DeltaA))` (identity 4.4).
- `set(train) & set(test) == set()` and `set(train) & set(cal) == set()`, and no train tile within Chebyshev 1 of a test/cal tile.
- `RAE >= -1`, and the area of every tile satisfies `0 <= A_hat_t <= A_tile_t`.
- `0 < theta1 < theta2 < theta3 < 1`; `abs(pi_j.sum() - 1) < 1e-9` for every unit.
- `abs(mask_area_cal - expected_area_cal) / expected_area_cal <= 1e-3` (5.4).
- Q3: `not np.any(F & W0)`, `np.all(F <= W1)`, `A_net <= A_new + 1e-9`, `rho == A_new / 3440.4`, `0 <= rho <= 1`.
- Q4: `abs(sum(A_reg_j) - 3440.4) / 3440.4 < 0.01`; `abs(sum(A_new_j) - A_new_city) / A_new_city < 0.005`; `w >= 0` and `abs(w.sum() - 1) < 1e-9`; `Omega_H >= Omega_E`; `CR < 0.1`; `0 <= S_j <= 1`.
- Q5: C10–C15 as in §8.5; `sum(eta) <= K`; `f1* in [0, 1]`; the status of every solve is Optimal.

### Expected ranges (estimates only; not error evidence)
- Q1 LightGBM: IoU about 0.55–0.80. The SAR in/out means are 45 vs 118 DN, so separability is strong. |RAE| ≤ 0.2 pooled; tile errors are worst on low-flood and built tiles.
- Q2 M6: pooled |RAE| below M0's. MAE$_A$ lower by 10–40%, which is not guaranteed. Conformal coverage 0.85–0.95.
- Thresholds: $\theta_1$ about 0.01–0.05, $\theta_3$ about 0.5–0.9 (the ρ quartiles are 0.012/0.038/0.206).
- Hengzhou $A_{new}$ on 07-16: of the order of 10–100 km², i.e. ρ ≈ 0.3–3%. The descriptive check found 115 km² of "new dark" pixels over the 6525 km² scene before guards. $A_{net}$ may be small or negative under the union W0.
- Q4: 100–300 affected units.
- Q5: total demand about $3.5\cdot3\cdot P/1000$ t. If 30,000 people are flooded, demand is about 315 t, comparable to the 360 t of 6 depots, so $f_1^*$ may be 0 or small.

## 14. Expected behaviour
- Time: $F_{0716}$ vs $F_{0704}$. 07-04 is pre-peak (rain onset) and 07-16 is post-peak (residual water), so either could be larger. Both are reported, and neither is called the peak.
- Monotonicity: $A_{new}$ is non-decreasing in $h^*$ and non-increasing in δ and in $\tau$. $f_1^*$ is non-increasing in $K$, $C$, $T_{max}$ and $N_v$, and non-decreasing in $q$.
- The Q2 ablation steps need not all improve the result; M4 may cost accuracy. A step that worsens the result is reported, not removed.
- Q4: the full ranking correlates strongly with the hazard-only ranking (Spearman expected > 0.6) but is not identical.
- Q5: with $f_1^*=0$, the Pareto front is anchored at full coverage.

## 15. Anomaly handling plans

**Folds.** If the fold flood shares differ by more than 0.05:
- Cause: a few large flooded blocks. Confirm from the block flood areas.
- Fix: re-run the greedy assignment with another `FOLD_SEED` (at most 5 seeds) and keep the most balanced one. Report the seed.
- Forbidden: random tile or pixel splits.

**Baseline A.** If it predicts large FP on bright tiles:
- Cause: Otsu fallback on unimodal tiles. Confirm from the Ashman pass rate.
- Fix: none. This is a baseline finding and is reported.
- Forbidden: tuning the baseline on test labels.

**P1-C7 fails** (label-1 share in $W_0$ ≥ 5% under a stricter $W_0$):
- Fix: switch `LABEL_SEMANTICS` to `"post_water"` for Q1/Q2 (train on $W_1$, subtract $\hat W_0$), and record it in UPSTREAM_ISSUES.md.
- Forbidden: keeping A1 silently.

**U-Net worse than LightGBM (M2 MAE$_A$ > M0):**
- Cause: overfitting on about 470 training tiles, or a CPU fallback with crops. Confirm from the train/val loss curves.
- Fix: within the fixed grid, use stronger augmentation (scale jitter 0.8–1.25) and dropout 0.3. Then apply the submission rule of §5.5.
- Forbidden: selecting by test score; changing folds.

**Isotonic / τ_A.** If τ_A lands at a bisection bound:
- Cause: probabilities are degenerate. Confirm from the histogram of $\bar p$ on the calibration fold.
- Fix: widen the bracket to [0.001, 0.999].
- Forbidden: a fixed τ = 0.5 reported as "area-preserving".

**Conformal coverage < 0.80 on test:**
- Cause: exchangeability between calibration and test folds fails because the regions differ. Confirm from the coverage per fold.
- Fix: report it as is; also report the leave-one-cluster-out coverage.
- Forbidden: enlarging α after looking at the test data.

**GMM gives a crossing outside $(\mu_g,\mu_{g+1})$, or the order fails:**
- Cause: overlapping components. Confirm from BIC and the component weights.
- Fix: use Jenks breaks on $x_t$ as the threshold rule, record the switch and keep the bootstrap.
- Forbidden: round-number thresholds.

**Q3 KS distances both > 0.3:**
- Cause: the sensor gap.
- Fix: none to the model. Report it, and rely on the transfer checks of §6.5.
- Forbidden: retraining on Hengzhou pseudo-labels; changing the Q2 weights (S16).

**Q3 $A_{net}<0$:**
- Cause: W0 includes dry river beds or seasonal ponds that are mapped as water in WorldCover or OSM. Confirm from the area of $R_d$ by source.
- Fix: none. Report it together with the `sar_only` case.
- Forbidden: forcing $A_{net}\ge0$.

**Q3 $A_{new}$ near 0:**
- Cause: τ_A too high under the domain shift, or guards too strict. Confirm from guard removal shares (§6.5c).
- Fix: report the sensitivity grid; keep the default.
- Forbidden: lowering defaults to "find" a flood.

**Q4: fewer than 10 affected units:**
- Fix: score at the 1 km grid instead (same formulas), and record it.
- Forbidden: inventing exposure.

**Q5 infeasible:** this cannot happen, because $u_j$ makes the model always feasible. If CBC reports infeasibility, check the bounds for data errors.

**Q5 $f_1^*=1$ (nothing reachable):**
- Cause: the network is over-cut by the buffer. Confirm the cut share.
- Fix: report it, and show the `ROAD_BUFFER_M = 0` case.
- Forbidden: re-adding flooded roads in the main plan.

## 16. Method specification

| Step | Method (unique) | Key parameters | Forbidden substitute |
|---|---|---|---|
| SAR normalization | per-tile 2–98% quantile stretch (3.2) | ε = 1e-6 | global min–max (it mixes tiles) |
| speckle | Lee 5 × 5 (3.3), ENL from the 10% lowest-CV 7 × 7 windows | – | median filter; no filter |
| geometry | nearest resample of the 9 anisotropic labels | – | ignoring the scale difference |
| W0 (tiles) | NDWI > 0 | – | MNDWI (no SWIR band) |
| W1 physics | split-based Otsu, 64-px blocks, Ashman D > 2, ≥ 10% minor class, median; fallback $\tau_g$ | – | single global Otsu |
| learned baseline | LightGBM per pixel, §4.2 | seed 2026 | logistic regression |
| folds | 5 × 5-tile blocks, 5 folds, buffer 1 | seed 2026 | random split |
| fused model | dual-branch gated U-Net (5.1), loss (5.2) | §5.2 | single-branch input concatenation as final model |
| calibration | isotonic (5.3) + area-preserving τ_A (5.4) | tol 1e-4 | Platt scaling |
| intervals | split conformal, normalized by MC σ (5.6) | α = 0.1 | Gaussian ±1.96σ |
| thresholds | GMM K = 4 on logit ρ, Bayes boundaries (5.8) | n_init 20, seed 2026 | quantiles; round numbers |
| grade uncertainty | residual-pool Monte Carlo (5.9) | M = 2000 | Gaussian noise ±x% |
| REM | Euclidean nearest drainage (6.1) | – | raw elevation threshold |
| Q4 weights | entropy + CRITIC game-theory (7.2) within criteria, AHP (7.3) between | – | equal weights |
| Q4 index | TOPSIS (7.5) | – | weighted sum only |
| Q5 | MILP (8.3), CBC, lexicographic + ε-constraint | gapRel = 0 | greedy assignment as the plan |

## 17. Verification checkpoints (for comp-code)

- □ 1000 tiles processed and the area grid uses (3.1). Fail → check the inputs.
- □ P1-C7 test passes. Fail → see the P1-C7 plan in §15.
- □ Fold disjointness and buffer. Fail → fold plan.
- □ Identity (4.4) holds per model.
- □ Every metric is computed from saved test masks.
- □ The M0–M6 metrics come from identical test tile sets (same ID list hash).
- □ The calibration equality (5.4) holds within 1e-3.
- □ Coverage reported per fold.
- □ $0<\theta_1<\theta_2<\theta_3<1$ and derived by (5.8) or the recorded Jenks fallback.
- □ $\sum_g\pi_{jg}=1$.
- □ Q3: the C1/C2 raster checks pass on both dates; $A_{net}\le A_{new}$; ρ = A_new/3440.4.
- □ Q3 uses the M6 weights (checkpoint file hash equals Q2's).
- □ Q4 partition error < 1%; $\sum_jA_{new,j}$ = city $A_{new}$ within 0.5%; weights sum to 1; CR < 0.1.
- □ Q5: Optimal status for all solves, and the audit passes (including the 07-04 recourse plan).
- □ The demand recomputation matches (C15).
- □ Final: all hard bounds hold; out-of-range expected values are explained in RESULTS.md.

## 18. Structural validation inputs (Q5 MILP)

**Constraint activity (expected).**
- Capacity (C11) of the open depots near large demand: active if $\sum d_j$ is comparable to $\sum C_i\eta_i$.
- $K$: active when capacity binds.
- Fleet: inactive in the base case (§12).
- $T_{max}$: it acts through $\mathcal A$; units beyond it cause $u_j>0$.
- If nothing is active and $f_1^*=0$, supply exceeds demand. The plan is then driven by $f_2$ alone, which is a valid finding.

**Decision variables.**

| Variable | Meaning | Bounds | Expected | At a bound means |
|---|---|---|---|---|
| $\eta_i$ | depot open | {0,1} | 3–6 open | $\sum \eta=K$: K binds |
| $f_{ij}$ | flow, t | [0, $d_j$] | sparse | – |
| $u_j$ | unmet, t | [0, $d_j$] | mostly 0 | $u_j=d_j$: the unit is unreachable or the lowest priority |

**Sensitivity directions.** $K\uparrow$ → $f_1^*$ ↓ or equal. $C\uparrow$ → $f_1^*$ ↓ or equal. $T_{max}\uparrow$ → $f_1^*$ ↓ or equal. $q\uparrow$ → $f_1^*$ ↑ or equal. A reversed direction indicates an indexing error.

**Stability.** The MILP is solved exactly, so no variation across runs is acceptable (CV = 0). Ties between equal-cost plans may differ in $y$; this is reported.

**Resource use.** Depot capacity use is 0–100% and is reported. Zero use of an open depot cannot occur at the optimum of stage 2, unless ties occur.

## 19. METHOD_CLAIMS

| ID | Claim in the paper | Required code feature |
|---|---|---|
| M1 | spatially blocked 5-fold CV with buffer | fold file built from tile-grid blocks; buffer exclusion; disjointness asserted |
| M2 | split-based Otsu two-stage physics baseline | `threshold_otsu` on 64-px sub-blocks with the Ashman D test; NDWI subtraction |
| M3 | LightGBM pixel classifier | `lightgbm` `LGBMClassifier` trained per fold |
| M4 | dual-branch gated-fusion U-Net with area-consistency loss | `torch.nn.Module` with two encoders, sigmoid gate (5.1), and loss containing the area term |
| M5 | MC-dropout + isotonic calibration + area-preserving threshold | dropout active at inference; `IsotonicRegression`; bisection for τ_A |
| M6 | split-conformal tile-area intervals | quantile of normalized calibration scores; coverage on test |
| M7 | GMM-based grade thresholds with block bootstrap | `GaussianMixture(n_components=4)`; Bayes-boundary root finding; 1000 resamples |
| M8 | Monte Carlo grade probabilities | residual-pool sampling, M = 2000 |
| M9 | REM/change/MMU-guarded transfer of the Q2 model | the same checkpoint loaded; `distance_transform_edt` for REM; ΔVV guard |
| M10 | entropy–CRITIC game-theory weights + AHP + TOPSIS + Dirichlet robustness | the 2 × 2 system (7.2); eigenvector CR; `dirichlet` |
| M11 | location–allocation MILP with ε-constraint on the flood-cut network | `LpBinary` depot variables; CBC; edges removed by F; Dijkstra |

<!-- METHOD_CLAIMS_MACHINE
M1 | must: folds\.json, chebyshev|buffer, isdisjoint | forbid: train_test_split\(, KFold\(.*shuffle=True
M2 | must: threshold_otsu, ashman | forbid: 
M3 | must: LGBMClassifier|lgb\.train | forbid: LogisticRegression\(
M4 | must: nn\.Module, torch\.sigmoid|nn\.Sigmoid, area | forbid: 
M5 | must: IsotonicRegression, \.train\(\)|enable_dropout|mc_dropout | forbid: tau_A\s*=\s*0\.5\b
M6 | must: conformal, coverage | forbid: 1\.96\s*\*\s*sigma
M7 | must: GaussianMixture\(, n_components\s*=\s*4, bootstrap | forbid: thresholds\s*=\s*\[\s*0\.1\s*,\s*0\.3
M8 | must: pi_|grade_prob, 2000 | forbid: 
M9 | must: distance_transform_edt, load_state_dict, delta_db|DELTA_DB | forbid: 
M10 | must: entropy, critic, topsis, dirichlet | forbid: 
M11 | must: LpBinary, PULP_CBC_CMD, dijkstra | forbid: greedy_plan\s*=.*final
-->

## 20. Logic contract

The equivalence claim is the area-preserving identity (5.4). The bound is that the union $W_0$ makes $A_{new}$ a lower bound relative to the `sar_only` $W_0$.

Bound derivation in four steps:
1. Inequality: $W_0^{union}\supseteq W_0^{sar}$.
2. Substitute into (6.3): $F^{union}=F^{*}\setminus W_0^{union}\subseteq F^{*}\setminus W_0^{sar}=F^{sar}$.
3. Bound: $A_{new}^{union}\le A_{new}^{sar}$, so the union value is a lower bound.
4. Numerical probe: shrink $W_0$ to `sar_only` and recompute. $A_{new}$ must increase or stay equal, i.e. the sign is ≥ 0. comp-code writes the sign into `figures/all_results.json` → `logic_probes.bounds`.

```json
{"LOGIC_CONTRACT_MACHINE":{
  "bounds":[{"quantity":"q3.A_new_0716_km2","is":"lower","reason":"union W0 is the largest pre-water set; removing it from the guarded model mask gives the smallest F, so A_new(union) <= A_new(sar_only)"}],
  "no_double_count":[],
  "must_features":["z_lee","ndwi","ndvi"],
  "train_range":{"label_flood_share_tile":[0.0,0.9999]},
  "monotonic":[{"more":"K_MAX","then":"q5.f1_weighted_unmet","dir":"better","sense":"min"},{"more":"CAP_TOWN_T","then":"q5.f1_weighted_unmet","dir":"better","sense":"min"}],
  "constraints_with_margin":[{"quantity":"q5.n_open_depots","limit":6,"kind":"le","min_margin":0}],
  "calibration_anchors":[],
  "equivalence_claims":[{"claim":"area-preserving threshold makes calibration mask area equal calibrated expected area","quantity_a":"q2.cal_mask_area_km2","quantity_b":"q2.cal_expected_area_km2","rel_tol":0.001,"explanation":""}]
}}
```

## 21. Figure and table pre-planning (图表预规划, carried from the analysis)

The figure plan of PROBLEM_ANALYSIS.md §6 is kept unchanged (14 data figures, 2 HTML diagrams, 1 TikZ). Updates from modeling:
- fig_q2_ablation uses the variants M0–M6 of §5.5.
- fig_q2_model_compare uses rows M0, M2, M3, M4, M6 plus baseline A, and the columns IoU, F1, RAE, MAE_A, median |RAE_t| and coverage.
- fig_q3_threshold_surface axes are $\Delta\tau$ × $h^*$ (§6.6).
- fig_q5_pareto shows the ε-front for 07-16 with the 07-04 recourse plan marked (§8.4).
- An extra table, TABLE_q5_voi (VoI and plan change), is added. fig_q1_error_drivers uses the TreeSHAP of §4.6.
- No figure is planned for Jenks, which is a cross-check only.

<!-- BEGIN FIGURE_MANIFEST -->
## FIGURE_MANIFEST

**Data figures:** fig_q1_visual_cmp, fig_q1_label_semantics, fig_q1_area_agreement, fig_q1_error_drivers, fig_q2_model_compare, fig_q2_ablation, fig_q2_grading, fig_q3_newflood_map, fig_q3_threshold_surface, fig_q3_area_budget, fig_q4_risk_map, fig_q4_rank_robustness, fig_q5_route_alloc, fig_q5_pareto

**HTML flow/architecture diagrams:** fig_roadmap, fig_arch_fusion

**TikZ:** tikz_q1_set_logic

**Total: DATA=14, HTML=2, TIKZ=1, ALL=17**
<!-- END FIGURE_MANIFEST -->

## 22. Programming notes (编程实现要点)

- Libraries: rasterio, geopandas, shapely, numpy, scipy.ndimage, scikit-image (`threshold_otsu`), lightgbm, shap, torch, scikit-learn (`IsotonicRegression`, `GaussianMixture`), networkx, pulp (CBC), pyproj.
- Files: `code/config.py` (all switches of §1), `code/common.py` (3.1)–(3.4), `q1_*.py` … `q5_*.py`, `q5_audit.py`, `validate_constraints()` per question.
- Every results JSON records the seed, run_id, versions and input hashes.
- Results go to `output/q{1..5}_results.json`, `figures/all_results.json` and RESULTS.md.
- The two datasets are never mixed: the official tiles feed Q1/Q2 only, and the Hengzhou layers feed Q3–Q5 only.

Problem types covered: [data mining / machine learning, statistics (calibration, conformal, mixture), evaluation / decision (entropy–CRITIC–AHP–TOPSIS), graph / network (shortest paths), optimization (MILP)]. Checked against the error-prevention manual (本题涉及题型：[数据挖掘/机器学习, 统计, 评价/决策, 图论/网络, 优化]，已对照防错手册审查。)
