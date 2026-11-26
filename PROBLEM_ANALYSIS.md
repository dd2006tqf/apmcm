# Problem Analysis Report — 2026 APMCM Problem A

**Selected problem**: Problem A, "Flood Disaster Assessment and Emergency Rescue Optimization Based on Image Recognition" (PDF pages 1–5; the PDF contains only Problem A).
**本赛题共 5 个子问题** (Problems 1–5; the recommendation letter is part of Problem 5).

**Materials used**
- Problem text: `user_data/2026 APMCM Problem A_extracted.txt`. This is the PDF text layer because vision OCR failed on all 5 pages. The problem contains almost no formulas or numeric parameters (only dates), so the text layer is sufficient. The dates were checked against the text: May 24, 2024; July 6, 2024; July 6, 2026.
- User notes: `user_data/PROBLEM_STATEMENT.md` (a summary of third-party "tips", which the user asked us to verify rather than follow) and `CUSTOM_REQUIREMENTS.md` (highest priority: aim for a national first prize, original innovations, advanced visual style, AI-use disclosure).
- Data: `user_data/a题数据汇总/`. The nested zips are already extracted on disk. The official data (Attachment 1) is split across `Problem A Dataset 1` (labels + SAR) and `Problem A Dataset 2` (optical). The self-collected Hengzhou data is in `公开数据/`. The two sets cover different regions and must never be mixed. See `README.md` in that folder.
- No submission template or commitment-letter file was found in the package. The APMCM template `apmcmthesis` will be used.

**Conflicts between user notes and verified data (all checked on the full 3000 tiles, see §3)**

| Claim | Source | Verified fact | Decision |
|---|---|---|---|
| Mean label flood share 24.5%, max 99.3% | data README | **18.99%** pixel-weighted, max 99.99% | use verified value |
| Mean 19.0%, median 3.8%, 15 empty tiles | user notes | 18.99%, median 3.76%, 15 empty tiles | confirmed |
| "label = post-disaster water, not new flood" | user notes | Only **2.37%** of label=1 pixels were water before the flood (pre-optical NDWI>0). Of pre-existing water pixels, **71.2%** are still SAR-dark after the flood, yet only **4.87%** are labelled 1 | **rejected**: labels are effectively *newly inundated* area (permanent water excluded). Downstream must re-confirm this with a stricter water index and report it as an empirical finding |
| Pixel ≈ 9.2 × 8.1 m ≈ 75 m² | user notes | scale 8.333e-5° → 9.26 m (N–S) × 8.11 m (E–W, cos 28.9°) ≈ 75.1 m² | confirmed; compute per-tile area from latitude |
| Region "inferred" as Dongting / Poyang lakes, 2020 flood | user notes | Tiles fall in 4 spatial clusters (512 + 450 + 22 + 16 tiles) at 112.69–116.91°E, 28.43–29.30°N, consistent with the Dongting and Poyang basins. The event date is **not** given by the problem | may say "located in the Dongting/Poyang lake region (from GeoTIFF tags)"; must not claim an event year |

## 1. Problem overview

Five chained tasks:
- **Q1** trains and validates a newly-inundated-area recognizer on Attachment 1, with an area-oriented evaluation.
- **Q2** improves it by multi-source fusion, then defines a four-level intensity grading with data-based thresholds and grade uncertainty.
- **Q3** transfers the model to Hengzhou (Guangxi), around 6 July 2026.
- **Q4** builds a regional disaster assessment (hazard + exposure).
- **Q5** builds one rescue optimization fed directly by Q3 and Q4, and writes a letter to the emergency authority.

Unified definition used everywhere:

$$F = W_1 \setminus W_0 = W_1 \cap \overline{W_0},\qquad A_{new}=\sum_{p\in F} a_p,\qquad A_{net}=|W_1|_a-|W_0|_a,\qquad \rho = A_{new}/A_{region}$$

Here $W_0$ is pre-flood water, $W_1$ is water at the flood observation time, and $a_p$ is the true ground area of pixel $p$. $A_{net}\le A_{new}$ always holds, because water that disappears (receding, drained or seasonal ponds) reduces the net value. The two quantities must be reported separately (the problem asks for both in Q3).

## 2. Sub-problem decomposition

### 问题一 (Problem 1): Preprocessing and newly-inundated-area identification

- **Input**: 1000 co-registered tile triplets: pre-disaster Sentinel-2 4-band uint8 (B, G, R, NIR), post-disaster GF-3 HH uint8 SAR, binary label.
- **Output**: pixel masks $\hat F$; pixel metrics (OA, Precision, Recall, F1, IoU, Kappa); area metrics (tile and total new-area error $\Delta A$, relative area error RAE, MAE of tile area); per-land-cover FP/FN area table; ranked error drivers.
- **Difficulty**: medium–hard. The difficulty comes from cross-sensor fusion, an extreme spread of SAR brightness (tile means 14–255), class imbalance (median flood share 3.8%), and spatial autocorrelation that makes random splits leak.
- **Method**:
  - Preprocessing: per-tile robust SAR normalization (2–98% quantile stretch), Lee/refined-Lee speckle filter, checking optical zero pixels and clouds, verifying co-registration (tie offsets < 0.47 px) and pixel-area computation.
  - Baseline A (physics, unsupervised): Otsu / split-based threshold on SAR gives $W_1$, NDWI gives $W_0$, and $\hat F=\hat W_1\setminus\hat W_0$.
  - Baseline B (learned): gradient boosting (LightGBM) on per-pixel multi-scale features (filtered SAR, local mean/std/GLCM texture, NDWI, NDVI, brightness).
  - Validation: **spatial block cross-validation** grouped by the 4 spatial clusters / longitude blocks (no random pixel split).
  - Pseudo land cover for the tiles (water / vegetation / cropland-bare / built-bright) comes from optical index clustering, because no land-cover product is supplied for these tiles. Downloading ESA WorldCover for 112–117°E is an optional upgrade.
  - Error drivers: regression or SHAP on tile-level area error versus tile descriptors (flood share, SAR mean/contrast, NDVI, built share, edge density, permanent-water share).
- **Relations**: Q1 fixes the preprocessing, the definition of $F$ and the metric set for all later problems. Its FP/FN hotspots define the "poor / uncertain cases" that Q2 targets.
- **论文重述素材 (restatement)**: Using the paired pre-disaster optical images, post-disaster SAR images and labels of Attachment 1, preprocess the images and build a model that identifies flood-inundated areas. Evaluate the model on independent samples, reporting the error of the estimated newly inundated area, the relative area error and the spatial overlap in addition to pixel accuracy. Analyse how false positives and false negatives under different land covers bias the area, and identify the key factors controlling accuracy.
- **论文分析素材 (analysis)**:
  - The task is a cross-sensor change-detection segmentation problem, not a plain water segmentation, because the quantity of interest is *new* water. Our verification shows the supplied labels already exclude permanent water. The model must therefore learn both "is water after" and "was not water before". Two-stage logic (water after minus water before) and end-to-end learning are both admissible, and comparing them exposes where errors come from.
  - The tiles are spatially clustered, so performance must be measured with spatially blocked folds.
  - Because the downstream questions depend on area, evaluation is organized around tile-level and total area errors, with pixel metrics secondary.
  - Errors are then stratified by land cover to explain which FP and FN mechanisms inflate or deflate the area. Examples: radar shadow and smooth bare or paved surfaces cause FP; flooded vegetation with double bounce and urban flooding cause FN.

### 问题二 (Problem 2): Multi-source fusion, improved model, intensity grading

- **Input**: Q1 data, Q1 error analysis, and optionally DEM-derived terrain (HAND, slope; Copernicus DEM, public).
- **Output**: an improved model, ablation results, a bias-corrected area estimator with uncertainty interval, four-level thresholds (general / relatively strong / strong / extremely strong) on new area and proportion, and grade-flip probabilities under area uncertainty.
- **Difficulty**: hard.
- **Method (recommended)**:
  - (i) A fusion segmentation network: a light U-Net with dual-branch encoder (SAR branch + optical branch, mid-level attention fusion), trained with a Dice + Focal + **area-consistency loss** $\lambda\,|\sum\hat p-\sum y|/N$. Alternative: LightGBM with fused features plus CRF/morphological post-processing.
  - (ii) Uncertainty-guided refinement on hard tiles: MC-dropout or ensemble variance, and tiles with high entropy get terrain or texture priors.
  - (iii) Bias correction: an error-adjusted (stratified, Olofsson-type) area estimator using the validation confusion matrix, and split-conformal prediction intervals for tile area.
  - (iv) Grading:
    - Indicator pair: $(A_{new}, \rho)$.
    - Thresholds come from the empirical distribution of $\rho$ over units: Gaussian mixture / Jenks natural breaks on $\log\rho$, cross-checked by quantiles. Absolute-area thresholds are derived as $A^{*}=\rho^{*}A_{unit}$ for a stated unit size.
    - Uncertainty: Monte Carlo resampling of area from the conformal or bias model produces a grade confusion matrix and flip probability.
- **Relations**: Q2 depends on Q1. It hands the model, the correction factor and the thresholds to Q3.
- **论文重述素材 (restatement)**: For the cases where the Problem 1 model performs poorly or is uncertain, use multi-source information fusion and model improvement, with the main aim of reducing the error of the estimated newly inundated area. Then build an intensity classification that uses the newly inundated area and its share of the study-region area as core indicators. Define explicit thresholds for four levels (general, relatively strong, strong, extremely strong) and analyse how area errors and their uncertainty affect the assigned level.
- **论文分析素材 (analysis)**:
  - Improvement targets the failure strata found in Q1, not average pixel accuracy. The objective explicitly includes area bias, so the loss and post-processing are area-aware, and a statistical bias correction converts pixel confusion into an unbiased area with a confidence interval.
  - Grading must be reproducible and data-driven. Thresholds come from the natural structure of the proportion distribution and are tested for stability.
  - Because a unit near a threshold can change grade under small area errors, the grade is reported together with its probability rather than as a deterministic label. This is what Q3 and Q4 consume.

### 问题三 (Problem 3): Hengzhou (Guangxi) application, flood around 6 July 2026

- **Input**: the Q2 model and thresholds; Sentinel-1 RTC VV/VH dB on 2026-06-27 (pre), 2026-07-04 and 2026-07-16 (UTM 49N, 10 m grid, 20 m effective); Sentinel-2 2026-05-31 (pre, 62.9% valid); WorldCover 2021, OSM water, Copernicus DEM; GADM boundary (3440.4 km²).
- **Output**: study-area definition; maps of pre-existing water $W_0$, flood-time water $W_1$, new inundation $F$ and receded water; $A_{new}$, $A_{net}$, $\rho$ with intervals; per-land-cover and per-unit breakdown; intensity grade.
- **Difficulty**: hard. The domain shift is GF-3 HH uint8 → Sentinel-1 VV/VH dB. There are no labels. The image dates bracket rather than hit 6 July.
- **Method**:
  - Domain adaptation:
    - Map S1 VV (the closest analogue to HH co-pol) into the training intensity space by per-tile quantile normalization, the same transform as in Q1. Apply the Q2 model on 2.56 km tiles.
    - Physically cross-check with split-based Otsu on VV/VH, a HAND/slope mask (floods cannot occur on steep high terrain) and a minimum mapping unit.
  - $W_0$: fusion of the 06-27 SAR water, S2 NDWI where cloud-free, and WorldCover/OSM permanent water. $W_1$: 07-04 and 07-16 (the multi-date design also gives the recession dynamics).
  - Study area: the whole municipality for totals, plus a Yu River floodplain zone (HAND ≤ h*) for detailed mapping.
  - Uncertainty: the Q2 bias model, plus sensitivity to the VV threshold, HAND threshold and $W_0$ source.
- **Relations**: Q3 depends on Q1 (definitions) and Q2 (model, thresholds). It feeds Q4 (hazard indicators) and Q5 (flooded roads and sites).
- **论文重述素材 (restatement)**: Apply the Problem 2 model to the flood around 6 July 2026 in Hengzhou City, Guangxi, to test it in a real event. Choose a suitable study area and representative images before and during the flood. Identify pre-existing water bodies and flood-induced new inundation, compute the newly inundated area, the net increase in inundated area and the proportion of newly inundated area, and map the new inundation.
- **论文分析素材 (analysis)**:
  - The core risk is transfer: the trained model saw 8-bit GF-3 HH, while Hengzhou has calibrated Sentinel-1 dB. We therefore transfer through a sensor-agnostic representation (normalized backscatter rank and change) and validate the transfer on physics: bimodality, terrain consistency, agreement with permanent-water maps, and consistency between 07-04 and 07-16.
  - Pre-existing water is estimated from independent sources so that $F$ is not contaminated by rivers and ponds.
  - Gross new area and net increase are reported separately, because receding seasonal water makes them differ.
  - The outputs are rasters plus unit summaries that Q4 and Q5 use directly.

### 问题四 (Problem 4): Regional disaster assessment

- **Input**: the Q3 $F$ raster, $\rho$ and grade; WorldPop 2020 (939,823 people inside the boundary); WorldCover built-up and cropland; OSM roads (2542 km), places (393, of which 16 towns), sparse amenities (5 schools only); DEM.
- **Output**: assessment units; an indicator system; a composite disaster index and grade per unit; rank robustness; a hotspot map.
- **Difficulty**: medium.
- **Method**:
  - Units: a 1 km grid (fine) aggregated to town-level Voronoi units built from the 16 town nodes clipped to the boundary, because no township boundary is available.
  - Framework: Risk = Hazard × Exposure × (1 + Vulnerability / Response deficit).
    - Hazard: $A_{new}$, $\rho$, grade probability.
    - Exposure: flooded population, flooded built-up area, flooded cropland, flooded road length.
    - Response conditions: travel time to the nearest town centre or hospital proxy, share of road cut, elevation/HAND.
  - Weights: combined entropy + CRITIC objective weights (game-theoretic combination) with TOPSIS ranking. Alternative: a projection pursuit / cloud model.
  - Robustness: Dirichlet weight perturbation gives rank intervals. Spatial: Moran's I / LISA.
- **Relations**: Q4 depends on Q3 and supplies demand and priority to Q5.
- **论文重述素材 (restatement)**: Based on the newly inundated areas of Problem 3, collect public data on population, buildings, roads, hospitals, shelters, terrain and other factors. Choose spatial assessment units and build a comprehensive regional disaster assessment model whose core indicators are the newly inundated area, the inundation proportion and the flood inundation intensity, while also accounting for the exposure of population and critical facilities.
- **论文分析素材 (analysis)**: This is a multi-criteria evaluation with a mandated hazard core. The design keeps the hazard indicators dominant: they form a separate first-level criterion and their weight is constrained to be at least that of exposure (an assumption, varied in the sensitivity analysis). Exposure enters through overlay of $F$ with population, built-up and road layers, so the assessment is directly driven by the image results. OSM facility data in China are sparse, so built-up and population rasters replace building and hospital counts, and the substitution is stated. Unit ranks and the affected population are the demand source of Q5.

### 问题五 (Problem 5): Rescue optimization and letter

- **Input**: Q3 flooded roads and areas; Q4 priority and affected population; the OSM road graph; candidate depot sites (town centres or schools outside $F$, on high ground).
- **Output**: a chosen task (recommended: **emergency supply allocation with depot activation on the flood-damaged road network**, i.e. a location–allocation MILP); a decision time point; objectives; demand model; resources; constraints; the optimal plan; a Pareto trade-off; sensitivity; the recommendation letter.
- **Difficulty**: medium–hard.
- **Method**:
  - Decision time: $T_0$ = 16 July 2026, after the Sentinel-1 pass is processed. A rolling update is shown with 4 July.
  - Network: OSM roads minus edges intersecting $F$ (with a passability buffer), with speeds by road class. Travel times come from shortest paths.
  - Demand: $d_j$ = affected population$_j$ × per-capita daily need (water, food, kits; standards from the Sphere Handbook) × horizon.
  - Priority: $w_j$ from the Q4 index.
  - MILP (PuLP/CBC):
    - Objective: minimize priority-weighted unmet demand and priority-weighted delivery time, solved as a bi-objective with the ε-constraint method.
    - Constraints: depot capacity, number of open depots $\le K$, fleet capacity, travel-time limit $T_{max}$, reachability only on intact roads.
- **Relations**: Q5 depends on Q3 and Q4. The letter synthesizes Q1–Q5.
- **论文重述素材 (restatement)**: Using the preceding identification and assessment results, choose one rescue task (site selection, supply allocation, route planning or evacuation) or propose a clearly defined one. State the decision time point, objectives, demand estimation, available resources and constraints, and build an optimization model that takes the recognition and assessment results directly as inputs. Finally, write a letter of recommendations to the local emergency management authority on flood remote sensing monitoring, disaster information updates and rescue resource allocation, including the implementation conditions and scope of applicability.
- **论文分析素材 (analysis)**:
  - Supply allocation with depot selection is chosen because it uses all three upstream products: $F$ removes roads and sites, $\rho$ and the grade scale demand, and the Q4 index sets priority. This makes the "monitoring → scheduling" link explicit and testable: re-running with the 07-04 map shows how decisions change when information updates.
  - Exact MILP solution of a moderate instance (tens of units, about 20 candidate sites) is feasible and gives an optimality gap.
  - The ε-constraint front exposes the trade-off between efficiency and coverage of the highest-priority units.

## 3. Data exploration (all files, full traversal)

`DATA_PROFILE.json` (machine profile) lists 3021 images in 12 folders, plus 13 vector files that the profiler did not profile. They were all read separately. Verification scripts: `_tmp/verify_official.py`, `_tmp/verify2.py`, `_tmp/verify3.py`, `_tmp/verify_public.py` (one-off, not part of `code/`).

### 3.1 Official Attachment 1 (training/validation only; Hunan–Jiangxi, not Hengzhou)

| Folder | Files | Shape / type | Notes |
|---|---|---|---|
| `Problem A Dataset 2/Dataset/pre-disaster optical images` | 1000 `before_s2_1001..2000.TIF` | 256×256×4 uint8 (B, G, R, NIR) | band means 58.6 / 68.4 / 75.1 / 118.1; 60 tiles have >5% zero (no-data) pixels |
| `Problem A Dataset 1/Dataset/post-disaster SAR images` | 1000 `after_gf3_hh_1001..2000.TIF` | 256×256 uint8 | tile mean range 14.2–254.9 (median 87.1) → per-tile normalization mandatory; 9 tiles have >5% zeros |
| `Problem A Dataset 1/Dataset/label information` | 1000 `label_1001..2000.tif` | 256×256 uint8 {0,1} | — |

- IDs 1001–2000 match one-to-one across the three folders.
- EPSG:4326. SAR and optical have pixel scale 8.333e-5°. Label pixel scales differ: 534 tiles at 8.333e-5°, 457 at 8.3349e-5° (a 0.005 px difference over a tile) and 9 tiles at 7.9119e-5° × 8.3349e-5°. The 9 anisotropic tiles need care (resample, or flag and exclude).
- Tie points differ between the three sets by at most 0.26 px (label vs SAR) and 0.47 px (label vs optical). The rasters are co-registered at sub-pixel level, so no resampling is needed apart from the 9 tiles.
- Labels: pixel-weighted flood share **18.99%**; tile median 3.76%; quartiles 1.21% / 20.57%; 15 empty tiles; 212 tiles <1%; 145 tiles >50%; 74 tiles >90%; max 99.99%.
- Space: 4 connected clusters (512, 450, 22 and 16 adjacent tiles), no overlapping tiles. Longitude blocks: 112.6–113.5°E has n = 383 and flood share 28.7%; 113.5–114.5°E has n = 151 and 2.3%; 115.5–117°E has n = 466 and 16.4%. Spatial folds should be built from cluster/longitude blocks, and fold-level class balance should be reported.
- **Label semantics** (pooled over all 65.5 M pixels, with pre-water = NDWI(G, NIR) > 0 and SAR-dark = normalized SAR < 0.2):

| Class | Pixels | SAR-dark share |
|---|---|---|
| label 1, pre non-water | 12,152,142 | 59.4% |
| label 1, pre water | 294,868 | 53.0% |
| label 0, pre water | 5,759,356 | **71.2%** |
| label 0, pre non-water | 47,329,634 | 12.8% |

  Permanent water is dark in the SAR yet labelled 0, so **labels encode new inundation**. Direct supervision therefore learns $F$, and a two-stage $W_1\setminus W_0$ model must reproduce this. Because the label already excludes permanent water, subtracting $\hat W_0$ from a model trained on the labels would double-subtract. This makes the Q1 design choice consequential.
- SAR mean inside labelled flood is 45.0, outside 118.1 (raw uint8), so separability is strong but tile-dependent.

### 3.2 Self-collected Hengzhou data (Q3–Q5)

| Layer | Spec (verified) | Key facts |
|---|---|---|
| Sentinel-1 RTC 2026-06-27 / 07-04 / 07-16 | 7500×8700×2 float32 dB (VV, VH), EPSG:32649, 10 m grid, valid ≈ 100%, scene 6525 km² | VV median −6.99 / −6.92 / −7.19 dB; VV < −16 dB covers 2.20% / 2.07% / 2.43% |
| Sentinel-2 L2A 2026-05-31 | 7500×8700×5 uint16 (B, G, R, NIR, SCL), same grid | SCL: 37% no-data, ~15% cloud (classes 8/9) → $W_0$ only partly from optics |
| WorldCover 2021 | 7675×9843 uint8, EPSG:4326, 10 m; 0 = outside boundary | classes 10 / 30 / 40 / 50 / 60 / 80; water 1,741,253 px; built 1,680,813 px (README: water 137.3 km², built 132.5 km², cropland 813.3 km²) |
| Copernicus DEM mosaic | 3600×4320 float32, 1″, tie 108.6°E 23.3°N | elevation median 101.7 m, 99th pct 521.7 m; 0 values over the outer area/water should be checked |
| WorldPop 2020 constrained | 769×985, 100 m | sum **939,822.6** inside boundary; max 91.2 people/pixel; valid 7.3% (unpopulated pixels are no-data) |
| GADM boundary | 1 multipolygon | 3440.4 km² |
| OSM | roads 2192 lines / 2542.3 km (motorway 324, tertiary 320, residential 532 …); places 393 (370 villages, 16 towns, 1 city); water surface 143.4 km²; waterways 593.1 km; buildings 609 (0.9 km² — very incomplete); amenities 5 (all schools) | buildings and hospitals cannot be counted from OSM |
| `coverage_gap.geojson`, `Browser_images*` | old partial exports (65.8% coverage) | obsolete; use the full-coverage `Sentinel1/*.tif` |

**Quick change check (descriptive, not a result)**: on 2× downsampled VV with a −16 dB threshold and one opening, "new dark" pixels versus 06-27 cover 91.4 km² (07-04) and 115.0 km² (07-16), while "disappeared dark" pixels cover 116.9 and 114.4 km² over the 6525 km² scene. Naive thresholding is dominated by speckle and seasonal noise, and the net dark-area change from 06-27 to 07-16 is only about +16 km². Implications:
- Q3 needs speckle filtering, multi-temporal stability, terrain masks and a minimum mapping unit.
- The Hengzhou flood signal may be moderate. Results must be reported honestly and not inflated.
- 07-04 is **before** the 6 July peak, and 07-16 is ten days after it. Both are "representative near-event" images, not peak images. This limitation must be stated. The user is asked to try to obtain a Sentinel-1 pass between 07-05 and 07-12 if one exists.

**Not available (stated as assumptions downstream)**: hospital and shelter POIs, township boundaries, observed flood extent for Hengzhou, and the S2 post-flood optical image (2026-07-13, still being collected).

## 3.5 Key concept alignment and task back-translation

| Problem text (sentence) | Our understanding | What the definition acts on | Symbol |
|---|---|---|---|
| "newly inundated area" (S8, S12, S19) | ground area that is water at flood time and was not water before | acts on the **set difference** of water masks, not on post-flood water | $A_{new}=\sum_{p\in W_1\setminus W_0}a_p$ [km²] |
| "net increase in inundated area" (S19) | change in total water area | acts on the **totals** $\lvert W_1\rvert-\lvert W_0\rvert$, which can be < $A_{new}$; not on $F$ | $A_{net}$ [km²] |
| "proportion of newly inundated area" / "its proportion of the total area of the study region" (S13, S19) | $A_{new}$ divided by the land area of the study region (or unit) | the denominator is the **study-region total area**, not the water area or the flooded area | $\rho=A_{new}/A_{reg}$ [–] |
| "relative area error" (S8) | (estimated − reference new area) / reference | acts on **new area**, signed; reported per tile and in total | $RAE=(\hat A-A)/A$ |
| "spatial overlap" (S8) | IoU / Dice of the new-inundation masks | acts on masks of $F$ | IoU |
| "independent samples" (S7) | spatially independent tiles (blocked CV / held-out clusters) | acts on the **sample split**, at tile/cluster level, not on pixels | — |
| "core indicators … newly inundated area and its proportion" (S13) | grading driven by $(A_{new},\rho)$ | thresholds act on **area and proportion**, not on depth, population or loss | $G(A_{new},\rho)$ |
| "core indicators … new area, proportion, intensity" (S23) | hazard criterion of the assessment model | exposure is an additional factor, not a replacement | $H_j$ |

**Task back-translation**: build a recognizer of new flood water whose *area* is accurate (Q1, Q2), turn area and proportion into a four-level grade with stated uncertainty (Q2), measure the new and net flood area in Hengzhou in July 2026 (Q3), combine that hazard with exposure in a unit-level assessment (Q4), and use it to optimize one rescue task (Q5) plus write a letter. The direction is to minimize area error, with no maximization target. Nothing is reversed. The easiest thing to drop is the separate *net* area and the *grade uncertainty* analysis, and both are kept.

## 3.6 Assumption sensitivity pre-check

### Ambiguous statements and interpretations
1. "a model for identifying flood-inundated areas" (Q1) + the binary labels
   - A: the labels are all post-flood water, so we must subtract pre-flood water.
   - B: the labels are already new inundation, so we learn $F$ directly and use $W_0$ only as an input feature or consistency check.
   - Choice: **B**, based on data evidence (§3.1: only 2.4% of label pixels were pre-water, and 71% of pre-water pixels are dark in the SAR but labelled 0). A is kept as the two-stage baseline for comparison.
   - 中文说明: 官方标签到底标的是"灾后所有水"还是"新增淹没" | A: 标签是灾后全部水体，要再减去灾前水体 | B: 标签本身已是新增淹没（原有河湖标为0），直接学新增
2. "proportion of the total area of the study region" (Q2 grading)
   - A: the region is each 2.56 km training tile.
   - B: the region is any assessment unit; thresholds are on $\rho$ (scale-free), with area thresholds given for a stated unit size.
   - Choice: **B**, because Q3 and Q4 apply the grading to Hengzhou units of different sizes, and a proportion is comparable across sizes.
   - 中文说明: 分级用的"研究区总面积"指什么范围 | A: 就是每张训练切片 | B: 任意评价单元，按比例分级，面积阈值按单元大小换算
3. "imagery acquired before and during the flood event" (Q3)
   - A: during = 07-04 (closest date, but before the 6 July peak).
   - B: during = 07-16 (after the peak, residual flooding).
   - C: use both as a time series.
   - Choice: **C**. Report 07-16 and 07-04 separately (whichever shows more new water is the event map) and state that neither date is the peak.
   - 中文说明: 横州"灾中"影像选哪一期 | A: 用7月4日（离洪峰最近但在峰前） | B: 用7月16日（峰后10天，残余积水） | C: 两期都用，做时间序列对比
4. "select one specific task" (Q5)
   - A: site selection only.
   - B: supply allocation with depot opening (location–allocation).
   - Choice: **B**. It is one task (supply allocation) whose depot sites are decision variables, and it uses all upstream outputs.
   - 中文说明: 第五问选哪个救援任务 | A: 只做救援点选址 | B: 做物资分配，同时决定开哪几个物资点

### Magnitude of the key ambiguity (estimate only, not a result)
- Interpretation A of item 1 (treating labels as all post-water and subtracting NDWI water): it would remove only 2.4% of label pixels but would misclassify the about 71% of permanent-water pixels that are SAR-dark wherever $\hat W_0$ misses them. On the pooled data, permanent water is about 9.2% of pixels, about half of the label share. A model trained under A that predicts all dark water would overestimate new area by up to $\approx 0.712\times5.76\text{M}/12.45\text{M}\approx 33\%$.
- Interpretation B: no such bias. This is why the choice matters.

### Expected progression across questions
- Q1 baseline: area error is expected to be the largest on low-contrast and urban or vegetated tiles.
- Q2 should reduce total |RAE| and the tile MAE versus Q1 on the same spatial folds. If it does not, the reason is reported, not hidden.
- Q3 transfers with larger uncertainty (domain shift and no labels), so its intervals should be wider than Q2's validation intervals.
- Q4 ranking must be dominated by the hazard core but differ from a pure hazard ranking where exposure differs.
- Q5's plan should change when Q3/Q4 inputs change (07-04 vs 07-16 information update). Equal plans would be a reportable finding.

### Final choices
B (labels = new inundation), B (proportion-based grading), C (multi-date Hengzhou), B (location–allocation).

## 4. Variables and symbols

| Symbol | Meaning | Unit | Type |
|---|---|---|---|
| $p$ | pixel index | – | index |
| $a_p$ | ground area of pixel $p$ ($\approx 75.1$ m² for the official tiles, 100 m² for S1) | m² | data |
| $I_{SAR}(p)$ | post-flood SAR intensity (GF-3 HH uint8 / S1 VV, VH dB) | DN / dB | observed |
| $\tilde I(p)$ | normalized, speckle-filtered SAR | – | derived |
| $B,G,R,N$ | pre-flood optical bands | DN / reflectance | observed |
| $NDWI$, $NDVI$ | $(G-N)/(G+N)$, $(N-R)/(N+R)$ | – | derived |
| $y_p$ | label (1 = newly inundated) | – | observed |
| $W_0, W_1$ | pre-flood and flood-time water masks | set | derived |
| $F$ | new-inundation mask $W_1\setminus W_0$ | set | output |
| $\hat P_p$ | predicted probability of new inundation | – | output |
| $\tau$ | decision threshold | – | parameter |
| $A_{new}, A_{net}$ | new and net inundated area | km² | output |
| $A_{reg}$ | area of the study region / unit | km² | data |
| $\rho$ | new-inundation proportion | – | output |
| IoU, F1, OA, κ | overlap and accuracy metrics | – | metric |
| RAE, MAE$_A$ | relative area error, mean absolute tile area error | –, km² | metric |
| $\theta_1<\theta_2<\theta_3$ | proportion thresholds of the 4 grades | – | output |
| $G_j$ | intensity grade of unit $j$ (1–4) | – | output |
| $\pi_{jg}$ | probability that unit $j$ is in grade $g$ | – | output |
| HAND$(p)$ | height above nearest drainage | m | derived (DEM) |
| $P_j, B_j, R_j$ | flooded population, built-up area, road length in unit $j$ | persons, km², km | derived |
| $w_k$ | indicator weights | – | derived |
| $S_j$ | composite disaster index of unit $j$ | – | output |
| $d_j$ | supply demand of unit $j$ | kg or kits | derived |
| $x_i$ | depot $i$ opened (binary) | – | decision |
| $f_{ij}$ | supply flow from depot $i$ to unit $j$ | kg | decision |
| $u_j$ | unmet demand | kg | decision |
| $t_{ij}$ | travel time on the flood-cut road network | min | derived |
| $C_i, K, T_{max}$ | depot capacity, max open depots, delivery time limit | kg, –, min | assumed parameters |

## 5. Modeling plan

### 5.1 Recommended chain and candidate methods

| Q | Recommended | Alternative / baseline | Validation |
|---|---|---|---|
| Q1 | Preprocessing (quantile stretch, refined-Lee filter, zero masking, area grid). Then (a) physics two-stage Otsu-SAR minus NDWI, (b) LightGBM pixel model on multi-scale features | Random forest; plain SAR threshold | spatial block CV over clusters / longitude blocks; pixel + area metrics; FP/FN area by pseudo land cover; tile error drivers (SHAP / regression) |
| Q2 | Dual-branch fusion U-Net (SAR + optical [+ HAND-free in Q1 tiles]) with Dice + Focal + area-consistency loss; uncertainty (MC-dropout or ensemble); error-adjusted area + split-conformal intervals; GMM / Jenks thresholds on $\log\rho$; MC grade-flip | LightGBM + CRF post-processing | same folds as Q1; ablation per module; interval coverage; threshold stability (bootstrap) |
| Q3 | Quantile-mapped S1 VV → Q2 model on tiles; split-based Otsu cross-check; HAND/slope mask; $W_0$ fusion (06-27 SAR ∩ S2 NDWI ∩ WorldCover/OSM); 07-04 and 07-16 | pure Otsu change detection (log-ratio) | agreement of model vs physics; permanent-water recall vs WorldCover/OSM; sensitivity to the thresholds |
| Q4 | 1 km grid → town-Voronoi units; H–E–R indicator system; entropy + CRITIC game-theory weights; TOPSIS; Dirichlet weight MC; LISA | AHP-entropy; cloud model | rank stability (Spearman, rank CI); hazard-only vs full ranking |
| Q5 | Location–allocation MILP on the flood-cut OSM graph; ε-constraint bi-objective; CBC exact | greedy heuristic as a comparator only | optimality gap; constraint audit; sensitivity to capacity, $K$, $T_{max}$, demand norm; 07-04 vs 07-16 information update |

GPU note: torch 2.13 is the CPU build; an RTX 4050 (6 GB) is present. A small U-Net (about 1–2 M parameters) on 1000×256² tiles is trainable on CPU in tens of minutes per fold. Installing CUDA torch is optional. LightGBM runs in minutes.

### 5.2 Innovation points (to avoid template solutions)
1. **Empirical label-semantics test**: a contingency analysis of label × pre-water × SAR-darkness shows that the official labels are *new* inundation. This settles the definition of $F$ with evidence instead of assumption.
2. **Area-first learning and inference**: an area-consistency loss plus an error-adjusted (stratified) area estimator. The area, not the pixel accuracy, is the primary deliverable and carries a confidence interval.
3. **Conformal uncertainty carried into grading**: thresholds come from the mixture structure of $\log\rho$, and each unit gets a *grade probability vector*. The effect of area error on classification is quantified by grade-flip rates.
4. **Physics-guarded cross-sensor transfer**: a sensor-agnostic normalized representation, plus a HAND/slope terrain prior and multi-date consistency, replaces unverifiable "fine-tuning" on a domain without labels.
5. **End-to-end information chain**: Q3 rasters remove road edges and candidate sites, Q4 scores set demand and priority in the MILP, and an "information-update" experiment (07-04 → 07-16) quantifies the value of new imagery for decisions.

## 5.5 Hard constraints (HARD_CONSTRAINTS)

| ID | Description | Expression | Variables | Type |
|---|---|---|---|---|
| C1 | new inundation excludes pre-existing water | `not np.any(F & W0)` | F, W0 | static (per date) |
| C2 | new ≤ flood-time water | `np.all(F <= W1)` | F, W1 | static |
| C3 | net ≤ new area | `A_net <= A_new + 1e-9` | areas | static |
| C4 | proportion bounds | `0 <= rho <= 1`, `rho == A_new/A_reg` | rho | static |
| C5 | independent validation: no tile in both train and test of a fold; folds spatially separated | `set(train_ids).isdisjoint(test_ids)` and the block rule | fold ids | static |
| C6 | grade thresholds strictly increasing; the 4 grades partition [0, 1] | `0 < t1 < t2 < t3 < 1` | thresholds | static |
| C7 | grade probabilities sum to 1 | `abs(pi_j.sum()-1) < 1e-6` | pi | static |
| C8 | areas use true pixel area (lat-dependent for EPSG:4326) | `a_p = dx*dy*cos(lat)` | a_p | derived |
| C9 | assessment weights non-negative, sum to 1; hazard criterion weight ≥ exposure weight (stated assumption) | `w>=0, sum(w)==1` | w | static |
| C10 | MILP flow conservation: delivered + unmet = demand | `sum_i f_ij + u_j == d_j` | f, u, d | static |
| C11 | depot capacity / only open depots ship | `sum_j f_ij <= C_i * x_i` | f, x | static |
| C12 | number of open depots | `sum_i x_i <= K` | x | static |
| C13 | only reachable pairs on the flood-cut network within $T_{max}$ | `f_ij == 0 if t_ij > T_max or unreachable` | f, t | **derived from Q3 F** (date-dependent) |
| C14 | candidate depots not inside $F$ | `F[site_i] == 0` | x | **derived from Q3 F** |
| C15 | Q5 demand derived from Q4 / Q3 outputs (no free demand) | `d_j == P_j * q * H` | d | derived |

## 5.6 Sentence-by-sentence decomposition

| 句号 | Sentence (abridged) | Type | Elements |
|---|---|---|---|
| S0 | Background on floods; Fig. 1 Huarong dike breach, May 24 and July 6, 2024 | 背景 | context only |
| S1 | "develop mathematical models for the specified study area and flood event" | 目标 | overall |
| S2 | "Except for Problem 1, imagery and reference data … may be independently obtained through Copernicus Browser …" | 机制（可选） | public data allowed |
| S3 | "data sources, acquisition times, spatial coverage, and processing methods should be clearly specified" | 约束 | data provenance table |
| S4 | "assumed parameters must be distinguished from actual observational data, and their impacts … analyzed" | 约束 | assumption register + sensitivity |
| S5 | "perform the necessary image preprocessing" | 决策 | normalization, speckle, registration |
| S6 | "develop a model for identifying flood-inundated areas" | 目标 | recognizer |
| S7 | "Evaluate the recognition performance … using independent samples" | 目标 | spatial split |
| S8 | "pixel-level accuracy … estimation error of newly inundated area, relative area error, … spatial overlap" | 目标 | metrics |
| S9 | "Analyze how FP and FN under different land-cover conditions affect … new area" | 目标 | stratified errors |
| S10 | "investigate the key factors influencing the accuracy" | 目标 | error drivers |
| S11 | "for cases … poor or uncertainty … high, investigate multi-source information fusion and model improvement" | 决策 | fusion model |
| S12 | "primary objectives of improving the accuracy of newly inundated area estimation and reducing area estimation errors" | 目标 | area-oriented |
| S13 | "establish a flood inundation intensity classification … new area and its proportion … as core indicators" | 目标 | grading |
| S14 | "Clearly define the area or proportion thresholds … general, relatively strong, strong, extremely strong" | 决策 | 4 thresholds |
| S15 | "analyze the effects of area identification errors and their uncertainties on the classification results" | 目标 | grade uncertainty |
| S16 | "Apply the model developed in Problem 2 to the flooding … around July 6, 2026, in Hengzhou City" | 机制 | transfer |
| S17 | "Independently determine an appropriate study area and select representative imagery … before and during" | 决策 | AOI, dates |
| S18 | "Identify pre-existing water bodies and newly inundated areas" | 目标 | W0, F |
| S19 | "quantitatively calculate the newly inundated area, the net increase …, and the proportion" | 目标 | A_new, A_net, ρ |
| S20 | "present the spatial distribution of the newly inundated areas" | 目标 | map |
| S21 | "obtain publicly available data on population, buildings, roads, hospitals, shelters, terrain …" | 数据 | exposure layers |
| S22 | "Independently determine appropriate spatial assessment units" | 决策 | units |
| S23 | "establish a comprehensive regional disaster assessment model … core indicators … considering exposure" | 目标 | assessment |
| S24 | "select one specific task for optimization …" | 决策 | task choice |
| S25 | "specify the decision-making time point, optimization objectives, demand estimation methods, available resources, and constraints, and establish … optimization model" | 目标 | MILP |
| S26 | "results of image recognition and disaster assessment are directly incorporated into rescue decision-making" | 机制 | data link |
| S27 | "write a letter of recommendations … remote sensing monitoring, disaster information updates, and … allocation of rescue resources" | 目标 | letter |
| S28 | "explain the implementation conditions and scope of applicability" | 约束 | letter content |

**Five-question review (key points)**: S9 needs land cover for the official tiles, which is not supplied, so it is derived from optical indices (an assumption, stated) with optional WorldCover download. S14 "area or proportion" is handled by proportion thresholds plus converted area thresholds. S16 has a single-mention risk, "model developed in Problem 2": Q3 must reuse the Q2 model and not switch to a new pure-threshold method, though physics checks are allowed. S21 hospitals and shelters are missing in OSM, so a proxy plus assumption is stated (severity: serious if ignored silently). S26 is the most-often-missed mechanism, and C13–C15 enforce it.

**Reverse mapping**: every model component traces to a sentence. HAND terrain prior → S11 (multi-source information) and S16. Conformal intervals → S15. Weight MC → S4. Info-update experiment → S26/S27. The components that S1–S28 do not ask for and that are our own additions are the LISA hotspot analysis and the ε-constraint front, both optional.

## 5.7 Classic-model upgrade check

| Initial mapping | Trigger | Final | Severity | Necessity | Effect if missing |
|---|---|---|---|---|---|
| water segmentation | "newly inundated" (S8, S18) | change segmentation $W_1\setminus W_0$ | 🔴 fatal | required | area overestimated (≈ +33%, §3.6) |
| pixel classification metrics | "area error", "relative area error" (S8, S12) | area-oriented loss + estimator | 🔴 fatal | required | the main objective is not addressed |
| deterministic grading | "effects of errors and uncertainties" (S15) | probabilistic grading | 🟡 important | required | S15 is not answered |
| in-domain model | different sensor in Q3 (S16) | domain adaptation + physical checks | 🟡 important | required | biased Hengzhou area |
| static TOPSIS | "exposure of populations and critical facilities" (S23) | H–E–R with overlay exposure | 🟡 important | required | the assessment ignores the image result |
| plain transport LP | "results … directly incorporated" (S26) | flood-cut network + Q4 demand | 🔴 fatal | required | Q5 is disconnected from Q3/Q4 |
| single-objective MILP | efficiency vs coverage | ε-constraint front | 🟢 optimization | suggested | less insight |

Missing-parameter plan:
- Depot capacity, fleet and $T_{max}$: no source is given, so they become parameters with ranges and sensitivity.
- Per-capita needs: Sphere Handbook minimum water of 15 L/person/day (cite).
- Road speed by class: Chinese design speeds (cite JTG B01), with a flood penalty as a sensitivity parameter.
- HAND threshold: varied over 5–20 m.

## 6. Figure pre-planning

The data-figure budget is 35 × 0.4 = 14, and the plan uses 14 (at most 4 per question). All data figures come from real computed results (official tiles or Hengzhou data). None is simulated. The style is Nature-engine. The language is English.

### 6.1 Data figures
- fig_q1_visual_cmp — qualitative comparison [4-panel] (academic #9) — four representative test tiles (cropland, urban, vegetated, high-flood), each showing pre-optical RGB, post-SAR, label, and prediction colour-coded TP/FP/FN with a zoom inset. It shows *where* area errors come from — Problem 1 results (facade figure, first).
- fig_q1_label_semantics — ridgeline (advanced #23) — distributions of normalized SAR intensity for the four classes (label × pre-water). Permanent water is dark but labelled 0, which proves the labels are new inundation — Problem 1 data analysis.
- fig_q1_area_agreement — Bland–Altman + predicted-vs-reference [2-panel] (advanced #8) — tile-level new area, predicted vs reference with limits of agreement, for the baseline models on spatial folds — Problem 1 evaluation.
- fig_q1_error_drivers — diverging bar + SHAP summary [2-panel] (advanced #20) — left: FP area (+) and FN area (−) contribution to the new-area error by pseudo land-cover class; right: SHAP of tile area error versus tile descriptors — Problem 1 error analysis.
- fig_q2_model_compare — method comparison heatmap (advanced #16) — models (threshold, two-stage, LightGBM, U-Net single, fusion U-Net, + area correction) × metrics (IoU, F1, RAE, MAE_A, interval coverage) — Problem 2 results.
- fig_q2_ablation — waterfall (advanced #6) — step-by-step reduction of total |area error| from the Q1 baseline as modules are added (fusion, area loss, uncertainty refinement, error-adjusted estimator) — Problem 2 ablation.
- fig_q2_grading — filled density with mixture components + confusion matrix [2-panel] (journal #32) — left: density of $\log\rho$ with fitted GMM components, the 3 thresholds and grade bands; right: grade transition matrix under Monte Carlo area uncertainty (competition #10 style) — Problem 2 grading.
- fig_q3_newflood_map — geographic map with zoom inset [2-panel] (custom) — Hengzhou boundary: pre-existing water $W_0$, new inundation $F$ (07-16) and receded water; zoom on the Yu River floodplain; a second panel for 07-04 — Problem 3 results.
- fig_q3_threshold_surface — contour / response surface (competition #14) — $A_{new}$ as a function of the VV threshold offset and the HAND threshold, with the chosen point and the grade boundaries overlaid — Problem 3 sensitivity.
- fig_q3_area_budget — dumbbell (advanced #2) — per land-cover class (cropland, built-up, grass, forest, bare): new inundated area on 07-04 vs 07-16, with net change marked — Problem 3 results.
- fig_q4_risk_map — bivariate choropleth (advanced #34) — assessment units coloured by hazard tercile × exposure tercile, with the final index rank labels for the top units — Problem 4 results.
- fig_q4_rank_robustness — dot plot with CI (advanced #13) — rank median and 90% interval per unit under Dirichlet weight perturbation, compared with a hazard-only rank — Problem 4 robustness.
- fig_q5_route_alloc — network path / route map (competition #16) — flood-cut road network (removed edges in red), open depots and allocation flows (width ∝ supply) to the demand units, coloured by priority — Problem 5 results.
- fig_q5_pareto — Pareto front (competition #3) — weighted delivery time vs weighted unmet demand from the ε-constraint method, for the 07-16 and 07-04 information states, with the knee solution — Problem 5 trade-off and information value.

### 6.2 Tables (planned)
- TABLE data provenance (source, date, resolution, coverage, processing; S3).
- TABLE Q1/Q2 metrics per fold.
- TABLE intensity thresholds (proportion and area for the stated unit size).
- TABLE Q3 areas ($A_{new}$, $A_{net}$, $\rho$, intervals, per date).
- TABLE Q4 indicator system and weights.
- TABLE Q5 parameters (observed vs assumed, S4) and optimal plan.
- TABLE assumption register with sensitivity results.

### 6.3 HTML flow / architecture diagrams
- fig_roadmap — technical roadmap (HTML) — data chain Q1 → Q2 → Q3 → Q4 → Q5: inputs (Attachment 1 / Hengzhou layers), methods per question, transferred objects (definition of $F$, model, thresholds, $F$ raster, priority and demand), letter — Introduction (end of restatement). [HTML]
- fig_arch_fusion — model architecture (HTML) — dual-branch SAR/optical encoder, attention fusion, decoder, uncertainty head, area-consistency loss and error-adjusted area estimator — Problem 2 model section. [HTML]

Per-question flowcharts are not planned (`MH_FLOW_PER_PROBLEM` not set).

### 6.4 TikZ
- tikz_q1_set_logic — set-relation diagram (TikZ) — $W_0$, $W_1$, $F=W_1\setminus W_0$, receded water $W_0\setminus W_1$, with FP and FN regions relative to the reference $F$. It defines new area and net increase geometrically — Problem 1 model formulation.

### 6.5 Diversity check
Each chart type is used once: academic #9, advanced #23, #8, #20, #16, #6, #2, #34, #13, journal #32, competition #14, #16, #3, plus one custom map. There are 3 geographic maps (Q3, Q4, Q5). Model comparison uses ≥2 types: heatmap and waterfall, plus Bland–Altman. Sensitivity uses a contour (Q3) and a rank-CI plot (Q4), not tornado charts.

Total: 14 data figures + 7 tables + 2 HTML + 1 TikZ + 0 GPT Image.

## 7. Work plan

| Step | Content | Time estimate |
|---|---|---|
| comp-modeling | formalize Q1–Q5 equations, losses, estimator, grading, assessment, MILP; assumption register | 1 session |
| comp-code Q1 | preprocessing + baselines + spatial CV + error analysis (full 1000 tiles) | ~1 h CPU |
| comp-code Q2 | fusion U-Net (CPU or optional CUDA), uncertainty, area estimator, grading MC | 1–2 h |
| comp-code Q3 | S1 transfer over the full 7500×8700 scenes (tiled), W0 fusion, sensitivity grid | ~1 h |
| comp-code Q4/Q5 | units, indicators, weights MC; network build, MILP + ε-constraint | ~0.5 h |
| figures / diagrams / paper / compile | per pipeline | remaining steps |

Downstream reminders:
- Re-verify the label semantics with a stricter $W_0$ (MNDWI is not available because there is no SWIR band, so use NDWI + SAR-dark consistency).
- Handle the 9 anisotropic label tiles.
- Never mix the official and Hengzhou datasets.
- Keep the area definition ($a_p$ by latitude) identical everywhere.
- All AI disclosure is inserted by script; the paper body does not mention AI tools.

<!-- BEGIN FIGURE_MANIFEST -->
## 图表清单（FIGURE_MANIFEST）

**数据图（matplotlib gen_fig_*.py，paper-figure 产出 .png/.pdf）：**
- fig_q1_visual_cmp — qualitative comparison [4-panel] (academic #9)
- fig_q1_label_semantics — ridgeline (advanced #23)
- fig_q1_area_agreement — Bland-Altman [2-panel] (advanced #8)
- fig_q1_error_drivers — diverging bar + SHAP [2-panel] (advanced #20)
- fig_q2_model_compare — method comparison heatmap (advanced #16)
- fig_q2_ablation — waterfall (advanced #6)
- fig_q2_grading — filled density + grade confusion matrix [2-panel] (journal #32)
- fig_q3_newflood_map — geographic map with zoom inset [2-panel] (custom)
- fig_q3_threshold_surface — contour response surface (competition #14)
- fig_q3_area_budget — dumbbell (advanced #2)
- fig_q4_risk_map — bivariate choropleth (advanced #34)
- fig_q4_rank_robustness — dot plot with CI (advanced #13)
- fig_q5_route_alloc — network route map (competition #16)
- fig_q5_pareto — Pareto front (competition #3)

**HTML 流程/架构图（paper-figure-html 产出 .png/.pdf）：**
- fig_roadmap — technical roadmap of the Q1→Q5 data chain [HTML]
- fig_arch_fusion — dual-branch fusion network architecture [HTML]

**TikZ 图（paper-figure 产出 tikz_*.pdf）：**
- tikz_q1_set_logic — set-relation diagram of W0, W1, F, FP, FN

**总数：DATA=14, DRAWIO=2, TIKZ=1, GPTIMG=0, ALL=17**
<!-- END FIGURE_MANIFEST -->
