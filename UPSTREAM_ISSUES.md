# Upstream issues found in the comp-code step

Each item: problem → evidence → affected question → what the code did. MODELING_REPORT.md was not edited.

1. **Self-collected Sentinel-1 rasters are mis-georeferenced (data issue).**
   Evidence: SAR-dark share on permanent water (WorldCover 80 ∩ OSM water) 0.034–0.049 at the stated geotransform; 0.346–0.378 after integer translations dr = −623 / −544 / −539 rows (06-27 / 07-04 / 07-16), i.e. 5.4–6.2 km north–south, with 0.8 km difference between dates (`q3_results.json::coregistration`). The user's preview PNG uses the same wrong transform, so it looks aligned with the boundary.
   Affects Q3–Q5. Code: co-registration step in `problem3.prepare_layers` before any use. The paper should report this as data verification (pre-processing "heterogeneous registration").

2. **Sentinel-2 2026-05-31 scene cannot be used (data issue).**
   Evidence: NDWI > 0 share on reference water 0.026 at zero shift, best 0.041 over translations ±800 × ±320 m (`q3_results.json::s2_check`); visual check shows the river pattern in a different position/shape.
   Affects Q3 (optical input, W0 S2 term). Code: optics off (m = 0), the modality-dropout model M4/M6 handles it. A correctly georeferenced optical scene would let the fused branch work in Hengzhou.

3. **Split-based Otsu rule of §4.2 fails on calibrated S1 VV.**
   Evidence: with 64-px blocks and no physical check, 52 % of the city was "water" at −6.9 dB (land-cover contrasts pass Ashman D > 2).
   Affects Q3 physics detector D_d, W0, transfer checks. Code: 128-px sub-blocks and dark-class mean ≤ −15 dB (`split_otsu_water`). The official-tile baseline A keeps the literal rule (DN scale unknown) and shows the same failure (RAE +1.41).

4. **(6.4) W1 definition counts mapped-but-never-SAR-visible water as receded.**
   Evidence: literal A_net = −153.2 km² (07-16), 86 km² of it WorldCover water that SAR never sees as water at 20 m effective resolution.
   Affects Q3 A_net. Code: observed-recession definition (W0 pixel SAR water on 06-27 and not on the flood date) → A_net = −52.5 km²; literal value reported alongside.

5. **Early-stopping criterion |Σ ΔA| (§5.2) selects untrained models.**
   Evidence: first run kept epoch 0 of M1 fold 0 (|ΣΔA| 32.8 km² by FP/FN cancellation).
   Affects Q2 training. Code: calibration tile MAE of area.

6. **CBC 2.10 stalls on Windows.** One sensitivity solve made no CPU progress for 20 min. Q5 uses HiGHS (exact, gapRel 0); CBC selectable.

7. **Comonotone interval aggregation (§6.4) is uninformative at city scale** ([0, 243.7] km² over 449 tiles). An independent-error aggregation [13.5, 40.3] km² is reported as a labelled secondary interval; neither is a calibrated city-level interval.

8. **Q5 unmet share is dominated by road-network isolation.** 75 affected units (Q4) / 57 demand units (Q5) have no road path within T_max after the flood cut (network splits into 745 components). f1* = 0.570 is insensitive to K, capacity, fleet and T_max. This is a finding for the letter (boats/air for isolated units), not a solver issue.

9. **P5-C4 (recommendation letter)** is a paper-step deliverable; all numbers it needs are in `problem_5_results.json` (plan, VoI, sensitivity) and `problem_3_results.json` (timing gap, guard dependence).

## Found in the figure step (nature-figure), not fixed here

10. **Spurious pre-flood water / recession strip at the northern co-registration edge (Q3).**
    Evidence: `output/q3_w0.tif` has 1071–1432 W0 pixels per row in rows 560–623 and only 5–32 per row from row 624 on; row 623 is exactly the 06-27 shift dr = −623, i.e. the edge of the no-data band created by the translation. The strip lies inside the Hengzhou boundary (boundary top ≈ row 170). Area in rows 600–700: W0 3.40 km², receded R (07-16) 3.36 km², new inundation F 0.00 km² (checked with a one-off script on the saved rasters). It is visible as a horizontal band at northing ≈ 2555 km in `fig_q3_newflood_map`.
    Impact: A_new / ρ / grade (Q3) and everything downstream (Q4, Q5) unaffected because F = 0 there. A_W0 (292.57 km²) and A_R (79.44 km²) each include ≈ 3.4 km² of edge artefact, so A_net (−52.53 km²) is about 3.4 km² too negative. Suggested fix in comp-code: mask the rows/cols left without valid pre-flood SAR after the shift (plus a few pixels of the Lee-filter footprint) before the SAR water term of W0 and the recession term.

## Found in the paper-writing step (comp-paper-en), not fixed here

11. **Tile count of the Q3 comonotone interval.** RESULTS.md (Q3 uncertainty) says "comonotone aggregation of the 449 active tile intervals". `problem_3_results.json::dates.20260716.n_active_tiles` = 344 (07-04: 314), out of 603 tiles in the boundary. The paper quotes only the interval values ([0, 243.7] km² and [13.48, 40.33] km², which match the JSON) and no tile count. Affects Q3 text only.

12. **Q3 water-balance definition in MODELING_REPORT (6.4) vs code.** (6.4) writes W1 = F ∪ (W0 ∩ D_d); the reported A_net = −52.53 km² equals A_W1 − A_W0 with A_W1 = 240.03 km², i.e. W1 = F ∪ (W0 \ R) with observed recession R = (W0 ∩ D_0627) \ D_d (A_R = 79.44 km²). The literal (6.4) set has A_W1_literal = 139.38 km². The paper states the code definition and reports the literal value as an alternative (already listed as item 4).

13. **Derived number used in the Q4 text.** The paper states that units with centroid northing 2520–2535 km hold 16.78 km² of new water and 8298 flooded people, and the 56 units south of northing 2505 km hold 1.15 km² and 237 people. These are sums over `output/q4_units.geojson` (H1, E1, centroid in EPSG:32649) made in the writing step for the description of `fig_q4_risk_map`; they are not stored in a results JSON.
