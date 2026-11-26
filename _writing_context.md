## 1 Introduction / 2 Assumptions / 3 Data
- Core: F = W1 \ W0 shared definition (eq:def_F); A_net = A_new - |R| (eq:net_identity); area error = FP - FN (eq:fpfn_identity)
- Numbers: 1000 tiles, 65,491,966 valid px, label-1 in pre-water 2.37% (2.06% strict), 70.1% of pre-water label0 SAR dark; flood share 18.99%, median 3.76%, 15 empty; pixel area 74.78-75.41 m2, mean tile 4.913 km2; ENL median 31.0; coreg official 491 tiles median shift 0, 63.3% within 2 px; Hengzhou boundary 3440.09 km2
- Symbols: shipments x_ij, indicators X_jk, eta_i, u_j, t_ij, A (admissible), f1,f2; Assumptions (1)-(6)
- Figures: fig:roadmap (sec1), fig:q1_label_semantics, fig:q1_set_logic (sec3); tab:hz_data

## 4 Problem 1
- Blocked 5-fold CV eq:folds; baseline A eq:ashman; LightGBM eq:tau_f1; metrics eq:area_metrics; FP/FN strata eq:fpfn_strata
- A: IoU 0.278 RAE +1.406 MAE 1.756; M0: OA 0.900 IoU 0.598 F1 0.749 kappa 0.686 RAE +0.092 MAE_A 0.430; ref 933.41, est 1019.12, dA +85.71; BA bias +0.086 [-1.42,1.60]; LOCO IoU 0.508; 397 hard tiles; SHAP sar_std 0.196
- Figures: fig:q1_visual_cmp, fig:q1_area_agreement, fig:q1_error_drivers; tables tab:q1_metrics, tab:q1_fpfn

## 5 Problem 2
- eq:gate, eq:loss, eq:isotonic, eq:tauA, eq:conformal, eq:area_thr, eq:gmm_bayes, eq:theta_values, eq:grade_mc
- M2 IoU 0.798 MAE 0.149; M6 IoU 0.732 RAE +0.0145 MAE 0.236 dA 13.52; hard IoU 0.266->0.562; theta (0.0593, 0.530, 0.969); coverage 0.878; grade acc 0.862 vs 0.543; MC 0.670
- Figures fig:arch_fusion, fig:q2_model_compare, fig:q2_ablation, fig:q2_grading; tables tab:q2_ablation, tab:grade_thr

## 6 Problem 3
- eq:coreg, eq:W0, eq:ks, eq:rem, eq:F_hz, eq:W1_R (R observed recession; W1 = F u (W0 \ R)), eq:areas_hz
- A_new 24.53 / 26.90; A_net -54.70 / -52.53; rho 0.713% / 0.782%; grade general, pi 0.72/0.26/0.016/0; guards remove 82.6%; tauA 0.557; KS 0.106 vs 0.241; shifts -623/-544/-539 rows
- Figures fig:q3_newflood_map, fig:q3_area_budget, fig:q3_threshold_surface; tables tab:hz_data (sec3), tab:q3_results

## 7 Problem 4
- eq:game_weights, eq:ahp, eq:topsis; 393 units, 259 affected; 13,954 people; weights table tab:q4_ind; Spearman 0.987
- Figures fig:q4_risk_map, fig:q4_rank_robustness

## 8 Problem 5 / Letter (in main.tex) / 10 Evaluation
- eq:demand, eq:admissible, eq:eta, eq:q5_obj, eq:q5_cons, eq:voi; f1* 0.570, f2 13.05, 7316 of 13,948 served, 57 isolated units; VoI1 0.039, VoI2 5.54 min; knee 0.662 / 4.79 min
- Figures fig:q5_route_alloc, fig:q5_pareto; tables tab:q5_plan, tab:sens_chain
