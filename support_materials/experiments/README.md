# experiments — improvement round v1

Branch `test/improvements-v1`（基于 `main@572ef93`，不改 main）。所有脚本只读 `output/` 与 `user_data/`，写 `experiments/results/`。

| 脚本 | 作用 |
|---|---|
| `e1_q1_uncertainty.py` | Q1 10 km 空间块 bootstrap（IoU/RAE）+ 斑块尺寸分层 + 误检碎斑结构 |
| `e2_q5_baselines.py` | Q5 MILP / 贪心 / 容量比例 / 无图反事实 / 无图方案在真实断路网回放 |
| `e3_q3_label_free.py` | Q3 无标签核验：W0 来源、S2 光学对照、VH 交叉极化、地形位置 |
| `e4_dual_grading.py` | Q4 双尺度分级（share vs 面积 GMM），输出逐单元 CSV |
| `e5_q2_rae_ci.py` | Q2 M0/M6 面积误差的块级与瓦片级 bootstrap，含配对差 |
| `e6_data_card.py` | 输入图层 CRS/分辨率/重采样数据卡 |
| `e7_capacity.py` | 从 Q2 训练日志做容量审计（参数量、早停、epoch 预算） |
| `e8_q1_postprocess.py` | Q1 LightGBM 掩膜的形态学/MMU 网格，找不重训的提分点 |
| `e9_q2_mask_combos.py` | Q2 缓存掩膜的并/交/投票/MMU 组合 |
| `e10_mmu_fairness.py` | 统一 50 px MMU 下 Q1/Q2 全模型（含基线）与硬集对比 |
| `e11_postprocess_bootstrap.py` | Q1 +MMU、Q2 M2+MMU vs M6 的配对块 bootstrap |
| `e12_mmu_table_metrics.py` | Q1/Q2 raw 与 50 px MMU 的完整表格指标（OA/F1/kappa/MAE/median） |
| `e13_q1_mmu_grid.py` | Q1 基线 A / LightGBM 的 MMU 扫描（论文图） |
| `e14_m2_area_calibration.py` | M2 逐折 isotonic + tau_A 面积校准，与 M2+MMU / M6 对比 |

运行：

```powershell
C:\Users\40461\miniconda3\python.exe experiments\e1_q1_uncertainty.py
C:\Users\40461\miniconda3\python.exe experiments\e2_q5_baselines.py
C:\Users\40461\miniconda3\python.exe experiments\e3_q3_label_free.py
C:\Users\40461\miniconda3\python.exe experiments\e4_dual_grading.py
C:\Users\40461\miniconda3\python.exe experiments\e5_q2_rae_ci.py
C:\Users\40461\miniconda3\python.exe experiments\e6_data_card.py
C:\Users\40461\miniconda3\python.exe experiments\e7_capacity.py
C:\Users\40461\miniconda3\python.exe experiments\e8_q1_postprocess.py
C:\Users\40461\miniconda3\python.exe experiments\e9_q2_mask_combos.py
C:\Users\40461\miniconda3\python.exe experiments\e10_mmu_fairness.py
C:\Users\40461\miniconda3\python.exe experiments\e11_postprocess_bootstrap.py
C:\Users\40461\miniconda3\python.exe experiments\e12_mmu_table_metrics.py
C:\Users\40461\miniconda3\python.exe experiments\e13_q1_mmu_grid.py
C:\Users\40461\miniconda3\python.exe experiments\e14_m2_area_calibration.py
```

依赖：本机 `pulp 3.3.0` 需要 `highspy` 才能用 HiGHS（`pip install highspy`）；仓库依赖未改。`problem5` 的导入链会拉起 `problem1` 的 `lightgbm`，`e2` 在缺包时用空模块 stub，不影响求解。

结果解释见 `reports/改进报告_v1.md`。
