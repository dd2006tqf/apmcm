# Figure generation code

这些脚本原先位于 `figures/`，现统一归档到本目录；`figures/` 只保留论文使用的成品 PDF/PNG。所有脚本均从**仓库根目录**运行，这样脚本内的 `figures/`、`output/`、`experiments/results/` 相对路径才会命中正确位置。

```bash
cd <repo-root>
python support_materials/figure_generation/gen_fig_q5_cvar_dist.py
```

`_figcommon.py` 负责 Nature 风格、路径和常用面板函数；`_utils/plot_utils.py` 是随脚本打包的最小绘图工具，避免依赖仓库外的同名模块。

## 论文使用的图

| 脚本 | 输出 |
|---|---|
| `gen_fig_data_samples.py` | `figures/fig_data_samples.pdf` |
| `gen_fig_mmu.py` | `figures/fig_q1_mmu.pdf`, `figures/fig_q2_mmu.pdf` |
| `gen_fig_q1_area_agreement.py` | `figures/fig_q1_area_agreement.pdf` |
| `gen_fig_q1_error_drivers.py` | `figures/fig_q1_error_drivers.pdf` |
| `gen_fig_q1_label_semantics.py` | `figures/fig_q1_label_semantics.pdf` |
| `gen_fig_q1_pred.py` | `figures/fig_q1_pred_examples.pdf` |
| `gen_fig_q1_visual_cmp.py` | `figures/fig_q1_visual_cmp.pdf` |
| `gen_fig_q2_ablation.py` | `figures/fig_q2_ablation.pdf` |
| `gen_fig_q2_flip_surface.py` | `figures/fig_q2_flip_surface.pdf` |
| `gen_fig_q2_grading.py` | `figures/fig_q2_grading.pdf` |
| `gen_fig_q2_model_compare.py` | `figures/fig_q2_model_compare.pdf` |
| `gen_fig_q3_newflood_map.py` | `figures/fig_q3_newflood_map.pdf` |
| `gen_fig_q3_terrain.py` | `figures/fig_q3_terrain.pdf` |
| `gen_fig_q3_threshold_surface.py` | `figures/fig_q3_threshold_surface.pdf` |
| `gen_fig_q4_grid1km.py` | `figures/fig_q4_grid1km.pdf` |
| `gen_fig_q4_rank_robustness.py` | `figures/fig_q4_rank_robustness.pdf` |
| `gen_fig_q4_risk_map.py` | `figures/fig_q4_risk_map.pdf` |
| `gen_fig_q5_cvar_dist.py` | `figures/fig_q5_cvar.pdf` |
| `gen_fig_q5_pareto.py` | `figures/fig_q5_pareto.pdf` |
| `gen_fig_q5_route_alloc.py` | `figures/fig_q5_route_alloc.pdf` |
| `gen_fig_scenario.py` | `figures/fig_scenario.pdf` |

## 备用图与探索性脚本

以下脚本生成过候选图或中间示意图，未进入最终论文，保留以便追溯：

- `gen_fig_q1_landcover.py` — 土地覆盖分层图
- `gen_fig_q4_exposure3d.py` — 暴露度三维示意
- `gen_fig_q5_equity.py` — 服务下限可行性网格
- `gen_fig_option1.py`, `gen_fig_option2.py` — 场景图候选方案
- `gen_fig_scenario_ai.py`, `gen_fig_scene_five.py`, `gen_fig_scene_shared_v2.py` — 场景图候选版
- `gen_fig_studyarea.py` — 研究区底图候选
- `compose_fig_scene3d.py` — 三维场景合成

## 数据依赖

- `figures/problem_*.json`：各图直接读取的汇总结果
- `output/`：Q1–Q5 原始结果、模型权重与栅格
- `experiments/results/`：补充实验 JSON（e10、e13、e15、e18 等）
- `_tmp/v2/grid1km.npz`：Q4 公里网格中间量
- `user_data/`：`gen_fig_data_samples.py` 与场景图脚本需要的公开数据/官方影像

缺少上述数据时脚本会直接报 `FileNotFoundError`，不会用占位数据静默重绘。
