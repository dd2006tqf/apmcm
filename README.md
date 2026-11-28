# 2026 APMCM Problem A —— 洪水识别 · 强度分级 · 救援优化（完整复现仓库）

本仓库包含论文全部源码、五问求解代码、结果文件、图件与已编译 PDF。除横州自收集原始数据（约 2.2 GB，见第 2 节）外，**克隆后即可重算全部数字、重绘全部图、重编论文**；官方附件已移出仓库，放在 `E:\华数数据\official_data_a\`。

## 1. 目录结构

```
paper/                论文 LaTeX 源码与成品
  main.tex            主文件（标题页 / 摘要 / 10 节 / 建议信 / 参考文献 / 附录）
  sections/           1_introduction … 10_evaluation, Z_ai_disclosure
  appendix/A_code.tex 代码结构与关键片段
  references.bib      17 条参考文献
  main.pdf            论文成品（49 页）
  AI工具使用详情.pdf   按赛事第 7 条要求单独提交的 AI 使用说明（2 页）
  ai_usage_detail.tex 该说明的源码
  apmcmthesis.cls/.cfg 官方模板
code/                 Python 求解代码（详见 code/README.md）
  01–05 对应 problem1.py … problem5.py；main.py 为入口；config.py 存路径/种子/假设开关 A1–A16
output/               全部结果（复现的核心证据，勿删）
  q1_results.json … q5_results.json   五问指标与中间量
  q2_models/M4_fold*.pt               五折模型权重（44 MB）
  q3_prob_*.tif, q3_w0.tif            概率图与前置水体栅格
  q4_units_*.geojson, q5_roads_cut_*.geojson  单元与断道网络
figures/              论文成品图件（PDF/PNG）；绘图脚本已归档到 support_materials/figure_generation/
support_materials/    支撑材料：数据、求解代码、补充实验与绘图代码
  data/               横州自收集数据（约 2.2 GB，不入 Git）
  code/               五问求解主程序快照
  experiments/        补充实验脚本与结果快照
  figure_generation/  全部 Figure 生成脚本与公共绘图模块
RESULTS.md            五问结果速查（论文数字的原始出处）
CROSS_PROBLEM_LEDGER.json  跨问一致性台账（面积口径、分母、等级阈值）
```

## 2. 数据准备（唯一未入库的部分）

横州自收集数据约 2.20 GB，超过 GitHub 单库承载能力，故不入库；官方附件已移出仓库，放在 `E:\华数数据\official_data_a\`。按下列路径放置：

```
E:\华数数据\official_data_a\
  Problem A Dataset 1/Dataset/{label information, post-disaster SAR images}
  Problem A Dataset 2/Dataset/pre-disaster optical images

user_data/a题数据汇总/
  公开数据/{Boundary, DEM, LandCover, OSM, Sentinel1, Sentinel2, WorldPop}
```

官方附件仅用于第 1–2 问，运行时把 `FLOOD_DATA_DIR` 指向 `E:\华数数据\official_data_a`；第 3–5 问只用公开数据层（Sentinel-1 RTC、ESA WorldCover v200、Copernicus DEM GLO-30、OSM/Geofabrik、WorldPop、GADM 4.1），清单见论文表 1。WorldCover 分块可用 `python code/fetch_worldcover.py` 通过 HTTP COG 自动拉取（约 15 min）。

## 3. 复现命令

```bash
pip install -r code/requirements.txt          # Python 3.11.9；CPU 可跑全流程，GPU 仅加速 Q2 训练
python code/main.py --data-dir user_data/a题数据汇总      # 全流程，约 2 h（GPU）/ 3 h（CPU）
python code/main.py --only 3 4 5                          # 复用 Q1/Q2 已有结果
```

单步耗时：Q1 20 min、Q2 60 min（GPU）、Q3 30 min、Q4 10 min、Q5 10 min、约束审计 2 min。随机种子 2026（+折号），`config.py` 中 A1–A16 为全部假设开关。

## 4. 论文编译

```bash
cd paper
xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
```

环境：TeX Live / MiKTeX（XeLaTeX + ctex + 模板自带 `apmcmthesis.cls`）。结果为 45 页、0 错误、0 未定义引用。单独说明：`xelatex ai_usage_detail.tex` 两遍，产物即 `AI工具使用详情.pdf`。

## 5. 论文数字与结果文件的对应

| 论文位置 | 结果文件 | 关键量 |
|---|---|---|
| 表 2（4.4 节） | `output/q1_results.json` | 阈值基线 IoU 0.2776 / RAE 1.4059；LightGBM IoU 0.5983 / RAE 0.0918 |
| 表 4（5.3 节） | `output/q2_results.json` | M2 IoU 0.7979；M4 0.7369；M6 0.7316；硬集 0.266→0.5618 |
| 表 5、表 6 | `output/q2_results.json` | τ_A 五折、θ=(0.0593,0.5298,0.9689)、BIC K2/K4 |
| 表 7（6.6 节） | `output/q3_results.json` | 07-16 A_new 26.9025 km²、A_net −52.5341 km²、ρ 0.78% |
| 表 8（7.4 节） | `output/q4_results.json` | 393 单元 / 259 受灾、马岭镇 S=0.6432、CR 0.0079 |
| 表 9、表 10 | `output/q5_results.json` | f1 0.5699、6 开仓点、VoI 0.0387 / 5.5405、57/164 单元不可达 |

图件由 `support_materials/figure_generation/gen_fig_*.py` 直接读取 `output/*.json` 重绘，编号与论文章节一一对应（图 1 场景示意，图 2 技术路线 … 图 25 CVaR 分布）。绘图脚本统一从仓库根目录运行。

## 6. 已知偏差（论文正文已说明）

- 第 3 问 07-16 的净增面积为负（−52.53 km²），是"退水主导"场景的真实结果；面积分母另给出洪泛平原口径 ρ=1.29%。
- 第 5 问 f1=0.5699 中，57/164 单元因路网断道不可达，属网络约束而非求解失败；约束审计 19 个方案 0 违规。

## 7. 未入库内容

`user_data/`（横州自收集数据约 2.2 GB）、`_tmp/`（运行缓存 8.6 GB）、`.mh/`（Agent 会话记录 214 MB）。前两者可由第 2–3 节重新获得，第三者与复现无关。官方数据不在仓库内，位于 `E:\华数数据\official_data_a`。

## 8. 支撑材料

`support_materials/` 是论文提交用材料归档，包含 `code/` 求解主程序、`experiments/` 补充实验与结果、`figure_generation/` 全部绘图脚本，以及 `data/` 中的横州自收集数据（39 个文件，约 2.2 GB）。官方附件已移出仓库。`figures/` 中的 Python 脚本已全部移入该目录，成品目录只保留 PDF/PNG，避免脚本与资产混放。数据目录不纳入 Git，提交时请整体压缩/拷贝；200 MB 提交版见 `support_materials_200mb/`。详细索引见 `support_materials/README.md`。
