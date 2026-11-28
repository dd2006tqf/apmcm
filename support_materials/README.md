# 论文支撑材料

本目录是 2026 APMCM Problem A 论文的支撑材料归档，包含五问求解主程序、补充实验脚本、全部绘图脚本，以及横州自收集数据。官方 Problem A Dataset 1/2 已移出仓库，放在 `E:\华数数据\official_data_a\`，不再随支撑材料提交。仓库根目录的 `code/`、`experiments/` 是工作副本；本目录中的同名目录是提交快照（仅把路径锚点指向仓库根目录），内容与工作副本一致。数据目录是 `user_data/a题数据汇总/公开数据` 的镜像，采用硬链接避免重复占用磁盘；模型权重、栅格中间结果和运行缓存不重复打包。

## 目录

```
support_materials/
├── code/                 五问求解主程序（problem1–problem5、main.py、config.py 等）
├── experiments/          论文补充实验与敏感性分析（e1–e18）
├── figure_generation/    全部 Figure 生成脚本与绘图公共模块
├── data/                 横州自收集数据（39 个文件，约 2.2 GB）
└── README.md
```

## 求解代码

`code/` 保留仓库 `code/` 的完整结构：

| 文件 | 作用 |
|---|---|
| `main.py` | 一键入口：Q1 → Q2 → Q3 → Q4 → Q5 → 汇总 → 约束审计 |
| `config.py` | 路径、随机种子、假设开关 A1–A16 |
| `problem1.py` | Q1 预处理、Otsu 基线、LightGBM、分层误差、SHAP、LOCO |
| `problem2.py` | Q2 双分支门控 U-Net、消融 M0–M6、面积校准、GMM 分级、蒙特卡洛等级概率 |
| `problem3.py` | Q3 Sentinel-1 配准、W0/F/W1、面积与守卫、迁移检验、敏感性网格 |
| `problem4.py` | Q4 评价单元、九项指标、熵权–CRITIC–AHP–TOPSIS、排序稳健性 |
| `problem5.py` | Q5 选址–分配 MILP、Pareto 前沿、信息价值、敏感性求解 |
| `network.py` | OSM 断道网络与 Dijkstra |
| `q5_audit.py`, `constraint_audit.py` | Q5 方案与约束的独立审计 |
| `summarize.py` | 汇总各问结果 JSON |

`experiments/` 保存论文表 5–表 12 与图 12、图 13、图 25 等所依赖的补充实验脚本，包括 MMU 公平性、面积校准、CVaR、服务下限、截距检验等。每个脚本头部注明输入、输出和口径。

运行方式（在仓库根目录）：

```bash
pip install -r support_materials/code/requirements.txt
pip install -r support_materials/figure_generation/requirements-figures.txt
python support_materials/code/main.py --only 3 4 5 --data-dir support_materials/data
```

Q1/Q2 需要官方数据，运行时把 `FLOOD_DATA_DIR` 指向 `E:\华数数据\official_data_a`，详见下面的数据说明。

补充实验从仓库根目录运行，例如：

```bash
python support_materials/experiments/e15_q5_cvar.py
```

## 数据

`data/` 只保留横州自收集数据，均为原始文件、未重采样：

| 目录 | 内容 | 规模 |
|---|---|---|
| `data/collected/公开数据` | 横州边界、DEM、WorldCover、OSM、Sentinel-1/2、WorldPop | 39 文件 / 约 2.2 GB |

`data/README.md` 给出逐图层说明，`data/MANIFEST.csv` 逐文件记录路径与字节数，`data/原始数据说明.md` 是采集阶段的完整口径。横州数据用于第 3–5 问；官方数据在外部目录 `E:\华数数据\official_data_a`，用于第 1–2 问，两者区域和传感器不同，不能混用。

支撑材料中的求解程序已适配 `data/collected/公开数据` 结构：

```bash
python support_materials/code/main.py --only 3 4 5 --data-dir support_materials/data
```

Q1/Q2 需要官方数据，运行时把 `FLOOD_DATA_DIR` 指向外部目录：

```powershell
$env:FLOOD_DATA_DIR = "E:\华数数据\official_data_a"
python support_materials/code/main.py --only 1 2
```

数据总量约 2.2 GB，不纳入 Git 跟踪；提交时直接压缩或拷贝整个 `support_materials/` 目录即可。

## 绘图代码

全部 `gen_fig_*.py`、`_figcommon.py` 与本地 `_utils/` 已从 `figures/` 移入 `figure_generation/`。`figures/` 现在只保留论文实际使用的 PDF/PNG 资产，不再混放脚本。

绘图脚本从**仓库根目录**运行，以保证相对路径 `figures/`、`output/`、`experiments/results/` 生效：

```bash
python support_materials/figure_generation/gen_fig_q5_cvar_dist.py
```

`figure_generation/README.md` 给出脚本与输出图件的逐项对照。

## 200 MB 提交版

如果上传平台限制 200 MB，使用仓库根目录的 `support_materials_200mb/`（目录 189.0 MB，压缩后 `support_materials_200mb.zip` 179.6 MB）。该版本**不含官方数据**，只收横州核心层：Boundary、30 m DEM、WorldCover、OSM、横州 WorldPop、30 m Sentinel-1/2 派生版和 Q3 派生结果；原始 10 m Sentinel-1/2、全国 WorldPop、DEM 原始瓦片、Browser 重复导出不进入包内。数据来源见 `support_materials_200mb/README.md`，被省略文件见 `support_materials_200mb/data/DATA_LINKS.md`。

## 说明

- 论文中的 TikZ 图（技术路线、SAR–光学融合结构、集合逻辑）由 `paper/` 中的 TeX 直接生成，不属于 Python 绘图脚本。
- `figure_generation/_utils/plot_utils.py` 是随脚本打包的最小绘图工具集，保证在干净环境中可直接导入；不再依赖仓库外的 `_utils`。
- 支撑材料只包含横州自收集数据（约 2.2 GB）；官方数据在 `E:\华数数据\official_data_a`。模型权重、概率栅格和 `_tmp/` 缓存不重复打包，可由代码重新生成。
