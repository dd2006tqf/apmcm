# 研究项目: 亚太 (APMCM) A题 - 2026 APMCM Problem A

## 项目说明
这是一个自动化研究管理的工作流引擎，当前正在执行「亚太数学建模 (APMCM)」流水线。

## 流水线步骤
  1. 赛题分析 (comp-prob-analysis)
  2. 建模求解 (comp-modeling)
  3. 编程实现 (comp-code)
  4. 图表生成 (paper-figure)
  5. 流程与架构图绘制 (paper-figure-html)
  6. 英文论文撰写 (comp-paper-en)
  7. 编译与合规检查 (comp-compile-en)

## 重要规则
- 所有产出文件都写入当前工作目录（不要写到其他位置）
- 用 Write 工具创建文件、Edit 工具修改；文件太长就先 Write 第一段（末尾留一行占位注释），再用 Edit 把占位换成下一段，每段不超过 150 行
- 不要用 Bash heredoc（cat << 'EOF'）写文件：本机 Bash 会把成对的反斜杠吃成一个，加了引号也一样
- 可能跑很久的脚本（模型拟合、Bootstrap、大循环）用 timeout 包着跑，如 `timeout 900 python code/xxx.py`：命令跑久了会被转到后台，卡住的后台进程不结束，这一步就一直等它。要停后台进程，先按命令行查出真实进程号再结束：`powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object CommandLine -like '*xxx.py*' | Select-Object ProcessId,CommandLine"`，不要凭记忆写进程号
- 每个步骤完成后至少产出一个文件
- 读取工作区中已有的文件，在前步骤的基础上继续工作
- 使用中文撰写论文（除非是 LaTeX 代码）

## 工作区文件
- [索引摘要] 工作区共 5143 个文件；展示关键文件和目录统计，不逐条展开同构原始数据。
- user_data/2026 APMCM Problem A.pdf
- user_data/2026 APMCM Problem A_extracted.txt
- user_data/2026 APMCM Problem A_extracted_check.md
- user_data/PROBLEM_STATEMENT.md
- user_data/a题数据汇总.zip
- [数据目录] user_data/a题数据汇总/：3040 个文件；类型 .tif×3013、.geojson×12、.tiff×6、.md×2、.png×2、其他×5；示例 user_data/a题数据汇总/Problem A Dataset 1/Dataset/label information/label_1001.tif、user_data/a题数据汇总/Problem A Dataset 1/Dataset/post-disaster SAR images/after_gf3_hh_1521.TIF、user_data/a题数据汇总/公开数据/数据清单.md
- _problem_file.txt
- CODE_CHECK.json
- CUSTOM_REQUIREMENTS.md
- DATA_FACTS.json
- DATA_PROFILE.json
- DELIVERABLES.json
- MODELING_REPORT.md
- PROBLEM_ANALYSIS.md
- RESULTS.md
- UPSTREAM_ISSUES.md
- figures/_figcommon.py
- figures/all_results.json
- figures/fig_arch_fusion.html
- figures/fig_arch_fusion.pdf
- figures/fig_q1_area_agreement.pdf
- figures/fig_q1_error_drivers.pdf
- figures/fig_q1_label_semantics.pdf
- figures/fig_q1_visual_cmp.pdf
- figures/fig_q2_ablation.pdf
- figures/fig_q2_grading.pdf
- figures/fig_q2_model_compare.pdf
- figures/fig_q3_area_budget.pdf
- figures/fig_q3_newflood_map.pdf
- figures/fig_q3_threshold_surface.pdf
- figures/fig_q4_rank_robustness.pdf
- figures/fig_q4_risk_map.pdf
- figures/fig_q5_pareto.pdf
- figures/fig_q5_route_alloc.pdf
- figures/fig_roadmap.html
- figures/fig_roadmap.pdf
- figures/FIGURES_REPORT.md
- figures/gen_fig_q1_area_agreement.py
- figures/gen_fig_q1_error_drivers.py
- figures/gen_fig_q1_label_semantics.py
- figures/gen_fig_q1_visual_cmp.py
- figures/gen_fig_q2_ablation.py
- figures/gen_fig_q2_grading.py
- figures/gen_fig_q2_model_compare.py
- figures/gen_fig_q3_area_budget.py
- figures/gen_fig_q3_newflood_map.py
- figures/gen_fig_q3_threshold_surface.py
- figures/gen_fig_q4_rank_robustness.py
- figures/gen_fig_q4_risk_map.py
- figures/gen_fig_q5_pareto.py
- figures/gen_fig_q5_route_alloc.py
- figures/latex_includes.tex
- figures/problem_1_results.json
- figures/problem_2_results.json
- figures/problem_3_results.json
- figures/problem_4_results.json
- figures/problem_5_results.json
- figures/sensitivity_results.json
- figures/tikz_q1_set_logic.pdf
- figures/tikz_q1_set_logic.tex
- output/coreg_official.json
- output/folds.json
- output/q1_results.json
- output/q1_worldcover_tiles.npz
- output/q2_results.json
- output/q3_newflood_0704.tif
- output/q3_newflood_0716.tif
- output/q3_newflood_0716_h30.tif
- output/q3_newflood_0716_sar_only.tif
- output/q3_prob_0704.tif
- output/q3_prob_0716.tif
- output/q3_receded_0704.tif
- output/q3_receded_0716.tif
- output/q3_results.json
- output/q3_w0.tif
- output/q3_w1_0704.tif
- output/q3_w1_0716.tif
- output/q4_results.json
- output/q4_units.geojson
- output/q4_units_0704.csv
- output/q5_results.json
- output/q5_roads_cut_0716.geojson
- output/q1_pred/A_1001.png
- output/q1_pred/A_1002.png
- output/q1_pred/A_1003.png
- output/q1_pred/A_1004.png
- output/q1_pred/A_1005.png
- output/q1_pred/A_1006.png
- output/q1_pred/A_1007.png
- output/q1_pred/A_1008.png
- output/q1_pred/A_1009.png
- output/q1_pred/A_1010.png
- output/q1_pred/A_1011.png
- output/q1_pred/A_1012.png
- output/q1_pred/A_1013.png
- output/q1_pred/A_1014.png
- output/q1_pred/A_1015.png
- output/q1_pred/A_1016.png
- output/q1_pred/A_1017.png
- output/q1_pred/A_1018.png
- output/q1_pred/A_1019.png
- output/q1_pred/A_1020.png
- output/q1_pred/A_1021.png
- output/q1_pred/A_1022.png
- output/q1_pred/A_1023.png
- output/q1_pred/A_1024.png
- [目录摘要] output/：1981 个文件；类型 .png×1976、.pt×5；示例 output/q1_pred/A_1025.png、output/q1_pred/B_1015.png、output/q2_models/M4_fold4.pt
- [目录摘要] code/：17 个文件；类型 .py×15、.md×1、.txt×1；示例 code/common.py、code/problem2.py、code/unet.py
- [全量处理约束] 本索引仅压缩重复路径；统计、训练、验证和产物差异检测仍必须由本地脚本遍历磁盘上的完整数据集。请用 Bash/Read 按需查看目录或样本，不要把索引摘要误当成抽样授权。
`user_data/` 下是用户上传的原始材料，请完整阅读并作为研究基础；其余是前面步骤的产物，按需读取，未经本步核对不要当成题目事实。目录摘要只压缩重复路径，不允许把全量数据分析降级为抽样分析。

## 📋 用户自定义需求/大纲（最高优先级）
用户上传了自定义要求或大纲文档 `CUSTOM_REQUIREMENTS.md`。
**请仔细按此文档的要求和思路执行工作流，此文档具有最高优先级指导。**
**当与文档内容发生冲突，以此文档为准。**

### 文档内容
```
对于这道题，我的要求是，我要冲国家级一等奖，然后你需要想出创新点，避免雷同要求，并存，可以用高级点的模型图片的可视化，你也要好好优化一下，用高级一点的风格，太基础的可视化所有使用AI工具生成的内容，应在正文相应位置明确标注，并在参考文献中列出所用AI工具，格式为：[编号]工具名称，版本/型号，开发机构/公司，使用日期。（示例：[1]DeepSeek,DeepSeek-R1-0528,深度求索（DeepSeek）,2026-6-12）

7、使用AI工具的参赛作品必须在提交作品时提供“AI工具使用详情说明”（PDF格式，以“AI工具使用详情”为文件名），内容包括：

（1）所用AI工具名称和版本

（2）具体使用目的和应用环节（如模型构建、数据处理、可视化、文献检索等）

（3）关键交互记录（重要提示词与回复）

（4）采纳内容和人工修改情况
同时，我上传给你的数据有一部分是官方给的有一部分是我自己去搜集的，你需要分开来看，在数据压缩包里面有readme；你也可以自己根据你的需要去搜集数据，搜集数据这部分我不给你设限。我上传给你的数据有些是嵌套的压缩包，你需要先解压。我上传的数据，你需要自己校验一遍，不一定要盲从，你也可以自己去搜集你需要用到的数据。
```

## 参数
- 研究主题: 亚太 (APMCM) A题 - 2026 APMCM Problem A
- template: comp_apmcm
- ai_disclosure: used
- problem_statement: 参考一下这些注意事项，你需要自己确认，不盲从；
## 1. 全局：五问是一条数据链

四家说法一致：五问不是五个独立题，而是一条接力链——**Q1 定口径与格网 → Q2 改模型、定分级 → Q3 迁移横州 → Q4 评估 → Q5 调度**。Q1 的口径一旦松动，后面全部返工。

统一口径只有一句：

$$F = W_1 \setminus W_0$$

即 **新增淹没 = 灾后水体 ∩ 灾前非水体**。老哥把"直接把灾后识别出的所有水体当新增"列为最大的坑；加油站用三张示意图表达同一件事：先记住灾前水体基数，再从灾后水体里减掉，剩下的才是题面要的那块面积。

五问之间的传递物：

| 传递    | 内容                                                       |
| ------- | ---------------------------------------------------------- |
| Q1 → Q2 | 基线模型、FP/FN 空间热点、分地类误差表（决定第二问往哪改） |
| Q2 → Q3 | 融合模型、面积校正系数、四级阈值                           |
| Q1 → Q3 | 预处理流程与面积口径（同一套定义，不重做）                 |
| Q3 → Q4 | 新增淹没栅格、占比、强度等级（评价单元的灾害强度分量）     |
| Q3 → Q5 | 淹没范围（判断哪条路被淹、哪个候选站不能用）               |
| Q4 → Q5 | 单元优先级与受灾人口（需求量 $d_j$ 的来源）                |
## 2. 数据事实（实测 + 博主判读）

### 2.1 附件 1 结构（本文实测）

| 目录                        | 命名                                 | 数量 | 规格                   |
| --------------------------- | ------------------------------------ | ---- | ---------------------- |
| pre-disaster optical images | `before_s2_1001.TIF` … `2000.TIF`    | 1000 | 256×256，4 波段，uint8 |
| post-disaster SAR images    | `after_gf3_hh_1001.TIF` … `2000.TIF` | 1000 | 256×256，1 波段，uint8 |
| label information           | `label_1001.tif` … `2000.tif`        | 1000 | 256×256，二值 {0,1}    |

- 编号 1001–2000 一一对应，同一编号三张为同一地面块，可配对叠置。
- 全部为带地理标签的 GeoTIFF（含 PixelScale、TiePoint），可对齐。
- 小助手据 GeoTag 推算：WGS-84，像元约 $9.2 \times 8.1$ m，单像元约 75 m²；样本中心约 112.7–116.9°E、28.4–29.3°N，推测对应洞庭湖与鄱阳湖区域、2020 年长江中游汛期洪水。**此为博主推测，题面未给该信息，论文引用须注明为推断。**

### 2.2 标签分布（本文实测）

| 项                 | 数值                                   |
| ------------------ | -------------------------------------- |
| 类别 0（非淹没）   | 81.0%                                  |
| 类别 1（淹没）     | 19.0%                                  |
| 单块淹没比例       | 中位数 3.8%，均值 19.0%，75 分位 20.8% |
| 全空块（比例为 0） | 15 块                                  |
| 全淹没块           | 0 块                                   |

结论：**严重类别不平衡，且大量切片几乎无目标**。训练需做选块（剔除空块/分层抽样）、加权或 Dice/Focal 损失；面积误差评估按块累计，不要只看像素级 IoU。

**注意**：label 只有二值"淹没/非淹没"，不区分"原有水体"与"新增淹没"。Q1 若直接用 label 监督，等于监督"灾后水体"而非"新增"；Q3 必须用灾前水体掩膜做差集。
## 9. 必须避开的坑

1. 把灾后全部水体当新增淹没（口径错，全线返工）
2. 随机抽像元划分训练/验证（数据泄漏，指标虚高）
3. 只报 OA / Accuracy，不报面积误差与 IoU
4. 分级阈值拍脑袋、没有数据依据
5. 把净增面积当新增面积
6. Q5 的需求与 Q4 评估结果脱节
7. 跳过 SAR 去噪与异源配准
- competition: apmcm
- language: en
- output_format: pdf
- flowchart_engine: html
- problem_id: A
- tools: python
- custom_requirements: （较长，全文见工作区 CUSTOM_REQUIREMENTS.md，也在 CLAUDE.md 的「用户自定义需求/大纲」段）
- min_figures: auto
- min_tables: auto
- min_models: auto
- figure_style: nature
- diagram_style: random

## Bash 环境变量（页数/字数等检查脚本会用到）
工作区已自动生成 `.env_skill`，含以下数值参数：

```bash
export SUBMISSION_PAGE_REFERENCE=35
```

**说明**：`SUBMISSION_PAGE_REFERENCE` 只供最终报告提示；正文不设页数目标，不据此扩写或压缩。其他数值按用户内容要求使用。脚本读取参数时，在 bash 块开头加一行 `source .env_skill 2>/dev/null || true` 就能让数值设置正确取到值。
**最终以论文实际质量为准**，不要为了凑页数刻意注水。

## ⛔ 流程图/架构图引擎 = HTML（用户选定，全程遵守）
- 本工作流的流程图 / 技术路线图 / 系统架构图**一律用 HTML+CSS 绘制**（`paper-figure-html` skill：flex/grid 自动布局 → Electron printToPDF 转矢量 PDF），**不使用 DrawIO**。
- 规划阶段（paper-plan / paper-plan-zh 等）写 FIGURE_MANIFEST 时，流程/架构图那一类的**章节标题请写「HTML 流程/架构图」**（不要写「DrawIO 流程/架构图」），与实际引擎保持一致，避免误导后续写作。图名前缀（fig_arch / fig_flow / fig_roadmap 等）不变。
- 写作阶段按 `figures/latex_includes.tex` 里已生成的 `fig_*.pdf` 直接引用即可，**无需也不要提及 DrawIO / .drawio 文件**（本工作流不产生任何 .drawio 文件）。

## HTML 流程图/架构图风格 = D 随机·跟随本篇数据图配色（仅限非数据图）
- 识别标记：`MH_DIAGRAM_STYLE=R`。`paper-figure-html` Step 1 按种子在 A/B 中选字体与造型，强调色 H0 取本篇数据图主色的色相（`python _utils/plot_utils.py --paper-accent`）。
- 同一篇的流程图、架构图、AI 总览图都用这同一个强调色，和本篇数据图同色系；不同论文因数据图配色按种子随机，颜色自然不同。
- 数据图仍只读下方 MH_DATA_FIG_* 配置，不受本节影响。
- LAYOUT/ARROW 等只允许在语义等价时改变结构或连线；同篇统一视觉 token，异篇差异来自题目真实依赖和种子。
- AI 画的总览图、流程/原理图同样按本篇随机配色（识别标记：`MH_DIAGRAM_STYLE_BY=default`）。

## 竞赛规则（亚太地区数学建模竞赛 (APMCM)）
- 语言: English
- 页数策略: 本地生成不设硬上限；最终提交参考为 35 页。超过只提醒，不失败、不删核心推导、不缩字号、不压小图；低于参考值也不按比例强制扩写。
- LaTeX 模板: apmcmthesis
- 机理题写作：原始定律/几何关系 → 关键代入与变形 → 完整方程及初边值 → 与代码一致的离散、事件切换和停止准则 → 真实结果与误差依据。先写数学骨架再嵌图；不以图表数量代替推导，通用算法科普可省但本题求解过程不能省。
- 按照官网上传的 PDF 模板格式提交
- 全英文撰写
- Summary 置首；正文按问题依赖链自适应组织 formulation—solution—results—validation，不固定章名与章数，不生成目录页
- 使用 apmcmthesis 或通用模板，pdflatex 编译
- 题号: A
- 编程工具: python

## 计算质量与有界验证
- 题设硬约束、真实数据、计算正确性必须同时满足，不通过改数据/放宽题设让结果显得合理。
- 编程实现步骤每问执行 validate_constraints()，从实际保存的结果核对必要条件、有限数值和任务能力。
- 编程实现步骤发现真实违例须定位并修受影响实现；模型变更要有依据，不能凭空加接触/公平性约束。
- 步骤授权边界：每一步只改本步产物。数据图/流程图/论文撰写/编译导出步骤不得修改 code/、figures/*_results.json、figures/all_results.json、output/、RESULTS.md、MODELING_REPORT.md、PROBLEM_ANALYSIS.md、PROBLEM_FACTS.json，也不重跑求解；发现上游结果有问题，写进 UPSTREAM_ISSUES.md（问题、证据、影响哪一问）后按现有结果完成本步。后台会在这些步骤结束时核对上游产物，被改动的会还原。
- 高分、负测试 R²、相同最优值、闲置资源、边界解仅是待解释现象，不单凭现象重算。
- 跨问单调比较须同目标同口径且可行域包含；最优值允许相等，启发式结果不冒充最优性证明。
- 用完整物理几何与正确单位；等价简化需要依据。全覆盖题不得以中心覆盖替代，局部覆盖题也不升级任务。
- 方法、步长、误差和验证次数按题设/建模方案确定，不统一强制 1e-6、双算法、五个种子或十维降维。
- 有可核对求解证书、解析对照或真实独立验证时可复用；必须做的验证不能以文件存在或关键词代替。
- 编程保存完整结果数据，图表排版在绘图步骤，章节在撰写步骤，最终排版/编译在导出步骤。
- 同版本输入、源码、参数和验证器的检查点可复用；变化只重验受影响算例，不在每层各跑一次全量求解。
- 本地检查失败先区分真实错误、缺证据与环境故障；缺说明补说明，环境故障修环境，不自动调用付费生成。
- 使用工作流统一的返修与时间预算；耗尽时保存进展、如实报告未完成，不假报成功也不无限重试。

## 等计算时的单次等待上限
- 等后台任务单次最多 4 分钟（`_utils/wait_output.py` 一段最多 230 秒；自己等时那次 Bash 的 timeout ≤ 240000 毫秒），没结束就再等同一个任务；不要一次等 10 分钟，也不要在一次前台调用里阻塞超过 4 分钟。超过 5 分钟没有新消息，提示缓存失效，下一轮要按全价重读整段上下文。
- 两次等待之间只看进度（日志末几行、已完成算例数），不重读大文件，不另起第二份求解。
- 赛题分析、建模步骤里的验证性试算单次不超过 3 分钟（命令前加 timeout 180），更长的求解留给编程实现步骤。

## 文档原文（已自动提取，按需读取）
用户上传的文档已经在 `user_data/` 目录下提取为文本/markdown 文件，**用 Read 工具按需读取，不要尝试一次性读所有文件**。
- 每个提取文件的第一行括号写明来源：`Vision OCR` / `部分识别` 是图像逐字转录（PDF 或赛题图片），`PDF 文本层` 是还没做图像识别的临时文本（上下标、公式会丢），`pandoc 转换` 基本保真，`python-docx 兜底` / `旧版 .doc 兜底解析` 会丢公式或符号
- 同名的 `*_extracted_check.md` 是图像识别与 PDF 原文的数字核对清单，有就要逐条核对

文件清单（按字符数排序）：

- **2026 APMCM Problem A** → `user_data/2026 APMCM Problem A_extracted.txt` (9,695 字符)
  - 开头: # 2026 APMCM Problem A.pdf 内容提取（部分识别 / 文本兜底） 原文 5 页；视觉成功 0 页；看图转录第 1–5 页。 没有识别成功的页，无法与 PDF 文本层交叉核对；详见 `2026 APMCM Proble…

**用法示例**: 如果你需要某篇论文的内容, 用 `Read` 工具读对应的 `user_data/xxx_extracted.md` 文件。可以用 `Grep` 工具按关键词搜索定位特定信息。

## ⛔ Output Language Override (HIGHEST PRIORITY)
All produced files (papers, sections, references, captions, headings, figure/table captions) must be in **English**. Do NOT write in Chinese. This overrides the default 'Use Chinese' rule above.

<!-- MH_AI_DISCLOSURE=used -->
## 📝 AI 工具使用声明（用户已开启，由脚本按用户确认的记录插入：Report on Use of AI（放在全文最后））
真实记录保存在 `.mh/ai_disclosure.json`，禁止自行增删、改写或猜测。声明段由确定性脚本`build_ai_disclosure.py` 按记录插进论文：写论文时不要自己写这一段，也不要删改脚本插进去的那段；论文其他地方不提 AI 工具的使用，不把 AI 工具列进参考文献。

## 数据图配色与版式（仅限数据图，与流程图独立）
<!-- MH_DATA_FIG_ENGINE=nature -->
数据图只读取 MH_DATA_FIG_* 配置，不读取 diagram_style / MH_DIAGRAM_STYLE。默认配色按工作区固定种子选择；Nature 引擎保留 Nature 版式。黑白打印可辨认是可读性要求，不等于必须导出灰度。除非用户明确选择数据图灰阶，不得用灰阶列表覆盖配色或将正式 PNG/PDF 强制转灰。黑色文字、参考线和合理的中性区域仍可使用；不要求每张图必须出现彩色。质检核对所属图种的配置，不得自行增加全图灰阶断言；更改颜色不得更改原始数值。

<!-- MH_DATA_FIG_PALETTE=random -->
数据图配色：未指定固定色卡，使用对应引擎的工作区稳定默认配色；不继承流程图的黑白选择。

<!-- MH_MEMO:BEGIN -->
## ⛔ 本工作区的记忆（引擎自动维护，每步刷新）

### 已确认的决定（贯穿全程，不得推翻）
- **官方标签到底标的是"灾后所有水"还是"新增淹没"**：B：标签本身已是新增淹没（原有河湖标为0），直接学新增
- **分级用的"研究区总面积"指什么范围**：B：任意评价单元，按比例分级，面积阈值按单元大小换算
- **横州"灾中"影像选哪一期**：C：两期都用，做时间序列对比
- **第五问选哪个救援任务**：B：做物资分配，同时决定开哪几个物资点
- **单独提交的「AI工具使用详情」PDF**：不做单独的「AI工具使用详情」PDF，只在论文末尾附 AI 使用声明（Report on Use of AI）（⛔ 用户已否决：另外生成一份「AI工具使用详情」PDF，不要加回来）

完整历史 → `.mh/JOURNAL.md`；结构化状态 → `.mh/STATE.json`（要溯源某个数字来自哪次运行时读它）
<!-- MH_MEMO:END -->

## 材料核对（后台自动检查，只提示不拦截）
- ⚠ 赛题开头写的是「MCM/ICM」，你选的是「APMCM 亚太赛」。照现在跑，页数、格式、摘要要求都会按选的这个比赛来写，请确认没有选错。
- 对照原题逐条确认；确实缺附件时在 PROBLEM_ANALYSIS.md 开头写明“缺失材料”，对应变量注明“附件未上传”，不要用构造数据冒充附件。
- 这些是软件报告给用户的提醒，**不要写进论文正文**（论文里别出现“附件没上传”“data files were not uploaded”这类话）；数据放在电脑数据位置里的、题目让自己找公开数据的、统计建模这类本来没有附件的都不算缺。

<!-- INJECT:PAPER_PLAN_BACKLINK -->


## ⛔ 撰写阶段强制规则（基于用户开关 + figures/ 实际产物）
**figures/ 中已有 16 张图，必须全部嵌入正文：**
- fig_arch_fusion.pdf
- fig_q1_area_agreement.pdf
- fig_q1_error_drivers.pdf
- fig_q1_label_semantics.pdf
- fig_q1_visual_cmp.pdf
- fig_q2_ablation.pdf
- fig_q2_grading.pdf
- fig_q2_model_compare.pdf
- fig_q3_area_budget.pdf
- fig_q3_newflood_map.pdf
- fig_q3_threshold_surface.pdf
- fig_q4_rank_robustness.pdf
- fig_q4_risk_map.pdf
- fig_q5_pareto.pdf
- fig_q5_route_alloc.pdf
- fig_roadmap.pdf

**铁律：**
1. 建立 figure use plan，把每张图分配到建模/求解/结果/验证过程并全部嵌入
2. **必须用 `.png` 扩展名**（Word 模式只支持 PNG）；不要写 `.pdf` 路径
3. 引用语「如图 X-Y 所示」必须有对应的图片嵌入
4. 每张正文图都要有图前问题/判断和图后完整解释；禁止连续贴图；任何生成图未嵌入都阻断

<!-- /INJECT:PAPER_PLAN_BACKLINK -->
