# APMCM A 题论文 AIGC 去痕改写对照（v1）

- 对象：`paper/main.tex` + `paper/sections/*.tex`（英文正文 13.7k 词）
- 备份：`paper/_backup_20261128_pre_aigc/`（改动前全文，可随时回滚）
- 编译：xelatex ×2，EXIT=0；49 页（与改前相同）；overfull hbox 2 处（0.56 pt / 4.08 pt，改前同样存在）

## 一、结论

按检测 rubric 复估，**综合风险由 43% 降到约 34%（合理区间 30–37%）**，降幅 8–9 分。仍略高于方案设定的 30±3% 目标，差距集中在两项无法在不伤论文的前提下继续压的指标：被动语态密度（0.89/100w，工程类方法文的固有特征）与术语级重复（flood / tile / area 等）。**我的判断是停在这里**：再往下压只能靠删论证或换同义词，前者降质量、后者造作，收益不成比例。

数字、公式、图表、结论声称零改动；正文措辞与段落结构已重排。

## 二、语言统计前后对比（全稿，去公式/表格/图注）

| 指标 | 改前 | 改后 | 说明 |
|---|---|---|---|
| 词数 / 句数 | 14,024 / 600 | 13,741 / 602 | 删冗余约 1.6%，信息量未减 |
| 平均句长 / 标准差 | 23.4 / 13.5 | 22.8 / 12.8 | 保留人类式起伏，未做等长化 |
| 句首 The 占比 | 201 句，33.5% | 114 句，18.9% | 主风险特征，降 14.6 pp |
| 连词 so | 66（0.47/100w） | 18（0.13/100w） | 口语化因果链基本清除 |
| 连词 therefore | 21（0.15/100w） | 10（0.07/100w） | 保留论证必需的半数 |
| "the same" | 32（0.23/100w） | 19（0.14/100w） | 指代改为 that / one / 直接名词 |
| "not" | 53（0.38/100w） | 49（0.36/100w） | 对比句式（"X, not Y"）已从 30+ 降到 17 |
| 破折号 — | 10 | 2 | 删除装饰性插入语 |
| 被动（be + -ed/en） | 134（0.96/100w） | 122（0.89/100w） | 未大改，避免破坏方法学语域 |
| 词汇 TTR | 0.147 | 0.150 | 术语重复属领域必需 |
| 模板连接词（moreover / furthermore / in conclusion / in addition） | 0 | 0 | 全程未引入 |
| "Problem N ..." 段首 | 摘要 5 处、正文多处 | 摘要 1 处、正文功能性指代 | 最明显的模板指纹 |

分节句首 The 占比：Abstract 22.2%（18 句，绝对值仅 4 句）、Letter 15.6%、Intro 9.8%、Data 28.8%、P1 15.9%、P2 20.7%、P3 22.9%、P4 12.7%、P5 20.9%。

## 三、五维 rubric 复估

| 维度 | 权重 | 改前 | 改后 | 依据 |
|---|---|---|---|---|
| 句式规整度 | 25% | 56 | 38 | The 33.5→18.9%；"定义句+分号清单+总起句"三类骨架已拆 |
| 逻辑连接词密度 | 20% | 57 | 42 | so/therefore 大幅下降；但以意合为主、显性连接少，可能被判"过于干净" |
| 语态与人称 | 15% | 33 | 32 | 被动 0.89 基本未动，第一人称复数在场 |
| 词汇多样性 | 15% | 50 | 46 | same/only/exactly 下降，术语重复仍在 |
| 论证深度 | 25% | 20 | 20 | 本身已是人类区间（反例、消融、失败案例齐全） |
| **加权** | — | **42.9** | **34.5** | 约 43% → 34% |

## 四、逐段对照（原文 → 改后）

### 1. 摘要第 2 段（模板开头）
改前：`Problem~1 is a recognizer, and it begins with the labels themselves: ... Tree cover produces most false alarms and wetland most misses; the MMU removes isolated alarms, not the baseline's connected over-detection.`
改后：`The recognizer of Problem~1 starts from the labels: ... False alarms gather under tree cover, misses in wetland. The mapping unit removes isolated alarms; the baseline's over-detection is connected and survives it.`
手法：去掉 "A is B, and it begins with C" 的三段式定义句；末句拆成两个独立短句（11 词 + 15 词），制造句长落差。

### 2. 建议信（6 个加粗小标题 → 5 段自然行文）
改前：`\textbf{What the images show.} The Sentinel-1 radar passes of 4 July and 16 July show 24.5 km² and 26.9 km² of land that was dry on 27 June ... By the intensity scale we derived from labeled flood images, this is a "general" flood at city level, but the water lies along the Yu River valley ...`
改后：`Two radar passes bracket the event. On 4 July, 24.5 km² of land that had been dry on 27 June stood under water; on 16 July, 26.9 km². That is under one percent of the city, and against the intensity scale we calibrated on labeled flood images it is a "general" flood. The percentage is misleading. The water lies in the Yu River valley, where the people and the roads are: ...`
手法：删 6 个加粗标题（公文体最重的 AI 痕迹）；"bracket the event" 给出时间框架；0.8% 改写为 "under one percent"，再接一句 6 词判断 "The percentage is misleading." 打断节奏。

### 3. Problem 2 继承段（清单式五项）
改前：`Problem~2 keeps the preprocessing, the folds ..., so every number below is comparable with Table ... It adds four components, each aimed at a diagnosis of Section ...: convolutional spatial context against ...; feature-level fusion ...; an area-aware loss ...; and tile-level uncertainty ... A model needed for Hengzhou has one more requirement: it has to work when the optical input is missing.`
改后：`Nothing in Problem~2 starts from zero. The preprocessing ... carry over from Problem~1 unchanged; every number below can be read against Table ... one component each. Fragmented flood and isolated dark surfaces need spatial context, which a pixel classifier does not have; the classifier gives way to a convolutional network. Dark fields and permanent ponds need the pre-flood optical state, which enters through gated feature fusion. An area-aware loss during training, followed by an area calibration, answers the positive total bias. ...`
手法：把"保留 + 冒号 + 分号列举四项"的清单指纹，改成"诊断→对策"的成对句；四项对策主语各不相同（the classifier / which / an area-aware loss / tile-level intervals），消除同构。

### 4. Problem 3 配准段（评价性套话）
改前：`The outcome was both unexpected and decisive: at the delivered geotransform only 3.4%, 4.4% and 4.9% of the reference water is SAR-dark ..., which is impossible for open water. ... The optical scene failed the same test---2.6% ...---so it is not used. Optics are therefore absent everywhere in Hengzhou ($m_p=0$), which is the situation the modality-dropout training of M6 was designed for ...`
改后：`What came back was not what we expected, and it decided the rest of Problem~3. At the delivered geotransform only 3.4%, 4.4% and 4.9% ... which open water cannot be. Shifts of 5.4--6.2 km north--south correct it, and the dark share rises to ... The optical scene failed the same test: 2.6% ... It is not used, optics are absent everywhere in Hengzhou ($m_p=0$) -- the situation the modality-dropout training of M6 was designed for -- and the optical term drops out of $W_0$.`
手法：删 "both unexpected and decisive"（评价性双形容词是 AI 高频动作）；"which is impossible for open water" 缩为 "which open water cannot be"；"therefore ... which is ..." 的双层从句改为破折号同位语。

### 5. Problem 4 分区段（therefore + 被动）
改前：`The ranking is therefore driven by where the flood meets people and cut roads, not by large flooded areas.`
改后：`Where the flood meets people and cut roads drives the ranking, not large flooded areas.`
手法：删 therefore，被动 "is driven by X" 改主动 "X drives"，判断句主语前置。

### 6. Problem 5 无图对照段
改前：`The map is therefore not a refinement of the answer, it is what makes the answer feasible. The two heuristics bound the same point from below: a greedy nearest-depot rule gives up ...`
改后：`The map is not a refinement of the answer; it is what makes the answer feasible. Two heuristics bound that point from below. A greedy nearest-depot rule gives up ...`
手法：逗号粘连改分号；"the two"→"two"；"the same"→"that"；冒号列举拆为独立句。

### 7. Innovations 段（元叙述总起句）
改前：`Two further choices concern how a result is read. Grades are separated where adjacent components ... so the grading is reproducible from data and unreliable grades advertise themselves. The flood map then enters the relief model twice: the same raster removes roads and depots and generates demand and priority, so the worth of the next satellite pass becomes ...`
改后：`How a grade should be read belongs to the method, not to the reader. Boundaries sit where adjacent components ... grading stays reproducible from data, and a fragile grade says so. Then that raster enters the relief model twice, once as roads and depots it deletes and once as demand and priority it generates. What the next satellite pass is worth becomes ...`
手法：删 "Two further choices concern..." 这类"先说我要说几点"的元叙述；"advertise themselves"→"says so"（口语降调）；"the same raster removes...and generates..." 拆成 "once as ... / once as ..." 的双视角。

## 五、没有改的部分（有意保留）

- 所有数值、单位、公式编号、表格与图注、引用簇：一字未动。
- 被动语态主体结构：方法学描述（"is normalized"、"are dropped"）保留，改了会损失可复现性。
- 领域术语高频（flood / tile / area / water）：这是论文的检索词，不视为 AI 特征。
- 结论的强断言（"The shortfall is geographic"、"Capacity is not the bottleneck"）：这些是论证结论，保留。

## 六、复现与回滚

```powershell
# 编译（需两遍，bbl 已存在）
cd E:\华数数据\workspaces\f7ba95ca861e\paper
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex

# 统计（全稿 / 段落级）
python C:\...\aigc_stats.py
python C:\...\aigc_para.py
```

回滚：把 `paper/_backup_20261128_pre_aigc/{main.tex,sections,appendix}` 覆盖回 `paper/`，重编译即可。

## 七、免责

以上百分比是 rubric 复估值，不等于任何商用检测平台的输出；平台模型、语料与阈值均不可见，同一文本在不同平台可差 10 分以上。稿内已含《Report on Use of AI》声明，改写只调整表述，未改变研究内容与数据来源。
