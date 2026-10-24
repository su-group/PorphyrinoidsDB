# Manuscript ML 部分修订思路（2026-10-18 审阅稿，未改动 docx）

对象：`C:\Users\admin\Downloads\PorScaDB-Manuscript-20261011.docx`（5.6 MB，150 段，1 张表，9 张图位）
本文档只提出方案与依据，**未对 docx 做任何修改**。所有"新数字"均取自本仓库四组已完成的 run：

| 简称 | 目录 | 划分 | 特征 |
|---|---|---|---|
| R-M | `experiments\20261016_R-M` | 随机 | Mordred |
| R-MF | `experiments\20261016_R-MF` | 随机 | 混合 |
| S-M | `experiments\20261016_S-M` | 骨架 | Mordred |
| S-MF | `experiments\20261016_S-MF` | 骨架 | 混合（GP 停机 39/100） |

---

## 0. 最关键的判断：正文 ML 数字不是当前代码能复现的

三条互相独立的证据：

1. **表 1 的数值在本机所有 run 里都找不到。** 全库 7 份 `model_performance_summary.csv`（`20261013_R-M`、`20261014_R-M`、`20261014_S-M`、`20261015_mixed20`、`20261016_R-M/R-MF/S-M`）逐一搜 0.9359 / 0.9095 / 0.9180 均无命中。最接近的是 Mordred-only LightGBM：正文 0.8913，现行 run 0.8917。
2. **正文自述的三个实现细节与现行管线全部不一致。**

| 细节 | 正文现在写的 | 现行管线实测 |
|---|---|---|
| Mordred 特征数 | 411 | 原始 1,613；折内保留 510–526（池化 523） |
| 混合特征数 | 3,825 | 原始 9,972；折内保留 3,815–4,055（池化 3,986–3,988） |
| CV | 3-fold | **5-fold**（1383 训练 / 351 测试） |

3. 正文把 SHAP 第二位描述为 `NsssCH`，而重跑后 `NsssCH` 已跌出前 10（见 §3）。

**结论：整段 ML 数字必须按四组 run 重写，并与 Methods 的实现细节一起替换；不能只改个别数字。**

---

## 1. 建议的新 Table 1（Test R² 排序，两特征集混合排名）

| # | Feature set | Model | Test R² | Test RMSE (eV) | Test MAE (eV) | CV MSE (eV²) |
|---|---|---|---|---|---|---|
| 1 | Concatenated | XGBoost | 0.9632 | 0.1923 | 0.1178 | 0.1038 |
| 2 | Concatenated | SVR | 0.9565 | 0.2090 | 0.1313 | **0.0913** |
| 3 | Concatenated | KernelRidge | 0.9443 | 0.2364 | 0.1583 | 0.1055 |
| 4 | Concatenated | Lasso | 0.9404 | 0.2447 | 0.1639 | 0.1187 |
| 5 | Concatenated | ElasticNet | 0.9399 | 0.2457 | 0.1668 | 0.1072 |
| 6 | Concatenated | GaussianProcess | 0.9362 | 0.2532 | 0.1630 | 0.1150 |
| 7 | Mordred-Only | XGBoost | 0.9321 | 0.2611 | 0.1662 | 0.1342 |
| 8 | Concatenated | LightGBM | 0.9207 | 0.2822 | 0.1469 | 0.1048 |
| 9 | Mordred-Only | LightGBM | 0.8917 | 0.3297 | 0.1991 | 0.1437 |
| 10 | Concatenated | RandomForest | 0.8886 | 0.3344 | 0.1394 | 0.1040 |

与旧表相比的**叙事变化**（这几条必须改，否则结论站不住）：

- "concatenated 占满前列"仍成立（前 10 中 8 个），但 **SVR 从缺席变为第 2**（0.9565），KernelRidge / Lasso / ElasticNet 也全部挤进前 6。
- "Gaussian Process 第 2" 不再成立：现为第 6（0.9362）。
- **Mordred-only XGBoost 从第 4 名降到第 7 名**，与最优的差距从 ΔR²=0.0264 扩大到 **ΔR²=0.0311**（ΔRMSE = 0.1923 → 0.2611，即 0.0688 eV）。旧文"仅低 0.0264，可忽略"的说法需要改成"中等但稳定的代价"。
- 若按 **CV MSE** 排序，第一是 SVR（0.0913）而不是 XGBoost。**排序主指标必须先定**（见 §6-③）。

---

## 2. 建议新增一节：骨架划分下的外推能力（本轮最有价值的新结果）

正文目前只有随机 80/20 一句话交代，而四组实验已经能把"指纹到底有没有用"讲成一条闭环论证：

| 指标 | 随机划分 | 骨架划分 |
|---|---|---|
| 最优 Test R²（Mordred） | +0.9321（XGBoost） | +0.3741（RandomForest） |
| 最优 Test R²（混合） | +0.9632（XGBoost） | +0.2843（SVR） |
| 同模型跨划分：XGBoost（Mordred） | 0.9321 | 0.0010 |
| 同模型跨划分：SVR（混合） | 0.9565 | 0.2843 |
| 训练/测试骨架重叠率 | 0.831 | **0.000** |
| 折内 valid 标签 sd 范围 | 0.875–1.156 | **0.315–1.621** |
| 模型 pooled MSE / 常数基线 | 24.7% | 50.9% |
| pooled CV R² | +0.748 | +0.431 |

**指纹主效应（同预算、同 seed、仅换特征集）：随机划分平均 CV MSE 下降 0.1206 eV²；骨架划分反而上升 0.2079 eV²（9 模型，含 GP 为 +0.1892）。**

配合 SHAP 侧的证据（§3）：指纹特征在随机划分与骨架划分中占据的贡献份额几乎相同（87.1% vs 89.1%，逐样本份额口径），但**重要的具体指纹位完全不迁移**（top-50 重叠 1/50，全局秩相关 −0.003）。

建议写法（一段即可）：
> 在按骨架划分的留出评估中，所有模型的解释方差显著下降（最优 R² 由 0.96 降至 0.37），指纹特征的加入不再带来收益，甚至使平均 CV MSE 上升 0.21 eV²。SHAP 对照显示指纹特征在两种划分中占据相近的贡献份额（约 87% 与 89%），但贡献最大的具体指纹位几乎不重叠，说明指纹编码的是划分内的插值信息而非可跨骨架迁移的化学信号。因此后续机理解释采用 Mordred-only 表示，其依据是**可解释性与可迁移性**，而非性能妥协。

这一节同时解决了正文现有的一处逻辑软肋：旧文用"指纹位化学含义不明确"来为 Mordred-only 辩护，属于主观取舍；现在有了可量化的实验依据。

---

## 3. SHAP 段落必须重写（不只是改数字）

重跑后 R-M XGBoost 的 SHAP（100 样本，TreeExplainer，expected_value = 6.2753 eV）：

| 排名 | 描述符 | mean\|SHAP\| (eV) | 方向（实测） |
|---|---|---|---|
| 1 | SlogP_VSA8 | 0.2678 | 值 0→+0.91、22.3→+0.43，≥27.9 全为负（−0.03～−0.37） |
| 2 | **C3SP3** | 0.1954 | 96 个样本取 0 → −0.097；**3 个样本取 4 → +3.38** |
| 3 | TopoShapeIndex | 0.1255 | 除最大值 1.000（+0.194）外全部为负 |
| 4 | NdsCH | 0.0751 | 0 → −0.150；非零 → +0.03～+0.09 并随计数饱和 |
| 5 | MATS1p | 0.0626 | 反相关（r = −0.50），低值 +0.067、高值 −0.040 |

需要改的具体位置：

- **集中度数字**：前 5 名 45.4%（正文 49%）、前 10 名 50.9%（正文 56%）。
- **第二位描述符**：`NsssCH` → **`C3SP3`**。`NsssCH` 现已不在前 10，原段落中"A04 子群驱动"的整段必须删除或改写。
- `C3SP3`（sp3 碳、且与三个碳相连）的贡献几乎全部来自 3/100 个样本（取值 4，这些分子同时带有 Se 相关描述符显著偏高），属**子群效应而非全局趋势**——建议保留旧文对 `NsssCH` 用过的"子群警示"写法，但对象换成 `C3SP3`。
- `SlogP_VSA8` / `TopoShapeIndex` / `NdsCH` / `MATS1p` 四段的方向描述与实测一致，**可基本保留**，只需按新数值微调。

另外两处口径：
- 正文写"shap v0.52.0，100 个训练样本"。实测 `shap 0.53.0`；且 **TreeExplainer/LinearExplainer 用 100 样本，KernelExplainer 用 30 样本**（`nsamples=512`、`l1_reg='num_features(10)'`）。若只有 Mordred XGBoost 一张图，改写为 100 即可；若按 §6-⑥ 扩成四组对照，则必须写清两种采样数。

---

## 4. Methods 需要改的实现细节

| 段 | 现状 | 建议 |
|---|---|---|
| P24 Models | "100 Optuna trials on an 80/20 train-test split" | 补：**训练集内部 5 折**交叉验证选参；最终 1383 训练 / 351 测试（79.7/20.3%） |
| P21 Mordred | "after feature selection before being fed" | 补：**折内拟合**（中位数填补 + VarianceThreshold 0.01 + \|r\|>0.95 去冗余 + StandardScaler 全部只在训练折上拟合），并给出逐折保留数 510–526 |
| P23 Fingerprint | 五个指纹 + 同一套筛选 | 补：原始 1,613（Mordred）+ Morgan/RDKitFP/AtomPair/TopoTorsion 各 2,048 + MACCS 167 = 9,972；折内保留 3,815–4,055 |
| P27 SHAP | 100 样本、v0.52.0 | 见 §3 |
| P61 Results | 411 / 3,825 / 3-fold | 与上同改；并说明"两特征集各自独立做折内筛选"，故筛后维数随折变化 |
| P48 | "Across the 1,743 valid entries" | 见 §6-①（1,743 与实测 1,734 不符；gap 统计 2.34–11.45、mean 6.28、sd 1.04 本身是对的） |

---

## 5. 图表替换清单

| 图 | 现状 | 建议 |
|---|---|---|
| Table 1 | 旧数字（不可复现） | 换成 §1 新表 |
| Fig 7 | 旧 Mordred / 混合 XGBoost 的 true-vs-pred | 用 `experiments\20261016_R-M\analysis_results\R-M\XGBoost_true_vs_predicted.png` 与 `experiments\20261016_R-MF\analysis_results\R-MF\XGBoost_true_vs_predicted.png`；建议统一两子图坐标范围后重绘以便并排比较 |
| Fig 8 | 旧 beeswarm | 用 `experiments\20261016_R-M\analysis_results\R-M\shap_analysis\XGBoost\shap_scatter.png`（现成 beeswarm，前 20 名）。注意 3 个 C3SP3 异常点把 x 轴拉到 3.4 eV，主体被压扁——建议保留但在图注注明异常点来自 3 个分子，或改用截断坐标 |
| 新增 Fig 9 | — | 骨架外推柱状图：四个 run 的最优 Test R² / 平均 CV MSE 对照（材料齐备） |
| 新增 Fig 10（可选） | — | `reports\figures\shap_family_share.png`（四组 SHAP 的指纹族占比）与 `reports\figures\shap_top15_by_run.png`（四组 top-15 对照） |
| SI | Table S4 被同时引作 NICS 表与模型全表 | 表号冲突，需重编 |

---

## 6. 需要你拍板的六件事

1. **数据库条目数**：原始 1,860 行 → `Status = Structure Optimization Finished` 1,734 行（ML 用）→ 正文写的是 1,743。1,743 既不是 1,860 也不是 1,734，需要你确认它指的是什么（是否含未收敛但保留的条目？）。这决定 Abstract / P40 / P48 / P74 四处怎么写。
2. **骨架外推写不写进正文**：我建议写（§2，一段 + 一张图 + SI 全表）。若期刊篇幅紧，至少要在 Discussion 里放一段，否则"指纹没用"这个结论在正文里没有支撑。
3. **Table 1 排序主指标**：Test R²（XGBoost 第一，现状写法）还是 CV MSE（SVR 第一）。建议主表用 Test R²、并加 CV MSE 列，正文明确"选参看 CV、报告看 Test"。
4. **S-MF 的 GaussianProcess 停机（39/100）**：正文只报 9 个模型（干净）还是表格加 † 脚注（透明）。建议表格加 †、正文一句话说明。
5. **特征数写法**：改成逐折范围（510–526 / 3,815–4,055）还是写池化值（523 / 3,986）。建议逐折范围 + 括注池化值，最不容易被审稿人挑。
6. **SHAP 扩不扩成四组对照**：只保留 Mordred XGBoost（改动最小）或加一段四组对照（支撑 §2 的"不迁移"结论）。建议加一段，图放 SI。

---

## 7. 建议的执行顺序（等你说"改"再动手）

1. 你拍板 §6 的 1/3/4 三项（其余我可按建议默认）。
2. 我生成新 Table 1、骨架表、SI 全表（直接从四组 CSV 出 Markdown + CSV，避免手抄错位）。
3. 重绘 Fig 7（统一坐标）、确认 Fig 8 取舍、生成 Fig 9。
4. 按 §1/§3/§4 逐段改文字：**用 python-docx 只替换 run 文本与表格单元格、只替换图片，不动样式与编号**；改前先复制一份 `PorScaDB-Manuscript-20261011-rev1.docx` 留底。
5. 交给你一份"改动对照表"（位置 / 原文 / 新文 / 依据），逐条核对后再定稿。

---

## 附：证据文件索引

- 四组汇总：`reports\四组总对照.md`
- SHAP 四组对照：`reports\四组SHAP对照.md`
- 骨架漂移量化：`reports\骨架漂移量化.md`
- 逐模型原始指标：`experiments\20261016_*\analysis_results\*\model_performance_summary.csv`
- SHAP 明细：`experiments\20261016_R-M\analysis_results\R-M\shap_analysis\XGBoost\{feature_importance.csv,shap_values.npy,samples.csv}`
