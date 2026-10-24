# Manuscript 修订记录（rev1）

| 项目 | 内容 |
|---|---|
| 原件 | `C:\Users\admin\Downloads\PorScaDB-Manuscript-20261011.docx`（**本流程只读取、从不写入**；rev1 由它复制而来） |
| 修订件 | `C:\Users\admin\Downloads\PorScaDB-Manuscript-20261011-rev1.docx`（4.94 MB，2026-10-18 13:10） |
| 改动范围 | 替换与四组重跑结果不再一致的数值与口径；骨架外推只作简要提及；与现结果一致的表述原样保留 |
| 编辑方式 | python-docx 逐 run 替换（保留上标、斜体、公式对象）＋ 表格单元格改写 ＋ 两张图位换图 ＋ 新增 1 段 |
| 生成脚本 | `.dsh_tmp\revise_manuscript.py`（**幂等**：每次从原件重新生成 rev1） |
| 数据来源 | `experiments\20261016_R-M`（随机·Mordred）、`20261016_R-MF`（随机·混合）、`20261016_S-M`（骨架·Mordred）、`20261016_S-MF`（骨架·混合） |

---

## 零、数据集规模：1,734 条

`data\Data-20261012.csv` 一共 1,860 行，但其中 **126 行的计算没有成功**，它们不属于数据集：

| Status | 行数 | 是否进数据集 |
|---|---|---|
| Structure Optimization Finished | **1,734** | 是 |
| Imaginary Frequencies | 47 | 否 |
| l9999 | 42 | 否 |
| File Not Found | 30 | 否 |
| Incomplete | 2 | 否 |
| l502 | 3 | 否 |
| l103 | 2 | 否 |

因此数据集规模 = **1,734 条**：gap 统计的 2.34–11.45 / 6.28 / 1.04 在这 1,734 行上算得，ML 的全部数据也是这 1,734 行（随机划分 1,387 训练 / 347 测试，骨架划分 1,383 / 351）。`load_valid_data` 就是按 `Status == Structure Optimization Finished` 过滤的。

原稿的 **1,743** 与任何口径都不符：不等于 1,860（含失败任务）、不等于 1,734（有效数据）、也不等于唯一骨架数 550 或去重后条数。已统一按 **1,734** 替换，**正文不再出现 1,860**（1,860 只是原始表的行数，不是数据库规模）。

---

## 一、正文文字改动（31 处）

### 1.1 第一轮（23 处，数值与口径）

#### 摘要 P0007
| 原文 | 修订 |
|---|---|
| achieve a prediction RMSE < 0.26 eV | achieve a test-set RMSE of **0.19 eV (R² = 0.96)** |

依据：最优模型（XGBoost·混合）Test RMSE = 0.1923 eV、R² = 0.9632。

#### 方法 P0021（Mordred）
新增两句：`The non‑3D descriptor matrix comprised 1,613 features …`、`refitted within every training fold, retaining 510–526 features per fold (523 distinct features over the pooled folds)`。

依据：原写"411 features"，与实测 1,613 原始 / 折内 510–526 差一个量级；且原文未交代特征筛选在折内拟合。

#### 方法 P0023（指纹）
在筛选句后追加 `both fitted within the training folds only`、`9,972 raw features (1,613 Mordred; Morgan/RDKitFP/AtomPair/TopoTorsion at 2,048 bits each; MACCS 167 keys)`、`retained 3,815–4,055 features per fold (3,986 pooled)`。

#### 方法 P0024（Models）
| 原文 | 修订 |
|---|---|
| 100 Optuna trials on an 80/20 train‑test split | 100 Optuna trials **of 5-fold cross-validation** on an 80/20 train‑test split |
| with a standard scaler applied to the features. | with **all preprocessing (median imputation, variance filtering, correlation pruning, and standard scaling) fitted on the training folds only**; the final models were refitted on the **1,383-molecule** training set and evaluated once on the **351-molecule** test set. |

#### 方法 P0027（SHAP）
| 原文 | 修订 |
|---|---|
| shap package (v0.52.0) | shap package (**v0.53.0**) |
| on a random subset of 100 training samples. | … for the tree-based and linear models, whereas **the kernel-based models were explained on 30 samples using 512 kernel-SHAP evaluations over a 25-cluster background**. |

#### 结果 P0048
`Across the 1,743 valid entries` → `Across the 1,734 valid entries`（gap 统计 2.34–11.45、6.28、1.04 正是在这 1,734 行上算的）。

#### 结果 P0061
`(411 features)` → `(1,613 raw features, of which 510–526 are retained per fold)`；`(3,825 features)` → `(9,972 raw features, of which 3,815–4,055 are retained per fold)`；`3-fold` → `5-fold`。

#### 结果 P0062
| 原文 | 修订 |
|---|---|
| R² = 0.9359, RMSE = 0.2544 eV, MAE = 0.1243 eV | R² = **0.9632**, RMSE = **0.1923** eV, MAE = **0.1178** eV |
| followed by LightGBM and Gaussian process | followed by **SVR and Kernel Ridge Regression** |
| mordred-only XGBoost ranks **fourth** … an R² of **0.9095** | ranks **seventh** … an R² of **0.9321** |
| only **0.0264** lower | only **0.0311** lower |
| improvement mainly concentrated in the densely populated low-gap region | **gain largest for high-gap molecules (test MSE 0.330 vs 0.194 eV² above 8.5 eV; 0.172 vs 0.057 eV² between 7.5 and 8.5 eV) rather than in the densely populated low-gap region** |
| This small performance gap | This **modest** performance gap |

逐区间实测（Mordred → 混合，347 个测试样本）：4.5–5.5 n=54 0.0641→0.0514；5.5–6.5 n=207 0.0347→0.0208；6.5–7.5 n=47 0.0975→0.0431；7.5–8.5 n=28 0.1718→0.0569；≥8.5 n=11 0.3299→0.1936。**收益随 gap 升高而增大**，与原文方向相反。

#### 表题 P0063
`indicators the two feature sets.` → `indicators **for** the two feature sets.`

#### 图题 P0068
`top 10 descriptors` → `top 12 descriptors`（重绘图按 top-12 出）。

#### 结论 P0075
`R² = 0.9359, RMSE = 0.2544 eV, MAE = 0.1243 eV` → 当前值；`provided competitive performance` → `provided competitive performance (R² = 0.9321)`。

### 1.2 第二轮（8 处，记录数 / SHAP 描述符 / 骨架外推）

| 位置 | 原文 | 修订 | 依据 |
|---|---|---|---|
| P0007 | comprising **1,743** porphyrin derivatives, all computed using a consistent QC method. | comprising **1,734** porphyrin derivatives, all computed using a consistent QC method. | 见第零节 |
| P0017 | comprising **1,743** structurally diverse porphyrin derivatives | comprising **1,734** structurally diverse porphyrin derivatives | 同上 |
| P0040 | The molecular design produced **1,743** database entries. | The molecular design produced **1,734** database entries. | 126 条失败任务不属于数据集 |
| P0074 | comprising **1,743** macrocycle-modified porphyrin derivatives | comprising **1,734** macrocycle-modified porphyrin derivatives | 同上 |
| P0069 | top five ≈ **49%**、top ten ≈ **56%** | top five ≈ **45%**、top ten ≈ **51%** | 由 `shap_values.npy` 现算：45.38% / 50.94% |
| P0070 | The second-ranked descriptor, **NsssCH** …（整段） | The second-ranked descriptor, **C3SP3**, counts the sp3-hybridized carbon atoms … 96/100 样本 C3SP3 = 0 贡献弱负，3 个 C3SP3 = 4 的样本强正（至 +3.38 eV） | NsssCH 已跌出前 12；C3SP3 mean\|SHAP\| = 0.195 排第 2 |
| P0071 | **MATS1p ranked fourth** … **NdsCH completed the five** … | **NdsCH ranked fourth** … **MATS1p completed the five** … | NdsCH 0.0751 > MATS1p 0.0626 |
| P0075 | …computational–experimental integration.（句末） | 追加一句：A scaffold-based split … **reduced the best test R² to 0.37**, indicating that the present models interpolate within known scaffolds rather than extrapolate to unseen ones. | 见下 |

### 1.3 新增段落（P0072 之后，1 段）

> To determine whether this accuracy extends beyond the scaffolds represented in the training data, the entire protocol was repeated with a scaffold-based split, in which none of the macrocyclic scaffolds of the test set occurs in the training set. The best test-set R² across the ten models decreased from 0.9632 (XGBoost, concatenated descriptors) to 0.3741 (random forest, Mordred descriptors), and the effect of the fingerprint block reversed: the mean cross-validation MSE over the nine converged models increased by 0.21 eV² when the fingerprint block was added, whereas it decreased by 0.12 eV² under the random split. The models therefore interpolate accurately within known scaffolds but do not extrapolate to unseen macrocyclic architectures, which defines the main scope limitation of the present machine-learning analysis.

依据：骨架划分下测试集与训练集的骨架重叠率 **0.000**（A_number 家族重叠率同为 0.000）；十模型 Test R² 最高 0.3741（RandomForest·Mordred）；指纹主效应在随机划分上平均 CV MSE −0.1206 eV²、在骨架划分上 **+0.2079 eV²**（9 个模型，S-MF 的 GaussianProcess 跑到 39/100 trial 后手动停机，其值 1.2136 未计入平均）。

按你的意见，骨架外推只作"简要提及"，**没有另起章节**；完整数据在 `reports\四组总对照.md` 与 `reports\骨架漂移量化.md`，可作 SI 素材。

---

## 二、Table 1（10 行 → 11 行）

结构未动（5 列），按 Test R² 重排并补齐第 10 行（原表只有 9 个数据行，与"Top-10"不符）。

| # | Feature | Model | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|---|
| 1 | Concatenated Descriptors | XGBoost | 0.9632 | 0.1923 | 0.1178 |
| 2 | Concatenated Descriptors | SVR | 0.9565 | 0.2090 | 0.1313 |
| 3 | Concatenated Descriptors | KernelRidge | 0.9443 | 0.2364 | 0.1583 |
| 4 | Concatenated Descriptors | Lasso | 0.9404 | 0.2447 | 0.1639 |
| 5 | Concatenated Descriptors | ElasticNet | 0.9399 | 0.2457 | 0.1668 |
| 6 | Concatenated Descriptors | GaussianProcess | 0.9362 | 0.2532 | 0.1630 |
| 7 | Mordred-Only | XGBoost | 0.9321 | 0.2611 | 0.1662 |
| 8 | Concatenated Descriptors | LightGBM | 0.9207 | 0.2822 | 0.1469 |
| 9 | Mordred-Only | LightGBM | 0.8917 | 0.3297 | 0.1991 |
| 10 | Concatenated Descriptors | RandomForest | 0.8886 | 0.3344 | 0.1394 |

---

## 三、图

| 图 | 处理 |
|---|---|
| Figure 7（`word/media/image7.png`） | 换成 `reports\figures\fig7_xgboost_true_vs_predicted.png`：两栏 (a) Mordred-only / (b) Concatenated，共享坐标，R²/RMSE 由 `test_predictions.csv` 现算（0.9321/0.2611 与 0.9632/0.1923，n=347） |
| Figure 8（`word/media/image8.png`） | 换成 `reports\figures\fig8_shap_beeswarm.png`：由重跑后的 `shap_values.npy` 现绘的横版 beeswarm，top-12，颜色编码描述符值。原图是旧模型、且首位是现已跌出前 12 的 `NsssCH` |

两图都按原图位长宽比出图（7: 2.147，8: 1.249），**不改 rId、不改 EMU**，直接重写 zip 里的图片，因此不会拉伸。

**Figure 1–6 一律不动**（用户 2026-10-18 明确要求"Fig1~6 不动，只修改 7 8"）。已逐字节核验：`image1.png`–`image6.png` 在 rev1 与原件的 SHA-256 前 16 位完全相同，只有 `image7.png` / `image8.png` 变了。8 张图的显示尺寸与原件完全一致。

> **已知残留（按用户要求保留）**：Figure 1 是图形摘要，图内文字仍写 `RMSE < 0.26 eV`、SHAP 清单仍列 `NsssCH`（现为第 2 位的 `C3SP3`）、gap 范围仍写 `1~12 eV`（实测 2.34–11.45）。这些是位图里的字，要改必须回到原始绘图文件重做，故本轮不动；正文与 Figure 7/8 均已是新结果。

---

## 四、刻意未改的部分

1. **Table S4 与 SI 表号冲突**（NICS 表与模型全表同用 S4）——SI 不在这一轮范围内。
2. 对 SHAP 描述符的**逐条展开仅到前五**（原文本来也只写到前五）；若要继续写第 6–12 位，需要新写段落。
3. 骨架外推的量化细节（逐折 MSE、常数基线、oracle 对比、指纹份额 87%/89% 但 top-50 只重叠 1/50）留在报告里，正文只留结论性一句。

---

## 五、已做的校验

| 校验 | 结果 |
|---|---|
| 31 处文字替换全部命中（脚本对每处断言唯一匹配） | 全部 `ok` |
| 旧值残留扫描（0.9359 / 0.2544 / 0.1243 / 0.9095 / 0.0264 / 411 features / 3,825 / 3-fold / **1,743** / **1,860** / v0.52.0 / NsssCH / 49% / 56% / ranks fourth） | `none` |
| 必改项确认（1,734 database entries / 1,734 porphyrin derivatives / 1,734 structurally diverse / 1,734 macrocycle-modified / C3SP3 / NdsCH ranked fourth / scaffold-based split / 0.3741 / test-set RMSE of 0.19 eV） | 全部存在 |
| 表格结构 | 11 行 × 5 列，表头未动 |
| 段落数 | 151（原件 150 + 新增 1 段），新增段样式继承 P0072 的 `TA_Main_Text` |
| 图片 | 8 张全部可解析；显示尺寸 5.77×2.69 in、5.43×4.34 in 未变 |
| 图片字节比对（rev1 vs 原件，SHA-256） | `image1`–`image6` 六个哈希全同 → **Fig 1–6 未被改动**；`image7`/`image8` 不同 → 只有这两张换了 |
| 正文旧值扫描（0.9359 / 0.2544 / 0.1243 / 0.9095 / 0.0264 / 411 features / 3,825 / 3-fold / 1,743 / 1,860 / v0.52.0 / NsssCH / 49% / 56% / ranks fourth） | `none`（图片内的位图文字不在扫描范围，见 §三 已知残留） |
| 文档可读性 | python-docx 可重新打开，表 1、图 8 与原件一致 |
| 上标/公式 | 逐 run 替换，未触碰同段其它 run（`Mg2+`、`R2`、公式对象保持原格式） |

环境没有 LibreOffice（`soffice` 不存在），因此**未做 PDF 渲染目检**，只做了结构与媒体解包核验。
