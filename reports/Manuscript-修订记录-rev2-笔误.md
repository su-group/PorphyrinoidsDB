# 稿件修订记录 · rev2（笔误修订 · 终版）

- 输入：`PorScaDB-Manuscript-20261011.docx`（原件，未改动）
- 输出：`C:\Users\admin\Downloads\PorScaDB-Manuscript-20261011-rev2.docx`（4.94 MB，151 段）
- 生成脚本：`.dsh_tmp\revise_manuscript.py`（幂等；从原件重建，先写 `.working.docx`，失败不落地）
- rev2 = rev1 的全部数值修订（31 处）+ 本轮 5 处笔误修订

---

## 一、本轮最终保留的改动（5 处 / 5 个段落）

| # | 位置 | 原文 | 改为 | 依据 |
|---|---|---|---|---|
| 1 | P0008 关键词行 | `KEYWORDS：`（全角冒号） | `KEYWORDS:` | 中文全角标点混入英文正文 |
| 2 | P0024 | `refitted on the 1,383-molecule training set and evaluated once on the 351-molecule test set` | `1,387-molecule` / `347-molecule` | 1,383/351 是**骨架划分**规模；表 1 与图 7/8 用的是**随机划分** 1,387/347（`splits.json` 实测） |
| 3 | P0057 | `strongly split in bacteriochlorin (10.07/1.95 eV)` | `(-0.07/1.95 eV)` | 条目 `A03N-B00-C00Y1` 实测 LUMO = −0.0697 eV、LUMO+1 = 1.9535 eV；`10.07` 全库不存在，HF 空轨道也不可能达 +10 eV |
| 4 | P0066 图题 | `Figure 7  Predicted`（双空格） | `Figure 7 Predicted` | 与 Figure 8 题注格式统一 |
| 5 | P0114 参考文献 (32) | `Geerlings, P.; . De Proft, F.` / `Phys, Chem, Chem, Phys, 2014, 16 (28)` | `Geerlings, P.; De Proft, F.` / `Phys. Chem. Chem. Phys., 2014, 16 (28)` | 多余的 `; .`；期刊缩写误用逗号且漏逗号 |

---

## 二、按作者要求**撤销**的改动

| 位置 | 上一版做过什么 | 现在 |
|---|---|---|
| P0019 | 删除中文标记【查找其他文献标记来源】 | **已加回**（作者工作批注，保留） |
| P0041 | 删除中文标记【添加文献】 | **已加回** |
| P0042 | 删除中文标记【这里的叙述需要等数据集完整处理好才能重新写】 | **已加回** |
| P0061（1 处）、P0062（6 处）、P0066、P0068、P0069 | 把 `mordred` 全局改成 `Mordred` | **全部改回小写**（共 11 处小写 `mordred`，与原件逐处一致） |

撤销理由（作者指示）：中文标记是作者本人的工作注记，数据集章节定稿时才处理；正文中不出现这种大写形式。

> 说明：原稿本身在 `Mordred`/`mordred` 上并不统一——P0017、P0021、P0023、P0029、P0062 末句、P0070、P0071、P0072、P0075、P0123 用的是大写 `Mordred`，而 P0061/P0062/P0066/P0068/P0069 的 “mordred descriptor set / mordred-only model” 用小写。rev2 保持原件这一分布不动。

---

## 三、按作者要求追加的两处改动（纯排版 / 大小写，不改文字含义）

### 3.1 P0056 的 `5.864 eV` 标红加粗（待作者核对）

- 位置：P0056 第 1 句 `…has a relatively low mean HOMO-LUMO gap of 5.864 eV.`
- 做法：只把 `5.864 eV` 这一段拆成独立 run，设 `bold=True` + 颜色 `FF0000`；**段落文字与 rev1 逐字相同**（脚本断言 `P0056.text == rev1.text` 为 True）。
- 动因：该均值用现有数据库的任何 (X_element × C_number × B_number) 切片都复现不出来（详见 §四(1)）。标红是留给作者核对用的，最终是改数、改口径还是保留，等作者定；正文一旦定稿，这一处红字需要处理掉。
- 全文档校验：**全文只有这 1 个红色加粗 run**，没有误染其他文字。

### 3.2 我新增文字里的 `Mordred` 改为小写 `mordred`

| 位置 | 改前 | 改后 |
|---|---|---|
| P0023 方法段 | `1,613 Mordred descriptors` | `1,613 mordred descriptors` |
| P0073 新增骨架划分段 | `random forest, Mordred descriptors` | `random forest, mordred descriptors` |

注意：P0070/P0071 中“as defined by Mordred”“Mordred dsCH atom type”指的是 **Mordred 这个软件包**，属专有名词，未改。

---

## 四、校验结果（脚本 exit 0）

- 逐段 diff（rev1 → rev2）：**只有 P0008、P0024、P0057、P0066、P0114 五个段落不同**，其余 146 段逐字相同
- 陈旧值扫描：`none`（0.9359 / 0.2544 / 0.1243 / 0.9095 / 0.0264 / 411 features / 3,825 / 3-fold / 1,743 / 1,860 / NsssCH / 49% / 56% / ranks fourth / 1,383-molecule / 351-molecule / 10.07 / `KEYWORDS：` / `Phys, Chem, Chem, Phys` / `Geerlings, P.; .`）
- 必需值存在：`none missing`（1,734 ×4、C3SP3、NdsCH ranked fourth、scaffold-based split、0.3741、0.19 eV、1,387/347、`(-0.07/1.95 eV)`）
- 结构：151 段、表 11×5、内嵌图 8、`word/media/` 8 个文件、4.94 MB
- **公式对象完好**：`<m:oMath` 计数 原件 12 / rev1 12 / rev2 12
- 中文标记：3 处，均在原位（P0019 / P0041 / P0042）
- 大小写：小写 `mordred` 13 处（原件 11 + 我新增文字 2）、大写 `Mordred` 12 处（均为原件用法，其中 P0070/P0071 指软件包）
- 红色标注：全文 1 处，即 P0056 的 `5.864 eV`

---

## 四、刻意未改、需作者拍板的两处

### (1) P0048 / P0056 的 `5.864 eV` 复现不出来

原文：“bacteriochlorin (A03) has a mean gap of 5.864 eV”（P0056 又说 “relatively low”）。

| 口径 | n | 均值 (eV) |
|---|---|---|
| A03 全部 | 148 | **6.134** |
| A03, X=C | 16 | 5.578 |
| A03, X=N | 48 | 4.939 |
| A03, 游离碱 (C=0) | 64 | 5.310 |
| A03, X=N 且 C=0 | 16 | 5.114 |
| 单条目 `A03N-B00-C00Y1` | 1 | 5.174 |

没有任何子集等于 5.864（最接近的是 A01 的 X=C 子集 5.868, n=16）。已把 (X_element × C_number × B_number) 全部 60 种组合的**均值与中位数**都枚举过，以 ±0.01 eV 为容差**无一命中** —— 也就是说它既不是任何子集的均值，也不是任何子集的中位数，最可能来自旧数据集版本。不敢直接改成 6.134，因为那样 “relatively low” 的前提不成立（6.134 vs 全库 6.280），而 P0056/P0057 的整段论证建立在“A03 明显更低”上。**建议**改用 A03 的 X=C 游离碱口径（5.578 vs A01 的 5.868），与 P0057 的配对比较同源。

### (2) 两处 `Table S4` 指向两张不同的表

- P0058：NICS 结果引 `(Table S4)`
- P0061：完整 ML 评估结果引 `Table S4`

需要 SI 的实际表序才能定谁改 S5，未动。

---

## 五、仍存在但不属笔误的问题（详见 `reports\审稿-初审意见.md`）

1. P0042 那段“数据集归档”描述与实际情况不符（声称含 XYZ 几何与原始输出），属数据集章节待重写范围。
2. P0056/P0057 把 aza 取代体（X=N）当作母体卟啉/细菌卟啉：−6.26/0.15/6.41 与 −5.24/−0.07/5.17 逐位等于 `A01N-B00-C00Y1` 与 `A03N-B00-C00Y1`；母体（X=C）是 6.329 与 5.599。
3. P0057 的机制归因应重排：Δgap = −1.24 eV 中 82% 来自 HOMO 上移（+1.02 eV）。
4. P0056 的 “counterintuitive” 措辞；P0024/P0061 缺超参搜索空间、缺基线、缺误差棒等。


---

## 六、2026-10-20 追加：Figure 7 改按作者提供的参考样式重绘

作者提供了一张参考散点图（两栏、密度着色、粗体轴标题、右下角斜体指标块），要求"之前的散点图图片样式参考这个"。

改动只在**图片本身与 docx 内的 `word/media/image7.png`**，正文文字未动：

- 新脚本 `.dsh_tmp\make_fig7_style.py`；交换脚本 `.dsh_tmp\swap_fig7.py`
- 样式：viridis 密度着色（稀疏深紫 → 密集黄）、灰色虚线 y = x、四边框、粗体轴标题 `ML-Predicted Gap (eV)` / `DFT-Computed Gap (eV)`、左上 `a)` / `b)` 面板字母与栏内粗体小标题、右下斜体粗体指标块
- 指标仍由 `test_predictions.csv` 现算，未手抄：Mordred R² = 0.9321 / RMSE = 0.2611 / MAE = 0.1662；Concatenated R² = 0.9632 / RMSE = 0.1923 / MAE = 0.1178（n = 347）
- 与参考图的差异：轴标题保留 "Gap" 而非 "Value"（本图预测的就是 HOMO–LUMO gap）；单位 "eV" 只写在 RMSE / MAE 行，R² 行不带单位

校验：

- `word/media/image7.png` 与 `reports\figures\fig7_xgboost_true_vs_predicted.png` 的 SHA-256 **完全相同**（206,521 B）
- 图片 extent 仍是 5.77 × 2.69 in（未改 rId、未改 EMU），新图 2160 × 1005 px、长宽比 2.149，与 2.147 匹配，不会被拉伸
- 结构不变：151 段、1 张表（11 × 5）、8 张图
- `PorScaDB-Manuscript-20261011-rev2.docx` 就地替换（5,212,342 B）
