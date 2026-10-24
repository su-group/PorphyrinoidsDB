# 报告索引（reports/）

本目录是**可交付的分析报告**。生成脚本随包收录在 `analysis/`（原 `.dsh_tmp/`），因此每张表都能追溯到代码。
所有数字都可从 `experiments/` 下的运行目录重新生成。少数脚本依赖未随包交付的中间件（`*.json` 汇总、`parity_original/` 原图备份等），逐条标注见 [`analysis/README.md`](../analysis/README.md)。

当前实验共 **6 组**：三条划分轴（随机 `random`、meso 取代基 `meso`、骨架 `scaffold`）× 两种特征集（Mordred / 混合）。

| 报告 | 内容 | 数据来源 | 生成脚本 |
|---|---|---|---|
| `可运行性验证报告.md` | 本机可运行性与可复现性总报告：两个 conda 环境、必设环境变量、复现命令、特征侧逐值复现（99.9991% 在 1e-6 内）、结果汇总、资源与风险、可清理项 | 全部 | 人工汇总 |
| `分组总对照.md` | **当前核心结果**：三轴 × 两特征集共 6 组、每模型 100 trial 的 CV MSE / Test R² 矩阵、CV 最优 vs Test 最优、特征集效应、划分轴效应 | `experiments\20261016_*` + `experiments\20261021_*` | `analysis/build_group_comparison.py` |
| `四组总对照.md` | 第一轮四组（随机/骨架 × Mordred/混合）的逐模型明细与主效应，`分组总对照.md` 的前身 | `experiments\20261016_{R-M,R-MF,S-M,S-MF}` | `analysis/build_four_group_comparison.py` |
| `四组SHAP对照.md` | 随机 vs 骨架两组最优模型的 SHAP 归因构成：指纹各族占 \|SHAP\| 份额（bootstrap CI）、头部族构成、换划分后排名一致性、份额差异的 Welch 检验 | 各 run 的 `analysis_results\<run>\shap_analysis\<model>` | `analysis/shap_four_group.py` |
| `B轴SHAP对照.md` | 加入 meso 轴后的六组 SHAP 构成与排名对照（B-MF 指纹合计 84.17%，R-MF 87.07%，S-MF 89.14%） | 同上，6 组 | `analysis/shap_b_axis.py` |
| `骨架漂移量化.md` | 骨架划分为什么难：骨架重叠率、折间标签均值偏移、逐折标签方差、常数基线与 oracle 基线、模型技能在高低方差折上的分布 | `experiments\20261016_{S-M,R-M}` 的 splits 与预处理 | `analysis/scaffold_drift_analysis.py` |
| `特征与超参分析.md` | 核方法“加指纹后变差”的根因：`config\hyperparameter_optimization.json` 搜索空间过窄，修正后 SVR/KernelRidge 提升 5.0×/5.8× | 折内矩阵重放 | `analysis/`（`fp_family_diagnostic.py`、`hp_vs_default.py`、`mixed_vs_mordred.py`） |
| `指纹-随机折内消融.md` | 随机划分下，在同一批折内矩阵上逐族加指纹的受控消融 | 折内矩阵重放 | `analysis/fp_scaffold_ablation.py` |
| `指纹-骨架折内消融.md` | 骨架划分下的同款受控消融（指纹对骨架外推无正贡献） | 折内矩阵重放 | `analysis/fp_scaffold_ablation.py` |
| `B-取代基外推测试.md` | meso 轴（B_number）的留一外推测试：树模型 R² 0.873/0.892、线性模型失效、B0 折被 A13/A14 混淆；§8.4 记录 B-M/B-MF 的正式结果 | 全部 1,734 条 + 特征缓存 | `analysis/bsplit_test.py`、`analysis/bsplit_test2.py`、`analysis/emit_b_tables.py` |
| `B轴留一组外推.md` | **meso 轴的无偏难度估计**：15 个取代基组重排成 5 折留一组（每组都当一次验证集），并与随机／骨架轴在同一协议同一超参下对照；单列 B0（母体）外推，树模型系统性高估 +2.8 eV | 全部 1,734 条 + Mordred 缓存 | `analysis/baxis_lobo_cv.py`、`analysis/split_reference_cv.py`、`analysis/render_b_lobo_report.py` |
| `R-M_修正前后对照.md` | 修正超参前后的 10 模型逐位对照（8 个逐位不变，仅 SVR/KernelRidge 变化） | `20261014_R-M` vs `20261016_R-M` | `analysis/compare_runs.py` |
| `R-M_原机对照_第一轮.md` | 与原机 `20261013-R-M` 的对照（第一轮，修正前配置） | `20261013-R-M` vs `20261014_R-M` | `analysis/compare_runs.py` |
| `六组结果汇总.xlsx` | 8 个工作表：说明 / 总览 / 全模型指标（60 行）/ 分组误差 / CV_MSE 矩阵 / Test_R2 矩阵 / SHAP 前 15 / SHAP 族份额 | 全部 6 组 | `analysis/build_four_group_xlsx.py` |
| `审稿-初审意见.md` | 五席模拟审稿（期刊匹配、方法、领域、宏观叙事、devil's advocate）+ 我复算后的更正，含可粘贴的修改意见 | 稿件 rev2 + 全部运行结果 | 人工汇总 + `analysis/verify_reviewer_claims.py` |
| `Manuscript-修订记录-rev1.md` | 稿件 rev1 的 31 处数值修订逐条对照（旧值→新值→依据） | `20261016_*` 与稿件原文 | `analysis/revise_manuscript.py` |
| `Manuscript-修订记录-rev2-笔误.md` | rev2 的笔误轮：5 处修订、被回退的中文标记、图表替换记录、待作者确认项 | 同上 | `analysis/revise_manuscript.py` |
| `Manuscript-ML修订思路.md` | 稿件 ML 部分与本地结果的差异清单与改写方案（含新 Table 1 草案） | 全部 | 人工汇总 |
| `SI-泛化能力评估.md` | **可直接并入 SI 的英文草稿**（Supplementary Note S-x）：协议、骨架划分与 meso 划分的 10 模型明细、三轴受控对照、留一组外推、B00 母体外推、两点解释（泛化边界在骨架；骨架是主导自由度故必须主动扩展）、术语定义；表格编号用 `Table S??` 占位 | 六组 `model_performance_summary.csv` + `analysis/` 重放的 `baxis_lobo_cv.json` + `split_reference_cv.json`（中间件，未随包交付） | `analysis/build_si_generalization.py` |
| `术语统一清单.md` | 全文术语统一（`macrocyclic scaffold` / `meso-substituent`）的逐段替换清单：8 处 auto、1 处待判断、1 处他人数据集须跳过、R00–R15 与 B00–B15 的编号冲突 | 稿件 `rev2.docx` | `analysis/build_terminology_checklist.py` |

`figures/` 存放报告引用的图片（PNG 300 dpi + SVG），由各脚本直接写出：
`parity_overview.png` 是 6 组最优模型的密度着色 parity 总览（每组按 Test R² 动态选取），
由 `analysis/make_parity_overview.py` 生成；`axis_split_comparison.png` 是三条划分轴在同一协议同一超参下的
合并 MSE 对照（`analysis/make_axis_figure.py`）；`fig7_xgboost_true_vs_predicted.png`、
`fig8_shap_beeswarm.png` 是稿件替换用图。

六组 `analysis_results\<run>\<Model>_true_vs_predicted.*` 共 **60 张** parity 图已统一按
密度着色风格重绘（`analysis/restyle_parity.py`），**共用同一坐标窗口**（0.963–12.049 eV），
因此骨架划分下预测塌缩成水平带的现象不会被重新缩放掩盖；重绘前的原图备份在 `parity_original/`（未随包交付）。

## 阅读顺序建议

1. `可运行性验证报告.md` —— 环境与可复现性，先确认真能跑。
2. `分组总对照.md` —— 三轴 × 两特征集的当前主结果，一切的基准。
3. `骨架漂移量化.md` —— 解释“为什么骨架划分下分数低”，不是模型坏了。
4. `B-取代基外推测试.md` —— 第三条轴：meso 取代基是插值友好轴，B0（母体）才是真外推。
5. `B轴留一组外推.md` —— 把 B 轴单次留出的运气平掉：meso 轴与随机划分同级（RF 0.170 vs 0.188），骨架轴才是唯一真外推失败；B0 上树模型崩。
6. `指纹-随机折内消融.md` / `指纹-骨架折内消融.md` / `四组SHAP对照.md` / `B轴SHAP对照.md` —— 指纹到底起了什么作用（同族插值，不是外推）。
7. `特征与超参分析.md` —— 核方法的搜索空间缺陷与修复。

## 口径备忘

- 表里的 `CV MSE` 是**管线口径**：`core/train_optimizer.py` 里五折 MSE 的算术平均（eV²）。
  受控消融脚本为排除各折标签方差差异，改用**五折合并 MSE**，所以同一配置会出现 0.6192（管线）与 0.6101（合并）两个数，两者都对。
- 折内特征筛选后的列数会有 ±1 抖动：`np.corrcoef` 的浮点归约顺序不固定，恰好卡在 `|corr|=0.95` 边界的列会进出。
- 各 run 的最优模型不同（XGBoost / SVR / KNeighbors / Lasso …），SHAP 值以 eV 计、量级跟随各自输出幅度，**只能比较份额与排名，不能比较绝对值**。