# PorScaDB-ML

**从 SMILES 预测卟啉类大环分子的 DFT HOMO–LUMO Gap，并量化模型究竟能沿哪个方向外推。**

[English](README.md) | 简体中文

本仓库包含完整的训练管线、模型、以及支撑一次受控基准的全部报告、表格与图。基准覆盖 **1,734 个卟啉类分子、10 种回归模型、3 条划分轴 × 2 套特征集**。核心问题不是"哪个模型最准"，而是**描述符模型能沿哪条变异轴外推、又会在哪里失效**。

---

## 目录

- [概览](#概览)
- [主要结论](#主要结论)
- [为什么用三条划分轴](#为什么用三条划分轴)
- [仓库结构](#仓库结构)
- [数据](#数据)
- [环境](#环境)
- [快速开始](#快速开始)
- [复现六组实验](#复现六组实验)
- [收录了什么、什么需要重跑](#收录了什么什么需要重跑)
- [结果](#结果)
- [可复现性说明与注意事项](#可复现性说明与注意事项)
- [引用与许可](#引用与许可)

---

## 概览

| 项目 | 数值 |
|---|---|
| 原始记录 | 1,860 条 DFT 计算（含表头 1,861 行） |
| 有效样本 | **1,734**（`C_number ∈ {0,1,2}`、Gap 有限且为正、SMILES 可解析） |
| 目标 | `Gap(eV)` = LUMO − HOMO |
| 特征集 A | 1,613 个 Mordred 2D 描述符 |
| 特征集 B（混合） | 9,972 列 = 1,613 Mordred + 5 族 RDKit 指纹（Morgan、MACCS、AtomPair、TopoTorsion、RDKitFP） |
| 模型 | 10 种：Lasso、ElasticNet、DecisionTree、RandomForest、XGBoost、LightGBM、SVR、KernelRidge、KNeighbors、GaussianProcess |
| 调参 | Optuna，每模型 100 trial，目标为五折 CV MSE |
| 验证 | 五折 CV，**折内**缺失填补 / 方差与相关性筛选 / 标准化；测试集只评估一次 |
| 划分 | `random` · `meso`（B_number 取代基）· `scaffold`（A_number 母体骨架） |
| 交付运行 | 3 轴 × 2 特征集 = **6 组 × 10 模型** |
| 随机种子 | 42，六组一致 |

全部可在 CPU 上运行。六组完整矩阵约 40 机时，单组 1–4 h。

## 主要结论

各组按交叉验证选出的最优模型（测试集从不参与选择）：

| 划分轴 | 特征集 | CV 最优模型 | CV MSE (eV²) | Test R² | Test 最优模型 | Test R² |
|---|---|---|---|---|---|---|
| `random` | Mordred | XGBoost | **0.1342** | 0.9321 | XGBoost | 0.9321 |
| `random` | 混合 | SVR | **0.0913** | 0.9565 | XGBoost | 0.9632 |
| `meso` | Mordred | KNeighbors | **0.4783** | 0.7369 | RandomForest | 0.8667 |
| `meso` | 混合 | KNeighbors | **0.1767** | 0.9412 | RandomForest | 0.9631 |
| `scaffold` | Mordred | Lasso | **0.6192** | 0.0909 | RandomForest | 0.3741 |
| `scaffold` | 混合 | SVR | **1.1748** | 0.2843 | SVR | 0.2843 |

六组实验支持的五个结论：

1. **指纹是插值工具，不是外推工具。** 加入指纹后平均 CV MSE 变化：随机轴 **−0.1206 eV²**（10/10 变好）、meso 轴 **−0.0599 eV²**（8/10 变好）、骨架轴 **+0.1892 eV²**（0/10 变好，十个模型全部变差）。
2. **划分轴的影响远大于模型选择。** 相对随机划分，CV MSE 中位数在 meso 轴放大 **3.14×**、骨架轴放大 **5.27×**（混合特征下为 4.54× 与 11.79×）。换更好的回归器，收益远小于换一个更诚实的划分。
3. **骨架外推基本失效。** 没有任何配置的 test R² 超过 0.38；parity 图塌缩成水平带——母体骨架未见时，模型退回训练集均值。
4. **CV 排名经不起换轴。** CV 最优同时也是 test 最优的只有 6 组中的 2 组。骨架划分下 CV 冠军（Lasso，CV MSE 0.6192）的 test R² 只有 0.0909，而 CV 上并不突出的 RandomForest 有 0.3741。
5. **搜索空间过窄会伪装成"模型不行"。** 核方法"加指纹就变差"的根因是 `config/hyperparameter_optimization.json` 里的搜索区间过窄；放宽后 SVR 与 KernelRidge 的 CV MSE 分别改善 **5.0×** 与 **5.8×**，本仓库交付的 `20261016_*`、`20261021_*` 均为修正后的结果。

SHAP 分析（每组最优模型）显示：混合特征组的指纹族合计占 |SHAP| 份额的 **84–89 %**（B-MF 84.17 %、R-MF 87.07 %、S-MF 89.14 %）。模型高度依赖子结构比特，这正是它只在训练化学空间内有效的原因。

## 为什么用三条划分轴

单一随机 80/20 只能回答"同一化学空间内的插值效果如何"，回答不了材料读者真正的问题：**面对一个从未算过的骨架会怎样？**

| 轴 | 分组键 | 测试集包含什么 |
|---|---|---|
| `random` | 无 | 独立同分布样本；骨架与取代基都已在训练集出现 |
| `meso` | `B_number` | 训练集未见过的 meso 取代基组合，但母体骨架见过（轻度外推） |
| `scaffold` | `A_number` | 完全未见过的母体骨架（真正的结构外推） |

划分冻结在 `data/splits-20261021.json`（含 `schema_version`、绑定 `data_sha256`、seed 42，每轴记录 `train_positions` / `test_positions` / `folds`）。六组实验都用同一份文件，带 `--split-file` 重跑即可得到完全相同的划分。

## 仓库结构

```text
.
├── prepare_features.py            # SMILES → Mordred 描述符（可选指纹），带缓存签名
├── run_pipeline.py           # 主入口：校验 → 划分 → 五折 CV → 拟合 → 保存 → SHAP → 出图
├── run_experiments.py             # R/S × M/MF 四组矩阵的串行启动器
├── core/
│   ├── data_processing.py         # 过滤、SMILES 规范化、缓存身份校验
│   ├── experiment_setup.py        # 划分清单、特征集选择、折构造
│   ├── molecular_features.py      # Mordred + 5 族 RDKit 指纹
│   └── train_optimizer.py         # 折内预处理、模型库、Optuna 目标函数、评估
├── config/hyperparameter_optimization.json
├── tests/                         # 快速回归测试（不跑全量训练）
├── MANIFEST.md                    # 清单：文件数、体积、未收录项
├── analysis/                      # 产出全部报告/表格/图的原始脚本
├── data/                          # 源数据、已签名特征缓存、冻结划分
├── experiments/<日期>_<运行>/      # 每组一个目录：models/、analysis_results/、logs/
└── reports/                       # 17 份报告（Markdown）、图、汇总工作簿、SI 表格
```

四个子目录 README 各自展开细节：[`data/README.md`](data/README.md)（列字典、缓存契约、三条划分）、[`experiments/README.md`](experiments/README.md)（每组目录里有什么、逐模型清单、重训命令）、[`analysis/README.md`](analysis/README.md)（41 个脚本各做什么、哪些需要改路径）、[`reports/README.md`](reports/README.md)（报告索引与每张表的口径）。

`experiments/<日期>_<运行>/` 与管线用 `--output-root` 写出的结构完全一致：

```text
experiments/20261021_B-MF/
├── models/B-MF/                   # <Model>.pkl、<Model>.result.pkl、<Model>.sqlite3（Optuna 试验库）、
│                                  # preprocessing.pkl、input_signature.json、run_metadata.json
├── analysis_results/B-MF/         # 指标汇总、测试集预测、分组误差、10 张 parity 图（PNG+SVG）、
│                                  # 特征相关性、shap_analysis/
└── logs/pipeline_B-MF.log    # 该次运行的完整日志
```

## 数据

`data/Data-20261012.csv`：一行一次 DFT 计算，共 1,860 行（含表头 1,861 行）。

| 列 | 含义 |
|---|---|
| `idx` | 样本编号，如 `A01XC-B00-C00Y1` |
| `SMILES` | 分子图 |
| `A_number` | 母体骨架编号（`scaffold` 轴的分组键） |
| `B_number` | meso 取代基编号（`meso` 轴的分组键） |
| `C_number` | 金属/轴向配体类别；管线保留 `{0,1,2}` |
| `X_element`、`Y_number` | 杂原子与轴向配体描述 |
| `Status` | DFT 任务状态；仅 `Structure Optimization Finished` 参与训练 |
| `Gap(eV)` | **目标量**，LUMO − HOMO（eV） |
| `HOMO(eV)`、`LUMO(eV)`、`SCF Done(Hartree)`、`Q band (nm)`、`S band (nm)` | 电子结构与光谱 |
| `MolMR`、`LogP`、`TPSA`、`NumRings` … | 理化描述符，供参考 |

过滤规则（1,860 → 1,734）：`C_number ∈ {0,1,2}` ∧ `Status == 'Structure Optimization Finished'` ∧ 目标有限且为正 ∧ SMILES 可解析并规范化。真正起筛选作用的是 Status：恰好 1,734 条带该状态（其余 126 条为 `Imaginary Frequencies` 47、`l9999` 42、`File Not Found` 30、`l502` 3、`Incomplete` 2、`l103` 2）。

`data/` 同时存放两份**已签名**特征缓存与冻结划分：

| 文件 | 大小 | 内容 |
|---|---|---|
| `Data-20261012_raw_features.csv` | 36 MB | 1,734 × 1,613 Mordred 描述符，**三组 `-M` 运行必需** |
| `Data-20261012_raw_features_fp.csv.xz` | 11 MB | 同批记录 + 指纹族，9,972 列；为控制仓库体积**以 xz 压缩交付，使用前需解压**（见[快速开始](#快速开始)） |
| `splits-20261021.json` | 0.5 MB | 冻结的 `random` / `scaffold` / `meso` 划分（seed 42） |

每份缓存都配 `.meta.json`，绑定源文件（`data_sha256`）、自身（`cache_sha256`）、逐行 SMILES/标签/自旋记录，以及生成环境。哈希、行序或源数据不一致时加载器会拒绝——防止在过期特征上静默训练。

## 环境

需要两个隔离环境，因为 Mordred 的 NumPy 版本钉死与现代训练栈冲突：

| 用途 | 文件 | Python |
|---|---|---|
| Mordred 特征生成 | `environment-features.yml` + `requirements-features.txt` | 3.8 |
| 训练、SHAP、绘图 | `environment-ml.yml` + `requirements-ml.txt` | 3.12 |

```bash
conda env create -f environment-features.yml
conda env create -f environment-ml.yml
```

交付产物的实际版本：numpy 2.5.4、pandas 3.0.6、scikit-learn 1.9.1、rdkit 2026.9.1、xgboost 3.4.1、lightgbm 4.7.0、optuna 5.0.0、shap 0.53.0（逐字记录在各 `run_metadata.json` 中）。`requirements-ml.txt` 钉的是最接近的公开版本。

> **pickle 兼容性。** `.pkl` 编码了 scikit-learn 内部结构。在更旧的栈上加载会失败——在 scikit-learn 1.6 下打开随包的 `preprocessing.pkl` 会报 `StringDtype.__init__() takes from 1 to 2 positional arguments but 3 were given`。请使用 `run_metadata.json` 记录的环境，或干脆从缓存重训而不要反序列化。

## 快速开始

```bash
# 0.（仅在需要指纹组时）解压混合特征缓存
python -c "import lzma,shutil; shutil.copyfileobj(lzma.open('data/Data-20261012_raw_features_fp.csv.xz','rb'), open('data/Data-20261012_raw_features_fp.csv','wb'))"

# 1. 冒烟测试：40 个真实样本、每模型 1 trial、3 个 SHAP 样本（约 2 分钟）
conda run -n porscadb-ml python -B run_pipeline.py --smoke

# 2. 单元测试
conda run -n porscadb-ml python -B -m unittest discover -s tests -v

# 3. 一次正式运行：冻结随机划分、10 个模型、100 trial
conda run -n porscadb-ml python -B run_pipeline.py \
    --split-file data/splits-20261021.json --split-name random \
    --run-name R-M --trials 100 --output-root experiments/20261024_R-M
```

冒烟模式的数字只用于验证流程，**不可作为模型性能引用**。

## 复现六组实验

六组之间只差 `--split-name` 与特征缓存。`--output-root` 会原样复现交付的目录结构。

```bash
PY=python   # 位于训练环境内

# --- Mordred 三组 -----------------------------------------------------------
$PY run_pipeline.py data/Data-20261012_raw_features.csv \
    --split-file data/splits-20261021.json --split-name random   --run-name R-M \
    --trials 100 --output-root experiments/20261024_R-M

$PY run_pipeline.py data/Data-20261012_raw_features.csv \
    --split-file data/splits-20261021.json --split-name scaffold --run-name S-M \
    --trials 100 --output-root experiments/20261024_S-M

$PY run_pipeline.py data/Data-20261012_raw_features.csv \
    --split-file data/splits-20261021.json --split-name meso     --run-name B-M \
    --trials 100 --output-root experiments/20261024_B-M

# --- 混合特征三组 -----------------------------------------------------------
$PY run_pipeline.py data/Data-20261012_raw_features_fp.csv \
    --split-file data/splits-20261021.json --split-name random   --run-name R-MF \
    --feature-mode mixed --trials 100 --output-root experiments/20261024_R-MF
# ... S-MF（--split-name scaffold）与 B-MF（--split-name meso）同理
```

常用开关：`--models XGBoost SVR`（只跑部分模型）、`--resume`（断点续跑）、`--fold-jobs N`（折间并行）、`--shap-mode {best,all,none}`、`--drop-fingerprint 族名` / `--only-fingerprint 族名`（单族消融）。

四组网格（R/S × M/MF）也可以用 `run_experiments.py` 串行启动（子进程隔离、可恢复、自动汇总）：

```bash
$PY run_experiments.py --cache data/Data-20261012_raw_features.csv \
    --output-root experiments --experiments R-M R-MF S-M S-MF --trials 100
```

## 收录了什么、什么需要重跑

为了让仓库一分钟内可克隆、同时每个结论都可核查，模型是**选择性**收录的。完整规则见 [`experiments/README.md`](experiments/README.md)：

| 收录 | 原因 |
|---|---|
| 6 组 × 10 模型的 `<Model>.sqlite3`（Optuna 试验库）、`.progress.json`、指标、预测、parity 图、SHAP | `reports/` 里每个数字的证据链 |
| `preprocessing.pkl`、`input_signature.json`、`filtered_feature_names.json`、`run_metadata.json` | 精确的特征清单、填补中位数、scaler、折归属、版本 |
| 体积小且可移植的估计器：DecisionTree、Lasso、ElasticNet、XGBoost、LightGBM、RandomForest | 合计 23 MB，克隆后可直接加载预测 |
| `GaussianProcess`、`KernelRidge`、`KNeighbors`、`SVR` 的 `.pkl` | **未收录**——每个都携带一份训练矩阵（单文件 5.5–57 MB；四类模型仅估计器合计 690 MB，连同其 `.result.pkl` 副本与 `prepared_data.pkl` 共约 2.12 GB）。它们的全部指标、预测、试验库与 SHAP 结果**都在**，用一条命令即可重训（见下） |

`prepared_data.pkl`（每组 28–207 MB 的设计矩阵缓存）同样未收录。

由于 Optuna 试验库保留，重训被排除的模型只需几分钟：

```bash
# 只重训 meso 混合组的核方法，复用冻结划分
python run_pipeline.py data/Data-20261012_raw_features_fp.csv \
    --split-file data/splits-20261021.json --split-name meso \
    --models SVR KernelRidge KNeighbors GaussianProcess \
    --feature-mode mixed --trials 100 --run-name B-MF-refit \
    --output-root experiments/20261024_B-MF-refit
```

## 结果

### CV MSE（eV²，越小越好；粗体为列内最优）

| 模型 | `R-M` | `R-MF` | `B-M` | `B-MF` | `S-M` | `S-MF` |
|---|---|---|---|---|---|---|
| XGBoost | **0.1342** | 0.1038 | 0.5465 | 0.4962 | 1.2542 | 1.2779 |
| LightGBM | 0.1437 | 0.1048 | 0.4855 | 0.5390 | 1.2649 | 1.2733 |
| RandomForest | 0.1554 | 0.1040 | 0.7400 | 1.1031 | 2.5315 | 2.8293 |
| SVR | 0.1980 | **0.0913** | 0.5733 | 0.4716 | 1.0219 | **1.1748** |
| KernelRidge | 0.1989 | 0.1055 | 0.5569 | 0.4082 | 1.0199 | 1.2053 |
| DecisionTree | 0.2195 | 0.1386 | 1.4671 | 1.4055 | 2.9328 | 2.9328 |
| GaussianProcess | 0.2221 | 0.1150 | 0.7834 | 0.4909 | 1.1927 | 1.2136 |
| ElasticNet | 0.2655 | 0.1072 | 0.4876 | 0.4597 | 0.6223 | 1.1846 |
| Lasso | 0.2706 | 0.1187 | 0.4878 | 0.4570 | **0.6192** | 1.2170 |
| KNeighbors | 0.6008 | 0.2133 | **0.4783** | **0.1767** | 1.1882 | 1.2308 |

### Test R²（越大越好；粗体为列内最优）

| 模型 | `R-M` | `R-MF` | `B-M` | `B-MF` | `S-M` | `S-MF` |
|---|---|---|---|---|---|---|
| XGBoost | **0.9321** | **0.9632** | 0.7198 | 0.8161 | 0.0010 | 0.0005 |
| LightGBM | 0.8917 | 0.9207 | 0.7979 | 0.8267 | 0.0530 | 0.0026 |
| RandomForest | 0.8533 | 0.8886 | **0.8667** | **0.9631** | **0.3741** | 0.0488 |
| SVR | 0.8880 | 0.9565 | 0.6227 | 0.9049 | -0.4093 | **0.2843** |
| KernelRidge | 0.8623 | 0.9443 | 0.6141 | 0.9274 | 0.0214 | 0.0982 |
| DecisionTree | 0.8064 | 0.8328 | 0.4848 | 0.4848 | -0.0208 | -0.0208 |
| GaussianProcess | 0.8725 | 0.9362 | 0.2841 | 0.9317 | -1.0759 | 0.2211 |
| ElasticNet | 0.8137 | 0.9399 | 0.4664 | 0.8426 | 0.0846 | 0.2468 |
| Lasso | 0.8119 | 0.9404 | 0.4661 | 0.8336 | 0.0909 | -0.0014 |
| KNeighbors | 0.4094 | 0.7800 | 0.7369 | 0.9412 | -0.0855 | 0.2337 |

### 特征集效应（同比划分内 混合 − Mordred，平均 ΔCV MSE）

| 划分轴 | 可比模型数 | 平均 ΔCV MSE | 变好 | 变差 |
|---|---|---|---|---|
| `random` | 10 | −0.1206 | 10 | 0 |
| `meso` | 10 | −0.0599 | 8 | 2 |
| `scaffold` | 10 | +0.1892 | 0 | 10 |

### 划分轴效应（同比特征集内，相对 `random` 的 ΔCV MSE）

| 特征集 | 轴 | 平均 ΔCV MSE | 中位倍数 |
|---|---|---|---|
| Mordred | `meso` | +0.4198 | 3.14× |
| Mordred | `scaffold` | +1.1239 | 5.27× |
| 混合 | `meso` | +0.4806 | 4.54× |
| 混合 | `scaffold` | +1.4337 | 11.79× |

逐模型指标、预测向量与图都在 `experiments/<运行>/analysis_results/`；叙述性报告索引见 [`reports/README.md`](reports/README.md)。关键图：`reports/figures/parity_overview.png`（六组最优模型共用同一坐标窗）、`axis_split_comparison.png`（轴效应）、`fig8_shap_beeswarm.png`。

## 可复现性说明与注意事项

引用数字前请先读这一节。

1. **CV MSE 是五折 MSE 的算术平均，不是池化 MSE。** 同一骨架运行，池化口径给出 0.6101，管线口径 0.6192。两者都对，但跨表不可混用。
2. **测试集指标只报告、不参与选择。** 模型选择只看 CV MSE；两者不一致时（6 组中有 4 组）并列给出。
3. **SHAP 单位是 eV，但绝对值随各自输出幅度变化。** 只比较份额与排名，不跨运行比较绝对值。
4. **指纹只改善插值。** 任何泛化性论断都应基于骨架轴——那里没有任何配置的 test R² 超过 0.38。
5. **实例型与核方法会"记住"训练集。** KNeighbors 的 train R² 为 1.0000、train RMSE 1.1e-7 eV；它给出的是覆盖度，不是误差估计。
6. **折内特征筛选有 ±1 列抖动**：`np.corrcoef` 的归约顺序不固定，恰好卡在 |corr| = 0.95 边界的列会进出。
7. **pickle 依赖记录的版本栈**（见[环境](#环境)）。缓存、划分与 CSV 结果与版本无关，才是长期可用的产物。
8. **绝对路径**：`run_metadata.json` 里记录的是原机器路径（`E:\Project-ZhangChengwei\…`），只作溯源，运行时不依赖它们。
9. **源表里有九列是由目标派生出来的。** `ChemicalHardness`、`ChemicalSoftness`、`ChemicalPotential`、`Electronegativity`、`ElectrophilicityIndex`、`NucleophilicityIndex`、`ElectronAffinity`、`IonizationPotential` 与 `Energy Min` 都是 `HOMO`/`LUMO`（也就是 `Gap(eV)`）的精确函数，例如 `hardness = Gap / 2`。本包中没有任何模型用到它们——所有特征都只从 `SMILES` 计算——但不要把它们喂给新模型。校验过的公式见 [`data/README.md`](data/README.md)。

## 引用与许可

引用信息见 [`CITATION.cff`](CITATION.cff)。

许可证为 [MIT](LICENSE)。数据集与衍生工件适用同一许可；如使用原始 DFT 数据，请同时标注其来源。
