# `experiments/` — 六组运行目录与模型清单

本目录保存三轴 × 两特征集共 **6 次完整运行**，每次 10 个模型、每模型 100 次 Optuna trial。
目录名格式为 `<日期>_<轴>-<特征集>`：

| 目录 | 划分轴 | 特征集 | 说明 |
|---|---|---|---|
| `20261016_R-M` | `random` | Mordred（1,613） | 随机划分，插值基线 |
| `20261016_R-MF` | `random` | 混合（9,972） | 随机划分 + 指纹 |
| `20261016_S-M` | `scaffold` | Mordred | 骨架外推 |
| `20261016_S-MF` | `scaffold` | 混合 | 骨架外推 + 指纹 |
| `20261021_B-M` | `meso` | Mordred | meso 取代基外推 |
| `20261021_B-MF` | `meso` | 混合 | meso 外推 + 指纹 |

六组用的是同一份冻结划分 [`../data/splits-20261021.json`](../data/splits-20261021.json)（seed 42）。

## 单次运行的目录结构

```text
20261021_B-MF/
├── models/B-MF/                 # 估计器、Optuna 试验库、预处理、运行元数据
├── analysis_results/B-MF/       # 指标、测试预测、分组误差、10 张 parity 图、SHAP
└── logs/pipeline_B-MF.log       # 该次运行的完整日志
```

与 `run_pipeline.py --output-root experiments/<目录>` 写出的结构**完全一致**，所以新的运行可以直接并列放进来对比。

### `models/<run>/` 各文件职责

| 文件 | 内容 |
|---|---|
| `<Model>.pkl` | 裸估计器。体积可接受的才收录，见下表 |
| `<Model>.result.pkl` | dict：`model`、`r2_train/rmse_train/mae_train`、`r2_test/rmse_test/mae_test`、`predictions`、`best_params`、`cv_mse` |
| `<Model>.sqlite3` | **Optuna 试验库**：100 个 trial 的参数与折内 CV 值——每个数字的源头 |
| `<Model>.progress.json` | 逐 trial 进度与耗时，供断点续跑与早停判断 |
| `<Model>.sampler.pkl` | TPESampler 状态，`--resume` 依赖它 |
| `preprocessing.pkl` | 折内预处理对象：缺失值中位数、方差/相关性筛选后的列、标准化参数 |
| `filtered_feature_names.json` | 实际进入模型的列名（可读版） |
| `input_signature.json` | 输入矩阵签名（行数、列数、列名哈希），加载时校验 |
| `feature_scaler.pkl` / `scaler.pkl` | 特征标准化器 / 目标标准化器 |
| `run_metadata.json` | 数据与缓存 SHA-256、seed、折归属 `cv_splits`、版本、`feature_names`、`best_model_by_cv` 等 |
| `status.json` | 运行状态标记（完成/中断） |

`prepared_data.pkl`（每组 28–207 MB 的设计矩阵缓存）未收录：它能由 `data/` 的缓存 + 冻结划分原样重建。

### `analysis_results/<run>/` 内容

| 文件 | 内容 |
|---|---|
| `model_performance_summary.csv` | 10 个模型的 CV MSE、Test R²/RMSE/MAE、训练集指标——所有结果表的输入 |
| `partial_model_metrics.csv` | 运行中断时的部分结果（仅 `S-MF` 有） |
| `test_predictions.csv` | 测试集逐样本 y_true / y_pred（parity 图与 R² 的原始数据） |
| `group_errors.csv` | 按母体骨架 `A_number` 分组的分组误差 |
| `<Model>_true_vs_predicted.{png,svg}` | 10 张 parity 图，**六组共用同一坐标窗口**（0.963–12.049 eV），骨架划分下的塌缩不会被重新缩放掩盖 |
| `feature_correlation_heatmap.*`、`feature_target_correlation.{csv,png,svg}` | 特征共线性与特征-目标相关性（全量特征，非折内筛选后） |
| `analysis_report.md` | 该次运行自动生成的摘要 |
| `shap_analysis/<Model>/` | `feature_importance.csv`、`samples.csv`、`shap_values.npy`、bar/scatter/force/dependence 图。每组只对 CV 最优模型计算（R-M XGBoost、R-MF SVR、S-M Lasso、S-MF SVR、B-M KNeighbors、B-MF KNeighbors） |

## 模型清单

✓ 数字为随包交付的 `<Model>.pkl` 体积（`.result.pkl` 同时交付）；
`— (…)` 表示该 `.pkl` 未收录，括号内是它在原运行中的体积。**未收录的模型仍然保留完整证据**：Optuna 试验库、`progress.json`、指标行、测试预测，以及该组最优模型的 SHAP 输出。

| Model | R-M | R-MF | S-M | S-MF | B-M | B-MF |
|---|---|---|---|---|---|---|
| DecisionTree | 0.03 MB | 0.02 MB | 0.00 MB | 0.00 MB | 0.00 MB | 0.00 MB |
| ElasticNet | 0.00 MB | 0.03 MB | 0.00 MB | 0.03 MB | 0.00 MB | 0.03 MB |
| GaussianProcess | — (20.24 MB) | — (56.88 MB) | — (20.14 MB) | — (56.70 MB) | — (20.54 MB) | — (57.03 MB) |
| KNeighbors | — (5.55 MB) | — (42.19 MB) | — (5.53 MB) | — (42.09 MB) | — (5.61 MB) | — (42.10 MB) |
| KernelRidge | — (5.55 MB) | — (42.19 MB) | — (5.53 MB) | — (42.09 MB) | — (5.61 MB) | — (42.10 MB) |
| Lasso | 0.00 MB | 0.03 MB | 0.00 MB | 0.03 MB | 0.00 MB | 0.03 MB |
| LightGBM | 0.42 MB | 0.48 MB | 0.05 MB | 0.22 MB | 0.24 MB | 0.26 MB |
| RandomForest | 6.51 MB | 8.77 MB | 0.22 MB | 0.80 MB | 0.13 MB | 1.14 MB |
| SVR | — (5.47 MB) | — (41.56 MB) | — (5.38 MB) | — (40.42 MB) | — (5.56 MB) | — (41.74 MB) |
| XGBoost | 0.75 MB | 0.93 MB | 0.02 MB | 0.02 MB | 0.19 MB | 0.69 MB |

合计：随包模型参数 **22.1 MB**；被排除的模型参数 **657.8 MB**（另有 `.result.pkl` 孪生文件 657.9 MB、`.sampler.pkl` 0.3 MB、`prepared_data.pkl` 708.4 MB，78 个文件共 2.0 GB）。

六组目录各自体积：`R-M` 22.6 MB、`R-MF` 28.4 MB、`S-M` 7.7 MB、`S-MF` 9.7 MB、`B-M` 7.2 MB、`B-MF` 12.1 MB。

### 为什么这四类模型不收 `.pkl`

`GaussianProcess`、`KernelRidge`、`KNeighbors`、`SVR` 都是**实例型 / 核方法**：pickle 里除了超参数，还带着整份训练设计矩阵。所以单文件从 5.5 MB（Mordred 组）涨到 57 MB（混合组），四类 × 6 组共 658 MB，其中 99% 是 `data/` 里已经存在的同一份特征。

为了这些模型"一条命令就能拿回来"，每条移除决定都逐条记录在 [`build_ledger.json`](build_ledger.json)（文件名 + 原始字节数），并且 Optuna 试验库没删——重训只需几分钟：

```bash
# 重训 meso + 混合组被排除的四个模型（复用冻结划分与同一套超参）
python run_pipeline.py data/Data-20261012_raw_features_fp.csv \
    --split-file data/splits-20261021.json --split-name meso \
    --models SVR KernelRidge KNeighbors GaussianProcess \
    --feature-mode mixed --trials 100 --run-name B-MF-refit \
    --output-root experiments/20261024_B-MF-refit
```

重训结果不必与原文件逐位一致（Optuna 采样与 BLAS 归约顺序都不保证跨机相同），但**指标应落在同一区间**——原始指标行就在 `analysis_results/` 里，可直接对照。

## 加载已交付的模型

```python
import joblib, json, pathlib
run = pathlib.Path("experiments/20261021_B-MF/models/B-MF")
pre = joblib.load(run / "preprocessing.pkl")   # 需 run_metadata.json 记录的环境
est = joblib.load(run / "XGBoost.pkl")         # 裸估计器
meta = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
print(meta["versions"], meta["best_model_by_cv"])
```

> pickle 与 scikit-learn 内部实现绑定。在比 `run_metadata.json` 记录的版本更旧的栈上加载会失败（在 scikit-learn 1.6 下打开 `preprocessing.pkl` 会报 `StringDtype.__init__() takes from 1 to 2 positional arguments but 3 were given`）。

## 完整性核对

每个 `run_metadata.json` 都记录：源数据 `data_sha256`、特征缓存 `cache_sha256`、`cv_splits`（逐折行归属）、`seed`、`trials`、`versions`（8 个库的精确版本）、`source_sha256`（管线源码哈希）。把 `data/` 的文件哈希与之一比对，就能确认报告里的数字对应的是不是同一份输入。

`run_metadata.json` 里的 `data_path` / `cache_path` 是**原始机器上的绝对路径**（`E:\Project-ZhangChengwei\…`），只作溯源线索，运行时不读取。

`source_sha256` 记录的是**运行当时**源码的哈希，不是本包内文件的哈希——发布整理时统一了文件命名与默认参数文本，文件内容随之改变，因此这些值对不上属于预期，请以它判断"哪一版代码产出了这次运行"，而不是"包里的文件有没有被改过"。它还显示六组用了**三个不同版本**的主流程：R-M / R-MF / S-M / S-MF 为 `41952e67…`，B-M 为 `0223a9e8…`，B-MF 为 `3a4768b1…`；本包交付的是最后一版（即 B-MF 所用的那一版）。
