# R-M 复现对照（原机 2026-10-13 vs 本机 2026-10-14）

- 原始：`experiments\20261013_R-M\analysis_results\R-M`（model_performance_summary.csv）
- 本机：`experiments\20261014_R-M\analysis_results\R-M`（model_performance_summary.csv）
- 同一 `splits.json` → 随机划分完全相同（1387 训练 / 347 测试），可逐模型比较。

### 五折 CV MSE（选型依据，越小越好）

| 模型 | 原始 | 本机重跑 | Δ |
|---|---|---|---|
| DecisionTree | 0.2223 | 0.2195 | -0.0028 |
| ElasticNet | 0.2657 | 0.2655 | -0.0002 |
| GaussianProcess | 0.2221 | 0.2221 | -0.0000 |
| KNeighbors | 0.6008 | 0.6008 | +0.0000 |
| KernelRidge | 0.2582 | 0.2568 | -0.0014 |
| Lasso | 0.2707 | 0.2706 | -0.0001 |
| LightGBM | 0.1454 | 0.1437 | -0.0017 |
| RandomForest | 0.1556 | 0.1554 | -0.0002 |
| SVR | 0.2411 | 0.2381 | -0.0031 |
| XGBoost | 0.1358 | 0.1342 | -0.0016 |

### 测试集 R²（越大越好）

| 模型 | 原始 | 本机重跑 | Δ |
|---|---|---|---|
| DecisionTree | 0.7514 | 0.8064 | +0.0550 |
| ElasticNet | 0.8147 | 0.8137 | -0.0011 |
| GaussianProcess | 0.8725 | 0.8725 | +0.0000 |
| KNeighbors | 0.4094 | 0.4094 | +0.0000 |
| KernelRidge | 0.8596 | 0.8610 | +0.0014 |
| Lasso | 0.8112 | 0.8119 | +0.0008 |
| LightGBM | 0.8799 | 0.8917 | +0.0118 |
| RandomForest | 0.8681 | 0.8533 | -0.0148 |
| SVR | 0.8832 | 0.8849 | +0.0017 |
| XGBoost | 0.9377 | 0.9321 | -0.0056 |

### 测试集 RMSE (eV)

| 模型 | 原始 | 本机重跑 | Δ |
|---|---|---|---|
| DecisionTree | 0.4996 | 0.4409 | -0.0587 |
| ElasticNet | 0.4313 | 0.4326 | +0.0012 |
| GaussianProcess | 0.3578 | 0.3578 | -0.0000 |
| KNeighbors | 0.7701 | 0.7701 | +0.0000 |
| KernelRidge | 0.3754 | 0.3735 | -0.0019 |
| Lasso | 0.4354 | 0.4346 | -0.0009 |
| LightGBM | 0.3472 | 0.3297 | -0.0175 |
| RandomForest | 0.3640 | 0.3838 | +0.0198 |
| SVR | 0.3425 | 0.3400 | -0.0025 |
| XGBoost | 0.2502 | 0.2611 | +0.0110 |

### 测试集 MAE (eV)

| 模型 | 原始 | 本机重跑 | Δ |
|---|---|---|---|
| DecisionTree | 0.2698 | 0.2401 | -0.0297 |
| ElasticNet | 0.3177 | 0.3195 | +0.0018 |
| GaussianProcess | 0.2435 | 0.2435 | -0.0000 |
| KNeighbors | 0.5188 | 0.5188 | +0.0000 |
| KernelRidge | 0.2605 | 0.2589 | -0.0016 |
| Lasso | 0.3255 | 0.3245 | -0.0010 |
| LightGBM | 0.2026 | 0.1991 | -0.0036 |
| RandomForest | 0.1903 | 0.1940 | +0.0036 |
| SVR | 0.2275 | 0.2260 | -0.0015 |
| XGBoost | 0.1598 | 0.1662 | +0.0064 |

### 训练集 R²

| 模型 | 原始 | 本机重跑 | Δ |
|---|---|---|---|
| DecisionTree | 0.9203 | 0.9604 | +0.0401 |
| ElasticNet | 0.8581 | 0.8586 | +0.0004 |
| GaussianProcess | 0.9977 | 0.9977 | -0.0000 |
| KNeighbors | 1.0000 | 1.0000 | +0.0000 |
| KernelRidge | 0.9684 | 0.9710 | +0.0025 |
| Lasso | 0.8309 | 0.8309 | -0.0000 |
| LightGBM | 0.9769 | 0.9944 | +0.0175 |
| RandomForest | 0.9705 | 0.9636 | -0.0069 |
| SVR | 0.9803 | 0.9834 | +0.0030 |
| XGBoost | 0.9995 | 0.9989 | -0.0005 |

### 过拟合信号（train R² − test R²，越大越严重）

| 模型 | 原始 | 本机重跑 |
|---|---|---|
| DecisionTree | 0.1690 | 0.1541 |
| ElasticNet | 0.0434 | 0.0449 |
| GaussianProcess | 0.1252 | 0.1252 |
| KNeighbors | 0.5906 | 0.5906 |
| KernelRidge | 0.1088 | 0.1099 |
| Lasso | 0.0198 | 0.0189 |
| LightGBM | 0.0969 | 0.1026 |
| RandomForest | 0.1024 | 0.1103 |
| SVR | 0.0971 | 0.0985 |
| XGBoost | 0.0618 | 0.0669 |

---

## R-MF（混合特征）本机结果

- 本机：`experiments\20261014_R-MF\analysis_results\R-MF`（partial_model_metrics.csv）

### 五折 CV MSE

| 模型 | 值 |
|---|---|
| DecisionTree | 0.1386 |
| ElasticNet | 0.1072 |
| KNeighbors | 0.2133 |
| KernelRidge | 0.4715 |
| Lasso | 0.1187 |
| SVR | 0.4565 |

### 测试集 R²

| 模型 | 值 |
|---|---|
| DecisionTree | 0.8328 |
| ElasticNet | 0.9399 |
| KNeighbors | 0.7800 |
| KernelRidge | 0.5601 |
| Lasso | 0.9404 |
| SVR | 0.5704 |

### 测试集 RMSE (eV)

| 模型 | 值 |
|---|---|
| DecisionTree | 0.4097 |
| ElasticNet | 0.2457 |
| KNeighbors | 0.4700 |
| KernelRidge | 0.6646 |
| Lasso | 0.2447 |
| SVR | 0.6568 |

### 测试集 MAE (eV)

| 模型 | 值 |
|---|---|
| DecisionTree | 0.1895 |
| ElasticNet | 0.1668 |
| KNeighbors | 0.3023 |
| KernelRidge | 0.3881 |
| Lasso | 0.1639 |
| SVR | 0.3950 |
