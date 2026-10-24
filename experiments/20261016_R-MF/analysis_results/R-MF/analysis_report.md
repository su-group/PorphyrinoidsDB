# C_number ∈ {0,1,2} 数据分析报告

生成时间: 2026-10-17 19:22:05

## 1. 数据概览

- 数据文件: `E:\Project-ZhangChengwei\PorScaDB-ML-compute-20261014\PorScaDB-ML\data\Data-20261012.csv`
- 过滤条件: C_number ∈ [0, 1, 2]
- 过滤后样本数: 1734
- 原始特征数: 9972
- 筛选后特征数: 3986
- 训练集: 1387, 测试集: 347
- 超参数优化轮次: 100；折内预处理，五折 CV

- SHAP 单位: eV；划分方式: 随机划分

- 特征: mixed；移除指纹: None；仅加指纹: None；Spin: 不使用

## 2. C_number 分布

| C_number | 样本数 |
|---|---|
| 0 | 944 |
| 1 | 390 |
| 2 | 400 |

## 3. 模型性能

| 模型 | CV MSE (eV²) | 训练R² | 训练RMSE | 训练MAE | 测试R² | 测试RMSE | 测试MAE |
|------|--------|--------|----------|---------|--------|----------|--------|
| SVR | 0.0913 | 0.9673 | 0.1888 | 0.0367 | 0.9565 | 0.2090 | 0.1313 |
| XGBoost | 0.1038 | 0.9994 | 0.0253 | 0.0181 | 0.9632 | 0.1923 | 0.1178 |
| RandomForest | 0.1040 | 0.9809 | 0.1441 | 0.0594 | 0.8886 | 0.3344 | 0.1394 |
| LightGBM | 0.1048 | 0.9877 | 0.1156 | 0.0609 | 0.9207 | 0.2822 | 0.1469 |
| KernelRidge | 0.1055 | 0.9787 | 0.1524 | 0.0841 | 0.9443 | 0.2364 | 0.1583 |
| ElasticNet | 0.1072 | 0.9629 | 0.2011 | 0.1191 | 0.9399 | 0.2457 | 0.1668 |
| GaussianProcess | 0.1150 | 0.9971 | 0.0561 | 0.0276 | 0.9362 | 0.2532 | 0.1630 |
| Lasso | 0.1187 | 0.9682 | 0.1862 | 0.1131 | 0.9404 | 0.2447 | 0.1639 |
| DecisionTree | 0.1386 | 0.9498 | 0.2339 | 0.1169 | 0.8328 | 0.4097 | 0.1895 |
| KNeighbors | 0.2133 | 1.0000 | 0.0000 | 0.0000 | 0.7800 | 0.4700 | 0.3023 |

**按五折 CV 选择的模型: SVR**

## 4. 超参数

- **Lasso**: {'alpha': 0.0020925531513550895, 'max_iter': 7891}
- **ElasticNet**: {'alpha': 0.09627697313029626, 'l1_ratio': 0.016820476725955685, 'max_iter': 8350}
- **SVR**: {'C': 31.261395982479396, 'gamma': 2.986418791375091e-05, 'epsilon': 0.0019764240776531617}
- **KernelRidge**: {'alpha': 0.00844168102702411, 'gamma': 1.9668603359495496e-05}
- **KNeighbors**: {'n_neighbors': 7, 'weights': 'distance', 'algorithm': 'auto'}
- **DecisionTree**: {'max_depth': 18, 'min_samples_split': 18, 'min_samples_leaf': 10}
- **RandomForest**: {'n_estimators': 174, 'max_depth': 20, 'min_samples_split': 4, 'min_samples_leaf': 2}
- **XGBoost**: {'n_estimators': 165, 'max_depth': 9, 'learning_rate': 0.05081719467696616, 'subsample': 0.8055018458373207}
- **LightGBM**: {'n_estimators': 128, 'max_depth': 9, 'learning_rate': 0.08039111146566126, 'subsample': 0.9780413086344616, 'num_leaves': 45}
- **GaussianProcess**: {'alpha': 0.009942767661062625}

## 5. 文件清单

- 模型: `experiments\20261016_R-MF\models\R-MF/`
- 分析结果: `experiments\20261016_R-MF\analysis_results\R-MF/`
- SHAP分析: `experiments\20261016_R-MF\analysis_results\R-MF/shap_analysis/`
