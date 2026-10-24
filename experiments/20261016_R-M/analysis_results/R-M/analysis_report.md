# C_number ∈ {0,1,2} 数据分析报告

生成时间: 2026-10-16 00:27:44

## 1. 数据概览

- 数据文件: `E:\Project-ZhangChengwei\PorScaDB-ML-compute-20261014\PorScaDB-ML\data\Data-20261012.csv`
- 过滤条件: C_number ∈ [0, 1, 2]
- 过滤后样本数: 1734
- 原始特征数: 1613
- 筛选后特征数: 523
- 训练集: 1387, 测试集: 347
- 超参数优化轮次: 100；折内预处理，五折 CV

- SHAP 单位: eV；划分方式: 随机划分

- 特征: mordred；移除指纹: None；仅加指纹: None；Spin: 不使用

## 2. C_number 分布

| C_number | 样本数 |
|---|---|
| 0 | 944 |
| 1 | 390 |
| 2 | 400 |

## 3. 模型性能

| 模型 | CV MSE (eV²) | 训练R² | 训练RMSE | 训练MAE | 测试R² | 测试RMSE | 测试MAE |
|------|--------|--------|----------|---------|--------|----------|--------|
| XGBoost | 0.1342 | 0.9989 | 0.0338 | 0.0256 | 0.9321 | 0.2611 | 0.1662 |
| LightGBM | 0.1437 | 0.9944 | 0.0783 | 0.0443 | 0.8917 | 0.3297 | 0.1991 |
| RandomForest | 0.1554 | 0.9636 | 0.1992 | 0.0906 | 0.8533 | 0.3838 | 0.1940 |
| SVR | 0.1980 | 0.9865 | 0.1213 | 0.0234 | 0.8880 | 0.3353 | 0.2237 |
| KernelRidge | 0.1989 | 0.9978 | 0.0494 | 0.0299 | 0.8623 | 0.3718 | 0.2514 |
| DecisionTree | 0.2195 | 0.9604 | 0.2076 | 0.0973 | 0.8064 | 0.4409 | 0.2401 |
| GaussianProcess | 0.2221 | 0.9977 | 0.0502 | 0.0298 | 0.8725 | 0.3578 | 0.2435 |
| ElasticNet | 0.2655 | 0.8586 | 0.3925 | 0.2810 | 0.8137 | 0.4326 | 0.3195 |
| Lasso | 0.2706 | 0.8309 | 0.4293 | 0.3084 | 0.8119 | 0.4346 | 0.3245 |
| KNeighbors | 0.6008 | 1.0000 | 0.0000 | 0.0000 | 0.4094 | 0.7701 | 0.5188 |

**按五折 CV 选择的模型: XGBoost**

## 4. 超参数

- **Lasso**: {'alpha': 0.001010171604205459, 'max_iter': 1945}
- **ElasticNet**: {'alpha': 0.001078154644581096, 'l1_ratio': 0.127330844167815, 'max_iter': 2109}
- **SVR**: {'C': 912.315859643007, 'gamma': 0.0003499123954435787, 'epsilon': 0.0030677893441494776}
- **KernelRidge**: {'alpha': 5.4968185525017745e-05, 'gamma': 0.0002611140669616281}
- **KNeighbors**: {'n_neighbors': 4, 'weights': 'distance', 'algorithm': 'brute'}
- **DecisionTree**: {'max_depth': 19, 'min_samples_split': 13, 'min_samples_leaf': 3}
- **RandomForest**: {'n_estimators': 198, 'max_depth': 20, 'min_samples_split': 4, 'min_samples_leaf': 3}
- **XGBoost**: {'n_estimators': 174, 'max_depth': 8, 'learning_rate': 0.06681473375334428, 'subsample': 0.7517577713720388}
- **LightGBM**: {'n_estimators': 182, 'max_depth': 10, 'learning_rate': 0.09466111597938345, 'subsample': 0.9283320449567347, 'num_leaves': 36}
- **GaussianProcess**: {'alpha': 0.009942767661062625}

## 5. 文件清单

- 模型: `experiments\20261016_R-M\models\R-M/`
- 分析结果: `experiments\20261016_R-M\analysis_results\R-M/`
- SHAP分析: `experiments\20261016_R-M\analysis_results\R-M/shap_analysis/`
