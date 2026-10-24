# C_number ∈ {0,1,2} 数据分析报告

生成时间: 2026-10-17 22:51:12

## 1. 数据概览

- 数据文件: `E:\Project-ZhangChengwei\PorScaDB-ML-compute-20261014\PorScaDB-ML\data\Data-20261012.csv`
- 过滤条件: C_number ∈ [0, 1, 2]
- 过滤后样本数: 1734
- 原始特征数: 1613
- 筛选后特征数: 523
- 训练集: 1383, 测试集: 351
- 超参数优化轮次: 100；折内预处理，五折 CV

- SHAP 单位: eV；划分方式: A_number

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
| Lasso | 0.6192 | 0.8001 | 0.4724 | 0.3340 | 0.0909 | 0.9047 | 0.6717 |
| ElasticNet | 0.6223 | 0.7993 | 0.4733 | 0.3347 | 0.0846 | 0.9078 | 0.6733 |
| KernelRidge | 1.0199 | 0.8385 | 0.4246 | 0.2954 | 0.0214 | 0.9386 | 0.7369 |
| SVR | 1.0219 | 0.8518 | 0.4067 | 0.1955 | -0.4093 | 1.1264 | 0.8554 |
| KNeighbors | 1.1882 | 1.0000 | 0.0000 | 0.0000 | -0.0855 | 0.9886 | 0.7016 |
| GaussianProcess | 1.1927 | 0.9975 | 0.0524 | 0.0287 | -1.0759 | 1.3671 | 1.0202 |
| XGBoost | 1.2542 | 0.0929 | 1.0062 | 0.6236 | 0.0010 | 0.9484 | 0.7113 |
| LightGBM | 1.2649 | 0.1511 | 0.9734 | 0.5960 | 0.0530 | 0.9234 | 0.6913 |
| RandomForest | 2.5315 | 0.9132 | 0.3112 | 0.1764 | 0.3741 | 0.7506 | 0.5494 |
| DecisionTree | 2.9328 | 0.4968 | 0.7495 | 0.5017 | -0.0208 | 0.9587 | 0.7350 |

**按五折 CV 选择的模型: Lasso**

## 4. 超参数

- **Lasso**: {'alpha': 0.007037542783164669, 'max_iter': 7151}
- **ElasticNet**: {'alpha': 0.0072603713631880775, 'l1_ratio': 0.97707370929739, 'max_iter': 3875}
- **SVR**: {'C': 16.68380548743517, 'gamma': 0.00023921058833668935, 'epsilon': 0.006087218942445001}
- **KernelRidge**: {'alpha': 0.01861777187418143, 'gamma': 7.48981295470159e-05}
- **KNeighbors**: {'n_neighbors': 4, 'weights': 'distance', 'algorithm': 'brute'}
- **DecisionTree**: {'max_depth': 1, 'min_samples_split': 19, 'min_samples_leaf': 9}
- **RandomForest**: {'n_estimators': 32, 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 7}
- **XGBoost**: {'n_estimators': 10, 'max_depth': 1, 'learning_rate': 0.01074752031268655, 'subsample': 0.5568813560152033}
- **LightGBM**: {'n_estimators': 10, 'max_depth': 6, 'learning_rate': 0.010169399585314022, 'subsample': 0.837924618369464, 'num_leaves': 27}
- **GaussianProcess**: {'alpha': 0.008008123007303159}

## 5. 文件清单

- 模型: `experiments\20261016_S-M\models\S-M/`
- 分析结果: `experiments\20261016_S-M\analysis_results\S-M/`
- SHAP分析: `experiments\20261016_S-M\analysis_results\S-M/shap_analysis/`
