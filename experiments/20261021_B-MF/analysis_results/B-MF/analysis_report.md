# C_number ∈ {0,1,2} 数据分析报告

生成时间: 2026-10-22 07:48:03

## 1. 数据概览

- 数据文件: `E:\Project-ZhangChengwei\PorScaDB-ML-compute-20261014\PorScaDB-ML\data\Data-20261012.csv`
- 过滤条件: C_number ∈ [0, 1, 2]
- 过滤后样本数: 1734
- 原始特征数: 9972
- 筛选后特征数: 3946
- 训练集: 1398, 测试集: 336
- 超参数优化轮次: 100；折内预处理，五折 CV

- SHAP 单位: eV；划分方式: B_number

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
| KNeighbors | 0.1767 | 1.0000 | 0.0000 | 0.0000 | 0.9412 | 0.2445 | 0.1354 |
| KernelRidge | 0.4082 | 0.9898 | 0.1051 | 0.0606 | 0.9274 | 0.2717 | 0.1758 |
| Lasso | 0.4570 | 0.7783 | 0.4906 | 0.3346 | 0.8336 | 0.4114 | 0.3062 |
| ElasticNet | 0.4597 | 0.7899 | 0.4776 | 0.3250 | 0.8426 | 0.4002 | 0.2973 |
| SVR | 0.4716 | 1.0000 | 0.0012 | 0.0011 | 0.9049 | 0.3111 | 0.2042 |
| GaussianProcess | 0.4909 | 1.0000 | 0.0000 | 0.0000 | 0.9317 | 0.2636 | 0.1682 |
| XGBoost | 0.4962 | 0.8297 | 0.4300 | 0.2758 | 0.8161 | 0.4325 | 0.2840 |
| LightGBM | 0.5390 | 0.7879 | 0.4799 | 0.3397 | 0.8267 | 0.4198 | 0.3235 |
| RandomForest | 1.1031 | 0.9365 | 0.2625 | 0.1184 | 0.9631 | 0.1936 | 0.1240 |
| DecisionTree | 1.4055 | 0.3925 | 0.8121 | 0.5568 | 0.4848 | 0.7239 | 0.5125 |

**按五折 CV 选择的模型: KNeighbors**

## 4. 超参数

- **Lasso**: {'alpha': 0.06542197859570792, 'max_iter': 6333}
- **ElasticNet**: {'alpha': 0.061113198723240746, 'l1_ratio': 0.9867568826397526, 'max_iter': 8507}
- **SVR**: {'C': 159.12500041232192, 'gamma': 0.00035127618408267556, 'epsilon': 0.0010318608792974876}
- **KernelRidge**: {'alpha': 0.11158822364813499, 'gamma': 0.0002883364807555473}
- **KNeighbors**: {'n_neighbors': 4, 'weights': 'distance', 'algorithm': 'brute'}
- **DecisionTree**: {'max_depth': 1, 'min_samples_split': 20, 'min_samples_leaf': 10}
- **RandomForest**: {'n_estimators': 113, 'max_depth': 14, 'min_samples_split': 8, 'min_samples_leaf': 9}
- **XGBoost**: {'n_estimators': 94, 'max_depth': 10, 'learning_rate': 0.010539761017734801, 'subsample': 0.999722776296009}
- **LightGBM**: {'n_estimators': 126, 'max_depth': 2, 'learning_rate': 0.027010527749605478, 'subsample': 0.6831809216468459, 'num_leaves': 51}
- **GaussianProcess**: {'alpha': 2.8787007443019176e-10}

## 5. 文件清单

- 模型: `experiments\20261021_B-MF\models\B-MF/`
- 分析结果: `experiments\20261021_B-MF\analysis_results\B-MF/`
- SHAP分析: `experiments\20261021_B-MF\analysis_results\B-MF/shap_analysis/`
