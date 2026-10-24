# C_number ∈ {0,1,2} 数据分析报告

生成时间: 2026-10-21 22:50:53

## 1. 数据概览

- 数据文件: `E:\Project-ZhangChengwei\PorScaDB-ML-compute-20261014\PorScaDB-ML\data\Data-20261012.csv`
- 过滤条件: C_number ∈ [0, 1, 2]
- 过滤后样本数: 1734
- 原始特征数: 1613
- 筛选后特征数: 525
- 训练集: 1398, 测试集: 336
- 超参数优化轮次: 100；折内预处理，五折 CV

- SHAP 单位: eV；划分方式: B_number

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
| KNeighbors | 0.4783 | 1.0000 | 0.0000 | 0.0000 | 0.7369 | 0.5173 | 0.3548 |
| LightGBM | 0.4855 | 0.8213 | 0.4405 | 0.2998 | 0.7979 | 0.4534 | 0.3338 |
| ElasticNet | 0.4876 | 0.5871 | 0.6696 | 0.4760 | 0.4664 | 0.7367 | 0.5589 |
| Lasso | 0.4878 | 0.5867 | 0.6699 | 0.4762 | 0.4661 | 0.7370 | 0.5588 |
| XGBoost | 0.5465 | 0.7908 | 0.4766 | 0.3169 | 0.7198 | 0.5339 | 0.3569 |
| KernelRidge | 0.5569 | 0.8423 | 0.4137 | 0.2946 | 0.6141 | 0.6266 | 0.4922 |
| SVR | 0.5733 | 0.8299 | 0.4298 | 0.2005 | 0.6227 | 0.6195 | 0.4705 |
| RandomForest | 0.7400 | 0.8952 | 0.3374 | 0.1775 | 0.8667 | 0.3682 | 0.2308 |
| GaussianProcess | 0.7834 | 1.0000 | 0.0000 | 0.0000 | 0.2841 | 0.8534 | 0.6640 |
| DecisionTree | 1.4671 | 0.3925 | 0.8121 | 0.5568 | 0.4848 | 0.7239 | 0.5125 |

**按五折 CV 选择的模型: KNeighbors**

## 4. 超参数

- **Lasso**: {'alpha': 0.06162574685394908, 'max_iter': 9184}
- **ElasticNet**: {'alpha': 0.06613079482361185, 'l1_ratio': 0.9157819222601741, 'max_iter': 4694}
- **SVR**: {'C': 2.404512534768291, 'gamma': 0.001534859284553775, 'epsilon': 0.0019639257228342907}
- **KernelRidge**: {'alpha': 0.23746945692027815, 'gamma': 0.001000090507849627}
- **KNeighbors**: {'n_neighbors': 6, 'weights': 'distance', 'algorithm': 'auto'}
- **DecisionTree**: {'max_depth': 1, 'min_samples_split': 10, 'min_samples_leaf': 2}
- **RandomForest**: {'n_estimators': 14, 'max_depth': 15, 'min_samples_split': 16, 'min_samples_leaf': 10}
- **XGBoost**: {'n_estimators': 35, 'max_depth': 8, 'learning_rate': 0.026737193301430567, 'subsample': 0.962031808805021}
- **LightGBM**: {'n_estimators': 153, 'max_depth': 5, 'learning_rate': 0.01142045442522202, 'subsample': 0.7141962485821065, 'num_leaves': 56}
- **GaussianProcess**: {'alpha': 1.0414493154220982e-10}

## 5. 文件清单

- 模型: `experiments\20261021_B-M\models\B-M/`
- 分析结果: `experiments\20261021_B-M\analysis_results\B-M/`
- SHAP分析: `experiments\20261021_B-M\analysis_results\B-M/shap_analysis/`
