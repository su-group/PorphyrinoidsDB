#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整流水线：仅使用 C_number ∈ {0,1,2} 的数据进行训练、评估和SHAP分析
所有执行代码必须放在 main() 中，避免Windows多进程重导入问题
"""
import os
import sys
import argparse
from pathlib import Path

def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='PorScaDB 五折 CV 训练与解释')
    parser.add_argument('features', nargs='?', help='带 .meta.json 的原始特征缓存')
    parser.add_argument('--data', type=Path, default=root / 'data/Data-20261012.csv')
    parser.add_argument('--output-root', type=Path, default=root)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--group-column', help='按此数据列分组划分和五折 CV，例如 A_number')
    parser.add_argument('--smoke', action='store_true', help='仅 40 样本、每模型 1 trial、小规模 SHAP')
    parser.add_argument('--feature-mode', choices=['mordred', 'mixed'], default='mordred')
    parser.add_argument('--drop-fingerprint', choices=['Morgan', 'MACCS', 'AtomPair', 'TopoTorsion', 'RDKitFP'])
    parser.add_argument('--only-fingerprint', choices=['Morgan', 'MACCS', 'AtomPair', 'TopoTorsion', 'RDKitFP'])
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--split-name', default='random',
                        help='固定划分清单中的键，例如 random / scaffold / meso')
    parser.add_argument('--models', nargs='+', help='仅运行这些模型；默认全部 10 个')
    parser.add_argument('--run-name', help='固定运行标识，供实验矩阵和断点恢复使用')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--fold-jobs', type=int, default=1,
                        help='五折 CV 的并行进程数；>1 时折间并行（各折独立，结果与串行一致）')
    parser.add_argument('--shap-mode', choices=['best', 'all', 'none'], default='best')
    args = parser.parse_args()
    if args.trials < 1 or args.threads < 1:
        parser.error('--trials 和 --threads 必须为正数')
    suffix = '_raw_smoke_features.csv' if args.smoke else '_raw_features.csv'
    if args.features is None:
        args.features = args.data.with_name(args.data.stem + suffix)
    return args


def main():
    args = parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')

    import logging
    from logging.handlers import RotatingFileHandler
    from datetime import datetime

    # 添加项目根目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    import pandas as pd
    import numpy as np
    import pickle
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    from tqdm import tqdm
    from scipy.stats import gaussian_kde

    # ==================== 出图规范：Nature 风格（评估图与 SHAP 图统一） ====================
    # Arial 字体、无上/右边框、无图内标题、无网格；SVG（矢量可编辑）+ PNG（300 dpi）双格式，
    # 覆盖同位置同名文件。坐标轴刻度数字与轴标题在原有基础上统一放大 TICK_SCALE 倍。
    TICK_SCALE     = 1.5
    FS_TICK_PARITY = 7.5 * TICK_SCALE   # 散点图：刻度数字
    FS_AXIS_PARITY = 8.5 * TICK_SCALE   # 散点图：坐标轴标题
    FS_METRIC      = 7.5 * TICK_SCALE   # 散点图：右下角指标注释
    FS_TICK_SMALL  = 6.0 * TICK_SCALE   # 条形图：刻度数字与特征名
    FS_LABEL_SMALL = 7.0 * TICK_SCALE   # 条形图：坐标轴标题
    FS_TICK_TINY   = 5.5 * TICK_SCALE   # 热图：刻度数字与特征名

    COLOR_BLUE = "#0F4D92"
    COLOR_RED  = "#B64342"
    COLOR_DARK = "#4D4D4D"
    COLOR_MID  = "#767676"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.labelweight": "bold",
        "legend.frameon": False,
        "axes.unicode_minus": False,
        "mathtext.default": "regular",
    })

    def save_nature_fig(fig, path_noext):
        """Nature 风格双格式输出：SVG（矢量）+ PNG（300 dpi），覆盖同名文件。"""
        fig.savefig(path_noext + '.svg', bbox_inches='tight')
        fig.savefig(path_noext + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ================================= 配置 =================================
    PATH = args.data.resolve()
    FEATURES_DIR = Path(args.features).resolve()
    C_NUMBER_VALUES = [0, 1, 2]
    HyPer_epoch = 1 if args.smoke else args.trials
    TIMESTAMP = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    run_name = args.run_name or TIMESTAMP + ('_smoke' if args.smoke else '')
    MODEL_DIR = str(args.output_root / 'models' / run_name)
    ANALYSIS_DIR = str(args.output_root / 'analysis_results' / run_name)
    LOG_DIR = str(args.output_root / 'logs')
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.DEBUG)
    log_file = os.path.join(LOG_DIR, f'pipeline_{run_name}.log')
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 70)
    logger.info(f"C_number in {C_NUMBER_VALUES} 完整流水线启动")
    logger.info(f"时间戳: {TIMESTAMP}")
    logger.info("=" * 70)

    from core.data_processing import load_valid_data, load_feature_cache, file_sha256
    from core.train_optimizer import (
        create_models, fit_preprocessing, transform_features, clean_selected_features,
        prepare_cv_folds, optimize_model_parameters, evaluate_model,
        target_correlations, shap_to_ev,
    )
    from sklearn.base import clone
    from threadpoolctl import threadpool_limits
    thread_limiter = threadpool_limits(limits=args.threads)
    from core.experiment_setup import (select_experiment_features, load_split_manifest,
                                       split_group_column)

    # 1. 数据与缓存：核对版本、样本身份、行顺序，不默许未知旧缓存。
    data_valid = load_valid_data(PATH, smoke=args.smoke)
    features, cache_metadata = load_feature_cache(FEATURES_DIR, data_valid, PATH)
    if data_valid['Y_number'].nunique() != 1:
        raise ValueError('本轮方案不使用 Spin，但数据出现多种自旋，请重新确认实验范围')
    X = select_experiment_features(features, args.feature_mode, args.drop_fingerprint, args.only_fingerprint)
    assert not any(c.startswith('Spin') for c in X)
    y = data_valid['Gap(eV)'].to_numpy()
    c_number_counts = data_valid['C_number'].value_counts().sort_index()
    logger.info(f'有效数据 {len(y)} 行，原始特征 {X.shape[1]} 列；smoke={args.smoke}')

    # 2. 外部测试集只用于最后评估；可按结构类别分组检验外推能力。
    positions = np.arange(len(y))
    groups = None
    fold_indices = None
    if args.split_file:
        train_idx, test_idx, fold_indices = load_split_manifest(
            args.split_file, data_valid, PATH, args.split_name, args.smoke)
        # 分组列由划分清单声明（scaffold→A_number，meso→B_number），用于分组误差汇总。
        group_column = split_group_column(args.split_file, args.split_name)
        if group_column:
            args.group_column = group_column
            groups = data_valid[group_column].astype(str).to_numpy()
    elif args.group_column:
        if args.group_column not in data_valid or data_valid[args.group_column].isna().any():
            raise ValueError('分组列不存在或含缺失值')
        groups = data_valid[args.group_column].astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))
    else:
        train_idx, test_idx = train_test_split(positions, test_size=0.2, random_state=42)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_original, y_test_original = y[train_idx], y[test_idx]
    overlap = set(data_valid.iloc[train_idx]['SMILES']) & set(data_valid.iloc[test_idx]['SMILES'])
    if overlap:
        raise ValueError('训练/测试存在相同规范化 SMILES，请使用 --group-column SMILES')

    # 3. 固定数据与源码身份，恢复时禁止混用另一套实验配置。
    source_root = Path(__file__).resolve().parent
    signature = {
        'data_sha256': file_sha256(PATH), 'cache_sha256': file_sha256(FEATURES_DIR),
        'feature_mode': args.feature_mode, 'drop_fingerprint': args.drop_fingerprint,
        'only_fingerprint': args.only_fingerprint, 'models': args.models,
        'train_positions': train_idx.tolist(), 'test_positions': test_idx.tolist(),
        'fold_indices': fold_indices, 'group_column': args.group_column,
        'trials': HyPer_epoch, 'smoke': args.smoke, 'threads': args.threads,
        'fold_jobs': args.fold_jobs,
        'shap_mode': args.shap_mode,
        'source_sha256': {str(p.relative_to(source_root)): file_sha256(p)
                          for p in [Path(__file__).resolve(), *sorted(source_root.glob('core/*.py')),
                                    source_root / 'config/hyperparameter_optimization.json']},
    }
    os.makedirs(MODEL_DIR, exist_ok=args.resume)
    os.makedirs(ANALYSIS_DIR, exist_ok=args.resume)
    signature_path = Path(MODEL_DIR) / 'input_signature.json'
    if signature_path.exists():
        if not args.resume or json.loads(signature_path.read_text(encoding='utf-8')) != signature:
            raise ValueError('已有运行与当前数据、配置或源码不一致，不能恢复；请使用新运行目录')
    else:
        signature_path.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    status_path = Path(MODEL_DIR) / 'status.json'

    def record_status(phase, **details):
        status = {'run_name': run_name, 'phase': phase,
                  'updated_at': datetime.now().isoformat(), **details}
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(status_path)

    record_status('preprocessing')
    prepared_path = Path(MODEL_DIR) / 'prepared_data.pkl'
    if args.resume and prepared_path.exists():
        with open(prepared_path, 'rb') as f:
            folds, preprocessing = pickle.load(f)
        logger.info('复用已校验的折内预处理检查点')
    else:
        logger.info('准备五折 CV（折内预处理）...')
        train_groups = groups[train_idx] if groups is not None else None
        folds = prepare_cv_folds(X_train, y_train_original, train_groups, fold_indices)
        preprocessing = fit_preprocessing(X_train, y_train_original)
        temp = prepared_path.with_suffix('.tmp')
        with open(temp, 'wb') as f:
            pickle.dump((folds, preprocessing), f)
        temp.replace(prepared_path)

    # 4. 最终预处理仅在外部训练集拟合，测试集只变换。
    filtered_feature_names = preprocessing['feature_names']
    X_train_filtered = clean_selected_features(X_train, preprocessing)
    X_train_scaled = transform_features(X_train, preprocessing)
    X_test_scaled = transform_features(X_test, preprocessing)
    feature_scaler_obj = preprocessing['feature_scaler']
    target_scaler = preprocessing['target_scaler']
    y_train = target_scaler.transform(y_train_original.reshape(-1, 1)).ravel()

    import platform
    import importlib.metadata
    import subprocess
    packages = ['numpy', 'pandas', 'scikit-learn', 'rdkit', 'xgboost', 'lightgbm', 'optuna', 'shap']
    try:
        git_revision = subprocess.check_output(
            ['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
    metadata = {
        'smoke': args.smoke, 'data_path': str(PATH), 'cache_path': str(FEATURES_DIR),
        'data_sha256': file_sha256(PATH), 'cache_sha256': file_sha256(FEATURES_DIR),
        'cache_metadata': cache_metadata, 'git_revision': git_revision,
        'cv_folds': 5, 'seed': 42, 'trials': HyPer_epoch, 'group_column': args.group_column,
        'cv_fold_jobs': args.fold_jobs,
        'target_unit': 'eV', 'shap_unit': 'eV',
        'train_source_rows': data_valid.iloc[train_idx]['source_row'].tolist(),
        'test_source_rows': data_valid.iloc[test_idx]['source_row'].tolist(),
        'cv_splits': [{'train_source_rows': data_valid.iloc[train_idx[f['train_positions']]]['source_row'].tolist(),
                       'valid_source_rows': data_valid.iloc[train_idx[f['valid_positions']]]['source_row'].tolist()}
                      for f in folds],
        'python': platform.python_version(),
        'versions': {name: importlib.metadata.version(name) for name in packages},
        'source_sha256': {str(p.relative_to(Path(__file__).resolve().parent)): file_sha256(p)
                          for p in [Path(__file__).resolve(),
                                    *sorted(Path(__file__).resolve().parent.glob('core/*.py')),
                                    Path(__file__).resolve().parent / 'config/hyperparameter_optimization.json']},
        'feature_names': filtered_feature_names, 'feature_mode': args.feature_mode,
        'drop_fingerprint': args.drop_fingerprint, 'only_fingerprint': args.only_fingerprint,
        'spin_used': False, 'shap_mode': args.shap_mode, 'threads': args.threads,
    }
    with open(os.path.join(MODEL_DIR, 'run_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(MODEL_DIR, 'preprocessing.pkl'), 'wb') as f:
        pickle.dump(preprocessing, f)

    # 5. 先按 CV 选参数。某模型失败就明确报错，不把失败伪装为完成。
    results = {}
    models = create_models(args.smoke)
    if args.models:
        if any(name not in models for name in args.models):
            raise ValueError(f'未知模型: {args.models}')
        models = {name: models[name] for name in args.models}
    for model_name, base_model in tqdm(models.items(), desc='训练模型'):
        record_status('training', model=model_name, completed_models=list(results))
        checkpoint = Path(MODEL_DIR) / f'{model_name}.result.pkl'
        if args.resume and checkpoint.exists():
            with open(checkpoint, 'rb') as f:
                res = pickle.load(f)
            logger.info(f'{model_name}: 复用已完成模型')
        else:
            best_params, cv_mse = optimize_model_parameters(
                model_name, base_model, folds, n_trials=HyPer_epoch, smoke=args.smoke,
                study_path=Path(MODEL_DIR) / f'{model_name}.sqlite3',
                fold_jobs=args.fold_jobs)
            model = clone(base_model).set_params(**best_params)
            model.fit(X_train_scaled, y_train)
            res = evaluate_model(model, X_train_scaled, X_test_scaled,
                                 y_train_original, y_test_original, target_scaler)
            res['best_params'], res['cv_mse'] = best_params, cv_mse
            temp = checkpoint.with_suffix('.tmp')
            with open(temp, 'wb') as f:
                pickle.dump(res, f)
            temp.replace(checkpoint)
            with open(Path(MODEL_DIR) / f'{model_name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        results[model_name] = res
        partial = [{'Model': name, 'CV MSE (eV²)': r['cv_mse'], 'Test R²': r['r2_test'],
                    'Test RMSE': r['rmse_test'], 'Test MAE': r['mae_test']} for name, r in results.items()]
        pd.DataFrame(partial).to_csv(Path(ANALYSIS_DIR) / 'partial_model_metrics.csv', index=False)
        logger.info(f'{model_name}: CV MSE={res["cv_mse"]:.4f} eV², Test R²={res["r2_test"]:.4f}')
    record_status('reports', completed_models=list(results))
    # 最佳模型由训练集 CV 决定，不使用测试成绩反向挑选模型。
    best_model_name = min(results, key=lambda name: results[name]['cv_mse'])
    metadata['best_model_by_cv'] = best_model_name
    metadata['models'] = {name: {'parameters': res['model'].get_params(deep=False),
                                 'best_params': res['best_params'], 'cv_mse_ev2': res['cv_mse']}
                          for name, res in results.items()}
    with open(os.path.join(MODEL_DIR, 'run_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    predictions = data_valid.iloc[test_idx][['source_row', 'idx', 'SMILES', 'A_number', 'C_number', 'Gap(eV)']].copy()
    for name, res in results.items():
        predictions[name] = res['predictions']
    predictions.to_csv(os.path.join(ANALYSIS_DIR, 'test_predictions.csv'), index=False)

    # 各测试骨架/金属的误差，避免整体分数掩盖小组。
    group_errors = []
    for column in ['A_number', 'C_number']:
        for group, rows in predictions.groupby(column):
            for name in results:
                error = rows[name] - rows['Gap(eV)']
                group_errors.append({'group_column': column, 'group': group, 'Model': name,
                                     'n': len(rows), 'MAE': float(error.abs().mean()),
                                     'RMSE': float(np.sqrt(np.mean(error ** 2)))})
    pd.DataFrame(group_errors).to_csv(os.path.join(ANALYSIS_DIR, 'group_errors.csv'), index=False)

    # ======================== 6. 性能汇总 ========================
    logger.info("\n" + "=" * 70)
    logger.info("步骤6: 模型性能汇总")
    logger.info("=" * 70)

    performance_df = pd.DataFrame([
        {'Model': name, 'Train R²': res['r2_train'], 'Train RMSE': res['rmse_train'],
         'Train MAE': res['mae_train'], 'CV MSE (eV²)': res['cv_mse'], 'Test R²': res['r2_test'],
         'Test RMSE': res['rmse_test'], 'Test MAE': res['mae_test']}
        for name, res in results.items()
    ]).sort_values('CV MSE (eV²)', ascending=True)

    logger.info("\n" + performance_df.to_string(index=False))
    performance_df.to_csv(os.path.join(ANALYSIS_DIR, "model_performance_summary.csv"), index=False)

    # ======================== 7. 保存模型 ========================
    logger.info("\n" + "=" * 70)
    logger.info("步骤7: 保存模型")
    logger.info("=" * 70)

    for model_name, res in results.items():
        if res['model'] is not None:
            model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(res['model'], f)
            logger.info(f"  已保存: {model_path}")

    if target_scaler is not None:
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), 'wb') as f:
            pickle.dump(target_scaler, f)
    if feature_scaler_obj is not None:
        with open(os.path.join(MODEL_DIR, "feature_scaler.pkl"), 'wb') as f:
            pickle.dump(feature_scaler_obj, f)
    with open(os.path.join(MODEL_DIR, "filtered_feature_names.json"), 'w', encoding='utf-8') as f:
        json.dump(filtered_feature_names, f)

    # ======================== 8. 真实值 vs 预测值图 ========================
    logger.info("\n" + "=" * 70)
    logger.info("步骤8: 真实值 vs 预测值散点图")
    logger.info("=" * 70)

    y_test_plot = y_test_original

    for model_name in results.keys():
        if model_name not in results or results[model_name]['model'] is None:
            continue
        res = results[model_name]
        yp = res['predictions']

        # 密度着色散点（viridis），保留完整四边框
        fig, ax = plt.subplots(figsize=(3.5, 3.4))
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        xy = np.vstack([y_test_plot, yp])
        try:
            z = gaussian_kde(xy)(xy)
        except (ValueError, np.linalg.LinAlgError):
            # 小样本或常数预测会产生奇异协方差，仍绘制普通散点。
            z = np.ones(len(yp))
        order = z.argsort()
        ax.scatter(y_test_plot[order], yp[order], c=z[order], s=16, cmap='viridis',
                   edgecolors='none', linewidths=0, rasterized=True)
        mn, mx = min(y_test_plot.min(), yp.min()), max(y_test_plot.max(), yp.max())
        margin = (mx - mn) * 0.08
        ax.plot([mn-margin, mx+margin], [mn-margin, mx+margin],
                ls='--', color=COLOR_MID, linewidth=0.8, zorder=0)
        ax.set_xlim(mn-margin, mx+margin)
        ax.set_ylim(mn-margin, mx+margin)
        ax.text(0.03, 0.97, model_name, transform=ax.transAxes, fontsize=9,
                fontweight='bold', va='top', ha='left', color='black')
        mt = (f"$R^2$ = {res['r2_test']:.4f}\n"
              f"RMSE = {res['rmse_test']:.4f}\n"
              f"MAE = {res['mae_test']:.4f}")
        ax.text(0.97, 0.03, mt, transform=ax.transAxes, fontsize=FS_METRIC,
                va='bottom', ha='right', color=COLOR_DARK)
        ax.set_xlabel('DFT-Computed Value (eV)', fontsize=FS_AXIS_PARITY)
        ax.set_ylabel('ML-Predicted Value (eV)', fontsize=FS_AXIS_PARITY)
        ax.tick_params(labelsize=FS_TICK_PARITY)
        fig.tight_layout(pad=0.5)
        save_nature_fig(fig, os.path.join(ANALYSIS_DIR, f"{model_name}_true_vs_predicted"))
        logger.info(f"  ✓ {model_name}")

    # ======================== 9. SHAP分析 ========================
    logger.info("\n" + "=" * 70)
    logger.info("步骤9: SHAP分析")
    logger.info("=" * 70)

    import shap
    shap_main_dir = os.path.join(ANALYSIS_DIR, "shap_analysis")
    os.makedirs(shap_main_dir, exist_ok=True)

    sample_size = min(3 if args.smoke else 100, X_train_scaled.shape[0])
    if X_train_scaled.shape[0] > sample_size:
        indices = np.random.RandomState(42).choice(X_train_scaled.shape[0], sample_size, replace=False)
        X_sample = X_train_scaled[indices]
    else:
        indices = np.arange(len(X_train_scaled))
        X_sample = X_train_scaled

    # 模型输入标准化数值，图中展示原始描述符数值。
    X_sample_df = X_train_filtered.iloc[indices].reset_index(drop=True)
    shap_errors = {}

    np.random.seed(42)
    record_status('shap', best_model=best_model_name, completed_models=list(results))
    for model_name, res in results.items():
        if args.shap_mode == 'none' or (args.shap_mode == 'best' and model_name != best_model_name):
            continue
        logger.info(f"\n--- {model_name} SHAP分析 ---")
        model = res['model']
        model_shap_dir = os.path.join(shap_main_dir, model_name)
        os.makedirs(model_shap_dir, exist_ok=True)

        try:
            if model_name in ['RandomForest', 'XGBoost', 'DecisionTree', 'LightGBM']:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                except Exception:
                    try:
                        explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
                        shap_values = explainer.shap_values(X_sample)
                    except Exception as e2:
                        raise RuntimeError(f'TreeExplainer失败: {e2}') from e2
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                if len(shap_values.shape) == 1:
                    shap_values = shap_values.reshape(-1, 1)
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)) and not np.isscalar(expected_value):
                    expected_value = expected_value[0]

            elif hasattr(model, 'coef_'):
                try:
                    explainer = shap.LinearExplainer(model, X_sample)
                    shap_values = explainer.shap_values(X_sample)
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, (list, np.ndarray)) and not np.isscalar(expected_value):
                        expected_value = float(expected_value[0])
                except Exception as e:
                    raise RuntimeError(f'LinearExplainer失败: {e}') from e
            else:
                try:
                    # 优化: kmeans(25)背景 + nsamples=100 + 仅30样本
                    # (默认KernelExplainer在此特征维度下需数小时/模型)
                    bg = shap.kmeans(X_sample, min(25, len(X_sample)))
                    n_kernel = min(30, X_sample.shape[0])
                    explainer = shap.KernelExplainer(model.predict, bg)
                    shap_values = explainer.shap_values(X_sample[:n_kernel], nsamples=32 if args.smoke else 512, l1_reg='num_features(10)', silent=True)
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, (list, np.ndarray)) and not np.isscalar(expected_value):
                        expected_value = float(expected_value[0])
                except Exception as e:
                    raise RuntimeError(f'KernelExplainer失败: {e}') from e

            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(-1, 1)

            shap_values, expected_value = shap_to_ev(shap_values, expected_value, target_scaler)
            np.save(os.path.join(model_shap_dir, "shap_values.npy"), shap_values)
            X_sample_df.iloc[:len(shap_values)].to_csv(os.path.join(model_shap_dir, 'samples.csv'), index=False)
            with open(os.path.join(model_shap_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump({'unit': 'eV', 'expected_value': expected_value,
                           'source_rows': data_valid.iloc[train_idx[indices[:len(shap_values)]]]['source_row'].tolist()}, f)

            # SHAP行数与绘图DataFrame对齐 (核模型仅计算前30个样本)
            X_plot_df = X_sample_df.iloc[:shap_values.shape[0]]

            # 特征重要性表（bar 图与 CSV 共用，避免重复计算）
            imp_df = pd.DataFrame({
                'Feature': X_plot_df.columns,
                'Mean_ABS_SHAP': np.abs(shap_values).mean(axis=0)
            }).sort_values('Mean_ABS_SHAP', ascending=False)

            # Bar plot（Nature 风格：横向条形、无图内标题）
            fi = imp_df.head(20).iloc[::-1]
            fig, ax = plt.subplots(figsize=(3.3, 3.3))
            ax.barh(range(len(fi)), fi['Mean_ABS_SHAP'].values, color=COLOR_BLUE, height=0.7)
            ax.set_yticks(range(len(fi)))
            ax.set_yticklabels(fi['Feature'].values, fontsize=FS_TICK_SMALL)
            ax.set_xlabel('Mean |SHAP value| (eV)', fontsize=FS_LABEL_SMALL)
            ax.tick_params(axis='x', labelsize=FS_TICK_SMALL)
            ax.spines['left'].set_visible(True)
            fig.tight_layout(pad=0.5)
            save_nature_fig(fig, os.path.join(model_shap_dir, "shap_bar"))

            # Summary beeswarm（Nature 风格：无图内标题）
            plt.figure(figsize=(3.6, 3.3))
            shap.summary_plot(shap_values, X_plot_df, show=False, max_display=20)
            plt.gca().set_xlabel('SHAP value (eV)')
            plt.gca().tick_params(labelsize=FS_TICK_SMALL)
            plt.tight_layout(pad=0.5)
            save_nature_fig(plt.gcf(), os.path.join(model_shap_dir, "shap_scatter"))

            # Dependence plots (top 3)
            mean_abs = np.abs(shap_values).mean(axis=0)
            top3 = np.argsort(mean_abs)[::-1][:3]
            for idx in top3:
                fname = X_plot_df.columns[idx]
                try:
                    plt.figure(figsize=(3.3, 3.0))
                    shap.dependence_plot(idx, shap_values, X_plot_df, show=False)
                    plt.gca().set_ylabel(f'SHAP value for {fname} (eV)')
                    plt.gca().tick_params(labelsize=FS_TICK_SMALL)
                    plt.tight_layout(pad=0.5)
                    save_nature_fig(plt.gcf(), os.path.join(model_shap_dir, f"shap_dependence_{fname}"))
                except Exception as e:
                    plt.close('all')
                    logger.warning(f"  依赖图失败: {e}")

            # Force plot
            try:
                shap.force_plot(expected_value, shap_values[0], X_plot_df.iloc[0],
                              show=False, matplotlib=True, out_names='Gap (eV)')
                plt.tight_layout(pad=0.5)
                save_nature_fig(plt.gcf(), os.path.join(model_shap_dir, "shap_force_1"))
            except Exception as e:
                plt.close('all')
                logger.warning(f"  Force plot失败: {e}")

            # Feature importance CSV
            imp_df.to_csv(os.path.join(model_shap_dir, "feature_importance.csv"), index=False)

            logger.info(f"  {model_name} SHAP完成 ✓")

        except Exception as e:
            shap_errors[model_name] = str(e)
            logger.error(f"  {model_name} SHAP失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 部分 SHAP 绘图函数会自行新建 Figure，统一关闭以免长流程积累。
            plt.close('all')

    # ======================== 10. 特征相关性 ========================
    logger.info("\n" + "=" * 70)
    logger.info("步骤10: 特征相关性分析")
    logger.info("=" * 70)

    corr_target = target_correlations(X_train_filtered, y_train_original).sort_values(ascending=False)
    corr_target.to_csv(os.path.join(ANALYSIS_DIR, "feature_target_correlation.csv"))

    # 特征-目标相关性条形图（Nature 风格：横向、正负分色）
    corr_sorted = corr_target.loc[corr_target.abs().nlargest(20).index].sort_values()
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    ax.barh(range(len(corr_sorted)), corr_sorted.values,
            color=[COLOR_RED if v < 0 else COLOR_BLUE for v in corr_sorted.values], height=0.7)
    ax.set_yticks(range(len(corr_sorted)))
    ax.set_yticklabels(corr_sorted.index.values, fontsize=FS_TICK_SMALL)
    ax.set_xlabel('Pearson correlation with target', fontsize=FS_LABEL_SMALL)
    ax.tick_params(axis='x', labelsize=FS_TICK_SMALL)
    ax.spines['left'].set_visible(True)
    ax.axvline(0, color=COLOR_MID, linewidth=0.6)
    fig.tight_layout(pad=0.5)
    save_nature_fig(fig, os.path.join(ANALYSIS_DIR, "feature_target_correlation"))

    # Top20 特征相关性热图（Nature 风格：无标注、RdBu_r 归一化色标）
    top20 = corr_target.abs().sort_values(ascending=False).head(20).index.tolist()
    cm = X_train_filtered[top20].corr()
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(top20)))
    ax.set_yticks(range(len(top20)))
    ax.set_xticklabels(top20, rotation=90, fontsize=FS_TICK_TINY)
    ax.set_yticklabels(top20, fontsize=FS_TICK_TINY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FS_TICK_TINY)
    fig.tight_layout(pad=0.5)
    save_nature_fig(fig, os.path.join(ANALYSIS_DIR, "feature_correlation_heatmap"))
    logger.info("  ✓ 特征相关性分析完成")

    # ======================== 11. 分析报告 ========================
    logger.info("\n" + "=" * 70)
    logger.info("步骤11: 生成分析报告")
    logger.info("=" * 70)

    report_path = os.path.join(ANALYSIS_DIR, "analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# C_number ∈ {{0,1,2}} 数据分析报告\n\n")
        f.write("**冒烟验证：不用于性能结论。**\n\n" if args.smoke else "")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 1. 数据概览\n\n")
        f.write(f"- 数据文件: `{PATH}`\n")
        f.write(f"- 过滤条件: C_number ∈ {C_NUMBER_VALUES}\n")
        f.write(f"- 过滤后样本数: {len(data_valid)}\n")
        f.write(f"- 原始特征数: {len(X.columns)}\n")
        f.write(f"- 筛选后特征数: {len(X_train_filtered.columns)}\n")
        f.write(f"- 训练集: {X_train_scaled.shape[0]}, 测试集: {X_test_scaled.shape[0]}\n")
        f.write(f"- 超参数优化轮次: {HyPer_epoch}；折内预处理，五折 CV\n\n")
        f.write(f"- SHAP 单位: eV；划分方式: {args.group_column or '随机划分'}\n\n")
        f.write(f"- 特征: {args.feature_mode}；移除指纹: {args.drop_fingerprint}；仅加指纹: {args.only_fingerprint}；Spin: 不使用\n\n")

        f.write(f"## 2. C_number 分布\n\n")
        f.write(f"| C_number | 样本数 |\n|---|---|\n")
        for val, cnt in c_number_counts.items():
            f.write(f"| {val} | {cnt} |\n")
        f.write("\n")

        f.write(f"## 3. 模型性能\n\n")
        f.write(f"| 模型 | CV MSE (eV²) | 训练R² | 训练RMSE | 训练MAE | 测试R² | 测试RMSE | 测试MAE |\n")
        f.write(f"|------|--------|--------|----------|---------|--------|----------|--------|\n")
        for _, row in performance_df.iterrows():
            f.write(f"| {row['Model']} | {row['CV MSE (eV²)']:.4f} | {row['Train R²']:.4f} | {row['Train RMSE']:.4f} | {row['Train MAE']:.4f} | {row['Test R²']:.4f} | {row['Test RMSE']:.4f} | {row['Test MAE']:.4f} |\n")
        f.write("\n")

        f.write(f"**按五折 CV 选择的模型: {best_model_name}**\n\n")

        f.write(f"## 4. 超参数\n\n")
        for mn, res in results.items():
            if res['best_params']:
                f.write(f"- **{mn}**: {res['best_params']}\n")
        f.write("\n")

        f.write(f"## 5. 文件清单\n\n")
        f.write(f"- 模型: `{MODEL_DIR}/`\n")
        f.write(f"- 分析结果: `{ANALYSIS_DIR}/`\n")
        f.write(f"- SHAP分析: `{ANALYSIS_DIR}/shap_analysis/`\n")

    logger.info(f"报告已保存: {report_path}")

    # ======================== 完成 ========================
    logger.info("\n" + "=" * 70)
    if shap_errors:
        with open(os.path.join(ANALYSIS_DIR, 'shap_errors.json'), 'w', encoding='utf-8') as f:
            json.dump(shap_errors, f, ensure_ascii=False, indent=2)
        raise RuntimeError(f'SHAP 未完整完成: {list(shap_errors)}；训练结果已保存')
    record_status('complete', best_model=best_model_name, completed_models=list(results))
    logger.info('冒烟流程完成（不代表正式模型性能）' if args.smoke else '流水线完成!')
    logger.info(f"模型: {MODEL_DIR}")
    logger.info(f"分析结果: {ANALYSIS_DIR}")
    logger.info("=" * 70)

    print("\n" + "=" * 70)
    print("模型结果 (C_number ∈ {0,1,2}, 按五折 CV MSE排序):")
    print("=" * 70)
    for i, (_, row) in enumerate(performance_df.iterrows(), 1):
        print(f"{i:2d}. {row['Model']:<18s} Test R²={row['Test R²']:.4f}  RMSE={row['Test RMSE']:.4f}  MAE={row['Test MAE']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
