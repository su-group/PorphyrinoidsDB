"""简单函数实现筛选、五折调参和评估；预处理只拟合当前训练部分。"""
import json
import sys
from pathlib import Path

# 支持直接运行这个历史入口；导入时不会配置日志或写文件。
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from core.data_processing import standardize_smiles


def load_hyperparameter_space():
    path = Path(__file__).resolve().parents[1] / 'config/hyperparameter_optimization.json'
    return json.loads(path.read_text(encoding='utf-8-sig'))


def create_models(smoke=False):
    from sklearn.linear_model import Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    return {
        'Lasso': Lasso(random_state=42),
        'ElasticNet': ElasticNet(random_state=42),
        'SVR': SVR(),
        'KernelRidge': KernelRidge(kernel='rbf'),
        'KNeighbors': KNeighborsRegressor(),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=1),
        'XGBoost': XGBRegressor(random_state=42, n_jobs=1),
        'LightGBM': LGBMRegressor(random_state=42, verbosity=-1, n_jobs=1, subsample_freq=1),
        'GaussianProcess': GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(1.0), random_state=42,
            n_restarts_optimizer=0 if smoke else 2),
    }


def target_correlations(X, y):
    """y 按行位置传入，显式绑定 X 索引，防止 corrwith 自动错误对齐。"""
    if len(X) != len(y):
        raise ValueError('特征与目标行数不一致')
    result = pd.Series(np.nan, index=X.columns, dtype=float)
    variable = X.columns[X.nunique(dropna=True) > 1]
    if len(variable) and np.std(y) > 0:
        result.loc[variable] = X[variable].corrwith(pd.Series(np.asarray(y), index=X.index))
    return result


def fit_preprocessing(X, y, min_features=50):
    """只从当前训练折学习填充值、特征列表、X/y 标准化参数。"""
    # 本轮 Y_number 恒为 1，Spin 不再进入任何训练或筛选分支。
    X = X.loc[:, [c for c in X if not str(c).startswith('Spin') and c != 'Y_number']]
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median().dropna()  # 全缺失列在当前训练折内移除。
    clean = X[medians.index].fillna(medians)
    molecular = list(clean.columns)
    variances = clean[molecular].var(ddof=0)
    nonconstant = variances[variances > 1e-10].index.tolist()
    variable = variances[variances > 0.01].index.tolist()
    candidates = variable if len(variable) >= min_features else nonconstant
    if candidates:
        # 按目标相关性排序，每次仅和已保留列比较，避免链式误删。
        relevance = target_correlations(clean[candidates], y).abs().fillna(0)
        ranked = relevance.sort_values(ascending=False, kind='stable').index.tolist()
        # 填补后是稠密有限数值，用矩阵运算避免 pandas 逐对循环的大量开销。
        correlation = np.abs(np.atleast_2d(np.corrcoef(clean[candidates].to_numpy(), rowvar=False)))
        column_positions = {name: i for i, name in enumerate(candidates)}
        kept = []
        for name in ranked:
            if not kept or not (correlation[column_positions[name], [column_positions[c] for c in kept]] > 0.95).any():
                kept.append(name)
        # 样本/特征少时不强行制造 50 个特征，也不恢复常量列。
        if len(kept) < min(min_features, len(candidates)):
            kept = ranked[:min(min_features, len(candidates))]
        names = [c for c in candidates if c in kept]
    else:
        names = []
    if not names:
        raise ValueError('当前训练折无可用特征')
    feature_scaler = StandardScaler().fit(clean[names])
    target_scaler = StandardScaler().fit(np.asarray(y).reshape(-1, 1))
    return {'feature_names': names, 'medians': medians[names],
            'feature_scaler': feature_scaler, 'target_scaler': target_scaler}


def clean_selected_features(X, preprocessing):
    names = preprocessing['feature_names']
    return X[names].replace([np.inf, -np.inf], np.nan).fillna(preprocessing['medians'])


def transform_features(X, preprocessing):
    return preprocessing['feature_scaler'].transform(clean_selected_features(X, preprocessing))


def prepare_cv_folds(X, y, groups=None, fold_indices=None):
    """固定五折；折内预处理与 trial 无关，计算一次供各模型复用。"""
    if groups is None:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    else:
        if len(np.unique(groups)) < 5:
            raise ValueError('分组五折交叉验证至少需要 5 个训练组')
        splitter = GroupKFold(n_splits=5)
    y = np.asarray(y)
    folds = []
    indices = splitter.split(X, y, groups) if fold_indices is None else [
        (np.asarray(f['train_positions']), np.asarray(f['valid_positions'])) for f in fold_indices]
    for train_idx, valid_idx in indices:
        prep = fit_preprocessing(X.iloc[train_idx], y[train_idx])
        folds.append({
            'X_train': transform_features(X.iloc[train_idx], prep),
            'X_valid': transform_features(X.iloc[valid_idx], prep),
            'y_train': prep['target_scaler'].transform(y[train_idx, None]).ravel(),
            'y_valid': y[valid_idx], 'target_scaler': prep['target_scaler'],
            'preprocessing': prep, 'train_positions': train_idx,
            'valid_positions': valid_idx,
        })
    return folds


def _fold_cv_loss(fold, params, model):
    """单折 CV 损失；放在模块级以便 joblib/loky 进程池可以序列化调用。"""
    candidate = clone(model).set_params(**params)
    candidate.fit(fold['X_train'], fold['y_train'])
    prediction = candidate.predict(fold['X_valid'])
    prediction = fold['target_scaler'].inverse_transform(prediction.reshape(-1, 1)).ravel()
    return mean_squared_error(fold['y_valid'], prediction)


def optimize_model_parameters(model_name, model, folds, n_trials=100, smoke=False, study_path=None,
                              fold_jobs=1):
    import optuna
    from optuna.samplers import TPESampler

    config = load_hyperparameter_space()[model_name]

    def objective(trial):
        params = {}
        for name, spec in config.items():
            if spec['type'] == 'float':
                params[name] = trial.suggest_float(name, spec['low'], spec['high'],
                                                   log=spec.get('log', False))
            elif spec['type'] == 'int':
                high = spec['high']
                if smoke and name == 'n_estimators':
                    high = min(high, 10)
                if name == 'n_neighbors':
                    high = min(high, min(len(f['y_train']) for f in folds))
                params[name] = trial.suggest_int(name, spec['low'], high)
            elif spec['type'] == 'categorical':
                params[name] = trial.suggest_categorical(name, spec['choices'])
            else:
                raise ValueError(f'未知参数类型: {spec}')
        # 五折彼此独立，fold_jobs>1 时并行评估；每折都在自己的克隆上拟合，结果与串行一致。
        if fold_jobs > 1 and len(folds) > 1:
            from joblib import Parallel, delayed
            losses = Parallel(n_jobs=min(fold_jobs, len(folds)))(
                delayed(_fold_cv_loss)(fold, params, model) for fold in folds)
        else:
            losses = [_fold_cv_loss(fold, params, model) for fold in folds]
        score = float(np.mean(losses))
        if not np.isfinite(score):
            raise ValueError(f'{model_name} 的 CV 损失不是有限数')
        return score

    sampler = TPESampler(seed=42)
    storage = None
    storage_backend = None
    if study_path is not None:
        import pickle
        study_path = Path(study_path).resolve()
        sampler_path = study_path.with_suffix('.sampler.pkl')
        storage_backend = optuna.storages.RDBStorage('sqlite:///' + study_path.as_posix())
        storage = storage_backend
        if sampler_path.exists():
            with open(sampler_path, 'rb') as stream:
                sampler = pickle.load(stream)
    study = optuna.create_study(direction='minimize', sampler=sampler, storage=storage,
                                study_name=model_name, load_if_exists=True)
    # 进程被中断时 RUNNING trial 不可续跑，将其明确标记为失败后继续。
    for trial in study.get_trials(deepcopy=False):
        if trial.state == optuna.trial.TrialState.RUNNING:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)

    def save_progress(study, trial):
        if study_path is not None:
            temp = sampler_path.with_suffix('.tmp')
            with open(temp, 'wb') as stream:
                pickle.dump(study.sampler, stream)
            temp.replace(sampler_path)
            complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
            progress = {'model': model_name, 'finished_trials': len(study.trials),
                        'complete_trials': complete, 'budget': n_trials,
                        'last_trial': trial.number, 'last_state': trial.state.name}
            progress_path = study_path.with_suffix('.progress.json')
            temp_progress = study_path.with_suffix('.progress.tmp')
            temp_progress.write_text(json.dumps(progress), encoding='utf-8')
            temp_progress.replace(progress_path)

    remaining = max(0, n_trials - len(study.trials))
    try:
        study.optimize(objective, n_trials=remaining, callbacks=[save_progress],
                       catch=(ValueError, np.linalg.LinAlgError))
        return study.best_params, float(study.best_value)
    finally:
        # Windows 上显式释放 SQLite 连接，避免恢复/清理时文件被占用。
        if storage_backend is not None:
            storage_backend.remove_session()
            storage_backend.engine.dispose()


def evaluate_model(model, X_train, X_test, y_train, y_test, target_scaler):
    result = {'model': model}
    for part, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        pred = target_scaler.inverse_transform(model.predict(X).reshape(-1, 1)).ravel()
        result['r2_' + part] = float(r2_score(y, pred))
        result['rmse_' + part] = float(np.sqrt(mean_squared_error(y, pred)))
        result['mae_' + part] = float(mean_absolute_error(y, pred))
        if part == 'test':
            result['predictions'] = pred
    return result


def shap_to_ev(values, expected_value, target_scaler):
    """线性反标准化：贡献乘标准差，基线另加均值。"""
    scale = float(target_scaler.scale_[0])
    mean = float(target_scaler.mean_[0])
    return np.asarray(values) * scale, float(np.asarray(expected_value).ravel()[0]) * scale + mean


if __name__ == '__main__':
    # 历史入口转到统一主流程，避免两套实现继续分歧。
    from run_pipeline import main
    main()
