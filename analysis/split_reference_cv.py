r"""Same-protocol reference CVs for the three split axes.

`baxis_lobo_cv.py` produces the meso-axis number.  To compare it against the random and
scaffold axes without the confound of a different script, the same preprocessing path,
feature matrix, models and hyperparameters are used here for:

  random    -- plain 5-fold KFold over rows (the interpolation ceiling)
  scaffold  -- 5-fold GroupKFold on A_number (the extrapolation floor)

Both use the tuned B-M hyperparameters on purpose: this table is about the *split axis*,
so the model must be held fixed.  Pooled MSE over the union of held-out rows is the
primary figure (ablation convention), with the mean of per-fold MSEs also reported.

Usage: python -B .dsh_tmp\split_reference_cv.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import GroupKFold, KFold

from core.data_processing import load_feature_cache, load_valid_data
from core.experiment_setup import select_experiment_features
from core.train_optimizer import create_models, fit_preprocessing, transform_features

TARGET = 'Gap(eV)'
CACHE = ROOT / 'data' / 'Data-20261012_raw_features.csv'
DATA = ROOT / 'data' / 'Data-20261012.csv'
MODEL_DIR = ROOT / 'experiments' / '20261021_B-M' / 'models' / 'B-M'
OUT_JSON = ROOT / '.dsh_tmp' / 'split_reference_cv.json'
SEED = 42


def tuned_params(model_name: str) -> dict:
    payload = pickle.loads((MODEL_DIR / f'{model_name}.result.pkl').read_bytes())
    return dict(payload['best_params'])


def build_model(model_name: str, params: dict):
    prototype = create_models(smoke=False)[model_name]
    usable = {k: v for k, v in params.items() if k in prototype.get_params()}
    prototype.set_params(**usable)
    return prototype


def main() -> int:
    data = load_valid_data(str(DATA))
    y_all = data[TARGET].to_numpy(dtype=float)
    groups_a = data['A_number'].astype(int).to_numpy()
    cache, _ = load_feature_cache(str(CACHE), data, str(DATA))
    matrix = select_experiment_features(cache, 'mordred')
    models = ['Lasso', 'RandomForest', 'XGBoost', 'KNeighbors']
    var_all = float(np.var(y_all))
    print(f'data {matrix.shape[0]} rows, {matrix.shape[1]} columns, variance {var_all:.4f}')

    splitters = {
        'random': list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(matrix)),
        'scaffold': list(GroupKFold(n_splits=5).split(matrix, groups=groups_a)),
    }

    report = {'seed': SEED, 'models': {}}
    for name in models:
        params = tuned_params(name)
        report['models'][name] = {'params': params, 'axes': {}}
        print(f'\n=== {name} ===')
        for axis, splits in splitters.items():
            pooled_true, pooled_pred, fold_mses, fold_ns = [], [], [], []
            for train_idx, valid_idx in splits:
                preprocessing = fit_preprocessing(matrix.iloc[train_idx], y_all[train_idx],
                                                  min_features=50)
                xs = transform_features(matrix.iloc[train_idx], preprocessing)
                xo = transform_features(matrix.iloc[valid_idx], preprocessing)
                model = build_model(name, params)
                scaled = preprocessing['target_scaler'].transform(
                    y_all[train_idx].reshape(-1, 1)).ravel()
                model.fit(xs, scaled)
                pred = preprocessing['target_scaler'].inverse_transform(
                    model.predict(xo).reshape(-1, 1)).ravel()
                pooled_true.append(y_all[valid_idx])
                pooled_pred.append(pred)
                fold_mses.append(float(np.mean((y_all[valid_idx] - pred) ** 2)))
                fold_ns.append(len(valid_idx))
            pooled = float(np.mean((np.concatenate(pooled_true) - np.concatenate(pooled_pred)) ** 2))
            report['models'][name]['axes'][axis] = {
                'pooled_mse': pooled,
                'pooled_r2': 1.0 - pooled / var_all,
                'fold_mse': fold_mses,
                'mean_of_fold_mse': float(np.mean(fold_mses)),
                'n_weighted_mse': float(np.average(fold_mses, weights=fold_ns)),
                'fold_n': fold_ns,
            }
            print(f'  {axis:9s} pooled MSE={pooled:.4f} (R2={1.0 - pooled / var_all:+.4f})  '
                  f'folds={[round(m, 3) for m in fold_mses]}')

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[written] {OUT_JSON}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
