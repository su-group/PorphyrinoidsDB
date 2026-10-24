r"""Leave-one-B-group-out on the meso axis, to explain the CV-vs-test gap.

The B axis holds out meso substituents.  The pipeline's five CV folds hold out subsets of
the *training* groups (B0/B6, B2/B4/B14, ...), while the single test split holds out
{B5, B11, B15}.  If some groups are intrinsically harder to extrapolate to, then the CV
number and the test number estimate different quantities, and the gap is a property of
which groups ended up on which side rather than a bug.

This script fits Lasso and RandomForest leave-one-group-out inside the training subset and
then scores the test groups with the model trained on everything -- same preprocessing code
path as the pipeline (`core.train_optimizer.fit_preprocessing`), so the numbers are directly
comparable to the published CV MSE.

Usage: python -B .dsh_tmp\bgroup_difficulty.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

from core.data_processing import load_feature_cache, load_valid_data
from core.experiment_setup import load_split_manifest, select_experiment_features
from core.train_optimizer import fit_preprocessing, transform_features

TARGET = 'Gap(eV)'
CACHE = ROOT / 'data' / 'Data-20261012_raw_features.csv'
DATA = ROOT / 'data' / 'Data-20261012.csv'
MANIFEST = ROOT / 'experiments' / '20261021_splits_meso.json'
MODEL_DIR = ROOT / 'experiments' / '20261021_B-M' / 'models' / 'B-M'


def best_params(model_name: str) -> dict:
    payload = pickle.loads((MODEL_DIR / f'{model_name}.result.pkl').read_bytes())
    return payload['best_params']


def evaluate(model_name: str, params: dict) -> None:
    data = load_valid_data(str(DATA))
    y_all = data[TARGET].to_numpy(dtype=float)
    groups = data['B_number'].to_numpy()
    cache, _ = load_feature_cache(str(CACHE), data, str(DATA))
    matrix = select_experiment_features(cache, 'mordred')
    train_pos, test_pos, _ = load_split_manifest(str(MANIFEST), data, str(DATA), 'meso')
    train_pos = np.asarray(train_pos)
    test_pos = np.asarray(test_pos)
    print(f'\n=== {model_name} params={params} ===')
    print(f'feature matrix {matrix.shape}, train {len(train_pos)}, test {len(test_pos)}')

    def make() -> object:
        if model_name == 'Lasso':
            usable = {k: v for k, v in params.items() if k in {'alpha'}}
            return Lasso(**usable, max_iter=50000, random_state=42)
        usable = {k: v for k, v in params.items() if k in {
            'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'}}
        return RandomForestRegressor(**usable, random_state=42, n_jobs=4)

    def fit_predict(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
        preprocessing = fit_preprocessing(matrix.loc[inner], y_all[inner], min_features=50)
        xs = transform_features(matrix.loc[inner], preprocessing)
        xo = transform_features(matrix.loc[outer], preprocessing)
        model = make()
        model.fit(xs, preprocessing['target_scaler'].transform(
            y_all[inner].reshape(-1, 1)).ravel())
        return preprocessing['target_scaler'].inverse_transform(
            model.predict(xo).reshape(-1, 1)).ravel()

    rows = []
    for group in sorted(set(groups[train_pos].tolist())):
        valid = train_pos[groups[train_pos] == group]
        inner = train_pos[groups[train_pos] != group]
        pred = fit_predict(inner, valid)
        rows.append({'group': int(group), 'role': 'LOSO-train', 'n': len(valid),
                     'mse': float(np.mean((y_all[valid] - pred) ** 2)),
                     'sd': float(np.std(y_all[valid], ddof=1))})

    for group in sorted(set(groups[test_pos].tolist())):
        valid = test_pos[groups[test_pos] == group]
        pred = fit_predict(train_pos, valid)
        rows.append({'group': int(group), 'role': 'test', 'n': len(valid),
                     'mse': float(np.mean((y_all[valid] - pred) ** 2)),
                     'sd': float(np.std(y_all[valid], ddof=1))})

    frame = pd.DataFrame(rows)
    train_rows = frame[frame['role'] == 'LOSO-train']
    test_rows = frame[frame['role'] == 'test']
    print(frame.to_string(index=False, float_format=lambda v: f'{v:7.4f}'))
    print(f'LOSO mean MSE over {len(train_rows)} train groups = {train_rows["mse"].mean():.4f}')
    print(f'MSE on the 3 test groups (unweighted)      = {test_rows["mse"].mean():.4f}')
    print(f'MSE on the 3 test groups (n-weighted)      = '
          f'{float(np.average(test_rows["mse"], weights=test_rows["n"])):.4f}')
    hardest = frame.sort_values('mse', ascending=False)['group'].tolist()
    print(f'groups ordered hardest -> easiest: {hardest}')
    print(f'rank of test groups: '
          f'{[(int(g), hardest.index(int(g)) + 1) for g in test_rows["group"]]}')


def main() -> int:
    evaluate('Lasso', best_params('Lasso'))
    evaluate('RandomForest', best_params('RandomForest'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
