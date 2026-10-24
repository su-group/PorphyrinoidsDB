r"""Leave-one-B-group-out cross-validation on the meso axis.

Why this exists: the formal B-axis experiment (`experiments\20261021_B-M*`) holds out a
single fixed set of meso groups {B5, B11, B15}, and those three happen to be the *easiest*
in the axis.  A single split therefore reports an optimistic number.  This script removes
that luck: it re-uses all 16 meso groups, partitions the 15 substituted groups (B1..B15)
into five folds of three, and always trains on the rest plus the unsubstituted parent B0.
The result is a five-fold estimate of "how well does a model place a *new* meso substituent",
where every group plays the validation role exactly once.

B0 is reported separately, because it is not a substituted group at all: it is the parent
porphyrin with no meso substituent, and A13/A14 only ever use it.  Its evaluation trains on
all 15 substituted groups and predicts B0 -- that is the one genuine extrapolation inside
this axis.

Protocol, for comparability: same preprocessing code path as the pipeline
(`core.train_optimizer.fit_preprocessing` / `transform_features`), features from the
Mordred-only cache, hyperparameters taken from the tuned B-M study.  Metrics are pooled
over the held-out rows (the ablation convention), not the mean of per-fold MSEs; both are
printed.

Usage: python -B .dsh_tmp\baxis_lobo_cv.py [--models Lasso,RandomForest,XGBoost,KNeighbors]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data_processing import load_feature_cache, load_valid_data
from core.experiment_setup import select_experiment_features
from core.train_optimizer import create_models, fit_preprocessing, transform_features

TARGET = 'Gap(eV)'
CACHE = ROOT / 'data' / 'Data-20261012_raw_features.csv'
DATA = ROOT / 'data' / 'Data-20261012.csv'
MODEL_DIR = ROOT / 'experiments' / '20261021_B-M' / 'models' / 'B-M'
OUT_JSON = ROOT / '.dsh_tmp' / 'baxis_lobo_cv.json'
N_FOLDS = 5
SEED = 42


def tuned_params(model_name: str) -> dict:
    payload = pickle.loads((MODEL_DIR / f'{model_name}.result.pkl').read_bytes())
    return dict(payload['best_params'])


def build_model(model_name: str, params: dict):
    prototype = create_models(smoke=False)[model_name]
    usable = {k: v for k, v in params.items() if k in prototype.get_params()}
    prototype.set_params(**usable)
    return prototype, usable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='Lasso,RandomForest,XGBoost,KNeighbors')
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(',') if m.strip()]

    data = load_valid_data(str(DATA))
    y_all = data[TARGET].to_numpy(dtype=float)
    groups = data['B_number'].astype(int).to_numpy()
    scaffolds = data['A_number'].astype(int).to_numpy()
    cache, _ = load_feature_cache(str(CACHE), data, str(DATA))
    matrix = select_experiment_features(cache, 'mordred')
    print(f'data {matrix.shape[0]} rows, feature matrix {matrix.shape[1]} columns')

    substituted = sorted({int(g) for g in groups if int(g) != 0})
    rng = np.random.RandomState(SEED)
    shuffled = list(substituted)
    rng.shuffle(shuffled)
    folds = [sorted(shuffled[i::N_FOLDS]) for i in range(N_FOLDS)]
    print(f'substituted groups ({len(substituted)}): {substituted}')
    for index, fold in enumerate(folds):
        print(f'  fold{index}: hold out B{fold} '
              f'(n={sum(int((groups == g).sum()) for g in fold)})')

    def fit_predict(inner_rows: np.ndarray, outer_rows: np.ndarray, params: dict, name: str):
        preprocessing = fit_preprocessing(matrix.loc[inner_rows], y_all[inner_rows], min_features=50)
        xs = transform_features(matrix.loc[inner_rows], preprocessing)
        xo = transform_features(matrix.loc[outer_rows], preprocessing)
        model, _ = build_model(name, params)
        scaled = preprocessing['target_scaler'].transform(y_all[inner_rows].reshape(-1, 1)).ravel()
        model.fit(xs, scaled)
        pred = model.predict(xo).reshape(-1, 1)
        return preprocessing['target_scaler'].inverse_transform(pred).ravel()

    report = {'seed': SEED, 'n_folds': N_FOLDS, 'folds': folds, 'models': {}}
    var_all = float(np.var(y_all))

    for name in models:
        params = tuned_params(name)
        print(f'\n=== {name} params={params} ===')
        fold_rows, per_group = [], []
        pooled_true, pooled_pred = [], []
        for index, fold in enumerate(folds):
            mask_valid = np.isin(groups, fold)
            rows_valid = np.flatnonzero(mask_valid)
            rows_train = np.flatnonzero(~mask_valid)
            pred = fit_predict(rows_train, rows_valid, params, name)
            mse = float(np.mean((y_all[rows_valid] - pred) ** 2))
            r2 = 1.0 - mse / float(np.var(y_all[rows_valid]))
            fold_rows.append({'fold': index, 'groups': fold, 'n': len(rows_valid),
                              'mse': mse, 'r2': r2,
                              'valid_mean': float(np.mean(y_all[rows_valid])),
                              'valid_sd': float(np.std(y_all[rows_valid], ddof=1)),
                              'const_mse': float(np.mean(
                                  (y_all[rows_valid] - np.mean(y_all[rows_train])) ** 2))})
            pooled_true.append(y_all[rows_valid])
            pooled_pred.append(pred)
            for group in fold:
                sel = groups[rows_valid] == group
                per_group.append({'group': group, 'n': int(sel.sum()),
                                  'mse': float(np.mean((y_all[rows_valid][sel] - pred[sel]) ** 2))})
            print(f'  fold{index} B{fold}: n={len(rows_valid)} MSE={mse:.4f} R2={r2:+.4f} '
                  f'(const {fold_rows[-1]["const_mse"]:.4f})')

        pooled = float(np.mean((np.concatenate(pooled_true) - np.concatenate(pooled_pred)) ** 2))
        mean_of_folds = float(np.mean([r['mse'] for r in fold_rows]))
        weighted = float(np.average([r['mse'] for r in fold_rows],
                                    weights=[r['n'] for r in fold_rows]))

        rows_nonzero = np.flatnonzero(groups != 0)
        rows_zero = np.flatnonzero(groups == 0)
        pred_b0 = fit_predict(rows_nonzero, rows_zero, params, name)
        mse_b0 = float(np.mean((y_all[rows_zero] - pred_b0) ** 2))
        r2_b0 = 1.0 - mse_b0 / float(np.var(y_all[rows_zero]))
        const_b0 = float(np.mean((y_all[rows_zero] - np.mean(y_all[rows_nonzero])) ** 2))
        # B0 also contains scaffolds that never appear among substituted groups (A13/A14 only
        # use the parent), which injects scaffold extrapolation into this number.  Report the
        # error both with and without those scaffolds.
        scaffolds_zero = set(scaffolds[rows_zero].tolist())
        unseen = scaffolds_zero - set(scaffolds[rows_nonzero].tolist())
        keep = ~np.isin(scaffolds[rows_zero], sorted(unseen))
        mse_b0_seen = float(np.mean((y_all[rows_zero][keep] - pred_b0[keep]) ** 2))
        print(f'  B0 extrapolation (train on 15 substituted groups): n={len(rows_zero)} '
              f'MSE={mse_b0:.4f} R2={r2_b0:+.4f} (const {const_b0:.4f})  '
              f'bias={float(np.mean(pred_b0 - y_all[rows_zero])):+.2f} eV  '
              f'pred={float(np.mean(pred_b0)):.2f} vs true={float(np.mean(y_all[rows_zero])):.2f}')
        print(f'     excluding unseen scaffolds {sorted(unseen)} (n={int(keep.sum())}): '
              f'MSE={mse_b0_seen:.4f}')

        per_group_frame = pd.DataFrame(per_group).sort_values('mse', ascending=False)
        print('  per-group MSE (hardest first): '
              + ', '.join(f"B{int(r.group)}={r.mse:.3f}" for r in per_group_frame.itertuples()))

        report['models'][name] = {
            'params': params,
            'folds': fold_rows,
            'pooled_mse': pooled,
            'pooled_r2': 1.0 - pooled / var_all,
            'mean_of_fold_mse': mean_of_folds,
            'n_weighted_mse': weighted,
            'per_group': per_group,
            'b0': {'n': int(len(rows_zero)), 'mse': mse_b0, 'r2': r2_b0,
                   'const_mse': const_b0,
                   'bias': float(np.mean(pred_b0 - y_all[rows_zero])),
                   'pred_mean': float(np.mean(pred_b0)),
                   'true_mean': float(np.mean(y_all[rows_zero])),
                   'true_sd': float(np.std(y_all[rows_zero], ddof=1)),
                   'unseen_scaffolds': sorted(int(s) for s in unseen),
                   'n_without_unseen': int(keep.sum()),
                   'mse_without_unseen': mse_b0_seen},
        }
        print(f'  --> pooled MSE={pooled:.4f} (R2={1.0 - pooled / var_all:+.4f}), '
              f'mean-of-folds={mean_of_folds:.4f}, n-weighted={weighted:.4f}')

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[written] {OUT_JSON}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
