r"""Second pass of the meso-substituent (B_number) extrapolation test.

The first pass showed a dramatic split between ridge on Mordred descriptors
(leave-one-B-out pooled MSE 5.12) and ridge on the concatenated set (1.06).  Two
explanations are possible and they lead to opposite conclusions:

  (a) the linear model simply blows up on held-out substituents (extreme z-scores
      multiplied by large collinear weights), or
  (b) there is no transferable signal along the substituent axis at all.

This pass separates them: every fold is scored by
  * ridge, raw prediction                (unbounded linear extrapolation)
  * ridge, prediction clipped to the training-fold target range
  * XGBoost                             (tree model: cannot extrapolate by design)
so a method that only fails when unclipped identifies (a).

B = 0 is reported separately: it holds the only examples of A13/A14, so that fold
is a scaffold hold-out in disguise and cannot be read as a substituent test.

Usage: python -B .dsh_tmp\bsplit_test2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from core.data_processing import load_valid_data  # noqa: E402
from core.train_optimizer import fit_preprocessing, transform_features  # noqa: E402

CACHE = ROOT / '.dsh_tmp' / 'fp-full' / 'Data-20261012_raw_features_fp.csv'
DATA = ROOT / 'data' / 'Data-20261012.csv'
OUT_CSV = ROOT / '.dsh_tmp' / 'bsplit_results2.csv'
FINGERPRINT_PREFIXES = ('Morgan_', 'RDKitFP_', 'AtomPair_', 'TopoTorsion_', 'MACCS_')

XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.9, colsample_bytree=0.25, reg_lambda=1.0,
                  min_child_weight=1, n_jobs=4, random_state=42, tree_method='hist')


def scores(y_true, y_pred):
    residual = np.asarray(y_true) - np.asarray(y_pred)
    mse = float(np.mean(residual ** 2))
    return mse, float(np.sqrt(mse)), float(np.mean(np.abs(residual)))


def pooled(y, predictions, mask=None):
    mask = np.ones(len(y), bool) if mask is None else mask
    mse, rmse, mae = scores(y[mask], predictions[mask])
    sst = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return dict(mse=mse, rmse=rmse, mae=mae,
                r2=float(1.0 - np.sum((y[mask] - predictions[mask]) ** 2) / sst))


def run(folds, X_raw, y, label):
    predictions = {key: np.full(len(y), np.nan) for key in ('ridge', 'clip', 'xgb')}
    rows = []
    for train_idx, test_idx in folds:
        train_X, test_X = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        train_y, test_y = y[train_idx], y[test_idx]
        prep = fit_preprocessing(train_X, train_y)
        train_scaled = transform_features(train_X, prep)
        test_scaled = transform_features(test_X, prep)
        scaled_y = prep['target_scaler'].transform(train_y.reshape(-1, 1)).ravel()
        invert = lambda values: prep['target_scaler'].inverse_transform(values.reshape(-1, 1)).ravel()

        ridge = Ridge(alpha=10.0).fit(train_scaled, scaled_y)
        raw = invert(ridge.predict(test_scaled))
        clip = np.clip(raw, train_y.min(), train_y.max())

        xgb = XGBRegressor(**XGB_PARAMS).fit(train_scaled, scaled_y)
        xgb_pred = invert(xgb.predict(test_scaled))

        predictions['ridge'][test_idx] = raw
        predictions['clip'][test_idx] = clip
        predictions['xgb'][test_idx] = xgb_pred

        rows.append({'fold': label, 'n_test': len(test_idx),
                     'ridge': scores(test_y, raw)[0],
                     'clip': scores(test_y, clip)[0],
                     'xgb': scores(test_y, xgb_pred)[0],
                     'const': scores(test_y, np.full(len(test_idx), train_y.mean()))[0]})
    return predictions, pd.DataFrame(rows)


def main() -> int:
    data = load_valid_data(str(DATA))
    y = data['Gap(eV)'].to_numpy(dtype=float)
    b = data['B_number'].to_numpy()

    cache = pd.read_csv(CACHE)
    descriptor_cols = [c for c in cache.columns if not c.startswith(FINGERPRINT_PREFIXES)]
    feature_sets = {'mordred': cache[descriptor_cols], 'mixed': cache}

    summary_rows = []
    fold_rows = []
    for set_name, X_raw in feature_sets.items():
        folds = [(np.where(b != value)[0], np.where(b == value)[0]) for value in np.unique(b)]
        predictions, fold_table = run(folds, X_raw, y, 'B')
        fold_table['B'] = np.unique(b)
        fold_table['feature_set'] = set_name
        fold_rows.append(fold_table)

        keep = np.ones(len(y), bool)
        exclude_zero = b != 0
        for label, mask in (('all 16', keep), ('excl. B0', exclude_zero)):
            row = {'feature_set': set_name, 'split': f'leave-one-B-out ({label})', 'n': int(mask.sum())}
            for key in ('ridge', 'clip', 'xgb'):
                row.update({f'{key}_{k}': v for k, v in pooled(y, predictions[key], mask).items()})
            summary_rows.append(row)

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        random_predictions, random_table = run(list(kfold.split(X_raw)), X_raw, y, 'random')
        random_table['feature_set'] = set_name
        fold_rows.append(random_table)
        row = {'feature_set': set_name, 'split': 'random 5-fold', 'n': int(len(y))}
        for key in ('ridge', 'clip', 'xgb'):
            row.update({f'{key}_{k}': v for k, v in pooled(y, random_predictions[key]).items()})
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    pd.concat(fold_rows, ignore_index=True).to_csv(OUT_CSV.with_name('bsplit_folds2.csv'),
                                                   index=False, encoding='utf-8')
    summary.to_csv(OUT_CSV, index=False, encoding='utf-8')

    pd.set_option('display.width', 210)
    for set_name in feature_sets:
        block = summary[summary['feature_set'] == set_name]
        print(f'\n=== {set_name} ===')
        for _, row in block.iterrows():
            print(f'{row["split"]:<34s} n={int(row["n"]):<5d} '
                  f'ridge {row["ridge_mse"]:7.4f} (R2 {row["ridge_r2"]:+.3f})  '
                  f'clip {row["clip_mse"]:7.4f} (R2 {row["clip_r2"]:+.3f})  '
                  f'xgb {row["xgb_mse"]:7.4f} (R2 {row["xgb_r2"]:+.3f})')
    print(f'\n[written] {OUT_CSV}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
