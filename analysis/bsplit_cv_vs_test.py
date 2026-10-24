r"""Quantify why the meso-B axis shows a large CV-vs-test gap.

The B axis holds out meso substituents (B_number groups).  The CV folds and the single
held-out test set exclude *different* B groups, so they are not the same task; this
script measures how different by reporting the label distribution each side is scored on
and the error of two trivial predictors (training mean, own-set mean) as a floor.

Usage: python -B .dsh_tmp\bsplit_cv_vs_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data_processing import load_valid_data

MANIFEST = ROOT / 'experiments' / '20261021_splits_meso.json'
RUN_DIR = ROOT / 'experiments' / '20261021_B-M'
TARGET = 'Gap(eV)'


def main() -> int:
    data = load_valid_data(str(ROOT / 'data' / 'Data-20261012.csv'))
    y = data[TARGET].to_numpy(dtype=float)
    b = data['B_number'].to_numpy()
    spec = json.loads(MANIFEST.read_text(encoding='utf-8'))['splits']['meso']

    train_pos = np.asarray(spec['train_positions'])
    test_pos = np.asarray(spec['test_positions'])
    folds = spec['folds']

    print('=== label distribution ===')
    print(f'train  n={len(train_pos):4d}  mean={y[train_pos].mean():.4f}  sd={y[train_pos].std(ddof=1):.4f}')
    print(f'test   n={len(test_pos):4d}  mean={y[test_pos].mean():.4f}  sd={y[test_pos].std(ddof=1):.4f}')
    print(f'test B groups: {sorted(set(b[test_pos].tolist()))}')
    print(f'train B groups: {sorted(set(b[train_pos].tolist()))}')

    print('\n=== per-fold validation slice (positions are relative to the train subset) ===')
    rows = []
    for index, fold in enumerate(folds):
        valid_local = np.asarray(fold['valid_positions'])
        train_local = np.asarray(fold['train_positions'])
        valid = train_pos[valid_local]
        inner = train_pos[train_local]
        fold_mean = y[inner].mean()
        const_mse = float(np.mean((y[valid] - fold_mean) ** 2))
        oracle_mse = float(np.mean((y[valid] - y[valid].mean()) ** 2))
        rows.append({
            'fold': index,
            'train_mean': fold_mean,
            'valid_mean': y[valid].mean(),
            'valid_sd': y[valid].std(ddof=1),
            'valid_B': len(set(b[valid].tolist())),
            'const_mse': const_mse,
            'oracle_mse': oracle_mse,
            'n': len(valid),
        })
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    test_mean = y[train_pos].mean()
    test_const = float(np.mean((y[test_pos] - test_mean) ** 2))
    test_oracle = float(np.mean((y[test_pos] - y[test_pos].mean()) ** 2))
    print('\n=== test slice ===')
    print(f'const(train mean) MSE = {test_const:.4f}   oracle(own mean) MSE = {test_oracle:.4f}')

    print('\n=== pooled vs mean-of-folds (B-M final models) ===')
    print(f'mean of per-fold const MSE = {frame["const_mse"].mean():.4f}')
    print(f'mean of per-fold oracle MSE = {frame["oracle_mse"].mean():.4f}')

    preds_path = RUN_DIR / 'analysis_results' / 'B-M' / 'test_predictions.csv'
    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        truth = preds[TARGET].to_numpy(dtype=float)
        print(f'\ntest set used by the pipeline: n={len(truth)} mean={truth.mean():.4f} sd={truth.std(ddof=1):.4f}')
        for model in ['KNeighbors', 'LightGBM', 'RandomForest', 'Lasso', 'DecisionTree']:
            if model not in preds.columns:
                continue
            residual = truth - preds[model].to_numpy(dtype=float)
            r2 = 1.0 - float(np.sum(residual ** 2)) / float(np.sum((truth - truth.mean()) ** 2))
            print(f'  {model:14s} test R2={r2:.4f}  MSE={np.mean(residual ** 2):.4f}  '
                  f'ratio to const={np.mean(residual ** 2) / test_const:.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
