r"""How much of the gap variance does each design factor carry, and how does the
composition-only baseline compare with the descriptor models on the same folds?

Answers two questions left open by the leave-one-B-out runs:
  1. Is B (meso substituent) even a large effect, or does the core dominate?
  2. On the folds where a substituent is unseen, do descriptors add anything over
     a one-hot model of A / X / C?

Usage: python -B .dsh_tmp\bsplit_diagnostics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import Ridge  # noqa: E402

from core.data_processing import load_valid_data  # noqa: E402


def group_r2(y, keys):
    frame = pd.DataFrame({'y': y})
    for name, values in keys.items():
        frame[name] = values
    grouped = frame.groupby(list(keys))['y']
    means = grouped.transform('mean').to_numpy()
    sse = float(np.sum((y - means) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - sse / sst, grouped.ngroups


def main() -> int:
    data = load_valid_data(str(ROOT / 'data' / 'Data-20261012.csv'))
    y = data['Gap(eV)'].to_numpy(dtype=float)
    a = data['A_number'].astype(str).to_numpy()
    x = data['X_element'].astype(str).to_numpy()
    b = data['B_number'].astype(str).to_numpy()
    c = data['C_number'].astype(str).to_numpy()

    print('=== in-sample R2 of a pure group-mean model (how much each factor carries) ===')
    for label, keys in [
        ('A only', {'A': a}),
        ('B only', {'B': b}),
        ('X only', {'X': x}),
        ('C only', {'C': c}),
        ('A + X + C', {'A': a, 'X': x, 'C': c}),
        ('A + B', {'A': a, 'B': b}),
        ('A + X + C + B', {'A': a, 'X': x, 'C': c, 'B': b}),
    ]:
        r2, groups = group_r2(y, keys)
        print(f'  {label:<14s} R2 {r2:6.3f}   groups {groups:5d}')

    print('\n=== how far does B move the gap inside one core? ===')
    core = pd.DataFrame({'A': a, 'X': x, 'C': c, 'B': b, 'y': y})
    spreads = []
    for _, block in core.groupby(['A', 'X', 'C']):
        if block['B'].nunique() >= 3:
            spreads.append(float(block.groupby('B')['y'].mean().std(ddof=0)))
    spreads = np.asarray(spreads)
    print(f'  cores with >=3 B patterns: {len(spreads)}')
    print(f'  sd of B-means within a core: median {np.median(spreads):.3f} eV, '
          f'90th pct {np.percentile(spreads, 90):.3f} eV')
    print(f'  overall gap sd: {np.std(y):.3f} eV')

    print('\n=== composition-only baseline on the leave-one-B-out folds ===')
    design = pd.concat([pd.get_dummies(pd.Series(a), prefix='A'),
                        pd.get_dummies(pd.Series(x), prefix='X'),
                        pd.get_dummies(pd.Series(c), prefix='C')],
                       axis=1).astype(float).to_numpy()
    predictions = np.full(len(y), np.nan)
    for value in np.unique(b):
        test = np.where(b == value)[0]
        train = np.where(b != value)[0]
        model = Ridge(alpha=1.0).fit(design[train], y[train])
        predictions[test] = model.predict(design[test])
    for label, mask in (('all 16', np.ones(len(y), bool)), ('excl. B0', b != '0')):
        sse = float(np.sum((y[mask] - predictions[mask]) ** 2))
        sst = float(np.sum((y[mask] - y[mask].mean()) ** 2))
        mse = sse / mask.sum()
        print(f'  {label:<9s} n={mask.sum():4d}  MSE {mse:.4f}  RMSE {np.sqrt(mse):.4f}  '
              f'R2 {1 - sse / sst:+.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
