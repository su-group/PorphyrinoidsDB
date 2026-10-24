"""Decisive test: on the EXACT fold matrices the pipeline used, compare the parameters
Optuna selected for SVR / KernelRidge against plain sklearn defaults.

Purpose: decide whether R-MF's collapse on SVR/KernelRidge comes from the feature set
(mordred + fingerprints) or from the hyper-parameter search itself.
"""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

warnings.filterwarnings('ignore')

RUN = Path(sys.argv[1])
WANT = sys.argv[2].split(',') if len(sys.argv) > 2 else ['SVR', 'KernelRidge']

with open(RUN / 'prepared_data.pkl', 'rb') as f:
    folds, _splits = pickle.load(f)

CAND = {
    'SVR': {
        'optuna-best': None,                       # filled below
        'sklearn-default': dict(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1),
        'plane-C10-g0.001': dict(kernel='rbf', C=10.0, gamma=0.001, epsilon=0.1),
    },
    'KernelRidge': {
        'optuna-best': None,
        'sklearn-default': dict(kernel='rbf', alpha=1.0, gamma=None),
        'plane-a0.01-g0.001': dict(kernel='rbf', alpha=0.01, gamma=0.001),
    },
}
BASE = {'SVR': SVR, 'KernelRidge': KernelRidge}

for name in WANT:
    with open(RUN / f'{name}.result.pkl', 'rb') as f:
        r = pickle.load(f)
    CAND[name]['optuna-best'] = r['best_params']
    print('%-13s pipeline cv_mse=%.4f  best_params=%s' % (name, r['cv_mse'], r['best_params']))
    rows = []
    for label, params in CAND[name].items():
        losses, r2s = [], []
        t0 = time.time()
        for fold in folds:
            m = BASE[name](**params)
            m.fit(fold['X_train'], fold['y_train'])
            pred = fold['target_scaler'].inverse_transform(m.predict(fold['X_valid']).reshape(-1, 1)).ravel()
            losses.append(mean_squared_error(fold['y_valid'], pred))
            r2s.append(r2_score(fold['y_valid'], pred))
        dt = time.time() - t0
        rows.append((np.mean(losses), np.mean(r2s), label, dt))
        print('    %-20s CV MSE=%.4f  CV R2=%+.4f  (%.1fs)' % (label, np.mean(losses), np.mean(r2s), dt))
        sys.stdout.flush()
    best = min(rows)
    print('    -> BEST on this exact matrix: %s (CV MSE=%.4f)\n' % (best[2], best[0]))
    sys.stdout.flush()
