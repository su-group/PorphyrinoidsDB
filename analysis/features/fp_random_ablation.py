"""Fingerprint ablation on the RANDOM split, built to mirror fp_scaffold_ablation.py.

Same models, same hyperparameters, same pooled metric, so the two tables can be
read side by side.  Fold positions come from the completed corrected R-M run;
the matrices are regenerated through the pipeline's own functions.

Anchor: Lasso with the corrected run's best params must reproduce the published
CV MSE 0.270567 (what `experiments/20261016_R-M` reports for Lasso).
"""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from core.data_processing import load_valid_data, load_feature_cache  # noqa: E402
from core.experiment_setup import load_split_manifest  # noqa: E402
from core.train_optimizer import fit_preprocessing, transform_features  # noqa: E402

DATA = 'data/Data-20261012.csv'
SPLITS = 'experiments/20261013_R-M/splits.json'
FP = '.dsh_tmp/fp-full/Data-20261012_raw_features_fp.csv'
RUN = 'experiments/20261016_R-M/models/R-M'
FAMILIES = ('Morgan', 'MACCS', 'AtomPair', 'TopoTorsion', 'RDKitFP')
ANCHOR = 0.270567020213          # published Lasso CV MSE in the corrected R-M run
ALPHA, MAX_ITER = 0.001010171604205459, 1945

data_valid = load_valid_data(DATA)
X_full, meta = load_feature_cache(FP, data_valid, DATA)
train_idx, test_idx, _ = load_split_manifest(SPLITS, data_valid, DATA, 'random', False)
train_idx = np.asarray(train_idx)
y_full = data_valid['Gap(eV)'].to_numpy()
y_train = y_full[train_idx]
mordred_cols = [c for c in X_full.columns if not str(c).startswith(FAMILIES)]
print('fp cache %s  mordred=%d  fingerprint=%d  train=%d test=%d'
      % (X_full.shape, len(mordred_cols), X_full.shape[1] - len(mordred_cols),
         len(train_idx), len(np.asarray(test_idx))))

folds_saved = pickle.load(open(RUN + '/prepared_data.pkl', 'rb'))[0]
positions = [(np.asarray(f['train_positions']), np.asarray(f['valid_positions']))
             for f in folds_saved]
print('fold sizes: %s' % [(len(tr), len(va)) for tr, va in positions])


def rebuild(Xcfg):
    assert len(Xcfg) == len(train_idx), \
        'rebuild needs the %d-row training subset' % len(train_idx)
    out = []
    for tr, va in positions:
        prep = fit_preprocessing(Xcfg.iloc[tr], y_train[tr])
        out.append((transform_features(Xcfg.iloc[tr], prep),
                    transform_features(Xcfg.iloc[va], prep), prep, tr, va))
    return out


def score(built, maker):
    pooled_p, pooled_y, mses, r2s = [], [], [], []
    for Xtr, Xva, prep, tr, va in built:
        ytr_scaled = prep['target_scaler'].transform(y_train[tr, None]).ravel()
        m = maker().fit(Xtr, ytr_scaled)
        pred = prep['target_scaler'].inverse_transform(m.predict(Xva)[:, None]).ravel()
        mses.append(mean_squared_error(y_train[va], pred))
        r2s.append(r2_score(y_train[va], pred))
        pooled_p.append(pred)
        pooled_y.append(y_train[va])
    pp, py = np.concatenate(pooled_p), np.concatenate(pooled_y)
    return (float(mean_squared_error(py, pp)), float(r2_score(py, pp)),
            float(np.mean(r2s)))


print('anchor check (Mordred-only, random split)')
built = rebuild(X_full[mordred_cols].iloc[train_idx])
cv, _, _ = score(built, lambda: Lasso(alpha=ALPHA, max_iter=MAX_ITER, random_state=42))
print('   pooled CV MSE=%.6f   published (fold-mean) =%.6f' % (cv, ANCHOR))
if cv > 0.5:
    raise SystemExit('ANCHOR FAILED: pooled CV MSE is implausible')
del built

CONFIGS = {
    '仅 Mordred (1613)': lambda c: not str(c).startswith(FAMILIES),
    'Mordred+Morgan': lambda c: not str(c).startswith(
        tuple(f for f in FAMILIES if f != 'Morgan')),
    'Mordred+MACCS': lambda c: not str(c).startswith(
        tuple(f for f in FAMILIES if f != 'MACCS')),
    'Mordred+RDKitFP': lambda c: not str(c).startswith(
        tuple(f for f in FAMILIES if f != 'RDKitFP')),
    'Mordred+AtomPair+TopoTorsion': lambda c: not str(c).startswith(('Morgan', 'MACCS', 'RDKitFP')),
    '全部指纹 (9972)': lambda c: True,
}
MODELS = [
    ('Lasso(R-M best)', lambda: Lasso(alpha=ALPHA, max_iter=MAX_ITER, random_state=42)),
    ('RandomForest(R-M best)', lambda: RandomForestRegressor(
        n_estimators=187, max_depth=20, min_samples_split=5, min_samples_leaf=3,
        random_state=42, n_jobs=4)),
    ('SVR(corrected best)', lambda: SVR(kernel='rbf', C=912.315859643007,
                                        gamma=0.0003499123954435787,
                                        epsilon=0.0030677893441494776)),
]

rows = []
for cfg, keep in CONFIGS.items():
    cols = [c for c in X_full.columns if keep(c)]
    built = rebuild(X_full[cols].iloc[train_idx])
    print('=== %s  raw=%d  kept(fold0)=%d' % (cfg, len(cols), built[0][0].shape[1]))
    for mname, maker in MODELS:
        t0 = time.time()
        cv, r2, fold_r2 = score(built, maker)
        rows.append((cfg, len(cols), built[0][0].shape[1], mname, cv, r2, fold_r2))
        print('   %-22s pooled CV MSE=%7.4f  pooled CV R2=%+7.4f  (fold-R2 mean %+.4f)  %.0fs'
              % (mname, cv, r2, fold_r2, time.time() - t0))
        sys.stdout.flush()
    del built

output = Path('analysis_outputs/features/random_ablation.csv')
output.parent.mkdir(parents=True, exist_ok=True)
import csv
with output.open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['config', 'raw_cols', 'kept_fold0', 'model', 'pooled_cv_mse', 'pooled_cv_r2', 'fold_r2_mean'])
    writer.writerows(rows)
print(output)
