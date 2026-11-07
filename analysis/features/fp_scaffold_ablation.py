"""Fingerprint ablation on the scaffold split — rewritten from scratch.

Uses the pipeline's own functions to rebuild the folds and the pipeline's own
scoring convention: models fit on the FOLD-SCALED target, and CV MSE is the mean
squared error of the inverse-transformed (eV) predictions, exactly as
`core/train_optimizer.optimize_model_parameters` computes it.

Anchors that must reproduce exactly, or the script exits:
  * S-M Lasso  CV MSE = 0.6191693213237417
"""
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
FAMILIES = ('Morgan', 'MACCS', 'AtomPair', 'TopoTorsion', 'RDKitFP')
OUT = Path('analysis_outputs/features/scaffold_ablation.csv')

# The pipeline records these numbers; reproducing them proves the harness.
ANCHORS = {
    ('scaffold', 'Lasso'): 0.6191693213237417,
    ('random', 'Lasso'): 0.270567,
}
LASSOS = {
    'scaffold': (0.007037542783164669, 7151),
    'random': (0.001010171604205459, 1945),
}


def build_folds(split_name, n_train_expected):
    """Load the fingerprint cache through the pipeline's own verifier.

    `load_feature_cache` checks schema_version, the SHA-256 of the data file, the
    SHA-256 of the cache and the full sample identity/order, so a stale or
    mismatched cache stops the run here rather than producing untraceable numbers.
    """
    data_valid = load_valid_data(DATA)
    X_full, cache_meta = load_feature_cache(FP, data_valid, DATA)
    train_idx, test_idx, fold_indices = load_split_manifest(SPLITS, data_valid, DATA, split_name, False)
    train_idx = np.asarray(train_idx)
    assert len(train_idx) == n_train_expected, (split_name, len(train_idx))
    y_full = data_valid['Gap(eV)'].to_numpy()
    assert cache_meta['use_fingerprints'] is True, 'fingerprint cache expected'
    mordred_cols = [c for c in X_full.columns if not str(c).startswith(FAMILIES)]
    print('cache verified: %s  mordred=%d  fingerprint=%d  train=%d test=%d'
          % (X_full.shape, len(mordred_cols), X_full.shape[1] - len(mordred_cols),
             len(train_idx), len(np.asarray(test_idx))))
    return data_valid, train_idx, y_full, X_full, mordred_cols, fold_indices


def main():
    (data_valid, train_idx, y_full, X_full, mordred_cols,
     fold_indices) = build_folds('scaffold', 1383)
    y_train = y_full[train_idx]

    # Fold positions come from the completed S-M run; the matrices themselves are
    # regenerated here and the *pooled* preprocessing is compared against
    # prepared_data.pkl's own copy as a consistency check.
    saved, prep_saved = pickle.load(
        open('experiments/20261014_S-M/models/S-M/prepared_data.pkl', 'rb'))
    positions = [(np.asarray(f['train_positions']), np.asarray(f['valid_positions']))
                 for f in saved]
    assert [len(tr) for tr, _ in positions] == [1118, 1123, 1120, 1085, 1086]
    # Cross-validate the two caches: the fp cache's Mordred columns must be the
    # pure Mordred cache verbatim (names and order), otherwise the anchor below
    # could not reproduce the published CV MSE on the same fold positions.
    pure, _ = load_feature_cache('data/Data-20261012_raw_features.csv',
                                 data_valid, DATA)
    assert list(pure.columns) == mordred_cols, 'fp cache Mordred columns differ from the pure cache'
    a, b = pure.to_numpy(), X_full[mordred_cols].to_numpy()
    both = np.isfinite(a) & np.isfinite(b)
    exact = float(np.mean(a[both] == b[both]))
    print('fp cache Mordred block vs pure cache: bit-exact=%.2f%%  max|diff|=%.2e (Mordred is not'
          ' bit-reproducible across runs, so the anchor below has a small tolerance)'
          % (100 * exact, np.abs(a[both] - b[both]).max()))

    def rebuild(Xcfg):
        # Fold positions are relative to the TRAINING SUBSET, so the frame handed
        # in must be exactly those 1383 rows.  Assert rather than silently index
        # the wrong molecules -- that mistake produced several wrong tables.
        assert len(Xcfg) == len(train_idx), \
            'rebuild expects the %d-row training subset, got %d rows' % (len(train_idx), len(Xcfg))
        out = []
        for tr, va in positions:
            prep = fit_preprocessing(Xcfg.iloc[tr], y_train[tr])
            out.append((transform_features(Xcfg.iloc[tr], prep),
                        transform_features(Xcfg.iloc[va], prep), prep, tr, va))
        return out

    print('anchor check (Mordred-only, scaffold split)')
    mordred = X_full[mordred_cols].iloc[train_idx]
    built = rebuild(mordred)
    alpha, max_iter = LASSOS['scaffold']
    losses = []
    for Xtr, Xva, prep, tr, va in built:
        m = Lasso(alpha=alpha, max_iter=max_iter, random_state=42).fit(Xtr, prep['target_scaler'].transform(y_train[tr, None]).ravel())
        pred = prep['target_scaler'].inverse_transform(m.predict(Xva)[:, None]).ravel()
        losses.append(mean_squared_error(y_train[va], pred))
    value = float(np.mean(losses))
    print('   Lasso CV MSE = %.6f  (pipeline reported %.6f)' % (value, ANCHORS[('scaffold', 'Lasso')]))
    if abs(value - ANCHORS[('scaffold', 'Lasso')]) > 5e-3:
        raise SystemExit('ANCHOR FAILED -- harness is not reproducing the pipeline')
    print('   ANCHOR OK (tolerance 5e-3)\n')

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
        ('Lasso(S-M best)', lambda: Lasso(alpha=alpha, max_iter=max_iter, random_state=42)),
        ('RandomForest(S-M best)', lambda: RandomForestRegressor(
            n_estimators=32, max_depth=8, min_samples_split=10, min_samples_leaf=7,
            random_state=42, n_jobs=4)),
        ('SVR(fixed C/gamma)', lambda: SVR(kernel='rbf', C=912.315859643007,
                                           gamma=0.0003499123954435787,
                                           epsilon=0.0030677893441494776)),
    ]

    rows = []
    for cfg, keep in CONFIGS.items():
        cols = [c for c in X_full.columns if keep(c)]
        built = rebuild(X_full[cols].iloc[train_idx])
        print('=== %s  raw=%d  kept(fold0)=%d' % (cfg, len(cols), built[0][0].shape[1]))
        # The Mordred-only configuration must reproduce the published S-M Lasso
        # CV MSE here too, not just in the standalone anchor above.
        if cfg.startswith('仅 Mordred'):
            m = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
            ref = []
            for Xtr, Xva, prep, tr, va in built:
                m.fit(Xtr, prep['target_scaler'].transform(y_train[tr, None]).ravel())
                p = prep['target_scaler'].inverse_transform(m.predict(Xva)[:, None]).ravel()
                ref.append(mean_squared_error(y_train[va], p))
            assert abs(float(np.mean(ref)) - ANCHORS[('scaffold', 'Lasso')]) < 5e-3, \
                'in-table Mordred-only Lasso does not match the published CV MSE'
            print('   in-table anchor reproduces %.6f  OK' % float(np.mean(ref)))
        for mname, maker in MODELS:
            t0, pooled_p, pooled_y = time.time(), [], []
            mses, r2s = [], []
            for Xtr, Xva, prep, tr, va in built:
                # Same convention as optimize_model_parameters: fit on the folded
                # (scaled) target, score after inverse-transform, in eV^2.  The
                # per-fold MSEs are kept for reference, but the headline numbers
                # are POOLED across folds, because each scaffold fold has a very
                # different target variance (sd 0.31 ... 1.62 eV) and averaging
                # MSEs would mostly measure that spread.
                ytr_scaled = prep['target_scaler'].transform(y_train[tr, None]).ravel()
                m = maker().fit(Xtr, ytr_scaled)
                pred = prep['target_scaler'].inverse_transform(m.predict(Xva)[:, None]).ravel()
                mses.append(mean_squared_error(y_train[va], pred))
                r2s.append(r2_score(y_train[va], pred))
                pooled_p.append(pred)
                pooled_y.append(y_train[va])
            pp = np.concatenate(pooled_p)
            py = np.concatenate(pooled_y)
            rows.append({'config': cfg, 'model': mname, 'raw_cols': len(cols),
                         'kept': built[0][0].shape[1],
                         'cv_mse': float(mean_squared_error(py, pp)),
                         'cv_r2': float(r2_score(py, pp)),
                         'fold_rmse_mean': float(np.mean(np.sqrt(mses))),
                         'fold_r2_mean': float(np.mean(r2s)),
                         'secs': round(time.time() - t0, 1)})
            print('   %-22s pooled CV MSE=%7.4f  pooled CV R2=%+7.4f  (fold-R2 mean %+.4f)  %.0fs'
                  % (mname, rows[-1]['cv_mse'], rows[-1]['cv_r2'], rows[-1]['fold_r2_mean'],
                     time.time() - t0))
            sys.stdout.flush()
        del built

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).rename(columns={'cv_mse': 'pooled_cv_mse', 'cv_r2': 'pooled_cv_r2', 'kept': 'kept_fold0'}).to_csv(OUT, index=False, encoding='utf-8-sig')
    print(OUT)


if __name__ == '__main__':
    main()
