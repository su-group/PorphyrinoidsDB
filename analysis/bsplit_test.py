r"""Quick out-of-sample test along the meso-substituent axis (B_number).

Question: the scaffold split fails; does the model transfer to a meso-substituent
pattern it has never seen, while every macrocyclic core stays familiar?

Protocol: leave-one-B-out over the 16 B patterns of the full 1,734-molecule set,
repeated for both feature sets, with exactly the in-fold preprocessing of the main
pipeline (fit_preprocessing / transform_features).  Three references are scored in
the identical folds:

  * ridge on descriptors                 -- the model
  * constant (training-fold mean)        -- no-information floor
  * ridge on one-hot A / X / C only      -- composition baseline; B is deliberately
                                            excluded because a dummy for an unseen
                                            substituent cannot exist

plus a random 5-fold run with the same estimator and preprocessing, to show what
the same pipeline scores when the substituent is familiar.

Usage: python -B .dsh_tmp\bsplit_test.py
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

from core.data_processing import load_valid_data  # noqa: E402
from core.train_optimizer import fit_preprocessing, transform_features  # noqa: E402

CACHE = ROOT / '.dsh_tmp' / 'fp-full' / 'Data-20261012_raw_features_fp.csv'
DATA = ROOT / 'data' / 'Data-20261012.csv'
OUT_CSV = ROOT / '.dsh_tmp' / 'bsplit_results.csv'

FINGERPRINT_PREFIXES = ('Morgan_', 'RDKitFP_', 'AtomPair_', 'TopoTorsion_', 'MACCS_')


def metrics(y_true, y_pred, reference=None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    mse = float(np.mean(residual ** 2))
    base = np.mean(y_true) if reference is None else reference
    sst = float(np.sum((y_true - base) ** 2))
    return {
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'mae': float(np.mean(np.abs(residual))),
        'r2': float(1.0 - np.sum(residual ** 2) / sst) if sst > 0 else float('nan'),
    }


def fit_predict(train_X, train_y, test_X):
    prep = fit_preprocessing(train_X, train_y)
    model = Ridge(alpha=10.0)
    model.fit(transform_features(train_X, prep), prep['target_scaler'].transform(
        np.asarray(train_y).reshape(-1, 1)).ravel())
    predicted = model.predict(transform_features(test_X, prep))
    return prep['target_scaler'].inverse_transform(predicted.reshape(-1, 1)).ravel(), prep


def design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """One-hot of the design factors that can be known for an unseen substituent."""
    blocks = [pd.get_dummies(frame['A_number'].astype(str), prefix='A'),
              pd.get_dummies(frame['X_element'].astype(str), prefix='X'),
              pd.get_dummies(frame['C_number'].astype(str), prefix='C')]
    return pd.concat(blocks, axis=1).astype(float)


def run_split(name, folds, X_raw, y, frame, verbose=True):
    predictions = np.full(len(y), np.nan)
    constant = np.full(len(y), np.nan)
    dummy = np.full(len(y), np.nan)
    design_all = design_matrix(frame)
    rows = []
    for train_idx, test_idx in folds:
        train_X, test_X = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        fold_pred, _ = fit_predict(train_X, y[train_idx], test_X)
        predictions[test_idx] = fold_pred
        constant[test_idx] = float(np.mean(y[train_idx]))

        design_model = Ridge(alpha=1.0)
        train_design = design_all.iloc[train_idx]
        test_design = design_all.iloc[test_idx]
        # Align columns: a held-out group may add a column that training lacks.
        test_design = test_design.reindex(columns=train_design.columns, fill_value=0.0)
        design_model.fit(train_design.to_numpy(), y[train_idx])
        dummy[test_idx] = design_model.predict(test_design.to_numpy())

        rows.append({
            'split': name,
            'n_test': int(len(test_idx)),
            'model_mse': metrics(y[test_idx], fold_pred)['mse'],
            'const_mse': metrics(y[test_idx], np.full(len(test_idx), np.mean(y[train_idx])))['mse'],
            'dummy_mse': metrics(y[test_idx], dummy[test_idx])['mse'],
            'model_r2': metrics(y[test_idx], fold_pred)['r2'],
        })
    result = {
        'split': name,
        'n': int(len(y)),
        'model': metrics(y, predictions),
        'constant': metrics(y, constant),
        'dummy': metrics(y, dummy),
    }
    if verbose:
        print(f'  {name:<22s} model MSE {result["model"]["mse"]:.4f}  R2 {result["model"]["r2"]:+.4f}'
              f'  | const {result["constant"]["mse"]:.4f}  dummy {result["dummy"]["mse"]:.4f}')
    return result, predictions, pd.DataFrame(rows)


def main() -> int:
    data = load_valid_data(str(DATA))
    y = data['Gap(eV)'].to_numpy(dtype=float)
    b = data['B_number'].to_numpy()
    a = data['A_number'].to_numpy()

    print('reading feature cache ...')
    cache = pd.read_csv(CACHE)
    print(f'  cache {cache.shape[0]} x {cache.shape[1]}')
    if len(cache) != len(data):
        print('!! cache and data row counts differ')
        return 1
    descriptor_cols = [c for c in cache.columns if not c.startswith(FINGERPRINT_PREFIXES)]
    feature_sets = {'mordred': cache[descriptor_cols], 'mixed': cache}

    all_rows = []
    summary = []
    for set_name, X_raw in feature_sets.items():
        print(f'\n=== {set_name} ({X_raw.shape[1]} raw columns) ===')

        b_folds = [(np.where(b != value)[0], np.where(b == value)[0]) for value in np.unique(b)]
        result, predictions, folds = run_split(f'leave-one-B-out ({set_name})', b_folds, X_raw, y, data)
        summary.append(result)
        folds['B'] = [int(value) for value in np.unique(b)]

        # per-B detail: composition of the held-out block and its neighbours
        for value in np.unique(b):
            mask = b == value
            train_mask = ~mask
            unseen_scaffold = len(set(a[mask]) - set(a[train_mask]))
            folds.loc[folds['B'] == value, 'n_scaffold'] = int(len(set(a[mask])))
            folds.loc[folds['B'] == value, 'unseen_scaffold'] = int(unseen_scaffold)
            folds.loc[folds['B'] == value, 'unseen_substituent_rows'] = int(np.sum(mask))

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        result, _, random_folds = run_split(f'random 5-fold ({set_name})',
                                            list(kfold.split(X_raw)), X_raw, y, data)
        summary.append(result)

        folds['feature_set'] = set_name
        all_rows.append(folds)

        order = folds.assign(abs_bias=(folds['model_mse'] / folds['const_mse']))
        print('  per-B skipped here; see the CSV')

    frame = pd.concat(all_rows, ignore_index=True)
    frame.to_csv(OUT_CSV, index=False, encoding='utf-8')
    print(f'\n[written] {OUT_CSV}')

    print('\n=== summary ===')
    for result in summary:
        model, const, dummy = result['model'], result['constant'], result['dummy']
        print(f'{result["split"]:<30s} model MSE {model["mse"]:.4f} R2 {model["r2"]:+.4f} '
              f'RMSE {model["rmse"]:.4f} MAE {model["mae"]:.4f} | '
              f'const {const["mse"]:.4f} (R2 {const["r2"]:+.4f}) | dummy {dummy["mse"]:.4f} '
              f'(R2 {dummy["r2"]:+.4f}) | model/const {model["mse"]/const["mse"]:.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
