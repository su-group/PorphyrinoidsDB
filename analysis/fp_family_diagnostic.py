"""Which fingerprint family actually earns its place, and what does each cost?

Runs the pipeline's own per-fold filter (variance + target correlation + 0.95
greedy de-correlation) and the project's actual model settings for the models
that matter (Lasso / XGBoost / SVR / KernelRidge / GaussianProcess) on:
  mordred              1613
  all-fp               mordred + 8359
  morgan-only          mordred + Morgan 2048
  maccs-only           mordred + MACCS 167
  rdkitfp-only         mordred + RDKitFP 2048
  topo-only            mordred + AtomPair + TopoTorsion 4096
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

sys.path.insert(0, '.')
from core.train_optimizer import target_correlations  # noqa: E402

CACHE = Path('.dsh_tmp/fp-full/Data-20261012_raw_features_fp.csv.xz')
META = Path('.dsh_tmp/fp-full/Data-20261012_raw_features_fp.csv.meta.json')
FAMILIES = ('Morgan', 'MACCS', 'AtomPair', 'TopoTorsion', 'RDKitFP')
MIN_FEATURES = 50


def family_of(column):
    for prefix in FAMILIES:
        if str(column).startswith(prefix):
            return prefix
    return 'mordred'


def select_like_pipeline(X, y):
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median().dropna()
    clean = X[medians.index].fillna(medians)
    variances = clean.var(ddof=0)
    nonconstant = variances[variances > 1e-10].index.tolist()
    variable = variances[variances > 0.01].index.tolist()
    candidates = variable if len(variable) >= MIN_FEATURES else nonconstant
    relevance = target_correlations(clean[candidates], y).abs().fillna(0)
    ranked = relevance.sort_values(ascending=False, kind='stable').index.tolist()
    corr = np.abs(np.atleast_2d(np.corrcoef(clean[candidates].to_numpy(), rowvar=False)))
    pos = {name: i for i, name in enumerate(candidates)}
    kept = []
    for name in ranked:
        if not kept or not (corr[pos[name], [pos[c] for c in kept]] > 0.95).any():
            kept.append(name)
    if len(kept) < min(MIN_FEATURES, len(candidates)):
        kept = ranked[:min(MIN_FEATURES, len(candidates))]
    return [c for c in candidates if c in kept]


def models():
    return {
        'Lasso': Lasso(alpha=0.0315, max_iter=10000, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                random_state=42, n_jobs=4),
        'SVR': SVR(C=10.0, epsilon=0.05),
        'KernelRidge': KernelRidge(kernel='rbf', alpha=1.0, gamma=0.01),
        'GaussianProcess': GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(1.0), random_state=42,
            n_restarts_optimizer=0),
    }


def evaluate(name, X, y, folds):
    out = {}
    for fold_no, (tr, va) in enumerate(folds):
        t0 = time.perf_counter()
        kept = select_like_pipeline(X.iloc[tr], y[tr])
        Xtr = X.iloc[tr][kept].replace([np.inf, -np.inf], np.nan)
        med = Xtr.median().dropna()
        Xtr = Xtr[med.index].fillna(med)
        Xva = X.iloc[va][med.index].replace([np.inf, -np.inf], np.nan).fillna(med)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
        counts = {}
        for c in kept:
            counts[family_of(c)] = counts.get(family_of(c), 0) + 1
        print(f'  [{name}] fold{fold_no}: kept={len(kept)} {counts} '
              f'select={time.perf_counter()-t0:.0f}s', flush=True)
        for model_name, model in models().items():
            t1 = time.perf_counter()
            model.fit(Xtr_s, y[tr])
            pred = model.predict(Xva_s)
            rec = out.setdefault(model_name, {'mse': [], 'r2': [], 'nfeat': [], 'sec': []})
            rec['mse'].append(mean_squared_error(y[va], pred))
            rec['r2'].append(r2_score(y[va], pred))
            rec['nfeat'].append(len(kept))
            rec['sec'].append(time.perf_counter() - t1)
            print(f'      {model_name:16s} mse={rec["mse"][-1]:.4f} '
                  f'r2={rec["r2"][-1]:.4f} {rec["sec"][-1]:.1f}s', flush=True)
    return out


def main():
    df = pd.read_csv(CACHE)
    meta = json.load(open(META, encoding='utf-8'))
    y = np.asarray([s['Gap(eV)'] for s in meta['samples']], dtype=float)
    cols_of = {f: [c for c in df.columns if family_of(c) == f] for f in FAMILIES}
    mordred = [c for c in df.columns if family_of(c) == 'mordred']
    print(f'mordred={len(mordred)} families=' +
          ', '.join(f'{f}={len(cols_of[f])}' for f in FAMILIES), flush=True)
    folds = list(KFold(5, shuffle=True, random_state=42).split(np.arange(len(df))))

    configs = {
        'mordred': mordred,
        'all-fp': mordred + [c for f in FAMILIES for c in cols_of[f]],
        'morgan-only': mordred + cols_of['Morgan'],
        'maccs-only': mordred + cols_of['MACCS'],
        'rdkitfp-only': mordred + cols_of['RDKitFP'],
        'topo-only': mordred + cols_of['AtomPair'] + cols_of['TopoTorsion'],
    }

    all_out = {}
    for name, cols in configs.items():
        print(f'\n=== {name}: {len(cols)} raw columns ===', flush=True)
        all_out[name] = {'ncols': len(cols), 'models': evaluate(name, df[cols], y, folds)}

    lines = []
    for name, info in all_out.items():
        for model_name, rec in info['models'].items():
            lines.append(f'{name:14s} {info["ncols"]:5d}raw {model_name:16s} '
                         f'CVMSE={np.mean(rec["mse"]):.4f} CVR2={np.mean(rec["r2"]):.4f} '
                         f'kept={np.mean(rec["nfeat"]):.0f} fit={np.mean(rec["sec"]):.1f}s')
    text = '\n'.join(lines)
    Path('.dsh_tmp/fp_family_diagnostic.txt').write_text(text, encoding='utf-8')
    print('\n' + text)


if __name__ == '__main__':
    main()
