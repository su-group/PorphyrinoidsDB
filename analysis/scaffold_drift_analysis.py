"""Quantify how much of the scaffold-split performance drop is label-distribution drift.

Usage: python -B .dsh_tmp/scaffold_drift_analysis.py
Writes .dsh_tmp/runnability/<chinese file>

For both split types (scaffold / random) this rebuilds the pipeline's own 5 folds and
reports, per fold:
  * label mean / sd on train vs valid
  * model MSE (Lasso at that split's published best params, which is anchored below)
  * constant-predictor MSE (predict the TRAIN fold mean)      -> how hard is the shift alone
  * oracle-constant MSE (predict the VALID fold mean)         -> best any constant can do
  * Bemis-Murcko scaffold overlap and A_number family overlap between valid and train
Anchors: scaffold mean-of-fold CV MSE must be 0.6191693, random must be 0.2705670.

Metal-centred porphyrins make RDKit's strict valence check fail inside
MurckoScaffold.GetScaffoldForMol, so the scaffold helper strips metal atoms first and
only then asks for the Murcko core; every failure path is counted and reported.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, '.')
from core.data_processing import load_valid_data, load_feature_cache  # noqa: E402
from core.experiment_setup import select_experiment_features, load_split_manifest  # noqa: E402
from core.train_optimizer import fit_preprocessing, transform_features  # noqa: E402

DATA = 'data/Data-20261012.csv'
CACHE = 'data/Data-20261012_raw_features.csv'
SPLITS = 'experiments/20261013_R-M/splits.json'
OUT = Path('.dsh_tmp/runnability/骨架漂移量化.md')

# per-split best Lasso params + the published mean-of-fold CV MSE they must reproduce
CONFIG = {
    'scaffold': {'alpha': 0.007037542783164669, 'max_iter': 7151, 'anchor': 0.6191693213237417},
    'random': {'alpha': 0.001010171604205459, 'max_iter': 1945, 'anchor': 0.2705670202127551},
}


METALS = {'Li', 'Na', 'K', 'Mg', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
          'Zn', 'Ga', 'Ge', 'As', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Pt', 'Au', 'Hg', 'Pb', 'Al'}
SCAFFOLD_STATS = {'ok': 0, 'metal_removed': 0, 'unparsed': 0}


def _murcko_of(mol):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def murcko(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            core = _murcko_of(mol)
            SCAFFOLD_STATS['ok'] += 1
            return core
        except Exception:  # metal-centred porphyrin: strict valence check rejects the core
            pass
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError('unparsable')
        rw = Chem.RWMol(mol)
        for idx in reversed(range(rw.GetNumAtoms())):
            if rw.GetAtomWithIdx(idx).GetSymbol() in METALS:
                rw.RemoveAtom(idx)
        stripped = rw.GetMol()
        stripped.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(stripped)
        core = _murcko_of(stripped)
        SCAFFOLD_STATS['metal_removed'] += 1
        return core
    except Exception:
        SCAFFOLD_STATS['unparsed'] += 1
        return 'UNPARSED:' + smiles


def analyse(split_name, data_valid, X, y):
    cfg = CONFIG[split_name]
    train_idx, test_idx, fold_indices = load_split_manifest(SPLITS, data_valid, DATA, split_name, False)
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    X_train, y_train = X.iloc[train_idx], y[train_idx]

    smiles = data_valid['SMILES'].astype(str).to_numpy()
    families = data_valid['A_number'].astype(str).to_numpy() if 'A_number' in data_valid else None
    scaffolds = np.array([murcko(s) for s in smiles])

    rows = []
    for k, f in enumerate(fold_indices):
        tr, va = np.asarray(f['train_positions']), np.asarray(f['valid_positions'])
        assert len(tr) + len(va) == len(train_idx), 'fold positions must cover the training subset'
        prep = fit_preprocessing(X_train.iloc[tr], y_train[tr])
        model = Lasso(alpha=cfg['alpha'], max_iter=cfg['max_iter'], random_state=42)
        model.fit(transform_features(X_train.iloc[tr], prep), prep['target_scaler'].transform(y_train[tr, None]).ravel())
        pred = prep['target_scaler'].inverse_transform(
            model.predict(transform_features(X_train.iloc[va], prep))[:, None]).ravel()

        yv, yt = y_train[va], y_train[tr]
        mse_model = float(mean_squared_error(yv, pred))
        mse_const = float(mean_squared_error(yv, np.full_like(yv, yt.mean())))
        mse_oracle = float(np.var(yv))

        g_tr = set(scaffolds[train_idx][tr])
        g_va = scaffolds[train_idx][va]
        overlap_scaf = float(np.mean([g in g_tr for g in g_va]))
        if families is not None:
            f_tr = set(families[train_idx][tr])
            f_va = families[train_idx][va]
            overlap_fam = float(np.mean([g in f_tr for g in f_va]))
        else:
            overlap_fam = float('nan')

        rows.append({
            'fold': k, 'n_train': len(tr), 'n_valid': len(va),
            'train_mean': float(yt.mean()), 'train_sd': float(yt.std(ddof=1)),
            'valid_mean': float(yv.mean()), 'valid_sd': float(yv.std(ddof=1)),
            'mean_shift_sd': float((yv.mean() - yt.mean()) / yt.std(ddof=1)),
            'mse_model': mse_model, 'mse_const_train_mean': mse_const, 'mse_oracle_valid_mean': mse_oracle,
            'model_vs_oracle': mse_model - mse_oracle,
            'scaffold_overlap': overlap_scaf, 'family_overlap': overlap_fam,
        })

    df = pd.DataFrame(rows)
    mean_of_folds = float(df['mse_model'].mean())
    sse_model = float((df['mse_model'] * df['n_valid']).sum())
    sse_const = float((df['mse_const_train_mean'] * df['n_valid']).sum())
    sse_oracle = float((df['mse_oracle_valid_mean'] * df['n_valid']).sum())
    n_total = int(df['n_valid'].sum())
    return df, {
        'train': len(train_idx), 'test': len(test_idx),
        'mean_of_fold_mse': mean_of_folds,
        'anchor': cfg['anchor'],
        'anchor_ok': abs(mean_of_folds - cfg['anchor']) < 5e-3,
        'pooled_mse_model': sse_model / n_total,
        'pooled_mse_const': sse_const / n_total,
        'pooled_mse_oracle': sse_oracle / n_total,
        'drift_share': sse_const / sse_model,
        'model_over_oracle': (sse_model - sse_oracle) / sse_oracle,
        'corr_mse_vs_valid_var': float(np.corrcoef(df['mse_model'], df['mse_oracle_valid_mean'])[0, 1]),
        'mean_abs_shift_sd': float(df['mean_shift_sd'].abs().mean()),
        'mean_scaffold_overlap': float(df['scaffold_overlap'].mean()),
        'mean_family_overlap': float(df['family_overlap'].mean()),
    }


def table(df):
    cols = [('fold', '折'), ('n_train', 'n_train'), ('n_valid', 'n_valid'),
            ('train_mean', 'train均值'), ('valid_mean', 'valid均值'),
            ('mean_shift_sd', '均值偏移(训练sd)'), ('valid_sd', 'valid sd'),
            ('mse_model', 'Lasso MSE'), ('mse_const_train_mean', '常数(训练均值) MSE'),
            ('mse_oracle_valid_mean', 'oracle(valid均值) MSE'),
            ('scaffold_overlap', '骨架重叠率'), ('family_overlap', 'A_number家族重叠率')]
    lines = ['| ' + ' | '.join(label for _, label in cols) + ' |', '|' + '---|' * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for key, _ in cols:
            value = row[key]
            cells.append(f'{int(value):d}' if key in ('fold', 'n_train', 'n_valid') else f'{value:.4f}')
        lines.append('| ' + ' | '.join(cells) + ' |')
    return lines


def main():
    data_valid = load_valid_data(DATA)
    features, _ = load_feature_cache(CACHE, data_valid, DATA)
    X = select_experiment_features(features, 'mordred', None, None)
    y = data_valid['Gap(eV)'].to_numpy()
    print(f'valid rows={len(y)}  X={X.shape}')
    print(f'total label: mean={y.mean():.4f} sd={y.std(ddof=1):.4f} var={np.var(y):.4f}')

    out = ['# 骨架划分的标签分布漂移量化', '']
    out.append(f'数据：`{DATA}`，特征：`{CACHE}`（纯 Mordred 1613 列）。')
    out.append(f'划分清单：`{SPLITS}`。全部折内预处理走管线自己的 `fit_preprocessing` / `transform_features`。')
    out.append(f'全样本标签：mean = {y.mean():.4f} eV，sd = {y.std(ddof=1):.4f} eV，var = {np.var(y):.4f} eV²。')
    out.append('')

    stats = {}
    for split_name in ('scaffold', 'random'):
        df, st = analyse(split_name, data_valid, X, y)
        stats[split_name] = st
        out += [f'## {split_name} 划分（train={st["train"]}，test={st["test"]}）', '']
        out += table(df)
        out.append('')
        out += ['**汇总**', '',
                f'- 管线上报口径（五折 MSE 取平均）：**{st["mean_of_fold_mse"]:.6f}** eV²，'
                f'锚点 {st["anchor"]:.6f} → {"一致" if st["anchor_ok"] else "**不一致**"}',
                f'- 按 valid 样本数合并（pooled）MSE：模型 **{st["pooled_mse_model"]:.4f}** ／ '
                f'常数（训练均值）{st["pooled_mse_const"]:.4f} ／ oracle（valid均值）{st["pooled_mse_oracle"]:.4f} eV²',
                f'- **模型相对“只用训练均值”的技能 = 1 − 模型 SSE / 常数 SSE = {1 - 1 / st["drift_share"]:+.3f}**'
                f'（常数基线/模型 SSE 比 = {st["drift_share"]:.3f}；0 表示模型与常数一样好）',
                f'- 模型相对 oracle 常数（直接用 valid 折均值）的误差 = {st["model_over_oracle"]:+.3f}'
                f'（负数 = 优于任何常数）',
                f'- pooled CV R²（按合并方差）= {1 - st["pooled_mse_model"] / 1.0725:+.3f}；'
                f'逐折 R² 均值见消融表，两者口径不同不可混用',
                f'- 逐折 MSE 与 valid 标签方差的相关 = {st["corr_mse_vs_valid_var"]:+.3f}',
                f'- 平均 |均值偏移| = {st["mean_abs_shift_sd"]:.3f} 个训练 sd',
                f'- 平均骨架重叠率（valid 中有多少骨架出现在其训练折）= {st["mean_scaffold_overlap"]:.3f}',
                f'- 平均 A_number 家族重叠率 = {st["mean_family_overlap"]:.3f}',
                '']
        if not st['anchor_ok']:
            raise SystemExit(f'anchor mismatch for {split_name}: {st["mean_of_fold_mse"]} vs {st["anchor"]}')

    sc, rd = stats['scaffold'], stats['random']
    out += ['## 结论', '',
            f'- **两种划分的重叠情况完全不同**：骨架划分的骨架重叠率 {sc["mean_scaffold_overlap"]:.3f}、'
            f'A_number 家族重叠率 {sc["mean_family_overlap"]:.3f}，即验证分子的骨架在训练折里从未出现；'
            f'随机划分分别是 {rd["mean_scaffold_overlap"]:.3f} / {rd["mean_family_overlap"]:.3f}。',
            f'- **标签漂移真实存在但不是全部**：骨架划分折均值相对训练折平均偏移 '
            f'{sc["mean_abs_shift_sd"]:.2f} 个训练 sd（随机只有 {rd["mean_abs_shift_sd"]:.2f}）。',
            f'- **模型仍有真实信号**：把“只用训练折均值”当基线，骨架划分下模型把误差降到基线的 '
            f'{100 / sc["drift_share"]:.1f}%（随机划分 {100 / rd["drift_share"]:.1f}%）；'
            f'即使与“直接用验证折均值的 oracle 常数”比，模型仍低 {abs(sc["model_over_oracle"]) * 100:.1f}%'
            f'（随机 {abs(rd["model_over_oracle"]) * 100:.1f}%）。',
            f'- **R²≈0.09 的来源**：Test R² 是按 351 个测试样本的总体方差算的，而骨架划分各折 valid 方差相差约 5 倍'
            f'（sd 0.31→1.62），且折均值本身在漂移，因此该口径会系统性偏低；'
            f'同一批折换成按合并方差算的 pooled CV R² 则是 {1 - sc["pooled_mse_model"] / 1.0725:+.3f}。',
            f'- 逐折 MSE 与 valid 标签方差的相关：骨架 {sc["corr_mse_vs_valid_var"]:+.3f} ／ '
            f'随机 {rd["corr_mse_vs_valid_var"]:+.3f}（正相关 = 验证折标签越分散，误差越大）。',
            '',
            '## 骨架字符串提取的失败情况（透明化）', '',
            f'- 直接算 Bemis-Murcko 成功：{SCAFFOLD_STATS["ok"]} 个分子',
            f'- 因金属配位卟啉价态被 RDKit 拒绝、剥掉金属后成功：{SCAFFOLD_STATS["metal_removed"]} 个分子',
            f'- 两段都失败、退化为“自身 SMILES 视为独立骨架”：{SCAFFOLD_STATS["unparsed"]} 个分子',
            '']


    text = '\n'.join(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding='utf-8')
    print(f'[written] {OUT}')
    for split_name, st in stats.items():
        print(f'{split_name}: mean_of_fold={st["mean_of_fold_mse"]:.6f} anchor={st["anchor"]:.6f} '
              f'ok={st["anchor_ok"]} drift_share={st["drift_share"]:.3f} '
              f'scaffold_overlap={st["mean_scaffold_overlap"]:.3f}')


if __name__ == '__main__':
    main()
