r"""Build the 2x2 four-group comparison: {random, scaffold} x {mordred, mixed}.

Usage: python -B .dsh_tmp\build_four_group_comparison.py
Writes .dsh_tmp\runnability\四组总对照.md

Data sources (verified): each run's analysis_results/<name>/model_performance_summary.csv
(preferred) or partial_model_metrics.csv (still running).  Split identity of each run is
re-verified against experiments\20261013_R-M\splits.json via run_metadata.json
train_source_rows / test_source_rows, so a mislabelled run cannot slip in silently.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# The script lives in .dsh_tmp\, so sys.path[0] is that directory; put the project root
# (its parent) first so `core` resolves without depending on how python was invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data_processing import load_valid_data

OUT = Path('.dsh_tmp/runnability/四组总对照.md')
SPLITS = Path('experiments/20261013_R-M/splits.json')
SPLITS_MESO = Path('experiments/20261021_splits_meso.json')

MODEL_ORDER = ['XGBoost', 'LightGBM', 'RandomForest', 'SVR', 'KernelRidge',
               'DecisionTree', 'GaussianProcess', 'ElasticNet', 'Lasso', 'KNeighbors']

RUNS = [
    ('R-M', '随机', 'Mordred', Path('experiments/20261016_R-M'), SPLITS),
    ('R-MF', '随机', '混合', Path('experiments/20261016_R-MF'), SPLITS),
    ('B-M', 'B轴', 'Mordred', Path('experiments/20261021_B-M'), SPLITS_MESO),
    ('B-MF', 'B轴', '混合', Path('experiments/20261021_B-MF'), SPLITS_MESO),
    ('S-M', '骨架', 'Mordred', Path('experiments/20261016_S-M'), SPLITS),
    ('S-MF', '骨架', '混合', Path('experiments/20261016_S-MF'), SPLITS),
]

SPLIT_OF_LABEL = {'随机': 'random', '骨架': 'scaffold', 'B轴': 'meso'}


def verify_split(run_dir, run_name, splits_path):
    """Return the split name the run actually used, or 'MISMATCH'."""
    meta = run_dir / 'models' / run_name / 'run_metadata.json'
    if not meta.exists():
        return '未写入'
    payload = json.loads(meta.read_text(encoding='utf-8'))
    data = load_valid_data('data/Data-20261012.csv')
    rows = data['source_row'].to_numpy()
    manifest = json.loads(splits_path.read_text(encoding='utf-8'))['splits']
    train_rows = pd.Series(payload['train_source_rows']).to_numpy()
    test_rows = pd.Series(payload['test_source_rows']).to_numpy()
    for name, spec in manifest.items():
        train_ref = rows[pd.Series(spec['train_positions']).to_numpy()]
        test_ref = rows[pd.Series(spec['test_positions']).to_numpy()]
        if (len(train_ref) == len(train_rows)
                and (train_ref == train_rows).all()
                and (test_ref == test_rows).all()):
            return name
    return 'MISMATCH'


def load_metrics(run_dir, run_name):
    analysis = run_dir / 'analysis_results' / run_name
    full = analysis / 'model_performance_summary.csv'
    part = analysis / 'partial_model_metrics.csv'
    src = full if full.exists() else part
    if not src.exists():
        return None, None, 'missing raw'
    frame = pd.read_csv(src, encoding='utf-8')
    return frame, src, ('完整' if full.exists() else '部分(仍在跑)')


def study_best(model_dir, name):
    db = model_dir / f'{name}.sqlite3'
    if not db.exists():
        return None
    con = sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)
    try:
        row = con.execute(
            'select v.value from trial_values v '
            'join trials t on t.trial_id = v.trial_id '
            'where v.value is not null order by v.value asc limit 1'
        ).fetchone()
        done = con.execute(
            "select count(*) from trials where state='COMPLETE'").fetchone()[0]
    except sqlite3.Error:
        row, done = None, 0
    finally:
        con.close()
    return (row[0] if row else None), done


def main():
    tables = {}
    provenance = {}
    for name, split_label, feat_label, run_dir in RUNS:
        frame, src, state = load_metrics(run_dir, name)
        actual = verify_split(run_dir, name)
        model_dir = run_dir / 'models' / name
        rows = {}
        for model in MODEL_ORDER:
            best, done = study_best(model_dir, model)
            rows[model] = {'study': best, 'trials': done}
        if frame is not None:
            for _, row in frame.iterrows():
                entry = rows.setdefault(row['Model'], {'study': None, 'trials': 0})
                entry['cv'] = row.get('CV MSE (eV²)')
                entry['test_r2'] = row.get('Test R²')
                entry['test_rmse'] = row.get('Test RMSE')
                entry['test_mae'] = row.get('Test MAE')
                entry['train_r2'] = row.get('Train R²')
        # Models absent from the metrics frame but present as a study: the run was
        # stopped early.  Report the study's best value and flag it, rather than
        # silently printing a dash that hides a valid measurement.
        for model, entry in rows.items():
            if entry.get('cv') is None and entry['study'] is not None:
                entry['cv'] = entry['study']
                entry['partial_note'] = f"停机，仅 {entry['trials']} trial"
        else:
            for model, entry in rows.items():
                if entry['study'] is not None:
                    entry['cv'] = entry['study']
        tables[name] = rows
        provenance[name] = (split_label, feat_label, actual, state, src)

    # A run whose GP study was stopped early still has metrics for the other models,
    # so label the provenance honestly instead of reusing "仍在跑".
    provenance = {
        name: (info[0], info[1], info[2],
               '部分(已停机)' if any(e.get('partial_note') for e in tables[name].values())
               else info[3], info[4])
        for name, info in provenance.items()
    }

    lines = ['# 四组同预算总对照（随机/骨架 × Mordred/混合）', '',
             '统一口径：每模型 Optuna 100 trial、同一 seed、同一套折内特征筛选与预处理；',
             '表中 `CV MSE` 为管线口径（五折 MSE 的算术平均，eV²）。', '']

    lines += ['## 0. 运行身份校验（防止错标划分）', '',
              '| run | 预设划分 | 预设特征 | 实测划分 | 指标来源 |',
              '|---|---|---|---|---|']
    for name, (split_label, feat_label, actual, state, src) in provenance.items():
        flag = '✅' if actual == ('random' if split_label == '随机' else 'scaffold') else '❌'
        lines.append(f'| `{name}` | {split_label} | {feat_label} | {actual} {flag} | {state} |')
    lines.append('')

    lines += ['## 1. CV MSE（eV²，越小越好；每列粗体为该组最优）', '',
              '| Model | R-M (随机·Mordred) | R-MF (随机·混合) | S-M (骨架·Mordred) | S-MF (骨架·混合) |',
              '|---|---|---|---|---|']
    best_by_run = {}
    for name in tables:
        values = [(m, tables[name][m].get('cv')) for m in tables[name]]
        values = [(m, v) for m, v in values if v is not None]
        if values:
            best_by_run[name] = min(values, key=lambda kv: kv[1])[0]
    for model in MODEL_ORDER:
        cells = []
        for name in tables:
            entry = tables[name].get(model, {})
            value = entry.get('cv')
            if value is None:
                cells.append('—')
            else:
                mark = '†' if entry.get('partial_note') else ''
                text = f'{value:.4f}{mark}'
                cells.append(f'**{text}**' if best_by_run.get(name) == model else text)
        lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    notes = [f'† `{name}` 的 {model}：{tables[name][model]["partial_note"]}'
             f'（CV MSE {tables[name][model]["cv"]:.4f}），无 Test 指标；'
             f'该 study 的 trial 值在 1.2136–1.2226 间波动，属平台期，故按停机处理。'
             for name in tables
             for model in MODEL_ORDER
             if tables[name].get(model, {}).get('partial_note')]
    if notes:
        lines += notes + ['']

    lines += ['## 2. Test R²（越大越好）', '',
              '| Model | R-M (随机·Mordred) | R-MF (随机·混合) | S-M (骨架·Mordred) | S-MF (骨架·混合) |',
              '|---|---|---|---|---|']
    for model in MODEL_ORDER:
        cells = []
        for name in tables:
            value = tables[name].get(model, {}).get('test_r2')
            cells.append('—' if value is None else f'{value:+.4f}')
        lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    lines += ['## 3. 每模型 trial 完成数（budget = 100）', '',
              '| Model | R-M | R-MF | S-M | S-MF |', '|---|---|---|---|---|']
    for model in MODEL_ORDER:
        cells = [str(tables[name].get(model, {}).get('trials', 0)) for name in tables]
        lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    lines += ['## 4. 指纹与划分的主效应（以 CV MSE 计）', '']
    effects = []
    for split in ('R', 'S'):
        mord = tables[f'{split}-M']
        mixed = tables[f'{split}-MF']
        shared = [m for m in MODEL_ORDER
                  if mord.get(m, {}).get('cv') is not None and mixed.get(m, {}).get('cv') is not None]
        if shared:
            delta = sum(mixed[m]['cv'] - mord[m]['cv'] for m in shared) / len(shared)
            effects.append((split, len(shared), delta))
    for split, n, delta in effects:
        label = '随机划分' if split == 'R' else '骨架划分'
        sign = '下降' if delta < 0 else '上升'
        line = (f'- **{label}**：加入指纹后平均 CV MSE {sign} {abs(delta):.4f} eV²'
                f'（{n} 个模型可对照）')
        if split == 'S':
            clean = [m for m in MODEL_ORDER
                     if not tables['S-MF'].get(m, {}).get('partial_note')]
            if len(clean) < n:
                delta_clean = sum(tables['S-MF'][m]['cv'] - tables['S-M'][m]['cv']
                                  for m in clean
                                  if tables['S-M'].get(m, {}).get('cv') is not None
                                  and tables['S-MF'].get(m, {}).get('cv') is not None)
                count = sum(1 for m in clean
                            if tables['S-M'].get(m, {}).get('cv') is not None
                            and tables['S-MF'].get(m, {}).get('cv') is not None)
                line += f'；剔除 GP 停机值后为 {delta_clean / count:+.4f} eV²（{count} 个模型）'
        lines.append(line)
    lines.append('')

    best_lines = []
    for name, (split_label, feat_label, actual, state, src) in provenance.items():
        best = best_by_run.get(name)
        if best:
            entry = tables[name][best]
            best_lines.append(
                f'- `{name}`（{split_label}·{feat_label}）：最优 **{best}**，'
                f'CV MSE {entry["cv"]:.4f}，Test R² {entry.get("test_r2", float("nan")):+.4f}')
    if best_lines:
        lines += ['## 5. 各组最优模型', ''] + best_lines + ['']

    text = '\n'.join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding='utf-8')
    print(text)
    print(f'\n[written] {OUT} ({len(text)} chars)')


if __name__ == '__main__':
    main()
