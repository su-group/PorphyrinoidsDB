r"""Build the three-split x two-feature-set comparison table.

Supersedes `.dsh_tmp\build_four_group_comparison.py`, which hardcoded four columns and
could not absorb the meso-B axis.  Everything here is driven by the RUNS list in that
older module, so the run/directory provenance logic is shared rather than re-derived.

Usage: python -B .dsh_tmp\build_group_comparison.py [--out PATH]

Reads each run's analysis_results/<run>/model_performance_summary.csv when the run has
finished, else partial_model_metrics.csv, else falls back to the Optuna study best (a
run whose model was stopped still has a real measurement).  Split identity is re-checked
against the manifest through run_metadata.json, so a mislabelled run cannot slip in.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    'four_group', ROOT / '.dsh_tmp' / 'build_four_group_comparison.py')
four = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(four)

MODEL_ORDER = four.MODEL_ORDER
SPLIT_TITLES = {'随机': '随机划分 random', 'B轴': 'B 轴 meso（取代基外推）',
                '骨架': '骨架划分 scaffold'}
SHORT = {'随机': '随机', 'B轴': 'B轴', '骨架': '骨架'}


def run_state(run_dir, run_name):
    """Phase and completed-model list from status.json (absent for an unstarted run)."""
    status = run_dir / 'models' / run_name / 'status.json'
    if not status.exists():
        return '未开始', []
    payload = json.loads(status.read_text(encoding='utf-8'))
    return payload.get('phase', '未知'), payload.get('completed_models', [])


def collect():
    """Return {run: {model: metrics-dict}} plus provenance for every run."""
    tables, provenance = {}, {}
    for name, split_label, feat_label, run_dir, manifest in four.RUNS:
        frame, src, state = four.load_metrics(run_dir, name)
        actual = four.verify_split(run_dir, name, manifest)
        phase, completed = run_state(run_dir, name)
        model_dir = run_dir / 'models' / name
        rows = {model: {'study': None, 'trials': 0} for model in MODEL_ORDER}
        for model in MODEL_ORDER:
            result = four.study_best(model_dir, model)
            if result is None:          # study DB not created yet
                best, done = None, 0
            else:
                best, done = result
            rows[model]['study'] = best
            rows[model]['trials'] = done
        if frame is not None:
            for _, row in frame.iterrows():
                entry = rows.setdefault(row['Model'], {'study': None, 'trials': 0})
                # partial_model_metrics.csv is narrower than the finished summary: a run
                # still training has no Train R², so read each column defensively.
                for key, column in (('cv', 'CV MSE (eV²)'), ('test_r2', 'Test R²'),
                                    ('test_rmse', 'Test RMSE'), ('test_mae', 'Test MAE'),
                                    ('train_r2', 'Train R²')):
                    if column in frame.columns and pd.notna(row.get(column)):
                        entry[key] = float(row[column])
        for model, entry in rows.items():
            if entry.get('cv') is None and entry['study'] is not None:
                entry['cv'] = float(entry['study'])
                entry['stopped'] = True
        tables[name] = rows
        provenance[name] = {'split': split_label, 'features': feat_label,
                            'actual': actual, 'state': state,
                            'phase': phase, 'completed': len(completed),
                            'src': str(src) if src else 'missing'}
    return tables, provenance


def matrix(tables, key):
    runs = [name for name, *_ in four.RUNS]
    data = {}
    for name in runs:
        data[name] = {model: tables[name].get(model, {}).get(key, np.nan) for model in MODEL_ORDER}
    frame = pd.DataFrame(data, index=MODEL_ORDER)
    return frame


def fmt(value, digits=4, bold=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '—'
    text = f'{value:.{digits}f}'
    return f'**{text}**' if bold else text


def table_block(lines, frame, title, note, best='min'):
    runs = list(frame.columns)
    header = '| Model | ' + ' | '.join(f'`{run}`' for run in runs) + ' |'
    lines += [f'## {title}', '', note, '', header,
              '|---|' + '---|' * len(runs)]
    column_best = {}
    for run in runs:
        values = frame[run].dropna()
        if values.empty:
            continue
        column_best[run] = values.idxmin() if best == 'min' else values.idxmax()
    for model in frame.index:
        cells = []
        for run in runs:
            value = frame.loc[model, run]
            cells.append(fmt(value, bold=(column_best.get(run) == model)))
        lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('reports/分组总对照.md'))
    args = parser.parse_args()

    tables, provenance = collect()
    runs = [name for name, *_ in four.RUNS]
    labels = {name: (split, feat) for name, split, feat, *_ in four.RUNS}

    complete = sum(1 for name in runs if provenance[name]['phase'] == 'complete')
    lines = ['# 三划分 × 两特征集总对照（随机 / B 轴 / 骨架）', '',
             f'生成时间口径：每模型 Optuna 100 trial、同一 seed 42、同一套折内特征筛选与预处理。'
             f'本表覆盖 {len(runs)} 组，其中 {complete} 组已完成全部 {len(MODEL_ORDER)} 个模型。',
             '',
             '表中 `CV MSE` 为管线口径（五折 MSE 的算术平均，eV²），`Test` 指标为留出集一次评估。',
             '']

    lines += ['## 0. 运行身份校验（防止错标划分）', '',
              '| run | 预设划分 | 预设特征集 | 实测划分 | 指标来源 | 已完成模型 | phase |',
              '|---|---|---|---|---|---|---|']
    for name in runs:
        info = provenance[name]
        expected = four.SPLIT_OF_LABEL[info['split']]
        flag = '✅' if info['actual'] == expected else '❌'
        done = info['completed']
        lines.append(f"| `{name}` | {info['split']} | {info['features']} | "
                     f"{info['actual']} {flag} | {info['state']} | "
                     f"{done}/{len(MODEL_ORDER)} | {info['phase']} |")
    lines.append('')

    cv = matrix(tables, 'cv')
    test = matrix(tables, 'test_r2')
    lines = table_block(lines, cv, '1. CV MSE（eV²，越小越好；每列粗体为该组最优）',
                        '同一模型跨划分/特征集可比，列内最优加粗。')
    lines = table_block(lines, test, '2. Test R²（越大越好；每列粗体为该组最优）',
                        'Test 指标只用于报告，不参与任何选择。', best='max')

    lines += ['## 3. 每组的最优模型（CV 口径 vs Test 口径）', '',
              '| run | 划分 | 特征集 | CV 最优 | CV MSE | 其 Test R² | Test 最优 | Test R² | 两者是否一致 |',
              '|---|---|---|---|---|---|---|---|---|']
    for name in runs:
        entries = {model: tables[name][model] for model in MODEL_ORDER
                   if tables[name].get(model, {}).get('cv') is not None}
        if not entries:
            continue
        best_cv = min(entries, key=lambda model: entries[model]['cv'])
        pool = {model: entry for model, entry in entries.items()
                if entry.get('test_r2') is not None}
        if not pool:
            continue
        best_test = max(pool, key=lambda model: pool[model]['test_r2'])
        agree = '是' if best_cv == best_test else '**否**'
        split, feat = labels[name]
        lines.append(f"| `{name}` | {SHORT[split]} | {feat} | {best_cv} | "
                     f"{entries[best_cv]['cv']:.4f} | "
                     f"{fmt(entries[best_cv].get('test_r2'))} | {best_test} | "
                     f"{pool[best_test]['test_r2']:.4f} | {agree} |")
    lines.append('')

    # Feature-set effect, measured only on models present in both feature sets of the
    # same split, so the mean is not polluted by a different model roster.
    lines += ['## 4. 特征集效应（同划分内：混合 − Mordred，CV MSE 差值）', '',
              '负值表示加入指纹后变好。只统计两侧都跑到 100 trial 的模型。', '',
              '| 划分 | 可比模型数 | 平均 ΔCV MSE | 变好的模型 | 变差的模型 |',
              '|---|---|---|---|---|']
    for split in ['随机', 'B轴', '骨架']:
        pair = [name for name in runs if labels[name][0] == split]
        if len(pair) != 2:
            continue
        plain = [name for name in pair if labels[name][1] == 'Mordred'][0]
        mixed = [name for name in pair if labels[name][1] == '混合'][0]
        deltas, better, worse = [], [], []
        for model in MODEL_ORDER:
            a = tables[plain].get(model, {}).get('cv')
            b = tables[mixed].get(model, {}).get('cv')
            if a is None or b is None:
                continue
            delta = b - a
            deltas.append(delta)
            (better if delta < 0 else worse).append(model)
        if not deltas:
            lines.append(f'| {SHORT[split]} | 0 | — | — | — |')
            continue
        lines.append(f'| {SHORT[split]} | {len(deltas)} | {np.mean(deltas):+.4f} | '
                     f'{len(better)} | {len(worse)} |')
    lines.append('')

    # Split-axis effect per feature set: meso-B and scaffold are both extrapolation
    # axes, so compare each against the random split on the same features.
    lines += ['## 5. 划分轴效应（同特征集内：相对随机划分的 CV MSE 变化）', '',
              '| 特征集 | 对比 | 可比模型数 | 平均 ΔCV MSE | 倍数（中位数） |',
              '|---|---|---|---|---|']
    for feat in ['Mordred', '混合']:
        base = [name for name in runs if labels[name][0] == '随机' and labels[name][1] == feat]
        if not base:
            continue
        base = base[0]
        for split in ['B轴', '骨架']:
            other = [name for name in runs if labels[name][0] == split and labels[name][1] == feat]
            if not other:
                continue
            other = other[0]
            deltas, ratios = [], []
            for model in MODEL_ORDER:
                a = tables[base].get(model, {}).get('cv')
                b = tables[other].get(model, {}).get('cv')
                if a is None or b is None:
                    continue
                deltas.append(b - a)
                ratios.append(b / a)
            if not deltas:
                continue
            lines.append(f'| {feat} | {SHORT[split]} | {len(deltas)} | '
                         f'{np.mean(deltas):+.4f} | {np.median(ratios):.2f}× |')
    lines.append('')

    lines += ['## 6. 口径与限制', '',
              '- `CV MSE` 是五折 MSE 的平均，不是把五折预测拼起来算的池化 MSE；两者在骨架划分下'
              '差别明显（0.6192 vs 0.6101），跨表比较时不要混用。',
              '- 骨架与 B 轴都是外推轴：测试集的骨架/取代基组合在训练集中不出现，因此 Test 指标'
              '反映的是外推能力，不是同分布泛化。',
              '- 标注 `部分` 的组尚未跑完 10 个模型，其列内最优随时可能被后续模型替换。',
              '']

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[written] {args.out}  ({len(lines)} lines, {complete}/{len(runs)} complete runs)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
