"""Side-by-side comparison of two pipeline result directories.

Usage: python -B compare_runs.py <old_analysis_dir> <new_analysis_dir> [out.md]

Reads model_performance_summary.csv from each and writes a markdown table with the
metric deltas plus a train/test gap table. The pipeline writes Chinese headers
("CV MSE (eV\u00b2)", "Test R\u00b2", ...), so every metric is matched by a list of
candidate header names instead of a single hard-coded key.
"""
import sys
from pathlib import Path

import pandas as pd

# metric key -> candidate header substrings, in priority order
METRICS = [
    ('cv_mse', ['cv mse']),
    ('test_r2', ['test r']),
    ('test_rmse', ['test rmse']),
    ('test_mae', ['test mae']),
    ('train_r2', ['train r']),
    ('train_rmse', ['train rmse']),
    ('train_mae', ['train mae']),
]

LABELS = {
    'cv_mse': 'CV MSE (eV\u00b2)',
    'test_r2': 'Test R\u00b2',
    'test_rmse': 'Test RMSE',
    'test_mae': 'Test MAE',
    'train_r2': 'Train R\u00b2',
    'train_rmse': 'Train RMSE',
    'train_mae': 'Train MAE',
}


def load(directory):
    path = Path(directory) / 'model_performance_summary.csv'
    if not path.exists():
        raise SystemExit(f'\u7f3a\u5c11 {path}')
    df = pd.read_csv(path, encoding='utf-8')
    key = 'Model' if 'Model' in df.columns else df.columns[0]
    df = df.set_index(key).sort_index()
    resolved = {}
    for metric, needles in METRICS:
        hit = None
        for needle in needles:
            for column in df.columns:
                low = str(column).lower()
                if needle in low and not (needle == 'test r' and 'rmse' in low):
                    hit = column
                    break
            if hit is not None:
                break
        if hit is not None:
            resolved[metric] = hit
    return df, resolved


def fmt(value, digits=4):
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return '\u2014'


def main():
    old_dir, new_dir = sys.argv[1], sys.argv[2]
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(new_dir) / 'comparison_vs_20261013.md'
    old, old_map = load(old_dir)
    new, new_map = load(new_dir)

    lines = ['# \u590d\u73b0\u5bf9\u6bd4\uff08\u539f\u59cb\u8fd0\u884c vs \u672c\u673a\u91cd\u8dd1\uff09', '']
    lines.append(f'- \u539f\u59cb\u7ed3\u679c: `{old_dir}`')
    lines.append(f'- \u672c\u673a\u7ed3\u679c: `{new_dir}`')
    lines.append('')
    if not old_map or not new_map:
        lines.append(f'\u26a0 \u5217\u540d\u672a\u8bc6\u522b: old={old_map} new={new_map}')
        lines.append('')
    models = sorted(set(old.index) | set(new.index))

    for metric, _ in METRICS:
        column_label = LABELS[metric]
        lines.append(f'## {column_label}')
        lines.append('')
        lines.append('| Model | \u539f\u59cb | \u672c\u673a | \u0394 |')
        lines.append('|---|---|---|---|')
        for model in models:
            o = old.at[model, old_map[metric]] if metric in old_map and model in old.index else None
            n = new.at[model, new_map[metric]] if metric in new_map and model in new.index else None
            try:
                delta = f'{float(n) - float(o):+.4f}'
            except (TypeError, ValueError):
                delta = '\u2014'
            lines.append(f'| {model} | {fmt(o)} | {fmt(n)} | {delta} |')
        lines.append('')

    lines.append('## \u8bad\u7ec3\u96c6-\u6d4b\u8bd5\u96c6\u5dee\u8ddd\uff08\u8fc7\u62df\u5408\u4fe1\u53f7\uff0c\u8d8a\u5927\u8d8a\u4e25\u91cd\uff09')
    lines.append('')
    lines.append('| Model | \u539f\u59cb train_R\u00b2 \u2212 test_R\u00b2 | \u672c\u673a train_R\u00b2 \u2212 test_R\u00b2 |')
    lines.append('|---|---|---|')

    def gap(df, mapping, model):
        if 'train_r2' not in mapping or 'test_r2' not in mapping or model not in df.index:
            return '\u2014'
        try:
            return f"{float(df.at[model, mapping['train_r2']]) - float(df.at[model, mapping['test_r2']]):.4f}"
        except (TypeError, ValueError, KeyError):
            return '\u2014'

    for model in models:
        lines.append(f'| {model} | {gap(old, old_map, model)} | {gap(new, new_map, model)} |')
    lines.append('')

    text = '\n'.join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    print(f'[written] {out}')
    print('[resolved] old=' + repr(sorted(old_map)).encode('ascii', 'backslashreplace').decode())
    print('[resolved] new=' + repr(sorted(new_map)).encode('ascii', 'backslashreplace').decode())


if __name__ == '__main__':
    main()
