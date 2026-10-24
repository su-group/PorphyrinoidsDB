r"""Emit the markdown tables quoted in `reports\B-取代基外推测试.md` section 8.

Numbers are read straight from each run's model_performance_summary.csv so that the
report cannot drift from the artifacts through manual transcription.

Usage: python -B .dsh_tmp\emit_b_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['KNeighbors', 'LightGBM', 'ElasticNet', 'Lasso', 'XGBoost', 'KernelRidge',
         'SVR', 'RandomForest', 'GaussianProcess', 'DecisionTree']
RUNS = [('B-M', ROOT / 'experiments' / '20261021_B-M'),
        ('B-MF', ROOT / 'experiments' / '20261021_B-MF')]


def emit(run: str, run_dir: Path) -> None:
    summary = run_dir / 'analysis_results' / run / 'model_performance_summary.csv'
    print(f'\n### {run}\n')
    if not summary.exists():
        status = run_dir / 'models' / run / 'status.json'
        phase = 'missing'
        if status.exists():
            phase = json.loads(status.read_text(encoding='utf-8')).get('phase', '?')
        print(f'(`model_performance_summary.csv` not written yet; phase={phase})')
        return
    frame = pd.read_csv(summary).set_index('Model')
    best_cv = frame['CV MSE (eV²)'].idxmin() if 'CV MSE (eV²)' in frame else None
    best_r2 = frame['Test R²'].idxmax() if 'Test R²' in frame else None
    print('| 模型 | CV MSE | Test R² | Test RMSE | Test MAE |')
    print('|---|---|---|---|---|')
    for model in ORDER:
        if model not in frame.index:
            print(f'| {model} | — | — | — | — |')
            continue
        row = frame.loc[model]
        cv = f"**{row['CV MSE (eV²)']:.4f}**" if model == best_cv else f"{row['CV MSE (eV²)']:.4f}"
        r2 = f"**{row['Test R²']:+.4f}**" if model == best_r2 else f"{row['Test R²']:+.4f}"
        print(f"| {model} | {cv} | {r2} | {row['Test RMSE']:.4f} | {row['Test MAE']:.4f} |")
    print(f'\nCV-best = {best_cv}; Test-R²-best = {best_r2}')


def main() -> int:
    for run, run_dir in RUNS:
        emit(run, run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
