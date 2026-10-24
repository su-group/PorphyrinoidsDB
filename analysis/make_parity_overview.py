r"""Four-panel overview: the best model of each run, same density-shaded style.

Best test-set model per run (from the four-group comparison):
  R-M  XGBoost (R2 0.9321)   R-MF SVR  (0.9565)
  S-M  RandomForest (0.3741) S-MF SVR  (0.2843)

All four panels share one axis window so the scaffold collapse is not rescaled
away.  Metrics are recomputed from test_predictions.csv.

Usage: python -B .dsh_tmp\make_parity_overview.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_ORDER = [
    ('R-M', 'random split · Mordred'),
    ('R-MF', 'random split · concatenated'),
    ('B-M', 'meso-B split · Mordred'),
    ('B-MF', 'meso-B split · concatenated'),
    ('S-M', 'scaffold split · Mordred'),
    ('S-MF', 'scaffold split · concatenated'),
]
RUN_DIRS = {
    'R-M': '20261016_R-M', 'R-MF': '20261016_R-MF',
    'S-M': '20261016_S-M', 'S-MF': '20261016_S-MF',
    'B-M': '20261021_B-M', 'B-MF': '20261021_B-MF',
}
OUT = ROOT / 'reports' / 'figures' / 'parity_overview.png'


def analysis_dir(run: str) -> Path:
    return ROOT / 'experiments' / RUN_DIRS[run] / 'analysis_results' / run


def best_panel(run: str) -> tuple[str, str]:
    """Return (model, caption) for the run's best test-set model.

    The model is read from model_performance_summary.csv rather than hardcoded, so a
    re-run cannot silently leave the overview showing a model that is no longer best.
    """
    summary = pd.read_csv(analysis_dir(run) / 'model_performance_summary.csv', encoding='utf-8')
    best = summary.loc[summary['Test R²'].idxmax(), 'Model']
    caption = dict(RUN_ORDER)[run]
    return str(best), caption


def main() -> int:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.linewidth': 0.9,
        'axes.unicode_minus': False,
    })
    viridis = matplotlib.colormaps['viridis']

    panels = [(run,) + best_panel(run) for run, _ in RUN_ORDER]
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(10.4, 3.3 * rows), dpi=300)
    low, high = np.inf, -np.inf
    data = {}
    for run, model, caption in panels:
        frame = pd.read_csv(analysis_dir(run) / 'test_predictions.csv')
        truth = frame['Gap(eV)'].to_numpy(dtype=float)
        predicted = frame[model].to_numpy(dtype=float)
        data[(run, model)] = (truth, predicted)
        low = min(low, truth.min(), predicted.min())
        high = max(high, truth.max(), predicted.max())
    margin = (high - low) * 0.06
    window = (low - margin, high + margin)

    for axis, (run, model, caption) in zip(axes.ravel(), panels):
        truth, predicted = data[(run, model)]
        residual = truth - predicted
        r2 = 1.0 - float(np.sum(residual ** 2)) / float(np.sum((truth - truth.mean()) ** 2))
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        mae = float(np.mean(np.abs(residual)))
        density = gaussian_kde(np.vstack([truth, predicted]))(np.vstack([truth, predicted]))
        density = (density - density.min()) / max(density.max() - density.min(), 1e-12)
        order = np.argsort(density)

        axis.plot(window, window, linestyle=(0, (4, 3)), color='#9E9E9E', linewidth=0.9, zorder=1)
        axis.scatter(truth[order], predicted[order], s=9, c=viridis(density[order] ** 0.85),
                     linewidths=0, zorder=2)
        axis.set_xlim(*window)
        axis.set_ylim(*window)
        axis.tick_params(direction='out', length=3.0, width=0.8, labelsize=8, pad=2.5)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
        axis.set_xlabel('DFT-Computed Gap (eV)', fontsize=9, fontweight='bold', labelpad=3)
        axis.set_ylabel('ML-Predicted Gap (eV)', fontsize=9, fontweight='bold', labelpad=3)
        axis.text(0.05, 0.95, f'{run} \u00b7 {model}', transform=axis.transAxes, fontsize=8.5,
                  fontweight='bold', va='top', ha='left')
        axis.text(0.05, 0.86, caption, transform=axis.transAxes, fontsize=7.5,
                  va='top', ha='left', style='italic')
        axis.text(0.97, 0.05,
                  f'R\u00b2 = {r2:.4f}\nRMSE = {rmse:.4f} eV\nMAE = {mae:.4f} eV',
                  transform=axis.transAxes, ha='right', va='bottom', fontsize=8,
                  fontweight='bold', fontstyle='italic', linespacing=1.4)

    figure.tight_layout(pad=0.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=300)
    plt.close(figure)
    print(f'[written] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
