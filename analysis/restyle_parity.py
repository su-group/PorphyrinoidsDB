r"""Re-render every true-vs-predicted parity plot in the density-shaded reference style.

The pipeline already shades by local density (viridis + gaussian_kde), but its
panels use plain axis text, grey metrics and data-driven limits.  This rewrites the
35 panels of the four runs (S-MF has no GaussianProcess) to match Figure 7:

  * boxed axes, outward ticks, bold axis titles "ML-Predicted Gap (eV)" /
    "DFT-Computed Gap (eV)"
  * bold-italic R2 / RMSE / MAE block in the lower-right corner
  * bold model name inside the upper-left corner
  * ONE shared axis window across all four runs, so the collapse under the
    scaffold split is visible instead of being rescaled away

Originals are copied to .dsh_tmp\parity_original\<run>\ before being overwritten.
Metrics are recomputed from test_predictions.csv, never transcribed.

Usage: python -B .dsh_tmp\restyle_parity.py
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = ['R-M', 'R-MF', 'B-M', 'B-MF', 'S-M', 'S-MF']
MODELS = ['XGBoost', 'LightGBM', 'RandomForest', 'SVR', 'KernelRidge',
          'GaussianProcess', 'ElasticNet', 'Lasso', 'DecisionTree', 'KNeighbors']
BACKUP = ROOT / '.dsh_tmp' / 'parity_original'
RUN_DIRS = {
    'R-M': '20261016_R-M', 'R-MF': '20261016_R-MF',
    'S-M': '20261016_S-M', 'S-MF': '20261016_S-MF',
    'B-M': '20261021_B-M', 'B-MF': '20261021_B-MF',
}


def analysis_dir(run: str) -> Path:
    return ROOT / 'experiments' / RUN_DIRS[run] / 'analysis_results' / run


def load_frame(run: str) -> pd.DataFrame:
    return pd.read_csv(analysis_dir(run) / 'test_predictions.csv')


def main() -> int:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        'axes.linewidth': 0.9,
        'axes.unicode_minus': False,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
    })
    viridis = matplotlib.colormaps['viridis']

    frames = {run: load_frame(run) for run in RUNS}

    low, high = np.inf, -np.inf
    for run, frame in frames.items():
        truth = frame['Gap(eV)'].to_numpy(dtype=float)
        low, high = min(low, truth.min()), max(high, truth.max())
        for model in MODELS:
            if model in frame.columns:
                values = frame[model].to_numpy(dtype=float)
                low, high = min(low, values.min()), max(high, values.max())
    margin = (high - low) * 0.06
    window = (low - margin, high + margin)
    print(f'shared window: {window[0]:.3f} .. {window[1]:.3f}')

    written, skipped = 0, 0
    for run, frame in frames.items():
        truth = frame['Gap(eV)'].to_numpy(dtype=float)
        backup_dir = BACKUP / run
        backup_dir.mkdir(parents=True, exist_ok=True)
        for model in MODELS:
            if model not in frame.columns:
                skipped += 1
                continue
            predicted = frame[model].to_numpy(dtype=float)
            residual = truth - predicted
            r2 = 1.0 - float(np.sum(residual ** 2)) / float(np.sum((truth - truth.mean()) ** 2))
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            mae = float(np.mean(np.abs(residual)))

            target = analysis_dir(run) / f'{model}_true_vs_predicted'
            for suffix in ('.png', '.svg'):
                source = target.with_suffix(suffix)
                copy = backup_dir / source.name
                if source.exists() and not copy.exists():
                    shutil.copy2(source, copy)

            try:
                density = gaussian_kde(np.vstack([truth, predicted]))(np.vstack([truth, predicted]))
                density = (density - density.min()) / max(density.max() - density.min(), 1e-12)
            except (ValueError, np.linalg.LinAlgError):
                density = np.ones(len(truth))
            order = np.argsort(density)

            figure, axis = plt.subplots(figsize=(3.5, 3.4))
            axis.plot(window, window, linestyle=(0, (4, 3)), color='#9E9E9E',
                      linewidth=0.9, zorder=1)
            axis.scatter(truth[order], predicted[order], s=14,
                         c=viridis(density[order] ** 0.85), linewidths=0, zorder=2)
            axis.set_xlim(*window)
            axis.set_ylim(*window)
            axis.tick_params(direction='out', length=3.0, width=0.8, labelsize=7.5, pad=2.5)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9)

            axis.set_xlabel('DFT-Computed Gap (eV)', fontsize=8.5, fontweight='bold', labelpad=4)
            axis.set_ylabel('ML-Predicted Gap (eV)', fontsize=8.5, fontweight='bold', labelpad=4)
            axis.text(0.05, 0.95, model, transform=axis.transAxes, fontsize=8.5,
                      fontweight='bold', va='top', ha='left')
            axis.text(0.97, 0.05,
                      f'R\u00b2 = {r2:.4f}\nRMSE = {rmse:.4f} eV\nMAE = {mae:.4f} eV',
                      transform=axis.transAxes, ha='right', va='bottom', fontsize=7.5,
                      fontweight='bold', fontstyle='italic', linespacing=1.4)
            figure.tight_layout(pad=0.4)
            figure.savefig(target.with_suffix('.svg'), bbox_inches='tight')
            figure.savefig(target.with_suffix('.png'), dpi=300, bbox_inches='tight')
            plt.close(figure)
            written += 1
        print(f'  {run}: done')

    print(f'[written] {written} panels, skipped {skipped} (no such model column)')
    print(f'[backup]  originals in {BACKUP}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
