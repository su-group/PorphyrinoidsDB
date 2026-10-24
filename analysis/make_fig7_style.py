r"""Figure 7 in the style of the reference scatter plot (density-shaded parity).

Two panels, one per feature set, XGBoost test predictions:
  * points coloured by local density (viridis: dark purple sparse -> yellow dense)
  * grey dashed identity line, boxed axes, bold ticks/labels
  * R2 / RMSE / MAE in the lower-right corner, bold italic

Writes reports\figures\fig7_xgboost_true_vs_predicted.png sized to the manuscript
figure box (5.77 x 2.69 in, aspect 2.147).

Usage: python -B .dsh_tmp\make_fig7_style.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PANELS = [
    ('R-M', 'Mordred-only', 'a)'),
    ('R-MF', 'Concatenated Descriptors', 'b)'),
]
MODEL = 'XGBoost'
OUT = ROOT / 'reports' / 'figures' / 'fig7_xgboost_true_vs_predicted.png'


def load_predictions(run):
    path = ROOT / 'experiments' / f'20261016_{run}' / 'analysis_results' / run / 'test_predictions.csv'
    frame = pd.read_csv(path)
    return frame['Gap(eV)'].to_numpy(dtype=float), frame[MODEL].to_numpy(dtype=float)


def main() -> int:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scipy.stats import gaussian_kde

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.linewidth': 0.9,
        'axes.unicode_minus': False,
        'mathtext.fontset': 'dejavusans',
        'mathtext.default': 'bfit',
    })
    viridis = matplotlib.colormaps['viridis']

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), dpi=300)
    for axis, (run, title, letter) in zip(axes, PANELS):
        truth, predicted = load_predictions(run)
        residual = truth - predicted
        r2 = 1.0 - float(np.sum(residual ** 2)) / float(np.sum((truth - truth.mean()) ** 2))
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        mae = float(np.mean(np.abs(residual)))

        density = gaussian_kde(np.vstack([truth, predicted]))(np.vstack([truth, predicted]))
        density = (density - density.min()) / (density.max() - density.min())
        order = np.argsort(density)
        colours = viridis(density[order] ** 0.85)

        axis.plot([2, 12], [2, 12], linestyle=(0, (4, 3)), color='#9E9E9E', linewidth=1.0, zorder=1)
        axis.scatter(truth[order], predicted[order], s=16, c=colours, linewidths=0, zorder=2)

        axis.set_xlim(2, 12)
        axis.set_ylim(2, 12)
        axis.set_xticks(range(2, 13, 2))
        axis.set_yticks(range(2, 13, 2))
        axis.tick_params(direction='out', length=3.5, width=0.9, labelsize=10, pad=3)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontweight('normal')
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)

        axis.set_xlabel('DFT-Computed Gap (eV)', fontsize=11, fontweight='bold', labelpad=5)
        axis.set_ylabel('ML-Predicted Gap (eV)', fontsize=11, fontweight='bold', labelpad=5)
        axis.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=6)

        axis.text(0.97, 0.04,
                  f'R\u00b2 = {r2:.4f}\nRMSE = {rmse:.4f} eV\nMAE = {mae:.4f} eV',
                  transform=axis.transAxes, ha='right', va='bottom',
                  fontsize=10.5, fontweight='bold', fontstyle='italic', linespacing=1.45)

        axis.text(-0.20, 1.08, letter, transform=axis.transAxes,
                  fontsize=14, fontweight='bold', va='top', ha='left')

    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.88, wspace=0.28)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=300)
    plt.close(figure)
    print(f'[written] {OUT}')

    from PIL import Image
    with Image.open(OUT) as image:
        print(f'  size {image.size[0]}x{image.size[1]}  aspect {image.size[0] / image.size[1]:.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
