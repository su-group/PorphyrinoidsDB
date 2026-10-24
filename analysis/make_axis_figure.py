r"""Grouped bar chart: the same models on the three split axes, same protocol.

Reads .dsh_tmp\baxis_lobo_cv.json (meso) and .dsh_tmp\split_reference_cv.json (random,
scaffold) and writes reports\figures\axis_split_comparison.png (+ .svg).

Usage: python -B .dsh_tmp\make_axis_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOBO = ROOT / '.dsh_tmp' / 'baxis_lobo_cv.json'
REF = ROOT / '.dsh_tmp' / 'split_reference_cv.json'
OUT = ROOT / 'reports' / 'figures' / 'axis_split_comparison.png'

MODEL_ORDER = ['RandomForest', 'XGBoost', 'KNeighbors', 'Lasso']
SHORT = {'RandomForest': 'RF', 'XGBoost': 'XGB',
         'KNeighbors': 'KNN', 'Lasso': 'Lasso'}
AXES = [('random', 'random 5-fold\n(interpolation ceiling)', '#4C72B0'),
        ('meso', 'meso leave-one-group-out', '#DD8452'),
        ('scaffold', 'scaffold GroupKFold\n(extrapolation floor)', '#C44E52')]


def main() -> int:
    lobo = json.loads(LOBO.read_text(encoding='utf-8'))
    ref = json.loads(REF.read_text(encoding='utf-8'))

    plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.8,
                         'svg.fonttype': 'none', 'axes.unicode_minus': False})
    figure, axis = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    positions = np.arange(len(MODEL_ORDER))
    width = 0.26
    for offset, (key, label, colour) in enumerate(AXES):
        values = [lobo['models'][name]['pooled_mse'] if key == 'meso'
                  else ref['models'][name]['axes'][key]['pooled_mse']
                  for name in MODEL_ORDER]
        bars = axis.bar(positions + (offset - 1) * width, values, width,
                        label=label, color=colour, edgecolor='white', linewidth=0.4)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value * 1.02, f'{value:.2f}',
                      ha='center', va='bottom', fontsize=5.2, rotation=90)

    axis.set_yscale('log')
    axis.set_ylabel('pooled CV MSE (eV$^2$, log scale)')
    axis.set_xticks(positions)
    axis.set_xticklabels([SHORT[n] for n in MODEL_ORDER])
    axis.set_ylim(0.1, 3.2)
    axis.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0])
    axis.set_yticklabels(['0.1', '0.2', '0.5', '1', '2'])
    axis.minorticks_off()
    axis.axhline(1.0725, color='#999999', linestyle=':', linewidth=0.7)
    from matplotlib.lines import Line2D
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='#999999', linestyle=':', linewidth=0.7))
    labels.append('constant baseline')

    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    axis.legend(handles, labels, fontsize=5.2, frameon=False,
                loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2,
                handlelength=1.4, columnspacing=0.9)
    figure.tight_layout(pad=0.3)
    for suffix in ('png', 'svg'):
        path = OUT.with_suffix(f'.{suffix}')
        figure.savefig(path, dpi=300, bbox_inches='tight')
        print(f'[written] {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
