r"""Rebuild manuscript Figure 7 from the current runs.

Left panel: XGBoost with Mordred descriptors (R-M).  Right panel: XGBoost with the
concatenated descriptor set (R-MF).  Both panels share axis limits so the visual
comparison is honest, and R2/RMSE are recomputed from the stored test predictions
rather than copied, so the annotation cannot drift from the underlying data.

Output aspect ratio is fixed to the manuscript's picture slot (5274310 x 2456180 EMU).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'figures' / 'fig7_xgboost_true_vs_predicted.png'

PANELS = [
    ('(a) Mordred-only descriptors',
     ROOT / 'experiments' / '20261016_R-M' / 'analysis_results' / 'R-M' / 'test_predictions.csv'),
    ('(b) Concatenated descriptors',
     ROOT / 'experiments' / '20261016_R-MF' / 'analysis_results' / 'R-MF' / 'test_predictions.csv'),
]

frames = []
for label, path in PANELS:
    frame = pd.read_csv(path, encoding='utf-8')
    observed = frame['Gap(eV)'].to_numpy(float)
    predicted = frame['XGBoost'].to_numpy(float)
    residual = observed - predicted
    r2 = 1.0 - np.sum(residual ** 2) / np.sum((observed - observed.mean()) ** 2)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    frames.append((label, observed, predicted, r2, rmse, len(observed)))

low = min(min(f[1].min(), f[2].min()) for f in frames)
high = max(max(f[1].max(), f[2].max()) for f in frames)
pad = 0.25 * (high - low)
limits = (low - pad, high + pad)

figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), dpi=300)
for axis, (label, observed, predicted, r2, rmse, count) in zip(axes, frames):
    axis.scatter(observed, predicted, s=11, alpha=0.65, edgecolors='none', color='#2f6fb5')
    axis.plot(limits, limits, '--', color='#b03a2e', linewidth=1.0)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect('equal')
    axis.set_xlabel('DFT HOMO–LUMO gap (eV)', fontsize=9)
    axis.set_ylabel('Predicted gap (eV)', fontsize=9)
    axis.tick_params(labelsize=8)
    axis.set_title(label, fontsize=9)
    axis.text(0.04, 0.96,
              f'$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f} eV\nn = {count}',
              transform=axis.transAxes, va='top', ha='left', fontsize=8,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                        edgecolor='0.7', linewidth=0.5))
    print(f'{label}: R2={r2:.4f} RMSE={rmse:.4f} n={count}')

figure.tight_layout(pad=0.6)
figure.savefig(OUT, dpi=300)
plt.close(figure)
print(f'[written] {OUT}')
