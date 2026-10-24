r"""Rebuild manuscript Figure 8 as a landscape beeswarm from the re-run SHAP values.

shap's own beeswarm is portrait, which does not fit the manuscript's picture slot
(4961255 x 3971499 EMU), so the swarm is drawn here directly from
shap_values.npy + samples.csv: one row per descriptor ordered by mean |SHAP|, dot
colour encoding the (normalised) descriptor value.

The huge positive contributions of the three C3SP3 = 4 molecules are clipped at the
axis limit and annotated, otherwise the bulk of the distribution collapses.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments' / '20261016_R-M' / 'analysis_results' / 'R-M' / 'shap_analysis' / 'XGBoost'
OUT = ROOT / 'reports' / 'figures' / 'fig8_shap_beeswarm.png'

TOP_N = 12
rng = np.random.default_rng(42)

importance = pd.read_csv(RUN / 'feature_importance.csv', encoding='utf-8')
samples = pd.read_csv(RUN / 'samples.csv', encoding='utf-8')
values = np.load(RUN / 'shap_values.npy')
assert values.shape[1] == samples.shape[1]

top = importance.head(TOP_N)
rows = list(top.itertuples())

figure, axis = plt.subplots(figsize=(6.4, 4.30), dpi=300)
try:                      # matplotlib >= 3.6
    colours = matplotlib.colormaps['coolwarm']
except Exception:         # older releases
    colours = cm.get_cmap('coolwarm')

# Clip range: keep the 99th percentile of |SHAP| so the swarm stays readable.
clip = float(np.percentile(np.abs(values[:, [samples.columns.get_loc(r.Feature) for r in rows]]), 99.5))
clip = max(clip, 0.35)
x_min = -clip
x_max = clip

for position, record in enumerate(rows):
    feature = record.Feature
    column_index = samples.columns.get_loc(feature)
    shap = values[:, column_index]
    feature_values = samples[feature].to_numpy(float)
    if feature_values.max() > feature_values.min():
        normalised = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
    else:
        normalised = np.zeros_like(feature_values)
    jitter = rng.uniform(-0.22, 0.22, size=len(shap))
    y = (len(rows) - position) + jitter
    clipped = np.clip(shap, x_min, x_max)
    axis.scatter(clipped, y, c=normalised, cmap=colours, s=13, alpha=0.9,
                 edgecolors='none', vmin=0, vmax=1)
    axis.text(x_max * 1.03, (len(rows) - position), f'{record.Mean_ABS_SHAP:.3f}',
              va='center', ha='left', fontsize=7.5, color='0.25')

axis.axvline(0.0, color='0.55', linewidth=0.8)
axis.set_yticks(range(1, len(rows) + 1))
axis.set_yticklabels([r.Feature for r in rows][::-1], fontsize=8)
axis.set_ylabel('Mordred descriptor', fontsize=13)
axis.set_ylim(0.4, len(rows) + 0.6)
axis.set_xlim(x_min - 0.02 * clip, x_max * 1.18)
axis.set_xlabel('SHAP value (eV)', fontsize=13)
axis.tick_params(axis='x', labelsize=11)
axis.set_title('Mordred-only XGBoost (top 12 by mean |SHAP|)', fontsize=9)

for spine in ('top', 'right'):
    axis.spines[spine].set_visible(False)

scalar = cm.ScalarMappable(cmap=colours, norm=plt.Normalize(0, 1))
colorbar = figure.colorbar(scalar, ax=axis, pad=0.02, fraction=0.035, aspect=30)
colorbar.set_label('Descriptor value (min → max)', fontsize=8)
colorbar.set_ticks([])

clipped_count = int(np.sum(np.abs(values[:, [samples.columns.get_loc(r.Feature) for r in rows]]) > clip))
if clipped_count:
    axis.text(0.0, -0.175,
              f'{clipped_count} contributions clipped at ±{clip:.2f} eV',
              transform=axis.transAxes, fontsize=7, color='0.35', va='top')

figure.tight_layout(pad=0.5)
figure.savefig(OUT, dpi=300)
plt.close(figure)
print(f'[written] {OUT}  clip={clip:.3f} eV  clipped={clipped_count}')
