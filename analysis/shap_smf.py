r"""Run the pipeline's SHAP step for a finished run whose analysis never ran.

Background: `--shap-mode best` computes SHAP inside the full pipeline.  The S-MF run
was stopped by hand right after its GaussianProcess study was judged degenerate
(trial values pinned at ~1.2136 while the constant baseline is 1.1981), so steps 9-11
of that pipeline never executed and `analysis_results/S-MF/` holds only
`partial_model_metrics.csv`.  Retraining is unnecessary and would overwrite the tuned
models: this script rebuilds exactly the same matrices from the persisted
`prepared_data.pkl` + saved estimator, then mirrors the pipeline's SHAP block
(run_pipeline.py:415-550) so the output is compatible with the other runs.

Usage:
    python -B .dsh_tmp\shap_smf.py [--run-dir experiments\20261016_S-MF] [--model SVR]

Writes <analysis>/shap_analysis/<Model>/{shap_values.npy,samples.csv,metadata.json,
feature_importance.csv,shap_bar.*,shap_scatter.*,shap_dependence_*.*,shap_force_1.*}
and <analysis>/feature_target_correlation.csv.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.data_processing import load_valid_data, load_feature_cache
from core.experiment_setup import select_experiment_features, load_split_manifest
from core.train_optimizer import (
    clean_selected_features, transform_features, target_correlations, shap_to_ev,
)

TICK_SCALE = 1.5
FS_TICK_SMALL = 6.0 * TICK_SCALE
FS_LABEL_SMALL = 7.0 * TICK_SCALE
COLOR_BLUE = '#0F4D92'
COLOR_RED = '#B64342'
COLOR_MID = '#767676'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
    'axes.labelweight': 'bold',
    'legend.frameon': False,
    'axes.unicode_minus': False,
    'mathtext.default': 'regular',
})


def save_nature_fig(fig, path_noext):
    fig.savefig(path_noext + '.svg', bbox_inches='tight')
    fig.savefig(path_noext + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path,
                        default=Path('experiments/20261016_S-MF'))
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--model', default=None,
                        help='默认取 partial/full 指标里 CV MSE 最小的模型')
    parser.add_argument('--data', type=Path, default=Path('data/Data-20261012.csv'))
    parser.add_argument('--features', type=Path,
                        default=Path('.dsh_tmp/fp-full/Data-20261012_raw_features_fp.csv'))
    parser.add_argument('--feature-mode', default='mixed')
    parser.add_argument('--split-file', type=Path,
                        default=Path('experiments/20261013_R-M/splits.json'))
    parser.add_argument('--split-name', default='scaffold')
    return parser.parse_args()


def pick_best_model(analysis_dir):
    for name in ('model_performance_summary.csv', 'partial_model_metrics.csv'):
        path = analysis_dir / name
        if path.exists():
            frame = pd.read_csv(path, encoding='utf-8')
            best = frame.loc[frame['CV MSE (eV²)'].idxmin(), 'Model']
            return best, str(path)
    raise SystemExit('no metrics file to pick the best model from')


def main():
    args = parse_args()
    run_name = args.run_name or args.run_dir.resolve().name.split('_', 1)[-1]
    model_dir = args.run_dir / 'models' / run_name
    analysis_dir = args.run_dir / 'analysis_results' / run_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    data_valid = load_valid_data(args.data)
    features, _ = load_feature_cache(args.features, data_valid, args.data)
    X = select_experiment_features(features, args.feature_mode, None, None)
    y = data_valid['Gap(eV)'].to_numpy()
    train_idx, test_idx, _ = load_split_manifest(
        args.split_file, data_valid, args.data, args.split_name, False)
    X_train = X.iloc[train_idx]
    y_train_original = y[train_idx]

    prepared = model_dir / 'prepared_data.pkl'
    with open(prepared, 'rb') as handle:
        folds, preprocessing = pickle.load(handle)
    print(f'[prepared] folds={len(folds)} features={len(preprocessing["feature_names"])}')

    X_train_filtered = clean_selected_features(X_train, preprocessing)
    X_train_scaled = transform_features(X_train, preprocessing)
    target_scaler = preprocessing['target_scaler']

    model_name = args.model
    if model_name is None:
        model_name, source = pick_best_model(analysis_dir)
        print(f'[best] {model_name} (from {source})')
    with open(model_dir / f'{model_name}.pkl', 'rb') as handle:
        model = pickle.load(handle)
    if not hasattr(model, 'predict'):
        raise SystemExit(f'{model_name}.pkl is not an estimator: {type(model)}')

    shap_main_dir = analysis_dir / 'shap_analysis'
    (shap_main_dir / model_name).mkdir(parents=True, exist_ok=True)
    model_shap_dir = shap_main_dir / model_name

    sample_size = min(100, X_train_scaled.shape[0])
    indices = np.random.RandomState(42).choice(
        X_train_scaled.shape[0], sample_size, replace=False) \
        if X_train_scaled.shape[0] > sample_size else np.arange(X_train_scaled.shape[0])
    X_sample = X_train_scaled[indices]
    X_sample_df = X_train_filtered.iloc[indices].reset_index(drop=True)

    import shap
    np.random.seed(42)
    if model_name in ['RandomForest', 'XGBoost', 'DecisionTree', 'LightGBM']:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
            shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and not np.isscalar(expected_value):
            expected_value = expected_value[0]
    elif hasattr(model, 'coef_'):
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and not np.isscalar(expected_value):
            expected_value = float(expected_value[0])
    else:
        print('[kernel] building kmeans(25) background, this is the slow path')
        bg = shap.kmeans(X_sample, min(25, len(X_sample)))
        n_kernel = min(30, X_sample.shape[0])
        explainer = shap.KernelExplainer(model.predict, bg)
        shap_values = explainer.shap_values(
            X_sample[:n_kernel], nsamples=512, l1_reg='num_features(10)', silent=True)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and not np.isscalar(expected_value):
            expected_value = float(expected_value[0])

    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(-1, 1)

    shap_values, expected_value = shap_to_ev(shap_values, expected_value, target_scaler)
    np.save(model_shap_dir / 'shap_values.npy', shap_values)
    X_sample_df.iloc[:len(shap_values)].to_csv(model_shap_dir / 'samples.csv', index=False)
    (model_shap_dir / 'metadata.json').write_text(json.dumps({
        'unit': 'eV',
        'expected_value': float(expected_value),
        'source_rows': data_valid.iloc[train_idx[indices[:len(shap_values)]]]['source_row'].tolist(),
        'shap_mode': 'standalone (pipeline steps 9-11 replayed after S-MF was stopped)',
    }, ensure_ascii=False), encoding='utf-8')

    X_plot_df = X_sample_df.iloc[:shap_values.shape[0]]
    imp_df = pd.DataFrame({
        'Feature': X_plot_df.columns,
        'Mean_ABS_SHAP': np.abs(shap_values).mean(axis=0),
    }).sort_values('Mean_ABS_SHAP', ascending=False)
    imp_df.to_csv(model_shap_dir / 'feature_importance.csv', index=False)

    fi = imp_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    ax.barh(range(len(fi)), fi['Mean_ABS_SHAP'].values, color=COLOR_BLUE, height=0.7)
    ax.set_yticks(range(len(fi)))
    ax.set_yticklabels(fi['Feature'].values, fontsize=FS_TICK_SMALL)
    ax.set_xlabel('Mean |SHAP value| (eV)', fontsize=FS_LABEL_SMALL)
    ax.tick_params(axis='x', labelsize=FS_TICK_SMALL)
    ax.spines['left'].set_visible(True)
    fig.tight_layout(pad=0.5)
    save_nature_fig(fig, str(model_shap_dir / 'shap_bar'))

    plt.figure(figsize=(3.6, 3.3))
    shap.summary_plot(shap_values, X_plot_df, show=False, max_display=20)
    plt.gca().set_xlabel('SHAP value (eV)')
    plt.gca().tick_params(labelsize=FS_TICK_SMALL)
    plt.tight_layout(pad=0.5)
    save_nature_fig(plt.gcf(), str(model_shap_dir / 'shap_scatter'))

    mean_abs = np.abs(shap_values).mean(axis=0)
    for idx in np.argsort(mean_abs)[::-1][:3]:
        fname = X_plot_df.columns[idx]
        try:
            plt.figure(figsize=(3.3, 3.0))
            shap.dependence_plot(idx, shap_values, X_plot_df, show=False)
            plt.gca().set_ylabel(f'SHAP value for {fname} (eV)')
            plt.gca().tick_params(labelsize=FS_TICK_SMALL)
            plt.tight_layout(pad=0.5)
            save_nature_fig(plt.gcf(), str(model_shap_dir / f'shap_dependence_{fname}'))
        except Exception as exc:
            plt.close('all')
            print(f'[warn] dependence plot failed for {fname}: {exc}')

    try:
        shap.force_plot(expected_value, shap_values[0], X_plot_df.iloc[0],
                        show=False, matplotlib=True, out_names='Gap (eV)')
        plt.tight_layout(pad=0.5)
        save_nature_fig(plt.gcf(), str(model_shap_dir / 'shap_force_1'))
    except Exception as exc:
        plt.close('all')
        print(f'[warn] force plot failed: {exc}')

    corr = target_correlations(X_train_filtered, y_train_original).sort_values(ascending=False)
    corr.to_csv(analysis_dir / 'feature_target_correlation.csv')

    top = imp_df.head(15)
    print(f'[done] {model_name}: shap {shap_values.shape}, '
          f'{len(imp_df)} features ranked')
    print(top.to_string(index=False))


if __name__ == '__main__':
    main()
