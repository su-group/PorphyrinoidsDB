# Analysis

Five scripts provide result summaries, feature ablations, and SHAP summaries. Outputs are CSV tables and optional PNG figures under `analysis_outputs/`. Run commands from the repository root.

## Environment

See the [installation instructions](../README.md#setup). Activate `porscadb-ml` for plotting and ablations. The two summary scripts need only the Python standard library when exporting CSV; `--plot` additionally requires Matplotlib.

## Result summary

```bash
python analysis/results/summarize.py
```

Reads the saved `model_performance_summary.csv` files and writes `analysis_outputs/results/model_metrics.csv`. Each row includes the run name and source path. Original metric columns and values are preserved; models are not refitted or selected by their test scores.

Use `--experiments PATH` to choose the experiment directory and `--output PATH` to choose the output directory. This script uses only the Python standard library.

## SHAP summary

```bash
python analysis/features/summarize_shap.py
python analysis/features/summarize_shap.py --plot --top 20
```

Reads saved `feature_importance.csv` files and writes:

- `shap_features.csv`: mean absolute SHAP importance by run, model, and feature.
- `shap_families.csv`: summed importance and its fraction of total importance by feature family.
- With `--plot`: a top-feature PNG for each saved SHAP model.

Family fractions are calculated from summed mean absolute importances. They are not averages of per-sample normalized SHAP fractions. The script summarizes saved values and does not recalculate SHAP. CSV export uses only the Python standard library; plotting additionally requires Matplotlib.

## Feature ablations

| Script | Output in `analysis_outputs/features/` |
|---|---|
| `features/fp_random_ablation.py` | `random_ablation.csv` |
| `features/fp_scaffold_ablation.py` | `scaffold_ablation.csv` |
| `features/fp_family_diagnostic.py` | `family_ablation.csv` |

These scripts fit models. Configure their input paths and model settings before running them in the training environment. Random and scaffold ablations export pooled CV metrics; the family diagnostic exports individual fold metrics. Output-format changes do not change their fitting or scoring logic.

The molecular data is documented in [`data/README.md`](../data/README.md), and saved runs in [`experiments/README.md`](../experiments/README.md).
