# Experiment outputs

This folder contains six archived runs comparing two feature sets across three data-splitting axes.

| Run | Split | Features |
|---|---|---|
| `20261016_R-M` | Random | Mordred descriptors |
| `20261016_R-MF` | Random | Mordred descriptors and fingerprints |
| `20261016_S-M` | Scaffold | Mordred descriptors |
| `20261016_S-MF` | Scaffold | Mordred descriptors and fingerprints |
| `20261021_B-M` | Meso substituent | Mordred descriptors |
| `20261021_B-MF` | Meso substituent | Mordred descriptors and fingerprints |

Each run contains model files, evaluation outputs, and logs:

```text
<run>/
├── models/             # Estimators, preprocessing, and Optuna study files
├── analysis_results/   # Metrics, predictions, plots, and SHAP outputs
└── logs/               # Run logs
```

Pickled estimators require compatible Python and package versions; see each run's `run_metadata.json`.

## Related resources

The molecular dataset is [`PorDB-20261018.csv`](../data/PorDB-20261018.csv), with structures in [`data/XYZ/`](../data/XYZ/). Field descriptions are provided in the [data README](../data/README.md).

Each run has a `run_metadata.json` file recording input information and software versions. Use the saved metrics and predictions to inspect its results.

See [`analysis/README.md`](../analysis/README.md) for result-summary and feature/SHAP scripts.
