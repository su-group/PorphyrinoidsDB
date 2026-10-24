# `analysis/` — the scripts behind the reports

About 6,300 lines of standalone scripts. This directory is **not a library and not an importable
package**: every file is a script you run from the package root, most of them print a table or write
a markdown/JSON artifact that was then pasted into [../reports/](../reports/). They are shipped
because a reader should be able to check any claim in the paper against the code that produced it.

```bash
python -B analysis/build_group_comparison.py      # prints the 3 x 2 table
python -B analysis/verify_reviewer_claims.py      # re-derives four reviewer-facing claims
```

## Environment

| Group of scripts | Needs |
| --- | --- |
| Most metric/table/figure scripts | `environment-ml.yml` (scikit-learn, xgboost, lightgbm, shap, matplotlib) |
| `fp_*`, `count_feature_families`, `shap_smf` (fingerprint cache) | also `environment-features.yml` (mordred 1.2.0, rdkit) to read the fingerprint cache |
| `check_cache_integrity` | rdkit only |

Scripts read only what this package ships, with the exceptions listed in *Portability* below.

## What each script does

### Result tables

| Script | Purpose | Reads | Writes |
| --- | --- | --- | --- |
| `build_group_comparison.py` | The 3-split × 2-feature table (CV MSE, test R²) with fallback from partial metrics to the Optuna best trial | `experiments/*/analysis_results/*/model_performance_summary.csv` | stdout / `--out` |
| `build_four_group_comparison.py` | The 2 × 2 random/scaffold × Mordred/mixed table; re-verifies split identity via `run_metadata.json` train/test rows | `experiments/`, `data/splits-20261021.json` | markdown |
| `build_four_group_xlsx.py` | 8-sheet workbook assembled from the same artifacts | `experiments/` | `.xlsx` |
| `compare_runs.py` | Side-by-side metrics of any two `analysis_results` directories, tolerant of Chinese headers | two run directories | stdout |
| `study_summary.py` | Best value per model for every Optuna study in a `models/` directory | `experiments/*/models/*/*.sqlite3` | stdout |
| `read_study_params.py` | Prints the best hyperparameters of every study | `.sqlite3` | stdout |
| `query_study.py` | Dumps one Optuna study (trials, params, values) | `.sqlite3` | stdout |
| `build_si_tables.py` | SI Tables S6–S8 (`.docx`): CV and test metrics per model per group | `experiments/` | `.docx` |
| `build_table_s5.py` | SI Table S5 (`.docx`) | a counts JSON | `.docx` |
| `build_si_generalization.py` | Renders the SI generalization section | intermediate JSONs | markdown |
| `emit_b_tables.py` | Markdown tables for section 8 of `../reports/B-取代基外推测试.md` | B-axis diagnostics | markdown |
| `build_terminology_checklist.py` | Paragraph-indexed terminology edits for the manuscript | manuscript copy | markdown |
| `revise_manuscript.py` | Applies the numeric revision that produced manuscript rev2 | manuscript copy | manuscript copy |

### Split and generalization diagnostics

| Script | Purpose | Reads |
| --- | --- | --- |
| `baxis_lobo_cv.py` | Leave-one-`B`-group-out CV: 15 substituted meso groups, 3 per fold, `B00` separate | `experiments/` |
| `bgroup_difficulty.py` | Leave-one-`B`-out *inside the training subset* to explain the CV-vs-test gap; fits Lasso and RandomForest | `experiments/` |
| `bsplit_cv_vs_test.py` | Label distribution and trivial-predictor floors on the CV side vs the test side | `experiments/` |
| `bsplit_diagnostics.py` | Variance decomposition over design factors + one-hot `A/X/C` composition baseline | `experiments/` |
| `bsplit_test.py` | Leave-one-`B`-out over 16 `B` patterns with in-fold preprocessing; scores ridge, constant and one-hot baselines | `experiments/`, fingerprint cache |
| `bsplit_test2.py` | Second pass separating a linear blow-up from genuinely absent transferable signal (raw ridge vs clipped ridge vs XGBoost) | `experiments/`, fingerprint cache |
| `scaffold_drift_analysis.py` | Per-fold label mean/sd, model MSE vs constant and oracle-constant, Bemis–Murcko and `A` overlap | `experiments/` |
| `split_reference_cv.py` | Same-protocol reference CV for the random and scaffold axes using the tuned B-axis hyperparameters (pooled MSE primary) | `experiments/` |
| `make_b_split.py` | The code that appended the `meso` split to `data/splits-20261021.json` | `data/` |
| `render_b_lobo_report.py` | Renders `../reports/B轴留一组外推.md` from the two JSONs | intermediate JSONs |
| `make_axis_figure.py` | Grouped bar chart of the three axes | intermediate JSONs |

### Feature-set ablations

| Script | Purpose |
| --- | --- |
| `fp_random_ablation.py` | Random-split ablation, anchored on Lasso CV MSE 0.270567 |
| `fp_scaffold_ablation.py` | Scaffold-split ablation, anchored on S-M Lasso CV MSE 0.6191693213237417 |
| `fp_family_diagnostic.py` | Per-family ablations — mordred, all fingerprints, Morgan only, MACCS only, RDKitFP only, topological only — for Lasso, XGBoost, SVR, KernelRidge and GaussianProcess |
| `count_feature_families.py` | Counts fingerprint-family columns |
| `mixed_vs_mordred.py` | Mixed vs Mordred comparison (references the earlier 20261013/20261014 runs) |
| `hp_vs_default.py` | Optuna-selected vs scikit-learn default hyperparameters on the exact fold matrices |

### SHAP

| Script | Purpose |
| --- | --- |
| `shap_four_group.py` | Cross-run SHAP composition shares, bootstrap CI and Welch test |
| `shap_b_axis.py` | `B`-axis SHAP composition (separate script because that axis' best model is KNeighbors, not SVR) |
| `shap_smf.py` | Re-runs the SHAP step for S-MF from `prepared_data.pkl` + the saved estimator; mirrors `run_pipeline.py:415-550` |
| `shap_detail.py` | Per-feature SHAP dump for `TopoShapeIndex` and `SlogP_VSA8` |

### Figures

| Script | Purpose |
| --- | --- |
| `make_fig7.py` | Manuscript Fig. 7 — XGBoost R-M vs R-MF, shared limits, R²/RMSE recomputed from predictions |
| `make_fig7_style.py` | Density-shaded variant sized to the manuscript box (5.77 × 2.69 in) |
| `make_fig8.py` | Landscape beeswarm Fig. 8 from `shap_values.npy` + `samples.csv`, clipping the `C3SP3 = 4` outliers |
| `make_parity_overview.py` | Four/five-panel best-model overview on one shared window |
| `restyle_parity.py` | Re-renders 35 parity panels in density style on the shared window 0.963–12.049 eV |

### Integrity and verification

| Script | Purpose |
| --- | --- |
| `check_cache_integrity.py` | Verifies cache columns against `.meta.json`, row count 1,734, both SHA-256s, the heavy-atom index binding and RDKit-recomputable descriptors |
| `verify_reviewer_claims.py` | Re-derives four reviewer-facing claims: target-derived columns, constant/design baselines, paired bootstrap ΔR², Optuna bound hits |

---

## Portability

Three tiers, honestly labelled:

**1. Runs as shipped, no editing** — `check_cache_integrity.py`, `compare_runs.py`,
`hp_vs_default.py`, `make_fig7.py`, `make_fig8.py`, `read_study_params.py`,
`shap_four_group.py`, `verify_reviewer_claims.py`. These read only `data/` and `experiments/`,
both of which are complete in this package.

**2. Runs as shipped, but needs a `models/` argument** — `study_summary.py`,
`query_study.py` (its default argument points at a run this package does not ship),
`build_si_tables.py`, `build_table_s5.py`, `count_feature_families.py`, `shap_detail.py`: they
carry a hard-coded `BASE`/path constant from the original working tree, so point it at this package
root (one-line edit) before running.

**3. Needs an intermediate that is not shipped** — the scripts that read
`.dsh_tmp/baxis_lobo_cv.json`, `.dsh_tmp/split_reference_cv.json`, `.dsh_tmp/s5_counts.json`
or the earlier 20261013/20261014 runs (which were superseded by the six runs in
`experiments/`): `baxis_lobo_cv.py`, `split_reference_cv.py`, `make_axis_figure.py`,
`render_b_lobo_report.py`, `build_si_generalization.py`, `build_table_s5.py`,
`fp_random_ablation.py`, `fp_scaffold_ablation.py`, `mixed_vs_mordred.py`, `make_b_split.py`,
`scaffold_drift_analysis.py`, `shap_smf.py`. They are included as the provenance of the numbers in
`../reports/`; `baxis_lobo_cv.py` and `split_reference_cv.py` regenerate their own JSONs (tens of
minutes each), after which the renderers work.

### Path conventions

Several scripts were written while the working tree used a scratch directory `.dsh_tmp/`, and they
still resolve inputs and outputs there. Nothing else changed. To satisfy those paths without editing
code:

```bash
mkdir -p .dsh_tmp/fp-full .dsh_tmp/runnability
cp data/Data-20261012_raw_features_fp.csv .dsh_tmp/fp-full/     # after decompressing the .xz
cp data/Data-20261012_raw_features_fp.csv.meta.json .dsh_tmp/fp-full/
```

On Windows use `New-Item -ItemType Directory` and `Copy-Item`. `.dsh_tmp/` is git-ignored by
design: it is scratch space, not a deliverable.

## Consistency with the reports

The numbers printed by these scripts are the ones quoted in `../reports/` and in the top-level
README. Where a quantity is recomputed from stored predictions rather than read from a summary CSV,
the two can differ in the last digit (for example CV MSE is the arithmetic mean of fold MSEs,
0.6192, while the pooled value is 0.6101 for the scaffold Lasso); `../reports/README.md` records
the caliber of each table.
