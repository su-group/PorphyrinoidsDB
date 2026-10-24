# PorScaDB-ML

**Predicting DFT HOMO–LUMO gaps of porphyrinoid macrocycles from SMILES — and measuring how far those models actually transfer.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn 1.9](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the complete pipeline, the trained models, and every report, table and figure behind a controlled benchmark of ten regression models on 1,734 porphyrinoids. The central question is not "which model is most accurate" but **which variation axis a descriptor-based model can and cannot extrapolate along**.

---

## Table of contents

- [Overview](#overview)
- [Headline results](#headline-results)
- [Why three split axes](#why-three-split-axes)
- [Repository layout](#repository-layout)
- [Dataset](#dataset)
- [Environments](#environments)
- [Quick start](#quick-start)
- [Reproducing the six experiments](#reproducing-the-six-experiments)
- [What is shipped, and what is regenerated](#what-is-shipped-and-what-is-regenerated)
- [Results](#results)
- [Reproducibility notes and caveats](#reproducibility-notes-and-caveats)
- [Citation](#citation)
- [License](#license)

---

## Overview

| Item | Value |
|---|---|
| Raw records | 1,860 DFT calculations (1,861 lines incl. header) |
| Valid samples | **1,734** (`C_number ∈ {0,1,2}`, finite positive gap, parseable SMILES) |
| Target | `Gap(eV)` = LUMO − HOMO (eV) |
| Feature set A | 1,613 Mordred 2D descriptors |
| Feature set B ("mixed") | 9,972 columns = 1,613 Mordred + 5 RDKit fingerprint families (Morgan, MACCS, AtomPair, TopoTorsion, RDKitFP) |
| Models | 10: Lasso, ElasticNet, DecisionTree, RandomForest, XGBoost, LightGBM, SVR, KernelRidge, KNeighbors, GaussianProcess |
| Hyper-parameter search | Optuna, 100 trials per model, objective = 5-fold CV MSE |
| Validation | 5-fold CV with **in-fold** imputation / variance & correlation filtering / scaling; one held-out test evaluation |
| Splits | `random` · `meso` (B_number substituent) · `scaffold` (A_number macrocycle) |
| Runs shipped | 3 axes × 2 feature sets = **6 runs × 10 models** |
| Seed | 42, identical across all runs |

Everything runs on CPU. The full 6-run matrix is roughly 40 GPU-free machine-hours; each individual run is 1–4 h.

## Headline results

Best model per run, selected by cross-validation (the test set is never used for selection):

| Axis | Features | CV-best model | CV MSE (eV²) | Test R² | Test-best model | Test R² |
|---|---|---|---|---|---|---|
| `random` | Mordred | XGBoost | **0.1342** | 0.9321 | XGBoost | 0.9321 |
| `random` | mixed | SVR | **0.0913** | 0.9565 | XGBoost | 0.9632 |
| `meso` | Mordred | KNeighbors | **0.4783** | 0.7369 | RandomForest | 0.8667 |
| `meso` | mixed | KNeighbors | **0.1767** | 0.9412 | RandomForest | 0.9631 |
| `scaffold` | Mordred | Lasso | **0.6192** | 0.0909 | RandomForest | 0.3741 |
| `scaffold` | mixed | SVR | **1.1748** | 0.2843 | SVR | 0.2843 |

Five findings that the six runs support:

1. **Fingerprints are interpolation aids, not extrapolation aids.** Adding them changes mean CV MSE by **−0.1206 eV²** on the random axis (10/10 models improve), **−0.0599 eV²** on the meso axis (8/10 improve), and **+0.1892 eV²** on the scaffold axis (0/10 improve, all ten degrade).
2. **The split axis dominates the model choice.** Relative to a random split, median CV MSE grows **3.14×** on the meso axis and **5.27×** on the scaffold axis with Mordred features (4.54× and 11.79× with fingerprints). Choosing a better regressor buys far less than choosing a fairer split.
3. **Scaffold extrapolation essentially fails.** No configuration reaches test R² > 0.38; parity plots collapse into horizontal bands because the models regress towards the training mean whenever the macrocyclic core is unseen.
4. **CV ranking does not survive the axis change.** The CV-best model is the test-best model in only 2 of 6 runs. Under the scaffold split the CV winner (Lasso, CV MSE 0.6192) delivers test R² 0.0909, while a CV-mediocre RandomForest delivers 0.3741.
5. **A narrow search space can masquerade as a modelling result.** The kernel methods' apparent penalty from fingerprints was traced to an under-specified Optuna space in `config/hyperparameter_optimization.json`; widening it improves SVR and KernelRidge by **5.0×** and **5.8×** in CV MSE, which is what the shipped `20261016_*` and `20261021_*` runs use.

SHAP analysis (per-run, best model) shows the fingerprint families absorb **84–89 %** of total |SHAP| mass in the mixed-feature runs (B-MF 84.17 %, R-MF 87.07 %, S-MF 89.14 %) — the models lean on substructure bits, and that is precisely why they transfer only within the chemistry they were trained on.

## Why three split axes

A single random 80/20 split answers "how well does this model interpolate the same chemical space". It cannot answer the question a materials reader actually asks: *what happens on a scaffold I have never computed?*

| Axis | Grouping key | What the held-out set contains |
|---|---|---|
| `random` | none | i.i.d. samples; all scaffolds and substituents seen in training |
| `meso` | `B_number` | unseen meso-substituent combinations on seen macrocycles (mild extrapolation) |
| `scaffold` | `A_number` | entirely unseen macrocyclic cores (true structural extrapolation) |

Splits are frozen in `data/splits-20261021.json` (schema `schema_version`, bound to `data_sha256`, seed 42, one entry per axis with `train_positions` / `test_positions` / `folds`). Every shipped run used this file, so re-running with `--split-file` reproduces the exact partitions.

## Repository layout

```text
.
├── prepare_features.py            # SMILES → Mordred descriptors (+ optional fingerprints), with cache signing
├── run_pipeline.py           # main entry: validate → split → 5-fold CV → fit → save → SHAP → figures
├── run_experiments.py             # matrix launcher for the R/S × M/MF four-run grid
├── core/
│   ├── data_processing.py         # filtering, SMILES standardisation, cache identity checks
│   ├── experiment_setup.py        # split manifests, feature-set selection, fold construction
│   ├── molecular_features.py      # Mordred + 5 RDKit fingerprint families
│   └── train_optimizer.py         # in-fold preprocessing, model zoo, Optuna objective, evaluation
├── config/hyperparameter_optimization.json
├── tests/                         # fast regression tests (no full training)
├── analysis/                      # the as-run scripts behind every report, table and figure
├── data/                          # source data, signed feature caches, frozen splits
├── experiments/<date>_<run>/      # one directory per run: models/, analysis_results/, logs/
├── reports/                       # 17 reports (Markdown), figures, summary workbook, SI tables
└── MANIFEST.md                    # inventory: file counts, sizes, what was excluded
```

Four sub-READMEs document the parts in depth: [`data/README.md`](data/README.md) (column dictionary, cache contracts, the three splits), [`experiments/README.md`](experiments/README.md) (what each run directory contains, per-model manifest, refit commands), [`analysis/README.md`](analysis/README.md) (what each of the 41 scripts does and which ones need a path edit), [`reports/README.md`](reports/README.md) (report index with the caliber of every table).

Each `experiments/<date>_<run>/` directory mirrors exactly what the pipeline writes with `--output-root`:

```text
experiments/20261021_B-MF/
├── models/B-MF/                   # <Model>.pkl, <Model>.result.pkl, <Model>.sqlite3 (Optuna study),
│                                  # preprocessing.pkl, input_signature.json, run_metadata.json
├── analysis_results/B-MF/         # model_performance_summary.csv, test_predictions.csv, group_errors.csv,
│                                  # 10 parity plots (PNG+SVG), feature correlations, shap_analysis/
└── logs/pipeline_B-MF.log    # full stdout log of the run
```

## Dataset

`data/Data-20261012.csv` — one row per DFT calculation, 1,860 rows (1,861 lines including the header).

| Column | Meaning |
|---|---|
| `idx` | sample identifier, e.g. `A01XC-B00-C00Y1` |
| `SMILES` | molecular graph |
| `A_number` | macrocyclic scaffold identifier (grouping key of the `scaffold` axis) |
| `B_number` | meso-substituent identifier (grouping key of the `meso` axis) |
| `C_number` | metal / axial-ligand class; the pipeline keeps `{0,1,2}` |
| `X_element`, `Y_number` | heteroatom and axial-ligand descriptors |
| `Status` | DFT job status; only `Structure Optimization Finished` rows are used |
| `Gap(eV)` | **target**, LUMO − HOMO in eV |
| `HOMO(eV)`, `LUMO(eV)`, `SCF Done(Hartree)`, `Q band (nm)`, `S band (nm)` | electronic structure and spectroscopy |
| `MolMR`, `LogP`, `TPSA`, `NumRings`, … | physico-chemical descriptors retained for reference |

Filtering (1,860 → 1,734) is: `C_number ∈ {0,1,2}` ∧ `Status == 'Structure Optimization Finished'` ∧ target finite and positive ∧ SMILES parses and normalises. The status check does almost all the work: exactly 1,734 rows carry that status (the other 126 are `Imaginary Frequencies` 47, `l9999` 42, `File Not Found` 30, `l502` 3, `Incomplete` 2, `l103` 2).

`data/` also holds two **signed** feature caches and the frozen split manifest:

| File | Size | Contents |
|---|---|---|
| `Data-20261012_raw_features.csv` | 36 MB | 1,734 × 1,613 Mordred descriptors, **needed for the three `-M` runs** |
| `Data-20261012_raw_features_fp.csv.xz` | 11 MB | the same records + fingerprint families, 9,972 columns; **compress-ed to stay repo-friendly — decompress before use** (see [Quick start](#quick-start)) |
| `splits-20261021.json` | 0.5 MB | frozen `random` / `scaffold` / `meso` partitions (seed 42) |

Each cache ships a `.meta.json` that binds it to the source file (`data_sha256`), to itself (`cache_sha256`), to the per-row SMILES/label/spin records, and to the environment that generated it. The loader refuses a cache whose hash, row order or source data no longer match — a guard against silently training on stale features.

## Environments

Two isolated environments are required, because Mordred's pinned NumPy conflicts with the modern training stack:

| Purpose | File | Python |
|---|---|---|
| Mordred feature generation | `environment-features.yml` + `requirements-features.txt` | 3.8 |
| Training, SHAP, plots | `environment-ml.yml` + `requirements-ml.txt` | 3.12 |

```bash
conda env create -f environment-features.yml   # feature generation
conda env create -f environment-ml.yml         # training
```

The shipped artifacts were produced under: numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, rdkit 2026.9.1, xgboost 3.4.1, lightgbm 4.7.0, optuna 5.0.0, shap 0.53.0 (recorded verbatim in every `run_metadata.json`). `requirements-ml.txt` pins the nearest public versions.

> **Pickle compatibility.** `.pkl` files encode scikit-learn internals. Loading them under an older stack fails — under scikit-learn 1.6 the shipped `preprocessing.pkl` raises `StringDtype.__init__() takes from 1 to 2 positional arguments but 3 were given`. Use the environment recorded in `run_metadata.json`, or refit from the caches instead of unpickling.

## Quick start

```bash
# 0. (only if you want the fingerprint runs) decompress the mixed cache
python -c "import lzma,shutil; shutil.copyfileobj(lzma.open('data/Data-20261012_raw_features_fp.csv.xz','rb'), open('data/Data-20261012_raw_features_fp.csv','wb'))"

# 1. smoke test: 40 real samples, 1 Optuna trial per model, 3 SHAP samples (~2 min)
conda run -n porscadb-ml python -B run_pipeline.py --smoke

# 2. unit tests
conda run -n porscadb-ml python -B -m unittest discover -s tests -v

# 3. one real run, frozen random split, all 10 models, 100 trials
conda run -n porscadb-ml python -B run_pipeline.py \
    --split-file data/splits-20261021.json --split-name random \
    --run-name R-M --trials 100 --output-root experiments/20261024_R-M
```

Smoke-mode numbers are for plumbing only — never quote them as model performance.

## Reproducing the six experiments

The six runs differ only in `--split-name` and the feature cache. `--output-root` reproduces the shipped directory layout exactly.

```bash
PY=python   # inside the training environment

# --- Mordred-only axis runs -------------------------------------------------
$PY run_pipeline.py data/Data-20261012_raw_features.csv \
    --data data/Data-20261012.csv \
    --split-file data/splits-20261021.json --split-name random   --run-name R-M  \
    --trials 100 --output-root experiments/20261024_R-M

$PY run_pipeline.py data/Data-20261012_raw_features.csv \
    --split-file data/splits-20261021.json --split-name scaffold --run-name S-M  \
    --trials 100 --output-root experiments/20261024_S-M

$PY run_pipeline.py data/Data-20261012_raw_features.csv \
    --split-file data/splits-20261021.json --split-name meso     --run-name B-M  \
    --trials 100 --output-root experiments/20261024_B-M

# --- mixed (descriptor + fingerprint) axis runs -----------------------------
$PY run_pipeline.py data/Data-20261012_raw_features_fp.csv \
    --split-file data/splits-20261021.json --split-name random   --run-name R-MF \
    --feature-mode mixed --trials 100 --output-root experiments/20261024_R-MF
# ... identical for S-MF (--split-name scaffold) and B-MF (--split-name meso)
```

Useful switches: `--models XGBoost SVR` (subset), `--resume` (continue an interrupted run), `--fold-jobs N` (parallel folds), `--shap-mode {best,all,none}`, `--drop-fingerprint FAMILY` / `--only-fingerprint FAMILY` (single-family ablations).

The four-run R/S × M/MF grid can instead be launched through `run_experiments.py` (sequential subprocesses, resumable, writes a comparison table):

```bash
$PY run_experiments.py --cache data/Data-20261012_raw_features.csv \
    --output-root experiments --experiments R-M R-MF S-M S-MF --trials 100
```

## What is shipped, and what is regenerated

To keep the repository clonable in under a minute while keeping every result auditable, models are included selectively. The rule is stated in full in [`experiments/README.md`](experiments/README.md):

| Included | Why |
|---|---|
| All 6 runs × 10 models: `<Model>.sqlite3` Optuna study, `.progress.json`, metrics, predictions, parity plots, SHAP | The evidence behind every number in `reports/` |
| `preprocessing.pkl`, `input_signature.json`, `filtered_feature_names.json`, `run_metadata.json` | Exact feature list, imputation medians, scalers, fold membership, versions |
| Estimators that are small and portable: DecisionTree, Lasso, ElasticNet, XGBoost, LightGBM, RandomForest | 23 MB total, load and predict immediately |
| `GaussianProcess`, `KernelRidge`, `KNeighbors`, `SVR` `.pkl` | **Excluded** — each carries a copy of the training matrix (5.5–57 MB per file; 690 MB for the four families' estimators alone, 2.12 GB once their `.result.pkl` twins and `prepared_data.pkl` are counted). Their full metrics, predictions, study databases and SHAP results *are* shipped; refit with one command (below) |

`prepared_data.pkl` (28–207 MB per run, a pickled design matrix) is likewise excluded.

Refitting any excluded model from the shipped caches takes minutes rather than hours, because the Optuna study is preserved:

```bash
# refit only the kernel models of the meso mixed run, reusing the frozen split
python run_pipeline.py data/Data-20261012_raw_features_fp.csv \
    --split-file data/splits-20261021.json --split-name meso \
    --models SVR KernelRidge KNeighbors GaussianProcess \
    --feature-mode mixed --trials 100 --run-name B-MF-refit \
    --output-root experiments/20261024_B-MF-refit
```

## Results

### CV MSE (eV², lower is better; bold = best in column)

| Model | `R-M` | `R-MF` | `B-M` | `B-MF` | `S-M` | `S-MF` |
|---|---|---|---|---|---|---|
| XGBoost | **0.1342** | 0.1038 | 0.5465 | 0.4962 | 1.2542 | 1.2779 |
| LightGBM | 0.1437 | 0.1048 | 0.4855 | 0.5390 | 1.2649 | 1.2733 |
| RandomForest | 0.1554 | 0.1040 | 0.7400 | 1.1031 | 2.5315 | 2.8293 |
| SVR | 0.1980 | **0.0913** | 0.5733 | 0.4716 | 1.0219 | **1.1748** |
| KernelRidge | 0.1989 | 0.1055 | 0.5569 | 0.4082 | 1.0199 | 1.2053 |
| DecisionTree | 0.2195 | 0.1386 | 1.4671 | 1.4055 | 2.9328 | 2.9328 |
| GaussianProcess | 0.2221 | 0.1150 | 0.7834 | 0.4909 | 1.1927 | 1.2136 |
| ElasticNet | 0.2655 | 0.1072 | 0.4876 | 0.4597 | 0.6223 | 1.1846 |
| Lasso | 0.2706 | 0.1187 | 0.4878 | 0.4570 | **0.6192** | 1.2170 |
| KNeighbors | 0.6008 | 0.2133 | **0.4783** | **0.1767** | 1.1882 | 1.2308 |

### Test R² (higher is better; bold = best in column)

| Model | `R-M` | `R-MF` | `B-M` | `B-MF` | `S-M` | `S-MF` |
|---|---|---|---|---|---|---|
| XGBoost | **0.9321** | **0.9632** | 0.7198 | 0.8161 | 0.0010 | 0.0005 |
| LightGBM | 0.8917 | 0.9207 | 0.7979 | 0.8267 | 0.0530 | 0.0026 |
| RandomForest | 0.8533 | 0.8886 | **0.8667** | **0.9631** | **0.3741** | 0.0488 |
| SVR | 0.8880 | 0.9565 | 0.6227 | 0.9049 | -0.4093 | **0.2843** |
| KernelRidge | 0.8623 | 0.9443 | 0.6141 | 0.9274 | 0.0214 | 0.0982 |
| DecisionTree | 0.8064 | 0.8328 | 0.4848 | 0.4848 | -0.0208 | -0.0208 |
| GaussianProcess | 0.8725 | 0.9362 | 0.2841 | 0.9317 | -1.0759 | 0.2211 |
| ElasticNet | 0.8137 | 0.9399 | 0.4664 | 0.8426 | 0.0846 | 0.2468 |
| Lasso | 0.8119 | 0.9404 | 0.4661 | 0.8336 | 0.0909 | -0.0014 |
| KNeighbors | 0.4094 | 0.7800 | 0.7369 | 0.9412 | -0.0855 | 0.2337 |

### Feature-set effect (mixed − Mordred, mean ΔCV MSE within the same axis)

| Axis | Comparable models | Mean ΔCV MSE | Improved | Worse |
|---|---|---|---|---|
| `random` | 10 | −0.1206 | 10 | 0 |
| `meso` | 10 | −0.0599 | 8 | 2 |
| `scaffold` | 10 | +0.1892 | 0 | 10 |

### Split-axis effect (ΔCV MSE relative to `random`, same feature set)

| Features | Axis | Mean ΔCV MSE | Median ratio |
|---|---|---|---|
| Mordred | `meso` | +0.4198 | 3.14× |
| Mordred | `scaffold` | +1.1239 | 5.27× |
| mixed | `meso` | +0.4806 | 4.54× |
| mixed | `scaffold` | +1.4337 | 11.79× |

Every per-model metric, prediction vector and plot is in `experiments/<run>/analysis_results/`; the narrative reports are indexed in [`reports/README.md`](reports/README.md). Figures: `reports/figures/parity_overview.png` (all six winning models on one shared axis), `axis_split_comparison.png` (axis effect), `fig8_shap_beeswarm.png`.

## Reproducibility notes and caveats

Read these before quoting a number.

1. **CV MSE is the arithmetic mean of five fold MSEs, not a pooled MSE.** Pooled aggregation gives 0.6101 where the pipeline reports 0.6192 for the same scaffold run. Both are correct; they are not interchangeable across tables.
2. **Test metrics are reported, never used for selection.** Model choice is by CV MSE only. Where the two disagree (4 of 6 runs), both are listed.
3. **SHAP values are in eV but their magnitude follows each model's output amplitude.** Compare *shares* and *ranks* within a run, never absolute values across runs.
4. **Fingerprints help interpolation only.** Any claim of generalisation must be made on the scaffold axis, where no configuration exceeds test R² 0.38.
5. **Instance-based and kernel models memorise.** KNeighbors reports train R² 1.0000 and train RMSE 1.1e-7 eV; its confidence is a coverage statement, not an error estimate.
6. **Feature filtering jitters by ±1 column** across folds: `np.corrcoef`'s reduction order is not fixed, so columns sitting exactly on the |corr| = 0.95 boundary enter or leave.
7. **Pickles require the recorded stack** (see [Environments](#environments)). The caches, splits and CSV results are version-independent and are the long-lived artifacts.
8. **Absolute paths.** `run_metadata.json` records the original machine paths (`E:\Project-ZhangChengwei\…`). They are provenance, not requirements — nothing resolves through them at run time.
9. **Nine columns in the source CSV are derived from the target.** `ChemicalHardness`, `ChemicalSoftness`, `ChemicalPotential`, `Electronegativity`, `ElectrophilicityIndex`, `NucleophilicityIndex`, `ElectronAffinity`, `IonizationPotential` and `Energy Min` are exact functions of `HOMO`/`LUMO`, and therefore of `Gap(eV)` (`hardness = Gap / 2`, for instance). No model here ever sees them — every feature is computed from `SMILES` alone — but do not feed them to a new model. See [`data/README.md`](data/README.md) for the checked formulas.

## Citation

If this work is useful, please cite it as described in [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE). The dataset and derived artifacts are released under the same terms; please also credit the original DFT data source.
