# MANIFEST

Inventory of this package as prepared for upload. Generated 2026-10-24.

- 654 files, 145,723,621 bytes (139.0 MB)
- Largest single file: 36.4 MB (`data/Data-20261012_raw_features.csv`) — comfortably under
  GitHub's 100 MB hard limit and under the 50 MB warning threshold

## By directory

| Path | Files | Bytes | What it holds |
| --- | ---: | ---: | --- |
| `experiments/` | 551 | 92,018,385 | Six run directories: per-model metrics, predictions, parity plots, SHAP values, Optuna studies, 21 shipped estimators, `run_metadata.json`, logs |
| `data/` | 7 | 51,487,636 | Source table, two feature caches (+ manifests), the frozen split manifest, `README.md` |
| `reports/` | 32 | 1,761,845 | 17 narrative reports, figure set, workbook, two SI documents, index |
| `analysis/` | 42 | 309,884 | 41 analysis scripts + `README.md` |
| package root | 15 | 99,157 | Entry points, requirements, environments, README (EN + zh-CN), LICENSE, CITATION, `.gitignore`, `.gitattributes` |
| `core/` | 4 | 34,758 | Data processing, experiment setup, molecular features, training/optimisation |
| `tests/` | 2 | 9,594 | Unit tests |
| `config/` | 1 | 2,362 | `hyperparameter_optimization.json` — the Optuna search space |

## By extension

| Extension | Files | | Extension | Files |
| --- | ---: | --- | --- | ---: |
| `.pkl` | 124 | | `.log` | 6 |
| `.png` | 114 | | `.npy` | 6 |
| `.svg` | 111 | | `.txt` | 3 |
| `.json` | 94 | | `.yml` | 2 |
| `.sqlite3` | 60 | | `.docx` | 2 |
| `.py` | 50 | | `.cff` | 1 |
| `.csv` | 44 | | `.xz` | 1 |
| `.md` | 32 | | `.xlsx` | 1 |
| | | | no extension | 3 |

The 124 `.pkl` files are the 21 shipped estimators and their `.result.pkl` twins, plus
`feature_scaler.pkl`, `scaler.pkl` and `preprocessing.pkl` per run. The 60 `.sqlite3` files are the
Optuna study databases — one per model per run, including the four families whose estimators are not
shipped, so every search is auditable.

## Data provenance

Each cache ships a `.meta.json` manifest holding the source table's `data_sha256`, the cache's own
`cache_sha256`, the column order and the RDKit version used to build it. Data and caches are
checked against those manifests by [`analysis/check_cache_integrity.py`](analysis/check_cache_integrity.py);
see [`data/README.md`](data/README.md) for the file list and byte sizes.

## What is not here, and why

[`experiments/build_ledger.json`](experiments/build_ledger.json) is the machine-readable record: 78
files, 2,122,628,075 bytes (2.12 GB) were deliberately left out.

| Excluded | Files | Bytes |
| --- | ---: | ---: |
| `prepared_data.pkl` (pickled design matrix) | 6 | 742,796,324 |
| `GaussianProcess` (`.pkl` + `.result.pkl` + `.sampler.pkl`) | 18 | 485,628,903 |
| `KNeighbors` (`.pkl` + `.result.pkl` + `.sampler.pkl`) | 18 | 300,118,234 |
| `KernelRidge` (`.pkl` + `.result.pkl` + `.sampler.pkl`) | 18 | 300,115,672 |
| `SVR` (`.pkl` + `.result.pkl` + `.sampler.pkl`) | 18 | 293,968,942 |
| **Total** | **78** | **2,122,628,075** (2.12 GB) |

These four model families store the training matrix inside the estimator, which is why a single
`.pkl` reaches 5.5–57 MB. Their metrics, predictions, study databases and SHAP outputs are all
shipped; only the pickled estimator is missing, and `experiments/README.md` gives the one-line refit
command. Exact per-file sizes are in the ledger and in the manifest table of that README.

## Verification performed on this tree

| Check | Result |
| --- | --- |
| `py_compile` on all 50 Python files | 0 failures |
| Markdown relative links (31 files) | 0 broken |
| `analysis/check_cache_integrity.py` | passes — hashes, shape (1734, 1613), 1734/1734 SMILES reparsed, 0 heavy-atom differences |
| Largest file | 36.4 MB < 50 MB |
| Non-ASCII / encoding | all text files UTF-8; CSV declared with BOM in the source table, read with `utf-8-sig` |

```bash
# from the package root
python analysis/check_cache_integrity.py
```

The data files are stored with CRLF line endings and [`.gitattributes`](.gitattributes) marks
them `-text`, so a clone cannot silently rewrite them.
