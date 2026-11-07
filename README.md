# PorScaDB-ML

PorScaDB-ML predicts DFT HOMO–LUMO gaps for porphyrinoid molecules from their SMILES. It includes feature-generation and training pipelines, trained models, and analysis outputs.

The project also compares model performance across three data splits:

- **Random** — samples from the same chemical space appear in training and test sets.
- **Meso-substituent** — test molecules contain unseen substituent combinations.
- **Scaffold** — test molecules contain unseen macrocyclic cores.

## Repository contents

- `core/` — data processing, molecular features, experiment setup, and training.
- `data/` — molecular data and structure files.
- `experiments/` — model artifacts and run outputs.
- `analysis/results/` — result summaries and run comparisons.
- `analysis/features/` — feature ablations and SHAP analyses.
- `run_pipeline.py` — run one modeling experiment.
- `run_experiments.py` — run a group of experiments.

## Setup

Install Miniconda or Anaconda first. Run the following commands from the repository root (the folder containing `run_pipeline.py`).

| Environment | Python | Purpose |
|---|---|---|
| `porscadb-ml` | 3.12 | Training, feature ablations, SHAP, and plotting |
| `porscadb-features` | 3.8 | Mordred feature generation |

```bash
conda env create -f environment-ml.yml
conda activate porscadb-ml
```

For molecular feature generation, create and activate the separate environment:

```bash
conda env create -f environment-features.yml
conda activate porscadb-features
```

The environment files install their corresponding `requirements-*.txt` dependencies automatically. Switch back with `conda activate porscadb-ml` for training and plotting.

To summarize saved results and generate SHAP figures:

```bash
conda activate porscadb-ml
python analysis/results/summarize.py
python analysis/features/summarize_shap.py --plot
```

CSV-only summaries use the Python standard library and do not require these scientific packages. PNG output requires Matplotlib.

## Run

Activate the training environment, then check the available options:

```bash
python run_pipeline.py --help
```

For example, to run a random-split experiment using a prepared feature table:

```bash
python run_pipeline.py path/to/features.csv \
  --split-file path/to/splits.json \
  --split-name random \
  --run-name example \
  --output-root experiments/example
```

See [`data/README.md`](data/README.md), [`experiments/README.md`](experiments/README.md), and [`analysis/README.md`](analysis/README.md) for data fields, experiment outputs, and analysis scripts.

## License

The code is released under the [MIT License](LICENSE).
