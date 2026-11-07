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
- `analysis/` and `reports/` — analysis scripts, figures, and summaries.
- `run_pipeline.py` — run one modeling experiment.
- `run_experiments.py` — run a group of experiments.

## Setup

The project uses separate Conda environments for feature generation and model training:

```bash
conda env create -f environment-features.yml
conda env create -f environment-ml.yml
```

See the environment files for the Python versions and dependencies.

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

See [`data/README.md`](data/README.md), [`experiments/README.md`](experiments/README.md), and [`analysis/README.md`](analysis/README.md) for details on the data, experiment outputs, and analysis scripts.

## Citation and license

If you use this project, please cite the work described in [`CITATION.cff`](CITATION.cff). The code is released under the [MIT License](LICENSE).
