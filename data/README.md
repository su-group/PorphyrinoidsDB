# Molecular data

This directory contains 1,734 molecular records and their corresponding XYZ structure files.

| Path | Contents |
|---|---|
| `PorDB-20261018.csv` | Molecular identifiers, SMILES, electronic properties, molecular descriptors, and structure-file paths |
| `XYZ/` | 1,734 XYZ files, named by molecular identifier |

## Main fields

| Field | Description |
|---|---|
| `idx` | Molecular identifier, such as `A01XC-B00-C00Y1` |
| `A_number`, `B_number` | Scaffold and meso-substituent identifiers used for grouped analyses |
| `X_element`, `C_number`, `Y_number` | Additional molecular design labels |
| `Status` | Calculation status; all included records are marked `Structure Optimization Finished` |
| `SMILES`, `CanonicalSMILES` | Molecular representations |
| `HOMO_eV`, `LUMO_eV`, `Gap_eV` | Frontier orbital energies and HOMO–LUMO gap, in eV |
| `HOMO-3_eV` through `LUMO+3_eV` | Additional orbital energies, in eV |
| `IP_eV`, `EA_eV`, `ChemicalPotential_mu_eV`, etc. | Electronic-property descriptors; units are indicated in column names where specified |
| `Formula`, `MolecularWeight`, `TPSA`, `MolLogP`, etc. | Molecular composition and physicochemical or graph descriptors |
| `xyz_file` | Structure-file path relative to this directory, for example `XYZ/A01XC-B00-C00Y1.xyz` |

Each XYZ file begins with the atom count, followed by a comment line and rows containing element symbols and Cartesian coordinates. All 1,734 paths listed in `xyz_file` have corresponding files in this directory.

## Read the data

From the repository root:

```python
from pathlib import Path
import pandas as pd

folder = Path("data")
data = pd.read_csv(folder / "PorDB-20261018.csv")
structure = folder / data.loc[0, "xyz_file"]
print(data[["idx", "SMILES", "Gap_eV"]].head())
print(structure.read_text())
```

## Using the training pipeline

This CSV uses `Gap_eV`. The loader in `core/data_processing.py` currently expects `Gap(eV)`, so column-name adaptation is required before training. Pass the chosen input path explicitly with `--data`.

For gap prediction, do not use HOMO/LUMO energies or properties derived from them as input features. The supplied feature-generation pipeline calculates descriptors from SMILES.
