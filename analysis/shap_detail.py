import json
import numpy as np
import pandas as pd
from pathlib import Path
RUN = Path(r"E:\Project-ZhangChengwei\PorScaDB-ML-compute-20261014\PorScaDB-ML\experiments\20261016_R-M\analysis_results\R-M\shap_analysis\XGBoost")
meta = json.loads((RUN / "metadata.json").read_text(encoding="utf-8"))
print("metadata:", {k: meta[k] for k in list(meta)[:12]})
imp = pd.read_csv(RUN / "feature_importance.csv", encoding="utf-8")
samples = pd.read_csv(RUN / "samples.csv", encoding="utf-8")
values = np.load(RUN / "shap_values.npy")

for feat in ["TopoShapeIndex", "SlogP_VSA8"]:
    col = samples[feat].to_numpy(float)
    sv = values[:, samples.columns.get_loc(feat)]
    print(f"--- {feat} ---")
    for v in sorted(pd.Series(col).unique()):
        m = col == v
        print(f"  value={v:>8.3f} n={m.sum():>3} mean_shap={sv[m].mean():+.4f}")

col = samples["C3SP3"].to_numpy(float)
mask = col == 4
print("--- the 3 molecules with C3SP3=4: distinctive descriptors ---")
sub = samples[mask]
base = samples[~mask]
diff = (sub.mean(axis=0) - base.mean(axis=0)).abs().sort_values(ascending=False)
print(diff.head(12).round(3).to_string())
print("row index of those samples:", list(np.where(mask)[0]))
