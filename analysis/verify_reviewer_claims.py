"""Re-verify the Methodology reviewer's numeric claims before adopting them.

Claims checked:
  1. target-derived columns in data/Data-20261012.csv
     (ChemicalHardness = Gap/2, ChemicalSoftness = 2/Gap, IP-EA = Gap)
  2. constant baseline and design-factorial baseline on the R-MF test rows
  3. paired bootstrap of dR2(XGBoost - SVR) over the same 347 rows
  4. Optuna best params for R-MF SVR / KernelRidge (bound hits)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data_processing import load_valid_data  # noqa: E402

DATA = ROOT / "data" / "Data-20261012.csv"
RUN = ROOT / "experiments" / "20261016_R-MF"


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- 1. leakage
rule("1. target-derived columns")
raw = pd.read_csv(DATA, encoding="utf-8-sig")
gap = raw["Gap(eV)"]
checks = {
    "ChemicalHardness = Gap/2": (raw["ChemicalHardness(eV)"] - gap / 2).abs().max(),
    "ChemicalSoftness = 2/Gap": (raw["ChemicalSoftness(1/eV)"] - 2.0 / gap).abs().max(),
    "IP - EA = Gap": (raw["IonizationPotential(eV)"] - raw["ElectronAffinity(eV)"] - gap).abs().max(),
}
for name, dev in checks.items():
    print(f"  {name:32s} max |dev| = {dev:.3e}")
print("  Gap == LUMO - HOMO max |dev| = %.3e"
      % (raw["Gap(eV)"] - (raw["LUMO(eV)"] - raw["HOMO(eV)"])).abs().max())

# ---------------------------------------------------------- 2/3. baselines
rule("2/3. baselines and bootstrap on the R-MF test rows")
data = load_valid_data(DATA)
meta = json.loads((RUN / "models" / "R-MF" / "run_metadata.json").read_text(encoding="utf-8"))
print("  run_metadata keys:", sorted(meta.keys())[:8], "...")
test_rows = np.asarray(meta["test_source_rows"])
train_rows = np.asarray(meta["train_source_rows"])
src = data["source_row"].to_numpy()
test_mask = np.isin(src, test_rows)
train_mask = np.isin(src, train_rows)
print(f"  train={train_mask.sum()}  test={test_mask.sum()}")

y = data["Gap(eV)"].to_numpy(float)
y_tr, y_te = y[train_mask], y[test_mask]


def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


# constant predictor: training mean
pred_const = np.full_like(y_te, y_tr.mean())
print(f"  constant(train mean): R2={r2(y_te, pred_const):+.4f} "
      f"RMSE={np.sqrt(np.mean((y_te-pred_const)**2)):.4f} "
      f"MAE={np.mean(np.abs(y_te-pred_const)):.4f}")

# design-factorial ridge: one-hot identities + A x B and A x C interactions
design = pd.get_dummies(
    data[["A_number", "B_number", "C_number", "X_element"]].astype(str),
    columns=["A_number", "B_number", "C_number", "X_element"],
)
a_cols = [c for c in design.columns if c.startswith("A_number_")]
b_cols = [c for c in design.columns if c.startswith("B_number_")]
c_cols = [c for c in design.columns if c.startswith("C_number_")]
inter = {}
for left, right, tag in ((a_cols, b_cols, "AB"), (a_cols, c_cols, "AC")):
    for lc in left:
        for rc in right:
            inter[f"{tag}_{lc.split('_')[-1]}_{rc.split('_')[-1]}"] = (
                design[lc].to_numpy(float) * design[rc].to_numpy(float)
            )
design = pd.concat([design, pd.DataFrame(inter, index=design.index)], axis=1)
X_design = design.to_numpy(float)
print(f"  design matrix: {X_design.shape[1]} columns")

best = None
for alpha in (1.0, 3.0, 10.0, 30.0):
    model = Ridge(alpha=alpha).fit(X_design[train_mask], y_tr)
    pred = model.predict(X_design[test_mask])
    score = r2(y_te, pred)
    rmse = float(np.sqrt(np.mean((y_te - pred) ** 2)))
    mae = float(np.mean(np.abs(y_te - pred)))
    print(f"  design ridge alpha={alpha:5.1f}: R2={score:+.4f} RMSE={rmse:.4f} MAE={mae:.4f}")
    if best is None or score > best[1]:
        best = (alpha, score)

# paired bootstrap XGBoost vs SVR
preds = pd.read_csv(RUN / "analysis_results" / "R-MF" / "test_predictions.csv")
colmap = {c.lower(): c for c in preds.columns}
xgb = preds[colmap.get("xgboost")].to_numpy(float)
svr = preds[colmap.get("svr")].to_numpy(float)
truth = preds[colmap.get("gap(ev)")].to_numpy(float)
print(f"  test_predictions: n={len(truth)}  "
      f"R2 xgb={r2(truth, xgb):+.4f}  svr={r2(truth, svr):+.4f}  "
      f"delta={r2(truth, xgb)-r2(truth, svr):+.4f}")

rng = np.random.default_rng(42)
deltas = np.empty(20000)
idx = np.arange(len(truth))
for i in range(20000):
    pick = rng.choice(idx, size=len(idx), replace=True)
    deltas[i] = r2(truth[pick], xgb[pick]) - r2(truth[pick], svr[pick])
lo, hi = np.percentile(deltas, [2.5, 97.5])
p_two = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
print(f"  paired bootstrap dR2(XGB-SVR) = {deltas.mean():+.4f} "
      f"95% CI [{lo:+.4f}, {hi:+.4f}]  p~{p_two:.3f}")

# ------------------------------------------------------- 4. Optuna bounds
rule("4. Optuna best params for R-MF SVR / KernelRidge")
import sqlite3

for model in ("SVR", "KernelRidge"):
    db = RUN / "models" / "R-MF" / f"{model}.sqlite3"
    if not db.exists():
        print(f"  {model}: study db missing ({db})")
        continue
    con = sqlite3.connect(str(db))
    row = con.execute(
        "select trial_id, value from trial_values order by value asc limit 1"
    ).fetchone()
    tid = row[0]
    params = con.execute(
        "select param_name, value_json from trial_params where trial_id=?", (tid,)
    ).fetchall()
    con.close()
    print(f"  {model}: best trial {tid}  value={row[1]:.6f}")
    for name, value in params:
        print(f"      {name:12s} {value}")
