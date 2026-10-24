"""Print mixed-feature vs mordred-only model comparison (matched splits)."""
import pandas as pd

CVMSE, R2, RMSE = 'CV MSE (eV\u00b2)', 'Test R\u00b2', 'Test RMSE'

old = pd.read_csv('experiments/20261013_R-M/analysis_results/R-M/model_performance_summary.csv').set_index('Model')
mix = pd.read_csv('experiments/20261014_R-MF/analysis_results/R-MF/partial_model_metrics.csv').set_index('Model')

lines = []
lines.append(f'{"model":14s} {"R-M cvMSE":>10s} {"mix cvMSE":>10s}  {"R-M R2":>7s} {"mix R2":>7s}  {"R-M RMSE":>9s} {"mix RMSE":>9s}')
for m in mix.index:
    if m not in old.index:
        continue
    o, n = old.loc[m], mix.loc[m]
    lines.append(f'{m:14s} {o[CVMSE]:10.4f} {n[CVMSE]:10.4f}  {o[R2]:7.4f} {n[R2]:7.4f}  {o[RMSE]:9.4f} {n[RMSE]:9.4f}')

text = '\n'.join(lines)
print(text)
open('.dsh_tmp/mixed_vs_mordred.txt', 'w', encoding='utf-8').write(text)
