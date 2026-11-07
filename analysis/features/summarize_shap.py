"""Export saved mean absolute SHAP importances to CSV and optional PNG figures."""
import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ('Morgan', 'MACCS', 'AtomPair', 'TopoTorsion', 'RDKitFP')


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', type=Path, default=ROOT / 'experiments')
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis_outputs/features')
    parser.add_argument('--plot', action='store_true', help='Create a top-feature PNG per saved SHAP model.')
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()
    if args.top < 1:
        parser.error('--top must be positive.')
    features, families, plots = [], [], []
    pattern = '*/analysis_results/*/shap_analysis/*/feature_importance.csv'
    for path in sorted(args.experiments.glob(pattern)):
        run, model = path.parents[4].name, path.parent.name
        records, totals = [], {}
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                value = float(row['Mean_ABS_SHAP'])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f'Invalid SHAP importance in {path}: {row}')
                feature = row['Feature']
                family = next((name for name in FAMILIES if feature.startswith(name)), 'Mordred')
                records.append(dict(run=run, model=model, feature=feature,
                                    family=family, mean_abs_shap=value))
                totals[family] = totals.get(family, 0.0) + value
        features.extend(records)
        total = sum(totals.values())
        for family, value in sorted(totals.items()):
            families.append(dict(run=run, model=model, family=family,
                                 sum_mean_abs_shap=value,
                                 share_of_total_importance=value / total if total else ''))
        plots.append((run, model, records))
    if not features:
        parser.error('No saved SHAP feature importances found.')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'shap_features.csv',
              ['run', 'model', 'feature', 'family', 'mean_abs_shap'], features)
    write_csv(args.output / 'shap_families.csv',
              ['run', 'model', 'family', 'sum_mean_abs_shap', 'share_of_total_importance'], families)
    if args.plot:
        for run, model, records in plots:
            top = sorted(records, key=lambda row: row['mean_abs_shap'], reverse=True)[:args.top][::-1]
            fig, ax = plt.subplots(figsize=(8, max(3, len(top) * 0.3)))
            ax.barh([row['feature'] for row in top], [row['mean_abs_shap'] for row in top])
            ax.set(xlabel='Mean absolute SHAP value', title=f'{run} / {model}')
            fig.tight_layout()
            fig.savefig(args.output / f'{run}_{model}_top_features.png', dpi=200)
            plt.close(fig)
    print(f'{len(features)} feature records exported to {args.output}')


if __name__ == '__main__':
    main()
