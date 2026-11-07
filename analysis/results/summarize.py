"""Export saved model metrics to CSV without fitting models."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', type=Path, default=ROOT / 'experiments')
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis_outputs/results')
    args = parser.parse_args()
    rows = []
    columns = ['run', 'source']
    for path in sorted(args.experiments.glob('*/analysis_results/*/model_performance_summary.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            for field in reader.fieldnames or []:
                if field not in columns:
                    columns.append(field)
            for record in reader:
                rows.append(dict(record, run=path.parents[2].name,
                                 source=path.relative_to(args.experiments).as_posix()))
    if not rows:
        parser.error('No model_performance_summary.csv records found.')
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'model_metrics.csv'
    with target.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} model records written to {target}')


if __name__ == '__main__':
    main()
