"""执行四组主实验及可选指纹消融；独立子进程、固定分割、可恢复进度。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
from core.data_processing import load_valid_data, load_feature_cache, file_sha256
from core.experiment_setup import make_split_manifest, FP_PREFIXES


def write_json(path, content):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def collect_results(root, jobs, statuses):
    tables = []
    for job in jobs:
        name = job['name']
        path = root / 'analysis_results' / name / 'model_performance_summary.csv'
        if statuses.get(name, {}).get('state') == 'complete' and path.exists():
            table = pd.read_csv(path)
            table.insert(0, 'Experiment', name)
            table.insert(1, 'Split', job['split'])
            table.insert(2, 'Features', job['mode'])
            table.insert(3, 'Drop', job.get('drop', ''))
            table.insert(4, 'Only', job.get('only', ''))
            tables.append(table)
    if tables:
        combined = pd.concat(tables, ignore_index=True)
        combined.to_csv(root / 'experiment_summary.csv', index=False)
        lines = ['# 四组实验与消融结果', '',
                 '按训练五折 CV MSE 选模型；测试成绩不参与选型。', '',
                 '| 实验 | 模型 | CV MSE (eV²) | Test R² | RMSE (eV) | MAE (eV) |',
                 '|---|---|---:|---:|---:|---:|']
        for _, row in combined.iterrows():
            lines.append(f"| {row['Experiment']} | {row['Model']} | {row['CV MSE (eV²)']:.6f} | "
                         f"{row['Test R²']:.6f} | {row['Test RMSE']:.6f} | {row['Test MAE']:.6f} |")
        lines += ['', '## 任务状态', '']
        for name, status in statuses.items():
            lines.append(f"- {name}: {status['state']}")
        lines += ['', '随机和骨架实验测试样本不同，不能作逐样本配对比较；同一种划分下可比较特征方案。',
                  'A_number 是预定义宏环骨架家族，未将其宣称为自动 Murcko 骨架。',
                  '消融固定各混合主组由 CV 选出的模型类型，各特征方案独立调参。']
        (root / 'experiment_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def execute_jobs(jobs, args, statuses, all_jobs):
    root = args.output_root
    pending = list(jobs)
    running = {}
    while pending or running:
        while pending and len(running) < args.workers:
            job = pending.pop(0)
            name = job['name']
            command = [sys.executable, '-B', str(Path(__file__).with_name('run_pipeline.py')),
                       str(args.cache), '--data', str(args.data), '--output-root', str(root),
                       '--run-name', name, '--feature-mode', job['mode'],
                       '--split-file', str(root / 'splits.json'), '--split-name', job['split'],
                       '--trials', str(args.trials), '--threads', str(args.threads), '--resume',
                       '--shap-mode', 'none' if job.get('model') else 'best']
            if args.smoke:
                command.append('--smoke')
            if job.get('model'):
                command.extend(['--models', job['model']])
            if job.get('drop'):
                command.extend(['--drop-fingerprint', job['drop']])
            if job.get('only'):
                command.extend(['--only-fingerprint', job['only']])
            env = os.environ.copy()
            for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                env[variable] = str(args.threads)
            log = open(root / 'logs' / (name + '.console.log'), 'a', encoding='utf-8')
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            running[name] = (process, log)
            statuses[name] = {'state': 'running', 'pid': process.pid,
                              'started_at': datetime.now().isoformat(), 'command': command}
            print(f'{datetime.now().isoformat()} START {name} pid={process.pid}', flush=True)
        for name, (process, log) in list(running.items()):
            code = process.poll()
            if code is None:
                progress_path = root / 'models' / name / 'status.json'
                if progress_path.exists():
                    try:
                        statuses[name]['progress'] = json.loads(progress_path.read_text(encoding='utf-8'))
                    except (json.JSONDecodeError, OSError):
                        pass
                continue
            log.close()
            final_path = root / 'models' / name / 'status.json'
            if final_path.exists():
                statuses[name]['progress'] = json.loads(final_path.read_text(encoding='utf-8'))
            if statuses[name].get('progress', {}).get('phase') != 'complete':
                code = code or 1
            statuses[name]['state'] = 'complete' if code == 0 else 'failed'
            statuses[name]['exit_code'] = code
            statuses[name]['finished_at'] = datetime.now().isoformat()
            del running[name]
            print(f'{datetime.now().isoformat()} END {name} code={code}', flush=True)
            collect_results(root, all_jobs, statuses)
        write_json(root / 'status.json', {'phase': 'running', 'updated_at': datetime.now().isoformat(),
                                         'jobs': statuses})
        if running:
            time.sleep(5)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=repo / 'data/Data-20261012.csv')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--ablation', choices=['none', 'drop', 'both'], default='drop')
    parser.add_argument('--experiments', nargs='+', choices=['R-M', 'R-MF', 'S-M', 'S-MF'],
                        default=['R-M', 'R-MF', 'S-M', 'S-MF'], help='本次执行的主实验')
    parser.add_argument('--feature-python', type=Path, help='缺少缓存时，使用该解释器计算全量特征')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.data, args.cache, args.output_root = args.data.resolve(), args.cache.resolve(), args.output_root.resolve()
    if min(args.trials, args.workers, args.threads) < 1:
        parser.error('预算、并发和线程数必须为正数')
    root = args.output_root
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    specification = {'data_sha256': file_sha256(args.data), 'trials': args.trials,
                     'smoke': args.smoke, 'cache': str(args.cache), 'ablation': args.ablation,
                     'threads': args.threads, 'experiments': args.experiments}
    config_path = root / 'experiment_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != specification:
        raise ValueError('实验目录已绑定其它配置，请使用新目录')
    write_json(config_path, specification)
    data = load_valid_data(args.data, smoke=args.smoke)
    if data.Y_number.nunique() != 1:
        raise ValueError('当前不使用 Spin 的方案要求单一自旋数据')
    split_path = root / 'splits.json'
    expected_splits = make_split_manifest(data, args.data, args.smoke)
    if split_path.exists() and json.loads(split_path.read_text(encoding='utf-8')) != expected_splits:
        raise ValueError('已保存划分与当前固定规则不一致')
    write_json(split_path, expected_splits)
    write_json(root / 'dataset_audit.json', {
        'valid_samples': len(data), 'spin_values': data.Y_number.unique().tolist(), 'spin_used': False,
        'A_counts': data.A_number.value_counts().sort_index().to_dict(),
        'C_counts': data.C_number.value_counts().sort_index().to_dict(),
        'splits': {name: {'train': len(s['train_positions']), 'test': len(s['test_positions']),
                          'train_A': s['train_A'], 'test_A': s['test_A']}
                   for name, s in expected_splits['splits'].items()}})
    if not args.cache.exists():
        if args.feature_python is None:
            raise ValueError('缺少特征缓存，请先生成或指定 --feature-python')
        write_json(root / 'status.json', {'phase': 'features', 'updated_at': datetime.now().isoformat()})
        command = [str(args.feature_python), '-B', str(repo/'prepare_features.py'), '--data', str(args.data),
                   '--output', str(args.cache), '--nproc', str(args.threads)]
        if any(name.endswith('MF') for name in args.experiments):
            command.append('--fingerprints')
        if args.smoke:
            command.append('--smoke')
        with open(root/'logs/features.log', 'a', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    # 先核对新缓存，而不是盲目信任同名文件。
    features, _ = load_feature_cache(args.cache, data, args.data)
    del features
    jobs = [{'name': 'R-M', 'split': 'random', 'mode': 'mordred'},
            {'name': 'R-MF', 'split': 'random', 'mode': 'mixed'},
            {'name': 'S-M', 'split': 'scaffold', 'mode': 'mordred'},
            {'name': 'S-MF', 'split': 'scaffold', 'mode': 'mixed'}]
    statuses = {}
    jobs = [job for job in jobs if job['name'] in args.experiments]
    write_json(root/'jobs.json', jobs)
    execute_jobs(jobs, args, statuses, jobs)
    ablations = []
    if args.ablation != 'none':
        for base in ['R-MF', 'S-MF']:
            if base not in statuses:
                continue
            if statuses[base]['state'] != 'complete':
                statuses[base + '-ablation'] = {'state': 'blocked', 'reason': '混合主组未完成'}
                continue
            metadata = json.loads((root/'models'/base/'run_metadata.json').read_text(encoding='utf-8'))
            model = metadata['best_model_by_cv']
            split = 'random' if base.startswith('R-') else 'scaffold'
            for family in FP_PREFIXES:
                ablations.append({'name': base+'-minus-'+family, 'split': split, 'mode': 'mixed',
                                  'model': model, 'drop': family})
                if args.ablation == 'both':
                    ablations.append({'name': base+'-only-'+family, 'split': split, 'mode': 'mixed',
                                      'model': model, 'only': family})
        all_jobs = jobs + ablations
        write_json(root/'jobs.json', all_jobs)
        execute_jobs(ablations, args, statuses, all_jobs)
    else:
        all_jobs = jobs
    collect_results(root, all_jobs, statuses)
    failed = any(s['state'] != 'complete' for s in statuses.values())
    write_json(root/'status.json', {'phase': 'failed' if failed else 'complete',
                                   'updated_at': datetime.now().isoformat(), 'jobs': statuses})
    print('FAILED' if failed else 'COMPLETE', root, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
