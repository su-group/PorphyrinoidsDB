"""四组实验共用的特征选择与固定划分；不依据目标值挑选测试集。"""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, GroupKFold
from core.data_processing import file_sha256, sample_records

FP_PREFIXES = {
    'Morgan': 'Morgan_', 'MACCS': 'MACCS_', 'AtomPair': 'AtomPair_',
    'TopoTorsion': 'TopoTorsion_', 'RDKitFP': 'RDKitFP_',
}


def select_experiment_features(features, mode='mordred', drop_fingerprint=None, only_fingerprint=None):
    if drop_fingerprint and only_fingerprint:
        raise ValueError('逐一移除和逐一添加不能同时指定')
    if mode == 'mordred' and (drop_fingerprint or only_fingerprint):
        raise ValueError('指纹消融必须使用 mixed 模式')
    all_prefixes = tuple(FP_PREFIXES.values())
    names = []
    for name in features.columns:
        if name.startswith('Spin') or name == 'Y_number':
            continue
        is_fp = name.startswith(all_prefixes)
        if is_fp and mode == 'mordred':
            continue
        if drop_fingerprint and name.startswith(FP_PREFIXES[drop_fingerprint]):
            continue
        if only_fingerprint and is_fp and not name.startswith(FP_PREFIXES[only_fingerprint]):
            continue
        names.append(name)
    if mode == 'mixed':
        required = [only_fingerprint] if only_fingerprint else [p for p in FP_PREFIXES if p != drop_fingerprint]
        for family in required:
            if not any(name.startswith(FP_PREFIXES[family]) for name in names):
                raise ValueError(f'缓存缺少所需指纹: {family}')
    if not names:
        raise ValueError('没有可用的分子特征')
    return features[names].copy()


def make_split_manifest(data, data_path, smoke=False):
    if data['A_number'].isna().any() or data['SMILES'].duplicated().any():
        raise ValueError('骨架编号缺失或存在重复规范 SMILES，需要先检查数据')
    positions = np.arange(len(data))
    random_train, random_test = train_test_split(positions, test_size=0.2, random_state=42)
    groups = data['A_number'].astype(str).to_numpy()
    candidates = GroupShuffleSplit(n_splits=200, test_size=0.2, random_state=42)
    best = None
    best_distance = float('inf')
    for train, test in candidates.split(positions, groups=groups):
        ratio = len(test) / len(data)
        if not 0.15 <= ratio <= 0.25:
            continue
        if len(set(groups[train])) < 5 or len(set(groups[test])) < 2:
            continue
        distance = abs(ratio - 0.2)
        if distance < best_distance:
            best, best_distance = (train, test), distance
    if best is None:
        raise ValueError('200 个候选中没有满足样本比例和骨架组数要求的划分')
    splits = {}
    for name, (train, test) in [('random', (random_train, random_test)), ('scaffold', best)]:
        if name == 'random':
            splitter = KFold(n_splits=5, shuffle=True, random_state=42)
            cv = splitter.split(train)
        else:
            splitter = GroupKFold(n_splits=5)
            cv = splitter.split(train, groups=groups[train])
        folds = []
        for fit, valid in cv:
            folds.append({'train_positions': fit.tolist(), 'valid_positions': valid.tolist()})
        splits[name] = {
            'train_positions': train.tolist(), 'test_positions': test.tolist(), 'folds': folds,
            'train_A': sorted(data.iloc[train]['A_number'].unique().tolist()),
            'test_A': sorted(data.iloc[test]['A_number'].unique().tolist()),
            'test_fraction': len(test) / len(data),
        }
    return {'schema_version': 1, 'data_sha256': file_sha256(data_path),
            'samples': sample_records(data), 'smoke': smoke, 'seed': 42,
            'split_selection': '200 candidates; closest to 20% by count; first tie; no targets',
            'splits': splits}


def split_group_column(path, name):
    """返回某个固定划分所使用的分组列；'random' 无分组，'scaffold' 默认按骨架。"""
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    split = manifest['splits'][name]
    column = split.get('group_column')
    if column is None and name == 'scaffold':
        column = 'A_number'
    return column


def load_split_manifest(path, data, data_path, name, smoke=False):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if (manifest['data_sha256'] != file_sha256(data_path) or
            manifest['samples'] != sample_records(data) or manifest['smoke'] != smoke):
        raise ValueError('固定划分与当前数据/样本顺序/冒烟模式不一致')
    split = manifest['splits'][name]
    group_column = split_group_column(path, name)
    if group_column is not None:
        groups_all = data[group_column].astype(str).to_numpy()
    train = np.asarray(split['train_positions'], dtype=int)
    test = np.asarray(split['test_positions'], dtype=int)
    if len(set(train) & set(test)) or sorted(np.r_[train, test].tolist()) != list(range(len(data))):
        raise ValueError('外部划分存在重复或遗漏样本')
    if group_column is not None and set(groups_all[train]) & set(groups_all[test]):
        raise ValueError(f'{group_column} 跨外部训练/测试集')
    seen = []
    if len(split['folds']) != 5:
        raise ValueError('必须恰好五折')
    for fold in split['folds']:
        fit, valid = fold['train_positions'], fold['valid_positions']
        if set(fit) & set(valid) or sorted(fit + valid) != list(range(len(train))):
            raise ValueError('CV 划分存在重复或遗漏')
        if group_column is not None and set(groups_all[train[fit]]) & set(groups_all[train[valid]]):
            raise ValueError(f'{group_column} 跨 CV 训练/验证折')
        seen.extend(valid)
    if sorted(seen) != list(range(len(train))):
        raise ValueError('五折验证未恰好覆盖一次训练集')
    return train, test, split['folds']
