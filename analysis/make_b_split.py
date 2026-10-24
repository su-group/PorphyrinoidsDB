r"""Append a meso-substituent (B-axis) split to a copy of the project split manifest.

Motivation: the manuscript only splits on the porphyrin core (A_number).  The
B axis (meso substituent, 16 groups) is a second out-of-distribution axis and
earlier probing suggested it generalises far better than the core axis.  This
script builds a manifest with the same schema/provenance fields as
`experiments/20261013_R-M/splits.json` so `core.experiment_setup.load_split_manifest`
accepts it unchanged, and adds one entry:

    'meso' : outer test = 2-3 held-out B groups, inner CV = GroupKFold over B

Design note (documented in the manifest as 'split_selection'): the `B00` group
(meso-unsubstituted; only A13/A14 use it) is kept on the training side, so this
split isolates the meso axis and does not double as a core split.

    python -B .dsh_tmp\make_b_split.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data_processing import load_valid_data                      # noqa: E402
from core.experiment_setup import sample_records, file_sha256          # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit      # noqa: E402

DATA = ROOT / 'data' / 'Data-20261012.csv'
BASE = ROOT / 'experiments' / '20261013_R-M' / 'splits.json'
OUT = ROOT / 'experiments' / '20261021_splits_meso.json'
GROUP_COLUMN = 'B_number'
# B_number 在数据里是 0..15 的整数（编号写作 B00..B15），所以这里是字符串 '0'。
PROTECTED = {'0'}


def main() -> int:
    data = load_valid_data(DATA)
    base = json.loads(BASE.read_text(encoding='utf-8'))
    if base['data_sha256'] != file_sha256(DATA) or base['samples'] != sample_records(data):
        raise SystemExit('基础划分清单与当前数据不一致，请勿复用')

    positions = np.arange(len(data))
    groups = data[GROUP_COLUMN].astype(str).to_numpy()
    unique = sorted(set(groups))
    print(f'rows={len(data)}  {GROUP_COLUMN} groups={len(unique)}  {unique}')

    # 先在“允许作为测试组”的组上搜索外层划分，再校验 B00 是否被误留在训练侧。
    pool = np.array([g not in PROTECTED for g in groups])
    candidates = GroupShuffleSplit(n_splits=200, test_size=0.2, random_state=42)
    best, best_distance = None, float('inf')
    for train, test in candidates.split(positions[pool], groups=groups[pool]):
        train, test = positions[pool][train], positions[pool][test]
        held = set(groups[test])
        if held & PROTECTED:
            continue
        ratio = len(test) / len(data)
        if not 0.15 <= ratio <= 0.25:
            continue
        if len(held) < 2 or len(set(groups[train])) < 5:
            continue
        distance = abs(ratio - 0.2)
        if distance < best_distance:
            best, best_distance = (train, test), distance
    if best is None:
        raise SystemExit('没有满足条件的 meso 划分候选')
    train, test = best
    # 搜索只在“允许作为测试组”的样本上进行，受保护组（B0）需要补回训练侧。
    train = np.setdiff1d(positions, test)
    print(f'outer: train={len(train)}  test={len(test)}  '
          f'held-out groups={sorted(set(groups[test]))}  '
          f'ratio={len(test) / len(data):.4f}')

    folds = []
    splitter = GroupKFold(n_splits=5)
    for fit, valid in splitter.split(train, groups=groups[train]):
        folds.append({'train_positions': fit.tolist(), 'valid_positions': valid.tolist()})

    # ---- 与 load_split_manifest 等价的断言，外加 B 轴专属的泄漏检查 ----
    if set(train) & set(test) or sorted(np.r_[train, test].tolist()) != list(range(len(data))):
        raise SystemExit('外层划分存在重复或遗漏样本')
    if set(groups[train]) & set(groups[test]):
        raise SystemExit('meso 组跨外部训练/测试集')
    seen = []
    for index, fold in enumerate(folds):
        fit, valid = fold['train_positions'], fold['valid_positions']
        if set(fit) & set(valid) or sorted(fit + valid) != list(range(len(train))):
            raise SystemExit(f'第 {index} 折存在重复或遗漏')
        if set(groups[train[fit]]) & set(groups[train[valid]]):
            raise SystemExit(f'第 {index} 折 meso 组跨越训练/验证')
        seen.extend(valid)
    if sorted(seen) != list(range(len(train))):
        raise SystemExit('五折验证未恰好覆盖一次训练集')
    print('assertions: outer disjoint, no B overlap in outer split or any fold, folds cover once')
    for index, fold in enumerate(folds):
        valid_groups = sorted(set(groups[train[fold['valid_positions']]]))
        print(f'  fold{index}: fit={len(fold["train_positions"])} '
              f'valid={len(fold["valid_positions"])} valid_groups={valid_groups}')

    manifest = {
        'schema_version': base['schema_version'],
        'data_sha256': base['data_sha256'],
        'samples': base['samples'],
        'smoke': base['smoke'],
        'seed': base['seed'],
        'split_selection': (base['split_selection'] +
                            ' | meso: GroupShuffleSplit(200, seed 42) over B_number, '
                            'closest to 20%, B00 kept on the training side so the split '
                            'isolates the meso axis; inner CV = GroupKFold(5) over B_number'),
        'splits': dict(base['splits']),
    }
    manifest['splits']['meso'] = {
        'train_positions': train.tolist(),
        'test_positions': test.tolist(),
        'folds': folds,
        'train_A': sorted(data.iloc[train]['A_number'].unique().tolist()),
        'test_A': sorted(data.iloc[test]['A_number'].unique().tolist()),
        'test_fraction': len(test) / len(data),
        'group_column': GROUP_COLUMN,
        'train_groups': sorted(set(groups[train])),
        'test_groups': sorted(set(groups[test])),
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    print(f'[written] {OUT}  splits={sorted(manifest["splits"])}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
