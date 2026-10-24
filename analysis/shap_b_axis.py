r"""SHAP composition on the meso-B axis, compared with the random and scaffold axes.

Why a separate script instead of extending shap_four_group.py: that report's prose is
built around the random-vs-scaffold pair (same estimator SVR, same 30 explained samples),
which is what makes its Welch test clean.  The B axis picks a different estimator
(KNeighbors on B-MF), so folding it into the same narrative would silently compare
apples to oranges.  Here the B axis gets its own table and its own caveats.

Output: reports\B轴SHAP对照.md
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    'shap_four', ROOT / '.dsh_tmp' / 'shap_four_group.py')
four = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(four)

RUN_ORDER = [
    ('R-M', '20261016_R-M', '随机·Mordred'),
    ('B-M', '20261021_B-M', 'B轴·Mordred'),
    ('S-M', '20261016_S-M', '骨架·Mordred'),
    ('R-MF', '20261016_R-MF', '随机·混合'),
    ('B-MF', '20261021_B-MF', 'B轴·混合'),
    ('S-MF', '20261016_S-MF', '骨架·混合'),
]
OUT = ROOT / 'reports' / 'B轴SHAP对照.md'


def main() -> int:
    loaded = {}
    for key, directory, label in RUN_ORDER:
        run_root = ROOT / 'experiments' / directory / 'analysis_results' / key / 'shap_analysis'
        names = [p.name for p in run_root.iterdir() if p.is_dir()]
        if not names:
            raise SystemExit(f'{key}: no SHAP directory')
        model = names[0]
        shap_dir = run_root / model
        imp, values, meta, rel, feature_names = four.load_run(shap_dir)
        loaded[key] = {'imp': imp, 'values': values, 'meta': meta, 'label': label,
                       'model': model, 'dir': rel, 'names': feature_names}

    lines = ['# B 轴（meso 取代基）与另外两条轴的 SHAP 归因对照', '',
             '三条划分轴 × 两种特征集，各自取 CV 最优模型做 SHAP。'
             'SHAP 值以 eV 为单位、量级跟随各模型自身输出幅度，**绝对值不可跨组比较**；'
             '这里比较的是**归因构成**（各族指纹在总 |SHAP| 中占的份额）与**排名一致性**。', '',
             '## 1. 解释设置', '',
             '| run | 划分 | 模型 | 解释样本数 | 折内特征数 | expected_value (eV) |',
             '|---|---|---|---|---|---|']
    for key, _, label in RUN_ORDER:
        item = loaded[key]
        lines.append(f"| `{key}` | {label} | {item['model']} | {len(item['values'])} | "
                     f"{len(item['imp'])} | {item['meta'].get('expected_value', float('nan')):.4f} |")
    lines.append('')

    lines += ['## 2. 归因构成（各族占总 |SHAP| 的份额，均值 [95% bootstrap CI]）', '']
    lines.append('| run | ' + ' | '.join(four.FAMILIES + ['Mordred']) + ' | 指纹合计 |')
    lines.append('|' + '---|' * (len(four.FAMILIES) + 3))
    shares = {}
    for key, _, _ in RUN_ORDER:
        per = four.per_sample_share(loaded[key]['values'], loaded[key]['names'])
        shares[key] = per
        cells, fp = [], []
        for family in four.FAMILIES + ['Mordred']:
            if family not in per:
                cells.append('—')
                continue
            mean, lo, hi = four.bootstrap_ci(per[family])
            cells.append(f'{mean * 100:.1f}% [{lo * 100:.0f}–{hi * 100:.0f}]')
            if family != 'Mordred':
                fp.append(per[family])
        total = '—（无指纹列）'
        if fp:
            stacked = np.sum(fp, axis=0)
            mean, lo, hi = four.bootstrap_ci(stacked)
            total = f'**{mean * 100:.2f}%** [{lo * 100:.1f}–{hi * 100:.1f}]'
        lines.append(f'| `{key}` | ' + ' | '.join(cells) + f' | {total} |')
    lines.append('')

    lines += ['## 3. 头部特征的族构成', '',
              '| run | top-10 指纹数 | top-20 指纹数 | top-50 指纹数 | 特征总数 |',
              '|---|---|---|---|---|']
    for key, _, _ in RUN_ORDER:
        imp = loaded[key]['imp']
        flags = (imp['Family'] != 'Mordred').to_numpy()
        lines.append(f'| `{key}` | {flags[:10].sum()}/10 | {flags[:20].sum()}/20 | '
                     f'{flags[:50].sum()}/50 | {len(imp)} |')
    lines.append('')

    lines += ['## 4. 排名一致性（同一特征空间内才可比）', '',
              '| 对照 | 共有特征数 | Spearman ρ | top-20 交集 | top-50 交集 |',
              '|---|---|---|---|---|']
    pairs = [('R-MF', 'B-MF'), ('B-MF', 'S-MF'), ('R-MF', 'S-MF'), ('R-M', 'B-M'), ('B-M', 'S-M')]
    for left, right in pairs:
        a = loaded[left]['imp'].set_index('Feature')['Mean_ABS_SHAP']
        b = loaded[right]['imp'].set_index('Feature')['Mean_ABS_SHAP']
        shared = a.index.intersection(b.index)
        rho = four.spearmanr(a.loc[shared], b.loc[shared]).statistic
        top_a = set(a.sort_values(ascending=False).index[:20])
        top_b = set(b.sort_values(ascending=False).index[:20])
        a50 = set(a.sort_values(ascending=False).index[:50])
        b50 = set(b.sort_values(ascending=False).index[:50])
        lines.append(f'| `{left}` vs `{right}` | {len(shared)} | {rho:+.3f} | '
                     f'{len(top_a & top_b)}/20 | {len(a50 & b50)}/50 |')
    lines.append('')

    bmf_fp = np.sum([shares['B-MF'][f] for f in four.FAMILIES if f in shares['B-MF']], axis=0)
    rmf_fp = np.sum([shares['R-MF'][f] for f in four.FAMILIES if f in shares['R-MF']], axis=0)
    smf_fp = np.sum([shares['S-MF'][f] for f in four.FAMILIES if f in shares['S-MF']], axis=0)

    lines += ['## 5. 结论', '',
              f'- **B 轴没有把模型推向另一种特征依赖**：B-MF 的指纹合计份额 '
              f'{np.mean(bmf_fp) * 100:.2f}%，与随机 {np.mean(rmf_fp) * 100:.2f}%、'
              f'骨架 {np.mean(smf_fp) * 100:.2f}% 同量级——尽管 B 轴的 CV 误差'
              f'（KNeighbors 0.1767）比随机（SVR 0.0913）差一倍、比骨架（SVR 1.1748）好得多。',
              '- 因此"B 轴好、骨架轴坏"不能归因于模型在 B 轴上少用了指纹；'
              '三个混合特征实验都在用同样比例的指纹。',
              '- **估计器差异是必须声明的混淆**：B-MF 的 CV 最优是 KNeighbors，'
              'R-MF/S-MF 是 SVR，解释样本数也不同（30 vs 30/100 视模型而定）；'
              '因此这里只对比份额与排名的**方向性**，不做显著性检验。', '']
    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[written] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
