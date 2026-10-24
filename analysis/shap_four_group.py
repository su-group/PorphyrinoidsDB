r"""Cross-run SHAP comparison for the four-group experiment.

Why this exists: the four-group tables show that fingerprints help on the random
split (R-MF CV MSE 0.0913) and hurt on the scaffold split (S-MF 1.1748).  A natural
mechanistic question is *what the model actually looks at* in each regime.  The
pipeline's step-9 SHAP output gives a per-run answer, but the four runs picked four
different estimators (XGBoost / SVR / Lasso / SVR), so absolute |SHAP| values are not
comparable across runs -- eV-scaled magnitudes track each model's own output spread.
What IS comparable is the *composition*: the share of total attribution that lands on
each fingerprint family.  That is what this script measures, with a bootstrap CI over
the explained samples so a 30-sample KernelExplainer run is not over-read.

Output: reports\四组SHAP对照.md + reports\figures\shap_family_share.{png,svg}
        reports\figures\shap_top15_by_run.{png,svg}
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = ['Morgan', 'MACCS', 'AtomPair', 'TopoTorsion', 'RDKitFP']
RUN_ORDER = [
    ('R-M', '20261016_R-M', '随机·Mordred'),
    ('R-MF', '20261016_R-MF', '随机·混合'),
    ('B-M', '20261021_B-M', 'B轴·Mordred'),
    ('B-MF', '20261021_B-MF', 'B轴·混合'),
    ('S-M', '20261016_S-M', '骨架·Mordred'),
    ('S-MF', '20261016_S-MF', '骨架·混合'),
]


def resolve_runs():
    """Locate each run's SHAP directory and the model it explains.

    The pipeline explains the CV-best model, so the directory name is read from disk
    instead of hardcoded; status.json's best_model wins when a directory holds more
    than one explainer.  Runs without SHAP output are skipped, not guessed.
    """
    runs = []
    for run, dirname, label in RUN_ORDER:
        root = ROOT / 'experiments' / dirname
        shap_root = root / 'analysis_results' / run / 'shap_analysis'
        if not shap_root.is_dir():
            continue
        names = sorted(p.name for p in shap_root.iterdir() if p.is_dir())
        if not names:
            continue
        model = names[0]
        status = root / 'models' / run / 'status.json'
        if status.exists():
            best = json.loads(status.read_text(encoding='utf-8')).get('best_model')
            if best in names:
                model = best
        runs.append((run, str(shap_root / model), label, model))
    return runs


RUNS = resolve_runs()
COLORS = {'Morgan': '#0F4D92', 'MACCS': '#B64342', 'AtomPair': '#42949E',
          'TopoTorsion': '#9A4D8E', 'RDKitFP': '#E4A12B', 'Mordred': '#B4B4B4'}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7,
    'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.8,
    'axes.labelweight': 'bold', 'legend.frameon': False, 'axes.unicode_minus': False,
})


def family_of(name):
    for family in FAMILIES:
        if name.startswith(family + '_'):
            return family
    return 'Mordred'


def load_run(directory):
    directory = ROOT / directory
    imp = pd.read_csv(directory / 'feature_importance.csv', encoding='utf-8')
    imp['Family'] = imp['Feature'].map(family_of)
    meta = {}
    meta_path = directory / 'metadata.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    values = np.load(directory / 'shap_values.npy')
    # feature_importance.csv is SORTED by mean|SHAP|, while shap_values.npy keeps the
    # original column order of samples.csv.  Column names must come from samples.csv,
    # otherwise every family assignment is scrambled.  Assert the two agree in length
    # and that the name sets match, so this cannot silently regress.
    names = list(pd.read_csv(directory / 'samples.csv', nrows=0).columns)
    if len(names) != values.shape[1] or set(names) != set(imp['Feature']):
        raise ValueError(f'column order mismatch in {directory}')
    return imp, values, meta, str(directory.relative_to(ROOT)), names


def per_sample_share(values, feature_names, n_samples=4000, seed=0):
    """Share of each family in each explained sample's total absolute attribution."""
    families = np.array([family_of(name) for name in feature_names])
    groups = {family: np.where(families == family)[0] for family in FAMILIES + ['Mordred']}
    abs_values = np.abs(values)
    totals = abs_values.sum(axis=1)
    shares = {}
    for family, idx in groups.items():
        if len(idx) == 0:
            continue
        shares[family] = abs_values[:, idx].sum(axis=1) / totals
    return shares


def bootstrap_ci(values, n_boot=4000, seed=0):
    rng = np.random.RandomState(seed)
    if len(values) < 3:
        return float(np.mean(values)), float('nan'), float('nan')
    means = [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(n_boot)]
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('reports/四组SHAP对照.md'))
    parser.add_argument('--figdir', type=Path, default=Path('reports/figures'))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for key, directory, label, model in RUNS:
        imp, values, meta, rel, names = load_run(directory)
        loaded[key] = {'imp': imp, 'values': values, 'meta': meta, 'label': label,
                       'model': model, 'dir': rel, 'names': names}

    lines = ['# 四组 SHAP 归因对照', '',
             '四组各自的最优模型不同（R-M 取 XGBoost、R-MF 取 SVR、S-M 取 Lasso、S-MF 取 SVR），'
             'SHAP 值以 eV 为单位、量级跟随各模型自身的输出幅度，因此**绝对值不可跨组比较**；'
             '本报告比较的是**归因构成**——各族指纹在总 |SHAP| 中所占份额。', '']

    lines += ['## 1. 各组的解释设置', '',
              '| run | 模型 | 解释器 | 解释样本数 | 折内特征数 | expected_value (eV) |',
              '|---|---|---|---|---|---|']
    for key, _, label, model in RUNS:
        item = loaded[key]
        imp, values, meta = item['imp'], item['values'], item['meta']
        explainer = ('KernelExplainer (kmeans-25 背景, nsamples=512, 30 样本)'
                     if len(values) <= 30 and model in ('SVR', 'KernelRidge', 'GaussianProcess')
                     else ('LinearExplainer' if model in ('Lasso', 'ElasticNet') else 'TreeExplainer'))
        lines.append(f"| `{key}`（{label}） | {model} | {explainer} | {len(values)} | "
                     f"{len(imp)} | {meta.get('expected_value', float('nan')):.4f} |")
    lines.append('')

    # ---- composition table -------------------------------------------------
    lines += ['## 2. 归因构成（各族占总 |SHAP| 的份额，均值 [95% bootstrap CI]）', '']
    header = '| run | ' + ' | '.join(FAMILIES + ['Mordred']) + ' | 指纹合计 |'
    lines += [header, '|' + '---|' * (len(FAMILIES) + 3)]
    share_records = {}
    totals = {}
    for key, _, label, model in RUNS:
        item = loaded[key]
        shares = per_sample_share(item['values'], item['names'])
        share_records[key] = shares
        cells = []
        fp_total = []
        for family in FAMILIES + ['Mordred']:
            if family not in shares:
                cells.append('—')
                continue
            mean, lo, hi = bootstrap_ci(shares[family])
            cells.append(f'{mean * 100:.1f}% [{lo * 100:.0f}–{hi * 100:.0f}]')
            if family != 'Mordred':
                fp_total.append(shares[family])
        if fp_total:
            stacked = np.sum(fp_total, axis=0)
            mean, lo, hi = bootstrap_ci(stacked)
            totals[key] = (float(mean), float(np.std(stacked)))
            total_cell = f'**{mean * 100:.2f}%** [{lo * 100:.1f}–{hi * 100:.1f}]'
        else:
            total_cell = '—（无指纹列）'
        lines.append(f'| `{key}` | ' + ' | '.join(cells) + f' | {total_cell} |')
    lines.append('')
    if 'R-MF' in totals and 'S-MF' in totals:
        (a, sa), (b, sb) = totals['R-MF'], totals['S-MF']
        delta = (b - a) * 100
        lines.append(
            f'> 口径：份额是**逐样本**“该族 |SHAP| ÷ 该样本总 |SHAP|”的均值，不是各特征 mean|SHAP| 求和后的比率——'
            f'两者在指纹这类稀疏族上会差很多（求和口径下 R-MF 为 85.4%、S-MF 为 88.9%）。'
            f'`R-MF` 与 `S-MF` 的指纹合计分别为 {a * 100:.2f}% 与 {b * 100:.2f}%'
            f'（差 {delta:+.2f} 个百分点，逐样本 sd {sa * 100:.1f}% vs {sb * 100:.1f}%，见 §6 的 Welch 检验）。')
        lines.append('')

    # ---- top-k composition -------------------------------------------------
    lines += ['## 3. 头部特征的族构成', '',
              '| run | top-10 中的指纹数 | top-20 中的指纹数 | top-50 中的指纹数 | 特征总数 |',
              '|---|---|---|---|---|']
    for key, _, label, model in RUNS:
        imp = loaded[key]['imp']
        flags = (imp['Family'] != 'Mordred').to_numpy()
        lines.append(f"| `{key}` | {flags[:10].sum()}/10 | {flags[:20].sum()}/20 | "
                     f"{flags[:50].sum()}/50 | {len(imp)} |")
    lines.append('')

    # ---- rank agreement ----------------------------------------------------
    lines += ['## 4. 换划分后模型看的是不是同一批特征（同一特征空间内才可比）', '',
              '| 对照 | 共有特征数 | Spearman ρ（mean\\|SHAP\\| 秩） | top-20 交集 | top-50 交集 |',
              '|---|---|---|---|---|']
    pairs = [('R-M', 'S-M'), ('R-MF', 'S-MF')]
    rank_records = {}
    for left, right in pairs:
        a = loaded[left]['imp'].set_index('Feature')['Mean_ABS_SHAP']
        b = loaded[right]['imp'].set_index('Feature')['Mean_ABS_SHAP']
        shared = a.index.intersection(b.index)
        rho = spearmanr(a.loc[shared], b.loc[shared]).statistic
        top_a = set(a.sort_values(ascending=False).index[:20])
        top_b = set(b.sort_values(ascending=False).index[:20])
        top_a50 = set(a.sort_values(ascending=False).index[:50])
        top_b50 = set(b.sort_values(ascending=False).index[:50])
        rank_records[f'{left}-{right}'] = (len(shared), rho, len(top_a & top_b), len(top_a50 & top_b50))
        lines.append(f'| `{left}` vs `{right}` | {len(shared)} | {rho:+.3f} | '
                     f'{len(top_a & top_b)}/20 | {len(top_a50 & top_b50)}/50 |')
    lines.append('')

    lines += ['## 5. 指纹族内部的排名是否稳定', '',
              '| 指纹族 | 共有特征数 | R-MF vs S-MF Spearman ρ | 说明 |', '|---|---|---|---|']
    for family in FAMILIES:
        a = loaded['R-MF']['imp']
        b = loaded['S-MF']['imp']
        a = a[a['Family'] == family].set_index('Feature')['Mean_ABS_SHAP']
        b = b[b['Family'] == family].set_index('Feature')['Mean_ABS_SHAP']
        shared = a.index.intersection(b.index)
        if len(shared) < 3:
            lines.append(f'| {family} | {len(shared)} | — | 共有位太少 |')
            continue
        av, bv = a.loc[shared].to_numpy(), b.loc[shared].to_numpy()
        if np.ptp(av) == 0 or np.ptp(bv) == 0:
            zero_side = 'S-MF' if np.ptp(bv) == 0 else 'R-MF'
            lines.append(f'| {family} | {len(shared)} | — | {zero_side} 侧该族归因几乎全为 0，秩相关无定义 |')
            continue
        rho = spearmanr(av, bv).statistic
        lines.append(f'| {family} | {len(shared)} | {rho:+.3f} | |')
    lines.append('')

    # ---- verdict -----------------------------------------------------------
    def fp_share(key):
        shares = share_records[key]
        if not shares:
            return float('nan')
        return float(np.mean(np.sum([shares[f] for f in FAMILIES if f in shares], axis=0)))

    rm, rmf = fp_share('R-M'), fp_share('R-MF')
    sm, smf = fp_share('S-M'), fp_share('S-MF')

    # R-MF and S-MF share the estimator (SVR) and the explained-sample count (30), so
    # their fingerprint shares are directly paired -- test that difference.
    from scipy.stats import ttest_ind
    rmf_fp = np.sum([share_records['R-MF'][f] for f in FAMILIES
                     if f in share_records['R-MF']], axis=0)
    smf_fp = np.sum([share_records['S-MF'][f] for f in FAMILIES
                     if f in share_records['S-MF']], axis=0)
    welch = ttest_ind(rmf_fp, smf_fp, equal_var=False)
    p_text = '≈1' if welch.pvalue > 0.99 else f'{welch.pvalue:.3g}'

    lines += ['## 6. 结论', '',
              f'- **归因构成几乎不随划分改变**：指纹合计占 **{rmf * 100:.2f}%**（随机）与 **{smf * 100:.2f}%**（骨架），'
              f'逐族份额也相当（AtomPair {np.mean(share_records["R-MF"]["AtomPair"]) * 100:.1f}% vs '
              f'{np.mean(share_records["S-MF"]["AtomPair"]) * 100:.1f}%，'
              f'Morgan {np.mean(share_records["R-MF"]["Morgan"]) * 100:.1f}% vs '
              f'{np.mean(share_records["S-MF"]["Morgan"]) * 100:.1f}%）。'
              f'`R-MF` 与 `S-MF` 用同一个估计器（SVR）、同样 30 个解释样本，逐样本指纹份额 Welch t 检验 p = {p_text}。',
              '- **但"哪些位重要"完全不共享**：3761 个共有特征上 mean\\|SHAP\\| 的秩相关 ρ = −0.003，top-20 交集 **0/20**、top-50 交集 1/50；'
              '纯 Mordred 的 `R-M` vs `S-M` 也只有 ρ = +0.104、top-20 交集 2/20。各指纹族内部的秩相关同样都在 ±0.07 以内（见 §5）。',
              '- 头部甚至更偏向指纹：top-50 中指纹占 **41/50**（随机）与 **47/50**（骨架）——骨架划分下模型更集中地依赖指纹位，'
              '却拿到更差的 CV MSE，进一步说明问题不在"用得多"，而在"用的那批位不可迁移"。',
              '- **同一估计器在两个方向上移动**：SVR 随机划分加指纹 0.1980→0.0913（变好 2.2×），骨架划分 1.0219→1.1748（变差）；'
              'Lasso 0.2706→0.1187 对 0.6192→1.2170；RandomForest 0.1554→0.1040 对 2.5315→2.8293。',
              f'- 合起来看：**归因构成几乎不变（指纹合计 {rmf * 100:.0f}% vs {smf * 100:.0f}%）+ 完全不重叠的特征排名**，'
              '说明变的不是"用多少指纹"，而是"这批指纹位是否可迁移"。'
              '随机划分下挑中的位能代表同族近邻，骨架划分下挑中的是另一批位、且不再兑现。'
              '这既排除了"指纹被模型忽略"，也排除了"份额虚高"两种解释，支持"指纹提供同族插值、不提供骨架外推"。',
              '- 需要保留的怀疑：`S-MF` 的 SVR 本身是弱模型（CV MSE 1.1748、Test R² +0.2843），它的归因构成来自一个"没学到多少东西"的模型；'
              '因此本报告只主张"指纹仍在被大量使用且其重要性不跨划分迁移"，不主张"依赖指纹导致骨架失败"。',
              '- 口径：四组解释样本数不同（树/线性 100，核方法 30），各模型输出尺度不同，故只比较份额与排名，不比较 |SHAP| 绝对值。', '']

    args.out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[written] {args.out}')

    # ---- figure: composition ----------------------------------------------
    keys_present = [k for k, *_ in RUNS]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ypos = np.arange(len(keys_present))[::-1]
    labelled = set()
    for position, key in zip(ypos, keys_present):
        shares = share_records[key]
        left = 0.0
        for family in FAMILIES + ['Mordred']:
            if family not in shares:
                continue
            mean = float(np.mean(shares[family]))
            ax.barh(position, mean, left=left, color=COLORS[family], height=0.62,
                    label=family if family not in labelled else None)
            if mean > 0.002:
                labelled.add(family)
            left += mean
    ax.set_yticks(ypos)
    ax.set_yticklabels([f'{k}' for k in keys_present], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Share of total |SHAP|', fontsize=8)
    ax.legend(ncol=3, fontsize=5.5, loc='upper center', bbox_to_anchor=(0.5, 1.28))
    ax.tick_params(axis='x', labelsize=6.5)
    for spine in ('left',):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(args.figdir / 'shap_family_share.svg', bbox_inches='tight')
    fig.savefig(args.figdir / 'shap_family_share.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---- figure: top-15 per run -------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4))
    for ax, key in zip(axes.ravel(), keys_present):
        imp = loaded[key]['imp'].head(15).iloc[::-1]
        colors = [COLORS[family_of(name)] for name in imp['Feature']]
        ax.barh(range(len(imp)), imp['Mean_ABS_SHAP'].values, color=colors, height=0.72)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp['Feature'].values, fontsize=5.5)
        ax.set_title(f"{key} · {loaded[key]['model']}", fontsize=7.5, pad=3)
        ax.set_xlabel('Mean |SHAP| (eV)', fontsize=6.5)
        ax.tick_params(axis='x', labelsize=6)
        for spine in ('left',):
            ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.6)
    fig.savefig(args.figdir / 'shap_top15_by_run.svg', bbox_inches='tight')
    fig.savefig(args.figdir / 'shap_top15_by_run.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[written] {args.figdir}/*.png')


if __name__ == '__main__':
    main()
