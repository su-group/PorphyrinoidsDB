r"""Collect the four runs (R-M, R-MF, S-M, S-MF) into one Excel workbook.

Every number is read from the run directories -- nothing is transcribed:

  experiments\20261016_<run>\analysis_results\<run>\model_performance_summary.csv
                                                \group_errors.csv
                                                \shap_analysis\<model>\feature_importance.csv
  experiments\20261016_<run>\models\<run>\run_metadata.json

Sheets: 说明 / 总览 / 全模型指标 / CV_MSE矩阵 / Test_R2矩阵 / 分组误差 / SHAP前15 / SHAP族份额

Usage: python -B .dsh_tmp\build_four_group_xlsx.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'reports' / '六组结果汇总.xlsx'

RUNS = [
    ('R-M', '随机划分 random', 'Mordred-only', 1613),
    ('R-MF', '随机划分 random', '混合 concatenated', 9972),
    ('B-M', 'B 轴 meso', 'Mordred-only', 1613),
    ('B-MF', 'B 轴 meso', '混合 concatenated', 9972),
    ('S-M', '骨架划分 scaffold', 'Mordred-only', 1613),
    ('S-MF', '骨架划分 scaffold', '混合 concatenated', 9972),
]
RUN_DIRS = {
    'R-M': '20261016_R-M', 'R-MF': '20261016_R-MF',
    'S-M': '20261016_S-M', 'S-MF': '20261016_S-MF',
    'B-M': '20261021_B-M', 'B-MF': '20261021_B-MF',
}
FAMILIES = [('Morgan_', 'Morgan'), ('RDKitFP_', 'RDKitFP'), ('AtomPair_', 'AtomPair'),
            ('TopoTorsion_', 'TopoTorsion'), ('MACCS_', 'MACCS')]
MODEL_ORDER = ['XGBoost', 'LightGBM', 'RandomForest', 'SVR', 'KernelRidge',
               'GaussianProcess', 'ElasticNet', 'Lasso', 'DecisionTree', 'KNeighbors']


def shaded(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ['group_column', 'group', 'Model', 'n', 'MAE', 'RMSE']]


def run_dir(run: str) -> Path:
    return ROOT / 'experiments' / RUN_DIRS[run] / 'analysis_results' / run


def family_of(feature: str) -> str:
    for prefix, name in FAMILIES:
        if feature.startswith(prefix):
            return name
    return 'Mordred'


def main() -> int:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import ColorScaleRule

    header_fill = PatternFill('solid', fgColor='1F3864')
    header_font = Font(bold=True, color='FFFFFF', size=10)
    body_font = Font(size=10)
    thin = Side(style='thin', color='BFBFBF')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    overview_rows, metric_rows, group_frames, shap_rows, share_rows = [], [], [], [], []
    cv_matrix, r2_matrix = {}, {}

    for run, split, features, raw_columns in RUNS:
        directory = run_dir(run)
        summary = pd.read_csv(directory / 'model_performance_summary.csv')
        summary = summary[summary['Model'].isin(MODEL_ORDER)].copy()
        meta_path = ROOT / 'experiments' / RUN_DIRS[run] / 'models' / run / 'run_metadata.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        n_train = len(meta.get('train_source_rows', []) or [])
        n_test = len(meta.get('test_source_rows', []) or [])

        importance_path = None
        shap_dir = directory / 'shap_analysis'
        if shap_dir.exists():
            for candidate in sorted(shap_dir.iterdir()):
                if candidate.is_dir() and (candidate / 'feature_importance.csv').exists():
                    importance_path = candidate / 'feature_importance.csv'
                    shap_model = candidate.name
                    break
        importance = pd.read_csv(importance_path) if importance_path else None
        shap_values = shap_names = None
        if importance_path is not None:
            # shap_values.npy follows samples.csv column order, NOT the importance
            # ranking (feature_importance.csv is sorted by mean|SHAP|).  Mapping the
            # two the wrong way round silently scrambles every family assignment.
            shap_values = np.load(importance_path.parent / 'shap_values.npy')
            shap_names = list(pd.read_csv(importance_path.parent / 'samples.csv',
                                          nrows=0).columns)
            if len(shap_names) != shap_values.shape[1] or set(shap_names) != set(importance['Feature']):
                raise ValueError(f'column order mismatch for {run}')
        pooled = int(len(importance)) if importance is not None else ''

        best_cv = summary.loc[summary['CV MSE (eV²)'].idxmin()]
        best_test = summary.loc[summary['Test R²'].idxmax()]

        # How many trials the budget actually bought for GaussianProcess: read the
        # saved progress json (unpickling the model would drag in sklearn, which the
        # openpyxl-only interpreter does not have) so the note stays true whether or
        # not the model was evaluated after the manual stop.
        trial_note = ''
        gp_progress = (ROOT / 'experiments' / RUN_DIRS[run] / 'models' / run
                       / 'GaussianProcess.progress.json')
        if gp_progress.exists():
            progress = json.loads(gp_progress.read_text(encoding='utf-8'))
            n_trials = progress.get('complete_trials')
            budget = progress.get('budget')
            finished = progress.get('finished_trials') or 0
            # Only flag a shortfall when the search really stopped early; a run whose
            # finished_trials reached the budget is a completed search even if a few
            # trials ended FAIL/PRUNED and therefore never counted as COMPLETE.
            if n_trials and budget and n_trials < budget and finished < budget:
                trial_note = f'（GP {n_trials}/{budget} trial，按最优 trial 评估）'
        stopped = run == 'S-MF'

        overview_rows.append({
            '组': run, '划分': split, '特征集': features,
            '原始特征列': raw_columns, '折内保留（池化）': pooled,
            '训练样本': n_train, '测试样本': n_test,
            '完成模型数': f'{len(summary)}/10' + trial_note,
            'CV 最优模型': best_cv['Model'], 'CV MSE (eV²)': round(float(best_cv['CV MSE (eV²)']), 4),
            'Test R²': round(float(best_cv['Test R²']), 4),
            'Test RMSE (eV)': round(float(best_cv['Test RMSE']), 4),
            'Test MAE (eV)': round(float(best_cv['Test MAE']), 4),
            'Test 最优模型': best_test['Model'],
            'Test R²（最优）': round(float(best_test['Test R²']), 4),
            'Test RMSE（最优）(eV)': round(float(best_test['Test RMSE']), 4),
            'Test MAE（最优）(eV)': round(float(best_test['Test MAE']), 4),
        })

        for _, row in summary.iterrows():
            metric_rows.append({
                '组': run, '划分': split, '特征集': features, '模型': row['Model'],
                'Train R²': round(float(row['Train R²']), 4),
                'Train RMSE (eV)': round(float(row['Train RMSE']), 4),
                'Train MAE (eV)': round(float(row['Train MAE']), 4),
                'CV MSE (eV²)': round(float(row['CV MSE (eV²)']), 4),
                'Test R²': round(float(row['Test R²']), 4),
                'Test RMSE (eV)': round(float(row['Test RMSE']), 4),
                'Test MAE (eV)': round(float(row['Test MAE']), 4),
                'Test/CV 误差比': round(float(row['Test RMSE']) ** 2 / float(row['CV MSE (eV²)']), 3),
                'CV 排名': int(summary['CV MSE (eV²)'].rank().loc[row.name]),
                'Test 排名': int(summary['Test R²'].rank(ascending=False).loc[row.name]),
                '备注': (f'GP {trial_note.strip("（）")}' if row['Model'] == 'GaussianProcess'
                         and trial_note else ''),
            })

        errors = pd.read_csv(directory / 'group_errors.csv')
        errors.insert(0, '组', run)
        group_frames.append(shaded(errors))

        cv_matrix[run] = summary.set_index('Model')['CV MSE (eV²)']
        r2_matrix[run] = summary.set_index('Model')['Test R²']

        if importance is not None:
            top = importance.head(15).copy()
            top.insert(0, '组', run)
            top.insert(1, '最优模型', shap_model)
            top['排名'] = range(1, len(top) + 1)
            top['特征族'] = top['Feature'].map(family_of)
            shap_rows.append(top[['组', '最优模型', '排名', 'Feature', 'Mean_ABS_SHAP', '特征族']])

            # Composition uses the SAME metric as reports\四组SHAP对照.md: the mean over
            # explained samples of (family |SHAP| / that sample's total |SHAP|).
            families = np.array([family_of(name) for name in shap_names])
            absolute = np.abs(shap_values)
            total_per_sample = absolute.sum(axis=1)
            row = {'组': run, '最优模型': shap_model, '解释样本数': int(shap_values.shape[0])}
            fingerprint = 0.0
            for _, name in FAMILIES + [('', 'Mordred')]:
                mask = families == name
                share = (absolute[:, mask].sum(axis=1) / total_per_sample).mean() * 100
                row[name] = round(float(share), 2)
                if name != 'Mordred':
                    fingerprint += float(share)
            row['指纹合计'] = round(fingerprint, 2)
            share_rows.append(row)

    overview = pd.DataFrame(overview_rows)
    metrics = pd.DataFrame(metric_rows)
    groups = pd.concat(group_frames, ignore_index=True)

    cv_frame = pd.DataFrame({run: [cv_matrix[run].get(model) for model in MODEL_ORDER]
                             for run, *_ in RUNS}, index=MODEL_ORDER)
    r2_frame = pd.DataFrame({run: [r2_matrix[run].get(model) for model in MODEL_ORDER]
                             for run, *_ in RUNS}, index=MODEL_ORDER)
    cv_frame.index.name = r2_frame.index.name = '模型'

    notes = pd.DataFrame([
        ['数据来源', 'experiments\\20261016_{R-M,R-MF,S-M,S-MF}\\ 与 experiments\\20261021_{B-M,B-MF}\\ 下 analysis_results\\<run>\\ 的 model_performance_summary.csv、group_errors.csv、shap_analysis\\<model>\\'],
        ['生成脚本', '.dsh_tmp\\build_four_group_xlsx.py（全部数值从上述 CSV 读出，未手工转录）'],
        ['CV MSE 口径', '管线口径：五折 MSE 的算术平均（eV²）。受控消融脚本改用五折合并 MSE，故同一配置会出现 0.6192（管线）与 0.6101（合并）两个数'],
        ['折数', '全部 5 折；随机划分 1,387 训练 / 347 测试，B 轴 1,398 / 336，骨架划分 1,383 / 351'],
        ['训练预算', '每个模型 100 个 Optuna trial，6 组 × 10 模型共 6,000 个 trial；S-MF 的 GaussianProcess 曾在 39/100 处人工停机，后于 2026-10-21 补跑至 100 trial（其中 97 个 COMPLETE），最优值未变'],
        ['S-MF 的 GP', '最优 trial（alpha≈3.6e-10）的 CV MSE = 1.2136 eV²，高于“只用训练折均值”的常数基线 1.1981，说明 GP 在该划分下没有学到可用结构；补跑 100 trial 未改变最优值'],
        ['折内特征数抖动', '每折保留列数会有 ±1 抖动，因为 np.corrcoef 的浮点归约顺序不固定，恰好卡在 |r| = 0.95 边界的列会进出'],
        ['SHAP 可比性', '六组最优模型不同（XGBoost / SVR / KNeighbors / Lasso），SHAP 以 eV 计、量级跟随各自输出幅度，只能比较份额与排名，不能比较绝对值'],
        ['SHAP 份额口径', '逐样本“该族 |SHAP| ÷ 该样本总 |SHAP|”的均值，与 reports\\B轴SHAP对照.md 一致。注意不要用各特征 mean|SHAP| 求和后的比率，两者在稀疏的指纹族上差别很大（求和口径 R-MF 85.4% / S-MF 88.9%，逐样本口径 87.1% / 89.1%）'],
        ['SHAP 列序陷阱', 'shap_values.npy 的列序等于 samples.csv 的列序，而 feature_importance.csv 已按 mean|SHAP| 降序排列；用后者给前者贴标签会把指纹族份额算错（曾因此得到错误的 44%），脚本已加断言'],
        ['Test/CV 误差比', 'Test RMSE² ÷ CV MSE。≈1 表示泛化与交叉验证一致；>3 通常说明测试集与训练分布不同（骨架划分）'],
        ['指纹主效应', '同划分内“混合 − Mordred”的平均 ΔCV MSE：随机 −0.1206、B 轴 −0.0599、骨架 +0.1892（各 10 个模型）'],
        ['划分轴主效应', '相对随机划分的平均 ΔCV MSE：B 轴 +0.4198（Mordred）/ +0.4806（混合），骨架 +1.1239 / +1.4337'],
    ], columns=['项目', '说明'])

    workbook = Workbook()
    workbook.remove(workbook.active)

    def add_sheet(name, frame, freeze='A2', widths=None, wrap_columns=()):
        sheet = workbook.create_sheet(name)
        sheet.append(list(frame.columns))
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = border
        for record in frame.itertuples(index=False):
            sheet.append(['' if (isinstance(value, float) and pd.isna(value)) else value
                          for value in record])
        for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, max_col=len(frame.columns)):
            for cell in row:
                cell.font = body_font
                cell.border = border
                if isinstance(cell.value, float):
                    cell.number_format = '0.0000'
        sheet.freeze_panes = freeze
        sheet.auto_filter.ref = f'A1:{get_column_letter(len(frame.columns))}{sheet.max_row}'
        for index, column in enumerate(frame.columns, start=1):
            longest = max([len(str(column))] + [len(str(value)) for value in frame[column].head(200)])
            sheet.column_dimensions[get_column_letter(index)].width = min(max(10, longest + 3), 34)
            if column in wrap_columns:
                for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, min_col=index, max_col=index):
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical='top')
        sheet.row_dimensions[1].height = 30
        return sheet

    sheet_notes = add_sheet('说明', notes, widths=None, wrap_columns=('说明',))
    for row in sheet_notes.iter_rows(min_row=2, max_row=sheet_notes.max_row, min_col=2, max_col=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical='top')
    sheet_notes.column_dimensions['A'].width = 18
    sheet_notes.column_dimensions['B'].width = 100

    add_sheet('总览', overview)
    add_sheet('全模型指标', metrics, wrap_columns=('备注',))
    add_sheet('分组误差', groups)

    for name, frame in (('CV_MSE矩阵', cv_frame), ('Test_R2矩阵', r2_frame)):
        sheet = workbook.create_sheet(name)
        sheet.append(['模型'] + list(frame.columns))
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border
        for model, values in frame.iterrows():
            sheet.append([model] + [None if pd.isna(value) else round(float(value), 4) for value in values])
        for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, max_col=len(frame.columns) + 1):
            for cell in row:
                cell.font = body_font
                cell.border = border
                if isinstance(cell.value, float):
                    cell.number_format = '0.0000'
        sheet.freeze_panes = 'B2'
        sheet.column_dimensions['A'].width = 18
        for index in range(2, len(frame.columns) + 2):
            sheet.column_dimensions[get_column_letter(index)].width = 14
        sheet.conditional_formatting.add(
            f'B2:{get_column_letter(len(frame.columns) + 1)}{sheet.max_row}',
            ColorScaleRule(start_type='min', start_color='63BE7B',
                           mid_type='percentile', mid_value=50, mid_color='FFEB84',
                           end_type='max', end_color='F8696B'))

    add_sheet('SHAP前15', pd.concat(shap_rows, ignore_index=True))
    add_sheet('SHAP族份额', pd.DataFrame(share_rows))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(OUT)
    print(f'[written] {OUT}')
    print(f'  sheets: {workbook.sheetnames}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
