r"""Apply the reviewed numeric revision to the manuscript, producing rev2.

The original file is never touched; every edit happens on a copy.  Edits are run
level (only the runs that carry the matched text are rewritten), so superscripts,
italics and equation objects elsewhere in the same paragraph survive untouched.

Round 1-2: values that no longer match the four re-run experiments.
Round 3:   typo pass (leftover editorial markers, one impossible orbital energy,
           one split-size mix-up, one capitalisation).

Usage: python -B .dsh_tmp\revise_manuscript.py
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.shared import RGBColor
from docx.text.paragraph import Paragraph
from docx.text.run import Run

RED = RGBColor(0xFF, 0x00, 0x00)

# Spans the author asked to see in bold red for manual checking.
# P0056: "a relatively low mean HOMO-LUMO gap of 5.864 eV" -- this mean cannot be
# reproduced from the current database under any (X, C, B) slicing of A03.
FLAGGED_SPANS = [(56, '5.864 eV')]


SRC = Path(r'C:\Users\admin\Downloads\PorScaDB-Manuscript-20261011.docx')
DST = Path(r'C:\Users\admin\Downloads\PorScaDB-Manuscript-20261011-rev2.docx')
ROOT = Path(__file__).resolve().parents[1]
FIG7 = ROOT / 'reports' / 'figures' / 'fig7_xgboost_true_vs_predicted.png'
FIG8 = ROOT / 'reports' / 'figures' / 'fig8_shap_beeswarm.png'

# (paragraph index, old, new, rationale)
PARAGRAPH_EDITS = [
    (7,
     'achieve a prediction RMSE < 0.26 eV',
     'achieve a test-set RMSE of 0.19 eV (R\u00b2 = 0.96)',
     'abstract: best test RMSE is 0.1923 eV (XGBoost, concatenated descriptors), not <0.26'),
    (21,
     'These descriptors were then subjected to feature selection before being fed into the ML models as input features.',
     'The non\u20113D descriptor matrix comprised 1,613 features after spin and metadata columns were removed. '
     'These descriptors were then subjected to feature selection, which was refitted within every training '
     'fold, retaining 510\u2013526 features per fold (523 distinct features over the pooled folds) before being '
     'fed into the ML models as input features.',
     'Methods: 411 features -> 1,613 raw, 510-526 retained per fold'),
    (23,
     '|Pearson r| > 0.95).',
     '|Pearson r| > 0.95), both fitted within the training folds only. The concatenated matrix contained '
     '9,972 raw features (1,613 mordred descriptors; Morgan, RDKitFP, AtomPair and TopoTorsion at 2,048 bits '
     'each; MACCS with 167 keys) and retained 3,815\u20134,055 features per fold (3,986 distinct features over the '
     'pooled folds).',
     'Methods: 3,825 features -> 9,972 raw, 3,815-4,055 retained per fold; in-fold fitting stated'),
    (24,
     '100 Optuna trials on an',
     '100 Optuna trials of 5-fold cross-validation on an',
     'Methods: 5-fold CV inside the training set, not a flat 80/20'),
    (24,
     'with a standard scaler applied to the features.',
     'with all preprocessing (median imputation, variance filtering, correlation pruning, and standard '
     'scaling) fitted on the training folds only; the final models were refitted on the 1,387-molecule '
     'training set and evaluated once on the 347-molecule test set.',
     'Methods: preprocessing is in-fold; Table 1 uses the RANDOM split (1,387/347), not the scaffold split'),
    (27,
     'shap package (v0.52.0)',
     'shap package (v0.53.0)',
     'Methods: installed shap is 0.53.0'),
    (27,
     'on a random subset of 100 training samples.',
     'on a random subset of 100 training samples for the tree-based and linear models, whereas the '
     'kernel-based models were explained on 30 samples using 512 kernel-SHAP evaluations over a '
     '25-cluster background.',
     'Methods: KernelExplainer used 30 samples / 512 evaluations'),
    (48,
     'Across the 1,743 valid entries',
     'Across the 1,734 valid entries',
     'the gap statistics were computed on the 1,734 converged structures that enter the ML set'),
    (61,
     'a mordred descriptor set (411 features)',
     'a mordred descriptor set (1,613 raw features, of which 510\u2013526 are retained per fold)',
     'Results: feature counts'),
    (61,
     'a concatenated descriptors set (3,825 features)',
     'a concatenated descriptors set (9,972 raw features, of which 3,815\u20134,055 are retained per fold)',
     'Results: feature counts'),
    (61,
     '100 trials of 3-fold cross-validation',
     '100 trials of 5-fold cross-validation',
     'Results: 5-fold, matching the pipeline'),
    (62,
     'followed by LightGBM and Gaussian process.',
     'followed by SVR and Kernel Ridge Regression.',
     'new ranking: SVR 0.9565 and KRR 0.9443 take ranks 2-3; GP drops to rank 6'),
    (62,
     '0.9359', '0.9632', 'Table 1 value'),
    (62,
     '0.2544', '0.1923', 'Table 1 value'),
    (62,
     '0.1243', '0.1178', 'Table 1 value'),
    (62,
     'ranks fourth', 'ranks seventh', 'mordred-only XGBoost is rank 7 in the re-run'),
    (62,
     'an R2 of 0.9095', 'an R2 of 0.9321', 're-run value'),
    (62,
     'only 0.0264 lower', 'only 0.0311 lower', '0.9632 - 0.9321 = 0.0311'),
    (62,
     'The concatenated descriptors model achieves slightly higher accuracy, with the improvement mainly '
     'concentrated in the densely populated low-gap region.',
     'The concatenated descriptors model achieves higher accuracy across the whole range, and the gain is '
     'largest for high-gap molecules (test MSE 0.330 vs 0.194 eV\u00b2 above 8.5 eV; 0.172 vs 0.057 eV\u00b2 between '
     '7.5 and 8.5 eV) rather than in the densely populated low-gap region.',
     'measured per-bin MSE: improvement grows with gap, contrary to the original claim'),
    (62,
     'This small performance gap', 'This modest performance gap', 'wording, 0.0311 is not negligible'),
    (63,
     'by test-set evaluation indicators the two feature sets.',
     'by test-set evaluation indicators for the two feature sets.',
     'caption grammar'),
    (75,
     'R\u00b2 = 0.9359, RMSE = 0.2544 eV, MAE = 0.1243 eV',
     'R\u00b2 = 0.9632, RMSE = 0.1923 eV, MAE = 0.1178 eV',
     'conclusion: current best-model metrics'),
    (75,
     'provided competitive performance with improved chemical interpretability',
     'provided competitive performance (R\u00b2 = 0.9321) with improved chemical interpretability',
     'conclusion: add the mordred-only value'),
    (68,
     'SHAP beeswarm plot for the top 10 descriptors.',
     'SHAP beeswarm plot for the top 12 descriptors.',
     'the regenerated figure shows the top 12'),
    # ---- round 2: entry counts, SHAP concentration, descriptor order ----
    (7,
     'comprising 1,743 porphyrin derivatives, all computed using a consistent quantum-chemical (QC) method.',
     'comprising 1,734 porphyrin derivatives, all computed using a consistent '
     'quantum-chemical (QC) method.',
     'the dataset is the 1,734 converged structures; 1,860 is the raw table including 126 failed jobs'),
    (17,
     'comprising 1,743 structurally diverse porphyrin derivatives',
     'comprising 1,734 structurally diverse porphyrin derivatives',
     'same count correction as the abstract'),
    (40,
     'The molecular design produced 1,743 database entries. All molecular geometries were '
     'optimized, and thermodynamic properties were computed',
     'The molecular design produced 1,734 database entries. All molecular geometries were '
     'optimized, and thermodynamic properties were computed',
     'the 126 non-converged jobs are not part of the dataset'),
    (74,
     'comprising 1,743 macrocycle-modified porphyrin derivatives',
     'comprising 1,734 macrocycle-modified porphyrin derivatives',
     'dataset size, consistent with P0040'),
    (69,
     'approximately 49% of the total mean absolute SHAP value, and the top ten for approximately 56%',
     'approximately 45% of the total mean absolute SHAP value, and the top ten for approximately 51%',
     'recomputed from shap_values.npy: top-5 45.38%, top-10 50.94%'),
    (70,
     'The second-ranked descriptor, NsssCH, denotes, as defined by Mordred, the number of atoms '
     'belonging to the sssCH atom type, that is, carbon atoms bearing one hydrogen and three '
     'single bonds. Its importance was driven almost entirely by a small number of uncommon '
     'structures with NsssCH = 4, in which the corresponding SHAP contributions were strongly '
     'positive, whereas the majority of samples (NsssCH = 0) contributed weakly and negatively. '
     'Its high global ranking therefore reflects substantial effects within a limited structural '
     'subgroup (A04) rather than a general trend across the full chemical space.',
     'The second-ranked descriptor, C3SP3, counts the sp3-hybridized carbon atoms. Its importance '
     'was likewise driven by a limited structural subgroup rather than by a general trend: 96 of '
     'the 100 explained samples carry C3SP3 = 0 and contribute weakly and negatively, whereas the '
     'three samples with C3SP3 = 4 contribute strongly positively (up to +3.38 eV).',
     'NsssCH has dropped out of the top 12; C3SP3 (mean |SHAP| 0.195) is now second'),
    (71,
     'MATS1p ranked fourth; it is the Moran coefficient at topological lag 1, weighted by atomic '
     'polarizability. Its dependence was inverse and asymmetric, with positive contributions '
     'confined to the samples with the lowest descriptor values and near-zero, slightly negative '
     'contributions elsewhere. NdsCH completed the five highest-ranked descriptors. It counts '
     'atoms of the Mordred dsCH atom type, that is, carbons bearing one hydrogen, one double bond '
     'and one single bond; samples with NdsCH = 0 contributed negatively, whereas nonzero values '
     'were uniformly positive and its influence saturated as the count increased, again indicating '
     'a threshold-like rather than a linear dependence.',
     'NdsCH ranked fourth. It counts atoms of the Mordred dsCH atom type, that is, carbons bearing '
     'one hydrogen, one double bond and one single bond; samples with NdsCH = 0 contributed '
     'negatively, whereas nonzero values were uniformly positive and its influence saturated as '
     'the count increased, again indicating a threshold-like rather than a linear dependence. '
     'MATS1p completed the five highest-ranked descriptors; it is the Moran coefficient at '
     'topological lag 1, weighted by atomic polarizability. Its dependence was inverse and '
     'asymmetric, with positive contributions confined to the samples with the lowest descriptor '
     'values and near-zero, slightly negative contributions elsewhere.',
     're-run means NdsCH (0.0751) outranks MATS1p (0.0626)'),
    (75,
     'to further improve model transferability and computational\u2013experimental integration.',
     'to further improve model transferability and computational\u2013experimental integration. '
     'A scaffold-based split, in which the test molecules share no macrocyclic scaffold with the '
     'training set, reduced the best test R\u00b2 to 0.37, indicating that the present models '
     'interpolate within known scaffolds rather than extrapolate to unseen ones.',
     'one sentence in the conclusion, per the request for a brief mention only'),
    # ---- round 3: typo pass ----
    # NOTE: the three Chinese editorial markers in P0019/P0041/P0042 are kept on
    # purpose (author's own working notes, to be resolved when the dataset section
    # is finalised).  Lower/upper case of "mordred" is likewise left exactly as
    # the author wrote it.
    (57,
     'is strongly split in bacteriochlorin (10.07/1.95 eV)',
     'is strongly split in bacteriochlorin (-0.07/1.95 eV)',
     'A03N LUMO is -0.0697 eV and LUMO+1 is 1.9535 eV; 10.07 eV exists nowhere in the database'),
    (8,
     'KEYWORDS\uff1a',
     'KEYWORDS: ',
     'fullwidth CJK colon in the keywords line'),
    (66,
     'Figure 7  Predicted',
     'Figure 7 Predicted',
     'double space in the caption'),
    (113,
     'Geerlings, P.; . De Proft, F.',
     'Geerlings, P.; De Proft, F.',
     'stray punctuation in reference 32'),
    (113,
     'Phys, Chem, Chem, Phys, 2014, 16 (28)',
     'Phys. Chem. Chem. Phys., 2014, 16 (28)',
     'journal abbreviation typo in reference 32'),
]

# Applied to every body paragraph after the explicit edits above (all occurrences).
# Intentionally empty: the manuscript's own mix of "Mordred"/"mordred" is kept.
GLOBAL_REPLACEMENTS = []

# A short paragraph inserted after P0072 (the closing paragraph of the ML results),
# so the scaffold result is stated once, briefly, without a dedicated section.
SCAFFOLD_PARAGRAPH_AFTER = 72
SCAFFOLD_PARAGRAPH = (
    'To determine whether this accuracy extends beyond the scaffolds represented in the training '
    'data, the entire protocol was repeated with a scaffold-based split, in which none of the '
    'macrocyclic scaffolds of the test set occurs in the training set. The best test-set '
    'R\u00b2 across the ten models decreased from 0.9632 (XGBoost, concatenated descriptors) '
    'to 0.3741 (random forest, mordred descriptors), and the effect of the fingerprint block '
    'reversed: '
    'the mean cross-validation MSE over the nine converged models increased by 0.21 eV\u00b2 '
    'when the fingerprint block was added, whereas it decreased by 0.12 eV\u00b2 under the random '
    'split. The models therefore interpolate accurately within known scaffolds but do not '
    'extrapolate to unseen macrocyclic architectures, which defines the main scope limitation of '
    'the present machine-learning analysis.'
)

TABLE_ROWS = [
    ('Concatenated Descriptors', 'XGBoost', '0.9632', '0.1923', '0.1178'),
    ('Concatenated Descriptors', 'SVR', '0.9565', '0.2090', '0.1313'),
    ('Concatenated Descriptors', 'KernelRidge', '0.9443', '0.2364', '0.1583'),
    ('Concatenated Descriptors', 'Lasso', '0.9404', '0.2447', '0.1639'),
    ('Concatenated Descriptors', 'ElasticNet', '0.9399', '0.2457', '0.1668'),
    ('Concatenated Descriptors', 'GaussianProcess', '0.9362', '0.2532', '0.1630'),
    ('Mordred-Only', 'XGBoost', '0.9321', '0.2611', '0.1662'),
    ('Concatenated Descriptors', 'LightGBM', '0.9207', '0.2822', '0.1469'),
    ('Mordred-Only', 'LightGBM', '0.8917', '0.3297', '0.1991'),
    ('Concatenated Descriptors', 'RandomForest', '0.8886', '0.3344', '0.1394'),
]


def replace_in_paragraph(paragraph, old: str, new: str, allow_many: bool = False) -> int:
    """Replace `old` with `new`, touching only the runs that carry the match."""
    runs = paragraph.runs
    text = ''.join(run.text for run in runs)
    count = text.count(old)
    if count == 0:
        return 0
    if count > 1 and not allow_many:
        raise ValueError(f'ambiguous match ({count}x) for {old!r}')
    start = text.index(old)
    end = start + len(old)

    spans = []
    cursor = 0
    for run in runs:
        spans.append((cursor, cursor + len(run.text), run))
        cursor += len(run.text)

    first_index = None
    for position, (span_start, span_end, _run) in enumerate(spans):
        if span_start <= start < span_end:
            first_index = position
            break
    if first_index is None:                      # match starts at a run boundary
        for position, (span_start, _span_end, _run) in enumerate(spans):
            if span_start == start:
                first_index = position
                break
    if first_index is None:
        raise ValueError(f'could not map match start for {old!r}')

    last_index = first_index
    for position in range(first_index, len(spans)):
        if spans[position][1] >= end:
            last_index = position
            break

    span_start, _span_end, first_run = spans[first_index]
    offset_in_first = start - span_start
    last_span_start, last_span_end, last_run = spans[last_index]
    tail = last_run.text[end - last_span_start:]

    first_run.text = first_run.text[:offset_in_first] + new
    if last_index == first_index:
        first_run.text = first_run.text + tail
    else:
        for position in range(first_index + 1, last_index):
            spans[position][2].text = ''
        last_run.text = tail
    return count


def flag_span(paragraph, needle: str) -> int:
    """Make `needle` bold + red, splitting runs so nothing else is recoloured."""
    runs = paragraph.runs
    text = ''.join(run.text for run in runs)
    if needle not in text:
        return 0
    start = text.index(needle)
    end = start + len(needle)

    spans = []
    cursor = 0
    for run in runs:
        spans.append((cursor, cursor + len(run.text), run))
        cursor += len(run.text)

    first = last = None
    for position, (span_start, span_end, _run) in enumerate(spans):
        if first is None and span_start <= start < span_end:
            first = position
        if span_start < end <= span_end:
            last = position
            break

    span_start, _e, first_run = spans[first]
    last_span_start, _e2, last_run = spans[last]
    head = first_run.text[:start - span_start]
    tail = last_run.text[end - last_span_start:]

    def clone_after(anchor, chunk):
        element = deepcopy(anchor._r)
        anchor._r.addnext(element)
        fresh = Run(element, anchor._parent)
        fresh.text = chunk
        fresh.bold = True
        fresh.font.color.rgb = RED
        return fresh

    def clone_plain(anchor, chunk):
        element = deepcopy(anchor._r)
        anchor._r.addnext(element)
        fresh = Run(element, anchor._parent)
        fresh.text = chunk
        fresh.bold = None
        fresh.font.color.rgb = None
        return fresh

    first_run.text = head
    anchor = clone_after(first_run, needle)
    if last is not None and last > first:
        for position in range(first + 1, last):
            spans[position][2].text = ''
        last_run.text = ''
    if tail:
        clone_plain(anchor, tail)
    return 1


def set_cell_text(cell, value: str) -> None:
    paragraph = cell.paragraphs[0]
    if paragraph.runs:
        paragraph.runs[0].text = value
        for run in paragraph.runs[1:]:
            run.text = ''
    else:
        paragraph.add_run(value)


def swap_media(path: Path, replacements: dict[str, Path]) -> None:
    """Rewrite image parts in place so relationships and extents stay valid."""
    buffer = path.with_suffix('.tmp')
    with zipfile.ZipFile(path, 'r') as source, zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            name = Path(item.filename).name
            if name in replacements:
                data = replacements[name].read_bytes()
                print(f'  media swapped: {item.filename} <- {replacements[name].name} ({len(data)/1024:.0f} KB)')
                item = zipfile.ZipInfo(item.filename, date_time=item.date_time)
                item.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(item, data)
    buffer.replace(path)


def main() -> int:
    if not SRC.exists():
        print(f'missing source: {SRC}')
        return 1
    # Work on a scratch copy: a failed edit must never leave a half-written
    # deliverable behind (an earlier failure overwrote the output with a bare copy).
    work = DST.with_suffix('.working.docx')
    shutil.copy2(SRC, work)
    document = Document(str(work))
    paragraphs = document.paragraphs

    applied = []
    for index, old, new, rationale in PARAGRAPH_EDITS:
        paragraph = paragraphs[index]
        hits = replace_in_paragraph(paragraph, old, new)
        if not hits:
            print(f'!! P{index:04d} not found: {old[:70]!r}')
            return 2
        applied.append((index, old, new, rationale))
        print(f'  P{index:04d} ok  {old[:58]!r} -> {new[:58]!r}')

    for old, new in GLOBAL_REPLACEMENTS:
        touched = 0
        for paragraph in paragraphs:
            while old in paragraph.text:
                replace_in_paragraph(paragraph, old, new, allow_many=True)
                touched += 1
        print(f'  global: {old!r} -> {new!r} ({touched} occurrences)')

    table = document.tables[0]
    while len(table.rows) - 1 < len(TABLE_ROWS):
        table._tbl.append(deepcopy(table.rows[-1]._tr))
    for row_index, values in enumerate(TABLE_ROWS, start=1):
        row = table.rows[row_index]
        for column_index, value in enumerate(values):
            set_cell_text(row.cells[column_index], value)
    print(f'  table rows written: {len(TABLE_ROWS)}')

    anchor = paragraphs[SCAFFOLD_PARAGRAPH_AFTER]
    holder = deepcopy(anchor._p)
    anchor._p.addnext(holder)
    inserted = Paragraph(holder, anchor._parent)
    for run in list(inserted.runs):
        run._r.getparent().remove(run._r)
    inserted.add_run(SCAFFOLD_PARAGRAPH)
    print(f'  inserted scaffold paragraph after P{SCAFFOLD_PARAGRAPH_AFTER:04d}')

    for index, needle in FLAGGED_SPANS:
        hits = flag_span(paragraphs[index], needle)
        if not hits:
            print(f'!! P{index:04d} flag not found: {needle!r}')
            return 4
        print(f'  P{index:04d} flagged bold-red: {needle!r}')

    document.save(str(work))
    swap_media(work, {'image7.png': FIG7, 'image8.png': FIG8})
    work.replace(DST)
    print(f'[saved] {DST}')

    # verify
    check = Document(str(DST))
    text = '\n'.join(p.text for p in check.paragraphs)
    leftovers = ['0.9359', '0.2544', '0.1243', '0.9095', '0.0264', '411 features',
                 '3,825', '3-fold', '1,743 valid entries', 'v0.52.0', '1,743',
                 '1,860', 'NsssCH', 'approximately 49%', 'approximately 56%',
                 'ranks fourth', '1,383-molecule', '351-molecule', '10.07',
                 'KEYWORDS\uff1a', 'Phys, Chem, Chem, Phys',
                 'Geerlings, P.; .']
    stale = [token for token in leftovers if token in text]
    print(f'stale tokens still present: {stale if stale else "none"}')
    required = ['1,734 database entries', '1,734 porphyrin derivatives',
                '1,734 structurally diverse', '1,734 macrocycle-modified',
                'C3SP3', 'NdsCH ranked fourth',
                'scaffold-based split', '0.3741', 'test-set RMSE of 0.19 eV',
                '1,387-molecule training set', '347-molecule test set',
                'bacteriochlorin (-0.07/1.95 eV)']
    missing = [token for token in required if token not in text]
    print(f'required tokens missing: {missing if missing else "none"}')
    print(f'paragraphs: {len(check.paragraphs)} (was 150 + 1 inserted)')
    return 0 if not stale and not missing else 3


if __name__ == '__main__':
    sys.exit(main())
