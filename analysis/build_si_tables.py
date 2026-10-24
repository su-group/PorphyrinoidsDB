
# -*- coding: utf-8 -*-
import os, sys, pandas as pd
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = r"E:\Project-ZhangChengwei\PorScaDB-ML-compute-20261014\PorScaDB-ML"
OUT  = os.path.join(BASE, "reports", "SI-Table-S6-S7-S8-three-axis.docx")

RUNS = [("S6a","20261016_R-M","R-M"), ("S6b","20261016_R-MF","R-MF"),
        ("S7a","20261016_S-M","S-M"), ("S7b","20261016_S-MF","S-MF"),
        ("S8a","20261021_B-M","B-M"), ("S8b","20261021_B-MF","B-MF")]

def load(exp, tag):
    p = os.path.join(BASE, "experiments", exp, "analysis_results", tag, "model_performance_summary.csv")
    return pd.read_csv(p, encoding="utf-8-sig")

DATA = {k: load(e,t) for k,e,t in RUNS}

# ---------- docx helpers ----------
def set_font(run, size, bold=False, italic=False, sup=False):
    run.font.size = Pt(size); run.font.name = "Times New Roman"
    run.bold = bold; run.italic = italic
    if sup: run.font.superscript = True
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts'); rpr.append(rf)
    rf.set(qn('w:ascii'), "Times New Roman"); rf.set(qn('w:hAnsi'), "Times New Roman")
    rf.set(qn('w:eastAsia'), "Times New Roman")

def add_rich(par, text, size=9, bold=False):
    """Render text, turning every 'R2' into italic R + italic superscript 2."""
    parts = text.split("R2")
    for i, chunk in enumerate(parts):
        if chunk:
            set_font(par.add_run(chunk), size, bold=bold)
        if i < len(parts) - 1:
            set_font(par.add_run("R"), size, bold=bold, italic=True)
            set_font(par.add_run("2"), size, bold=bold, italic=True, sup=True)

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hexcolor)
    tcPr.append(shd)

def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement('w:tblHeader'); el.set(qn('w:val'), "true"); trPr.append(el)

def cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement('w:cantSplit'); el.set(qn('w:val'), "true"); trPr.append(el)


def fix_layout(t, widths_cm):
    tblPr = t._tbl.tblPr
    for tag in ("w:tblLayout", "w:tblW"):
        for el in tblPr.findall(qn(tag)):
            tblPr.remove(el)
    layout = OxmlElement('w:tblLayout'); layout.set(qn('w:type'), 'fixed'); tblPr.append(layout)
    tw = OxmlElement('w:tblW'); tw.set(qn('w:type'), 'dxa')
    tw.set(qn('w:w'), str(int(sum(widths_cm) * 567))); tblPr.append(tw)
    mar = OxmlElement('w:tblCellMar')
    for side in ("left", "right"):
        el = OxmlElement('w:' + side); el.set(qn('w:w'), '28'); el.set(qn('w:type'), 'dxa'); mar.append(el)
    for side in ("top", "bottom"):
        el = OxmlElement('w:' + side); el.set(qn('w:w'), '14'); el.set(qn('w:type'), 'dxa'); mar.append(el)
    tblPr.append(mar)
    grid = t._tbl.find(qn('w:tblGrid'))
    if grid is not None:
        for gc, w in zip(grid.findall(qn('w:gridCol')), widths_cm):
            gc.set(qn('w:w'), str(int(w * 567)))

COLS = ["Model", "CV MSE\n(eV\u00b2)", "Train R2", "Train RMSE\n(eV)", "Train MAE\n(eV)",
        "Test R2", "Test RMSE\n(eV)", "Test MAE\n(eV)"]
WIDTHS = [2.70, 2.00, 1.82, 2.05, 2.00, 1.82, 2.05, 2.00]

def build_table(doc, df, width_list):
    d = df.sort_values("Test R\u00b2", ascending=False)
    t = doc.add_table(rows=1, cols=len(COLS))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    hdr = t.rows[0]
    repeat_header(hdr); cant_split(hdr)
    for i, name in enumerate(COLS):
        cell = hdr.cells[i]
        cell.width = Cm(width_list[i])
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shade(cell, "EDEDED")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1); p.paragraph_format.space_after = Pt(1)
        add_rich(p, name, size=8.5, bold=True)
    for _, r in d.iterrows():
        cells = t.add_row().cells
        cant_split(t.rows[-1])
        vals = [r["Model"],
                "%.4f" % r["CV MSE (eV\u00b2)"], "%.4f" % r["Train R\u00b2"], "%.4f" % r["Train RMSE"],
                "%.4f" % r["Train MAE"], "%.4f" % r["Test R\u00b2"], "%.4f" % r["Test RMSE"], "%.4f" % r["Test MAE"]]
        for i, v in enumerate(vals):
            cell = cells[i]
            cell.width = Cm(width_list[i])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0.5); p.paragraph_format.space_after = Pt(0.5)
            add_rich(p, v, size=8.5)
    fix_layout(t, width_list)
    return t

def caption(doc, label, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(4)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_font(p.add_run(label + "  "), 9.5, bold=True)
    add_rich(p, text, size=9.5)

def note(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(12)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_rich(p, text, size=8.5)

# ---------- document ----------
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"; st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn('w:eastAsia'), "Times New Roman")

sec = doc.sections[0]
sec.page_width = Cm(21.0); sec.page_height = Cm(29.7)
sec.left_margin = Cm(2.1); sec.right_margin = Cm(2.1)
sec.top_margin = Cm(2.2); sec.bottom_margin = Cm(2.0)

t = doc.add_paragraph()
t.paragraph_format.space_after = Pt(6)
set_font(t.add_run("Supporting Information \u2014 Tables S6, S7 and S8"), 14, bold=True)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(10)
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
add_rich(p, "Three-axis evaluation of the ten regressors: the random split (Table S6), the scaffold split (A axis, Table S7) and the meso-substituent split (B axis, Table S8). Each table is split into part (a), Mordred descriptors only, and part (b), the concatenated descriptor set, so that the cross-validation error can be reported alongside the training and test metrics without exceeding the page width. Part (a) and part (b) of the same table share one split definition and one set of experimental conditions.", size=10)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(6)
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
add_rich(p, "All ten regressors were tuned with Optuna (100 trials, seed 42) using five-fold cross-validation inside the training partition and then refitted on that partition. CV MSE is the mean squared error of that inner cross-validation and is the quantity on which the final model is selected; it is computed on the training partition only and therefore never sees a test molecule. The Test columns are computed on the held-out partition. RMSE and MAE are in eV, CV MSE in eV\u00b2, and all values are given to four decimal places. Within each table the rows are ordered by Test R2, so that the three tables can be read against one another. Train RMSE and MAE of 0.0000 denote values below 5 \u00d7 10\u207b\u2075 eV; they occur for KNeighbors and GaussianProcess, which reproduce the training set to numerical precision.", size=10)

# ---- Table S6 ----
caption(doc, "Table S6 (a).", "Predictive performance under the random split, Mordred descriptors only. Before model development the 1,734 molecules were partitioned at random into 1,387 training and 347 test molecules.")
build_table(doc, DATA["S6a"], WIDTHS)
note(doc, "Note: The random split ignores group membership, so the same macrocyclic scaffold and the same meso substituent may occur on both sides of the split; this table therefore measures interpolation, not extrapolation. The inner cross-validation is a random five-fold split of the training partition.")

caption(doc, "Table S6 (b).", "Continued: Mordred descriptors concatenated with four fingerprint families (9,972 columns before feature selection).")
build_table(doc, DATA["S6b"], WIDTHS)
note(doc, "Note: The remarks to part (a) apply unchanged; only the descriptor set differs. Adding fingerprints lowers the mean CV MSE of the ten regressors from 0.2409 to 0.1202 eV\u00b2 and improves ten of the ten models.")

# ---- Table S7 ----
caption(doc, "Table S7 (a).", "Predictive performance under the scaffold split (A axis), Mordred descriptors only. Before model development the dataset was divided into 1,383 training and 351 test molecules by macrocyclic scaffold, so that no scaffold occurs on both sides.")
build_table(doc, DATA["S7a"], WIDTHS)
note(doc, "Note: Four scaffolds are withheld in the test partition (A02, A03, A14, A18), and the inner cross-validation is grouped by scaffold index, so no scaffold is split across folds. The model chosen by cross-validation is not the model that performs best on the held-out scaffolds: Lasso has the lowest CV MSE (0.6192 eV\u00b2) but reaches a Test R2 of only +0.0909, whereas RandomForest reaches +0.3741 with a CV MSE of 2.5315 eV\u00b2. Part (b) does not show the same inversion: with fingerprints SVR is both the cross-validation and the test optimum, so the selection failure is specific to the Mordred descriptor set rather than to the split.")

caption(doc, "Table S7 (b).", "Continued: the concatenated descriptor set.")
build_table(doc, DATA["S7b"], WIDTHS)
note(doc, "Note: The split and the cross-validation grouping are those of part (a). DecisionTree reports entries identical to those of Table S7a, including the CV MSE. This is not a transcription error, and the two models are not the same file: both Optuna optima are two-leaf trees (max_depth = 1) whose single split lands on a descriptor present in both feature sets, and the two fitted models return identical predictions on all 351 test molecules (maximum absolute difference 0.0 over the stored prediction files). The same coincidence recurs in Table S8, where the held-out predictions again coincide exactly while the CV MSE differs slightly, because the cross-validation models are refitted inside each fold; it is a property of a depth-one tree, not duplicated data.")

# ---- Table S8 ----
caption(doc, "Table S8 (a).", "Predictive performance under the meso-substituent split (B axis), Mordred descriptors only. Before model development the dataset was divided into 1,398 training and 336 test molecules by meso-substituent group, so that three entire substituent groups were withheld.")
build_table(doc, DATA["S8a"], WIDTHS)
note(doc, "Note: The held-out groups are B05, B11 and B15, and the inner cross-validation is grouped by substituent index. B00, the unsubstituted parent, is retained on the training side in every fold, so the split does not double as a scaffold extrapolation: A13 and A14 occur only in combination with B00 and would otherwise be withheld twice.")

caption(doc, "Table S8 (b).", "Continued: the concatenated descriptor set.")
build_table(doc, DATA["S8b"], WIDTHS)
note(doc, "Note: Two entries require a remark. First, the CV MSE of RandomForest here (1.1031 eV\u00b2) is higher than in Table S8a (0.7400 eV\u00b2) although its test performance is better (+0.9631 against +0.8667); the two numbers are not inconsistent. Cross-validation is computed on the training partition, which retains B00, whereas the held-out groups contain no molecule without a meso substituent. When B00 is withheld as a whole group, RandomForest errs by +2.78 eV with an MSE of 9.7130 eV\u00b2, 8.9 times the constant baseline (Section S-x.3), so the inner folds that contain B00 inflate the pooled cross-validation error of the tree models. CV MSE therefore ranks the tree ensembles pessimistically on this axis and prefers KNeighbors (0.1767 eV\u00b2) over RandomForest, which is nonetheless the better model on the held-out groups. Second, DecisionTree reports the same test metrics as Table S8a; the reason is given in the note to Table S7b.")

doc.save(OUT)
print("saved:", OUT, os.path.getsize(OUT), "bytes")
print("tables:", len(doc.tables), "rows:", [len(tb.rows) for tb in doc.tables])
