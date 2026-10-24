
# -*- coding: utf-8 -*-
"""Render Table S5 (feature extraction and selection) as a standalone docx.
Counts are read from .dsh_tmp/s5_counts.json, which verify_s5_final.py wrote
after asserting every number against the cache, the split file and the artefacts."""
import json, os
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = r"E:\Project-ZhangChengwei\PorScaDB-ML-compute-20261014\PorScaDB-ML"
OUT = os.path.join(BASE, "reports", "SI-Table-S5-feature-selection.docx")
C = json.load(open(os.path.join(BASE, ".dsh_tmp", "s5_counts.json"), encoding="utf-8"))
fam = C["fam"]; sm = C["stages_mordred"]; sh = C["stages_hybrid"]
MORD_ONLY = C["mordred_only"]; MIXED = C["mixed"]

def pct(a, b):
    return "%.1f%%" % (100.0 * a / b)

# ---------- docx helpers ----------
def set_font(run, size, bold=False, italic=False, sup=False):
    run.font.size = Pt(size); run.font.name = "Times New Roman"
    run.bold = bold; run.italic = italic
    if sup: run.font.superscript = True
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(k), "Times New Roman")

def add_rich(par, text, size=9, bold=False):
    parts = text.split("R2")
    for i, chunk in enumerate(parts):
        if chunk:
            set_font(par.add_run(chunk), size, bold=bold)
        if i < len(parts) - 1:
            set_font(par.add_run("R"), size, bold=bold, italic=True)
            set_font(par.add_run("2"), size, bold=bold, italic=True, sup=True)

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)

def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true"); trPr.append(el)

def cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:cantSplit"); el.set(qn("w:val"), "true"); trPr.append(el)

def fix_layout(t, widths_cm):
    tblPr = t._tbl.tblPr
    for tag in ("w:tblLayout", "w:tblW", "w:tblCellMar"):
        for el in tblPr.findall(qn(tag)):
            tblPr.remove(el)
    layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed"); tblPr.append(layout)
    tw = OxmlElement("w:tblW"); tw.set(qn("w:type"), "dxa")
    tw.set(qn("w:w"), str(int(sum(widths_cm) * 567))); tblPr.append(tw)
    mar = OxmlElement("w:tblCellMar")
    for side, w in (("left", 40), ("right", 40), ("top", 14), ("bottom", 14)):
        el = OxmlElement("w:" + side); el.set(qn("w:w"), str(w)); el.set(qn("w:type"), "dxa"); mar.append(el)
    tblPr.append(mar)
    grid = t._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths_cm):
            gc.set(qn("w:w"), str(int(w * 567)))

def caption(doc, label, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(11); p.paragraph_format.space_after = Pt(4)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_font(p.add_run(label + "  "), 9.5, bold=True)
    add_rich(p, text, size=9.5)

def note(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(13)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_rich(p, text, size=8.5)

def body(doc, text, size=10, after=8, justify=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    if justify: p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_rich(p, text, size=size)
    return p

def th(t): return "{:,}".format(t)

# ---------- document ----------
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"; st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
sec = doc.sections[0]
sec.page_width = Cm(21.0); sec.page_height = Cm(29.7)
sec.left_margin = Cm(2.1); sec.right_margin = Cm(2.1)
sec.top_margin = Cm(2.2); sec.bottom_margin = Cm(2.0)

set_font(doc.add_paragraph().add_run("Supporting Information" + " \u2014 Table S5"), 14, bold=True)

body(doc, "Table S5 documents how the design matrix of the random split is assembled and reduced before model development. "
     "Two descriptor sets are compared. In the first, the design matrix consists of the 2-D Mordred descriptors only. "
     "In the second, the same " + th(sm["raw"]) + " Mordred columns are concatenated with five RDKit fingerprint families "
     "(" + th(sh["raw"] - sm["raw"]) + " bits), giving " + th(sh["raw"]) + " columns in total. The Mordred block is placed first and the "
     "fingerprint blocks follow in the order Morgan, MACCS, atom pair, topological torsion and RDKit path, so the column order "
     "is deterministic and the two selection results can be compared column by column.", size=10, after=6)

body(doc, "One model-independent procedure is applied to both sets. Infinite values are replaced by missing values, "
     "columns that are missing for every molecule are dropped, and the remaining gaps are imputed with the training-set median. "
     "Constant columns are then removed and a variance threshold of 0.01 is applied. Finally the surviving columns are ranked by "
     "the absolute Pearson correlation with the target, and a column is discarded when its |r| with an already retained column "
     "exceeds 0.95. The procedure is fitted inside every cross-validation fold and refitted once on the pooled training partition "
     "(1,387 of the 1,734 molecules of the random split) for the final models; the 347 test molecules are only transformed. "
     "Because a single matrix is shared, all ten regressors are trained on exactly the same columns and no model-specific "
     "(embedded or tree-based) selection is applied, including for XGBoost.", size=10, after=6)

# ---- Table S5 (a) ----
caption(doc, "Table S5 (a).",
        "Composition of the two descriptor sets and the number of columns retained by the selection procedure "
        "(random split, 1,387 training molecules).")
COLS_A = ["Feature block", "RDKit implementation and hyperparameters", "Computed",
          "Retained\n(hybrid)", "Retention\n(hybrid)", "Retained\n(Mordred\nonly)"]
W_A = [3.20, 6.05, 1.70, 1.70, 1.65, 2.10]
ROWS_A = [
    ("Morgan (circular)", "AllChem.GetMorganFingerprintAsBitVect\nradius = 2, nBits = 2048",
     2048, fam["Morgan"], None, False),
    ("MACCS keys", "MACCSkeys.GenMACCSKeys\n167 bits, bit 0 always zero",
     167, fam["MACCS"], None, False),
    ("Atom pair", "rdFingerprintGenerator.GetAtomPairGenerator\nfpSize = 2048, 1\u201330 bonds (defaults)",
     2048, fam["AtomPair"], None, False),
    ("Topological torsion", "GetTopologicalTorsionGenerator\nfpSize = 2048, 4-atom torsions (defaults)",
     2048, fam["Torsion"], None, False),
    ("RDKit path", "rdmolops.RDKFingerprint\nminPath = 1, maxPath = 7, fpSize = 2048",
     2048, fam["Path"], None, False),
    ("Fingerprints, subtotal", "", sh["raw"] - sm["raw"],
     sum(fam[k] for k in ("Morgan", "MACCS", "AtomPair", "Torsion", "Path")), None, True),
    ("Mordred, 2-D", "mordred.Calculator(ignore_3D = True)\n" + th(sm["raw"]) + " 2-D descriptors",
     sm["raw"], fam["Mordred"], MORD_ONLY, True),
    ("Total design matrix", "Mordred block followed by the five fingerprint blocks",
     sh["raw"], MIXED, MORD_ONLY, True),
]
FP_SUB = sh["raw"] - sm["raw"]
assert [r[2] for r in ROWS_A] == [2048, 167, 2048, 2048, 2048, 8359, 1613, 9972], [r[2] for r in ROWS_A]
assert [r[3] for r in ROWS_A] == [245, 44, 1041, 106, 2039, 3475, 511, 3986], [r[3] for r in ROWS_A]
t = doc.add_table(rows=1, cols=len(COLS_A))
t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
for i, name in enumerate(COLS_A):
    cell = t.rows[0].cells[i]; cell.width = Cm(W_A[i]); cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    shade(cell, "EDEDED")
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(1); p.paragraph_format.space_after = Pt(1)
    for j, line in enumerate(name.split("\n")):
        if j:
            p.add_run().add_break()
        set_font(p.add_run(line), 8.5, bold=True)
repeat_header(t.rows[0]); cant_split(t.rows[0])
for label, impl, comp, ret, mret, bold in ROWS_A:
    cells = t.add_row().cells
    cant_split(t.rows[-1])
    vals = [label, impl, th(comp), th(ret), pct(ret, comp), ("\u2014" if mret is None else th(mret))]
    for i, v in enumerate(vals):
        cell = cells[i]; cell.width = Cm(W_A[i]); cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i in (0, 1) else WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0.5); p.paragraph_format.space_after = Pt(0.5)
        for j, line in enumerate(str(v).split("\n")):
            if j:
                p = cell.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i in (0, 1) else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0.5)
            set_font(p.add_run(line), 8.5, bold=bold)
fix_layout(t, W_A)

note(doc, "Note: The counts refer to the random split and to the pooled training partition. \u201cComputed\u201d is the nominal "
     "dimensionality of each block and is the denominator of the retention rate: 118 of the " + th(sm["raw"]) + " Mordred "
     "descriptors are missing for every molecule of this dataset and are removed before ranking, so " + th(sm["var"]) +
     " Mordred columns actually enter the variance and collinearity filters. MACCS provides 167 bits of which bit 0 is always "
     "zero, leaving 166 informative positions. Fingerprints supply " + th(sum(fam[k] for k in ("Morgan", "MACCS", "AtomPair", "Torsion", "Path"))) +
     " of the " + th(MIXED) + " retained columns; the RDKit path block alone supplies " + th(fam["Path"]) + " (51.2% of the "
     "retained matrix), because sparse path bits rarely exceed the variance threshold and are seldom collinear with a retained "
     "column, so a high retention rate does not by itself indicate predictive value. Without fingerprints the identical "
     "procedure retains " + th(MORD_ONLY) + " Mordred columns (" + pct(MORD_ONLY, sm["raw"]) + " of " + th(sm["raw"]) + ").")

# ---- Table S5 (b) ----
caption(doc, "Table S5 (b).", "Stage-wise attrition of the design matrix under the same procedure.")
COLS_B = ["Selection stage", "Mordred only", "Mordred + fingerprints"]
W_B = [8.40, 4.00, 4.00]
ROWS_B = [
    ("Columns in the raw feature matrix", sm["raw"], sh["raw"], False),
    ("After removing columns that are missing for every molecule", sm["nonnan"], sh["nonnan"], False),
    ("After removing constant columns", sm["nonconst"], sh["nonconst"], False),
    ("After the variance threshold", sm["var"], sh["var"], False),
    ("After collinearity pruning (|r| > 0.95)", sm["final"], sh["final"], True),
]
t2 = doc.add_table(rows=1, cols=len(COLS_B))
t2.style = "Table Grid"; t2.alignment = WD_TABLE_ALIGNMENT.CENTER; t2.autofit = False
for i, name in enumerate(COLS_B):
    cell = t2.rows[0].cells[i]; cell.width = Cm(W_B[i]); cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    shade(cell, "EDEDED")
    p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(1); p.paragraph_format.space_after = Pt(1)
    set_font(p.add_run(name), 8.5, bold=True)
repeat_header(t2.rows[0]); cant_split(t2.rows[0])
for label, a, b, bold in ROWS_B:
    cells = t2.add_row().cells
    cant_split(t2.rows[-1])
    for i, v in enumerate([label, th(a), th(b)]):
        cell = cells[i]; cell.width = Cm(W_B[i]); cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0.5); p.paragraph_format.space_after = Pt(0.5)
        set_font(p.add_run(v), 8.5, bold=bold)
fix_layout(t2, W_B)

note(doc, "Note: Constant columns dominate the attrition of the fingerprint blocks: of the " +
     th(sh["nonnan"] - sh["final"]) + " columns removed from the concatenated matrix, " + th(sh["nonnan"] - sh["nonconst"]) +
     " are constant across the training set, " + th(sh["nonconst"] - sh["var"]) + " fall below the variance threshold and " +
     th(sh["var"] - sh["final"]) + " are removed as collinear with an already retained column. In the Mordred-only matrix the "
     "collinearity filter removes more columns (" + th(sm["var"] - sm["final"]) + ") than the variance threshold (" +
     th(sm["nonconst"] - sm["var"]) + ") and the constant-column filter (" + th(sm["nonnan"] - sm["nonconst"]) + ").")

doc.save(OUT)
print("saved:", OUT, os.path.getsize(OUT), "bytes")
print("tables:", [(len(x.rows), len(x.columns)) for x in doc.tables])
print("rows(a):", len(ROWS_A), "rows(b):", len(ROWS_B))
