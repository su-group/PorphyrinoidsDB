"""Build the terminology-unification checklist from the current manuscript.

Reads the revised manuscript (rev2), finds every occurrence of the terms that
compete with ``macrocyclic scaffold`` / ``meso-substituent``, and emits a
table of paragraph-indexed edits.  Nothing is modified in the manuscript.

Run with the interpreter that has python-docx:

    C:\\Users\\admin\\.dsh\\dsh-runtimes\\dsh-primary-runtime\\dependencies\\python\\python.exe ^
        -B .dsh_tmp\\build_terminology_checklist.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SOURCE = Path(r"C:\Users\admin\Downloads\PorScaDB-Manuscript-20261011-rev2.docx")

# (label, compiled pattern, replacement template, verdict)
RULES = [
    ("macrocyclic topology", re.compile(r"macrocyclic topolog(y|ies)\b", re.I),
     {"y": "macrocyclic scaffold", "ies": "macrocyclic scaffolds"}, "auto"),
    ("macrocyclic core topology", re.compile(r"macrocyclic core topolog(y|ies)\b", re.I),
     {"y": "macrocyclic scaffold", "ies": "macrocyclic scaffolds"}, "review"),
    ("molecular backbone", re.compile(r"molecular backbones?\b", re.I),
     "macrocyclic scaffolds", "auto"),
    ("side groups", re.compile(r"side groups?\b", re.I),
     "meso-substituents", "skip-if-legacy"),
    ("substituent patterns", re.compile(r"(?<!meso-)substituents? patterns?\b", re.I),
     "meso-substituent patterns", "auto"),
    ("macrocycle-modified", re.compile(r"macrocycle-modified\b", re.I),
     "—", "keep"),
    ("bare scaffold", re.compile(r"(?<!macrocyclic )scaffolds?\b", re.I),
     "—", "keep-shortform"),
    ("macrocyclic scaffold", re.compile(r"macrocyclic scaffolds?\b", re.I),
     "—", "canonical"),
]

ACTIONS = {
    "macrocyclic topology": "改为 `macrocyclic scaffold(s)`",
    "macrocyclic core topology": "改为 `macrocyclic scaffold`（句首要补 The）",
    "molecular backbone": "改为 `macrocyclic scaffolds`",
    "side groups": "仅当指本研究 B 轴时改；他人数据集保留",
    "substituent patterns": "改为 `meso-substituent patterns`",
    "macrocycle-modified": "保留（关键词 / 论文自称）",
    "bare scaffold": "保留（首次定义后可用简称）",
    "macrocyclic scaffold": "保持不动（规范写法）",
}

LEGACY_MARKERS = ("8 porphyrin scaffolds", "9 side groups", "Ørnsø", "Ornso",
                  "legacy", "previous dataset", "earlier dataset")


def sentence_around(text: str, start: int, end: int, width: int = 90) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    snippet = text[left:right].strip()
    return ("…" if left > 0 else "") + snippet + ("…" if right < len(text) else "")


def main() -> int:
    document = Document(SOURCE)
    blocks: list[tuple[str, str, str]] = []      # (id, style, text)
    for index, paragraph in enumerate(document.paragraphs):
        blocks.append((f"P{index:04d}", paragraph.style.name or "Normal", paragraph.text))
    for t_index, table in enumerate(document.tables):
        for r_index, row in enumerate(table.rows):
            for c_index, cell in enumerate(row.cells):
                blocks.append((f"T{t_index}r{r_index}c{c_index}", "Table", cell.text))

    rows: list[dict] = []
    for block_id, style, text in blocks:
        if not text.strip():
            continue
        lowered = text.lower()
        legacy = any(marker.lower() in lowered for marker in LEGACY_MARKERS)
        for label, pattern, replacement, verdict in RULES:
            for match in pattern.finditer(text):
                old = match.group(0)
                if isinstance(replacement, dict):
                    key = "ies" if old.lower().endswith("ies") else "y"
                    new = replacement[key]
                else:
                    new = replacement
                if label == "macrocyclic topology" and match.start() == 0 and old[0].isupper():
                    new = "The " + new
                if verdict == "skip-if-legacy" and legacy:
                    effective = "skip"
                else:
                    effective = verdict
                rows.append({
                    "id": block_id,
                    "style": style,
                    "label": label,
                    "old": old,
                    "new": new,
                    "verdict": effective,
                    "context": sentence_around(text, match.start(), match.end()),
                })

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1

    out: list[str] = []
    add = out.append
    add("# 术语统一清单（macrocyclic scaffold / meso-substituent）")
    add("")
    add(f"扫描对象：`{SOURCE.name}`（151 段 + 1 表）；扫描时间由脚本生成，"
        "所有段落号均为该文件中段落的自然序号（P0000 起）。")
    add("")
    add("## 1. 术语定义（英文，可直接放进正文首次出现处）")
    add("")
    add("> **macrocyclic scaffold** — the porphyrinoid ring system (the conjugated "
        "macrocycle together with its peripheral ring modifications and the central "
        "metalation state), indexed A01–A18 in this work.")
    add(">")
    add("> **meso-substituent** — the substituent at the four meso positions, "
        "grouped as B00–B15 in 16 substitution patterns.")
    add("")
    add("首次出现后，正文可用简称 `scaffold` / `substituent`，但不得再引入 "
        "`macrocyclic topology`、`molecular backbone`、`side group` 等同义写法。")
    add("")
    add("## 2. 出现次数汇总")
    add("")
    add("| 写法 | 次数 | 处理 |")
    add("|---|---|---|")
    labels = [(label, verdict) for label, _, _, verdict in RULES]
    for label, verdict in labels:
        add(f"| `{label}` | {counts.get(label, 0)} | {ACTIONS[label]} |")
    add("")

    add("## 3. 必改清单（auto）")
    add("")
    auto_rows = [row for row in rows if row["verdict"] == "auto"]
    if auto_rows:
        add("| 段落 | 样式 | 原文 | 改为 | 上下文 |")
        add("|---|---|---|---|---|")
        for row in auto_rows:
            context = row["context"].replace("|", "\\|")
            add(f"| {row['id']} | {row['style']} | `{row['old']}` | `{row['new']}` | {context} |")
    else:
        add("（无）")
    add("")

    add("## 4. 需人工判断（review）")
    add("")
    review_rows = [row for row in rows if row["verdict"] == "review"]
    if review_rows:
        add("| 段落 | 样式 | 原文 | 建议 | 上下文 |")
        add("|---|---|---|---|---|")
        for row in review_rows:
            context = row["context"].replace("|", "\\|")
            add(f"| {row['id']} | {row['style']} | `{row['old']}` | `{row['new']}` | {context} |")
    else:
        add("（无）")
    add("")

    add("## 5. 明确保留 / 跳过")
    add("")
    skip_rows = [row for row in rows if row["verdict"] == "skip"]
    keep_rows = [row for row in rows if row["verdict"] == "keep"]
    add("### 5.1 他人先前数据集中的 `side group`（**不得**改成 meso-substituent）")
    add("")
    if skip_rows:
        add("| 段落 | 原文 | 上下文 |")
        add("|---|---|---|")
        for row in skip_rows:
            context = row["context"].replace("|", "\\|")
            add(f"| {row['id']} | `{row['old']}` | {context} |")
    else:
        add("（未检到）")
    add("")
    add("### 5.2 保留的写法")
    add("")
    if keep_rows:
        add("| 段落 | 写法 | 上下文 |")
        add("|---|---|---|")
        for row in keep_rows:
            context = row["context"].replace("|", "\\|")
            add(f"| {row['id']} | `{row['old']}` | {context} |")
    else:
        add("（无）")
    add("")

    add("## 6. 建议在正文插入的一句（指向 SI）")
    add("")
    add("> The metrics reported in the main text are obtained under a random "
        "five-fold split and therefore describe interpolation within the covered "
        "macrocyclic scaffolds; the generalization behaviour across macrocyclic "
        "scaffolds and across meso-substituents is quantified in Supplementary "
        "Note S-x.")
    add("")
    add("## 7. 图片注意事项")
    add("")
    add("Figure 5、Figure 6 的内嵌图片轴标签仍写着 `Molecular backbone Index`，"
        "图片本体按你的要求不改动；但正文/图注引用该轴时应统一称 "
        "`macrocyclic scaffold index`，避免正文与图内标签出现两套术语。")
    add("")

    add("## 8. 编号体系冲突（R00–R15 vs B00–B15）")
    add("")
    id_hits = []
    for block_id, style, text in blocks:
        tags = sorted(set(re.findall(r"\b[RB]\d{2}\b", text)))
        if tags:
            id_hits.append((block_id, tags, sentence_around(text, 0, 0, width=110)))
    if id_hits:
        add("数据集里 meso 取代基的编号是 **B00–B15**；正文至少有一处写成了 "
            "`R00–R15`，与图 2 的 R 标签一致但与数据文件不一致。建议全文统一为 "
            "`B00–B15`，或在图注/SI 中显式声明 `R` 为图内标签、`B` 为数据字段值。")
        add("")
        add("| 段落 | 该段出现的编号 | 上下文 |")
        add("|---|---|---|")
        for block_id, tags, context in id_hits:
            context = context.replace("|", "\\|")
            add(f"| {block_id} | {', '.join(f'`{tag}`' for tag in tags)} | {context} |")
    else:
        add("（未检到 R/B 编号）")
    add("")

    target = REPORTS / "术语统一清单.md"
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[written] {target}  ({len(out)} lines)")
    for label, verdict in labels:
        print(f"  {label:28s} {counts.get(label, 0):3d}  {verdict}")
    print(f"  auto={len(auto_rows)} review={len(review_rows)} skip={len(skip_rows)} keep={len(keep_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
