import csv, re
from pathlib import Path
root = Path(r"E:\Project-ZhangChengwei\PorScaDB-ML-compute-20261014\PorScaDB-ML")
paths = {
    "mixed(fp-full)": root / ".dsh_tmp/fp-full/Data-20261012_raw_features_fp.csv",
    "mordred(data/)": root / "data/Data-20261012_raw_features.csv",
}
for label, p in paths.items():
    if not p.exists():
        print(f"{label}: MISSING {p}")
        continue
    with open(p, encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    fams = {}
    for c in header:
        m = re.match(r"^(Morgan|MACCS|AtomPair|TopoTorsion|RDKitFP)_", c)
        fams[m.group(1) if m else "Mordred/other"] = fams.get(m.group(1) if m else "Mordred/other", 0) + 1
    print(f"{label}: total={len(header)}  {fams}")
