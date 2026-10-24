"""Independent structural + numerical integrity check of the shipped Mordred feature cache.

The cache CSV holds ONLY the 1613 raw Mordred descriptor columns (index=False);
sample identity lives in the sibling .meta.json. So this script verifies:

  A. cache columns == meta["columns"] (exact names and order, 1613 of them)
  B. cache row count == len(meta["samples"]) == 1734, and meta SHA-256 values
     still match the shipped files
  C. row identity: meta["descriptor_smiles"][i] re-parses and its heavy-atom
     count agrees with that row's Mordred descriptors (index-binding test)
  D. independently recomputable descriptors (RDKit) match the cached values
     (guards against the historical index-misalignment defect class)
  E. NaN / inf / constant-column profile (matters for linear vs tree models)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RDLogger.DisableLog('rdApp.*')

raw_path = Path('data/Data-20261012_raw_features.csv')
src_path = Path('data/Data-20261012.csv')
meta_path = Path(str(raw_path) + '.meta.json')


def sha256(p, block=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


meta = json.loads(meta_path.read_text(encoding='utf-8'))
df = pd.read_csv(raw_path)

print(f'[A] cache shape={df.shape}  meta columns={len(meta["columns"])}  '
      f'exact order match={list(df.columns) == meta["columns"]}')

print(f'[B] meta samples={len(meta["samples"])}  rows match={len(meta["samples"]) == len(df)}  '
      f'data_sha256 ok={sha256(src_path) == meta["data_sha256"]}  '
      f'cache_sha256 ok={sha256(raw_path) == meta["cache_sha256"]}')
print(f'    versions={meta["versions"]}  use_fingerprints={meta["use_fingerprints"]}')

rows = meta['samples']
smi = meta['descriptor_smiles']
mols = [Chem.MolFromSmiles(s) for s in smi]
unparsed = int(sum(m is None for m in mols))
heavy = np.array([m.GetNumHeavyAtoms() if m else -1 for m in mols], dtype=float)
print(f'[C] descriptor_smiles reparse: {len(smi) - unparsed}/{len(smi)} ok, {unparsed} failed')

labels = np.array([r['Gap(eV)'] for r in rows], dtype=float)
print(f'    labels: n={len(labels)} mean={labels.mean():.3f} min={labels.min():.3f} '
      f'max={labels.max():.3f}  any non-positive={bool((labels <= 0).any())}')
src_rows = np.array([r['source_row'] for r in rows], dtype=int)
print(f'    source_row: strictly increasing={bool((np.diff(src_rows) > 0).all())} '
      f'range=[{src_rows.min()}, {src_rows.max()}]  unique={len(set(src_rows.tolist()))}')
for group, want in (('C_number', None), ('Y_number', None)):
    vals, counts = np.unique([r[group] for r in rows], return_counts=True)
    print(f'    {group} distribution: {dict(zip(vals.tolist(), counts.tolist()))}')

if 'nHeavyAtom' in df.columns:
    d = np.abs(df['nHeavyAtom'].to_numpy(dtype=float) - heavy)
    print(f'[C2] nHeavyAtom (cache) vs RDKit heavy atoms: max|diff|={np.nanmax(d):.0f} '
          f'rows differing={int((d > 0).sum())}')

checkers = {
    'MW': lambda m: Descriptors.MolWt(m),
    'LogP': lambda m: Crippen.MolLogP(m),
    'nHBAcc': lambda m: rdMolDescriptors.CalcNumHBA(m),
    'nHBDon': lambda m: rdMolDescriptors.CalcNumHBD(m),
    'nRing': lambda m: rdMolDescriptors.CalcNumRings(m),
    'nRot': lambda m: Lipinski.NumRotatableBonds(m),
    'nAromRings': lambda m: rdMolDescriptors.CalcNumAromaticRings(m),    'TopoPSA': lambda m: rdMolDescriptors.CalcTPSA(m),
    'nHetero': lambda m: rdMolDescriptors.CalcNumHeteroatoms(m),
}
idx = list(range(0, len(df), 87))
print(f'[D] recompute comparison on {len(idx)} spread rows:')
for col, fn in checkers.items():
    if col not in df.columns:
        print(f'    {col:<10} column absent')
        continue
    ref = np.array([fn(mols[i]) if mols[i] else np.nan for i in idx], dtype=float)
    got = df[col].to_numpy(dtype=float)[idx]
    ok = np.isfinite(ref) & np.isfinite(got)
    if ok.sum() == 0:
        print(f'    {col:<10} no finite comparison')
        continue
    rel = np.abs(got[ok] - ref[ok]) / np.maximum(np.abs(ref[ok]), 1e-9)
    verdict = 'MATCH' if rel.max() < 1e-6 else 'DIFF'
    print(f'    {col:<10} n={int(ok.sum()):>3} max rel.diff={rel.max():.3e} {verdict}')

num = df.apply(pd.to_numeric, errors='raise')
nan_cols = num.isna().any()
inf_mask = np.isinf(num.to_numpy(dtype=float))
const = [c for c in num.columns if num[c].nunique(dropna=False) <= 1]
print(f'[E] NaN: {int(num.isna().sum().sum())} cells in {int(nan_cols.sum())} columns '
      f'{(nan_cols[nan_cols].index.tolist()[:6])}')
print(f'    inf: {int(inf_mask.sum())} cells')
print(f'    constant columns: {len(const)} {const[:8]}')
print(f'    value range: min={np.nanmin(num.to_numpy(dtype=float)):.4g} '
      f'max={np.nanmax(num.to_numpy(dtype=float)):.4g}')
