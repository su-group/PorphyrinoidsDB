"""数据过滤和特征缓存校验。导入模块不会写文件或修改控制台。"""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


def standardize_smiles(smiles):
    """无效结构返回 None，由调用者统一过滤，不能退出整个程序。"""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None
    except (ValueError, RuntimeError):
        return None


def load_valid_data(path, smoke=False):
    data = pd.read_csv(path)
    required = ['SMILES', 'Gap(eV)', 'Y_number', 'C_number', 'Status']
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f'数据缺少必需列: {missing}')
    data['source_row'] = np.arange(len(data))  # 源文件行号用于样本追踪。
    data['C_number'] = pd.to_numeric(data['C_number'], errors='coerce')
    data['Gap(eV)'] = pd.to_numeric(data['Gap(eV)'], errors='coerce')
    data = data[data['C_number'].isin([0, 1, 2]) &
                data['Status'].eq('Structure Optimization Finished') &
                np.isfinite(data['Gap(eV)']) & data['Gap(eV)'].gt(0)].copy()
    data['source_smiles'] = data['SMILES']  # 跨 RDKit 版本稳定的身份键。
    data['SMILES'] = data['SMILES'].apply(standardize_smiles)
    invalid = int(data['SMILES'].isna().sum())
    if invalid:
        warnings.warn(f'已跳过 {invalid} 个无效 SMILES', stacklevel=2)
    data = data.dropna(subset=['SMILES']).reset_index(drop=True)
    if data.empty or data['Y_number'].isna().any():
        raise ValueError('有效样本为空或存在缺失的 Y_number')
    if smoke:
        # 两个环境按同一顺序选取相同的少量真实样本。
        data = data.sample(n=min(40, len(data)), random_state=42).reset_index(drop=True)
    return data


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sample_records(data):
    columns = ['source_row', 'source_smiles', 'Gap(eV)', 'Y_number', 'C_number']
    # 同时绑定标签与自旋，避免更新数据后误用旧缓存。
    return json.loads(data[columns].to_json(orient='records', double_precision=15))


def save_feature_cache(features, data, data_path, cache_path, use_fingerprints=False):
    """只由实际计算特征的程序生成证明文件，不给未知旧缓存补签名。"""
    import platform
    import rdkit
    import mordred

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if len(features) != len(data):
        raise ValueError('特征与样本行数不一致')
    features.to_csv(cache_path, index=False)
    metadata = {
        'schema_version': 1, 'raw_descriptors': True,
        'data_sha256': file_sha256(data_path),
        'cache_sha256': file_sha256(cache_path),
        'samples': sample_records(data), 'columns': list(features.columns),
        # 不同 RDKit 版本对配位原子的规范写法可能不同，单独记录生成时的写法。
        'descriptor_smiles': data['SMILES'].tolist(),
        'use_fingerprints': use_fingerprints,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'rdkit': rdkit.__version__,
                     'mordred': mordred.__version__},
    }
    Path(str(cache_path) + '.meta.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


def load_feature_cache(cache_path, data, data_path):
    cache_path = Path(cache_path)
    meta_path = Path(str(cache_path) + '.meta.json')
    if not cache_path.exists() or not meta_path.exists():
        raise ValueError('缺少经过校验的原始特征缓存。请先运行 prepare_features.py；'
                         '旧缓存不能仅补元数据后复用。冒烟缓存需要 --smoke。')
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    checks = [metadata.get('schema_version') == 1,
              metadata.get('raw_descriptors') is True,
              metadata.get('data_sha256') == file_sha256(data_path),
              metadata.get('cache_sha256') == file_sha256(cache_path),
              metadata.get('samples') == sample_records(data)]
    if not all(checks):
        raise ValueError('缓存校验失败：数据版本、样本身份/顺序或缓存内容已变化，请重新计算。')
    features = pd.read_csv(cache_path)
    if len(features) != len(data) or list(features.columns) != metadata['columns']:
        raise ValueError('缓存行数或列名与元数据不一致')
    # 不在全数据上丢列；缺失值处理、筛选都由训练折决定。
    features = features.apply(pd.to_numeric, errors='raise')
    return features.replace([np.inf, -np.inf], np.nan), metadata
