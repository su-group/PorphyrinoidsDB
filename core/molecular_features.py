"""
使用mordred库计算分子特征，包含多种分子指纹
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs as AtomPairs
from rdkit.Chem.AtomPairs import Torsions


def calculate_morgan_fingerprint(mols, radius=2, n_bits=2048):
    """
    计算Morgan指纹（圆形指纹）

    Parameters:
    mols (list): RDKit分子对象列表
    radius (int): 指纹半径，默认2
    n_bits (int): 指纹位数，默认2048

    Returns:
    pandas.DataFrame: 指纹特征数据框
    """
    fps = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((1,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    
    fps_array = np.vstack(fps)
    fps_df = pd.DataFrame(fps_array, columns=[f'Morgan_{i}' for i in range(n_bits)])
    print(f"计算得到Morgan指纹特征数量: {n_bits}")
    
    return fps_df


def calculate_maccs_fingerprint(mols):
    """
    计算MACCS键指纹（166位预定义子结构键）

    Parameters:
    mols (list): RDKit分子对象列表

    Returns:
    pandas.DataFrame: 指纹特征数据框
    """
    fps = []
    for mol in mols:
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((1,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    
    fps_array = np.vstack(fps)
    # MACCS有167位，第0位始终为0，实际使用166位
    fps_df = pd.DataFrame(fps_array, columns=[f'MACCS_{i}' for i in range(fps_array.shape[1])])
    print(f"计算得到MACCS指纹特征数量: {fps_array.shape[1]}")
    
    return fps_df


def calculate_atom_pairs_fingerprint(mols, n_bits=2048):
    """
    计算原子对指纹（基于原子类型和拓扑距离）

    Parameters:
    mols (list): RDKit分子对象列表
    n_bits (int): 指纹位数，默认2048

    Returns:
    pandas.DataFrame: 指纹特征数据框
    """
    from rdkit.Chem import rdFingerprintGenerator
    fp_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    fps = []
    for mol in mols:
        fp = fp_gen.GetFingerprint(mol)
        arr = np.zeros((1,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    
    fps_array = np.vstack(fps)
    fps_df = pd.DataFrame(fps_array, columns=[f'AtomPair_{i}' for i in range(n_bits)])
    print(f"计算得到原子对指纹特征数量: {n_bits}")
    
    return fps_df


def calculate_topological_torsion_fingerprint(mols, n_bits=2048):
    """
    计算拓扑扭转指纹（基于连续4原子的拓扑信息）

    Parameters:
    mols (list): RDKit分子对象列表
    n_bits (int): 指纹位数，默认2048

    Returns:
    pandas.DataFrame: 指纹特征数据框
    """
    from rdkit.Chem import rdFingerprintGenerator
    fp_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
    fps = []
    for mol in mols:
        fp = fp_gen.GetFingerprint(mol)
        arr = np.zeros((1,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    
    fps_array = np.vstack(fps)
    fps_df = pd.DataFrame(fps_array, columns=[f'TopoTorsion_{i}' for i in range(n_bits)])
    print(f"计算得到拓扑扭转指纹特征数量: {n_bits}")
    
    return fps_df


def calculate_rdkit_fingerprint(mols, min_path=1, max_path=7, fp_size=2048):
    """
    计算RDKit拓扑指纹（基于路径的指纹）

    Parameters:
    mols (list): RDKit分子对象列表
    min_path (int): 最小路径长度，默认1
    max_path (int): 最大路径长度，默认7
    fp_size (int): 指纹位数，默认2048

    Returns:
    pandas.DataFrame: 指纹特征数据框
    """
    from rdkit.Chem import rdmolops
    fps = []
    for mol in mols:
        fp = rdmolops.RDKFingerprint(mol, minPath=min_path, maxPath=max_path, fpSize=fp_size)
        arr = np.zeros((1,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    
    fps_array = np.vstack(fps)
    fps_df = pd.DataFrame(fps_array, columns=[f'RDKitFP_{i}' for i in range(fp_size)])
    print(f"计算得到RDKit拓扑指纹特征数量: {fp_size}")
    
    return fps_df


def calculate_molecular_features(smiles_list, output_file=None, remove_constant_features=True, use_fingerprints=False,
                                 clean_features=True, nproc=None):
    """
    使用mordred计算分子特征，可选包含多种分子指纹

    Parameters:
    smiles_list (list): SMILES字符串列表
    output_file (str): 特征保存文件路径
    remove_constant_features (bool): 是否删除冗余特征（所有值都相同的列）
    use_fingerprints (bool): 是否计算分子指纹特征（默认 False）
    clean_features (bool): 历史清理开关；训练缓存必须为 False
    nproc (int): Mordred 进程数；冒烟检查用 1

    Returns:
    pandas.DataFrame: 分子特征数据框
    """
    # 创建分子对象，过滤掉无效的SMILES
    mols = []
    valid_smiles = []
    for i, smiles in enumerate(smiles_list):
        # 先检查smiles是否为None或空字符串
        if not isinstance(smiles, str) or not smiles.strip():
            print(f"警告: SMILES为空或None: {smiles}")
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                valid_smiles.append(smiles)
            else:
                print(f"警告: 无法解析SMILES: {smiles}")
        except Exception as e:
            print(f"警告: 解析SMILES时出错: {smiles}, 错误: {e}")

    if len(mols) == 0:
        raise ValueError("没有有效的SMILES分子结构")

    print(f"成功解析 {len(mols)} 个分子，忽略 {len(smiles_list) - len(mols)} 个无效SMILES")

    # 延迟导入：模型训练/SHAP环境只读取预计算特征，不应被 Mordred
    # 与新版 NumPy/NetworkX 的依赖冲突阻塞。
    from mordred import Calculator, descriptors

    # 创建mordred计算器
    calc = Calculator(descriptors, ignore_3D=True)

    # 计算mordred特征
    print("正在计算mordred分子特征...")
    mordred_features = calc.pandas(mols, nproc=nproc)
    print(f"原始mordred特征数量: {len(mordred_features.columns)}")

    # === 关键修复步骤 ===

    # 1. 强制转换为数值型，无法转换的设为 NaN（包括 "None", inf, -inf）
    mordred_features = mordred_features.apply(pd.to_numeric, errors='coerce')

    # 2. 将 inf 和 -inf 显式替换为 NaN
    mordred_features = mordred_features.replace([np.inf, -np.inf], np.nan)

    # 新训练缓存设置 clean_features=False，保留原始列，由每个训练折处理缺失值。
    # clean_features=True 仅兼容历史分析脚本。
    initial_cols = len(mordred_features.columns)
    if clean_features:
        mordred_features = mordred_features.dropna(axis=1, how='any')
    print(f"缺失列处理后，mordred特征数量: {len(mordred_features.columns)} (删除了 {initial_cols - len(mordred_features.columns)} 列)")

    # 4. 删除常量特征（所有值相同）
    if clean_features and remove_constant_features:
        initial_cols = len(mordred_features.columns)
        mordred_features = mordred_features.loc[:, mordred_features.var() > 1e-10]
        print(f"删除常量特征后，剩余mordred特征数量: {len(mordred_features.columns)} (删除了 {initial_cols - len(mordred_features.columns)} 列)")

    # 5. 确保所有列都是 float64（避免 object 类型）
    mordred_features = mordred_features.astype(np.float64)

    # 6. 计算多种分子指纹并合并
    if not use_fingerprints:
        print(f"仅使用Mordred描述符，特征数量: {len(mordred_features.columns)}")
        if output_file:
            mordred_features.to_csv(output_file, index=False)
        return mordred_features

    print("正在计算分子指纹...")
    fingerprint_dfs = []

    # 6.1 Morgan指纹（圆形指纹，捕捉局部原子环境）
    print("  计算Morgan指纹...")
    morgan_fp = calculate_morgan_fingerprint(mols, radius=2, n_bits=2048)
    fingerprint_dfs.append(morgan_fp)

    # 6.2 MACCS键指纹（166位预定义子结构键）
    print("  计算MACCS指纹...")
    maccs_fp = calculate_maccs_fingerprint(mols)
    fingerprint_dfs.append(maccs_fp)

    # 6.3 原子对指纹（基于原子类型和拓扑距离）
    print("  计算原子对指纹...")
    atompair_fp = calculate_atom_pairs_fingerprint(mols, n_bits=2048)
    fingerprint_dfs.append(atompair_fp)

    # 6.4 拓扑扭转指纹（基于连续4原子的拓扑信息）
    print("  计算拓扑扭转指纹...")
    toto_fp = calculate_topological_torsion_fingerprint(mols, n_bits=2048)
    fingerprint_dfs.append(toto_fp)

    # 6.5 RDKit拓扑指纹（基于路径的指纹）
    print("  计算RDKit拓扑指纹...")
    rdkit_fp = calculate_rdkit_fingerprint(mols, min_path=1, max_path=7, fp_size=2048)
    fingerprint_dfs.append(rdkit_fp)

    # 合并所有指纹特征
    all_fingerprints = pd.concat(fingerprint_dfs, axis=1)
    print(f"所有分子指纹总特征数量: {len(all_fingerprints.columns)}")

    # 合并mordred特征和所有指纹特征
    features = pd.concat([mordred_features, all_fingerprints], axis=1)
    print(f"合并后的总特征数量: {len(features.columns)}")


    if output_file:
        features.to_csv(output_file, index=False)

    return features


def load_features_from_file(file_path):
    """
    从文件加载预计算的特征（增强版）

    Parameters:
    file_path (str): 特征文件路径

    Returns:
    pandas.DataFrame: 分子特征数据框
    """
    # 检查文件是否存在
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"特征文件不存在: {file_path}")

    # 读取时自动将 "None", "null" 等视为 NaN
    df = pd.read_csv(file_path, na_values=['None', 'null', '', 'NaN', 'nan'])

    # 再次确保数值转换和 inf 处理
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)

    # 删除剩余的 NaN 列（防御性编程）
    initial_cols = len(df.columns)
    df = df.dropna(axis=1, how='any')
    print(f"从文件加载特征: {file_path}")
    print(f"删除包含NaN的列后，剩余特征数量: {len(df.columns)} (删除了 {initial_cols - len(df.columns)} 列)")

    return df.astype(np.float64)

if __name__ == '__main__':
    # 示例数据
    try:
        data = pd.read_csv("data/Labeled-Gap-IsoelectronicSubstitution.csv")
        smiles_list = data["SMILES"]
        
        # 计算特征
        features = calculate_molecular_features(smiles_list, output_file="data/Labeled-Gap-IsoelectronicSubstitution_feature.csv")
        print("特征计算完成")
        print(features.head())
        
    except FileNotFoundError:
        print("示例数据文件不存在，请确保数据文件路径正确")
    except Exception as e:
        print(f"计算特征时出错: {e}")
        import traceback
        traceback.print_exc()