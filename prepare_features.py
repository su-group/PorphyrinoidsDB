"""在 Mordred 兼容环境生成原始特征及对应关系证明，训练环境只读取缓存。"""
import argparse
from pathlib import Path

from core.data_processing import load_valid_data, save_feature_cache
from core.molecular_features import calculate_molecular_features


def main():
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=root / 'data/Data-20261012.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fingerprints', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='只计算固定的 40 个真实样本')
    parser.add_argument('--nproc', type=int, default=1)
    args = parser.parse_args()
    suffix = '_raw_smoke_features.csv' if args.smoke else '_raw_features.csv'
    if args.output is None:
        args.output = args.data.with_name(args.data.stem + suffix)
    data = load_valid_data(args.data, smoke=args.smoke)
    # 原始描述符不能按整个数据集的缺失率、方差筛选。训练时每折独立处理。
    features = calculate_molecular_features(
        data['SMILES'], remove_constant_features=False, clean_features=False,
        use_fingerprints=args.fingerprints, nproc=args.nproc)
    save_feature_cache(features, data, args.data, args.output, args.fingerprints)
    print(f'缓存已保存: {args.output}；shape={features.shape}；smoke={args.smoke}')


if __name__ == '__main__':
    main()
