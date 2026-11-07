# PorScaDB-ML

PorScaDB-ML 根据卟啉及相关大环分子的 SMILES 预测其 DFT HOMO–LUMO 能隙。仓库包含特征生成与模型训练流程、训练好的模型，以及分析结果。

项目使用三种数据划分比较模型表现：

- **随机划分**：训练集和测试集包含相同化学空间中的样本。
- **meso 取代基划分**：测试集包含训练中未出现的取代基组合。
- **骨架划分**：测试集包含训练中未出现的大环骨架。

## 仓库内容

- `core/`：数据处理、分子特征、实验设置与模型训练代码。
- `data/`：分子数据和结构文件。
- `experiments/`：模型文件和实验输出。
- `analysis/`、`reports/`：分析脚本、图表和结果摘要。
- `run_pipeline.py`：运行单组建模实验。
- `run_experiments.py`：批量运行多组实验。

## 环境配置

特征生成与模型训练使用不同的 Conda 环境：

```bash
conda env create -f environment-features.yml
conda env create -f environment-ml.yml
```

Python 版本和依赖项见对应的环境文件。

## 运行

激活训练环境后，可先查看命令参数：

```bash
python run_pipeline.py --help
```

例如，使用准备好的特征表运行随机划分实验：

```bash
python run_pipeline.py path/to/features.csv \
  --split-file path/to/splits.json \
  --split-name random \
  --run-name example \
  --output-root experiments/example
```

数据、实验输出和分析脚本的详细说明见 [`data/README.md`](data/README.md)、[`experiments/README.md`](experiments/README.md) 和 [`analysis/README.md`](analysis/README.md)。

## 引用与许可

如使用本项目，请按 [`CITATION.cff`](CITATION.cff) 中的信息引用。代码采用 [MIT 许可证](LICENSE)。
