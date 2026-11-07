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
- `analysis/results/`：结果汇总与实验比较。
- `analysis/features/`：特征消融与 SHAP 分析。
- `run_pipeline.py`：运行单组建模实验。
- `run_experiments.py`：批量运行多组实验。

## 环境配置

先安装 Miniconda 或 Anaconda。在仓库根目录（包含 `run_pipeline.py` 的文件夹）运行以下命令。

| 环境 | Python | 用途 |
|---|---|---|
| `porscadb-ml` | 3.12 | 模型训练、特征消融、SHAP 和绘图 |
| `porscadb-features` | 3.8 | Mordred 分子特征生成 |

创建并激活训练环境：

```bash
conda env create -f environment-ml.yml
conda activate porscadb-ml
```

需要计算分子特征时，创建并激活另一个环境：

```bash
conda env create -f environment-features.yml
conda activate porscadb-features
```

环境文件会自动安装对应 `requirements-*.txt` 中的依赖。训练和绘图时，用 `conda activate porscadb-ml` 切回训练环境。

汇总已有实验结果并生成 SHAP 图片：

```bash
conda activate porscadb-ml
python analysis/results/summarize.py
python analysis/features/summarize_shap.py --plot
```

两个汇总脚本仅导出 CSV 时只需 Python 标准库；输出 PNG 还需要 Matplotlib。

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

数据字段、实验输出和分析脚本的说明见 [`data/README.md`](data/README.md)、[`experiments/README.md`](experiments/README.md) 和 [`analysis/README.md`](analysis/README.md)。

## 许可

代码采用 [MIT 许可证](LICENSE)。
