# KCalResFormer：校准 Kronecker LMMSE 与 CNN–Transformer 残差信道估计

[返回英文项目首页](../README.md)

面向多种 DM-RS 布局的 SISO OFDM 信道估计模型。

## 项目概述

KCalResFormer 将校准 Kronecker LMMSE 估计器与共享 CNN–Transformer 残差网络结合。物理估计器根据导频观测重建完整信道，神经网络学习对重建结果的修正，并使用标称 SNR、均方根时延扩展和最大多普勒频移作为辅助信息。

当前实现使用固定的 **120 × 14** 资源网格，神经网络包含 **61,741 个可训练参数**。物理校准和神经训练使用 DM-RS 布局 A/C，拟合后的模型直接用于 B/D，无需重新训练。

本仓库提供模型代码、MATLAB 数据生成器以及训练和推理脚本，不附带预训练权重。

## 模型结构

估计器由三个部分组成：

1. **校准 K-LMMSE 物理前端**：在假定的可分离协方差下，通过精确 Kronecker 求解重建信道。
2. **CNN–Transformer 残差网络**：结合局部卷积处理和注意力，预测信道修正量。
3. **残差融合**：对神经修正量进行缩放，再加到物理估计上。

$$
\widehat{H} = \widehat{H}_{\mathrm{K\text{-}LMMSE}} + \alpha\,\Delta_{\mathrm{NN}}.
$$

## 安装

克隆仓库并安装 Python 依赖：

```bash
git clone https://github.com/pigeon5511/KCalResFormer.git
cd KCalResFormer
python -m pip install -r requirements.txt
```

环境要求：

- **Python 3.10 或更新版本**，以及 PyTorch、NumPy 和 SciPy。
- 数据生成需要 **MATLAB 和 5G Toolbox**。
- 完整训练需要 **支持 CUDA 的 PyTorch**；小规模流程检查和推理可以使用 CPU。

GPU 训练前请安装与 CUDA 环境匹配的 PyTorch。MATLAB 生成器已在 R2021b 和 5G Toolbox 2.3 下检查，无需安装 Python MATLAB Engine。

## 快速开始

使用随包提供的合成导频观测检查安装：

```bash
python examples/quickstart.py
```

程序会输出参数数量和形状为 `(4, 120, 14)` 的复数信道张量。该示例不加载训练权重，用于检查模型接口。

## 项目结构

```text
KCalResFormer/
├── models/              # 物理估计器与 CNN–Transformer 残差网络
├── configs/             # 模型、训练及数据配置
├── scripts/
│   ├── matlab/          # MATLAB 信道和 OFDM 数据生成器
│   └── _training/       # 训练流程
├── examples/            # 合成输入和最小示例
├── checkpoints/         # 导出的模型权重
├── docs/                # 中文指南和详细数据说明
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

以下命令均在项目根目录运行。

## 数据生成

生成训练、物理校准和验证数据：

```bash
python scripts/generate_data.py all --output-root data/formal_dmrs
```

生成器使用 SISO TDL-A 信道、120 × 14 OFDM 网格、15 kHz 子载波间隔及 QPSK 符号。

| 数据划分 | 独立信道数 |
|---|---:|
| 训练 | 100,000 |
| 物理校准 | 5,000 |
| 验证 | 10,000 |

需要先检查小规模流程时，使用 `--smoke`，并指定单独的输出目录：

```bash
python scripts/generate_data.py all --smoke --output-root data/smoke
```

可以通过 `MATLAB_EXECUTABLE` 或 `--matlab` 参数指定 MATLAB。分阶段执行和中断恢复方法见[数据生成指南](DATA_GENERATION.md)。

## 训练

使用生成的数据训练共享 KCalResFormer 模型：

```bash
python scripts/train.py train --generated-data --device cuda --data-root data/formal_dmrs --output-root runs/formal
```

批大小、优化器、学习率和早停等设置位于 `configs/training.json`。选模和残差缩放系数拟合使用验证集内互不重叠的部分，且只使用 A/C 观测。中断后，使用相同数据和输出目录重复执行上述命令即可继续。

训练完成后，导出用于推理的模型：

```bash
python scripts/import_checkpoint.py --fit-root runs/formal --seed 2026102301 --output checkpoints/kcalresformer.pt
```

导出文件包含训练后的网络、校准物理参数及拟合得到的残差缩放系数。

使用小数据集在 CPU 上检查训练流程：

```bash
python scripts/train.py train --smoke --generated-data --device cpu --data-root data/smoke --output-root runs/smoke
```

## 推理与评估

下面的示例准备少量验证数据，估计其信道并计算误差指标：

```bash
python scripts/prepare_data.py --data-root data/formal_dmrs --split validation --channels 4 --layouts A B C D --output runs/validation_subset.npz
python scripts/infer.py --checkpoint checkpoints/kcalresformer.pt --input runs/validation_subset.npz --output runs/channel_estimates.npy --device cpu
python scripts/evaluate.py --checkpoint checkpoints/kcalresformer.pt --input runs/validation_subset.npz --output runs/evaluation --device cpu
```

评估器输出复数 MSE、NMSE 及分布局汇总。输入包含必要的接收符号、参考比特和数据掩码时，还会计算未编码 QPSK BER。报告测试性能时，应使用独立于训练与验证的测试集。

推理结果为形状 `(batch, 120, 14)` 的复数信道张量。输入字段和单位见[数据格式说明](DATA.md)。

## 配置

| 文件 | 用途 |
|---|---|
| `configs/model.json` | 模型结构和推理默认参数 |
| `configs/training.json` | 训练、校准及选模设置 |
| `configs/dataset_protocol.json` | 信道条件和数据规模 |
| `configs/geometry.mat` | 资源网格坐标及导频布局 |

修改仓库文件后，开始新的训练前先更新源码清单：

```bash
python scripts/update_manifest.py
```

## 第三方依赖

PyTorch、NumPy 和 SciPy 需单独安装，并遵循各自许可。数据生成还需要单独安装并取得 MATLAB 和 5G Toolbox 的使用许可。本仓库不分发这些依赖的实现。

## 许可

本项目采用 [MIT 许可](../LICENSE)。

Copyright (c) 2026 Yusheng Zhang.
