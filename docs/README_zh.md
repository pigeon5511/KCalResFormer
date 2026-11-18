# KCalResFormer 代码整理版

[返回项目首页](../README.md)

本目录对应当前论文中的共享 KCalResFormer：精确 K-LMMSE 物理重建加 CNN–Transformer 神经残差，固定 120×14 资源栅格，神经参数 61,741。物理校准、神经训练、选模和 alpha 拟合均仅使用 A/C，拟合后冻结参数并迁移至 B/D。

原研究工程及实验数据未移动。这里是独立整理出的源码目录，运行时不导入原工程。

| 目录 | 内容 |
|---|---|
| `models/` | 物理前端、共享残差网络、统一模型接口 |
| `configs/` | 模型参数、训练配置、数据协议、几何、权重哈希及源码校验清单 |
| `scripts/` | 数据生成、转换、训练、检查点导入、推理和评估入口 |
| `scripts/matlab/` | 原正式实验的信道与 OFDM 波形生成器 |
| `examples/` | 人工合成输入与最小可运行示例 |
| `checkpoints/` | 放置经校验转换后的预训练权重 |
| `docs/` | 数据合同、数据生成指南、复现说明及整理状态 |

以下命令均在项目根目录运行。先安装依赖并检查接口：

```powershell
python -m pip install -r requirements.txt
python examples/quickstart.py
python scripts/train.py plan
```

正式训练命令、推理命令和 Python API 见英文 [README](../README.md)。MAT 数据转换使用 `scripts/prepare_data.py`；训练仅运行共享精确 K 版，保留原正式实验的 A/C 数据路由、复数 MSE、选模池、alpha 池和断点续训逻辑。

## 从头生成数据并训练

数据生成需要 **MATLAB 和 5G Toolbox**。本地已使用 MATLAB R2021b / 5G Toolbox 2.3 检查生成流程；不需要安装 Python 的 MATLAB Engine。Python 负责种子计划、启动 MATLAB 和数据验收。

先在小样本上检查完整流程（24 个信道，只用于检查代码）：

```powershell
python scripts/generate_data.py all --smoke --output-root data/smoke
python scripts/train.py train --smoke --generated-data --device cpu --data-root data/smoke --output-root runs/smoke
```

再按论文源数据规格生成完整数据、训练并导出自己的权重：

```powershell
python scripts/generate_data.py all --output-root data/formal_dmrs
python scripts/train.py train --generated-data --device cuda --data-root data/formal_dmrs --output-root runs/formal
python scripts/import_checkpoint.py --fit-root runs/formal --seed 2026102301 --output checkpoints/retrained_seed_2026102301.pt
```

如果系统找不到 MATLAB，在生成命令后加 `--matlab "MATLAB可执行文件的完整路径"`，也可以设置 `MATLAB_EXECUTABLE`。正式生成包含训练 100,000、校准 5,000、验证 10,000 个独立信道；验证前 8,000 选模、后 2,000 拟合 alpha。支持校验后续跑；完整命令和恢复说明见 [数据生成指南](DATA_GENERATION.md)。独立最终测试集需另行冻结协议，本入口不生成。

`--generated-data` 会把新数据的协议、分片哈希和几何绑定到本次训练，避免要求别人重新生成的数据具有原 MAT 文件的字节哈希。未加此选项的旧数据路径仍校验原正式数据清单。

**当前权重状态：**已找到论文正式冻结记录、系数和 alpha，但未找到哈希对应的三个正式检查点。本目录没有把早期 15-epoch 权重或后续 A/B/C/D 专用模型当作论文共享权重。`examples/quickstart.py` 明确使用随机初始化，只证明接口可运行。正式权重恢复到本地后，使用 `scripts/import_checkpoint.py` 校验并转换。

本次只整理和本地验证，没有创建或上传 GitHub 仓库。完整数据、训练缓存、实验日志、旧压缩包和虚拟环境未收进开源目录。别人可以通过生成器产生自己的源数据；正式论文链接和引用尚待补充。

源码校验清单位于 `configs/SOURCE_MANIFEST.json`。修改源码或说明文件后，运行 `python scripts/update_manifest.py` 更新清单，再使用新的输出目录开始实验。

本项目采用 [MIT 许可](../LICENSE)，版权署名为 `Copyright (c) 2026 Yusheng Zhang`。外部依赖说明见 [项目首页](../README.md#third-party-dependencies)。
