# 发布内容

本仓库提供共享 KCalResFormer 的模型实现、配置、MATLAB 数据生成器、训练流程、推理与评估脚本，以及中英文使用指南。

完整数据和预训练权重不随仓库分发。按照[数据生成指南](DATA_GENERATION.md)生成源数据后，可运行训练入口并导出自己的模型。

中文指南见 [README_zh.md](README_zh.md)，英文说明及第三方依赖要求见[项目首页](../README.md#third-party-dependencies)。源码清单位于 `configs/SOURCE_MANIFEST.json`。

小样本流程用于检查生成、训练、导出和评估接口；模型性能应在独立测试集上评估。数据生成需要 MATLAB 和 5G Toolbox，完整训练需要支持 CUDA 的 PyTorch。

本项目采用 MIT 许可，版权署名为 `Copyright (c) 2026 Yusheng Zhang`。
