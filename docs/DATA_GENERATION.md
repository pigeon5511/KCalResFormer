# 数据生成与从头训练

生成入口是 `scripts/generate_data.py`，MATLAB 核心是 `scripts/matlab/generate_formal.m`。Python 负责数据协议、条件和种子计划、分片校验、调用 MATLAB 和最终审计；MATLAB 负责 NR OFDM 波形、TDL-A 信道、理想同步、信道真值和接收网格。

## 环境

- Python 3.10 或更新版本，以及 `requirements.txt` 中的依赖。
- MATLAB 和 5G Toolbox；本地检查版本为 R2021b / 5G Toolbox 2.3。
- 数据生成使用 CPU。完整训练入口使用 CUDA PyTorch；小样本检查可用 CPU。
- 无需 Python MATLAB Engine，也无需 Parallel Computing Toolbox。`--workers 2` 启动两个独立 MATLAB 进程，请根据内存与许可条件选择；默认一个进程。

MATLAB 可以通过 PATH、环境变量 `MATLAB_EXECUTABLE` 或命令参数 `--matlab` 指定。示例中的路径请替换为自己安装的可执行文件路径：

```powershell
python scripts/generate_data.py all --smoke --output-root data/smoke --matlab "MATLAB可执行文件的完整路径"
```

## 小样本：检查完整流程

以下命令从项目根目录运行。它实际调用 MATLAB，生成 24 个独立信道：校准 6、训练 8、验证 10（选模 6、alpha 4）。每片 2 个信道。这些数据只用于检查流程。

```powershell
python scripts/generate_data.py all --smoke --output-root data/smoke
python scripts/train.py train --smoke --generated-data --device cpu --data-root data/smoke --output-root runs/smoke
python scripts/import_checkpoint.py --fit-root runs/smoke --seed 2026102301 --output checkpoints/smoke_seed_2026102301.pt
python scripts/prepare_data.py --data-root data/smoke --split validation --start-channel 1 --channels 10 --layouts A B C D --output runs/smoke_validation.npz
python scripts/evaluate.py --checkpoint checkpoints/smoke_seed_2026102301.pt --input runs/smoke_validation.npz --output runs/smoke_evaluation
```

数据与权重均标记为 `SMOKE_ONLY` 或对应的 smoke 状态。上述验证数据包括选模和 alpha 开发池，评估仅用于检查接口，不是独立精度结果。正式训练不会误接收 smoke 数据；必须明确提供 `--smoke`。

## 完整源数据：训练自己的权重

```powershell
python scripts/generate_data.py plan
python scripts/generate_data.py all --output-root data/formal_dmrs
python scripts/train.py train --generated-data --device cuda --data-root data/formal_dmrs --output-root runs/formal
python scripts/import_checkpoint.py --fit-root runs/formal --seed 2026102301 --output checkpoints/retrained_seed_2026102301.pt
```

完整规格保持原源数据协议：

| 用途 | 独立信道数 | 使用规则 |
|---|---:|---|
| 物理前端校准 | 5,000 | 仅 A/C |
| 神经网络训练 | 100,000 | 仅 A/C，每信道两个配对视图 |
| 验证选模 | 8,000 | validation 前 8,000，仅 A/C |
| alpha 拟合 | 2,000 | validation 后 2,000，仅 A/C |
| 合计 | 115,000 | 230 个 MAT 分片，每片 500 信道 |

固定 SISO、TDL-A、120×14、15 kHz、normal CP。条件为 SNR `0:5:25` dB、时延 `25:25:300` ns、多普勒 `50:50:1000` Hz，共 1,440 组合。随机条件计划按每 1,440 信道重新排列，保留原始随机种子及三个互不重叠的信道、数据、噪声种子流。

A/B/C/D 是同一物理信道的观测布局；训练加载器仅暴露 A/C。完整 OFDM 波形中的导频并集为 200 RE，公共 BER 掩码为 1,480 RE。AWGN 在 OFDM 解调后的资源网格加入。高多普勒下有 ICI，不能假定无噪接收网格严格等于 `H_true * X`。

## 输出、续跑和审计

```text
data/formal_dmrs/
├── protocol.json
├── geometry.mat
├── manifest.json               # 所有分片、重放和审计通过后生成
├── audit.json
├── calibration/
├── train/
├── validation/
└── generation/
    ├── generate_formal.m       # 随数据冻结的生成器副本
    ├── protocol.json
    ├── schedule_*.mat
    ├── schedule_audit.json
    ├── smoke.json
    ├── freeze.json
    ├── replay.json
    └── logs/
```

中断或已经完成时，重复 `all` 命令，或执行：

```powershell
python scripts/generate_data.py resume --output-root data/formal_dmrs --workers 2
```

每个已完成分片必须存在 `.mat.complete.json`，并与协议、生成器和 MAT 哈希匹配，才会跳过。校验不通过会停止。若发生“最终 MAT 已写好，但完成标记尚未提交”的中断，程序报告孤立分片并保留现场，需要先检查该分片，不能直接假定它已完成。正常写入中断留下的 `.partial.mat` 可在重新计算同一未提交分片时覆盖。

也可以分阶段执行，使用同一个输出目录：

```powershell
python scripts/generate_data.py setup --output-root data/formal_dmrs
python scripts/generate_data.py smoke --output-root data/formal_dmrs
python scripts/generate_data.py freeze --output-root data/formal_dmrs
python scripts/generate_data.py generate --output-root data/formal_dmrs
python scripts/generate_data.py replay --output-root data/formal_dmrs
python scripts/generate_data.py audit --output-root data/formal_dmrs
```

`setup` 只建立协议和种子计划，不调用 MATLAB。`smoke` 在单独保留的种子上检查几何、静态重建、重放和高多普勒有限性。`replay` 重建三个源划分首分片中的第一个信道，并逐元素检查 `H/X/Y/tx_bits`。`audit` 扫描所有存储数组，验证形状、数据类型、条件、种子互斥、真值去重、QPSK 映射、目标能量、噪声功率和哈希。

新输出目录应专用于本数据集。完成后应保留整个目录与生成日志。分片本体约 6.6 GB；完整训练还要存物理前端缓存和断点，建议至少预留 20 GB。生成器和训练器都用执行锁阻止多个入口同时写同一目录。

## 与原论文数据的关系

默认源数据规格、种子计划和信道生成公式来自原正式实验。不同 MATLAB 版本、协议时间戳和 MAT 文件头可能导致文件哈希变化，数值逐位相同也不能默认保证。

训练时的 `--generated-data` 会验证新数据的冻结记录和审计结果，并把新清单哈希绑定到本次训练输出；不要求它冒充原论文的字节完全相同数据。原已存数据仍可通过未加此选项的入口校验原清单。

本入口不生成独立最终测试集，也不把 B/D 开发视图当作新的独立信道测试。独立最终实验需要在模型、校准、alpha 和评价规则冻结后，另行规定不相交的测试种子。
