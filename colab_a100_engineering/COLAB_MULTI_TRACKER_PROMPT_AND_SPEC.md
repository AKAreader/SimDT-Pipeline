# SimDT 围点打援｜多算法跑分对比与 SiamDT 改进工程 Colab A100 部署规范与提示词模板

> **适用目标**：本文件将原本面向“单一 SiamDT 算法”的 Colab A100 部署方案，正式升级为面向 **“原始 SiamDT 基线 + 2025–2026 核心实验池多算法横向跑分对比 + 逐帧故障诊断（Rescue Matrix）+ 后续单变量受控改进”** 的统一工程架构。

---

## 一、 升级版提示词（可直接复制用于 Codex / Colab Agent）

```markdown
请为我的 **SimDT 围点打援（SiamDT 基线 + 2025–2026 多类别跟踪算法横向跑分对比与诊断改进）+ Anti-UAV410** 项目建立规范、可复现的 Google Colab A100 深度学习运行工程，并实际完成可执行的部署准备工作。

官方主仓库：https://github.com/HwangBo94/Anti-UAV410 （锁定 Commit: `8a8eb04d976e9386b7c9c3ada5c85e5086013d52`）
主诊断对象位置：`trackers/SiamDT/`（原始 245 参数张量 `SiamDTRCNN` / Swin-Tiny + 5层 FPN）
对比算法池（Core Experimental Pool）：
- Group I（同域 Anti-UAV410 训练组）：`SiamDT_Baseline`、`FocusTrack`（含同库重训 `OSTrack`/`ROMTrack`/`ZoomTrack`/`DropTrack`）、`MemLoTrack`（`scratch` & `retrained`）
- Group II（零样本/基础模型机制探针组）：`DAM4SAM`（SAM 2.1++）、`SAMOSA` & `SAMURAI`（SAM 2.1 + 运动预测/恢复）、`UETrack`（CVPR 2026）/ `MCITrack`（AAAI 2025）
- Group III（后续受控改进组）：`SiamDT_Ours`（严禁混入师兄方法 `SiamISDT`）

### 1. 官方代码与多算法依赖检查

阅读 Anti-UAV410 仓库 README、SiamDT 安装文档、训练与测试入口、数据加载代码、配置文件及对比算法依赖。

明确：
- 官方数据集（`train` 200 / `val` 90 / `test` 120）及诊断子集（`Discovery Set` / `Holdout Verification Set`）、SiamDT 训练预训练权重（`cascade_mask_rcnn_swin_tiny.pth.tar`）、官方测试 Checkpoint（`siamdt_swin_tiny_sgd.pth`）及对比算法权重的下载来源、SHA-256 校验和目录要求。
- 双轨环境隔离方案：
  1. **Tier-1 (`env_siamdt_cu118`)**：SiamDT 专用的 Python 3.10 + PyTorch 2.0.0+cu118 + MMCV-full 1.7.2 + Swin Transformer + Apex (`opt_level="O1"`) 适配。
  2. **Tier-2 (`env_modern_pyt2`)**：2025–2026 对比算法专用的 PyTorch >= 2.3/2.4 (CUDA 12.1) + `timm` + `SAM 2.1` + `DINOv2` + `filterpy` 环境。
- 官方旧版 SiamDT 环境在当前 Colab A100 (`sm_80`) 上的兼容性及最小零语义适配（仅允许版本门禁、跨设备索引修复、纯 Python Apex `torch._six` 垫片及数据集路径参数化，严格保留官方 SiamDT 的 245 个参数张量、模型结构、训练配置和实验协议，不为安装方便而擅自替换算法或修改训练超参数）。

### 2. 建立支持多算法对比的 Google Drive 工程目录

使用 `/content/drive/MyDrive/SiamDT_Project/` 作为默认持久化根目录，采用支持多算法横向评测与逐帧遥测诊断的目录结构：

SiamDT_Project/
├── datasets/
│   ├── AntiUAV410/
│   │   ├── train/                      # 200 序列
│   │   ├── val/                        # 90 序列
│   │   ├── test/                       # 120 序列（Phase 1-6 期间冻结只读）
│   │   └── annos/                      # 官方属性标注 (TC/OV/SV/FM/OC/DBC/TS/SS/MS/NS)
│   ├── diagnostic_splits/              # 冻结的数据集划分清单 (JSON)
│   │   ├── val90_full.json
│   │   ├── discovery_set.json          # 故障诊断与 Rescue Matrix 分析集
│   │   ├── holdout_verification.json   # 改进方案独立验证集
│   │   ├── challenge_windows.json      # 典型挑战时间窗 (Canonical Windows)
│   │   └── test120_lock.json           # 最终测试集防泄漏锁
│   └── archives/                       # 原始 ZIP 压缩包与 Range 缓存
├── repos/                              # Git 仓库冷备份与补丁记录（运行期同步至 /content/runtime/）
│   ├── Anti-UAV410_official/           # 官方主库（含 trackers/SiamDT 及统一评测工具箱）
│   ├── core_pool/                      # 核心对比算法源码（FocusTrack, MemLoTrack, DAM4SAM, SAMOSA, UETrack）
│   └── patches/                        # 最小兼容性补丁与 git diff 审计记录
├── weights/
│   ├── foundation_backbones/           # 通用底座预训练权重（Swin-Tiny, SAM2.1, DINOv2 等）
│   │   └── swin_tiny/cascade_mask_rcnn_swin_tiny.pth.tar
│   ├── in_domain_antiuav410/           # Group I: 同域训练模型权重
│   │   ├── SiamDT_Baseline/siamdt_swin_tiny_sgd.pth
│   │   ├── FocusTrack/
│   │   └── MemLoTrack/
│   └── zero_shot_sota/                 # Group II: 零样本/跨域模型权重
│       ├── DAM4SAM/
│       ├── SAMOSA/
│       └── UETrack_MCITrack/
├── environments/
│   ├── requirements/                   # Tier-1 (SiamDT) 与 Tier-2 (Modern) 锁定依赖清单
│   ├── wheels/                         # 预编译 whl 离线缓存 (mmcv-full 1.7.2, uv, apex 等)
│   └── diagnostics/                    # A100 硬件、CUDA 算子与严格权重加载核验 JSON
├── experiments/
│   ├── runs_training/                  # 训练实验独立目录（按时间戳+实验名隔离，含 checkpoints/ 与 logs/）
│   │   ├── siamdt_baseline_repro/
│   │   └── siamdt_ours_ablations/
│   ├── benchmark_eval/                 # 多算法统一评测输出（严格区分训练域协议）
│   │   ├── group1_in_domain/           # 同域组预测框 (.txt) 与逐帧遥测 (.jsonl)
│   │   ├── group2_zero_shot/           # 零样本组预测框 (.txt) 与逐帧遥测 (.jsonl)
│   │   └── reference_static/           # 外部静态参考框（如 TAPTrack、官方50+基线）
│   ├── failure_diagnosis/              # 围点打援核心分析产物
│   │   ├── per_frame_telemetry/        # 统一格式逐帧记录 (IoU, 中心误差, 置信度, up_flag, bg_penalty)
│   │   ├── rescue_matrix/              # Type A/B/C 失败分类与跨算法救援矩阵
│   │   └── reports/                    # SA / SR / PR / 10属性对比表与曲线
├── colab/                              # Colab 一键初始化脚本与 Notebook
└── docs/                               # 实验协议契约、目录规范与环境变更日志

先扫描 Drive 中已有的 SiamDT、Anti-UAV410 数据、压缩包、权重与环境文件（包含旧版目录结构下的 `data/`、`weights/checkpoints/`、`archives/`）。优先通过软链接或登记表无损复用并检查 SHA-256 / CRC 完整性，避免重新下载或解压已有数据。不覆盖或删除原有文件；需要迁移大型文件时先确认。

### 3. 安装与持久化

- 从 GitHub 获取官方源码，锁定并验证 commit ID (`8a8eb04d976e9386b7c9c3ada5c85e5086013d52`)。
- 将数据集及必要权重持久化到 Drive，下载前核对来源、空间和已有文件。
- 运行代码与 GPU 虚拟环境部署在 Colab 本地高速盘 `/content/runtime/` 与 `/content/envs/`，不要直接运行 Drive 中可能因 FUSE I/O 变慢或软链接失效的虚拟环境。
- 将 `pip-cache`、`uv-cache`、预编译 `.whl`、环境依赖清单（`pip freeze`）和构建记录保存到 Drive 的 `environments/`。
- 针对 A100 (`sm_80`) 实测验证 PyTorch CUDA、MMCV CUDA 算子（`nms`, `roi_align` 前向与反向）及 SiamDT 245 个参数张量的 `strict=True` 加载。

### 4. 自动化与验证

制作可重复运行的 Colab 初始化入口，实现挂载 Drive、检查 A100 GPU、恢复工程、准备双轨环境、检查数据与权重、注入非侵入路径配置。

要求：
- 已完成步骤自动跳过，支持中断后重试（Idempotent）。
- 运行环境重建后能够利用 Drive Wheel 缓存在 1~2 分钟内快速恢复。
- 检查 `init_paths`、关键模块导入、MMCV CUDA 算子、数据读取和模型加载。
- 使用带时间戳与 Git Hash 的独立实验目录，避免不同算法或不同消融轮次的日志、预测框和 Checkpoint 相互覆盖。
- 本阶段只进行最小功能闭环测试（Smoke Test），不启动完整模型训练或大规模评测。

### 5. 交付

提供实际建立的目录树、数据与权重清单、环境版本及验证结果，以及 Colab 一键初始化与后续训练/多算法评测启动方法。
如果无法直接操作 Colab，请将需要执行的完整代码单元直接发在对话中，说明执行顺序，不要求额外下载部署脚本。
区分已完成、未验证和需要人工授权的步骤，不宣称未执行的操作已经成功。
```
