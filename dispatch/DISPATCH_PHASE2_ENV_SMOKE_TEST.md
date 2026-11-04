# 【总控派工单 B】Phase 2 前置基建｜Tier-1 核心实验池环境就位、统一接口封装与 5 帧冒烟测试任务书

## 1. 会话角色与权限边界
- **你的角色**：**Tier-1 核心实验池环境与统一评测接口工程师（Subsession-B）**。
- **上级总控**：项目总工（Master Control Session，工作区 `Z:\SimDT围点打援`）。
- **核心铁律（Gate 1 前置边界）**：
  1. **只做基建与 5 帧冒烟测试（Smoke Test），严禁提前跑大规模评测**！在总控会话正式批准 **Gate 1（冻结诊断子集）** 之前，你只负责把代码、环境、权重和统一逐帧输出包装器（Wrapper）调通，并在 **单条测试序列的前 5 帧** 上验证输出格式正确。
  2. **不碰师兄方法**：当前核心基线是**原始 `SiamDT` (`SiamDTRCNN`)**，严禁运行师兄的 `SiamISDT` / `auto_SCLT_menchan-new-cuda2`。
  3. **严禁污染上游只读目录**：对 `E:\PHD\YB` 与 `D:\CodexProjects\SiamDT_Project` 保持**只读（Read-Only）**，所有克隆的第三方代码、下载的权重与封装脚本统一存放在 `Z:\SimDT围点打援\phase2_unified_eval\`（或对应子目录 `repos/`、`checkpoints/`、`wrappers/`）。

---

## 2. 需要就位的模型清单（1 个原始基线 + Tier-1 核心实验池）
请按优先级依次完成以下模型的仓库拉取、权重就位与环境核验（详见 `Z:\SimDT围点打援\phase0_method_census\PHASE0_COMPARISON_AND_MERGED_POOL_20261104.md`）：

1. **【Baseline】原始 `SiamDT` (`SiamDTRCNN`)**
   - 源码参考：`D:\CodexProjects\SiamDT_Project\code\Anti-UAV410\trackers\SiamDT` 或 `E:\PHD\YB\Anti_yb\trackers\SiamDT`
   - 现有可用环境核验：检查本机 `siamdt` Conda 环境及原始官方权重路径。
2. **【Core-1｜轴心 A1】`FocusTrack`（2025）**
   - 仓库：`https://github.com/vero1925/FocusTrack`
   - 权重：Google Drive / 百度网盘（提取码 `1106`，优先获取 `Anti-UAV410` 域内权重）。
3. **【Core-2｜轴心 A2】`MemLoTrack`（2025.11）**
   - 仓库：`https://github.com/SeoulTech-HCIRLab/MemLoTrack`
   - 权重：Google Drive（`Anti-UAV410` `retrained` / `scratch` 权重）。
4. **【Core-3｜轴心 A3】`DAM4SAM`（CVPR 2025 / IJCV 2026）**
   - 仓库：`https://github.com/jovanavidenovic/DAM4SAM`
   - 权重：`SAM 2.1` 官方公開权重（`sam2.1_hiera_base_plus.pt` / `large.pt`）。
5. **【Core-4｜轴心 A4】`SAMOSA + SAMURAI`（2026.05 / TIP 2026）**
   - 仓库：`https://github.com/DurYi/SAMOSA` 及 `https://github.com/yangchris11/samurai`
   - 权重：复用同一套 `SAM 2.1` 权重 + `SAMOSA` 仓库内置 `sam2/checkpoints/mp.pth`。
6. **【Core-5｜轴心 A5】`UETrack`（CVPR 2026，配 `LoRAT` / `SPMTrack`）**
   - 仓库：`https://github.com/kangben258/UETrack`
   - 权重：HuggingFace `https://huggingface.co/kangben258/UETrack`。
7. **【Core-6｜红外/多模态补充】`ReFocus-TIR` / `SUTrack` / `UTPTrack-S`**
   - 仓库：`https://github.com/laisimiao/ReFocus_TIR_Tracking`、`https://github.com/chenxin-dlut/SUTrack`、`https://github.com/EIT-NLP/UTPTrack`
   - 权重：HuggingFace `xche32/SUTrack`、`HarrisonWu/UTPTrack` 等。

*(注：若个别百度网盘或 Google Drive 链接在命令行自动下载受阻，请明确记录具体 URL、文件 ID 与提取码，生成清单一并汇报总控会话。)*

---

## 3. 统一逐帧中间结果输出契约（Unified Frame-Level Telemetry Schema）
为支持 Phase 2–3 的 **Failure Episode 提取** 与 **Rescue Matrix 计算**，所有被测跟踪器的推理 Wrapper 必须统一输出满足以下字段的逐序列 JSONL / Parquet 记录：
- `seq_id`：序列名称
- `frame_id`：帧序号（0-indexed）
- `exist`：GT 目标是否存在（`1` 或 `0`）
- `gt_bbox`：`[x, y, w, h]`（当 `exist == 0` 时记录原始标注）
- `pred_bbox`：`[x, y, w, h]`（统一 4 元组格式）
- `pred_absent_flag`：模型是否判定目标消失（`0` 或 `1`，与 `pred_bbox` 解耦记录以备阈值敏感性分析）
- `iou`：当前帧预测框与 GT 框的 IoU（`exist == 1` 时计算）
- `sa_score`：按 Anti-UAV410 官方公式计算的当帧 `SA` 得分
- `center_error`：中心点欧氏距离误差（像素）
- `norm_center_error`：相对 GT 尺寸归一化的中心误差
- `target_size`：$\sqrt{w_{\text{gt}} \cdot h_{\text{gt}}}$
- `confidence`：模型当前帧最高响应得分 / 置信度（若模型可导出）
- `template_updated`：当前帧是否触发了模板/记忆库更新（`true`/`false`，若模型支持）
- `latency_ms`：单帧推理耗时（毫秒）

---

## 4. 必须产出的交付物
请在 `Z:\SimDT围点打援\phase2_unified_eval\` 下生成：
1. **`ENV_AND_SMOKE_TEST_REPORT.md`**：记录每个候选模型的代码拉取状态、权重就位状态、RTX 5080 环境兼容性实测结果，以及 5 帧冒烟测试通过情况；
2. **`SMOKE_TEST_5FRAMES_VERIFICATION.json`**：在 1 条序列前 5 帧上跑出的标准 Schema 样板数据；
3. 完成冒烟测试后**立即停止**，等待总控会话完成 Gate 1 审查后再下达全诊断子集运行指令。
