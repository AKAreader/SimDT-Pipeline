# Phase 0 双会话普查结果对比审查与总工合流裁定报告（2026-11-04）

- **对比对象 A（总工自研普查）**：[PHASE0_REPORT_20261104.md](file:///Z:/SimDT%E5%9B%B4%E7%82%B9%E6%89%93%E6%8F%B4/phase0_method_census/PHASE0_REPORT_20261104.md) 及机器可读账本 [PHASE0_METHOD_POOL_CENSUS_20261104.json](file:///Z:/SimDT%E5%9B%B4%E7%82%B9%E6%89%93%E6%8F%B4/phase0_method_census/PHASE0_METHOD_POOL_CENSUS_20261104.json)
- **对比对象 B（调研会话报告）**：[PHASE0_SOTA_SURVEY_REPORT.md](file:///Z:/SimDT%E5%9B%B4%E7%82%B9%E6%89%93%E6%8F%B4/PHASE0_SOTA_SURVEY_REPORT.md)

---

## 一、 两份报告的核心差异与互补性审计（`【事实】`）

### 1.1 高度共识项（双方独立核验完全一致的 `【事实】`）
1. **对原始 `SiamDT` (`SiamDTRCNN`) 代码结构的解构完全一致**：
   - 双方均基于 [siamdt_swin_tiny_sgd.py](file:///D:/CodexProjects/SiamDT_Project/code/Anti-UAV410/trackers/SiamDT/configs/siamdt_swin_tiny_sgd.py) 与 [siamdt_rcnn.py](file:///D:/CodexProjects/SiamDT_Project/code/Anti-UAV410/trackers/SiamDT/trackers/siamdt_rcnn.py) 锁定了 5 大机制：全图 `1333x800` 输入 + `Swin-Tiny` + 5 层 `FPN`、`Dual-Semantic RPN`（1000 提议）、`Versatile RCNN`（`tra_bboxes` 与 `det_bboxes` 合并截断 `Top_NUM = 5`）、11 个随机 `IoU <= 0` RPN 背景框线性减分（`_bbox_feats_bg`）、无运动门控的单状态 EMA 更新（`lr = 0.01`，更新时整批覆写 11 个背景框）。
2. **对 Anti-UAV410 评测协议风险的判断完全一致**：
   - 双方均指出 `SA` 公式在 `exist == 0`（目标消失帧）上的判分机制差异，以及 **In-Domain（域内重训练）** 与 **Zero-Shot（零样本跨域）** 必须分轨比较、严禁混排的铁律。
3. **共同验证通过的 2025–2026 开源有权重方法（10 个）**：
   - `UETrack` (CVPR 2026)、`UTPTrack` (CVPR 2026)、`SUTrack` (AAAI 2025)、`SPMTrack` (CVPR 2025)、`ORTrack` (CVPR 2025)、`SGLATrack` (CVPR 2025)、`AsymTrack` (AAAI 2025)、`MPT` (ICML 2025)、`DAM4SAM` (CVPR 2025 / IJCV 2026)、`MCITrack` (AAAI 2025) / `SAMURAI` (TIP 2026)。
4. **共同拦截的“伪开源 / 404 / 缺权重”陷阱（8 个）**：
   - `MCATrack` (ICCV 2025, HTTP 404)、`FARTrack` (ICLR 2026, HTTP 404)、`TGTrack` (CVPR 2026, 9字节空壳)、`TemTrack` (AAAI 2025, 仅结果无源码)、`LoRATv2` (NeurIPS 2025, coming soon 无代码无权重)、`DTPTrack` (CVPR 2026, 有源码但无训练好的跟踪器权重)。

---

### 1.2 总工自研普查独有发现、纠偏与深度优势（`【事实】`）

| 维度 | 调研会话（`PHASE0_SOTA_SURVEY_REPORT.md`）结论 | 总工自研普查（`PHASE0_REPORT_20261104.md`）核验事实与纠偏 | 科研影响裁定 |
| --- | --- | --- | --- |
| **1. `FocusTrack` (2025) 开源状态误判纠偏** | 列入 Reference Pool `R-05`，声称“arXiv 预印本（无公开源码）、权重未公开” | **总工实测纠正（`【事实】`）**：官方仓库 `https://github.com/vero1925/FocusTrack` 返回 **HTTP 200**，不仅完整开源 `lib/test/tracking/experiments` 代码，且在 Google Drive 与百度网盘（提取码 `1106`）公开了 **Anti-UAV410 域内训练权重**及同批重训练的 `OSTrack`/`ROMTrack`/`ZoomTrack`/`DropTrack` 权重！ | **重大纠偏**：`FocusTrack` 从 Reference 池正式提拔回 **Core Experimental Pool 第一梯队（In-Domain 轨道）** |
| **2. `MemLoTrack` (2025.11) 漏检补齐** | 未收录 | **总工独有核验（`【事实】`）**：`https://github.com/SeoulTech-HCIRLab/MemLoTrack` 返回 **HTTP 200**，代码完整，且 Google Drive 直接提供 **Anti-UAV410 专用权重（`retrained` 与 `scratch` 双版本）**，含 DINOv2+LoRA + 卡尔曼运动一致性门控 FIFO 记忆库 + MAL 损失 | **核心补齐**：补齐了 **In-Domain（Anti-UAV410 域内权重）** 最关键的记忆门控对比标杆 |
| **3. `SAMOSA` (2026.05) 漏检补齐** | 仅收录了前作 `SAMURAI` (卡尔曼线性滤波) | **总工独有核验（`【事实】`）**：`https://github.com/DurYi/SAMOSA` 返回 **HTTP 200**，仓库内直接内置高阶马尔可夫非线性运动预测器权重 `sam2/checkpoints/mp.pth` + 错误检测恢复模块（EDRM），针对无人机急停急转非线性机动 | **核心升级**：将 `SAMURAI` (线性卡尔曼) 与 `SAMOSA` (高阶马尔可夫 + EDRM) 组合为统一对照单元 |
| **4. 额外核验的 2025–2026 储备方法** | 未收录 `MambaLCT`、`SEATrack`、`DPTracker`、`TAPTrack`、`SATATrack` | **总工独有核验（`【事实】`）**：核实 `MambaLCT` (AAAI 2025)、`SEATrack` (CVPR 2026 Oral)、`DPTracker` (ICRA 2026) 代码与权重齐全；拦截 `TAPTrack` (仅网盘 txt 结果无源码) 与 `SATATrack` (HTTP 404) | 扩大了储备实验池，同时消除了 `TAPTrack`/`SATATrack` 的排期隐患 |
| **5. 与前期 `val90` 真实失分账本的定量挂钩** | 仅做源码静态结构阅读，未结合本项目前期 `val90` 逐帧失分统计与已证伪路线 | **总工深度绑定（`【事实】` + `【实验结论】`）**：将 5 大对比轴心直接锚定在 `val90`（`SA=70.3344%`）的 **19 段 `101+` 帧超长漂移（占零重叠失分 60.10%）**、**`<16 px` 极小目标 42.39% 零重叠率**、**180 帧候选池召回坍塌 vs 135 帧 Top-5 截断误杀**，以及前期已证伪的 4 条单点修补路线上 | 确保 Phase 0 选池不是泛泛的文献综述，而是精准服务于“围点打援”归因与救援矩阵 |

---

### 1.3 调研会话独有贡献与增量吸收（`【事实】`）

调研会话在以下 4 个方面提供了非常有价值的增量信息，总工予以采纳并合流：

1. **新增 2 个验证通过的 Core Pool 候选方法**：
   - **`LETrack` (CVPR 2026, `GXNU-ZhongLab/LETrack`)**：与 `SGLATrack` 同属广西师大钟必能团队，针对无人机低开销跨帧时序令牌传播，代码完整开源，提供百度网盘（提取码 `6n35`）与 Google Drive 权重，纯 PyTorch 实现。
   - **`ReFocus-TIR` (IEEE TNNLS 2024/2025, `laisimiao/ReFocus_TIR_Tracking`)**：专门针对热红外（TIR）无纹理目标导致的 ViT 自注意力弥散问题设计参数高效重聚焦模块，代码与百度网盘（提取码 `v27s`）/ Google Drive 权重齐全，且在 `CST-Anti-UAV` (ICCVW 2025, `PCwenyue/CST-Anti-UAV`) 中有 Anti-UAV410 对比基线。
2. **新增 1 个高价值跨任务开源 Reference 机制源**：
   - **`NS-FPN` (CVPR 2026, `mengduann/NS-FPN`)**：虽为单帧红外小目标检测分割（IRSTDS）方法（不可直接作为端到端跟踪器跑 OPE），但代码与权重 100% 开源，其 **低频引导特征净化（LFP，滤除浅层高频云层/地面热噪声）** 与 **螺旋感知特征采样（SFS）** 恰好对应 `SiamDT` 的 5 层 `FPN` 在 `<16 px` 极小目标上召回退化的痛点，是 Phase 5–6 诊断 FPN 特征信噪比的极佳跨任务参考。
3. **补充了 `SiamDT` 同脉络文献演化链（Reference Pool）**：
   - 梳理了 `StrongSiamTracker` (CVPRW 2025，全图小目标检测 + 局部孪生 + 轨迹引导空间约束提议抑制 SCPS)、`Enhanced-SiamDT` (Electronics 2026，多注意力 FPN `MA-FPN`)、`SiamDiff` (2025，尺度感知扩散候选框)、`DSATrack` (NeurIPS 2025，动态语义互相关，开源代码无跟踪器权重)。
4. **补充拦截了更多 2026 空壳/404 仓库**：
   - 核实 `EATrack` (`GXNU-ZhongLab/EATrack`, 0 commit)、`ETCTrack` (`PJD-WJ/ETCTrack`, 42字节 README)、`ARTrack-AC` (`MIV-XJTU/ARTrackAC`, 404)、`MamTrack` (`scy0712/MamTrack`, 仅结果无源码)。
5. **本地 GPU 硬件（RTX 5080 `sm_120` Blackwell）算子兼容性提示**：
   - 明确记录了本地显卡为 RTX 5080（16GB，`sm_120`），纯 PyTorch 2.x 算子方法可直接运行，而含自定义 CUDA 算子的方法（如 `MCITrack`/`MambaLCT` 的 `mamba-ssm`）需注意编译或 PyTorch fallback。

---

## 二、 总工合流裁定后的最终实验池（Merged Master Pool）

综合总工自研普查与调研会话成果，去伪存真后形成最终的 **Gate 0 合流实验池**：

### 2.1 第一批核心实验池（Tier-1 Core Experimental Pool：6 组标杆，覆盖五大诊断轴心 + 双评测轨道）

| 最终编号 | 方法名称 | 年份/出处 | 评测轨道 (`Track`) | 对应 SiamDT 诊断轴心 | 权重与代码核验状态 (`【事实】`) | 来源归属 |
| --- | --- | --- | --- | --- | --- | --- |
| **Core-1** | **FocusTrack** *(含同仓重训 OSTrack / ROMTrack / ZoomTrack / DropTrack)* | 2025.04 | **In-Domain** (`Anti-UAV410` 原生权重) | **A1｜搜索视野与小目标保持**（自适应局部缩放 SRA + 注意力掩码 ATM vs. 全图 1000 提议） | `vero1925/FocusTrack` 代码完整 + 百度网盘(`1106`)/GDrive 权重 | **总工独有核验（纠正调研会话误判）** |
| **Core-2** | **MemLoTrack** | 2025.11 | **In-Domain** (`Anti-UAV410` 原生权重) | **A2｜模板记忆更新与抗污染门控**（DINOv2+LoRA + 卡尔曼运动一致性门控 FIFO 记忆库 + MAL vs. 单状态 EMA） | `SeoulTech-HCIRLab/MemLoTrack` 代码完整 + GDrive `Anti-UAV410` 权重 | **总工独有核验** |
| **Core-3** | **DAM4SAM** | CVPR 2025 / IJCV 2026 | **Zero-Shot** (免训练基础模型) | **A3｜背景干扰物抑制与记忆解耦**（RAM 近期外观 + DRM 分心物记忆解耦 + 竞争不确定性冻结 vs. 11 个随机 RPN 背景框线性减分） | `jovanavidenovic/DAM4SAM` 代码完整 + `SAM 2.1` 官方权重 (`run_on_box_dataset.py`) | **双方共识** |
| **Core-4** | **SAMOSA + SAMURAI** | 2026.05 / TIP 2026 | **Zero-Shot** (免训练基础模型) | **A4｜非线性运动约束与错误恢复**（高阶马尔可夫/卡尔曼运动预测 + EDRM 错误检测恢复 vs. 零运动约束全图跳变） | `DurYi/SAMOSA` (`mp.pth`) + `yangchris11/samurai` 代码与 `SAM 2.1` 权重完整 | **总工(`SAMOSA`) + 双方(`SAMURAI`)合流** |
| **Core-5** | **UETrack** *(配 `LoRAT` / `SPMTrack` 对照)* | CVPR 2026 / CVPR 2025 | **Zero-Shot & Fine-tunable** | **A5｜跨帧上下文与混合专家路由**（相似度驱动令牌池化 TP-MoE + 目标感知蒸馏 TAD vs. 离散 7x7 RoI 互相关） | `kangben258/UETrack` (HuggingFace) + `LitingLin/LoRAT` / `WenRuiCai/SPMTrack` (GDrive) | **双方共识** |
| **Core-6** | **ReFocus-TIR + SUTrack / UTPTrack-S** | TNNLS 2025 / AAAI 2025 / CVPR 2026 | **TIR-Specific & Multi-Modal** | **A1/A2 跨模态补充｜热红外注意力重聚焦与模板背景令牌剪枝** | `laisimiao/ReFocus_TIR_Tracking` (`v27s`) + `chenxin-dlut/SUTrack` (`jqpb`) + `EIT-NLP/UTPTrack` (HF) | **调研会话(`ReFocus-TIR`) + 双方(`SUTrack`/`UTPTrack`)合流** |

### 2.2 第二批储备核心池（Tier-2 Reserve Core Pool：均已核验代码与权重齐全，按需调用）
- **双流孪生调制对照**：`AsymTrack` (AAAI 2025, `jiawen-zhu/AsymTrack`, 提取码 `asym`)
- **UAV 抗遮挡/层自适应/低成本时序家族**：`ORTrack` (CVPR 2025, `cvpr`)、`SGLATrack` (CVPR 2025)、`LETrack` (CVPR 2026, `6n35`)
- **显式长程运动提示对照**：`MPT` (ICML 2025, `zj5559/Motion-Prompt-Tracking`, `8j2g`)
- **Mamba 状态空间上下文家族**：`MCITrack` (AAAI 2025)、`MambaLCT` (AAAI 2025)
- **测试时训练/动态提示家族**：`SEATrack` (CVPR 2026 Oral)、`DPTracker` (ICRA 2026)

### 2.3 参考机制池（Reference Pool：严禁混用论文数字，仅作 Phase 5–6 机制提取与负面备案）
- **跨任务开源五星级机制参考（有代码有权重，任务为单帧 IRSTDS）**：`NS-FPN` (CVPR 2026, `mengduann/NS-FPN`：LFP 低频引导净化高频热噪 + SFS 螺旋采样)
- **开源代码但缺跟踪器权重**：`DTPTrack` (CVPR 2026, `NorahGreen/DTPTrack`)、`DSATrack` (NeurIPS 2025, `zxyyxzz/DSATrack`)
- **高相关未开源文献机制**：`MCATrack` (ICCV 2025, 404)、`StrongSiamTracker` (CVPRW 2025)、`Enhanced-SiamDT` (Electronics 2026)、`SiamDiff` (2025)、`FFTR` (TCSVT 2025)、`ASMTrack` (TITS 2025)
- **已拦截空壳/404 负面清单**：`TAPTrack`、`LoRATv2`、`TGTrack`、`TemTrack`、`MamTrack`、`EATrack`、`ETCTrack`、`SATATrack`、`FARTrack`、`ARTrack-AC`
