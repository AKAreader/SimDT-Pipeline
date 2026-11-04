# Phase 2 失败判定标准与三层失败提取协议（Pre-registered & Frozen Protocol）

- **文档版本**：`v1.1-phase2-protocol-rev20261104`（在 `v1.0-phase2-protocol-frozen` 基础上修订；第一、二章及第三章 3.2 步骤 4 的单算法状态分级阈值**未改动**）
- **编制单元**：`Subsession-Phase2`（Phase 2 统一评测协议、失败事件提取器与云端资产预置工程师）
- **生成时间**：`2026-11-04 18:50:00 +08:00`；**修订时间**：`2026-11-04`（见下方修订记录）
- **适用范围**：`Anti-UAV410` 代表性诊断子集（`Core 40`: `24 Discovery + 16 Holdout`）及全量验证集（`Full Val90`: `54 Discovery + 36 Holdout`）上的所有被测跟踪算法（`Original SiamDT Baseline` + Group I 同域训练组 + Group II 零样本/基础模型组）
- **主口径（2026-11-04 确定）**：Phase 2–5 主口径为 **Core 40（Discovery 24 / Holdout 16）**，Holdout 结果必须封存；Full Val90 保留为扩展验证用。
- **科研铁律**：本协议在正式跨算法失败统计（Phase 3/4）启动前**预先定义并严格冻结**。对所有被测算法执行 **100% 相同的阈值、桥接规则与窗口对齐准则**，严禁针对单一算法调整失败判定标准。

> [!IMPORTANT]
> **修订记录 v1.1（2026-11-04，冻结）**
> - **修订原因**：Claude × Gemini 交叉审查（`review-think-20261104-001`，P0-7）指出 v1.0 的"统一挑战时间窗"由全部参赛算法失败段取并集，(1) 每新增一个算法窗口边界就会漂移；(2) 同域组与零样本组被混在一起计算存活率，零样本算法大面积失败会把"SiamDT 独有短板"（Type C）误判为其他类型，误导创新方向；另外 v1.0 缺少 Holdout 封存的代码护栏。
> - **修订内容**：① 主口径改为 **Subject-Anchored Windows**（窗口边界仅由 `SiamDT_Baseline` 自身失败 episode 决定），原 Pool-Union 窗口降为补充视图并记录 `tracker_set` 及其 SHA-256；② 存活率按 **同域 / 零样本分轨** 计算，Type A/B/C/D 按分轨规则重写（新增 D0/E/U/X）；③ 新增 **Holdout 封存护栏**（默认拒绝读取，`--unlock-holdout` 解封且输出目录分开）。
> - **未改动**：逐帧指标、四态分类、τ_fail=0.10、敏感性集合、g_bridge=3、前导 15 帧、恢复判定（K_rec=5, IoU≥0.50）、窗口上下文 [-15, +10]、合并间隔 5 帧、单算法窗口状态分级阈值（SUCCESS / ROBUST_SURVIVAL / TRANSIENT_FAILURE_RECOVERED / FAILED_UNRECOVERED）。

---

## 一、 统一输入契约与逐帧遥测（Per-Frame Telemetry）标准格式

### 1.1 统一评测运行边界（One-Pass Evaluation, OPE）
1. **零重置初始化（Zero-Reset OPE）**：所有算法均在序列首帧（`frame_id = 0`）使用完全相同的官方物理真值框 $\text{GT}_0 = [x_0, y_0, w_0, h_0]$ 初始化，从 `frame_id = 1` 至序列末帧连续推理，**严禁在序列中间任何帧读取真值进行重初始化或状态重置**。
2. **训练域严格分轨（Domain Track Separation）**：
   - **`group1_in_domain`（同域组）**：使用 `Anti-UAV410` 训练集训练或微调的模型（`SiamDT_Baseline`、`FocusTrack` 及同库重训模型、`MemLoTrack` 的 `scratch` 与 `retrained` 版本）。
   - **`group2_zero_shot`（零样本/跨域组）**：未在 `Anti-UAV410` 训练集上微调的基础模型与通用/热红外预训练模型（`DAM4SAM`、`SAMOSA`、`SAMURAI`、`UETrack`、`SUTrack`、`UTPTrack-S`、`ReFocus-TIR`）。
3. **Discovery / Holdout 防泄漏硬隔离**：
   - `discovery_set.json`（`24` 条核心序列，源自 `22` 个独立录制组，共 `27,791` 帧）：用于 Phase 2–5 逐帧失败提取、跨算法救援矩阵（Rescue Matrix）构建、风险因子回归与内部特征诊断；
   - `holdout_verification.json`（`16` 条核心序列，源自 `16` 个独立录制组，共 `17,937` 帧）：仅用于验证机理规律与 Phase 6 模块改进的泛化性，**严禁利用 Holdout 序列反向定制阈值或超参数**。

### 1.2 逐帧基础几何指标定义
设视频序列总帧数为 $T$，帧索引 $t \in \{0, 1, \dots, T-1\}$。第 $t$ 帧的真值存在标志为 $v_t \in \{0, 1\}$（对应 `IR_label.json` 中的 `exist` 字段），真值边界框为 $B_t^{\text{gt}} = [x_t^{\text{gt}}, y_t^{\text{gt}}, w_t^{\text{gt}}, h_t^{\text{gt}}]$；被测跟踪器输出的预测边界框为 $B_t^{\text{pred}} = [x_t^{\text{pred}}, y_t^{\text{pred}}, w_t^{\text{pred}}, h_t^{\text{pred}}]$，预测消失标志为 $p_t \in \{0, 1\}$（当模型显式预测目标消失或输出 $w_t^{\text{pred}} \le 0 \lor h_t^{\text{pred}} \le 0$ 时 $p_t = 1$，否则 $p_t = 0$）。

1. **逐帧交并比（Intersection-over-Union, $\text{IoU}_t$）**：
   - 当 $v_t = 1$（目标存在且 $w_t^{\text{gt}} > 0, h_t^{\text{gt}} > 0$）且 $p_t = 0$ 时：
     $$\text{IoU}_t = \frac{\text{Area}(B_t^{\text{pred}} \cap B_t^{\text{gt}})}{\text{Area}(B_t^{\text{pred}} \cup B_t^{\text{gt}})} \in [0, 1]$$
   - 当 $v_t = 1$ 但模型漏检预测为空（$p_t = 1$）时，$\text{IoU}_t = 0.0$。
2. **官方状态精度得分（State Accuracy, $\text{SA}_t$）**：
   遵循 `Anti-UAV410` 官方评测公式：
   $$\text{SA}_t = \text{IoU}_t \cdot \delta(v_t > 0) + p_t \cdot \big(1 - \delta(v_t > 0)\big)$$
   其中：
   - 当 $v_t = 1$（目标存在）时，$\text{SA}_t = \text{IoU}_t$；
   - 当 $v_t = 0$（目标消失）时，若模型正确输出空框（$p_t = 1$），则 $\text{SA}_t = 1.0$；若模型仍输出非空实框（$p_t = 0$，即原始 `SiamDT` 等密集框输出模型），则 $\text{SA}_t = 0.0$。
   - **官方分母对齐**：对于极少数虽标注 $v_t = 1$ 但真值宽高异常（$w_t^{\text{gt}} \le 0 \lor h_t^{\text{gt}} \le 0$）的帧，严格按官方 `eval_validation.py` 口径不计入序列 $\text{SA}$ 分母（对应 `sa_denominator`）。
3. **中心定位误差（$\text{CE}_t$）与尺度归一化中心误差（$\text{NCE}_t$）**：
   当 $v_t = 1$ 且 $w_t^{\text{gt}} > 0, h_t^{\text{gt}} > 0$ 时，定义目标等效几何尺度 $s_t = \sqrt{w_t^{\text{gt}} \cdot h_t^{\text{gt}}}$：
   $$\text{CE}_t = \sqrt{\left(c_{x,t}^{\text{pred}} - c_{x,t}^{\text{gt}}\right)^2 + \left(c_{y,t}^{\text{pred}} - c_{y,t}^{\text{gt}}\right)^2}, \quad \text{NCE}_t = \frac{\text{CE}_t}{s_t}$$
   其中 $(c_{x,t}, c_{y,t}) = (x_t + 0.5 w_t, y_t + 0.5 h_t)$。若 $v_t = 1$ 且 $p_t = 1$（预测空框），记 $\text{CE}_t = +\infty, \text{NCE}_t = +\infty$（在 `.jsonl` 中存为 `-1.0` 哨兵值，并在统计判定中视为超出任何有限误差阈值）。

### 1.3 统一 `.jsonl` 逐帧遥测文件 Schema
每个算法在每条序列上生成 `<seq_id>.jsonl`，每行对应一帧，强制包含以下核心字段（`SiamDT` 额外附加内部诊断字段 `tra_score`, `det_score`, `bg_max_penalty`）：

```json
{
  "seq_id": "03_7951_1500-2999",
  "frame_id": 1,
  "exist": 1,
  "gt_bbox": [312.0, 244.0, 14.0, 11.0],
  "pred_bbox": [311.62, 243.85, 14.21, 11.15],
  "pred_absent_flag": 0,
  "iou": 0.914285,
  "sa_score": 0.914285,
  "center_error": 0.4123,
  "norm_center_error": 0.0332,
  "target_size": 12.4097,
  "confidence": 0.9642,
  "template_updated": true,
  "latency_ms": 28.415
}
```

---

## 二、 三层失败定义体系（Sequence $\to$ Frame $\to$ Failure Episode）

### 2.1 第一层（Level 1｜Sequence）：序列级性能与失分归因指标
对每条序列 $s$，定义有效存在帧集合 $T_{\text{exist}}(s) = \{t \in [0, T_s-1] \mid v_t = 1 \land w_t^{\text{gt}} > 0 \land h_t^{\text{gt}} > 0\}$，目标消失帧集合 $T_{\text{absent}}(s) = \{t \in [0, T_s-1] \mid v_t = 0\}$，官方有效分母帧集合 $T_{\text{sa}}(s) = T_{\text{exist}}(s) \cup T_{\text{absent}}(s)$：

1. **序列官方状态精度（$\text{SA}_s$）**：$\text{SA}_s = \frac{1}{|T_{\text{sa}}(s)|} \sum_{t \in T_{\text{sa}}(s)} \text{SA}_t$。
2. **存在帧平均重叠率（$\text{MeanIoU}_{\text{exist}, s}$）**：$\frac{1}{|T_{\text{exist}}(s)|} \sum_{t \in T_{\text{exist}}(s)} \text{IoU}_t$。
3. **序列失败帧率（$\text{FR}_s(\tau)$）**：在给定阈值 $\tau$ 下存在帧中的失败比例：
   $$\text{FR}_s(\tau) = \frac{1}{|T_{\text{exist}}(s)|} \sum_{t \in T_{\text{exist}}(s)} \mathbb{I}\big(\text{IoU}_t < \tau\big)$$
   特别地，当 $\tau = 0.0$ 时，定义**零重叠丢失率（Zero-Overlap Rate, $\text{ZOR}_s$）**：$\text{ZOR}_s = \frac{1}{|T_{\text{exist}}(s)|} \sum_{t \in T_{\text{exist}}(s)} \mathbb{I}(\text{IoU}_t = 0.0)$。
4. **中心精度（$\text{PR}_{20\text{px}, s}$）与归一化中心精度（$\text{NPR}_{0.5, s}$）**：存在帧中满足 $\text{CE}_t \le 20\text{ px}$ 与 $\text{NCE}_t \le 0.5$ 的帧占比。
5. **消失期误报率（Absent False Positive Rate, $\text{AFPR}_s$）**：当 $|T_{\text{absent}}(s)| > 0$ 时，$\text{AFPR}_s = \frac{1}{|T_{\text{absent}}(s)|} \sum_{t \in T_{\text{absent}}(s)} \mathbb{I}(p_t = 0)$。
6. **序列失分预算拆解（SA Loss Decomposition）**：将每条序列相对于满分 $1.0$ 的总失分 $1 - \text{SA}_s$ 精确分解为四项之和（严守可加性恒等式）：
   $$1 - \text{SA}_s = \underbrace{\Delta \text{SA}_{\text{fail}}(\text{IoU} < 0.10)}_{\text{失败帧失分}} + \underbrace{\Delta \text{SA}_{\text{imprecise}}(0.10 \le \text{IoU} < 0.50)}_{\text{粗糙定位失分}} + \underbrace{\Delta \text{SA}_{\text{high}}(\text{IoU} \ge 0.50)}_{\text{高质量锁定残差}} + \underbrace{\Delta \text{SA}_{\text{absent}}(v_t = 0)}_{\text{消失期误报失分}}$$

---

### 2.2 第二层（Level 2｜Frame）：逐帧状态分类与预注册敏感性阈值

#### A. 当 `exist == 1`（目标存在帧，$t \in T_{\text{exist}}$）时：
将每一帧严格划分为以下互斥的四级定位状态，同时冻结主失败判定标准与预注册敏感性集合：

| 状态代码 | 状态名称 | 数学判定准则（主标准冻结） | 物理与诊断含义 |
| :--- | :--- | :--- | :--- |
| **`ZERO_OVERLAP_LOST`** | **完全丢失帧** | $\text{IoU}_t = 0.0$（或 $p_t = 1$） | 预测框与真值框完全无交集（零重叠漂移或漏检） |
| **`FAILURE_FRAME`** | **主失败判定帧** *(包含 `ZERO_OVERLAP_LOST`)* | **$\text{IoU}_t < \tau_{\text{fail}}$，主阈值冻结为 $\tau_{\text{fail}} = 0.10$** | 预测框已脱离目标主体（重叠不足 $10\%$），处于跟丢或擦边失锁状态 |
| **`IMPRECISE_LOCALIZATION`** | **定位粗糙帧** | $0.10 \le \text{IoU}_t < 0.50$ | 预测框仍覆盖部分目标或近邻区域，但存在明显尺度膨胀/收缩或中心偏移 |
| **`HIGH_QUALITY_LOCK`** | **高质量锁定帧** | $\text{IoU}_t \ge 0.50$ | 预测框紧密锁定目标，符合标准视觉跟踪成功准则（$\text{OP}_{0.50}$） |

> [!IMPORTANT]
> **预注册敏感性阈值集合（Pre-registered Sensitivity Thresholds）**：
> 为防止审稿人质疑“结论是否仅在 $\tau_{\text{fail}} = 0.10$ 下成立”，提取器在冻结主阈值 $\tau_{\text{fail}} = 0.10$ 的同时，强制同步计算并输出以下 **5 组预注册敏感性准则** 下的失败帧率与事件段统计：
> 1. **$\tau_0 = 0.0$（严格零重叠准则）**：$\text{IoU}_t = 0.0$；
> 2. **$\tau_1 = 0.10$（主失败准则，Primary Frozen）**：$\text{IoU}_t < 0.10$；
> 3. **$\tau_2 = 0.30$（宽松重叠失败准则）**：$\text{IoU}_t < 0.30$；
> 4. **$\tau_3 = 0.50$（严格成功反转准则）**：$\text{IoU}_t < 0.50$；
> 5. **$\eta_{\text{NCE}} = 1.0$（尺度归一化中心误差准则）**：$\text{NCE}_t > 1.0$（中心偏移超过目标等效尺寸 $\sqrt{w\cdot h}$ 的 $100\%$），以及复合准则 $(\text{IoU}_t < 0.10) \lor (\text{NCE}_t > 1.0)$。

#### B. 当 `exist == 0`（目标消失帧，$t \in T_{\text{absent}}$）时：
1. **消失期状态分类**：
   - **`false_positive_in_absent`（消失期误报帧）**：$v_t = 0 \land p_t = 0$（模型未输出空框，在背景中盲目预测实框，得分为 $\text{SA}_t = 0$，且若此时发生模板更新则会直接把纯背景写入记忆库）。
   - **`true_negative_in_absent`（消失期正确拒识帧）**：$v_t = 0 \land p_t = 1$（模型正确输出空框，得分为 $\text{SA}_t = 1$）。
2. **消失重入后的重捕获延迟（Post-Absence Re-acquisition Latency, $L_{\text{reacq}}$）**：
   - 对于序列中任意一段连续消失区间 $[t_{\text{abs\_start}}, t_{\text{abs\_end}}]$（期间恒有 $v_t = 0$），若目标在 $t_{\text{reentry}} = t_{\text{abs\_end}} + 1$ 重新进入视野（$v_{t_{\text{reentry}}} = 1$）：
   - 定义**重捕获稳定锁定条件**：从某帧 $t_{\text{lock}} \ge t_{\text{reentry}}$ 起，跟踪器在后续连续 $K_{\text{rec}} = 5$ 个有效存在帧上均满足高质量锁定（$\text{IoU} \ge 0.50$）（若距离下一次消失或序列结束不足 $5$ 帧，则要求剩余全部存在帧均满足 $\text{IoU} \ge 0.50$）。
   - 定义 **Re-acquisition Latency（重捕获延迟帧数）**：
     $$L_{\text{reacq}} = t_{\text{lock}} - t_{\text{reentry}}$$
     - 若目标重入首帧 $t_{\text{reentry}}$ 即开始连续 $5$ 帧稳定锁定，则 $L_{\text{reacq}} = 0$（瞬时重捕获）；
     - 若在下一次目标消失或序列结束前始终未能达成连续 $K_{\text{rec}} = 5$ 帧 $\text{IoU} \ge 0.50$，则标记 `reacquired = False`，并记录截尾延迟 $L_{\text{reacq}} = T_{\text{avail}}$（重入后永久失锁）。

---

### 2.3 第三层（Level 3｜Failure Episode）：失败事件段聚合与恢复判定协议

单帧统计会掩盖“偶发 1 帧抖动”与“连续 300 帧不可逆漂移”的本质差异。为此，本协议将逐帧异常聚合成具有完整时间上下文的 **失败事件段（Failure Episode）**：

#### 1. 瞬时波动桥接规则（Temporal Gap Bridging, $g_{\text{bridge}} = 3$）
- 给定某序列上的二值失败帧指示序列 $f_t = \mathbb{I}(\text{IoU}_t < \tau_{\text{fail}})$（仅在有效存在帧 $t \in T_{\text{exist}}$ 上定义，主标准 $\tau_{\text{fail}} = 0.10$）：
- 首先提取所有连续满足 $f_t = 1$ 的初始失败片段 $\{[a_k, b_k]\}_{k=1}^M$（排除首帧 $t=0$ 初始化帧）。
- **桥接条件**：若相邻两个失败片段 $[a_k, b_k]$ 与 $[a_{k+1}, b_{k+1}]$ 之间夹隔的非失败存在帧数满足：
  $$N_{\text{non\_fail\_present}}(b_k, a_{k+1}) = \big|\{t \in (b_k, a_{k+1}) \cap T_{\text{exist}} \mid f_t = 0\}\big| \le g_{\text{bridge}} = 3$$
  则将该短暂间隙视为漂移过程中的瞬时偶发擦边波动（Transient Accidental Overlap），将两个片段合并为一个完整失败事件段 $[a_k, b_{k+1}]$。
- **跨消失期关联标注**：若两个失败片段之间夹隔了真值消失区间（$v_t = 0$），提取器不仅保留各自的物理帧边界，同时记录事件内包含的 `absent_frames_inside` 以及是否为“消失引发的重入失锁（`triggered_by_absence_reentry`）”。

#### 2. 事件前导上下文窗口（Pre-failure Context Window, $W_{\text{pre}} = 15$）
对每一个聚合后的失败事件段 $E_i = [t_{\text{start}}, t_{\text{end}}]$，强制提取其前导时间窗：
$$\text{PreContext}(E_i) = \big[\max(1, t_{\text{start}} - 15),\; t_{\text{start}} - 1\big]$$
并在该前导窗口内自动统计：
- `pre_mean_iou` / `pre_min_iou`：失锁前 15 帧的平均/最低 IoU（区分“突发跳变失锁 Sudden Jump”与“渐进腐蚀漂移 Gradual Erosion”）；
- `pre_iou_slope`：失锁前 IoU 的线性退化斜率；
- `pre_template_updates_count`：失锁前 15 帧内触发模板更新（`template_updated == True`）的次数及最后一次更新时的真实 IoU（`last_pre_update_iou`，用于捕捉“低质量帧误触发更新污染模板”的直接证据）；
- `pre_mean_target_size`：失锁前的目标平均等效尺寸 $\sqrt{w \cdot h}$。

#### 3. 失败事件持续时长分桶（Episode Duration Buckets）
定义事件段有效存在帧跨度 $L_{\text{ep}} = \big|[t_{\text{start}}, t_{\text{end}}] \cap T_{\text{exist}}\big|$ 及日历帧跨度 $L_{\text{span}} = t_{\text{end}} - t_{\text{start}} + 1$，按 $L_{\text{ep}}$ 划分为 4 个标准档位：
- **`Bucket_1_5_Transient`（瞬时抖动）**：$1 \le L_{\text{ep}} \le 5$ 帧；
- **`Bucket_6_20_Short`（短程失锁）**：$6 \le L_{\text{ep}} \le 20$ 帧；
- **`Bucket_21_100_Medium`（中程漂移）**：$21 \le L_{\text{ep}} \le 100$ 帧；
- **`Bucket_101_Plus_Long_Drift`（超长程漂移）**：$L_{\text{ep}} \ge 101$ 帧。

#### 4. 恢复判定标准（Recovered vs. Unrecovered Terminal Drift）
在失败事件结束帧 $t_{\text{end}}$ 之后，检查后续有效存在帧 $t \in (t_{\text{end}}, T_s - 1] \cap T_{\text{exist}}$：
1. **`recovered`（成功恢复稳定锁定）**：
   - 在 $t_{\text{end}}$ 之后（且在下一个独立失败事件爆发前，或在 $t_{\text{end}}$ 之后整体后续序列中），存在起始帧 $t_{\text{rec}} > t_{\text{end}}$ 使得跟踪器连续 $K_{\text{rec}} = 5$ 个有效存在帧满足 $\text{IoU}_t \ge 0.50$（若 $t_{\text{end}}$ 距序列末尾剩余有效存在帧数 $R \in [1, 4]$，且这 $R$ 帧全部满足 $\text{IoU}_t \ge 0.50$，亦判定为 `recovered`）。
   - 同时记录 `recovery_latency_frames` $= t_{\text{rec}} - t_{\text{end}} - 1$（从脱离失败阈值 $\ge 0.10$ 到重新进入高质量稳定锁定 $\ge 0.50$ 所需的过渡帧数）。
2. **`partial_recovery_unstable`（仅恢复至粗糙定位，未达稳定高质锁定）**：
   - $t_{\text{end}} < t_{\text{last\_present}}$，后续帧虽脱离了 $\text{IoU} < 0.10$，但在序列结束或下一次彻底失锁前始终徘徊于 $0.10 \le \text{IoU}_t < 0.50$，未能达成连续 $5$ 帧 $\text{IoU} \ge 0.50$。
3. **`unrecovered_terminal_drift`（不可逆终局漂移）**：
   - $t_{\text{end}}$ 直接延伸至序列最后一个有效存在帧 $t_{\text{last\_present}}$，或自 $t_{\text{start}}$ 起直至序列结束再未达成连续 $K_{\text{rec}} = 5$ 帧 $\text{IoU} \ge 0.50$。

---

## 三、 挑战时间窗协议（v1.1：主口径 Subject-Anchored Windows + 补充视图 Pool-Union）

### 3.0 v1.1 修订：主口径 Subject-Anchored Windows（2026-11-04 冻结）
1. **窗口边界（主口径）**：对序列 $s$，种子区间**仅**取 `SiamDT_Baseline` 自身的主失败事件段 $\{[t_{\text{start}}^{(k)}, t_{\text{end}}^{(k)}]\}$（$\tau_{\text{fail}}=0.10, g_{\text{bridge}}=3$）；相邻种子间隔 $\le G_{\text{canon}}=5$ 帧合并为核心跨度 $[u_{\text{start}}, u_{\text{end}}]$，再附加 $[\max(1,u_{\text{start}}-15), \min(T_s-1,u_{\text{end}}+10)]$ 上下文。窗口边界**与参与算法集合无关**，后加入的算法不改变已锁定窗口；输出 `window_set_sha256`（窗口 ID 与跨度的哈希）供核验。GT 消失重入区间不作为主口径种子（其仍出现在补充视图中）。
2. **补充视图 Pool-Union**：保留 v1.0 的 3.2 四步法（全部算法失败段 ∪ GT 消失重入区间），输出文件中写入 `tracker_set`（算法名 + 分组）及 `tracker_set_sha256`，仅作补充分析，不作为 Type 统计主口径。
3. **单算法状态分级**：沿用 3.2 步骤 4 的 4 级状态与阈值（未改动）。记 *survived* = {`SUCCESS`, `ROBUST_SURVIVAL`}，*failed* = {`FAILED_UNRECOVERED`, `TRANSIENT_FAILURE_RECOVERED`}。
4. **分轨存活率**：
   - `in_domain_peer_survival_rate` = 同域组（`group1_in_domain`：FocusTrack、MemLoTrack 各版本等）中 survived 的比例；
   - `zero_shot_peer_survival_rate` = 零样本组（`group2_zero_shot`：DAM4SAM、SAMOSA、SAMURAI、UETrack、SUTrack、UTPTrack、ReFocus-TIR）中 survived 的比例；
   - 同时输出两轨的 `failure_rate`、`best_in_domain_peer`、`best_zero_shot_peer` 与 `best_peer`（按窗口平均 SA）；
   - 分组由提取器内置前缀注册表决定，可用 `--tracker-groups` JSON 覆盖；**未登记的算法记为 `unassigned`，不计入任何一轨**并在输出中列出（例如 AiATrack 需人工确认训练域后再指定分组）。
5. **Type 判定规则（分轨，阈值冻结）**：*rescuer* = 对手 survived，或（SiamDT 为 `FAILED_UNRECOVERED` 且对手为 `TRANSIENT_FAILURE_RECOVERED`）。

| 类型 | 判定条件（按顺序） |
| :--- | :--- |
| `Type_X_Subject_Not_Evaluated` | `SiamDT_Baseline` 在该窗口无逐帧记录 |
| `Type_U_No_Peer_Evidence` | 两轨均无可用对手记录（不做 A/B/C/D 推断） |
| `Type_C_SiamDT_Unique_Failure` | SiamDT failed，且同域对手数 $\ge 1$，且 `in_domain_peer_survival_rate` **> 0.5**（同域对手多数存活；零样本轨结果不影响 C 的判定） |
| `Type_B_Rescued_SiamDT_Failure` | SiamDT failed，非 C，且至少 1 个 rescuer（任一轨；`rescue_tracks` 标注来源轨） |
| `Type_A_Common_Hard_Failure` | SiamDT failed，且两轨均无 rescuer |
| `Type_D_InDomain_Peer_Failure_SiamDT_Success` | SiamDT survived，且同域对手数 $\ge 1$，且同域 `failure_rate` **> 0.5** |
| `Type_D0_ZeroShot_Only_Peer_Failure_SiamDT_Success` | SiamDT survived，非 D，且零样本对手数 $\ge 1$，且零样本 `failure_rate` **> 0.5** |
| `Type_E_All_Survived` | SiamDT survived，非 D / D0 |

   - 当同域对手仅 1 个时，输出 `type_c_evidence_note = single_in_domain_peer`，论文中需注明证据强度。
   - 主口径窗口由 SiamDT 自身失败 episode 产生，因此 D/D0/E 主要出现在补充视图中。
6. **Holdout 封存护栏**：提取器与六层解剖器默认拒绝读取 Holdout 序列（`holdout_verification.json` 的 16 条 ∪ `val90_full.json` 的 `holdout_full_36`）。纯 Holdout 划分直接拒绝；混合划分（如 `val90_full`）自动排除 Holdout 序列且**不打开其遥测文件**，并在输出 `holdout_guard.excluded_holdout_sequences` 中列出；找不到 Holdout 清单时按 fail-closed 拒绝执行。显式 `--unlock-holdout` 后结果写入 `<output_dir>__HOLDOUT_UNSEALED/` 并生成 `HOLDOUT_UNSEAL_RECORD.json`，与 Discovery 结果分开存放。

### 3.1 为什么必须构建统一挑战时间窗？（v1.0 原文，保留）
在同一条视频序列的同一次物理困难事件（如无人机掠过热辐射高光建筑、进入树冠遮挡或出视野重入）中，不同跟踪器发生抖动或失锁的起止帧往往存在数帧至数十帧的相位差（例如算法 A 在第 `410` 帧失锁，算法 B 在第 `416` 帧失锁并在第 `435` 帧恢复）。
若直接按各算法自身的 Failure Episode 独立统计，将导致事件分母不一致。通过构建对所有被测算法完全对齐的 **Canonical Challenge Windows（统一挑战时间窗）**，Phase 3 可以在**完全相同的序列区间 $[u_{\text{start}}, u_{\text{end}}]$** 内公平判定每个算法是“稳健存活”、“短暂失锁后恢复”还是“彻底漂移”，从而严密计算 **Failure Matrix** 与 **Rescue Matrix（围点打援救援矩阵）**。

### 3.2 Pool-Union 窗口生成算法（四步法；v1.1 起仅作补充视图，步骤 4 的状态分级阈值仍为两种视图共用）
对于给定子集（如 `discovery_set` 24 序列）中的每一条序列 $s$：

1. **步骤 1｜收集多源种子事件区间（Seed Intervals Collection）**：
   - 收集当前已评测的所有跟踪器 $m \in \mathcal{M}$ 在序列 $s$ 上提取出的主失败事件段 $\{[t_{\text{start}}^{(m,k)}, t_{\text{end}}^{(m,k)}]\}$（基于主阈值 $\tau_{\text{fail}} = 0.10, g_{\text{bridge}} = 3$）；
   - 同时纳入序列 $s$ 真值标注中的**目标消失与重入考题区间**：对于每一段真值消失区间 $[t_{\text{abs\_start}}, t_{\text{abs\_end}}]$，将其与重入后 $15$ 帧合并为种子区间 $[t_{\text{abs\_start}}, \min(T_s - 1, t_{\text{abs\_end}} + 15)]$，确保所有算法在“目标消失防误报 + 重入重捕获”考题上接受统一检验。
2. **步骤 2｜跨算法时间并集与近邻合并（Temporal Union & Proximity Merging）**：
   - 将序列 $s$ 上的所有种子区间按起始帧升序排列；
   - 若两个区间 $[a_1, b_1]$ 与 $[a_2, b_2]$ 存在时间重叠或间隔不超过 $G_{\text{canon}} = 5$ 帧（即 $a_2 \le b_1 + G_{\text{canon}} + 1$），则合并为 $[\min(a_1, a_2), \max(b_1, b_2)]$；
   - 迭代合并直至生成两两不重叠的**核心挑战跨度（Core Challenge Span）** $[u_{\text{start}}, u_{\text{end}}]$。
3. **步骤 3｜统一前导与后置观测窗口扩展（Standardized Context Padding）**：
   - 为每个核心挑战跨度 $[u_{\text{start}}, u_{\text{end}}]$ 附加统一的前导窗口 $W_{\text{pre}}^{\text{canon}} = 15$ 帧与后置恢复观测窗口 $W_{\text{post}}^{\text{canon}} = 10$ 帧：
     $$\text{ContextSpan} = \big[\max(1, u_{\text{start}} - 15),\; \min(T_s - 1, u_{\text{end}} + 10)\big]$$
   - 其中，窗口内的性能分级严格基于核心挑战跨度 $[u_{\text{start}}, u_{\text{end}}]$ 计算，而后置窗口 $[u_{\text{end}}+1, \min(T_s-1, u_{\text{end}}+10)]$ 专门用于核验算法在挑战结束时是否实现稳定重锁定（`post_window_locked`）。
4. **步骤 4｜各算法在统一窗口内的表现分级与 Rescue 归类**：
   对任意被测算法 $m \in \mathcal{M}$ 在统一挑战窗口 $W_j$（核心跨度 $[u_{\text{start}}, u_{\text{end}}]$）内的表现，计算：
   - 核心跨度存在帧失败率 $r_{\text{fail}}^{(m)} = \text{FR}_{[u_{\text{start}}, u_{\text{end}}]}^{(m)}(0.10)$；
   - 核心跨度平均状态精度 $\overline{\text{SA}}^{(m)}_{[u_{\text{start}}, u_{\text{end}}]}$ 与存在帧平均重叠率 $\overline{\text{IoU}}^{(m)}_{[u_{\text{start}}, u_{\text{end}}]}$；
   - 窗口末端/后置恢复标志 `post_locked`：在 $[\max(u_{\text{start}}, u_{\text{end}}-4), \min(T_s-1, u_{\text{end}}+10)]$ 区间内是否达成连续 $\ge 5$ 帧（或序列末尾剩余全部存在帧）$\text{IoU} \ge 0.50$。
   
   据此将算法 $m$ 在窗口 $W_j$ 上的状态严格映射为 4 个等级：
   - **`SUCCESS`（高质量通过）**：$r_{\text{fail}}^{(m)} = 0.0$ 且 $\overline{\text{SA}}^{(m)} \ge 0.50$；
   - **`ROBUST_SURVIVAL`（稳健存活）**：$r_{\text{fail}}^{(m)} \le 0.15$ 且（$\overline{\text{SA}}^{(m)} \ge 0.35$ 或 `post_locked == True`）；
   - **`TRANSIENT_FAILURE_RECOVERED`（中途失锁但自救恢复）**：$r_{\text{fail}}^{(m)} > 0.15$，但在窗口结束时已成功恢复锁定（`post_locked == True`）；
   - **`FAILED_UNRECOVERED`（失锁且未恢复）**：$r_{\text{fail}}^{(m)} > 0.15$ 且 `post_locked == False`。

   > [!WARNING]
   > **以下 v1.0 Type 判定规则已于 2026-11-04 被 3.0 第 5 条的分轨规则取代**（原因：混合计算同域/零样本存活率会误判 Type C）。保留原文仅供追溯，提取器不再使用。

   在此基础上，以 `SiamDT_Baseline` 为锚点，自动将每个 Canonical Challenge Window 归入 Phase 3 的四类诊断象限（v1.0，已废止）：
   - **`Type_A_Common_Hard_Failure`（共性极限难题）**：`SiamDT_Baseline` 失败（`FAILED_UNRECOVERED` 或 `TRANSIENT_FAILURE_RECOVERED`），且池中**没有任何对比算法**达到 `SUCCESS` 或 `ROBUST_SURVIVAL`；
   - **`Type_B_Rescued_SiamDT_Failure`（可被异构机制救援的失败——“围点打援”核心靶标）**：`SiamDT_Baseline` 失败，而**至少有 1 个对比算法**在该窗口内达到 `SUCCESS` 或 `ROBUST_SURVIVAL`（或当 `SiamDT_Baseline` 为 `FAILED_UNRECOVERED` 时对比算法实现了 `TRANSIENT_FAILURE_RECOVERED` 自救）；
   - **`Type_C_SiamDT_Unique_Failure`（SiamDT 特有结构性失败）**：`Type_B` 的高优先级子集——`SiamDT_Baseline` 失败，而池中**$\ge 80\%$ 的已评测对比算法**均稳健通过（`SUCCESS` / `ROBUST_SURVIVAL`）；
   - **`Type_D_Peer_Failure_SiamDT_Success`（SiamDT 优势窗口）**：`SiamDT_Baseline` 稳健通过（`SUCCESS` / `ROBUST_SURVIVAL`），而至少 1 个对比算法发生失败（体现 `SiamDT` 全图双阶段检测在大跨度重捕获上的固有优势，Phase 6 改进时严禁破坏此类能力）。

---

## 四、 配套代码与标准输出产物契约

本协议由同目录下的 [`failure_episode_extractor.py`](file:///Z:/SimDT围点打援/phase2_unified_eval/failure_episode_extractor.py) 100% 忠实实现，标准输入与输出如下：

1. **输入**：
   - 逐帧遥测目录：`<drive_or_local>/experiments/failure_diagnosis/per_frame_telemetry/<TrackerName>_<SplitName>/<seq_id>.jsonl`
   - 冻结划分清单：`Z:\SimDT围点打援\datasets\diagnostic_splits\discovery_set.json`（或 `holdout_verification.json` / `val90_full.json`）
2. **输出**（未解封时写入 `--output-dir`；解封 Holdout 后写入 `<output-dir>__HOLDOUT_UNSEALED/`）：
   - **`FAILURE_EPISODE_DATABASE.json`**：包含所有被测算法在 Level 1（序列级指标与失分拆解）、Level 2（逐帧四态分布、消失重入延迟、$\tau \in \{0.0, 0.10, 0.30, 0.50\}$ 与 $\text{NCE} > 1.0$ 敏感性矩阵）及 Level 3（全部 Failure Episodes 详情、前导 15 帧退化特征与恢复状态）的完整结构化数据库；v1.1 起附带 `holdout_guard`、`tracker_groups`、`tracker_set_sha256`。
   - **`CANONICAL_CHALLENGE_WINDOWS.json`**（v1.1 起 = **主口径 Subject-Anchored Windows**，`window_mode = "subject_anchored"`）：窗口列表、物理属性、各算法状态分级、分轨存活率（`in_domain_peer_survival_rate` / `zero_shot_peer_survival_rate`）、`best_peer`、`rescue_tracks`、Type 归类以及 `window_set_sha256`。
   - **`POOL_UNION_CHALLENGE_WINDOWS.json`**（v1.1 新增，补充视图）：v1.0 并集窗口，附 `tracker_set` 与 `tracker_set_sha256`。
   - **`HOLDOUT_UNSEAL_RECORD.json`**（仅解封时）：解封时间、涉及序列与"严禁据此回调阈值"的声明。
3. **自检**：`python failure_episode_extractor.py --self-test` 覆盖桥接/重入延迟、Subject-Anchored 窗口不随新增算法变化、分轨分类、Holdout 默认拒绝 / fail-closed / 解封目录分离。
