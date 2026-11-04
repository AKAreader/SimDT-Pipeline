# 【总控派工单 A】Phase 1｜Anti-UAV 代表性诊断子集（Diagnostic Subset）构建专项任务书

## 1. 会话角色与权限边界
- **你的角色**：**Phase 1 数据工程与代表性诊断子集构建专员（Subsession-A）**。
- **上级总控**：项目总工（Master Control Session，工作区 `Z:\SimDT围点打援`）。
- **核心铁律**：
  1. **只执行 Phase 1**：你的唯一任务是构建并冻结 **`Anti-UAV Diagnostic Subset`（代表性诊断子集）**，划分为 **`Discovery Set`（发现集）** 与 **`Holdout Verification Set`（独立验证集）**。
  2. **禁止反向挑数据（No Cherry-picking by Score）**：必须优先依据数据集官方场景属性标签及可客观计算的物理属性（目标尺寸、局部热对比度/信杂比、运动速度/加速度、尺度变化率、背景杂波复杂度、消失/重入段）进行分层客观抽样；**严禁根据某一个算法的最终排名或为了凑某个模块有利结果去反向挑选序列**。
  3. **严禁污染上游只读目录**：对 `E:\PHD\YB` 与 `D:\CodexProjects\SiamDT_Project` 仅允许**只读（Read-Only）**访问，所有脚本、中间统计与交付报告必须全部写入 `Z:\SimDT围点打援\phase1_diagnostic_subset\`。
  4. **证据标签纪律**：报告中所有陈述必须严格区分 `【事实】`、`【实验结论】`、`【推测】`、`【建议】`。

---

## 2. 本地数据源只读路径指引
- **Anti-UAV410 数据集路径**：
  - `E:\PHD\YB\dataset\Anti-UAV410` 或 `D:\CodexProjects\SiamDT_Project\data\Anti-UAV410`（含 `train`、`val` 90 序列、`test` 120 序列及 JSON/txt 标注文件）。
  - **数据划分红线**：诊断子集应优先从 **`val`（90 序列）** 中科学分层构建（或明确说明 `val` 内的 `Discovery Set` 与 `Holdout Verification Set` 切分方案），保持 `test`（120 序列）作为最终论文盲测集不被泄露。
- **可参考的已有客观统计事实（只读）**：
  - `E:\PHD\YB\Anti_yb\experiments\anti_uav.py`（查看官方属性标签定义，如 `OV`, `OC`, `FM`, `SV`, `LI`, `DBC`, `TS`, `TC` 等）；
  - `E:\PHD\YB\yb\research_20261103\failure_diagnosis\BASELINE_FULLVAL90_STATS.json`（仅用于核对 `val90` 总序列数、帧数与尺寸分布，不得直接按单一模型成绩挑序列）。

---

## 3. 诊断子集必须覆盖的 10 类典型困难场景
你设计的 `Anti-UAV Diagnostic Subset` 必须确保对以下 10 类具有科研诊断价值的困难情况形成可量化的覆盖：
1. **极小目标（Tiny Target）**：如等效尺寸 $\sqrt{w \cdot h} < 16\text{ px}$ 甚至 $< 10\text{ px}$ 的高占比序列；
2. **低对比度 / 热交叉（Low Contrast / Thermal Crossover）**：目标灰度与周围背景均值差异极小、局部信杂比（SCR）极低；
3. **遮挡（Occlusion, `OC`）**：树木、建筑物、云层遮挡导致目标局部或短暂不可见；
4. **快速运动与剧烈机动（Fast Motion & High Acceleration, `FM`）**：帧间位移大或存在急停急转非线性加速度；
5. **大尺度变化（Scale Variation, `SV`）**：序列内目标面积跨度大；
6. **强背景干扰（Dynamic Background Clutter, `DBC`）**：复杂云层、城市建筑、山体林地热辐射杂波；
7. **类目标干扰物（Target-like Distractors）**：目标周围存在尺寸或热辐射强度相近的孤立热源/假目标；
8. **易漂移场景（Drift-Prone Scenarios）**：目标经过强边缘或高热结构物附近；
9. **易误锁场景（False-Lock Prone Scenarios）**：初始或中途背景中存在强显著性热源；
10. **丢失后难恢复 / 出视野重入（Out-of-View & Hard Recovery, `OV`）**：包含 `exist == 0` 消失段及重入视野后的恢复考验。

---

## 4. Discovery Set 与 Holdout Verification Set 划分要求
- 将入选的诊断子集科学划分为两个互不重叠的子集（建议总规模控制在 `30~45` 条代表性序列，或给出全量 `val90` 分层索引 + 核心精简子集双层结构）：
  1. **`Discovery Set`（发现集，约占 60%–67%）**：用于 Phase 2–5 发现失败模式、提取 Canonical Challenge Windows、进行统计归因与内部特征诊断，以及产生创新假设。
  2. **`Holdout Verification Set`（独立验证集，约占 33%–40%，立即冻结）**：用于 Phase 6 候选模块形成后的独立验证；**严禁在 Phase 3–5 的特征查看和超参调节中使用 Holdout 集**。
- 必须用统计检验（如均值/分位数对比或 KS 检验）证明：`Discovery Set` 与 `Holdout Verification Set` 在上述 10 类困难属性与目标尺寸分布上是**同分布对齐（Stratified & Balanced）**的！

---

## 5. 必须产出的交付物（Gate 1 审查材料）
请在 `Z:\SimDT围点打援\phase1_diagnostic_subset\` 目录下生成以下文件：
1. **`PHASE1_SUBSET_DESIGN_REPORT.md`**：包含：
   - 子集构建原则与客观筛选公式；
   - 入选序列完整清单；
   - 每个序列覆盖的困难属性与客观物理指标表；
   - `Discovery Set` / `Holdout Verification Set` 划分结果及分布平衡性验证；
   - 选择依据与潜在偏差（Potential Bias）分析。
2. **`DIAGNOSTIC_SUBSET_MANIFEST.json`**：机器可读的 JSON 账本（含 `discovery_sequences`、`holdout_sequences`、每条序列的帧数、路径、属性标签向量、客观统计指标及冻结时间戳）。
3. **`build_diagnostic_subset.py`**：可一键复现该抽样与统计结果的数据构建脚本。

完成上述交付物后**立即停下**，向总控会话汇报并等待 **Gate 1** 审查。
