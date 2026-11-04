#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 三层失败提取器与挑战时间窗构建器 (failure_episode_extractor.py) —— v1.1（2026-11-04 修订）
================================================================================
严格实现《PHASE2_FAILURE_DEFINITION_PROTOCOL.md》（v1.1-phase2-protocol-rev20261104）：
1. Level 1 (Sequence): SA、存在帧 Mean IoU、ZOR、FR(0.10)、PR@20px、NPR@0.5、消失期误报率及 SA 失分四项可加拆解；
2. Level 2 (Frame): 四态分类、消失期误报与重入延迟、预注册敏感性分析（tau ∈ {0.0, 0.10, 0.30, 0.50}、NCE > 1.0、复合准则）；
3. Level 3 (Failure Episode): g_bridge=3 桥接、前导 15 帧上下文、恢复状态判定（以上三层与 v1.0 冻结协议完全一致）；
4. 挑战时间窗（2026-11-04 修订，依据 Claude × Gemini 交叉审查 P0-7）：
   - 【主口径】Subject-Anchored Windows：窗口边界仅由 SiamDT_Baseline 自身的失败 episode（加前后上下文）决定，
     与参与算法集合无关 —— 后加入的算法不会改变已锁定的窗口（输出 window_set_sha256 便于核验）；
   - 【补充视图】Pool-Union Windows：v1.0 的"全部算法失败段 + GT 消失重入"并集窗口，输出 tracker_set 及其 sha256；
   - 存活率分轨：in_domain_peer_survival_rate（group1 同域）与 zero_shot_peer_survival_rate（group2 零样本）分别计算，
     并给出 best_peer / best_in_domain_peer / best_zero_shot_peer；Type A/B/C/D/D0/E/U/X 按分轨规则判定（见 CLASSIFICATION_RULES_DOC）；
5. Holdout 封存护栏：默认拒绝读取 holdout 序列（holdout_verification.json ∪ val90_full.holdout_full_36），
   需 --unlock-holdout 才放行，解封结果写入独立的 <output_dir>__HOLDOUT_UNSEALED 目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# 冻结协议常量 (Pre-registered & Frozen Protocol Constants)
# ==============================================================================
PRIMARY_TAU_FAIL: float = 0.10
IMPRECISE_UPPER_IOU: float = 0.50
HIGH_QUALITY_IOU: float = 0.50
SENSITIVITY_TAU_LIST: Tuple[float, ...] = (0.0, 0.10, 0.30, 0.50)
PRIMARY_NCE_FAIL: float = 1.0

BRIDGE_GAP_FRAMES: int = 3
PRE_CONTEXT_FRAMES: int = 15
RECOVERY_CONSECUTIVE_FRAMES: int = 5
RECOVERY_IOU_THRESHOLD: float = 0.50

CANONICAL_MERGE_GAP: int = 5
CANONICAL_PRE_PAD: int = 15
CANONICAL_POST_PAD: int = 10
ABSENT_REENTRY_SEED_HORIZON: int = 15


# ==============================================================================
# 基础几何与单帧计算工具
# ==============================================================================
def calc_iou_xywh(pred: List[float], gt: List[float]) -> float:
    """计算 [x, y, w, h] 格式边界框的 IoU。"""
    if len(pred) < 4 or len(gt) < 4:
        return 0.0
    px, py, pw, ph = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
    gx, gy, gw, gh = float(gt[0]), float(gt[1]), float(gt[2]), float(gt[3])
    if pw <= 0.0 or ph <= 0.0 or gw <= 0.0 or gh <= 0.0:
        return 0.0
    ox0 = max(px, gx)
    oy0 = max(py, gy)
    ox1 = min(px + pw, gx + gw)
    oy1 = min(py + ph, gy + gh)
    if ox1 <= ox0 or oy1 <= oy0:
        return 0.0
    inter = (ox1 - ox0) * (oy1 - oy0)
    union = (pw * ph) + (gw * gh) - inter
    return float(inter / union) if union > 0.0 else 0.0


def classify_present_frame_state(iou: float, tau_fail: float = PRIMARY_TAU_FAIL) -> str:
    """对 exist == 1 的有效帧进行四态分类。"""
    if iou <= 0.0:
        return "ZERO_OVERLAP_LOST"
    if iou < tau_fail:
        return "FAILURE_FRAME"
    if iou < IMPRECISE_UPPER_IOU:
        return "IMPRECISE_LOCALIZATION"
    return "HIGH_QUALITY_LOCK"


def is_frame_failure_under_criterion(row: Dict[str, Any], criterion: str) -> bool:
    """
    在给定敏感性准则下判定 exist == 1 的有效帧是否属于失败帧。
    支持 criterion:
      - 'tau_0.00': IoU == 0.0
      - 'tau_0.10': IoU < 0.10 (Primary Frozen)
      - 'tau_0.30': IoU < 0.30
      - 'tau_0.50': IoU < 0.50
      - 'nce_gt_1.0': norm_center_error > 1.0 (或 pred_absent_flag == 1)
      - 'composite_iou0.10_or_nce1.0': (IoU < 0.10) or (norm_center_error > 1.0)
    """
    iou = float(row.get("iou", 0.0))
    nce = float(row.get("norm_center_error", -1.0))
    pred_absent = int(row.get("pred_absent_flag", 0)) == 1
    if pred_absent:
        return True
    if criterion == "tau_0.00":
        return iou <= 0.0
    if criterion == "tau_0.10":
        return iou < 0.10
    if criterion == "tau_0.30":
        return iou < 0.30
    if criterion == "tau_0.50":
        return iou < 0.50
    if criterion == "nce_gt_1.0":
        return (nce < 0.0) or (nce > PRIMARY_NCE_FAIL)
    if criterion == "composite_iou0.10_or_nce1.0":
        return (iou < 0.10) or (nce < 0.0) or (nce > PRIMARY_NCE_FAIL)
    raise ValueError(f"Unknown failure criterion: {criterion}")


def get_duration_bucket(present_len: int) -> str:
    """按有效存在帧长度划分失败事件持续时间分桶。"""
    if present_len <= 5:
        return "Bucket_1_5_Transient"
    if present_len <= 20:
        return "Bucket_6_20_Short"
    if present_len <= 100:
        return "Bucket_21_100_Medium"
    return "Bucket_101_Plus_Long_Drift"


def _linear_slope(values: List[float]) -> float:
    """计算序列值的最小二乘线性退化斜率。"""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = 0.5 * (n - 1)
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return float(num / den) if den > 0.0 else 0.0


# ==============================================================================
# 消失重入延迟 (Post-Absence Re-acquisition Latency) 提取
# ==============================================================================
def extract_absence_and_reacquisition_events(
    rows: List[Dict[str, Any]],
    k_rec: int = RECOVERY_CONSECUTIVE_FRAMES,
    rec_iou: float = RECOVERY_IOU_THRESHOLD,
) -> Dict[str, Any]:
    """
    针对序列中的 exist == 0 帧：
    1. 统计 false_positive_in_absent（模型未输出空框）与消失期间发生的错误模板更新次数；
    2. 提取所有连续消失区间 [t_abs_start, t_abs_end]，若后续存在重入视野帧 t_reentry，
       计算恢复至连续 k_rec=5 帧 IoU >= 0.50 所需的 Re-acquisition Latency（帧数）。
    """
    total_absent = 0
    fp_in_absent = 0
    tn_in_absent = 0
    updates_in_absent = 0

    n = len(rows)
    absent_intervals: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        ex = int(rows[i].get("exist", 1))
        if ex == 0:
            start_i = i
            while i < n and int(rows[i].get("exist", 1)) == 0:
                total_absent += 1
                p_abs = int(rows[i].get("pred_absent_flag", 0))
                if p_abs == 0:
                    fp_in_absent += 1
                else:
                    tn_in_absent += 1
                if bool(rows[i].get("template_updated", False)):
                    updates_in_absent += 1
                i += 1
            absent_intervals.append((start_i, i - 1))
        else:
            i += 1

    reacquisition_events: List[Dict[str, Any]] = []
    for abs_idx, (abs_start, abs_end) in enumerate(absent_intervals, start=1):
        abs_len = abs_end - abs_start + 1
        fp_seg = sum(
            1 for t in range(abs_start, abs_end + 1)
            if int(rows[t].get("pred_absent_flag", 0)) == 0
        )
        upd_seg = sum(
            1 for t in range(abs_start, abs_end + 1)
            if bool(rows[t].get("template_updated", False))
        )

        # 查找重入后的有效存在帧（直至下一次消失或序列结束）
        reentry_frame = abs_end + 1
        if reentry_frame >= n:
            reacquisition_events.append({
                "absence_index": abs_idx,
                "absent_start_frame": int(rows[abs_start]["frame_id"]),
                "absent_end_frame": int(rows[abs_end]["frame_id"]),
                "absent_duration_frames": abs_len,
                "false_positive_frames_in_absence": fp_seg,
                "template_updates_in_absence": upd_seg,
                "has_target_reentry": False,
                "reentry_frame": None,
                "reacquired": None,
                "reacquisition_latency_frames": None,
                "stabilized_lock_frame": None,
            })
            continue

        # 收集从 reentry_frame 开始直到下一段长时间消失或序列末尾的有效存在帧
        post_present_indices: List[int] = []
        for t in range(reentry_frame, n):
            if int(rows[t].get("exist", 1)) == 1:
                gt_b = rows[t].get("gt_bbox", [0, 0, 0, 0])
                if len(gt_b) >= 4 and float(gt_b[2]) > 0 and float(gt_b[3]) > 0:
                    post_present_indices.append(t)
            else:
                # 若紧接着再次进入消失段且尚未恢复，停止当前重入窗口扫描
                break

        if not post_present_indices:
            reacquisition_events.append({
                "absence_index": abs_idx,
                "absent_start_frame": int(rows[abs_start]["frame_id"]),
                "absent_end_frame": int(rows[abs_end]["frame_id"]),
                "absent_duration_frames": abs_len,
                "false_positive_frames_in_absence": fp_seg,
                "template_updates_in_absence": upd_seg,
                "has_target_reentry": False,
                "reentry_frame": None,
                "reacquired": None,
                "reacquisition_latency_frames": None,
                "stabilized_lock_frame": None,
            })
            continue

        req_consec = min(k_rec, len(post_present_indices))
        reacquired = False
        latency_present_frames = len(post_present_indices)
        lock_frame_id: Optional[int] = None

        for p_pos in range(0, len(post_present_indices) - req_consec + 1):
            window_idxs = post_present_indices[p_pos : p_pos + req_consec]
            if all(float(rows[w].get("iou", 0.0)) >= rec_iou for w in window_idxs):
                reacquired = True
                latency_present_frames = p_pos
                lock_frame_id = int(rows[window_idxs[0]]["frame_id"])
                break

        reacquisition_events.append({
            "absence_index": abs_idx,
            "absent_start_frame": int(rows[abs_start]["frame_id"]),
            "absent_end_frame": int(rows[abs_end]["frame_id"]),
            "absent_duration_frames": abs_len,
            "false_positive_frames_in_absence": fp_seg,
            "template_updates_in_absence": upd_seg,
            "has_target_reentry": True,
            "reentry_frame": int(rows[post_present_indices[0]]["frame_id"]),
            "available_post_reentry_frames": len(post_present_indices),
            "reacquired": reacquired,
            "reacquisition_latency_frames": latency_present_frames,
            "stabilized_lock_frame": lock_frame_id,
        })

    valid_reentries = [e for e in reacquisition_events if e["has_target_reentry"]]
    reacquired_list = [e for e in valid_reentries if e["reacquired"] is True]
    latencies = [int(e["reacquisition_latency_frames"]) for e in valid_reentries if e["reacquisition_latency_frames"] is not None]

    return {
        "total_absent_frames": total_absent,
        "false_positive_in_absent_frames": fp_in_absent,
        "true_negative_in_absent_frames": tn_in_absent,
        "false_positive_in_absent_rate": round(fp_in_absent / total_absent, 6) if total_absent > 0 else 0.0,
        "template_updates_in_absent_frames": updates_in_absent,
        "num_absence_intervals": len(absent_intervals),
        "num_reentry_events": len(valid_reentries),
        "num_reacquired_events": len(reacquired_list),
        "reacquisition_success_rate": round(len(reacquired_list) / len(valid_reentries), 6) if valid_reentries else None,
        "mean_reacquisition_latency_frames": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "max_reacquisition_latency_frames": max(latencies) if latencies else None,
        "events": reacquisition_events,
    }


# ==============================================================================
# Level 3: 失败事件段聚合 (Failure Episode Extraction with Bridging & Context)
# ==============================================================================
def extract_failure_episodes_for_sequence(
    rows: List[Dict[str, Any]],
    criterion: str = "tau_0.10",
    bridge_gap: int = BRIDGE_GAP_FRAMES,
    pre_context_len: int = PRE_CONTEXT_FRAMES,
    k_rec: int = RECOVERY_CONSECUTIVE_FRAMES,
    rec_iou: float = RECOVERY_IOU_THRESHOLD,
    include_detailed_context: bool = True,
) -> List[Dict[str, Any]]:
    """
    将单条序列的逐帧遥测按指定失败准则提取并桥接为 Level 3 Failure Episodes。
    - 仅在有效存在帧 (exist == 1 且 gt_w > 0, gt_h > 0) 上判定定位失败（排除 frame_id == 0 首帧真值初始化帧）；
    - 允许桥接 <= bridge_gap (3) 个非失败有效存在帧的瞬时偶发波动；
    - 附带 Pre-failure Context [t_start - 15, t_start - 1] 与恢复状态判定。
    """
    # 识别所有有效存在帧索引（排除第0帧初始化）
    valid_present_pos: List[int] = []
    for idx, r in enumerate(rows):
        if int(r.get("exist", 1)) == 1:
            gt_b = r.get("gt_bbox", [0, 0, 0, 0])
            if len(gt_b) >= 4 and float(gt_b[2]) > 0 and float(gt_b[3]) > 0:
                valid_present_pos.append(idx)

    if not valid_present_pos:
        return []

    last_present_row_idx = valid_present_pos[-1]

    # 在有效存在帧序列上提取初始失败游程 (排除首帧 frame_id == 0)
    raw_runs: List[Tuple[int, int]] = []  # 存储 valid_present_pos 上的下标区间 (p_start, p_end)
    p = 0
    num_p = len(valid_present_pos)
    while p < num_p:
        row_idx = valid_present_pos[p]
        if int(rows[row_idx].get("frame_id", row_idx)) == 0:
            p += 1
            continue
        if is_frame_failure_under_criterion(rows[row_idx], criterion):
            p_start = p
            while p < num_p and is_frame_failure_under_criterion(rows[valid_present_pos[p]], criterion):
                p += 1
            raw_runs.append((p_start, p - 1))
        else:
            p += 1

    if not raw_runs:
        return []

    # 桥接规则：若相邻两个失败游程之间的非失败存在帧数量 <= bridge_gap (3)，合并为一个 Episode
    merged_runs: List[Tuple[int, int]] = [raw_runs[0]]
    for cur_start, cur_end in raw_runs[1:]:
        prev_start, prev_end = merged_runs[-1]
        non_fail_present_gap = cur_start - prev_end - 1
        if non_fail_present_gap <= bridge_gap:
            merged_runs[-1] = (prev_start, cur_end)
        else:
            merged_runs.append((cur_start, cur_end))

    episodes: List[Dict[str, Any]] = []
    seq_id = str(rows[0].get("seq_id", "unknown"))

    for ep_idx, (p_start, p_end) in enumerate(merged_runs, start=1):
        row_start = valid_present_pos[p_start]
        row_end = valid_present_pos[p_end]
        t_start = int(rows[row_start].get("frame_id", row_start))
        t_end = int(rows[row_end].get("frame_id", row_end))

        ep_present_indices = valid_present_pos[p_start : p_end + 1]
        ep_all_rows = rows[row_start : row_end + 1]

        present_len = len(ep_present_indices)
        calendar_span = t_end - t_start + 1
        absent_inside = sum(1 for r in ep_all_rows if int(r.get("exist", 1)) == 0)

        ep_ious = [float(rows[i].get("iou", 0.0)) for i in ep_present_indices]
        ep_ces = [float(rows[i].get("center_error", -1.0)) for i in ep_present_indices if float(rows[i].get("center_error", -1.0)) >= 0.0]
        ep_nces = [float(rows[i].get("norm_center_error", -1.0)) for i in ep_present_indices if float(rows[i].get("norm_center_error", -1.0)) >= 0.0]
        ep_confs = [float(rows[i].get("confidence", 0.0)) for i in ep_present_indices]
        ep_sizes = [float(rows[i].get("target_size", 0.0)) for i in ep_present_indices if float(rows[i].get("target_size", 0.0)) > 0.0]
        ep_zero_count = sum(1 for v in ep_ious if v <= 0.0)
        ep_bridged_nonfail_count = sum(1 for i in ep_present_indices if not is_frame_failure_under_criterion(rows[i], criterion))
        ep_wrong_updates = sum(1 for r in ep_all_rows if bool(r.get("template_updated", False)))

        # 恢复判定：检查 p_end 之后的有效存在帧是否达成连续 k_rec=5 帧 IoU >= 0.50
        remaining_present_positions = valid_present_pos[p_end + 1 :]
        if row_end >= last_present_row_idx or not remaining_present_positions:
            recovery_status = "unrecovered_terminal_drift"
            recovered_flag = False
            recovery_latency = None
            recovery_lock_frame = None
        else:
            req_c = min(k_rec, len(remaining_present_positions))
            found_lock = False
            rec_lat: Optional[int] = None
            rec_frame: Optional[int] = None
            for offset in range(0, len(remaining_present_positions) - req_c + 1):
                cand_idxs = remaining_present_positions[offset : offset + req_c]
                if all(float(rows[c].get("iou", 0.0)) >= rec_iou for c in cand_idxs):
                    found_lock = True
                    rec_lat = offset
                    rec_frame = int(rows[cand_idxs[0]].get("frame_id", cand_idxs[0]))
                    break
            if found_lock:
                recovery_status = "recovered"
                recovered_flag = True
                recovery_latency = rec_lat
                recovery_lock_frame = rec_frame
            else:
                # 检查后续是否完全未再达到 >=0.50 直到终点
                any_subsequent_fail_to_end = is_frame_failure_under_criterion(rows[last_present_row_idx], criterion)
                recovery_status = "unrecovered_terminal_drift" if any_subsequent_fail_to_end else "partial_recovery_unstable"
                recovered_flag = False
                recovery_latency = None
                recovery_lock_frame = None

        ep_record: Dict[str, Any] = {
            "episode_id": f"{seq_id}__EP{ep_idx:03d}",
            "seq_id": seq_id,
            "episode_index": ep_idx,
            "criterion": criterion,
            "t_start": t_start,
            "t_end": t_end,
            "calendar_span_frames": calendar_span,
            "present_duration_frames": present_len,
            "duration_bucket": get_duration_bucket(present_len),
            "zero_overlap_frames_count": ep_zero_count,
            "bridged_transient_frames_count": ep_bridged_nonfail_count,
            "absent_frames_inside": absent_inside,
            "mean_iou": round(sum(ep_ious) / len(ep_ious), 6) if ep_ious else 0.0,
            "min_iou": round(min(ep_ious), 6) if ep_ious else 0.0,
            "mean_center_error": round(sum(ep_ces) / len(ep_ces), 4) if ep_ces else -1.0,
            "max_center_error": round(max(ep_ces), 4) if ep_ces else -1.0,
            "mean_norm_center_error": round(sum(ep_nces) / len(ep_nces), 4) if ep_nces else -1.0,
            "max_norm_center_error": round(max(ep_nces), 4) if ep_nces else -1.0,
            "mean_confidence": round(sum(ep_confs) / len(ep_confs), 6) if ep_confs else 0.0,
            "mean_target_size": round(sum(ep_sizes) / len(ep_sizes), 4) if ep_sizes else 0.0,
            "erroneous_template_updates_during_episode": ep_wrong_updates,
            "recovered": recovered_flag,
            "recovery_status": recovery_status,
            "recovery_latency_frames": recovery_latency,
            "recovery_stabilized_frame": recovery_lock_frame,
        }

        if include_detailed_context:
            # 提取 Pre-failure Context Window [max(1, row_start - 15), row_start - 1]
            pre_start_idx = max(1, row_start - pre_context_len) if row_start > 1 else 0
            pre_end_idx = max(0, row_start - 1)
            pre_rows = rows[pre_start_idx : pre_end_idx + 1] if row_start > 0 else []
            pre_present_rows = [r for r in pre_rows if int(r.get("exist", 1)) == 1]
            pre_ious = [float(r.get("iou", 0.0)) for r in pre_present_rows]
            pre_confs = [float(r.get("confidence", 0.0)) for r in pre_present_rows]
            pre_sizes = [float(r.get("target_size", 0.0)) for r in pre_present_rows if float(r.get("target_size", 0.0)) > 0.0]
            pre_updates = [r for r in pre_rows if bool(r.get("template_updated", False))]
            pre_absent_cnt = sum(1 for r in pre_rows if int(r.get("exist", 1)) == 0)

            # 计算前导帧间真值中心最大位移（速度峰值）
            pre_displacements: List[float] = []
            for k in range(1, len(pre_present_rows)):
                b0 = pre_present_rows[k - 1].get("gt_bbox", [0, 0, 0, 0])
                b1 = pre_present_rows[k].get("gt_bbox", [0, 0, 0, 0])
                if len(b0) >= 4 and len(b1) >= 4:
                    cx0, cy0 = float(b0[0]) + 0.5 * float(b0[2]), float(b0[1]) + 0.5 * float(b0[3])
                    cx1, cy1 = float(b1[0]) + 0.5 * float(b1[2]), float(b1[1]) + 0.5 * float(b1[3])
                    pre_displacements.append(math.hypot(cx1 - cx0, cy1 - cy0))

            ep_record["pre_failure_context"] = {
                "context_start_frame": int(pre_rows[0]["frame_id"]) if pre_rows else 0,
                "context_end_frame": int(pre_rows[-1]["frame_id"]) if pre_rows else 0,
                "num_context_frames": len(pre_rows),
                "absent_frames_in_pre_context": pre_absent_cnt,
                "triggered_immediately_after_absence": (row_start > 0 and int(rows[row_start - 1].get("exist", 1)) == 0),
                "pre_mean_iou": round(sum(pre_ious) / len(pre_ious), 6) if pre_ious else None,
                "pre_min_iou": round(min(pre_ious), 6) if pre_ious else None,
                "pre_iou_degradation_slope": round(_linear_slope(pre_ious), 6) if len(pre_ious) >= 2 else 0.0,
                "pre_mean_confidence": round(sum(pre_confs) / len(pre_confs), 6) if pre_confs else None,
                "pre_mean_target_size": round(sum(pre_sizes) / len(pre_sizes), 4) if pre_sizes else None,
                "pre_max_gt_step_displacement": round(max(pre_displacements), 4) if pre_displacements else 0.0,
                "pre_template_updates_count": len(pre_updates),
                "last_pre_update_iou": round(float(pre_updates[-1].get("iou", 0.0)), 6) if pre_updates else None,
                "onset_pattern": (
                    "post_absence_reentry_miss"
                    if (row_start > 0 and int(rows[row_start - 1].get("exist", 1)) == 0)
                    else (
                        "sudden_jump_loss"
                        if (pre_ious and pre_ious[-1] >= 0.50)
                        else "gradual_erosion_drift"
                    )
                ),
            }

        episodes.append(ep_record)

    return episodes


# ==============================================================================
# 单序列全层级分析 (Level 1 + Level 2 + Level 3 + Sensitivity)
# ==============================================================================
def analyze_single_sequence(
    rows: List[Dict[str, Any]],
    seq_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """对单个算法在单条序列上的逐帧遥测执行完整三层统计分析与敏感性检验。"""
    if not rows:
        return {}

    seq_id = str(rows[0].get("seq_id", "unknown"))
    total_frames = len(rows)

    present_rows: List[Dict[str, Any]] = []
    absent_rows: List[Dict[str, Any]] = []
    invalid_gt_rows: List[Dict[str, Any]] = []

    for r in rows:
        ex = int(r.get("exist", 1))
        if ex == 1:
            gt_b = r.get("gt_bbox", [0, 0, 0, 0])
            if len(gt_b) >= 4 and float(gt_b[2]) > 0 and float(gt_b[3]) > 0:
                present_rows.append(r)
            else:
                invalid_gt_rows.append(r)
        else:
            absent_rows.append(r)

    sa_valid_rows = present_rows + absent_rows
    sa_denom = len(sa_valid_rows)
    n_present = len(present_rows)
    n_absent = len(absent_rows)

    # Level 1 & Level 2 帧级分布统计
    state_counts = {
        "ZERO_OVERLAP_LOST": 0,
        "FAILURE_FRAME": 0,  # 注意：FAILURE_FRAME 为主失败判定 IoU < 0.10（包含 ZERO_OVERLAP_LOST）
        "FAILURE_NONZERO_LT_0_10": 0,
        "IMPRECISE_LOCALIZATION": 0,
        "HIGH_QUALITY_LOCK": 0,
    }

    fail_sa_loss_sum = 0.0
    imprecise_sa_loss_sum = 0.0
    high_sa_loss_sum = 0.0
    absent_sa_loss_sum = 0.0

    pr_20_hits = 0
    npr_05_hits = 0

    for r in present_rows:
        iou = float(r.get("iou", 0.0))
        ce = float(r.get("center_error", -1.0))
        nce = float(r.get("norm_center_error", -1.0))
        if 0.0 <= ce <= 20.0:
            pr_20_hits += 1
        if 0.0 <= nce <= 0.50:
            npr_05_hits += 1

        if iou <= 0.0:
            state_counts["ZERO_OVERLAP_LOST"] += 1
            state_counts["FAILURE_FRAME"] += 1
            fail_sa_loss_sum += (1.0 - iou)
        elif iou < PRIMARY_TAU_FAIL:
            state_counts["FAILURE_NONZERO_LT_0_10"] += 1
            state_counts["FAILURE_FRAME"] += 1
            fail_sa_loss_sum += (1.0 - iou)
        elif iou < IMPRECISE_UPPER_IOU:
            state_counts["IMPRECISE_LOCALIZATION"] += 1
            imprecise_sa_loss_sum += (1.0 - iou)
        else:
            state_counts["HIGH_QUALITY_LOCK"] += 1
            high_sa_loss_sum += (1.0 - iou)

    for r in absent_rows:
        sa_v = float(r.get("sa_score", 0.0))
        absent_sa_loss_sum += (1.0 - sa_v)

    seq_sa = sum(float(r.get("sa_score", 0.0)) for r in sa_valid_rows) / sa_denom if sa_denom > 0 else 0.0
    mean_iou_exist = sum(float(r.get("iou", 0.0)) for r in present_rows) / n_present if n_present > 0 else 0.0

    # Level 2 消失重入专项统计
    absence_reacq = extract_absence_and_reacquisition_events(rows)

    # Level 3 主协议失败事件段提取 (tau_fail = 0.10, g_bridge = 3)
    primary_episodes = extract_failure_episodes_for_sequence(
        rows,
        criterion="tau_0.10",
        bridge_gap=BRIDGE_GAP_FRAMES,
        pre_context_len=PRE_CONTEXT_FRAMES,
        include_detailed_context=True,
    )

    # 预注册敏感性分析 (tau in {0.0, 0.10, 0.30, 0.50}, NCE > 1.0, Composite)
    sensitivity_criteria = [
        "tau_0.00",
        "tau_0.10",
        "tau_0.30",
        "tau_0.50",
        "nce_gt_1.0",
        "composite_iou0.10_or_nce1.0",
    ]
    sensitivity_stats: Dict[str, Any] = {}
    for crit in sensitivity_criteria:
        # 统计非首帧存在帧中的失败帧数
        eval_present = [r for r in present_rows if int(r.get("frame_id", 0)) > 0]
        fail_cnt = sum(1 for r in eval_present if is_frame_failure_under_criterion(r, crit))
        crit_eps = (
            primary_episodes
            if crit == "tau_0.10"
            else extract_failure_episodes_for_sequence(
                rows, criterion=crit, bridge_gap=BRIDGE_GAP_FRAMES, include_detailed_context=False
            )
        )
        ep_durations = [int(e["present_duration_frames"]) for e in crit_eps]
        sensitivity_stats[crit] = {
            "failed_present_frames": fail_cnt,
            "failure_frame_rate": round(fail_cnt / len(eval_present), 6) if eval_present else 0.0,
            "episode_count": len(crit_eps),
            "mean_episode_duration": round(sum(ep_durations) / len(ep_durations), 3) if ep_durations else 0.0,
            "max_episode_duration": max(ep_durations) if ep_durations else 0,
            "unrecovered_terminal_episodes": sum(
                1 for e in crit_eps if e["recovery_status"] == "unrecovered_terminal_drift"
            ),
        }

    bucket_counts = {
        "Bucket_1_5_Transient": 0,
        "Bucket_6_20_Short": 0,
        "Bucket_21_100_Medium": 0,
        "Bucket_101_Plus_Long_Drift": 0,
    }
    for ep in primary_episodes:
        bname = ep["duration_bucket"]
        bucket_counts[bname] = bucket_counts.get(bname, 0) + 1

    ep_lens = [int(e["present_duration_frames"]) for e in primary_episodes]
    latencies_ms = [float(r.get("latency_ms", 0.0)) for r in rows if float(r.get("latency_ms", 0.0)) > 0.0]

    return {
        "seq_id": seq_id,
        "recording_group": seq_meta.get("recording_group") if seq_meta else None,
        "attributes": seq_meta.get("attributes") if seq_meta else None,
        "primary_challenge_tags": seq_meta.get("primary_challenge_tags") if seq_meta else None,
        "level1_sequence_metrics": {
            "total_frames": total_frames,
            "sa_denominator_frames": sa_denom,
            "present_valid_frames": n_present,
            "absent_frames": n_absent,
            "ignored_invalid_gt_frames": len(invalid_gt_rows),
            "sa_score": round(seq_sa, 6),
            "mean_iou_present": round(mean_iou_exist, 6),
            "zero_overlap_rate": round(state_counts["ZERO_OVERLAP_LOST"] / n_present, 6) if n_present > 0 else 0.0,
            "primary_failure_rate_tau010": round(state_counts["FAILURE_FRAME"] / n_present, 6) if n_present > 0 else 0.0,
            "imprecise_localization_rate": round(state_counts["IMPRECISE_LOCALIZATION"] / n_present, 6) if n_present > 0 else 0.0,
            "high_quality_lock_rate": round(state_counts["HIGH_QUALITY_LOCK"] / n_present, 6) if n_present > 0 else 0.0,
            "precision_rate_20px": round(pr_20_hits / n_present, 6) if n_present > 0 else 0.0,
            "norm_precision_rate_050": round(npr_05_hits / n_present, 6) if n_present > 0 else 0.0,
            "mean_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 3) if latencies_ms else None,
            "sa_loss_decomposition": {
                "total_sa_loss": round(1.0 - seq_sa, 6),
                "failure_frames_sa_loss": round(fail_sa_loss_sum / sa_denom, 6) if sa_denom > 0 else 0.0,
                "imprecise_frames_sa_loss": round(imprecise_sa_loss_sum / sa_denom, 6) if sa_denom > 0 else 0.0,
                "high_quality_residual_sa_loss": round(high_sa_loss_sum / sa_denom, 6) if sa_denom > 0 else 0.0,
                "absent_false_positive_sa_loss": round(absent_sa_loss_sum / sa_denom, 6) if sa_denom > 0 else 0.0,
            },
        },
        "level2_frame_and_absence_analysis": {
            "present_frame_state_counts": state_counts,
            "absence_and_reacquisition": absence_reacq,
            "sensitivity_analysis": sensitivity_stats,
        },
        "level3_failure_episodes_summary": {
            "primary_criterion": "tau_0.10 (IoU < 0.10, bridge_gap <= 3)",
            "num_episodes": len(primary_episodes),
            "total_episode_present_frames": sum(ep_lens),
            "max_episode_duration_frames": max(ep_lens) if ep_lens else 0,
            "mean_episode_duration_frames": round(sum(ep_lens) / len(ep_lens), 3) if ep_lens else 0.0,
            "recovered_episodes_count": sum(1 for e in primary_episodes if e["recovery_status"] == "recovered"),
            "partial_recovery_episodes_count": sum(
                1 for e in primary_episodes if e["recovery_status"] == "partial_recovery_unstable"
            ),
            "unrecovered_terminal_drift_count": sum(
                1 for e in primary_episodes if e["recovery_status"] == "unrecovered_terminal_drift"
            ),
            "duration_bucket_counts": bucket_counts,
            "episodes": primary_episodes,
        },
    }

# ==============================================================================
# 分轨注册表与窗口判定常量（2026-11-04 修订冻结，见协议文档第三章 3.0 修订记录）
# ==============================================================================
PROTOCOL_VERSION: str = "v1.1-phase2-protocol-rev20261104"

GROUP_SUBJECT = "subject"
GROUP_IN_DOMAIN = "group1_in_domain"
GROUP_ZERO_SHOT = "group2_zero_shot"
GROUP_UNASSIGNED = "unassigned"
VALID_GROUPS = (GROUP_SUBJECT, GROUP_IN_DOMAIN, GROUP_ZERO_SHOT, GROUP_UNASSIGNED)

# 名称规范化（小写、去掉非字母数字）后按前缀匹配；未匹配的算法记为 unassigned，
# 不计入任何一轨的存活率，并在输出中显式列出（绝不静默归组）。
DEFAULT_TRACKER_GROUP_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("FocusTrack", GROUP_IN_DOMAIN),
    ("MemLoTrack", GROUP_IN_DOMAIN),
    ("DAM4SAM", GROUP_ZERO_SHOT),
    ("SAMOSA", GROUP_ZERO_SHOT),
    ("SAMURAI", GROUP_ZERO_SHOT),
    ("UETrack", GROUP_ZERO_SHOT),
    ("SUTrack", GROUP_ZERO_SHOT),
    ("UTPTrack", GROUP_ZERO_SHOT),
    ("ReFocus", GROUP_ZERO_SHOT),
)

SURVIVED_STATUSES: Tuple[str, ...] = ("SUCCESS", "ROBUST_SURVIVAL")
RECOVERED_STATUSES: Tuple[str, ...] = ("TRANSIENT_FAILURE_RECOVERED",)
FAILED_STATUSES: Tuple[str, ...] = ("FAILED_UNRECOVERED", "TRANSIENT_FAILURE_RECOVERED")

TYPE_C_IN_DOMAIN_SURVIVAL_GT: float = 0.5   # Type C：同域对手存活率严格 > 0.5（多数存活）
TYPE_D_IN_DOMAIN_FAILURE_GT: float = 0.5    # Type D：同域对手失败率严格 > 0.5
TYPE_D0_ZERO_SHOT_FAILURE_GT: float = 0.5   # Type D0：仅零样本对手失败率严格 > 0.5

WINDOW_TYPE_KEYS: Tuple[str, ...] = (
    "Type_A_Common_Hard_Failure",
    "Type_B_Rescued_SiamDT_Failure",
    "Type_C_SiamDT_Unique_Failure",
    "Type_D_InDomain_Peer_Failure_SiamDT_Success",
    "Type_D0_ZeroShot_Only_Peer_Failure_SiamDT_Success",
    "Type_E_All_Survived",
    "Type_U_No_Peer_Evidence",
    "Type_X_Subject_Not_Evaluated",
)

CLASSIFICATION_RULES_DOC: Dict[str, str] = {
    "survived": "窗口核心跨度内状态 ∈ {SUCCESS, ROBUST_SURVIVAL}",
    "failed": "窗口核心跨度内状态 ∈ {FAILED_UNRECOVERED, TRANSIENT_FAILURE_RECOVERED}",
    "rescuer": "对手 survived，或 (SiamDT 为 FAILED_UNRECOVERED 且对手为 TRANSIENT_FAILURE_RECOVERED)",
    "Type_X": "SiamDT_Baseline 在该窗口无逐帧记录",
    "Type_U": "两轨均无可用对手记录（不做 A/B/C/D 推断）",
    "Type_C": f"SiamDT failed 且 同域对手数 >= 1 且 in_domain_peer_survival_rate > {TYPE_C_IN_DOMAIN_SURVIVAL_GT}（只看同域轨，零样本轨结果不影响）",
    "Type_B": "SiamDT failed 且 非 Type_C 且 至少 1 个 rescuer（任一轨；rescue_tracks 记录来源轨）",
    "Type_A": "SiamDT failed 且 两轨均无 rescuer",
    "Type_D": f"SiamDT survived 且 同域对手数 >= 1 且 in_domain_peer_failure_rate > {TYPE_D_IN_DOMAIN_FAILURE_GT}",
    "Type_D0": f"SiamDT survived 且 非 Type_D 且 零样本对手数 >= 1 且 zero_shot_peer_failure_rate > {TYPE_D0_ZERO_SHOT_FAILURE_GT}",
    "Type_E": "SiamDT survived 且 非 D/D0",
}


class HoldoutSealedError(RuntimeError):
    """Holdout 封存护栏：默认拒绝读取 holdout 序列。"""


def _norm_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_tracker_groups(
    tracker_names: List[str],
    anchor_tracker: str,
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """返回 {tracker: group}；overrides 优先，其次前缀注册表，否则 unassigned。"""
    overrides = overrides or {}
    out: Dict[str, str] = {}
    for tk in tracker_names:
        if tk == anchor_tracker:
            out[tk] = GROUP_SUBJECT
            continue
        if tk in overrides:
            grp = overrides[tk]
            if grp not in VALID_GROUPS:
                raise ValueError(f"--tracker-groups 中 {tk} 的分组 {grp} 非法，应为 {VALID_GROUPS}")
            out[tk] = grp
            continue
        n = _norm_name(tk)
        grp = GROUP_UNASSIGNED
        for prefix, g in DEFAULT_TRACKER_GROUP_PREFIXES:
            if n.startswith(_norm_name(prefix)):
                grp = g
                break
        out[tk] = grp
    return out


def tracker_set_fingerprint(groups: Dict[str, str]) -> Dict[str, Any]:
    items = [{"tracker": tk, "group": groups[tk]} for tk in sorted(groups)]
    digest = hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"tracker_set": items, "tracker_set_sha256": digest}


def _is_valid_present(r: Dict[str, Any]) -> bool:
    gt = r.get("gt_bbox", [])
    return int(r.get("exist", 1)) == 1 and len(gt) >= 4 and float(gt[2]) > 0 and float(gt[3]) > 0


def evaluate_tracker_outcome_in_span(
    tk_rows: List[Dict[str, Any]], u_start: int, u_end: int, ctx_end: int,
) -> Dict[str, Any]:
    """在核心跨度 [u_start, u_end] 内评估单个算法状态（阈值与 v1.0 冻结协议完全相同）。"""
    w_rows = tk_rows[u_start: u_end + 1]
    w_present = [r for r in w_rows if _is_valid_present(r)]
    w_absent = [r for r in w_rows if int(r.get("exist", 1)) == 0]
    w_sa_valid = w_present + w_absent
    mean_sa = sum(float(r.get("sa_score", 0.0)) for r in w_sa_valid) / len(w_sa_valid) if w_sa_valid else 0.0
    mean_iou = sum(float(r.get("iou", 0.0)) for r in w_present) / len(w_present) if w_present else 0.0
    min_iou = min((float(r.get("iou", 0.0)) for r in w_present), default=0.0)
    fail_cnt = sum(1 for r in w_present if float(r.get("iou", 0.0)) < PRIMARY_TAU_FAIL)
    fail_rate = fail_cnt / len(w_present) if w_present else 0.0
    hq_cnt = sum(1 for r in w_present if float(r.get("iou", 0.0)) >= HIGH_QUALITY_IOU)
    hq_rate = hq_cnt / len(w_present) if w_present else 0.0
    absent_fp_cnt = sum(1 for r in w_absent if int(r.get("pred_absent_flag", 0)) == 0)

    tail_start = max(u_start, u_end - (RECOVERY_CONSECUTIVE_FRAMES - 1))
    tail_present = [r for r in tk_rows[tail_start: ctx_end + 1] if _is_valid_present(r)]
    req_k = min(RECOVERY_CONSECUTIVE_FRAMES, len(tail_present))
    post_locked = False
    if req_k > 0:
        for pos in range(0, len(tail_present) - req_k + 1):
            if all(float(x.get("iou", 0.0)) >= RECOVERY_IOU_THRESHOLD for x in tail_present[pos: pos + req_k]):
                post_locked = True
                break

    if not w_present:
        if mean_sa >= 0.80 and post_locked:
            status = "SUCCESS"
        elif post_locked:
            status = "ROBUST_SURVIVAL"
        else:
            status = "FAILED_UNRECOVERED"
    else:
        if fail_rate == 0.0 and mean_sa >= 0.50:
            status = "SUCCESS"
        elif fail_rate <= 0.15 and (mean_sa >= 0.35 or post_locked):
            status = "ROBUST_SURVIVAL"
        elif post_locked:
            status = "TRANSIENT_FAILURE_RECOVERED"
        else:
            status = "FAILED_UNRECOVERED"
    return {
        "status": status,
        "window_mean_sa": round(mean_sa, 6),
        "window_mean_iou_present": round(mean_iou, 6),
        "window_min_iou_present": round(min_iou, 6),
        "window_failure_rate_tau010": round(fail_rate, 6),
        "window_high_quality_rate": round(hq_rate, 6),
        "absent_false_positives": absent_fp_cnt,
        "post_window_locked": post_locked,
    }


def summarize_peer_tracks(
    per_tracker_outcomes: Dict[str, Dict[str, Any]], groups: Dict[str, str],
) -> Dict[str, Any]:
    """按同域 / 零样本两轨分别统计对手存活、恢复、失败与最佳对手。"""
    def _track(group: str) -> Dict[str, Any]:
        names = sorted(tk for tk, g in groups.items() if g == group and tk in per_tracker_outcomes)
        survived = [tk for tk in names if per_tracker_outcomes[tk]["status"] in SURVIVED_STATUSES]
        recovered = [tk for tk in names if per_tracker_outcomes[tk]["status"] in RECOVERED_STATUSES]
        failed = [tk for tk in names if per_tracker_outcomes[tk]["status"] in FAILED_STATUSES]
        n = len(names)
        best = max(names, key=lambda tk: per_tracker_outcomes[tk]["window_mean_sa"]) if names else None
        return {
            "evaluated": names,
            "survived": survived,
            "recovered": recovered,
            "failed": failed,
            "survival_rate": round(len(survived) / n, 6) if n else None,
            "failure_rate": round(len(failed) / n, 6) if n else None,
            "best_peer": (
                {"tracker": best, "group": group, "status": per_tracker_outcomes[best]["status"],
                 "window_mean_sa": per_tracker_outcomes[best]["window_mean_sa"]} if best else None
            ),
        }

    ind = _track(GROUP_IN_DOMAIN)
    zs = _track(GROUP_ZERO_SHOT)
    cands = [b for b in (ind["best_peer"], zs["best_peer"]) if b]
    best_overall = max(cands, key=lambda b: b["window_mean_sa"]) if cands else None
    unassigned = sorted(tk for tk, g in groups.items() if g == GROUP_UNASSIGNED and tk in per_tracker_outcomes)
    return {"in_domain": ind, "zero_shot": zs, "best_peer": best_overall, "unassigned_excluded": unassigned}


def classify_window_type(
    anchor_out: Optional[Dict[str, Any]],
    per_tracker_outcomes: Dict[str, Dict[str, Any]],
    groups: Dict[str, str],
) -> Dict[str, Any]:
    """分轨判定 Type A/B/C/D/D0/E/U/X（规则见 CLASSIFICATION_RULES_DOC）。"""
    tracks = summarize_peer_tracks(per_tracker_outcomes, groups)
    ind, zs = tracks["in_domain"], tracks["zero_shot"]
    n_in, n_zs = len(ind["evaluated"]), len(zs["evaluated"])
    rescuers: List[str] = []
    rescue_tracks: List[str] = []
    if anchor_out is None:
        wtype = "Type_X_Subject_Not_Evaluated"
    elif n_in == 0 and n_zs == 0:
        wtype = "Type_U_No_Peer_Evidence"
    elif anchor_out["status"] in FAILED_STATUSES:
        for label, tr in (("in_domain", ind), ("zero_shot", zs)):
            for tk in tr["evaluated"]:
                st = per_tracker_outcomes[tk]["status"]
                if st in SURVIVED_STATUSES or (anchor_out["status"] == "FAILED_UNRECOVERED" and st in RECOVERED_STATUSES):
                    rescuers.append(tk)
                    if label not in rescue_tracks:
                        rescue_tracks.append(label)
        if n_in >= 1 and ind["survival_rate"] is not None and ind["survival_rate"] > TYPE_C_IN_DOMAIN_SURVIVAL_GT:
            wtype = "Type_C_SiamDT_Unique_Failure"
        elif rescuers:
            wtype = "Type_B_Rescued_SiamDT_Failure"
        else:
            wtype = "Type_A_Common_Hard_Failure"
    else:
        if n_in >= 1 and ind["failure_rate"] is not None and ind["failure_rate"] > TYPE_D_IN_DOMAIN_FAILURE_GT:
            wtype = "Type_D_InDomain_Peer_Failure_SiamDT_Success"
        elif n_zs >= 1 and zs["failure_rate"] is not None and zs["failure_rate"] > TYPE_D0_ZERO_SHOT_FAILURE_GT:
            wtype = "Type_D0_ZeroShot_Only_Peer_Failure_SiamDT_Success"
        else:
            wtype = "Type_E_All_Survived"
    return {
        "window_diagnostic_type": wtype,
        "rescuing_trackers": sorted(rescuers),
        "rescue_tracks": rescue_tracks,
        "type_c_evidence_note": ("single_in_domain_peer" if n_in == 1 else ("no_in_domain_peer" if n_in == 0 else "multi_in_domain_peers")),
        "in_domain_peer_survival_rate": ind["survival_rate"],
        "zero_shot_peer_survival_rate": zs["survival_rate"],
        "in_domain_peer_failure_rate": ind["failure_rate"],
        "zero_shot_peer_failure_rate": zs["failure_rate"],
        "best_peer": tracks["best_peer"],
        "best_in_domain_peer": ind["best_peer"],
        "best_zero_shot_peer": zs["best_peer"],
        "peer_evidence": {
            "in_domain": {k: ind[k] for k in ("evaluated", "survived", "recovered", "failed")},
            "zero_shot": {k: zs[k] for k in ("evaluated", "survived", "recovered", "failed")},
            "unassigned_excluded": tracks["unassigned_excluded"],
        },
    }


# ==============================================================================
# 窗口构建（主口径 Subject-Anchored + 补充视图 Pool-Union）
# ==============================================================================
def _merge_seed_intervals(seed_intervals: List[Tuple[int, int, str]], merge_gap: int) -> List[Dict[str, Any]]:
    seed_intervals = sorted(seed_intervals, key=lambda x: (x[0], x[1]))
    merged: List[Dict[str, Any]] = []
    for s_start, s_end, tag in seed_intervals:
        if merged and s_start <= merged[-1]["u_end"] + merge_gap + 1:
            merged[-1]["u_end"] = max(merged[-1]["u_end"], s_end)
            if tag not in merged[-1]["triggers"]:
                merged[-1]["triggers"].append(tag)
        else:
            merged.append({"u_start": s_start, "u_end": s_end, "triggers": [tag]})
    return merged


def _gt_absence_seed_intervals(ref_rows: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
    seeds: List[Tuple[int, int, str]] = []
    total_f = len(ref_rows)
    i = 0
    while i < total_f:
        if int(ref_rows[i].get("exist", 1)) == 0:
            a_start = int(ref_rows[i].get("frame_id", i))
            while i < total_f and int(ref_rows[i].get("exist", 1)) == 0:
                i += 1
            a_end = int(ref_rows[i - 1].get("frame_id", i - 1))
            seeds.append((a_start, min(total_f - 1, a_end + ABSENT_REENTRY_SEED_HORIZON), "gt_absence_reentry"))
        else:
            i += 1
    return seeds


def _manifest_sequence_details(split_manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {it["seq_name"]: it for it in split_manifest.get("sequence_details", []) if "seq_name" in it}


def _build_windows_for_spans(
    seq_id: str,
    spans: List[Dict[str, Any]],
    ref_rows: List[Dict[str, Any]],
    tracker_seq_rows: Dict[str, Dict[str, List[Dict[str, Any]]]],
    groups: Dict[str, str],
    anchor_tracker: str,
    s_meta: Dict[str, Any],
    pre_pad: int,
    post_pad: int,
    id_tag: str,
) -> List[Dict[str, Any]]:
    total_f = len(ref_rows)
    windows: List[Dict[str, Any]] = []
    for w_idx, sp in enumerate(spans, start=1):
        u_start, u_end = int(sp["u_start"]), int(sp["u_end"])
        ctx_start, ctx_end = max(1, u_start - pre_pad), min(total_f - 1, u_end + post_pad)
        core_rows = ref_rows[u_start: u_end + 1]
        core_present = [r for r in core_rows if _is_valid_present(r)]
        core_sizes = [float(r.get("target_size", 0.0)) for r in core_present if float(r.get("target_size", 0.0)) > 0]
        outcomes: Dict[str, Dict[str, Any]] = {}
        for tk in sorted(tracker_seq_rows):
            tk_rows = tracker_seq_rows[tk].get(seq_id)
            if not tk_rows or len(tk_rows) <= u_end:
                continue
            outcomes[tk] = {**evaluate_tracker_outcome_in_span(tk_rows, u_start, u_end, ctx_end), "group": groups.get(tk, GROUP_UNASSIGNED)}
        cls = classify_window_type(outcomes.get(anchor_tracker), outcomes, groups)
        windows.append({
            "window_id": f"{seq_id}__{id_tag}{w_idx:03d}",
            "seq_id": seq_id,
            "window_index_in_seq": w_idx,
            "recording_group": s_meta.get("recording_group"),
            "sequence_attributes": s_meta.get("attributes"),
            "primary_challenge_tags": s_meta.get("primary_challenge_tags"),
            "core_challenge_span": [u_start, u_end],
            "core_duration_frames": u_end - u_start + 1,
            "core_present_frames": len(core_present),
            "core_absent_frames": sum(1 for r in core_rows if int(r.get("exist", 1)) == 0),
            "core_mean_target_size": round(sum(core_sizes) / len(core_sizes), 4) if core_sizes else 0.0,
            "padded_context_span": [ctx_start, ctx_end],
            "pre_context_span": [ctx_start, max(ctx_start, u_start - 1)],
            "post_observation_span": [min(ctx_end, u_end + 1), ctx_end],
            "trigger_sources": sp["triggers"],
            **cls,
            "per_tracker_outcomes": outcomes,
        })
    return windows


def _windows_payload(
    mode: str,
    windows: List[Dict[str, Any]],
    split_manifest: Dict[str, Any],
    anchor_tracker: str,
    groups: Dict[str, str],
    merge_gap: int,
    pre_pad: int,
    post_pad: int,
    boundary_note: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    counts = {k: 0 for k in WINDOW_TYPE_KEYS}
    for w in windows:
        counts[w["window_diagnostic_type"]] = counts.get(w["window_diagnostic_type"], 0) + 1
    span_sig = [[w["window_id"], w["core_challenge_span"], w["padded_context_span"]] for w in windows]
    fp = tracker_set_fingerprint(groups)
    payload = {
        "schema_version": "1.1.0",
        "protocol_version": PROTOCOL_VERSION,
        "window_mode": mode,
        "is_primary_view": mode == "subject_anchored",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_name": split_manifest.get("split_name", "unknown"),
        "anchor_tracker": anchor_tracker,
        "window_boundary_determined_by": boundary_note,
        "evaluated_trackers": sorted(groups),
        "tracker_groups": groups,
        **fp,
        "unassigned_trackers_excluded_from_tracks": sorted(tk for tk, g in groups.items() if g == GROUP_UNASSIGNED),
        "window_set_sha256": hashlib.sha256(json.dumps(span_sig, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "window_construction_params": {
            "primary_tau_fail": PRIMARY_TAU_FAIL,
            "episode_bridge_gap": BRIDGE_GAP_FRAMES,
            "canonical_merge_gap": merge_gap,
            "pre_context_pad": pre_pad,
            "post_observation_pad": post_pad,
            "recovery_consecutive_frames": RECOVERY_CONSECUTIVE_FRAMES,
            "recovery_iou_threshold": RECOVERY_IOU_THRESHOLD,
        },
        "classification_rules": CLASSIFICATION_RULES_DOC,
        "summary": {
            "total_windows": len(windows),
            "sequences_with_windows": len({w["seq_id"] for w in windows}),
            "window_type_counts": counts,
        },
        "windows": windows,
    }
    if extra:
        payload.update(extra)
    return payload


def build_subject_anchored_windows(
    tracker_seq_rows: Dict[str, Dict[str, List[Dict[str, Any]]]],
    tracker_seq_analysis: Dict[str, Dict[str, Dict[str, Any]]],
    split_manifest: Dict[str, Any],
    anchor_tracker: str = "SiamDT_Baseline",
    tracker_groups: Optional[Dict[str, str]] = None,
    merge_gap: int = CANONICAL_MERGE_GAP,
    pre_pad: int = CANONICAL_PRE_PAD,
    post_pad: int = CANONICAL_POST_PAD,
) -> Dict[str, Any]:
    """
    【主口径】Subject-Anchored Windows：窗口边界只由 anchor_tracker（SiamDT_Baseline）自身的主失败 episode
    （τ=0.10、g_bridge=3，间隔 <= merge_gap 合并）加 [-pre_pad, +post_pad] 上下文决定，与参与算法集合无关。
    """
    groups = tracker_groups or resolve_tracker_groups(sorted(tracker_seq_rows), anchor_tracker)
    seq_details = _manifest_sequence_details(split_manifest)
    windows: List[Dict[str, Any]] = []
    missing_subject: List[str] = []
    for seq_id in split_manifest.get("sequences", []):
        ref_rows = tracker_seq_rows.get(anchor_tracker, {}).get(seq_id)
        seq_ana = tracker_seq_analysis.get(anchor_tracker, {}).get(seq_id)
        if not ref_rows or not seq_ana:
            missing_subject.append(seq_id)
            continue
        seeds = [(int(ep["t_start"]), int(ep["t_end"]), f"failure:{anchor_tracker}")
                 for ep in seq_ana["level3_failure_episodes_summary"]["episodes"]]
        if not seeds:
            continue
        spans = _merge_seed_intervals(seeds, merge_gap)
        windows.extend(_build_windows_for_spans(seq_id, spans, ref_rows, tracker_seq_rows, groups, anchor_tracker,
                                                seq_details.get(seq_id, {}), pre_pad, post_pad, "SAW"))
    return _windows_payload(
        "subject_anchored", windows, split_manifest, anchor_tracker, groups, merge_gap, pre_pad, post_pad,
        boundary_note=f"仅 {anchor_tracker} 自身主失败 episode（与参与算法集合无关）",
        extra={"sequences_missing_subject_telemetry": missing_subject},
    )


def build_pool_union_challenge_windows(
    tracker_seq_rows: Dict[str, Dict[str, List[Dict[str, Any]]]],
    tracker_seq_analysis: Dict[str, Dict[str, Dict[str, Any]]],
    split_manifest: Dict[str, Any],
    anchor_tracker: str = "SiamDT_Baseline",
    tracker_groups: Optional[Dict[str, str]] = None,
    merge_gap: int = CANONICAL_MERGE_GAP,
    pre_pad: int = CANONICAL_PRE_PAD,
    post_pad: int = CANONICAL_POST_PAD,
) -> Dict[str, Any]:
    """【补充视图】Pool-Union：全部参与算法的失败 episode + GT 消失重入区间取并集（会随 tracker_set 变化，见 tracker_set_sha256）。"""
    groups = tracker_groups or resolve_tracker_groups(sorted(tracker_seq_rows), anchor_tracker)
    seq_details = _manifest_sequence_details(split_manifest)
    windows: List[Dict[str, Any]] = []
    for seq_id in split_manifest.get("sequences", []):
        ref_rows = tracker_seq_rows.get(anchor_tracker, {}).get(seq_id)
        if not ref_rows:
            ref_rows = next((tracker_seq_rows[tk][seq_id] for tk in sorted(tracker_seq_rows) if seq_id in tracker_seq_rows[tk]), None)
        if not ref_rows:
            continue
        seeds: List[Tuple[int, int, str]] = []
        for tk in sorted(tracker_seq_rows):
            seq_ana = tracker_seq_analysis.get(tk, {}).get(seq_id)
            if seq_ana:
                seeds.extend((int(ep["t_start"]), int(ep["t_end"]), f"failure:{tk}")
                             for ep in seq_ana["level3_failure_episodes_summary"]["episodes"])
        seeds.extend(_gt_absence_seed_intervals(ref_rows))
        if not seeds:
            continue
        spans = _merge_seed_intervals(seeds, merge_gap)
        windows.extend(_build_windows_for_spans(seq_id, spans, ref_rows, tracker_seq_rows, groups, anchor_tracker,
                                                seq_details.get(seq_id, {}), pre_pad, post_pad, "PUW"))
    return _windows_payload(
        "pool_union_supplementary", windows, split_manifest, anchor_tracker, groups, merge_gap, pre_pad, post_pad,
        boundary_note="全部参与算法失败 episode ∪ GT 消失重入区间（补充视图；随 tracker_set 漂移）",
    )


# 向后兼容：旧名称指向主口径（Subject-Anchored）
build_canonical_challenge_windows = build_subject_anchored_windows


# ==============================================================================
# 划分清单解析与 Holdout 封存护栏
# ==============================================================================
def resolve_split_sequences(split_manifest: Dict[str, Any]) -> Tuple[List[str], set]:
    """返回 (序列列表, 清单自身标注为 holdout 的序列集合)。兼容 discovery/holdout（sequences 字段）与 val90_full（分区字段）。"""
    role = str(split_manifest.get("split_role", "")).lower()
    name = str(split_manifest.get("split_name", "")).lower()
    if "sequences" in split_manifest:
        seqs = [str(s) for s in split_manifest.get("sequences", [])]
        own_holdout = set(seqs) if ("holdout" in role or "holdout" in name) else set()
        return seqs, own_holdout
    disc = [str(s) for s in split_manifest.get("discovery_full_54", {}).get("sequences", [])]
    hold = [str(s) for s in split_manifest.get("holdout_full_36", {}).get("sequences", [])]
    return disc + hold, set(hold)


def load_holdout_set_for_split(split_json_path: Path, holdout_json: Optional[Path] = None) -> Tuple[Optional[set], List[str]]:
    """holdout 参考集合 = holdout_verification.json（显式指定或与划分清单同目录）∪ 同目录 val90_full.json 的 holdout_full_36。"""
    hj = holdout_json if holdout_json else split_json_path.parent / "holdout_verification.json"
    if not hj.exists():
        return None, []
    seqs = set(str(s) for s in json.loads(hj.read_text(encoding="utf-8")).get("sequences", []))
    sources = [str(hj)]
    v90 = hj.parent / "val90_full.json"
    if v90.exists():
        seqs |= set(str(s) for s in json.loads(v90.read_text(encoding="utf-8")).get("holdout_full_36", {}).get("sequences", []))
        sources.append(str(v90))
    return seqs, sources


def apply_holdout_guard(
    split_json_path: Path,
    split_manifest: Dict[str, Any],
    unlock_holdout: bool,
    holdout_json: Optional[Path] = None,
) -> Dict[str, Any]:
    all_seqs, own_holdout = resolve_split_sequences(split_manifest)
    ref_set, sources = load_holdout_set_for_split(split_json_path, holdout_json)
    if ref_set is None and not unlock_holdout:
        raise HoldoutSealedError(
            f"未在 {split_json_path.parent} 找到 holdout_verification.json，无法确认划分中是否含 Holdout 序列，按封存规则拒绝执行；"
            "请用 --holdout-json 指定，或确认需要解封时显式传入 --unlock-holdout。"
        )
    holdout_set = (ref_set or set()) | own_holdout
    holdout_in_split = [s for s in all_seqs if s in holdout_set]
    if holdout_in_split and not unlock_holdout:
        if len(holdout_in_split) == len(all_seqs):
            raise HoldoutSealedError(
                f"划分 {split_manifest.get('split_name')} 全部为 Holdout 序列（{len(all_seqs)} 条），默认拒绝读取；如确需解封请显式传入 --unlock-holdout。"
            )
        targets = [s for s in all_seqs if s not in holdout_set]
        excluded = holdout_in_split
    else:
        targets, excluded = list(all_seqs), []
    return {
        "unlock_holdout": bool(unlock_holdout),
        "holdout_reference_sources": sources,
        "holdout_reference_found": ref_set is not None,
        "target_sequences": targets,
        "excluded_holdout_sequences": excluded,
        "holdout_sequences_included": [s for s in targets if s in holdout_set],
        "holdout_unsealed": bool(unlock_holdout and any(s in holdout_set for s in targets)) or (unlock_holdout and ref_set is None),
    }


# ==============================================================================
# 预测框 (.txt) 到标准逐帧遥测 (.jsonl) 转换与加载工具
# ==============================================================================
def build_telemetry_rows_from_boxes(
    seq_id: str,
    pred_bboxes: List[List[float]],
    gt_rects: List[List[float]],
    exist_flags: List[int],
    confidences: Optional[List[float]] = None,
    template_updates: Optional[List[bool]] = None,
    latencies_ms: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    将任意跟踪器的预测框序列转换为符合 Phase 2 协议的标准逐帧遥测行列表。
    支持：
    - 若某帧预测框 w <= 0 或 h <= 0，自动标记 pred_absent_flag = 1；
    - 严格按 Anti-UAV410 官方公式计算 iou, sa_score, center_error, norm_center_error, target_size。
    """
    total_f = min(len(pred_bboxes), len(gt_rects))
    rows: List[Dict[str, Any]] = []
    for f in range(total_f):
        pb = [round(float(x), 4) for x in pred_bboxes[f][:4]] if len(pred_bboxes[f]) >= 4 else [0.0, 0.0, 0.0, 0.0]
        gb_raw = gt_rects[f] if isinstance(gt_rects[f], list) and len(gt_rects[f]) >= 4 else [0.0, 0.0, 0.0, 0.0]
        gb = [round(float(x), 4) for x in gb_raw[:4]]
        ex = int(exist_flags[f]) if f < len(exist_flags) else 1

        pred_absent = 1 if (pb[2] <= 0.0 or pb[3] <= 0.0) else 0
        if ex == 1 and gb[2] > 0.0 and gb[3] > 0.0:
            iou_f = calc_iou_xywh(pb, gb) if pred_absent == 0 else 0.0
            sa_f = iou_f
            if pred_absent == 0:
                cx_p, cy_p = pb[0] + 0.5 * pb[2], pb[1] + 0.5 * pb[3]
                cx_g, cy_g = gb[0] + 0.5 * gb[2], gb[1] + 0.5 * gb[3]
                ce_f = math.hypot(cx_p - cx_g, cy_p - cy_g)
                ts_f = math.sqrt(max(1e-6, gb[2] * gb[3]))
                nce_f = ce_f / ts_f
            else:
                ce_f = -1.0
                ts_f = math.sqrt(max(1e-6, gb[2] * gb[3]))
                nce_f = -1.0
        else:
            iou_f = 0.0
            sa_f = 1.0 if pred_absent == 1 else 0.0
            ce_f = -1.0
            nce_f = -1.0
            ts_f = 0.0

        conf_f = float(confidences[f]) if (confidences and f < len(confidences)) else (1.0 if f == 0 else 0.0)
        up_f = bool(template_updates[f]) if (template_updates and f < len(template_updates)) else (f == 0)
        lat_f = float(latencies_ms[f]) if (latencies_ms and f < len(latencies_ms)) else 0.0

        rows.append({
            "seq_id": seq_id,
            "frame_id": f,
            "exist": ex,
            "gt_bbox": gb,
            "pred_bbox": pb,
            "pred_absent_flag": pred_absent,
            "iou": round(iou_f, 6),
            "sa_score": round(sa_f, 6),
            "center_error": round(ce_f, 4),
            "norm_center_error": round(nce_f, 4),
            "target_size": round(ts_f, 4),
            "confidence": round(conf_f, 6),
            "template_updated": up_f,
            "latency_ms": round(lat_f, 3),
        })
    return rows


def load_jsonl_telemetry(jsonl_path: Path) -> List[Dict[str, Any]]:
    """读取单个 <seq_id>.jsonl 逐帧遥测文件。"""
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # 补齐可能缺失的派生字段（向后兼容早期冒烟测试生成的 jsonl）
            if "pred_absent_flag" not in row:
                pb = row.get("pred_bbox", [0, 0, 0, 0])
                row["pred_absent_flag"] = 1 if (len(pb) < 4 or float(pb[2]) <= 0 or float(pb[3]) <= 0) else 0
            if "norm_center_error" not in row:
                ce = float(row.get("center_error", -1.0))
                ts = float(row.get("target_size", 0.0))
                row["norm_center_error"] = round(ce / ts, 4) if (ce >= 0.0 and ts > 0.0) else -1.0
            rows.append(row)
    return rows



# ==============================================================================
# 批量目录扫描与顶层数据库构建入口
# ==============================================================================
SENSITIVITY_CRITERIA: Tuple[str, ...] = (
    "tau_0.00", "tau_0.10", "tau_0.30", "tau_0.50", "nce_gt_1.0", "composite_iou0.10_or_nce1.0",
)


def _discover_tracker_dirs(telemetry_root: Path, split_name: str, split_stem: str, anchor_tracker: str) -> Dict[str, Path]:
    tracker_dirs: Dict[str, Path] = {}
    if not telemetry_root.exists():
        return tracker_dirs
    if list(telemetry_root.glob("*.jsonl")):
        tracker_dirs[anchor_tracker] = telemetry_root
        return tracker_dirs
    own_suffixes = tuple(sorted({f"_{split_name}", f"_{split_stem}"}, key=len, reverse=True))
    other_suffixes = ("_val90_full", "_holdout_verification", "_discovery_set")
    for sub in sorted(telemetry_root.iterdir()):
        if not sub.is_dir():
            continue
        dn = sub.name
        own = next((s for s in own_suffixes if dn.endswith(s)), None)
        if own:
            tk_name = dn[: -len(own)]
        elif any(dn.endswith(s) for s in other_suffixes):
            continue  # 属于其他划分（含 holdout 目录）——不读取
        else:
            tk_name = dn
        if list(sub.glob("*.jsonl")):
            tracker_dirs[tk_name] = sub
    return tracker_dirs


def run_phase2_extraction_pipeline(
    telemetry_root: Path,
    split_json_path: Path,
    output_dir: Path,
    anchor_tracker: str = "SiamDT_Baseline",
    unlock_holdout: bool = False,
    holdout_json: Optional[Path] = None,
    tracker_groups_override: Optional[Dict[str, str]] = None,
) -> Tuple[Path, Path]:
    """
    输出（返回值仍为 (db_path, cw_path) 以兼容调用方）：
    1. FAILURE_EPISODE_DATABASE.json
    2. CANONICAL_CHALLENGE_WINDOWS.json —— 主口径 Subject-Anchored Windows
    3. POOL_UNION_CHALLENGE_WINDOWS.json —— 补充视图（含 tracker_set 及其 hash）
    Holdout：默认拒绝读取；解封后输出目录改为 <output_dir>__HOLDOUT_UNSEALED，与 discovery 结果分开。
    """
    split_manifest = json.loads(split_json_path.read_text(encoding="utf-8"))
    split_name = split_manifest.get("split_name", split_json_path.stem)
    guard = apply_holdout_guard(split_json_path, split_manifest, unlock_holdout, holdout_json)
    target_sequences: List[str] = guard["target_sequences"]
    if guard["excluded_holdout_sequences"]:
        print(f"[Holdout 封存] 已排除 {len(guard['excluded_holdout_sequences'])} 条 Holdout 序列（未读取其遥测）。", flush=True)
    seq_meta_map = _manifest_sequence_details(split_manifest)
    filtered_manifest = {"split_name": split_name, "sequences": target_sequences,
                         "sequence_details": [seq_meta_map[s] for s in target_sequences if s in seq_meta_map]}

    tracker_dirs = _discover_tracker_dirs(telemetry_root, split_name, split_json_path.stem, anchor_tracker)
    groups = resolve_tracker_groups(sorted(tracker_dirs), anchor_tracker, tracker_groups_override)

    tracker_seq_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    tracker_seq_analysis: Dict[str, Dict[str, Dict[str, Any]]] = {}
    tracker_aggregates: Dict[str, Any] = {}

    for tk_name, tk_dir in tracker_dirs.items():
        tracker_seq_rows[tk_name] = {}
        tracker_seq_analysis[tk_name] = {}
        seq_sa_list: List[float] = []
        tot = {"present": 0, "absent": 0, "zero": 0, "fail": 0, "imprecise": 0, "hq": 0, "absent_fp": 0}
        all_episodes: List[Dict[str, Any]] = []
        agg_sens = {c: {"failed_present_frames": 0, "eval_present_frames": 0, "episode_count": 0,
                        "max_episode_duration": 0, "unrecovered_terminal_episodes": 0} for c in SENSITIVITY_CRITERIA}
        for seq_id in target_sequences:  # 只遍历已通过 holdout 护栏的序列
            jf = tk_dir / f"{seq_id}.jsonl"
            if not jf.exists():
                continue
            rows = load_jsonl_telemetry(jf)
            if not rows:
                continue
            tracker_seq_rows[tk_name][seq_id] = rows
            seq_ana = analyze_single_sequence(rows, seq_meta_map.get(seq_id))
            tracker_seq_analysis[tk_name][seq_id] = seq_ana
            l1 = seq_ana["level1_sequence_metrics"]
            l2 = seq_ana["level2_frame_and_absence_analysis"]
            l3 = seq_ana["level3_failure_episodes_summary"]
            seq_sa_list.append(float(l1["sa_score"]))
            tot["present"] += int(l1["present_valid_frames"])
            tot["absent"] += int(l1["absent_frames"])
            sc = l2["present_frame_state_counts"]
            tot["zero"] += int(sc["ZERO_OVERLAP_LOST"])
            tot["fail"] += int(sc["FAILURE_FRAME"])
            tot["imprecise"] += int(sc["IMPRECISE_LOCALIZATION"])
            tot["hq"] += int(sc["HIGH_QUALITY_LOCK"])
            tot["absent_fp"] += int(l2["absence_and_reacquisition"]["false_positive_in_absent_frames"])
            all_episodes.extend(l3["episodes"])
            eval_p_cnt = max(0, int(l1["present_valid_frames"]) - 1)
            for crit, cdict in l2["sensitivity_analysis"].items():
                agg_sens[crit]["failed_present_frames"] += int(cdict["failed_present_frames"])
                agg_sens[crit]["eval_present_frames"] += eval_p_cnt
                agg_sens[crit]["episode_count"] += int(cdict["episode_count"])
                agg_sens[crit]["max_episode_duration"] = max(agg_sens[crit]["max_episode_duration"], int(cdict["max_episode_duration"]))
                agg_sens[crit]["unrecovered_terminal_episodes"] += int(cdict["unrecovered_terminal_episodes"])

        ep_durations = [int(e["present_duration_frames"]) for e in all_episodes]
        buckets = ("Bucket_1_5_Transient", "Bucket_6_20_Short", "Bucket_21_100_Medium", "Bucket_101_Plus_Long_Drift")
        rate = lambda a, b: round(a / b, 6) if b > 0 else 0.0  # noqa: E731
        tracker_aggregates[tk_name] = {
            "tracker_name": tk_name,
            "tracker_group": groups.get(tk_name, GROUP_UNASSIGNED),
            "telemetry_source_dir": str(tk_dir),
            "evaluated_sequences_count": len(seq_sa_list),
            "target_sequences_count": len(target_sequences),
            "macro_mean_sa": round(sum(seq_sa_list) / len(seq_sa_list), 6) if seq_sa_list else 0.0,
            "total_present_frames": tot["present"],
            "total_absent_frames": tot["absent"],
            "overall_zero_overlap_rate": rate(tot["zero"], tot["present"]),
            "overall_primary_failure_rate_tau010": rate(tot["fail"], tot["present"]),
            "overall_imprecise_rate": rate(tot["imprecise"], tot["present"]),
            "overall_high_quality_rate": rate(tot["hq"], tot["present"]),
            "overall_absent_false_positive_rate": rate(tot["absent_fp"], tot["absent"]),
            "total_failure_episodes": len(all_episodes),
            "total_episode_present_frames": sum(ep_durations),
            "mean_episode_duration_frames": round(sum(ep_durations) / len(ep_durations), 3) if ep_durations else 0.0,
            "max_episode_duration_frames": max(ep_durations) if ep_durations else 0,
            "recovered_episodes_count": sum(1 for e in all_episodes if e["recovery_status"] == "recovered"),
            "unrecovered_terminal_drift_count": sum(1 for e in all_episodes if e["recovery_status"] == "unrecovered_terminal_drift"),
            "episode_duration_buckets": {b: sum(1 for e in all_episodes if e["duration_bucket"] == b) for b in buckets},
            "sensitivity_analysis_summary": {
                crit: {
                    "failed_present_frames": c["failed_present_frames"],
                    "failure_frame_rate": rate(c["failed_present_frames"], c["eval_present_frames"]),
                    "total_episodes": c["episode_count"],
                    "max_episode_duration": c["max_episode_duration"],
                    "unrecovered_terminal_episodes": c["unrecovered_terminal_episodes"],
                } for crit, c in agg_sens.items()
            },
            "per_sequence_analysis": tracker_seq_analysis[tk_name],
        }

    if guard["holdout_unsealed"]:
        output_dir = output_dir.with_name(output_dir.name + "__HOLDOUT_UNSEALED")
    output_dir.mkdir(parents=True, exist_ok=True)

    fp = tracker_set_fingerprint(groups)
    failure_db_payload = {
        "schema_version": "1.1.0",
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_name": split_name,
        "split_manifest_path": str(split_json_path),
        "holdout_guard": guard,
        "tracker_groups": groups,
        **fp,
        "frozen_protocol_parameters": {
            "primary_tau_fail": PRIMARY_TAU_FAIL,
            "imprecise_localization_band": [PRIMARY_TAU_FAIL, IMPRECISE_UPPER_IOU],
            "high_quality_lock_threshold": HIGH_QUALITY_IOU,
            "sensitivity_tau_set": list(SENSITIVITY_TAU_LIST),
            "normalized_center_error_threshold": PRIMARY_NCE_FAIL,
            "episode_bridge_gap_frames": BRIDGE_GAP_FRAMES,
            "pre_failure_context_frames": PRE_CONTEXT_FRAMES,
            "recovery_consecutive_frames": RECOVERY_CONSECUTIVE_FRAMES,
            "recovery_iou_threshold": RECOVERY_IOU_THRESHOLD,
        },
        "evaluated_trackers_count": len(tracker_aggregates),
        "trackers": tracker_aggregates,
    }
    subject_payload = build_subject_anchored_windows(tracker_seq_rows, tracker_seq_analysis, filtered_manifest, anchor_tracker, groups)
    pool_payload = build_pool_union_challenge_windows(tracker_seq_rows, tracker_seq_analysis, filtered_manifest, anchor_tracker, groups)
    for p in (subject_payload, pool_payload):
        p["holdout_guard"] = {k: guard[k] for k in ("unlock_holdout", "excluded_holdout_sequences", "holdout_sequences_included", "holdout_unsealed")}

    db_path = output_dir / "FAILURE_EPISODE_DATABASE.json"
    cw_path = output_dir / "CANONICAL_CHALLENGE_WINDOWS.json"
    pu_path = output_dir / "POOL_UNION_CHALLENGE_WINDOWS.json"
    db_path.write_text(json.dumps(failure_db_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    cw_path.write_text(json.dumps(subject_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pu_path.write_text(json.dumps(pool_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if guard["holdout_unsealed"]:
        (output_dir / "HOLDOUT_UNSEAL_RECORD.json").write_text(json.dumps({
            "unsealed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "split_manifest_path": str(split_json_path),
            "holdout_sequences_included": guard["holdout_sequences_included"],
            "note": "Holdout 结果已解封；严禁据此反向调整阈值或超参数（协议 1.1-3）。",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    return db_path, cw_path


# ==============================================================================
# 内置自检单元测试 (--self-test)
# ==============================================================================
def _make_synthetic_sequence() -> Tuple[str, List[List[float]], List[int]]:
    seq_id = "03_7951_1500-2999"
    total_f = 80
    gt_rects = [[100.0 + f * 1.0, 100.0, 20.0, 20.0] for f in range(total_f)]
    exist_flags = [1] * total_f
    for t in range(40, 45):
        exist_flags[t] = 0
        gt_rects[t] = [0.0, 0.0, 0.0, 0.0]
    return seq_id, gt_rects, exist_flags


def _run_self_test() -> None:
    report: Dict[str, Any] = {}
    seq_id, gt_rects, exist_flags = _make_synthetic_sequence()
    total_f = len(gt_rects)

    # ---------------- 1. 桥接 / 重入延迟（v1.0 原有用例，保持不变） ----------------
    pred_siamdt = [list(gt_rects[f]) if exist_flags[f] == 1 else [150.0, 150.0, 20.0, 20.0] for f in range(total_f)]
    for t in list(range(15, 19)) + list(range(21, 26)) + list(range(45, 48)):
        pred_siamdt[t] = [10.0, 10.0, 20.0, 20.0]
    for t in range(19, 21):
        pred_siamdt[t] = [gt_rects[t][0] + 12.0, gt_rects[t][1], 20.0, 20.0]
    rows_s = build_telemetry_rows_from_boxes(seq_id, pred_siamdt, gt_rects, exist_flags)
    ana_s = analyze_single_sequence(rows_s)
    eps = ana_s["level3_failure_episodes_summary"]["episodes"]
    assert len(eps) == 2, f"期望 2 个桥接后失败事件段，实际 {len(eps)}"
    assert eps[0]["t_start"] == 15 and eps[0]["t_end"] == 25 and eps[0]["bridged_transient_frames_count"] == 2
    assert eps[0]["recovery_status"] == "recovered" and eps[0]["recovery_latency_frames"] == 0
    reacq = ana_s["level2_frame_and_absence_analysis"]["absence_and_reacquisition"]
    assert reacq["false_positive_in_absent_frames"] == 5
    assert reacq["events"][0]["reacquired"] is True and reacq["events"][0]["reacquisition_latency_frames"] == 3
    report["1_bridging_and_reacquisition"] = {"episode_0": [eps[0]["t_start"], eps[0]["t_end"]], "reacq_latency": 3, "pass": True}

    # 对手：FocusTrack（同域）全程完美；DAM4SAM（零样本）在 [15,25] 与 [60,70] 失败（后者 SiamDT 正常）
    pred_focus = [list(gt_rects[f]) if exist_flags[f] == 1 else [0.0, 0.0, 0.0, 0.0] for f in range(total_f)]
    pred_dam = [list(gt_rects[f]) if exist_flags[f] == 1 else [0.0, 0.0, 0.0, 0.0] for f in range(total_f)]
    for t in list(range(15, 26)) + list(range(60, 71)):
        pred_dam[t] = [300.0, 300.0, 20.0, 20.0]
    rows_f = build_telemetry_rows_from_boxes(seq_id, pred_focus, gt_rects, exist_flags)
    rows_d = build_telemetry_rows_from_boxes(seq_id, pred_dam, gt_rects, exist_flags)
    manifest = {"split_name": "self_test", "sequences": [seq_id], "sequence_details": [{"seq_name": seq_id}]}
    rows2 = {"SiamDT_Baseline": {seq_id: rows_s}, "FocusTrack": {seq_id: rows_f}}
    ana2 = {"SiamDT_Baseline": {seq_id: ana_s}, "FocusTrack": {seq_id: analyze_single_sequence(rows_f)}}
    rows3 = {**rows2, "DAM4SAM": {seq_id: rows_d}}
    ana3 = {**ana2, "DAM4SAM": {seq_id: analyze_single_sequence(rows_d)}}

    # ---------------- 2. Subject-Anchored 窗口不随新增算法变化 ----------------
    sa2 = build_subject_anchored_windows(rows2, ana2, manifest)
    sa3 = build_subject_anchored_windows(rows3, ana3, manifest)
    sig = lambda p: [(w["window_id"], w["core_challenge_span"], w["padded_context_span"]) for w in p["windows"]]  # noqa: E731
    assert sig(sa2) == sig(sa3), f"Subject-anchored 窗口随算法集合漂移: {sig(sa2)} vs {sig(sa3)}"
    assert sa2["window_set_sha256"] == sa3["window_set_sha256"]
    assert sa2["tracker_set_sha256"] != sa3["tracker_set_sha256"]
    pu2 = build_pool_union_challenge_windows(rows2, ana2, manifest)
    pu3 = build_pool_union_challenge_windows(rows3, ana3, manifest)
    assert pu3["window_set_sha256"] != pu2["window_set_sha256"], "补充视图应体现 tracker_set 引起的窗口漂移"
    pu_sig = lambda p: [w["core_challenge_span"] for w in p["windows"]]  # noqa: E731
    report["2_subject_anchored_invariance"] = {
        "subject_windows_2_trackers": [list(map(str, s)) for s in sig(sa2)],
        "subject_windows_3_trackers": [list(map(str, s)) for s in sig(sa3)],
        "window_set_sha256_equal": True,
        "tracker_set_sha256_differs": True,
        "pool_union_core_spans_2_vs_3_trackers (补充视图随算法集合漂移)": [pu_sig(pu2), pu_sig(pu3)],
        "pass": True,
    }

    # ---------------- 3. 分轨分类 ----------------
    w0 = sa3["windows"][0]
    assert w0["core_challenge_span"] == [15, 25]
    assert w0["window_diagnostic_type"] == "Type_C_SiamDT_Unique_Failure", w0["window_diagnostic_type"]
    assert w0["in_domain_peer_survival_rate"] == 1.0 and w0["zero_shot_peer_survival_rate"] == 0.0
    assert w0["best_peer"]["tracker"] == "FocusTrack"
    g = {"SiamDT_Baseline": GROUP_SUBJECT, "FocusTrack": GROUP_IN_DOMAIN, "MemLoTrack": GROUP_IN_DOMAIN, "DAM4SAM": GROUP_ZERO_SHOT}

    def _o(status: str, sa: float = 0.5) -> Dict[str, Any]:
        return {"status": status, "window_mean_sa": sa}

    cases = {
        "B_zero_shot_rescue": ({"SiamDT_Baseline": _o("FAILED_UNRECOVERED"), "FocusTrack": _o("FAILED_UNRECOVERED"), "DAM4SAM": _o("SUCCESS", 0.9)},
                               "Type_B_Rescued_SiamDT_Failure", ["zero_shot"]),
        "A_all_failed": ({"SiamDT_Baseline": _o("FAILED_UNRECOVERED"), "FocusTrack": _o("FAILED_UNRECOVERED"), "DAM4SAM": _o("FAILED_UNRECOVERED")},
                         "Type_A_Common_Hard_Failure", []),
        "U_no_peers": ({"SiamDT_Baseline": _o("FAILED_UNRECOVERED")}, "Type_U_No_Peer_Evidence", []),
        "D_in_domain_failed": ({"SiamDT_Baseline": _o("SUCCESS"), "FocusTrack": _o("FAILED_UNRECOVERED"), "DAM4SAM": _o("SUCCESS")},
                               "Type_D_InDomain_Peer_Failure_SiamDT_Success", []),
        "D0_zero_shot_only_failed": ({"SiamDT_Baseline": _o("ROBUST_SURVIVAL"), "FocusTrack": _o("SUCCESS"), "DAM4SAM": _o("FAILED_UNRECOVERED")},
                                     "Type_D0_ZeroShot_Only_Peer_Failure_SiamDT_Success", []),
        "B_in_domain_half_not_majority": ({"SiamDT_Baseline": _o("FAILED_UNRECOVERED"), "FocusTrack": _o("SUCCESS"), "MemLoTrack": _o("FAILED_UNRECOVERED"),
                                           "DAM4SAM": _o("FAILED_UNRECOVERED")}, "Type_B_Rescued_SiamDT_Failure", ["in_domain"]),
        "C_despite_zero_shot_mass_failure": ({"SiamDT_Baseline": _o("TRANSIENT_FAILURE_RECOVERED"), "FocusTrack": _o("SUCCESS"), "MemLoTrack": _o("ROBUST_SURVIVAL"),
                                              "DAM4SAM": _o("FAILED_UNRECOVERED")}, "Type_C_SiamDT_Unique_Failure", ["in_domain"]),
        "X_subject_missing": ({"FocusTrack": _o("SUCCESS")}, "Type_X_Subject_Not_Evaluated", []),
    }
    case_out = {}
    for name, (outs, expect, exp_tracks) in cases.items():
        res = classify_window_type(outs.get("SiamDT_Baseline"), outs, {k: v for k, v in g.items() if k in outs})
        assert res["window_diagnostic_type"] == expect, f"{name}: {res['window_diagnostic_type']} != {expect}"
        assert res["rescue_tracks"] == exp_tracks, f"{name}: rescue_tracks {res['rescue_tracks']} != {exp_tracks}"
        case_out[name] = res["window_diagnostic_type"]
    assert resolve_tracker_groups(["SiamDT_Baseline", "UTPTrack-S", "MemLoTrack_scratch", "AiATrack"], "SiamDT_Baseline") == {
        "SiamDT_Baseline": GROUP_SUBJECT, "UTPTrack-S": GROUP_ZERO_SHOT, "MemLoTrack_scratch": GROUP_IN_DOMAIN, "AiATrack": GROUP_UNASSIGNED}
    report["3_track_separated_classification"] = {
        "end_to_end_window_0": {"type": w0["window_diagnostic_type"], "in_domain_survival": w0["in_domain_peer_survival_rate"],
                                "zero_shot_survival": w0["zero_shot_peer_survival_rate"], "best_peer": w0["best_peer"]["tracker"],
                                "note": "v1.0 规则（全体对手 >=80% 存活）下此窗口会被判为 Type_B；分轨后正确判为 Type_C"},
        "unit_cases": case_out,
        "unassigned_tracker_kept_out_of_tracks": "AiATrack -> unassigned",
        "pass": True,
    }

    # ---------------- 4. Holdout 默认拒绝 ----------------
    with tempfile.TemporaryDirectory(prefix="phase2_selftest_") as td:
        tdp = Path(td)
        splits = tdp / "splits"
        splits.mkdir()
        (splits / "holdout_verification.json").write_text(json.dumps(
            {"split_name": "holdout_verification", "split_role": "holdout_verification", "sequences": ["HOLD_SEQ"]}), encoding="utf-8")
        (splits / "mixed_val.json").write_text(json.dumps(
            {"split_name": "mixed_val", "discovery_full_54": {"sequences": ["DISC_SEQ"]}, "holdout_full_36": {"sequences": ["HOLD_SEQ"]}}), encoding="utf-8")
        tele = tdp / "telemetry" / "SiamDT_Baseline"
        tele.mkdir(parents=True)
        disc_rows = build_telemetry_rows_from_boxes("DISC_SEQ", pred_siamdt, gt_rects, exist_flags)
        (tele / "DISC_SEQ.jsonl").write_text("\n".join(json.dumps(r) for r in disc_rows) + "\n", encoding="utf-8")
        (tele / "HOLD_SEQ.jsonl").write_text("THIS FILE MUST NEVER BE READ UNLESS UNLOCKED\n", encoding="utf-8")
        out = tdp / "out" / "reports"

        refused_pure = False
        try:
            run_phase2_extraction_pipeline(tdp / "telemetry", splits / "holdout_verification.json", out)
        except HoldoutSealedError:
            refused_pure = True
        assert refused_pure, "纯 holdout 划分应被默认拒绝"

        db_p, cw_p = run_phase2_extraction_pipeline(tdp / "telemetry", splits / "mixed_val.json", out)
        db = json.loads(db_p.read_text(encoding="utf-8"))
        assert db["holdout_guard"]["excluded_holdout_sequences"] == ["HOLD_SEQ"]
        assert list(db["trackers"]["SiamDT_Baseline"]["per_sequence_analysis"].keys()) == ["DISC_SEQ"]
        assert db_p.parent == out, "未解封时应写入 discovery 输出目录"
        assert (out / "POOL_UNION_CHALLENGE_WINDOWS.json").exists()

        lonely = tdp / "lonely"
        lonely.mkdir()
        (lonely / "discovery_set.json").write_text(json.dumps({"split_name": "discovery_set", "sequences": ["DISC_SEQ"]}), encoding="utf-8")
        refused_missing = False
        try:
            run_phase2_extraction_pipeline(tdp / "telemetry", lonely / "discovery_set.json", out)
        except HoldoutSealedError:
            refused_missing = True
        assert refused_missing, "找不到 holdout 清单时应 fail-closed"

        tele2 = tdp / "telemetry_unlock" / "SiamDT_Baseline"
        tele2.mkdir(parents=True)
        (tele2 / "HOLD_SEQ.jsonl").write_text("\n".join(json.dumps({**r, "seq_id": "HOLD_SEQ"}) for r in disc_rows) + "\n", encoding="utf-8")
        db_u, _ = run_phase2_extraction_pipeline(tdp / "telemetry_unlock", splits / "holdout_verification.json", out, unlock_holdout=True)
        assert db_u.parent.name == "reports__HOLDOUT_UNSEALED", db_u.parent
        assert (db_u.parent / "HOLDOUT_UNSEAL_RECORD.json").exists()
        report["4_holdout_guard"] = {
            "pure_holdout_split_refused_by_default": True,
            "mixed_split_holdout_excluded_and_file_not_read": True,
            "fail_closed_when_holdout_manifest_missing": True,
            "unlock_writes_to_separate_dir": db_u.parent.name,
            "pass": True,
        }

    print(json.dumps({"self_test": "PASS", "protocol_version": PROTOCOL_VERSION, **report}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 三层失败提取器（主口径 Subject-Anchored Windows，分轨存活率，Holdout 封存）")
    parser.add_argument("--telemetry-root", type=Path,
                        default=Path("G:/我的云端硬盘/SiamDT_Project/experiments/failure_diagnosis/per_frame_telemetry"),
                        help="包含各算法逐帧遥测子目录 (<Tracker>_<Split>/<seq_id>.jsonl) 的根目录")
    parser.add_argument("--split-json", type=Path,
                        default=Path("Z:/SimDT围点打援/datasets/diagnostic_splits/discovery_set.json"),
                        help="冻结划分清单（discovery_set.json / val90_full.json；holdout 需 --unlock-holdout）")
    parser.add_argument("--output-dir", type=Path, default=Path("Z:/SimDT围点打援/phase2_unified_eval/outputs"),
                        help="输出目录（解封 holdout 时自动改为 <output-dir>__HOLDOUT_UNSEALED）")
    parser.add_argument("--anchor-tracker", type=str, default="SiamDT_Baseline", help="主体算法（窗口锚点）")
    parser.add_argument("--tracker-groups", type=Path, default=None,
                        help="可选 JSON：{tracker: group}，group ∈ group1_in_domain / group2_zero_shot / unassigned（覆盖内置前缀注册表）")
    parser.add_argument("--holdout-json", type=Path, default=None, help="holdout_verification.json 路径（默认取划分清单同目录）")
    parser.add_argument("--unlock-holdout", action="store_true", help="显式解封 Holdout 序列（结果写入独立目录并留下解封记录）")
    parser.add_argument("--self-test", action="store_true", help="运行内置确定性自检")
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        return
    overrides = json.loads(args.tracker_groups.read_text(encoding="utf-8")) if args.tracker_groups else None
    try:
        db_path, cw_path = run_phase2_extraction_pipeline(
            telemetry_root=args.telemetry_root, split_json_path=args.split_json, output_dir=args.output_dir,
            anchor_tracker=args.anchor_tracker, unlock_holdout=args.unlock_holdout,
            holdout_json=args.holdout_json, tracker_groups_override=overrides,
        )
    except HoldoutSealedError as exc:
        print(f"[Holdout 封存] {exc}", file=sys.stderr)
        sys.exit(3)
    print(json.dumps({
        "status": "COMPLETED",
        "failure_episode_database": str(db_path),
        "canonical_challenge_windows_subject_anchored": str(cw_path),
        "pool_union_challenge_windows_supplementary": str(cw_path.parent / "POOL_UNION_CHALLENGE_WINDOWS.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
