#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 三层失败提取器与跨算法统一挑战时间窗构建器 (failure_episode_extractor.py)
================================================================================
严格实现《PHASE2_FAILURE_DEFINITION_PROTOCOL.md》预先定义并冻结的判定标准：
1. Level 1 (Sequence): SA、存在帧 Mean IoU、ZOR、FR(0.10)、PR@20px、NPR@0.5、消失期误报率及 SA 失分四项可加拆解；
2. Level 2 (Frame):
   - exist == 1: ZERO_OVERLAP_LOST (IoU=0.0) / FAILURE_FRAME (IoU<0.10) / IMPRECISE (0.10<=IoU<0.50) / HIGH_QUALITY (IoU>=0.50)
   - exist == 0: false_positive_in_absent 统计 + Post-Absence Re-acquisition Latency (重入后连续 K_rec=5 帧 IoU>=0.50 延迟帧数)
   - 预注册敏感性分析: tau in {0.0, 0.10, 0.30, 0.50}, NCE > 1.0, 以及复合准则 (IoU < 0.10 or NCE > 1.0)
3. Level 3 (Failure Episode):
   - 允许桥接 <= g_bridge=3 帧瞬时偶发波动，聚合连续失败段 [t_start, t_end]
   - 提取前导退化窗口 Pre-failure Context [t_start-15, t_start-1] 与恢复状态判定 (Recovered vs. Partial vs. Unrecovered Terminal Drift)
4. Canonical Challenge Windows (跨算法统一挑战时间窗):
   - 将所有被测算法在同一序列上的 Failure Episodes 与真值消失重入窗口取时间并集（间隔 <= 5 帧合并）并扩展 [-15, +10] 上下文
   - 在完全对齐的时间窗内评估各算法的存活/自救/失锁状态，自动标注 Type A / Type B (Rescue) / Type C / Type D 事件。
"""

from __future__ import annotations

import argparse
import json
import math
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
# 跨算法统一挑战时间窗 (Canonical Challenge Windows) 构建器
# ==============================================================================
def build_canonical_challenge_windows(
    tracker_seq_rows: Dict[str, Dict[str, List[Dict[str, Any]]]],
    tracker_seq_analysis: Dict[str, Dict[str, Dict[str, Any]]],
    split_manifest: Dict[str, Any],
    anchor_tracker: str = "SiamDT_Baseline",
    merge_gap: int = CANONICAL_MERGE_GAP,
    pre_pad: int = CANONICAL_PRE_PAD,
    post_pad: int = CANONICAL_POST_PAD,
) -> Dict[str, Any]:
    """
    将所有已评测跟踪器在同一序列上的 Failure Episodes 与真值消失重入窗口取时间并集，
    生成对所有算法 100% 对齐的 Canonical Challenge Windows，并自动判定 각算法状态与 Type A/B/C/D。
    """
    seq_details_map: Dict[str, Dict[str, Any]] = {}
    for item in split_manifest.get("sequence_details", []):
        if "seq_name" in item:
            seq_details_map[item["seq_name"]] = item

    target_sequences: List[str] = split_manifest.get("sequences", sorted(seq_details_map.keys()))
    available_trackers = sorted(tracker_seq_rows.keys())

    all_canonical_windows: List[Dict[str, Any]] = []
    window_type_counts = {
        "Type_A_Common_Hard_Failure": 0,
        "Type_B_Rescued_SiamDT_Failure": 0,
        "Type_C_SiamDT_Unique_Failure": 0,
        "Type_D_Peer_Failure_SiamDT_Success": 0,
        "Type_E_All_Trackers_Succeeded_Absent_Window": 0,
    }

    for seq_id in target_sequences:
        # 找到任意一个拥有该序列逐帧记录的算法作为真值帧参考
        ref_rows: Optional[List[Dict[str, Any]]] = None
        if anchor_tracker in tracker_seq_rows and seq_id in tracker_seq_rows[anchor_tracker]:
            ref_rows = tracker_seq_rows[anchor_tracker][seq_id]
        else:
            for tk in available_trackers:
                if seq_id in tracker_seq_rows[tk]:
                    ref_rows = tracker_seq_rows[tk][seq_id]
                    break
        if not ref_rows:
            continue

        total_f = len(ref_rows)
        seed_intervals: List[Tuple[int, int, str]] = []

        # 1. 收集所有算法在该序列上的主失败事件段 [t_start, t_end]
        for tk in available_trackers:
            seq_ana = tracker_seq_analysis.get(tk, {}).get(seq_id)
            if not seq_ana:
                continue
            eps = seq_ana["level3_failure_episodes_summary"]["episodes"]
            for ep in eps:
                seed_intervals.append((int(ep["t_start"]), int(ep["t_end"]), f"failure:{tk}"))

        # 2. 收集真值目标消失与重入考题区间 [t_abs_start, min(T-1, t_abs_end + 15)]
        i = 0
        while i < total_f:
            if int(ref_rows[i].get("exist", 1)) == 0:
                a_start = int(ref_rows[i].get("frame_id", i))
                while i < total_f and int(ref_rows[i].get("exist", 1)) == 0:
                    i += 1
                a_end = int(ref_rows[i - 1].get("frame_id", i - 1))
                seed_end = min(total_f - 1, a_end + ABSENT_REENTRY_SEED_HORIZON)
                seed_intervals.append((a_start, seed_end, "gt_absence_reentry"))
            else:
                i += 1

        if not seed_intervals:
            continue

        # 3. 按起始帧排序并合并间隔 <= merge_gap (5) 帧的区间
        seed_intervals.sort(key=lambda x: (x[0], x[1]))
        merged_spans: List[Dict[str, Any]] = []
        for s_start, s_end, src_tag in seed_intervals:
            if not merged_spans:
                merged_spans.append({
                    "u_start": s_start,
                    "u_end": s_end,
                    "triggers": [src_tag],
                })
            else:
                prev = merged_spans[-1]
                if s_start <= prev["u_end"] + merge_gap + 1:
                    prev["u_end"] = max(prev["u_end"], s_end)
                    if src_tag not in prev["triggers"]:
                        prev["triggers"].append(src_tag)
                else:
                    merged_spans.append({
                        "u_start": s_start,
                        "u_end": s_end,
                        "triggers": [src_tag],
                    })

        # 4. 为每个合并后的核心跨度扩展上下文并计算各算法在窗口内的表现
        s_meta = seq_details_map.get(seq_id, {})
        for w_idx, mspan in enumerate(merged_spans, start=1):
            u_start = int(mspan["u_start"])
            u_end = int(mspan["u_end"])
            ctx_start = max(1, u_start - pre_pad)
            ctx_end = min(total_f - 1, u_end + post_pad)

            # 统计窗口内真值物理特征（尺寸、消失帧数、帧间位移）
            core_rows_ref = ref_rows[u_start : u_end + 1]
            core_present_ref = [
                r for r in core_rows_ref
                if int(r.get("exist", 1)) == 1
                and len(r.get("gt_bbox", [])) >= 4
                and float(r["gt_bbox"][2]) > 0
                and float(r["gt_bbox"][3]) > 0
            ]
            core_absent_cnt = sum(1 for r in core_rows_ref if int(r.get("exist", 1)) == 0)
            core_sizes = [float(r.get("target_size", 0.0)) for r in core_present_ref if float(r.get("target_size", 0.0)) > 0]

            per_tracker_outcomes: Dict[str, Any] = {}
            for tk in available_trackers:
                tk_rows = tracker_seq_rows.get(tk, {}).get(seq_id)
                if not tk_rows or len(tk_rows) <= u_end:
                    continue
                w_rows = tk_rows[u_start : u_end + 1]
                w_present = [
                    r for r in w_rows
                    if int(r.get("exist", 1)) == 1
                    and len(r.get("gt_bbox", [])) >= 4
                    and float(r["gt_bbox"][2]) > 0
                    and float(r["gt_bbox"][3]) > 0
                ]
                w_absent = [r for r in w_rows if int(r.get("exist", 1)) == 0]
                w_sa_valid = w_present + w_absent

                mean_sa = (
                    sum(float(r.get("sa_score", 0.0)) for r in w_sa_valid) / len(w_sa_valid)
                    if w_sa_valid else 0.0
                )
                mean_iou = (
                    sum(float(r.get("iou", 0.0)) for r in w_present) / len(w_present)
                    if w_present else 0.0
                )
                min_iou = min((float(r.get("iou", 0.0)) for r in w_present), default=0.0)
                fail_cnt = sum(1 for r in w_present if float(r.get("iou", 0.0)) < PRIMARY_TAU_FAIL)
                fail_rate = fail_cnt / len(w_present) if w_present else 0.0
                hq_cnt = sum(1 for r in w_present if float(r.get("iou", 0.0)) >= HIGH_QUALITY_IOU)
                hq_rate = hq_cnt / len(w_present) if w_present else 0.0
                absent_fp_cnt = sum(1 for r in w_absent if int(r.get("pred_absent_flag", 0)) == 0)

                # 检查窗口末端及后置观测区间 [max(u_start, u_end-4), ctx_end] 是否达成连续 >=5 帧高质锁定
                tail_start = max(u_start, u_end - (RECOVERY_CONSECUTIVE_FRAMES - 1))
                tail_present = [
                    r for r in tk_rows[tail_start : ctx_end + 1]
                    if int(r.get("exist", 1)) == 1
                    and len(r.get("gt_bbox", [])) >= 4
                    and float(r["gt_bbox"][2]) > 0
                    and float(r["gt_bbox"][3]) > 0
                ]
                req_k = min(RECOVERY_CONSECUTIVE_FRAMES, len(tail_present))
                post_locked = False
                if req_k > 0:
                    for pos in range(0, len(tail_present) - req_k + 1):
                        if all(float(x.get("iou", 0.0)) >= RECOVERY_IOU_THRESHOLD for x in tail_present[pos : pos + req_k]):
                            post_locked = True
                            break

                # 如果整个核心窗口全是 absent 帧（无 present 帧），则以 mean_sa 与 post_locked 判定
                if not w_present:
                    if mean_sa >= 0.80 and post_locked:
                        status_label = "SUCCESS"
                    elif post_locked:
                        status_label = "ROBUST_SURVIVAL"
                    else:
                        status_label = "FAILED_UNRECOVERED"
                else:
                    if fail_rate == 0.0 and mean_sa >= 0.50:
                        status_label = "SUCCESS"
                    elif fail_rate <= 0.15 and (mean_sa >= 0.35 or post_locked):
                        status_label = "ROBUST_SURVIVAL"
                    elif post_locked:
                        status_label = "TRANSIENT_FAILURE_RECOVERED"
                    else:
                        status_label = "FAILED_UNRECOVERED"

                per_tracker_outcomes[tk] = {
                    "status": status_label,
                    "window_mean_sa": round(mean_sa, 6),
                    "window_mean_iou_present": round(mean_iou, 6),
                    "window_min_iou_present": round(min_iou, 6),
                    "window_failure_rate_tau010": round(fail_rate, 6),
                    "window_high_quality_rate": round(hq_rate, 6),
                    "absent_false_positives": absent_fp_cnt,
                    "post_window_locked": post_locked,
                }

            # 判定相对于 anchor_tracker (SiamDT_Baseline) 的 Type A / B / C / D 类别
            anchor_out = per_tracker_outcomes.get(anchor_tracker)
            peer_trackers = [tk for tk in per_tracker_outcomes.keys() if tk != anchor_tracker]

            rescuing_trackers: List[str] = []
            best_rescuer: Optional[str] = None
            best_rescuer_sa: float = -1.0

            if anchor_out:
                anchor_failed = anchor_out["status"] in ("FAILED_UNRECOVERED", "TRANSIENT_FAILURE_RECOVERED")
                if anchor_failed:
                    for ptk in peer_trackers:
                        p_out = per_tracker_outcomes[ptk]
                        # 若对比算法达到 SUCCESS / ROBUST_SURVIVAL，或当 SiamDT 完全未恢复时对比算法成功自救恢复
                        is_rescuer = (
                            p_out["status"] in ("SUCCESS", "ROBUST_SURVIVAL")
                            or (
                                anchor_out["status"] == "FAILED_UNRECOVERED"
                                and p_out["status"] == "TRANSIENT_FAILURE_RECOVERED"
                            )
                        )
                        if is_rescuer:
                            rescuing_trackers.append(ptk)
                            if p_out["window_mean_sa"] > best_rescuer_sa:
                                best_rescuer_sa = p_out["window_mean_sa"]
                                best_rescuer = ptk

                    if not peer_trackers or len(rescuing_trackers) == 0:
                        window_type = "Type_A_Common_Hard_Failure"
                    else:
                        robust_peers = sum(
                            1 for ptk in peer_trackers
                            if per_tracker_outcomes[ptk]["status"] in ("SUCCESS", "ROBUST_SURVIVAL")
                        )
                        if len(peer_trackers) > 0 and (robust_peers / len(peer_trackers)) >= 0.80:
                            window_type = "Type_C_SiamDT_Unique_Failure"
                        else:
                            window_type = "Type_B_Rescued_SiamDT_Failure"
                else:
                    # SiamDT 自身稳健通过
                    any_peer_failed = any(
                        per_tracker_outcomes[ptk]["status"] in ("FAILED_UNRECOVERED", "TRANSIENT_FAILURE_RECOVERED")
                        for ptk in peer_trackers
                    )
                    if any_peer_failed:
                        window_type = "Type_D_Peer_Failure_SiamDT_Success"
                    else:
                        window_type = "Type_E_All_Trackers_Succeeded_Absent_Window"
            else:
                window_type = "Type_A_Common_Hard_Failure"

            window_type_counts[window_type] = window_type_counts.get(window_type, 0) + 1

            all_canonical_windows.append({
                "window_id": f"{seq_id}__CW{w_idx:03d}",
                "seq_id": seq_id,
                "window_index_in_seq": w_idx,
                "recording_group": s_meta.get("recording_group"),
                "sequence_attributes": s_meta.get("attributes"),
                "primary_challenge_tags": s_meta.get("primary_challenge_tags"),
                "core_challenge_span": [u_start, u_end],
                "core_duration_frames": u_end - u_start + 1,
                "core_present_frames": len(core_present_ref),
                "core_absent_frames": core_absent_cnt,
                "core_mean_target_size": round(sum(core_sizes) / len(core_sizes), 4) if core_sizes else 0.0,
                "padded_context_span": [ctx_start, ctx_end],
                "pre_context_span": [ctx_start, max(ctx_start, u_start - 1)],
                "post_observation_span": [min(ctx_end, u_end + 1), ctx_end],
                "trigger_sources": mspan["triggers"],
                "window_diagnostic_type": window_type,
                "rescuing_trackers": rescuing_trackers,
                "best_rescuer": best_rescuer,
                "best_rescuer_mean_sa": round(best_rescuer_sa, 6) if best_rescuer else None,
                "per_tracker_outcomes": per_tracker_outcomes,
            })

    return {
        "schema_version": "1.0.0",
        "protocol_version": "v1.0-phase2-protocol-frozen",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_name": split_manifest.get("split_name", "unknown"),
        "anchor_tracker": anchor_tracker,
        "evaluated_trackers": available_trackers,
        "window_construction_params": {
            "primary_tau_fail": PRIMARY_TAU_FAIL,
            "episode_bridge_gap": BRIDGE_GAP_FRAMES,
            "canonical_merge_gap": merge_gap,
            "pre_context_pad": pre_pad,
            "post_observation_pad": post_pad,
            "recovery_consecutive_frames": RECOVERY_CONSECUTIVE_FRAMES,
            "recovery_iou_threshold": RECOVERY_IOU_THRESHOLD,
        },
        "summary": {
            "total_canonical_windows": len(all_canonical_windows),
            "sequences_with_windows": len({w["seq_id"] for w in all_canonical_windows}),
            "window_type_counts": window_type_counts,
        },
        "canonical_windows": all_canonical_windows,
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
def run_phase2_extraction_pipeline(
    telemetry_root: Path,
    split_json_path: Path,
    output_dir: Path,
    anchor_tracker: str = "SiamDT_Baseline",
) -> Tuple[Path, Path]:
    """
    扫描 telemetry_root 下所有符合 `<TrackerName>_<SplitName>`（或直接 `<TrackerName>`）的子目录，
    对 split_json_path 指定的序列集合执行三层失败分析，输出：
    1. FAILURE_EPISODE_DATABASE.json
    2. CANONICAL_CHALLENGE_WINDOWS.json
    """
    split_manifest = json.loads(split_json_path.read_text(encoding="utf-8"))
    split_name = split_manifest.get("split_name", split_json_path.stem)
    target_sequences: List[str] = split_manifest.get("sequences", [])
    seq_meta_map = {
        item["seq_name"]: item
        for item in split_manifest.get("sequence_details", [])
        if "seq_name" in item
    }

    # 发现所有算法遥测目录
    tracker_dirs: Dict[str, Path] = {}
    if telemetry_root.exists():
        # 若 telemetry_root 本身就包含 *.jsonl，则视为单算法目录
        direct_jsonls = list(telemetry_root.glob("*.jsonl"))
        if direct_jsonls:
            tracker_dirs[anchor_tracker] = telemetry_root
        else:
            for sub in sorted(telemetry_root.iterdir()):
                if not sub.is_dir():
                    continue
                # 支持形如 SiamDT_Baseline_discovery_set 或 SiamDT_Baseline
                dir_name = sub.name
                suffix = f"_{split_name}"
                if dir_name.endswith(suffix):
                    tk_name = dir_name[: -len(suffix)]
                elif "_" in dir_name and any(
                    dir_name.endswith(s) for s in ("_val90_full", "_holdout_verification", "_discovery_set")
                ):
                    # 属于其他划分的目录则跳过
                    continue
                else:
                    tk_name = dir_name
                if list(sub.glob("*.jsonl")):
                    tracker_dirs[tk_name] = sub

    tracker_seq_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    tracker_seq_analysis: Dict[str, Dict[str, Dict[str, Any]]] = {}
    tracker_aggregates: Dict[str, Any] = {}

    for tk_name, tk_dir in tracker_dirs.items():
        tracker_seq_rows[tk_name] = {}
        tracker_seq_analysis[tk_name] = {}

        seq_sa_list: List[float] = []
        total_present_frames = 0
        total_absent_frames = 0
        total_zero_frames = 0
        total_fail_frames = 0
        total_imprecise_frames = 0
        total_hq_frames = 0
        total_absent_fp = 0
        all_episodes: List[Dict[str, Any]] = []

        # 聚合级敏感性计数器
        agg_sens = {
            crit: {
                "failed_present_frames": 0,
                "eval_present_frames": 0,
                "episode_count": 0,
                "max_episode_duration": 0,
                "unrecovered_terminal_episodes": 0,
            }
            for crit in (
                "tau_0.00",
                "tau_0.10",
                "tau_0.30",
                "tau_0.50",
                "nce_gt_1.0",
                "composite_iou0.10_or_nce1.0",
            )
        }

        for seq_id in target_sequences:
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
            total_present_frames += int(l1["present_valid_frames"])
            total_absent_frames += int(l1["absent_frames"])
            sc = l2["present_frame_state_counts"]
            total_zero_frames += int(sc["ZERO_OVERLAP_LOST"])
            total_fail_frames += int(sc["FAILURE_FRAME"])
            total_imprecise_frames += int(sc["IMPRECISE_LOCALIZATION"])
            total_hq_frames += int(sc["HIGH_QUALITY_LOCK"])
            total_absent_fp += int(l2["absence_and_reacquisition"]["false_positive_in_absent_frames"])
            all_episodes.extend(l3["episodes"])

            eval_p_cnt = max(0, int(l1["present_valid_frames"]) - 1)
            for crit, cdict in l2["sensitivity_analysis"].items():
                agg_sens[crit]["failed_present_frames"] += int(cdict["failed_present_frames"])
                agg_sens[crit]["eval_present_frames"] += eval_p_cnt
                agg_sens[crit]["episode_count"] += int(cdict["episode_count"])
                agg_sens[crit]["max_episode_duration"] = max(
                    agg_sens[crit]["max_episode_duration"], int(cdict["max_episode_duration"])
                )
                agg_sens[crit]["unrecovered_terminal_episodes"] += int(cdict["unrecovered_terminal_episodes"])

        # 汇总算法级指标
        ep_durations = [int(e["present_duration_frames"]) for e in all_episodes]
        bucket_totals = {
            "Bucket_1_5_Transient": sum(1 for e in all_episodes if e["duration_bucket"] == "Bucket_1_5_Transient"),
            "Bucket_6_20_Short": sum(1 for e in all_episodes if e["duration_bucket"] == "Bucket_6_20_Short"),
            "Bucket_21_100_Medium": sum(1 for e in all_episodes if e["duration_bucket"] == "Bucket_21_100_Medium"),
            "Bucket_101_Plus_Long_Drift": sum(
                1 for e in all_episodes if e["duration_bucket"] == "Bucket_101_Plus_Long_Drift"
            ),
        }

        sensitivity_summary = {}
        for crit, ctot in agg_sens.items():
            ep_cnt = ctot["eval_present_frames"]
            sensitivity_summary[crit] = {
                "failed_present_frames": ctot["failed_present_frames"],
                "failure_frame_rate": round(ctot["failed_present_frames"] / ep_cnt, 6) if ep_cnt > 0 else 0.0,
                "total_episodes": ctot["episode_count"],
                "max_episode_duration": ctot["max_episode_duration"],
                "unrecovered_terminal_episodes": ctot["unrecovered_terminal_episodes"],
            }

        tracker_aggregates[tk_name] = {
            "tracker_name": tk_name,
            "telemetry_source_dir": str(tk_dir),
            "evaluated_sequences_count": len(seq_sa_list),
            "target_sequences_count": len(target_sequences),
            "macro_mean_sa": round(sum(seq_sa_list) / len(seq_sa_list), 6) if seq_sa_list else 0.0,
            "total_present_frames": total_present_frames,
            "total_absent_frames": total_absent_frames,
            "overall_zero_overlap_rate": (
                round(total_zero_frames / total_present_frames, 6) if total_present_frames > 0 else 0.0
            ),
            "overall_primary_failure_rate_tau010": (
                round(total_fail_frames / total_present_frames, 6) if total_present_frames > 0 else 0.0
            ),
            "overall_imprecise_rate": (
                round(total_imprecise_frames / total_present_frames, 6) if total_present_frames > 0 else 0.0
            ),
            "overall_high_quality_rate": (
                round(total_hq_frames / total_present_frames, 6) if total_present_frames > 0 else 0.0
            ),
            "overall_absent_false_positive_rate": (
                round(total_absent_fp / total_absent_frames, 6) if total_absent_frames > 0 else 0.0
            ),
            "total_failure_episodes": len(all_episodes),
            "total_episode_present_frames": sum(ep_durations),
            "mean_episode_duration_frames": round(sum(ep_durations) / len(ep_durations), 3) if ep_durations else 0.0,
            "max_episode_duration_frames": max(ep_durations) if ep_durations else 0,
            "recovered_episodes_count": sum(1 for e in all_episodes if e["recovery_status"] == "recovered"),
            "unrecovered_terminal_drift_count": sum(
                1 for e in all_episodes if e["recovery_status"] == "unrecovered_terminal_drift"
            ),
            "episode_duration_buckets": bucket_totals,
            "sensitivity_analysis_summary": sensitivity_summary,
            "per_sequence_analysis": tracker_seq_analysis[tk_name],
        }

    # 构建 FAILURE_EPISODE_DATABASE.json
    failure_db_payload = {
        "schema_version": "1.0.0",
        "protocol_version": "v1.0-phase2-protocol-frozen",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_name": split_name,
        "split_manifest_path": str(split_json_path),
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

    # 构建 CANONICAL_CHALLENGE_WINDOWS.json
    canonical_windows_payload = build_canonical_challenge_windows(
        tracker_seq_rows=tracker_seq_rows,
        tracker_seq_analysis=tracker_seq_analysis,
        split_manifest=split_manifest,
        anchor_tracker=anchor_tracker,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "FAILURE_EPISODE_DATABASE.json"
    cw_path = output_dir / "CANONICAL_CHALLENGE_WINDOWS.json"

    db_path.write_text(json.dumps(failure_db_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    cw_path.write_text(json.dumps(canonical_windows_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return db_path, cw_path


# ==============================================================================
# 内置自检单元测试 (--self-test)
# ==============================================================================
def _run_self_test() -> None:
    """
    构造合成多算法遥测数据并验证：
    1. 瞬时波动桥接 (<= 3 帧非失败间隙自动合并)；
    2. 消失期 false_positive_in_absent 与重入连续 5 帧 IoU>=0.50 的 Re-acquisition Latency；
    3. 敏感性阈值集合 {0.0, 0.10, 0.30, 0.50} 与 NCE > 1.0；
    4. 跨算法 Canonical Challenge Windows 对齐与 Type B (Rescue) 判定。
    """
    seq_id = "03_7951_1500-2999"
    total_f = 80
    gt_rects = [[100.0 + f * 1.0, 100.0, 20.0, 20.0] for f in range(total_f)]
    exist_flags = [1] * total_f
    # 设定第 40..44 帧 (5帧) 目标消失 (exist == 0)
    for t in range(40, 45):
        exist_flags[t] = 0
        gt_rects[t] = [0.0, 0.0, 0.0, 0.0]

    # 算法 A (SiamDT_Baseline):
    # - 第 15..18 帧失败 (IoU=0)，第 19..20 帧瞬时擦边 (IoU=0.25, 2帧 <= 3帧桥接)，第 21..25 帧失败 (IoU=0)
    #   -> 应桥接为单个 Failure Episode [15, 25]，并在第 26..30 帧连续 5 帧 IoU=1.0 实现 recovered (latency=0)
    # - 第 40..44 帧消失期间继续输出实框 (false_positive_in_absent = 5)
    # - 第 45 帧重入后，第 45..47 帧丢失 (3帧)，第 48..52 帧恢复锁定 -> Re-acquisition Latency = 3 帧
    pred_siamdt = [list(gt_rects[f]) if exist_flags[f] == 1 else [150.0, 150.0, 20.0, 20.0] for f in range(total_f)]
    for t in range(15, 19):
        pred_siamdt[t] = [10.0, 10.0, 20.0, 20.0]
    for t in range(19, 21):
        # 构造 IoU ~ 0.25 (位移 12px)
        pred_siamdt[t] = [gt_rects[t][0] + 12.0, gt_rects[t][1], 20.0, 20.0]
    for t in range(21, 26):
        pred_siamdt[t] = [10.0, 10.0, 20.0, 20.0]
    for t in range(45, 48):
        pred_siamdt[t] = [10.0, 10.0, 20.0, 20.0]

    rows_siamdt = build_telemetry_rows_from_boxes(seq_id, pred_siamdt, gt_rects, exist_flags)
    ana_siamdt = analyze_single_sequence(rows_siamdt)

    eps = ana_siamdt["level3_failure_episodes_summary"]["episodes"]
    assert len(eps) == 2, f"期望 2 个桥接后失败事件段，实际得到 {len(eps)}"
    assert eps[0]["t_start"] == 15 and eps[0]["t_end"] == 25, f"桥接边界错误: {eps[0]}"
    assert eps[0]["bridged_transient_frames_count"] == 2, f"桥接帧数错误: {eps[0]}"
    assert eps[0]["recovery_status"] == "recovered" and eps[0]["recovery_latency_frames"] == 0

    reacq = ana_siamdt["level2_frame_and_absence_analysis"]["absence_and_reacquisition"]
    assert reacq["false_positive_in_absent_frames"] == 5
    assert reacq["events"][0]["reacquired"] is True
    assert reacq["events"][0]["reacquisition_latency_frames"] == 3, f"重捕获延迟错误: {reacq['events'][0]}"

    # 算法 B (FocusTrack): 在 [15, 25] 保持完美跟踪 -> 应在 Canonical Challenge Window 中触发 Type_B / Type_C Rescue
    pred_focus = [list(gt_rects[f]) if exist_flags[f] == 1 else [0.0, 0.0, 0.0, 0.0] for f in range(total_f)]
    rows_focus = build_telemetry_rows_from_boxes(seq_id, pred_focus, gt_rects, exist_flags)
    ana_focus = analyze_single_sequence(rows_focus)

    cw = build_canonical_challenge_windows(
        tracker_seq_rows={"SiamDT_Baseline": {seq_id: rows_siamdt}, "FocusTrack": {seq_id: rows_focus}},
        tracker_seq_analysis={"SiamDT_Baseline": {seq_id: ana_siamdt}, "FocusTrack": {seq_id: ana_focus}},
        split_manifest={"split_name": "self_test", "sequences": [seq_id], "sequence_details": [{"seq_name": seq_id}]},
        anchor_tracker="SiamDT_Baseline",
    )
    assert cw["summary"]["total_canonical_windows"] == 2
    w0 = cw["canonical_windows"][0]
    assert "FocusTrack" in w0["rescuing_trackers"], f"期望 FocusTrack 成功 Rescue 窗口 0: {w0}"
    print(json.dumps({
        "self_test": "PASS",
        "verified_episode_bridging": [eps[0]["t_start"], eps[0]["t_end"]],
        "verified_reacquisition_latency": reacq["events"][0]["reacquisition_latency_frames"],
        "verified_canonical_windows_count": cw["summary"]["total_canonical_windows"],
        "verified_window_0_rescuer": w0["best_rescuer"],
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 三层失败提取器与跨算法统一挑战时间窗生成工具"
    )
    parser.add_argument(
        "--telemetry-root",
        type=Path,
        default=Path("G:/我的云端硬盘/SiamDT_Project/experiments/failure_diagnosis/per_frame_telemetry"),
        help="包含各算法逐帧遥测子目录 (<Tracker>_<Split>/<seq_id>.jsonl) 的根目录",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("Z:/SimDT围点打援/datasets/diagnostic_splits/discovery_set.json"),
        help="冻结的数据集划分清单 JSON 路径 (如 discovery_set.json / holdout_verification.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Z:/SimDT围点打援/phase2_unified_eval/outputs"),
        help="输出 FAILURE_EPISODE_DATABASE.json 与 CANONICAL_CHALLENGE_WINDOWS.json 的目录",
    )
    parser.add_argument(
        "--anchor-tracker",
        type=str,
        default="SiamDT_Baseline",
        help="用于判定 Type A/B/C/D 救援关系的主锚点算法名称",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="运行内置确定性单元测试以核验桥接、重入延迟与 Canonical Window 逻辑",
    )
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        return

    db_path, cw_path = run_phase2_extraction_pipeline(
        telemetry_root=args.telemetry_root,
        split_json_path=args.split_json,
        output_dir=args.output_dir,
        anchor_tracker=args.anchor_tracker,
    )
    print(json.dumps({
        "status": "COMPLETED",
        "failure_episode_database": str(db_path),
        "canonical_challenge_windows": str(cw_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
