#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SiamDT 六层模块特征解剖器 (siamdt_6layer_feature_probe.py) —— v2.0（2026-11-04 交叉审查修订版）
================================================================================
本版本依据 Claude × Gemini 交叉审查（review-think-20261104-001，P0-1 / P0-2）全面重写：
- 删除 v1 中自定义的"二阶 Gram 矩阵响应图"（SiamDT 源码 similarity_encoders.py 中不存在该机制）；
- 删除 v1 Layer 5 中按排名递减 4%、按奇偶分配分支、统一减全局 penalty 等编造公式；
- Layer 5 / Layer 6 全部改为从 SiamDTRCNN 真实计算路径中捕获的值，拿不到的字段写 null，绝不估算。

科研诚信约定（每个输出字段都带 provenance 标签，见 FIELD_PROVENANCE）：
- `captured_model_path`        : 模型真实计算路径中的量（forward hook / 方法包装直接捕获）；
- `recomputed_by_source_formula`: 用真实捕获量按源码同一公式复算，仅用于一致性核对；
- `posthoc_derived_not_model_path`: 事后派生量（如通道 L2 范数图、与 GT 的 IoU、T/BG 比值），非模型路径；
- `analyst_heuristic_label`    : 分析者启发式标签（阈值由分析者设定，非模型行为）。

六层与真实源码路径的对应关系（源码：trackers/SiamDT/trackers/siamdt_rcnn.py、similarity_encoders.py、siamdt_tracking.py）：
- Layer 1 `model.backbone`：Swin-Tiny 4 个 Stage 输出（out_indices=(0,1,2,3)）；
- Layer 2 `model.neck`：FPN 5 层输出（num_outs=5）；
- Layer 3A `model.rpn_similarity_learning.proj_out[k]`：RPN_Similarity_Learning 的调制响应
  x_corr[k] = proj_out[k](proj_query[k](template) * x[k])（similarity_encoders.py:30-32），
  该响应在 _process_gallary:437-441 中与 x[k] 相加后送入 RPN；
  注意 forward 是 generator，本工具只在 proj_out 子模块上挂 hook，绝不消费 generator；
- Layer 3B `model.rcnn_similarity_learning.proj_out`：Stage-2 RCNN_Similarity_Learning 的响应
  proj_out(proj_x(x) * proj_z(z))（similarity_encoders.py:53），按调用阶段区分
  `rcnn_main`（simple_test_bboxes:203，模板 vs 全部 RPN 候选）与 `bg_matching`（simple_matching:276，候选 vs 背景记忆）；
- Layer 4 RPN 候选：simple_test_bboxes 的 proposals 入参（网络输入尺度）→ 除以 img_meta['scale_factor'] 还原原图；
- Layer 5 决赛圈：源码 _process_gallary:452-460 把 tra_bboxes 与 det_bboxes 拼接后按分数降序取 Top_NUM=5
  （注意：源码是"两分支合并后取前 5"，并不是"每分支各取前 5 共 10 个"；本工具如实记录这 5 个，
  另外把两分支各自前 5 作为 `branch_top5` 参考列表输出，两者含义不同，不可混用）；
  每个候选的真实背景惩罚 = simple_matching 返回分数的最大值（:472-478）；
  IoU 奖励是同一帧 5 个候选之间 IoU>0.8 的互相奖励（:482-497，并非"历史框"奖励）；
  最终得分取自 _process_gallary 的真实返回值；由于 update 传入 rescale=True，tra/det/最终框已是原图坐标；
- Layer 6 模板更新：源码 :500 条件为 tra_bboxes[0,-1] + det_bboxes[0,-1] > 1.9 且 IoU(det[0], tra[0]) > 0.8，
  更新为 EMA：_bbox_feats_z = (1 - lr) * _bbox_feats_z + lr * new（:402，lr 由 _process_query:353 设为 0.01，
  本工具运行时直接读取 model._learning_rate），同时背景记忆 _bbox_feats_bg 被整体替换（:421-426）。

运行模式：
- `--offline-geometry-check`：仅几何/IO 纯函数离线检查（纯标准库，可在无 GPU 本机运行；不加载模型）；
- `--model-smoke`：在 Colab GPU 上实例化真实 SiamDTRCNN，跑 1 条序列前 5 帧，断言每个 hook/包装都捕获到非空且形状合理的张量；
- 默认模式：对指定序列关键帧输出 6 层解剖图与 JSON 指标。
Holdout 封存：默认拒绝处理 holdout 序列（holdout_verification.json 以及 val90_full.json 的 holdout_full_36），
除非显式传入 `--unlock-holdout`，且解封结果写入独立的 HOLDOUT_UNSEALED 目录。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROBE_VERSION = "2.0.0-rev20261104"

# ==============================================================================
# 0. 源码常量（全部注明出处；运行时能从模型读到的量以运行时为准）
# ==============================================================================
SOURCE_TOP_NUM = 5                    # siamdt_rcnn.py:456  Top_NUM = 5
SOURCE_IOU_REWARD_THR = 0.8           # siamdt_rcnn.py:496  if iouvalue > 0.8
SOURCE_UPDATE_SCORE_SUM_THR = 1.9     # siamdt_rcnn.py:500  tra[0,-1] + det[0,-1] > 1.9
SOURCE_UPDATE_IOU_THR = 0.8           # siamdt_rcnn.py:500  computeiou(det[0], tra[0]) > 0.8
SOURCE_EMA_LR_AT_INIT = 0.01          # siamdt_rcnn.py:353  self._learning_rate = 0.01（运行时以 model._learning_rate 为准）
SOURCE_BG_BANK_MAX = 11               # siamdt_rcnn.py:382/418  len(bgproposals) > 10 -> break（最多 11 个）

# 分析者阈值（非模型行为；仅用于事后诊断标签，必须在输出中标注）
ANALYST_UPDATE_GT_IOU_THR = 0.5       # 模板更新所用框与 GT 的 IoU 低于该值 -> 事后标记"可疑更新"
ANALYST_RPN_MISS_IOU = 0.30
ANALYST_RANK_INV_BEST_IOU = 0.50
ANALYST_RANK_INV_SELECTED_IOU = 0.30

PROV_CAPTURED = "captured_model_path"
PROV_RECOMPUTED = "recomputed_by_source_formula"
PROV_DERIVED = "posthoc_derived_not_model_path"
PROV_ANALYST = "analyst_heuristic_label"

FIELD_PROVENANCE: Dict[str, str] = {
    "layer1.*.l2_map / shape": PROV_CAPTURED + "（shape）；l2_map 为通道 L2 范数 = " + PROV_DERIVED,
    "layer1/2/3a.*.target_to_bg_activation_ratio / margin / entropy": PROV_DERIVED,
    "layer2.*": "同 layer1",
    "layer3a_rpn_modulation.*": "proj_out[k] 输出张量为 " + PROV_CAPTURED + "；空间图为通道 L2 范数 = " + PROV_DERIVED,
    "layer3b_rcnn_similarity.per_roi_l2_energy": "proj_out 输出为 " + PROV_CAPTURED + "；逐 ROI L2 能量 = " + PROV_DERIVED,
    "layer4.proposals_orig_xyxy / score": PROV_CAPTURED + "（坐标按 scale_factor 还原原图）",
    "layer4.*_max_iou / hit_count": PROV_DERIVED + "（与 GT 比较）",
    "layer5.candidates[].bbox_xyxy / final_score": PROV_CAPTURED + "（_process_gallary 返回值）",
    "layer5.candidates[].branch / pre_bg_supp_score": PROV_CAPTURED + "（与 simple_test_bboxes 返回的 tra/det 行做精确坐标匹配）",
    "layer5.candidates[].bg_penalty": PROV_CAPTURED + "（simple_matching 返回分数的 max）",
    "layer5.candidates[].post_bg_score / iou_reward": PROV_RECOMPUTED + "（源码 :478 / :497 的加减关系，由真实量反推）",
    "layer5.candidates[].gt_iou / selected_by_tracker": PROV_DERIVED + " / " + PROV_CAPTURED,
    "layer5.branch_top5": PROV_CAPTURED + "（两分支各自前 5，仅参考，并非源码决赛圈）",
    "layer6.template_updated": PROV_CAPTURED + "（_process_gallary 返回的 up_flag）",
    "layer6.source_update_condition": PROV_RECOMPUTED + "（源码 :500 公式，用于与 up_flag 一致性核对）",
    "layer6.learning_rate": PROV_CAPTURED + "（model._learning_rate）",
    "layer6.update_box_orig_xyxy": PROV_CAPTURED + "（_update_query 入参，按 scale_factor 还原原图）",
    "layer6.update_box_gt_iou / posthoc_suspicious_update": PROV_DERIVED + " / " + PROV_ANALYST,
    "layer6.template_cosine_sim_to_init / l2_drift / init_template_residual_weight": PROV_DERIVED,
    "root_cause_verdict": PROV_ANALYST,
}


# ==============================================================================
# 1. 纯标准库几何 / IO 工具（--offline-geometry-check 覆盖范围）
# ==============================================================================

def xywh_to_xyxy(box_xywh: Sequence[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box_xywh[:4]]
    return [x, y, x + max(0.0, w), y + max(0.0, h)]


def compute_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def source_computeiou(bbox1: Sequence[float], bbox2: Sequence[float]) -> float:
    """逐行复刻 siamdt_rcnn.py:318-349 SiamDTRCNN.computeiou（用于复算源码 :496 / :500 条件）。"""
    x0_1, y0_1, x1_1, y1_1 = [float(x) for x in bbox1[:4]]
    x0_2, y0_2, x1_2, y1_2 = [float(x) for x in bbox2[:4]]
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0.0
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection
    return size_intersection / size_union


def normalize_scale_factor(scale_factor: Any) -> Optional[List[float]]:
    """把 img_meta['scale_factor']（标量或长度 4 的数组）规范为 4 元列表；无法识别则返回 None（不猜测）。"""
    if scale_factor is None:
        return None
    raw = scale_factor.tolist() if hasattr(scale_factor, "tolist") else scale_factor
    if isinstance(raw, (int, float)):
        vals = [float(raw)] * 4
    else:
        try:
            vals = [float(v) for v in raw]
        except Exception:
            return None
        if len(vals) == 1:
            vals = vals * 4
    if len(vals) != 4 or any((not math.isfinite(v)) or v <= 0.0 for v in vals):
        return None
    return vals


def boxes_net_to_orig(boxes: Sequence[Sequence[float]], sf4: Optional[List[float]]) -> Optional[List[List[float]]]:
    """网络输入尺度 -> 原图尺度：前 4 列除以 scale_factor，其余列（如分数）原样保留。sf4 为 None 时返回 None。"""
    if sf4 is None:
        return None
    out: List[List[float]] = []
    for b in boxes:
        row = [float(b[0]) / sf4[0], float(b[1]) / sf4[1], float(b[2]) / sf4[2], float(b[3]) / sf4[3]]
        row.extend(float(v) for v in b[4:])
        out.append(row)
    return out


def match_candidate_provenance(
    final_rows: Sequence[Sequence[float]],
    tra_rows: Sequence[Sequence[float]],
    det_rows: Sequence[Sequence[float]],
) -> List[Dict[str, Any]]:
    """
    对 _process_gallary 返回的 Top-5 候选做"精确坐标匹配"溯源：
    源码 :463-497 只修改 newens_bboxes 的分数列，坐标列与 tra/det 中被选中的行逐位相同，
    因此坐标精确相等即可唯一确定分支来源与抑制前原始分数；若两分支出现相同坐标则标记 ambiguous 且不猜分数。
    """
    out: List[Dict[str, Any]] = []
    tra_boxes = [[float(v) for v in r[:4]] for r in tra_rows]
    det_boxes = [[float(v) for v in r[:4]] for r in det_rows]
    for i, row in enumerate(final_rows):
        box = [float(v) for v in row[:4]]
        t_idx = [j for j, b in enumerate(tra_boxes) if b == box]
        d_idx = [j for j, b in enumerate(det_boxes) if b == box]
        if t_idx and not d_idx:
            branch = "similarity_tracking_branch(tra)"
            scores = {float(tra_rows[j][4]) for j in t_idx}
        elif d_idx and not t_idx:
            branch = "detection_branch(det)"
            scores = {float(det_rows[j][4]) for j in d_idx}
        elif t_idx and d_idx:
            branch = "ambiguous_both_branches"
            scores = {float(tra_rows[j][4]) for j in t_idx} | {float(det_rows[j][4]) for j in d_idx}
        else:
            branch = "unmatched"
            scores = set()
        out.append({
            "rank_pre_suppression": i + 1,
            "branch": branch,
            "tra_row_indices": t_idx,
            "det_row_indices": d_idx,
            "pre_bg_supp_score": scores.pop() if len(scores) == 1 else None,
        })
    return out


def decompose_candidate_scores(
    final_rows: Sequence[Sequence[float]],
    provenance: Sequence[Dict[str, Any]],
    bg_penalties: Optional[Sequence[Optional[float]]],
) -> List[Dict[str, Any]]:
    """组合真实捕获量：pre（捕获）、bg_penalty（捕获）、final（捕获）；post_bg 与 iou_reward 为源码算术关系反推。"""
    cands: List[Dict[str, Any]] = []
    pens_ok = bg_penalties is not None and len(bg_penalties) == len(final_rows)
    for i, row in enumerate(final_rows):
        prov = provenance[i] if i < len(provenance) else {}
        pre = prov.get("pre_bg_supp_score")
        pen = float(bg_penalties[i]) if (pens_ok and bg_penalties[i] is not None) else None
        final = float(row[4]) if len(row) > 4 else None
        post = (pre - pen) if (pre is not None and pen is not None) else None
        reward = (final - post) if (final is not None and post is not None) else None
        cands.append({
            "rank_pre_suppression": i + 1,
            "bbox_xyxy": [float(v) for v in row[:4]],
            "branch": prov.get("branch"),
            "pre_bg_supp_score": pre,
            "bg_penalty": pen,
            "post_bg_score": post,
            "iou_reward": reward,
            "final_score": final,
        })
    return cands


def source_update_condition(
    tra_rows: Sequence[Sequence[float]],
    det_rows: Sequence[Sequence[float]],
) -> Optional[Dict[str, Any]]:
    """复算源码 :500 的模板更新条件（使用 tra/det 第 0 行，与源码一致；不排序、不改阈值）。"""
    if not tra_rows or not det_rows:
        return None
    tra0, det0 = tra_rows[0], det_rows[0]
    s_sum = float(tra0[4]) + float(det0[4])
    iou = source_computeiou(det0[:4], tra0[:4])
    return {
        "tra_top1_score": float(tra0[4]),
        "det_top1_score": float(det0[4]),
        "score_sum": s_sum,
        "score_sum_threshold_source": SOURCE_UPDATE_SCORE_SUM_THR,
        "tra_det_top1_iou": iou,
        "iou_threshold_source": SOURCE_UPDATE_IOU_THR,
        "condition_met": bool(s_sum > SOURCE_UPDATE_SCORE_SUM_THR and iou > SOURCE_UPDATE_IOU_THR),
    }


def rows_without_iou_reward_partner(final_rows: Sequence[Sequence[float]]) -> List[int]:
    """返回与其它候选的 source_computeiou 均 <= 0.8 的行号；对这些行源码 :497 不加奖励，可用于核对 final == pre - penalty。"""
    idxs: List[int] = []
    for i, a in enumerate(final_rows):
        if all(source_computeiou(a[:4], b[:4]) <= SOURCE_IOU_REWARD_THR for j, b in enumerate(final_rows) if j != i):
            idxs.append(i)
    return idxs


def select_diagnostic_keyframes(
    total_frames: int,
    onset_frame: int,
    explicit_frames: Optional[List[int]] = None,
) -> List[Tuple[int, str]]:
    """围绕失败起点帧截取 [-10, -3, 0, +5, +15] 关键帧（或使用显式帧号）。"""
    if explicit_frames:
        out: List[Tuple[int, str]] = []
        for f in explicit_frames:
            idx = max(1, min(total_frames - 1, int(f)))
            rel = idx - onset_frame
            if rel < 0:
                tag = f"Pre-Failure [{rel:+d}]"
            elif rel == 0:
                tag = "Failure Onset [0]"
            elif rel <= 8:
                tag = f"Drift [{rel:+d}]"
            else:
                tag = f"Recovery/Terminal [{rel:+d}]"
            out.append((idx, tag))
        return out
    offsets = [(-10, "Pre-Failure [-10]"), (-3, "Pre-Failure [-3]"), (0, "Failure Onset [0]"),
               (+5, "Drift [+5]"), (+15, "Recovery/Terminal [+15]")]
    seen = set()
    selected: List[Tuple[int, str]] = []
    for delta, label in offsets:
        f_idx = max(1, min(total_frames - 1, onset_frame + delta))
        if f_idx not in seen:
            seen.add(f_idx)
            selected.append((f_idx, label))
    return selected


# ------------------------------------------------------------------------------
# Holdout 封存护栏（fail-closed：找不到 holdout 清单时也拒绝，除非显式解封）
# ------------------------------------------------------------------------------
class HoldoutSealedError(RuntimeError):
    pass


def _default_split_dirs() -> List[Path]:
    dirs: List[Path] = []
    try:
        here = Path(__file__).resolve()
        dirs.append(here.parents[1] / "datasets" / "diagnostic_splits")
    except Exception:
        pass
    dirs.extend([
        Path("/content/runtime/SimDT_Pipeline/datasets/diagnostic_splits"),
        Path("/content/drive/MyDrive/SiamDT_Project/datasets/diagnostic_splits"),
        Path("/content/drive/MyDrive/SimDT围点打援/datasets/diagnostic_splits"),
        Path(r"G:\我的云端硬盘\SimDT围点打援\datasets\diagnostic_splits"),
    ])
    return dirs


def load_holdout_sequence_set(holdout_json: Optional[Path] = None) -> Tuple[Optional[set], List[str]]:
    """
    读取 holdout 序列集合：holdout_verification.json 的 sequences（Core 16）
    ∪ 同目录 val90_full.json 的 holdout_full_36.sequences（若存在）。返回 (集合或 None, 来源文件列表)。
    """
    cands: List[Path] = [holdout_json] if holdout_json else [d / "holdout_verification.json" for d in _default_split_dirs()]
    for hj in cands:
        if hj is None or not hj.exists():
            continue
        data = json.loads(hj.read_text(encoding="utf-8"))
        seqs = set(str(s) for s in data.get("sequences", []))
        sources = [str(hj)]
        v90 = hj.parent / "val90_full.json"
        if v90.exists():
            try:
                vdata = json.loads(v90.read_text(encoding="utf-8"))
                seqs |= set(str(s) for s in vdata.get("holdout_full_36", {}).get("sequences", []))
                sources.append(str(v90))
            except Exception:
                pass
        if seqs:
            return seqs, sources
    return None, []


def enforce_holdout_guard(seq_id: str, unlock_holdout: bool, holdout_json: Optional[Path] = None) -> Dict[str, Any]:
    holdout_set, sources = load_holdout_sequence_set(holdout_json)
    if holdout_set is None:
        if unlock_holdout:
            return {"holdout_check": "UNVERIFIED_UNLOCKED", "is_holdout": None, "sources": []}
        raise HoldoutSealedError(
            "未找到 holdout_verification.json，无法确认该序列是否属于 Holdout，按封存规则拒绝执行；"
            "请用 --holdout-json 指定清单，或在确认需要解封时显式传入 --unlock-holdout。"
        )
    is_holdout = seq_id in holdout_set
    if is_holdout and not unlock_holdout:
        raise HoldoutSealedError(
            f"序列 {seq_id} 属于 Holdout（封存集），默认拒绝处理；如确需解封请显式传入 --unlock-holdout（结果将写入独立 HOLDOUT_UNSEALED 目录）。"
        )
    return {"holdout_check": "PASSED", "is_holdout": is_holdout, "sources": sources}


def load_discovery_first_sequence() -> Optional[str]:
    for d in _default_split_dirs():
        f = d / "discovery_set.json"
        if f.exists():
            seqs = json.loads(f.read_text(encoding="utf-8")).get("sequences", [])
            if seqs:
                return str(seqs[0])
    return None


# ==============================================================================
# 2. 需要 numpy 的派生指标（事后派生，非模型路径）
# ==============================================================================

def compute_spatial_map_metrics(activation_map: Any, gt_xywh: List[float], img_h: int, img_w: int,
                                guard_dilation_factor: float = 1.5) -> Dict[str, Any]:
    """【事后派生，非模型路径】对二维图计算 GT 区域/背景能量比、峰值裕度与归一化熵。"""
    import numpy as np

    arr = np.asarray(activation_map, dtype=np.float64)
    if arr.ndim > 2:
        arr = arr.squeeze()
    hm, wm = arr.shape[:2]
    empty = {"target_to_bg_activation_ratio": None, "target_vs_strongest_distractor_margin": None,
             "response_entropy": None, "target_peak": None, "strongest_distractor_peak": None,
             "provenance": PROV_DERIVED}
    if hm == 0 or wm == 0:
        return empty
    span = max(1e-8, float(arr.max()) - float(arr.min()))
    norm_arr = (arr - float(arr.min())) / span
    prob = norm_arr + 1e-12
    prob = prob / np.sum(prob)
    entropy = float(-np.sum(prob * np.log(prob)) / math.log(max(2, hm * wm)))
    gx, gy, gw, gh = [float(v) for v in gt_xywh[:4]]
    if gw <= 0.0 or gh <= 0.0 or img_w <= 0 or img_h <= 0:
        empty["response_entropy"] = round(entropy, 6)
        return empty
    sx, sy = wm / float(img_w), hm / float(img_h)
    u1 = max(0, min(wm - 1, int(math.floor(gx * sx))))
    v1 = max(0, min(hm - 1, int(math.floor(gy * sy))))
    u2 = max(u1 + 1, min(wm, int(math.ceil((gx + gw) * sx))))
    v2 = max(v1 + 1, min(hm, int(math.ceil((gy + gh) * sy))))
    cx, cy = gx + 0.5 * gw, gy + 0.5 * gh
    guard_w, guard_h = max(gw * guard_dilation_factor, 16.0), max(gh * guard_dilation_factor, 16.0)
    gu1 = max(0, min(wm - 1, int(math.floor((cx - 0.5 * guard_w) * sx))))
    gv1 = max(0, min(hm - 1, int(math.floor((cy - 0.5 * guard_h) * sy))))
    gu2 = max(gu1 + 1, min(wm, int(math.ceil((cx + 0.5 * guard_w) * sx))))
    gv2 = max(gv1 + 1, min(hm, int(math.ceil((cy + 0.5 * guard_h) * sy))))
    target_patch = norm_arr[v1:v2, u1:u2]
    bg_mask = np.ones((hm, wm), dtype=bool)
    bg_mask[gv1:gv2, gu1:gu2] = False
    bg_vals = norm_arr[bg_mask]
    if bg_vals.size == 0:
        bg_mask = np.ones((hm, wm), dtype=bool)
        bg_mask[v1:v2, u1:u2] = False
        bg_vals = norm_arr[bg_mask]
    t_mean = float(np.mean(target_patch)) if target_patch.size else None
    b_mean = float(np.mean(bg_vals)) if bg_vals.size else None
    t_peak = float(np.max(target_patch)) if target_patch.size else None
    d_peak = float(np.max(bg_vals)) if bg_vals.size else None
    return {
        "target_to_bg_activation_ratio": round(t_mean / max(1e-6, b_mean), 6) if (t_mean is not None and b_mean is not None) else None,
        "target_vs_strongest_distractor_margin": round(t_peak - d_peak, 6) if (t_peak is not None and d_peak is not None) else None,
        "response_entropy": round(entropy, 6),
        "target_peak": round(t_peak, 6) if t_peak is not None else None,
        "strongest_distractor_peak": round(d_peak, 6) if d_peak is not None else None,
        "provenance": PROV_DERIVED,
    }


# ==============================================================================
# 3. 非侵入式 hook + 方法包装管理器（只读捕获，不改变任何返回值）
# ==============================================================================

def _to_py(v: Any) -> Any:
    if v is None:
        return None
    if hasattr(v, "detach"):
        v = v.detach().float().cpu()
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_to_py(x) for x in v]
    return v


class SiamDT6LayerHookManager:
    """
    - forward hook：backbone / neck / rpn_similarity_learning.proj_out[k] / rpn_head / rcnn_similarity_learning.proj_out；
    - 实例级方法包装（只观测入参与返回值，原样返回）：_process_gallary / simple_test_bboxes / simple_matching / _update_query。
    包装通过在模型实例 __dict__ 中放置同名可调用对象实现，remove_hooks() 会删除这些实例属性，恢复类方法。
    """

    WRAPPED_METHODS = ("_process_gallary", "simple_test_bboxes", "simple_matching", "_update_query")

    def __init__(self, model: Any) -> None:
        self.model = model
        self.handles: List[Any] = []
        self._phase: str = "other"
        self.frame_buffer: Dict[str, Any] = {}
        self.initial_template_feat: Any = None
        self.cumulative_update_count: int = 0
        self.clear_frame_buffer()
        self._register_hooks()
        self._install_wrappers()

    def clear_frame_buffer(self) -> None:
        self.frame_buffer = {
            "layer1_backbone_stages": [],
            "layer2_fpn_levels": [],
            "layer3a_rpn_modulation_levels": [None] * 5,
            "layer3a_call_count": 0,
            "rpn_objectness_maps": [],
            "layer3b_rcnn_similarity_calls": [],
            "img_meta_x": None,
            "learning_rate": None,
            "rpn_proposals_net": None,
            "simple_test_bboxes_rescale": None,
            "tra_bboxes": None,
            "det_bboxes": None,
            "bg_matching_calls": [],
            "final_results": None,
            "up_flag": None,
            "update_query_calls": [],
            "process_gallary_calls": 0,
        }

    @staticmethod
    def _l2_spatial_map(feat_tensor: Any) -> Any:
        import torch
        t = feat_tensor.detach().float()
        if t.dim() == 4:
            t = t[0]
        return torch.norm(t, p=2, dim=0).cpu().numpy()

    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        import torch
        m = self.model

        def _hook_backbone(module: Any, inputs: Any, output: Any) -> None:
            if isinstance(output, (tuple, list)):
                self.frame_buffer["layer1_backbone_stages"] = [
                    {"stage_index": i + 1, "stage_name": f"Stage{i + 1}_C{i + 2}", "shape": list(f.shape),
                     "l2_map": self._l2_spatial_map(f)}
                    for i, f in enumerate(output) if torch.is_tensor(f)
                ]

        self.handles.append(m.backbone.register_forward_hook(_hook_backbone))

        def _hook_neck(module: Any, inputs: Any, output: Any) -> None:
            if isinstance(output, (tuple, list)):
                self.frame_buffer["layer2_fpn_levels"] = [
                    {"level_index": i, "level_name": f"P{i + 2}", "stride": 2 ** (i + 2), "shape": list(f.shape),
                     "l2_map": self._l2_spatial_map(f)}
                    for i, f in enumerate(output) if torch.is_tensor(f)
                ]

        self.handles.append(m.neck.register_forward_hook(_hook_neck))

        # Layer 3A：RPN_Similarity_Learning.forward 是 generator —— 只挂 proj_out[k] 子模块，绝不调用/消费 generator
        for lvl_idx, conv_mod in enumerate(m.rpn_similarity_learning.proj_out):
            def _make_hook(k: int):
                def _hook(module: Any, inputs: Any, output: Any) -> None:
                    if torch.is_tensor(output):
                        self.frame_buffer["layer3a_call_count"] += 1
                        self.frame_buffer["layer3a_rpn_modulation_levels"][k] = {
                            "level_name": f"P{k + 2}",
                            "shape": list(output.shape),
                            "l2_map": self._l2_spatial_map(output),
                            "abs_mean": float(output.detach().float().abs().mean().item()),
                        }
                return _hook
            self.handles.append(conv_mod.register_forward_hook(_make_hook(lvl_idx)))

        use_sigmoid = bool(getattr(m.rpn_head, "use_sigmoid_cls", False))

        def _hook_rpn_head(module: Any, inputs: Any, output: Any) -> None:
            if isinstance(output, (tuple, list)) and len(output) == 2:
                maps = []
                for i, sc in enumerate(output[0]):
                    if torch.is_tensor(sc):
                        maps.append({
                            "level_name": f"P{i + 2}",
                            "shape": list(sc.shape),
                            # 仅当 rpn_head.use_sigmoid_cls=True 时 sigmoid 才与 get_bboxes 的打分方式一致；否则写 null
                            "objectness_max_over_anchors": (
                                torch.sigmoid(sc.detach().float()[0]).max(dim=0)[0].cpu().numpy() if use_sigmoid else None
                            ),
                        })
                self.frame_buffer["rpn_objectness_maps"] = maps

        self.handles.append(m.rpn_head.register_forward_hook(_hook_rpn_head))

        # Layer 3B：Stage-2 RCNN_Similarity_Learning.proj_out（输出即 proj_out(proj_x(x) * proj_z(z))）
        def _hook_rcnn_sim(module: Any, inputs: Any, output: Any) -> None:
            if torch.is_tensor(output):
                o = output.detach().float()
                self.frame_buffer["layer3b_rcnn_similarity_calls"].append({
                    "phase": self._phase,
                    "bg_call_index": len(self.frame_buffer["bg_matching_calls"]) if self._phase == "bg_matching" else None,
                    "out_shape": list(o.shape),
                    "per_roi_l2_energy": torch.norm(o.reshape(o.shape[0], -1), p=2, dim=1).cpu().tolist() if o.dim() >= 2 else [],
                })

        self.handles.append(m.rcnn_similarity_learning.proj_out.register_forward_hook(_hook_rcnn_sim))

    # ------------------------------------------------------------------
    def _install_wrappers(self) -> None:
        m = self.model
        orig_pg = m._process_gallary
        orig_stb = m.simple_test_bboxes
        orig_sm = m.simple_matching
        orig_uq = m._update_query
        hm = self

        def pg_wrapper(img_x: Any, img_meta_x: Any, **kwargs: Any) -> Any:
            buf = hm.frame_buffer
            buf["process_gallary_calls"] += 1
            meta0 = img_meta_x[0] if isinstance(img_meta_x, (list, tuple)) and img_meta_x else {}
            buf["img_meta_x"] = {
                "img_shape": list(meta0.get("img_shape", []) or []),
                "ori_shape": list(meta0.get("ori_shape", []) or []),
                "scale_factor_raw": _to_py(meta0.get("scale_factor")),
            }
            lr = getattr(m, "_learning_rate", None)
            buf["learning_rate"] = float(lr) if lr is not None else None
            out = orig_pg(img_x, img_meta_x, **kwargs)
            results, up_flag = out
            arr = results.tolist() if hasattr(results, "tolist") else _to_py(results)
            buf["final_results"] = [list(r[:5]) for r in arr]
            buf["up_flag"] = bool(up_flag)
            return out

        def stb_wrapper(*args: Any, **kwargs: Any) -> Any:
            buf = hm.frame_buffer
            proposals = args[3] if len(args) > 3 else kwargs.get("proposals")
            if proposals is not None and len(proposals) > 0:
                buf["rpn_proposals_net"] = _to_py(proposals[0])
            buf["simple_test_bboxes_rescale"] = bool(kwargs.get("rescale", False))
            prev = hm._phase
            hm._phase = "rcnn_main"
            try:
                out = orig_stb(*args, **kwargs)
            finally:
                hm._phase = prev
            tra_b, _tra_l, det_b, _det_l = out
            buf["tra_bboxes"] = _to_py(tra_b)
            buf["det_bboxes"] = _to_py(det_b)
            return out

        def sm_wrapper(*args: Any, **kwargs: Any) -> Any:
            buf = hm.frame_buffer
            rois_bg = args[2] if len(args) > 2 else kwargs.get("rois_x")
            prev = hm._phase
            hm._phase = "bg_matching"
            try:
                out = orig_sm(*args, **kwargs)
            finally:
                hm._phase = prev
            match_bboxes = out[0]
            scores = _to_py(match_bboxes[:, -1]) if hasattr(match_bboxes, "shape") and match_bboxes.shape[0] > 0 else []
            buf["bg_matching_calls"].append({
                "call_index": len(buf["bg_matching_calls"]),
                "num_bg_rois": int(rois_bg.shape[0]) if hasattr(rois_bg, "shape") else None,
                "num_returned": len(scores),
                "scores": scores,
                "max_score": max(scores) if scores else None,
            })
            return out

        def uq_wrapper(*args: Any, **kwargs: Any) -> Any:
            import torch
            buf = hm.frame_buffer
            gt_bboxes_z = args[2] if len(args) > 2 else kwargs.get("gt_bboxes_z")
            z_before = m._bbox_feats_z.detach().float().clone() if torch.is_tensor(getattr(m, "_bbox_feats_z", None)) else None
            out = orig_uq(*args, **kwargs)
            z_after = m._bbox_feats_z.detach().float() if torch.is_tensor(getattr(m, "_bbox_feats_z", None)) else None
            box_net = None
            try:
                box_net = [float(v) for v in _to_py(gt_bboxes_z[0])[0][:4]]
            except Exception:
                box_net = None
            buf["update_query_calls"].append({
                "update_box_net_xyxy": box_net,
                "learning_rate": float(getattr(m, "_learning_rate", float("nan"))),
                "template_delta_l2": float(torch.norm(z_after - z_before).item()) if (z_before is not None and z_after is not None) else None,
                "bg_bank_size_after": int(m._bbox_feats_bg.shape[0]) if torch.is_tensor(getattr(m, "_bbox_feats_bg", None)) else None,
            })
            hm.cumulative_update_count += 1
            return out

        m.__dict__["_process_gallary"] = pg_wrapper
        m.__dict__["simple_test_bboxes"] = stb_wrapper
        m.__dict__["simple_matching"] = sm_wrapper
        m.__dict__["_update_query"] = uq_wrapper

    def snapshot_initial_template(self) -> None:
        z = getattr(self.model, "_bbox_feats_z", None)
        if z is not None:
            self.initial_template_feat = z.detach().float().clone()

    def remove_hooks(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()
        for name in self.WRAPPED_METHODS:
            self.model.__dict__.pop(name, None)


# ==============================================================================
# 4. 从捕获缓冲区构建单帧记录（真实值 + 明确标注的派生量）
# ==============================================================================

def build_layer5_and_6_records(buf: Dict[str, Any], gt_xywh: List[float], exist_f: int) -> Dict[str, Any]:
    """纯 Python：Layer 4/5/6 的真实值整理与派生指标（可脱离 torch 单测）。"""
    gt_xyxy = xywh_to_xyxy(gt_xywh)
    gt_valid = exist_f == 1 and gt_xywh[2] > 0 and gt_xywh[3] > 0
    meta = buf.get("img_meta_x") or {}
    sf4 = normalize_scale_factor(meta.get("scale_factor_raw"))
    tra = buf.get("tra_bboxes") or []
    det = buf.get("det_bboxes") or []
    final_rows = buf.get("final_results") or []

    # ---------------- Layer 4：RPN 候选（网络尺度 -> 原图） ----------------
    props_net = buf.get("rpn_proposals_net") or []
    props_orig = boxes_net_to_orig(props_net, sf4)
    l4: Dict[str, Any] = {
        "num_proposals": len(props_net),
        "scale_factor": sf4,
        "coordinate_note": "proposals 来自 simple_test_bboxes 入参（网络输入尺度），已除以 scale_factor 还原原图；scale_factor 无法识别时坐标写 null",
        "proposals_orig_xyxy_top64": [[round(v, 3) for v in p[:5]] for p in props_orig[:64]] if props_orig else None,
        "rpn_top64_max_iou": None, "rpn_all_max_iou": None, "rpn_top64_hits_iou_05": None,
        "provenance": {"boxes": PROV_CAPTURED, "ious": PROV_DERIVED},
    }
    if props_orig and gt_valid:
        ious = [compute_iou_xyxy(p, gt_xyxy) for p in props_orig]
        l4["rpn_all_max_iou"] = round(max(ious), 6)
        l4["rpn_top64_max_iou"] = round(max(ious[:64]), 6)
        l4["rpn_top64_hits_iou_05"] = int(sum(1 for v in ious[:64] if v >= 0.5))

    # ---------------- Layer 5：源码决赛圈 Top-5（真实值） ----------------
    prov = match_candidate_provenance(final_rows, tra, det)
    bg_calls = buf.get("bg_matching_calls") or []
    penalties = [c.get("max_score") for c in bg_calls] if len(bg_calls) == len(final_rows) else None
    cands = decompose_candidate_scores(final_rows, prov, penalties)
    selected_idx = max(range(len(final_rows)), key=lambda i: float(final_rows[i][4])) if final_rows else None
    for i, c in enumerate(cands):
        c["selected_by_tracker"] = (i == selected_idx)
        c["gt_iou"] = round(compute_iou_xyxy(c["bbox_xyxy"], gt_xyxy), 6) if gt_valid else None
        c["bg_matching_num_bg_rois"] = bg_calls[i].get("num_bg_rois") if i < len(bg_calls) else None
    order_final = sorted(range(len(cands)), key=lambda i: -float(cands[i]["final_score"])) if cands else []
    for r, i in enumerate(order_final, start=1):
        cands[i]["rank_by_final_score"] = r

    def _top5(rows: List[List[float]]) -> List[Dict[str, Any]]:
        srt = sorted(rows, key=lambda r: -float(r[4]))[:5]
        return [{"bbox_xyxy": [round(float(v), 3) for v in r[:4]], "score": float(r[4]),
                 "gt_iou": round(compute_iou_xyxy(r, gt_xyxy), 6) if gt_valid else None} for r in srt]

    best_iou_rank_final = None
    best_iou = None
    selected_iou = None
    if cands and gt_valid:
        best_i = max(range(len(cands)), key=lambda i: cands[i]["gt_iou"])
        best_iou = cands[best_i]["gt_iou"]
        best_iou_rank_final = cands[best_i]["rank_by_final_score"]
        selected_iou = cands[selected_idx]["gt_iou"] if selected_idx is not None else None

    no_partner_rows = rows_without_iou_reward_partner(final_rows)
    identity_checks = []
    for i in no_partner_rows:
        c = cands[i]
        if c["post_bg_score"] is not None and c["final_score"] is not None:
            identity_checks.append(abs(c["final_score"] - c["post_bg_score"]) <= 1e-4)
    l5 = {
        "source_semantics": "Top-5 = tra 与 det 拼接后按分数降序取前 5（siamdt_rcnn.py:452-460），非每分支各 5 个",
        "top_num_source": SOURCE_TOP_NUM,
        "num_candidates_captured": len(cands),
        "num_bg_matching_calls": len(bg_calls),
        "candidates": cands,
        "branch_top5": {"similarity_tracking_branch(tra)": _top5(tra), "detection_branch(det)": _top5(det)},
        "best_gt_iou_among_top5": best_iou,
        "best_gt_iou_candidate_rank_by_final_score": best_iou_rank_final,
        "selected_candidate_gt_iou": selected_iou,
        "capture_consistency": {
            "bg_matching_calls_equals_top_num": len(bg_calls) == len(final_rows) == SOURCE_TOP_NUM,
            "all_candidates_provenance_matched": all(p["branch"] not in ("unmatched",) for p in prov) if prov else False,
            "rows_without_iou_reward_partner": no_partner_rows,
            "final_equals_pre_minus_penalty_on_those_rows": all(identity_checks) if identity_checks else None,
            "simple_test_bboxes_rescale_flag": buf.get("simple_test_bboxes_rescale"),
        },
        "coordinate_note": "tra/det/最终框来自 rescale=True 的 get_bboxes，已是原图坐标（siamdt_tracking.py:77-78）",
    }

    # ---------------- Layer 6：模板更新审计（源码条件 + 运行时 lr） ----------------
    cond = source_update_condition(tra, det)
    up_flag = buf.get("up_flag")
    uq_calls = buf.get("update_query_calls") or []
    upd_box_orig = None
    upd_box_gt_iou = None
    if uq_calls and uq_calls[-1].get("update_box_net_xyxy") is not None:
        conv = boxes_net_to_orig([uq_calls[-1]["update_box_net_xyxy"]], sf4)
        upd_box_orig = [round(v, 3) for v in conv[0][:4]] if conv else None
        if upd_box_orig is not None and gt_valid:
            upd_box_gt_iou = round(compute_iou_xyxy(upd_box_orig, gt_xyxy), 6)
    l6 = {
        "template_updated": up_flag,
        "update_query_called": bool(uq_calls),
        "learning_rate": buf.get("learning_rate"),
        "ema_rule_source": "_bbox_feats_z = (1 - lr) * _bbox_feats_z + lr * new  (siamdt_rcnn.py:402)",
        "bg_bank_refreshed_on_update": "update 时 _bbox_feats_bg 整体替换为新的 <=11 个背景 ROI（siamdt_rcnn.py:410-426）",
        "source_update_condition": cond,
        "source_condition_consistent_with_up_flag": (bool(cond["condition_met"]) == bool(up_flag)) if (cond is not None and up_flag is not None) else None,
        "update_box_orig_xyxy": upd_box_orig,
        "update_box_gt_iou": upd_box_gt_iou,
        "posthoc_suspicious_update": (bool(up_flag) and upd_box_gt_iou is not None and upd_box_gt_iou < ANALYST_UPDATE_GT_IOU_THR) if up_flag else False,
        "posthoc_suspicious_update_rule": f"分析者阈值：更新框与 GT 的 IoU < {ANALYST_UPDATE_GT_IOU_THR}（非模型行为，模型推理时看不到 GT）",
        "template_delta_l2_this_frame": uq_calls[-1].get("template_delta_l2") if uq_calls else None,
        "bg_bank_size_after_update": uq_calls[-1].get("bg_bank_size_after") if uq_calls else None,
    }
    return {"layer4_rpn_proposals": l4, "layer5_rcnn_top5": l5, "layer6_template_update_audit": l6}


def diagnose_primary_bottleneck(layers: Dict[str, Any]) -> Dict[str, Any]:
    """【分析者启发式标签，非模型输出】依据真实捕获量给出诊断标签；缺失字段时返回 INSUFFICIENT_DATA。"""
    l4 = layers.get("layer4_rpn_proposals", {})
    l5 = layers.get("layer5_rcnn_top5", {})
    l6 = layers.get("layer6_template_update_audit", {})
    l3 = layers.get("layer3a_rpn_modulation", {}).get("combined", {})
    rpn_all = l4.get("rpn_all_max_iou")
    best = l5.get("best_gt_iou_among_top5")
    sel = l5.get("selected_candidate_gt_iou")
    best_rank = l5.get("best_gt_iou_candidate_rank_by_final_score")
    base = {"provenance": PROV_ANALYST}
    if l6.get("posthoc_suspicious_update"):
        return {**base, "verdict_code": "L6_SUSPICIOUS_TEMPLATE_UPDATE",
                "summary_zh": f"源码条件触发了模板更新，但更新框与 GT 的 IoU={l6.get('update_box_gt_iou')}（< 分析者阈值 {ANALYST_UPDATE_GT_IOU_THR}），疑似把非目标写入模板。"}
    if rpn_all is None or best is None or sel is None:
        return {**base, "verdict_code": "INSUFFICIENT_DATA", "summary_zh": "GT 不可用（目标消失/无效）或捕获字段缺失，不给出诊断。"}
    if rpn_all < ANALYST_RPN_MISS_IOU:
        margin = l3.get("target_vs_strongest_distractor_margin")
        hint = f"；RPN 调制响应（派生）峰值裕度={margin}" if margin is not None else ""
        return {**base, "verdict_code": "L4_RPN_PROPOSAL_MISS",
                "summary_zh": f"全部 RPN 候选与 GT 的最大 IoU={rpn_all:.3f} (< {ANALYST_RPN_MISS_IOU})，目标在第一阶段即未被召回{hint}。"}
    if best < ANALYST_RPN_MISS_IOU:
        return {**base, "verdict_code": "L5_TARGET_NOT_IN_TOP5",
                "summary_zh": f"RPN 已召回目标（max IoU={rpn_all:.3f}），但源码决赛圈 Top-5 中最大 IoU 仅 {best:.3f}，目标未进入背景抑制环节。"}
    if best >= ANALYST_RANK_INV_BEST_IOU and sel < ANALYST_RANK_INV_SELECTED_IOU and best_rank and best_rank > 1:
        cands = l5.get("candidates", [])
        sel_c = next((c for c in cands if c.get("selected_by_tracker")), None)
        best_c = max(cands, key=lambda c: c.get("gt_iou") or -1.0) if cands else None
        cause = "unknown"
        if sel_c and best_c and None not in (sel_c.get("pre_bg_supp_score"), best_c.get("pre_bg_supp_score"), sel_c.get("bg_penalty"), best_c.get("bg_penalty")):
            if best_c["pre_bg_supp_score"] >= sel_c["pre_bg_supp_score"]:
                cause = "background_suppression_or_iou_reward_reversed_order"
            else:
                cause = "rcnn_raw_score_already_lower"
        return {**base, "verdict_code": "L5_RANK_INVERSION", "inversion_cause": cause,
                "summary_zh": f"真目标在 Top-5 中（IoU={best:.3f}，按最终得分排第 {best_rank}），但被选中的框 IoU={sel:.3f}；原因判别（真实分数对比）：{cause}。"}
    if sel < 0.5:
        return {**base, "verdict_code": "L5_LOCALIZATION_DRIFT", "summary_zh": f"被选中框 IoU={sel:.3f}（Top-5 最佳 {best:.3f}）。"}
    return {**base, "verdict_code": "HEALTHY_TRACKING", "summary_zh": f"被选中框 IoU={sel:.3f}，RPN 召回 IoU={rpn_all:.3f}。"}


def build_frame_anatomy_from_hook_buffer(
    seq_id: str, frame_id: int, phase_tag: str, img_rgb: Any, gt_xywh: List[float], exist_f: int,
    pred_xywh: List[float], iou_val: Optional[float], conf_val: float, up_flag: bool,
    hook_mgr: SiamDT6LayerHookManager,
) -> Dict[str, Any]:
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    img_h, img_w = img_rgb.shape[:2]
    buf = hook_mgr.frame_buffer
    model = hook_mgr.model
    gt_for_maps = gt_xywh if exist_f == 1 else [0.0, 0.0, 0.0, 0.0]

    def _resize(mp: Any) -> Any:
        arr = np.asarray(mp, dtype=np.float32).squeeze()
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            return np.zeros((img_h, img_w), dtype=np.float32)
        return cv2.resize(arr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    def _norm01(mp: Any) -> Any:
        arr = np.asarray(mp, dtype=np.float32)
        return (arr - float(arr.min())) / max(1e-8, float(arr.max()) - float(arr.min()))

    # 注意：特征图覆盖的是"缩放+补边后的网络输入"，映射回原图时需按 img_shape（未补边部分）裁剪后再缩放
    meta = buf.get("img_meta_x") or {}
    img_shape = meta.get("img_shape") or []

    def _crop_valid(mp: Any, stride: int) -> Any:
        arr = np.asarray(mp, dtype=np.float32).squeeze()
        if len(img_shape) >= 2 and arr.ndim == 2:
            vh = max(1, int(math.ceil(img_shape[0] / float(stride))))
            vw = max(1, int(math.ceil(img_shape[1] / float(stride))))
            arr = arr[:vh, :vw]
        return arr

    l1_per, l1_maps = {}, {}
    for st in buf.get("layer1_backbone_stages", []):
        stride = 2 ** (st["stage_index"] + 1)
        mp = _crop_valid(st["l2_map"], stride)
        l1_per[st["stage_name"]] = {**compute_spatial_map_metrics(mp, gt_for_maps, img_h, img_w), "shape": st["shape"], "stride": stride}
        l1_maps[st["stage_name"]] = _norm01(_resize(mp))

    l2_per, strips = {}, []
    for lv in buf.get("layer2_fpn_levels", []):
        mp = _crop_valid(lv["l2_map"], lv["stride"])
        l2_per[lv["level_name"]] = {**compute_spatial_map_metrics(mp, gt_for_maps, img_h, img_w), "shape": lv["shape"], "stride": lv["stride"]}
        strips.append(cv2.resize(_norm01(mp), (120, 120), interpolation=cv2.INTER_NEAREST))
    fpn_montage = np.concatenate(strips, axis=1) if strips else np.zeros((120, 600), dtype=np.float32)

    l3a_per, acc, n_acc = {}, np.zeros((img_h, img_w), dtype=np.float32), 0
    for k, xc in enumerate(buf.get("layer3a_rpn_modulation_levels", [])):
        if xc is None:
            continue
        mp = _crop_valid(xc["l2_map"], 2 ** (k + 2))
        l3a_per[xc["level_name"]] = {**compute_spatial_map_metrics(mp, gt_for_maps, img_h, img_w), "shape": xc["shape"], "abs_mean": xc["abs_mean"]}
        acc += _norm01(_resize(mp))
        n_acc += 1
    if n_acc:
        acc /= float(n_acc)
    l3a = {
        "source": "rpn_similarity_learning.proj_out[k] 输出 = x_corr[k]（similarity_encoders.py:30-32；_process_gallary:437-441 与 x[k] 相加）",
        "per_level": l3a_per,
        "combined": compute_spatial_map_metrics(acc, gt_for_maps, img_h, img_w) if n_acc else {},
        "combined_map_note": "combined 为各层通道 L2 图归一化后取平均 —— " + PROV_DERIVED,
        "num_levels_captured": n_acc,
    }

    # Layer 3B：Stage-2 相似度响应（逐 ROI 能量为派生量）
    rcnn_main = [c for c in buf.get("layer3b_rcnn_similarity_calls", []) if c["phase"] == "rcnn_main"]
    bg_sim = [c for c in buf.get("layer3b_rcnn_similarity_calls", []) if c["phase"] == "bg_matching"]
    sf4 = normalize_scale_factor(meta.get("scale_factor_raw"))
    props_orig = boxes_net_to_orig(buf.get("rpn_proposals_net") or [], sf4) or []
    energy_vs_iou: List[Tuple[float, float]] = []
    if rcnn_main and props_orig and exist_f == 1 and len(rcnn_main[0]["per_roi_l2_energy"]) == len(props_orig):
        gx = xywh_to_xyxy(gt_xywh)
        energy_vs_iou = [(e, compute_iou_xyxy(p, gx)) for e, p in zip(rcnn_main[0]["per_roi_l2_energy"], props_orig)]
    l3b: Dict[str, Any] = {
        "source": "rcnn_similarity_learning.proj_out 输出 = proj_out(proj_x(x) * proj_z(z))（similarity_encoders.py:53）",
        "rcnn_main_out_shape": rcnn_main[0]["out_shape"] if rcnn_main else None,
        "bg_matching_out_shapes": [c["out_shape"] for c in bg_sim],
        "per_roi_l2_energy_note": PROV_DERIVED,
        "energy_of_best_iou_proposal": None, "max_energy": None, "rank_of_best_iou_proposal_by_energy": None,
    }
    if energy_vs_iou:
        best_j = max(range(len(energy_vs_iou)), key=lambda j: energy_vs_iou[j][1])
        energies = [e for e, _ in energy_vs_iou]
        l3b["energy_of_best_iou_proposal"] = round(energies[best_j], 6)
        l3b["max_energy"] = round(max(energies), 6)
        l3b["rank_of_best_iou_proposal_by_energy"] = 1 + sum(1 for e in energies if e > energies[best_j])

    l456 = build_layer5_and_6_records(buf, gt_xywh, exist_f)

    # Layer 6 模板派生统计
    z_curr = getattr(model, "_bbox_feats_z", None)
    z_init = hook_mgr.initial_template_feat
    bg_bank = getattr(model, "_bbox_feats_bg", None)
    cos_sim, drift = None, None
    z_init_map = np.zeros((7, 7), dtype=np.float32)
    z_curr_map = np.zeros((7, 7), dtype=np.float32)
    bg_mean_map = np.zeros((7, 7), dtype=np.float32)
    if torch.is_tensor(z_curr):
        zc = z_curr.detach().float()
        z_curr_map = _norm01(torch.norm(zc[0], p=2, dim=0).cpu().numpy())
        if torch.is_tensor(z_init):
            zi = z_init.to(zc.device)
            z_init_map = _norm01(torch.norm(zi[0], p=2, dim=0).cpu().numpy())
            cos_sim = float(F.cosine_similarity(zc.reshape(1, -1), zi.reshape(1, -1), dim=1).item())
            drift = float(torch.norm(zc - zi).item() / max(1e-6, torch.norm(zi).item()))
    if torch.is_tensor(bg_bank):
        bg_mean_map = _norm01(torch.norm(bg_bank.detach().float(), p=2, dim=1).mean(dim=0).cpu().numpy())
    lr = l456["layer6_template_update_audit"].get("learning_rate")
    n_upd = hook_mgr.cumulative_update_count
    l456["layer6_template_update_audit"].update({
        "cumulative_update_count": n_upd,
        "init_template_residual_weight": round((1.0 - lr) ** n_upd, 6) if lr is not None else None,
        "template_cosine_sim_to_init": round(cos_sim, 6) if cos_sim is not None else None,
        "template_l2_drift_ratio": round(drift, 6) if drift is not None else None,
        "bg_bank_size_current": int(bg_bank.shape[0]) if torch.is_tensor(bg_bank) else None,
        "derived_note": "cosine/drift/residual_weight 为 " + PROV_DERIVED,
    })
    sep = np.zeros((7, 1), dtype=np.float32)
    template_comp_map = np.concatenate([z_init_map, sep, z_curr_map, sep, bg_mean_map], axis=1)

    layers = {
        "layer1_backbone": {"per_stage": l1_per},
        "layer2_fpn": {"per_level": l2_per},
        "layer3a_rpn_modulation": l3a,
        "layer3b_rcnn_similarity": l3b,
        **l456,
    }
    layers["root_cause_verdict"] = diagnose_primary_bottleneck(layers)

    return {
        "seq_id": seq_id, "frame_id": frame_id, "phase_tag": phase_tag, "exist": exist_f,
        "gt_bbox": [round(float(x), 3) for x in gt_xywh],
        "pred_bbox": [round(float(x), 3) for x in pred_xywh],
        "iou": round(float(iou_val), 6) if iou_val is not None else None,
        "confidence": round(float(conf_val), 6),
        "template_updated": bool(up_flag),
        "metrics": layers,
        "visual_maps": {
            "layer1_stage1": l1_maps.get("Stage1_C2", np.zeros((img_h, img_w), dtype=np.float32)),
            "layer1_stage4": l1_maps.get("Stage4_C5", np.zeros((img_h, img_w), dtype=np.float32)),
            "layer2_fpn_montage": fpn_montage,
            "layer3a_rpn_modulation_map": acc,
            "layer3b_energy_vs_iou": energy_vs_iou,
            "layer6_template_comparison_map": template_comp_map,
            "proposals_orig": props_orig[:64],
        },
    }


# ==============================================================================
# 5. 全景图（标题中明确标注派生量）
# ==============================================================================

def render_6layer_anatomy_figure(frame_rgb: Any, fd: Dict[str, Any], output_png_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    m = fd["metrics"]
    maps = fd["visual_maps"]
    verdict = m["root_cause_verdict"]
    img_h, img_w = frame_rgb.shape[:2]
    fig, axes = plt.subplots(3, 4, figsize=(24, 16.5), dpi=140)
    fig.patch.set_facecolor("#0f172a")
    iou_txt = f"{fd['iou']:.3f}" if fd["iou"] is not None else "n/a"
    fig.suptitle(
        f"SiamDT 6-Layer Anatomy v{PROBE_VERSION} | {fd['seq_id']} #{fd['frame_id']:04d} ({fd['phase_tag']}) | IoU={iou_txt} | "
        f"Updated={fd['template_updated']} | [{verdict['verdict_code']}] (analyst heuristic)\n"
        "Legend: [captured]=model path value; [derived]=post-hoc, NOT model path",
        fontsize=12.5, fontweight="bold", color="#f8fafc", y=0.985)

    def _style(ax: Any, title: str) -> None:
        ax.set_facecolor("#1e293b")
        ax.set_title(title, fontsize=9.8, fontweight="bold", color="#e2e8f0", pad=6)
        ax.tick_params(colors="#cbd5e1", labelsize=7.5)

    def _boxes(ax: Any) -> None:
        gx, gy, gw, gh = fd["gt_bbox"]
        px, py, pw, ph = fd["pred_bbox"]
        if gw > 0 and gh > 0:
            ax.add_patch(patches.Rectangle((gx, gy), gw, gh, lw=2.0, ec="#22c55e", fc="none"))
        if pw > 0 and ph > 0:
            ax.add_patch(patches.Rectangle((px, py), pw, ph, lw=2.0, ls="--", ec="#ef4444", fc="none"))

    def _fmt(v: Any, spec: str = ".2f") -> str:
        return format(v, spec) if isinstance(v, (int, float)) else "n/a"

    ax = axes[0, 0]
    ax.imshow(frame_rgb)
    _boxes(ax)
    _style(ax, f"[Input] Frame #{fd['frame_id']:04d}  Green=GT  Red=Pred(selected)")

    for ax, key, sname, ttl in ((axes[0, 1], "layer1_stage1", "Stage1_C2", "L1 Swin Stage-1 (C2,s=4)"),
                                (axes[0, 2], "layer1_stage4", "Stage4_C5", "L1 Swin Stage-4 (C5,s=32)")):
        im = ax.imshow(maps[key], cmap="magma", extent=[0, img_w, img_h, 0])
        _boxes(ax)
        st = m["layer1_backbone"]["per_stage"].get(sname, {})
        _style(ax, f"[{ttl}] channel-L2 [derived]\nT/BG={_fmt(st.get('target_to_bg_activation_ratio'))} Margin={_fmt(st.get('target_vs_strongest_distractor_margin'), '+.2f')}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 3]
    im = ax.imshow(maps["layer2_fpn_montage"], cmap="viridis", aspect="auto")
    ratios = " | ".join(f"P{k}:{_fmt(m['layer2_fpn']['per_level'].get(f'P{k}', {}).get('target_to_bg_activation_ratio'))}" for k in range(2, 7))
    _style(ax, f"[L2 FPN P2..P6] channel-L2 [derived]\nT/BG {ratios}")
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    im = ax.imshow(maps["layer3a_rpn_modulation_map"], cmap="jet", extent=[0, img_w, img_h, 0])
    _boxes(ax)
    c3 = m["layer3a_rpn_modulation"].get("combined", {})
    _style(ax, f"[L3A] RPN_Similarity proj_out[k] response [captured tensor; L2 map derived]\n"
               f"levels={m['layer3a_rpn_modulation']['num_levels_captured']} T/BG={_fmt(c3.get('target_to_bg_activation_ratio'))} Margin={_fmt(c3.get('target_vs_strongest_distractor_margin'), '+.3f')}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    ev = maps["layer3b_energy_vs_iou"]
    if ev:
        ax.scatter([e for e, _ in ev], [i for _, i in ev], s=6, c="#38bdf8", alpha=0.6)
        ax.set_xlabel("per-ROI L2 energy of Stage-2 sim response [derived]", color="#cbd5e1", fontsize=8)
        ax.set_ylabel("proposal GT IoU [derived]", color="#cbd5e1", fontsize=8)
    l3b = m["layer3b_rcnn_similarity"]
    _style(ax, f"[L3B] RCNN_Similarity proj_out (template vs proposals)\n"
               f"out={l3b.get('rcnn_main_out_shape')} best-IoU ROI energy rank={l3b.get('rank_of_best_iou_proposal_by_energy')}")

    ax = axes[1, 2]
    ax.imshow(frame_rgb, alpha=0.6)
    for p in maps["proposals_orig"][:25]:
        ax.add_patch(patches.Rectangle((p[0], p[1]), max(1.0, p[2] - p[0]), max(1.0, p[3] - p[1]), lw=0.8, ec="#38bdf8", fc="none", alpha=0.7))
    _boxes(ax)
    l4 = m["layer4_rpn_proposals"]
    _style(ax, f"[L4] RPN proposals (orig coords, /scale_factor) [captured]\n"
               f"N={l4['num_proposals']} maxIoU@64={_fmt(l4['rpn_top64_max_iou'], '.3f')} maxIoU@all={_fmt(l4['rpn_all_max_iou'], '.3f')} [derived]")

    ax = axes[1, 3]
    ax.imshow(frame_rgb)
    _boxes(ax)
    l5 = m["layer5_rcnn_top5"]
    palette = ["#ef4444", "#f97316", "#eab308", "#06b6d4", "#a855f7"]
    for i, c in enumerate(l5["candidates"]):
        x1, y1, x2, y2 = c["bbox_xyxy"]
        col = palette[i % 5]
        ax.add_patch(patches.Rectangle((x1, y1), max(2.0, x2 - x1), max(2.0, y2 - y1), lw=2.2 if c["selected_by_tracker"] else 1.3, ec=col, fc="none"))
        br = "tra" if str(c["branch"]).startswith("similarity") else ("det" if str(c["branch"]).startswith("detection") else "?")
        ax.text(x1, max(10.0, y1 - 3.0), f"#{c['rank_pre_suppression']}({br}) S={_fmt(c['final_score'])} IoU={_fmt(c['gt_iou'])}",
                fontsize=7, color="white", bbox=dict(boxstyle="round,pad=0.15", fc=col, alpha=0.8, ec="none"))
    _style(ax, "[L5A] Source Top-5 (tra+det merged, :452-460) [captured]\nthick=selected by tracker; IoU [derived]")

    ax = axes[2, 0]
    cands = l5["candidates"]
    if cands:
        x = np.arange(len(cands))
        w = 0.2
        vals = lambda k: [c[k] if isinstance(c[k], (int, float)) else 0.0 for c in cands]  # noqa: E731
        ax.bar(x - 1.5 * w, vals("pre_bg_supp_score"), w, label="pre-supp [captured]", color="#60a5fa")
        ax.bar(x - 0.5 * w, vals("bg_penalty"), w, label="bg penalty [captured]", color="#f43f5e")
        ax.bar(x + 0.5 * w, vals("final_score"), w, label="final [captured]", color="#a855f7")
        ax.bar(x + 1.5 * w, [c["gt_iou"] or 0.0 for c in cands], w, label="GT IoU [derived]", color="#10b981")
        ax.set_xticks(x, [f"#{c['rank_pre_suppression']}" for c in cands], color="white")
        ax.legend(fontsize=7, facecolor="#0f172a", labelcolor="white")
    _style(ax, "[L5B] Real score decomposition (null shown as 0)\npost_bg/iou_reward = source arithmetic [recomputed]")

    ax = axes[2, 1]
    im = ax.imshow(maps["layer6_template_comparison_map"], cmap="inferno", aspect="auto")
    l6 = m["layer6_template_update_audit"]
    _style(ax, f"[L6A] z_init | z_current | BG-bank mean (channel-L2) [derived]\n"
               f"cos(z_t,z_0)={_fmt(l6.get('template_cosine_sim_to_init'), '.3f')} residual(1-lr)^n={_fmt(l6.get('init_template_residual_weight'), '.3f')}")
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2, 2]
    ax.axis("off")
    cond = l6.get("source_update_condition") or {}
    txt = [
        "[L6B] Template update audit (source :500, :402)",
        f"up_flag [captured]          : {l6.get('template_updated')}",
        f"lr (model._learning_rate)   : {l6.get('learning_rate')}",
        f"tra0+det0 score [recomputed]: {_fmt(cond.get('score_sum'), '.4f')}  (> {SOURCE_UPDATE_SCORE_SUM_THR})",
        f"IoU(det0,tra0) [recomputed] : {_fmt(cond.get('tra_det_top1_iou'), '.4f')}  (> {SOURCE_UPDATE_IOU_THR})",
        f"consistent with up_flag     : {l6.get('source_condition_consistent_with_up_flag')}",
        f"update box GT IoU [derived] : {l6.get('update_box_gt_iou')}",
        f"suspicious [analyst<{ANALYST_UPDATE_GT_IOU_THR}]    : {l6.get('posthoc_suspicious_update')}",
        f"cumulative updates          : {l6.get('cumulative_update_count')}",
    ]
    ax.text(0.02, 0.95, "\n".join(txt), transform=ax.transAxes, va="top", family="monospace", fontsize=8.5, color="#f1f5f9")
    _style(ax, "[L6B] Template update audit")

    ax = axes[2, 3]
    ax.axis("off")
    ax.text(0.02, 0.95, f"Verdict (analyst heuristic, NOT model output):\n{verdict['verdict_code']}\n\n{verdict['summary_zh']}",
            transform=ax.transAxes, va="top", fontsize=9, color="#fde047", wrap=True)
    _style(ax, "[Scorecard]")

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png_path), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 6. 真实序列推理主流程
# ==============================================================================

def find_failure_onset_from_telemetry(telemetry_jsonl: Path) -> int:
    rows = [json.loads(line) for line in telemetry_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows[1:]:
        if int(r.get("exist", 1)) == 1 and float(r.get("iou", 1.0)) < 0.10:
            return int(r["frame_id"])
    valid = [r for r in rows[1:] if int(r.get("exist", 1)) == 1]
    if valid:
        return int(min(valid, key=lambda x: float(x.get("iou", 1.0)))["frame_id"])
    return max(1, len(rows) // 2)


def _load_sequence(seq_dir: Path) -> Tuple[List[Path], List[List[float]], List[int]]:
    lbl_file = seq_dir / "IR_label.json"
    if not lbl_file.exists():
        raise FileNotFoundError(f"未找到序列标注文件: {lbl_file}")
    lbl = json.loads(lbl_file.read_text(encoding="utf-8"))
    gt_rects = lbl["gt_rect"]
    exist_flags = lbl.get("exist", [1] * len(gt_rects))
    frame_files = sorted(seq_dir.glob("*.jpg"))
    return frame_files, gt_rects, exist_flags


def _build_tracker(siamdt_dir: Path, config_rel: str, ckpt_rel: str) -> Tuple[Any, Any]:
    os.chdir(siamdt_dir)
    if str(siamdt_dir) not in sys.path:
        sys.path.insert(0, str(siamdt_dir))
    import init_paths  # noqa: F401
    import libs.data as data
    import libs.ops as ops
    from trackers import SiamDTTracker
    transforms = data.BasicPairTransforms(train=False)
    tracker = SiamDTTracker(config_rel, ckpt_rel, transforms, name_suffix="6layer_probe", visualize=False)
    return tracker, ops


def _gt_of(gt_rects: List[Any], exist_flags: List[int], f_idx: int) -> Tuple[List[float], int]:
    g = gt_rects[f_idx] if f_idx < len(gt_rects) else []
    gt = [float(x) for x in g] if isinstance(g, (list, tuple)) and len(g) == 4 else [0.0, 0.0, 0.0, 0.0]
    ex = int(exist_flags[f_idx]) if f_idx < len(exist_flags) else 1
    return gt, ex


def _init_tracker_on_seq(tracker: Any, ops: Any, frame_files: List[Path], gt_rects: List[Any]) -> None:
    import numpy as np
    gt0 = [float(x) for x in gt_rects[0]]
    init_xyxy = np.array([gt0[0], gt0[1], gt0[0] + gt0[2] - 1.0, gt0[1] + gt0[3] - 1.0], dtype=np.float32)
    tracker.init(ops.read_image(str(frame_files[0]), tracker.color_fmt), init_xyxy)


def run_live_siamdt_probe_on_sequence(
    siamdt_dir: Path, seq_dir: Path, output_dir: Path, config_rel: str, ckpt_rel: str,
    onset_frame: int = 0, explicit_frames: Optional[List[int]] = None, telemetry_jsonl: Optional[Path] = None,
    holdout_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    orig_cwd = Path.cwd()
    seq_dir = seq_dir.resolve()
    output_dir = output_dir.resolve()
    try:
        frame_files, gt_rects, exist_flags = _load_sequence(seq_dir)
        total_f = min(len(frame_files), len(gt_rects))
        if total_f < 2:
            raise RuntimeError(f"序列 {seq_dir.name} 帧数不足 ({total_f})")
        if onset_frame <= 0:
            onset_frame = find_failure_onset_from_telemetry(telemetry_jsonl) if (telemetry_jsonl and telemetry_jsonl.exists()) \
                else min(total_f - 1, max(10, total_f // 3))
        keyframes = dict(select_diagnostic_keyframes(total_f, onset_frame, explicit_frames))
        tracker, ops = _build_tracker(siamdt_dir, config_rel, ckpt_rel)
        hook_mgr = SiamDT6LayerHookManager(tracker.model)
        _init_tracker_on_seq(tracker, ops, frame_files, gt_rects)
        hook_mgr.snapshot_initial_template()
        output_dir.mkdir(parents=True, exist_ok=True)
        records: List[Dict[str, Any]] = []
        pngs: List[str] = []
        for f_idx in range(1, max(keyframes) + 1):
            hook_mgr.clear_frame_buffer()
            img = ops.read_image(str(frame_files[f_idx]), tracker.color_fmt)
            results, up_flag = tracker.update(img, return_all=True)
            max_ind = int(results[:, 4].argmax())
            bx = results[max_ind, :4]
            conf = float(results[max_ind, 4])
            pred_xywh = [float(bx[0]), float(bx[1]), float(bx[2] - bx[0] + 1.0), float(bx[3] - bx[1] + 1.0)]
            gt_f, ex_f = _gt_of(gt_rects, exist_flags, f_idx)
            iou_f = compute_iou_xyxy(xywh_to_xyxy(pred_xywh), xywh_to_xyxy(gt_f)) if (ex_f == 1 and gt_f[2] > 0 and gt_f[3] > 0) else None
            if f_idx not in keyframes:
                continue
            fa = build_frame_anatomy_from_hook_buffer(seq_dir.name, f_idx, keyframes[f_idx], img, gt_f, ex_f,
                                                      pred_xywh, iou_f, conf, bool(up_flag), hook_mgr)
            png = output_dir / f"6layer_anatomy_{seq_dir.name}_frame_{f_idx:04d}.png"
            render_6layer_anatomy_figure(img, fa, png)
            pngs.append(str(png))
            records.append({k: fa[k] for k in ("seq_id", "frame_id", "phase_tag", "exist", "gt_bbox", "pred_bbox", "iou", "confidence", "template_updated")}
                           | {"anatomy_figure_png": str(png), "layers": fa["metrics"]})
            print(f"[6-Layer Probe] {seq_dir.name} #{f_idx:04d} ({keyframes[f_idx]}) -> {png.name} | {fa['metrics']['root_cause_verdict']['verdict_code']}", flush=True)
        hook_mgr.remove_hooks()
        payload = {
            "seq_id": seq_dir.name,
            "probe_version": PROBE_VERSION,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verified_on_real_model": True,
            "holdout_guard": holdout_info,
            "onset_frame": onset_frame,
            "field_provenance": FIELD_PROVENANCE,
            "source_constants": {
                "top_num": SOURCE_TOP_NUM, "iou_reward_thr": SOURCE_IOU_REWARD_THR,
                "update_score_sum_thr": SOURCE_UPDATE_SCORE_SUM_THR, "update_iou_thr": SOURCE_UPDATE_IOU_THR,
                "ema_lr_at_init_source": SOURCE_EMA_LR_AT_INIT, "bg_bank_max": SOURCE_BG_BANK_MAX,
            },
            "analyst_thresholds": {
                "update_gt_iou": ANALYST_UPDATE_GT_IOU_THR, "rpn_miss_iou": ANALYST_RPN_MISS_IOU,
                "rank_inversion_best_iou": ANALYST_RANK_INV_BEST_IOU, "rank_inversion_selected_iou": ANALYST_RANK_INV_SELECTED_IOU,
            },
            "probed_keyframes_count": len(records),
            "generated_anatomy_figures": pngs,
            "keyframes": records,
        }
        (output_dir / f"6layer_metrics_{seq_dir.name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return payload
    finally:
        os.chdir(orig_cwd)


def run_model_smoke(siamdt_dir: Path, seq_dir: Path, output_dir: Path, config_rel: str, ckpt_rel: str,
                    num_frames: int = 5, holdout_info: Optional[Dict[str, Any]] = None) -> int:
    """在 Colab GPU 上实例化真实 SiamDTRCNN，跑 num_frames 帧，逐项断言捕获结果非空且形状合理。"""
    orig_cwd = Path.cwd()
    seq_dir = seq_dir.resolve()
    output_dir = output_dir.resolve()
    checks: List[Dict[str, Any]] = []

    def chk(frame: int, name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"frame": frame, "check": name, "pass": bool(ok), "detail": detail})

    env_info: Dict[str, Any] = {}
    try:
        import torch
        env_info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        frame_files, gt_rects, exist_flags = _load_sequence(seq_dir)
        tracker, ops = _build_tracker(siamdt_dir, config_rel, ckpt_rel)
        hm = SiamDT6LayerHookManager(tracker.model)
        _init_tracker_on_seq(tracker, ops, frame_files, gt_rects)
        hm.snapshot_initial_template()
        chk(0, "initial_template_captured", hm.initial_template_feat is not None,
            list(hm.initial_template_feat.shape) if hm.initial_template_feat is not None else None)
        for f_idx in range(1, min(num_frames, len(frame_files) - 1) + 1):
            hm.clear_frame_buffer()
            img = ops.read_image(str(frame_files[f_idx]), tracker.color_fmt)
            img_h, img_w = img.shape[:2]
            results, up_flag = tracker.update(img, return_all=True)
            b = hm.frame_buffer
            st = b["layer1_backbone_stages"]
            chk(f_idx, "L1_backbone_4_stages_nonempty", len(st) == 4 and all(len(s["shape"]) == 4 and s["shape"][0] == 1 and min(s["shape"][1:]) > 0 for s in st), [s["shape"] for s in st])
            lv = b["layer2_fpn_levels"]
            chk(f_idx, "L2_fpn_5_levels_256ch", len(lv) == 5 and all(l["shape"][1] == 256 and min(l["shape"][2:]) > 0 for l in lv), [l["shape"] for l in lv])
            l3 = b["layer3a_rpn_modulation_levels"]
            ok3 = all(x is not None for x in l3) and len(lv) == 5 and all(l3[k]["shape"] == lv[k]["shape"] for k in range(5))
            chk(f_idx, "L3A_proj_out_5_levels_shape_eq_fpn", ok3, [x["shape"] if x else None for x in l3])
            chk(f_idx, "L3A_proj_out_called_exactly_5_times", b["layer3a_call_count"] == 5, b["layer3a_call_count"])
            props = b["rpn_proposals_net"] or []
            chk(f_idx, "L4_proposals_nonempty_5cols", len(props) > 0 and all(len(p) == 5 for p in props[:5]), len(props))
            main = [c for c in b["layer3b_rcnn_similarity_calls"] if c["phase"] == "rcnn_main"]
            chk(f_idx, "L3B_rcnn_main_once_shape", len(main) == 1 and main[0]["out_shape"][1:] == [256, 7, 7] and main[0]["out_shape"][0] == len(props),
                [c["out_shape"] for c in main])
            bgsim = [c for c in b["layer3b_rcnn_similarity_calls"] if c["phase"] == "bg_matching"]
            bgc = b["bg_matching_calls"]
            chk(f_idx, "L3B_bg_matching_5_calls_shape", len(bgsim) == SOURCE_TOP_NUM and all(c["out_shape"][1:] == [256, 7, 7] for c in bgsim)
                and all(bgsim[i]["out_shape"][0] == bgc[i]["num_bg_rois"] for i in range(min(len(bgsim), len(bgc)))),
                [c["out_shape"] for c in bgsim])
            chk(f_idx, "L5_tra_det_nonempty", bool(b["tra_bboxes"]) and bool(b["det_bboxes"]) and len(b["tra_bboxes"][0]) == 5 and len(b["det_bboxes"][0]) == 5,
                [len(b["tra_bboxes"] or []), len(b["det_bboxes"] or [])])
            chk(f_idx, "L5_rescale_true", b["simple_test_bboxes_rescale"] is True, b["simple_test_bboxes_rescale"])
            chk(f_idx, "L5_bg_matching_calls_eq_top_num", len(bgc) == SOURCE_TOP_NUM and all(c["max_score"] is not None for c in bgc),
                [(c["num_bg_rois"], c["num_returned"]) for c in bgc])
            fr = b["final_results"] or []
            chk(f_idx, "L5_final_results_5x5", len(fr) == SOURCE_TOP_NUM and all(len(r) >= 5 for r in fr), len(fr))
            ret = results.tolist()
            chk(f_idx, "L5_wrapper_return_identical_to_tracker", [list(r[:5]) for r in ret] == fr)
            gt_f, ex_f = _gt_of(gt_rects, exist_flags, f_idx)
            rec = build_layer5_and_6_records(b, gt_f, ex_f)
            cc = rec["layer5_rcnn_top5"]["capture_consistency"]
            chk(f_idx, "L5_all_candidates_provenance_matched", cc["all_candidates_provenance_matched"],
                [c["branch"] for c in rec["layer5_rcnn_top5"]["candidates"]])
            chk(f_idx, "L5_final_eq_pre_minus_penalty_where_no_iou_partner", cc["final_equals_pre_minus_penalty_on_those_rows"] is not False,
                {"rows": cc["rows_without_iou_reward_partner"], "result": cc["final_equals_pre_minus_penalty_on_those_rows"]})
            l6 = rec["layer6_template_update_audit"]
            chk(f_idx, "L6_source_condition_consistent_with_up_flag", l6["source_condition_consistent_with_up_flag"] is True, l6["source_update_condition"])
            chk(f_idx, "L6_update_query_called_iff_up_flag", bool(b["update_query_calls"]) == bool(up_flag), [bool(up_flag), len(b["update_query_calls"])])
            chk(f_idx, "L6_learning_rate_captured", b["learning_rate"] is not None, b["learning_rate"])
            sf4 = normalize_scale_factor((b["img_meta_x"] or {}).get("scale_factor_raw"))
            chk(f_idx, "scale_factor_valid", sf4 is not None, sf4)
            mg = 0.25 * max(img_w, img_h)
            chk(f_idx, "final_boxes_within_image_bounds", all(-mg <= r[0] <= img_w + mg and -mg <= r[1] <= img_h + mg and -mg <= r[2] <= img_w + mg and -mg <= r[3] <= img_h + mg for r in fr),
                {"img_wh": [img_w, img_h]})
            po = boxes_net_to_orig(props[:64], sf4) or []
            chk(f_idx, "proposals_orig_within_image_bounds", bool(po) and all(-mg <= p[0] and p[2] <= img_w + mg and -mg <= p[1] and p[3] <= img_h + mg for p in po))
        hm.remove_hooks()
        chk(-1, "wrappers_removed", all(n not in tracker.model.__dict__ for n in SiamDT6LayerHookManager.WRAPPED_METHODS))
    except Exception as exc:  # 任何异常都算 smoke 失败，不吞掉
        import traceback
        chk(-1, "exception", False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}")
    finally:
        os.chdir(orig_cwd)
    passed = bool(checks) and all(c["pass"] for c in checks)
    report = {
        "mode": "model_smoke", "probe_version": PROBE_VERSION, "seq_id": seq_dir.name, "num_frames": num_frames,
        "environment": env_info, "holdout_guard": holdout_info,
        "result": "PASS" if passed else "FAIL",
        "num_checks": len(checks), "num_failed": sum(1 for c in checks if not c["pass"]),
        "failed_checks": [c for c in checks if not c["pass"]],
        "checks": checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"model_smoke_{seq_dir.name}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("mode", "seq_id", "environment", "result", "num_checks", "num_failed")} | {"report": str(out), "failed_checks": report["failed_checks"][:10]},
                     indent=2, ensure_ascii=False, default=str))
    return 0 if passed else 1


# ==============================================================================
# 7. 离线几何/IO 检查（不加载模型；不代表模型验证）
# ==============================================================================

def _run_offline_geometry_check() -> int:
    results: List[Tuple[str, bool]] = []

    def ok(name: str, cond: bool) -> None:
        results.append((name, bool(cond)))

    ok("iou_identity", abs(compute_iou_xyxy([10, 10, 30, 30], [10, 10, 30, 30]) - 1.0) < 1e-9)
    ok("iou_disjoint", compute_iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0)
    ok("source_computeiou_matches_generic", abs(source_computeiou([0, 0, 10, 10], [5, 5, 15, 15]) - compute_iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15])) < 1e-12)
    ok("xywh_to_xyxy", xywh_to_xyxy([1, 2, 3, 4]) == [1.0, 2.0, 4.0, 6.0])
    kf = select_diagnostic_keyframes(total_frames=200, onset_frame=50)
    ok("keyframes_default", len(kf) == 5 and kf[2] == (50, "Failure Onset [0]"))
    ok("scale_factor_scalar", normalize_scale_factor(1.5625) == [1.5625] * 4)
    ok("scale_factor_array4", normalize_scale_factor([2.0, 1.5, 2.0, 1.5]) == [2.0, 1.5, 2.0, 1.5])
    ok("scale_factor_invalid_is_none", normalize_scale_factor([1.0, 2.0]) is None and normalize_scale_factor(None) is None and normalize_scale_factor(0.0) is None)
    conv = boxes_net_to_orig([[156.25, 78.125, 312.5, 156.25, 0.9]], [1.5625] * 4)
    ok("net_to_orig_restore", conv == [[100.0, 50.0, 200.0, 100.0, 0.9]])
    ok("net_to_orig_none_without_sf", boxes_net_to_orig([[1, 2, 3, 4]], None) is None)

    tra = [[10, 10, 20, 20, 0.95], [50, 50, 60, 60, 0.80], [70, 70, 80, 80, 0.30]]
    det = [[11, 11, 21, 21, 0.97], [90, 90, 99, 99, 0.60], [50, 50, 60, 60, 0.55]]
    # 源码排序：0.97(det0) 0.95(tra0) 0.80(tra1) 0.60(det1) 0.55(det2 坐标与 tra1 相同 -> ambiguous)
    pens = [0.10, 0.20, 0.05, 0.40, 0.05]
    pre = [0.97, 0.95, 0.80, 0.60, 0.55]
    final_rows = [[11, 11, 21, 21, None], [10, 10, 20, 20, None], [50, 50, 60, 60, None], [90, 90, 99, 99, None], [50, 50, 60, 60, None]]
    post = [p - q for p, q in zip(pre, pens)]
    rewards = [0.0] * 5
    for i in range(5):
        for j in range(5):
            if i != j:
                v = source_computeiou(final_rows[i][:4], final_rows[j][:4])
                if v > SOURCE_IOU_REWARD_THR:
                    rewards[i] += post[j] * v
    for i in range(5):
        final_rows[i][4] = post[i] + rewards[i]
    prov = match_candidate_provenance(final_rows, tra, det)
    ok("provenance_branches", [p["branch"] for p in prov] == ["detection_branch(det)", "similarity_tracking_branch(tra)",
                                                              "ambiguous_both_branches", "detection_branch(det)", "ambiguous_both_branches"])
    ok("provenance_pre_scores_or_null", [p["pre_bg_supp_score"] for p in prov] == [0.97, 0.95, None, 0.60, None])
    dec = decompose_candidate_scores(final_rows, prov, pens)
    ok("decompose_identity", all(abs(d["pre_bg_supp_score"] - d["bg_penalty"] + d["iou_reward"] - d["final_score"]) < 1e-9 for d in dec if d["pre_bg_supp_score"] is not None))
    ok("decompose_null_propagation", dec[2]["post_bg_score"] is None and dec[2]["iou_reward"] is None and dec[2]["final_score"] is not None)
    ok("decompose_penalty_length_mismatch_gives_null", all(d["bg_penalty"] is None for d in decompose_candidate_scores(final_rows, prov, [0.1, 0.2])))
    # 行 0/1 的 IoU=0.68（<=0.8，无奖励），行 2/4 坐标相同（IoU=1，互相奖励），行 3 孤立
    ok("no_partner_rows", rows_without_iou_reward_partner(final_rows) == [0, 1, 3])
    cond = source_update_condition([[10, 10, 20, 20, 0.96]], [[10, 10, 20, 20, 0.95]])
    ok("update_condition_met", cond is not None and cond["condition_met"] is True and abs(cond["score_sum"] - 1.91) < 1e-9)
    cond2 = source_update_condition([[10, 10, 20, 20, 0.96]], [[30, 30, 40, 40, 0.99]])
    ok("update_condition_not_met_iou", cond2 is not None and cond2["condition_met"] is False)
    ok("update_condition_empty_is_none", source_update_condition([], [[1, 1, 2, 2, 0.5]]) is None)

    buf = {
        "img_meta_x": {"scale_factor_raw": 1.5625}, "tra_bboxes": tra, "det_bboxes": det, "final_results": final_rows,
        "bg_matching_calls": [{"max_score": p, "num_bg_rois": 11} for p in pens], "rpn_proposals_net": [[15.625, 15.625, 31.25, 31.25, 0.9]],
        "up_flag": False, "update_query_calls": [], "learning_rate": 0.01, "simple_test_bboxes_rescale": True,
    }
    rec = build_layer5_and_6_records(buf, [10.0, 10.0, 10.0, 10.0], 1)
    ok("records_rpn_restored_iou", rec["layer4_rpn_proposals"]["rpn_all_max_iou"] == 1.0)
    ok("records_selected_is_argmax_final", sum(1 for c in rec["layer5_rcnn_top5"]["candidates"] if c["selected_by_tracker"]) == 1)
    ok("records_condition_consistency", rec["layer6_template_update_audit"]["source_condition_consistent_with_up_flag"] is True)

    with tempfile.TemporaryDirectory() as td:
        hj = Path(td) / "holdout_verification.json"
        hj.write_text(json.dumps({"sequences": ["SEQ_HOLD"]}), encoding="utf-8")
        refused = False
        try:
            enforce_holdout_guard("SEQ_HOLD", unlock_holdout=False, holdout_json=hj)
        except HoldoutSealedError:
            refused = True
        ok("holdout_default_refused", refused)
        ok("holdout_unlock_allowed", enforce_holdout_guard("SEQ_HOLD", True, hj)["is_holdout"] is True)
        ok("discovery_allowed", enforce_holdout_guard("SEQ_DISC", False, hj)["is_holdout"] is False)
        missing_refused = False
        try:
            enforce_holdout_guard("SEQ_X", False, Path(td) / "missing.json")
        except HoldoutSealedError:
            missing_refused = True
        ok("holdout_fail_closed_when_manifest_missing", missing_refused)

    # 静态检查：源码中不得残留 v1 的 Gram 相关标识符或字典键（模块 docstring 除外）
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    doc_node = tree.body[0] if tree.body and isinstance(tree.body[0], ast.Expr) else None
    bad: List[str] = []
    needle = "".join(["g", "r", "a", "m"])  # 动态拼接，避免检查代码自身命中
    for node in ast.walk(tree):
        if node is doc_node or (doc_node is not None and node is getattr(doc_node, "value", None)):
            continue
        name = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 64:
            name = node.value
        if name and needle in name.lower():
            bad.append(name)
    ok("v1_second_order_matrix_identifiers_removed", not bad)

    passed = all(r for _, r in results)
    print(json.dumps({
        "offline_geometry_check": "PASS" if passed else "FAIL",
        "scope": "仅几何/IO/溯源/护栏纯函数的离线检查（纯标准库）；未加载真实 SiamDTRCNN，未在 GPU 上验证任何 hook。",
        "num_checks": len(results),
        "failed": [n for n, r in results if not r],
        "checks": {n: r for n, r in results},
    }, indent=2, ensure_ascii=False))
    return 0 if passed else 1


# ==============================================================================
# 8. CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SiamDT 六层模块特征解剖器（v2.0，真实路径捕获版）")
    p.add_argument("--seq-id", type=str, default="", help="序列名（如 20190925_101846_1_2）")
    p.add_argument("--seq-dir", type=str, default="", help="序列目录完整路径（不填则按 --seq-id 在常见位置查找）")
    p.add_argument("--siamdt-dir", type=str, default="/content/runtime/Anti-UAV410/trackers/SiamDT", help="SiamDT 源码运行目录")
    p.add_argument("--config", type=str, default="configs/siamdt_swin_tiny_sgd.py")
    p.add_argument("--ckpt", type=str, default="checkpoints/siamdt_swin_tiny_sgd.pth")
    p.add_argument("--onset-frame", type=int, default=0)
    p.add_argument("--frames", type=str, default="", help="显式帧号列表，逗号分隔")
    p.add_argument("--telemetry-jsonl", type=str, default="")
    p.add_argument("--output-dir", type=str, default="/content/drive/MyDrive/SiamDT_Project/experiments/failure_diagnosis/feature_anatomy_6layer")
    p.add_argument("--holdout-json", type=str, default="", help="holdout_verification.json 路径（默认自动查找；找不到则拒绝执行）")
    p.add_argument("--unlock-holdout", action="store_true", help="显式解封 Holdout 序列（结果写入独立 HOLDOUT_UNSEALED 目录）")
    p.add_argument("--offline-geometry-check", action="store_true", help="仅运行几何/IO 纯函数离线检查（不加载模型，不代表模型验证）")
    p.add_argument("--model-smoke", action="store_true", help="在 GPU 上实例化真实 SiamDTRCNN 跑 1 条序列 5 帧并断言所有捕获点")
    p.add_argument("--smoke-frames", type=int, default=5)
    return p.parse_args()


def _resolve_seq_dir(args: argparse.Namespace, seq_id: str) -> Path:
    if args.seq_dir:
        return Path(args.seq_dir)
    for cand in (Path(f"/content/runtime/datasets/AntiUAV410/val/{seq_id}"),
                 Path(f"/content/drive/MyDrive/SiamDT_Project/datasets/AntiUAV410/val/{seq_id}")):
        if cand.exists():
            return cand
    return Path(f"/content/drive/MyDrive/SiamDT_Project/datasets/AntiUAV410/val/{seq_id}")


def main() -> None:
    args = parse_args()
    if args.offline_geometry_check:
        sys.exit(_run_offline_geometry_check())

    seq_id = args.seq_id or (Path(args.seq_dir).name if args.seq_dir else "")
    if not seq_id and args.model_smoke:
        seq_id = load_discovery_first_sequence() or ""
    if not seq_id:
        print("错误: 请提供 --seq-id 或 --seq-dir（或使用 --offline-geometry-check / --model-smoke）。", file=sys.stderr)
        sys.exit(2)
    seq_dir = _resolve_seq_dir(args, seq_id)
    try:
        holdout_info = enforce_holdout_guard(seq_dir.name, args.unlock_holdout, Path(args.holdout_json) if args.holdout_json else None)
    except HoldoutSealedError as exc:
        print(f"[Holdout 封存] {exc}", file=sys.stderr)
        sys.exit(3)
    out_root = Path(args.output_dir)
    if holdout_info.get("is_holdout") or holdout_info.get("holdout_check") == "UNVERIFIED_UNLOCKED":
        out_root = out_root / "HOLDOUT_UNSEALED"

    if args.model_smoke:
        sys.exit(run_model_smoke(Path(args.siamdt_dir), seq_dir, out_root / "_model_smoke", args.config, args.ckpt,
                                 num_frames=args.smoke_frames, holdout_info=holdout_info))

    explicit = [int(x.strip()) for x in args.frames.split(",") if x.strip()] if args.frames.strip() else None
    summary = run_live_siamdt_probe_on_sequence(
        siamdt_dir=Path(args.siamdt_dir), seq_dir=seq_dir, output_dir=out_root / seq_dir.name,
        config_rel=args.config, ckpt_rel=args.ckpt, onset_frame=args.onset_frame, explicit_frames=explicit,
        telemetry_jsonl=Path(args.telemetry_jsonl) if args.telemetry_jsonl else None, holdout_info=holdout_info,
    )
    print(json.dumps({"status": "COMPLETED", "seq_id": summary["seq_id"], "probed_keyframes_count": summary["probed_keyframes_count"],
                      "generated_anatomy_figures": summary["generated_anatomy_figures"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
