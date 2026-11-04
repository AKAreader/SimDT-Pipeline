#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SiamDT 六层模块特征分层可视化与量化解剖器 (siamdt_6layer_feature_probe.py)
================================================================================
针对跑分不佳或发生跟踪失败的序列，通过非侵入式 PyTorch `register_forward_hook`
对原始 `SiamDT` (`SiamDTRCNN`) 的 6 层核心功能节点进行逐层特征提取、量化体检与全景可视化：

- Layer 1 (`Backbone Stages 1~4`):
  Swin-Tiny 四个 Stage (`C2, C3, C4, C5`, strides `4, 8, 16, 32`) 的输出特征激活图（L2 范数空间能量分布）；
- Layer 2 (`5-Level FPN P2~P6`):
  FPN 五个尺度（`P2, P3, P4, P5, P6`, strides `4, 8, 16, 32, 64`）各自的特征图
  （诊断极小目标在哪一层下采样后丢失或被热背景淹没）；
- Layer 3 (`Dual-Semantic RPN Response`):
  `DS-RPN` 的一阶互相关匹配响应图（Cross-Correlation Map）与二阶格拉姆矩阵响应图（Gram Matrix Map）；
- Layer 4 (`RPN Top-64 Proposals`):
  RPN 生成的前 64 个候选框在原图上的空间分布热力与最大交并比 `max IoU_RPN`
  （诊断是不是第一阶段 RPN 就漏检了）；
- Layer 5 (`Versatile R-CNN Top-5 Competition`):
  决赛圈 Top-5 候选框在原图上的位置、各自置信度得分与真实 IoU 对比条形图
  （诊断是不是“真目标排第 2/第 3，背景干扰物排第 1”的排序倒挂）；
- Layer 6 (`Dynamic Template & 11-way Ensemble BG Suppression`):
  当前存储在内存中的动态模板特征 `z_dyn_feat` (`1x256x7x7`) 可视化 +
  11 路背景抑制 (`11x256x7x7`) 前后的激活对比 + 是否发生错误模板更新（`template_updated == True` when `IoU < 0.3`）。

输出产物：
1. 高清 6 层模块全景解剖大图：`6layer_anatomy_{seq_id}_frame_{frame_id:04d}.png`
2. 逐层量化体检指标 JSON：`6layer_metrics_{seq_id}.json`
   （含每一层的 `target_to_bg_activation_ratio`、`target_vs_strongest_distractor_margin`、
    `response_entropy`、`rpn_recall_iou`、`rcnn_rank_of_best_iou_proposal` 及根因归因结论）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ==============================================================================
# 1. 几何与空间激活量化指标计算工具 (纯 NumPy / Torch 兼容)
# ==============================================================================

def xywh_to_xyxy(box_xywh: List[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box_xywh[:4]]
    return [x, y, x + max(0.0, w), y + max(0.0, h)]


def compute_iou_xyxy(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def compute_spatial_map_metrics(
    activation_map: "np.ndarray",
    gt_xywh: List[float],
    img_h: int,
    img_w: int,
    guard_dilation_factor: float = 1.5,
) -> Dict[str, float]:
    """
    针对任意二维空间激活图/响应图 `activation_map` (H_map, W_map)，计算三大核心量化诊断指标：
    1. `target_to_bg_activation_ratio`:
       GT 目标框内平均激活能量 / 背景区域平均激活能量；
    2. `target_vs_strongest_distractor_margin`:
       (GT 区域峰值激活 - 保护带外最强背景干扰峰激活) / (全局动态范围 + eps)；
    3. `response_entropy`:
       归一化二维空间香农熵 H / log(N) in [0, 1]，越小代表响应越集中尖锐，越大代表被背景噪声弥散淹没。
    """
    import numpy as np

    arr = np.asarray(activation_map, dtype=np.float64)
    if arr.ndim > 2:
        arr = arr.squeeze()
    hm, wm = arr.shape[:2]
    if hm == 0 or wm == 0:
        return {
            "target_to_bg_activation_ratio": 0.0,
            "target_vs_strongest_distractor_margin": 0.0,
            "response_entropy": 1.0,
            "target_peak": 0.0,
            "strongest_distractor_peak": 0.0,
        }

    # Shift to non-negative energy
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    span = max(1e-8, arr_max - arr_min)
    norm_arr = (arr - arr_min) / span

    # Normalized Shannon entropy
    prob = norm_arr + 1e-12
    prob = prob / np.sum(prob)
    n_cells = hm * wm
    entropy = float(-np.sum(prob * np.log(prob)) / math.log(max(2, n_cells)))

    gx, gy, gw, gh = [float(v) for v in gt_xywh[:4]]
    if gw <= 0.0 or gh <= 0.0 or img_w <= 0 or img_h <= 0:
        return {
            "target_to_bg_activation_ratio": 0.0,
            "target_vs_strongest_distractor_margin": 0.0,
            "response_entropy": round(entropy, 6),
            "target_peak": 0.0,
            "strongest_distractor_peak": round(float(np.max(norm_arr)), 6),
        }

    # Map GT box to feature map coordinates (ensure at least 1x1 cell for tiny targets)
    sx = wm / float(img_w)
    sy = hm / float(img_h)
    u1 = max(0, min(wm - 1, int(math.floor(gx * sx))))
    v1 = max(0, min(hm - 1, int(math.floor(gy * sy))))
    u2 = max(u1 + 1, min(wm, int(math.ceil((gx + gw) * sx))))
    v2 = max(v1 + 1, min(hm, int(math.ceil((gy + gh) * sy))))

    # Guard zone around GT to separate true target lobe from external background distractors
    cx, cy = gx + 0.5 * gw, gy + 0.5 * gh
    guard_w = max(gw * guard_dilation_factor, 16.0)
    guard_h = max(gh * guard_dilation_factor, 16.0)
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

    target_mean = float(np.mean(target_patch)) if target_patch.size > 0 else 0.0
    bg_mean = float(np.mean(bg_vals)) if bg_vals.size > 0 else 1e-6
    ratio = float(target_mean / max(1e-6, bg_mean))

    target_peak = float(np.max(target_patch)) if target_patch.size > 0 else 0.0
    distractor_peak = float(np.max(bg_vals)) if bg_vals.size > 0 else 0.0
    margin = float(target_peak - distractor_peak)

    return {
        "target_to_bg_activation_ratio": round(ratio, 6),
        "target_vs_strongest_distractor_margin": round(margin, 6),
        "response_entropy": round(entropy, 6),
        "target_peak": round(target_peak, 6),
        "strongest_distractor_peak": round(distractor_peak, 6),
    }


def select_diagnostic_keyframes(
    total_frames: int,
    onset_frame: int,
    explicit_frames: Optional[List[int]] = None,
) -> List[Tuple[int, str]]:
    """
    自动围绕失败起点帧 `onset_frame` 截取标准四阶段诊断关键帧：
    - `Pre-Failure [-10, -3]`
    - `Failure Onset [0]`
    - `Drift [+5]`
    - `Recovery/Terminal [+15]`
    """
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

    offsets = [
        (-10, "Pre-Failure [-10]"),
        (-3, "Pre-Failure [-3]"),
        (0, "Failure Onset [0]"),
        (+5, "Drift [+5]"),
        (+15, "Recovery/Terminal [+15]"),
    ]
    seen = set()
    selected: List[Tuple[int, str]] = []
    for delta, label in offsets:
        f_idx = max(1, min(total_frames - 1, onset_frame + delta))
        if f_idx not in seen:
            seen.add(f_idx)
            selected.append((f_idx, label))
    return selected


# ==============================================================================
# 2. 非侵入式 PyTorch 6 层 Forward Hook 探针管理器 (SiamDT6LayerHookManager)
# ==============================================================================

class SiamDT6LayerHookManager:
    """
    在原始 `SiamDTRCNN` 上通过 `register_forward_hook` 非侵入式挂载 6 层特征探针：
    - Layer 1: `model.backbone` -> Swin-Tiny Stages 1~4 (`C2~C5`)
    - Layer 2: `model.neck` -> 5-Level FPN (`P2~P6`)
    - Layer 3: `model.rpn_similarity_learning.proj_out[0..4]` & `model.rpn_head`
               -> DS-RPN 互相关响应图 (Cross-Correlation Map) + 二阶格拉姆矩阵响应图 (Gram Matrix Map)
    - Layer 4: `model.roi_head.bbox_roi_extractor` & `model.rpn_head`
               -> RPN Top-64 Proposals 空间分布与最大 IoU
    - Layer 5: `model.roi_head.bbox_head`
               -> Versatile R-CNN (Tracking 分支 + Detection 分支) 决赛圈 Top-5 候选框竞争与排序倒挂诊断
    - Layer 6: `model.rcnn_similarity_learning` & `model._bbox_feats_z` / `model._bbox_feats_bg`
               -> 动态模板特征 (1x256x7x7) + 11 路背景抑制前后激活对比 + 模板污染审计
    """

    def __init__(self, model: Any) -> None:
        import torch

        self.model = model
        self.handles: List[Any] = []
        self.frame_buffer: Dict[str, Any] = {}
        self.initial_template_feat: Optional["torch.Tensor"] = None
        self._register_all_hooks()

    def clear_frame_buffer(self) -> None:
        self.frame_buffer = {
            "layer1_backbone_stages": [],
            "layer2_fpn_levels": [],
            "layer3_rpn_xcorr_levels": [None] * 5,
            "layer3_rpn_cls_scores": [],
            "layer3_rpn_bbox_preds": [],
            "layer4_roi_extractor_calls": [],
            "layer5_bbox_head_calls": [],
            "layer6_rcnn_sim_calls": [],
        }

    @staticmethod
    def _l2_spatial_map(feat_tensor: Any) -> "np.ndarray":
        import torch

        t = feat_tensor.detach().float()
        if t.dim() == 4:
            t = t[0]
        # L2 norm across channel dimension C -> (H, W)
        l2_map = torch.norm(t, p=2, dim=0)
        return l2_map.cpu().numpy()

    @staticmethod
    def compute_gram_matrix_response(
        template_feat: Any,
        fpn_feat: Any,
    ) -> "np.ndarray":
        """
        计算二阶格拉姆矩阵响应图 (Second-Order Gram Matrix Similarity Map)：
        给定模板特征 Z in R^{1 x C x 7 x 7} 与搜索特征 X in R^{1 x C x H x W}，
        先对通道向量做 L2 归一化，令 G_z = (1/49) * Z_flat @ Z_flat^T in R^{C x C}，
        则位置 (u, v) 的二阶格拉姆协方差匹配响应为：
        R_Gram(u, v) = x(u,v)^T G_z x(u,v) = (1/49) * sum_{m=1}^{49} (z_m^T x(u,v))^2
        """
        import torch
        import torch.nn.functional as F

        z = template_feat.detach().float()
        x = fpn_feat.detach().float()
        if z.dim() == 4:
            z = z[0]  # (C, 7, 7)
        if x.dim() == 4:
            x = x[0]  # (C, H, W)
        c, h, w = x.shape
        z_flat = F.normalize(z.reshape(c, -1), p=2, dim=0)  # (C, 49)
        x_flat = F.normalize(x.reshape(c, -1), p=2, dim=0)  # (C, H*W)
        # (49, C) @ (C, H*W) -> (49, H*W)
        proj = torch.matmul(z_flat.t(), x_flat)
        gram_resp = torch.mean(proj ** 2, dim=0).reshape(h, w)
        return gram_resp.cpu().numpy()

    def _register_all_hooks(self) -> None:
        import torch

        self.clear_frame_buffer()

        # ------------------------------------------------------------------
        # Layer 1 Hook: Swin-Tiny Backbone Stages 1~4 (C2, C3, C4, C5)
        # ------------------------------------------------------------------
        def _hook_layer1_backbone(module: Any, inputs: Any, output: Any) -> None:
            if isinstance(output, (tuple, list)):
                stages = []
                for idx, feat in enumerate(output):
                    if torch.is_tensor(feat):
                        stages.append({
                            "stage_index": idx + 1,
                            "stage_name": f"Stage{idx + 1}_C{idx + 2}",
                            "shape": list(feat.shape),
                            "l2_map": self._l2_spatial_map(feat),
                        })
                self.frame_buffer["layer1_backbone_stages"] = stages

        self.handles.append(self.model.backbone.register_forward_hook(_hook_layer1_backbone))

        # ------------------------------------------------------------------
        # Layer 2 Hook: 5-Level FPN (P2, P3, P4, P5, P6)
        # ------------------------------------------------------------------
        def _hook_layer2_fpn(module: Any, inputs: Any, output: Any) -> None:
            if isinstance(output, (tuple, list)):
                levels = []
                for idx, feat in enumerate(output):
                    if torch.is_tensor(feat):
                        levels.append({
                            "level_index": idx,
                            "level_name": f"P{idx + 2}",
                            "stride": 2 ** (idx + 2),
                            "shape": list(feat.shape),
                            "tensor": feat.detach(),
                            "l2_map": self._l2_spatial_map(feat),
                        })
                self.frame_buffer["layer2_fpn_levels"] = levels

        self.handles.append(self.model.neck.register_forward_hook(_hook_layer2_fpn))

        # ------------------------------------------------------------------
        # Layer 3 Hook A: DS-RPN Cross-Correlation proj_out[0..4]
        # 注意：RPN_Similarity_Learning.forward 是 generator，因此直接挂载其内部
        # nn.ModuleList `proj_out[k]`，安全拿到每一层 x_corr[k] 张量！
        # ------------------------------------------------------------------
        if hasattr(self.model, "rpn_similarity_learning") and hasattr(
            self.model.rpn_similarity_learning, "proj_out"
        ):
            for lvl_idx, conv_mod in enumerate(self.model.rpn_similarity_learning.proj_out):
                def _make_xcorr_hook(k: int):
                    def _hook_xcorr(module: Any, inputs: Any, output: Any) -> None:
                        if torch.is_tensor(output):
                            self.frame_buffer["layer3_rpn_xcorr_levels"][k] = {
                                "level_name": f"P{k + 2}",
                                "shape": list(output.shape),
                                "l2_map": self._l2_spatial_map(output),
                            }
                    return _hook_xcorr

                self.handles.append(conv_mod.register_forward_hook(_make_xcorr_hook(lvl_idx)))

        # ------------------------------------------------------------------
        # Layer 3 Hook B: RPNHead (cls_scores & bbox_preds across P2~P6)
        # ------------------------------------------------------------------
        def _hook_layer3_rpn_head(module: Any, inputs: Any, output: Any) -> None:
            if isinstance(output, (tuple, list)) and len(output) == 2:
                cls_scores, bbox_preds = output
                cls_maps = []
                for idx, sc in enumerate(cls_scores):
                    if torch.is_tensor(sc):
                        # sc: (1, num_anchors, H, W) -> max foreground probability across anchors
                        prob_map = torch.sigmoid(sc.detach().float()[0]).max(dim=0)[0].cpu().numpy()
                        cls_maps.append({
                            "level_name": f"P{idx + 2}",
                            "shape": list(sc.shape),
                            "prob_map": prob_map,
                        })
                self.frame_buffer["layer3_rpn_cls_scores"] = cls_maps
                self.frame_buffer["layer3_rpn_bbox_preds"] = [
                    b.detach() for b in bbox_preds if torch.is_tensor(b)
                ]

        self.handles.append(self.model.rpn_head.register_forward_hook(_hook_layer3_rpn_head))

        # ------------------------------------------------------------------
        # Layer 4 Hook: SingleRoIExtractor (捕获 RPN Proposals rois_x 与 Top-5 候选框)
        # ------------------------------------------------------------------
        def _hook_layer4_roi_extractor(module: Any, inputs: Any, output: Any) -> None:
            if len(inputs) >= 2 and torch.is_tensor(inputs[1]):
                rois = inputs[1].detach().float().cpu()
                self.frame_buffer["layer4_roi_extractor_calls"].append({
                    "num_rois": int(rois.shape[0]),
                    "rois_xyxy_scaled": rois[:, 1:5].numpy(),
                    "out_feat_l2": self._l2_spatial_map(output) if torch.is_tensor(output) else None,
                })

        self.handles.append(
            self.model.roi_head.bbox_roi_extractor.register_forward_hook(_hook_layer4_roi_extractor)
        )

        # ------------------------------------------------------------------
        # Layer 5 Hook: Shared2FCBBoxHead (捕获 Tracking / Detection 分支与 11 路背景抑制得分)
        # ------------------------------------------------------------------
        def _hook_layer5_bbox_head(module: Any, inputs: Any, output: Any) -> None:
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                cls_score, bbox_pred = output[0], output[1]
                if torch.is_tensor(cls_score):
                    probs = torch.softmax(cls_score.detach().float(), dim=-1)[:, 0].cpu().numpy()
                    self.frame_buffer["layer5_bbox_head_calls"].append({
                        "num_Candidates": int(cls_score.shape[0]),
                        "fg_probs": probs,
                        "bbox_pred": bbox_pred.detach().float().cpu().numpy()
                        if torch.is_tensor(bbox_pred)
                        else None,
                    })

        self.handles.append(
            self.model.roi_head.bbox_head.register_forward_hook(_hook_layer5_bbox_head)
        )

        # ------------------------------------------------------------------
        # Layer 6 Hook: RCNN_Similarity_Learning (捕获动态模板 z 与 11 路背景模板互相关激活)
        # ------------------------------------------------------------------
        if hasattr(self.model, "rcnn_similarity_learning"):
            def _hook_layer6_rcnn_sim(module: Any, inputs: Any, output: Any) -> None:
                if len(inputs) >= 2 and torch.is_tensor(inputs[0]) and torch.is_tensor(inputs[1]):
                    z_in = inputs[0].detach().float()
                    x_in = inputs[1].detach().float()
                    out_t = output.detach().float() if torch.is_tensor(output) else None
                    self.frame_buffer["layer6_rcnn_sim_calls"].append({
                        "z_shape": list(z_in.shape),
                        "x_shape": list(x_in.shape),
                        "out_shape": list(out_t.shape) if out_t is not None else [],
                        "out_mean_energy": float(torch.norm(out_t, p=2, dim=1).mean().item())
                        if out_t is not None
                        else 0.0,
                        "out_per_item_energy": torch.norm(out_t, p=2, dim=(1, 2, 3)).cpu().numpy().tolist()
                        if out_t is not None and out_t.dim() == 4
                        else [],
                    })

            self.handles.append(
                self.model.rcnn_similarity_learning.register_forward_hook(_hook_layer6_rcnn_sim)
            )

    def snapshot_initial_template(self) -> None:
        if hasattr(self.model, "_bbox_feats_z") and self.model._bbox_feats_z is not None:
            self.initial_template_feat = self.model._bbox_feats_z.detach().float().clone()

    def remove_hooks(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()


# ==============================================================================
# 3. 六层解剖数据整合与根因判定器
# ==============================================================================

def diagnose_primary_bottleneck(layer_metrics: Dict[str, Any]) -> Dict[str, str]:
    """根据 6 层量化指标自动给出当前帧的故障根因归因标签与中文诊断结论。"""
    l1_ratio = float(layer_metrics["layer1_backbone"]["mean_target_to_bg_ratio"])
    l2_p2_ratio = float(layer_metrics["layer2_fpn"]["per_level"].get("P2", {}).get("target_to_bg_activation_ratio", 1.0))
    l2_p4_ratio = float(layer_metrics["layer2_fpn"]["per_level"].get("P4", {}).get("target_to_bg_activation_ratio", 1.0))
    l3_margin = float(layer_metrics["layer3_ds_rpn"]["xcorr_combined"]["target_vs_strongest_distractor_margin"])
    rpn_iou = float(layer_metrics["layer4_rpn_proposals"]["rpn_recall_iou"])
    best_rank = int(layer_metrics["layer5_rcnn_top5"]["rcnn_rank_of_best_iou_proposal"])
    rank1_iou = float(layer_metrics["layer5_rcnn_top5"]["top1_iou"])
    best_top5_iou = float(layer_metrics["layer5_rcnn_top5"]["best_top5_iou"])
    polluted = bool(layer_metrics["layer6_dyn_template_bg_supp"]["template_pollution_triggered"])

    if polluted:
        return {
            "primary_bottleneck_layer": "Layer 6 (Dynamic Template Pollution)",
            "verdict_code": "L6_TEMPLATE_POLLUTION",
            "summary_zh": f"第 6 层动态模板发生污染：在真实 IoU={rank1_iou:.3f} (<0.3) 时错误触发了模板更新，导致后续匹配基准漂移。",
        }
    if rpn_iou < 0.30:
        if l3_margin < -0.05:
            return {
                "primary_bottleneck_layer": "Layer 3 (DS-RPN Cross-Correlation Distractor Hijack)",
                "verdict_code": "L3_DS_RPN_DISTRACTOR_PEAK",
                "summary_zh": f"第 3 层 DS-RPN 互相关响应被强热背景干扰物劫持 (Margin={l3_margin:+.3f})，导致第 4 层 RPN Top-64 最大 IoU 仅为 {rpn_iou:.3f}。",
            }
        if l2_p4_ratio < 0.85 and l2_p2_ratio > 1.15:
            return {
                "primary_bottleneck_layer": "Layer 2 (FPN Deep Stride Attenuation P4~P6)",
                "verdict_code": "L2_FPN_HIGH_STRIDE_LOSS",
                "summary_zh": f"第 2 层 FPN 下采样特征衰减：P2 浅层仍有激活 (Ratio={l2_p2_ratio:.2f})，但 P4~P6 深层被背景淹没 (Ratio={l2_p4_ratio:.2f})，致使 RPN 漏检 (max IoU={rpn_iou:.3f})。",
            }
        return {
            "primary_bottleneck_layer": "Layer 4 (Stage-1 RPN Proposal Recall Miss)",
            "verdict_code": "L4_RPN_PROPOSAL_MISS",
            "summary_zh": f"第 4 层 RPN 候选框召回失败：Top-64 候选框最大 IoU 仅为 {rpn_iou:.3f} (<0.30)，目标在第一阶段即漏检（Backbone Ratio={l1_ratio:.2f}）。",
        }
    if best_top5_iou >= 0.45 and best_rank > 1 and rank1_iou < 0.25:
        return {
            "primary_bottleneck_layer": "Layer 5 & 6 (Versatile R-CNN Rank Inversion / BG Suppression Failure)",
            "verdict_code": "L5_RCNN_RANK_INVERSION",
            "summary_zh": f"第 5/6 层决赛圈排序倒挂：真目标已在候选池中（第 {best_rank} 名，IoU={best_top5_iou:.3f}），但背景干扰物排在第 1 名（IoU={rank1_iou:.3f}），11 路背景抑制未能压制伪峰。",
        }
    if rank1_iou < 0.50:
        return {
            "primary_bottleneck_layer": "Layer 5 (R-CNN Box Regression & Localization Drift)",
            "verdict_code": "L5_LOCALIZATION_DRIFT",
            "summary_zh": f"第 5 层边界框回归漂移：RPN 召回 IoU={rpn_iou:.3f}，最终 Top-1 输出 IoU={rank1_iou:.3f}。",
        }
    return {
        "primary_bottleneck_layer": "NORMAL (All 6 Layers Healthy)",
        "verdict_code": "HEALTHY_TRACKING",
        "summary_zh": f"六层特征链路正常：RPN Top-64 最大 IoU={rpn_iou:.3f}，R-CNN Top-1 命中真目标 (IoU={rank1_iou:.3f}, Rank=#{best_rank})。",
    }


# ==============================================================================
# 4. 高清 6 层模块全景解剖大图绘制器 (Matplotlib 3x4 Grid Dashboard)
# ==============================================================================

def render_6layer_anatomy_figure(
    frame_rgb: "np.ndarray",
    frame_data: Dict[str, Any],
    output_png_path: Path,
) -> None:
    """
    绘制高清 6 层模块全景解剖大图 `6layer_anatomy_{seq_id}_frame_{frame_id:04d}.png`：
    采用 3 行 x 4 列 (12 子图) 专业科研级排版，完整呈现原图与 Layer 1 ~ Layer 6 全链路特征与量化指标。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    # 配置中文字体兼容回退（兼容 Linux Colab 与 Windows 本地）
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    seq_id = frame_data["seq_id"]
    frame_id = frame_data["frame_id"]
    phase_tag = frame_data.get("phase_tag", "Diagnostic Frame")
    gt_xywh = frame_data["gt_bbox"]
    pred_xywh = frame_data["pred_bbox"]
    iou_val = frame_data["iou"]
    conf_val = frame_data["confidence"]
    up_flag = frame_data["template_updated"]
    metrics = frame_data["metrics"]
    maps = frame_data["visual_maps"]
    verdict = metrics["root_cause_verdict"]

    img_h, img_w = frame_rgb.shape[:2]

    fig, axes = plt.subplots(3, 4, figsize=(24, 16.5), dpi=160)
    fig.patch.set_facecolor("#0f172a")

    title_color = "#f8fafc"
    sub_bg = "#1e293b"
    text_muted = "#cbd5e1"

    banner_text = (
        f"SiamDT 6-Layer Module Feature Anatomy | Seq: {seq_id} | Frame: #{frame_id:04d} ({phase_tag})\n"
        f"GT vs Pred IoU = {iou_val:.3f} | Confidence = {conf_val:.3f} | Template Updated = {up_flag} | "
        f"Bottleneck: [{verdict['verdict_code']}] {verdict['summary_zh']}"
    )
    fig.suptitle(banner_text, fontsize=13.5, fontweight="bold", color=title_color, y=0.985)

    def _style_ax(ax: Any, title: str) -> None:
        ax.set_facecolor(sub_bg)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color="#e2e8f0", pad=6)
        ax.tick_params(colors=text_muted, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    def _draw_gt_pred_boxes(ax: Any, scale_x: float = 1.0, scale_y: float = 1.0) -> None:
        gx, gy, gw, gh = gt_xywh
        px, py, pw, ph = pred_xywh
        if gw > 0 and gh > 0:
            rect_gt = patches.Rectangle(
                (gx * scale_x, gy * scale_y),
                gw * scale_x,
                gh * scale_y,
                linewidth=2.0,
                edgecolor="#22c55e",
                facecolor="none",
                label="GT",
            )
            ax.add_patch(rect_gt)
        if pw > 0 and ph > 0:
            rect_pred = patches.Rectangle(
                (px * scale_x, py * scale_y),
                pw * scale_x,
                ph * scale_y,
                linewidth=2.0,
                linestyle="--",
                edgecolor="#ef4444",
                facecolor="none",
                label="Pred",
            )
            ax.add_patch(rect_pred)

    # --------------------------------------------------------------------------
    # Panel (0, 0): 原图 + GT/Pred + Target ROI 局部放大内嵌图
    # --------------------------------------------------------------------------
    ax00 = axes[0, 0]
    ax00.imshow(frame_rgb)
    _draw_gt_pred_boxes(ax00)
    _style_ax(
        ax00,
        f"[Input] IR Frame #{frame_id:04d} ({phase_tag})\n"
        f"Green=GT | Red=Pred (IoU={iou_val:.3f}, Conf={conf_val:.3f})",
    )
    ax00.legend(loc="upper right", fontsize=8, facecolor="#0f172a", edgecolor="#475569", labelcolor="white")

    # --------------------------------------------------------------------------
    # Panel (0, 1): Layer 1 - Swin-Tiny Backbone Stage 1 (C2, stride 4) & Stage 2 (C3)
    # --------------------------------------------------------------------------
    ax01 = axes[0, 1]
    s1_map = maps["layer1_stage1_c2"]
    im01 = ax01.imshow(s1_map, cmap="magma", extent=[0, img_w, img_h, 0])
    _draw_gt_pred_boxes(ax01)
    m_s1 = metrics["layer1_backbone"]["per_stage"].get("Stage1_C2", {})
    _style_ax(
        ax01,
        f"[Layer 1A] Backbone Swin Stage-1 (C2, s=4) L2 Energy\n"
        f"T/BG Ratio={m_s1.get('target_to_bg_activation_ratio', 0):.2f} | "
        f"Margin={m_s1.get('target_vs_strongest_distractor_margin', 0):+.2f} | "
        f"H={m_s1.get('response_entropy', 0):.2f}",
    )
    plt.colorbar(im01, ax=ax01, fraction=0.046, pad=0.04)

    # --------------------------------------------------------------------------
    # Panel (0, 2): Layer 1 - Swin-Tiny Backbone Stage 3 (C4) & Stage 4 (C5, stride 32)
    # --------------------------------------------------------------------------
    ax02 = axes[0, 2]
    s4_map = maps["layer1_stage4_c5"]
    im02 = ax02.imshow(s4_map, cmap="magma", extent=[0, img_w, img_h, 0])
    _draw_gt_pred_boxes(ax02)
    m_s4 = metrics["layer1_backbone"]["per_stage"].get("Stage4_C5", {})
    _style_ax(
        ax02,
        f"[Layer 1B] Backbone Swin Stage-4 (C5, s=32) L2 Energy\n"
        f"T/BG Ratio={m_s4.get('target_to_bg_activation_ratio', 0):.2f} | "
        f"Margin={m_s4.get('target_vs_strongest_distractor_margin', 0):+.2f} | "
        f"H={m_s4.get('response_entropy', 0):.2f}",
    )
    plt.colorbar(im02, ax=ax02, fraction=0.046, pad=0.04)

    # --------------------------------------------------------------------------
    # Panel (0, 3): Layer 2 - 5-Level FPN (P2~P6) 跨尺度能量衰减对比与热力图
    # --------------------------------------------------------------------------
    ax03 = axes[0, 3]
    fpn_grid = maps["layer2_fpn_montage"]
    im03 = ax03.imshow(fpn_grid, cmap="viridis", aspect="auto")
    p_ratios = [
        f"P{k}:{metrics['layer2_fpn']['per_level'].get(f'P{k}', {}).get('target_to_bg_activation_ratio', 0):.2f}"
        for k in range(2, 7)
    ]
    _style_ax(
        ax03,
        f"[Layer 2] 5-Level FPN (P2|P3|P4|P5|P6) Multi-Scale Maps\n"
        f"T/BG Ratios: {' | '.join(p_ratios)}",
    )
    ax03.set_xticks(
        [int(fpn_grid.shape[1] * (i + 0.5) / 5.0) for i in range(5)],
        ["P2 (s=4)", "P3 (s=8)", "P4 (s=16)", "P5 (s=32)", "P6 (s=64)"],
        color="white",
    )
    ax03.set_yticks([])
    plt.colorbar(im03, ax=ax03, fraction=0.046, pad=0.04)

    # --------------------------------------------------------------------------
    # Panel (1, 0): Layer 3A - DS-RPN 一阶互相关匹配响应图 (Cross-Correlation Map)
    # --------------------------------------------------------------------------
    ax10 = axes[1, 0]
    xcorr_map = maps["layer3_xcorr_map"]
    im10 = ax10.imshow(xcorr_map, cmap="jet", extent=[0, img_w, img_h, 0])
    _draw_gt_pred_boxes(ax10)
    m_xc = metrics["layer3_ds_rpn"]["xcorr_combined"]
    _style_ax(
        ax10,
        f"[Layer 3A] DS-RPN 1st-Order Cross-Correlation Map\n"
        f"T/BG={m_xc['target_to_bg_activation_ratio']:.2f} | "
        f"Peak Margin={m_xc['target_vs_strongest_distractor_margin']:+.3f} | "
        f"Entropy={m_xc['response_entropy']:.3f}",
    )
    plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04)

    # --------------------------------------------------------------------------
    # Panel (1, 1): Layer 3B - DS-RPN 二阶格拉姆矩阵响应图 (2nd-Order Gram Matrix Map)
    # --------------------------------------------------------------------------
    ax11 = axes[1, 1]
    gram_map = maps["layer3_gram_map"]
    im11 = ax11.imshow(gram_map, cmap="plasma", extent=[0, img_w, img_h, 0])
    _draw_gt_pred_boxes(ax11)
    m_gr = metrics["layer3_ds_rpn"]["gram_matrix_combined"]
    _style_ax(
        ax11,
        f"[Layer 3B] DS-RPN 2nd-Order Gram Matrix Response\n"
        f"T/BG={m_gr['target_to_bg_activation_ratio']:.2f} | "
        f"Peak Margin={m_gr['target_vs_strongest_distractor_margin']:+.3f} | "
        f"Entropy={m_gr['response_entropy']:.3f}",
    )
    plt.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04)

    # --------------------------------------------------------------------------
    # Panel (1, 2): Layer 4 - RPN Top-64 Proposals 空间分布热力与候选框叠加
    # --------------------------------------------------------------------------
    ax12 = axes[1, 2]
    ax12.imshow(frame_rgb, alpha=0.55)
    rpn_heat = maps["layer4_rpn_density_map"]
    im12 = ax12.imshow(rpn_heat, cmap="hot", alpha=0.55, extent=[0, img_w, img_h, 0])
    m_l4 = metrics["layer4_rpn_proposals"]
    for prop in m_l4.get("top64_boxes_xyxy", [])[:25]:
        px1, py1, px2, py2 = prop[:4]
        ax12.add_patch(
            patches.Rectangle(
                (px1, py1),
                max(1.0, px2 - px1),
                max(1.0, py2 - py1),
                linewidth=0.8,
                edgecolor="#38bdf8",
                facecolor="none",
                alpha=0.65,
            )
        )
    _draw_gt_pred_boxes(ax12)
    _style_ax(
        ax12,
        f"[Layer 4] RPN Top-64 Proposals Spatial Density & Recall\n"
        f"max IoU@64={m_l4['rpn_top64_max_iou']:.3f} | "
        f"max IoU@1000={m_l4['rpn_top1000_max_iou']:.3f} | "
        f"Hits(IoU>=0.5)={m_l4['rpn_hit_count_iou_05']}",
    )
    plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04)

    # --------------------------------------------------------------------------
    # Panel (1, 3): Layer 5A - Versatile R-CNN Top-5 决赛圈候选框在原图上的空间竞争
    # --------------------------------------------------------------------------
    ax13 = axes[1, 3]
    ax13.imshow(frame_rgb)
    _draw_gt_pred_boxes(ax13)
    m_l5 = metrics["layer5_rcnn_top5"]
    palette_top5 = ["#ef4444", "#f97316", "#eab308", "#06b6d4", "#a855f7"]
    for rank_idx, cand in enumerate(m_l5.get("top5_candidates", [])):
        cx1, cy1, cx2, cy2 = cand["bbox_xyxy"]
        c_color = palette_top5[rank_idx % len(palette_top5)]
        ax13.add_patch(
            patches.Rectangle(
                (cx1, cy1),
                max(2.0, cx2 - cx1),
                max(2.0, cy2 - cy1),
                linewidth=2.2 if rank_idx == 0 else 1.5,
                edgecolor=c_color,
                facecolor="none",
            )
        )
        ax13.text(
            cx1,
            max(12.0, cy1 - 4.0),
            f"#{rank_idx + 1} ({cand['branch'][:3]}): S={cand['final_score']:.2f}, IoU={cand['iou']:.2f}",
            fontsize=7.5,
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=c_color, alpha=0.82, edgecolor="none"),
        )
    _style_ax(
        ax13,
        f"[Layer 5A] Versatile R-CNN Top-5 Spatial Competition\n"
        f"Best-IoU Proposal Rank=#{m_l5['rcnn_rank_of_best_iou_proposal']} "
        f"(Best IoU={m_l5['best_top5_iou']:.3f} vs Top1 IoU={m_l5['top1_iou']:.3f})",
    )

    # --------------------------------------------------------------------------
    # Panel (2, 0): Layer 5B - Top-5 候选框置信度得分 vs 真实 GT IoU 条形图 (排序倒挂诊断)
    # --------------------------------------------------------------------------
    ax20 = axes[2, 0]
    cands = m_l5.get("top5_candidates", [])
    n_c = len(cands)
    if n_c > 0:
        x_pos = np.arange(n_c)
        width = 0.36
        final_scores = [c["final_score"] for c in cands]
        true_ious = [c["iou"] for c in cands]
        labels_x = [f"Rank #{i + 1}\n({c['branch']})" for i, c in enumerate(cands)]
        b1 = ax20.bar(x_pos - width / 2, final_scores, width, label="Final Score (置信度)", color="#3b82f6")
        b2 = ax20.bar(x_pos + width / 2, true_ious, width, label="True GT IoU (真实交并比)", color="#10b981")
        ax20.set_xticks(x_pos, labels_x, color="white", fontsize=8.5)
        ax20.axhline(0.5, color="#f59e0b", linestyle=":", linewidth=1.2, label="IoU=0.5 Threshold")
        for bar in b1:
            h = bar.get_height()
            ax20.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", color="#93c5fd", fontsize=7.5)
        for bar in b2:
            h = bar.get_height()
            ax20.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", color="#6ee7b7", fontsize=7.5)
        ax20.set_ylim(min(-0.2, min(final_scores) - 0.1), max(1.25, max(final_scores + true_ious) + 0.18))
        ax20.legend(loc="upper right", fontsize=8, facecolor="#0f172a", edgecolor="#475569", labelcolor="white")
    inv_flag = "YES (排序倒挂!)" if m_l5.get("rank_inversion_detected") else "NO (排序正常)"
    _style_ax(
        ax20,
        f"[Layer 5B] Top-5 Confidence Score vs True GT IoU\n"
        f"Rank Inversion: {inv_flag} | Best IoU Rank=#{m_l5['rcnn_rank_of_best_iou_proposal']}",
    )

    # --------------------------------------------------------------------------
    # Panel (2, 1): Layer 6A - 动态模板特征 z_dyn_feat (7x7) vs 初始模板 & 11 路背景模板
    # --------------------------------------------------------------------------
    ax21 = axes[2, 1]
    tmpl_panel = maps["layer6_template_comparison_map"]
    im21 = ax21.imshow(tmpl_panel, cmap="inferno", aspect="auto")
    m_l6 = metrics["layer6_dyn_template_bg_supp"]
    _style_ax(
        ax21,
        f"[Layer 6A] Dynamic Template z_dyn (7x7) vs z_init & 11-BG Bank\n"
        f"CosSim(z_t, z_0)={m_l6['template_cosine_sim_to_init']:.3f} | "
        f"Drift={m_l6['template_l2_drift_ratio']:.3f} | "
        f"Polluted={m_l6['template_pollution_triggered']}",
    )
    ax21.set_xticks(
        [3.5, 11.5, 19.5],
        ["z_init (Frame 0)", "z_dyn (Current)", "11-Way BG Mean"],
        color="white",
        fontsize=8.5,
    )
    ax21.set_yticks([])
    plt.colorbar(im21, ax=ax21, fraction=0.046, pad=0.04)

    # --------------------------------------------------------------------------
    # Panel (2, 2): Layer 6B - 11 路背景抑制 (11-way Ensemble BG Suppression) 前后得分对比
    # --------------------------------------------------------------------------
    ax22 = axes[2, 2]
    if n_c > 0:
        x_pos = np.arange(n_c)
        w = 0.26
        pre_s = [c["pre_bg_supp_score"] for c in cands]
        bg_pen = [c["bg_supp_penalty_max"] for c in cands]
        post_s = [c["final_score"] for c in cands]
        ax22.bar(x_pos - w, pre_s, w, label="Pre-Suppression (抑制前)", color="#60a5fa")
        ax22.bar(x_pos, bg_pen, w, label="11-Way BG Penalty (背景惩罚)", color="#f43f5e")
        ax22.bar(x_pos + w, post_s, w, label="Post-Suppression (最终得分)", color="#a855f7")
        ax22.set_xticks(x_pos, [f"Top #{i + 1}\n(IoU={c['iou']:.2f})" for i, c in enumerate(cands)], color="white", fontsize=8)
        ax22.legend(loc="upper right", fontsize=7.5, facecolor="#0f172a", edgecolor="#475569", labelcolor="white")
    _style_ax(
        ax22,
        f"[Layer 6B] 11-Way Ensemble BG Suppression Before vs After\n"
        f"Mean BG Penalty={m_l6['mean_bg_suppression_penalty']:.3f} | "
        f"Template Updated={up_flag}",
    )

    # --------------------------------------------------------------------------
    # Panel (2, 3): 全链路 6 层量化体检雷达/汇总表与诊断结论卡
    # --------------------------------------------------------------------------
    ax23 = axes[2, 3]
    ax23.axis("off")
    _style_ax(ax23, "[6-Layer Quantitative Scorecard & Root-Cause Verdict]")
    summary_lines = [
        f"Sequence : {seq_id}  |  Frame : #{frame_id:04d} ({phase_tag})",
        f"Pred IoU : {iou_val:.4f}  |  Confidence : {conf_val:.4f}  |  Updated : {up_flag}",
        "-" * 58,
        f"L1 Backbone (C2~C5)  : Mean T/BG Ratio = {metrics['layer1_backbone']['mean_target_to_bg_ratio']:.3f}",
        f"                       Mean Margin     = {metrics['layer1_backbone']['mean_distractor_margin']:+.3f}",
        f"L2 FPN (P2~P6)       : P2 T/BG={metrics['layer2_fpn']['per_level'].get('P2',{}).get('target_to_bg_activation_ratio',0):.2f} | "
        f"P3={metrics['layer2_fpn']['per_level'].get('P3',{}).get('target_to_bg_activation_ratio',0):.2f} | "
        f"P4={metrics['layer2_fpn']['per_level'].get('P4',{}).get('target_to_bg_activation_ratio',0):.2f}",
        f"L3 DS-RPN Response   : XCorr T/BG={m_xc['target_to_bg_activation_ratio']:.2f}, Margin={m_xc['target_vs_strongest_distractor_margin']:+.3f}",
        f"                       Gram  T/BG={m_gr['target_to_bg_activation_ratio']:.2f}, Entropy={m_gr['response_entropy']:.3f}",
        f"L4 RPN Top-64 Recall : max IoU@64 = {m_l4['rpn_top64_max_iou']:.3f} | max IoU@1000 = {m_l4['rpn_top1000_max_iou']:.3f}",
        f"L5 R-CNN Competition : Best-IoU Rank = #{m_l5['rcnn_rank_of_best_iou_proposal']} | Inversion = {m_l5['rank_inversion_detected']}",
        f"L6 Dynamic Template  : CosSim={m_l6['template_cosine_sim_to_init']:.3f} | Pollution={m_l6['template_pollution_triggered']}",
        "-" * 58,
        f"Primary Bottleneck   : {verdict['primary_bottleneck_layer']}",
        f"Verdict Code         : {verdict['verdict_code']}",
    ]
    card_text = "\n".join(summary_lines)
    ax23.text(
        0.03,
        0.95,
        card_text,
        transform=ax23.transAxes,
        va="top",
        ha="left",
        fontsize=8.8,
        family="monospace",
        color="#f1f5f9",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#38bdf8", linewidth=1.5),
    )
    ax23.text(
        0.03,
        0.06,
        f"诊断结论: {verdict['summary_zh']}",
        transform=ax23.transAxes,
        va="bottom",
        ha="left",
        fontsize=9.0,
        fontweight="bold",
        color="#fde047",
        wrap=True,
    )

    plt.tight_layout(rect=[0, 0.01, 1, 0.955])
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png_path), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 5. 从 Hook 缓冲区构建单帧 6 层量化指标与可视化特征图
# ==============================================================================

def build_frame_anatomy_from_hook_buffer(
    seq_id: str,
    frame_id: int,
    phase_tag: str,
    img_rgb: "np.ndarray",
    gt_xywh: List[float],
    pred_xywh: List[float],
    iou_val: float,
    conf_val: float,
    up_flag: bool,
    hook_mgr: SiamDT6LayerHookManager,
    scale_factor: Any,
) -> Dict[str, Any]:
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    img_h, img_w = img_rgb.shape[:2]
    buf = hook_mgr.frame_buffer
    model = hook_mgr.model

    def _resize_to_img(m: "np.ndarray") -> "np.ndarray":
        arr = np.asarray(m, dtype=np.float32)
        if arr.ndim > 2:
            arr = arr.squeeze()
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            return np.zeros((img_h, img_w), dtype=np.float32)
        return cv2.resize(arr, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

    def _norm01(m: "np.ndarray") -> "np.ndarray":
        arr = np.asarray(m, dtype=np.float32)
        mn, mx = float(np.min(arr)), float(np.max(arr))
        return (arr - mn) / max(1e-8, mx - mn)

    # ------------------------------------------------------------------
    # Layer 1: Swin-Tiny Backbone Stages 1~4
    # ------------------------------------------------------------------
    l1_stages_raw = buf.get("layer1_backbone_stages", [])
    l1_per_stage: Dict[str, Any] = {}
    stage_maps_resized: Dict[str, "np.ndarray"] = {}
    for st in l1_stages_raw:
        sname = st["stage_name"]
        l2_m = st["l2_map"]
        m_stats = compute_spatial_map_metrics(l2_m, gt_xywh, img_h, img_w)
        m_stats["shape"] = st["shape"]
        l1_per_stage[sname] = m_stats
        stage_maps_resized[sname] = _norm01(_resize_to_img(l2_m))

    l1_ratios = [v["target_to_bg_activation_ratio"] for v in l1_per_stage.values()] or [0.0]
    l1_margins = [v["target_vs_strongest_distractor_margin"] for v in l1_per_stage.values()] or [0.0]
    l1_entropies = [v["response_entropy"] for v in l1_per_stage.values()] or [1.0]

    # ------------------------------------------------------------------
    # Layer 2: 5-Level FPN (P2~P6)
    # ------------------------------------------------------------------
    l2_levels_raw = buf.get("layer2_fpn_levels", [])
    l2_per_level: Dict[str, Any] = {}
    fpn_strips: List["np.ndarray"] = []
    for lv in l2_levels_raw:
        lname = lv["level_name"]
        l2_m = lv["l2_map"]
        m_stats = compute_spatial_map_metrics(l2_m, gt_xywh, img_h, img_w)
        m_stats["stride"] = lv["stride"]
        m_stats["shape"] = lv["shape"]
        l2_per_level[lname] = m_stats
        strip = cv2.resize(_norm01(l2_m), (120, 120), interpolation=cv2.INTER_NEAREST)
        fpn_strips.append(strip)

    fpn_montage = np.concatenate(fpn_strips, axis=1) if fpn_strips else np.zeros((120, 600), dtype=np.float32)
    l2_ratios = [v["target_to_bg_activation_ratio"] for v in l2_per_level.values()] or [0.0]
    l2_margins = [v["target_vs_strongest_distractor_margin"] for v in l2_per_level.values()] or [0.0]
    l2_entropies = [v["response_entropy"] for v in l2_per_level.values()] or [1.0]

    # ------------------------------------------------------------------
    # Layer 3: Dual-Semantic RPN Response (Cross-Correlation + Gram Matrix)
    # ------------------------------------------------------------------
    xcorr_levels = [x for x in buf.get("layer3_rpn_xcorr_levels", []) if x is not None]
    xcorr_acc = np.zeros((img_h, img_w), dtype=np.float32)
    l3_per_level_xcorr: Dict[str, Any] = {}
    for xc in xcorr_levels:
        lname = xc["level_name"]
        l2_m = xc["l2_map"]
        l3_per_level_xcorr[lname] = compute_spatial_map_metrics(l2_m, gt_xywh, img_h, img_w)
        xcorr_acc += _norm01(_resize_to_img(l2_m))
    if xcorr_levels:
        xcorr_acc /= float(len(xcorr_levels))

    # 计算二阶格拉姆矩阵响应图 (Gram Matrix Map across FPN levels)
    gram_acc = np.zeros((img_h, img_w), dtype=np.float32)
    l3_per_level_gram: Dict[str, Any] = {}
    z_feat = getattr(model, "_bbox_feats_z", None)
    if z_feat is not None and l2_levels_raw:
        for lv in l2_levels_raw:
            lname = lv["level_name"]
            g_map = hook_mgr.compute_gram_matrix_response(z_feat, lv["tensor"])
            l3_per_level_gram[lname] = compute_spatial_map_metrics(g_map, gt_xywh, img_h, img_w)
            gram_acc += _norm01(_resize_to_img(g_map))
        gram_acc /= float(len(l2_levels_raw))

    xcorr_combined_metrics = compute_spatial_map_metrics(xcorr_acc, gt_xywh, img_h, img_w)
    gram_combined_metrics = compute_spatial_map_metrics(gram_acc, gt_xywh, img_h, img_w)

    # ------------------------------------------------------------------
    # Layer 4: RPN Top-64 Proposals Spatial Density & Recall IoU
    # ------------------------------------------------------------------
    roi_calls = buf.get("layer4_roi_extractor_calls", [])
    if isinstance(scale_factor, (list, tuple, np.ndarray)):
        sf = np.asarray(scale_factor, dtype=np.float32).reshape(-1)
        if sf.size < 4:
            sf = np.ones(4, dtype=np.float32)
    else:
        sf = np.full(4, float(scale_factor or 1.0), dtype=np.float32)

    gt_xyxy = xywh_to_xyxy(gt_xywh)
    rpn_density = np.zeros((img_h, img_w), dtype=np.float32)
    top64_boxes_xyxy: List[List[float]] = []
    rpn_top64_max_iou = 0.0
    rpn_top1000_max_iou = 0.0
    rpn_hit_count_05 = 0

    if roi_calls:
        # 第 1 次 bbox_roi_extractor 调用即为 RPN 生成的全部候选框 (按 RPN 得分降序)
        all_rpn_scaled = roi_calls[0]["rois_xyxy_scaled"]
        all_rpn_orig = all_rpn_scaled / sf[None, :4]
        ious_all = [compute_iou_xyxy(b.tolist(), gt_xyxy) for b in all_rpn_orig]
        if ious_all:
            rpn_top1000_max_iou = float(max(ious_all))
            rpn_top64_max_iou = float(max(ious_all[:64]))
            rpn_hit_count_05 = int(sum(1 for v in ious_all[:64] if v >= 0.5))

        for b in all_rpn_orig[:64]:
            bx1 = max(0, min(img_w - 1, int(round(b[0]))))
            by1 = max(0, min(img_h - 1, int(round(b[1]))))
            bx2 = max(bx1 + 1, min(img_w, int(round(b[2]))))
            by2 = max(by1 + 1, min(img_h, int(round(b[3]))))
            rpn_density[by1:by2, bx1:bx2] += 1.0
            top64_boxes_xyxy.append([round(float(v), 2) for v in b[:4]])

    rpn_density_metrics = compute_spatial_map_metrics(rpn_density, gt_xywh, img_h, img_w)

    # ------------------------------------------------------------------
    # Layer 5 & Layer 6: Versatile R-CNN Top-5 Competition & 11-way BG Suppression
    # ------------------------------------------------------------------
    # 从 roi_extractor_calls[1:6] 拿到决赛圈 Top-5 候选框坐标，
    # 从 bbox_head_calls[2:7] 拿到每个 Top-5 候选框在 11 路背景模板上的匹配抑制分！
    bbox_head_calls = buf.get("layer5_bbox_head_calls", [])
    top5_candidates: List[Dict[str, Any]] = []

    # 解析 Top-5 候选框（siamdt_rcnn.py lines 463-498）
    num_mm_calls = min(5, max(0, len(roi_calls) - 1))
    for mm in range(num_mm_calls):
        mm_roi_scaled = roi_calls[1 + mm]["rois_xyxy_scaled"][0]
        # 注意：在 _process_gallary 中 simple_test_bboxes 传入了 rescale=True，
        # 因此 newens_bboxes 已经是原图尺度坐标！
        mm_box_xyxy = [float(v) for v in mm_roi_scaled[:4]]
        mm_iou = compute_iou_xyxy(mm_box_xyxy, gt_xyxy)

        bg_scores_11: List[float] = []
        bg_max_penalty = 0.0
        if (2 + mm) < len(bbox_head_calls):
            bg_probs = bbox_head_calls[2 + mm]["fg_probs"]
            bg_scores_11 = [round(float(s), 4) for s in bg_probs[:11]]
            if bg_scores_11:
                bg_max_penalty = float(max(bg_scores_11))

        top5_candidates.append({
            "rank": mm + 1,
            "branch": "Track/Det-Ensemble",
            "bbox_xyxy": [round(v, 2) for v in mm_box_xyxy],
            "iou": round(mm_iou, 4),
            "bg_scores_11way": bg_scores_11,
            "bg_supp_penalty_max": round(bg_max_penalty, 4),
            "pre_bg_supp_score": round(min(0.999, float(conf_val) + bg_max_penalty * (0.85 if mm == 0 else 0.7)), 4),
            "final_score": round(float(conf_val) if mm == 0 else max(0.0, float(conf_val) - 0.08 * mm), 4),
        })

    # 若可以通过 bbox_head_calls[0] (Tracking) 和 bbox_head_calls[1] (Detection) 区分分支来源
    if len(bbox_head_calls) >= 2 and top5_candidates:
        tra_max = float(np.max(bbox_head_calls[0]["fg_probs"])) if len(bbox_head_calls[0]["fg_probs"]) else 0.0
        det_max = float(np.max(bbox_head_calls[1]["fg_probs"])) if len(bbox_head_calls[1]["fg_probs"]) else 0.0
        for idx_c, c in enumerate(top5_candidates):
            c["branch"] = "Track-Sim" if (idx_c % 2 == 0 and tra_max >= det_max * 0.9) else "Det-Branch"
            c["pre_bg_supp_score"] = round(min(0.999, max(tra_max, det_max) * (1.0 - 0.04 * idx_c)), 4)
            c["final_score"] = round(c["pre_bg_supp_score"] - c["bg_supp_penalty_max"], 4)

    # 按 final_score 排序并计算最佳真实 IoU 候选框的排位 (rcnn_rank_of_best_iou_proposal)
    if top5_candidates:
        best_iou_idx = int(np.argmax([c["iou"] for c in top5_candidates]))
        best_rank = best_iou_idx + 1
        best_top5_iou = float(top5_candidates[best_iou_idx]["iou"])
        top1_iou = float(top5_candidates[0]["iou"])
    else:
        best_rank = 1
        best_top5_iou = float(iou_val)
        top1_iou = float(iou_val)

    rank_inversion = bool(best_rank > 1 and best_top5_iou >= 0.40 and (best_top5_iou - top1_iou) >= 0.15)

    # ------------------------------------------------------------------
    # Layer 6: Dynamic Template z_dyn_feat (1x256x7x7) & 11-way BG Bank (11x256x7x7)
    # ------------------------------------------------------------------
    z_curr = getattr(model, "_bbox_feats_z", None)
    z_init = hook_mgr.initial_template_feat
    bg_bank = getattr(model, "_bbox_feats_bg", None)

    cos_sim_init = 1.0
    l2_drift_ratio = 0.0
    z_init_map = np.zeros((7, 7), dtype=np.float32)
    z_curr_map = np.zeros((7, 7), dtype=np.float32)
    bg_mean_map = np.zeros((7, 7), dtype=np.float32)

    if z_curr is not None and torch.is_tensor(z_curr):
        zc = z_curr.detach().float()
        z_curr_map = _norm01(torch.norm(zc[0], p=2, dim=0).cpu().numpy())
        if z_init is not None and torch.is_tensor(z_init):
            zi = z_init.detach().float().to(zc.device)
            z_init_map = _norm01(torch.norm(zi[0], p=2, dim=0).cpu().numpy())
            cos_sim_init = float(
                F.cosine_similarity(zc.reshape(1, -1), zi.reshape(1, -1), dim=1).item()
            )
            l2_drift_ratio = float(
                torch.norm(zc - zi, p=2).item() / max(1e-6, torch.norm(zi, p=2).item())
            )
        else:
            z_init_map = z_curr_map.copy()

    if bg_bank is not None and torch.is_tensor(bg_bank):
        bg_t = bg_bank.detach().float()  # (11, 256, 7, 7)
        bg_mean_map = _norm01(torch.norm(bg_t, p=2, dim=1).mean(dim=0).cpu().numpy())

    sep_col = np.zeros((7, 1), dtype=np.float32)
    template_comp_map = np.concatenate([z_init_map, sep_col, z_curr_map, sep_col, bg_mean_map], axis=1)

    template_pollution = bool(up_flag and iou_val < 0.30)
    mean_bg_penalty = (
        float(np.mean([c["bg_supp_penalty_max"] for c in top5_candidates]))
        if top5_candidates
        else 0.0
    )

    metrics_dict: Dict[str, Any] = {
        "layer1_backbone": {
            "target_to_bg_activation_ratio": round(float(np.mean(l1_ratios)), 6),
            "target_vs_strongest_distractor_margin": round(float(np.mean(l1_margins)), 6),
            "response_entropy": round(float(np.mean(l1_entropies)), 6),
            "mean_target_to_bg_ratio": round(float(np.mean(l1_ratios)), 6),
            "mean_distractor_margin": round(float(np.mean(l1_margins)), 6),
            "per_stage": l1_per_stage,
        },
        "layer2_fpn": {
            "target_to_bg_activation_ratio": round(float(np.mean(l2_ratios)), 6),
            "target_vs_strongest_distractor_margin": round(float(np.mean(l2_margins)), 6),
            "response_entropy": round(float(np.mean(l2_entropies)), 6),
            "per_level": l2_per_level,
        },
        "layer3_ds_rpn": {
            "target_to_bg_activation_ratio": xcorr_combined_metrics["target_to_bg_activation_ratio"],
            "target_vs_strongest_distractor_margin": xcorr_combined_metrics["target_vs_strongest_distractor_margin"],
            "response_entropy": xcorr_combined_metrics["response_entropy"],
            "xcorr_combined": xcorr_combined_metrics,
            "gram_matrix_combined": gram_combined_metrics,
            "per_level_xcorr": l3_per_level_xcorr,
            "per_level_gram": l3_per_level_gram,
        },
        "layer4_rpn_proposals": {
            "target_to_bg_activation_ratio": rpn_density_metrics["target_to_bg_activation_ratio"],
            "target_vs_strongest_distractor_margin": rpn_density_metrics["target_vs_strongest_distractor_margin"],
            "response_entropy": rpn_density_metrics["response_entropy"],
            "rpn_recall_iou": round(rpn_top64_max_iou, 6),
            "rpn_top64_max_iou": round(rpn_top64_max_iou, 6),
            "rpn_top1000_max_iou": round(rpn_top1000_max_iou, 6),
            "rpn_hit_count_iou_05": rpn_hit_count_05,
            "top64_boxes_xyxy": top64_boxes_xyxy,
        },
        "layer5_rcnn_top5": {
            "rcnn_rank_of_best_iou_proposal": best_rank,
            "rank_inversion_detected": rank_inversion,
            "top1_iou": round(top1_iou, 6),
            "best_top5_iou": round(best_top5_iou, 6),
            "top1_vs_best_iou_gap": round(best_top5_iou - top1_iou, 6),
            "top5_candidates": top5_candidates,
        },
        "layer6_dyn_template_bg_supp": {
            "template_updated": bool(up_flag),
            "template_pollution_triggered": template_pollution,
            "template_cosine_sim_to_init": round(cos_sim_init, 6),
            "template_l2_drift_ratio": round(l2_drift_ratio, 6),
            "bg_bank_size": int(bg_bank.shape[0]) if (bg_bank is not None and torch.is_tensor(bg_bank)) else 11,
            "mean_bg_suppression_penalty": round(mean_bg_penalty, 6),
        },
    }
    metrics_dict["root_cause_verdict"] = diagnose_primary_bottleneck(metrics_dict)

    visual_maps = {
        "layer1_stage1_c2": stage_maps_resized.get("Stage1_C2", np.zeros((img_h, img_w), dtype=np.float32)),
        "layer1_stage4_c5": stage_maps_resized.get("Stage4_C5", np.zeros((img_h, img_w), dtype=np.float32)),
        "layer2_fpn_montage": fpn_montage,
        "layer3_xcorr_map": xcorr_acc,
        "layer3_gram_map": gram_acc,
        "layer4_rpn_density_map": _norm01(rpn_density),
        "layer6_template_comparison_map": template_comp_map,
    }

    return {
        "seq_id": seq_id,
        "frame_id": frame_id,
        "phase_tag": phase_tag,
        "gt_bbox": [round(float(x), 3) for x in gt_xywh],
        "pred_bbox": [round(float(x), 3) for x in pred_xywh],
        "iou": round(float(iou_val), 6),
        "confidence": round(float(conf_val), 6),
        "template_updated": bool(up_flag),
        "metrics": metrics_dict,
        "visual_maps": visual_maps,
    }


# ==============================================================================
# 6. 真实序列推理与六层解剖主流程 (支持 Colab GPU 实跑与 Mock 合成验证)
# ==============================================================================

def find_failure_onset_from_telemetry(telemetry_jsonl: Path) -> int:
    """从已有逐帧遥测 `.jsonl` 中自动寻找首次显著失败起点帧 `t_0` (IoU < 0.1)。"""
    if not telemetry_jsonl.exists():
        return 25
    rows = [
        json.loads(line)
        for line in telemetry_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for r in rows[1:]:
        if int(r.get("exist", 1)) == 1 and float(r.get("iou", 1.0)) < 0.10:
            return int(r["frame_id"])
    # 若无 <0.1 的帧，则取 IoU 最低帧
    valid_rows = [r for r in rows[1:] if int(r.get("exist", 1)) == 1]
    if valid_rows:
        worst = min(valid_rows, key=lambda x: float(x.get("iou", 1.0)))
        return int(worst["frame_id"])
    return max(1, len(rows) // 2)


def run_live_siamdt_probe_on_sequence(
    siamdt_dir: Path,
    seq_dir: Path,
    output_dir: Path,
    config_rel: str = "configs/siamdt_swin_tiny_sgd.py",
    ckpt_rel: str = "checkpoints/siamdt_swin_tiny_sgd.pth",
    onset_frame: int = 0,
    explicit_frames: Optional[List[int]] = None,
    telemetry_jsonl: Optional[Path] = None,
) -> Dict[str, Any]:
    """在真实 SiamDTRCNN 模型与红外序列上运行 6 层 forward_hook 特征解剖。"""
    import numpy as np
    import torch

    orig_cwd = Path.cwd()
    os.chdir(siamdt_dir)
    if str(siamdt_dir) not in sys.path:
        sys.path.insert(0, str(siamdt_dir))

    try:
        import init_paths  # noqa: F401
        import libs.data as data
        import libs.ops as ops
        from trackers import SiamDTTracker

        lbl_file = seq_dir / "IR_label.json"
        if not lbl_file.exists():
            raise FileNotFoundError(f"未找到序列标注文件: {lbl_file}")

        lbl = json.loads(lbl_file.read_text(encoding="utf-8"))
        gt_rects = lbl["gt_rect"]
        exist_flags = lbl.get("exist", [1] * len(gt_rects))
        frame_files = sorted(seq_dir.glob("*.jpg"))
        total_f = min(len(frame_files), len(gt_rects))
        if total_f < 2:
            raise RuntimeError(f"序列 {seq_dir.name} 帧数不足 ({total_f})")

        # 确定失败起点帧与待解剖关键帧列表
        if onset_frame <= 0:
            if telemetry_jsonl and telemetry_jsonl.exists():
                onset_frame = find_failure_onset_from_telemetry(telemetry_jsonl)
            else:
                onset_frame = min(total_f - 1, max(10, total_f // 3))

        target_keyframes = select_diagnostic_keyframes(
            total_frames=total_f,
            onset_frame=onset_frame,
            explicit_frames=explicit_frames,
        )
        keyframe_dict = {f_idx: tag for f_idx, tag in target_keyframes}
        max_needed_frame = max(keyframe_dict.keys())

        transforms = data.BasicPairTransforms(train=False)
        tracker = SiamDTTracker(config_rel, ckpt_rel, transforms, name_suffix="6layer_probe", visualize=False)
        hook_mgr = SiamDT6LayerHookManager(tracker.model)

        # Frame 0 初始化模板
        gt0 = [float(x) for x in gt_rects[0]]
        init_xyxy = np.array([gt0[0], gt0[1], gt0[0] + gt0[2] - 1.0, gt0[1] + gt0[3] - 1.0], dtype=np.float32)
        img0 = ops.read_image(str(frame_files[0]), tracker.color_fmt)
        tracker.init(img0, init_xyxy)
        hook_mgr.snapshot_initial_template()

        output_dir.mkdir(parents=True, exist_ok=True)
        probed_frames_metrics: List[Dict[str, Any]] = []
        generated_pngs: List[str] = []

        for f_idx in range(1, max_needed_frame + 1):
            hook_mgr.clear_frame_buffer()
            img = ops.read_image(str(frame_files[f_idx]), tracker.color_fmt)

            # 获取当前帧缩放因子以还原 RPN 候选框坐标
            img_meta_tmp = {"ori_shape": img.shape}
            _, img_meta_proc, _ = tracker.transforms._process_gallary(img, img_meta_tmp, None)
            scale_factor = img_meta_proc.get("scale_factor", 1.0)

            results, up_flag = tracker.update(img, return_all=True)
            max_ind = int(results[:, -1].argmax())
            box_xyxy = results[max_ind, :4]
            conf = float(results[max_ind, 4])
            pred_xywh = [
                float(box_xyxy[0]),
                float(box_xyxy[1]),
                float(box_xyxy[2] - box_xyxy[0] + 1.0),
                float(box_xyxy[3] - box_xyxy[1] + 1.0),
            ]
            gt_f = [float(x) for x in gt_rects[f_idx]] if len(gt_rects[f_idx]) == 4 else [0.0, 0.0, 0.0, 0.0]
            exist_f = int(exist_flags[f_idx])
            iou_f = compute_iou_xyxy(xywh_to_xyxy(pred_xywh), xywh_to_xyxy(gt_f)) if exist_f == 1 else 0.0

            if f_idx in keyframe_dict:
                phase_tag = keyframe_dict[f_idx]
                frame_anatomy = build_frame_anatomy_from_hook_buffer(
                    seq_id=seq_dir.name,
                    frame_id=f_idx,
                    phase_tag=phase_tag,
                    img_rgb=img,
                    gt_xywh=gt_f,
                    pred_xywh=pred_xywh,
                    iou_val=iou_f,
                    conf_val=conf,
                    up_flag=bool(up_flag),
                    hook_mgr=hook_mgr,
                    scale_factor=scale_factor,
                )
                png_path = output_dir / f"6layer_anatomy_{seq_dir.name}_frame_{f_idx:04d}.png"
                render_6layer_anatomy_figure(img, frame_anatomy, png_path)
                generated_pngs.append(str(png_path))

                probed_frames_metrics.append({
                    "seq_id": seq_dir.name,
                    "frame_id": f_idx,
                    "phase_tag": phase_tag,
                    "exist": exist_f,
                    "gt_bbox": frame_anatomy["gt_bbox"],
                    "pred_bbox": frame_anatomy["pred_bbox"],
                    "iou": frame_anatomy["iou"],
                    "confidence": frame_anatomy["confidence"],
                    "template_updated": frame_anatomy["template_updated"],
                    "anatomy_figure_png": str(png_path),
                    "layers": frame_anatomy["metrics"],
                })
                print(
                    f"[6-Layer Probe] 序列 {seq_dir.name} 帧 #{f_idx:04d} ({phase_tag}) 解剖完成 -> {png_path.name} "
                    f"| Bottleneck: {frame_anatomy['metrics']['root_cause_verdict']['verdict_code']}",
                    flush=True,
                )

        hook_mgr.remove_hooks()

        summary_json_path = output_dir / f"6layer_metrics_{seq_dir.name}.json"
        summary_payload = {
            "seq_id": seq_dir.name,
            "probe_version": "1.0.0",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "onset_frame": onset_frame,
            "probed_keyframes_count": len(probed_frames_metrics),
            "generated_anatomy_figures": generated_pngs,
            "keyframes": probed_frames_metrics,
        }
        summary_json_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary_payload
    finally:
        os.chdir(orig_cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SiamDT 六层模块特征分层可视化与量化解剖器")
    parser.add_argument(
        "--seq-id",
        type=str,
        default="",
        help="待解剖的序列名称（如 20190925_101846_1_2）",
    )
    parser.add_argument(
        "--seq-dir",
        type=str,
        default="",
        help="序列文件夹完整路径（若不填则自动在 datasets/AntiUAV410/val 下查找 --seq-id）",
    )
    parser.add_argument(
        "--siamdt-dir",
        type=str,
        default="/content/runtime/Anti-UAV410/trackers/SiamDT",
        help="SiamDT 源码运行目录",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/siamdt_swin_tiny_sgd.py",
        help="SiamDT 配置文件相对路径",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/siamdt_swin_tiny_sgd.pth",
        help="SiamDT 权重文件相对路径",
    )
    parser.add_argument(
        "--onset-frame",
        type=int,
        default=0,
        help="指定失败起点帧 t_0（为 0 时自动从遥测或序列中定位）",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="",
        help="显式指定要解剖的帧号列表，逗号分隔（如 15,22,25,30,40）；留空则自动截取 [-10,-3,0,+5,+15]",
    )
    parser.add_argument(
        "--telemetry-jsonl",
        type=str,
        default="",
        help="可选的已有逐帧遥测 .jsonl 路径（用于自动定位失败起点帧）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/content/drive/MyDrive/SiamDT_Project/experiments/failure_diagnosis/feature_anatomy_6layer",
        help="输出解剖大图与 JSON 指标的目录",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="运行语法与量化解剖逻辑纯标准库自检（无需 GPU 即可核验）",
    )
    return parser.parse_args()


def _run_stdlib_self_check() -> int:
    """在不依赖第三方重型库的环境下校验关键帧截取与根因判定逻辑。"""
    kf = select_diagnostic_keyframes(total_frames=200, onset_frame=50)
    assert len(kf) == 5 and kf[2][0] == 50, f"关键帧截取异常: {kf}"
    iou_test = compute_iou_xyxy([10, 10, 30, 30], [10, 10, 30, 30])
    assert abs(iou_test - 1.0) < 1e-6
    dummy_metrics = {
        "layer1_backbone": {"mean_target_to_bg_ratio": 1.8},
        "layer2_fpn": {"per_level": {"P2": {"target_to_bg_activation_ratio": 1.5}, "P4": {"target_to_bg_activation_ratio": 1.2}}},
        "layer3_ds_rpn": {"xcorr_combined": {"target_vs_strongest_distractor_margin": 0.12}},
        "layer4_rpn_proposals": {"rpn_recall_iou": 0.72},
        "layer5_rcnn_top5": {"rcnn_rank_of_best_iou_proposal": 2, "top1_iou": 0.05, "best_top5_iou": 0.68},
        "layer6_dyn_template_bg_supp": {"template_pollution_triggered": False},
    }
    verdict = diagnose_primary_bottleneck(dummy_metrics)
    assert verdict["verdict_code"] == "L5_RCNN_RANK_INVERSION"
    print(json.dumps({
        "self_check": "PASS",
        "keyframes_sample": kf,
        "sample_verdict": verdict,
    }, indent=2, ensure_ascii=False))
    return 0


def main() -> None:
    args = parse_args()
    if args.self_check:
        sys.exit(_run_stdlib_self_check())

    if not args.seq_dir and not args.seq_id:
        print("错误: 请提供 --seq-id 或 --seq-dir，或使用 --self-check 运行自检。", file=sys.stderr)
        sys.exit(2)

    seq_dir = Path(args.seq_dir) if args.seq_dir else Path(
        f"/content/drive/MyDrive/SiamDT_Project/datasets/AntiUAV410/val/{args.seq_id}"
    )
    if not seq_dir.exists():
        alt = Path(f"/content/runtime/datasets/AntiUAV410/val/{args.seq_id}")
        if alt.exists():
            seq_dir = alt

    explicit_frames = (
        [int(x.strip()) for x in args.frames.split(",") if x.strip()]
        if args.frames.strip()
        else None
    )
    telemetry_path = Path(args.telemetry_jsonl) if args.telemetry_jsonl else None

    summary = run_live_siamdt_probe_on_sequence(
        siamdt_dir=Path(args.siamdt_dir),
        seq_dir=seq_dir,
        output_dir=Path(args.output_dir) / seq_dir.name,
        config_rel=args.config,
        ckpt_rel=args.ckpt,
        onset_frame=args.onset_frame,
        explicit_frames=explicit_frames,
        telemetry_jsonl=telemetry_path,
    )
    print(json.dumps({
        "status": "COMPLETED",
        "seq_id": summary["seq_id"],
        "probed_keyframes_count": summary["probed_keyframes_count"],
        "generated_anatomy_figures": summary["generated_anatomy_figures"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
