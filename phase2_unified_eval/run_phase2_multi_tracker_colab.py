#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Colab 多算法统一评测与权重拉取调度器 (run_phase2_multi_tracker_colab.py)
================================================================================
供 Google Colab (主力 A100 sm_80 / 辅助 L4 sm_89) 及本地总控直接调用的统一调度脚本。
支持在冻结的 `discovery_set.json` (24 序列) 和 `holdout_verification.json` (16 序列) 上统一运行：
  - Group I (In-Domain 同域组):
      1. SiamDT_Baseline (原始 245 参数张量 SiamDTRCNN, Swin-Tiny + 5-level FPN)
      2. FocusTrack (自适应局部采样缩放 + 同仓重训 OSTrack/ROMTrack/ZoomTrack/DropTrack)
      3. MemLoTrack (DINOv2+LoRA + 卡尔曼运动门控 FIFO 记忆库, retrained & scratch)
  - Group II (Zero-Shot / 多模态组):
      4. DAM4SAM (SAM 2.1 + RAM/DRM 干扰物感知记忆解耦 + 不确定性更新冻结)
      5. SAMOSA (SAM 2.1 + 高阶马尔可夫非线性运动预测 mp.pth + EDRM 误差恢复)
      6. SAMURAI (SAM 2.1 + 线性卡尔曼运动感知记忆)
      7. UETrack (CVPR 2026 五模态统一 Token-Pooling MoE + TAD)
      8. SUTrack (AAAI 2025 统一单流多模态跟踪器)
      9. UTPTrack-S (CVPR 2026 三组件统一背景 Token 剪枝)
     10. ReFocus-TIR (TNNLS 热红外参数高效注意力重聚焦)

核心特性：
1. A100 算力吃满混合并发调度 (`--concurrency-mode hybrid_b_then_a`, `--gpu-slots`):
   - **方案 B 优先（多算法同卡并发）**：当待评测算法数 M >= GPU 算力槽位数 S 时，优先并发启动 S 个不同算法进程；
   - **方案 A 自动补位（单算法内多视频分片并发）**：
     a) 当 1 <= M < S（如只跑单个算法或最后剩余少数算法）时，自动将每个待跑算法的未完成序列拆分为 K = floor(S / M)
        （含余数分配使总分片数等于 S）个并行视频分片 (`pending_seqs[shard_idx::num_shards]`) 同时运行，吃满 100% 算力槽位；
     b) 在方案 B 多算法并发运行过程中，进程池实时扫描 `telemetry_eval_dir/*.jsonl`：一旦某个快算法先跑完腾出空闲槽位，
        而某个慢算法仍有 > 4 条未完成序列，立即在空闲槽位上为该慢算法动态派生补充的方案 A 分片 Worker（从末尾反向抢占未完成序列，
        配合 `.jsonl` 与本地盘 `.lock` 原子级双重检查防重跑），直至全部序列完成并统一汇总生成 `eval_progress_summary.json`！
2. 权重与源码持久化金库：源码位于 Google Drive `repos/core_pool/`，权重位于 `weights/<组>/<算法>/`（每个算法独立子目录，
   按“精确文件名 + 字节数”核验），评测数据优先解压到 Colab 本地高速盘 `/content/runtime/datasets/AntiUAV410/val`；
3. 逐序列断点续评 (Auto-Resume)：每跑完 1 条序列立即将 `.txt` 预测框、官方兼容 JSON 与统一 `.jsonl`
   逐帧遥测实时 fsync 刷入 Google Drive，断线重连自动跳过已完成序列；
4. 统一遥测转换与三层失败提取：内置原生推理适配器与通用预测框摄入器 (`--ingest-raw-txt-dir`)，
   评测完成后可一键调用 `failure_episode_extractor.py` 生成 `FAILURE_EPISODE_DATABASE.json`
   与 `CANONICAL_CHALLENGE_WINDOWS.json`。

2026-11-04 交叉审查 P0 修复要点（review-think-20261104-001）：
  - 权重：严格加载 `WEIGHT_DIR / PRIMARY_WEIGHT`，缺失即报错退出，不再按字母序盲取，也不回退到其他文件；
  - 数据：优先 /content/runtime 本地盘（必要时从 Drive tar 解压并写 .extracted 标记），仅非 Colab 时回退 Drive；
  - 环境：专用 env 的 python 不存在即 fail-fast（除非显式 `--allow-sys-python-fallback`）；
  - 锁：`.lock` 放在本地盘 /content/runtime/locks/，任何 OSError 均视为“未拿到锁”；
  - 多模态：统一传 3 通道 RGB，由模型内部 multi_modal_vision 自行复制成 6 通道；UETrack 的 dataset_name 可配置（默认 lasher）；
  - 遥测：每帧记录 concurrency_slots；计时前后 torch.cuda.synchronize()；并发耗时只用于吞吐监控，FPS 结论须用
    `--benchmark-latency` 单进程测速模式另行测得。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 导入同目录的三层失败提取器与遥测生成函数
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from failure_episode_extractor import (  # noqa: E402
    build_telemetry_rows_from_boxes,
    calc_iou_xywh,
    run_phase2_extraction_pipeline,
)

try:
    _colab_eng_dir = Path(__file__).resolve().parents[1] / "colab_a100_engineering"
    if _colab_eng_dir.exists() and str(_colab_eng_dir) not in sys.path:
        sys.path.insert(0, str(_colab_eng_dir))
    import colab_ntfy_telemetry as _ntfy_telem
    _ntfy_telem.ensure_daemon_started()
    _ntfy_telem.push_once(_ntfy_telem.collect_snapshot())
except Exception:
    pass


# ==============================================================================
# 并发调度常量
# ==============================================================================
CONCURRENCY_MODES = (
    "serial",
    "plan_b_multi_tracker",
    "plan_a_seq_shards",
    "hybrid_b_then_a",
)
DYNAMIC_BACKFILL_MIN_REMAINING_SEQS = 4  # 慢算法未完成序列数 > 4 时触发动态方案 A 补位 Worker
CONCURRENT_TIMING_NOTE = "并发模式耗时仅作吞吐监控，不可用于 FPS 结论（FPS 请使用 --benchmark-latency 单进程测速模式）"

# ==============================================================================
# 运行环境分组（详见 phase2_unified_eval/ENV_MATRIX.md）
# 同一分组内的跟踪器可共用一个 Python 环境；不同分组的依赖版本互相冲突，必须分别建 env。
# ==============================================================================
ENV_GROUPS: Dict[str, Dict[str, str]] = {
    "A_siamdt_cu118": {
        "python_bin_colab": "/content/envs/siamdt-cu118/bin/python",
        "summary": "SiamDT 原版 mmdet 风格栈（沿用既有 siamdt-cu118 环境）",
    },
    "B1_pytracking_pt20_cu118": {
        "python_bin_colab": "/content/envs/pytracking-pt20-cu118/bin/python",
        "summary": "Python 3.9 + torch 2.0.0/cu118（FocusTrack README 官方版本；ReFocus-TIR 为 OSTrack 系，未给官方版本，暂归此组）",
    },
    "B2_sutrack_pt111_cu113": {
        "python_bin_colab": "/content/envs/sutrack-pt111-cu113/bin/python",
        "summary": "Python 3.8 + torch 1.11.0+cu113 + timm 0.5.4 + OpenAI CLIP（SUTrack/UETrack install.sh 官方版本；UTPTrack-S README 指定 python=3.8）",
    },
    "C_sam2_pt25_cu124": {
        "python_bin_colab": "/content/envs/sam2-pt25-cu124/bin/python",
        "summary": "Python 3.10+ + torch>=2.3.1（DAM4SAM requirements 锁 cu124 系）+ numpy 1.26.4 + hydra-core 1.3.2 + vot-toolkit 0.7.1；"
                   "三个 SAM2 改版仓库都自带同名 `sam2` 包，禁止 pip install 任何 sam2，靠每个进程的 sys.path 隔离",
    },
    "D_trackit_memlo_pt2": {
        "python_bin_colab": "/content/envs/trackit-memlo-pt2/bin/python",
        "summary": "Python 3.10+ + torch>=2.0 + timm/safetensors/pyturbojpeg(apt libturbojpeg)/fvcore/wandb(禁用)",
    },
}


def _env_bin(group: str) -> str:
    return ENV_GROUPS[group]["python_bin_colab"]


# ==============================================================================
# 多算法元数据、仓库地址与官方权重清单 (Locked Phase 2 Tracker Registry)
# weight_files: 每个文件 {role, rel_path(相对 drive_root), bytes, source}；stage 时按“精确文件名 + 字节数”判断是否已就位。
# 权重 SHA256、来源 URL 的完整账本见 weights/WEIGHTS_LEDGER.json。
# ==============================================================================
TRACKER_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --------------------------------------------------------------------------
    # Group I: In-Domain 同域训练组 (Anti-UAV410 Trained)
    # --------------------------------------------------------------------------
    "SiamDT_Baseline": {
        "group": "group1_in_domain",
        "track_label": "In-Domain (Anti-UAV410 Official Baseline)",
        "diagnostic_axis": "Subject Baseline (Full-Image Swin-T + 5-Level FPN + DS-RPN + VR-CNN Top-5 + 11-BG Subtraction + Single EMA)",
        "env_tier": "A_siamdt_cu118",
        "python_bin_colab": _env_bin("A_siamdt_cu118"),
        "git_url": "https://github.com/HwangBo94/Anti-UAV410.git",
        "pinned_commit": "8a8eb04d976e9386b7c9c3ada5c85e5086013d52",
        "repo_rel_dir": "repos/Anti-UAV410_official",
        "weight_rel_dir": "weights/in_domain_antiuav410/SiamDT_Baseline",
        "primary_weight_file": "siamdt_swin_tiny_sgd.pth",
        "expected_bytes": 499427901,
        "expected_sha256": "b4c7e0e0490881f81a64352f4db213934b15194f3ad76c3937aee0204e4c6c38",
        "weight_files": [
            {
                "role": "primary",
                "rel_path": "weights/in_domain_antiuav410/SiamDT_Baseline/siamdt_swin_tiny_sgd.pth",
                "bytes": 499427901,
                "source": {
                    "type": "gdrive_zip",
                    "gdrive_id": "1ovjDnsB5FDGIY0D1smqW9U08J4ROMjW2",
                    "archive_rel_path": "datasets/archives/checkpoints.zip",
                    "member_suffix": "siamdt_swin_tiny_sgd.pth",
                },
            },
        ],
        "adapter_type": "siamdt_native",
        "default_config": "configs/siamdt_swin_tiny_sgd.py",
    },
    "FocusTrack": {
        "group": "group1_in_domain",
        "track_label": "In-Domain (Anti-UAV410 Trained)",
        "diagnostic_axis": "A1_Search_Scope_and_Resolution (Self-Adaptive Local Sampling vs. Full-Image RPN)",
        "env_tier": "B1_pytracking_pt20_cu118",
        "python_bin_colab": _env_bin("B1_pytracking_pt20_cu118"),
        "git_url": "https://github.com/vero1925/FocusTrack.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/FocusTrack",
        "weight_rel_dir": "weights/in_domain_antiuav410/FocusTrack",
        # 磁盘真实文件名（大小写敏感）；GDrive 文件夹 1ndhwjIIymsppHknSjLUd40NmPmQ4XZhi / focustrack_single
        "primary_weight_file": "FocusTrack_ep0030.pth.tar",
        "expected_bytes": 1413875893,
        "weight_files": [
            {
                "role": "primary",
                "rel_path": "weights/in_domain_antiuav410/FocusTrack/FocusTrack_ep0030.pth.tar",
                "bytes": 1413875893,
                "source": {"type": "gdrive_file", "gdrive_id": "1wWeP4SyCLm5hFuKcBPTsBYGwsFlCrCtj"},
            },
        ],
        "adapter_type": "pytracking_ostrack_family",
        "tracker_module": "focustrack",
        "default_config": "focustrack_single",
        "pytracking_dataset_name": "antiuav410",
    },
    "MemLoTrack": {
        "group": "group1_in_domain",
        "track_label": "In-Domain (Anti-UAV410 pretrain->finetune, retrained weight)",
        "diagnostic_axis": "A2_Temporal_Memory_and_Update_Gate (DINOv2+LoRA + Kalman-Gated FIFO Memory + MAL)",
        "env_tier": "D_trackit_memlo_pt2",
        "python_bin_colab": _env_bin("D_trackit_memlo_pt2"),
        "git_url": "https://github.com/SeoulTech-HCIRLab/MemLoTrack.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/MemLoTrack",
        "weight_rel_dir": "weights/in_domain_antiuav410/MemLoTrack",
        "primary_weight_file": "antiuav410_retrained_weight.bin",
        "expected_bytes": 508283437,
        "weight_files": [
            {
                "role": "primary",
                "rel_path": "weights/in_domain_antiuav410/MemLoTrack/antiuav410_retrained_weight.bin",
                "bytes": 508283437,
                "source": {"type": "gdrive_file", "gdrive_id": "1EKpgyXYgtNj6xayLoLzNXwFDiaJ1cYJt"},
            },
        ],
        # 官方批量入口: ./run.sh MemLoTrack dinov2 --mixin evaluation --mixin anti_uav_test --weight_path <w>
        # 适配器按分片生成数据子集 + mixin，调用 main.py 后把 results.zip 中的逐序列 XYWH 转为逐帧 jsonl
        "adapter_type": "trackit_memlotrack",
        "default_config": "dinov2",
    },
    "MemLoTrack-Scratch": {
        # P1-7：两个官方权重版本拆成两个条目；不计入 all/group1 默认集合，需显式点名
        "group": "group1_in_domain",
        "include_in_default_sets": False,
        "track_label": "In-Domain (Anti-UAV410 trained from scratch)",
        "diagnostic_axis": "A2_Temporal_Memory_and_Update_Gate (scratch weight ablation)",
        "env_tier": "D_trackit_memlo_pt2",
        "python_bin_colab": _env_bin("D_trackit_memlo_pt2"),
        "git_url": "https://github.com/SeoulTech-HCIRLab/MemLoTrack.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/MemLoTrack",
        "weight_rel_dir": "weights/in_domain_antiuav410/MemLoTrack",
        "primary_weight_file": "antiuav410_scratch_weight.bin",
        "expected_bytes": 508283437,
        "weight_files": [
            {
                "role": "primary",
                "rel_path": "weights/in_domain_antiuav410/MemLoTrack/antiuav410_scratch_weight.bin",
                "bytes": 508283437,
                "source": {"type": "gdrive_file", "gdrive_id": "1wDnnxWzfZzr62-zs1i6Rm6ZutOvWUGeH"},
            },
        ],
        "adapter_type": "trackit_memlotrack",
        "default_config": "dinov2",
    },
    # --------------------------------------------------------------------------
    # Group II: Zero-Shot / 基础模型与多模态机制探针组
    # --------------------------------------------------------------------------
    "DAM4SAM": {
        "group": "group2_zero_shot",
        "track_label": "Zero-Shot Foundation Model (SAM 2.1 + Distractor-Aware Memory)",
        "diagnostic_axis": "A3_Background_Distractor_Suppression (Decoupled RAM/DRM + Uncertainty Update Freeze)",
        "env_tier": "C_sam2_pt25_cu124",
        "python_bin_colab": _env_bin("C_sam2_pt25_cu124"),
        "git_url": "https://github.com/jovanavidenovic/DAM4SAM.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/DAM4SAM",
        "weight_rel_dir": "weights/foundation_backbones/sam2_1",
        "primary_weight_file": "sam2.1_hiera_large.pt",
        "expected_bytes": 898083611,
        "weight_files": [
            {
                "role": "primary",
                "rel_path": "weights/foundation_backbones/sam2_1/sam2.1_hiera_large.pt",
                "bytes": 898083611,
                "source": {"type": "url", "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"},
            },
        ],
        "adapter_type": "sam2_dam4sam",
        # DAM4SAMTracker(tracker_name)；utils/utils.py::determine_tracker 期望 <repo>/checkpoints/sam2.1_hiera_large.pt
        "default_config": "sam21pp-L",
        "repo_checkpoint_rel": "checkpoints/sam2.1_hiera_large.pt",
    },
    "SAMOSA": {
        "group": "group2_zero_shot",
        "track_label": "Zero-Shot Visual + Trajectory Motion Prior (SAM 2.1 + Markov MP + EDRM)",
        "diagnostic_axis": "A4_Motion_Modeling_and_Recovery (Higher-Order Markov Motion Predictor + Error Detection-Recovery)",
        "env_tier": "C_sam2_pt25_cu124",
        "python_bin_colab": _env_bin("C_sam2_pt25_cu124"),
        "git_url": "https://github.com/DurYi/SAMOSA.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/SAMOSA",
        # 官方 scripts/test_antiuav410.py 默认: --model_path sam2.1_hiera_large.pt --mode samosa_a，点+框提示，首个 exist 帧初始化
        "weight_rel_dir": "weights/foundation_backbones/sam2_1",
        "primary_weight_file": "sam2.1_hiera_large.pt",
        "expected_bytes": 898083611,
        "weight_files": [
            {
                "role": "primary",
                "rel_path": "weights/foundation_backbones/sam2_1/sam2.1_hiera_large.pt",
                "bytes": 898083611,
                "source": {"type": "url", "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"},
            },
            {
                "role": "motion_predictor",
                "rel_path": "weights/zero_shot_sota/SAMOSA/mp.pth",
                "bytes": 480127,
                "source": {"type": "repo_builtin", "repo_rel": "sam2/checkpoints/mp.pth"},
            },
        ],
        "adapter_type": "sam2_samosa_samurai",
        "default_config": "samosa_a",
    },
    "SAMURAI": {
        "group": "group2_zero_shot",
        "track_label": "Zero-Shot Foundation Model (SAM 2.1 + Linear Kalman Motion-Aware Memory)",
        "diagnostic_axis": "A4_Motion_Modeling_and_Recovery (Linear Kalman Filter Baseline for SAMOSA)",
        "env_tier": "C_sam2_pt25_cu124",
        "python_bin_colab": _env_bin("C_sam2_pt25_cu124"),
        "git_url": "https://github.com/yangchris11/samurai.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/SAMURAI",
        # 官方 scripts/main_inference.py 默认: model_name=base_plus, configs/samurai/sam2.1_hiera_b+.yaml, 每条视频重建 predictor
        "weight_rel_dir": "weights/foundation_backbones/sam2_1",
        "primary_weight_file": "sam2.1_hiera_base_plus.pt",
        "expected_bytes": 323606802,
        "weight_files": [
            {
                "role": "primary",
                "rel_path": "weights/foundation_backbones/sam2_1/sam2.1_hiera_base_plus.pt",
                "bytes": 323606802,
                "source": {"type": "url", "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"},
            },
        ],
        "adapter_type": "sam2_samosa_samurai",
        "default_config": "configs/samurai/sam2.1_hiera_b+.yaml",
    },
    "UETrack": {
        "group": "group2_zero_shot",
        "track_label": "Zero-Shot Multi-Modal Unified Tracker (CVPR 2026 TP-MoE + TAD)",
        "diagnostic_axis": "A5_Continuous_Context_and_Thermal_Representation (5-Modality Token-Pooling MoE)",
        "env_tier": "B2_sutrack_pt111_cu113",
        "python_bin_colab": _env_bin("B2_sutrack_pt111_cu113"),
        "git_url": "https://github.com/kangben258/UETrack.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/UETrack",
        "weight_rel_dir": "weights/zero_shot_sota/UETrack",
        "primary_weight_file": "uetrack_base.tar",
        "expected_bytes": 1267204495,
        "weight_files": [
            {
                "role": "primary",
                "rel_path": "weights/zero_shot_sota/UETrack/uetrack_base.tar",
                "bytes": 1267204495,
                "source": {"type": "hf", "repo_id": "kangben258/UETrack", "filename": "uetrack/checkpoints/uetrack_base.tar"},
            },
            {
                # build_encoder 以 pretrained=is_main_process()=True 调用 torch.load(PRETRAIN_TYPE)；uetrack_base 编码器为 fastitpnt_layer6
                "role": "encoder_pretrain",
                "rel_path": "weights/zero_shot_sota/UETrack/pretrained/fast_itpn_tiny_1600e_1k.pt",
                "bytes": 57814887,
                "source": {"type": "hf", "repo_id": "kangben258/UETrack", "filename": "uetrack/pretrained_models/fast_itpn_tiny_1600e_1k.pt"},
            },
        ],
        "adapter_type": "pytracking_ostrack_family",
        "tracker_module": "uetrack",
        "default_config": "uetrack_base",
        # UETrack 按 dataset_name 选择任务专家路由: LASHER -> task_index 3（RGB-T 热红外专家）。
        # 默认 "lasher"，可用 --uetrack-dataset-name 覆盖（例如 "lasot" -> RGB 专家 task_index 0）。
        "pytracking_dataset_name": "lasher",
        "encoder_pretrain_role": "encoder_pretrain",
    },
    "SUTrack": {
        "group": "group2_zero_shot",
        "track_label": "Zero-Shot Unified Multi-Modal Tracker (AAAI 2025)",
        "diagnostic_axis": "A5_MultiModal_Unified_Representation (Unified RGB/Thermal Single-Stream ViT)",
        "env_tier": "B2_sutrack_pt111_cu113",
        "python_bin_colab": _env_bin("B2_sutrack_pt111_cu113"),
        "git_url": "https://github.com/chenxin-dlut/SUTrack.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/SUTrack",
        "weight_rel_dir": "weights/zero_shot_sota/SUTrack",
        "primary_weight_file": "SUTRACK_ep0180.pth.tar",
        "expected_bytes": 1291057340,
        "weight_files": [
            {
                "role": "primary",
                "rel_path": "weights/zero_shot_sota/SUTrack/SUTRACK_ep0180.pth.tar",
                "bytes": 1291057340,
                "source": {"type": "hf", "repo_id": "xche32/SUTrack", "filename": "checkpoints/train/sutrack/sutrack_b224/SUTRACK_ep0180.pth.tar"},
            },
            {
                "role": "encoder_pretrain",
                "rel_path": "weights/zero_shot_sota/SUTrack/pretrained/fast_itpn_base_clipl_e1600.pt",
                "bytes": 180830695,
                "source": {"type": "hf", "repo_id": "xche32/SUTrack", "filename": "pretrained/itpn/fast_itpn_base_clipl_e1600.pt"},
            },
        ],
        "adapter_type": "pytracking_ostrack_family",
        "tracker_module": "sutrack",
        "default_config": "sutrack_b224",
        # 无 ANTIUAV410 专用键 -> 走 cfg.TEST.*.DEFAULT（MULTI_MODAL_VISION.DEFAULT=True，模型内部 3->6 通道复制）
        "pytracking_dataset_name": "antiuav410",
        "encoder_pretrain_role": "encoder_pretrain",
    },
    "UTPTrack-S": {
        "group": "group2_zero_shot",
        "track_label": "Zero-Shot Token-Pruning Multi-Modal Tracker (CVPR 2026)",
        "diagnostic_axis": "A1_A3_Early_Token_Pruning (>65% Background Token Pruning across Search & Templates)",
        "env_tier": "B2_sutrack_pt111_cu113",
        "python_bin_colab": _env_bin("B2_sutrack_pt111_cu113"),
        "git_url": "https://github.com/EIT-NLP/UTPTrack.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/UTPTrack/UTPTrack-S",
        "weight_rel_dir": "weights/zero_shot_sota/UTPTrack_S",
        "primary_weight_file": "SUTRACKCMP_ep0180.pth.tar",
        "expected_bytes": 2008607182,
        "weight_files": [
            {
                "role": "primary",
                "rel_path": "weights/zero_shot_sota/UTPTrack_S/SUTRACKCMP_ep0180.pth.tar",
                "bytes": 2008607182,
                "source": {"type": "hf", "repo_id": "HarrisonWu/UTPTrack", "filename": "UTPTrack-S-224/SUTRACKCMP_ep0180.pth.tar"},
            },
            {
                # ceatettamma_b224.yaml 的 PRETRAIN_TYPE 是作者集群绝对路径，运行时改指向本地同名文件（strict=True 加载会整体覆盖）
                "role": "encoder_pretrain",
                "rel_path": "weights/zero_shot_sota/UTPTrack_S/pretrained/fast_itpn_base_clipl_e1600.pt",
                "bytes": 180830695,
                "source": {"type": "hf", "repo_id": "xche32/SUTrack", "filename": "pretrained/itpn/fast_itpn_base_clipl_e1600.pt"},
            },
        ],
        "adapter_type": "pytracking_ostrack_family",
        # 官方权重名 SUTRACKCMP_ep0180 对应 lib/test/parameter/sutrackcmp.py；类 SUTRACKCMP 位于 lib/test/tracker/sutrackcmp.py；
        # 配置依据官方 VOT 入口 lib/test/vot/ceatettamma_b224.py: run_vot_exp('sutrackcmp', 'ceatettamma_b224', ...)
        "tracker_module": "sutrackcmp",
        "default_config": "ceatettamma_b224",
        "pytracking_dataset_name": "antiuav410",
        "encoder_pretrain_role": "encoder_pretrain",
    },
    "ReFocus-TIR": {
        "group": "group2_zero_shot",
        "track_label": "Zero-Shot Thermal-Specific Tracker (IEEE TNNLS Parameter-Efficient TIR Attention Refocusing)",
        "diagnostic_axis": "A1_Thermal_Attention_Refocusing (Mitigating ViT Self-Attention Dispersion on Textureless TIR Targets)",
        "env_tier": "B1_pytracking_pt20_cu118",
        "python_bin_colab": _env_bin("B1_pytracking_pt20_cu118"),
        "git_url": "https://github.com/laisimiao/ReFocus_TIR_Tracking.git",
        "pinned_commit": "df47763684a96c5160b09a88d780fb878a2d107b",
        "repo_rel_dir": "repos/core_pool/ReFocus_TIR",
        "weight_rel_dir": "weights/zero_shot_sota/ReFocus_TIR",
        "primary_weight_file": "OSTrack_ep0060.pth.tar",
        "expected_bytes": 378880109,
        "weight_files": [
            {
                "role": "primary",
                "rel_path": "weights/zero_shot_sota/ReFocus_TIR/OSTrack_ep0060.pth.tar",
                "bytes": 378880109,
                # README: GDrive 文件夹 1Q3NYwKwm-NBPKLxxApRHxNfyCgTbkOy2 / Model/OSTrack_ep0060.pth.tar
                "source": {"type": "gdrive_file", "gdrive_id": "1OtoNXFnUXmLIhtXYQMXDNQ_DC95YLh4U"},
            },
        ],
        "adapter_type": "pytracking_ostrack_family",
        "tracker_module": "refocus",
        "default_config": "vitb_refocus",
        "pytracking_dataset_name": "antiuav410",
    },
}


# ==============================================================================
# 通用工具与硬件槽位自动探测函数
# ==============================================================================
def resolve_default_drive_root() -> Path:
    """自动识别 Colab `/content/drive/MyDrive/SiamDT_Project` 或本地 `Z:/SiamDT_Project` / `G:/我的云端硬盘/SiamDT_Project`。"""
    env_val = os.environ.get("SIAMDT_DRIVE_ROOT")
    if env_val:
        return Path(env_val)
    colab_path = Path("/content/drive/MyDrive/SiamDT_Project")
    if colab_path.exists() or Path("/content/drive").exists():
        return colab_path
    z_drive = Path("Z:/SiamDT_Project")
    if z_drive.exists():
        return z_drive
    return Path("G:/我的云端硬盘/SiamDT_Project")


def is_colab_runtime() -> bool:
    """判断当前是否运行在 Colab（存在 /content 且有 Colab 环境变量或已挂载 Drive）。"""
    if not Path("/content").exists():
        return False
    return bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU")) or Path("/content/drive").exists()


def fsync_write_text(path: Path, content: str) -> None:
    """将文本写入文件并强制执行 os.fsync 确保落盘 Google Drive。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


def calc_sha256_prefix(path: Path, max_bytes: int = 0) -> str:
    """计算文件 SHA-256。"""
    h = hashlib.sha256()
    read_so_far = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
            read_so_far += len(chunk)
            if max_bytes > 0 and read_so_far >= max_bytes:
                break
    return h.hexdigest()


def parse_tracker_selection(selection: str) -> List[str]:
    """解析 `--trackers` 参数（支持 `all`, `group1`, `group2` 或逗号分隔列表）。
    注：`include_in_default_sets=False` 的条目（如 MemLoTrack-Scratch）不进入 all/group1/group2，需显式点名。"""
    sel = selection.strip()
    default_keys = [k for k, v in TRACKER_REGISTRY.items() if v.get("include_in_default_sets", True)]
    if sel.lower() == "all":
        return default_keys
    if sel.lower() in ("group1", "group1_in_domain", "in_domain"):
        return [k for k in default_keys if TRACKER_REGISTRY[k]["group"] == "group1_in_domain"]
    if sel.lower() in ("group2", "group2_zero_shot", "zero_shot"):
        return [k for k in default_keys if TRACKER_REGISTRY[k]["group"] == "group2_zero_shot"]

    resolved: List[str] = []
    lower_map = {k.lower(): k for k in TRACKER_REGISTRY.keys()}
    for part in sel.split(","):
        p = part.strip()
        if not p:
            continue
        if p in TRACKER_REGISTRY:
            resolved.append(p)
        elif p.lower() in lower_map:
            resolved.append(lower_map[p.lower()])
        else:
            raise ValueError(f"未注册的跟踪器名称: {p}。支持列表: {list(TRACKER_REGISTRY.keys())}")
    return resolved


def load_split_sequences(split_path: Path) -> Tuple[List[str], Dict[str, Any]]:
    """读取冻结划分 JSON 清单，返回序列列表与原始字典。"""
    if not split_path.exists():
        raise FileNotFoundError(f"找不到冻结子集清单文件: {split_path}")
    raw = json.loads(split_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "sequences" in raw and isinstance(raw["sequences"], list):
        return [str(x) for x in raw["sequences"]], raw
    # val90_full.json 结构：discovery_full_54.sequences + holdout_full_36.sequences（无 sequence_details，帧数回退读 IR_label.json）
    if isinstance(raw, dict) and isinstance(raw.get("discovery_full_54"), dict) and isinstance(raw.get("holdout_full_36"), dict):
        seqs = [str(x) for x in raw["discovery_full_54"].get("sequences", [])] + \
               [str(x) for x in raw["holdout_full_36"].get("sequences", [])]
        if seqs and len(seqs) == len(set(seqs)):
            return seqs, raw
    raise ValueError(f"无效的子集 JSON 结构: {split_path}")


def required_weight_paths(drive_root: Path, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """返回某跟踪器全部所需权重文件 [{role, path, expected_bytes, exists, size_ok, actual_bytes}]。"""
    out = []
    for wf in spec.get("weight_files", []):
        p = drive_root / wf["rel_path"]
        exists = p.exists() and p.is_file()
        actual = p.stat().st_size if exists else None
        exp = wf.get("bytes")
        out.append({
            "role": wf["role"],
            "path": str(p),
            "expected_bytes": exp,
            "actual_bytes": actual,
            "exists": exists,
            "size_ok": bool(exists and (exp is None or actual == exp)),
        })
    return out


def load_weights_ledger(drive_root: Path) -> Dict[str, Any]:
    """读取 weights/WEIGHTS_LEDGER.json（文件名 -> sha256 等），不存在则返回空表。"""
    lp = drive_root / "weights/WEIGHTS_LEDGER.json"
    if not lp.exists():
        return {}
    try:
        raw = json.loads(lp.read_text(encoding="utf-8"))
    except Exception:
        return {}
    table = {}
    for ent in raw.get("files", []):
        rel = ent.get("rel_path")
        if rel:
            table[rel.replace("\\", "/")] = ent
    return table


def resolve_lock_root(runtime_root: Path) -> Path:
    """`.lock` 根目录：Colab 用 /content/runtime/locks/；非 Colab 用本机临时目录（均为本地盘，避免 Drive FUSE 语义问题）。"""
    if is_colab_runtime():
        return runtime_root / "locks"
    return Path(tempfile.gettempdir()) / "phase2_locks"


def _split_archive_name(split_json_path: Path) -> str:
    stem = split_json_path.stem.lower()
    return "AntiUAV410_Full_val90.tar" if stem.startswith("val90") else "AntiUAV410_Core40_val.tar"


def resolve_dataset_val_dir(
    drive_root: Path,
    runtime_root: Path,
    split_json_path: Path,
    dataset_root_override: Optional[Path] = None,
    allow_extract: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    """
    评测数据目录解析（返回 (val_dir, 决策说明)）：
      1. `--dataset-root` 显式覆盖（指向直接包含各序列子目录的 val 目录）；
      2. /content/runtime/datasets/AntiUAV410/val 已存在且含所需 .extracted 标记 -> 直接使用；
      3. 在 Colab 上且 Drive 有对应 tar（val90_full.json -> AntiUAV410_Full_val90.tar，其余 -> AntiUAV410_Core40_val.tar）
         -> 解压到 /content/runtime/datasets/AntiUAV410/val 并写入 `.extracted_<tar名>` 标记；
      4. 仅当不在 Colab 时才回退 drive_root/datasets/AntiUAV410/val。
    """
    info: Dict[str, Any] = {"in_colab": is_colab_runtime()}
    if dataset_root_override is not None:
        info["decision"] = "CLI_OVERRIDE"
        return Path(dataset_root_override), info

    rt_val = runtime_root / "datasets/AntiUAV410/val"
    archive_name = _split_archive_name(split_json_path)
    archive = drive_root / "datasets/archives" / archive_name
    marker = rt_val / f".extracted_{archive_name}"
    info.update({"runtime_val_dir": str(rt_val), "archive": str(archive), "archive_exists": archive.exists(), "marker": str(marker)})

    if marker.exists():
        info["decision"] = "RUNTIME_LOCAL_DISK (marker present)"
        return rt_val, info

    if info["in_colab"]:
        if archive.exists():
            if not allow_extract:
                info["decision"] = "WOULD_EXTRACT_ARCHIVE_TO_RUNTIME (dry-run, not extracted)"
                return rt_val, info
            rt_val.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            print(f"[dataset] 正在把 {archive} 解压到本地盘 {rt_val} ...", flush=True)
            with tarfile.open(archive, "r:*") as tf:
                for m in tf:
                    name = m.name.replace("\\", "/")
                    if name.startswith("/") or ".." in Path(name).parts:
                        raise RuntimeError(f"tar 成员路径不安全: {m.name}")
                    tf.extract(m, rt_val)
            marker.write_text(json.dumps({"archive": str(archive), "extracted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                          "elapsed_sec": round(time.time() - t0, 1)}), encoding="utf-8")
            info["decision"] = "EXTRACTED_ARCHIVE_TO_RUNTIME"
            return rt_val, info
        if rt_val.exists() and any(rt_val.iterdir()):
            info["decision"] = "RUNTIME_LOCAL_DISK (pre-populated, no marker, archive missing)"
            return rt_val, info
        raise FileNotFoundError(
            f"Colab 上既没有本地数据 {rt_val}，Drive 上也没有 {archive}。拒绝直接从 Drive 逐帧读取（P0-5）。"
            f"如确需从 Drive 读，请显式传 --dataset-root。"
        )

    if rt_val.exists() and any(rt_val.iterdir()):
        info["decision"] = "RUNTIME_DIR_EXISTS (non-colab)"
        return rt_val, info
    info["decision"] = "NON_COLAB_FALLBACK_DRIVE_ROOT"
    return drive_root / "datasets/AntiUAV410/val", info


def detect_optimal_gpu_slots(requested_slots: int = 0) -> Dict[str, Any]:
    """
    根据 GPU 显存与型号自动计算并发算力槽位数 S (`--gpu-slots`):
      - A100 80GB (VRAM >= 70GB) -> 6 槽位
      - A100 40GB (VRAM >= 35GB) -> 4 槽位
      - L4 24GB   (VRAM >= 20GB) -> 2 槽位
      - 其他 GPU / CPU            -> 1 槽位
    若用户显式传入 requested_slots > 0，则以用户指定值为准。
    """
    gpu_name = "NO_GPU_DETECTED"
    vram_mb = 0.0
    auto_slots = 1
    tier_label = "FALLBACK_1_SLOT"

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        if out:
            first_line = out.splitlines()[0]
            parts = [x.strip() for x in first_line.split(",")]
            gpu_name = parts[0]
            vram_mb = float(parts[1]) if len(parts) > 1 else 0.0
    except Exception:
        pass

    vram_gb = round(vram_mb / 1024.0, 2) if vram_mb > 0 else 0.0
    name_upper = gpu_name.upper()

    if vram_gb >= 70.0 or ("A100" in name_upper and "80GB" in name_upper):
        auto_slots = 6
        tier_label = "A100_80GB_6_SLOTS"
    elif vram_gb >= 35.0 or "A100" in name_upper:
        auto_slots = 4
        tier_label = "A100_40GB_4_SLOTS"
    elif vram_gb >= 20.0 or "L4" in name_upper:
        auto_slots = 2
        tier_label = "L4_24GB_2_SLOTS"
    else:
        auto_slots = 1
        tier_label = f"GENERIC_1_SLOT ({gpu_name})"

    effective_slots = requested_slots if requested_slots > 0 else auto_slots
    return {
        "detected_gpu_name": gpu_name,
        "detected_vram_gb": vram_gb,
        "hardware_tier_label": tier_label,
        "auto_recommended_slots": auto_slots,
        "requested_slots_arg": requested_slots,
        "effective_gpu_slots": effective_slots,
    }


# ==============================================================================
# 功能 1: 盘点各算法源码、权重、数据集与评测进度状态 (--action status)
# ==============================================================================
def inspect_phase2_status(
    drive_root: Path,
    split_json_path: Path,
    selected_trackers: List[str],
    runtime_root: Path = Path("/content/runtime"),
    dataset_root_override: Optional[Path] = None,
) -> Dict[str, Any]:
    """扫描 Google Drive 金库中各算法的源码、权重（精确文件名 + 字节数）、数据集与评测遥测完成度。"""
    seqs, split_meta = load_split_sequences(split_json_path)
    split_name = split_meta.get("split_name", split_json_path.stem)
    val_dir, ds_info = resolve_dataset_val_dir(drive_root, runtime_root, split_json_path, dataset_root_override, allow_extract=False)
    ledger = load_weights_ledger(drive_root)

    seq_expected_frames = {
        item["seq_name"]: int(item.get("frames", 0))
        for item in split_meta.get("sequence_details", [])
        if "seq_name" in item
    }

    labels_ready_seqs = []
    full_frames_ready_seqs = []
    missing_val_seqs = []
    for s in seqs:
        sdir = val_dir / s
        if (sdir / "IR_label.json").exists():
            labels_ready_seqs.append(s)
            exp_f = seq_expected_frames.get(s, 0)
            jpg_cnt = 0
            try:
                with os.scandir(sdir) as it:
                    for entry in it:
                        if entry.is_file() and entry.name.lower().endswith(".jpg"):
                            jpg_cnt += 1
            except OSError:
                pass
            if exp_f > 0 and jpg_cnt >= exp_f:
                full_frames_ready_seqs.append(s)
            elif exp_f == 0 and jpg_cnt > 0:
                full_frames_ready_seqs.append(s)
            elif ds_info.get("archive_exists"):
                full_frames_ready_seqs.append(s)
            else:
                missing_val_seqs.append(s)
        else:
            missing_val_seqs.append(s)

    trackers_report: Dict[str, Any] = {}
    for tk in selected_trackers:
        spec = TRACKER_REGISTRY[tk]
        repo_dir = drive_root / spec["repo_rel_dir"]
        diag_telemetry_dir = drive_root / "experiments/failure_diagnosis/per_frame_telemetry" / f"{tk}_{split_name}"
        weights = required_weight_paths(drive_root, spec)
        for w in weights:
            rel = str(Path(w["path"]).relative_to(drive_root)).replace("\\", "/")
            w["ledger_sha256_12"] = (ledger.get(rel, {}).get("sha256") or "")[:12] or None

        completed_seqs = []
        if diag_telemetry_dir.exists():
            for s in seqs:
                jf = diag_telemetry_dir / f"{s}.jsonl"
                if jf.exists() and jf.stat().st_size > 0:
                    completed_seqs.append(s)

        trackers_report[tk] = {
            "group": spec["group"],
            "env_tier": spec["env_tier"],
            "python_bin_colab": spec["python_bin_colab"],
            "python_bin_present_here": Path(spec["python_bin_colab"]).exists(),
            "adapter_type": spec["adapter_type"],
            "tracker_module": spec.get("tracker_module"),
            "default_config": spec["default_config"],
            "repo_dir": str(repo_dir),
            "repo_present": repo_dir.exists(),
            "repo_cloned_git": (repo_dir / ".git").exists(),
            "weights_all_ready": all(w["size_ok"] for w in weights) and bool(weights),
            "weights": weights,
            "completed_sequences_count": len(completed_seqs),
            "pending_sequences_count": len(seqs) - len(completed_seqs),
        }

    return {
        "action": "status",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "drive_root": str(drive_root),
        "split_name": split_name,
        "split_total_sequences": len(seqs),
        "dataset_val_dir": str(val_dir),
        "dataset_resolution": ds_info,
        "dataset_labels_ready_sequences_count": len(labels_ready_seqs),
        "dataset_full_frames_ready_sequences_count": len(full_frames_ready_seqs),
        "dataset_pending_frame_sync_sequences_count": len(missing_val_seqs),
        "sample_pending_sequences": missing_val_seqs[:5],
        "trackers_status": trackers_report,
    }


# ==============================================================================
# 功能 2: 自动拉取多算法源码与官方公开权重 (--action stage-repos-and-weights)
# ==============================================================================
def _download_weight_file(drive_root: Path, repo_drive: Path, wf: Dict[str, Any], dst: Path) -> str:
    """按注册表 source 拉取单个权重文件到 dst（精确文件名）。返回状态字符串；失败抛异常由上层如实记录。"""
    src = wf["source"]
    st = src["type"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    if st == "hf":
        from huggingface_hub import hf_hub_download
        cache_dir = Path(tempfile.gettempdir()) / "phase2_hf_cache"
        local = hf_hub_download(repo_id=src["repo_id"], filename=src["filename"], local_dir=str(cache_dir))
        shutil.copyfile(local, tmp)
    elif st == "gdrive_file":
        import gdown
        out = gdown.download(id=src["gdrive_id"], output=str(tmp), quiet=False, resume=True)
        if not out:
            raise RuntimeError(f"gdown 未返回文件 (id={src['gdrive_id']})，可能为配额限制或需要权限")
    elif st == "url":
        subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(tmp), src["url"]], check=True)
    elif st == "repo_builtin":
        builtin = repo_drive / src["repo_rel"]
        if not builtin.exists():
            raise FileNotFoundError(f"仓库内置权重不存在: {builtin}")
        shutil.copyfile(builtin, tmp)
    elif st == "gdrive_zip":
        import zipfile
        zpath = drive_root / src["archive_rel_path"]
        if not zpath.exists():
            import gdown
            zpath.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(id=src["gdrive_id"], output=str(zpath), resume=True, quiet=False)
        found = False
        with zipfile.ZipFile(zpath, "r") as zf:
            for m in zf.infolist():
                if m.filename.endswith(src["member_suffix"]):
                    with zf.open(m) as s_, tmp.open("wb") as d_:
                        shutil.copyfileobj(s_, d_, 8 * 1024 * 1024)
                    found = True
                    break
        if not found:
            raise FileNotFoundError(f"压缩包 {zpath} 中找不到 *{src['member_suffix']}")
    else:
        raise ValueError(f"未知 source 类型: {st}")

    exp = wf.get("bytes")
    got = tmp.stat().st_size
    if exp is not None and got != exp:
        raise RuntimeError(f"下载后大小不符: {dst.name} 期望 {exp} 实得 {got}（已保留 {tmp.name} 供排查）")
    os.replace(tmp, dst)
    return f"DOWNLOADED: {dst.name} ({got} bytes)"


def stage_repos_and_weights(
    drive_root: Path,
    runtime_root: Path,
    selected_trackers: List[str],
    sync_to_runtime: bool = True,
) -> Dict[str, Any]:
    """
    在 Colab (或本地联网环境) 自动克隆所选算法的官方 GitHub 仓库至 Google Drive `repos/`，
    并按注册表 `weight_files` 逐个核验/拉取权重：仅当“精确文件名存在且字节数等于注册值”才判定为已就位，
    不再出现“目录里有任意 .pth.tar 就跳过”的情况。
    """
    results: Dict[str, Any] = {}

    for tk in selected_trackers:
        spec = TRACKER_REGISTRY[tk]
        repo_drive = drive_root / spec["repo_rel_dir"]

        tk_res: Dict[str, Any] = {
            "repo_status": "SKIPPED_ALREADY_EXISTS",
            "weight_status": [],
            "runtime_repo": None,
        }

        # 1. 在 Colab 运行期优先直接克隆至本地高速盘 /content/runtime/repos/...（规避 Drive 半同步目录问题）
        runtime_repo = runtime_root / spec["repo_rel_dir"]
        clone_rel = "repos/core_pool/UTPTrack" if spec["repo_rel_dir"].startswith("repos/core_pool/UTPTrack/") else spec["repo_rel_dir"]
        clone_target = (runtime_root if is_colab_runtime() else drive_root) / clone_rel

        if is_colab_runtime() and tk == "SiamDT_Baseline" and Path("/content/runtime/Anti-UAV410/trackers/SiamDT").exists() and not runtime_repo.exists():
            runtime_repo.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(Path("/content/runtime/Anti-UAV410"), runtime_repo, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)
            tk_res["repo_status"] = "COPIED_FROM_PATCHED_RUNTIME_ANTIUAV410"
            tk_res["runtime_repo"] = str(runtime_repo)
        elif not (runtime_repo if is_colab_runtime() else repo_drive).exists():
            clone_target.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{tk}] 正在克隆源码仓库 {spec['git_url']} -> {clone_target} ...")
            try:
                if clone_target.exists():
                    shutil.rmtree(clone_target, ignore_errors=True)
                subprocess.run(
                    ["git", "clone", "--depth", "1", spec["git_url"], str(clone_target)],
                    check=True,
                    timeout=300,
                )
                if spec["pinned_commit"] != "HEAD":
                    subprocess.run(["git", "-C", str(clone_target), "fetch", "--unshallow"], check=False)
                    subprocess.run(["git", "-C", str(clone_target), "checkout", spec["pinned_commit"]], check=True)
                tk_res["repo_status"] = "CLONED_SUCCESS"
            except Exception as exc:
                tk_res["repo_status"] = f"CLONE_FAILED: {exc}"

        if is_colab_runtime() and runtime_repo.exists():
            tk_res["runtime_repo"] = str(runtime_repo)

        # 3. 逐个核验或拉取权重文件（精确文件名 + 字节数）
        for wf in spec.get("weight_files", []):
            dst = drive_root / wf["rel_path"]
            exp = wf.get("bytes")
            if dst.exists() and (exp is None or dst.stat().st_size == exp):
                tk_res["weight_status"].append(f"VERIFIED_READY: {wf['role']} {dst.name} ({dst.stat().st_size} bytes)")
                continue
            if dst.exists():
                tk_res["weight_status"].append(
                    f"SIZE_MISMATCH_KEEP_AND_REFETCH: {dst.name} 现有 {dst.stat().st_size} != 期望 {exp}"
                )
            try:
                tk_res["weight_status"].append(_download_weight_file(drive_root, repo_drive, wf, dst))
            except Exception as exc:
                tk_res["weight_status"].append(f"WEIGHT_PULL_FAILED: {wf['role']} {dst.name}: {type(exc).__name__}: {exc}")

        results[tk] = tk_res

    return {
        "action": "stage-repos-and-weights",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }


# ==============================================================================
# 功能 3: 外部/原生跟踪器 `.txt` 预测框目录一键摄入与标准 `.jsonl` 遥测生成
# ==============================================================================
def ingest_raw_predictions_to_telemetry(
    tracker_name: str,
    raw_pred_dir: Path,
    split_json_path: Path,
    dataset_val_dir: Path,
    drive_root: Path,
    git_short_sha: str = "phase2",
) -> Dict[str, Any]:
    """
    将任意跟踪器跑出的原始预测框目录（支持每帧 `x,y,w,h` 或空格分隔 `.txt`，或 `{"res": [[x,y,w,h], ...]}` JSON）
    无损转换为统一 `.jsonl` 逐帧遥测文件，并同步写入：
      1. `experiments/benchmark_eval/<group>/<tracker>_<split>_<sha>/`
      2. `experiments/failure_diagnosis/per_frame_telemetry/<tracker>_<split>/`
    """
    if tracker_name not in TRACKER_REGISTRY:
        raise ValueError(f"未知的 tracker_name: {tracker_name}")
    spec = TRACKER_REGISTRY[tracker_name]
    seqs, split_meta = load_split_sequences(split_json_path)
    split_stem = split_meta.get("split_name", split_json_path.stem)

    eval_root = drive_root / "experiments/benchmark_eval" / spec["group"] / f"{tracker_name}_{split_stem}_{git_short_sha}"
    txt_box_dir = eval_root / "pred_bboxes_txt"
    uavtir_json_dir = eval_root / "pred_bboxes_uavtir_json"
    telemetry_eval_dir = eval_root / "per_frame_telemetry"
    telemetry_diag_dir = drive_root / "experiments/failure_diagnosis/per_frame_telemetry" / f"{tracker_name}_{split_stem}"

    for d in (txt_box_dir, uavtir_json_dir, telemetry_eval_dir, telemetry_diag_dir):
        d.mkdir(parents=True, exist_ok=True)

    seq_sa_scores: Dict[str, float] = {}
    missing_preds: List[str] = []

    for seq_id in seqs:
        lbl_file = dataset_val_dir / seq_id / "IR_label.json"
        if not lbl_file.exists():
            missing_preds.append(f"{seq_id} (missing IR_label.json)")
            continue
        lbl = json.loads(lbl_file.read_text(encoding="utf-8"))
        gt_rects = lbl["gt_rect"]
        exist_flags = lbl.get("exist", [1] * len(gt_rects))

        cand_files = [
            raw_pred_dir / f"{seq_id}.txt",
            raw_pred_dir / f"{seq_id}.json",
            raw_pred_dir / seq_id / "groundtruth.txt",
            raw_pred_dir / f"{seq_id}_pred.txt",
        ]
        pred_file = next((p for p in cand_files if p.exists()), None)
        if pred_file is None:
            missing_preds.append(seq_id)
            continue

        raw_text = pred_file.read_text(encoding="utf-8").strip()
        pred_boxes: List[List[float]] = []
        if raw_text.startswith("{") or raw_text.startswith("["):
            parsed = json.loads(raw_text)
            box_list = parsed.get("res", parsed) if isinstance(parsed, dict) else parsed
            pred_boxes = [[float(v) for v in b[:4]] for b in box_list]
        else:
            for ln in raw_text.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.replace(",", " ").replace("\t", " ").split()
                if len(parts) >= 4:
                    pred_boxes.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])

        if not pred_boxes:
            missing_preds.append(f"{seq_id} (empty predictions)")
            continue

        conf_file = raw_pred_dir / f"{seq_id}_score.txt"
        confs: Optional[List[float]] = None
        if conf_file.exists():
            confs = [float(x.strip()) for x in conf_file.read_text(encoding="utf-8").splitlines() if x.strip()]

        time_file = raw_pred_dir / f"{seq_id}_time.txt"
        lats_ms: Optional[List[float]] = None
        if time_file.exists():
            lats_ms = [float(x.strip()) * 1000.0 for x in time_file.read_text(encoding="utf-8").splitlines() if x.strip()]

        telemetry_rows = build_telemetry_rows_from_boxes(
            seq_id=seq_id,
            pred_bboxes=pred_boxes,
            gt_rects=gt_rects,
            exist_flags=exist_flags,
            confidences=confs,
            latencies_ms=lats_ms,
        )

        txt_content = "\n".join([f"{b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f}" for b in pred_boxes[: len(telemetry_rows)]]) + "\n"
        uavtir_json_content = json.dumps({"res": pred_boxes[: len(telemetry_rows)]})
        jsonl_content = "\n".join([json.dumps(r, ensure_ascii=False) for r in telemetry_rows]) + "\n"

        fsync_write_text(txt_box_dir / f"{seq_id}.txt", txt_content)
        fsync_write_text(uavtir_json_dir / f"{seq_id}.txt", uavtir_json_content)
        fsync_write_text(telemetry_eval_dir / f"{seq_id}.jsonl", jsonl_content)
        fsync_write_text(telemetry_diag_dir / f"{seq_id}.jsonl", jsonl_content)

        seq_sa_scores[seq_id] = float(sum(r["sa_score"] for r in telemetry_rows) / len(telemetry_rows))

    summary_payload = {
        "tracker_name": tracker_name,
        "tracker_group": spec["group"],
        "split_name": split_stem,
        "stage_git_sha": git_short_sha,
        "completed_sequences_count": len(seq_sa_scores),
        "total_target_sequences": len(seqs),
        "mean_sa_score": round(sum(seq_sa_scores.values()) / len(seq_sa_scores), 6) if seq_sa_scores else 0.0,
        "missing_sequences": missing_preds,
        "per_sequence_sa": seq_sa_scores,
        "timing_note": "摄入的外部结果：latency 字段来自 *_time.txt（若有），其测量条件未知，不可直接用于 FPS 结论",
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    fsync_write_text(eval_root / "eval_progress_summary.json", json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return summary_payload


# ==============================================================================
# 功能 4: 混合并发调度器 (方案 B 多算法并发优先 + 方案 A 单算法多视频分片自动补位)
# ==============================================================================
def effective_eval_tag(git_short_sha: str, benchmark_latency: bool) -> str:
    """单进程测速模式写入独立目录（tag 加 `_latbench`），不污染并发评测结果，也不会因“已完成”被跳过。"""
    return f"{git_short_sha}_latbench" if benchmark_latency else git_short_sha


def get_tracker_eval_dirs(
    drive_root: Path,
    tracker_name: str,
    split_stem: str,
    git_short_sha: str,
    benchmark_latency: bool = False,
) -> Dict[str, Path]:
    """获取某算法在 Google Drive 上的标准评测与遥测输出目录字典。"""
    spec = TRACKER_REGISTRY[tracker_name]
    tag = effective_eval_tag(git_short_sha, benchmark_latency)
    eval_root = drive_root / "experiments/benchmark_eval" / spec["group"] / f"{tracker_name}_{split_stem}_{tag}"
    diag_dir = (
        eval_root / "per_frame_telemetry"  # 测速模式不写入失败诊断目录
        if benchmark_latency
        else drive_root / "experiments/failure_diagnosis/per_frame_telemetry" / f"{tracker_name}_{split_stem}"
    )
    return {
        "eval_root": eval_root,
        "txt_box_dir": eval_root / "pred_bboxes_txt",
        "uavtir_json_dir": eval_root / "pred_bboxes_uavtir_json",
        "telemetry_eval_dir": eval_root / "per_frame_telemetry",
        "telemetry_diag_dir": diag_dir,
    }


def get_tracker_lock_dir(runtime_root: Path, tracker_name: str, split_stem: str, tag: str) -> Path:
    return resolve_lock_root(runtime_root) / f"{tracker_name}_{split_stem}_{tag}"


def scan_realtime_pending_sequences(
    drive_root: Path,
    dataset_val_dir: Path,
    tracker_name: str,
    split_stem: str,
    target_sequences: List[str],
    git_short_sha: str = "phase2",
    expected_frames_map: Optional[Dict[str, int]] = None,
    runtime_root: Path = Path("/content/runtime"),
    benchmark_latency: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """
    实时扫描 `telemetry_eval_dir/*.jsonl` 与 `telemetry_diag_dir/*.jsonl` 及本地盘 `.lock` 文件，
    返回 `(completed_seqs, unfinished_seqs, unclaimed_unfinished_seqs)`。
    """
    dirs = get_tracker_eval_dirs(drive_root, tracker_name, split_stem, git_short_sha, benchmark_latency)
    tel_eval_dir = dirs["telemetry_eval_dir"]
    tel_diag_dir = dirs["telemetry_diag_dir"]
    lock_dir = get_tracker_lock_dir(runtime_root, tracker_name, split_stem, effective_eval_tag(git_short_sha, benchmark_latency))

    completed: List[str] = []
    unfinished: List[str] = []
    unclaimed_unfinished: List[str] = []
    now_ts = time.time()

    for seq_id in target_sequences:
        exp_len = expected_frames_map.get(seq_id, 0) if expected_frames_map else 0
        if exp_len <= 0:
            lbl_p = dataset_val_dir / seq_id / "IR_label.json"
            if lbl_p.exists():
                try:
                    exp_len = len(json.loads(lbl_p.read_text(encoding="utf-8")).get("gt_rect", []))
                except Exception:
                    exp_len = 0

        # 检查是否已完成落盘
        is_done = False
        for check_dir in (tel_eval_dir, tel_diag_dir):
            jf = check_dir / f"{seq_id}.jsonl"
            if jf.exists() and jf.stat().st_size > 0:
                try:
                    line_cnt = sum(1 for ln in jf.open("r", encoding="utf-8") if ln.strip())
                    if (exp_len > 0 and line_cnt == exp_len) or (exp_len == 0 and line_cnt > 0):
                        is_done = True
                        break
                except Exception:
                    pass

        if is_done:
            completed.append(seq_id)
            continue

        unfinished.append(seq_id)

        # 检查是否有其他活跃分片 Worker 正在锁定推理该序列 (本地盘 .{seq_id}.lock 且未超时 30 分钟)
        lf = lock_dir / f".{seq_id}.lock"
        is_claimed = False
        if lf.exists():
            try:
                if (now_ts - lf.stat().st_mtime) < 1800.0:
                    is_claimed = True
            except OSError:
                is_claimed = True  # 状态不明时保守视为已被占用，避免重复跑
        if not is_claimed:
            unclaimed_unfinished.append(seq_id)

    return completed, unfinished, unclaimed_unfinished


def finalize_tracker_eval_summary(
    drive_root: Path,
    tracker_name: str,
    split_stem: str,
    target_sequences: List[str],
    git_short_sha: str = "phase2",
    concurrency_mode: str = "hybrid_b_then_a",
    gpu_slots: int = 1,
    benchmark_latency: bool = False,
) -> Dict[str, Any]:
    """
    当某算法的所有分片 Worker 完成后，统一扫描全部 `{seq_id}.jsonl` 汇总生成无冲突的 `eval_progress_summary.json`。
    """
    spec = TRACKER_REGISTRY[tracker_name]
    dirs = get_tracker_eval_dirs(drive_root, tracker_name, split_stem, git_short_sha, benchmark_latency)
    tel_eval_dir = dirs["telemetry_eval_dir"]
    tel_diag_dir = dirs["telemetry_diag_dir"]
    eval_root = dirs["eval_root"]

    seq_sa_scores: Dict[str, float] = {}
    lat_all: List[float] = []
    for seq_id in target_sequences:
        jf = tel_eval_dir / f"{seq_id}.jsonl"
        if not jf.exists():
            jf = tel_diag_dir / f"{seq_id}.jsonl"
        if jf.exists() and jf.stat().st_size > 0:
            try:
                rows = [json.loads(ln) for ln in jf.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if rows:
                    seq_sa_scores[seq_id] = round(float(sum(r["sa_score"] for r in rows) / len(rows)), 6)
                    lat_all.extend(
                        float(r.get("latency_ms", 0.0)) for r in rows[1:]
                        if r.get("latency_available", True) and float(r.get("latency_ms", 0.0)) > 0
                    )
            except Exception:
                pass

    mean_sa = round(sum(seq_sa_scores.values()) / len(seq_sa_scores), 6) if seq_sa_scores else 0.0
    summary_payload: Dict[str, Any] = {
        "tracker_name": tracker_name,
        "tracker_group": spec["group"],
        "split_name": split_stem,
        "stage_git_sha": git_short_sha,
        "completed_sequences_count": len(seq_sa_scores),
        "total_target_sequences": len(target_sequences),
        "mean_sa_score": mean_sa,
        "per_sequence_sa": seq_sa_scores,
        "concurrency_mode": "benchmark_latency_single_process" if benchmark_latency else concurrency_mode,
        "concurrency_slots": 1 if benchmark_latency else gpu_slots,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if benchmark_latency:
        mean_lat = (sum(lat_all) / len(lat_all)) if lat_all else None
        summary_payload["timing_note"] = "单进程测速模式（concurrency_slots=1，计时前后 torch.cuda.synchronize；首帧 init 不计入），可用于 FPS 结论"
        summary_payload["mean_track_latency_ms_excl_init"] = round(mean_lat, 3) if mean_lat else None
        summary_payload["fps_excl_init"] = round(1000.0 / mean_lat, 3) if mean_lat else None
    else:
        summary_payload["timing_note"] = CONCURRENT_TIMING_NOTE
        summary_payload["throughput_monitor_mean_latency_ms"] = round(sum(lat_all) / len(lat_all), 3) if lat_all else None
    if eval_root.exists():
        fsync_write_text(eval_root / "eval_progress_summary.json", json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return summary_payload


def build_hybrid_schedule_plan(
    tracker_pending_map: Dict[str, List[str]],
    gpu_slots: int,
    concurrency_mode: str,
) -> Dict[str, Any]:
    """
    根据待跑算法集合 `tracker_pending_map` (算法名 -> 待测序列列表)、总算力槽位数 `gpu_slots` (S)
    与并发模式 `concurrency_mode` 生成初始并发波次计划与动态补位策略表：
      - 若 M >= S 且为 `hybrid_b_then_a` / `plan_b_multi_tracker`：
        首批启动 S 个不同算法进程（每个算法 1 个主 Worker，占满 S 个槽位），其余 M - S 个算法进入等待队列；
      - 若 1 <= M < S 且为 `hybrid_b_then_a`：
        自动触发【方案 A 视频分片补位】，为每个算法分配 K_i = floor(S / M) (+ 余数分配) 个并行分片 Worker，
        使首批并发启动的总进程数直接等于 S，100% 吃满 A100 算力槽位！
    """
    S = max(1, int(gpu_slots))
    active_trackers = [tk for tk, pseqs in tracker_pending_map.items() if len(pseqs) > 0]
    M = len(active_trackers)

    initial_wave_jobs: List[Dict[str, Any]] = []
    queued_trackers: List[str] = []
    strategy_summary = ""

    if M == 0:
        strategy_summary = "ALL_TRACKERS_ALREADY_COMPLETED"
    elif concurrency_mode == "serial":
        strategy_summary = f"SERIAL_MODE (1 slot active at a time across {M} trackers)"
        first_tk = active_trackers[0]
        initial_wave_jobs.append({
            "slot_id": 0,
            "tracker_name": first_tk,
            "shard_idx": 0,
            "num_shards": 1,
            "dynamic_helper": False,
            "assigned_sequences_preview": tracker_pending_map[first_tk][:4],
            "assigned_sequences_count": len(tracker_pending_map[first_tk]),
        })
        queued_trackers = active_trackers[1:]

    elif concurrency_mode == "plan_b_multi_tracker":
        launch_tks = active_trackers[:S]
        queued_trackers = active_trackers[S:]
        strategy_summary = (
            f"PLAN_B_MULTI_TRACKER_ONLY (Launching {len(launch_tks)}/{M} trackers across {S} GPU slots, 1 shard/tracker)"
        )
        for slot_idx, tk in enumerate(launch_tks):
            pseqs = tracker_pending_map[tk]
            initial_wave_jobs.append({
                "slot_id": slot_idx,
                "tracker_name": tk,
                "shard_idx": 0,
                "num_shards": 1,
                "dynamic_helper": False,
                "assigned_sequences_preview": pseqs[:4],
                "assigned_sequences_count": len(pseqs),
            })

    elif concurrency_mode == "plan_a_seq_shards":
        first_tk = active_trackers[0]
        queued_trackers = active_trackers[1:]
        pseqs = tracker_pending_map[first_tk]
        k_shards = min(S, len(pseqs))
        strategy_summary = (
            f"PLAN_A_SEQ_SHARDS_ONLY (Tracker {first_tk} split into {k_shards} parallel video shards across {S} GPU slots)"
        )
        for s_idx in range(k_shards):
            shard_seqs = pseqs[s_idx::k_shards]
            initial_wave_jobs.append({
                "slot_id": s_idx,
                "tracker_name": first_tk,
                "shard_idx": s_idx,
                "num_shards": k_shards,
                "dynamic_helper": False,
                "assigned_sequences_preview": shard_seqs[:4],
                "assigned_sequences_count": len(shard_seqs),
            })

    elif concurrency_mode == "hybrid_b_then_a":
        if M >= S:
            launch_tks = active_trackers[:S]
            queued_trackers = active_trackers[S:]
            strategy_summary = (
                f"HYBRID_STAGE1_PLAN_B_PRIORITY (M={M} >= S={S}: Launching {S} distinct trackers in parallel to fill all {S} slots; "
                f"{len(queued_trackers)} trackers queued; dynamic Plan-A shard helpers auto-spawn when free slots appear & remaining_seqs > {DYNAMIC_BACKFILL_MIN_REMAINING_SEQS})"
            )
            for slot_idx, tk in enumerate(launch_tks):
                pseqs = tracker_pending_map[tk]
                initial_wave_jobs.append({
                    "slot_id": slot_idx,
                    "tracker_name": tk,
                    "shard_idx": 0,
                    "num_shards": 1,
                    "dynamic_helper": False,
                    "assigned_sequences_preview": pseqs[:4],
                    "assigned_sequences_count": len(pseqs),
                })
        else:
            # 1 <= M < S: 自动触发方案 A 视频分片补位，将 S 个槽位按 floor(S / M) + 余数分配给 M 个待跑算法
            base_k = S // M
            rem = S % M
            # 按待跑序列数降序排列以优先获得余数分片槽位
            sorted_tks = sorted(active_trackers, key=lambda x: len(tracker_pending_map[x]), reverse=True)
            shards_per_tk: Dict[str, int] = {}
            for i, tk in enumerate(sorted_tks):
                alloc = base_k + (1 if i < rem else 0)
                shards_per_tk[tk] = max(1, min(alloc, len(tracker_pending_map[tk])))

            strategy_summary = (
                f"HYBRID_PLAN_B_PLUS_PLAN_A_SHARD_BACKFILL (1 <= M={M} < S={S}: Auto-splitting {M} trackers into "
                f"{sum(shards_per_tk.values())} parallel video shards {shards_per_tk} to fill 100% of {S} GPU slots)"
            )
            slot_cursor = 0
            for tk in active_trackers:
                k_shards = shards_per_tk[tk]
                pseqs = tracker_pending_map[tk]
                for s_idx in range(k_shards):
                    shard_seqs = pseqs[s_idx::k_shards]
                    initial_wave_jobs.append({
                        "slot_id": slot_cursor,
                        "tracker_name": tk,
                        "shard_idx": s_idx,
                        "num_shards": k_shards,
                        "dynamic_helper": False,
                        "assigned_sequences_preview": shard_seqs[:4],
                        "assigned_sequences_count": len(shard_seqs),
                    })
                    slot_cursor += 1
    else:
        raise ValueError(f"未知的 concurrency_mode: {concurrency_mode}")

    return {
        "concurrency_mode": concurrency_mode,
        "total_gpu_slots": S,
        "active_trackers_count_M": M,
        "strategy_decision": strategy_summary,
        "initial_wave_slots_occupied": len(initial_wave_jobs),
        "slot_utilization_rate_wave1": round(len(initial_wave_jobs) / S, 4) if S > 0 else 0.0,
        "initial_wave_jobs": initial_wave_jobs,
        "queued_trackers_after_wave1": queued_trackers,
        "dynamic_plan_a_backfill_policy": {
            "enabled": (concurrency_mode == "hybrid_b_then_a"),
            "min_unfinished_sequences_to_trigger_helper": DYNAMIC_BACKFILL_MIN_REMAINING_SEQS + 1,
            "condition_rule": f"remaining_unfinished_seqs > {DYNAMIC_BACKFILL_MIN_REMAINING_SEQS}",
            "realtime_monitor_source": "telemetry_eval_dir/*.jsonl + local-disk lock_dir/.{seq_id}.lock",
            "helper_steal_direction": "tail_reverse_order (pending_seqs[::-1] with atomic .lock & .jsonl double-check)",
        },
    }


def _build_worker_script_for_tracker(tracker_name: str, spec: Dict[str, Any]) -> str:
    """
    为指定跟踪器生成在对应虚拟环境中运行的并发分片 Worker 脚本。
    支持：
    - 接收环境变量 `_P2_SHARD_IDX` (0..K-1)、`_P2_NUM_SHARDS` (K) 与 `_P2_DYNAMIC_HELPER` ("0" 或 "1")；
    - 对 `pending_seqs[shard_idx::num_shards]` 进行无冲突分片推理；
    - 若为动态派生的方案 A 补位 Worker (`_P2_DYNAMIC_HELPER="1"`)，从序列列表末尾反向抢占未完成序列；
    - 启动每一条序列前均再次检查 `{seq_id}.jsonl` 与本地盘 `.{seq_id}.lock` 原子锁，杜绝多 Worker 重复跑同一序列；
    - 权重严格使用 `WEIGHT_DIR / PRIMARY_WEIGHT`（及注册表声明的辅助权重），缺失即退出，不做任何回退。
    """
    return r'''
import os, sys, json, math, time, shutil, zipfile, subprocess
import numpy as np
from pathlib import Path

DRIVE_ROOT = Path(os.environ["_P2_DRIVE_ROOT"])
RUNTIME_ROOT = Path(os.environ["_P2_RUNTIME_ROOT"])
TRACKER_NAME = os.environ["_P2_TRACKER_NAME"]
TRACKER_GROUP = os.environ["_P2_TRACKER_GROUP"]
ADAPTER_TYPE = os.environ["_P2_ADAPTER_TYPE"]
REPO_DIR = Path(os.environ["_P2_REPO_DIR"])
WEIGHT_DIR = Path(os.environ["_P2_WEIGHT_DIR"])
PRIMARY_WEIGHT = os.environ["_P2_PRIMARY_WEIGHT"]
EXPECTED_BYTES = int(os.environ.get("_P2_EXPECTED_BYTES", "0") or 0)
AUX_WEIGHTS = json.loads(os.environ.get("_P2_AUX_WEIGHTS", "{}"))  # role -> 绝对路径
DEFAULT_CFG = os.environ["_P2_DEFAULT_CFG"]
TRACKER_MOD = os.environ.get("_P2_TRACKER_MOD", "")
PYTRACKING_DATASET = os.environ.get("_P2_PYTRACKING_DATASET", "antiuav410")
ENCODER_PRETRAIN_ROLE = os.environ.get("_P2_ENCODER_PRETRAIN_ROLE", "")
REPO_CKPT_REL = os.environ.get("_P2_REPO_CKPT_REL", "")
SPLIT_JSON = Path(os.environ["_P2_SPLIT_JSON"])
DATASET_DIR = Path(os.environ["_P2_DATASET_DIR"])
GIT_SHA = os.environ.get("_P2_GIT_SHA", "phase2")
EVAL_TAG = os.environ.get("_P2_EVAL_TAG", GIT_SHA)
MAX_SEQS = int(os.environ.get("_P2_MAX_SEQS", "0"))
LOCK_DIR = Path(os.environ["_P2_LOCK_DIR"])
CONCURRENCY_SLOTS = int(os.environ.get("_P2_CONCURRENCY_SLOTS", "1"))
CONCURRENCY_MODE = os.environ.get("_P2_CONCURRENCY_MODE", "serial")
BENCHMARK = os.environ.get("_P2_BENCHMARK", "0") == "1"

SHARD_IDX = int(os.environ.get("_P2_SHARD_IDX", "0"))
NUM_SHARDS = max(1, int(os.environ.get("_P2_NUM_SHARDS", "1")))
IS_DYNAMIC_HELPER = os.environ.get("_P2_DYNAMIC_HELPER", "0") == "1"
WORKER_TAG = os.environ.get("_P2_WORKER_TAG", f"shard{SHARD_IDX}of{NUM_SHARDS}")


def fatal(msg):
    print(f"[{TRACKER_NAME} | {WORKER_TAG}] FATAL: {msg}", flush=True)
    sys.exit(2)


# 0. 权重严格核验：只认 WEIGHT_DIR / PRIMARY_WEIGHT，不存在或字节数不符即退出（不回退到任何其他文件）
CKPT_PATH = WEIGHT_DIR / PRIMARY_WEIGHT
if not CKPT_PATH.is_file():
    fatal(f"主权重不存在: {CKPT_PATH}")
if EXPECTED_BYTES > 0 and CKPT_PATH.stat().st_size != EXPECTED_BYTES:
    fatal(f"主权重字节数不符: {CKPT_PATH} 实际 {CKPT_PATH.stat().st_size} != 注册 {EXPECTED_BYTES}")
for _role, _p in AUX_WEIGHTS.items():
    if not Path(_p).is_file():
        fatal(f"辅助权重 [{_role}] 不存在: {_p}")

raw_split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
split_stem = raw_split.get("split_name", SPLIT_JSON.stem)
seq_list = [str(x) for x in raw_split["sequences"]]
if MAX_SEQS > 0:
    seq_list = seq_list[:MAX_SEQS]

expected_len_map = {
    item["seq_name"]: int(item.get("frames", 0))
    for item in raw_split.get("sequence_details", [])
    if "seq_name" in item
}

eval_root = DRIVE_ROOT / "experiments/benchmark_eval" / TRACKER_GROUP / f"{TRACKER_NAME}_{split_stem}_{EVAL_TAG}"
txt_box_dir = eval_root / "pred_bboxes_txt"
uavtir_json_dir = eval_root / "pred_bboxes_uavtir_json"
telemetry_eval_dir = eval_root / "per_frame_telemetry"
if BENCHMARK:
    telemetry_diag_dir = telemetry_eval_dir  # 测速模式不写失败诊断目录
else:
    telemetry_diag_dir = DRIVE_ROOT / "experiments/failure_diagnosis/per_frame_telemetry" / f"{TRACKER_NAME}_{split_stem}"

for d in (txt_box_dir, uavtir_json_dir, telemetry_eval_dir, telemetry_diag_dir, LOCK_DIR):
    d.mkdir(parents=True, exist_ok=True)

try:
    import torch as _torch
    _CUDA_SYNC = bool(_torch.cuda.is_available())
except Exception:
    _torch = None
    _CUDA_SYNC = False


def cuda_sync():
    """计时前后同步 CUDA（仅在 torch 可用且有 CUDA 时），确保 latency_ms 反映真实 GPU 耗时。"""
    if _CUDA_SYNC:
        _torch.cuda.synchronize()


def fsync_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def calc_iou_xywh(pred, gt):
    px, py, pw, ph = [float(v) for v in pred[:4]]
    gx, gy, gw, gh = [float(v) for v in gt[:4]]
    if pw <= 0 or ph <= 0 or gw <= 0 or gh <= 0:
        return 0.0
    ox0, oy0 = max(px, gx), max(py, gy)
    ox1, oy1 = min(px + pw, gx + gw), min(py + ph, gy + gh)
    if ox1 <= ox0 or oy1 <= oy0:
        return 0.0
    inter = (ox1 - ox0) * (oy1 - oy0)
    union = (pw * ph) + (gw * gh) - inter
    return float(inter / union) if union > 0 else 0.0


def is_seq_already_done(seq_id: str) -> bool:
    exp_len = expected_len_map.get(seq_id, 0)
    if exp_len <= 0:
        lbl_file = DATASET_DIR / seq_id / "IR_label.json"
        if lbl_file.exists():
            try:
                exp_len = len(json.loads(lbl_file.read_text(encoding="utf-8"))["gt_rect"])
            except Exception:
                exp_len = 0
    for cdir in (telemetry_eval_dir, telemetry_diag_dir):
        jf = cdir / f"{seq_id}.jsonl"
        if jf.exists() and jf.stat().st_size > 0:
            try:
                cnt = sum(1 for ln in jf.open("r", encoding="utf-8") if ln.strip())
                if (exp_len > 0 and cnt == exp_len) or (exp_len == 0 and cnt > 0):
                    return True
            except Exception:
                pass
    return False


def try_acquire_seq_lock(seq_id: str) -> bool:
    """在本地盘 LOCK_DIR 上以 O_CREAT|O_EXCL 原子创建 .{seq_id}.lock；任何 OSError 一律视为未拿到锁。"""
    if is_seq_already_done(seq_id):
        return False
    lock_file = LOCK_DIR / f".{seq_id}.lock"
    now_ts = time.time()
    try:
        if lock_file.exists():
            if (now_ts - lock_file.stat().st_mtime) < 1800.0:
                return False
            lock_file.unlink()
    except OSError:
        return False
    payload = json.dumps({"pid": os.getpid(), "worker_tag": WORKER_TAG, "claimed_at": now_ts})
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as lf:
            lf.write(payload)
            lf.flush()
        return True
    except OSError:
        return False


def release_seq_lock(seq_id: str):
    lock_file = LOCK_DIR / f".{seq_id}.lock"
    try:
        if lock_file.exists():
            lock_file.unlink()
    except OSError:
        pass


def load_sequence(seq_id):
    """读取序列标注与帧列表；帧不齐返回 None。"""
    seq_dir = DATASET_DIR / seq_id
    lbl_file = seq_dir / "IR_label.json"
    if not lbl_file.exists():
        print(f"[{TRACKER_NAME} | {WORKER_TAG}] 序列 {seq_id} 缺少 IR_label.json，跳过。")
        return None
    lbl = json.loads(lbl_file.read_text(encoding="utf-8"))
    gt_rects = lbl["gt_rect"]
    exist_flags = lbl.get("exist", [1] * len(gt_rects))
    frames = sorted(seq_dir.glob("*.jpg"))
    total_f = min(len(frames), len(gt_rects))
    if total_f < len(gt_rects) or total_f == 0:
        print(f"[{TRACKER_NAME} | {WORKER_TAG}] 序列 {seq_id} JPG 帧数 ({len(frames)}) 未齐备 (期望 {len(gt_rects)})，跳过。")
        return None
    return seq_dir, gt_rects, exist_flags, frames[:total_f], total_f


def write_sequence_outputs(seq_id, gt_rects, exist_flags, total_f, pred_boxes, confs, up_flags, lats_ms, extra):
    """把一条序列的逐帧结果写成 .txt / 官方 JSON / 逐帧 jsonl（含 concurrency_slots 等遥测字段）。"""
    telemetry_rows = []
    for f in range(total_f):
        pb = [float(v) for v in pred_boxes[f][:4]] if len(pred_boxes[f]) >= 4 else [0.0, 0.0, 0.0, 0.0]
        gb = [float(x) for x in gt_rects[f][:4]] if len(gt_rects[f]) >= 4 else [0.0, 0.0, 0.0, 0.0]
        ex = int(exist_flags[f])
        p_abs = 1 if (pb[2] <= 0.0 or pb[3] <= 0.0) else 0
        if ex == 1 and gb[2] > 0.0 and gb[3] > 0.0:
            iou_f = calc_iou_xywh(pb, gb) if p_abs == 0 else 0.0
            sa_f = iou_f
            if p_abs == 0:
                ce_f = math.hypot((pb[0] + 0.5 * pb[2]) - (gb[0] + 0.5 * gb[2]), (pb[1] + 0.5 * pb[3]) - (gb[1] + 0.5 * gb[3]))
                ts_f = math.sqrt(max(1e-6, gb[2] * gb[3]))
                nce_f = ce_f / ts_f
            else:
                ce_f, ts_f, nce_f = -1.0, math.sqrt(max(1e-6, gb[2] * gb[3])), -1.0
        else:
            iou_f = 0.0
            sa_f = 1.0 if p_abs == 1 else 0.0
            ce_f, nce_f, ts_f = -1.0, -1.0, 0.0

        row = {
            "seq_id": seq_id,
            "frame_id": f,
            "exist": ex,
            "gt_bbox": gb,
            "pred_bbox": pb,
            "pred_absent_flag": p_abs,
            "iou": round(iou_f, 6),
            "sa_score": round(sa_f, 6),
            "center_error": round(ce_f, 4),
            "norm_center_error": round(nce_f, 4),
            "target_size": round(ts_f, 4),
            "confidence": round(float(confs[f]), 6),
            "template_updated": bool(up_flags[f]),
            "latency_ms": round(float(lats_ms[f]), 3),
            "concurrency_slots": 1 if BENCHMARK else CONCURRENCY_SLOTS,
            "concurrency_mode": "benchmark_latency_single_process" if BENCHMARK else CONCURRENCY_MODE,
            "timing_cuda_synchronized": _CUDA_SYNC,
        }
        row.update(extra)
        telemetry_rows.append(row)

    txt_content = "\n".join([f"{b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f}" for b in pred_boxes[:total_f]]) + "\n"
    uavtir_json_content = json.dumps({"res": [list(map(float, b[:4])) for b in pred_boxes[:total_f]]})
    jsonl_content = "\n".join([json.dumps(r, ensure_ascii=False) for r in telemetry_rows]) + "\n"

    fsync_write(txt_box_dir / f"{seq_id}.txt", txt_content)
    fsync_write(uavtir_json_dir / f"{seq_id}.txt", uavtir_json_content)
    fsync_write(telemetry_eval_dir / f"{seq_id}.jsonl", jsonl_content)
    if telemetry_diag_dir != telemetry_eval_dir:
        fsync_write(telemetry_diag_dir / f"{seq_id}.jsonl", jsonl_content)

    seq_sa = float(np.mean([r["sa_score"] for r in telemetry_rows]))
    print(f"[{TRACKER_NAME} | {WORKER_TAG}] 序列 {seq_id} ({total_f} 帧) 落盘完成 | SA={seq_sa:.4f}", flush=True)


def write_pytracking_local_py():
    """为 pytracking 系仓库生成 lib/test/evaluation/local.py（prj_dir 指向 REPO_DIR，network_path 指向 WEIGHT_DIR）。"""
    save_dir = RUNTIME_ROOT / "pytracking_out" / TRACKER_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 由 run_phase2_multi_tracker_colab.py 运行时自动生成，请勿手改",
        "from lib.test.evaluation.environment import EnvSettings",
        "",
        "def local_env_settings():",
        "    settings = EnvSettings()",
        f"    settings.prj_dir = {str(REPO_DIR)!r}",
        f"    settings.save_dir = {str(save_dir)!r}",
        f"    settings.results_path = {str(save_dir / 'test/tracking_results')!r}",
        f"    settings.segmentation_path = {str(save_dir / 'test/segmentation_results')!r}",
        f"    settings.result_plot_path = {str(save_dir / 'test/result_plots')!r}",
        f"    settings.network_path = {str(WEIGHT_DIR)!r}",
        "    return settings",
        "",
    ]
    loc = REPO_DIR / "lib/test/evaluation/local.py"
    tmp = loc.with_name(f"local.py.{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, loc)  # 同一算法多个分片同时写入时内容相同，原子替换即可
    return loc


def mask_to_xywh_official_samosa(mask_bool):
    """SAMOSA/SAMURAI 官方脚本口径: [x_min, y_min, x_max - x_min, y_max - y_min]（不加 1），空掩码 -> [0,0,0,0]。"""
    nz = np.argwhere(mask_bool)
    if len(nz) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    y_min, x_min = nz.min(axis=0).tolist()
    y_max, x_max = nz.max(axis=0).tolist()
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def sam2_object_score(state, frame_idx):
    """只读地从 SAM2 inference_state 中取该帧 object_score_logits 并做 sigmoid；取不到返回 None。"""
    try:
        od = state["output_dict"]
        out = od["cond_frame_outputs"].get(frame_idx) or od["non_cond_frame_outputs"].get(frame_idx)
        if out is None or out.get("object_score_logits") is None:
            return None
        return float(_torch.sigmoid(out["object_score_logits"].float()).flatten()[0].item())
    except Exception:
        return None


# 1. 扫描当前尚未完成的序列池，并按分片模式确定本 Worker 的执行队列
pending_seqs = [s for s in seq_list if not is_seq_already_done(s)]
if not pending_seqs:
    print(f"[{TRACKER_NAME} | {WORKER_TAG}] 所有目标序列均已完成，直接退出。")
    sys.exit(0)

if IS_DYNAMIC_HELPER:
    # 动态派生的方案 A 补位 Worker 从末尾反向切片抢占，最大化避免与正向主 Worker 碰撞
    rev_pending = list(reversed(pending_seqs))
    primary_queue = rev_pending[SHARD_IDX::NUM_SHARDS]
    fallback_tail = [s for s in rev_pending if s not in primary_queue]
else:
    # 常规分片 Worker: 先执行固定分片 pending_seqs[shard_idx::num_shards]，若提前跑完再从末尾反向协助剩余未认领序列
    primary_queue = pending_seqs[SHARD_IDX::NUM_SHARDS]
    fallback_tail = [s for s in reversed(pending_seqs) if s not in primary_queue] if NUM_SHARDS > 1 else []
worker_seq_queue = primary_queue + fallback_tail

print(
    f"[{TRACKER_NAME} | {WORKER_TAG}] 启动分片推理 | primary_slice={len(primary_queue)} 条 "
    f"| total_pending={len(pending_seqs)} 条 | dynamic_helper={IS_DYNAMIC_HELPER} | ckpt={CKPT_PATH} "
    f"| concurrency_slots={1 if BENCHMARK else CONCURRENCY_SLOTS} | benchmark={BENCHMARK}",
    flush=True,
)


# 2. 构建统一单帧步进跟踪器接口
class UnifiedTrackerStepper:
    def __init__(self):
        self.adapter = ADAPTER_TYPE
        os.chdir(REPO_DIR)
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))

        if self.adapter == "siamdt_native":
            siamdt_dir = REPO_DIR / "trackers/SiamDT"
            os.chdir(siamdt_dir)
            sys.path.insert(0, str(siamdt_dir))
            import init_paths  # noqa: F401
            import libs.data as data
            import libs.ops as ops
            from trackers import SiamDTTracker
            self.ops = ops
            transforms = data.BasicPairTransforms(train=False)
            self.tracker = SiamDTTracker(DEFAULT_CFG, str(CKPT_PATH), transforms, name_suffix=f"{TRACKER_NAME}_{WORKER_TAG}", visualize=False)

        elif self.adapter == "pytracking_ostrack_family":
            import cv2
            self.cv2 = cv2
            loc = write_pytracking_local_py()
            print(f"[{TRACKER_NAME} | {WORKER_TAG}] 已生成 {loc}", flush=True)
            from importlib import import_module
            param_mod = import_module(f"lib.test.parameter.{TRACKER_MOD}")
            tracker_mod = import_module(f"lib.test.tracker.{TRACKER_MOD}")
            params = param_mod.parameters(DEFAULT_CFG)
            # 官方 Tracker.get_parameters/run_sequence 会设置 debug/tracker_name/param_name
            params.debug = 0
            params.tracker_name = TRACKER_MOD
            params.param_name = DEFAULT_CFG
            # 严格指定权重，不按目录盲取
            params.checkpoint = str(CKPT_PATH)
            if ENCODER_PRETRAIN_ROLE:
                # build_encoder(pretrained=is_main_process()=True) 会 torch.load(PRETRAIN_TYPE)；改指向本地官方同名文件
                params.cfg.MODEL.ENCODER.PRETRAIN_TYPE = AUX_WEIGHTS[ENCODER_PRETRAIN_ROLE]
            tracker_cls = tracker_mod.get_tracker_class()
            self.tracker = tracker_cls(params, PYTRACKING_DATASET)
            print(f"[{TRACKER_NAME} | {WORKER_TAG}] pytracking dataset_name={PYTRACKING_DATASET!r}", flush=True)

        elif self.adapter == "sam2_dam4sam":
            from PIL import Image
            self.Image = Image
            try:
                import vot.region  # noqa: F401  DAM4SAM 的 DRM 判定用到 vot.region.Mask / calculate_overlaps
            except Exception as exc:
                fatal(f"DAM4SAM 依赖 vot-toolkit==0.7.1（见 DAM4SAM/requirements.txt 与 ENV_MATRIX.md），当前不可用: {exc}")
            # utils/utils.py::determine_tracker 期望 <repo>/checkpoints/<文件名>；以符号链接（失败则复制）方式就位
            ck_dst = REPO_DIR / REPO_CKPT_REL
            ck_dst.parent.mkdir(parents=True, exist_ok=True)
            if not (ck_dst.exists() and ck_dst.stat().st_size == CKPT_PATH.stat().st_size):
                try:
                    if ck_dst.is_symlink() or ck_dst.exists():
                        ck_dst.unlink()
                    os.symlink(CKPT_PATH, ck_dst)
                except OSError:
                    shutil.copyfile(CKPT_PATH, ck_dst)
            from dam4sam_tracker import DAM4SAMTracker
            # 模型只实例化一次；initialize() 内部每条序列都会新建 inference_state 并 reset_state
            self.tracker = DAM4SAMTracker(DEFAULT_CFG)

        elif self.adapter == "sam2_samosa_samurai":
            # 各 SAM2 改版仓库自带同名 sam2 包：把 <repo>/sam2 放在 sys.path 最前，确保导入的是本仓库版本
            sys.path.insert(0, str(REPO_DIR / "sam2"))
            from sam2.build_sam import build_sam2_video_predictor
            self.build_predictor = build_sam2_video_predictor
            self.predictor = None
            if TRACKER_NAME == "SAMOSA":
                mode = DEFAULT_CFG
                if not (REPO_DIR / "sam2/sam2/configs" / mode).is_dir():
                    fatal(f"SAMOSA mode 配置目录不存在: {mode}")
                # 与官方 utils.determine_model_cfg 一致：large + sam2.1 -> configs/<mode>/sam2.1_hiera_l.yaml
                name = PRIMARY_WEIGHT
                if "large" in name and "sam2.1_" in name:
                    self.model_cfg = f"configs/{mode}/sam2.1_hiera_l.yaml"
                elif "base_plus" in name:
                    self.model_cfg = f"configs/{mode}/sam2.1_hiera_b+.yaml"
                else:
                    fatal(f"无法从权重名推断 SAMOSA 模型尺寸: {name}")
                # 官方 test_antiuav410.py 在循环外只构建一次 predictor
                self.predictor = self.build_predictor(self.model_cfg, ckpt_path=str(CKPT_PATH),
                                                      mp_ckpt_path=AUX_WEIGHTS["motion_predictor"], device="cuda:0")
            else:
                # SAMURAI 官方 main_inference.py 每条视频重建 predictor（卡尔曼状态挂在模型上），这里保持一致
                self.model_cfg = DEFAULT_CFG

        elif self.adapter == "trackit_memlotrack":
            self.memlo_out_root = RUNTIME_ROOT / "memlo_runs" / f"{TRACKER_NAME}_{split_stem}_{EVAL_TAG}" / WORKER_TAG
            consts = REPO_DIR / "consts.yaml"
            if not consts.exists():
                tpl = REPO_DIR / "consts.template.yaml"
                tmp = consts.with_name(f"consts.yaml.{os.getpid()}.tmp")
                shutil.copyfile(tpl, tmp)
                os.replace(tmp, consts)

        else:
            fatal(f"未知 adapter: {self.adapter}")

    # ---------------- 逐帧接口（siamdt / pytracking / dam4sam） ----------------
    def run_sequence(self, seq_id, seq_dir: Path, frames: list, gt_rects: list, exist_flags: list):
        gt0 = [float(x) for x in gt_rects[0][:4]]
        pred_boxes = [gt0]
        confs = [1.0]
        up_flags = [True]
        lats_ms = [0.0]
        extra = {"confidence_available": True, "latency_available": True}

        if self.adapter == "siamdt_native":
            init_xyxy = np.array([gt0[0], gt0[1], gt0[0] + gt0[2] - 1.0, gt0[1] + gt0[3] - 1.0], dtype=np.float32)
            img0 = self.ops.read_image(str(frames[0]), self.tracker.color_fmt)
            cuda_sync(); t0 = time.perf_counter()
            self.tracker.init(img0, init_xyxy)
            cuda_sync(); lats_ms[0] = (time.perf_counter() - t0) * 1000.0
            for fpath in frames[1:]:
                img = self.ops.read_image(str(fpath), self.tracker.color_fmt)
                cuda_sync(); ts = time.perf_counter()
                results, up_flag = self.tracker.update(img, return_all=True)
                cuda_sync(); dt = (time.perf_counter() - ts) * 1000.0
                max_ind = int(results[:, -1].argmax())
                b = results[max_ind, :4]
                conf = float(results[max_ind, 4])
                pred_boxes.append([
                    round(float(b[0]), 4),
                    round(float(b[1]), 4),
                    round(float(b[2] - b[0] + 1.0), 4),
                    round(float(b[3] - b[1] + 1.0), 4),
                ])
                confs.append(conf)
                up_flags.append(bool(up_flag))
                lats_ms.append(dt)
            return pred_boxes, confs, up_flags, lats_ms, extra

        if self.adapter == "pytracking_ostrack_family":
            # 统一传 3 通道 RGB（与官方 Tracker._read_image 一致）；多模态模型内部按 multi_modal_vision 自行 3->6 通道复制
            img0 = self.cv2.cvtColor(self.cv2.imread(str(frames[0])), self.cv2.COLOR_BGR2RGB)
            cuda_sync(); t0 = time.perf_counter()
            if TRACKER_MOD == "refocus":
                self.tracker.initialize(img0, {"init_bbox": gt0}, seqname=seq_id)
            else:
                self.tracker.initialize(img0, {"init_bbox": gt0})
            cuda_sync(); lats_ms[0] = (time.perf_counter() - t0) * 1000.0
            has_score = False
            prev_out = None
            for fpath in frames[1:]:
                img = self.cv2.cvtColor(self.cv2.imread(str(fpath)), self.cv2.COLOR_BGR2RGB)
                cuda_sync(); ts = time.perf_counter()
                out = self.tracker.track(img, {"previous_output": prev_out})
                cuda_sync(); dt = (time.perf_counter() - ts) * 1000.0
                prev_out = out
                tb = out.get("target_bbox", [0.0, 0.0, 0.0, 0.0])
                sc = out.get("best_score", out.get("conf_score"))
                if sc is not None:
                    has_score = True
                    conf = float(sc.item() if hasattr(sc, "item") else sc)
                else:
                    conf = 0.0
                pred_boxes.append([round(float(x), 4) for x in tb[:4]])
                confs.append(conf)
                up_flags.append(False)
                lats_ms.append(dt)
            extra["confidence_available"] = has_score
            return pred_boxes, confs, up_flags, lats_ms, extra

        if self.adapter == "sam2_dam4sam":
            # 与官方 run_on_box_dataset.py 一致：main 带 @torch.inference_mode() + @torch.cuda.amp.autocast()；
            # 逐帧 tracker.track(PIL)，掩码 -> mask2box（x1-x0+1）；空掩码 -> 目标缺席
            with _torch.inference_mode(), _torch.autocast("cuda", dtype=_torch.float16):
                return self._run_dam4sam(frames, gt0, pred_boxes, confs, up_flags, lats_ms, extra)

        fatal(f"adapter {self.adapter} 不支持逐帧接口")

    def _run_dam4sam(self, frames, gt0, pred_boxes, confs, up_flags, lats_ms, extra):
        img0 = self.Image.open(str(frames[0])).convert("RGB")
        cuda_sync(); t0 = time.perf_counter()
        self.tracker.initialize(img0, None, bbox=gt0)
        cuda_sync(); lats_ms[0] = (time.perf_counter() - t0) * 1000.0
        last_added = self.tracker.last_added
        for fpath in frames[1:]:
            img = self.Image.open(str(fpath)).convert("RGB")
            cuda_sync(); ts = time.perf_counter()
            out = self.tracker.track(img)
            cuda_sync(); dt = (time.perf_counter() - ts) * 1000.0
            mask = out.get("pred_mask")
            if mask is not None and np.any(mask > 0):
                ys, xs = np.where(mask > 0)
                x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                pb = [x0, y0, x1 - x0 + 1.0, y1 - y0 + 1.0]
            else:
                pb = [0.0, 0.0, 0.0, 0.0]
            pred_boxes.append(pb)
            confs.append(0.0)
            up_flags.append(self.tracker.last_added != last_added)  # 本帧是否写入 DRM
            last_added = self.tracker.last_added
            lats_ms.append(dt)
        extra["confidence_available"] = False
        return pred_boxes, confs, up_flags, lats_ms, extra

    # ---------------- 官方批量视频接口 -> 逐帧（SAMOSA / SAMURAI） ----------------
    def run_sequence_sam2_video(self, seq_id, seq_dir: Path, frames: list, gt_rects: list, exist_flags: list):
        total_f = len(frames)
        if TRACKER_NAME == "SAMOSA":
            # 官方 load_json_point：以首个 exist==1 帧为提示帧，框 + 中心点双提示
            idx = next((i for i in range(total_f) if int(exist_flags[i]) == 1 and len(gt_rects[i]) >= 4), None)
        else:
            # SAMURAI 官方脚本以第 0 帧框为提示；若第 0 帧无目标，则同样取首个 exist 帧
            idx = 0 if (len(gt_rects[0]) >= 4 and int(exist_flags[0]) == 1) else next(
                (i for i in range(total_f) if int(exist_flags[i]) == 1 and len(gt_rects[i]) >= 4), None)
        extra = {"confidence_available": True, "latency_available": True, "prompt_frame": idx,
                 "latency_note": "init_state 视频解码不计入；prompt 帧计 add_new_points_or_box 耗时，其余帧为相邻两次 yield 的间隔"}
        if idx is None:
            z = [[0.0, 0.0, 0.0, 0.0] for _ in range(total_f)]
            extra["confidence_available"] = False
            return z, [0.0] * total_f, [False] * total_f, [0.0] * total_f, extra

        x, y, w, h = [float(v) for v in gt_rects[idx][:4]]
        if TRACKER_NAME == "SAMURAI":
            predictor = self.build_predictor(self.model_cfg, str(CKPT_PATH), device="cuda:0")
        else:
            predictor = self.predictor

        pred_boxes = [[0.0, 0.0, 0.0, 0.0] for _ in range(total_f)]
        confs = [0.0] * total_f
        up_flags = [False] * total_f
        lats_ms = [0.0] * total_f
        any_score = False
        with _torch.inference_mode(), _torch.autocast("cuda", dtype=_torch.float16):
            if TRACKER_NAME == "SAMURAI":
                state = predictor.init_state(str(seq_dir), offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
                bbox = (int(x), int(y), int(x + w), int(y + h))  # 官方 load_lasot_gt: map(int, ...)
                cuda_sync(); t0 = time.perf_counter()
                predictor.add_new_points_or_box(state, box=bbox, frame_idx=idx, obj_id=0)
                cuda_sync(); lats_ms[idx] = (time.perf_counter() - t0) * 1000.0
            else:
                state = predictor.init_state(str(seq_dir), offload_video_to_cpu=True)
                bbox = (x, y, x + w, y + h)
                point = np.array([[int(x + w / 2), int(y + h / 2)]], np.int32)
                labels = np.array([1], np.int32)
                cuda_sync(); t0 = time.perf_counter()
                predictor.add_new_points_or_box(state, points=point, box=bbox, labels=labels, frame_idx=idx, obj_id=0)
                cuda_sync(); lats_ms[idx] = (time.perf_counter() - t0) * 1000.0
            up_flags[idx] = True
            cuda_sync(); t_prev = time.perf_counter()
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                cuda_sync(); t_now = time.perf_counter()
                m = (masks[0][0] > 0.0).cpu().numpy()
                if frame_idx < total_f:
                    pred_boxes[frame_idx] = mask_to_xywh_official_samosa(m)
                    sc = sam2_object_score(state, frame_idx)
                    if sc is not None:
                        any_score = True
                        confs[frame_idx] = sc
                    if frame_idx != idx:
                        lats_ms[frame_idx] = (t_now - t_prev) * 1000.0
                t_prev = time.perf_counter()
        extra["confidence_available"] = any_score
        extra["confidence_source"] = "sigmoid(object_score_logits)" if any_score else "unavailable"
        del state
        if TRACKER_NAME == "SAMURAI":
            del predictor
        import gc
        gc.collect()
        _torch.cuda.empty_cache()
        return pred_boxes, confs, up_flags, lats_ms, extra

    # ---------------- 官方批量入口 -> 逐帧（MemLoTrack / TrackIt main.py） ----------------
    def run_batch_memlotrack(self, batch_seq_ids, batch_idx):
        tag = f"{WORKER_TAG}_b{batch_idx}"
        ds_root = RUNTIME_ROOT / "memlo_shards" / f"{TRACKER_NAME}_{split_stem}_{EVAL_TAG}_{tag}"
        if ds_root.exists():
            shutil.rmtree(ds_root)
        (ds_root / "val").mkdir(parents=True)
        for s in batch_seq_ids:
            os.symlink(DATASET_DIR / s, ds_root / "val" / s, target_is_directory=True)
        mixin_name = f"p2_{TRACKER_NAME.replace('-', '_')}_{tag}"
        mixin_path = REPO_DIR / "config/MemLoTrack/_mixin" / f"{mixin_name}.yaml"
        # 与官方 config/MemLoTrack/_mixin/anti_uav_test.yaml 的 EVAL 段一致，仅把数据根目录指向本分片子集、split 改为 val
        mixin_rules = [{
            "path": "run.data.eval.source",
            "value": {
                "type": "dataset",
                "parameters": {
                    "filters": [{"type": "data_cleaning.annotation_standard",
                                 "parameters": {"bounding_box_format": "XYXY", "bounding_box_coordinate_system": "Continuous"}}],
                    "datasets": [{"name": "Anti_UAV_410", "type": "SOT", "splits": "val", "path": str(ds_root),
                                  "parameters": {"evaluation_mode": "state_accuracy"}}],
                },
            },
        }]
        mixin_path.write_text(json.dumps(mixin_rules, indent=2), encoding="utf-8")  # JSON 是合法 YAML
        out_dir = self.memlo_out_root / tag
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        cmd = [sys.executable, "main.py", "MemLoTrack", DEFAULT_CFG, "--output_dir", str(out_dir), "--run_id", tag,
               "--mixin_config", "evaluation", "--mixin_config", mixin_name,
               "--weight_path", str(CKPT_PATH), "--disable_wandb", "--device", "cuda:0"]
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
        print(f"[{TRACKER_NAME} | {WORKER_TAG}] 官方批量入口: {' '.join(cmd)}", flush=True)
        cuda_sync(); t0 = time.perf_counter()
        ret = subprocess.run(cmd, cwd=str(REPO_DIR), env=env).returncode
        wall_ms = (time.perf_counter() - t0) * 1000.0
        try:
            mixin_path.unlink()
        except OSError:
            pass
        if ret != 0:
            raise RuntimeError(f"MemLoTrack main.py 退出码 {ret}")
        zips = sorted(out_dir.rglob("results.zip"))
        if not zips:
            raise RuntimeError(f"未在 {out_dir} 下找到 results.zip")
        ext = out_dir / "_extracted"
        with zipfile.ZipFile(zips[-1]) as zf:
            zf.extractall(ext)
        found = {}
        for p in ext.rglob("*.txt"):
            if p.stem in batch_seq_ids:
                found[p.stem] = p
        return found, wall_ms


def parse_xywh_txt(p: Path):
    boxes = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.replace(",", " ").replace("\t", " ").split()
        boxes.append([float(v) for v in parts[:4]])
    return boxes


stepper = None

if ADAPTER_TYPE == "trackit_memlotrack":
    # 批量模式：先认领主分片，再认领尾部剩余未认领序列；每批调用一次官方 main.py
    for b_idx, queue in enumerate((primary_queue, fallback_tail)):
        claimed = [s for s in queue if try_acquire_seq_lock(s)]
        if not claimed:
            continue
        try:
            if stepper is None:
                stepper = UnifiedTrackerStepper()
            seq_meta = {s: load_sequence(s) for s in claimed}
            runnable = [s for s, m in seq_meta.items() if m is not None]
            if not runnable:
                continue
            found, wall_ms = stepper.run_batch_memlotrack(runnable, b_idx)
            for s in runnable:
                seq_dir, gt_rects, exist_flags, frames, total_f = seq_meta[s]
                if s not in found:
                    print(f"[{TRACKER_NAME} | {WORKER_TAG}] 序列 {s} 在 results.zip 中缺失，未写遥测。", flush=True)
                    continue
                boxes = parse_xywh_txt(found[s])
                if len(boxes) != total_f:
                    print(f"[{TRACKER_NAME} | {WORKER_TAG}] 序列 {s} 帧数不符 ({len(boxes)} != {total_f})，未写遥测。", flush=True)
                    continue
                write_sequence_outputs(
                    s, gt_rects, exist_flags, total_f, boxes,
                    [0.0] * total_f, [False] * total_f, [0.0] * total_f,
                    {"confidence_available": False, "latency_available": False,
                     "latency_note": f"官方批量入口不提供逐帧耗时；本批 {len(runnable)} 条序列总墙钟 {wall_ms:.0f} ms（含模型加载）"},
                )
        finally:
            for s in claimed:
                release_seq_lock(s)
    sys.exit(0)

for seq_id in worker_seq_queue:
    # 启动前再次双重检查 .jsonl 与 .lock，防止与其他分片或动态补位 Worker 冲突
    if not try_acquire_seq_lock(seq_id):
        continue

    try:
        meta = load_sequence(seq_id)
        if meta is None:
            continue
        seq_dir, gt_rects, exist_flags, frames, total_f = meta

        if stepper is None:
            stepper = UnifiedTrackerStepper()

        if ADAPTER_TYPE == "sam2_samosa_samurai":
            res = stepper.run_sequence_sam2_video(seq_id, seq_dir, frames, gt_rects, exist_flags)
        else:
            res = stepper.run_sequence(seq_id, seq_dir, frames, gt_rects, exist_flags)
        pred_boxes, confs, up_flags, lats_ms, extra = res
        write_sequence_outputs(seq_id, gt_rects, exist_flags, total_f, pred_boxes, confs, up_flags, lats_ms, extra)
    finally:
        release_seq_lock(seq_id)
'''


def _spawn_tracker_shard_worker(
    tracker_name: str,
    shard_idx: int,
    num_shards: int,
    dynamic_helper: bool,
    slot_id: int,
    drive_root: Path,
    runtime_root: Path,
    split_json_path: Path,
    dataset_val_dir: Path,
    max_sequences: int,
    git_short_sha: str,
    split_stem: str,
    concurrency_mode: str,
    gpu_slots: int,
    benchmark_latency: bool = False,
    allow_sys_python_fallback: bool = False,
    uetrack_dataset_name: str = "lasher",
) -> Tuple[subprocess.Popen, Dict[str, Any]]:
    """启动单个算法分片子进程并返回 `(Popen, job_meta)`。"""
    spec = TRACKER_REGISTRY[tracker_name]
    py_bin = spec["python_bin_colab"]
    if not Path(py_bin).exists():
        if not allow_sys_python_fallback:
            raise SystemExit(
                f"[{tracker_name}] 专用环境 python 不存在: {py_bin}（分组 {spec['env_tier']}，见 ENV_MATRIX.md）。"
                f"拒绝静默改用 sys.executable；如确需回退请显式加 --allow-sys-python-fallback。"
            )
        print(f"[{tracker_name}] 警告: 已按 --allow-sys-python-fallback 改用 {sys.executable}", flush=True)
        py_bin = sys.executable
    runtime_repo = runtime_root / spec["repo_rel_dir"]
    drive_repo = drive_root / spec["repo_rel_dir"]
    if is_colab_runtime() and not runtime_repo.exists():
        # 优先在 Colab 本地盘直接从 GitHub 克隆完整源码（或复用 Cell 2 已打补丁的 /content/runtime/Anti-UAV410），规避 Drive 半同步目录
        if tracker_name == "SiamDT_Baseline" and Path("/content/runtime/Anti-UAV410/trackers/SiamDT").exists():
            runtime_repo.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(Path("/content/runtime/Anti-UAV410"), runtime_repo, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)
        else:
            clone_rel = "repos/core_pool/UTPTrack" if spec["repo_rel_dir"].startswith("repos/core_pool/UTPTrack/") else spec["repo_rel_dir"]
            clone_target = runtime_root / clone_rel
            clone_target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if clone_target.exists():
                    shutil.rmtree(clone_target, ignore_errors=True)
                subprocess.run(["git", "clone", "--depth", "1", spec["git_url"], str(clone_target)], check=True, timeout=300)
                if spec["pinned_commit"] != "HEAD":
                    subprocess.run(["git", "-C", str(clone_target), "fetch", "--unshallow"], check=False)
                    subprocess.run(["git", "-C", str(clone_target), "checkout", spec["pinned_commit"]], check=True)
            except Exception:
                if drive_repo.exists():
                    runtime_repo.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(drive_repo, runtime_repo, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)
    repo_dir = runtime_repo if runtime_repo.exists() else drive_repo
    weight_dir = drive_root / spec["weight_rel_dir"]
    aux = {wf["role"]: str(drive_root / wf["rel_path"]) for wf in spec.get("weight_files", []) if wf["role"] != "primary"}
    ds_name = uetrack_dataset_name if tracker_name == "UETrack" else spec.get("pytracking_dataset_name", "antiuav410")
    tag = effective_eval_tag(git_short_sha, benchmark_latency)
    worker_tag = (
        f"dyn_helper_slot{slot_id}_s{shard_idx}of{num_shards}"
        if dynamic_helper
        else f"slot{slot_id}_s{shard_idx}of{num_shards}"
    )

    worker_code = _build_worker_script_for_tracker(tracker_name, spec)
    env = os.environ.copy()
    env.update({
        "_P2_DRIVE_ROOT": str(drive_root),
        "_P2_RUNTIME_ROOT": str(runtime_root),
        "_P2_TRACKER_NAME": tracker_name,
        "_P2_TRACKER_GROUP": spec["group"],
        "_P2_ADAPTER_TYPE": spec["adapter_type"],
        "_P2_REPO_DIR": str(repo_dir),
        "_P2_WEIGHT_DIR": str(weight_dir),
        "_P2_PRIMARY_WEIGHT": spec["primary_weight_file"],
        "_P2_EXPECTED_BYTES": str(spec.get("expected_bytes", 0)),
        "_P2_AUX_WEIGHTS": json.dumps(aux, ensure_ascii=False),
        "_P2_DEFAULT_CFG": spec["default_config"],
        "_P2_TRACKER_MOD": spec.get("tracker_module", ""),
        "_P2_PYTRACKING_DATASET": ds_name,
        "_P2_ENCODER_PRETRAIN_ROLE": spec.get("encoder_pretrain_role", ""),
        "_P2_REPO_CKPT_REL": spec.get("repo_checkpoint_rel", ""),
        "_P2_SPLIT_JSON": str(split_json_path),
        "_P2_DATASET_DIR": str(dataset_val_dir),
        "_P2_GIT_SHA": git_short_sha,
        "_P2_EVAL_TAG": tag,
        "_P2_MAX_SEQS": str(max_sequences),
        "_P2_LOCK_DIR": str(get_tracker_lock_dir(runtime_root, tracker_name, split_stem, tag)),
        "_P2_CONCURRENCY_SLOTS": str(gpu_slots),
        "_P2_CONCURRENCY_MODE": concurrency_mode,
        "_P2_BENCHMARK": "1" if benchmark_latency else "0",
        "_P2_SHARD_IDX": str(shard_idx),
        "_P2_NUM_SHARDS": str(num_shards),
        "_P2_DYNAMIC_HELPER": "1" if dynamic_helper else "0",
        "_P2_WORKER_TAG": worker_tag,
    })

    # Worker 代码写入本地临时文件再执行（避免 `python -c <长代码>` 触发 Windows 32K / Linux 128KB 单参数长度上限）
    script_dir = Path(tempfile.gettempdir()) / "phase2_worker_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"worker_{tracker_name.replace('-', '_')}_{os.getpid()}.py"
    script_path.write_text(worker_code, encoding="utf-8")
    proc = subprocess.Popen([py_bin, str(script_path)], env=env)
    meta = {
        "slot_id": slot_id,
        "tracker_name": tracker_name,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "dynamic_helper": dynamic_helper,
        "worker_tag": worker_tag,
        "python_bin": py_bin,
        "pid": proc.pid,
        "started_at": time.time(),
    }
    return proc, meta


def preflight_check_weights(drive_root: Path, trackers: List[str]) -> Dict[str, Any]:
    """正式启动前核验所有被选算法的权重（精确文件名 + 字节数），缺任何一个即中止，避免在 A100 上浪费时间。"""
    problems = {}
    for tk in trackers:
        bad = [w for w in required_weight_paths(drive_root, TRACKER_REGISTRY[tk]) if not w["size_ok"]]
        if bad:
            problems[tk] = bad
    return problems


def execute_multi_tracker_eval(
    drive_root: Path,
    runtime_root: Path,
    split_json_path: Path,
    selected_trackers: List[str],
    concurrency_mode: str = "hybrid_b_then_a",
    requested_gpu_slots: int = 0,
    max_sequences: int = 0,
    git_short_sha: str = "phase2",
    dry_run: bool = False,
    dataset_root_override: Optional[Path] = None,
    allow_sys_python_fallback: bool = False,
    benchmark_latency: bool = False,
    uetrack_dataset_name: str = "lasher",
) -> Dict[str, Any]:
    """
    在 Colab A100/L4 上执行混合并发统一评测 (`hybrid_b_then_a`):
      1. 自动探测或使用 `--gpu-slots` 确定总算力槽位数 S；
      2. 实时扫描各算法未完成序列，生成方案 B (多算法并发) + 方案 A (多视频分片补位) 混合调度计划；
      3. 运行期通过动态进程池监控：一旦快算法跑完腾出空闲槽位，而慢算法仍有 > 4 条未完成序列，
         立即在空闲槽位派生方案 A 动态补位 Worker；
      4. 所有分片完成后统一汇总生成各算法的 `eval_progress_summary.json`。
    `benchmark_latency=True` 时强制 serial + 单槽位，写入独立 `_latbench` 目录，用于 FPS 结论。
    """
    if concurrency_mode not in CONCURRENCY_MODES:
        raise ValueError(f"不支持的 concurrency_mode: {concurrency_mode}，可选: {CONCURRENCY_MODES}")

    if benchmark_latency:
        concurrency_mode = "serial"
        requested_gpu_slots = 1
    gpu_slot_info = detect_optimal_gpu_slots(requested_gpu_slots)
    S = int(gpu_slot_info["effective_gpu_slots"])

    all_seqs, split_meta = load_split_sequences(split_json_path)
    split_stem = split_meta.get("split_name", split_json_path.stem)
    target_seqs = all_seqs[:max_sequences] if max_sequences > 0 else all_seqs
    expected_frames_map = {
        item["seq_name"]: int(item.get("frames", 0))
        for item in split_meta.get("sequence_details", [])
        if "seq_name" in item
    }
    dataset_val_dir, ds_info = resolve_dataset_val_dir(
        drive_root, runtime_root, split_json_path, dataset_root_override, allow_extract=not dry_run
    )

    # 扫描每个被选算法的实时已完成与待评测序列
    tracker_pending_map: Dict[str, List[str]] = {}
    tracker_completed_map: Dict[str, List[str]] = {}
    for tk in selected_trackers:
        comp, unfin, _ = scan_realtime_pending_sequences(
            drive_root=drive_root,
            dataset_val_dir=dataset_val_dir,
            tracker_name=tk,
            split_stem=split_stem,
            target_sequences=target_seqs,
            git_short_sha=git_short_sha,
            expected_frames_map=expected_frames_map,
            runtime_root=runtime_root,
            benchmark_latency=benchmark_latency,
        )
        tracker_completed_map[tk] = comp
        tracker_pending_map[tk] = unfin

    # 生成初始并发波次与混合补位计划表
    schedule_plan = build_hybrid_schedule_plan(
        tracker_pending_map=tracker_pending_map,
        gpu_slots=S,
        concurrency_mode=concurrency_mode,
    )
    weight_problems = preflight_check_weights(drive_root, selected_trackers)

    if dry_run:
        # 构造详细的 Dry-Run 验证报告（同时模拟展示：若 M >= S 时 Wave 1 方案 B 满载 + Wave 2 方案 A 动态补位触发演练，
        # 以及当 M < S 时直接触发方案 A 分片补位的对照预演）
        single_tracker_example = {selected_trackers[0]: tracker_pending_map[selected_trackers[0]]} if selected_trackers else {}
        plan_a_backfill_demo = build_hybrid_schedule_plan(
            tracker_pending_map=single_tracker_example,
            gpu_slots=S,
            concurrency_mode=concurrency_mode,
        ) if single_tracker_example else {}

        return {
            "action": "eval",
            "mode": "DRY_RUN_HYBRID_SCHEDULE_VERIFIED",
            "benchmark_latency": benchmark_latency,
            "split_name": split_stem,
            "target_sequences_count": len(target_seqs),
            "dataset_val_dir": str(dataset_val_dir),
            "dataset_resolution": ds_info,
            "lock_root": str(resolve_lock_root(runtime_root)),
            "gpu_slot_detection": gpu_slot_info,
            "selected_trackers_count": len(selected_trackers),
            "per_tracker_python_bin": {tk: TRACKER_REGISTRY[tk]["python_bin_colab"] for tk in selected_trackers},
            "per_tracker_python_bin_present_here": {tk: Path(TRACKER_REGISTRY[tk]["python_bin_colab"]).exists() for tk in selected_trackers},
            "preflight_weight_problems": weight_problems,
            "per_tracker_pending_counts": {tk: len(tracker_pending_map[tk]) for tk in selected_trackers},
            "hybrid_schedule_plan": schedule_plan,
            "plan_a_backfill_when_M_lt_S_preview": plan_a_backfill_demo,
            "timing_note": CONCURRENT_TIMING_NOTE if not benchmark_latency else "单进程测速模式",
        }

    if weight_problems:
        raise SystemExit("权重预检失败（精确文件名 + 字节数），拒绝启动:\n" + json.dumps(weight_problems, indent=2, ensure_ascii=False))

    spawn_common = dict(
        drive_root=drive_root,
        runtime_root=runtime_root,
        split_json_path=split_json_path,
        dataset_val_dir=dataset_val_dir,
        max_sequences=max_sequences,
        git_short_sha=git_short_sha,
        split_stem=split_stem,
        concurrency_mode=concurrency_mode,
        gpu_slots=S,
        benchmark_latency=benchmark_latency,
        allow_sys_python_fallback=allow_sys_python_fallback,
        uetrack_dataset_name=uetrack_dataset_name,
    )
    scan_common = dict(
        drive_root=drive_root,
        dataset_val_dir=dataset_val_dir,
        split_stem=split_stem,
        target_sequences=target_seqs,
        git_short_sha=git_short_sha,
        expected_frames_map=expected_frames_map,
        runtime_root=runtime_root,
        benchmark_latency=benchmark_latency,
    )

    # 正式并发执行循环
    active_procs: List[Tuple[subprocess.Popen, Dict[str, Any]]] = []
    occupied_slots: Dict[int, str] = {}
    completed_worker_logs: List[Dict[str, Any]] = []
    dynamic_backfill_events: List[Dict[str, Any]] = []
    helper_counter_per_tracker: Dict[str, int] = {tk: 0 for tk in selected_trackers}

    # 1. 启动 Wave 1 初始并发作业
    for job in schedule_plan["initial_wave_jobs"]:
        slot_id = int(job["slot_id"])
        tk = str(job["tracker_name"])
        proc, meta = _spawn_tracker_shard_worker(
            tracker_name=tk,
            shard_idx=int(job["shard_idx"]),
            num_shards=int(job["num_shards"]),
            dynamic_helper=bool(job["dynamic_helper"]),
            slot_id=slot_id,
            **spawn_common,
        )
        active_procs.append((proc, meta))
        occupied_slots[slot_id] = tk

    queued_trackers: List[str] = list(schedule_plan["queued_trackers_after_wave1"])

    # 2. 动态进程池监控与方案 B -> 方案 A 自动补位循环
    while active_procs or queued_trackers:
        time.sleep(0.5)
        still_running: List[Tuple[subprocess.Popen, Dict[str, Any]]] = []
        freed_slots: List[int] = []

        for proc, meta in active_procs:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, meta))
            else:
                slot_id = int(meta["slot_id"])
                freed_slots.append(slot_id)
                occupied_slots.pop(slot_id, None)
                meta_done = dict(meta)
                meta_done["returncode"] = ret
                meta_done["elapsed_sec"] = round(time.time() - float(meta["started_at"]), 2)
                completed_worker_logs.append(meta_done)

        active_procs = still_running

        if not freed_slots:
            # 队列非空但没有空闲槽位且无活跃进程的情况在 serial/plan_b 首波为空时可能出现，此处补一个空闲槽位
            if not active_procs and queued_trackers:
                freed_slots = [0]
            else:
                continue

        # 优先阶段 1：若等待队列 queued_trackers 中仍有未启动算法，优先按方案 B（或尾盘分片）填补空闲槽位
        while freed_slots and queued_trackers:
            next_tk = queued_trackers.pop(0)
            _, unfin, _ = scan_realtime_pending_sequences(tracker_name=next_tk, **scan_common)
            if not unfin:
                continue

            if concurrency_mode == "plan_a_seq_shards":
                # plan_a 模式下，当前算法独占所有空闲槽位分片运行
                k_shards = min(len(freed_slots), len(unfin))
                for s_idx in range(k_shards):
                    s_slot = freed_slots.pop(0)
                    p, m = _spawn_tracker_shard_worker(
                        tracker_name=next_tk, shard_idx=s_idx, num_shards=k_shards,
                        dynamic_helper=False, slot_id=s_slot, **spawn_common,
                    )
                    active_procs.append((p, m))
                    occupied_slots[s_slot] = next_tk
            else:
                # 方案 B：启动下一个待测算法
                s_slot = freed_slots.pop(0)
                p, m = _spawn_tracker_shard_worker(
                    tracker_name=next_tk, shard_idx=0, num_shards=1,
                    dynamic_helper=False, slot_id=s_slot, **spawn_common,
                )
                active_procs.append((p, m))
                occupied_slots[s_slot] = next_tk

        # 优先阶段 2 (方案 A 动态补位)：当 queued_trackers 已空且仍有空闲槽位 freed_slots 时，
        # 实时扫描正在运行的慢算法的 `telemetry_eval_dir/*.jsonl`；若某个慢算法仍有 > 4 条未完成序列，
        # 立即在空闲槽位上为该慢算法派生补充的方案 A 分片 Worker！
        if freed_slots and concurrency_mode == "hybrid_b_then_a" and active_procs:
            running_tks = sorted({str(m["tracker_name"]) for _, m in active_procs})
            for s_slot in list(freed_slots):
                # 实时统计各活跃算法的剩余未完成序列数与未锁定序列数
                candidates: List[Tuple[int, int, str]] = []
                for rtk in running_tks:
                    _, unfin, unclaimed = scan_realtime_pending_sequences(tracker_name=rtk, **scan_common)
                    cur_workers_for_rtk = sum(1 for _, m in active_procs if m["tracker_name"] == rtk)
                    # 严格执行阈值条件：慢算法还有超过 4 条未完成序列，且存在尚未被锁定的剩余序列
                    if len(unfin) > DYNAMIC_BACKFILL_MIN_REMAINING_SEQS and len(unclaimed) > 0:
                        candidates.append((len(unfin), -cur_workers_for_rtk, rtk))

                if not candidates:
                    break

                # 选择剩余未完成序列最多的慢算法派生方案 A 补位 Worker
                candidates.sort(reverse=True)
                best_unfin_cnt, _, slow_tk = candidates[0]
                freed_slots.remove(s_slot)
                helper_idx = helper_counter_per_tracker.get(slow_tk, 0)
                helper_counter_per_tracker[slow_tk] = helper_idx + 1
                cur_workers_total = sum(1 for _, m in active_procs if m["tracker_name"] == slow_tk) + 1

                p, m = _spawn_tracker_shard_worker(
                    tracker_name=slow_tk,
                    shard_idx=helper_idx,
                    num_shards=max(2, cur_workers_total),
                    dynamic_helper=True,
                    slot_id=s_slot,
                    **spawn_common,
                )
                active_procs.append((p, m))
                occupied_slots[s_slot] = slow_tk
                backfill_evt = {
                    "slot_id": s_slot,
                    "slow_tracker": slow_tk,
                    "remaining_unfinished_sequences_at_spawn": best_unfin_cnt,
                    "helper_worker_tag": m["worker_tag"],
                    "pid": m["pid"],
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                dynamic_backfill_events.append(backfill_evt)
                print(
                    f"[Hybrid B->A 动态补位] 空闲槽位 Slot-{s_slot} 已为慢算法 [{slow_tk}] "
                    f"(剩余 {best_unfin_cnt} 条未完成序列 > 4) 派生分片 Worker ({m['worker_tag']}, PID={m['pid']})"
                )

    # 3. 全部 Worker 结束后，统一汇总生成各算法的 eval_progress_summary.json
    final_summaries: Dict[str, Any] = {}
    for tk in selected_trackers:
        final_summaries[tk] = finalize_tracker_eval_summary(
            drive_root=drive_root,
            tracker_name=tk,
            split_stem=split_stem,
            target_sequences=target_seqs,
            git_short_sha=git_short_sha,
            concurrency_mode=concurrency_mode,
            gpu_slots=S,
            benchmark_latency=benchmark_latency,
        )

    return {
        "action": "eval",
        "concurrency_mode": concurrency_mode,
        "benchmark_latency": benchmark_latency,
        "gpu_slot_detection": gpu_slot_info,
        "dataset_val_dir": str(dataset_val_dir),
        "dataset_resolution": ds_info,
        "initial_schedule_plan": schedule_plan,
        "dynamic_plan_a_backfill_events": dynamic_backfill_events,
        "completed_workers_count": len(completed_worker_logs),
        "completed_workers": completed_worker_logs,
        "nonzero_exit_workers": [w for w in completed_worker_logs if w.get("returncode") not in (0, None)],
        "tracker_final_summaries": final_summaries,
        "timing_note": CONCURRENT_TIMING_NOTE if not benchmark_latency else "单进程测速模式，可用于 FPS 结论",
    }


# ==============================================================================
# CLI 主入口
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 Colab A100/L4 多算法统一评测与混合并发调度器 (方案 B 优先 + 方案 A 自动补位)"
    )
    parser.add_argument(
        "--action",
        choices=["status", "stage-repos-and-weights", "eval", "ingest-txt", "extract-episodes"],
        default="status",
        help="执行动作: status(盘点就绪状态) | stage-repos-and-weights(拉取源码与权重) | eval(运行统一评测) | ingest-txt(导入原始预测框为.jsonl) | extract-episodes(运行三层失败提取)",
    )
    parser.add_argument(
        "--trackers",
        type=str,
        default="all",
        help="目标算法: all | group1 | group2 | 或逗号分隔的算法名 (如 SiamDT_Baseline,FocusTrack,MemLoTrack,DAM4SAM)；MemLoTrack-Scratch 需显式点名",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="discovery_set.json",
        help="冻结子集 JSON 文件名或路径 (默认: discovery_set.json，可选: holdout_verification.json / val90_full.json)",
    )
    parser.add_argument(
        "--concurrency-mode",
        choices=list(CONCURRENCY_MODES),
        default="hybrid_b_then_a",
        help="并发调度模式: hybrid_b_then_a (默认: 方案B多算法同卡并发优先 + 方案A单算法多视频分片自动补位) | plan_b_multi_tracker | plan_a_seq_shards | serial",
    )
    parser.add_argument(
        "--gpu-slots",
        type=int,
        default=0,
        help="GPU 并发槽位数 S (默认 0 表示按显存自动检测: A100 80GB->6, A100 40GB->4, L4 24GB->2, 其他->1)",
    )
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=None,
        help="Google Drive 持久化根目录 (默认自动检测 /content/drive/MyDrive/SiamDT_Project 或 Z:/SiamDT_Project)",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("/content/runtime"),
        help="Colab 本地高速运行盘目录 (默认: /content/runtime)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="手动覆盖评测数据目录（直接包含各序列子目录的 val 目录）。默认: /content/runtime/datasets/AntiUAV410/val（必要时从 Drive tar 解压）",
    )
    parser.add_argument(
        "--allow-sys-python-fallback",
        action="store_true",
        help="专用 env 的 python 不存在时允许改用当前解释器（默认 fail-fast）",
    )
    parser.add_argument(
        "--benchmark-latency",
        action="store_true",
        help="单进程测速模式：强制 serial + 1 槽位，结果写入独立 *_latbench 目录，汇总给出可用于结论的 FPS",
    )
    parser.add_argument(
        "--uetrack-dataset-name",
        type=str,
        default="lasher",
        help="UETrack 的 dataset_name（决定任务专家路由；默认 lasher=RGB-T 热红外专家，lasot=RGB 专家）",
    )
    parser.add_argument(
        "--ingest-raw-txt-dir",
        type=Path,
        default=None,
        help="当 --action ingest-txt 时，指定某算法输出的原始 .txt 预测框目录",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="限制最多运行的序列数 (0 表示运行划分内全部序列；设为 1 或 2 可在 L4 上快速冒烟)",
    )
    parser.add_argument(
        "--run-extractor",
        action="store_true",
        help="评测或导入完成后，立即自动调用 failure_episode_extractor 生成 FAILURE_EPISODE_DATABASE.json 与 CANONICAL_CHALLENGE_WINDOWS.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅验证混合并发调度计划表与输出路径，不实际启动 GPU 推理进程，也不解压数据",
    )
    args = parser.parse_args()

    drive_root = args.drive_root if args.drive_root else resolve_default_drive_root()
    selected_trackers = parse_tracker_selection(args.trackers)

    # 解析 split_json 路径
    split_cand = Path(args.split)
    if split_cand.exists():
        split_json_path = split_cand
    elif (drive_root / "datasets/diagnostic_splits" / args.split).exists():
        split_json_path = drive_root / "datasets/diagnostic_splits" / args.split
    else:
        split_json_path = CURRENT_DIR.parent / "datasets/diagnostic_splits" / args.split

    if args.action == "status":
        rep = inspect_phase2_status(drive_root, split_json_path, selected_trackers, args.runtime_root, args.dataset_root)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return

    if args.action == "stage-repos-and-weights":
        rep = stage_repos_and_weights(drive_root, args.runtime_root, selected_trackers)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return

    if args.action == "ingest-txt":
        if len(selected_trackers) != 1 or not args.ingest_raw_txt_dir:
            raise ValueError("使用 --action ingest-txt 时必须指定单个 --trackers <Name> 及 --ingest-raw-txt-dir <Path>")
        val_dir, _ = resolve_dataset_val_dir(drive_root, args.runtime_root, split_json_path, args.dataset_root)
        rep = ingest_raw_predictions_to_telemetry(
            tracker_name=selected_trackers[0],
            raw_pred_dir=args.ingest_raw_txt_dir,
            split_json_path=split_json_path,
            dataset_val_dir=val_dir,
            drive_root=drive_root,
        )
        print(json.dumps(rep, indent=2, ensure_ascii=False))

    elif args.action == "eval":
        rep = execute_multi_tracker_eval(
            drive_root=drive_root,
            runtime_root=args.runtime_root,
            split_json_path=split_json_path,
            selected_trackers=selected_trackers,
            concurrency_mode=args.concurrency_mode,
            requested_gpu_slots=args.gpu_slots,
            max_sequences=args.max_sequences,
            dry_run=args.dry_run,
            dataset_root_override=args.dataset_root,
            allow_sys_python_fallback=args.allow_sys_python_fallback,
            benchmark_latency=args.benchmark_latency,
            uetrack_dataset_name=args.uetrack_dataset_name,
        )
        print(json.dumps(rep, indent=2, ensure_ascii=False))

    if args.action == "extract-episodes" or args.run_extractor:
        telemetry_root = drive_root / "experiments/failure_diagnosis/per_frame_telemetry"
        out_dir = drive_root / "experiments/failure_diagnosis/reports" / split_json_path.stem
        db_path, cw_path = run_phase2_extraction_pipeline(
            telemetry_root=telemetry_root,
            split_json_path=split_json_path,
            output_dir=out_dir,
        )
        print(json.dumps({
            "extractor_status": "COMPLETED",
            "failure_episode_database": str(db_path),
            "canonical_challenge_windows": str(cw_path),
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
