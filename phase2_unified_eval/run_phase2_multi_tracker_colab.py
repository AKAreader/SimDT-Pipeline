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
        配合 `.jsonl` 与 `.lock` 原子级双重检查防重跑），直至全部序列完成并统一汇总生成 `eval_progress_summary.json`！
2. 权重与源码持久化金库：自动将源码拉取至 Google Drive `repos/core_pool/`，权重拉取至 `weights/`，
   并在运行期同步至 Colab 本地高速盘 `/content/runtime/`；
3. 逐序列断点续评 (Auto-Resume)：每跑完 1 条序列立即将 `.txt` 预测框、官方兼容 JSON 与统一 `.jsonl`
   逐帧遥测实时 fsync 刷入 Google Drive，断线重连自动跳过已完成序列；
4. 统一遥测转换与三层失败提取：内置原生推理适配器与通用预测框摄入器 (`--ingest-raw-txt-dir`)，
   评测完成后可一键调用 `failure_episode_extractor.py` 生成 `FAILURE_EPISODE_DATABASE.json`
   与 `CANONICAL_CHALLENGE_WINDOWS.json`。
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


# ==============================================================================
# 多算法元数据、仓库地址与官方权重拉取清单 (Locked Phase 2 Tracker Registry)
# ==============================================================================
TRACKER_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --------------------------------------------------------------------------
    # Group I: In-Domain 同域训练组 (Anti-UAV410 Trained)
    # --------------------------------------------------------------------------
    "SiamDT_Baseline": {
        "group": "group1_in_domain",
        "track_label": "In-Domain (Anti-UAV410 Official Baseline)",
        "diagnostic_axis": "Subject Baseline (Full-Image Swin-T + 5-Level FPN + DS-RPN + VR-CNN Top-5 + 11-BG Subtraction + Single EMA)",
        "env_tier": "tier1_siamdt_cu118",
        "python_bin_colab": "/content/envs/siamdt-cu118/bin/python",
        "git_url": "https://github.com/HwangBo94/Anti-UAV410.git",
        "pinned_commit": "8a8eb04d976e9386b7c9c3ada5c85e5086013d52",
        "repo_rel_dir": "repos/Anti-UAV410_official",
        "weight_rel_dir": "weights/in_domain_antiuav410/SiamDT_Baseline",
        "primary_weight_file": "siamdt_swin_tiny_sgd.pth",
        "expected_bytes": 499427901,
        "expected_sha256": "b4c7e0e0490881f81a64352f4db213934b15194f3ad76c3937aee0204e4c6c38",
        "weight_source": {
            "type": "gdrive_zip",
            "gdrive_id": "1ovjDnsB5FDGIY0D1smqW9U08J4ROMjW2",
            "archive_rel_path": "datasets/archives/checkpoints.zip",
            "member_suffix": "siamdt_swin_tiny_sgd.pth",
        },
        "adapter_type": "siamdt_native",
        "default_config": "configs/siamdt_swin_tiny_sgd.py",
    },
    "FocusTrack": {
        "group": "group1_in_domain",
        "track_label": "In-Domain (Anti-UAV410 Trained)",
        "diagnostic_axis": "A1_Search_Scope_and_Resolution (Self-Adaptive Local Sampling vs. Full-Image RPN)",
        "env_tier": "tier2_modern_pyt2",
        "python_bin_colab": "/content/envs/modern-pyt2/bin/python",
        "git_url": "https://github.com/vero1925/FocusTrack.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/FocusTrack",
        "weight_rel_dir": "weights/in_domain_antiuav410/FocusTrack",
        "primary_weight_file": "FOCUSTRACK_ep0030.pth.tar",
        "weight_source": {
            "type": "gdrive_folder",
            "gdrive_folder_id": "1ndhwjIIymsppHknSjLUd40NmPmQ4XZhi",
            "gdrive_raw_results_folder_id": "1YmdjnSCKsmLKuGZ9MJysM7jiM8EUhQA2",
            "baidu_url": "https://pan.baidu.com/s/1vVNZtYes-V6HJS0bWh0DJA?pwd=1106",
        },
        "adapter_type": "pytracking_ostrack_family",
        "tracker_module": "focustrack",
        "default_config": "focustrack_single",
    },
    "MemLoTrack": {
        "group": "group1_in_domain",
        "track_label": "In-Domain (Anti-UAV410 Trained, retrained & scratch)",
        "diagnostic_axis": "A2_Temporal_Memory_and_Update_Gate (DINOv2+LoRA + Kalman-Gated FIFO Memory + MAL)",
        "env_tier": "tier2_modern_pyt2",
        "python_bin_colab": "/content/envs/modern-pyt2/bin/python",
        "git_url": "https://github.com/SeoulTech-HCIRLab/MemLoTrack.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/MemLoTrack",
        "weight_rel_dir": "weights/in_domain_antiuav410/MemLoTrack",
        "primary_weight_file": "antiuav410_retrained_weight.bin",
        "secondary_weight_file": "antiuav410_scratch_weight.bin",
        "weight_source": {
            "type": "gdrive_folder",
            "gdrive_folder_id": "1cr2yX1UagM48qIuWzvYUSfl2V1CUZuAK",
        },
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
        "env_tier": "tier2_modern_pyt2",
        "python_bin_colab": "/content/envs/modern-pyt2/bin/python",
        "git_url": "https://github.com/jovanavidenovic/DAM4SAM.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/DAM4SAM",
        "weight_rel_dir": "weights/foundation_backbones/sam2_1",
        "primary_weight_file": "sam2.1_hiera_large.pt",
        "weight_source": {
            "type": "direct_url",
            "urls": {
                "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
                "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
            },
        },
        "adapter_type": "sam2_dam4sam",
        "default_config": "sam21_L",
    },
    "SAMOSA": {
        "group": "group2_zero_shot",
        "track_label": "Zero-Shot Visual + Trajectory Motion Prior (SAM 2.1 + Markov MP + EDRM)",
        "diagnostic_axis": "A4_Motion_Modeling_and_Recovery (Higher-Order Markov Motion Predictor + Error Detection-Recovery)",
        "env_tier": "tier2_modern_pyt2",
        "python_bin_colab": "/content/envs/modern-pyt2/bin/python",
        "git_url": "https://github.com/DurYi/SAMOSA.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/SAMOSA",
        "weight_rel_dir": "weights/zero_shot_sota/SAMOSA",
        "primary_weight_file": "mp.pth",
        "weight_source": {
            "type": "repo_builtin_plus_sam2",
            "repo_builtin_rel": "sam2/checkpoints/mp.pth",
            "sam2_checkpoint": "weights/foundation_backbones/sam2_1/sam2.1_hiera_base_plus.pt",
            "sam2_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        },
        "adapter_type": "sam2_samosa_samurai",
        "default_config": "sam2.1_hiera_b+",
    },
    "SAMURAI": {
        "group": "group2_zero_shot",
        "track_label": "Zero-Shot Foundation Model (SAM 2.1 + Linear Kalman Motion-Aware Memory)",
        "diagnostic_axis": "A4_Motion_Modeling_and_Recovery (Linear Kalman Filter Baseline for SAMOSA)",
        "env_tier": "tier2_modern_pyt2",
        "python_bin_colab": "/content/envs/modern-pyt2/bin/python",
        "git_url": "https://github.com/yangchris11/samurai.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/SAMURAI",
        "weight_rel_dir": "weights/foundation_backbones/sam2_1",
        "primary_weight_file": "sam2.1_hiera_base_plus.pt",
        "weight_source": {
            "type": "direct_url",
            "urls": {
                "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
            },
        },
        "adapter_type": "sam2_samosa_samurai",
        "default_config": "sam2.1_hiera_b+",
    },
    "UETrack": {
        "group": "group2_zero_shot",
        "track_label": "Zero-Shot Multi-Modal Unified Tracker (CVPR 2026 TP-MoE + TAD)",
        "diagnostic_axis": "A5_Continuous_Context_and_Thermal_Representation (5-Modality Token-Pooling MoE)",
        "env_tier": "tier2_modern_pyt2",
        "python_bin_colab": "/content/envs/modern-pyt2/bin/python",
        "git_url": "https://github.com/kangben258/UETrack.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/UETrack",
        "weight_rel_dir": "weights/zero_shot_sota/UETrack_MCITrack",
        "primary_weight_file": "uetrack_base.pth.tar",
        "weight_source": {
            "type": "huggingface",
            "hf_repo_id": "kangben258/UETrack",
            "hf_allow_patterns": ["*uetrack*checkpoint*", "*.pth*", "*.pt", "*.tar"],
        },
        "adapter_type": "pytracking_ostrack_family",
        "tracker_module": "uetrack",
        "default_config": "uetrack_base",
    },
    "SUTrack": {
        "group": "group2_zero_shot",
        "track_label": "Zero-Shot Unified Multi-Modal Tracker (AAAI 2025)",
        "diagnostic_axis": "A5_MultiModal_Unified_Representation (Unified RGB/Thermal Single-Stream ViT)",
        "env_tier": "tier2_modern_pyt2",
        "python_bin_colab": "/content/envs/modern-pyt2/bin/python",
        "git_url": "https://github.com/chenxin-dlut/SUTrack.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/SUTrack",
        "weight_rel_dir": "weights/zero_shot_sota/ReFocus_TIR_SUTrack",
        "primary_weight_file": "SUTRACK_ep0180.pth.tar",
        "weight_source": {
            "type": "huggingface",
            "hf_repo_id": "xche32/SUTrack",
            "hf_allow_patterns": ["*b224*", "*b256*", "*.pth.tar"],
        },
        "adapter_type": "pytracking_ostrack_family",
        "tracker_module": "sutrack",
        "default_config": "sutrack_b224",
    },
    "UTPTrack-S": {
        "group": "group2_zero_shot",
        "track_label": "Zero-Shot Token-Pruning Multi-Modal Tracker (CVPR 2026)",
        "diagnostic_axis": "A1_A3_Early_Token_Pruning (>65% Background Token Pruning across Search & Templates)",
        "env_tier": "tier2_modern_pyt2",
        "python_bin_colab": "/content/envs/modern-pyt2/bin/python",
        "git_url": "https://github.com/EIT-NLP/UTPTrack.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/UTPTrack",
        "weight_rel_dir": "weights/zero_shot_sota/ReFocus_TIR_SUTrack",
        "primary_weight_file": "UTPTrack_S_b224.pth.tar",
        "weight_source": {
            "type": "huggingface",
            "hf_repo_id": "HarrisonWu/UTPTrack",
            "hf_allow_patterns": ["*UTPTrack*", "*.pth*", "*.tar"],
        },
        "adapter_type": "pytracking_ostrack_family",
        "tracker_module": "sutrack",
        "default_config": "utptrack_s_b224",
    },
    "ReFocus-TIR": {
        "group": "group2_zero_shot",
        "track_label": "Zero-Shot Thermal-Specific Tracker (IEEE TNNLS Parameter-Efficient TIR Attention Refocusing)",
        "diagnostic_axis": "A1_Thermal_Attention_Refocusing (Mitigating ViT Self-Attention Dispersion on Textureless TIR Targets)",
        "env_tier": "tier2_modern_pyt2",
        "python_bin_colab": "/content/envs/modern-pyt2/bin/python",
        "git_url": "https://github.com/laisimiao/ReFocus_TIR_Tracking.git",
        "pinned_commit": "HEAD",
        "repo_rel_dir": "repos/core_pool/ReFocus_TIR",
        "weight_rel_dir": "weights/zero_shot_sota/ReFocus_TIR_SUTrack",
        "primary_weight_file": "OSTrack_ep0060.pth.tar",
        "weight_source": {
            "type": "gdrive_folder",
            "gdrive_folder_id": "1Q3NYwKwm-NBPKLxxApRHxNfyCgTbkOy2",
            "baidu_url": "https://pan.baidu.com/s/1beeHhL9wj7R8TMg3VdGvbg?pwd=v27s",
        },
        "adapter_type": "pytracking_ostrack_family",
        "tracker_module": "ostrack",
        "default_config": "vitb_256_mae_ce_32x4_ep300",
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
    """解析 `--trackers` 参数（支持 `all`, `group1`, `group2` 或逗号分隔列表）。"""
    sel = selection.strip()
    if sel.lower() == "all":
        return list(TRACKER_REGISTRY.keys())
    if sel.lower() in ("group1", "group1_in_domain", "in_domain"):
        return [k for k, v in TRACKER_REGISTRY.items() if v["group"] == "group1_in_domain"]
    if sel.lower() in ("group2", "group2_zero_shot", "zero_shot"):
        return [k for k, v in TRACKER_REGISTRY.items() if v["group"] == "group2_zero_shot"]

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
    raise ValueError(f"无效的子集 JSON 结构: {split_path}")


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
) -> Dict[str, Any]:
    """扫描 Google Drive 金库中各算法的源码、权重、Core 40 序列与评测遥测完成度。"""
    seqs, split_meta = load_split_sequences(split_json_path)
    split_name = split_meta.get("split_name", split_json_path.stem)
    val_dir = drive_root / "datasets/AntiUAV410/val"

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
            else:
                missing_val_seqs.append(s)
        else:
            missing_val_seqs.append(s)

    trackers_report: Dict[str, Any] = {}
    for tk in selected_trackers:
        spec = TRACKER_REGISTRY[tk]
        repo_dir = drive_root / spec["repo_rel_dir"]
        weight_dir = drive_root / spec["weight_rel_dir"]
        diag_telemetry_dir = drive_root / "experiments/failure_diagnosis/per_frame_telemetry" / f"{tk}_{split_name}"

        weight_files = []
        if weight_dir.exists():
            for wf in sorted(weight_dir.rglob("*")):
                if wf.is_file() and wf.suffix.lower() in (".pth", ".pt", ".tar", ".bin", ".safetensors", ".ckpt"):
                    weight_files.append({"name": wf.name, "bytes": wf.stat().st_size})

        completed_seqs = []
        if diag_telemetry_dir.exists():
            for s in seqs:
                jf = diag_telemetry_dir / f"{s}.jsonl"
                if jf.exists() and jf.stat().st_size > 0:
                    completed_seqs.append(s)

        trackers_report[tk] = {
            "group": spec["group"],
            "track_label": spec["track_label"],
            "env_tier": spec["env_tier"],
            "adapter_type": spec["adapter_type"],
            "repo_dir": str(repo_dir),
            "repo_cloned": (repo_dir / ".git").exists(),
            "weight_dir": str(weight_dir),
            "weight_files_ready_count": len(weight_files),
            "weight_files": weight_files,
            "telemetry_dir": str(diag_telemetry_dir),
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
        "dataset_labels_ready_sequences_count": len(labels_ready_seqs),
        "dataset_full_frames_ready_sequences_count": len(full_frames_ready_seqs),
        "dataset_full_frames_ready_sequences": full_frames_ready_seqs,
        "dataset_pending_frame_sync_sequences_count": len(missing_val_seqs),
        "sample_pending_sequences": missing_val_seqs[:5],
        "trackers_status": trackers_report,
    }


# ==============================================================================
# 功能 2: 自动拉取多算法源码与官方公开权重 (--action stage-repos-and-weights)
# ==============================================================================
def stage_repos_and_weights(
    drive_root: Path,
    runtime_root: Path,
    selected_trackers: List[str],
    sync_to_runtime: bool = True,
) -> Dict[str, Any]:
    """
    在 Colab (或本地联网环境) 自动克隆所选算法的官方 GitHub 仓库至 Google Drive `repos/`，
    并按注册表拉取对应的 Google Drive / HuggingFace / Meta 官方预训练权重至 `weights/`。
    已完成克隆或已存在的权重自动跳过（幂等安全）。
    """
    results: Dict[str, Any] = {}

    for tk in selected_trackers:
        spec = TRACKER_REGISTRY[tk]
        repo_drive = drive_root / spec["repo_rel_dir"]
        weight_drive = drive_root / spec["weight_rel_dir"]
        weight_drive.mkdir(parents=True, exist_ok=True)

        tk_res: Dict[str, Any] = {
            "repo_status": "SKIPPED_ALREADY_EXISTS",
            "weight_status": [],
            "runtime_repo": None,
        }

        # 1. 克隆源码仓库至 Google Drive 冷备区
        if not (repo_drive / ".git").exists():
            repo_drive.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{tk}] 正在克隆源码仓库 {spec['git_url']} -> {repo_drive} ...")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", spec["git_url"], str(repo_drive)],
                    check=True,
                    timeout=300,
                )
                if spec["pinned_commit"] != "HEAD":
                    subprocess.run(["git", "-C", str(repo_drive), "fetch", "--unshallow"], check=False)
                    subprocess.run(["git", "-C", str(repo_drive), "checkout", spec["pinned_commit"]], check=True)
                tk_res["repo_status"] = "CLONED_SUCCESS"
            except Exception as exc:
                tk_res["repo_status"] = f"CLONE_FAILED: {exc}"

        # 2. 若在 Colab 运行期，将源码同步至本地高速盘 /content/runtime/
        if sync_to_runtime and (repo_drive / ".git").exists():
            runtime_repo = runtime_root / spec["repo_rel_dir"]
            if not (runtime_repo / ".git").exists():
                runtime_repo.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(repo_drive, runtime_repo, dirs_exist_ok=True)
            tk_res["runtime_repo"] = str(runtime_repo)

        # 3. 拉取或核验官方模型权重
        wsrc = spec["weight_source"]
        wtype = wsrc["type"]

        try:
            if wtype == "gdrive_zip":
                target_f = weight_drive / spec["primary_weight_file"]
                if target_f.exists() and target_f.stat().st_size == spec.get("expected_bytes", target_f.stat().st_size):
                    tk_res["weight_status"].append(f"VERIFIED_READY: {target_f.name}")
                else:
                    import gdown
                    import zipfile
                    zpath = drive_root / wsrc["archive_rel_path"]
                    zpath.parent.mkdir(parents=True, exist_ok=True)
                    if not zpath.exists():
                        gdown.download(id=wsrc["gdrive_id"], output=str(zpath), resume=True, quiet=False)
                    with zipfile.ZipFile(zpath, "r") as zf:
                        for m in zf.infolist():
                            if m.filename.endswith(wsrc["member_suffix"]):
                                with zf.open(m) as src, target_f.open("wb") as dst:
                                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                    tk_res["weight_status"].append(f"EXTRACTED: {target_f.name}")

            elif wtype == "gdrive_folder":
                existing = [
                    p.name for p in weight_drive.rglob("*")
                    if p.is_file() and p.suffix.lower() in (".pth", ".pt", ".tar", ".bin", ".safetensors")
                ]
                if existing:
                    tk_res["weight_status"].append(f"EXISTING_WEIGHTS: {existing}")
                else:
                    import gdown
                    print(f"[{tk}] 正在通过 gdown.download_folder 拉取 Google Drive 权重目录 ({wsrc['gdrive_folder_id']})...")
                    gdown.download_folder(
                        id=wsrc["gdrive_folder_id"],
                        output=str(weight_drive),
                        quiet=False,
                        use_cookies=False,
                        remaining_ok=True,
                    )
                    downloaded = [p.name for p in weight_drive.rglob("*") if p.is_file()]
                    tk_res["weight_status"].append(f"GDRIVE_FOLDER_PULLED: {downloaded}")

            elif wtype == "huggingface":
                existing = [
                    p.name for p in weight_drive.rglob("*")
                    if p.is_file() and p.suffix.lower() in (".pth", ".pt", ".tar", ".bin", ".safetensors")
                ]
                if existing:
                    tk_res["weight_status"].append(f"EXISTING_HF_WEIGHTS: {existing}")
                else:
                    from huggingface_hub import snapshot_download
                    print(f"[{tk}] 正在从 HuggingFace ({wsrc['hf_repo_id']}) 拉取模型权重...")
                    snapshot_download(
                        repo_id=wsrc["hf_repo_id"],
                        local_dir=str(weight_drive),
                        allow_patterns=wsrc.get("hf_allow_patterns"),
                    )
                    downloaded = [p.name for p in weight_drive.rglob("*") if p.is_file()]
                    tk_res["weight_status"].append(f"HF_PULLED: {downloaded}")

            elif wtype == "direct_url":
                for fname, url in wsrc["urls"].items():
                    dst_p = weight_drive / fname
                    if dst_p.exists() and dst_p.stat().st_size > 10_000_000:
                        tk_res["weight_status"].append(f"READY: {fname}")
                    else:
                        print(f"[{tk}] 正在下载 {fname} <- {url} ...")
                        subprocess.run(
                            ["curl", "-L", "-C", "-", "-o", str(dst_p), url],
                            check=True,
                        )
                        tk_res["weight_status"].append(f"DOWNLOADED: {fname} ({dst_p.stat().st_size} bytes)")

            elif wtype == "repo_builtin_plus_sam2":
                builtin_mp = repo_drive / wsrc["repo_builtin_rel"]
                dst_mp = weight_drive / "mp.pth"
                if builtin_mp.exists() and not dst_mp.exists():
                    shutil.copy2(builtin_mp, dst_mp)
                    tk_res["weight_status"].append(f"COPIED_REPO_BUILTIN: {dst_mp.name}")
                elif dst_mp.exists():
                    tk_res["weight_status"].append(f"READY: {dst_mp.name}")
                sam2_dst = drive_root / wsrc["sam2_checkpoint"]
                sam2_dst.parent.mkdir(parents=True, exist_ok=True)
                if not (sam2_dst.exists() and sam2_dst.stat().st_size > 10_000_000):
                    subprocess.run(["curl", "-L", "-C", "-", "-o", str(sam2_dst), wsrc["sam2_url"]], check=True)
                tk_res["weight_status"].append(f"SAM2_BASE_READY: {sam2_dst.name}")

        except Exception as exc:
            tk_res["weight_status"].append(f"WEIGHT_PULL_WARNING: {exc}")

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
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    fsync_write_text(eval_root / "eval_progress_summary.json", json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return summary_payload


# ==============================================================================
# 功能 4: 混合并发调度器 (方案 B 多算法并发优先 + 方案 A 单算法多视频分片自动补位)
# ==============================================================================
def get_tracker_eval_dirs(
    drive_root: Path,
    tracker_name: str,
    split_stem: str,
    git_short_sha: str,
) -> Dict[str, Path]:
    """获取某算法在 Google Drive 上的标准评测与遥测输出目录字典。"""
    spec = TRACKER_REGISTRY[tracker_name]
    eval_root = drive_root / "experiments/benchmark_eval" / spec["group"] / f"{tracker_name}_{split_stem}_{git_short_sha}"
    return {
        "eval_root": eval_root,
        "txt_box_dir": eval_root / "pred_bboxes_txt",
        "uavtir_json_dir": eval_root / "pred_bboxes_uavtir_json",
        "telemetry_eval_dir": eval_root / "per_frame_telemetry",
        "telemetry_diag_dir": drive_root / "experiments/failure_diagnosis/per_frame_telemetry" / f"{tracker_name}_{split_stem}",
    }


def scan_realtime_pending_sequences(
    drive_root: Path,
    dataset_val_dir: Path,
    tracker_name: str,
    split_stem: str,
    target_sequences: List[str],
    git_short_sha: str = "phase2",
    expected_frames_map: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    实时扫描 `telemetry_eval_dir/*.jsonl` 与 `telemetry_diag_dir/*.jsonl` 及 `.lock` 文件，
    返回 `(completed_seqs, unfinished_seqs, unclaimed_unfinished_seqs)`。
    """
    dirs = get_tracker_eval_dirs(drive_root, tracker_name, split_stem, git_short_sha)
    tel_eval_dir = dirs["telemetry_eval_dir"]
    tel_diag_dir = dirs["telemetry_diag_dir"]

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

        # 检查是否有其他活跃分片 Worker 正在锁定推理该序列 (.{seq_id}.lock 且未超时 30 分钟)
        lock_eval = tel_eval_dir / f".{seq_id}.lock"
        lock_diag = tel_diag_dir / f".{seq_id}.lock"
        is_claimed = False
        for lf in (lock_eval, lock_diag):
            if lf.exists():
                try:
                    if (now_ts - lf.stat().st_mtime) < 1800.0:
                        is_claimed = True
                        break
                except OSError:
                    pass
        if not is_claimed:
            unclaimed_unfinished.append(seq_id)

    return completed, unfinished, unclaimed_unfinished


def finalize_tracker_eval_summary(
    drive_root: Path,
    tracker_name: str,
    split_stem: str,
    target_sequences: List[str],
    git_short_sha: str = "phase2",
) -> Dict[str, Any]:
    """
    当某算法的所有分片 Worker 完成后，统一扫描全部 `{seq_id}.jsonl` 汇总生成无冲突的 `eval_progress_summary.json`。
    """
    spec = TRACKER_REGISTRY[tracker_name]
    dirs = get_tracker_eval_dirs(drive_root, tracker_name, split_stem, git_short_sha)
    tel_eval_dir = dirs["telemetry_eval_dir"]
    tel_diag_dir = dirs["telemetry_diag_dir"]
    eval_root = dirs["eval_root"]

    seq_sa_scores: Dict[str, float] = {}
    for seq_id in target_sequences:
        jf = tel_eval_dir / f"{seq_id}.jsonl"
        if not jf.exists():
            jf = tel_diag_dir / f"{seq_id}.jsonl"
        if jf.exists() and jf.stat().st_size > 0:
            try:
                rows = [json.loads(ln) for ln in jf.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if rows:
                    seq_sa_scores[seq_id] = round(float(sum(r["sa_score"] for r in rows) / len(rows)), 6)
            except Exception:
                pass

    mean_sa = round(sum(seq_sa_scores.values()) / len(seq_sa_scores), 6) if seq_sa_scores else 0.0
    summary_payload = {
        "tracker_name": tracker_name,
        "tracker_group": spec["group"],
        "split_name": split_stem,
        "stage_git_sha": git_short_sha,
        "completed_sequences_count": len(seq_sa_scores),
        "total_target_sequences": len(target_sequences),
        "mean_sa_score": mean_sa,
        "per_sequence_sa": seq_sa_scores,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
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
            "realtime_monitor_source": "telemetry_eval_dir/*.jsonl + .{seq_id}.lock",
            "helper_steal_direction": "tail_reverse_order (pending_seqs[::-1] with atomic .lock & .jsonl double-check)",
        },
    }


def _build_worker_script_for_tracker(tracker_name: str, spec: Dict[str, Any]) -> str:
    """
    为指定跟踪器生成在对应虚拟环境 (Tier-1 或 Tier-2) 中运行的并发分片 Worker 脚本。
    支持：
    - 接收环境变量 `_P2_SHARD_IDX` (0..K-1)、`_P2_NUM_SHARDS` (K) 与 `_P2_DYNAMIC_HELPER` ("0" 或 "1")；
    - 对 `pending_seqs[shard_idx::num_shards]` 进行无冲突分片推理；
    - 若为动态派生的方案 A 补位 Worker (`_P2_DYNAMIC_HELPER="1"`)，从序列列表末尾反向抢占未完成序列；
    - 启动每一条序列前均再次检查 `{seq_id}.jsonl` 与 `.{seq_id}.lock` 原子锁，彻底杜绝多 Worker 重复跑同一序列！
    """
    return r'''
import os, sys, json, math, time
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
DEFAULT_CFG = os.environ["_P2_DEFAULT_CFG"]
TRACKER_MOD = os.environ.get("_P2_TRACKER_MOD", "")
SPLIT_JSON = Path(os.environ["_P2_SPLIT_JSON"])
DATASET_DIR = Path(os.environ["_P2_DATASET_DIR"])
GIT_SHA = os.environ.get("_P2_GIT_SHA", "phase2")
MAX_SEQS = int(os.environ.get("_P2_MAX_SEQS", "0"))

SHARD_IDX = int(os.environ.get("_P2_SHARD_IDX", "0"))
NUM_SHARDS = max(1, int(os.environ.get("_P2_NUM_SHARDS", "1")))
IS_DYNAMIC_HELPER = os.environ.get("_P2_DYNAMIC_HELPER", "0") == "1"
WORKER_TAG = os.environ.get("_P2_WORKER_TAG", f"shard{SHARD_IDX}of{NUM_SHARDS}")

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

eval_root = DRIVE_ROOT / "experiments/benchmark_eval" / TRACKER_GROUP / f"{TRACKER_NAME}_{split_stem}_{GIT_SHA}"
txt_box_dir = eval_root / "pred_bboxes_txt"
uavtir_json_dir = eval_root / "pred_bboxes_uavtir_json"
telemetry_eval_dir = eval_root / "per_frame_telemetry"
telemetry_diag_dir = DRIVE_ROOT / "experiments/failure_diagnosis/per_frame_telemetry" / f"{TRACKER_NAME}_{split_stem}"

for d in (txt_box_dir, uavtir_json_dir, telemetry_eval_dir, telemetry_diag_dir):
    d.mkdir(parents=True, exist_ok=True)

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
    """在启动序列推理前尝试获取 .{seq_id}.lock，防止多个分片或动态补位 Worker 重复跑同一条序列。"""
    if is_seq_already_done(seq_id):
        return False
    lock_file = telemetry_eval_dir / f".{seq_id}.lock"
    now_ts = time.time()
    if lock_file.exists():
        try:
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
    except FileExistsError:
        return False
    except OSError:
        # 若底层 FUSE 不支持 O_EXCL，退化为读写校验
        try:
            lock_file.write_text(payload, encoding="utf-8")
            return True
        except OSError:
            return False

def release_seq_lock(seq_id: str):
    lock_file = telemetry_eval_dir / f".{seq_id}.lock"
    try:
        if lock_file.exists():
            lock_file.unlink()
    except OSError:
        pass

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
    worker_seq_queue = primary_queue + fallback_tail
else:
    # 常规分片 Worker: 先执行固定分片 pending_seqs[shard_idx::num_shards]，若提前跑完再从末尾反向协助剩余未认领序列
    primary_queue = pending_seqs[SHARD_IDX::NUM_SHARDS]
    fallback_tail = [s for s in reversed(pending_seqs) if s not in primary_queue] if NUM_SHARDS > 1 else []
    worker_seq_queue = primary_queue + fallback_tail

print(
    f"[{TRACKER_NAME} | {WORKER_TAG}] 启动分片推理 | primary_slice={len(primary_queue)} 条 "
    f"| total_pending={len(pending_seqs)} 条 | dynamic_helper={IS_DYNAMIC_HELPER}"
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
            import init_paths
            import libs.data as data
            import libs.ops as ops
            from trackers import SiamDTTracker
            self.ops = ops
            ckpt_p = WEIGHT_DIR / PRIMARY_WEIGHT
            if not ckpt_p.exists():
                ckpt_p = siamdt_dir / "checkpoints" / PRIMARY_WEIGHT
            transforms = data.BasicPairTransforms(train=False)
            self.tracker = SiamDTTracker(DEFAULT_CFG, str(ckpt_p), transforms, name_suffix=f"{TRACKER_NAME}_{WORKER_TAG}", visualize=False)

        elif self.adapter == "pytracking_ostrack_family":
            import cv2
            self.cv2 = cv2
            from importlib import import_module
            param_mod = import_module(f"lib.test.parameter.{TRACKER_MOD}")
            tracker_mod = import_module(f"lib.test.tracker.{TRACKER_MOD}")
            params = param_mod.parameters(DEFAULT_CFG)
            ckpts = sorted([
                p for p in WEIGHT_DIR.rglob("*")
                if p.is_file() and p.suffix.lower() in (".tar", ".pth", ".pt")
            ])
            if ckpts:
                params.checkpoint = str(ckpts[0])
            tracker_cls = tracker_mod.get_tracker_class()
            self.tracker = tracker_cls(params, "antiuav410")

        elif self.adapter == "sam2_dam4sam":
            import cv2
            from PIL import Image
            self.cv2 = cv2
            self.Image = Image
            from dam4sam_tracker import DAM4SAMTracker
            self.tracker_cls = DAM4SAMTracker

        elif self.adapter == "sam2_samosa_samurai":
            import cv2
            self.cv2 = cv2
            sys.path.insert(0, str(REPO_DIR / "sam2"))
            from sam2.build_sam import build_sam2_video_predictor
            self.build_predictor = build_sam2_video_predictor

        elif self.adapter == "trackit_memlotrack":
            import cv2
            self.cv2 = cv2
            self.weight_files = sorted([p for p in WEIGHT_DIR.rglob("*.bin")] + [p for p in WEIGHT_DIR.rglob("*.safetensors")])

    def run_sequence(self, seq_dir: Path, frames: list, gt0: list):
        pred_boxes = [gt0]
        confs = [1.0]
        up_flags = [True]
        lats_ms = [0.0]

        if self.adapter == "siamdt_native":
            init_xyxy = np.array([gt0[0], gt0[1], gt0[0] + gt0[2] - 1.0, gt0[1] + gt0[3] - 1.0], dtype=np.float32)
            img0 = self.ops.read_image(str(frames[0]), self.tracker.color_fmt)
            t0 = time.perf_counter()
            self.tracker.init(img0, init_xyxy)
            lats_ms[0] = (time.perf_counter() - t0) * 1000.0
            for fpath in frames[1:]:
                img = self.ops.read_image(str(fpath), self.tracker.color_fmt)
                ts = time.perf_counter()
                results, up_flag = self.tracker.update(img, return_all=True)
                dt = (time.perf_counter() - ts) * 1000.0
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
            return pred_boxes, confs, up_flags, lats_ms

        elif self.adapter == "pytracking_ostrack_family":
            img0 = self.cv2.cvtColor(self.cv2.imread(str(frames[0])), self.cv2.COLOR_BGR2RGB)
            if TRACKER_NAME in ("SUTrack", "UETrack", "UTPTrack-S"):
                img0_in = np.concatenate([img0, img0], axis=2)
            else:
                img0_in = img0
            t0 = time.perf_counter()
            try:
                self.tracker.initialize(img0_in, {"init_bbox": gt0})
            except Exception:
                self.tracker.initialize(img0, {"init_bbox": gt0})
                img0_in = None
            lats_ms[0] = (time.perf_counter() - t0) * 1000.0

            for fpath in frames[1:]:
                img = self.cv2.cvtColor(self.cv2.imread(str(fpath)), self.cv2.COLOR_BGR2RGB)
                img_in = np.concatenate([img, img], axis=2) if img0_in is not None else img
                ts = time.perf_counter()
                out = self.tracker.track(img_in)
                dt = (time.perf_counter() - ts) * 1000.0
                tb = out.get("target_bbox", [0.0, 0.0, 0.0, 0.0])
                conf = float(out.get("best_score", out.get("conf_score", 1.0)))
                pred_boxes.append([round(float(x), 4) for x in tb[:4]])
                confs.append(conf)
                up_flags.append(False)
                lats_ms.append(dt)
            return pred_boxes, confs, up_flags, lats_ms

        elif self.adapter == "sam2_dam4sam":
            tracker = self.tracker_cls(DEFAULT_CFG)
            img0 = self.Image.open(str(frames[0]))
            t0 = time.perf_counter()
            tracker.initialize(img0, None, bbox=gt0)
            lats_ms[0] = (time.perf_counter() - t0) * 1000.0
            for fpath in frames[1:]:
                img = self.Image.open(str(fpath))
                ts = time.perf_counter()
                out = tracker.track(img)
                dt = (time.perf_counter() - ts) * 1000.0
                mask = out.get("pred_mask")
                if mask is not None and np.any(mask > 0):
                    ys, xs = np.where(mask > 0)
                    x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                    pb = [x0, y0, max(1.0, x1 - x0 + 1.0), max(1.0, y1 - y0 + 1.0)]
                else:
                    pb = [0.0, 0.0, 0.0, 0.0]
                pred_boxes.append(pb)
                confs.append(float(out.get("iou_score", 1.0)))
                up_flags.append(bool(out.get("memory_updated", False)))
                lats_ms.append(dt)
            return pred_boxes, confs, up_flags, lats_ms

        else:
            raise RuntimeError(f"Adapter {self.adapter} 需要通过专用批量入口或 --ingest-raw-txt-dir 执行。")

stepper = None

for seq_id in worker_seq_queue:
    # 启动前再次双重检查 .jsonl 与 .lock，防止与其他分片或动态补位 Worker 冲突
    if not try_acquire_seq_lock(seq_id):
        continue

    try:
        seq_dir = DATASET_DIR / seq_id
        lbl_file = seq_dir / "IR_label.json"
        if not lbl_file.exists():
            print(f"[{TRACKER_NAME} | {WORKER_TAG}] 序列 {seq_id} 缺少 IR_label.json，跳过。")
            continue
        lbl = json.loads(lbl_file.read_text(encoding="utf-8"))
        gt_rects = lbl["gt_rect"]
        exist_flags = lbl.get("exist", [1] * len(gt_rects))
        frames = sorted(seq_dir.glob("*.jpg"))
        total_f = min(len(frames), len(gt_rects))
        if total_f < len(gt_rects) or total_f == 0:
            print(f"[{TRACKER_NAME} | {WORKER_TAG}] 序列 {seq_id} JPG 帧数 ({len(frames)}) 未齐备 (期望 {len(gt_rects)})，跳过。")
            continue

        if stepper is None:
            stepper = UnifiedTrackerStepper()

        gt0 = [float(x) for x in gt_rects[0][:4]]
        pred_boxes, confs, up_flags, lats_ms = stepper.run_sequence(seq_dir, frames[:total_f], gt0)

        telemetry_rows = []
        for f in range(total_f):
            pb = pred_boxes[f]
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

            telemetry_rows.append({
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
            })

        txt_content = "\n".join([f"{b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f}" for b in pred_boxes]) + "\n"
        uavtir_json_content = json.dumps({"res": pred_boxes})
        jsonl_content = "\n".join([json.dumps(r, ensure_ascii=False) for r in telemetry_rows]) + "\n"

        fsync_write(txt_box_dir / f"{seq_id}.txt", txt_content)
        fsync_write(uavtir_json_dir / f"{seq_id}.txt", uavtir_json_content)
        fsync_write(telemetry_eval_dir / f"{seq_id}.jsonl", jsonl_content)
        fsync_write(telemetry_diag_dir / f"{seq_id}.jsonl", jsonl_content)

        seq_sa = float(np.mean([r["sa_score"] for r in telemetry_rows]))
        print(f"[{TRACKER_NAME} | {WORKER_TAG}] 序列 {seq_id} ({total_f} 帧) 落盘完成 | SA={seq_sa:.4f}")
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
) -> Tuple[subprocess.Popen, Dict[str, Any]]:
    """启动单个算法分片子进程并返回 `(Popen, job_meta)`。"""
    spec = TRACKER_REGISTRY[tracker_name]
    py_bin = spec["python_bin_colab"] if Path(spec["python_bin_colab"]).exists() else sys.executable
    repo_dir = (
        (runtime_root / spec["repo_rel_dir"])
        if (runtime_root / spec["repo_rel_dir"]).exists()
        else (drive_root / spec["repo_rel_dir"])
    )
    weight_dir = drive_root / spec["weight_rel_dir"]
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
        "_P2_DEFAULT_CFG": spec["default_config"],
        "_P2_TRACKER_MOD": spec.get("tracker_module", ""),
        "_P2_SPLIT_JSON": str(split_json_path),
        "_P2_DATASET_DIR": str(dataset_val_dir),
        "_P2_GIT_SHA": git_short_sha,
        "_P2_MAX_SEQS": str(max_sequences),
        "_P2_SHARD_IDX": str(shard_idx),
        "_P2_NUM_SHARDS": str(num_shards),
        "_P2_DYNAMIC_HELPER": "1" if dynamic_helper else "0",
        "_P2_WORKER_TAG": worker_tag,
    })

    proc = subprocess.Popen([py_bin, "-c", worker_code], env=env)
    meta = {
        "slot_id": slot_id,
        "tracker_name": tracker_name,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "dynamic_helper": dynamic_helper,
        "worker_tag": worker_tag,
        "pid": proc.pid,
        "started_at": time.time(),
    }
    return proc, meta


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
) -> Dict[str, Any]:
    """
    在 Colab A100/L4 上执行混合并发统一评测 (`hybrid_b_then_a`):
      1. 自动探测或使用 `--gpu-slots` 确定总算力槽位数 S；
      2. 实时扫描各算法未完成序列，生成方案 B (多算法并发) + 方案 A (多视频分片补位) 混合调度计划；
      3. 运行期通过动态进程池监控：一旦快算法跑完腾出空闲槽位，而慢算法仍有 > 4 条未完成序列，
         立即在空闲槽位派生方案 A 动态补位 Worker；
      4. 所有分片完成后统一汇总生成各算法的 `eval_progress_summary.json`。
    """
    if concurrency_mode not in CONCURRENCY_MODES:
        raise ValueError(f"不支持的 concurrency_mode: {concurrency_mode}，可选: {CONCURRENCY_MODES}")

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
    dataset_val_dir = drive_root / "datasets/AntiUAV410/val"

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
        )
        tracker_completed_map[tk] = comp
        tracker_pending_map[tk] = unfin

    # 生成初始并发波次与混合补位计划表
    schedule_plan = build_hybrid_schedule_plan(
        tracker_pending_map=tracker_pending_map,
        gpu_slots=S,
        concurrency_mode=concurrency_mode,
    )

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
            "split_name": split_stem,
            "target_sequences_count": len(target_seqs),
            "gpu_slot_detection": gpu_slot_info,
            "selected_trackers_count": len(selected_trackers),
            "per_tracker_pending_counts": {tk: len(tracker_pending_map[tk]) for tk in selected_trackers},
            "hybrid_schedule_plan": schedule_plan,
            "plan_a_backfill_when_M_lt_S_preview": plan_a_backfill_demo,
        }

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
            drive_root=drive_root,
            runtime_root=runtime_root,
            split_json_path=split_json_path,
            dataset_val_dir=dataset_val_dir,
            max_sequences=max_sequences,
            git_short_sha=git_short_sha,
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
            continue

        # 优先阶段 1：若等待队列 queued_trackers 中仍有未启动算法，优先按方案 B（或尾盘分片）填补空闲槽位
        while freed_slots and queued_trackers:
            next_tk = queued_trackers.pop(0)
            _, unfin, _ = scan_realtime_pending_sequences(
                drive_root=drive_root,
                dataset_val_dir=dataset_val_dir,
                tracker_name=next_tk,
                split_stem=split_stem,
                target_sequences=target_seqs,
                git_short_sha=git_short_sha,
                expected_frames_map=expected_frames_map,
            )
            if not unfin:
                continue

            if concurrency_mode == "plan_a_seq_shards":
                # plan_a 模式下，当前算法独占所有空闲槽位分片运行
                k_shards = min(len(freed_slots), len(unfin))
                for s_idx in range(k_shards):
                    s_slot = freed_slots.pop(0)
                    p, m = _spawn_tracker_shard_worker(
                        tracker_name=next_tk,
                        shard_idx=s_idx,
                        num_shards=k_shards,
                        dynamic_helper=False,
                        slot_id=s_slot,
                        drive_root=drive_root,
                        runtime_root=runtime_root,
                        split_json_path=split_json_path,
                        dataset_val_dir=dataset_val_dir,
                        max_sequences=max_sequences,
                        git_short_sha=git_short_sha,
                    )
                    active_procs.append((p, m))
                    occupied_slots[s_slot] = next_tk
            else:
                # 方案 B：启动下一个待测算法
                s_slot = freed_slots.pop(0)
                p, m = _spawn_tracker_shard_worker(
                    tracker_name=next_tk,
                    shard_idx=0,
                    num_shards=1,
                    dynamic_helper=False,
                    slot_id=s_slot,
                    drive_root=drive_root,
                    runtime_root=runtime_root,
                    split_json_path=split_json_path,
                    dataset_val_dir=dataset_val_dir,
                    max_sequences=max_sequences,
                    git_short_sha=git_short_sha,
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
                    _, unfin, unclaimed = scan_realtime_pending_sequences(
                        drive_root=drive_root,
                        dataset_val_dir=dataset_val_dir,
                        tracker_name=rtk,
                        split_stem=split_stem,
                        target_sequences=target_seqs,
                        git_short_sha=git_short_sha,
                        expected_frames_map=expected_frames_map,
                    )
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
                    drive_root=drive_root,
                    runtime_root=runtime_root,
                    split_json_path=split_json_path,
                    dataset_val_dir=dataset_val_dir,
                    max_sequences=max_sequences,
                    git_short_sha=git_short_sha,
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
        )

    return {
        "action": "eval",
        "concurrency_mode": concurrency_mode,
        "gpu_slot_detection": gpu_slot_info,
        "initial_schedule_plan": schedule_plan,
        "dynamic_plan_a_backfill_events": dynamic_backfill_events,
        "completed_workers_count": len(completed_worker_logs),
        "completed_workers": completed_worker_logs,
        "tracker_final_summaries": final_summaries,
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
        help="目标算法: all | group1 | group2 | 或逗号分隔的算法名 (如 SiamDT_Baseline,FocusTrack,MemLoTrack,DAM4SAM)",
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
        help="仅验证混合并发调度计划表与输出路径，不实际启动 GPU 推理进程",
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
        rep = inspect_phase2_status(drive_root, split_json_path, selected_trackers)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return

    if args.action == "stage-repos-and-weights":
        rep = stage_repos_and_weights(drive_root, args.runtime_root, selected_trackers)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return

    if args.action == "ingest-txt":
        if len(selected_trackers) != 1 or not args.ingest_raw_txt_dir:
            raise ValueError("使用 --action ingest-txt 时必须指定单个 --trackers <Name> 及 --ingest-raw-txt-dir <Path>")
        rep = ingest_raw_predictions_to_telemetry(
            tracker_name=selected_trackers[0],
            raw_pred_dir=args.ingest_raw_txt_dir,
            split_json_path=split_json_path,
            dataset_val_dir=drive_root / "datasets/AntiUAV410/val",
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
