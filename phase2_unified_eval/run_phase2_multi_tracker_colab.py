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
1. 权重与源码持久化金库：自动将源码拉取至 Google Drive `repos/core_pool/`，权重拉取至 `weights/`，
   并在运行期同步至 Colab 本地高速盘 `/content/runtime/`；
2. 逐序列断点续评 (Auto-Resume)：每跑完 1 条序列立即将 `.txt` 预测框、官方兼容 JSON 与统一 `.jsonl`
   逐帧遥测实时 fsync 刷入 Google Drive，断线重连自动跳过已完成序列；
3. 统一遥测转换与三层失败提取：内置原生日式推理适配器与通用预测框摄入器 (`--ingest-raw-txt-dir`)，
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
# 通用工具函数
# ==============================================================================
def resolve_default_drive_root() -> Path:
    """自动识别 Colab `/content/drive/MyDrive/SiamDT_Project` 或本地 `G:\\我的云端硬盘\\SiamDT_Project`。"""
    env_val = os.environ.get("SIAMDT_DRIVE_ROOT")
    if env_val:
        return Path(env_val)
    colab_path = Path("/content/drive/MyDrive/SiamDT_Project")
    if colab_path.exists() or Path("/content/drive").exists():
        return colab_path
    win_gdrive = Path("G:/我的云端硬盘/SiamDT_Project")
    return win_gdrive


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
    # 支持大小写不敏感匹配
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
                # 1. 复制仓库内置的 mp.pth (非线性运动预测器权重)
                builtin_mp = repo_drive / wsrc["repo_builtin_rel"]
                dst_mp = weight_drive / "mp.pth"
                if builtin_mp.exists() and not dst_mp.exists():
                    shutil.copy2(builtin_mp, dst_mp)
                    tk_res["weight_status"].append(f"COPIED_REPO_BUILTIN: {dst_mp.name}")
                elif dst_mp.exists():
                    tk_res["weight_status"].append(f"READY: {dst_mp.name}")
                # 2. 确保 SAM 2.1 基座权重就绪
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

        # 查找预测框文件 (<seq_id>.txt 或 <seq_id>.json)
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

        # 读取可选置信度与耗时文件 (如 <seq_id>_score.txt / <seq_id>_time.txt)
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
# 功能 4: Colab A100/L4 统一序列推理与遥测落盘调度器 (--action eval)
# ==============================================================================
def _build_worker_script_for_tracker(tracker_name: str, spec: Dict[str, Any]) -> str:
    """
    为指定跟踪器生成在对应虚拟环境 (Tier-1 或 Tier-2) 中运行的单序列/批序列统一推理 Python 脚本。
    所有适配器严格遵循：
    - 仅在 frame_id == 0 读取 gt_rect[0] 初始化；
    - 逐帧输出 pred_bbox [x, y, w, h]、confidence、template_updated、latency_ms；
    - 每完成 1 条序列立即 fsync 写入 Google Drive，支持断点续评。
    """
    return r'''
import os, sys, json, math, time
import numpy as np
from pathlib import Path

DRIVE_ROOT = Path(os.environ[" _P2_DRIVE_ROOT ".strip()])
RUNTIME_ROOT = Path(os.environ[" _P2_RUNTIME_ROOT ".strip()])
TRACKER_NAME = os.environ[" _P2_TRACKER_NAME ".strip()]
TRACKER_GROUP = os.environ[" _P2_TRACKER_GROUP ".strip()]
ADAPTER_TYPE = os.environ[" _P2_ADAPTER_TYPE ".strip()]
REPO_DIR = Path(os.environ[" _P2_REPO_DIR ".strip()])
WEIGHT_DIR = Path(os.environ[" _P2_WEIGHT_DIR ".strip()])
PRIMARY_WEIGHT = os.environ[" _P2_PRIMARY_WEIGHT ".strip()]
DEFAULT_CFG = os.environ[" _P2_DEFAULT_CFG ".strip()]
TRACKER_MOD = os.environ.get(" _P2_TRACKER_MOD ".strip(), "")
SPLIT_JSON = Path(os.environ[" _P2_SPLIT_JSON ".strip()])
DATASET_DIR = Path(os.environ[" _P2_DATASET_DIR ".strip()])
GIT_SHA = os.environ.get(" _P2_GIT_SHA ".strip(), "phase2")
MAX_SEQS = int(os.environ.get(" _P2_MAX_SEQS ".strip(), "0"))

raw_split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
split_stem = raw_split.get("split_name", SPLIT_JSON.stem)
seq_list = [str(x) for x in raw_split["sequences"]]
if MAX_SEQS > 0:
    seq_list = seq_list[:MAX_SEQS]

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

# 1. 扫描已完成序列（支持断线重连无缝跳过）
completed_seqs = []
pending_seqs = []
seq_sa_scores = {}

for seq_id in seq_list:
    lbl_file = DATASET_DIR / seq_id / "IR_label.json"
    jsonl_file = telemetry_diag_dir / f"{seq_id}.jsonl"
    txt_file = txt_box_dir / f"{seq_id}.txt"
    if lbl_file.exists() and jsonl_file.exists() and txt_file.exists():
        try:
            gt_len = len(json.loads(lbl_file.read_text(encoding="utf-8"))["gt_rect"])
            rows = [json.loads(ln) for ln in jsonl_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if gt_len > 0 and len(rows) == gt_len:
                completed_seqs.append(seq_id)
                seq_sa_scores[seq_id] = float(np.mean([r["sa_score"] for r in rows]))
                continue
        except Exception:
            pass
    pending_seqs.append(seq_id)

print(f"[{TRACKER_NAME} | {split_stem}] 已完成序列: {len(completed_seqs)}/{len(seq_list)} | 待评测序列: {len(pending_seqs)}")
if not pending_seqs:
    sys.exit(0)

# 2. 构建统一单帧步进跟踪器接口 (init(img_path, gt_xywh) -> step(img_path) -> (pred_xywh, conf, up_flag))
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
            self.tracker = SiamDTTracker(DEFAULT_CFG, str(ckpt_p), transforms, name_suffix=TRACKER_NAME, visualize=False)

        elif self.adapter == "pytracking_ostrack_family":
            import cv2
            self.cv2 = cv2
            from importlib import import_module
            param_mod = import_module(f"lib.test.parameter.{TRACKER_MOD}")
            tracker_mod = import_module(f"lib.test.tracker.{TRACKER_MOD}")
            params = param_mod.parameters(DEFAULT_CFG)
            # 自动绑定 weight_dir 下找到的权重文件
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
            # MemLoTrack 支持通过 trackit 模型或批量推理导入
            self.weight_files = sorted([p for p in WEIGHT_DIR.rglob("*.bin")] + [p for p in WEIGHT_DIR.rglob("*.safetensors")])

    def run_sequence(self, seq_dir: Path, frames: list, gt0: list):
        """对单条序列执行从第0帧初始化到末帧的完整 OPE 推理，返回 (pred_boxes, confs, up_flags, lats_ms)。"""
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
            # 若为多模态 6 通道输入 (SUTrack / UETrack / UTPTrack-S)，自动拼接双路三通道红外帧 [RGB, TIR]
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

stepper = UnifiedTrackerStepper()

for idx, seq_id in enumerate(pending_seqs, start=1):
    seq_dir = DATASET_DIR / seq_id
    lbl_file = seq_dir / "IR_label.json"
    if not lbl_file.exists():
        print(f"[警告] 序列 {seq_id} 尚未同步至 {seq_dir}，跳过。")
        continue
    lbl = json.loads(lbl_file.read_text(encoding="utf-8"))
    gt_rects = lbl["gt_rect"]
    exist_flags = lbl.get("exist", [1] * len(gt_rects))
    frames = sorted(seq_dir.glob("*.jpg"))
    total_f = min(len(frames), len(gt_rects))
    if total_f == 0:
        continue

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
    seq_sa_scores[seq_id] = seq_sa
    print(f"[{TRACKER_NAME} | {idx}/{len(pending_seqs)}] 序列 {seq_id} ({total_f} 帧) 落盘完成 | SA={seq_sa:.4f}")

    summary_payload = {
        "tracker_name": TRACKER_NAME,
        "tracker_group": TRACKER_GROUP,
        "split_name": split_stem,
        "stage_git_sha": GIT_SHA,
        "completed_sequences_count": len(seq_sa_scores),
        "total_target_sequences": len(seq_list),
        "mean_sa_score": round(float(np.mean(list(seq_sa_scores.values()))), 6) if seq_sa_scores else 0.0,
        "per_sequence_sa": seq_sa_scores,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    fsync_write(eval_root / "eval_progress_summary.json", json.dumps(summary_payload, indent=2, ensure_ascii=False))
'''


def execute_multi_tracker_eval(
    drive_root: Path,
    runtime_root: Path,
    split_json_path: Path,
    selected_trackers: List[str],
    max_sequences: int = 0,
    git_short_sha: str = "phase2",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """在 Colab A100/L4 上按顺序调度所选跟踪器执行统一评测并实时写入 `.jsonl` 遥测。"""
    dataset_val_dir = drive_root / "datasets/AntiUAV410/val"
    core40_tar = drive_root / "datasets/archives/AntiUAV410_Core40_val.tar"
    runtime_val_dir = runtime_root / "datasets/AntiUAV410/val"
    if not dry_run and core40_tar.exists() and str(runtime_root).startswith("/content"):
        marker = runtime_val_dir / ".core40_extracted"
        if not marker.exists():
            print(f"[NVMe 极速预热] 正在将 {core40_tar.name} 解压至本地高速盘 {runtime_val_dir} ...")
            runtime_val_dir.mkdir(parents=True, exist_ok=True)
            import tarfile
            with tarfile.open(core40_tar, "r") as tf:
                tf.extractall(path=runtime_val_dir)
            marker.write_text("ok", encoding="utf-8")
        dataset_val_dir = runtime_val_dir
    exec_logs: Dict[str, Any] = {}

    for tk in selected_trackers:
        spec = TRACKER_REGISTRY[tk]
        py_bin = spec["python_bin_colab"] if Path(spec["python_bin_colab"]).exists() else sys.executable
        repo_dir = (runtime_root / spec["repo_rel_dir"]) if (runtime_root / spec["repo_rel_dir"]).exists() else (drive_root / spec["repo_rel_dir"])
        weight_dir = drive_root / spec["weight_rel_dir"]

        if dry_run:
            exec_logs[tk] = {
                "mode": "DRY_RUN",
                "python_bin": py_bin,
                "repo_dir": str(repo_dir),
                "weight_dir": str(weight_dir),
                "split_json": str(split_json_path),
            }
            continue

        worker_code = _build_worker_script_for_tracker(tk, spec)
        env = os.environ.copy()
        env.update({
            "_P2_DRIVE_ROOT": str(drive_root),
            "_P2_RUNTIME_ROOT": str(runtime_root),
            "_P2_TRACKER_NAME": tk,
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
        })

        print(f"\n>>> 启动 [{tk}] ({spec['track_label']}) 统一评测 | Split={split_json_path.name} <<<")
        t0 = time.time()
        ret = subprocess.run([py_bin, "-c", worker_code], env=env, check=False)
        elapsed = round(time.time() - t0, 2)
        exec_logs[tk] = {
            "returncode": ret.returncode,
            "elapsed_sec": elapsed,
        }

    return exec_logs


# ==============================================================================
# CLI 主入口
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 Colab A100/L4 多算法统一评测与权重拉取调度器"
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
        "--drive-root",
        type=Path,
        default=None,
        help="Google Drive 持久化根目录 (默认自动检测 /content/drive/MyDrive/SiamDT_Project 或 G:/我的云端硬盘/SiamDT_Project)",
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
        help="仅验证调度配置与输出路径，不实际启动 GPU 推理",
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
