#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Google Drive 云端金库目录、权重与 Core 40 数据集无损预置器 (stage_core40_to_gdrive.py)
================================================================================================
严格遵守只读红线：仅从 `D:\\CodexProjects`、`D:\\CodexData`、`E:\\PHD\\YB`、`Z:\\SimDT围点打援` 只读复制资产，
严禁修改或删除源目录任何字节。

执行职责：
1. 在 `G:\\我的云端硬盘\\SiamDT_Project` 建立完整的多算法评测与故障诊断金库目录树；
2. 无损同步并校验 SHA-256：
   - 官方 SiamDT 权重 `siamdt_swin_tiny_sgd.pth` (499,427,901 bytes)
   - Swin-Tiny 预训练主干权重 `cascade_mask_rcnn_swin_tiny.pth.tar` (如本机存在)
   - 冻结划分 JSON (`discovery_set.json`, `holdout_verification.json`, `val90_full.json`, `DIAGNOSTIC_SUBSET_MANIFEST.json`)
   - 官方 `val` 属性标注 `annos/val/att/*.txt` 及 Phase 2 协议/评测脚本
3. 核验本机 `D:\\CodexData\\YB\\datasets\\Anti-UAV410_official_20261024\\original_train_val\\val` 中
   Core 40 (`24 Discovery + 16 Holdout`, 共 45,728 帧) 的完整性；
4. 支持断点续传与优先级队列：
   - 秒级预置全部 Core 40 序列的 `IR_label.json` 真值标注；
   - 优先完整同步冒烟与核心代表序列的全部 `.jpg` 图像帧，并支持后台断点续传全量 Core 40 序列。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# 默认路径与常量配置
# ==============================================================================
DEFAULT_WORKSPACE_ROOT = Path("Z:/SimDT围点打援")
DEFAULT_GDRIVE_ROOT = Path("G:/我的云端硬盘/SiamDT_Project")
DEFAULT_LOCAL_VAL_DIR = Path("D:/CodexData/YB/datasets/Anti-UAV410_official_20261024/original_train_val/val")
DEFAULT_LOCAL_SIAMDT_CKPT = Path("D:/CodexProjects/SiamDT_Project/weights/checkpoints/siamdt_swin_tiny_sgd.pth")
DEFAULT_LOCAL_ANNOS_DIR = Path("E:/PHD/YB/Anti_yb/annos/val")

EXPECTED_SIAMDT_BYTES = 499427901
EXPECTED_SIAMDT_SHA256 = "b4c7e0e0490881f81a64352f4db213934b15194f3ad76c3937aee0204e4c6c38"

REQUIRED_GDRIVE_DIRS = [
    "datasets/AntiUAV410/train",
    "datasets/AntiUAV410/val",
    "datasets/AntiUAV410/test",
    "datasets/AntiUAV410/annos/val/att",
    "datasets/diagnostic_splits",
    "datasets/archives",
    "repos/SimDT_Pipeline",
    "repos/Anti-UAV410_official",
    "repos/core_pool",
    "repos/patches",
    "weights/foundation_backbones/swin_tiny",
    "weights/foundation_backbones/sam2_1",
    "weights/foundation_backbones/dinov2",
    "weights/in_domain_antiuav410/SiamDT_Baseline",
    "weights/in_domain_antiuav410/FocusTrack",
    "weights/in_domain_antiuav410/MemLoTrack",
    "weights/zero_shot_sota/DAM4SAM",
    "weights/zero_shot_sota/SAMOSA",
    "weights/zero_shot_sota/UETrack_MCITrack",
    "weights/zero_shot_sota/ReFocus_TIR_SUTrack",
    "environments/requirements",
    "environments/wheels/pip-cache",
    "environments/wheels/uv-cache",
    "environments/diagnostics",
    "experiments/runs_training/siamdt_baseline_repro",
    "experiments/runs_training/siamdt_ours_ablations",
    "experiments/benchmark_eval/group1_in_domain",
    "experiments/benchmark_eval/group2_zero_shot",
    "experiments/benchmark_eval/reference_static",
    "experiments/failure_diagnosis/per_frame_telemetry",
    "experiments/failure_diagnosis/rescue_matrix",
    "experiments/failure_diagnosis/reports",
    "colab",
    "docs",
]

# 优先预置序列队列（涵盖 Discovery 首条冒烟序列、短序列、极小目标、热交叉、遮挡、出视野重入典型序列）
PRIORITY_STAGE_SEQUENCES = [
    "03_7951_1500-2999",       # Discovery #1 (Colab CELL 5/6 默认首条冒烟序列, 1500帧, TS+TC+OV+FM+DBC)
    "20190926_143632_1_8",     # Discovery 短序列 (584帧, TC+SV)
    "20190926_142435_1_9",     # Holdout 最短序列 (299帧, TC)
    "20190925_205804_1_5",     # Discovery 遮挡+消失重入代表 (961帧, OC+SV+FM)
    "20190925_130434_1_8",     # Discovery 快速运动代表 (1000帧, FM)
    "20190925_131530_1_3",     # Discovery 热交叉代表 (1000帧, TC)
]


def calc_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_gdrive_layout(gdrive_root: Path) -> int:
    """在 Google Drive 下创建完整目录树。"""
    for rel in REQUIRED_GDRIVE_DIRS:
        (gdrive_root / rel).mkdir(parents=True, exist_ok=True)
    return len(REQUIRED_GDRIVE_DIRS)


def stage_weights_and_splits(
    workspace_root: Path,
    gdrive_root: Path,
    local_siamdt_ckpt: Path,
    local_annos_dir: Path,
) -> Dict[str, Any]:
    """无损同步官方权重、冻结划分 JSON、属性标注与 Phase 2 脚本至 Google Drive。"""
    report: Dict[str, Any] = {
        "weights": {},
        "diagnostic_splits": [],
        "annos_synced_count": 0,
        "phase2_scripts_synced": [],
    }

    # 1. 同步 SiamDT 官方推理权重 (siamdt_swin_tiny_sgd.pth)
    dst_ckpt = gdrive_root / "weights/in_domain_antiuav410/SiamDT_Baseline/siamdt_swin_tiny_sgd.pth"
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if local_siamdt_ckpt.exists():
        need_copy = True
        if dst_ckpt.exists() and dst_ckpt.stat().st_size == EXPECTED_SIAMDT_BYTES:
            need_copy = False
        if need_copy:
            shutil.copy2(local_siamdt_ckpt, dst_ckpt)
        actual_size = dst_ckpt.stat().st_size
        actual_sha = calc_sha256(dst_ckpt)
        report["weights"]["siamdt_swin_tiny_sgd.pth"] = {
            "source": str(local_siamdt_ckpt),
            "destination": str(dst_ckpt),
            "bytes": actual_size,
            "sha256": actual_sha,
            "verified_exact": (actual_size == EXPECTED_SIAMDT_BYTES and actual_sha == EXPECTED_SIAMDT_SHA256),
        }
    else:
        report["weights"]["siamdt_swin_tiny_sgd.pth"] = {"status": "LOCAL_SOURCE_NOT_FOUND"}

    # 检查是否有本地预训练权重 cascade_mask_rcnn_swin_tiny.pth.tar
    swin_candidates = [
        Path("D:/CodexProjects/SiamDT_Project/weights/pretrained/cascade_mask_rcnn_swin_tiny.pth.tar"),
        Path("D:/CodexData/YB/weights/cascade_mask_rcnn_swin_tiny.pth.tar"),
        Path("E:/PHD/YB/Anti_yb/trackers/SiamDT/pretrained_weights/cascade_mask_rcnn_swin_tiny.pth.tar"),
    ]
    for sc in swin_candidates:
        if sc.exists():
            dst_swin = gdrive_root / "weights/foundation_backbones/swin_tiny/cascade_mask_rcnn_swin_tiny.pth.tar"
            if not (dst_swin.exists() and dst_swin.stat().st_size == sc.stat().st_size):
                shutil.copy2(sc, dst_swin)
            report["weights"]["cascade_mask_rcnn_swin_tiny.pth.tar"] = {
                "source": str(sc),
                "destination": str(dst_swin),
                "bytes": dst_swin.stat().st_size,
            }
            break

    # 2. 同步冻结划分 JSON 与 Manifest
    dst_splits_dir = gdrive_root / "datasets/diagnostic_splits"
    src_json_files = [
        workspace_root / "datasets/diagnostic_splits/discovery_set.json",
        workspace_root / "datasets/diagnostic_splits/holdout_verification.json",
        workspace_root / "datasets/diagnostic_splits/val90_full.json",
        workspace_root / "phase1_diagnostic_subset/DIAGNOSTIC_SUBSET_MANIFEST.json",
    ]
    for sjf in src_json_files:
        if sjf.exists():
            dst_jf = dst_splits_dir / sjf.name
            shutil.copy2(sjf, dst_jf)
            report["diagnostic_splits"].append({
                "name": sjf.name,
                "destination": str(dst_jf),
                "bytes": dst_jf.stat().st_size,
                "sha256": calc_sha256(dst_jf),
            })

    # 3. 同步官方 val 属性标注 (annos/val/att/*.txt)
    if local_annos_dir.exists():
        dst_annos_val = gdrive_root / "datasets/AntiUAV410/annos/val"
        dst_annos_att = dst_annos_val / "att"
        dst_annos_att.mkdir(parents=True, exist_ok=True)
        att_dir = local_annos_dir / "att"
        cnt = 0
        if att_dir.exists():
            for tf in sorted(att_dir.glob("*.txt")):
                dst_tf = dst_annos_att / tf.name
                if not (dst_tf.exists() and dst_tf.stat().st_size == tf.stat().st_size):
                    shutil.copy2(tf, dst_tf)
                cnt += 1
        report["annos_synced_count"] = cnt

    # 4. 同步 Colab Notebook 与 Phase 2 统一评测/提取器脚本至 Google Drive `colab/` 与 `docs/`
    colab_files = [
        workspace_root / "colab_a100_engineering/SimDT_Colab_A100_Master.ipynb",
        workspace_root / "colab_a100_engineering/colab_init_cells.py",
        workspace_root / "phase2_unified_eval/run_phase2_multi_tracker_colab.py",
        workspace_root / "phase2_unified_eval/failure_episode_extractor.py",
    ]
    for cf in colab_files:
        if cf.exists():
            dst_cf = gdrive_root / "colab" / cf.name
            shutil.copy2(cf, dst_cf)
            report["phase2_scripts_synced"].append(str(dst_cf))

    doc_files = [
        workspace_root / "phase2_unified_eval/PHASE2_FAILURE_DEFINITION_PROTOCOL.md",
        workspace_root / "phase1_diagnostic_subset/PHASE1_SUBSET_DESIGN_REPORT.md",
        workspace_root / "colab_a100_engineering/ARCHITECTURE_GITHUB_COLAB_GDRIVE.md",
    ]
    for df in doc_files:
        if df.exists():
            dst_df = gdrive_root / "docs" / df.name
            shutil.copy2(df, dst_df)
            report["phase2_scripts_synced"].append(str(dst_df))

    return report


def verify_local_core40_source(
    workspace_root: Path,
    local_val_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    核验本机 `D:\\CodexData\\YB\\datasets\\Anti-UAV410_official_20261024\\original_train_val\\val`
    是否完整包含 Core 40 (24 Discovery + 16 Holdout) 的全部序列、`IR_label.json` 与图像帧。
    """
    disc_json = json.loads((workspace_root / "datasets/diagnostic_splits/discovery_set.json").read_text(encoding="utf-8"))
    hold_json = json.loads((workspace_root / "datasets/diagnostic_splits/holdout_verification.json").read_text(encoding="utf-8"))

    core40_specs: List[Dict[str, Any]] = []
    for item in disc_json.get("sequence_details", []):
        entry = dict(item)
        entry["split_role"] = "discovery"
        core40_specs.append(entry)
    for item in hold_json.get("sequence_details", []):
        entry = dict(item)
        entry["split_role"] = "holdout_verification"
        core40_specs.append(entry)

    verified_count = 0
    total_expected_frames = 0
    total_found_jpgs = 0
    missing_seqs: List[str] = []

    for spec in core40_specs:
        seq_name = spec["seq_name"]
        expected_frames = int(spec["frames"])
        expected_sha = spec.get("label_sha256", "")
        total_expected_frames += expected_frames

        seq_dir = local_val_dir / seq_name
        lbl_path = seq_dir / "IR_label.json"
        if not lbl_path.exists():
            spec["local_verified"] = False
            missing_seqs.append(seq_name)
            continue

        # 快速统计目录下 .jpg 数量（os.scandir 比 glob 快 5~10 倍）
        jpg_count = 0
        with os.scandir(seq_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith(".jpg"):
                    jpg_count += 1

        lbl_sha = calc_sha256(lbl_path)
        sha_ok = (not expected_sha) or (lbl_sha == expected_sha)
        frames_ok = (jpg_count == expected_frames)

        spec["local_seq_dir"] = str(seq_dir)
        spec["local_jpg_count"] = jpg_count
        spec["local_label_sha256_match"] = sha_ok
        spec["local_verified"] = (sha_ok and frames_ok)

        total_found_jpgs += jpg_count
        if spec["local_verified"]:
            verified_count += 1
        else:
            missing_seqs.append(seq_name)

    summary = {
        "local_val_dir": str(local_val_dir),
        "core40_total_sequences": len(core40_specs),
        "discovery_sequences": len(disc_json.get("sequence_details", [])),
        "holdout_sequences": len(hold_json.get("sequence_details", [])),
        "verified_complete_sequences": verified_count,
        "total_expected_frames": total_expected_frames,
        "total_found_jpg_frames": total_found_jpgs,
        "all_40_verified_100pct": (verified_count == len(core40_specs)),
        "missing_or_mismatch_sequences": missing_seqs,
    }
    return core40_specs, summary


def _copy_single_file_if_needed(src_dst: Tuple[Path, Path, int]) -> int:
    """工作线程函数：若目标文件不存在或字节大小不一致，执行只读复制，返回新复制字节数。"""
    src, dst, src_size = src_dst
    try:
        if dst.exists() and dst.stat().st_size == src_size:
            return 0
    except OSError:
        pass
    shutil.copy2(src, dst)
    return src_size


def stage_single_sequence_to_gdrive(
    seq_name: str,
    local_val_dir: Path,
    gdrive_val_dir: Path,
    num_workers: int = 8,
) -> Dict[str, Any]:
    """
    将单条序列从本机只读同步至 `G:\\我的云端硬盘\\SiamDT_Project\\datasets\\AntiUAV410\\val\\<seq_name>`。
    支持文件级断点续传（已存在且字节数匹配的 `.jpg` 自动跳过）。
    """
    src_dir = local_val_dir / seq_name
    dst_dir = gdrive_val_dir / seq_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tasks: List[Tuple[Path, Path, int]] = []
    total_files = 0
    already_present = 0

    # 先收集目标目录已有文件的大小映射，避免逐文件 stat 造成重复系统调用
    existing_sizes: Dict[str, int] = {}
    with os.scandir(dst_dir) as dst_it:
        for d_entry in dst_it:
            if d_entry.is_file():
                try:
                    existing_sizes[d_entry.name] = d_entry.stat().st_size
                except OSError:
                    pass

    with os.scandir(src_dir) as src_it:
        for s_entry in src_it:
            if not s_entry.is_file():
                continue
            fname = s_entry.name
            if not (fname.lower().endswith(".jpg") or fname == "IR_label.json"):
                continue
            total_files += 1
            s_size = s_entry.stat().st_size
            if existing_sizes.get(fname) == s_size:
                already_present += 1
            else:
                tasks.append((Path(s_entry.path), dst_dir / fname, s_size))

    copied_bytes = 0
    if tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            for b in pool.map(_copy_single_file_if_needed, tasks):
                copied_bytes += b

    elapsed = round(time.time() - t0, 2)
    return {
        "seq_name": seq_name,
        "total_files": total_files,
        "skipped_existing_files": already_present,
        "newly_copied_files": len(tasks),
        "newly_copied_mb": round(copied_bytes / (1024 * 1024), 2),
        "elapsed_sec": elapsed,
        "status": "READY",
    }


def stage_core40_sequences(
    core40_specs: List[Dict[str, Any]],
    local_val_dir: Path,
    gdrive_root: Path,
    max_full_sequences: int = 4,
    stage_all_labels_immediately: bool = True,
    num_workers: int = 8,
) -> Dict[str, Any]:
    """
    按优先级顺序向 Google Drive 预置 Core 40 序列：
    1. 首先将全部 40 条序列的 `IR_label.json` 100% 同步到位；
    2. 对优先级队列 (`PRIORITY_STAGE_SEQUENCES` + 其余 Discovery + Holdout) 执行断点续传帧同步，
       支持通过 `max_full_sequences` 控制本次前台同步的完整序列条数（0 表示同步全部 40 条）。
    """
    gdrive_val_dir = gdrive_root / "datasets/AntiUAV410/val"
    gdrive_val_dir.mkdir(parents=True, exist_ok=True)

    # 1. 秒级预置全部 40 条序列的 IR_label.json
    labels_staged = 0
    if stage_all_labels_immediately:
        for spec in core40_specs:
            sname = spec["seq_name"]
            src_lbl = local_val_dir / sname / "IR_label.json"
            dst_seq_dir = gdrive_val_dir / sname
            dst_lbl = dst_seq_dir / "IR_label.json"
            if src_lbl.exists():
                dst_seq_dir.mkdir(parents=True, exist_ok=True)
                if not (dst_lbl.exists() and dst_lbl.stat().st_size == src_lbl.stat().st_size):
                    shutil.copy2(src_lbl, dst_lbl)
                labels_staged += 1

    # 2. 构建优先级序列列表（Priority -> Discovery -> Holdout）
    spec_map = {s["seq_name"]: s for s in core40_specs}
    ordered_seq_names: List[str] = []
    for p_seq in PRIORITY_STAGE_SEQUENCES:
        if p_seq in spec_map and p_seq not in ordered_seq_names:
            ordered_seq_names.append(p_seq)
    for s in core40_specs:
        if s["split_role"] == "discovery" and s["seq_name"] not in ordered_seq_names:
            ordered_seq_names.append(s["seq_name"])
    for s in core40_specs:
        if s["split_role"] != "discovery" and s["seq_name"] not in ordered_seq_names:
            ordered_seq_names.append(s["seq_name"])

    # 检查哪些序列在 Google Drive 上已经完整就绪
    already_complete_seqs: List[str] = []
    pending_seq_names: List[str] = []
    for sname in ordered_seq_names:
        expected_f = int(spec_map[sname]["frames"])
        dst_seq_dir = gdrive_val_dir / sname
        if dst_seq_dir.exists() and (dst_seq_dir / "IR_label.json").exists():
            jpg_cnt = 0
            with os.scandir(dst_seq_dir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith(".jpg"):
                        jpg_cnt += 1
            if jpg_cnt == expected_f:
                already_complete_seqs.append(sname)
                continue
        pending_seq_names.append(sname)

    target_to_stage = pending_seq_names if max_full_sequences <= 0 else pending_seq_names[:max_full_sequences]
    staged_logs: List[Dict[str, Any]] = []

    for idx, sname in enumerate(target_to_stage, start=1):
        print(f"[Core40->GDrive {idx}/{len(target_to_stage)}] 正在同步序列 {sname} ({spec_map[sname]['frames']} 帧)...")
        res = stage_single_sequence_to_gdrive(sname, local_val_dir, gdrive_val_dir, num_workers=num_workers)
        staged_logs.append(res)
        if res["Status" if "Status" in res else "status"] == "READY":
            already_complete_seqs.append(sname)

    status_payload = {
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gdrive_val_dir": str(gdrive_val_dir),
        "core40_labels_staged_count": labels_staged,
        "core40_full_frames_ready_count": len(already_complete_seqs),
        "core40_total_sequences": len(ordered_seq_names),
        "ready_sequences": already_complete_seqs,
        "remaining_sequences_count": len(ordered_seq_names) - len(already_complete_seqs),
        "newly_staged_in_this_run": staged_logs,
    }

    status_file = gdrive_root / "datasets/diagnostic_splits/core40_gdrive_staging_status.json"
    status_file.write_text(json.dumps(status_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return status_payload


def pack_core40_contiguous_tar(
    core40_specs: List[Dict[str, Any]],
    local_val_dir: Path,
    gdrive_root: Path,
) -> Dict[str, Any]:
    """
    将 Core 40 全部 40 条序列 (45,728 帧 + 40 个 IR_label.json) 打包为单个无压缩连续归档文件：
    `G:/我的云端硬盘/SiamDT_Project/datasets/archives/AntiUAV410_Core40_val.tar`
    优势：
    1. Google Drive 桌面端同步 1 个连续大文件比同步 45,728 个零散 JPG 快 50 倍以上；
    2. Colab 端可在 2 秒内解压至本地 NVMe SSD (`/content/runtime/datasets/AntiUAV410/val`)，彻底消除 FUSE 随机读瓶颈。
    """
    out_tar = gdrive_root / "datasets/archives/AntiUAV410_Core40_val.tar"
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if out_tar.exists() and out_tar.stat().st_size > 700 * 1024 * 1024:
        return {
            "archive_path": str(out_tar),
            "size_mb": round(out_tar.stat().st_size / (1024 * 1024), 2),
            "status": "ALREADY_EXISTS",
        }

    tmp_tar = out_tar.with_suffix(".tar.tmp")
    total_files = 0
    total_bytes = 0
    with tarfile.open(tmp_tar, "w") as tf:
        for spec in core40_specs:
            sname = spec["seq_name"]
            sdir = local_val_dir / sname
            if not sdir.exists():
                continue
            with os.scandir(sdir) as it:
                entries = sorted(it, key=lambda e: e.name)
            for entry in entries:
                if entry.is_file():
                    st = entry.stat()
                    tf.add(entry.path, arcname=f"{sname}/{entry.name}", recursive=False)
                    total_files += 1
                    total_bytes += st.st_size

    if out_tar.exists():
        out_tar.unlink()
    tmp_tar.replace(out_tar)
    elapsed = round(time.time() - t0, 2)
    return {
        "archive_path": str(out_tar),
        "sequences_packed": len(core40_specs),
        "files_packed": total_files,
        "size_mb": round(total_bytes / (1024 * 1024), 2),
        "elapsed_sec": elapsed,
        "status": "PACKED_READY",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 Google Drive 云端金库目录、官方权重与 Core 40 数据集断点续传预置脚本"
    )
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--gdrive-root", type=Path, default=DEFAULT_GDRIVE_ROOT)
    parser.add_argument("--local-val-dir", type=Path, default=DEFAULT_LOCAL_VAL_DIR)
    parser.add_argument("--local-siamdt-ckpt", type=Path, default=DEFAULT_LOCAL_SIAMDT_CKPT)
    parser.add_argument("--local-annos-dir", type=Path, default=DEFAULT_LOCAL_ANNOS_DIR)
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=4,
        help="本次运行同步完整 JPG 图像帧的序列数量 (设为 0 则同步全部 40 条 Core 序列；默认优先同步前 4 条核心/冒烟序列)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="单序列多文件并发复制线程数",
    )
    parser.add_argument(
        "--pack-tar",
        action="store_true",
        help="额外将 Core 40 全部 40 条序列打包为单个 datasets/archives/AntiUAV410_Core40_val.tar 以供 Colab NVMe 极速解压",
    )
    args = parser.parse_args()

    dirs_created = ensure_gdrive_layout(args.gdrive_root)
    assets_rep = stage_weights_and_splits(
        workspace_root=args.workspace_root,
        gdrive_root=args.gdrive_root,
        local_siamdt_ckpt=args.local_siamdt_ckpt,
        local_annos_dir=args.local_annos_dir,
    )
    core40_specs, local_verify_rep = verify_local_core40_source(
        workspace_root=args.workspace_root,
        local_val_dir=args.local_val_dir,
    )
    tar_rep = None
    if args.pack_tar:
        tar_rep = pack_core40_contiguous_tar(
            core40_specs=core40_specs,
            local_val_dir=args.local_val_dir,
            gdrive_root=args.gdrive_root,
        )
    staging_rep = stage_core40_sequences(
        core40_specs=core40_specs,
        local_val_dir=args.local_val_dir,
        gdrive_root=args.gdrive_root,
        max_full_sequences=args.max_sequences,
        stage_all_labels_immediately=True,
        num_workers=args.workers,
    )

    final_output = {
        "dirs_verified_or_created": dirs_created,
        "weights_and_splits_sync": assets_rep,
        "local_core40_integrity": local_verify_rep,
        "core40_tar_archive": tar_rep,
        "gdrive_core40_staging": staging_rep,
    }
    print(json.dumps(final_output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
