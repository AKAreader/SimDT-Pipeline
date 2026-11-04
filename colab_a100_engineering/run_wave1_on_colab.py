#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimDT 围点打援 —— Colab 端第一轮 (Wave 1) 闭环自动执行器 (run_wave1_on_colab.py)
==============================================================================
直接在 Colab A100 实例本地按序执行 Wave 1 全流程，并实时将步骤进度、序列完成数与预计结束时间 (ETA)
写入 `/content/runtime/wave_progress.json`，由 `colab_remote_bridge.py` 随 `heartbeat.json` 每 15 秒
同步回本地 VS Code 监控面板 (`COLAB_A100_LIVE_DASHBOARD.md`)。
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

RUNTIME_ROOT = Path("/content/runtime")
DRIVE_ROOT = Path(os.environ.get("SIAMDT_DRIVE_ROOT", "/content/drive/MyDrive/SiamDT_Project"))
PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROGRESS_FILE = RUNTIME_ROOT / "wave_progress.json"
DRIVE_PROGRESS_FILE = DRIVE_ROOT / "colab/bridge/wave_progress.json"

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import colab_ntfy_telemetry as _ntfy_telem
except Exception:
    _ntfy_telem = None


def beijing_time_str(epoch: float | None = None) -> str:
    ts = epoch if epoch is not None else time.time()
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone(datetime.timedelta(hours=8)))
    return dt.strftime("%Y-%m-%d %H:%M:%S (北京时间)")


def write_wave_progress(payload: Dict[str, Any]) -> None:
    payload["updated_at_beijing"] = beijing_time_str()
    payload["updated_at_epoch"] = round(time.time(), 2)
    for target in (PROGRESS_FILE, DRIVE_PROGRESS_FILE):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(str(tmp), str(target))
        except Exception:
            pass
    if _ntfy_telem is not None:
        try:
            _ntfy_telem.ensure_daemon_started()
        except Exception:
            pass


def count_completed_sequences(tracker_name: str = "SiamDT_Baseline", split_name: str = "discovery_set_24") -> int:
    tele_dir = DRIVE_ROOT / "experiments/failure_diagnosis/per_frame_telemetry" / f"{tracker_name}_{split_name}"
    if not tele_dir.exists():
        # 兼容不同 split_name 后缀
        parent = DRIVE_ROOT / "experiments/failure_diagnosis/per_frame_telemetry"
        if parent.exists():
            cands = sorted(parent.glob(f"{tracker_name}_*"))
            if cands:
                tele_dir = cands[0]
    if not tele_dir.exists():
        return 0
    return sum(1 for p in tele_dir.glob("*.jsonl") if p.is_file() and p.stat().st_size > 0)


def main() -> None:
    t_wave_start = time.time()
    total_seqs = 24
    steps_plan = [
        ("Step 1/5: 数据集与官方仓库准备 (Cells 1-3)", 60.0),
        ("Step 2/5: 双轨 GPU 环境恢复 (Tier-1 SiamDT + Tier-2 Modern)", 180.0),
        ("Step 3/5: 原始 SiamDT 6槽位满载并发评测 (24个核心视频, 27,791帧)", 420.0),
        ("Step 4/5: 单进程纯净延迟基准抽测 (2个视频)", 60.0),
        ("Step 5/5: 真实 A100 GPU 6层模块特征探针自检 (--model-smoke)", 45.0),
    ]
    total_est_sec = sum(x[1] for x in steps_plan)

    state: Dict[str, Any] = {
        "wave_name": "第一轮 (Wave 1): 原始 SiamDT A100 满载基准评测 + 6层特征探针真机核验",
        "status": "RUNNING",
        "started_at_beijing": beijing_time_str(t_wave_start),
        "started_at_epoch": round(t_wave_start, 2),
        "current_step_index": 1,
        "total_steps": 5,
        "current_step_title": steps_plan[0][0],
        "completed_sequences": 0,
        "total_sequences": total_seqs,
        "sequence_progress_pct": 0.0,
        "overall_progress_pct": 2.0,
        "elapsed_sec": 0.0,
        "eta_remaining_sec": total_est_sec,
        "eta_finish_beijing": beijing_time_str(t_wave_start + total_est_sec),
        "steps_history": [],
    }
    write_wave_progress(state)

    stop_monitor = threading.Event()

    def _bg_progress_updater() -> None:
        while not stop_monitor.wait(5.0):
            now = time.time()
            elapsed = now - t_wave_start
            idx = state["current_step_index"]
            done_seqs = count_completed_sequences("SiamDT_Baseline")
            state["completed_sequences"] = done_seqs
            state["sequence_progress_pct"] = round(done_seqs * 100.0 / max(1, total_seqs), 1)
            state["elapsed_sec"] = round(elapsed, 1)

            # 动态计算剩余时间 ETA
            step_elapsed = max(1.0, now - float(state.get("_step_t0") or now))
            if idx == 1:
                rem = max(30.0, total_est_sec - elapsed)
                pct = min(10.0, round(step_elapsed / 60.0 * 10.0, 1))
            elif idx == 2:
                rem = max(60.0, (total_est_sec - 60.0) - step_elapsed)
                pct = min(25.0, 10.0 + round(step_elapsed / 180.0 * 15.0, 1))
            elif idx == 3:
                if done_seqs > 0:
                    sec_per_seq = step_elapsed / done_seqs
                    rem_seqs = max(0, total_seqs - done_seqs)
                    rem = rem_seqs * sec_per_seq + 105.0
                else:
                    rem = max(120.0, 525.0 - step_elapsed)
                pct = 25.0 + round(done_seqs / total_seqs * 60.0, 1)
            elif idx == 4:
                rem = max(45.0, 105.0 - step_elapsed)
                pct = 90.0
            else:
                rem = max(10.0, 45.0 - step_elapsed)
                pct = 96.0

            state["overall_progress_pct"] = min(99.0, max(state["overall_progress_pct"], pct))
            state["eta_remaining_sec"] = round(rem, 1)
            state["eta_finish_beijing"] = beijing_time_str(now + rem)
            write_wave_progress(state)

    mon_thread = threading.Thread(target=_bg_progress_updater, daemon=True)
    mon_thread.start()

    def _run_step(idx: int, title: str, argv: List[str]) -> None:
        t0 = time.time()
        state["current_step_index"] = idx
        state["current_step_title"] = title
        state["_step_t0"] = t0
        write_wave_progress(state)
        print(f"\n>>> [{beijing_time_str()}] 开始 {title}\n    CMD: {' '.join(argv)}", flush=True)
        proc = subprocess.run(argv, cwd=str(PIPELINE_DIR))
        dt = round(time.time() - t0, 2)
        state["steps_history"].append({
            "step_index": idx,
            "title": title,
            "returncode": proc.returncode,
            "elapsed_sec": dt,
            "finished_at_beijing": beijing_time_str(),
        })
        write_wave_progress(state)
        if proc.returncode != 0:
            state["status"] = f"FAILED_AT_STEP_{idx}_RC_{proc.returncode}"
            write_wave_progress(state)
            raise SystemExit(proc.returncode)

    try:
        init_script = str(PIPELINE_DIR / "colab_a100_engineering/colab_init_cells.py")
        eval_script = str(PIPELINE_DIR / "phase2_unified_eval/run_phase2_multi_tracker_colab.py")
        probe_script = str(PIPELINE_DIR / "phase5_feature_diagnosis/siamdt_6layer_feature_probe.py")
        tier1_py = "/content/envs/siamdt-cu118/bin/python"

        # Step 1: Cells 1,2,3 (若已在本地 NVMe 解压完毕则秒级跳过)
        ckpt_ready = Path("/content/runtime/Anti-UAV410/trackers/SiamDT/checkpoints/siamdt_swin_tiny_sgd.pth").exists()
        val_dir = Path("/content/runtime/Anti-UAV410/data/Anti-UAV410/val")
        val_ready = val_dir.exists() and sum(1 for p in val_dir.iterdir() if p.is_dir()) >= 40
        if ckpt_ready and val_ready:
            print(f"\n>>> [{beijing_time_str()}] Step 1/5 已在 /content/runtime 就绪，秒级跳过！", flush=True)
            state["steps_history"].append({
                "step_index": 1,
                "title": steps_plan[0][0],
                "returncode": 0,
                "elapsed_sec": 0.1,
                "finished_at_beijing": beijing_time_str(),
            })
            write_wave_progress(state)
        else:
            _run_step(1, steps_plan[0][0], [sys.executable, init_script, "--run-cell", "1,2,3"])

        # Step 2: Cell 4 (Environment Setup)
        _run_step(2, steps_plan[1][0], [sys.executable, init_script, "--run-cell", "4"])

        # Step 3: SiamDT_Baseline Full Evaluation on discovery_set (6 slots on A100-80GB)
        _run_step(3, steps_plan[2][0], [
            sys.executable, eval_script,
            "--action", "eval",
            "--trackers", "SiamDT_Baseline",
            "--split", "discovery_set.json",
            "--concurrency-mode", "hybrid_b_then_a",
            "--gpu-slots", "6",
        ])

        # Step 4: 1-Slot Latency Benchmark (2 sequences)
        _run_step(4, steps_plan[3][0], [
            sys.executable, eval_script,
            "--action", "eval",
            "--trackers", "SiamDT_Baseline",
            "--split", "discovery_set.json",
            "--benchmark-latency",
            "--max-sequences", "2",
        ])

        # Step 5: 6-Layer Feature Probe Real GPU Model Smoke
        py_for_probe = tier1_py if Path(tier1_py).exists() else sys.executable
        _run_step(5, steps_plan[4][0], [py_for_probe, probe_script, "--model-smoke"])

        stop_monitor.set()
        state["status"] = "COMPLETED_SUCCESS"
        state["completed_sequences"] = count_completed_sequences("SiamDT_Baseline")
        state["sequence_progress_pct"] = 100.0
        state["overall_progress_pct"] = 100.0
        state["elapsed_sec"] = round(time.time() - t_wave_start, 1)
        state["eta_remaining_sec"] = 0.0
        state["eta_finish_beijing"] = beijing_time_str()
        write_wave_progress(state)
        print(f"\n🎉 [{beijing_time_str()}] Wave 1 全部 5 个步骤成功完成！总耗时 {state['elapsed_sec']} 秒。", flush=True)
    finally:
        stop_monitor.set()


if __name__ == "__main__":
    main()
