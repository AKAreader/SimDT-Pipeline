#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab A100 零延迟 HTTP 遥测直推守护进程 (colab_ntfy_telemetry.py)
==================================================================
在 Colab 端独立后台运行，每 5 秒采集一次：
1. A100 GPU 实时利用率 (%)、已用显存 (MB)、总显存 (MB)
2. `/content/runtime/wave_progress.json`（当前步骤、已完成视频数、预计结束时间 ETA）
3. `/content/drive/MyDrive/SiamDT_Project/colab/bridge/cmd_results/*.log` 最新日志尾部
并通过 HTTPS 直接推送到 `https://ntfy.sh/simdt_a100_telem_9fe17727e2a3`，
完全绕开 Google Drive 桌面端同步队列延迟，供本地 VS Code 与桌面 GUI 实时秒级读取。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

NTFY_TOPIC = os.environ.get("SIMDT_NTFY_TOPIC", "simdt_a100_telem_9fe17727e2a3")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"
PID_FILE = Path("/content/runtime/ntfy_telemetry.pid")
DRIVE_ROOT = Path(os.environ.get("SIAMDT_DRIVE_ROOT", "/content/drive/MyDrive/SiamDT_Project"))


def query_gpu() -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        first = out.splitlines()[0]
        parts = [x.strip() for x in first.split(",")]
        return {
            "gpu_available": True,
            "gpu_name": parts[0],
            "vram_total_mb": float(parts[1]),
            "vram_used_mb": float(parts[2]),
            "vram_free_mb": float(parts[3]),
            "gpu_utilization_pct": float(parts[4]),
        }
    except Exception:
        return {
            "gpu_available": False,
            "gpu_name": "NO_GPU",
            "vram_total_mb": 0.0,
            "vram_used_mb": 0.0,
            "vram_free_mb": 0.0,
            "gpu_utilization_pct": 0.0,
        }


def collect_snapshot() -> Dict[str, Any]:
    now = time.time()
    gpu = query_gpu()
    wave_prog = None
    for wp in (Path("/content/runtime/wave_progress.json"), DRIVE_ROOT / "colab/bridge/wave_progress.json"):
        if wp.exists():
            try:
                wave_prog = json.loads(wp.read_text(encoding="utf-8"))
                break
            except Exception:
                pass

    latest_log_name = ""
    latest_log_tail = ""
    res_dir = DRIVE_ROOT / "colab/bridge/cmd_results"
    if res_dir.exists():
        try:
            logs = sorted(res_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            if logs:
                latest_log_name = logs[0].name
                lines = logs[0].read_text(encoding="utf-8", errors="replace").splitlines()
                latest_log_tail = "\n".join(lines[-25:])[-2500:]
        except Exception:
            pass

    return {
        "status": "ONLINE",
        "source": "NTFY_DIRECT_STREAM",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "timestamp_epoch": round(now, 3),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "gpu": gpu,
        "gpu_name": gpu["gpu_name"],
        "vram_total_mb": gpu["vram_total_mb"],
        "vram_used_mb": gpu["vram_used_mb"],
        "vram_free_mb": gpu["vram_free_mb"],
        "wave_progress": wave_prog,
        "latest_log_name": latest_log_name,
        "latest_log_tail": latest_log_tail,
    }


def push_once(payload: Dict[str, Any]) -> bool:
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(NTFY_URL, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def ensure_daemon_started() -> None:
    if os.name == "nt":
        return
    try:
        if PID_FILE.exists():
            old_pid = int(PID_FILE.read_text().strip())
            if Path(f"/proc/{old_pid}").exists():
                return
    except Exception:
        pass
    try:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--daemon"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        pass


def main() -> None:
    if "--daemon" not in sys.argv:
        ensure_daemon_started()
        push_once(collect_snapshot())
        return
    try:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass
    while True:
        snap = collect_snapshot()
        push_once(snap)
        time.sleep(8.0)


if __name__ == "__main__":
    main()
