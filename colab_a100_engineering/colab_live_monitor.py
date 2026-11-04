#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimDT 围点打援 —— VS Code 实时显卡监控与结束时间倒计时面板 (colab_live_monitor.py)
================================================================================
实时读取 Colab 回传的 `heartbeat.json` 与 `wave_progress.json`（及逐序列遥测文件 `*.jsonl`），
在 VS Code 终端打印动态进度条，并每 5 秒自动刷新工作区根目录下的可视化看板文件：
  `G:/我的云端硬盘/SimDT围点打援/COLAB_A100_LIVE_DASHBOARD.md`
您可以在 VS Code 中打开 `COLAB_A100_LIVE_DASHBOARD.md`（或按 Ctrl+Shift+V 开启预览），
实时直观查看：
  1. Colab A100-SXM4-80GB 显卡利用率 (%)、显存占用 (MB / 81920 MB)；
  2. 当前执行阶段、已完成视频序列进度条；
  3. 预计结束的具体北京时间（精确到秒）与剩余倒计时！
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_MD = WORKSPACE_ROOT / "COLAB_A100_LIVE_DASHBOARD.md"


def resolve_bridge_dir() -> Path:
    for cand in (
        Path(r"G:\我的云端硬盘\SiamDT_Project\colab\bridge"),
        Path(r"Z:\SiamDT_Project\colab\bridge"),
        Path("/content/drive/MyDrive/SiamDT_Project/colab/bridge"),
    ):
        if cand.exists():
            return cand
    return Path(r"G:\我的云端硬盘\SiamDT_Project\colab\bridge")


def beijing_now_str() -> str:
    dt = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(sec: float) -> str:
    sec = max(0, int(round(sec)))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}小时 {m:02d}分 {s:02d}秒"
    return f"{m}分 {s:02d}秒"


def make_bar(pct: float, width: int = 24) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct / 100.0 * width))
    return "█" * filled + "░" * (width - filled) + f" {pct:5.1f}%"


def read_json_safe(p: Path) -> Optional[Dict[str, Any]]:
    try:
        os.listdir(p.parent)
    except Exception:
        pass
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def render_dashboard(bridge_dir: Path) -> Dict[str, Any]:
    hb = read_json_safe(bridge_dir / "heartbeat.json") or {}
    wp = hb.get("wave_progress") or read_json_safe(bridge_dir / "wave_progress.json") or {}

    now = time.time()
    hb_epoch = float(hb.get("timestamp_epoch") or 0.0)
    hb_age = round(max(0.0, now - hb_epoch), 1) if hb_epoch > 0 else 9999.0
    online = hb_age <= 180.0 and hb.get("status") == "ONLINE"

    gpu = hb.get("gpu") or {}
    gpu_name = gpu.get("gpu_name") or hb.get("gpu_name") or "未知 GPU"
    gpu_util = float(gpu.get("gpu_utilization_pct") or 0.0)
    vram_used = float(gpu.get("vram_used_mb") or hb.get("vram_used_mb") or 0.0)
    vram_total = float(gpu.get("vram_total_mb") or hb.get("vram_total_mb") or 81920.0)
    vram_pct = round(vram_used * 100.0 / max(1.0, vram_total), 1)

    active_cmd = hb.get("active_cmd_id") or "空闲 / 等待指令触发"
    active_act = hb.get("active_action") or "-"
    bridge_ver = hb.get("bridge_version") or "2.0.0"

    # 解析任务进度与预计结束时间
    wave_title = wp.get("wave_name") or "第一轮 (Wave 1): 原始 SiamDT A100-80GB 6槽位满载评测 + 6层特征探针"
    step_title = wp.get("current_step_title") or (
        f"正在执行远程动作: {active_act} ({active_cmd})" if active_act != "-" else "等待启动 Wave 1 自动任务（请在 Colab 点击一次刷新后的 Cell 0）"
    )
    done_seqs = int(wp.get("completed_sequences") or 0)
    total_seqs = int(wp.get("total_sequences") or 24)
    overall_pct = float(wp.get("overall_progress_pct") or 0.0)
    elapsed_sec = float(wp.get("elapsed_sec") or 0.0)
    eta_rem_sec = float(wp.get("eta_remaining_sec") or 765.0)
    eta_finish_str = wp.get("eta_finish_beijing") or datetime.datetime.fromtimestamp(
        now + eta_rem_sec, tz=datetime.timezone(datetime.timedelta(hours=8))
    ).strftime("%Y-%m-%d %H:%M:%S (北京时间)")
    wave_status = wp.get("status") or ("RUNNING" if active_act != "-" else "STANDBY")

    md = f"""# 🚀 SimDT 围点打援｜Colab A100 实时显卡与进度监控看板

> **看板最后刷新时间（北京时间）**：`{beijing_now_str()}` | **远端心跳延迟**：`{hb_age} 秒` | **接管桥版本**：`v{bridge_ver}`

---

## 1. 🖥️ 远端服务器与 A100 显卡实时状态

| 监控指标 | 实时数值 | 状态可视化 |
| :--- | :--- | :--- |
| **服务器连接状态** | **{"🟢 在线 (ONLINE)" if online else "🔴 离线 (OFFLINE)"}** | 主机 `{hb.get("hostname", "-")}` (PID `{hb.get("pid", "-")}`) |
| **分配显卡型号** | **`{gpu_name}`** | 算力架构 `sm_{gpu.get("compute_cap", "8.0")}` / 驱动 `{gpu.get("driver_version", "-")}` |
| **GPU 核心利用率** | **`{gpu_util:.1f}%`** | `{make_bar(gpu_util)}` |
| **GPU 显存占用 (VRAM)** | **`{vram_used:.0f} MB / {vram_total:.0f} MB`** (`{vram_used/1024:.1f} GB / {vram_total/1024:.0f} GB`) | `{make_bar(vram_pct)}` |
| **Colab 本地高速盘剩余** | **`{hb.get("runtime_disk_free_gb", "-")} GB`** | Google 云端硬盘剩余：`{hb.get("drive_free_gb", "-")} GB` |

---

## 2. ⏱️ 当前运行阶段与预计结束时间 (ETA)

| 进度指标 | 实时状态 |
| :--- | :--- |
| **当前实验波次** | **{wave_title}** |
| **任务运行状态** | **`{wave_status}`**（当前活动指令：`{active_cmd}`） |
| **当前执行步骤** | **{step_title}** |
| **总进度条** | `{make_bar(overall_pct)}` |
| **视频序列评测进度** | **`{done_seqs} / {total_seqs} 个序列`** (`{make_bar(done_seqs * 100.0 / max(1, total_seqs))}`) |
| **已运行时间** | **`{format_duration(elapsed_sec)}`** |
| **预计剩余时间** | **`{format_duration(eta_rem_sec)}`** |
| **🎯 预计完成时间（北京时间）** | **`{eta_finish_str}`** |

---

## 3. 📋 阶段步骤执行明细

| 步骤序号 | 步骤名称 | 状态 | 耗时 | 完成时间 |
| :---: | :--- | :---: | :---: | :--- |
"""
    hist = wp.get("steps_history") or []
    if hist:
        for item in hist:
            st_icon = "✅ 完成" if item.get("returncode") == 0 else f"❌ 异常({item.get('returncode')})"
            md += f"| Step {item.get('step_index')} | {item.get('title')} | {st_icon} | `{item.get('elapsed_sec')}s` | `{item.get('finished_at_beijing')}` |\n"
    else:
        md += "| - | 正在执行或等待触发第 1 步 | ⏳ 运行中 | - | - |\n"

    # 读取最新运行日志尾部
    log_tail = ""
    res_dir = bridge_dir / "cmd_results"
    if res_dir.exists():
        logs = sorted(res_dir.glob("*.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        if logs:
            try:
                log_tail = "".join(logs[0].read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-25:])
            except Exception:
                pass

    md += f"""
---

## 4. 📜 远端实时输出日志尾部 (`cmd_results/*.log`)

```text
{log_tail.strip() or "(暂无子任务日志输出 — 任务启动后此处每 5 秒自动刷新远端最新标准输出)"}
```
"""
    tmp_md = DASHBOARD_MD.with_suffix(".md.tmp")
    tmp_md.write_text(md, encoding="utf-8")
    os.replace(str(tmp_md), str(DASHBOARD_MD))

    return {
        "online": online,
        "hb_age_sec": hb_age,
        "bridge_version": bridge_ver,
        "gpu_name": gpu_name,
        "gpu_util_pct": gpu_util,
        "vram_used_mb": vram_used,
        "vram_total_mb": vram_total,
        "wave_status": wave_status,
        "current_step": step_title,
        "done_seqs": done_seqs,
        "total_seqs": total_seqs,
        "overall_pct": overall_pct,
        "eta_finish_beijing": eta_finish_str,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Colab A100 实时显卡与 ETA 监控器")
    p.add_argument("--once", action="store_true", help="仅刷新一次看板并打印摘要后退出")
    p.add_argument("--interval", type=float, default=5.0, help="刷新间隔秒（默认 5 秒）")
    p.add_argument("--duration-min", type=float, default=180.0, help="持续监控时长（分钟）")
    args = p.parse_args()

    bridge_dir = resolve_bridge_dir()
    if args.once:
        summary = render_dashboard(bridge_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    t0 = time.time()
    print(f"[LiveMonitor] 已启动！正在实时刷新看板: {DASHBOARD_MD}", flush=True)
    while time.time() - t0 < args.duration_min * 60.0:
        s = render_dashboard(bridge_dir)
        line = (
            f"[{beijing_now_str()}] {'🟢ONLINE' if s['online'] else '🔴OFFLINE'} (v{s['bridge_version']}) | "
            f"{s['gpu_name']} Util={s['gpu_util_pct']:.0f}% VRAM={s['vram_used_mb']:.0f}/{s['vram_total_mb']:.0f}MB | "
            f"进度={s['overall_pct']:.1f}% ({s['done_seqs']}/{s['total_seqs']}序列) | "
            f"预计完成={s['eta_finish_beijing']} | {s['current_step']}"
        )
        print(line, flush=True)
        if s["wave_status"] == "COMPLETED_SUCCESS":
            print("🎉 [LiveMonitor] 检测到 Wave 1 已全部完成！", flush=True)
            break
        time.sleep(max(2.0, args.interval))


if __name__ == "__main__":
    main()
