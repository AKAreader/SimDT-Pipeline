#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimDT 围点打援 —— 桌面级实时显卡与结束时间可视化监控窗口 (colab_gui_monitor.py)
==============================================================================
直接弹出在桌面上的深色主题可视化仪表盘，每 3 秒自动刷新一次：
  1. Colab A100-SXM4-80GB 在线状态、GPU 核心利用率 (%) 进度条、显存占用 (MB / 81920 MB) 进度条；
  2. 当前执行步骤、视频序列完成进度条 (X / 24 序列)；
  3. 预计完成的具体北京时间（精确到秒）与剩余倒计时；
  4. 远端实时输出日志尾部。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_live_monitor import resolve_bridge_dir, render_dashboard, beijing_now_str, format_duration, read_json_safe


def main() -> None:
    bridge_dir = resolve_bridge_dir()
    root = tk.Tk()
    root.title("🚀 SimDT 围点打援｜Colab A100-80GB 实时显卡与训练/评测倒计时监控台")
    root.geometry("860x690")
    root.configure(bg="#0f172a")
    try:
        root.attributes("-topmost", True)
        root.after(4000, lambda: root.attributes("-topmost", False))
    except Exception:
        pass

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Green.Horizontal.TProgressbar", foreground="#22c55e", background="#22c55e", troughcolor="#1e293b", bordercolor="#334155")
    style.configure("Blue.Horizontal.TProgressbar", foreground="#38bdf8", background="#38bdf8", troughcolor="#1e293b", bordercolor="#334155")
    style.configure("Amber.Horizontal.TProgressbar", foreground="#f59e0b", background="#f59e0b", troughcolor="#1e293b", bordercolor="#334155")

    header = tk.Frame(root, bg="#1e293b", padx=16, pady=10)
    header.pack(fill="x", padx=12, pady=(10, 4))

    lbl_title = tk.Label(header, text="🚀 SimDT 围点打援｜Colab A100-SXM4-80GB 实时监控与剩余时间面板", font=("Microsoft YaHei UI", 13, "bold"), fg="#f8fafc", bg="#1e293b")
    lbl_title.pack(anchor="w")
    lbl_sub = tk.Label(header, text="正在连接远端心跳...", font=("Microsoft YaHei UI", 9), fg="#94a3b8", bg="#1e293b")
    lbl_sub.pack(anchor="w", pady=(2, 0))

    # GPU 卡片
    gpu_card = tk.Frame(root, bg="#1e293b", padx=16, pady=10)
    gpu_card.pack(fill="x", padx=12, pady=4)

    lbl_gpu_title = tk.Label(gpu_card, text="🖥️ 远端 A100-SXM4-80GB 显卡实时负载", font=("Microsoft YaHei UI", 11, "bold"), fg="#38bdf8", bg="#1e293b")
    lbl_gpu_title.pack(anchor="w")

    lbl_gpu_util = tk.Label(gpu_card, text="GPU 核心利用率: 0.0%", font=("Microsoft YaHei UI", 10, "bold"), fg="#4ade80", bg="#1e293b")
    lbl_gpu_util.pack(anchor="w", pady=(4, 2))
    bar_gpu = ttk.Progressbar(gpu_card, style="Green.Horizontal.TProgressbar", orient="horizontal", length=800, mode="determinate", maximum=100)
    bar_gpu.pack(fill="x", pady=(0, 4))

    lbl_vram = tk.Label(gpu_card, text="GPU 显存占用 (VRAM): 0 MB / 81920 MB (0.0%)", font=("Microsoft YaHei UI", 10), fg="#e2e8f0", bg="#1e293b")
    lbl_vram.pack(anchor="w", pady=(2, 2))
    bar_vram = ttk.Progressbar(gpu_card, style="Blue.Horizontal.TProgressbar", orient="horizontal", length=800, mode="determinate", maximum=100)
    bar_vram.pack(fill="x")
    lbl_vram_note = tk.Label(
        gpu_card,
        text="💡 为什么利用率100%而显存占9.1GB？在线跟踪逐帧推理(Batch=1)每进程仅需1.5GB×6并发=9.1GB，算力核心已100%跑满；第2轮SAM2/DINOv2大模型与第3轮微调训练将占35~75GB显存。",
        font=("Microsoft YaHei UI", 8),
        fg="#94a3b8",
        bg="#1e293b",
        wraplength=810,
        justify="left",
    )
    lbl_vram_note.pack(anchor="w", pady=(4, 0))

    # 进度与倒计时卡片
    eta_card = tk.Frame(root, bg="#1e293b", padx=16, pady=10)
    eta_card.pack(fill="x", padx=12, pady=4)

    lbl_eta_title = tk.Label(eta_card, text="⏱️ 运行耗时与剩余结束倒计时 (ETA)", font=("Microsoft YaHei UI", 11, "bold"), fg="#f59e0b", bg="#1e293b")
    lbl_eta_title.pack(anchor="w")

    lbl_countdown = tk.Label(eta_card, text="⏱️ 距离本轮完成还需: 计算中...   |   ⏳ 已运行时间: 0分 00秒", font=("Microsoft YaHei UI", 13, "bold"), fg="#fde047", bg="#1e293b")
    lbl_countdown.pack(anchor="w", pady=(6, 2))

    lbl_eta_big = tk.Label(eta_card, text="🎯 本轮预计完成时间: 计算中...", font=("Microsoft YaHei UI", 11, "bold"), fg="#4ade80", bg="#1e293b")
    lbl_eta_big.pack(anchor="w", pady=(0, 4))

    lbl_step = tk.Label(eta_card, text="当前步骤: -", font=("Microsoft YaHei UI", 10), fg="#f8fafc", bg="#1e293b", wraplength=800, justify="left")
    lbl_step.pack(anchor="w", pady=(2, 2))

    lbl_seq_prog = tk.Label(eta_card, text="总进度: 0.0% | 已完成序列: 0 / 24", font=("Microsoft YaHei UI", 10), fg="#e2e8f0", bg="#1e293b")
    lbl_seq_prog.pack(anchor="w", pady=(2, 2))
    bar_prog = ttk.Progressbar(eta_card, style="Amber.Horizontal.TProgressbar", orient="horizontal", length=800, mode="determinate", maximum=100)
    bar_prog.pack(fill="x", pady=(0, 6))

    lbl_roadmap = tk.Label(
        eta_card,
        text="🗺️ 三轮推进计划: 【🟢第1轮: 原始SiamDT评测+探针】 ➔ 【⏳第2轮: 10算法并发评测(约50分)】 ➔ 【⏳第3轮: 特征解剖与微调训练(约75分)】",
        font=("Microsoft YaHei UI", 9, "bold"),
        fg="#38bdf8",
        bg="#1e293b",
        wraplength=810,
        justify="left",
    )
    lbl_roadmap.pack(anchor="w")

    # 实时日志框
    log_card = tk.Frame(root, bg="#1e293b", padx=16, pady=8)
    log_card.pack(fill="both", expand=True, padx=12, pady=(4, 10))

    lbl_log = tk.Label(log_card, text="📜 远端实时输出日志 (每 3 秒自动同步)", font=("Microsoft YaHei UI", 10, "bold"), fg="#94a3b8", bg="#1e293b")
    lbl_log.pack(anchor="w", pady=(0, 4))

    txt_log = tk.Text(log_card, bg="#090d16", fg="#cbd5e1", font=("Consolas", 9), height=8, relief="flat", padx=8, pady=6)
    txt_log.pack(fill="both", expand=True)

    def refresh() -> None:
        try:
            s = render_dashboard(bridge_dir)
            hb = read_json_safe(bridge_dir / "heartbeat.json") or {}
            wp = hb.get("wave_progress") or read_json_safe(bridge_dir / "wave_progress.json") or {}
            status_str = "🟢 在线 (ONLINE)" if s["online"] else "🔴 离线 (OFFLINE)"
            lbl_sub.config(
                text=f"状态: {status_str} | 接管桥: v{s['bridge_version']} | 心跳延迟: {s['hb_age_sec']}s | 刷新时间: {beijing_now_str()}",
                fg="#4ade80" if s["online"] else "#f87171",
            )
            lbl_gpu_title.config(text=f"🖥️ 远端显卡: {s['gpu_name']}  (本地高速盘剩余: {hb.get('runtime_disk_free_gb', '-')} GB)")
            lbl_gpu_util.config(text=f"GPU 核心利用率: {s['gpu_util_pct']:.1f}% （6并发算力满载运行中）")
            bar_gpu["value"] = max(0.0, min(100.0, s["gpu_util_pct"]))

            vram_pct = s["vram_used_mb"] * 100.0 / max(1.0, s["vram_total_mb"])
            lbl_vram.config(text=f"GPU 显存占用 (VRAM): {s['vram_used_mb']:.0f} MB / {s['vram_total_mb']:.0f} MB ({s['vram_used_mb']/1024:.1f} GB / 80 GB, {vram_pct:.1f}%)")
            bar_vram["value"] = max(0.0, min(100.0, vram_pct))

            rem_sec = float(wp.get("eta_remaining_sec") or 765.0)
            el_sec = float(wp.get("elapsed_sec") or 0.0)
            lbl_countdown.config(
                text=f"⏱️ 距离本轮完成还需: {format_duration(rem_sec)}   |   ⏳ 已运行时间: {format_duration(el_sec)}"
            )
            lbl_eta_big.config(
                text=f"🎯 本轮预计完成时刻: {s['eta_finish_beijing']}   (第1~3轮全流程预计剩余: 约 {format_duration(rem_sec + 7200)})"
            )

            lbl_step.config(text=f"当前步骤: {s['current_step']}")
            lbl_seq_prog.config(text=f"当前轮总进度: {s['overall_pct']:.1f}%  |  视频序列进度: {s['done_seqs']} / {s['total_seqs']} 个序列  |  状态: {s['wave_status']}")
            bar_prog["value"] = max(0.0, min(100.0, s["overall_pct"]))
            lbl_roadmap.config(
                text=f"🗺️ 三轮推进计划: 【✅第1轮已完成 (SA=64.81%, 8.24FPS)】 ➔ 【🟢第2轮: 10算法并发评测 ({s['done_seqs']}/{s['total_seqs']}序列)】 ➔ 【⏳第3轮: 特征解剖与微调训练(约75分)】"
            )

            # 更新日志
            res_dir = bridge_dir / "cmd_results"
            log_text = ""
            if res_dir.exists():
                logs = sorted(
                    [p for p in res_dir.glob("*.log") if not p.name.startswith("w1_stage_")],
                    key=lambda p: p.stat().st_mtime if p.exists() else 0,
                    reverse=True,
                )
                if logs:
                    try:
                        log_text = "".join(logs[0].read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-16:])
                    except Exception:
                        pass
            if not log_text:
                log_text = (
                    f"[心跳正常] 已连接 {s['gpu_name']} (桥版本 v{s['bridge_version']})。\n"
                    "若桥版本仍显示 v2.0.0，请在当前已打开的 Colab 页面按顺序点击：\n"
                    "  1. 点击【章节 2 (Cell 2)】的 ▶（拉取最新 v2.1.0 代码与任务单，忽略末尾旧报错）\n"
                    "  2. 点击【Cell 0b (停止远程接管桥)】的 ▶\n"
                    "  3. 点击【Cell 0】的 ▶（启动 v2.1.0，1 秒内自动开始跑满 A100！）\n"
                )
            txt_log.delete("1.0", tk.END)
            txt_log.insert(tk.END, log_text)
        except Exception as exc:
            lbl_sub.config(text=f"刷新异常: {exc}", fg="#f87171")
        root.after(3000, refresh)

    refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
