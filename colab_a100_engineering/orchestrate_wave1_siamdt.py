#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimDT 围点打援 —— 第一轮 (Wave 1) Colab A100-80GB 全自动总控执行脚本
===================================================================
执行步骤：
  1. stage_assets (自动自更新 GitHub v2.3-colab-nvme-fix + 准备 /content/runtime/Anti-UAV410 与官方权重)
  2. setup_env    (构建 Tier-1 /content/envs/siamdt-cu118 与 Tier-2 /content/envs/modern-pyt2)
  3. run_phase2   (Round 1: 原始 SiamDT_Baseline 在 A100-80GB 上开启 6 分片满载并发评测 discovery_set 24 序列)
  4. run_phase2   (单进程 1-slot 真实延迟基准测速 --benchmark-latency 2 序列)
  5. run_probe    (在真实 A100 GPU 上执行 siamdt_6layer_feature_probe.py --model-smoke 真机 6 层特征探针自检)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PYTHON_EXE = sys.executable
CONTROLLER = Path(__file__).resolve().parent / "local_colab_controller.py"
PROGRESS_JSON = Path(__file__).resolve().parents[1] / "phase2_unified_eval" / "WAVE1_LIVE_PROGRESS.json"


def save_progress(state: Dict[str, Any]) -> None:
    PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    PROGRESS_JSON.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_remote_step(
    step_name: str,
    cmd_id: str,
    action: str,
    timeout_min: int,
    remote_args: List[str] | None = None,
    state: Dict[str, Any] | None = None,
) -> int:
    t0 = time.time()
    print(f"\n================ [{step_name}] cmd_id={cmd_id} action={action} ================", flush=True)
    if state is not None:
        state["current_step"] = step_name
        state["current_cmd_id"] = cmd_id
        state["steps"].append({
            "step": step_name,
            "cmd_id": cmd_id,
            "action": action,
            "remote_args": remote_args or [],
            "status": "RUNNING",
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        })
        save_progress(state)

    argv = [
        PYTHON_EXE,
        str(CONTROLLER),
        "--action",
        action,
        "--cmd-id",
        cmd_id,
        "--wait",
        "--timeout-min",
        str(timeout_min),
        "--pickup-timeout-min",
        "15",
        "--poll-interval",
        "8",
    ]
    if remote_args:
        argv.append("--")
        argv.extend(remote_args)

    proc = subprocess.run(argv)
    elapsed = round(time.time() - t0, 2)
    rc = int(proc.returncode)
    print(f"================ [{step_name}] 完成 rc={rc} elapsed={elapsed}s ================\n", flush=True)
    if state is not None and state["steps"]:
        state["steps"][-1]["status"] = "COMPLETED" if rc == 0 else f"FAILED_RC_{rc}"
        state["steps"][-1]["returncode"] = rc
        state["steps"][-1]["elapsed_sec"] = elapsed
        state["steps"][-1]["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_progress(state)
    return rc


def main() -> None:
    ts = time.strftime("%H%M%S")
    state: Dict[str, Any] = {
        "wave": "Wave1_SiamDT_Baseline_Full_A100_80GB",
        "gpu": "NVIDIA A100-SXM4-80GB",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current_step": "INIT",
        "current_cmd_id": None,
        "overall_status": "RUNNING",
        "steps": [],
    }
    save_progress(state)

    # Step 1: stage_assets (Pass 1 拉取最新 GitHub commit 0639c7e；若旧内存脚本在 Part B 退出则自动执行 Pass 2)
    rc1 = run_remote_step(
        step_name="Step1_StageAssets_Pass1",
        cmd_id=f"w1_stage_p1_{ts}",
        action="stage_assets",
        timeout_min=30,
        state=state,
    )
    if rc1 != 0:
        print("[Wave1] Pass 1 已完成 GitHub 代码自更新，立即启动 Pass 2 执行新版 colab_init_cells.py ...", flush=True)
        rc1b = run_remote_step(
            step_name="Step1_StageAssets_Pass2",
            cmd_id=f"w1_stage_p2_{ts}",
            action="stage_assets",
            timeout_min=30,
            state=state,
        )
        if rc1b != 0:
            state["overall_status"] = "FAILED_AT_STAGE_ASSETS"
            save_progress(state)
            sys.exit(rc1b)

    # Step 2: setup_env (恢复 Tier-1 /content/envs/siamdt-cu118 与 Tier-2 /content/envs/modern-pyt2)
    rc2 = run_remote_step(
        step_name="Step2_SetupEnv_Tier1_Tier2",
        cmd_id=f"w1_setup_env_{ts}",
        action="setup_env",
        timeout_min=45,
        state=state,
    )
    if rc2 != 0:
        state["overall_status"] = "FAILED_AT_SETUP_ENV"
        save_progress(state)
        sys.exit(rc2)

    # Step 3: Round 1 原始 SiamDT_Baseline 在 A100-80GB 上开启 6 分片满载并发评测 discovery_set (24 序列)
    rc3 = run_remote_step(
        step_name="Step3_Eval_SiamDT_Baseline_Discovery24_6Slots",
        cmd_id=f"w1_eval_siamdt_disc24_{ts}",
        action="run_phase2",
        timeout_min=180,
        remote_args=[
            "--action", "eval",
            "--trackers", "SiamDT_Baseline",
            "--split", "discovery_set.json",
            "--concurrency-mode", "hybrid_b_then_a",
            "--gpu-slots", "6",
        ],
        state=state,
    )
    if rc3 != 0:
        state["overall_status"] = "FAILED_AT_SIAMDT_EVAL"
        save_progress(state)
        sys.exit(rc3)

    # Step 4: 单进程 1-slot 延迟基准测速 (2 条序列)
    rc4 = run_remote_step(
        step_name="Step4_LatencyBench_SiamDT_Baseline_1Slot",
        cmd_id=f"w1_latbench_siamdt_{ts}",
        action="run_phase2",
        timeout_min=45,
        remote_args=[
            "--action", "eval",
            "--trackers", "SiamDT_Baseline",
            "--split", "discovery_set.json",
            "--benchmark-latency",
            "--max-sequences", "2",
        ],
        state=state,
    )

    # Step 5: 真实 A100 GPU 上运行 6 层特征探针真机冒烟验证 (--model-smoke)
    rc5 = run_remote_step(
        step_name="Step5_Probe6Layer_ModelSmoke_A100",
        cmd_id=f"w1_probe_smoke_{ts}",
        action="run_probe",
        timeout_min=30,
        remote_args=["--model-smoke"],
        state=state,
    )
    if rc5 != 0:
        state["overall_status"] = "FAILED_AT_PROBE_SMOKE"
        save_progress(state)
        sys.exit(rc5)

    state["current_step"] = "ALL_DONE"
    state["overall_status"] = "COMPLETED_SUCCESS"
    save_progress(state)
    print("\n🎉 [Wave 1 全部完成] 原始 SiamDT 6 槽位满载评测 + 延迟基准 + 6 层特征探针真机验证均已通过！", flush=True)


if __name__ == "__main__":
    main()
