#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimDT 围点打援 —— 本地总控遥控客户端 v2.0 (Local Colab Remote Controller，2026-11-04 修订)
====================================================================
与 colab_remote_bridge.py v2.0 配套，默认只走 Google Drive 文件队列通道，只能下发白名单动作：

本地动作：
  --action ping          读取 heartbeat.json，判断在线（默认心跳年龄 <= 180 s 视为在线）
  --action result        查询某个 cmd_id 的状态/结果与日志尾部（--cmd-id）
  --action stop-bridge   在 bridge 目录写入 STOP_BRIDGE 标志，远端桥会停止并退出
远端白名单动作（投递到 cmd_queue/）：
  status | tail_log | cancel | setup_env | stage_assets | run_phase2 | run_probe | extract_episodes | raw_shell(需远端 --allow-raw-shell)

示例：
  python local_colab_controller.py --action status --wait
  python local_colab_controller.py --action run_phase2 --wait --timeout-min 720 -- --action evaluate --split discovery_set
  python local_colab_controller.py --action tail_log --param cmd_id=cmd_20261104_xxx --param lines=200 --wait
所有等待均按分钟级设计（Drive 同步延迟通常为数十秒到数分钟）。
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CONTROLLER_VERSION = "2.0.0"
DEFAULT_MAX_AGE_SEC = 180.0
DEFAULT_POLL_SEC = 10.0
DEFAULT_PICKUP_TIMEOUT_MIN = 10.0
REMOTE_ACTIONS = ("status", "tail_log", "cancel", "setup_env", "stage_assets", "run_phase2", "run_probe", "extract_episodes", "raw_shell")
LOCAL_ACTIONS = ("ping", "result", "stop-bridge")
BUILTIN_REMOTE = ("status", "tail_log", "cancel")


def resolve_default_bridge_dir() -> Path:
    env_dir = os.environ.get("SIAMDT_BRIDGE_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    env_drive = os.environ.get("SIAMDT_DRIVE_ROOT", "").strip()
    if env_drive:
        return Path(env_drive) / "colab" / "bridge"
    for cand in (Path(r"Z:\SiamDT_Project\colab\bridge"), Path(r"G:\我的云端硬盘\SiamDT_Project\colab\bridge"),
                 Path("/content/drive/MyDrive/SiamDT_Project/colab/bridge")):
        if cand.parent.exists():
            return cand
    return Path(r"Z:\SiamDT_Project\colab\bridge")


def refresh_dir(path: Path) -> None:
    """先 os.listdir 父目录以刷新元数据（Drive for Desktop 下否则可能看不到新文件）。"""
    try:
        os.listdir(path)
    except Exception:
        pass


def atomic_fsync_write_json(target_path: Path, payload: Dict[str, Any]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    tmp_path = target_path.with_name(target_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp_path), str(target_path))


def read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    refresh_dir(path.parent)
    if not path.exists():
        return None
    for _ in range(3):
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return json.loads(text)
        except Exception:
            time.sleep(0.5)
    return None


def heartbeat_state(bridge_dir: Path, max_age_sec: float) -> Dict[str, Any]:
    hb = read_json_file(bridge_dir / "heartbeat.json")
    if hb is None:
        return {"online": False, "reason": f"未找到心跳文件 {bridge_dir / 'heartbeat.json'}（请在 Colab 运行 Cell 0）", "heartbeat": None}
    age = round(max(0.0, time.time() - float(hb.get("timestamp_epoch") or 0.0)), 1)
    status = str(hb.get("status", "ONLINE"))
    online = age <= max_age_sec and status == "ONLINE"
    return {"online": online, "heartbeat_age_sec": age, "max_allowed_age_sec": max_age_sec, "bridge_status": status, "heartbeat": hb}


def action_ping(bridge_dir: Path, max_age_sec: float) -> int:
    st = heartbeat_state(bridge_dir, max_age_sec)
    hb = st.get("heartbeat") or {}
    report = {k: v for k, v in st.items() if k != "heartbeat"}
    report.update({k: hb.get(k) for k in ("bridge_version", "gpu_name", "gpu_tier", "vram_total_mb", "vram_used_mb", "drive_free_gb",
                                          "runtime_disk_free_gb", "channels_active", "tunnel_enabled", "raw_shell_enabled",
                                          "allowed_actions", "active_cmd_id", "active_action", "processed_cmd_count",
                                          "timestamp_utc", "heartbeat_interval_sec")})
    report["bridge_dir"] = str(bridge_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if st["online"] else 2


def http_json(url: str, token: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Tuple[int, Optional[Dict[str, Any]]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"X-Bridge-Token": token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8", "replace"))
        except Exception:
            return e.code, None
    except Exception:
        return -1, None


def _poll_results(bridge_dir: Path, cmd_id: str, wait_timeout_sec: float, pickup_timeout_sec: float, poll_sec: float) -> Dict[str, Any]:
    results_dir = bridge_dir / "cmd_results"
    log_p, status_p, res_p = (results_dir / f"{cmd_id}.log", results_dir / f"{cmd_id}.status.json", results_dir / f"{cmd_id}.result.json")
    t0 = time.time()
    offset = 0
    picked = False
    while True:
        refresh_dir(results_dir)
        if log_p.exists():
            picked = True
            try:
                raw = log_p.read_bytes()
                if offset < len(raw):
                    sys.stdout.write(raw[offset:].decode("utf-8", "replace"))
                    sys.stdout.flush()
                    offset = len(raw)
            except Exception:
                pass
        if status_p.exists():
            picked = True
        res = read_json_file(res_p)
        if res is not None:
            try:
                raw = log_p.read_bytes() if log_p.exists() else b""
                if offset < len(raw):
                    sys.stdout.write(raw[offset:].decode("utf-8", "replace"))
            except Exception:
                pass
            return res
        elapsed = time.time() - t0
        if not picked and elapsed > pickup_timeout_sec:
            return {"cmd_id": cmd_id, "status": "CONTROLLER_PICKUP_TIMEOUT", "returncode": 3,
                    "error": f"{pickup_timeout_sec / 60:.1f} 分钟内远端未接收（可能离线或 Drive 同步延迟）；指令仍在队列中，可稍后用 --action result 查询"}
        if elapsed > wait_timeout_sec:
            return {"cmd_id": cmd_id, "status": "CONTROLLER_WAIT_TIMEOUT", "returncode": 4,
                    "error": f"等待超过 {wait_timeout_sec / 60:.1f} 分钟；远端可能仍在运行，可稍后用 --action result 查询"}
        time.sleep(poll_sec)


def action_submit(bridge_dir: Path, args: argparse.Namespace, remote_args: List[str]) -> int:
    action = args.action
    params: Dict[str, str] = {}
    for kv in args.param or []:
        if "=" not in kv:
            print(f"错误: --param 需为 key=value 形式: {kv}", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        params[k.strip()] = v
    if action == "raw_shell" and not args.cmd.strip():
        print("错误: raw_shell 需要 --cmd（且远端必须以 --allow-raw-shell 启动）", file=sys.stderr)
        return 2
    cmd_id = args.cmd_id.strip() or f"cmd_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"
    spec: Dict[str, Any] = {"cmd_id": cmd_id, "action": action, "args": remote_args, "params": params,
                            "created_at": time.time(), "controller_version": CONTROLLER_VERSION}
    if args.timeout_min is not None:
        spec["timeout_sec"] = float(args.timeout_min) * 60.0
    if action == "raw_shell":
        spec["command"] = args.cmd

    st = heartbeat_state(bridge_dir, args.max_age_sec)
    if not st["online"]:
        print(f"[Controller-警告] 远端当前不在线（{st.get('reason') or ('心跳年龄 %ss' % st.get('heartbeat_age_sec'))}），指令将留在队列中等待桥上线。", flush=True)

    hb = st.get("heartbeat") or {}
    if args.prefer_channel == "tunnel" or (args.prefer_channel == "auto" and hb.get("tunnel_url")):
        url, token = str(hb.get("tunnel_url") or ""), str(hb.get("bridge_token") or "")
        if url and token:
            code, resp = http_json(f"{url.rstrip('/')}/exec", token, "POST", spec)
            if code in (200, 202):
                print(f"[Controller] 已通过隧道投递 {cmd_id}（state={resp.get('state') if resp else None}）", flush=True)
                if code == 200 or not args.wait:
                    print(json.dumps(resp, ensure_ascii=False, indent=2))
                    return int(((resp or {}).get("result") or {}).get("returncode", 0))
        if args.prefer_channel == "tunnel":
            print("[Controller] 隧道不可用（默认关闭），改用 Drive 队列。", flush=True)

    queue_dir = bridge_dir / "cmd_queue"
    cmd_file = queue_dir / f"{cmd_id}.cmd.json"
    atomic_fsync_write_json(cmd_file, spec)
    print(f"[Controller] 已投递到 Drive 队列: {cmd_file}", flush=True)
    if not args.wait:
        print(json.dumps({"submitted": True, "cmd_id": cmd_id, "cmd_file": str(cmd_file)}, ensure_ascii=False))
        return 0
    if args.wait_timeout_min is not None:
        wait_min = float(args.wait_timeout_min)
    elif action in BUILTIN_REMOTE:
        wait_min = 15.0
    else:
        wait_min = (float(args.timeout_min) if args.timeout_min is not None else 60.0) + 10.0
    res = _poll_results(bridge_dir, cmd_id, wait_min * 60.0, float(args.pickup_timeout_min) * 60.0, float(args.poll_interval))
    print("\n[Controller-Result] " + json.dumps({k: v for k, v in res.items() if k != "tail_output"}, ensure_ascii=False))
    return int(res.get("returncode", 1))


def action_result(bridge_dir: Path, cmd_id: str) -> int:
    results_dir = bridge_dir / "cmd_results"
    payload: Dict[str, Any] = {"cmd_id": cmd_id, "bridge_dir": str(bridge_dir)}
    res = read_json_file(results_dir / f"{cmd_id}.result.json")
    st = read_json_file(results_dir / f"{cmd_id}.status.json")
    if res is not None:
        payload["result"] = res
    elif st is not None:
        payload["status_info"] = st
    else:
        refresh_dir(bridge_dir / "cmd_queue")
        payload["status"] = "QUEUED" if (bridge_dir / "cmd_queue" / f"{cmd_id}.cmd.json").exists() else "NOT_FOUND"
    log_p = results_dir / f"{cmd_id}.log"
    if log_p.exists():
        payload["log_tail"] = log_p.read_text(encoding="utf-8", errors="replace")[-2000:]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def action_stop_bridge(bridge_dir: Path) -> int:
    flag = bridge_dir / "STOP_BRIDGE"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " stop requested by local controller\n", encoding="utf-8")
    print(json.dumps({"stop_flag_written": str(flag), "note": "远端桥在下一次轮询（Drive 同步后）检测到标志即退出；运行中的任务会被终止"},
                     ensure_ascii=False, indent=2))
    return 0


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(description="SimDT 本地总控 Colab 遥控客户端 v2.0（白名单动作 + Drive 队列）")
    p.add_argument("--action", required=True, choices=list(LOCAL_ACTIONS) + list(REMOTE_ACTIONS))
    p.add_argument("--bridge-dir", type=str, default="")
    p.add_argument("--wait", action="store_true", help="等待远端结果并实时打印日志")
    p.add_argument("--timeout-min", type=float, default=None, help="远端任务超时（分钟；默认按动作：setup_env 60、stage_assets 120、run_phase2 720、run_probe 180、extract_episodes 60）")
    p.add_argument("--wait-timeout-min", type=float, default=None, help="本地等待上限（分钟；默认 内建动作 15，脚本动作 timeout+10）")
    p.add_argument("--pickup-timeout-min", type=float, default=DEFAULT_PICKUP_TIMEOUT_MIN, help="远端未接收的判定时间（分钟，默认 10）")
    p.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_SEC, help="轮询间隔秒（默认 10）")
    p.add_argument("--max-age-sec", type=float, default=DEFAULT_MAX_AGE_SEC, help="心跳在线阈值秒（默认 180）")
    p.add_argument("--prefer-channel", choices=["auto", "gdrive", "tunnel"], default="gdrive",
                   help="默认 gdrive；auto 仅在远端显式开启隧道时才走隧道")
    p.add_argument("--param", action="append", help="内建动作参数 key=value（如 tail_log: cmd_id=... / path=... / lines=200；cancel: cmd_id=...）")
    p.add_argument("--cmd", type=str, default="", help="仅 raw_shell 使用")
    p.add_argument("--cmd-id", type=str, default="", help="指定/查询的 cmd_id")
    argv = sys.argv[1:]
    remote_args: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, remote_args = argv[:i], argv[i + 1:]
    return p.parse_args(argv), remote_args


def main() -> None:
    args, remote_args = parse_args()
    bridge_dir = Path(args.bridge_dir) if args.bridge_dir else resolve_default_bridge_dir()
    if args.action == "ping":
        sys.exit(action_ping(bridge_dir, args.max_age_sec))
    if args.action == "result":
        if not args.cmd_id.strip():
            print("错误: --action result 需要 --cmd-id", file=sys.stderr)
            sys.exit(2)
        sys.exit(action_result(bridge_dir, args.cmd_id.strip()))
    if args.action == "stop-bridge":
        sys.exit(action_stop_bridge(bridge_dir))
    sys.exit(action_submit(bridge_dir, args, remote_args))


if __name__ == "__main__":
    main()
