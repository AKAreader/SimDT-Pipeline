#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimDT 围点打援 —— 本地总控遥控客户端 (Local Colab Remote Controller)
====================================================================
供本地总控 AI 直接调用，全自动控制远端 Google Colab A100/L4 服务器：
1. `--action ping`:
   检查 `Z:\\SiamDT_Project\\colab\\bridge\\heartbeat.json`，判断 Colab 是否在线、GPU 型号（A100/L4）、
   显存与磁盘空间、心跳距今秒数、以及是否有可用直连 `tunnel_url`。
2. `--action exec --cmd "<shell command>" --wait`:
   - 优先尝试通过 `heartbeat.json` 中的 `tunnel_url` (通道 B: Cloudflare HTTP 隧道) 直接下发命令并实时流式拉取 `/logs`；
   - 若无 `tunnel_url` 或直连超时，自动无缝切换至 通道 A (Google Drive 文件队列)，向 `cmd_queue/` 投递 `{cmd_id}.cmd.json`，
     并实时轮询流式打印 `cmd_results/{cmd_id}.log` 直至拿到 `cmd_results/{cmd_id}.result.json` 回执！
3. `--action status --cmd-id <cmd_id>`:
   查询指定任务的状态与日志尾部输出。
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def resolve_default_bridge_dir() -> Path:
    env_dir = os.environ.get("SIAMDT_BRIDGE_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    env_drive = os.environ.get("SIAMDT_DRIVE_ROOT", "").strip()
    if env_drive:
        return Path(env_drive) / "colab" / "bridge"
    for cand in (
        Path(r"Z:\SiamDT_Project\colab\bridge"),
        Path(r"G:\我的云端硬盘\SiamDT_Project\colab\bridge"),
        Path("/content/drive/MyDrive/SiamDT_Project/colab/bridge"),
    ):
        if cand.parent.exists():
            return cand
    return Path(r"Z:\SiamDT_Project\colab\bridge")


def atomic_fsync_write_json(target_path: Path, payload: Dict[str, Any]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(target_path))
    except Exception:
        with target_path.open("w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())


def read_heartbeat(bridge_dir: Path) -> Optional[Dict[str, Any]]:
    hb_file = bridge_dir / "heartbeat.json"
    if not hb_file.exists():
        return None
    for _ in range(3):
        try:
            text = hb_file.read_text(encoding="utf-8").strip()
            if text:
                return json.loads(text)
        except Exception:
            time.sleep(0.15)
    return None


def http_json_request(
    url: str,
    token: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    headers = {
        "X-Bridge-Token": token,
        "Accept": "application/json",
    }
    data_bytes = None
    if payload is not None:
        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return True, json.loads(raw), ""
    except Exception as exc:
        return False, None, str(exc)


def action_ping(bridge_dir: Path, max_age_sec: float = 45.0, check_http: bool = True) -> int:
    hb = read_heartbeat(bridge_dir)
    if hb is None:
        report = {
            "online": False,
            "reason": f"未找到心跳文件: {bridge_dir / 'heartbeat.json'} (请在 Colab 中点击 Cell 0 启动远程接管桥)",
            "bridge_dir": str(bridge_dir),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    now = time.time()
    hb_epoch = float(hb.get("timestamp_epoch") or 0.0)
    age_sec = round(max(0.0, now - hb_epoch), 2)
    is_fresh = age_sec <= max_age_sec

    tunnel_url = str(hb.get("tunnel_url") or "").strip()
    http_listen = str(hb.get("http_listen") or "").strip()
    token = str(hb.get("bridge_token") or "").strip()
    same_host = (str(hb.get("hostname") or "") == socket.gethostname())

    direct_endpoint = ""
    direct_reachable = False
    if check_http and token:
        url_candidates = [tunnel_url] if not same_host else [tunnel_url, http_listen]
        for base_url in [u for u in url_candidates if u]:
            ok, data, _ = http_json_request(f"{base_url.rstrip('/')}/status", token=token, method="GET", timeout=3.0)
            if ok and isinstance(data, dict):
                direct_endpoint = base_url.rstrip("/")
                direct_reachable = True
                break

    report = {
        "online": bool(is_fresh or direct_reachable),
        "heartbeat_fresh": is_fresh,
        "heartbeat_age_sec": age_sec,
        "max_allowed_age_sec": max_age_sec,
        "gpu_name": hb.get("gpu_name", "UNKNOWN"),
        "gpu_tier": hb.get("gpu_tier", "UNKNOWN"),
        "vram_total_mb": hb.get("vram_total_mb", 0.0),
        "vram_used_mb": hb.get("vram_used_mb", 0.0),
        "vram_free_mb": hb.get("vram_free_mb", 0.0),
        "drive_free_gb": hb.get("drive_free_gb", 0.0),
        "runtime_disk_free_gb": hb.get("runtime_disk_free_gb", 0.0),
        "tunnel_url": tunnel_url,
        "direct_http_endpoint": direct_endpoint,
        "direct_http_reachable": direct_reachable,
        "channels_active": hb.get("channels_active", ["GDRIVE_FILE_QUEUE"]),
        "active_cmd_id": hb.get("active_cmd_id"),
        "processed_cmd_count": hb.get("processed_cmd_count", 0),
        "hostname": hb.get("hostname"),
        "pid": hb.get("pid"),
        "heartbeat_timestamp_utc": hb.get("timestamp_utc"),
        "bridge_dir": str(bridge_dir),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["online"] else 2


def _exec_via_http_endpoint(
    base_url: str,
    token: str,
    spec: Dict[str, Any],
    wait: bool,
    poll_interval: float,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    cmd_id = spec["cmd_id"]
    payload = dict(spec)
    payload["async"] = True
    ok, resp, err = http_json_request(
        f"{base_url.rstrip('/')}/exec",
        token=token,
        method="POST",
        payload=payload,
        timeout=6.0,
    )
    if not ok or not resp or not resp.get("accepted"):
        return False, None

    print(f"[Controller] 已通过直连 HTTP 通道 ({base_url}) 投递指令: {cmd_id}", flush=True)
    if not wait:
        return True, resp

    offset = 0
    t0 = time.time()
    timeout_sec = float(spec.get("timeout_sec") or 3600.0)
    while True:
        qs = urllib.parse.urlencode({"cmd_id": cmd_id, "offset": offset})
        ok_log, log_data, _ = http_json_request(
            f"{base_url.rstrip('/')}/logs?{qs}",
            token=token,
            method="GET",
            timeout=6.0,
        )
        if not ok_log or not isinstance(log_data, dict):
            # 若 HTTP 隧道中途波动，交由调用方无缝回退至 Google Drive 文件轮询
            return False, None
        chunk = log_data.get("log_chunk") or ""
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        offset = int(log_data.get("next_offset") or offset)
        if log_data.get("finished") and isinstance(log_data.get("result"), dict):
            return True, log_data["result"]
        if (time.time() - t0) > (timeout_sec + 30.0):
            return False, None
        time.sleep(poll_interval)


def _poll_gdrive_results(
    bridge_dir: Path,
    cmd_id: str,
    timeout_sec: float,
    poll_interval: float,
    initial_offset: int = 0,
) -> Dict[str, Any]:
    results_dir = bridge_dir / "cmd_results"
    log_path = results_dir / f"{cmd_id}.log"
    res_path = results_dir / f"{cmd_id}.result.json"

    offset = initial_offset
    t0 = time.time()

    while True:
        if log_path.exists():
            try:
                raw_bytes = log_path.read_bytes()
                if offset < len(raw_bytes):
                    chunk = raw_bytes[offset:].decode("utf-8", errors="replace")
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    offset = len(raw_bytes)
            except Exception:
                pass

        if res_path.exists():
            try:
                res_text = res_path.read_text(encoding="utf-8").strip()
                if res_text:
                    res_obj = json.loads(res_text)
                    # 再读一次确保尾部日志全部打出
                    if log_path.exists():
                        raw_bytes = log_path.read_bytes()
                        if offset < len(raw_bytes):
                            sys.stdout.write(raw_bytes[offset:].decode("utf-8", errors="replace"))
                            sys.stdout.flush()
                    return res_obj
            except Exception:
                pass

        if (time.time() - t0) > (timeout_sec + 60.0):
            return {
                "cmd_id": cmd_id,
                "status": "CONTROLLER_WAIT_TIMEOUT",
                "returncode": -99,
                "elapsed_sec": round(time.time() - t0, 2),
                "error": f"等待 Google Drive 回执超时 ({timeout_sec}s): {res_path}",
            }

        time.sleep(poll_interval)


def action_exec(
    bridge_dir: Path,
    command: str,
    cwd: str = "",
    timeout_sec: float = 3600.0,
    wait: bool = True,
    prefer_channel: str = "auto",
    poll_interval: float = 1.0,
    cmd_id: str = "",
) -> int:
    queue_dir = bridge_dir / "cmd_queue"
    results_dir = bridge_dir / "cmd_results"
    queue_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    resolved_cmd_id = cmd_id.strip() or f"cmd_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"
    spec: Dict[str, Any] = {
        "cmd_id": resolved_cmd_id,
        "command": command,
        "cwd": cwd if cwd else None,
        "timeout_sec": timeout_sec,
        "created_at": time.time(),
    }

    # 1. 若允许使用直连隧道 (auto 或 tunnel)，优先尝试 HTTP 通道
    hb = read_heartbeat(bridge_dir)
    if prefer_channel in ("auto", "tunnel") and hb is not None:
        token = str(hb.get("bridge_token") or "").strip()
        tunnel_url = str(hb.get("tunnel_url") or "").strip()
        http_listen = str(hb.get("http_listen") or "").strip()
        same_host = (str(hb.get("hostname") or "") == socket.gethostname())
        candidates = [u for u in ([tunnel_url] if not same_host else [tunnel_url, http_listen]) if u]
        for endpoint in candidates:
            ok, res_obj = _exec_via_http_endpoint(
                base_url=endpoint,
                token=token,
                spec=spec,
                wait=wait,
                poll_interval=poll_interval,
            )
            if ok and res_obj is not None:
                print("\n[Controller-Result] " + json.dumps(res_obj, ensure_ascii=False))
                return int(res_obj.get("returncode", 0)) if wait else 0

    # 2. 通道 A：向 Google Drive 文件夹队列写入 {cmd_id}.cmd.json
    cmd_file = queue_dir / f"{resolved_cmd_id}.cmd.json"
    atomic_fsync_write_json(cmd_file, spec)
    print(f"[Controller] 已向 Google Drive 队列投递指令: {cmd_file}", flush=True)

    if not wait:
        print(json.dumps({"submitted": True, "cmd_id": resolved_cmd_id, "cmd_file": str(cmd_file)}, ensure_ascii=False))
        return 0

    res_obj = _poll_gdrive_results(
        bridge_dir=bridge_dir,
        cmd_id=resolved_cmd_id,
        timeout_sec=timeout_sec,
        poll_interval=poll_interval,
    )
    print("\n[Controller-Result] " + json.dumps(res_obj, ensure_ascii=False))
    return int(res_obj.get("returncode", 1))


def action_status(bridge_dir: Path, cmd_id: str) -> int:
    results_dir = bridge_dir / "cmd_results"
    res_file = results_dir / f"{cmd_id}.result.json"
    status_file = results_dir / f"{cmd_id}.status.json"
    log_file = results_dir / f"{cmd_id}.log"

    payload: Dict[str, Any] = {"cmd_id": cmd_id, "bridge_dir": str(bridge_dir)}
    if res_file.exists():
        payload["result"] = json.loads(res_file.read_text(encoding="utf-8"))
    elif status_file.exists():
        payload["status_info"] = json.loads(status_file.read_text(encoding="utf-8"))
    else:
        payload["status"] = "PENDING_OR_NOT_FOUND"

    if log_file.exists():
        text = log_file.read_text(encoding="utf-8", errors="replace")
        payload["log_tail"] = text[-2000:]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimDT 本地总控 Colab 遥控客户端")
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=["ping", "exec", "status"],
        help="执行动作: ping (检查心跳与GPU状态) | exec (投递并执行远程命令) | status (查询任务状态)",
    )
    parser.add_argument(
        "--bridge-dir",
        type=str,
        default="",
        help="网桥目录路径 (默认 Z:\\SiamDT_Project\\colab\\bridge)",
    )
    parser.add_argument("--cmd", type=str, default="", help="--action exec 时要执行的 Shell 命令")
    parser.add_argument("--cwd", type=str, default="", help="远程命令工作目录 (可选)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="远程命令超时秒数")
    parser.add_argument("--wait", action="store_true", help="等待远程命令执行完成并实时流式打印日志")
    parser.add_argument(
        "--prefer-channel",
        type=str,
        default="auto",
        choices=["auto", "tunnel", "gdrive"],
        help="通道偏好: auto (优先隧道，失败自动回退 Google Drive 队列) | tunnel | gdrive",
    )
    parser.add_argument("--cmd-id", type=str, default="", help="指定或查询的 cmd_id")
    parser.add_argument("--max-age-sec", type=float, default=45.0, help="ping 判定在线的最大心跳年龄(秒)")
    parser.add_argument("--poll-interval", type=float, default=0.8, help="轮询日志间隔(秒)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bridge_dir = Path(args.bridge_dir) if args.bridge_dir else resolve_default_bridge_dir()

    if args.action == "ping":
        rc = action_ping(bridge_dir=bridge_dir, max_age_sec=args.max_age_sec)
        sys.exit(rc)
    elif args.action == "exec":
        if not args.cmd.strip():
            print("错误: --action exec 必须提供 --cmd 参数", file=sys.stderr)
            sys.exit(2)
        rc = action_exec(
            bridge_dir=bridge_dir,
            command=args.cmd,
            cwd=args.cwd,
            timeout_sec=args.timeout,
            wait=args.wait,
            prefer_channel=args.prefer_channel,
            poll_interval=args.poll_interval,
            cmd_id=args.cmd_id,
        )
        sys.exit(rc)
    elif args.action == "status":
        if not args.cmd_id.strip():
            print("错误: --action status 必须提供 --cmd-id 参数", file=sys.stderr)
            sys.exit(2)
        rc = action_status(bridge_dir=bridge_dir, cmd_id=args.cmd_id.strip())
        sys.exit(rc)


if __name__ == "__main__":
    main()
