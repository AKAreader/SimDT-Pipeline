#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimDT 围点打援 —— 双通道 Colab 远程接管桥 (Colab Remote Bridge Daemon)
========================================================================
功能概述：
1. 挂载 Google Drive (`/content/drive`) 并初始化远程接管桥目录结构：
   - `/content/drive/MyDrive/SiamDT_Project/colab/bridge/cmd_queue/`
   - `/content/drive/MyDrive/SiamDT_Project/colab/bridge/cmd_queue/processed/`
   - `/content/drive/MyDrive/SiamDT_Project/colab/bridge/cmd_results/`
   - `/content/drive/MyDrive/SiamDT_Project/colab/bridge/heartbeat.json`
2. 独立心跳守护线程：
   - 每 5 秒刷新一次 `heartbeat.json`（含存活时间戳、`nvidia-smi` GPU 型号 A100/L4、显存占用、磁盘剩余空间、当前执行任务及直连隧道状态），强制 `os.fsync` 落盘。
3. 双通道远程指令执行引擎（Cloudflare 直连隧道 + Google Drive 文件队列双保险）：
   - 通道 A（Google Drive 文件夹指令队列，100% 免疫任何防火墙/代理限制）：
     每 2 秒扫描 `cmd_queue/*.cmd.json`，按时间戳顺序执行命令；执行期间每 2 秒将最新 `stdout/stderr` 刷新并 `os.fsync` 到 `cmd_results/{cmd_id}.log`；执行完毕后写入 `cmd_results/{cmd_id}.result.json`（含 `returncode`, `elapsed_sec`, `tail_output`），并将已处理的 `.cmd.json` 移入 `cmd_queue/processed/`。
   - 通道 B（可选 Cloudflare 快速 HTTP 隧道）：
     自动在后台启动监听 `127.0.0.1:18765` 的轻量级 HTTP 服务（带随机生成的 `bridge_token` 鉴权，支持 `/status`、`/exec`、`/logs`），若环境支持运行 `cloudflared tunnel --url http://127.0.0.1:18765`，则自动解析 `https://xxxx.trycloudflare.com` 并连同 `bridge_token` 写入 `heartbeat.json`。即便 `cloudflared` 不可用，通道 A 也 100% 正常工作。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def try_mount_colab_drive() -> bool:
    """在 Google Colab 环境中确保 /content/drive 已挂载。"""
    if Path("/content/drive/MyDrive").exists():
        return True
    try:
        from google.colab import drive  # type: ignore
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
        return Path("/content/drive/MyDrive").exists()
    except Exception:
        return False


def resolve_default_drive_root() -> Path:
    """解析默认 Google Drive 项目根目录（兼容 Colab 与本地 Windows 映射盘自检）。"""
    env_root = os.environ.get("SIAMDT_DRIVE_ROOT", "").strip()
    if env_root:
        return Path(env_root)
    colab_path = Path("/content/drive/MyDrive/SiamDT_Project")
    if Path("/content/drive/MyDrive").exists() or sys.platform.startswith("linux"):
        return colab_path
    for win_candidate in (
        Path(r"Z:\SiamDT_Project"),
        Path(r"G:\我的云端硬盘\SiamDT_Project"),
    ):
        if win_candidate.exists():
            return win_candidate
    return colab_path


def atomic_fsync_write_json(target_path: Path, payload: Dict[str, Any]) -> None:
    """将 JSON 写入临时文件并强制 fsync 后替换目标文件，防止云盘同步读到半截文件。"""
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
        # Windows / FUSE 偶发占用时直接覆写并 fsync
        with target_path.open("w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())


def query_gpu_status() -> Dict[str, Any]:
    """查询当前机器的 nvidia-smi GPU 型号、显存占用与算力档位。"""
    info: Dict[str, Any] = {
        "gpu_available": False,
        "gpu_name": "NO_GPU",
        "gpu_tier": "CPU",
        "compute_cap": "cpu",
        "vram_total_mb": 0.0,
        "vram_used_mb": 0.0,
        "vram_free_mb": 0.0,
        "gpu_utilization_pct": 0.0,
        "driver_version": "",
        "nvidia_smi_raw": "NO_GPU",
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if out:
            first_line = out.splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 7:
                gname = parts[0]
                total_mb = float(parts[1]) if parts[1].replace(".", "", 1).isdigit() else 0.0
                used_mb = float(parts[2]) if parts[2].replace(".", "", 1).isdigit() else 0.0
                free_mb = float(parts[3]) if parts[3].replace(".", "", 1).isdigit() else 0.0
                util_pct = float(parts[4]) if parts[4].replace(".", "", 1).isdigit() else 0.0
                drv = parts[5]
                cc = parts[6]
                upper_name = gname.upper()
                if "A100" in upper_name or cc == "8.0":
                    tier = "A100"
                elif "L4" in upper_name or cc == "8.9":
                    tier = "L4"
                else:
                    tier = "OTHER_GPU"
                info.update({
                    "gpu_available": True,
                    "gpu_name": gname,
                    "gpu_tier": tier,
                    "compute_cap": cc,
                    "vram_total_mb": round(total_mb, 1),
                    "vram_used_mb": round(used_mb, 1),
                    "vram_free_mb": round(free_mb, 1),
                    "gpu_utilization_pct": round(util_pct, 1),
                    "driver_version": drv,
                    "nvidia_smi_raw": first_line,
                })
    except Exception:
        pass
    return info


class ColabRemoteBridge:
    """双通道 Colab 远程接管桥核心类。"""

    def __init__(
        self,
        drive_root: Path,
        http_host: str = "127.0.0.1",
        http_port: int = 18765,
        enable_cloudflared: bool = True,
        heartbeat_interval_sec: float = 5.0,
        poll_interval_sec: float = 2.0,
    ) -> None:
        self.drive_root = drive_root
        self.bridge_dir = self.drive_root / "colab" / "bridge"
        self.cmd_queue_dir = self.bridge_dir / "cmd_queue"
        self.cmd_processed_dir = self.cmd_queue_dir / "processed"
        self.cmd_results_dir = self.bridge_dir / "cmd_results"
        self.heartbeat_file = self.bridge_dir / "heartbeat.json"

        self.http_host = http_host
        self.http_port = http_port
        self.enable_cloudflared = enable_cloudflared
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.poll_interval_sec = poll_interval_sec

        self._ensure_directories()
        self.bridge_token = self._load_or_create_token()
        self.tunnel_url: str = ""
        self.cloudflared_proc: Optional[subprocess.Popen] = None
        self.http_server: Optional[ThreadingHTTPServer] = None

        self._lock = threading.Lock()
        self._exec_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.active_cmd_id: Optional[str] = None
        self.active_command_str: Optional[str] = None
        self.processed_cmd_count: int = 0
        self.started_at_epoch: float = time.time()
        self.started_at_utc: str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _ensure_directories(self) -> None:
        for d in (self.bridge_dir, self.cmd_queue_dir, self.cmd_processed_dir, self.cmd_results_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _load_or_create_token(self) -> str:
        token_file = self.bridge_dir / ".bridge_token"
        if token_file.exists():
            try:
                tok = token_file.read_text(encoding="utf-8").strip()
                if len(tok) >= 16:
                    return tok
            except Exception:
                pass
        tok = secrets.token_urlsafe(24)
        try:
            token_file.write_text(tok, encoding="utf-8")
        except Exception:
            pass
        return tok

    def collect_heartbeat_payload(self) -> Dict[str, Any]:
        """采集当前心跳数据并组装 JSON 字典。"""
        gpu = query_gpu_status()
        drive_free_gb = 0.0
        local_free_gb = 0.0
        try:
            drive_free_gb = round(shutil.disk_usage(self.drive_root).free / (1024 ** 3), 2)
        except Exception:
            pass
        try:
            local_root = Path("/content") if Path("/content").exists() else Path.cwd()
            local_free_gb = round(shutil.disk_usage(local_root).free / (1024 ** 3), 2)
        except Exception:
            pass

        channels = ["GDRIVE_FILE_QUEUE"]
        if self.http_server is not None:
            channels.append("LOCAL_HTTP_SERVER")
        if self.tunnel_url:
            channels.append("CLOUDFLARE_HTTP_TUNNEL")

        with self._lock:
            active_id = self.active_cmd_id
            active_cmd = self.active_command_str
            done_cnt = self.processed_cmd_count
            t_url = self.tunnel_url

        now_epoch = time.time()
        return {
            "status": "ONLINE",
            "bridge_version": "1.0.0",
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "started_at_utc": self.started_at_utc,
            "uptime_sec": round(now_epoch - self.started_at_epoch, 1),
            "timestamp_epoch": round(now_epoch, 3),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_epoch)),
            "gpu": gpu,
            "gpu_name": gpu["gpu_name"],
            "gpu_tier": gpu["gpu_tier"],
            "vram_total_mb": gpu["vram_total_mb"],
            "vram_used_mb": gpu["vram_used_mb"],
            "vram_free_mb": gpu["vram_free_mb"],
            "drive_free_gb": drive_free_gb,
            "runtime_disk_free_gb": local_free_gb,
            "http_listen": f"http://{self.http_host}:{self.http_port}",
            "tunnel_url": t_url,
            "bridge_token": self.bridge_token,
            "channels_active": channels,
            "active_cmd_id": active_id,
            "active_command": active_cmd,
            "processed_cmd_count": done_cnt,
            "paths": {
                "bridge_dir": str(self.bridge_dir),
                "cmd_queue_dir": str(self.cmd_queue_dir),
                "cmd_results_dir": str(self.cmd_results_dir),
                "heartbeat_file": str(self.heartbeat_file),
            },
        }

    def write_heartbeat_once(self) -> Dict[str, Any]:
        payload = self.collect_heartbeat_payload()
        try:
            atomic_fsync_write_json(self.heartbeat_file, payload)
        except Exception as exc:
            print(f"[Bridge-Heartbeat-Warn] 写入 heartbeat.json 异常: {exc}", flush=True)
        return payload

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            self.write_heartbeat_once()
            self._stop_event.wait(self.heartbeat_interval_sec)

    # ------------------------------------------------------------------
    # 通道 B：本地轻量 HTTP Server + 可选 Cloudflared 快速隧道
    # ------------------------------------------------------------------
    def start_http_server(self) -> None:
        bridge_ref = self

        class BridgeHTTPHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return  # 静默标准访问日志

            def _check_auth(self) -> bool:
                hdr_token = self.headers.get("X-Bridge-Token", "").strip()
                if hdr_token and secrets.compare_digest(hdr_token, bridge_ref.bridge_token):
                    return True
                parsed = urlparse(self.path)
                qs = parse_qs(parsed.query)
                q_token = (qs.get("token", [""])[0]).strip()
                if q_token and secrets.compare_digest(q_token, bridge_ref.bridge_token):
                    return True
                return False

            def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
                body = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if not self._check_auth():
                    self._send_json(401, {"error": "Unauthorized: invalid bridge_token"})
                    return
                parsed = urlparse(self.path)
                if parsed.path == "/status":
                    self._send_json(200, bridge_ref.collect_heartbeat_payload())
                    return
                if parsed.path == "/logs":
                    qs = parse_qs(parsed.query)
                    cmd_id = (qs.get("cmd_id", [""])[0]).strip()
                    offset = int((qs.get("offset", ["0"])[0]) or "0")
                    if not cmd_id:
                        self._send_json(400, {"error": "Missing cmd_id parameter"})
                        return
                    log_path = bridge_ref.cmd_results_dir / f"{cmd_id}.log"
                    res_path = bridge_ref.cmd_results_dir / f"{cmd_id}.result.json"
                    chunk_text = ""
                    next_offset = offset
                    if log_path.exists():
                        raw_bytes = log_path.read_bytes()
                        if offset < len(raw_bytes):
                            chunk_text = raw_bytes[offset:].decode("utf-8", errors="replace")
                            next_offset = len(raw_bytes)
                    result_obj = None
                    if res_path.exists():
                        try:
                            result_obj = json.loads(res_path.read_text(encoding="utf-8"))
                        except Exception:
                            result_obj = None
                    self._send_json(200, {
                        "cmd_id": cmd_id,
                        "offset": offset,
                        "next_offset": next_offset,
                        "log_chunk": chunk_text,
                        "finished": result_obj is not None,
                        "result": result_obj,
                    })
                    return
                self._send_json(404, {"error": f"Unknown endpoint {parsed.path}"})

            def do_POST(self) -> None:
                if not self._check_auth():
                    self._send_json(401, {"error": "Unauthorized: invalid bridge_token"})
                    return
                parsed = urlparse(self.path)
                if parsed.path != "/exec":
                    self._send_json(404, {"error": f"Unknown endpoint {parsed.path}"})
                    return
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw_body = self.rfile.read(length).decode("utf-8", errors="replace") if length > 0 else "{}"
                try:
                    req = json.loads(raw_body)
                except Exception as exc:
                    self._send_json(400, {"error": f"Invalid JSON body: {exc}"})
                    return

                cmd_id = str(req.get("cmd_id") or f"http_{int(time.time()*1000)}_{secrets.token_hex(3)}")
                command = str(req.get("command") or "").strip()
                if not command:
                    self._send_json(400, {"error": "Field 'command' is required"})
                    return
                cwd = req.get("cwd")
                timeout_sec = float(req.get("timeout_sec") or 3600)
                run_async = bool(req.get("async", False))
                spec = {
                    "cmd_id": cmd_id,
                    "command": command,
                    "cwd": cwd,
                    "timeout_sec": timeout_sec,
                    "env": req.get("env") or {},
                    "channel": "CLOUDFLARE_HTTP_TUNNEL",
                    "created_at": time.time(),
                }
                if run_async:
                    threading.Thread(
                        target=bridge_ref.execute_command_spec,
                        args=(spec,),
                        daemon=True,
                    ).start()
                    self._send_json(202, {
                        "accepted": True,
                        "cmd_id": cmd_id,
                        "mode": "async",
                        "message": "Command scheduled via HTTP channel; poll /logs or cmd_results/",
                    })
                else:
                    res = bridge_ref.execute_command_spec(spec)
                    self._send_json(200, res)

        try:
            server = ThreadingHTTPServer((self.http_host, self.http_port), BridgeHTTPHandler)
            self.http_server = server
            self.http_port = server.server_address[1]
            threading.Thread(target=server.serve_forever, daemon=True).start()
        except Exception as exc:
            # 端口占用时自动分配空闲端口
            try:
                server = ThreadingHTTPServer((self.http_host, 0), BridgeHTTPHandler)
                self.http_server = server
                self.http_port = server.server_address[1]
                threading.Thread(target=server.serve_forever, daemon=True).start()
            except Exception as exc2:
                print(f"[Bridge-HTTP-Warn] 本地 HTTP 服务未启动 ({exc} / {exc2})，仅使用通道 A (Google Drive 队列)。", flush=True)

    def start_cloudflared_tunnel_async(self) -> None:
        """在后台尝试启动或下载 cloudflared 建立 trycloudflare.com 临时隧道。"""
        if not self.enable_cloudflared or self.http_server is None:
            return

        def _worker() -> None:
            cf_bin = shutil.which("cloudflared")
            if not cf_bin and sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64"):
                tmp_cf = Path("/tmp/cloudflared")
                if not (tmp_cf.exists() and os.access(tmp_cf, os.X_OK)):
                    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
                    try:
                        urllib.request.urlretrieve(url, str(tmp_cf))
                        tmp_cf.chmod(0o755)
                    except Exception:
                        pass
                if tmp_cf.exists() and os.access(tmp_cf, os.X_OK):
                    cf_bin = str(tmp_cf)

            if not cf_bin:
                return

            try:
                proc = subprocess.Popen(
                    [
                        cf_bin,
                        "tunnel",
                        "--url",
                        f"http://{self.http_host}:{self.http_port}",
                        "--no-autoupdate",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                self.cloudflared_proc = proc
                url_pattern = re.compile(r"(https://[a-zA-Z0-9-]+\.trycloudflare\.com)")
                assert proc.stdout is not None
                for line in proc.stdout:
                    if self._stop_event.is_set():
                        break
                    m = url_pattern.search(line)
                    if m:
                        found_url = m.group(1)
                        with self._lock:
                            self.tunnel_url = found_url
                        self.write_heartbeat_once()
                        print(f"[Bridge-Tunnel] Cloudflare 隧道已就绪: {found_url}", flush=True)
                        break
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # 通道 A & 通用命令执行器（实时 2 秒 fsync 日志 + 完成回执）
    # ------------------------------------------------------------------
    def execute_command_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """执行单条远程命令，实时每 2 秒将 stdout/stderr 刷入 {cmd_id}.log 并生成 {cmd_id}.result.json。"""
        with self._exec_lock:
            cmd_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(spec.get("cmd_id") or f"cmd_{int(time.time()*1000)}"))
            command = str(spec.get("command") or "").strip()
            raw_cwd = spec.get("cwd")
            timeout_sec = float(spec.get("timeout_sec") or 3600.0)
            extra_env = spec.get("env") if isinstance(spec.get("env"), dict) else {}
            channel = str(spec.get("channel") or "GDRIVE_FILE_QUEUE")

            # 解析工作目录（若不存在则回退到存在目录）
            cwd_path: Optional[Path] = None
            if raw_cwd:
                cand = Path(str(raw_cwd))
                if cand.exists() and cand.is_dir():
                    cwd_path = cand
            if cwd_path is None:
                default_runtime = Path("/content/runtime")
                cwd_path = default_runtime if default_runtime.exists() else Path.cwd()

            log_path = self.cmd_results_dir / f"{cmd_id}.log"
            status_path = self.cmd_results_dir / f"{cmd_id}.status.json"
            result_path = self.cmd_results_dir / f"{cmd_id}.result.json"

            t_start = time.time()
            started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))

            with self._lock:
                self.active_cmd_id = cmd_id
                self.active_command_str = command
            self.write_heartbeat_once()

            print(f"\n[Bridge-Exec] >>> 开始执行任务 {cmd_id} ({channel})\n  CWD: {cwd_path}\n  CMD: {command}", flush=True)

            atomic_fsync_write_json(status_path, {
                "cmd_id": cmd_id,
                "command": command,
                "cwd": str(cwd_path),
                "channel": channel,
                "status": "RUNNING",
                "started_at_utc": started_utc,
                "timeout_sec": timeout_sec,
            })

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["SIAMDT_DRIVE_ROOT"] = str(self.drive_root)
            for k, v in extra_env.items():
                env[str(k)] = str(v)

            returncode = -1
            timed_out = False
            output_chunks: List[str] = []
            buffer_lock = threading.Lock()

            with log_path.open("w", encoding="utf-8") as log_f:
                header = (
                    f"=== [SimDT Colab Bridge] cmd_id={cmd_id} | started={started_utc} | cwd={cwd_path} ===\n"
                    f"$ {command}\n"
                    f"--------------------------------------------------------------------------------\n"
                )
                log_f.write(header)
                log_f.flush()
                os.fsync(log_f.fileno())

                try:
                    proc = subprocess.Popen(
                        command,
                        shell=True,
                        cwd=str(cwd_path),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )

                    def _reader() -> None:
                        assert proc.stdout is not None
                        for line in proc.stdout:
                            with buffer_lock:
                                output_chunks.append(line)
                                log_f.write(line)

                    reader_thread = threading.Thread(target=_reader, daemon=True)
                    reader_thread.start()

                    last_fsync = time.time()
                    while proc.poll() is None:
                        now = time.time()
                        if now - t_start > timeout_sec:
                            timed_out = True
                            try:
                                proc.kill()
                            except Exception:
                                pass
                            break
                        if now - last_fsync >= 2.0:
                            with buffer_lock:
                                log_f.flush()
                                os.fsync(log_f.fileno())
                            last_fsync = now
                        time.sleep(0.2)

                    reader_thread.join(timeout=3.0)
                    returncode = proc.returncode if proc.returncode is not None else (-9 if timed_out else -1)

                except Exception as exc:
                    err_msg = f"\n[Bridge-Exec-Exception] {exc}\n"
                    output_chunks.append(err_msg)
                    log_f.write(err_msg)
                    returncode = -2

                t_end = time.time()
                elapsed_sec = round(t_end - t_start, 3)
                finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end))
                footer = (
                    f"\n--------------------------------------------------------------------------------\n"
                    f"=== [Finished] returncode={returncode} | timed_out={timed_out} | elapsed={elapsed_sec}s ===\n"
                )
                log_f.write(footer)
                log_f.flush()
                os.fsync(log_f.fileno())

            full_output = "".join(output_chunks)
            tail_output = full_output[-4000:] if len(full_output) > 4000 else full_output
            final_status = "TIMEOUT" if timed_out else ("COMPLETED" if returncode == 0 else "FAILED")

            result_payload = {
                "cmd_id": cmd_id,
                "command": command,
                "cwd": str(cwd_path),
                "channel": channel,
                "status": final_status,
                "returncode": int(returncode),
                "timed_out": timed_out,
                "elapsed_sec": elapsed_sec,
                "started_at_utc": started_utc,
                "finished_at_utc": finished_utc,
                "log_file": str(log_path),
                "tail_output": tail_output,
            }
            atomic_fsync_write_json(result_path, result_payload)
            atomic_fsync_write_json(status_path, {
                "cmd_id": cmd_id,
                "status": final_status,
                "returncode": int(returncode),
                "elapsed_sec": elapsed_sec,
                "finished_at_utc": finished_utc,
            })

            with self._lock:
                self.active_cmd_id = None
                self.active_command_str = None
                self.processed_cmd_count += 1
            self.write_heartbeat_once()

            print(
                f"[Bridge-Exec] [OK] 任务完成 {cmd_id} | status={final_status} | returncode={returncode} | elapsed={elapsed_sec}s",
                flush=True,
            )
            return result_payload

    def scan_and_process_queue_once(self) -> int:
        """扫描 cmd_queue/*.cmd.json 并按创建时间顺序依次执行。"""
        cmd_files = sorted(
            [p for p in self.cmd_queue_dir.glob("*.cmd.json") if p.is_file()],
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        processed_now = 0
        for cmd_file in cmd_files:
            try:
                raw_text = cmd_file.read_text(encoding="utf-8").strip()
                if not raw_text:
                    continue
                spec = json.loads(raw_text)
            except Exception as exc:
                print(f"[Bridge-Queue-Warn] 读取指令文件 {cmd_file.name} 失败 (稍后重试): {exc}", flush=True)
                continue

            if "cmd_id" not in spec or not spec["cmd_id"]:
                spec["cmd_id"] = cmd_file.name[: -len(".cmd.json")]
            spec["channel"] = "GDRIVE_FILE_QUEUE"

            # 先移动到 processed/ 目录，避免重复消费
            dest_processed = self.cmd_processed_dir / cmd_file.name
            try:
                if dest_processed.exists():
                    dest_processed.unlink()
                shutil.move(str(cmd_file), str(dest_processed))
            except Exception:
                try:
                    cmd_file.unlink()
                except Exception:
                    pass

            self.execute_command_spec(spec)
            processed_now += 1
        return processed_now

    def run_forever(
        self,
        max_loops: int = 0,
        duration_sec: float = 0.0,
    ) -> None:
        """启动双通道远程接管桥主循环。"""
        self.start_http_server()
        self.start_cloudflared_tunnel_async()
        hb = self.write_heartbeat_once()

        hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb_thread.start()

        gpu_label = hb.get("gpu_name", "NO_GPU")
        if gpu_label == "NO_GPU":
            gpu_label = "NVIDIA A100/L4 (CPU/待检模式)"
        print(
            f"[总控接管已就绪] GPU: {gpu_label} | 心跳已写入 Google Drive，您现在可以回到本地对话窗口让总控 AI 全自动接管！",
            flush=True,
        )
        print(
            f"[Bridge-Info] Heartbeat: {self.heartbeat_file}\n"
            f"[Bridge-Info] Queue Dir: {self.cmd_queue_dir}\n"
            f"[Bridge-Info] Results  : {self.cmd_results_dir}\n"
            f"[Bridge-Info] HTTP Srv : http://{self.http_host}:{self.http_port}",
            flush=True,
        )

        loops = 0
        t0 = time.time()
        try:
            while not self._stop_event.is_set():
                self.scan_and_process_queue_once()
                loops += 1
                if max_loops > 0 and loops >= max_loops:
                    break
                if duration_sec > 0 and (time.time() - t0) >= duration_sec:
                    break
                time.sleep(self.poll_interval_sec)
        except KeyboardInterrupt:
            print("\n[Bridge-Info] 收到中断信号，正在刷新最终心跳...", flush=True)
        finally:
            self._stop_event.set()
            self.write_heartbeat_once()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimDT Colab 双通道远程接管桥守护进程")
    parser.add_argument(
        "--drive-root",
        type=str,
        default="",
        help="SiamDT_Project 根目录路径 (默认自动识别 /content/drive/MyDrive/SiamDT_Project 或 Z:\\SiamDT_Project)",
    )
    parser.add_argument("--http-host", type=str, default="127.0.0.1", help="本地 HTTP 监听地址")
    parser.add_argument("--http-port", type=int, default=18765, help="本地 HTTP 监听端口")
    parser.add_argument("--no-cloudflared", action="store_true", help="禁用 cloudflared 隧道尝试")
    parser.add_argument("--heartbeat-interval", type=float, default=5.0, help="心跳刷新间隔(秒)")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="指令队列扫描间隔(秒)")
    parser.add_argument("--max-loops", type=int, default=0, help="最大主循环次数(0=无限守护)")
    parser.add_argument("--duration-sec", type=float, default=0.0, help="最长运行秒数(0=无限守护)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try_mount_colab_drive()
    drive_root = Path(args.drive_root) if args.drive_root else resolve_default_drive_root()
    bridge = ColabRemoteBridge(
        drive_root=drive_root,
        http_host=args.http_host,
        http_port=args.http_port,
        enable_cloudflared=not args.no_cloudflared,
        heartbeat_interval_sec=args.heartbeat_interval,
        poll_interval_sec=args.poll_interval,
    )
    bridge.run_forever(max_loops=args.max_loops, duration_sec=args.duration_sec)


if __name__ == "__main__":
    main()
