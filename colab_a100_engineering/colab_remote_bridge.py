#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimDT 围点打援 —— Colab 远程接管桥 v2.0 (Colab Remote Bridge Daemon，2026-11-04 合规修订版)
========================================================================
依据 Claude × Gemini 交叉审查（review-think-20261104-001，P1-2 / P1-6）修订：
1. 默认只用 **Google Drive 文件队列通道**（cmd_queue/*.cmd.json -> cmd_results/）；
   cloudflared 公网隧道默认**关闭**，只有显式 `--enable-tunnel` 才启动，并在启动时打印 Colab FAQ
   关于 remote proxies 的禁止条款提醒（"disallowed from all managed Colab runtimes"）。
2. 取消 `shell=True` 任意命令执行，改为**动作白名单**：
   - 内建（不起子进程）：status / tail_log / cancel
   - 脚本动作（subprocess 列表参数，shell=False）：setup_env / stage_assets / run_phase2 / run_probe / extract_episodes
   - 调试用 raw_shell：仅当启动参数显式 `--allow-raw-shell` 时可用，默认拒绝。
3. 心跳间隔 30 s（控制器离线阈值 180 s）；队列扫描前先 os.listdir 刷新目录元数据；所有超时按分钟级设计。
4. 停止方式：在 bridge 目录写 `STOP_BRIDGE` 标志文件（或 `--local-stop-file` 指定的本地标志），
   桥会终止正在运行的任务（结果记为 CANCELLED_BY_BRIDGE_STOP）、写最终心跳后退出。

队列文件格式（由 local_colab_controller.py 生成）：
{"cmd_id": "...", "action": "run_phase2", "args": ["--action", "evaluate"], "params": {}, "timeout_sec": 3600}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BRIDGE_VERSION = "2.0.0"
DEFAULT_HEARTBEAT_SEC = 30.0
DEFAULT_POLL_SEC = 5.0
LOG_FSYNC_SEC = 10.0
MAX_TIMEOUT_SEC = 24 * 3600
MAX_PARSE_RETRIES = 6

COLAB_FAQ_REMOTE_PROXY_NOTICE = (
    "[合规提醒] Google Colab FAQ「Which activities are restricted in Colab?」写明以下行为"
    "\"disallowed from all managed Colab runtimes\"，其中包括 connecting to remote proxies（连接远程代理）。"
    "cloudflared/ngrok 一类公网隧道可能被认定为此类行为，可能导致运行时被终止或账号受限。"
    "仅在您已自行确认合规时才使用 --enable-tunnel；默认的 Google Drive 文件队列通道不需要隧道。"
    " 参考：https://research.google.com/colaboratory/faq.html"
)

BUILTIN_ACTIONS = ("status", "tail_log", "cancel")
SCRIPT_ACTIONS: Dict[str, Dict[str, Any]] = {
    "setup_env": {"script": "colab_a100_engineering/colab_init_cells.py", "fixed_args": ["--run-cell", "4"],
                  "allow_extra_args": False, "python": "system", "default_timeout_min": 60},
    "stage_assets": {"script": "colab_a100_engineering/colab_init_cells.py", "fixed_args": ["--run-cell", "1,2,3"],
                     "allow_extra_args": False, "python": "system", "default_timeout_min": 120},
    "run_phase2": {"script": "phase2_unified_eval/run_phase2_multi_tracker_colab.py", "fixed_args": [],
                   "allow_extra_args": True, "python": "system", "default_timeout_min": 720},
    "run_probe": {"script": "phase5_feature_diagnosis/siamdt_6layer_feature_probe.py", "fixed_args": [],
                  "allow_extra_args": True, "python": "tier1", "default_timeout_min": 180},
    "extract_episodes": {"script": "phase2_unified_eval/failure_episode_extractor.py", "fixed_args": [],
                         "allow_extra_args": True, "python": "system", "default_timeout_min": 60},
}
RAW_SHELL_ACTION = "raw_shell"
ALL_ACTIONS = BUILTIN_ACTIONS + tuple(SCRIPT_ACTIONS) + (RAW_SHELL_ACTION,)

# 参数白名单字符：字母数字、中文、常见路径/选项符号与空格；拒绝引号、反引号、$ ; | & < > 换行等
SAFE_ARG_RE = re.compile(r"^[A-Za-z0-9_./:=,+@%\- \u4e00-\u9fff]*$")
SAFE_ENV_KEY_RE = re.compile(r"^(SIAMDT_[A-Z0-9_]+|SIMDT_[A-Z0-9_]+|ANTI_UAV410_[A-Z0-9_]+|CUDA_VISIBLE_DEVICES)$")
TIER1_PYTHON_DEFAULT = "/content/envs/siamdt-cu118/bin/python"


def try_mount_colab_drive() -> bool:
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
    env_root = os.environ.get("SIAMDT_DRIVE_ROOT", "").strip()
    if env_root:
        return Path(env_root)
    colab_path = Path("/content/drive/MyDrive/SiamDT_Project")
    if Path("/content/drive/MyDrive").exists() or sys.platform.startswith("linux"):
        return colab_path
    for cand in (Path(r"Z:\SiamDT_Project"), Path(r"G:\我的云端硬盘\SiamDT_Project")):
        if cand.exists():
            return cand
    return colab_path


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


def sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def refresh_dir(path: Path) -> None:
    """os.listdir 刷新目录元数据（Drive FUSE / Drive for Desktop 下有助于看到新文件）。"""
    try:
        os.listdir(path)
    except Exception:
        pass


def query_gpu_status() -> Dict[str, Any]:
    info: Dict[str, Any] = {"gpu_available": False, "gpu_name": "NO_GPU", "gpu_tier": "CPU", "compute_cap": "cpu",
                            "vram_total_mb": 0.0, "vram_used_mb": 0.0, "vram_free_mb": 0.0, "gpu_utilization_pct": 0.0,
                            "driver_version": ""}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,driver_version,compute_cap",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
        parts = [p.strip() for p in out.splitlines()[0].split(",")] if out else []
        if len(parts) >= 7:
            num = lambda s: float(s) if s.replace(".", "", 1).isdigit() else 0.0  # noqa: E731
            up = parts[0].upper()
            tier = "A100" if ("A100" in up or parts[6] == "8.0") else ("L4" if ("L4" in up or parts[6] == "8.9") else "OTHER_GPU")
            info.update({"gpu_available": True, "gpu_name": parts[0], "gpu_tier": tier, "compute_cap": parts[6],
                         "vram_total_mb": num(parts[1]), "vram_used_mb": num(parts[2]), "vram_free_mb": num(parts[3]),
                         "gpu_utilization_pct": num(parts[4]), "driver_version": parts[5]})
    except Exception:
        pass
    return info


def validate_script_args(args: Any) -> List[str]:
    if args is None:
        return []
    if not isinstance(args, list) or len(args) > 64:
        raise ValueError("args 必须是长度 <= 64 的字符串列表")
    out: List[str] = []
    for a in args:
        if not isinstance(a, str) or len(a) > 512 or not SAFE_ARG_RE.match(a):
            raise ValueError(f"参数含非白名单字符或过长: {a!r}")
        out.append(a)
    return out


class ColabRemoteBridge:
    def __init__(self, drive_root: Path, enable_tunnel: bool = False, allow_raw_shell: bool = False,
                 http_host: str = "127.0.0.1", http_port: int = 18765,
                 heartbeat_interval_sec: float = DEFAULT_HEARTBEAT_SEC, poll_interval_sec: float = DEFAULT_POLL_SEC,
                 local_stop_file: Optional[Path] = None) -> None:
        self.drive_root = drive_root
        self.bridge_dir = drive_root / "colab" / "bridge"
        self.cmd_queue_dir = self.bridge_dir / "cmd_queue"
        self.cmd_processed_dir = self.cmd_queue_dir / "processed"
        self.cmd_results_dir = self.bridge_dir / "cmd_results"
        self.heartbeat_file = self.bridge_dir / "heartbeat.json"
        self.stop_files = [self.bridge_dir / "STOP_BRIDGE"] + ([local_stop_file] if local_stop_file else [])
        for d in (self.bridge_dir, self.cmd_queue_dir, self.cmd_processed_dir, self.cmd_results_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.enable_tunnel = enable_tunnel
        self.allow_raw_shell = allow_raw_shell
        self.http_host, self.http_port = http_host, http_port
        self.heartbeat_interval_sec = max(5.0, heartbeat_interval_sec)
        self.poll_interval_sec = max(0.5, poll_interval_sec)
        self.bridge_token = secrets.token_urlsafe(24) if enable_tunnel else ""
        self.tunnel_url = ""
        self.http_server: Optional[ThreadingHTTPServer] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._job_thread: Optional[threading.Thread] = None
        self._job_proc: Optional[subprocess.Popen] = None
        self._job_cancel_reason: Optional[str] = None
        self.active_cmd_id: Optional[str] = None
        self.active_action: Optional[str] = None
        self.processed_cmd_count = 0
        self._parse_failures: Dict[str, int] = {}
        self._queued_notified: set = set()
        self.started_at_epoch = time.time()
        self.started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.self_sha256 = sha256_file(Path(__file__).resolve())

    # ------------------------------------------------------------------ 心跳
    def collect_heartbeat_payload(self, status: str = "ONLINE") -> Dict[str, Any]:
        gpu = query_gpu_status()
        def _free(p: Path) -> float:
            try:
                return round(shutil.disk_usage(p).free / (1024 ** 3), 2)
            except Exception:
                return 0.0
        with self._lock:
            active_id, active_action, done = self.active_cmd_id, self.active_action, self.processed_cmd_count
        now = time.time()
        channels = ["GDRIVE_FILE_QUEUE"] + (["LOCAL_HTTP_SERVER"] if self.http_server else []) + (["CLOUDFLARE_HTTP_TUNNEL"] if self.tunnel_url else [])
        payload = {
            "status": status,
            "bridge_version": BRIDGE_VERSION,
            "bridge_script_sha256": self.self_sha256,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "started_at_utc": self.started_at_utc,
            "uptime_sec": round(now - self.started_at_epoch, 1),
            "timestamp_epoch": round(now, 3),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "heartbeat_interval_sec": self.heartbeat_interval_sec,
            "gpu": gpu, "gpu_name": gpu["gpu_name"], "gpu_tier": gpu["gpu_tier"],
            "vram_total_mb": gpu["vram_total_mb"], "vram_used_mb": gpu["vram_used_mb"], "vram_free_mb": gpu["vram_free_mb"],
            "drive_free_gb": _free(self.drive_root),
            "runtime_disk_free_gb": _free(Path("/content") if Path("/content").exists() else Path.cwd()),
            "channels_active": channels,
            "tunnel_enabled": self.enable_tunnel,
            "tunnel_url": self.tunnel_url,
            "raw_shell_enabled": self.allow_raw_shell,
            "allowed_actions": list(BUILTIN_ACTIONS) + list(SCRIPT_ACTIONS) + ([RAW_SHELL_ACTION] if self.allow_raw_shell else []),
            "active_cmd_id": active_id,
            "active_action": active_action,
            "processed_cmd_count": done,
            "paths": {"bridge_dir": str(self.bridge_dir), "cmd_queue_dir": str(self.cmd_queue_dir),
                      "cmd_results_dir": str(self.cmd_results_dir), "stop_files": [str(p) for p in self.stop_files]},
        }
        if self.enable_tunnel:
            payload["bridge_token"] = self.bridge_token
            payload["http_listen"] = f"http://{self.http_host}:{self.http_port}"
        return payload

    def write_heartbeat_once(self, status: str = "ONLINE") -> Dict[str, Any]:
        payload = self.collect_heartbeat_payload(status)
        try:
            atomic_fsync_write_json(self.heartbeat_file, payload)
        except Exception as exc:
            print(f"[Bridge-Heartbeat-Warn] {exc}", flush=True)
        return payload

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval_sec):
            self.write_heartbeat_once()

    # ------------------------------------------------------------------ 结果写入
    def _paths(self, cmd_id: str) -> Tuple[Path, Path, Path]:
        return (self.cmd_results_dir / f"{cmd_id}.log", self.cmd_results_dir / f"{cmd_id}.status.json",
                self.cmd_results_dir / f"{cmd_id}.result.json")

    def _finish(self, cmd_id: str, action: str, status: str, returncode: int, tail: str, extra: Optional[Dict[str, Any]] = None,
                t_start: Optional[float] = None) -> Dict[str, Any]:
        log_p, status_p, result_p = self._paths(cmd_id)
        t_end = time.time()
        payload = {
            "cmd_id": cmd_id, "action": action, "status": status, "returncode": int(returncode),
            "elapsed_sec": round(t_end - (t_start or t_end), 3),
            "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end)),
            "log_file": str(log_p), "tail_output": tail[-4000:],
            "bridge_version": BRIDGE_VERSION,
        }
        if extra:
            payload.update(extra)
        atomic_fsync_write_json(result_p, payload)
        atomic_fsync_write_json(status_p, {"cmd_id": cmd_id, "action": action, "status": status, "returncode": int(returncode),
                                           "finished_at_utc": payload["finished_at_utc"]})
        with self._lock:
            self.processed_cmd_count += 1
        print(f"[Bridge] {cmd_id} ({action}) -> {status} rc={returncode}", flush=True)
        return payload

    def _reject(self, cmd_id: str, action: str, reason: str) -> Dict[str, Any]:
        log_p, _, _ = self._paths(cmd_id)
        log_p.write_text(f"[REJECTED] {reason}\n", encoding="utf-8")
        return self._finish(cmd_id, action or "unknown", "REJECTED", 126, f"[REJECTED] {reason}")

    # ------------------------------------------------------------------ 动作解析
    def resolve_pipeline_script(self, rel: str) -> Optional[Path]:
        cands = [Path("/content/runtime/SimDT_Pipeline") / rel]
        try:
            cands.append(Path(__file__).resolve().parents[1] / rel)
        except Exception:
            pass
        cands.append(self.drive_root / "colab" / Path(rel).name)
        return next((c for c in cands if c.exists()), None)

    def _python_for(self, kind: str) -> Optional[str]:
        if kind == "tier1":
            py = os.environ.get("SIAMDT_TIER1_PY", TIER1_PYTHON_DEFAULT)
            return py if Path(py).exists() else None
        return sys.executable

    def _safe_env(self, extra: Any) -> Dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["SIAMDT_DRIVE_ROOT"] = str(self.drive_root)
        if isinstance(extra, dict):
            for k, v in extra.items():
                if SAFE_ENV_KEY_RE.match(str(k)) and SAFE_ARG_RE.match(str(v)):
                    env[str(k)] = str(v)
        return env

    def _timeout_for(self, spec: Dict[str, Any], default_min: float) -> float:
        try:
            t = float(spec.get("timeout_sec") or default_min * 60.0)
        except Exception:
            t = default_min * 60.0
        return max(60.0, min(MAX_TIMEOUT_SEC, t))

    def build_job(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """返回 {argv | command, shell, cwd, env, timeout, meta}；不合法则抛 ValueError。"""
        action = str(spec.get("action") or "")
        if action == RAW_SHELL_ACTION:
            if not self.allow_raw_shell:
                raise ValueError("raw_shell 未启用（需桥启动时显式 --allow-raw-shell）")
            command = str(spec.get("command") or "").strip()
            if not command:
                raise ValueError("raw_shell 需要 command 字段")
            return {"command": command, "shell": True, "cwd": str(Path("/content/runtime") if Path("/content/runtime").exists() else Path.cwd()),
                    "env": self._safe_env(spec.get("env")), "timeout": self._timeout_for(spec, 60), "meta": {"raw_shell": True}}
        conf = SCRIPT_ACTIONS.get(action)
        if conf is None:
            raise ValueError(f"未知或不在白名单中的动作: {action!r}；允许: {list(ALL_ACTIONS)}")
        extra_args = validate_script_args(spec.get("args"))
        if extra_args and not conf["allow_extra_args"]:
            raise ValueError(f"动作 {action} 不接受额外参数")
        script = self.resolve_pipeline_script(conf["script"])
        if script is None:
            raise ValueError(f"找不到动作 {action} 对应脚本 {conf['script']}")
        py = self._python_for(conf["python"])
        if py is None:
            raise ValueError("Tier-1 环境 Python 不存在（请先执行 setup_env 动作）")
        argv = [py, str(script)] + list(conf["fixed_args"]) + extra_args
        return {"argv": argv, "shell": False, "cwd": str(script.parent), "env": self._safe_env(spec.get("env")),
                "timeout": self._timeout_for(spec, conf["default_timeout_min"]),
                "meta": {"script": str(script), "script_sha256": sha256_file(script), "argv": argv}}

    # ------------------------------------------------------------------ 内建动作
    def handle_builtin(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        cmd_id, action = spec["cmd_id"], spec["action"]
        params = spec.get("params") if isinstance(spec.get("params"), dict) else {}
        t0 = time.time()
        log_p, _, _ = self._paths(cmd_id)
        if action == "status":
            hb = self.collect_heartbeat_payload()
            refresh_dir(self.cmd_queue_dir)
            hb["queued_cmd_files"] = sorted(n for n in os.listdir(self.cmd_queue_dir) if n.endswith(".cmd.json"))
            text = json.dumps(hb, indent=2, ensure_ascii=False)
            log_p.write_text(text + "\n", encoding="utf-8")
            return self._finish(cmd_id, action, "COMPLETED", 0, text, {"status_payload": hb}, t0)
        if action == "tail_log":
            lines = max(1, min(2000, int(params.get("lines", 200)))) if str(params.get("lines", "200")).isdigit() else 200
            target: Optional[Path] = None
            if params.get("cmd_id"):
                target = self.cmd_results_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', str(params['cmd_id']))}.log"
            elif params.get("path"):
                cand = Path(str(params["path"])).resolve()
                roots = [self.drive_root.resolve()] + ([Path("/content/runtime").resolve()] if Path("/content/runtime").exists() else [])
                if not any(str(cand).startswith(str(r) + os.sep) or cand == r for r in roots):
                    return self._reject(cmd_id, action, f"tail_log 路径不在允许范围内: {cand}")
                target = cand
            if target is None or not target.exists():
                return self._reject(cmd_id, action, f"tail_log 目标不存在: {target}")
            text = "".join(target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-lines:])
            log_p.write_text(text, encoding="utf-8")
            return self._finish(cmd_id, action, "COMPLETED", 0, text, {"target": str(target)}, t0)
        if action == "cancel":
            with self._lock:
                running, proc = self.active_cmd_id, self._job_proc
            want = params.get("cmd_id")
            if not running or proc is None:
                return self._finish(cmd_id, action, "COMPLETED", 0, "当前没有运行中的任务", {"cancelled": None}, t0)
            if want and want != running:
                return self._reject(cmd_id, action, f"运行中的任务是 {running}，与请求的 {want} 不一致")
            self._job_cancel_reason = "CANCELLED_BY_REQUEST"
            self._kill_proc(proc)
            log_p.write_text(f"已请求取消 {running}\n", encoding="utf-8")
            return self._finish(cmd_id, action, "COMPLETED", 0, f"已请求取消 {running}", {"cancelled": running}, t0)
        return self._reject(cmd_id, action, "未知内建动作")

    # ------------------------------------------------------------------ 子进程任务
    @staticmethod
    def _kill_proc(proc: subprocess.Popen) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _run_job(self, spec: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
        cmd_id, action = spec["cmd_id"], spec["action"]
        log_p, status_p, _ = self._paths(cmd_id)
        t0 = time.time()
        with self._lock:
            self.active_cmd_id, self.active_action = cmd_id, action
        self._job_cancel_reason = None
        self.write_heartbeat_once()
        atomic_fsync_write_json(status_p, {"cmd_id": cmd_id, "action": action, "status": "RUNNING",
                                           "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                                           "timeout_sec": job["timeout"], **job["meta"]})
        chunks: List[str] = []
        buf_lock = threading.Lock()
        returncode, timed_out = -1, False
        with log_p.open("w", encoding="utf-8") as log_f:
            log_f.write(f"=== [SimDT Bridge v{BRIDGE_VERSION}] cmd_id={cmd_id} action={action} ===\n"
                        f"argv={json.dumps(job.get('argv') or job.get('command'), ensure_ascii=False)}\n{'-' * 78}\n")
            log_f.flush()
            try:
                popen_kw: Dict[str, Any] = dict(cwd=job["cwd"], env=job["env"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1)
                if os.name == "posix":
                    popen_kw["start_new_session"] = True
                proc = subprocess.Popen(job["command"] if job["shell"] else job["argv"], shell=bool(job["shell"]), **popen_kw)
                with self._lock:
                    self._job_proc = proc

                def _reader() -> None:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        with buf_lock:
                            chunks.append(line)
                            log_f.write(line)

                rt = threading.Thread(target=_reader, daemon=True)
                rt.start()
                last_sync = time.time()
                while proc.poll() is None:
                    now = time.time()
                    if now - t0 > job["timeout"]:
                        timed_out = True
                        self._kill_proc(proc)
                        break
                    if self._stop_event.is_set() and self._job_cancel_reason is None:
                        self._job_cancel_reason = "CANCELLED_BY_BRIDGE_STOP"
                        self._kill_proc(proc)
                    if now - last_sync >= LOG_FSYNC_SEC:
                        with buf_lock:
                            log_f.flush()
                            os.fsync(log_f.fileno())
                        last_sync = now
                    time.sleep(0.5)
                try:
                    proc.wait(timeout=30)
                except Exception:
                    proc.kill()
                rt.join(timeout=10)
                returncode = proc.returncode if proc.returncode is not None else -9
            except Exception as exc:
                msg = f"\n[Bridge-Exec-Exception] {exc}\n"
                chunks.append(msg)
                log_f.write(msg)
                returncode = -2
            log_f.write(f"\n{'-' * 78}\n=== [Finished] rc={returncode} timed_out={timed_out} ===\n")
            log_f.flush()
            os.fsync(log_f.fileno())
        with self._lock:
            self.active_cmd_id = self.active_action = None
            self._job_proc = None
        status = "TIMEOUT" if timed_out else (self._job_cancel_reason or ("COMPLETED" if returncode == 0 else "FAILED"))
        res = self._finish(cmd_id, action, status, returncode, "".join(chunks), {"timed_out": timed_out, **job["meta"]}, t0)
        self.write_heartbeat_once()
        return res

    def job_running(self) -> bool:
        return self._job_thread is not None and self._job_thread.is_alive()

    def dispatch(self, spec: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """返回 (状态, 结果)。状态：DONE / STARTED / BUSY。"""
        spec["cmd_id"] = re.sub(r"[^A-Za-z0-9_.-]", "_", str(spec.get("cmd_id") or f"cmd_{int(time.time() * 1000)}"))
        action = str(spec.get("action") or "")
        if not action:
            reason = "缺少 action 字段（v2.0 起不再接受任意 command；请用 local_colab_controller.py --action <白名单动作>）"
            return "DONE", self._reject(spec["cmd_id"], "", reason)
        if action in BUILTIN_ACTIONS:
            return "DONE", self.handle_builtin(spec)
        try:
            job = self.build_job(spec)
        except ValueError as exc:
            return "DONE", self._reject(spec["cmd_id"], action, str(exc))
        if self.job_running():
            return "BUSY", None
        self._job_thread = threading.Thread(target=self._run_job, args=(spec, job), daemon=True)
        self._job_thread.start()
        return "STARTED", None

    # ------------------------------------------------------------------ 队列
    def scan_and_process_queue_once(self) -> int:
        refresh_dir(self.cmd_queue_dir)
        try:
            names = [n for n in os.listdir(self.cmd_queue_dir) if n.endswith(".cmd.json")]
        except Exception:
            return 0
        files = sorted((self.cmd_queue_dir / n for n in names), key=lambda p: (p.stat().st_mtime if p.exists() else 0, p.name))
        handled = 0
        for cmd_file in files:
            try:
                spec = json.loads(cmd_file.read_text(encoding="utf-8"))
                if not isinstance(spec, dict):
                    raise ValueError("队列文件不是 JSON 对象")
            except Exception as exc:
                n = self._parse_failures.get(cmd_file.name, 0) + 1
                self._parse_failures[cmd_file.name] = n
                if n >= MAX_PARSE_RETRIES:
                    self._move_processed(cmd_file)
                    self._reject(cmd_file.name[: -len(".cmd.json")], "", f"队列文件连续 {n} 次解析失败: {exc}")
                continue
            spec.setdefault("cmd_id", cmd_file.name[: -len(".cmd.json")])
            action = str(spec.get("action") or "")
            if action not in BUILTIN_ACTIONS and action and action in ALL_ACTIONS and self.job_running():
                if cmd_file.name not in self._queued_notified:
                    _, status_p, _ = self._paths(re.sub(r"[^A-Za-z0-9_.-]", "_", str(spec["cmd_id"])))
                    atomic_fsync_write_json(status_p, {"cmd_id": spec["cmd_id"], "action": action, "status": "QUEUED_WAITING_FOR_RUNNING_JOB"})
                    self._queued_notified.add(cmd_file.name)
                continue
            self._move_processed(cmd_file)
            self.dispatch(spec)
            handled += 1
        return handled

    def _move_processed(self, cmd_file: Path) -> None:
        dest = self.cmd_processed_dir / cmd_file.name
        try:
            if dest.exists():
                dest.unlink()
            shutil.move(str(cmd_file), str(dest))
        except Exception:
            try:
                cmd_file.unlink()
            except Exception:
                pass

    def stop_requested(self) -> bool:
        for p in self.stop_files:
            refresh_dir(p.parent)
            if p.exists():
                return True
        return False

    # ------------------------------------------------------------------ 可选 HTTP + 隧道（默认关闭）
    def start_http_server(self) -> None:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                return

            def _auth(self) -> bool:
                tok = self.headers.get("X-Bridge-Token", "").strip()
                return bool(tok) and secrets.compare_digest(tok, bridge.bridge_token)

            def _send(self, code: int, data: Dict[str, Any]) -> None:
                body = (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if not self._auth():
                    return self._send(401, {"error": "Unauthorized"})
                u = urlparse(self.path)
                if u.path == "/status":
                    return self._send(200, bridge.collect_heartbeat_payload())
                if u.path == "/logs":
                    qs = parse_qs(u.query)
                    cid = re.sub(r"[^A-Za-z0-9_.-]", "_", qs.get("cmd_id", [""])[0])
                    off = int(qs.get("offset", ["0"])[0] or 0)
                    log_p, _, res_p = bridge._paths(cid)
                    raw = log_p.read_bytes() if log_p.exists() else b""
                    res = json.loads(res_p.read_text(encoding="utf-8")) if res_p.exists() else None
                    return self._send(200, {"cmd_id": cid, "next_offset": len(raw), "log_chunk": raw[off:].decode("utf-8", "replace"),
                                            "finished": res is not None, "result": res})
                return self._send(404, {"error": "unknown endpoint"})

            def do_POST(self) -> None:
                if not self._auth():
                    return self._send(401, {"error": "Unauthorized"})
                if urlparse(self.path).path != "/exec":
                    return self._send(404, {"error": "unknown endpoint"})
                try:
                    spec = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0") or 0)).decode("utf-8") or "{}")
                except Exception as exc:
                    return self._send(400, {"error": f"bad json: {exc}"})
                state, res = bridge.dispatch(spec)
                code = {"DONE": 200, "STARTED": 202, "BUSY": 409}[state]
                return self._send(code, {"state": state, "cmd_id": spec.get("cmd_id"), "result": res})

        try:
            self.http_server = ThreadingHTTPServer((self.http_host, self.http_port), Handler)
        except Exception:
            self.http_server = ThreadingHTTPServer((self.http_host, 0), Handler)
        self.http_port = self.http_server.server_address[1]
        threading.Thread(target=self.http_server.serve_forever, daemon=True).start()

    def start_cloudflared_tunnel_async(self) -> None:
        def _worker() -> None:
            cf_bin = shutil.which("cloudflared")
            if not cf_bin and sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64"):
                tmp_cf = Path("/tmp/cloudflared")
                if not tmp_cf.exists():
                    try:
                        urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", str(tmp_cf))
                        tmp_cf.chmod(0o755)
                    except Exception:
                        return
                cf_bin = str(tmp_cf)
            if not cf_bin:
                return
            proc = subprocess.Popen([cf_bin, "tunnel", "--url", f"http://{self.http_host}:{self.http_port}", "--no-autoupdate"],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            pat = re.compile(r"(https://[a-zA-Z0-9-]+\.trycloudflare\.com)")
            assert proc.stdout is not None
            for line in proc.stdout:
                m = pat.search(line)
                if m:
                    self.tunnel_url = m.group(1)
                    self.write_heartbeat_once()
                    print(f"[Bridge-Tunnel] {self.tunnel_url}", flush=True)
                    break

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------ 主循环
    def run_forever(self, max_loops: int = 0, duration_sec: float = 0.0) -> None:
        if self.enable_tunnel:
            print(COLAB_FAQ_REMOTE_PROXY_NOTICE, flush=True)
            self.start_http_server()
            self.start_cloudflared_tunnel_async()
        if self.allow_raw_shell:
            print("[警告] 已启用 --allow-raw-shell：raw_shell 动作可执行任意 shell 命令，仅限调试，用完请重启桥并关闭该开关。", flush=True)
        hb = self.write_heartbeat_once()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        print(f"[接管已就绪] Bridge v{BRIDGE_VERSION} | GPU: {hb['gpu_name']} | 通道: {hb['channels_active']} | 心跳 {self.heartbeat_interval_sec:.0f}s\n"
              f"[Bridge-Info] Queue: {self.cmd_queue_dir}\n[Bridge-Info] Results: {self.cmd_results_dir}\n"
              f"[Bridge-Info] 停止: 写入 {self.stop_files[0]}（或本地 Cell 0b）", flush=True)
        loops, t0 = 0, time.time()
        final_status = "STOPPED"
        try:
            while True:
                if self.stop_requested():
                    print("[Bridge-Info] 检测到 STOP_BRIDGE 标志，正在停止...", flush=True)
                    break
                self.scan_and_process_queue_once()
                loops += 1
                if (max_loops > 0 and loops >= max_loops) or (duration_sec > 0 and time.time() - t0 >= duration_sec):
                    final_status = "EXITED_LIMIT_REACHED"
                    break
                time.sleep(self.poll_interval_sec)
        except KeyboardInterrupt:
            final_status = "STOPPED_KEYBOARD_INTERRUPT"
        finally:
            self._stop_event.set()
            if self._job_thread is not None:
                self._job_thread.join(timeout=60)
            for p in self.stop_files:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            self.write_heartbeat_once(final_status)
            print(f"[Bridge-Info] 已退出（{final_status}）", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SimDT Colab 远程接管桥 v2.0（默认仅 Drive 文件队列 + 动作白名单）")
    p.add_argument("--drive-root", type=str, default="", help="SiamDT_Project 根目录")
    p.add_argument("--enable-tunnel", action="store_true", help="启用本地 HTTP + cloudflared 隧道（默认关闭；请先阅读 Colab FAQ 合规提醒）")
    p.add_argument("--allow-raw-shell", action="store_true", help="调试开关：允许 raw_shell 动作执行任意 shell 命令（默认关闭）")
    p.add_argument("--http-host", type=str, default="127.0.0.1")
    p.add_argument("--http-port", type=int, default=18765)
    p.add_argument("--heartbeat-interval", type=float, default=DEFAULT_HEARTBEAT_SEC, help="心跳间隔秒（默认 30）")
    p.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_SEC, help="队列扫描间隔秒（默认 5）")
    p.add_argument("--local-stop-file", type=str, default="", help="额外的本地停止标志文件路径")
    p.add_argument("--max-loops", type=int, default=0)
    p.add_argument("--duration-sec", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try_mount_colab_drive()
    drive_root = Path(args.drive_root) if args.drive_root else resolve_default_drive_root()
    bridge = ColabRemoteBridge(
        drive_root=drive_root, enable_tunnel=args.enable_tunnel, allow_raw_shell=args.allow_raw_shell,
        http_host=args.http_host, http_port=args.http_port, heartbeat_interval_sec=args.heartbeat_interval,
        poll_interval_sec=args.poll_interval, local_stop_file=Path(args.local_stop_file) if args.local_stop_file else None,
    )
    bridge.run_forever(max_loops=args.max_loops, duration_sec=args.duration_sec)


if __name__ == "__main__":
    main()
