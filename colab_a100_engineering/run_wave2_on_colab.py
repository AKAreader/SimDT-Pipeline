#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimDT 围点打援 —— Colab 端第二轮 (Wave 2) 10大算法混合并发评测与三层失败提取自动执行器 (run_wave2_on_colab.py)
========================================================================================================
在第一轮 (Wave 1: SiamDT_Baseline 24序列满载评测 + 延迟基准 + 6层探针) 完成后无缝接力运行：
  - Step 1/4: 10大算法源码仓库克隆与官方权重完整性核验 (--action stage-repos-and-weights --trackers all)
  - Step 2/4: 构建并门禁自检 4 组对比算法 GPU 环境 (B1: FocusTrack/ReFocus-TIR, B2: SUTrack/UETrack/UTPTrack-S,
              C: DAM4SAM/SAMOSA/SAMURAI, D: MemLoTrack)
  - Step 3/4: 10大算法 A100-80GB 6槽位混合并发评测 (方案B多算法同卡并发优先 + 方案A慢算法动态分片补位, 共 10×24=240 序列对)
  - Step 4/4: 自动运行三层失败片段提取器 (--action extract-episodes)，生成 FAILURE_EPISODE_DATABASE.json 与 CANONICAL_CHALLENGE_WINDOWS.json
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
from typing import Any, Dict, List, Tuple

RUNTIME_ROOT = Path("/content/runtime")
DRIVE_ROOT = Path(os.environ.get("SIAMDT_DRIVE_ROOT", "/content/drive/MyDrive/SiamDT_Project"))
PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROGRESS_FILE = RUNTIME_ROOT / "wave_progress.json"
DRIVE_PROGRESS_FILE = DRIVE_ROOT / "colab/bridge/wave_progress.json"
os.environ["MPLBACKEND"] = "Agg"

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import colab_ntfy_telemetry as _ntfy_telem
except Exception:
    _ntfy_telem = None

ALL_10_TRACKERS = [
    "SiamDT_Baseline",
    "FocusTrack",
    "MemLoTrack",
    "DAM4SAM",
    "SAMOSA",
    "SAMURAI",
    "UETrack",
    "SUTrack",
    "UTPTrack-S",
    "ReFocus-TIR",
]


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


def count_all_trackers_completed_sequences() -> Tuple[int, Dict[str, int]]:
    parent = DRIVE_ROOT / "experiments/failure_diagnosis/per_frame_telemetry"
    per_tk: Dict[str, int] = {}
    total = 0
    for tk in ALL_10_TRACKERS:
        cnt = 0
        if parent.exists():
            for d in parent.glob(f"{tk}_discovery_set*"):
                if d.is_dir():
                    c = sum(1 for p in d.glob("*.jsonl") if p.is_file() and p.stat().st_size > 0)
                    cnt = max(cnt, c)
        cnt = min(24, cnt)
        per_tk[tk] = cnt
        total += cnt
    return total, per_tk


BUILD_WAVE2_ENVS_BASH = r'''set -euo pipefail

export MPLBACKEND="Agg"
export PIP_CACHE_DIR="/content/cache/pip"
export UV_CACHE_DIR="/content/cache/uv"
export UV_LINK_MODE="copy"
export UV_PYTHON_INSTALL_DIR="/content/uv-python"
mkdir -p /content/envs /content/cache/pip /content/cache/uv

UV="/content/uv-bootstrap/bin/uv"
if [ ! -x "$UV" ]; then
    python3 -m pip install --quiet --target /content/uv-bootstrap uv==0.9.7
fi

# 0. 安装系统级 libturbojpeg (供 MemLoTrack / jpeg4py 调用)
if ! dpkg -s libturbojpeg0-dev >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq libturbojpeg0-dev libturbojpeg >/dev/null 2>&1 || true
fi

# 1. B1 环境: /content/envs/pytracking-pt20-cu118 (FocusTrack, ReFocus-TIR)
ENV_B1="/content/envs/pytracking-pt20-cu118"
if [ ! -x "$ENV_B1/bin/python" ]; then
    "$UV" python install 3.10.19
    "$UV" venv --seed --python 3.10.19 "$ENV_B1"
fi
PY_B1="$ENV_B1/bin/python"
if ! "$PY_B1" -c "import torch, torchvision, timm, easydict, yacs, cv2, lmdb, jpeg4py, pandas, yaml, tikzplotlib, thop" 2>/dev/null; then
    echo "[Wave2-Env B1] 正在构建 FocusTrack / ReFocus-TIR 环境..."
    "$UV" pip install --python "$PY_B1" setuptools==68.2.2 wheel==0.41.3 cython==0.29.35 numpy==1.24.4
    if ! "$PY_B1" -c "import torch; assert '2.0.0' in torch.__version__" 2>/dev/null; then
        "$UV" pip install --python "$PY_B1" torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    fi
    "$UV" pip install --no-build-isolation --python "$PY_B1" \
        numpy==1.24.4 scipy==1.10.1 matplotlib==3.7.5 opencv-python-headless==4.8.1.78 pillow==10.2.0 \
        timm==0.6.13 easydict yacs lmdb jpeg4py pandas tqdm pyyaml packaging requests six \
        pycocotools==2.0.7 tensorboardX thop tikzplotlib visdom==0.2.4 wandb
fi
"$PY_B1" -c "import torch, timm, easydict, yacs, cv2, lmdb, jpeg4py, yaml; print('[B1 PASS] torch=' + torch.__version__)"

# 2. B2 环境: /content/envs/sutrack-pt111-cu113 (SUTrack, UETrack, UTPTrack-S)
ENV_B2="/content/envs/sutrack-pt111-cu113"
if [ ! -x "$ENV_B2/bin/python" ]; then
    "$UV" venv --seed --python 3.10.19 "$ENV_B2"
fi
PY_B2="$ENV_B2/bin/python"
if ! "$PY_B2" -c "import torch, torchvision, timm, clip, easydict, yacs, einops, cv2, lmdb, jpeg4py, skimage, yaml" 2>/dev/null; then
    echo "[Wave2-Env B2] 正在构建 SUTrack / UETrack / UTPTrack-S 环境..."
    "$UV" pip install --python "$PY_B2" setuptools==68.2.2 wheel==0.41.3 cython==0.29.35 numpy==1.24.4
    if ! "$PY_B2" -c "import torch; assert '2.0.0' in torch.__version__" 2>/dev/null; then
        "$UV" pip install --python "$PY_B2" torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    fi
    "$UV" pip install --no-build-isolation --python "$PY_B2" \
        timm==0.5.4 yacs easydict einops opencv-python-headless==4.8.1.78 pillow==10.2.0 \
        numpy==1.24.4 scipy==1.10.1 scikit-image matplotlib==3.7.5 pandas pycocotools==2.0.7 \
        jpeg4py lmdb tqdm pyyaml packaging six ftfy regex tensorboardX thop tikzplotlib \
        pytorch-pretrained-bert==0.6.2 visdom==0.2.4 wandb
    "$UV" pip install --no-build-isolation --python "$PY_B2" --no-deps git+https://github.com/openai/CLIP.git
fi
"$PY_B2" -c "import torch, timm, clip, easydict, yacs, einops, cv2, skimage; print('[B2 PASS] timm=' + timm.__version__)"

# 3. C 环境: /content/envs/sam2-pt25-cu124 (DAM4SAM, SAMOSA, SAMURAI)
ENV_C="/content/envs/sam2-pt25-cu124"
if [ ! -x "$ENV_C/bin/python" ]; then
    "$UV" venv --system-site-packages --seed "$ENV_C"
fi
PY_C="$ENV_C/bin/python"
if ! "$PY_C" -c "import torch, torchvision, hydra, omegaconf, iopath, loguru, shapely, vot.region, cv2, lmdb, jpeg4py" 2>/dev/null; then
    echo "[Wave2-Env C] 正在构建 DAM4SAM / SAMOSA / SAMURAI (SAM 2.1) 环境..."
    "$UV" pip install --python "$PY_C" setuptools==68.2.2 wheel==0.41.3
    "$UV" pip install --no-build-isolation --python "$PY_C" \
        torch torchvision hydra-core==1.3.2 omegaconf==2.3.0 iopath loguru shapely \
        vot-toolkit==0.7.1 vot-trax==4.0.2 lmdb jpeg4py decord tensordict submitit \
        opencv-python-headless pillow tqdm scipy pandas matplotlib pycocotools scikit-image tikzplotlib wandb
fi
"$PY_C" -c "import torch, torchvision, hydra, omegaconf, iopath, loguru, shapely, vot.region; print('[C PASS] torch=' + torch.__version__ + ' tv=' + torchvision.__version__)"

# 4. D 环境: /content/envs/trackit-memlo-pt2 (MemLoTrack)
ENV_D="/content/envs/trackit-memlo-pt2"
if [ ! -x "$ENV_D/bin/python" ]; then
    "$UV" venv --system-site-packages --seed "$ENV_D"
fi
PY_D="$ENV_D/bin/python"
if ! "$PY_D" -c "import torch, torchvision, timm, safetensors, fvcore, tabulate, exifread, zmq, psutil, turbojpeg, shapely" 2>/dev/null; then
    echo "[Wave2-Env D] 正在构建 MemLoTrack (DINOv2+LoRA) 环境..."
    "$UV" pip install --python "$PY_D" setuptools==68.2.2 wheel==0.41.3
    "$UV" pip install --no-build-isolation --python "$PY_D" \
        torch torchvision timm safetensors fvcore tabulate exifread pyzmq psutil PyTurboJPEG shapely \
        wandb pyyaml matplotlib tqdm pillow opencv-python-headless scipy pandas einops pycocotools zstandard
fi
"$PY_D" -c "import torch, torchvision, timm, safetensors, fvcore, tabulate, exifread, zmq, psutil, turbojpeg; print('[D PASS] timm=' + timm.__version__ + ' tv=' + torchvision.__version__)"

echo "[完成] Wave 2 全部 4 组对比算法环境 (B1/B2/C/D) 已就绪！"
'''


def main() -> None:
    t_wave_start = time.time()
    total_seqs = 240  # 10 个算法 × 24 条序列
    steps_plan = [
        ("Step 1/4: 10大算法源码克隆与官方权重完整性核验 (stage-repos-and-weights)", 90.0),
        ("Step 2/4: 构建并自检 4 组对比算法专用 GPU 环境 (B1/B2/C/D)", 150.0),
        ("Step 3/4: 10大算法 A100-80GB 6槽位混合并发评测 (方案B+A, 240个算法-视频对)", 2700.0),
        ("Step 4/4: 三层失败片段数据库与典型挑战窗口提取 (extract-episodes)", 60.0),
    ]
    total_est_sec = sum(x[1] for x in steps_plan)

    done_init, per_tk_init = count_all_trackers_completed_sequences()
    state: Dict[str, Any] = {
        "wave_name": "第二轮 (Wave 2): 10大算法同台混合并发评测 (方案B+A) + 三层失败片段提取",
        "status": "RUNNING",
        "started_at_beijing": beijing_time_str(t_wave_start),
        "started_at_epoch": round(t_wave_start, 2),
        "current_step_index": 1,
        "total_steps": 4,
        "current_step_title": steps_plan[0][0],
        "completed_sequences": done_init,
        "total_sequences": total_seqs,
        "per_tracker_completed": per_tk_init,
        "sequence_progress_pct": round(done_init * 100.0 / total_seqs, 1),
        "overall_progress_pct": max(5.0, round(done_init * 100.0 / total_seqs, 1)),
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
            done_seqs, per_tk = count_all_trackers_completed_sequences()
            state["completed_sequences"] = done_seqs
            state["per_tracker_completed"] = per_tk
            state["sequence_progress_pct"] = round(done_seqs * 100.0 / max(1, total_seqs), 1)
            state["elapsed_sec"] = round(elapsed, 1)

            step_elapsed = max(1.0, now - float(state.get("_step_t0") or now))
            if idx == 1:
                rem = max(60.0, total_est_sec - elapsed)
                pct = min(8.0, round(step_elapsed / 90.0 * 8.0, 1))
            elif idx == 2:
                rem = max(120.0, (total_est_sec - 90.0) - step_elapsed)
                pct = min(15.0, 8.0 + round(step_elapsed / 150.0 * 7.0, 1))
            elif idx == 3:
                newly_done = max(0, done_seqs - 24)
                rem_seqs = max(0, total_seqs - done_seqs)
                if newly_done >= 3:
                    sec_per_seq = step_elapsed / newly_done
                    rem = rem_seqs * sec_per_seq + 60.0
                else:
                    rem = max(300.0, 2700.0 - step_elapsed)
                pct = 15.0 + round(done_seqs / total_seqs * 80.0, 1)
            else:
                rem = max(10.0, 60.0 - step_elapsed)
                pct = 97.0

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
        eval_script = str(PIPELINE_DIR / "phase2_unified_eval/run_phase2_multi_tracker_colab.py")

        # Step 0 (if VM was reset after Colab disconnect): Ensure local NVMe dataset cache is staged
        val_cache = RUNTIME_ROOT / "antiuav410_val"
        if not val_cache.exists() or len(list(val_cache.glob("*/IR_label.json"))) < 24:
            _run_step(1, "Step 0/4: 恢复重连后 Colab 本地高速盘数据集缓存 (Cells 1-3)", [
                sys.executable, str(PIPELINE_DIR / "colab_a100_engineering/colab_init_cells.py"),
                "--run-cell", "1,2,3",
            ])

        # Step 1: Stage repos and verify weights for all 10 trackers
        _run_step(1, steps_plan[0][0], [
            sys.executable, eval_script,
            "--action", "stage-repos-and-weights",
            "--trackers", "all",
        ])

        # Step 2: Build B1, B2, C, D environments on local NVMe
        _run_step(2, steps_plan[1][0], ["bash", "-c", BUILD_WAVE2_ENVS_BASH])

        # Step 3: 10-Tracker Unified Evaluation (Plan B priority + Plan A dynamic backfill)
        _run_step(3, steps_plan[2][0], [
            sys.executable, eval_script,
            "--action", "eval",
            "--trackers", "all",
            "--split", "discovery_set.json",
            "--concurrency-mode", "hybrid_b_then_a",
            "--gpu-slots", "6",
        ])

        # Step 4: Extract 3-tier failure episodes and canonical challenge windows
        _run_step(4, steps_plan[3][0], [
            sys.executable, eval_script,
            "--action", "extract-episodes",
            "--split", "discovery_set.json",
        ])

        stop_monitor.set()
        done_final, per_tk_final = count_all_trackers_completed_sequences()
        state["status"] = "COMPLETED_SUCCESS"
        state["completed_sequences"] = done_final
        state["per_tracker_completed"] = per_tk_final
        state["sequence_progress_pct"] = 100.0
        state["overall_progress_pct"] = 100.0
        state["elapsed_sec"] = round(time.time() - t_wave_start, 1)
        state["eta_remaining_sec"] = 0.0
        state["eta_finish_beijing"] = beijing_time_str()
        write_wave_progress(state)
        print(f"\n🎉 [{beijing_time_str()}] Wave 2 (10大算法混合并发评测 + 三层失败片段提取) 全部成功完成！总耗时 {state['elapsed_sec']} 秒。", flush=True)
    finally:
        stop_monitor.set()


if __name__ == "__main__":
    main()
