"""
SimDT 围点打援 (SiamDT + Multi-Tracker Benchmark on Anti-UAV410)
“本地总控 (10%~15%) + GitHub 阶段代码 + Colab A100 主力算力 (70%~80%) / L4 辅助算力 (10%~15%) + Google Drive 永久金库” 四位一体工程套件。

算力调度分工契约：
- 主力算力 (70%~80%)：Colab A100 (sm_80, 40GB/80GB) —— 承担全部模型训练、消融实验、多算法全量统一评测及高负载任务；
- 辅助算力 (10%~15%)：Colab L4 (sm_89, 24GB) —— 承担轻量级子集推理、单模型 5 帧冒烟验证、Phase 5 重点序列内部特征导出等任务；
- 本地调度 (10%~15%)：本机 VS Code —— 仅承担总控调度、Phase 1/3/4 轻量级 JSON/CSV 统计分析、代码编写与向 GitHub 推送阶段 Tag。

各章节 (CELL 1 ~ CELL 7) 既可通过 SimDT_Colab_A100_Master.ipynb 在 Colab 一键运行，
也可单独复制至 Colab Notebook 单元格按序执行。
"""

# ==============================================================================
# [CELL 0] 后台启动远程接管桥（仅 Google Drive 文件队列通道；不开公网隧道；立即返回）
# ==============================================================================
CELL_0_REMOTE_BRIDGE = r'''
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from google.colab import drive
    if not os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
except ImportError:
    print('[提示] 当前非 Google Colab 交互环境，跳过 drive.mount。')

DRIVE_ROOT = Path(os.environ.get('SIAMDT_DRIVE_ROOT', '/content/drive/MyDrive/SiamDT_Project'))
RUNTIME_BRIDGE = Path('/content/runtime/SimDT_Pipeline/colab_a100_engineering/colab_remote_bridge.py')  # Cell 2 从 GitHub 拉取的版本
DRIVE_BRIDGE = DRIVE_ROOT / 'colab/colab_remote_bridge.py'                                            # Drive 副本（回退）
STATE_DIR = Path('/content/runtime/bridge_state')
PID_FILE = STATE_DIR / 'bridge.pid'
LOG_FILE = STATE_DIR / 'bridge_stdout.log'
LOCAL_STOP = STATE_DIR / 'STOP_BRIDGE'
DRIVE_STOP = DRIVE_ROOT / 'colab/bridge/STOP_BRIDGE'


def _sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


shas = {}
for tag, p in (('runtime_github', RUNTIME_BRIDGE), ('drive_copy', DRIVE_BRIDGE)):
    shas[tag] = _sha256(p) if p.exists() else None
    print(f'[Bridge-SHA256] {tag:15s} {shas[tag] or "(不存在)"}  {p}')
if shas['runtime_github'] and shas['drive_copy'] and shas['runtime_github'] != shas['drive_copy']:
    print('[警告] GitHub 运行时版本与 Drive 副本 SHA256 不一致（存在漂移）；本次使用 GitHub 运行时版本。')

BRIDGE = RUNTIME_BRIDGE if RUNTIME_BRIDGE.exists() else DRIVE_BRIDGE
assert BRIDGE.exists(), f'未找到远程接管脚本: {BRIDGE}'
STATE_DIR.mkdir(parents=True, exist_ok=True)

old_txt = PID_FILE.read_text().strip() if PID_FILE.exists() else ''
old_pid = int(old_txt) if old_txt.isdigit() else None
if old_pid and _pid_alive(old_pid):
    print(f'[接管已就绪] 远程接管桥已在后台运行 (PID={old_pid})，无需重复启动。')
else:
    for flag in (LOCAL_STOP, DRIVE_STOP):
        if flag.exists():
            flag.unlink()
    log_f = open(LOG_FILE, 'a', encoding='utf-8')
    proc = subprocess.Popen(
        [sys.executable, str(BRIDGE), '--drive-root', str(DRIVE_ROOT), '--local-stop-file', str(LOCAL_STOP)],
        stdout=log_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True, close_fds=True,
    )
    PID_FILE.write_text(str(proc.pid))
    time.sleep(3)
    if proc.poll() is not None:
        print(LOG_FILE.read_text(encoding='utf-8', errors='replace')[-3000:])
        raise RuntimeError(f'远程接管桥启动失败 (returncode={proc.returncode})，日志: {LOG_FILE}')
    print(f'[接管已就绪] 远程接管桥已在后台运行 (PID={proc.pid}，脚本={BRIDGE})。本单元已返回，可继续运行其它单元。')
    print(f'             通道: Google Drive 文件队列（cloudflared 隧道默认关闭）；心跳 30 秒；日志: {LOG_FILE}')
print('[如何停止] 运行下方 Cell 0b（写停止标志，超时后结束进程）；或在本地执行: python local_colab_controller.py --action stop-bridge')
'''

# ==============================================================================
# [CELL 0b] 停止远程接管桥（写停止标志；60 秒内未退出则结束进程组）
# ==============================================================================
CELL_0B_STOP_BRIDGE = r'''
import os
import signal
import time
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get('SIAMDT_DRIVE_ROOT', '/content/drive/MyDrive/SiamDT_Project'))
STATE_DIR = Path('/content/runtime/bridge_state')
PID_FILE = STATE_DIR / 'bridge.pid'
STATE_DIR.mkdir(parents=True, exist_ok=True)
for flag in (STATE_DIR / 'STOP_BRIDGE', DRIVE_ROOT / 'colab/bridge/STOP_BRIDGE'):
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' stop requested by Cell 0b\n')

txt = PID_FILE.read_text().strip() if PID_FILE.exists() else ''
pid = int(txt) if txt.isdigit() else None


def _alive(p):
    try:
        os.kill(p, 0)
        return True
    except OSError:
        return False


if pid is None or not _alive(pid):
    print('[Cell 0b] 未发现运行中的远程接管桥（已写入停止标志，若桥稍后启动也会立即退出；重新运行 Cell 0 会自动清除标志）。')
else:
    for _ in range(60):
        if not _alive(pid):
            break
        time.sleep(1)
    if _alive(pid):
        print(f'[Cell 0b] 桥 60 秒内未退出，发送 SIGTERM 到进程组 {pid} ...')
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception as exc:
            print(f'[Cell 0b] SIGTERM 失败: {exc}')
        time.sleep(10)
        if _alive(pid):
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    print(f'[Cell 0b] 远程接管桥已停止 (PID={pid})。运行中的远程任务（如有）已被终止，结果记为 CANCELLED_BY_BRIDGE_STOP。')
    PID_FILE.unlink(missing_ok=True)
'''

# ==============================================================================
# [CELL 1] 挂载 Google Drive 与建立持久化金库目录树（零覆盖资产扫描与无损复用）
# ==============================================================================
CELL_1_MOUNT_AND_LAYOUT = r'''
import os
import json
import time
import shutil
import hashlib
import subprocess
from pathlib import Path

# 1. 挂载 Google Drive（如在 Colab 环境）
try:
    from google.colab import drive
    if not os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
except ImportError:
    print("[提示] 当前非 Google Colab 交互环境，跳过 drive.mount。")

DRIVE_ROOT = Path(os.environ.get("SIAMDT_DRIVE_ROOT", "/content/drive/MyDrive/SiamDT_Project"))
RUNTIME_ROOT = Path("/content/runtime")

# 2. 定义支持“GitHub 阶段代码 + 多算法跑分对比 + 逐帧遥测诊断 + SiamDT 断点续训”的金库目录树
REQUIRED_DIRS = [
    # 数据集与冻结划分契约 (JSON)
    "datasets/AntiUAV410/train",
    "datasets/AntiUAV410/val",
    "datasets/AntiUAV410/test",
    "datasets/AntiUAV410/annos",
    "datasets/diagnostic_splits",
    "datasets/archives",
    # 源码冷备与非侵入兼容补丁审计
    "repos/SimDT_Pipeline",
    "repos/Anti-UAV410_official",
    "repos/core_pool",
    "repos/patches",
    # 按训练域与角色严格分仓的权重库
    "weights/foundation_backbones/swin_tiny",
    "weights/foundation_backbones/sam2_1",
    "weights/foundation_backbones/dinov2",
    "weights/in_domain_antiuav410/SiamDT_Baseline",
    "weights/in_domain_antiuav410/FocusTrack",
    "weights/in_domain_antiuav410/MemLoTrack",
    "weights/zero_shot_sota/DAM4SAM",
    "weights/zero_shot_sota/SAMOSA",
    "weights/zero_shot_sota/UETrack_MCITrack",
    "weights/zero_shot_sota/ReFocus_TIR_SUTrack",
    # 双轨环境缓存与诊断记录
    "environments/requirements",
    "environments/wheels",
    "environments/wheels/pip-cache",
    "environments/wheels/uv-cache",
    "environments/diagnostics",
    # 实验输出（严格隔离训练、分域跑分与故障诊断）
    "experiments/runs_training",
    "experiments/runs_training/siamdt_baseline_repro",
    "experiments/runs_training/siamdt_ours_ablations",
    "experiments/benchmark_eval",
    "experiments/benchmark_eval/group1_in_domain",
    "experiments/benchmark_eval/group2_zero_shot",
    "experiments/benchmark_eval/reference_static",
    "experiments/failure_diagnosis/per_frame_telemetry",
    "experiments/failure_diagnosis/rescue_matrix",
    "experiments/failure_diagnosis/reports",
    # Colab 脚本与工程文档
    "colab",
    "colab/bridge/cmd_queue/processed",
    "colab/bridge/cmd_results",
    "docs",
]

for rel in REQUIRED_DIRS:
    (DRIVE_ROOT / rel).mkdir(parents=True, exist_ok=True)
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

# 3. 扫描 Drive 中已有的旧版路径资产（零覆盖、无损软链/登记复用）
LEGACY_CANDIDATES = {
    "siamdt_official_ckpt": [
        DRIVE_ROOT / "weights/in_domain_antiuav410/SiamDT_Baseline/siamdt_swin_tiny_sgd.pth",
        DRIVE_ROOT / "weights/official_checkpoints/siamdt_swin_tiny_sgd.pth",
        DRIVE_ROOT / "weights/checkpoints/siamdt_swin_tiny_sgd.pth",
    ],
    "swin_tiny_pretrained": [
        DRIVE_ROOT / "weights/foundation_backbones/swin_tiny/cascade_mask_rcnn_swin_tiny.pth.tar",
        DRIVE_ROOT / "weights/pretrained/cascade_mask_rcnn_swin_tiny.pth.tar",
    ],
    "ckpt_zip_archive": [
        DRIVE_ROOT / "datasets/archives/checkpoints.zip",
        DRIVE_ROOT / "archives/checkpoints.zip",
    ],
    "legacy_repo": [
        DRIVE_ROOT / "repos/Anti-UAV410_official",
        DRIVE_ROOT / "code/Anti-UAV410",
    ],
    "legacy_data": [
        DRIVE_ROOT / "datasets/AntiUAV410",
        DRIVE_ROOT / "data/Anti-UAV",
        DRIVE_ROOT / "data",
    ],
    "diagnostic_splits": [
        DRIVE_ROOT / "datasets/diagnostic_splits",
    ],
}

discovered = {}
for key, paths in LEGACY_CANDIDATES.items():
    found = [str(p) for p in paths if p.exists()]
    discovered[key] = found

# 检查磁盘剩余空间与 GPU 档位信息（原生识别主力 A100 sm_80 与辅助 L4 sm_89）
total_b, used_b, free_b = shutil.disk_usage(DRIVE_ROOT)
gpu_info = "NO_GPU"
gpu_tier_role = "LOCAL_OR_CPU_CONTROL (10%~15% 调度/统计探针)"
try:
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"],
        text=True
    ).strip()
    if "A100" in gpu_info.upper() or "8.0" in gpu_info:
        gpu_tier_role = "COLAB_A100_MAIN (sm_80, 70%~80% 主力算力：承担全量训练/消融/多算法全量评测)"
    elif "L4" in gpu_info.upper() or "8.9" in gpu_info:
        gpu_tier_role = "COLAB_L4_AUX (sm_89, 10%~15% 辅助算力：承担轻量级子集推理/5帧冒烟/Phase5特征导出)"
    else:
        gpu_tier_role = f"COLAB_GENERIC_GPU ({gpu_info})"
except Exception:
    pass

scan_report = {
    "cell": "CELL_1_MOUNT_AND_LAYOUT",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "drive_root": str(DRIVE_ROOT),
    "runtime_root": str(RUNTIME_ROOT),
    "drive_free_gb": round(free_b / (1024**3), 2),
    "gpu_detected": gpu_info,
    "gpu_compute_tier_role": gpu_tier_role,
    "verified_dirs_count": len(REQUIRED_DIRS),
    "discovered_existing_assets": discovered,
}
print(json.dumps(scan_report, indent=2, ensure_ascii=False))
'''

# ==============================================================================
# [CELL 2] 双源 Git 同步（GitHub 阶段代码库 SimDT_Pipeline + 官方 Anti-UAV410 锁定仓库）
# ==============================================================================
CELL_2_DUAL_GIT_SYNC_AND_PATCH = r'''
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get("SIAMDT_DRIVE_ROOT", "/content/drive/MyDrive/SiamDT_Project"))
RUNTIME_ROOT = Path("/content/runtime")
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Part A: 同步用户 GitHub 阶段代码库 (SimDT_Pipeline) 并同步冻结子集契约至 Google Drive
# ------------------------------------------------------------------------------
# 可在运行前通过环境变量或在此直接修改默认 GitHub 仓库地址与阶段 Tag (如 v1.0-phase1-subset-frozen)
SIMDT_GITHUB_REPO = os.environ.get(
    "SIMDT_GITHUB_REPO",
    "https://github.com/AKAreader/SimDT-Pipeline.git"
).strip()
SIMDT_GIT_REF = os.environ.get(
    "SIMDT_GIT_REF",
    "main"  # 支持填写分支名 (main) 或阶段里程碑 Tag (如 v1.0-phase1-subset-frozen)
).strip()

PERSIST_STAGE_REPO = DRIVE_ROOT / "repos/SimDT_Pipeline"
RUNTIME_STAGE_REPO = RUNTIME_ROOT / "SimDT_Pipeline"
DRIVE_SPLITS_DIR = DRIVE_ROOT / "datasets/diagnostic_splits"
DRIVE_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

stage_sync_status = {
    "configured_repo": SIMDT_GITHUB_REPO,
    "requested_ref": SIMDT_GIT_REF,
    "sync_mode": "UNINITIALIZED",
    "commit_sha": "local_untracked",
    "short_sha": "local00",
    "synced_split_jsons": [],
    "mounted_innovations": [],
}

def _try_git_sync_stage_repo() -> None:
    if not SIMDT_GITHUB_REPO:
        return
    try:
        r_check = subprocess.run(["git", "-C", str(RUNTIME_STAGE_REPO), "rev-parse", "HEAD"], capture_output=True, text=True) if (RUNTIME_STAGE_REPO / ".git").exists() else None
        if r_check is None or r_check.returncode != 0:
            if RUNTIME_STAGE_REPO.exists():
                shutil.rmtree(RUNTIME_STAGE_REPO, ignore_errors=True)
            print(f"[Stage-Git] 正在从 GitHub 克隆阶段代码库: {SIMDT_GITHUB_REPO} ...")
            subprocess.run(
                ["git", "clone", SIMDT_GITHUB_REPO, str(RUNTIME_STAGE_REPO)],
                check=True, timeout=180
            )
        # 拉取全部分支与阶段 Tag
        subprocess.run(
            ["git", "-C", str(RUNTIME_STAGE_REPO), "remote", "set-url", "origin", SIMDT_GITHUB_REPO],
            check=False
        )
        subprocess.run(
            ["git", "-C", str(RUNTIME_STAGE_REPO), "fetch", "--all", "--tags", "--prune"],
            check=True, timeout=120
        )
        subprocess.run(
            ["git", "-C", str(RUNTIME_STAGE_REPO), "checkout", SIMDT_GIT_REF],
            check=True
        )
        # 若当前在分支上而非 detached Tag，则执行 fast-forward pull
        branch_check = subprocess.run(
            ["git", "-C", str(RUNTIME_STAGE_REPO), "symbolic-ref", "-q", "HEAD"],
            capture_output=True, text=True
        )
        if branch_check.returncode == 0:
            subprocess.run(
                ["git", "-C", str(RUNTIME_STAGE_REPO), "pull", "--ff-only", "origin", SIMDT_GIT_REF],
                check=False, timeout=120
            )
        # 备份最新阶段代码到 Google Drive 冷备目录（忽略 .git 避免产生大量小文件阻塞云盘同步）
        shutil.copytree(RUNTIME_STAGE_REPO, PERSIST_STAGE_REPO, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)
        stage_sync_status["sync_mode"] = "GITHUB_LIVE_SYNCED"
    except Exception as exc:
        print(f"[Stage-Git 提示] GitHub 远端拉取未就绪或需要鉴权 ({exc})，尝试使用 Drive 冷备...")
        if not (RUNTIME_STAGE_REPO / ".git").exists() and (PERSIST_STAGE_REPO / ".git").exists():
            shutil.copytree(PERSIST_STAGE_REPO, RUNTIME_STAGE_REPO, dirs_exist_ok=True)
            stage_sync_status["sync_mode"] = "GDRIVE_COLD_BACKUP_FALLBACK"
        elif (RUNTIME_STAGE_REPO / ".git").exists():
            stage_sync_status["sync_mode"] = "RUNTIME_CACHED_REPO"

_try_git_sync_stage_repo()

if (RUNTIME_STAGE_REPO / ".git").exists():
    try:
        full_sha = subprocess.check_output(
            ["git", "-C", str(RUNTIME_STAGE_REPO), "rev-parse", "HEAD"], text=True
        ).strip()
        stage_sync_status["commit_sha"] = full_sha
        stage_sync_status["short_sha"] = full_sha[:7]
    except Exception:
        pass

# 将阶段代码库中的 datasets/diagnostic_splits/*.json 与 phase1_diagnostic_subset/*.json 同步至 Google Drive
if RUNTIME_STAGE_REPO.exists():
    candidate_json_dirs = [
        RUNTIME_STAGE_REPO / "datasets/diagnostic_splits",
        RUNTIME_STAGE_REPO / "phase1_diagnostic_subset",
    ]
    for cdir in candidate_json_dirs:
        if cdir.exists():
            for jf in sorted(cdir.glob("*.json")):
                dest_jf = DRIVE_SPLITS_DIR / jf.name
                shutil.copy2(jf, dest_jf)
                stage_sync_status["synced_split_jsons"].append(str(dest_jf))

# 写入阶段版本元信息文件，供后续评测 (Cell 6) 与训练 (Cell 7) 自动绑定 Git SHA
STAGE_META_FILE = RUNTIME_ROOT / "simdt_stage_version.json"
STAGE_META_FILE.write_text(json.dumps(stage_sync_status, indent=2, ensure_ascii=False), encoding="utf-8")
os.environ["SIMDT_STAGE_COMMIT_SHA"] = stage_sync_status["short_sha"]

# ------------------------------------------------------------------------------
# Part B: 同步官方 HwangBo94/Anti-UAV410 仓库（锁定 Commit）并施加 3 项非侵入兼容补丁
# ------------------------------------------------------------------------------
PERSIST_REPO = DRIVE_ROOT / "repos/Anti-UAV410_official"
LEGACY_REPO = DRIVE_ROOT / "code/Anti-UAV410"
RUNTIME_REPO = RUNTIME_ROOT / "Anti-UAV410"

OFFICIAL_URL = "https://github.com/HwangBo94/Anti-UAV410.git"
PINNED_COMMIT = "8a8eb04d976e9386b7c9c3ada5c85e5086013d52"

# 1. 直接在 Colab 本地高速盘 /content/runtime/Anti-UAV410 克隆并锁定官方 Commit（规避 Drive FUSE 半同步 .git 目录导致 exit 128）
def _is_valid_git_repo(p: Path) -> bool:
    if not (p / ".git").exists():
        return False
    r = subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.returncode == 0

if not _is_valid_git_repo(RUNTIME_REPO):
    if RUNTIME_REPO.exists():
        shutil.rmtree(RUNTIME_REPO, ignore_errors=True)
    if _is_valid_git_repo(PERSIST_REPO):
        print(f"[复用] 从完整 Drive 冷备恢复 {PERSIST_REPO} -> {RUNTIME_REPO} ...")
        shutil.copytree(PERSIST_REPO, RUNTIME_REPO, dirs_exist_ok=True)
    elif _is_valid_git_repo(LEGACY_REPO):
        print(f"[复用] 从已有旧版仓库恢复 {LEGACY_REPO} -> {RUNTIME_REPO} ...")
        shutil.copytree(LEGACY_REPO, RUNTIME_REPO, dirs_exist_ok=True)
    else:
        print(f"[克隆] 直接从 GitHub 克隆官方仓库至本地高速盘 {RUNTIME_REPO} ...")
        subprocess.run(["git", "clone", OFFICIAL_URL, str(RUNTIME_REPO)], check=True)

# 锁定官方 Commit
subprocess.run(["git", "-C", str(RUNTIME_REPO), "checkout", PINNED_COMMIT], check=True)
head_commit = subprocess.check_output(["git", "-C", str(RUNTIME_REPO), "rev-parse", "HEAD"], text=True).strip()
assert head_commit == PINNED_COMMIT, f"Commit 不匹配: {head_commit} != {PINNED_COMMIT}"

# 3. 施加 3 项最小非语义兼容补丁（仅作用于 /content/runtime，并生成审计补丁文件存入 Drive）
patches_applied = []

# 补丁 A: 允许 MMDetection 2.11 与 MMCV-full 1.7.2 (同属 MMCV 1.x API) 协同工作
mmdet_init = RUNTIME_REPO / "trackers/SiamDT/libs/swintransformer/mmdet/__init__.py"
init_text = mmdet_init.read_text(encoding="utf-8")
if "mmcv_maximum_version = '1.4.0'" in init_text:
    mmdet_init.write_text(
        init_text.replace("mmcv_maximum_version = '1.4.0'", "mmcv_maximum_version = '1.7.2'"),
        encoding="utf-8"
    )
    patches_applied.append("mmdet/__init__.py: mmcv_maximum_version 1.4.0 -> 1.7.2")

# 补丁 B: 修复 PyTorch >= 1.11 下 multiclass_nms 中 labels (CPU) 与 inds (CUDA) 跨设备索引异常
bbox_nms = RUNTIME_REPO / "trackers/SiamDT/libs/swintransformer/mmdet/core/post_processing/bbox_nms.py"
nms_text = bbox_nms.read_text(encoding="utf-8")
target_line = "labels = torch.arange(num_classes, dtype=torch.long)"
replacement_line = "labels = torch.arange(num_classes, dtype=torch.long, device=multi_scores.device)"
if target_line in nms_text:
    bbox_nms.write_text(nms_text.replace(target_line, replacement_line), encoding="utf-8")
    patches_applied.append("bbox_nms.py: labels device aligned with multi_scores.device")

# 补丁 C: 数据集路径环境变量化（优先读取环境变量 ANTI_UAV410_ROOT，默认指向统一数据集目录，不破坏原有接口）
wrappers_py = RUNTIME_REPO / "trackers/SiamDT/datasets/wrappers.py"
w_text = wrappers_py.read_text(encoding="utf-8")
hardcoded_uav_path = "'/media/data2/TrackingDatasets/Anti-UAV410/Anti-UAV/'"
env_uav_path = "os.environ.get('ANTI_UAV410_ROOT', '/content/drive/MyDrive/SiamDT_Project/datasets/AntiUAV410/')"
if hardcoded_uav_path in w_text:
    if "import os\n" not in w_text:
        w_text = "import os\n" + w_text
    w_text = w_text.replace(hardcoded_uav_path, env_uav_path)
    wrappers_py.write_text(w_text, encoding="utf-8")
    patches_applied.append("datasets/wrappers.py: root_dir parameterized via ANTI_UAV410_ROOT")

# 4. 如阶段代码库存在 phase6_siamdt_innovations，自动将新增配置/模块软链或同步至运行目录（不覆盖原始基线文件）
innov_dir = RUNTIME_STAGE_REPO / "phase6_siamdt_innovations"
if innov_dir.exists():
    for sub in ("configs", "trackers"):
        src_sub = innov_dir / sub
        dst_sub = RUNTIME_REPO / "trackers/SiamDT" / sub
        if src_sub.exists() and dst_sub.exists():
            for item in src_sub.glob("*.py"):
                dst_item = dst_sub / item.name
                shutil.copy2(item, dst_item)
                stage_sync_status["mounted_innovations"].append(str(dst_item))

# 同步一份到 /content/runtime/repos/Anti-UAV410_official 供 run_phase2_multi_tracker_colab.py 直接使用
runtime_phase2_siamdt = RUNTIME_ROOT / "repos/Anti-UAV410_official"
if not runtime_phase2_siamdt.exists():
    runtime_phase2_siamdt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUNTIME_REPO, runtime_phase2_siamdt, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)

diff_output = subprocess.check_output(["git", "-C", str(RUNTIME_REPO), "diff"], text=True)
patch_file = DRIVE_ROOT / "repos/patches" / f"runtime_compat_{PINNED_COMMIT[:8]}.patch"
patch_file.parent.mkdir(parents=True, exist_ok=True)
patch_file.write_text(diff_output, encoding="utf-8")

print(json.dumps({
    "cell": "CELL_2_DUAL_GIT_SYNC_AND_PATCH",
    "simdt_stage_repo": stage_sync_status,
    "official_antiuav410_repo": {
        "pinned_commit": head_commit,
        "runtime_repo": str(RUNTIME_REPO),
        "patches_applied": patches_applied,
        "audit_patch_saved": str(patch_file),
    }
}, indent=2, ensure_ascii=False))
'''

# 兼容旧引用别名
CELL_2_SYNC_REPO_AND_PATCH = CELL_2_DUAL_GIT_SYNC_AND_PATCH

# ==============================================================================
# [CELL 3] 数据集与多算法权重完整性核验（SHA-256 校验与软链接）
# ==============================================================================
CELL_3_ASSETS_VERIFY_AND_PREPARE = r'''
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get("SIAMDT_DRIVE_ROOT", "/content/drive/MyDrive/SiamDT_Project"))
RUNTIME_SIAMDT = Path("/content/runtime/Anti-UAV410/trackers/SiamDT")

# 1. 官方 SiamDT 权重 Google Drive ID 与已知权威 SHA-256 哈希
OFFICIAL_ASSETS = {
    "siamdt_official_ckpt": {
        "gdrive_id": "1ovjDnsB5FDGIY0D1smqW9U08J4ROMjW2",
        "is_zip": True,
        "zip_path": DRIVE_ROOT / "datasets/archives/checkpoints.zip",
        "zip_sha256": "2ec8c111397e7d143c0e4f4996c1c0f173bd05dc44f3e680f13a5b1387034bf2",
        "target_path": DRIVE_ROOT / "weights/in_domain_antiuav410/SiamDT_Baseline/siamdt_swin_tiny_sgd.pth",
        "expected_bytes": 499427901,
        "expected_sha256": "b4c7e0e0490881f81a64352f4db213934b15194f3ad76c3937aee0204e4c6c38",
        "runtime_link": RUNTIME_SIAMDT / "checkpoints/siamdt_swin_tiny_sgd.pth",
        "legacy_search": [
            DRIVE_ROOT / "weights/checkpoints/siamdt_swin_tiny_sgd.pth",
            DRIVE_ROOT / "weights/official_checkpoints/siamdt_swin_tiny_sgd.pth",
        ]
    },
    "swin_tiny_pretrained": {
        "gdrive_id": "160tU2B5VproDAQSz6lYw-7jrNC3BsAHj",
        "is_zip": False,
        "target_path": DRIVE_ROOT / "weights/foundation_backbones/swin_tiny/cascade_mask_rcnn_swin_tiny.pth.tar",
        "expected_bytes": 121266712,
        "expected_sha256": "f8f9a52b47d689bc3e6d0ea97e5001846366353fdbf2e9d36f83dfe5c7bf4000",
        "runtime_link": RUNTIME_SIAMDT / "pretrained_weights/cascade_mask_rcnn_swin_tiny.pth.tar",
        "legacy_search": [
            DRIVE_ROOT / "weights/pretrained/cascade_mask_rcnn_swin_tiny.pth.tar",
        ]
    }
}

def calc_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

asset_status = {}
for name, spec in OFFICIAL_ASSETS.items():
    target = spec["target_path"]
    target.parent.mkdir(parents=True, exist_ok=True)

    # 若目标不存在或大小不完整，先从旧目录查找并无损复制（不删除原文件）
    if not target.exists() or target.stat().st_size != spec["expected_bytes"]:
        for legacy in spec["legacy_search"]:
            if legacy.exists() and legacy.stat().st_size == spec["expected_bytes"]:
                print(f"[复用] 从已有路径复用 {legacy} -> {target}")
                shutil.copy2(legacy, target)
                break

    # 若仍不存在或大小不完整，检查是否有已下载的 checkpoints.zip 或使用 gdown 下载
    if not target.exists() or target.stat().st_size != spec["expected_bytes"]:
        import gdown
        if spec["is_zip"]:
            zpath = spec["zip_path"]
            if not zpath.exists() and (DRIVE_ROOT / "archives/checkpoints.zip").exists():
                zpath = DRIVE_ROOT / "archives/checkpoints.zip"
            if not zpath.exists() or zpath.stat().st_size < 1000000:
                zpath.parent.mkdir(parents=True, exist_ok=True)
                gdown.download(id=spec["gdrive_id"], output=str(zpath), resume=True, quiet=False)
            with zipfile.ZipFile(zpath, "r") as zf:
                for m in zf.infolist():
                    if m.filename.endswith("siamdt_swin_tiny_sgd.pth"):
                        with zf.open(m) as src, target.open("wb") as dst:
                            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        else:
            gdown.download(id=spec["gdrive_id"], output=str(target), resume=True, quiet=False)

    # 校验字节数与 SHA-256
    actual_bytes = target.stat().st_size
    actual_sha = calc_sha256(target)
    verified = (actual_bytes == spec["expected_bytes"]) and (actual_sha == spec["expected_sha256"])

    # 建立到 /content/runtime/Anti-UAV410/trackers/SiamDT 的软链接
    rlink = spec["runtime_link"]
    rlink.parent.mkdir(parents=True, exist_ok=True)
    if rlink.exists() or rlink.is_symlink():
        rlink.unlink()
    rlink.symlink_to(target)

    asset_status[name] = {
        "drive_path": str(target),
        "runtime_symlink": str(rlink),
        "bytes": actual_bytes,
        "sha256": actual_sha,
        "verified_exact": verified,
    }

# 2. 扫描并核验 Tier-1 核心实验池（多算法横向对比池）已就位权重
CORE_POOL_WEIGHT_DIRS = {
    "Core1_FocusTrack": DRIVE_ROOT / "weights/in_domain_antiuav410/FocusTrack",
    "Core2_MemLoTrack": DRIVE_ROOT / "weights/in_domain_antiuav410/MemLoTrack",
    "Core3_DAM4SAM": DRIVE_ROOT / "weights/zero_shot_sota/DAM4SAM",
    "Core4_SAMOSA_SAMURAI": DRIVE_ROOT / "weights/zero_shot_sota/SAMOSA",
    "Core5_UETrack_MCITrack": DRIVE_ROOT / "weights/zero_shot_sota/UETrack_MCITrack",
    "Core6_ReFocus_TIR_SUTrack": DRIVE_ROOT / "weights/zero_shot_sota/ReFocus_TIR_SUTrack",
    "Foundation_SAM2_1": DRIVE_ROOT / "weights/foundation_backbones/sam2_1",
    "Foundation_DINOv2": DRIVE_ROOT / "weights/foundation_backbones/dinov2",
}

core_pool_inventory = {}
for pool_key, wdir in CORE_POOL_WEIGHT_DIRS.items():
    wdir.mkdir(parents=True, exist_ok=True)
    files_info = []
    for wf in sorted(wdir.rglob("*")):
        if wf.is_file() and wf.suffix.lower() in (".pth", ".pt", ".tar", ".safetensors", ".bin", ".ckpt"):
            files_info.append({
                "name": wf.name,
                "bytes": wf.stat().st_size,
                "sha256_prefix": calc_sha256(wf)[:16],
            })
    core_pool_inventory[pool_key] = {
        "dir": str(wdir),
        "ready_files_count": len(files_info),
        "files": files_info,
    }

# 3. 检查数据集目录与已准备序列（支持无损软链复用旧版 data/ 下的单序列）
DATASET_ROOT = DRIVE_ROOT / "datasets/AntiUAV410"
legacy_seq_dirs = list((DRIVE_ROOT / "data").rglob("IR_label.json")) if (DRIVE_ROOT / "data").exists() else []
for lbl in legacy_seq_dirs:
    seq_dir = lbl.parent
    split_name = seq_dir.parent.name if seq_dir.parent.name in ("train", "val", "test") else "test"
    dest_seq = DATASET_ROOT / split_name / seq_dir.name
    if not dest_seq.exists():
        dest_seq.parent.mkdir(parents=True, exist_ok=True)
        dest_seq.symlink_to(seq_dir, target_is_directory=True)

split_counts = {}
for sp in ("train", "val", "test"):
    sp_dir = DATASET_ROOT / sp
    seqs = [
        p.name for p in sp_dir.iterdir()
        if (p.is_dir() or p.is_symlink()) and (p / "IR_label.json").exists()
    ] if sp_dir.exists() else []
    split_counts[sp] = {"ready_sequences_count": len(seqs), "sample_sequences": sorted(seqs)[:5]}

# 检查冻结子集清单文件 (datasets/diagnostic_splits/*.json)
splits_dir = DRIVE_ROOT / "datasets/diagnostic_splits"
frozen_splits = [p.name for p in sorted(splits_dir.glob("*.json"))] if splits_dir.exists() else []

print(json.dumps({
    "cell": "CELL_3_ASSETS_VERIFY_AND_PREPARE",
    "siamdt_official_weights": asset_status,
    "core_pool_weights_inventory": core_pool_inventory,
    "datasets_summary": split_counts,
    "frozen_diagnostic_split_manifests": frozen_splits,
}, indent=2, ensure_ascii=False))
'''

# ==============================================================================
# [CELL 4] 双轨运行环境秒级恢复（Tier-1: env_siamdt_cu118 + Tier-2: env_modern_pyt2）
# ==============================================================================
# 说明（2026-11-04）：原单元使用 IPython 专用的 %%bash 魔法命令，经远程接管桥用普通 Python 进程执行时会报语法错误；
# 现改为把同一段 bash 脚本交给 subprocess.run(['bash', '-c', ...]) 执行（脚本内容未改动），在 Notebook 与普通 Python 中都可运行。
CELL_4_BASH_SCRIPT = r'''set -euo pipefail

DRIVE_ROOT="${SIAMDT_DRIVE_ROOT:-/content/drive/MyDrive/SiamDT_Project}"
SIAMDT_ENV="/content/envs/siamdt-cu118"
MODERN_ENV="/content/envs/modern-pyt2"
REQ_DIR="$DRIVE_ROOT/environments/requirements"
mkdir -p "$REQ_DIR" /content/envs /content/uv-bootstrap /content/cache/pip /content/cache/uv

# 使用 Colab 本地 NVMe 缓存目录，避免 Google Drive FUSE 不支持 flock/hardlink 导致 uv 报 Operation not permitted (os error 1)
export PIP_CACHE_DIR="/content/cache/pip"
export UV_CACHE_DIR="/content/cache/uv"
export UV_LINK_MODE="copy"
export UV_PYTHON_INSTALL_DIR="/content/uv-python"

T_START=$(date +%s)

# 1. 安装 uv 极速包管理引导器以实现 45~60 秒级环境恢复
if [ ! -x "/content/uv-bootstrap/bin/uv" ]; then
    python3 -m pip install --quiet --target /content/uv-bootstrap uv==0.9.7
fi
UV="/content/uv-bootstrap/bin/uv"

# 2. 创建并恢复 Tier-1 环境 (env_siamdt_cu118: Python 3.10.19 + Torch 2.0.0+cu118 + MMCV-full 1.7.2)
if [ ! -x "$SIAMDT_ENV/bin/python" ]; then
    "$UV" python install 3.10.19
    "$UV" venv --seed --python 3.10.19 "$SIAMDT_ENV"
fi
PY1="$SIAMDT_ENV/bin/python"

# 幂等检查：若已通过全套导入门禁则直接跳过安装
if ! "$PY1" -c "import torch, torchvision, mmcv, timm, shapely, visdom; from mmcv.ops import nms, roi_align; assert torch.__version__=='2.0.0+cu118' and mmcv.__version__=='1.7.2'" 2>/dev/null; then
    echo "[Tier-1: env_siamdt_cu118] 正在构建恢复 SiamDT 专用环境..."
    "$UV" pip install --python "$PY1" setuptools==68.2.2 wheel==0.41.3 cython==0.29.35 yapf==0.40.1
    "$UV" pip install --python "$PY1" torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    "$PY1" -m pip install --no-deps --only-binary=:all: mmcv-full==1.7.2 \
        -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
    "$UV" pip install --python "$PY1" numpy==1.24.4 scipy==1.10.1 matplotlib==3.7.5 opencv-python-headless==4.8.1.78 \
        pillow==10.2.0 pycocotools==2.0.7 shapely==2.0.3 six==1.16.0 terminaltables==3.1.10 \
        timm==0.4.12 tqdm==4.66.2 visdom==0.2.4 addict==2.4.0
fi

# 3. 为 PyTorch 2.0 注入 torch._six 兼容垫片（供旧版 Apex O1 混合精度无缝调用）
SITE_PKGS=$("$PY1" -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_PKGS/torch/_six.py" <<'EOF'
import collections.abc as container_abcs
import math
inf = math.inf
nan = math.nan
string_classes = (str, bytes)
int_classes = int
EOF

# 4. 构建 Swin Transformer Object Detection (develop 模式)
if [ -d "/content/runtime/Anti-UAV410/trackers/SiamDT/libs/swintransformer" ]; then
    cd /content/runtime/Anti-UAV410/trackers/SiamDT/libs/swintransformer
    "$PY1" setup.py develop --no-deps >/dev/null
fi

# 5. 创建并恢复 Tier-2 环境 (env_modern_pyt2: 复用 Colab 原生 PyTorch 2.x + 现代对比算法依赖)
if [ ! -x "$MODERN_ENV/bin/python" ]; then
    "$UV" venv --system-site-packages --seed "$MODERN_ENV"
fi
PY2="$MODERN_ENV/bin/python"

if ! "$PY2" -c "import torch, timm, einops, filterpy, huggingface_hub, safetensors" 2>/dev/null; then
    echo "[Tier-2: env_modern_pyt2] 正在安装现代对比算法依赖..."
    "$UV" pip install --python "$PY2" timm einops filterpy huggingface_hub safetensors hydra-core iopath lmdb pandas pyyaml tqdm
fi

# 6. 保存锁定依赖清单到 Drive
"$PY1" -m pip freeze --all > "$REQ_DIR/tier1_siamdt_cu118_freeze.txt"
"$PY2" -m pip freeze > "$REQ_DIR/tier2_modern_pyt2_freeze.txt"

T_ELAPSED=$(( $(date +%s) - T_START ))
echo "[完成] 双轨环境 (Tier-1: $SIAMDT_ENV | Tier-2: $MODERN_ENV) 已就绪，耗时 ${T_ELAPSED} 秒。"
'''

CELL_4_BUILD_ENVIRONMENTS = (
    "import subprocess\n"
    "import sys\n"
    "\n"
    "# 原 %%bash 单元内容，原样交给 bash 执行（普通 Python 进程与 Notebook 均可运行）\n"
    "BASH_SCRIPT = r\"\"\"" + CELL_4_BASH_SCRIPT + "\"\"\"\n"
    "\n"
    "# 逐行转发输出（Notebook 的 sys.stdout 没有 fileno，不能直接作为子进程 stdout）\n"
    "proc = subprocess.Popen(['bash', '-c', BASH_SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
    "                        text=True, encoding='utf-8', errors='replace', bufsize=1)\n"
    "for line in proc.stdout:\n"
    "    print(line, end='', flush=True)\n"
    "if proc.wait() != 0:\n"
    "    raise RuntimeError(f'Cell 4 环境恢复脚本失败，returncode={proc.returncode}')\n"
)

# ==============================================================================
# [CELL 5] A100 (sm_80) / L4 (sm_89) 算子、245 参数张量 strict=True 加载与 5 帧冒烟测试
# ==============================================================================
CELL_5_MINIMAL_VERIFICATION = r'''
import subprocess
import sys

DRIVE_ROOT = "/content/drive/MyDrive/SiamDT_Project"
PY1 = "/content/envs/siamdt-cu118/bin/python"

verify_code = r"""
import os, sys, json, math, time, torch, torchvision, mmcv
import numpy as np
from pathlib import Path
from mmcv.ops import nms, roi_align, get_compiling_cuda_version

DRIVE_ROOT = Path(os.environ.get("SIAMDT_DRIVE_ROOT", "/content/drive/MyDrive/SiamDT_Project"))
SIAMDT_DIR = Path("/content/runtime/Anti-UAV410/trackers/SiamDT")
os.chdir(SIAMDT_DIR)
sys.path.insert(0, str(SIAMDT_DIR))

# 1. 验证 init_paths 与关键模块导入
import init_paths
import libs.data as data
import libs.ops as ops
from mmdet.models import build_detector
from mmcv import Config
from trackers import SiamDTTracker

arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
report = {
    "cell": "CELL_5_MINIMAL_VERIFICATION",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "mmcv": mmcv.__version__,
    "mmcv_cuda": str(get_compiling_cuda_version()),
    "cuda_available": torch.cuda.is_available(),
    "compiled_cuda_archs_support_sm80_sm89": {
        "sm_80_a100_supported": any("80" in a for a in arch_list),
        "sm_89_l4_supported": any(("89" in a or "86" in a or "90" in a) for a in arch_list),
        "arch_list": arch_list,
    },
    "init_paths_import": "PASS",
}

# 2. 验证 A100 (sm_80 主力 70%~80%) / L4 (sm_89 辅助 10%~15%) GPU 与 MMCV CUDA 算子 (NMS + RoIAlign 前向与反向)
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    if (cc_major, cc_minor) == (8, 0) or "A100" in gpu_name.upper():
        gpu_profile = "A100_HIGH_THROUGHPUT (sm_80, 主力算力 70%~80%：承担全量训练/消融/多算法全量评测)"
    elif (cc_major, cc_minor) == (8, 9) or "L4" in gpu_name.upper():
        gpu_profile = "L4_24GB_SAFE (sm_89, 辅助算力 10%~15%：承担轻量级子集推理/5帧冒烟/Phase5特征导出)"
    else:
        gpu_profile = f"GENERIC_CUDA_GPU (sm_{cc_major}{cc_minor}, {vram_gb}GB)"

    report["gpu_name"] = gpu_name
    report["vram_total_gb"] = vram_gb
    report["compute_capability"] = [cc_major, cc_minor]
    report["compute_arch_tag"] = f"sm_{cc_major}{cc_minor}"
    report["auto_gpu_profile"] = gpu_profile

    boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10], [30, 30, 40, 40]], dtype=torch.float32, device=dev)
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32, device=dev)
    dets, keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [0, 2], f"NMS 异常: {keep.tolist()}"
    report["mmcv_cuda_nms"] = "PASS"

    feat = torch.arange(16, dtype=torch.float32, device=dev).view(1, 1, 4, 4).requires_grad_()
    rois = torch.tensor([[0, 0, 0, 3, 3]], dtype=torch.float32, device=dev)
    out = roi_align(feat, rois, (2, 2), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    out.sum().backward()
    assert out.shape == (1, 1, 2, 2) and feat.grad is not None and torch.isfinite(feat.grad).all()
    torch.cuda.synchronize()
    report["mmcv_cuda_roi_align_fwd_bwd"] = "PASS"
else:
    report["gpu_warning"] = "当前为 CPU 运行时，CUDA 算子与 GPU 推理待切换至 Colab A100 (sm_80) 或 L4 (sm_89) 后自动执行验证"

# 3. 验证官方 SiamDTRCNN 模型构建与 245 个参数张量的 strict=True 严格加载
cfg = Config.fromfile("configs/siamdt_swin_tiny_sgd.py")
model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
ckpt_path = Path("checkpoints/siamdt_swin_tiny_sgd.pth")
if ckpt_path.exists():
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    incompatible = model.load_state_dict(state_dict, strict=True)
    report["strict_245_tensor_load"] = {
        "status": "PASS",
        "tensor_count": len(state_dict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    assert len(state_dict) == 245, f"参数张量数量异常: {len(state_dict)}"

# 4. 5 帧冒烟测试（Smoke Test）：若存在已就绪序列且有 GPU，则执行 5 帧真实跟踪并核验统一遥测 Schema
candidate_dirs = [
    DRIVE_ROOT / "datasets/AntiUAV410/val",
    DRIVE_ROOT / "datasets/AntiUAV410/test",
]
ready_seqs = []
for cdir in candidate_dirs:
    if cdir.exists():
        ready_seqs.extend([p for p in sorted(cdir.iterdir()) if (p / "IR_label.json").exists()])

if ready_seqs and torch.cuda.is_available() and ckpt_path.exists():
    seq = ready_seqs[0]
    lbl = json.loads((seq / "IR_label.json").read_text(encoding="utf-8"))
    frames = sorted(seq.glob("*.jpg"))[:5]
    transforms = data.BasicPairTransforms(train=False)
    tracker = SiamDTTracker(
        "configs/siamdt_swin_tiny_sgd.py",
        str(ckpt_path),
        transforms,
        name_suffix="smoke",
        visualize=False
    )

    gt0 = [float(x) for x in lbl["gt_rect"][0]]
    init_xyxy = np.array([gt0[0], gt0[1], gt0[0] + gt0[2] - 1.0, gt0[1] + gt0[3] - 1.0], dtype=np.float32)
    img0 = ops.read_image(str(frames[0]), tracker.color_fmt)
    t0 = time.perf_counter()
    tracker.init(img0, init_xyxy)
    dt0_ms = (time.perf_counter() - t0) * 1000.0

    smoke_telemetry = [{
        "seq_id": seq.name,
        "frame_id": 0,
        "exist": int(lbl["exist"][0]) if "exist" in lbl else 1,
        "gt_bbox": gt0,
        "pred_bbox": gt0,
        "iou": 1.0,
        "sa_score": 1.0,
        "center_error": 0.0,
        "target_size": round(math.sqrt(max(0.0, gt0[2] * gt0[3])), 4),
        "confidence": 1.0,
        "template_updated": True,
        "latency_ms": round(dt0_ms, 3),
    }]

    for idx, fpath in enumerate(frames[1:], start=1):
        img = ops.read_image(str(fpath), tracker.color_fmt)
        t_start = time.perf_counter()
        results, up_flag = tracker.update(img, return_all=True)
        dt_ms = (time.perf_counter() - t_start) * 1000.0

        max_ind = int(results[:, -1].argmax())
        box_xyxy = results[max_ind, :4]
        conf = float(results[max_ind, 4])
        pred_xywh = [
            round(float(box_xyxy[0]), 4),
            round(float(box_xyxy[1]), 4),
            round(float(box_xyxy[2] - box_xyxy[0] + 1.0), 4),
            round(float(box_xyxy[3] - box_xyxy[1] + 1.0), 4),
        ]
        gt_f = [float(x) for x in lbl["gt_rect"][idx]]
        exist_f = int(lbl["exist"][idx]) if "exist" in lbl else 1
        # 计算 IoU 与中心误差
        if exist_f == 1 and len(gt_f) == 4 and gt_f[2] > 0 and gt_f[3] > 0:
            ox0 = max(pred_xywh[0], gt_f[0])
            oy0 = max(pred_xywh[1], gt_f[1])
            ox1 = min(pred_xywh[0] + pred_xywh[2], gt_f[0] + gt_f[2])
            oy1 = min(pred_xywh[1] + pred_xywh[3], gt_f[1] + gt_f[3])
            if ox1 > ox0 and oy1 > oy0:
                inter = (ox1 - ox0) * (oy1 - oy0)
                union = pred_xywh[2] * pred_xywh[3] + gt_f[2] * gt_f[3] - inter
                iou_f = float(inter / union) if union > 0 else 0.0
            else:
                iou_f = 0.0
            sa_f = iou_f
            ce_f = math.hypot(
                (pred_xywh[0] + pred_xywh[2] * 0.5) - (gt_f[0] + gt_f[2] * 0.5),
                (pred_xywh[1] + pred_xywh[3] * 0.5) - (gt_f[1] + gt_f[3] * 0.5),
            )
            ts_f = math.sqrt(max(0.0, gt_f[2] * gt_f[3]))
        else:
            iou_f = 0.0
            sa_f = 0.0
            ce_f = -1.0
            ts_f = 0.0

        smoke_telemetry.append({
            "seq_id": seq.name,
            "frame_id": idx,
            "exist": exist_f,
            "gt_bbox": gt_f,
            "pred_bbox": pred_xywh,
            "iou": round(iou_f, 6),
            "sa_score": round(sa_f, 6),
            "center_error": round(ce_f, 4),
            "target_size": round(ts_f, 4),
            "confidence": round(conf, 6),
            "template_updated": bool(up_flag),
            "latency_ms": round(dt_ms, 3),
        })

    report["smoke_5frame_inference"] = {
        "status": "PASS",
        "sequence": seq.name,
        "frames_tested": len(smoke_telemetry),
        "sample_telemetry": smoke_telemetry,
    }

diag_file = DRIVE_ROOT / "environments/diagnostics" / f"a100_verification_{int(time.time())}.json"
diag_file.parent.mkdir(parents=True, exist_ok=True)
diag_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
"""

subprocess.run([PY1, "-c", verify_code], check=True)
'''

# ==============================================================================
# [CELL 6] Phase 2 统一逐帧遥测评测器（支持 discovery_set.json / val90_full.json 及断点续评）
# ==============================================================================
CELL_6_UNIFIED_TELEMETRY_EVALUATOR = r'''
import json
import os
import subprocess
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get("SIAMDT_DRIVE_ROOT", "/content/drive/MyDrive/SiamDT_Project"))
PY1 = "/content/envs/siamdt-cu118/bin/python"

# ------------------------------------------------------------------------------
# 评测任务配置参数（支持环境变量覆盖或直接在单元格修改）
# ------------------------------------------------------------------------------
# 开关：Gate 1 批准后将 RUN_PHASE2_BATCH_EVAL 设为 True（或设置环境变量 SIMDT_RUN_EVAL=1）即可启动全子集断点续评
RUN_PHASE2_BATCH_EVAL = os.environ.get("SIMDT_RUN_EVAL", "0") == "1"
EVAL_SPLIT_JSON = os.environ.get("SIMDT_EVAL_SPLIT_JSON", "discovery_set.json")  # 可选: discovery_set.json / val90_full.json / holdout_verification.json
DATASET_SUBSET = os.environ.get("SIMDT_DATASET_SUBSET", "val")                   # val 或 test (Phase 1-6 期间锁定 val)
TRACKER_GROUP = os.environ.get("SIMDT_TRACKER_GROUP", "group1_in_domain")        # group1_in_domain 或 group2_zero_shot
TRACKER_NAME = os.environ.get("SIMDT_TRACKER_NAME", "SiamDT_Baseline")
CONFIG_REL_PATH = os.environ.get("SIMDT_CONFIG_PATH", "configs/siamdt_swin_tiny_sgd.py")
CKPT_REL_PATH = os.environ.get("SIMDT_CKPT_PATH", "checkpoints/siamdt_swin_tiny_sgd.pth")
MAX_SEQUENCES = int(os.environ.get("SIMDT_MAX_SEQUENCES", "0"))                  # 0 表示跑完清单内全部序列

evaluator_script = r"""
import os, sys, json, math, time
import numpy as np
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get("SIAMDT_DRIVE_ROOT", "/content/drive/MyDrive/SiamDT_Project"))
SIAMDT_DIR = Path("/content/runtime/Anti-UAV410/trackers/SiamDT")
STAGE_META_FILE = Path("/content/runtime/simdt_stage_version.json")

os.chdir(SIAMDT_DIR)
sys.path.insert(0, str(SIAMDT_DIR))

run_batch = os.environ.get("_CELL6_RUN_BATCH", "0") == "1"
split_json_name = os.environ.get("_CELL6_SPLIT_JSON", "discovery_set.json")
dataset_subset = os.environ.get("_CELL6_SUBSET", "val")
tracker_group = os.environ.get("_CELL6_GROUP", "group1_in_domain")
tracker_name = os.environ.get("_CELL6_TRACKER", "SiamDT_Baseline")
cfg_file = os.environ.get("_CELL6_CFG", "configs/siamdt_swin_tiny_sgd.py")
ckp_file = os.environ.get("_CELL6_CKPT", "checkpoints/siamdt_swin_tiny_sgd.pth")
max_seqs = int(os.environ.get("_CELL6_MAX_SEQS", "0"))

# 读取当前阶段代码的 Git Commit SHA
git_short_sha = "local00"
if STAGE_META_FILE.exists():
    try:
        git_short_sha = json.loads(STAGE_META_FILE.read_text(encoding="utf-8")).get("short_sha", "local00")
    except Exception:
        pass

split_stem = Path(split_json_name).stem
split_file = DRIVE_ROOT / "datasets/diagnostic_splits" / split_json_name
dataset_dir = DRIVE_ROOT / "datasets/AntiUAV410" / dataset_subset

# 解析冻结子集序列清单（兼容多种 JSON 结构）
def resolve_sequence_list(split_path: Path, fallback_dir: Path):
    if split_path.exists():
        raw = json.loads(split_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [x["seq_id"] if isinstance(x, dict) and "seq_id" in x else str(x) for x in raw], str(split_path)
        if isinstance(raw, dict):
            for key in ("sequences", "discovery_sequences", "holdout_sequences", "val_sequences", "seq_ids"):
                if key in raw and isinstance(raw[key], list):
                    items = raw[key]
                    return [x["seq_id"] if isinstance(x, dict) and "seq_id" in x else str(x) for x in items], str(split_path)
    # 若子集 JSON 尚未同步，则扫描 dataset_dir 下已就绪序列作为兜底预览
    if fallback_dir.exists():
        seqs = sorted([p.name for p in fallback_dir.iterdir() if (p / "IR_label.json").exists()])
        return seqs, f"FALLBACK_DIR_SCAN:{fallback_dir}"
    return [], "NONE"

seq_list, manifest_source = resolve_sequence_list(split_file, dataset_dir)
if max_seqs > 0:
    seq_list = seq_list[:max_seqs]

# 建立 Google Drive 持久化评测与逐帧遥测输出目录
eval_root = DRIVE_ROOT / "experiments/benchmark_eval" / tracker_group / f"{tracker_name}_{split_stem}_{git_short_sha}"
txt_box_dir = eval_root / "pred_bboxes_txt"
uavtir_json_dir = eval_root / "pred_bboxes_uavtir_json"
telemetry_eval_dir = eval_root / "per_frame_telemetry"
telemetry_diag_dir = DRIVE_ROOT / "experiments/failure_diagnosis/per_frame_telemetry" / f"{tracker_name}_{split_stem}"

for d in (txt_box_dir, uavtir_json_dir, telemetry_eval_dir, telemetry_diag_dir):
    d.mkdir(parents=True, exist_ok=True)

# 扫描已完成序列（断线重连自动跳过校验）
already_completed = []
pending_sequences = []
for seq_id in seq_list:
    seq_dir = dataset_dir / seq_id
    lbl_file = seq_dir / "IR_label.json"
    txt_file = txt_box_dir / f"{seq_id}.txt"
    jsonl_file = telemetry_eval_dir / f"{seq_id}.jsonl"
    if lbl_file.exists() and txt_file.exists() and jsonl_file.exists():
        try:
            expected_len = len(json.loads(lbl_file.read_text(encoding="utf-8"))["gt_rect"])
            txt_lines = [ln for ln in txt_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            jsonl_lines = [ln for ln in jsonl_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if expected_len > 0 and len(txt_lines) == expected_len and len(jsonl_lines) == expected_len:
                already_completed.append(seq_id)
                continue
        except Exception:
            pass
    pending_sequences.append(seq_id)

# 自适应 GPU 档位检测 (Auto GPU Profile: A100 sm_80 主力 70%~80% vs L4 sm_89 辅助 10%~15%)
def detect_eval_gpu_profile():
    try:
        import torch
        if torch.cuda.is_available():
            gname = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            torch.backends.cudnn.benchmark = True
            if cc == (8, 0) or "A100" in gname.upper() or vram_gb >= 38.0:
                return {
                    "profile": "A100_HIGH_THROUGHPUT",
                    "gpu_name": gname,
                    "compute_cap": f"sm_{cc[0]}{cc[1]}",
                    "vram_gb": vram_gb,
                    "compute_allocation_role": "Colab A100 主力算力 (70%~80%)：全量统一评测 / 多算法横向对比 / 高负载推理",
                    "empty_cache_interval": 25,
                }
            elif cc == (8, 9) or "L4" in gname.upper():
                return {
                    "profile": "L4_24GB_SAFE",
                    "gpu_name": gname,
                    "compute_cap": f"sm_{cc[0]}{cc[1]}",
                    "vram_gb": vram_gb,
                    "compute_allocation_role": "Colab L4 辅助算力 (10%~15%)：24GB 显存安全模式，适合轻量级子集推理 (discovery_set) / 5帧冒烟 / Phase5特征导出",
                    "empty_cache_interval": 5,
                }
            return {
                "profile": "GENERIC_GPU_PROFILE",
                "gpu_name": gname,
                "compute_cap": f"sm_{cc[0]}{cc[1]}",
                "vram_gb": vram_gb,
                "compute_allocation_role": "通用 GPU 评测模式",
                "empty_cache_interval": 10,
            }
    except Exception:
        pass
    return {
        "profile": "CPU_DRY_RUN",
        "gpu_name": "NO_GPU",
        "compute_cap": "cpu",
        "vram_gb": 0.0,
        "compute_allocation_role": "本地/CPU 探针模式 (10%~15% 调度与清单检查)",
        "empty_cache_interval": 0,
    }

gpu_profile_cfg = detect_eval_gpu_profile()

if not run_batch:
    print(json.dumps({
        "cell": "CELL_6_UNIFIED_TELEMETRY_EVALUATOR",
        "mode": "DRY_RUN_CHECKPOINT_INSPECTION (将 RUN_PHASE2_BATCH_EVAL 设为 True 可立即执行评测)",
        "auto_gpu_profile": gpu_profile_cfg,
        "tracker_name": tracker_name,
        "tracker_group": tracker_group,
        "split_manifest": manifest_source,
        "stage_git_sha": git_short_sha,
        "eval_output_dir": str(eval_root),
        "diagnosis_telemetry_dir": str(telemetry_diag_dir),
        "total_target_sequences": len(seq_list),
        "completed_sequences_skipped_on_resume": len(already_completed),
        "pending_sequences_count": len(pending_sequences),
        "sample_pending_sequences": pending_sequences[:5],
    }, indent=2, ensure_ascii=False))
    sys.exit(0)

import torch
import init_paths
import libs.data as data
import libs.ops as ops
from trackers import SiamDTTracker

transforms = data.BasicPairTransforms(train=False)
tracker = SiamDTTracker(cfg_file, ckp_file, transforms, name_suffix=tracker_name, visualize=False)

def calc_iou_xywh(pred, gt):
    px, py, pw, ph = [float(v) for v in pred]
    gx, gy, gw, gh = [float(v) for v in gt]
    ox0 = max(px, gx)
    oy0 = max(py, gy)
    ox1 = min(px + pw, gx + gw)
    oy1 = min(py + ph, gy + gh)
    if ox1 <= ox0 or oy1 <= oy0:
        return 0.0
    inter = (ox1 - ox0) * (oy1 - oy0)
    union = (pw * ph) + (gw * gh) - inter
    return float(inter / union) if union > 0 else 0.0

def fsync_write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())

seq_sa_scores = {}

# 1. 先载入已完成序列的 SA 得分，实现断点无缝汇总
for seq_id in already_completed:
    jsonl_file = telemetry_eval_dir / f"{seq_id}.jsonl"
    rows = [json.loads(ln) for ln in jsonl_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if rows:
        seq_sa_scores[seq_id] = float(np.mean([r["sa_score"] for r in rows]))
    print(f"[SKIP-RESUMED] 序列 {seq_id} 已存在完整预测框与遥测 ({len(rows)} 帧), SA={seq_sa_scores.get(seq_id, 0.0):.4f}")

# 2. 逐序列执行待测序列并实时落盘 Google Drive
for idx, seq_id in enumerate(pending_sequences, start=1):
    seq_dir = dataset_dir / seq_id
    lbl_file = seq_dir / "IR_label.json"
    if not lbl_file.exists():
        print(f"[警告] 序列 {seq_id} 缺少 {lbl_file}，跳过。")
        continue

    lbl = json.loads(lbl_file.read_text(encoding="utf-8"))
    gt_rects = lbl["gt_rect"]
    exist_flags = lbl.get("exist", [1] * len(gt_rects))
    frames = sorted(seq_dir.glob("*.jpg"))
    total_f = min(len(frames), len(gt_rects))
    if total_f == 0:
        continue

    gt0 = [float(x) for x in gt_rects[0]]
    init_xyxy = np.array([gt0[0], gt0[1], gt0[0] + gt0[2] - 1.0, gt0[1] + gt0[3] - 1.0], dtype=np.float32)
    img0 = ops.read_image(str(frames[0]), tracker.color_fmt)
    t0 = time.perf_counter()
    tracker.init(img0, init_xyxy)
    dt0_ms = (time.perf_counter() - t0) * 1000.0

    pred_bboxes_xywh = [gt0]
    telemetry_rows = [{
        "seq_id": seq_id,
        "frame_id": 0,
        "exist": int(exist_flags[0]),
        "gt_bbox": gt0,
        "pred_bbox": gt0,
        "pred_absent_flag": 0,
        "iou": 1.0 if int(exist_flags[0]) == 1 else 0.0,
        "sa_score": 1.0 if int(exist_flags[0]) == 1 else 0.0,
        "center_error": 0.0 if int(exist_flags[0]) == 1 else -1.0,
        "norm_center_error": 0.0 if int(exist_flags[0]) == 1 else -1.0,
        "target_size": round(math.sqrt(max(0.0, gt0[2] * gt0[3])), 4) if int(exist_flags[0]) == 1 else 0.0,
        "confidence": 1.0,
        "template_updated": True,
        "latency_ms": round(dt0_ms, 3),
    }]

    for f in range(1, total_f):
        img = ops.read_image(str(frames[f]), tracker.color_fmt)
        t_start = time.perf_counter()
        results, up_flag = tracker.update(img, return_all=True)
        dt_ms = (time.perf_counter() - t_start) * 1000.0

        max_ind = int(results[:, -1].argmax())
        box_xyxy = results[max_ind, :4]
        conf = float(results[max_ind, 4])
        pred_xywh = [
            round(float(box_xyxy[0]), 4),
            round(float(box_xyxy[1]), 4),
            round(float(box_xyxy[2] - box_xyxy[0] + 1.0), 4),
            round(float(box_xyxy[3] - box_xyxy[1] + 1.0), 4),
        ]
        pred_bboxes_xywh.append(pred_xywh)

        exist_f = int(exist_flags[f])
        gt_f = [float(x) for x in gt_rects[f]] if len(gt_rects[f]) == 4 else [0.0, 0.0, 0.0, 0.0]
        pred_absent = 1 if (pred_xywh[2] <= 0 or pred_xywh[3] <= 0) else 0

        if exist_f == 1 and gt_f[2] > 0 and gt_f[3] > 0:
            iou_f = calc_iou_xywh(pred_xywh, gt_f) if pred_absent == 0 else 0.0
            sa_f = iou_f
            cx_p, cy_p = pred_xywh[0] + pred_xywh[2] * 0.5, pred_xywh[1] + pred_xywh[3] * 0.5
            cx_g, cy_g = gt_f[0] + gt_f[2] * 0.5, gt_f[1] + gt_f[3] * 0.5
            ce_f = math.hypot(cx_p - cx_g, cy_p - cy_g)
            ts_f = math.sqrt(max(1e-6, gt_f[2] * gt_f[3]))
            nce_f = ce_f / ts_f
        else:
            iou_f = 0.0
            sa_f = 1.0 if pred_absent == 1 else 0.0
            ce_f = -1.0
            nce_f = -1.0
            ts_f = 0.0

        telemetry_rows.append({
            "seq_id": seq_id,
            "frame_id": f,
            "exist": exist_f,
            "gt_bbox": gt_f,
            "pred_bbox": pred_xywh,
            "pred_absent_flag": pred_absent,
            "iou": round(iou_f, 6),
            "sa_score": round(sa_f, 6),
            "center_error": round(ce_f, 4),
            "norm_center_error": round(nce_f, 4),
            "target_size": round(ts_f, 4),
            "confidence": round(conf, 6),
            "template_updated": bool(up_flag),
            "latency_ms": round(dt_ms, 3),
        })

    # 每跑完 1 条序列立即将 .txt 预测框、官方兼容 JSON 和逐帧 .jsonl 实时刷入 Google Drive
    txt_content = "\n".join([f"{b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f}" for b in pred_bboxes_xywh]) + "\n"
    uavtir_json_content = json.dumps({"res": pred_bboxes_xywh})
    jsonl_content = "\n".join([json.dumps(r, ensure_ascii=False) for r in telemetry_rows]) + "\n"

    fsync_write_text(txt_box_dir / f"{seq_id}.txt", txt_content)
    fsync_write_text(uavtir_json_dir / f"{seq_id}.txt", uavtir_json_content)
    fsync_write_text(telemetry_eval_dir / f"{seq_id}.jsonl", jsonl_content)
    fsync_write_text(telemetry_diag_dir / f"{seq_id}.jsonl", jsonl_content)

    seq_sa = float(np.mean([r["sa_score"] for r in telemetry_rows]))
    seq_sa_scores[seq_id] = seq_sa
    if gpu_profile_cfg.get("empty_cache_interval", 0) > 0 and idx % gpu_profile_cfg["empty_cache_interval"] == 0:
        torch.cuda.empty_cache()
    print(f"[{idx}/{len(pending_sequences)}] 序列 {seq_id} ({total_f} 帧) 实时落盘完成 | SA={seq_sa:.4f} | GPU档位={gpu_profile_cfg['profile']}")

    # 实时更新评测进度总表
    summary_payload = {
        "tracker_name": tracker_name,
        "tracker_group": tracker_group,
        "split_name": split_stem,
        "stage_git_sha": git_short_sha,
        "auto_gpu_profile": gpu_profile_cfg,
        "completed_sequences_count": len(seq_sa_scores),
        "total_target_sequences": len(seq_list),
        "mean_sa_score": round(float(np.mean(list(seq_sa_scores.values()))), 6) if seq_sa_scores else 0.0,
        "per_sequence_sa": seq_sa_scores,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    fsync_write_text(eval_root / "eval_progress_summary.json", json.dumps(summary_payload, indent=2, ensure_ascii=False))
"""

env_vars = os.environ.copy()
env_vars.update({
    "_CELL6_RUN_BATCH": "1" if RUN_PHASE2_BATCH_EVAL else "0",
    "_CELL6_SPLIT_JSON": EVAL_SPLIT_JSON,
    "_CELL6_SUBSET": DATASET_SUBSET,
    "_CELL6_GROUP": TRACKER_GROUP,
    "_CELL6_TRACKER": TRACKER_NAME,
    "_CELL6_CFG": CONFIG_REL_PATH,
    "_CELL6_CKPT": CKPT_REL_PATH,
    "_CELL6_MAX_SEQS": str(MAX_SEQUENCES),
})
subprocess.run([PY1, "-c", evaluator_script], env=env_vars, check=True)
'''

# 兼容旧引用别名
CELL_6_LAUNCH_TEMPLATES = CELL_6_UNIFIED_TELEMETRY_EVALUATOR

# ==============================================================================
# [CELL 7] Phase 6 具备自动断点续训 (Auto-Resume) 与 Google Drive 实时落盘的训练引擎
# ==============================================================================
CELL_7_AUTO_RESUME_TRAINING_ENGINE = r'''
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get("SIAMDT_DRIVE_ROOT", "/content/drive/MyDrive/SiamDT_Project"))
SIAMDT_DIR = Path("/content/runtime/Anti-UAV410/trackers/SiamDT")
STAGE_META_FILE = Path("/content/runtime/simdt_stage_version.json")
PY1 = "/content/envs/siamdt-cu118/bin/python"

# ------------------------------------------------------------------------------
# 训练引擎参数配置（支持环境变量覆盖或直接在单元格修改）
# ------------------------------------------------------------------------------
# 开关：将 EXECUTE_TRAINING 设为 True（或设置环境变量 SIMDT_EXECUTE_TRAINING=1）即可启动/续训
EXECUTE_TRAINING = os.environ.get("SIMDT_EXECUTE_TRAINING", "0") == "1"
PHASE_TAG = os.environ.get("SIMDT_TRAIN_PHASE", "phase6")
EXP_NAME = os.environ.get("SIMDT_TRAIN_EXP_NAME", "siamdt_baseline_repro")
TRAIN_CONFIG = os.environ.get("SIMDT_TRAIN_CONFIG", "configs/siamdt_swin_tiny_sgd.py")
PRETRAINED_CKPT = os.environ.get("SIMDT_PRETRAINED_CKPT", "pretrained_weights/cascade_mask_rcnn_swin_tiny.pth.tar")
BASE_DATASET = os.environ.get("SIMDT_BASE_DATASET", "uavtir_train")
RANDOM_SEED = int(os.environ.get("SIMDT_RANDOM_SEED", "42"))

# 1. 读取阶段代码 Git Commit SHA（绑定实验目录与 run_metadata.json）
stage_meta = {"commit_sha": "local_untracked", "short_sha": "local00", "requested_ref": "main"}
if STAGE_META_FILE.exists():
    try:
        stage_meta.update(json.loads(STAGE_META_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
git_sha = os.environ.get("SIMDT_STAGE_COMMIT_SHA", stage_meta.get("short_sha", "local00"))

# 2. 将 --work_dir 严格绑定到 Google Drive: /MyDrive/SiamDT_Project/experiments/runs_training/{phase}_{exp_name}_{git_sha}/
TRAIN_WORK_DIR = DRIVE_ROOT / "experiments/runs_training" / f"{PHASE_TAG}_{EXP_NAME}_{git_sha}"
TRAIN_WORK_DIR.mkdir(parents=True, exist_ok=True)

# 3. 自动断点续训检测逻辑 (Auto-Resume)：
#    优先检查 latest.pth；若 Google Drive FUSE 断联导致 latest.pth 缺失但已有 epoch_*.pth，自动按最大 Epoch 恢复 latest.pth
latest_pth = TRAIN_WORK_DIR / "latest.pth"
if not (latest_pth.exists() and latest_pth.stat().st_size > 0):
    epoch_ckpts = []
    for p in TRAIN_WORK_DIR.glob("epoch_*.pth"):
        m = re.match(r"^epoch_(\d+)\.pth$", p.name)
        if m and p.stat().st_size > 0:
            epoch_ckpts.append((int(m.group(1)), p))
    if epoch_ckpts:
        epoch_ckpts.sort(key=lambda x: x[0])
        newest_epoch, newest_ckpt = epoch_ckpts[-1]
        print(f"[Auto-Resume 修复] 发现已落盘权重 {newest_ckpt.name}，正在同步至 {latest_pth} ...")
        shutil.copy2(newest_ckpt, latest_pth)

has_resume_ckpt = latest_pth.exists() and latest_pth.stat().st_size > 0

# 4. 自适应 GPU 档位配置 (Auto GPU Profile)：
#    - 主力 70%~80% Colab A100 (sm_80, 40GB/80GB)：启用高吞吐配置 (workers=4, cudnn_benchmark=True)
#    - 辅助 10%~15% Colab L4 (sm_89, 24GB)：自动适配 24GB 显存安全参数 (workers=2, max_split_size_mb=128 防显存碎片)
gpu_name = "NO_GPU"
try:
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"], text=True
    ).strip()
except Exception:
    pass

if "A100" in gpu_name.upper() or "8.0" in gpu_name:
    auto_gpu_profile = {
        "profile_name": "A100_HIGH_THROUGHPUT",
        "compute_arch": "sm_80",
        "compute_role": "主力算力 (70%~80%)：承担全部模型训练、消融实验及高负载任务",
        "workers": int(os.environ.get("SIMDT_NUM_WORKERS", "4")),
        "samples_per_gpu": int(os.environ.get("SIMDT_SAMPLES_PER_GPU", "1")),
        "cuda_alloc_conf": "max_split_size_mb:512",
    }
elif "L4" in gpu_name.upper() or "8.9" in gpu_name:
    auto_gpu_profile = {
        "profile_name": "L4_24GB_SAFE",
        "compute_arch": "sm_89",
        "compute_role": "辅助算力 (10%~15%)：自动适配 24GB 显存安全参数（推荐用于短程冒烟验证，完整多 Epoch 训练建议切换至 A100）",
        "workers": int(os.environ.get("SIMDT_NUM_WORKERS", "2")),
        "samples_per_gpu": 1,
        "cuda_alloc_conf": "max_split_size_mb:128",
    }
else:
    auto_gpu_profile = {
        "profile_name": "DEFAULT_SAFE_PROFILE",
        "compute_arch": "generic",
        "compute_role": "通用/待命档位（本地 10%~15% 仅作总控调度，不执行重型训练）",
        "workers": int(os.environ.get("SIMDT_NUM_WORKERS", "2")),
        "samples_per_gpu": 1,
        "cuda_alloc_conf": "max_split_size_mb:256",
    }

# 构建训练命令行参数
train_cmd = [
    PY1,
    "tracking_train_demo.py",
    "--config", TRAIN_CONFIG,
    "--work_dir", str(TRAIN_WORK_DIR),
    "--base_dataset", BASE_DATASET,
    "--seed", str(RANDOM_SEED),
    "--workers", str(auto_gpu_profile["workers"]),
]
if has_resume_ckpt:
    resume_mode = "AUTO_RESUME_FROM_LATEST_EPOCH"
    train_cmd.extend(["--resume_from", str(latest_pth)])
else:
    resume_mode = "FRESH_START_FROM_PRETRAINED"
    train_cmd.extend(["--load_from", PRETRAINED_CKPT])

train_cmd.extend([
    "--cfg-options",
    "optimizer_config.use_fp16=False",
    "checkpoint_config.interval=1",
    f"data.samples_per_gpu={auto_gpu_profile['samples_per_gpu']}",
    f"data.workers_per_gpu={auto_gpu_profile['workers']}",
    "cudnn_benchmark=True",
])

# 5. 生成并实时刷盘 run_metadata.json 到 Google Drive
run_metadata = {
    "cell": "CELL_7_AUTO_RESUME_TRAINING_ENGINE",
    "phase": PHASE_TAG,
    "exp_name": EXP_NAME,
    "simdt_stage_git_commit_sha": stage_meta.get("commit_sha", git_sha),
    "simdt_stage_git_short_sha": git_sha,
    "simdt_stage_git_ref": stage_meta.get("requested_ref", "main"),
    "official_antiuav410_commit": "8a8eb04d976e9386b7c9c3ada5c85e5086013d52",
    "work_dir": str(TRAIN_WORK_DIR),
    "config_file": TRAIN_CONFIG,
    "base_dataset": BASE_DATASET,
    "seed": RANDOM_SEED,
    "resume_mode": resume_mode,
    "latest_pth_detected": has_resume_ckpt,
    "checkpoint_entry": str(latest_pth) if has_resume_ckpt else PRETRAINED_CKPT,
    "resolved_command": " ".join(train_cmd),
    "gpu_info": gpu_name,
    "auto_gpu_profile": auto_gpu_profile,
    "execute_training_flag": EXECUTE_TRAINING,
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

meta_path = TRAIN_WORK_DIR / "run_metadata.json"
with meta_path.open("w", encoding="utf-8") as mf:
    json.dump(run_metadata, mf, indent=2, ensure_ascii=False)
    mf.flush()
    os.fsync(mf.fileno())

print(json.dumps(run_metadata, indent=2, ensure_ascii=False))

# 6. 若开启训练执行开关，启动流式日志双通道实时刷盘 (train.log + train.log.json 每 50 行强制 fsync)
if EXECUTE_TRAINING:
    train_log_path = TRAIN_WORK_DIR / "train.log"
    train_json_log_path = TRAIN_WORK_DIR / "train.log.json"
    train_env = os.environ.copy()
    train_env["ANTI_UAV410_ROOT"] = str(DRIVE_ROOT / "datasets/AntiUAV410") + "/"
    train_env["PYTHONUNBUFFERED"] = "1"
    train_env["PYTORCH_CUDA_ALLOC_CONF"] = auto_gpu_profile["cuda_alloc_conf"]

    print(f"\n[启动训练] 模式: {resume_mode} | GPU档位: {auto_gpu_profile['profile_name']} | 实时落盘目录: {TRAIN_WORK_DIR}")
    with train_log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n=== Training Session Started at {run_metadata['timestamp_utc']} ({resume_mode}) ===\n")
        log_f.flush()
        os.fsync(log_f.fileno())

        proc = subprocess.Popen(
            train_cmd,
            cwd=str(SIAMDT_DIR),
            env=train_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        line_count = 0
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            line_count += 1
            if line_count % 50 == 0:
                log_f.flush()
                os.fsync(log_f.fileno())
                # 同步 MMDetection 生成的最新 *.log.json 到固定名称 train.log.json
                json_logs = sorted(
                    [p for p in TRAIN_WORK_DIR.glob("*.log.json") if p.name != "train.log.json"],
                    key=lambda p: p.stat().st_mtime,
                )
                if json_logs:
                    shutil.copy2(json_logs[-1], train_json_log_path)

        ret = proc.wait()
        log_f.flush()
        os.fsync(log_f.fileno())
        json_logs = sorted(
            [p for p in TRAIN_WORK_DIR.glob("*.log.json") if p.name != "train.log.json"],
            key=lambda p: p.stat().st_mtime,
        )
        if json_logs:
            shutil.copy2(json_logs[-1], train_json_log_path)
        if ret != 0:
            raise RuntimeError(f"训练进程异常退出，返回码: {ret}")
else:
    print("\n[就绪待命] 训练目录与 run_metadata.json 已写入 Google Drive。如需启动/续训，请将 EXECUTE_TRAINING 设为 True。")
'''

ALL_COLAB_CELLS = [
    ("CELL 0｜后台启动远程接管桥（Drive 文件队列；隧道默认关闭）", CELL_0_REMOTE_BRIDGE),
    ("CELL 0b｜停止远程接管桥", CELL_0B_STOP_BRIDGE),
    ("CELL 1｜挂载 Google Drive 与建立持久化金库目录树", CELL_1_MOUNT_AND_LAYOUT),
    ("CELL 2｜双源 Git 同步（GitHub 阶段代码库 + 官方 Anti-UAV410 锁定仓库）", CELL_2_DUAL_GIT_SYNC_AND_PATCH),
    ("CELL 3｜数据集与多算法权重完整性核验（SHA-256 校验与软链接）", CELL_3_ASSETS_VERIFY_AND_PREPARE),
    ("CELL 4｜双轨运行环境秒级恢复（Tier-1: env_siamdt_cu118 + Tier-2: env_modern_pyt2）", CELL_4_BUILD_ENVIRONMENTS),
    ("CELL 5｜A100 算子、245 参数张量 strict=True 加载与 5 帧冒烟测试", CELL_5_MINIMAL_VERIFICATION),
    ("CELL 6｜Phase 2 统一逐帧遥测评测器（支持 discovery_set.json / val90_full.json 及断点续评）", CELL_6_UNIFIED_TELEMETRY_EVALUATOR),
    ("CELL 7｜Phase 6 具备自动断点续训（Auto-Resume）与 Google Drive 实时落盘的训练引擎", CELL_7_AUTO_RESUME_TRAINING_ENGINE),
]

CELLS_BY_KEY = {
    "0": CELL_0_REMOTE_BRIDGE, "0b": CELL_0B_STOP_BRIDGE, "1": CELL_1_MOUNT_AND_LAYOUT,
    "2": CELL_2_DUAL_GIT_SYNC_AND_PATCH, "3": CELL_3_ASSETS_VERIFY_AND_PREPARE, "4": CELL_4_BUILD_ENVIRONMENTS,
    "5": CELL_5_MINIMAL_VERIFICATION, "6": CELL_6_UNIFIED_TELEMETRY_EVALUATOR, "7": CELL_7_AUTO_RESUME_TRAINING_ENGINE,
}


def check_cells_plain_python() -> dict:
    """确认每个单元都是普通 Python：无行首 ! / % 魔法命令，且能被 compile() 编译。"""
    import re as _re
    report = {}
    for key, code in CELLS_BY_KEY.items():
        magics = [ln for ln in code.splitlines() if _re.match(r"^\s*[!%]", ln)]
        try:
            compile(code, f"<colab_cell_{key}>", "exec")
            compiled = True
        except SyntaxError as exc:
            compiled = f"SyntaxError: {exc}"
        report[key] = {"ipython_magic_lines": magics, "compiles_as_python": compiled}
    return report


def run_cells(keys: str) -> None:
    """按顺序在独立命名空间中执行指定单元（供远程接管桥 setup_env / stage_assets 动作调用）。"""
    for key in [k.strip() for k in keys.split(",") if k.strip()]:
        if key not in CELLS_BY_KEY:
            raise SystemExit(f"未知单元: {key}；可选 {list(CELLS_BY_KEY)}")
        if key in ("0", "0b"):
            raise SystemExit("Cell 0 / 0b 管理远程接管桥本身，不允许经由 --run-cell 执行")
        print(f"\n===== [colab_init_cells] 执行 CELL {key} =====", flush=True)
        exec(compile(CELLS_BY_KEY[key], f"<colab_cell_{key}>", "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    import argparse
    import json as _json

    _p = argparse.ArgumentParser(description="SimDT Colab 单元模板（可被普通 Python 进程执行）")
    _p.add_argument("--run-cell", type=str, default="", help="按顺序执行的单元编号，逗号分隔，如 4 或 1,2,3")
    _p.add_argument("--check-plain-python", action="store_true", help="检查所有单元无 IPython 魔法且可编译")
    _a = _p.parse_args()
    if _a.check_plain_python:
        _rep = check_cells_plain_python()
        _ok = all(not v["ipython_magic_lines"] and v["compiles_as_python"] is True for v in _rep.values())
        print(_json.dumps({"plain_python_check": "PASS" if _ok else "FAIL", "cells": _rep}, ensure_ascii=False, indent=2))
        raise SystemExit(0 if _ok else 1)
    if _a.run_cell:
        run_cells(_a.run_cell)
    else:
        print(f"Colab initialization template module verified successfully ({len(ALL_COLAB_CELLS)} cells ready).")
