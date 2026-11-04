# Phase 2 跟踪器运行环境矩阵（ENV_MATRIX）

> 生成时间：2026-11-04。版本依据均来自各仓库自带的 README / install.sh / requirements.txt / setup.py（路径见“依据”列）。
> **以下环境均未在 Colab GPU 上实际创建和验证**，第一次建 env 时请按本表执行，并把实际装上的版本回填到“实测”列。

## 1. 分组总览

| 分组（注册表 `env_tier`） | Colab 解释器路径（`python_bin_colab`） | Python | PyTorch / CUDA | 关键依赖 | 成员 |
|---|---|---|---|---|---|
| **A_siamdt_cu118** | `/content/envs/siamdt-cu118/bin/python` | 沿用既有环境 | 沿用既有 cu118 环境 | SiamDT 自带 mmdet 风格 libs | SiamDT_Baseline |
| **B1_pytracking_pt20_cu118** | `/content/envs/pytracking-pt20-cu118/bin/python` | 3.9 | torch 2.0.0 / torchvision 0.15.0 / cu118 | numpy 1.23.5、matplotlib 3.6、timm、easydict、yacs、opencv-python、lmdb、jpeg4py | FocusTrack、ReFocus-TIR |
| **B2_sutrack_pt111_cu113** | `/content/envs/sutrack-pt111-cu113/bin/python` | 3.8 | torch 1.11.0+cu113 / torchvision 0.12.0+cu113 | timm==0.5.4、OpenAI CLIP（git）、pytorch-pretrained-bert==0.6.2、yacs、easydict、opencv-python、scikit-image、pycocotools、jpeg4py | SUTrack、UETrack、UTPTrack-S |
| **C_sam2_pt25_cu124** | `/content/envs/sam2-pt25-cu124/bin/python` | ≥3.10 | torch ≥2.3.1（DAM4SAM requirements 的 nvidia-*-cu12 12.4 系 → 建议 2.5.1+cu124） | numpy==1.26.4、hydra-core==1.3.2、omegaconf==2.3.0、iopath、opencv-python、pillow、tqdm、**vot-toolkit==0.7.1 + vot-trax==4.0.2**（DAM4SAM 必需） | DAM4SAM、SAMOSA、SAMURAI |
| **D_trackit_memlo_pt2** | `/content/envs/trackit-memlo-pt2/bin/python` | ≥3.10（README 未锁定） | torch ≥2.0（README 原文） | timm、safetensors、fvcore、tabulate、exifread、pyzmq、psutil、pyturbojpeg（+ `apt install libturbojpeg`）、wandb（运行时 `--disable_wandb`）、pyyaml、matplotlib、tqdm、pillow | MemLoTrack、MemLoTrack-Scratch |

## 2. 逐个跟踪器明细

| 跟踪器 | 分组 | 官方给的版本依据 | 冲突点 / 备注 |
|---|---|---|---|
| SiamDT_Baseline | A | 既有 `siamdt-cu118` 环境（第 1 波只跑它） | 老 mmcv/mmdet 栈，不能和任何 torch≥2 的仓库混用 |
| FocusTrack | B1 | `FocusTrack/README.md`：Python 3.9、CUDA 11.8、`pytorch==2.0.0 torchvision==0.15.0`、`numpy==1.23.5` | 与 B2 的 torch 1.11 冲突 |
| ReFocus-TIR | B1 | `ReFocus_TIR/README.md` 没给版本（只写 OSTrack 系 `create_default_local_file.py`） | OSTrack 系代码在 torch 2.0 上通常可跑，**但无官方确认**；若出错可改放 B2（torch 1.11） |
| SUTrack | B2 | `SUTrack/install.sh`：`torch==1.11.0+cu113`、`timm==0.5.4`、CLIP（git） | `vot-toolkit` 为 git 版，仅 VOT 评测需要，本调度器不需要 |
| UETrack | B2 | `UETrack/install.sh`：同上（`vot-toolkit==0.5.3`，本调度器不需要） | 推理时会构建 CLIP ViT-L/14 文本编码器（`MULTI_MODAL_LANGUAGE: True`），首次运行需联网下载 CLIP 权重 |
| UTPTrack-S | B2 | `UTPTrack/UTPTrack-S/README.md`：`conda create -n utptrack-s python=3.8` + `bash install.sh`（SUTrack 系） | 同上需要 CLIP |
| DAM4SAM | C | `DAM4SAM/requirements.txt`：numpy 1.26.4、hydra-core 1.3.2、vot-toolkit 0.7.1、nvidia-*-cu12 12.4；`setup.py`：torch≥2.1、python≥3.10 | **必须装 vot-toolkit==0.7.1**（`dam4sam_tracker.py` 用 `vot.region.Mask` / `calculate_overlaps` 做 DRM 判定）。适配器启动时若 `import vot.region` 失败会直接报错退出 |
| SAMOSA | C | `SAMOSA/README.md`：python≥3.10、torch≥2.3.1、`numpy==1.26.4`、`matplotlib==3.7`、loguru、shapely、scipy、pandas、lmdb、jpeg4py | 自带改版 `sam2` 包 |
| SAMURAI | C | `SAMURAI/sam2/setup.py`：python≥3.10、torch≥2.3.1、numpy≥1.24.4 | 自带改版 `sam2` 包 |
| MemLoTrack(-Scratch) | D | `MemLoTrack/README.md`：“PyTorch version should be >= 2.0” + `requirements.txt`（未锁版本）+ `apt install libturbojpeg` | timm 需较新版本（DINOv2），与 B2 的 timm 0.5.4 冲突 |

## 3. 分组理由（为什么不能全塞进一个 env）

1. **torch 主版本冲突**：A（老 mmdet）/ B2（torch 1.11+cu113，timm 0.5.4）/ B1（torch 2.0+cu118）/ C（torch≥2.3.1）四者互斥。
2. **timm 冲突**：B2 锁 `timm==0.5.4`；D（DINOv2+LoRA）需要新版 timm。
3. **`sam2` 同名包冲突（C 组内部）**：DAM4SAM、SAMOSA、SAMURAI 三个仓库各带一份**改过的** `sam2`（DAM4SAM 加了 `add_to_drm`/`return_all_masks`，SAMOSA 加了 `motion_predictor`/`mp_ckpt_path`，SAMURAI 加了卡尔曼滤波）。
   - **C 组 env 里禁止 `pip install` 任何 sam2**（包括 `pip install -e .`）。
   - 调度器每个 Worker 是独立进程，适配器会把 `<该仓库>/sam2`（或 DAM4SAM 仓库根）插到 `sys.path` 最前面，各进程只看到自己仓库的那份 `sam2`。
   - SAM2 的 CUDA 扩展 `_C` 可以不编译（官方 INSTALL.md：缺失时只跳过掩码小孔填充后处理）。
4. **可选合并**：D 组在技术上可以并入 C 组（都是 torch≥2），但 MemLoTrack 未锁版本，为了不影响 C 组 numpy 1.26.4 / vot-toolkit 0.7.1 的锁定，建议单独建 env。

## 4. 建 env 参考命令（未在 Colab 上验证）

Colab 自带 Python 3.12，3.8/3.9 需要用 micromamba/conda 或 uv 建独立解释器：

```bash
# B1：FocusTrack / ReFocus-TIR
micromamba create -y -p /content/envs/pytracking-pt20-cu118 python=3.9
/content/envs/pytracking-pt20-cu118/bin/pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
/content/envs/pytracking-pt20-cu118/bin/pip install numpy==1.23.5 matplotlib==3.6 timm easydict yacs opencv-python lmdb jpeg4py pandas tqdm

# B2：SUTrack / UETrack / UTPTrack-S
micromamba create -y -p /content/envs/sutrack-pt111-cu113 python=3.8
/content/envs/sutrack-pt111-cu113/bin/pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
/content/envs/sutrack-pt111-cu113/bin/pip install timm==0.5.4 yacs easydict opencv-python pandas pycocotools jpeg4py lmdb scipy scikit-image pytorch-pretrained-bert==0.6.2 git+https://github.com/openai/CLIP.git

# C：DAM4SAM / SAMOSA / SAMURAI（不要 pip install 任何 sam2！）
micromamba create -y -p /content/envs/sam2-pt25-cu124 python=3.10
/content/envs/sam2-pt25-cu124/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
/content/envs/sam2-pt25-cu124/bin/pip install numpy==1.26.4 hydra-core==1.3.2 omegaconf==2.3.0 iopath opencv-python pillow tqdm scipy pandas loguru shapely matplotlib==3.7 vot-toolkit==0.7.1 vot-trax==4.0.2

# D：MemLoTrack
apt-get install -y libturbojpeg
micromamba create -y -p /content/envs/trackit-memlo-pt2 python=3.10
/content/envs/trackit-memlo-pt2/bin/pip install torch torchvision
/content/envs/trackit-memlo-pt2/bin/pip install -r /content/runtime/repos/core_pool/MemLoTrack/requirements.txt
```

## 5. 运行时注意（调度器已自动处理的部分）

- pytracking 系（FocusTrack / SUTrack / UETrack / UTPTrack-S / ReFocus-TIR）：Worker 启动时自动生成 `lib/test/evaluation/local.py`（`prj_dir=REPO_DIR`，`network_path=WEIGHT_DIR`），并把 `params.checkpoint` 严格设为 `WEIGHT_DIR/PRIMARY_WEIGHT`。
- SUTrack / UETrack / UTPTrack-S 的编码器在推理构建时会以 `pretrained=is_main_process()=True` 调用 `torch.load(MODEL.ENCODER.PRETRAIN_TYPE)`：UETrack 配置里该项为空、UTPTrack-S 配置里是作者集群的绝对路径，直接跑会崩。调度器把它改指向 `weights/.../pretrained/` 下的官方同名文件（SUTrack/UTPTrack-S 之后以 `strict=True` 加载完整权重，预训练值会被全部覆盖；UETrack 为 `strict=False`，与官方一样使用官方预训练做初始化）。
- 专用 env 的 python 不存在时，调度器默认直接报错（不再静默用系统 Python），除非显式加 `--allow-sys-python-fallback`。
