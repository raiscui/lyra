# RTX 4090 Baseline 与 Continuation Task A 操作手册

## 1. 文档定位

这份手册是给"编程智能体"和"接手执行的人"用的.

它只解决两件事:

1. 在 `RTX 4090 24G` 上,先把 `sample.py -> baseline .ply` 跑通.
2. baseline 一旦成功,立刻继续 `Continuation Task A`,验证真实 external SR reference.

这不是一份泛泛的说明文档.
它是一份带明确输入、命令、通过条件、失败分流和交接格式的执行手册.

## 2. 执行契约

### 2.1 目标

- 在 `RTX 4090 24G` 上完成一次 baseline smoke test.
- 成功导出 baseline `.ply`.
- 在同一条 `joint_refinement_camera_gaussians_v2` 主线上继续执行 `Continuation Task A`.
- 输出一组可用于判断 selective SR 是否有效的真实诊断产物.

### 2.2 非目标

- 不改 `sample.py` 主链.
- 不新增第二套后处理入口.
- 不在本轮处理 `sample.py` 的旧 GPU fallback.
- 不把 `Continuation Task A` 和 `Continuation Task B` 混成一个任务.
- 不跳过 SR reference 生成前的逐帧人工核对.

### 2.3 成功判定

本手册执行成功,至少应满足:

- baseline smoke test 成功结束.
- 找到 `gaussians_orig/gaussians_0.ply`.
- `FlashVSR-Pro` 成功导出一份真实 SR reference.
- `FlashVSR-Pro` 输出目录里存在:
  - `native_frames/`
  - `sr_frames/`
  - `compare_frames/`
- `refine_robust_v2.py` 在真实 SR reference 下成功结束.
- 输出目录里存在:
  - `metrics_phase3s.json`
  - `metrics_stage3sr.json`
  - `gaussian_fidelity_histogram.json`
  - `sr_selection_maps/`
  - `gaussians/gaussians_stage3sr.ply`

### 2.4 阶段命名与成功信号约定

这份手册里最容易让人误会的一点是:

- 命令里会写 `--stop-after stage2a`
- 但成功后的 `diagnostics.json` 里,常见却是:
  - `phase_reached = stage3sr`

这不是矛盾。

当前约定是:

- `stage2a`
  - 对外主线阶段名
- `Stage 3A -> Phase 3S -> Stage 3SR`
  - `stage2a` 内部的增强子阶段

因此在本手册这条 `Task A` 路线上:

- `--stop-after stage2a`
  - 表示先不要继续 `Stage 2B`
  - 但仍然允许 `stage2a` 内部把 enhanced 链路跑完
- 如果你启用了:
  - `--stage2a-mode enhanced`
  - patch supervision
  - SR reference
- 那么成功信号更应理解为:
  - `phase_reached = stage3sr`
  - 而不是强行要求 `phase_reached = stage2a`

这轮最该检查的产物也因此是:

- `metrics_stage2a.json`
- `metrics_phase3s.json`
- `metrics_stage3sr.json`
- `gaussians/gaussians_stage3sr.ply`
- `diagnostics.json`

## 3. 输入约定

### 3.1 必需输入

- 仓库根目录: 当前仓库
- GPU: `RTX 4090 24G`
- baseline 配置:
  - `configs/demo/lyra_static.yaml`
- baseline 数据集:
  - `lyra_static_demo_generated`
- baseline 真实资产:
  - `assets/demo/static/diffusion_output_generated`
- checkpoints:
  - `checkpoints/`

### 3.2 Task A 额外输入

- `FlashVSR-Pro` 仓库本地路径
- `FlashVSR-Pro` 对应模型权重
- 一份和 native 视频内容完全对等的 SR reference
  - 推荐直接由 `assets/demo/static/diffusion_output_generated/*/rgb/*.mp4` 生成
  - 最终仍允许两种形态:
    - 本地视频文件
    - 帧目录

如果 SR reference 是"纯整数倍放大",默认不需要额外 `reference_intrinsics`.

如果不是纯整数倍放大,或者做过 crop / pad / reprojection,则必须同时给:

- `--reference-intrinsics-path`

## 4. 变量模板

执行前先在仓库根目录设置这些变量.

```bash
export CUDA_VISIBLE_DEVICES=0
export BASE_CONFIG="configs/demo/lyra_static.yaml"
export DATASET_NAME="lyra_static_demo_generated"
export VIEW_ID="3"
export SCENE_STEM="00172"
export SCENE_ROOT="assets/demo/static/diffusion_output_generated/${VIEW_ID}"
export POSE_PATH="${SCENE_ROOT}/pose/${SCENE_STEM}.npz"
export INTRINSICS_PATH="${SCENE_ROOT}/intrinsics/${SCENE_STEM}.npz"
export RGB_PATH="${SCENE_ROOT}/rgb/${SCENE_STEM}.mp4"

# `FlashVSR-Pro` 仓库本地路径.
export FLASHVSR_REPO="/ABS/PATH/TO/FlashVSR-Pro"

# `FlashVSR-Pro` 本机环境里的 python.
# 在当前 48G 主机上的实际值是:
# /usr/local/miniconda3/envs/flashvsr/bin/python3
export FLASHVSR_LOCAL_PYTHON="/ABS/PATH/TO/flashvsr/bin/python3"

# `FlashVSR-Pro` 产物根目录.
export FLASHVSR_OUTPUT_ROOT="outputs/flashvsr_reference"

# 如果按本手册默认流程跑,后面会自动得到这个路径.
export SR_REFERENCE_PATH="$FLASHVSR_OUTPUT_ROOT/full_scale2x/${VIEW_ID}/rgb/${SCENE_STEM}.mp4"

# 如果 SR 不是纯整数倍放大,就填真实内参; 否则留空.
export SR_INTRINSICS_PATH=""
```

## 5. 阶段 A: 4090 预检

### 5.1 确认选中的卡真的是 4090

```bash
nvidia-smi -L
```

### 5.2 让 Python 再确认一次

```bash
pixi run python3 - <<'PY'
import torch
print("cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_name =", torch.cuda.get_device_name(0))
    print("capability =", torch.cuda.get_device_capability(0))
    print("bf16_supported =", torch.cuda.is_bf16_supported())
PY
```

### 5.3 通过条件

- `device_name` 包含 `RTX 4090`
- `capability` 的 major version `>= 8`

### 5.4 如果失败

- 如果不是 4090:
  - 修正 `CUDA_VISIBLE_DEVICES`
- 如果 `torch.cuda.is_available() == False`:
  - 先修 CUDA / 驱动 / 容器映射

## 6. 阶段 B: baseline smoke test

### 6.1 输入存在性检查

```bash
test -f "$POSE_PATH" && echo "pose ok"
test -f "$INTRINSICS_PATH" && echo "intrinsics ok"
test -f "$RGB_PATH" && echo "rgb ok"
find checkpoints -maxdepth 2 -type f | sed -n '1,20p'
```

### 6.2 执行 baseline smoke test

```bash
pixi run bash -lc "CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) accelerate launch sample.py \
  --config \"$BASE_CONFIG\" \
  dataset_name=\"$DATASET_NAME\" \
  save_gaussians_orig=true \
  save_gaussians=true"
```

### 6.3 推荐并行观察

另开一个终端观察显存和是否真的跑在 4090 上:

```bash
watch -n 1 nvidia-smi
```

### 6.4 baseline 产物定位

```bash
find outputs -path '*gaussians_orig/gaussians_0.ply' | sort
```

如果你只跑了推荐的 demo 静态链路,通常会看到类似路径:

```bash
outputs/demo/lyra_static/static_view_indices_fixed_3/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply
```

执行成功后,立刻固化变量:

```bash
export BASELINE_PLY="$(find outputs -path '*gaussians_orig/gaussians_0.ply' | sort | tail -n 1)"
echo "$BASELINE_PLY"
```

### 6.5 baseline 通过条件

- 命令退出码为 `0`
- 没有出现以下错误:
  - `.bf16 requires .target sm_80 or higher`
  - `no kernel image is available for execution on the device`
- `BASELINE_PLY` 指向真实存在的 `.ply`

## 7. 阶段 C: Continuation Task A

### 7.1 任务目标

这一阶段现在拆成两段:

1. 先用 `FlashVSR-Pro` 生成真实 SR reference.
2. 先看逐帧图,确认问题是不是在 SR 阶段就已经出现.
3. 只有 SR reference 看起来基本靠谱,才继续跑 `Stage 2A -> Phase 3S -> Stage 3SR`.

也就是说:

- 现在不推荐一上来就直接跑 `refine_robust_v2.py`.
- 先把 SR reference 做出来.
- 先把 `native / sr / compare` 三组图看一遍.

### 7.2 先准备 `FlashVSR-Pro` 环境

如果本机还没有准备好 `FlashVSR-Pro`,先做这一步:

```bash
git clone https://github.com/LujiaJin/FlashVSR-Pro "$FLASHVSR_REPO"
git -C "$FLASHVSR_REPO" lfs install
git -C "$FLASHVSR_REPO" lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1 "$FLASHVSR_REPO/models/FlashVSR-v1.1"
docker build -t flashvsr-pro:latest "$FLASHVSR_REPO"
```

如果你已经准备过这套环境,可以直接跳到下一步.

#### 2026-03-07 已验证的主机结论

- 当前这台 `48G` 主机上:
  - Docker 包安装是成功的
  - daemon 也能起来
  - 但真实 `docker run` 会在 layer 注册阶段报:
    - `failed to register layer: unshare: operation not permitted`
- 这说明当前环境更像 nested 容器 / 外层 seccomp 约束.
- 因此在这台机器上跑 `FlashVSR-Pro`, 当前推荐路线不是 docker runner, 而是:
  - 本机 conda / venv 环境
  - `scripts/run_flashvsr_reference.py --runner local --local-python "$FLASHVSR_LOCAL_PYTHON"`

### 7.3 生成 SR reference + 逐帧排查图

在 `48G` 显存主机上,默认先用 `full` 模式.
先不做 tile.
只有真的遇到 OOM,才自动回退到 `tile_size=512`、`overlap=128`.

```bash
PYTHONPATH="$(pwd)" pixi run python3 scripts/run_flashvsr_reference.py \
  --input-root assets/demo/static/diffusion_output_generated \
  --output-root "$FLASHVSR_OUTPUT_ROOT" \
  --flashvsr-repo "$FLASHVSR_REPO" \
  --runner local \
  --local-python "$FLASHVSR_LOCAL_PYTHON" \
  --view-ids "$VIEW_ID" \
  --scene-stem "$SCENE_STEM" \
  --mode full \
  --debug-every 8
```

这条脚本会同时落盘:

- `rgb/${SCENE_STEM}.mp4`
- `debug/${SCENE_STEM}/native_frames/`
- `debug/${SCENE_STEM}/sr_frames/`
- `debug/${SCENE_STEM}/compare_frames/`
- `manifests/${SCENE_STEM}.json`

#### 2026-03-07 已验证的真实结果

- 输入:
  - `assets/demo/static/diffusion_output_generated/3/rgb/00172.mp4`
- 输出:
  - `outputs/flashvsr_reference/full_scale2x/3/rgb/00172.mp4`
- 实际结果:
  - `2560x1408`
  - `121 frames`
  - `duration=5.039567`
  - `full` 模式成功
  - 没有触发 tiled fallback
- 逐帧目录也已完整生成:
  - `native_frames/`
  - `sr_frames/`
  - `compare_frames/`

### 7.4 先做逐帧人工核对

```bash
export FLASHVSR_DEBUG_DIR="$FLASHVSR_OUTPUT_ROOT/full_scale2x/${VIEW_ID}/debug/${SCENE_STEM}"
find "$FLASHVSR_DEBUG_DIR" -maxdepth 2 -type f | sort | sed -n '1,200p'
```

重点先看:

- `native_frames/`
- `sr_frames/`
- `compare_frames/`

这里的判断目标不是“是否已经最终最优”.
而是先确认:

- SR 是否引入了明显错位
- SR 是否已经开始出现奇怪纹理
- 问题是否在进入 refinement 之前就已经能看出来

### 7.5 再跑 selective SR smoke

只有在 `FlashVSR-Pro` 输出看起来没有明显时序错位后,再继续这一步.

先不要把 `Stage 2B` 混进来.
先把 selective SR 本身跑清楚.

```bash
PYTHONPATH="$(pwd)" pixi run python3 scripts/refine_robust_v2.py \
  --config "$BASE_CONFIG" \
  --dataset-name "$DATASET_NAME" \
  --gaussians "$BASELINE_PLY" \
  --view-id "$VIEW_ID" \
  --reference-mode super_resolved \
  --reference-path "$SR_REFERENCE_PATH" \
  --frame-indices 0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120 \
  --patch-size 256 \
  --lambda-patch-rgb 0.25 \
  --lambda-sampling-smooth 0.0005 \
  --enable-pruning \
  --stop-after stage2a \
  --outdir "outputs/refine_v2/view${VIEW_ID}_task_a_sr_smoke"
```

如果 SR reference 不是纯整数倍放大,就在上面命令里追加:

```bash
  --reference-intrinsics-path "$SR_INTRINSICS_PATH"
```

### 7.6 为什么先这样跑

- `--stop-after stage2a`
  - 先把观察范围限制在 `stage2a` 这条对外主线内部
  - 在当前 enhanced 口径下, 重点其实是看 `Stage 3A -> Phase 3S -> Stage 3SR`
  - 不让 `Stage 2B` 把问题搅混
- `--frame-indices ...`
  - 先跑稀疏子集
  - 更适合作为 smoke test
- `--lambda-patch-rgb 0.25`
  - 当前这是更稳的困难轨迹起点
- `--enable-pruning`
  - 这条主线已经验证过值得保留

#### 2026-03-07 已验证通过的 direct-input 主基线 v1

在确认 `FlashVSR-Pro` 视频本身正常后, 当前更推荐先用 direct inputs 固定一版 `Stage 3SR` 基线:

```bash
PYTHONPATH="$(pwd)" pixi run python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --pose-path assets/demo/static/diffusion_output_generated/3/pose/00172.npz \
  --intrinsics-path assets/demo/static/diffusion_output_generated/3/intrinsics/00172.npz \
  --rgb-path assets/demo/static/diffusion_output_generated/3/rgb/00172.mp4 \
  --view-id 3 \
  --frame-indices 0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120 \
  --reference-mode super_resolved \
  --sr-scale 2.0 \
  --reference-path outputs/flashvsr_reference/full_scale2x/3/rgb/00172.mp4 \
  --start-stage stage2a \
  --stage2a-mode enhanced \
  --patch-size 256 \
  --lambda-patch-rgb 0.25 \
  --lambda-sampling-smooth 0.0005 \
  --enable-pruning \
  --iters-stage2a 60 \
  --stop-after stage2a \
  --outdir outputs/refine_v2/view3_stage3sr_real_sr_baseline_v1
```

这版命令已经验证成功, 关键结果为:

- `phase_reached = stage3sr`
  - 这是符合当前命名约定的正常结果
  - 因为 `stage2a` 的 enhanced 内部链路已经一路跑到了 `Stage 3SR`
- `stopped_reason = metrics_plateau`
- `PSNR: 15.9834 -> 18.4991`
- `residual_mean: 0.09453 -> 0.05571`
- `sharpness: 0.002266 -> 0.004078`

2026-03-07 后续复盘又发现一处 direct-input 相机契约问题:

- `--pose-path` 提供的是 raw pose / `c2w` 风格 `pose.npz`
- direct-input 路径必须在内部转换成 provider 一致的:
  - `cam_view = inverse(c2w).T`
- 修复后, 同一组主线参数重新验证输出为:
  - `outputs/refine_v2/view3_stage3sr_real_sr_baseline_v1_fixed_cam_view`
- 修复后关键结果:
  - `phase_reached = stage3sr`
  - `PSNR: 18.9714 -> 28.4424`
  - `residual_mean: 0.06698 -> 0.01910`
  - `sharpness: 0.002229 -> 0.005211`
- `scale_tail_ratio: 0.01946 -> 0.00747`

因此当前建议是:

- 先把这组参数作为 `Stage 3SR` 的 SR 变体验证命令
- 先不要急着继续 `Stage 2B`
- 先检查:
  - `metrics_phase3s.json`
  - `metrics_stage3sr.json`
  - `videos/final_render.mp4`
  - `sr_selection_maps/`

如果你的目标是比较“native reference”和“super-resolved reference”谁更好, 当前正式 baseline 应改按下面理解:

- 正式 baseline:
  - `outputs/refine_v2/view3_stage3sr_native_reference_v1_fixed_cam_view`
- SR 变体:
  - `outputs/refine_v2/view3_stage3sr_real_sr_baseline_v1_fixed_cam_view`
- 说明:
  - 后者目录名中的 `baseline` 是历史命名残留.
  - 语义上不再把它当“正式 baseline”.

在这个口径下, 后续已经完成一轮真实 `Stage 2B` 续跑:

- 起点:
  - `outputs/refine_v2/view3_stage3sr_native_reference_v1_fixed_cam_view`
- 续跑结果:
  - `outputs/refine_v2/view3_stage2b_from_native_reference_v1_fixed_cam_view_20260307_1248`
- 结束状态:
  - `phase_reached = stage2b`
  - `warm_start_stage2b = true`
- 相对 native baseline 的最终增益:
  - `PSNR: 28.5179 -> 30.7697`
  - `residual_mean: 0.018922 -> 0.014952`
  - `sharpness: 0.005127 -> 0.006509`
  - `scale_tail_ratio: 0.006442 -> 0.004440`
  - `opacity_lowconf_ratio: 0.812334 -> 0.800099`
- 结论:
  - 当前 48G 主机上, native baseline 往下继续开 `Stage 2B` 是值得的
  - 而且收益明显大于当前 “native vs SR reference” 的微小差异

同一轮还补跑了:

- `outputs/refine_v2/view3_stage2b_from_sr_reference_v1_fixed_cam_view_20260307_1301`

结果说明:

- `SR -> Stage 2B` 也能继续提升
- 但当前最终仍是:
  - native `Stage 2B` 在 `PSNR / residual_mean` 上更优
  - SR `Stage 2B` 在 `sharpness` 上略优
- 因此当前 runbook 的正式推荐主线,应固定为:
  - `native baseline -> Stage 2B`

## 8. 阶段 D: Task A 结果检查

### 8.1 关键产物

```bash
export TASK_A_OUT="outputs/refine_v2/view${VIEW_ID}_task_a_sr_smoke"

find "$TASK_A_OUT" -maxdepth 2 -type f | sort | sed -n '1,200p'
```

重点确认:

```bash
test -f "$TASK_A_OUT/metrics_phase3s.json" && echo "metrics_phase3s ok"
test -f "$TASK_A_OUT/metrics_stage3sr.json" && echo "metrics_stage3sr ok"
test -f "$TASK_A_OUT/gaussian_fidelity_histogram.json" && echo "fidelity histogram ok"
test -f "$TASK_A_OUT/gaussians/gaussians_stage3sr.ply" && echo "stage3sr ply ok"
test -d "$TASK_A_OUT/sr_selection_maps" && echo "sr_selection_maps ok"
```

### 8.2 快速读取 metrics

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

task_out = Path(os.environ["TASK_A_OUT"])
for name in ["metrics_stage2a.json", "metrics_phase3s.json", "metrics_stage3sr.json", "diagnostics.json"]:
    path = task_out / name
    if not path.exists():
        print(f"{name}: missing")
        continue
    print(f"===== {name} =====")
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        for key in ["psnr", "residual_mean", "sharpness", "sr_selection_mean", "loss_patch_rgb", "loss_sampling_smooth"]:
            if key in data:
                print(f"{key} = {data[key]}")
PY
```

### 8.3 肉眼检查建议

先看 `FlashVSR-Pro` 的逐帧图:

- `"$FLASHVSR_DEBUG_DIR/native_frames/*.png"`
- `"$FLASHVSR_DEBUG_DIR/sr_frames/*.png"`
- `"$FLASHVSR_DEBUG_DIR/compare_frames/*.png"`

再看 refinement 产物:

- `videos/baseline_render.mp4`
- `videos/final_render.mp4`
- `sr_selection_maps/*.png`

如果你的判断目标是 selective SR 是否真的有帮助,重点看:

- 局部高频区域有没有更清楚
- 有没有引入新的双轮廓
- `sr_selection_maps` 是否不是全白

## 9. 失败分流

### 9.1 baseline 阶段失败

#### 症状: 不是 4090

动作:

- 修 `CUDA_VISIBLE_DEVICES`
- 重新跑阶段 A

#### 症状: `sample.py` 报 OOM

动作:

- 先检查是不是别的进程占了显存
- 先清空占用后重跑
- 先不要直接改模型或代码

#### 症状: 仍然出现 Triton / Mamba 错误

动作:

- 这时问题已不再是旧 GPU 架构本身
- 需要转去核对:
  - 驱动版本
  - CUDA runtime
  - Triton 版本
  - `pixi` 环境是否一致

### 9.2 Task A 阶段失败

#### 症状: reference 对齐报错

动作:

- 检查 SR reference 是否保持:
  - 同时序
  - 同 crop
  - 同 aspect ratio
- 如果不是纯放大:
  - 补 `--reference-intrinsics-path`

#### 症状: `FlashVSR-Pro` 逐帧图里已经能看出问题

动作:

- 先不要继续跑 `refine_robust_v2.py`
- 先只看这三组图:
  - `native_frames`
  - `sr_frames`
  - `compare_frames`
- 如果问题只在 `sr_frames` 里出现:
  - 先把根因归到 SR 生成阶段
  - 再考虑是否需要调整:
    - `--mode`
    - tiled fallback
    - 输入视频选择
- 如果 `native_frames` 本身就已经怪:
  - 问题更可能在 diffusion 输出,不是 refinement

#### 症状: `patch_size ... must be divisible by sr_scale`

动作:

- 改用能被整数倍率整除的 `patch_size`
- 当前推荐起点:
  - `sr_scale=2` 时用 `256`

#### 症状: 成功跑完,但 `sr_selection_maps` 接近全白

动作:

- 先确认 `render_meta` 是否真实返回
- 再检查 `gaussian_fidelity_histogram.json`
- 这类情况通常说明:
  - fidelity 区分度不够
  - 或当前轨迹本身没有明显 selective SR 收益

## 10. 交接输出格式

执行人或智能体结束后,应至少交付下面这个 JSON 摘要.

```json
{
  "gpu_name": "RTX 4090",
  "gpu_capability": [8, 9],
  "baseline_smoke": {
    "status": "pass",
    "baseline_ply": "outputs/.../gaussians_orig/gaussians_0.ply"
  },
  "flashvsr_reference": {
    "status": "pass",
    "reference_path": "outputs/flashvsr_reference/full_scale2x/3/rgb/00172.mp4",
    "debug_dir": "outputs/flashvsr_reference/full_scale2x/3/debug/00172",
    "artifacts_present": [
      "native_frames/",
      "sr_frames/",
      "compare_frames/",
      "manifests/00172.json"
    ]
  },
  "task_a": {
    "status": "pass",
    "outdir": "outputs/refine_v2/view3_task_a_sr_smoke",
    "artifacts_present": [
      "metrics_phase3s.json",
      "metrics_stage3sr.json",
      "gaussian_fidelity_histogram.json",
      "gaussians/gaussians_stage3sr.ply",
      "sr_selection_maps/"
    ],
    "metrics": {
      "psnr": null,
      "residual_mean": null,
      "sharpness": null,
      "sr_selection_mean": null
    },
    "notes": "简短记录最重要的观察"
  }
}
```

## 11. 当前推荐执行顺序

1. 先做阶段 A 和阶段 B.
2. baseline `.ply` 一旦成功,立刻记录 `BASELINE_PLY`.
3. 再做阶段 C 的 `Continuation Task A`.
4. 先只停在 `stage2a`.
5. selective SR 结果看清后,再决定要不要继续 `Stage 2B`.
