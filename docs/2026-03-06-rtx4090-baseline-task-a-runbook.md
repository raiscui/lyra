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

### 2.3 成功判定

本手册执行成功,至少应满足:

- baseline smoke test 成功结束.
- 找到 `gaussians_orig/gaussians_0.ply`.
- `refine_robust_v2.py` 在真实 SR reference 下成功结束.
- 输出目录里存在:
  - `metrics_phase3s.json`
  - `metrics_stage3sr.json`
  - `gaussian_fidelity_histogram.json`
  - `sr_selection_maps/`
  - `gaussians/gaussians_stage3sr.ply`

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

- 一份和 native 视频内容完全对等的 SR reference
- 允许两种形态:
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

# 这里填你真实的 SR reference 路径.
# 它可以是 mp4,也可以是一个帧目录.
export SR_REFERENCE_PATH="/ABS/PATH/TO/SR_REFERENCE.mp4"

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

这一阶段不是为了做最终最优调参.

这一阶段的目标更单纯:

- 让真实 SR reference 进入 `reference_images`
- 让 `Stage 2A -> Phase 3S -> Stage 3SR` 走完
- 拿到 selective SR 的真实诊断证据

### 7.2 推荐第一版命令

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

### 7.3 为什么先这样跑

- `--stop-after stage2a`
  - 先只看 `Stage 3SR`
  - 不让 `Stage 2B` 把问题搅混
- `--frame-indices ...`
  - 先跑稀疏子集
  - 更适合作为 smoke test
- `--lambda-patch-rgb 0.25`
  - 当前这是更稳的困难轨迹起点
- `--enable-pruning`
  - 这条主线已经验证过值得保留

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

至少看这些文件:

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
