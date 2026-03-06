# `joint_refinement_camera_gaussians_v2` 使用说明

## 1. 这份文档解决什么问题

这份文档只回答一件事:

- `joint_refinement_camera_gaussians_v2` 已经实现的增强部分,到底该怎么用。

如果你前面一直是按 `README.md` 的 `Example 1` 在跑,最容易混淆的一点是:

- `README` 里的两步命令,只覆盖了 `SDG 生成 + baseline 3DGS 重建`
- 这两步**不会自动触发** `joint_refinement_camera_gaussians_v2`
- 要用到增强部分,还需要第 3 步单独运行 `scripts/refine_robust_v2.py`

一句话说完:

- `README 两步 = baseline`
- `README 两步 + refine_robust_v2 第三步 = baseline 之上的 v2 增强`

---

## 2. 先看整体流程

```text
单张输入图像
  -> SDG 生成多视角视频 latent / diffusion 输出
  -> sample.py 重建初始 3DGS
  -> 导出 gaussians_orig/gaussians_0.ply
  -> refine_robust_v2.py 做联合细化
  -> 输出更干净的 gaussians_stage2a / gaussians_stage3sr / gaussians_stage2b / gaussians_refined
```

这里面第 3 步 `refine_robust_v2.py` 才是 `joint_refinement_camera_gaussians_v2` 的实际入口。

---

## 3. 当前代码里已经可用的增强能力

当前已经做完并可直接使用的能力包括:

- `Phase 0` baseline 诊断
- `Phase 1` 视图/帧/参考图对齐
- `Stage 2A` 颜色与透明度主清理
- `Stage 2A` 内部增强子阶段
  - `Stage 3A` native cleanup
  - `Phase 3S` Gaussian fidelity / SR selection
  - `Stage 3SR` selective SR patch supervision
- `Stage 2B` limited geometry refinement
- `Phase 3` tiny pose-only diagnostic
- `Phase 4` joint fallback
- pruning
- patch supervision
- selective SR
- `L_sampling_smooth`
- `--start-stage stage2b`
- external reference contract
  - `--reference-path`
  - `--reference-intrinsics-path`
- direct file inputs v1
  - `--pose-path`
  - `--intrinsics-path`
  - `--rgb-path`

这些能力都从同一个脚本进入:

- `scripts/refine_robust_v2.py`

---

## 4. 最常见的误区

### 误区 1: 只跑 README 的两条命令就已经用了 v2

不是。

README 里的 `Example 1` 只做到:

1. 生成 diffusion 输出
2. 用 `sample.py` 重建初始高斯

这一步结束后,你拿到的是 baseline 高斯与 baseline render。
还没有进入 v2 refinement。

### 误区 2: `sample.py` 输出的视频就是 refinement 的最终结果

不是。

`sample.py` 输出的 `main_gaussians_renderings/rgb_0.mp4` 只是 baseline render。
它可以用来观察当前质量,但还不是 `joint_refinement_camera_gaussians_v2` 细化后的最终结果。

### 误区 3: refinement 一定要重新生成 diffusion

也不是。

refinement 的输入核心是:

- 场景配置
- baseline 导出的 `.ply`
- 对应的 dataset / 视角 / 参考图像

也就是说,只要 baseline 产物还在,你可以反复调 refinement 参数,不需要每次都重新跑 diffusion。

### 误区 4: `stage3a / phase3s / stage3sr` 是另一套独立程序

不是。

这几个名字现在更准确的理解是:

- `stage2a` 是对外主线阶段名
- `stage3a / phase3s / stage3sr` 是 `stage2a` 内部的增强子阶段

也就是说:

- 你仍然只运行同一个脚本:
  - `scripts/refine_robust_v2.py`
- 不是先跑一套 `stage2a`,再切去另一套 `Long-LRM` 程序
- 而是当前代码把 `stage2a` 内部拆得更细了, 这样 selective SR 和 diagnostics 才能单独落盘

默认理解可以记成:

- `stage2a`
  - 先做 native cleanup
  - 再算 `gaussian_fidelity_score`
  - 再做 selective SR patch supervision

---

## 5. 你至少要跑哪几步

### 5.1 最小必需链路

如果你要真正用上 v2 增强,最少是 3 步:

### 第 1 步: 生成或准备 SDG 输出

如果你要自己从单张图生成:

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py \
    --checkpoint_dir checkpoints \
    --num_gpus 1 \
    --input_image_path assets/demo/static/diffusion_input/images/00172.png \
    --video_save_folder assets/demo/static/diffusion_output_generated \
    --foreground_masking \
    --multi_trajectory \
    --total_movement_distance_factor 1.0
```

如果你不想自己生成,也可以直接用仓库自带的:

- `assets/demo/static/diffusion_output`

### 第 2 步: 用 `sample.py` 重建 baseline 3DGS

```bash
accelerate launch sample.py --config configs/demo/lyra_static.yaml \
  dataset_name=lyra_static_demo_generated \
  save_gaussians_orig=true save_gaussians=true
```

如果你当前机器是较老的 GPU,这里要多注意一件事:

- `sample.py` 仍然依赖 `Mamba + Triton`
- 在 `sm_61` 这类旧卡上,可能在 baseline 生成时就被 Triton kernel 卡住
- 这不是 `refine_robust_v2.py` 本身的问题
- 如果你已经有 baseline `.ply`,后面的 v2 refinement 仍然可以继续跑

### 第 3 步: 用 `refine_robust_v2.py` 做增强细化

```bash
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --dataset-name lyra_static_demo_generated \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_3/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --view-id 3 \
  --outdir outputs/refine_v2/view3_full \
  --enable-pruning \
  --enable-stage2b \
  --enable-pose-diagnostic \
  --enable-joint-fallback
```

这 3 步都跑完,才算真正用到了 `joint_refinement_camera_gaussians_v2`。

### 5.2 不想依赖 provider 时,可以直接走 file inputs

如果你已经手里有成套文件:

- `pose/*.npz`
- `intrinsics/*.npz`
- `rgb/*.mp4` 或帧目录

那么现在同一个入口也支持 direct file inputs:

```bash
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_3/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --pose-path assets/demo/static/diffusion_output_generated/3/pose/00172.npz \
  --intrinsics-path assets/demo/static/diffusion_output_generated/3/intrinsics/00172.npz \
  --rgb-path assets/demo/static/diffusion_output_generated/3/rgb/00172.mp4 \
  --view-id 3 \
  --outdir outputs/refine_v2/view3_direct_inputs
```

这条模式的意义是:

- 仍然是同一个 `scripts/refine_robust_v2.py`
- 不是新程序
- 只是 `build_scene_bundle(...)` 不再强依赖 provider / dataloader

---

## 6. 为什么第 3 步必须单独跑

因为当前工程结构就是这样设计的:

- `sample.py` 负责 baseline 3DGS 推理与导出
- `scripts/refine_robust_v2.py` 负责 v2 联合细化

也就是:

- `sample.py` 不会自动顺手再帮你跑 refinement
- `refine_robust_v2.py` 也不会替代 `sample.py` 直接从 latent 端做重建

它们是前后串联的两个阶段,不是一个脚本里的两个开关。

## 6.1 现在怎么理解 `Stage 2A`

如果你只看对外流程, 仍然可以把它理解成一个阶段:

- `Stage 2A`

但如果你开始看输出目录和 metrics, 现在最好按下面这个映射理解:

| 对外主线阶段 | 内部子阶段 | 作用 | 常见产物 |
| --- | --- | --- | --- |
| `Stage 2A` | `Stage 3A` | native cleanup | `metrics_stage2a.json`, `gaussians_stage2a.ply` |
| `Stage 2A` | `Phase 3S` | fidelity / SR selection 诊断 | `metrics_phase3s.json`, `gaussian_fidelity_histogram.json`, `sr_selection_maps/` |
| `Stage 2A` | `Stage 3SR` | selective SR patch supervision | `metrics_stage3sr.json`, `gaussians_stage3sr.ply` |

最重要的一点是:

- 这 3 个内部名字不是新的程序入口
- 它们只是当前 `stage2a` 主线被拆开的可观测子阶段

---

## 7. baseline 输出里,refinement 要接哪一个文件

refinement 直接接 baseline 导出的 `.ply`。

最推荐接的是:

- `gaussians_orig/gaussians_0.ply`

原因很简单:

- 这是 `sample.py` 导出的原始格式高斯
- `refine_robust_v2.py` 当前就是按这个 `.ply` 入口来读的

常见路径形态是:

```text
outputs/demo/lyra_static/
  static_view_indices_fixed_<view>/
    <dataset_name>/
      gaussians_orig/
        gaussians_0.ply
```

比如当前仓库里已经存在的真实路径就包括:

- `outputs/demo/lyra_static/static_view_indices_fixed_3/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply`
- `outputs/demo/lyra_static/static_view_indices_fixed_5/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply`

---

## 8. 当前仓库有一个配置差异,要特别注意

`README.md` 里的描述,和当前仓库里的 `configs/demo/lyra_static.yaml` 不完全一样。

当前这个 YAML 里:

- `dataset_name: lyra_static_demo_generated_one`
- `static_view_indices_fixed: ['3']`

也就是说:

1. 它默认不是 `README` 文案里说的 `lyra_static_demo`
2. 它也不是常见的 `lyra_static_demo_generated`
3. 它当前固定主视角是 `3`

所以更稳妥的做法是:

- 在命令行里显式覆盖 `dataset_name`
- 并且让 `--view-id` 和你实际生成的轨迹保持一致

如果你当前就是按 `view 3` 在跑,那上面的示例命令可以直接照抄。

如果你改成 `view 5`,那么至少要一起改这几处:

- `sample.py` 输出目录里的 `static_view_indices_fixed_5`
- refinement 的 `--view-id 5`
- `--gaussians` 指向 `view 5` 对应的 `gaussians_0.ply`

---

## 9. 三种最实用的使用方式

### 9.1 方式 A: 最快验证,直接吃仓库自带 demo

适合场景:

- 你先只想确认 pipeline 能通
- 不关心是否用自己生成的 diffusion 输出

### 命令

```bash
accelerate launch sample.py --config configs/demo/lyra_static.yaml \
  dataset_name=lyra_static_demo \
  save_gaussians_orig=true save_gaussians=true
```

然后:

```bash
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --dataset-name lyra_static_demo \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_3/lyra_static_demo/gaussians_orig/gaussians_0.ply \
  --view-id 3 \
  --outdir outputs/refine_v2/view3_demo_builtin \
  --enable-pruning \
  --enable-stage2b \
  --enable-pose-diagnostic \
  --enable-joint-fallback
```

### 特点

- 最快
- 不需要重新生成 diffusion
- 适合先确认环境和 refinement 主线都正常

---

### 9.2 方式 B: 用你自己生成的 `diffusion_output_generated`

适合场景:

- 你已经跑了 `gen3c_single_image_sdg.py`
- 输出落在 `assets/demo/static/diffusion_output_generated`

### 命令

先跑 baseline:

```bash
accelerate launch sample.py --config configs/demo/lyra_static.yaml \
  dataset_name=lyra_static_demo_generated \
  save_gaussians_orig=true save_gaussians=true
```

再跑 refinement:

```bash
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --dataset-name lyra_static_demo_generated \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_3/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --view-id 3 \
  --outdir outputs/refine_v2/view3_generated_full \
  --enable-pruning \
  --enable-stage2b \
  --enable-pose-diagnostic \
  --enable-joint-fallback
```

### 特点

- 这是最接近“README Example 1 + v2 增强”的标准用法
- 推荐优先用这一条作为主工作流

---

### 9.3 方式 C: 接入外部超分视频做 patch supervision

适合场景:

- 你已经有一个外部工具生成的更高分辨率参考视频
- 想把它作为 refinement 的参考图像

### 最小命令

```bash
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --dataset-name lyra_static_demo_generated \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_3/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --view-id 3 \
  --outdir outputs/refine_v2/view3_external_reference \
  --reference-mode super_resolved \
  --sr-scale 2.0 \
  --reference-path /path/to/external_rgb.mp4 \
  --enable-pruning \
  --enable-stage2b \
  --enable-pose-diagnostic \
  --enable-joint-fallback
```

如果你的外部 reference 对应的内参也变了,再额外加:

```bash
--reference-intrinsics-path /path/to/reference_intrinsics.npz
```

### 什么时候一定要给 `reference_intrinsics_path`

如果你的外部工具做了下面这些事,最好明确给:

- 裁剪
- 非等比缩放
- 改了视场角
- 改了主点位置
- 不只是简单 2x 放大

如果只是严格同视角、同构图、同中心点的纯放大,当前实现可以按分辨率自动推断缩放后的 `intrinsics_ref`。

### 这条路的边界

external reference 已经可以真实进入 patch supervision 主线。

但前提仍然是:

- 轨迹一致
- 帧顺序一致
- 每一帧和原始轨迹是同一相机位姿

如果外部工具偷偷改了镜头语言,即便画面更“好看”,几何上也会不对齐。

---

## 10. `--start-stage stage2b` 什么时候用

这个参数不是第一次跑就要开。

它是给下面这种情况准备的:

- 你已经有上一轮 refinement 导出的 `gaussians_stage2a.ply`
- 想直接从 `Stage 2B` 往后续跑

例如:

```bash
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --dataset-name lyra_static_demo_generated \
  --gaussians outputs/refine_v2/previous_run/gaussians/gaussians_stage2a.ply \
  --view-id 3 \
  --outdir outputs/refine_v2/view3_stage2b_resume \
  --start-stage stage2b \
  --enable-stage2b
```

适合理解成:

- 第一次跑: 从 baseline `.ply` 开始
- 第二次增量跑: 从 `gaussians_stage2a.ply` 开始

这里也顺手说明一下:

- `--start-stage` 目前仍然只有:
  - `stage2a`
  - `stage2b`
- 不需要也不能把它理解成:
  - `--start-stage stage3a`
  - `--start-stage phase3s`
  - `--start-stage stage3sr`

因为这些是 `stage2a` 内部子阶段,不是对外 CLI 的独立入口

---

## 11. refinement 跑完后你会看到什么

输出目录大致会包含这些内容:

```text
outdir/
  diagnostics.json
  metrics_phase0.json
  metrics_phase1.json
  metrics_stage2a.json
  metrics_phase3s.json
  metrics_stage3sr.json
  metrics_stage2b.json
  state/
  gaussians/
    gaussians_stage2a.ply
    gaussians_stage3sr.ply
    gaussians_stage2b.ply
    gaussians_refined.ply
  videos/
    baseline_render.mp4
    gt_reference.mp4
    final_render.mp4
```

注意:

- 这里列的是“常见全量产物”
- 实际某次运行会不会全部出现,取决于:
  - 你有没有开 patch supervision
  - 有没有真的进入 `Stage 2B`
  - 有没有继续进入 `Phase 3 / Phase 4`

你最常会关心的是:

- `gaussians/gaussians_stage2a.ply`
- `gaussians/gaussians_stage3sr.ply`
- `gaussians/gaussians_stage2b.ply`
- `gaussians/gaussians_refined.ply`
- `videos/baseline_render.mp4`
- `videos/final_render.mp4`
- `metrics_phase3s.json`
- `metrics_stage3sr.json`

---

## 12. 推荐的实际使用顺序

如果你不是在做特别复杂的对比实验,推荐按这个顺序来:

### 路线 1: 普通使用

1. 跑自己的 `diffusion_output_generated`
2. 跑 `sample.py`
3. 跑一次完整 refinement
4. 看 `baseline_render.mp4` 和 `final_render.mp4`

### 路线 2: 外部 SR 接入

1. 先完成 baseline 3DGS
2. 先跑一版不带 external reference 的 refinement
3. 再加 `--reference-path` 比较差异
4. 如果外部视频不是纯放大,补 `--reference-intrinsics-path`

### 路线 3: 继续增量细化

1. 先拿到 `gaussians_stage2a.ply`
2. 用 `--start-stage stage2b` 续跑
3. 再观察 `Stage 2B` 是否有真实收益

如果你当前就在调 selective SR, 还可以加一条更细的观察顺序:

1. 先看 `metrics_stage2a.json`
2. 再看 `metrics_phase3s.json`
3. 再看 `metrics_stage3sr.json`
4. 最后再决定要不要继续观察 `Stage 2B`

---

## 13. 一份最推荐直接照抄的命令

如果你现在问“那我最该先跑哪一套”,推荐就是这一套:

### 1) 生成 diffusion 输出

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py \
    --checkpoint_dir checkpoints \
    --num_gpus 1 \
    --input_image_path assets/demo/static/diffusion_input/images/00172.png \
    --video_save_folder assets/demo/static/diffusion_output_generated \
    --foreground_masking \
    --multi_trajectory \
    --total_movement_distance_factor 1.0
```

### 2) 重建 baseline 3DGS

```bash
accelerate launch sample.py --config configs/demo/lyra_static.yaml \
  dataset_name=lyra_static_demo_generated \
  save_gaussians_orig=true save_gaussians=true
```

### 3) 运行 v2 enhancement

```bash
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --dataset-name lyra_static_demo_generated \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_3/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --view-id 3 \
  --outdir outputs/refine_v2/view3_generated_full \
  --enable-pruning \
  --enable-stage2b \
  --enable-pose-diagnostic \
  --enable-joint-fallback
```

---

## 14. 最后给一个很短的判断标准

如果你只看命令层面,记住下面这句就够了:

- 想得到 baseline,跑 `README` 两步
- 想得到 `joint_refinement_camera_gaussians_v2` 增强结果,必须再跑 `scripts/refine_robust_v2.py`

这就是当前实现的标准使用方式。
