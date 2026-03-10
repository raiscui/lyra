# 笔记: 2026-03-08 A800 `stage2a` `target_index_subsample=4` 基准

## 2026-03-08 07:04 UTC: A800 `stage2a` `sub4` 基准前置判断

### 现象

- 旧文档 `docs/cmd.md` 的 full-view 正式起点还是 `target_subsample=16`。
- `outputs/refine_v2/full_view_native_stage2a_fair_v2_20260308_0526` 与 `fair_v3_20260308_0630` 的 baseline/final 视频都是 `48` 帧, 因而不是 `sub4`。
- 当前机器 GPU 已确认:
  - `NVIDIA A800-SXM4-80GB`
  - `81920 MiB`

### 假设

- `target_subsample=4` 对 full-view 会形成约 `31 frames/view * 6 views = 186 observations`。
- 这比 `sub16` 的 `48 observations` 高很多, 风险主要在显存与 wall time, 不在数据契约。
- 因此更稳的推进方式是:
  1. 先 smoke
  2. 再正式 benchmark

### 验证计划

- 先用 full-view native `stage2a` 路径跑 `--target-subsample 4` 的最小 smoke。
- 若 smoke 成功, 再跑增强口径的正式 benchmark, 并采样 GPU memory used / utilization。

## 2026-03-08 07:04 UTC: A800 `sub4` smoke 验证结果

### 验证命令

1. 不带 allocator 调整:

```bash
env PYTHONPATH="$(pwd)" pixi run python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --scene-stem 00172 \
  --view-ids 5,0,1,2,3,4 \
  --pose-root assets/demo/static/diffusion_output_generated \
  --intrinsics-root assets/demo/static/diffusion_output_generated \
  --rgb-root assets/demo/static/diffusion_output_generated \
  --reference-mode native \
  --target-subsample 4 \
  --stage2a-mode enhanced \
  --patch-size 256 \
  --lambda-patch-rgb 0.25 \
  --lambda-sampling-smooth 0.0005 \
  --enable-pruning \
  --iters-stage2a 2 \
  --save-every 1 \
  --stop-after stage2a \
  --outdir outputs/refine_v2/full_view_native_stage2a_smoke_sub4_20260308_0707
```

2. 只增加 allocator 试验:

```bash
env PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  pixi run python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --scene-stem 00172 \
  --view-ids 5,0,1,2,3,4 \
  --pose-root assets/demo/static/diffusion_output_generated \
  --intrinsics-root assets/demo/static/diffusion_output_generated \
  --rgb-root assets/demo/static/diffusion_output_generated \
  --reference-mode native \
  --target-subsample 4 \
  --stage2a-mode enhanced \
  --patch-size 256 \
  --lambda-patch-rgb 0.25 \
  --lambda-sampling-smooth 0.0005 \
  --enable-pruning \
  --iters-stage2a 2 \
  --save-every 1 \
  --stop-after stage2a \
  --outdir outputs/refine_v2/full_view_native_stage2a_smoke_sub4_expandseg_20260308_0710
```

### 动态证据

- 两次 run 都在 `Phase 0` 失败。
- 失败调用链一致:
  - `runner.run_phase0()`
  - `runner.render_scene()`
  - `src/rendering/gs.py:GaussianRenderer.render()`
  - `src/rendering/gs.py:_merge_chunk_meta()`
- 第一次 OOM 关键输出:
  - `Tried to allocate 4.25 GiB`
  - `Process 20354 has 75.63 GiB memory in use`
  - `allocated=71.11 GiB`
  - `reserved but unallocated=4.04 GiB`
  - wall time: `17s`
- 第二次 OOM 关键输出:
  - `Tried to allocate 4.25 GiB`
  - `Process 22024 has 78.06 GiB memory in use`
  - `allocated=74.73 GiB`
  - `reserved but unallocated=2.86 GiB`
  - `gpu_mem_smoke.csv` 采样峰值: `79945 MiB`
  - wall time: `18s`

### 静态证据

- [`src/rendering/gs.py`](/workspace/lyra/src/rendering/gs.py) 当前会在渲染完每个 view chunk 后, 对以下 dense meta 做全量回拼:
  - `radii`
  - `means2d`
  - `depths`
  - `conics`
  - `opacities`
  - `tiles_per_gauss`
- OOM 正好发生在 `_merge_chunk_meta()` 的 `torch.cat` / `torch.stack`。
- 因为当前 `gs_view_chunk_size=1`, 所以 chunk 内存已经被切到最小; 爆点不是“单 chunk 太大”, 而是“所有 chunk 的 dense meta 合并回全视图张量”。

### 结论

- 主假设成立:
  - 当前实现下, full-view native `stage2a` 的 `target_subsample=4` 已经超过 A800 80GB 可承受上限。
- 备选解释被推翻:
  - 这不只是 allocator 碎片化。
  - 证据是 `expandable_segments:True` 仍在同一路径 OOM, 且进程峰值已接近整卡上限。
- 因此本轮“基准”已经建立为:
  - `sub4` 不是“慢但能跑”, 而是“当前代码在 Phase 0 就 OOM 的容量边界”。
- 当前最接近根因的候选判断是:
  - 高 observation 密度下, `render_meta` 的 dense 合并策略本身就是主要内存瓶颈。

## [2026-03-09 04:06 UTC] add-refinement-v2-depth-anchor 可行性分析 - 历史上下文摘录

### 已观察到的事实
- OpenSpec 当前只有一个活跃 change: `add-refinement-v2-depth-anchor`。
- 六文件最近主线集中在 `joint_refinement_camera_gaussians_v2`、`Stage 2B`、external reference contract、以及 selective SR 后续路线。
- `LATER_PLANS.md` 已明确记录: selective SR 主线待补 `gaussian_fidelity_score`、`W_sr_select`、`W_final_sr`、`L_sampling_smooth`、`Phase 3S / Stage 3SR`。
- `EPIPHANY_LOG.md` 已明确记录: `Stage 2A` 不是几何修复阶段,baseline 空间厚化要优先回看初始化链路,不能把几何问题直接归因给后续 appearance refinement。
- 目前六文件里没有直接提到 `depth-anchor` 或 `add-refinement-v2-depth-anchor` 的既有结论。

### 初步含义
- 这条 change 很可能是一次新探索,不是单纯把现有 selective SR 任务换个名字。
- 其可行性判断不能只看 loss 设计,还必须对照当前几何初始化、深度来源、以及 stage 语义是否允许它真正改变问题。

## [2026-03-09 04:18 UTC] add-refinement-v2-depth-anchor 可行性分析 - 代码与验证证据

### 现象
- `src/refinement_v2/losses.py` 当前只有 RGB / scale / opacity / patch / means / rotation / pose 相关损失,没有 depth loss。
- `src/refinement_v2/gaussian_adapter.py` 当前把 `stage2a/stage3a/stage3sr` 的可训练参数限制为 `opacity/scales/colors`,`means/rotations` 冻结。
- `src/rendering/gs.py` 当前通过 `gsplat.rasterization(..., render_mode="RGB+ED")` 返回 `images_pred/alphas_pred/depths_pred`。
- `src/refinement_v2/runner.py` 中 `Phase 0 / Phase 1` 已经使用 `_render_scene_for_evaluation(...)` 做无 backward 的 baseline 渲染; 多卡时还会把结果聚到 CPU。
- `src/refinement_v2/data_loader.py` 的 `SceneBundle` 不含 depth 字段,且 refinement overrides 显式 `use_depth=false`。

### 动态证据
- 运行 `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_runner_phase0.py tests/refinement_v2/test_runner_stage2a.py tests/refinement_v2/test_patch_supervision.py`
- 结果: `18 passed in 3.20s`
- 这些测试已覆盖:
  - `Stage 2A` 不改 `means`,会改 appearance 参数
  - `Stage 2A` 与 `Stage 3SR` 当前是分阶段留痕
  - 多设备 `Phase 0 / Phase 1 / Phase 3S` 走 evaluation render 路径
- 额外最小脚本验证:
  - `Stage 2A` 在 synthetic 场景下 `opacity_changed=True`, `color_changed=True`, `scale_changed=False`
  - 含义: 即使不动 `means`,appearance 阶段也确实会改变会影响 alpha / depth 聚合行为的参数

### 推断与约束
- 基于 `gsplat` 官方文档,`RGB+ED` 的 depth 是 expected depth,而不是 GT depth 或独立几何标签。
- 因此 baseline_render depth anchor 的本质更接近“防止 appearance 过程把当前几何分布继续带偏”,而不是“修复 baseline 初始化错误”。
- 若 baseline 本身已有厚表面/沿视线拉长,把 baseline depth 当 reference 也可能把这类问题一起锚住。
- 因为当前 appearance 阶段显式包含 `loss_opacity_sparse`,depth anchor 与 opacity 稀疏化之间存在真实耦合,必须谨慎处理低 alpha 像素。

### 设计层面的可行性判断
- 工程接入层面: 可行,且侵入面比 proposal 里写的还小。
- 价值层面: 适合作为 V1 的 anti-drift loss,不适合作为 geometry correction 方案来承诺效果。
- 最稳切入点:
  1. baseline reference 优先复用 `Phase 0/Phase 1` evaluation render,不要额外发明新的数据输入链
  2. reference depth / alpha 建议缓存到 runner CPU 状态,多卡时用现有 `_slice_view_tensor(...)` 下发到 shard
  3. V1 最好先默认只在 `Stage 3SR` 或者保守权重下启用,再观察是否扩到 `Stage 2A`

## [2026-03-09 04:37 UTC] `moge_version` 默认行为调用链核对

### 现象
- README Example 1 的第 1 步实际运行的是 `cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py`。
- README Example 1 的第 2 步 `sample.py` 中没有 `moge` 相关参数或加载逻辑。
- `gen3c_single_image.py` 与 `gen3c_single_image_sdg.py` 都在 parser 初始化时调用 `add_moge_arguments(parser)`。
- `gen3c_persistent.py` 直接复用 `gen3c_single_image.create_parser()`。
- `add_moge_arguments(...)` 把 `--moge_version` 默认值设为 `auto`。
- `load_moge_model(...)` 在未显式提供 `moge_model_id` 时,会调用 `_resolve_default_moge_model_id(moge_version)`。
- `_resolve_default_moge_model_id(...)` 对 `auto` 返回 `DEFAULT_MOGE_MODEL_IDS["v2"]`,即 `Ruicheng/moge-2-vitl`。
- 如果显式给了 `--moge_checkpoint_path` 或把 `--moge_model_id` 写成本地文件路径,则 `auto` 会先读取 checkpoint 的 `model_config["encoder"]` 结构,再自动识别是 `v1` 还是 `v2`。

### 结论
- CLI 层默认值: `auto`
- 内建 repo 选择默认值: `auto -> v2`
- 本地 checkpoint 自动识别: `auto -> 按权重结构识别 v1/v2`
- 对 README 常用链路而言:
  1. 第 1 步 `gen3c_single_image_sdg.py` 若不传 `--moge_version`,实际会走 `v2`
  2. 第 2 步 `sample.py` 不再参与 `MoGe` 版本选择
- `scripts/bash/static_sdg.sh` 也没有额外传 `--moge_version`,因此同样继承 `auto -> v2`。

## 2026-03-09 视角索引对应关系排查

### 现象

- 顶层配置常用 `static_view_indices_fixed: ['5', '0', '1', '2', '3', '4']`.
- `dataset_registry` 中 `sampling_buckets` 仍是 `[['0'], ['1'], ['2'], ['3'], ['4'], ['5']]`, `start_view_idx=0`.

### 静态证据

- `Provider._sample_view_indices_bucket()` 在 `static_view_indices_sampling == 'fixed'` 时直接返回 `self.opt.static_view_indices_fixed`.
- `sampling_buckets` 只在 `static_view_indices_sampling == 'random_bucket'` 时进入 `_sample_view_indices_from_bucket()`.
- `start_view_idx` 只用于:
  - 随机生成 bucket 索引时的偏移
  - 未显式给 view 时默认从哪个视角开始读数据
- `Radym._read_data()` 对多视角数据会直接用传入的 `view_idx` 拼路径 `<root>/<view_idx>/...`.

### 动态证据

- 用 `.pixi/envs/default/bin/python` 实例化 `Provider('lyra_static_demo_generated', training=False)`.
- 覆盖:
  - `static_view_indices_sampling='fixed'`
  - `static_view_indices_fixed=['5','0','1','2','3','4']`
  - `num_input_multi_views=6`
- `provider._get_indices_static(0)` 输出:
  - `num_input_multi_views = 6`
  - `input_view_indices = ['5', '0', '1', '2', '3', '4']`
- 资产目录 `assets/demo/static/diffusion_output_generated/{0..5}/rgb/00172.mp4` 全部存在.

### 中间判断

- 当前不是“bucket 第 0 位一定要对上 fixed 第 0 位”的设计.
- fixed 模式下, 顶层配置给出的字符串列表就是最终视角 ID 顺序.
- 因此 `['5','0','1','2','3','4']` 与 registry 中顺序不同, 本身不会导致错位.

### 补充风险

- 如果未来改回 `static_view_indices_sampling='random_bucket'`,这时才会走 `sampling_buckets` + `start_view_idx` 的映射逻辑.
- 如果未来数据集目录不是 `0..5`,而是别的编号起点,才需要同步修改 `start_view_idx` 和 bucket 内容.
