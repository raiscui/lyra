# Long-LRM Style Post Refinement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** 基于 `specs/long_lrm_style_post_refinement.md`, 落地一条可运行的后置 refinement 路线: `sample.py` 继续做 feed-forward 初始化, `scripts/refine_robust_v2.py` 负责 `Long-LRM` 风格的 post-prediction optimization, 默认先做 `gaussian-only` 的 `appearance-first` 优化, 再通过 `SplatSuRe-style selective SR` 选择性吸收超分后的对等视频高频监督,并用 `Mip-inspired sampling-aware smoothing` 约束不受支持的高频细节.

**Architecture:** 沿用现有 `refinement_v2` 骨架, 不另开新入口. 第一版执行顺序固定为: `SceneBundle` 装配与对齐 -> baseline render / diagnostics -> `W_robust` 构造 -> Stage 3A native cleanup -> 可选 opacity / pruning -> `gaussian_fidelity_score + W_sr_select` -> selective SR patch supervision(`W_final_sr = W_robust * W_sr_select`) -> `L_sampling_smooth` 约束 -> 可选 Stage 3B limited geometry -> state / export / resume. `sample.py` 不改主链, V1 不做 pose optimization, 也不引入 Long-LRM 的长序列 token budget 控制.

**Tech Stack:** Python 3.10, PyTorch, existing Lyra provider / renderer (`src/models/data/provider.py`, `src/rendering/gs_deferred.py`, `src/rendering/gs_deferred_patch.py`), OmegaConf / YAML, pathlib / json, pytest.

---

### Task 1: 对齐 CLI 与配置层到新 spec

**Files:**
- Modify: `scripts/refine_robust_v2.py`
- Modify: `src/refinement_v2/config.py`
- Test: `tests/refinement_v2/test_config.py`
- Read: `specs/long_lrm_style_post_refinement.md`

**Step 1: 先写失败测试, 锁住新 CLI 形状**

测试至少覆盖:

- `--gaussians`
- `--pose`
- `--intrinsics`
- `--reference-video`
- `--reference-mode`
- `--sr-scale`
- `--scene-index`
- `--view-id`
- `--outdir`
- `--frame-indices`
- `--target-subsample`
- `--patch-size`
- `--dry-run`
- `--resume`
- `--enable-stage3b`

建议先在 `tests/refinement_v2/test_config.py` 增加类似断言:

```python
def test_cli_maps_long_lrm_post_refinement_args() -> None:
    run_config, stage_hparams = load_effective_config_from_cli(
        [
            "--config", "configs/demo/lyra_static.yaml",
            "--gaussians", "outputs/demo/gaussians_0.ply",
            "--pose", "assets/demo/pose/demo.npz",
            "--intrinsics", "assets/demo/intrinsics/demo.npz",
            "--reference-video", "assets/demo/rgb/demo.mp4",
            "--reference-mode", "super_resolved",
            "--sr-scale", "2.0",
            "--scene-index", "0",
            "--view-id", "3",
            "--outdir", "outputs/refine_v2/demo",
            "--patch-size", "256",
            "--enable-stage3b",
        ]
    )

    assert str(run_config.gaussians_path).endswith("gaussians_0.ply")
    assert run_config.reference_mode == "super_resolved"
    assert run_config.sr_scale == 2.0
    assert run_config.enable_stage3b is True
    assert stage_hparams.patch_size == 256
```

**Step 2: 跑测试, 确认当前确实还不满足**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_config.py
```

Expected:

- 新增断言先失败
- 失败点集中在字段缺失、默认值不完整, 或 CLI 参数未注册

**Step 3: 实现最小配置层**

在 `src/refinement_v2/config.py` 至少明确这几个对象:

```python
@dataclass
class RefinementRunConfig:
    config_path: Path
    gaussians_path: Path
    pose_path: Path
    intrinsics_path: Path
    reference_video_path: Path
    reference_mode: str
    sr_scale: float
    scene_index: int
    view_id: str | None
    outdir: Path
    frame_indices: list[int] | None
    target_subsample: int | None
    dry_run: bool
    resume: bool
    enable_stage3b: bool


@dataclass
class StageHyperParams:
    patch_size: int
    weight_floor: float
    residual_tau: float
    opacity_prune_threshold: float
    prune_every: int
    means_delta_cap: float
```

并让 `scripts/refine_robust_v2.py` 只负责:

- 解析 CLI
- 加载配置
- 把 config 交给 runner

**Step 4: 再跑测试和帮助信息**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_config.py
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py --help
```

Expected:

- `test_config.py` 全部通过
- `--help` 能打印出新参数

**Step 5: 提交本任务(可选, 若按 task 落地)**

```bash
git add scripts/refine_robust_v2.py src/refinement_v2/config.py tests/refinement_v2/test_config.py
git commit -s -m "feat: align refine cli with post refinement spec"
```

**Done when:**

- 新 spec 需要的 CLI 全部能落到 `RefinementRunConfig`
- `reference_mode` / `sr_scale` / `enable_stage3b` 有明确默认值
- 明确声明 V1 不含 pose optimization 和 token budget 控制

---

### Task 2: 实现 `SceneBundle` 与 reference video 对齐层

**Files:**
- Modify: `src/refinement_v2/data_loader.py`
- Modify: `src/refinement_v2/config.py`
- Test: `tests/refinement_v2/test_data_loader.py`
- Read: `src/models/data/provider.py`

**Step 1: 先写 synthetic 单测**

至少覆盖:

- `frame_indices` 过滤
- `target_subsample` 生效
- `reference_mode=native` 时 `reference_hw == native_hw`
- `reference_mode=super_resolved` 时 `intrinsics_ref == intrinsics_native * sr_scale`
- 帧数不一致时报错
- aspect ratio / crop 不匹配时报错

建议补一个显式的失败测试:

```python
def test_super_resolved_reference_requires_scaled_intrinsics() -> None:
    bundle = build_scene_bundle(
        gaussians_path=Path("dummy.ply"),
        pose_path=pose_path,
        intrinsics_path=intrinsics_path,
        reference_video_path=reference_path,
        reference_mode="super_resolved",
        sr_scale=2.0,
        scene_index=0,
        view_id="3",
    )

    assert torch.allclose(bundle.intrinsics_ref, bundle.intrinsics_native * 2.0)
```

**Step 2: 跑测试, 确认对齐约束尚未落地**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_data_loader.py
```

Expected:

- 新增 SR / mismatch 测试先失败

**Step 3: 实现 `SceneBundle`**

建议把 `data_loader.py` 里的核心对象收紧成:

```python
@dataclass
class SceneBundle:
    gaussians_path: Path
    reference_frames: torch.Tensor
    cam_view: torch.Tensor
    intrinsics_native: torch.Tensor
    intrinsics_ref: torch.Tensor
    native_hw: tuple[int, int]
    reference_hw: tuple[int, int]
    sr_scale: float
    reference_mode: str
    frame_indices: list[int]
    scene_index: int
    view_id: str | None
```

同时实现:

- `load_reference_video(...)`
- `scale_intrinsics(...)`
- `validate_reference_alignment(...)`
- `build_scene_bundle(...)`

这里要坚持 fail-fast:

- 帧数不一致直接报错
- SR 不是整数倍率直接报错
- crop / aspect ratio 不匹配直接报错

**Step 4: 再跑单测**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_data_loader.py
```

Expected:

- `SceneBundle` 相关测试全部通过

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/data_loader.py src/refinement_v2/config.py tests/refinement_v2/test_data_loader.py
git commit -s -m "feat: build aligned scene bundle for post refinement"
```

**Done when:**

- runner 后面只接 `SceneBundle`, 不直接碰 provider 原始 batch
- native / SR 的 intrinsics 约束被统一固定

---

### Task 3: 打通 baseline render 与第一批 diagnostics

**Files:**
- Modify: `src/refinement_v2/runner.py`
- Modify: `src/refinement_v2/diagnostics.py`
- Modify: `tests/refinement_v2/test_diagnostics.py`
- Create: `tests/refinement_v2/test_runner_baseline.py`
- Read: `src/rendering/gs_deferred.py`

**Step 1: 先写 baseline 层测试**

至少覆盖:

- `render_baseline()` 能返回与 `frame_indices` 对应的渲染结果
- `compute_baseline_metrics()` 输出:
  - `psnr`
  - `sharpness`
  - `opacity_lowconf_ratio`
  - `scale_tail_ratio`
- baseline 失败时 runner 不进入后续优化阶段

建议用 monkeypatch 或 synthetic renderer stub, 不要一开始依赖真实大模型.

**Step 2: 跑测试, 让 baseline contract 先红**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_diagnostics.py tests/refinement_v2/test_runner_baseline.py
```

Expected:

- baseline / diagnostics 的新 contract 先失败

**Step 3: 实现最小 baseline 路径**

在 `runner.py` 增加类似骨架:

```python
class RefinementRunner:
    def render_baseline(self, bundle: SceneBundle) -> BaselineResult:
        ...

    def run(self) -> None:
        bundle = build_scene_bundle(...)
        baseline = self.render_baseline(bundle)
        self.write_baseline_artifacts(baseline)
        if self.config.dry_run:
            return
```

在 `diagnostics.py` 实现:

- `summarize_baseline_metrics(...)`
- `summarize_opacity_stats(...)`
- `summarize_scale_stats(...)`

先把结构稳定下来, 指标值可以先依赖 synthetic / stub 输出做验证.

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_diagnostics.py tests/refinement_v2/test_runner_baseline.py
```

Expected:

- baseline contract 通过
- dry-run 有 baseline 产物, 但不会继续进入优化

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/runner.py src/refinement_v2/diagnostics.py tests/refinement_v2/test_diagnostics.py tests/refinement_v2/test_runner_baseline.py
git commit -s -m "feat: add baseline render and diagnostics"
```

**Done when:**

- `Phase 1: baseline render and diagnostics` 可独立跑通
- baseline 失败会中止流程, 不会硬进优化阶段

---

### Task 4: 落地 `W_robust` 构造器

**Files:**
- Modify: `src/refinement_v2/weight_builder.py`
- Modify: `src/refinement_v2/losses.py`
- Test: `tests/refinement_v2/test_weight_builder.py`

**Step 1: 先写失败测试**

至少覆盖:

- residual 全 0 时, 权重接近全 1
- residual 极高时, 权重下降但不低于 `weight_floor`
- quantile normalize 后结果稳定在预期范围
- EMA 第二次更新比第一次更平稳
- full-frame 和 patch tensor 的 shape 都能通过

**Step 2: 跑测试, 确认 `WeightBuilder` 还没长成 spec 里的样子**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_weight_builder.py
```

Expected:

- 新增 residual / EMA / patch 测试先失败

**Step 3: 实现 V1 最小版**

第一版只做 spec 已经收敛过的部分:

- RGB residual
- quantile normalize
- `exp(-r_norm / tau)`
- `clamp(weight_floor, 1.0)`
- EMA
- optional blur

不要在这一步把 LPIPS / DINO / feature residual 一起塞进来.

建议对外接口固定成:

```python
class WeightBuilder:
    def build_from_residual(self, residual: torch.Tensor) -> torch.Tensor: ...
    def update_ema(self, weights: torch.Tensor) -> torch.Tensor: ...
```

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_weight_builder.py
```

Expected:

- 所有 residual / weight map 单测通过

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/weight_builder.py src/refinement_v2/losses.py tests/refinement_v2/test_weight_builder.py
git commit -s -m "feat: add residual weight builder"
```

**Done when:**

- `Phase 2: robust residual / trust weight build` 已可独立验证
- `W_robust` 的语义已经稳定
- 后面 Stage 3A / selective SR 都能复用同一套基础权重接口

---

### Task 5: 打通 `Long-LRM` 风格的 Stage 3A appearance-first 优化主线

**Files:**
- Modify: `src/refinement_v2/runner.py`
- Modify: `src/refinement_v2/stage_controller.py`
- Modify: `src/refinement_v2/gaussian_adapter.py`
- Modify: `src/refinement_v2/losses.py`
- Test: `tests/refinement_v2/test_gaussian_adapter.py`
- Test: `tests/refinement_v2/test_losses.py`
- Create: `tests/refinement_v2/test_runner_stage3a.py`

**Step 1: 先写失败测试**

至少覆盖:

- `Stage 3A` 默认只更新:
  - opacity
  - color
  - limited scale
- `means` / `rotation` 在 Stage 3A 中被冻结
- `L_rgb_weighted` 与 `L_opacity_sparse` 会一起出现在 loss dict
- runner 在一次 stage 内的顺序固定为:
  - render
  - residual
  - weight
  - loss
  - backward
  - step

**Step 2: 跑测试, 让 Stage 3A 的 contract 先失败**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_gaussian_adapter.py tests/refinement_v2/test_losses.py tests/refinement_v2/test_runner_stage3a.py
```

Expected:

- 参数冻结与 stage loop 测试先失败

**Step 3: 实现 Stage 3A 最小闭环**

这一步是整个计划的第一优先级, 也是最直接吸收 `Long-LRM` post-prediction optimization 的位置.

必须明确:

- `GaussianAdapter.freeze_for_stage("stage3a")`
- `GaussianAdapter.build_optimizer("stage3a", ...)`
- `StageController.should_stop_stage3a(...)`
- `compute_stage3a_losses(...)`

推荐最小 loss 组合:

- `L_rgb_weighted_native`
- `L_perceptual_patch` 的占位接口(允许先 stub)
- `L_opacity_sparse`
- `L_scale_ceiling`

这里先把“先 appearance, 后 geometry”的工程节奏固定死.

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_gaussian_adapter.py tests/refinement_v2/test_losses.py tests/refinement_v2/test_runner_stage3a.py
```

Expected:

- Stage 3A 的冻结 / loss / loop contract 全部通过

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/runner.py src/refinement_v2/stage_controller.py src/refinement_v2/gaussian_adapter.py src/refinement_v2/losses.py tests/refinement_v2/test_gaussian_adapter.py tests/refinement_v2/test_losses.py tests/refinement_v2/test_runner_stage3a.py
git commit -s -m "feat: implement appearance first post refinement"
```

**Done when:**

- `Long-LRM` 风格的 post-prediction optimization 主线已经有了最小工程闭环
- Stage 3A 明确只做 native cleanup,不提前把 SR 混进来
- 默认流程可以只跑到 Stage 3A 就结束

---

### Task 6: 单独补上 opacity / pruning 方法论

**Files:**
- Modify: `src/refinement_v2/gaussian_adapter.py`
- Modify: `src/refinement_v2/diagnostics.py`
- Modify: `src/refinement_v2/stage_controller.py`
- Create: `tests/refinement_v2/test_pruning.py`

**Step 1: 先写测试, 别让 pruning 只停留在口头**

至少覆盖:

- 低 opacity 高斯会被标成 prune candidate
- prune 前后统计会写入 diagnostics
- pruning 不会在 warmup 前触发
- 一次 pruning 不会删掉超过上限比例

建议锁住一个最小 contract:

```python
def test_pruning_respects_max_fraction() -> None:
    adapter = make_adapter_with_opacity([0.9, 0.01, 0.02, 0.03])
    kept = adapter.prune_low_opacity(
        threshold=0.05,
        max_fraction=0.25,
    )
    assert kept.num_gaussians == 3
```

**Step 2: 跑测试, 让 pruning 行为先红**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_pruning.py tests/refinement_v2/test_diagnostics.py
```

Expected:

- pruning 行为与 diagnostics 输出先失败

**Step 3: 实现最小 pruning controller**

这一步是第二优先级, 不能并入别的任务里含混带过.

至少实现:

- `GaussianAdapter.collect_prune_candidates(...)`
- `GaussianAdapter.prune_low_opacity(...)`
- `diagnostics.write_prune_summary(...)`
- `StageController.should_prune_now(iteration=...)`

要加的保护:

- warmup 迭代前不 prune
- 每次最多 prune 固定比例
- prune 后立刻重算 `opacity_lowconf_ratio`

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_pruning.py tests/refinement_v2/test_diagnostics.py
```

Expected:

- pruning 单测通过
- diagnostics 能记录 prune 前后统计

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/gaussian_adapter.py src/refinement_v2/diagnostics.py src/refinement_v2/stage_controller.py tests/refinement_v2/test_pruning.py tests/refinement_v2/test_diagnostics.py
git commit -s -m "feat: add opacity pruning controller"
```

**Done when:**

- opacity / pruning 已经从“方法论”变成可执行逻辑
- 这条能力可以和 Stage 3A 一起跑, 也可以单独开关

---

### Task 7: 接入 `SplatSuRe-style` selective SR patch supervision

**Files:**
- Modify: `src/refinement_v2/data_loader.py`
- Modify: `src/refinement_v2/runner.py`
- Modify: `src/refinement_v2/losses.py`
- Modify: `src/refinement_v2/weight_builder.py`
- Modify: `src/refinement_v2/gaussian_adapter.py`
- Modify: `src/refinement_v2/diagnostics.py`
- Create: `tests/refinement_v2/test_patch_supervision.py`
- Read: `src/rendering/gs_deferred_patch.py`

**Step 1: 先写 patch 监督测试**

至少覆盖:

- patch window 采样不会越界
- SR patch 坐标能映射回 native / reference 尺度
- patch 渲染使用 `intrinsics_ref`
- `reference_mode=native` 时也能复用同一 patch path
- `gaussian_fidelity_score` 能转成每视图 `W_sr_select`
- `W_final_sr = W_robust * W_sr_select` 的 shape 与数值范围正确
- `L_sampling_smooth` 不会替换掉原有 `L_scale_ceiling`

**Step 2: 跑测试, 让 patch contract 先失败**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_patch_supervision.py
```

Expected:

- patch 采样或 intrinsics 对齐相关断言先失败

**Step 3: 实现最小 selective SR patch supervision**

在 `runner.py` 里加:

- `sample_patch_windows(...)`
- `render_patch_prediction(...)`
- `gather_reference_patch(...)`
- `compute_gaussian_fidelity_score(...)`
- `build_sr_selection_maps(...)`

在 `weight_builder.py` 里加:

- `build_sr_selection_weight(...)`
- `combine_sr_weights(w_robust, w_sr_select)`

在 `losses.py` 里让下面几项都支持 patch 张量:

- `L_rgb_weighted_sr`
- `L_perceptual_patch_sr`
- `L_sampling_smooth`

强约束:

- patch 渲染与全帧渲染共用同一套高斯参数解释
- SR 模式下不把 supervision 偷偷缩回 native 全帧
- `W_sr_select` 不是 `W_robust` 的替代
- `W_final_sr` 必须显式由两者组合而来

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_patch_supervision.py
```

Expected:

- SR patch / native patch 两类测试都通过
- fidelity / selection / final SR weight 三类测试都通过

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/data_loader.py src/refinement_v2/runner.py src/refinement_v2/losses.py src/refinement_v2/weight_builder.py src/refinement_v2/gaussian_adapter.py src/refinement_v2/diagnostics.py tests/refinement_v2/test_patch_supervision.py
git commit -s -m "feat: add selective sr patch supervision"
```

**Done when:**

- 超分后的对等视频终于能以“选择性高分辨率监督”的身份进入后置 refinement
- `gaussian_fidelity_score`、`W_sr_select`、`W_final_sr` 三个对象都已落地
- `L_sampling_smooth` 已作为 `Mip-inspired` 互补约束进入主线
- `sample.py` 主链依然不用改

---

### Task 8: 实现可选的 Stage 3B limited geometry

**Files:**
- Modify: `src/refinement_v2/gaussian_adapter.py`
- Modify: `src/refinement_v2/runner.py`
- Modify: `src/refinement_v2/stage_controller.py`
- Modify: `src/refinement_v2/losses.py`
- Test: `tests/refinement_v2/test_stage_controller.py`
- Create: `tests/refinement_v2/test_runner_stage3b.py`

**Step 1: 先写失败测试**

至少覆盖:

- 未显式启用 `--enable-stage3b` 时不会进入 Stage 3B
- Stage 3B 允许更新:
  - scale
  - rotation
  - means
- `means` 更新被 `means_delta_cap` 限制
- `L_means_anchor` 和 `L_rotation_reg` 会加入总 loss

**Step 2: 跑测试, 让 Stage 3B contract 先红**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_stage_controller.py tests/refinement_v2/test_runner_stage3b.py
```

Expected:

- Stage 3B 开关与位移约束测试先失败

**Step 3: 实现 limited geometry**

这一步仍然保持克制:

- 只做小幅 geometry release
- 不做 pose
- 不做 densify-and-clone

至少实现:

- `GaussianAdapter.enable_stage3b_geometry(...)`
- `GaussianAdapter.clamp_means_delta(...)`
- `compute_stage3b_losses(...)`
- `StageController.should_enter_stage3b(...)`

进入条件要保守:

- Stage 3A 已完成
- selective SR 已完成
- ghosting 仍明显
- diagnostics 更像局部结构重叠, 不是整体 pose 错位或 SR 假细节污染

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_stage_controller.py tests/refinement_v2/test_runner_stage3b.py
```

Expected:

- Stage 3B 的 gating 与几何约束测试通过

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/gaussian_adapter.py src/refinement_v2/runner.py src/refinement_v2/stage_controller.py src/refinement_v2/losses.py tests/refinement_v2/test_stage_controller.py tests/refinement_v2/test_runner_stage3b.py
git commit -s -m "feat: add limited geometry refinement stage"
```

**Done when:**

- Stage 3B 只是可选后续阶段, 不会抢走 Stage 3A 的默认地位

---

### Task 9: 补齐 state / resume / 输出目录布局

**Files:**
- Modify: `src/refinement_v2/state_io.py`
- Modify: `src/refinement_v2/runner.py`
- Modify: `src/refinement_v2/diagnostics.py`
- Create: `tests/refinement_v2/test_state_io.py`

**Step 1: 先写失败测试**

至少覆盖:

- `config_effective.yaml` 会落盘
- `scene_bundle.json` 会落盘
- `diagnostics.json` / `metrics.json` 会落盘
- `state/*.pt` 能保存和恢复当前 stage / iter / gaussian 参数
- `--resume` 时能找到最近一次 checkpoint

**Step 2: 跑测试, 让状态恢复路径先失败**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_state_io.py
```

Expected:

- state / resume / artifact layout 相关测试先失败

**Step 3: 实现状态与产物写入**

输出目录结构必须与 spec 对齐:

```text
outdir/
  config_effective.yaml
  scene_bundle.json
  gaussian_fidelity_histogram.json
  sr_selection_stats.json
  diagnostics.json
  metrics.json
  gaussians_refined.ply
  videos/
    render_baseline.mp4
    render_refined.mp4
  residual_maps/
  weight_maps/
  sr_selection_maps/
  state/
```

至少实现:

- `save_checkpoint(...)`
- `load_latest_checkpoint(...)`
- `write_effective_config(...)`
- `write_metrics(...)`
- `write_diagnostics(...)`

**Step 4: 再跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_state_io.py
```

Expected:

- resume 和输出布局测试全部通过

**Step 5: 提交本任务(可选)**

```bash
git add src/refinement_v2/state_io.py src/refinement_v2/runner.py src/refinement_v2/diagnostics.py tests/refinement_v2/test_state_io.py
git commit -s -m "feat: add refinement state and resume outputs"
```

**Done when:**

- 后置 refinement 可以断点续跑
- dry-run / baseline / refined 三类产物边界清楚

---

### Task 10: 做最小集成验证与 dry-run 验收

**Files:**
- Modify: `scripts/refine_robust_v2.py`
- Modify: `src/refinement_v2/runner.py`
- Create: `tests/refinement_v2/test_smoke_dry_run.py`

**Step 1: 先写 smoke 测试**

至少覆盖:

- `--dry-run` 会执行:
  - CLI parse
  - SceneBundle build
  - baseline render
  - diagnostics write
- `--dry-run` 不会执行:
  - optimizer.step()
  - pruning
  - Stage 3B
- `--resume --dry-run` 组合不会崩溃

**Step 2: 跑测试, 先把 smoke contract 定死**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_smoke_dry_run.py
```

Expected:

- dry-run contract 先失败

**Step 3: 实现 dry-run 最终链路**

这里不要追求真实大场景全跑通.
先追求“流程是闭的”.

需要做到:

- `python3 scripts/refine_robust_v2.py ... --dry-run` 能稳定结束
- 输出至少包含:
  - `config_effective.yaml`
  - `scene_bundle.json`
  - `diagnostics.json`
  - `metrics.json`

**Step 4: 运行完整验证命令**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --gaussians outputs/demo/gaussians_0.ply \
  --pose assets/demo/static/diffusion_output_generated_one/3/pose/demo.npz \
  --intrinsics assets/demo/static/diffusion_output_generated_one/3/intrinsics/demo.npz \
  --reference-video assets/demo/static/diffusion_output_generated_one/3/rgb/demo.mp4 \
  --reference-mode native \
  --scene-index 0 \
  --view-id 3 \
  --outdir outputs/refine_v2/dry_run_demo \
  --dry-run
```

Expected:

- `tests/refinement_v2` 全部通过
- dry-run 退出码 0
- 输出目录按 spec 生成最小产物

**Step 5: 提交本任务(可选)**

```bash
git add scripts/refine_robust_v2.py src/refinement_v2/runner.py tests/refinement_v2/test_smoke_dry_run.py
git commit -s -m "feat: add dry run validation for post refinement"
```

**Done when:**

- 新路线已经从“spec + 任务清单”推进到“可验证的最小工程闭环”

---

## 实施顺序建议

1. 必做主线:
   - Task 1
   - Task 2
   - Task 3
   - Task 4
   - Task 5
2. 第二优先级:
   - Task 6
3. 接选择性高分辨率监督:
   - Task 7
4. 可选受限几何:
   - Task 8
5. 可靠性和交付:
   - Task 9
   - Task 10

## 明确不在本计划内

- `sample.py` 主链改造成高分辨率输入
- pose optimization
- joint pose + gaussian optimization
- Long-LRM 的长序列 token budget 控制
- `Mip-Splatting` 的 2D Mip filter(renderer-level 改造)
- `EDGS-style` local reinitialization
- `tttLRM` 的 fast-weight memory / LaCT / streaming reconstruction

## 一句话落地顺序

先把 `Long-LRM` 风格的 post-prediction optimization 主线跑通, 再把 opacity / pruning 变成硬逻辑, 然后接 `SplatSuRe-style` selective SR patch supervision 与 `Mip-inspired` smoothing, 最后才给有限 geometry 一个保守入口.
