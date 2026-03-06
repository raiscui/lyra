# Joint Refinement Camera Gaussians V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** 实现 `joint_refinement_camera_gaussians_v2` 的第一版可运行工程,支持 `Phase 0 + Phase 1 + Stage 2A` 为主线,并为 `Stage 2B / Phase 3 / Phase 4` 预留清晰扩展点.

**Architecture:** 以 `scripts/refine_robust_v2.py` 为唯一入口,把配置解析、数据对齐、权重构造、高斯适配、阶段控制、诊断落盘拆到 `src/refinement_v2/` 下的独立模块. 第一版先做 `gaussian-only refinement`,不默认动 pose,不默认做 joint fallback,优先证明 `residual weighting + appearance-first refinement` 能有效减轻 ghosting.

**Tech Stack:** Python 3.10, PyTorch, existing Lyra provider/renderer, OmegaConf/YAML config loading, pytest, pathlib/json.

---

### Task 1: 搭好入口脚手架和包结构

**Files:**
- Create: `scripts/refine_robust_v2.py`
- Create: `src/refinement_v2/__init__.py`
- Create: `src/refinement_v2/config.py`
- Create: `src/refinement_v2/data_loader.py`
- Create: `src/refinement_v2/runner.py`
- Create: `src/refinement_v2/stage_controller.py`
- Create: `src/refinement_v2/weight_builder.py`
- Create: `src/refinement_v2/gaussian_adapter.py`
- Create: `src/refinement_v2/losses.py`
- Create: `src/refinement_v2/diagnostics.py`
- Create: `src/refinement_v2/state_io.py`
- Create: `tests/refinement_v2/__init__.py`

**Step 1: 创建目录和空模块**

- 只放最小 import 和中文模块注释.
- 不写业务逻辑.

**Step 2: 写入口脚本最小 stub**

```python
def main() -> None:
    raise NotImplementedError("refine_robust_v2 is not implemented yet")


if __name__ == "__main__":
    main()
```

**Step 3: 跑语法检查**

Run:

```bash
python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py
```

Expected:

- 无输出

**Done when:**

- 所有目标文件存在
- `py_compile` 通过

---

### Task 2: 实现配置层和 CLI 映射

**Files:**
- Modify: `scripts/refine_robust_v2.py`
- Modify: `src/refinement_v2/config.py`
- Create: `tests/refinement_v2/test_config.py`

**Step 1: 先写配置层测试**

测试点:

- CLI 参数能映射到 `RefinementRunConfig`
- 未传值时使用默认超参数
- `--dry-run` / `--resume` / `--enable-stage2b` 等布尔值正确生效

**Step 2: 实现 dataclass**

至少实现:

- `RefinementRunConfig`
- `StageHyperParams`
- `load_effective_config_from_cli()`

建议签名:

```python
def load_effective_config_from_cli(argv: list[str] | None = None) -> tuple[RefinementRunConfig, StageHyperParams]:
    ...
```

**Step 3: 跑单测**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_config.py
```

Expected:

- `test_config.py` 全部通过

**Step 4: 跑入口帮助信息**

Run:

```bash
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py --help
```

Expected:

- 正常打印参数帮助,退出码 0

**Done when:**

- CLI 参数集合和规格一致
- 帮助信息可用

---

### Task 3: 实现 `SceneBundle` 和数据读取标准化

**Files:**
- Modify: `src/refinement_v2/config.py`
- Modify: `src/refinement_v2/data_loader.py`
- Create: `tests/refinement_v2/test_data_loader.py`
- Read: `src/models/data/provider.py`

**Step 1: 先写纯单元测试**

不要一开始就依赖真实大数据.
先用 synthetic tensors 验证:

- `view_id` 解析
- `frame_indices` 过滤
- `target_subsample` 生效
- 输出 shape 标准化

**Step 2: 实现 `SceneBundle`**

建议字段:

```python
@dataclass
class SceneBundle:
    gt_images: torch.Tensor
    cam_view: torch.Tensor
    intrinsics: torch.Tensor
    frame_indices: list[int]
    scene_index: int
    view_id: str | None
```

**Step 3: 实现 `build_scene_bundle(...)`**

要求:

- 不在 runner 里直接操作 provider 原始 batch
- 统一把 shape 转成 refinement 层使用的格式

**Step 4: 跑单测**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_data_loader.py
```

Expected:

- 所有 frame/filter/shape 测试通过

**Done when:**

- runner 可以只接收 `SceneBundle`,不用关心 provider 细节

---

### Task 4: 实现 `GaussianAdapter` 的最小可用版本

**Files:**
- Modify: `src/refinement_v2/gaussian_adapter.py`
- Create: `tests/refinement_v2/test_gaussian_adapter.py`

**Step 1: 先写测试**

覆盖:

- 从最小 synthetic gaussian 参数构建 adapter
- `freeze_for_stage("stage2a")` 后 `means` / `rotation` 不进 optimizer
- `summarize_gaussian_stats()` 返回 `scale_tail_ratio` 等键

**Step 2: 实现最小接口**

至少实现:

```python
class GaussianAdapter:
    @classmethod
    def from_ply(cls, path: Path) -> "GaussianAdapter": ...
    def freeze_for_stage(self, stage_name: str) -> None: ...
    def build_optimizer(self, stage_name: str, hparams: StageHyperParams) -> torch.optim.Optimizer: ...
    def summarize_gaussian_stats(self) -> dict: ...
    def export_ply(self, path: Path) -> None: ...
```

**Step 3: 跑单测**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_gaussian_adapter.py
```

Expected:

- 阶段冻结逻辑测试通过

**Done when:**

- `Stage 2A` 所需参数组能稳定冻结/解冻

---

### Task 5: 实现 `WeightBuilder`

**Files:**
- Modify: `src/refinement_v2/weight_builder.py`
- Create: `tests/refinement_v2/test_weight_builder.py`

**Step 1: 先写测试**

至少覆盖:

- residual 全 0 时权重接近全 1
- 极端高 residual 时权重明显下降但不低于 `weight_floor`
- 加 EMA 后第二次结果更平稳

**Step 2: 实现 V2.0 最小版**

只做:

- RGB residual
- quantile normalize
- `exp(-r_norm / tau)`
- `clamp(weight_floor, 1.0)`
- EMA

不要第一版就加 LPIPS/DINO.

**Step 3: 跑单测**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_weight_builder.py
```

Expected:

- 权重构造测试通过

**Done when:**

- `WeightBuilder` 能稳定输出 soft trust map

---

### Task 6: 实现损失函数层

**Files:**
- Modify: `src/refinement_v2/losses.py`
- Create: `tests/refinement_v2/test_losses.py`

**Step 1: 先写测试**

覆盖:

- `compute_weighted_rgb_loss(...)`
- `compute_scale_tail_loss(...)`
- `compute_opacity_sparse_loss(...)`
- `compute_pose_regularization(...)`

**Step 2: 实现最小损失**

第一版只要求:

- weighted RGB loss
- scale tail regularization
- opacity sparse regularization

pose regularization 先写函数,不一定接主流程.

**Step 3: 跑单测**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_losses.py
```

Expected:

- 损失公式测试通过

**Done when:**

- runner 中不再手写任何损失细节

---

### Task 7: 实现 `DiagnosticsWriter`

**Files:**
- Modify: `src/refinement_v2/diagnostics.py`
- Create: `tests/refinement_v2/test_diagnostics.py`

**Step 1: 先写测试**

覆盖:

- `diagnostics.json` 可落盘
- `metrics_stage*.json` 可按阶段写入
- `weight_map` / `residual_map` PNG 能写出文件

**Step 2: 实现最小 contract**

必须支持:

- `log_stage_metrics(...)`
- `save_weight_map(...)`
- `save_residual_map(...)`
- `finalize(...)`

**Step 3: 跑单测**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_diagnostics.py
```

Expected:

- 所有输出文件测试通过

**Done when:**

- 诊断输出与主训练循环解耦

---

### Task 8: 实现 `StageController`

**Files:**
- Modify: `src/refinement_v2/stage_controller.py`
- Create: `tests/refinement_v2/test_stage_controller.py`

**Step 1: 先写测试**

覆盖:

- `ghosting_acceptable` 时停止
- 未启用 `enable_pose_diagnostic` 时不能进 Phase 3
- 未启用 `enable_joint_fallback` 时不能进 Phase 4
- `global_shift_detected` 时才允许进入 pose-only

**Step 2: 实现门控逻辑**

至少实现:

```python
class StageController:
    def should_stop_stage(self, stage_name: str, metrics_history: list[dict]) -> bool: ...
    def should_enter_stage2b(self, diagnostics: dict) -> bool: ...
    def should_enter_pose_diagnostic(self, diagnostics: dict) -> bool: ...
    def should_enter_joint_fallback(self, diagnostics: dict) -> bool: ...
    def summarize_stop_reason(self, diagnostics: dict) -> str: ...
```

**Step 3: 跑单测**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_stage_controller.py
```

Expected:

- 阶段门控测试通过

**Done when:**

- 下一阶段选择不再散落在 runner 里

---

### Task 9: 实现 `Phase 0` 和 `--dry-run`

**Files:**
- Modify: `src/refinement_v2/runner.py`
- Modify: `scripts/refine_robust_v2.py`
- Read: `src/rendering/gs.py`
- Read: `src/rendering/gs_deferred.py`

**Step 1: 先写集成测试**

Create:

- `tests/refinement_v2/test_runner_phase0.py`

覆盖:

- `--dry-run` 能跑到 baseline 诊断
- 输出 `diagnostics.json`
- 输出 baseline residual / metrics

**Step 2: 实现 `run_phase0_only()`**

要求:

- 只做读取、渲染、baseline 统计、落盘
- 不进行参数更新

**Step 3: 跑集成测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_runner_phase0.py
```

Expected:

- `dry-run` 相关测试通过

**Done when:**

- 用户可以先验证数据契约和输出路径,不必立即跑完整优化

---

### Task 10: 实现 `Phase 1 + Stage 2A`

**Files:**
- Modify: `src/refinement_v2/runner.py`
- Modify: `src/refinement_v2/weight_builder.py`
- Modify: `src/refinement_v2/losses.py`
- Modify: `src/refinement_v2/gaussian_adapter.py`
- Create: `tests/refinement_v2/test_runner_stage2a.py`

**Step 1: 先写集成测试**

覆盖:

- `Stage 2A` 能跑固定迭代数
- 会更新 `opacity/color/scale`
- 不会更新 `means/rotation`
- 会输出 `metrics_stage2a.json`

**Step 2: 实现 `run_stage2a()`**

要求:

- 每 iter 都走 `render -> residual -> weight -> loss -> step`
- 每次进入新阶段重新创建 optimizer
- 支持 `stop_reason`

**Step 3: 跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_runner_stage2a.py
```

Expected:

- `Stage 2A` 行为测试通过

**Step 4: 跑最小语法和导入检查**

Run:

```bash
PYTHONPATH="$(pwd)" python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py
```

Expected:

- 无输出

**Done when:**

- 第一版主价值路径已经可用

---

### Task 11: 实现 `Stage 2B` 受限几何优化

**Files:**
- Modify: `src/refinement_v2/runner.py`
- Modify: `src/refinement_v2/gaussian_adapter.py`
- Create: `tests/refinement_v2/test_runner_stage2b.py`

**Step 1: 先写测试**

覆盖:

- 只有开启 `--enable-stage2b` 才会进入
- `means_delta_cap` 生效
- `lr_means` 明显低于外观参数

**Step 2: 实现 `run_stage2b()`**

要求:

- 只开放受限几何
- 优先支持 along-ray 或受限 `means`
- 默认不开放完整自由 3D 位移

**Step 3: 跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_runner_stage2b.py
```

Expected:

- `Stage 2B` 门控和 cap 测试通过

**Done when:**

- geometry 只在必要时进入,且不会轻易跑飞

---

### Task 12: 实现 `Phase 3`、`Phase 4` 和最小 `state_io`

**Files:**
- Modify: `src/refinement_v2/runner.py`
- Modify: `src/refinement_v2/stage_controller.py`
- Modify: `src/refinement_v2/state_io.py`
- Create: `tests/refinement_v2/test_runner_phase3_phase4.py`
- Create: `tests/refinement_v2/test_state_io.py`

**Step 1: 先写测试**

覆盖:

- 未开启标志时不能进 `Phase 3/4`
- 开启 pose-only 后会保存 `pose_delta_summary`
- `state/latest.pt` 能保存并恢复最小状态

**Step 2: 实现最小状态恢复**

第一版只恢复:

- 当前阶段
- 高斯参数
- 关键指标摘要

不要第一版就恢复 optimizer state.

**Step 3: 实现 `Phase 3` 和 `Phase 4`**

要求:

- `Phase 3` 只做 tiny pose-only diagnostic
- `Phase 4` 只在强证据且显式允许时进入

**Step 4: 跑测试**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2/test_runner_phase3_phase4.py tests/refinement_v2/test_state_io.py
```

Expected:

- pose/joint/state 三类测试通过

**Done when:**

- 完整阶段链闭合
- 恢复能力具备最小可用性

---

### Task 13: 跑最小验证矩阵并同步文档

**Files:**
- Modify: `specs/joint_refinement_camera_gaussians_v2.md`
- Modify: `WORKLOG.md`
- Modify: `notes.md`
- Optional Create: `outputs/refine_v2/...` 运行产物

**Step 1: 验证组 A**

Run on:

- `view "3"`

目标:

- 确认最稳轨迹没有被修坏

**Step 2: 验证组 B**

Run on:

- `view "5"` 或当前最差轨迹

目标:

- 确认 ghosting 有实质改善

**Step 3: 记录结果**

至少记录:

- `psnr`
- `sharpness`
- `scale_tail_ratio`
- 主观 before/after 截图路径
- 停在哪个阶段
- `stop_reason`

**Step 4: 跑整个新增测试集**

Run:

```bash
PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2
```

Expected:

- 全部通过

**Step 5: 跑语法检查**

Run:

```bash
PYTHONPATH="$(pwd)" python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py
```

Expected:

- 无输出

**Done when:**

- 第一版实现已被真实轨迹验证
- 文档与工作记录同步

---

## 执行优先级

如果要压缩实现范围,严格按下面优先级:

1. `Task 1 -> Task 10`
2. `Task 11`
3. `Task 12`
4. `Task 13`

也就是说,**最小可用版本** 只要求:

- CLI 可跑
- `Phase 0`
- `Phase 1`
- `Stage 2A`
- 诊断产物
- 对应测试

`Stage 2B / Phase 3 / Phase 4 / resume` 都可以后置.

## 交付标准

完成第一版 MVP 时,必须满足:

- 能用一条命令启动 refinement
- 能在 `view "3"` 和最差轨迹上输出 before/after 与指标
- `Stage 2A` 参数冻结正确
- `weight_map` 行为稳定
- 诊断文件完整
- `tests/refinement_v2/` 全绿

---

## Implementation Notes After Real Validation

- 真实验证表明, refinement 不能直接把 demo 顶层 YAML 当成完整训练配置使用.
  - 必须先解析并合并 `config_path` 指向的训练配置链.
- 真实 demo 资产与 demo YAML 的 `dataset_name` 可能不一致.
  - 当前实现已补 `--dataset-name`,用于在不改全局 demo YAML 的前提下切换到本机真实存在的数据集.
- refinement 专用 dataloader 不应继承训练链路里的全部 IO 依赖.
  - 当前实现已显式覆盖:
    - `use_depth = False`
    - `load_latents = False`
    - `target_index_manual = None`
- 第一轮真实验证使用了 `3-frame subset`:
  - `outputs/refine_v2/view3_phase0_subset`
  - `outputs/refine_v2/view3_stage2a_subset`
  - `outputs/refine_v2/view5_stage2a_subset`
  - 其中 `Stage 2A` 5 iter 已能在 `view 3` / `view 5` 上稳定降低 `residual_mean`,并提升 `PSNR`.
- 后续实现已补充正式可视化导出:
  - `videos/baseline_render.mp4`
  - `videos/final_render.mp4`
  - `videos/gt_reference.mp4`
  - `renders_before_after/*_frame_0000.png`
- 完整 target 序列验证也已完成:
  - `outputs/refine_v2/view3_stage2a_full`
  - `outputs/refine_v2/view5_stage2a_full`
  - `Stage 2A` 5 iter 在两组全序列上依然稳定降低 `residual_mean`,并提升 `PSNR`.
