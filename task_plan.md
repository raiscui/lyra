# 任务计划: 将 `Long-LRM style post refinement` spec 拆成 implementation tasks

## 目标

把 `specs/long_lrm_style_post_refinement.md` 进一步落成一份可直接执行的 implementation task list。

结束状态应满足:

- 新计划文件已保存到 `docs/plans/`.
- 任务拆分明确区分当前新路线与旧的 `joint_refinement_camera_gaussians_v2` 计划.
- 任务顺序覆盖:
  - CLI / config
  - SceneBundle / reference video 对齐
  - baseline render / diagnostics
  - residual / weight map
  - Stage 3A appearance-first
  - opacity / pruning
  - SR patch supervision
  - Stage 3B limited geometry
  - state / resume / export
  - 集成验证

## 阶段

- [x] 阶段1: 旧上下文回读与规格边界确认
- [x] 阶段2: 四文件续档与持续学习摘要
- [x] 阶段3: 编写 implementation plan 文档
- [x] 阶段4: 校对、留痕与交付

## 关键问题

1. 新计划如何和 `docs/plans/2026-03-06-joint-refinement-camera-gaussians-v2.md` 拉开边界?
2. 如何把 spec 里的 phase / stage 转成可执行文件任务, 而不是重复摘要一遍规格?
3. 哪些任务应该优先体现用户确认过的重点:
   - `Long-LRM` 的 post-prediction optimization
   - opacity / pruning 方法论

## 两个方向

1. 不惜代价, 最佳方案:
   - 把现有 `refinement_v2` 的每个模块都重新按新 spec 映射一遍.
   - 为每个 task 写出测试入口、失败预期、最小实现和验证命令.
   - 明确 V1 非目标, 避免实现时又把 pose / token budget 混回主线.
2. 先能用, 后面再优雅:
   - 只输出一份粗粒度 task 列表.
   - 不区分已有骨架文件和新增文件.
   - 不写测试路径和验证命令.

## 做出的决定

- 决定: 采用方向1.
- 理由: 用户现在要的不是继续讨论, 而是把 spec 变成后续可以直接开工的实现任务.
- 决定: 新计划独立保存为 `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`.
- 理由: 避免和旧 V2 计划互相覆盖, 后面实现时也更容易按路线选择.
- 决定: 先续档 `task_plan.md` 与 `notes.md`, 再写新计划.
- 理由: 两个文件都已超过 1000 行, 继续追加会让后续上下文检索越来越脏.

## 遇到错误

- 暂无代码错误.
- 过程约束: `task_plan.md` 与 `notes.md` 已超过 1000 行, 本轮先执行续档与轻量持续学习摘要.

## 状态

**已完成**: `Long-LRM style post refinement` 的 spec 已正式并入 `SplatSuRe-style selective SR`、`Mip-inspired sampling-aware smoothing` 与 `EDGS deferred idea`, implementation plan 也已同步.

---

## 记录(按时间追加)

### 2026-03-06 06:15 UTC

- 本轮目标: 不实现代码, 只把 `specs/long_lrm_style_post_refinement.md` 拆成 implementation tasks.
- 已确认:
  - 当前用户明确不再考虑 `tttLRM`.
  - 当前主线只沿 `Long-LRM style post refinement` 继续.
  - 旧计划 `docs/plans/2026-03-06-joint-refinement-camera-gaussians-v2.md` 不能直接复用成新计划, 因为范围更泛.
- 已执行:
  - 回读 `task_plan.md`、`WORKLOG.md`、`notes.md`、`LATER_PLANS.md`.
  - 回读 `specs/long_lrm_style_post_refinement.md` 与旧 implementation plan.
  - 检查当前 `refinement_v2` 骨架与测试文件集合.
  - 因 `task_plan.md` / `notes.md` 超过 1000 行, 已先续档.

### 2026-03-06 06:15 UTC

- 已完成:
  - 新增 `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`.
  - 这份文档已把 spec 进一步拆成 implementation tasks, 并明确了执行顺序.
- 本轮计划覆盖的重点:
  1. `Long-LRM` 风格的 post-prediction optimization 主线.
  2. opacity / pruning 单独成任务, 不再只放在原则描述里.
  3. SR patch supervision 与 Stage 3B limited geometry 分开落位.
  4. state / resume / dry-run 验证作为最后收口任务.
- 续档处理:
  - `task_plan.md` 与 `notes.md` 已换新.
  - 旧长文件已转入 `archive/`, 避免后续继续污染根目录上下文.

### 2026-03-06 07:20 UTC

- 用户已确认继续, 本轮不再停留在 explore 对话, 而是直接做文档落盘.
- 本轮目标:
  1. 正式更新 `specs/long_lrm_style_post_refinement.md`.
  2. 同步更新 `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`.
- 本轮要写进主文档的核心结论:
  - `SplatSuRe-style selective SR` 进入主线, 不再把 SR 视为全图统一监督.
  - `Mip-Splatting` 的 `3D smoothing` 不替换当前 `scale_reg`, 而是作为第二条互补约束.
  - `EDGS` 只保留为 deferred idea, 不进入当前 implementation plan.
- 本轮额外约束:
  - 需要同步更新 mermaid flowchart.
  - 更新后要做 mermaid 语法验证, 避免文档内容与图表脱节.

### 2026-03-06 07:28 UTC

- 已完成 spec 正式更新:
  - `specs/long_lrm_style_post_refinement.md`
- 本轮正式写入的内容包括:
  1. `SplatSuRe-style selective SR`
  2. `gaussian_fidelity_score`
  3. `W_sr_select`
  4. `W_final_sr = W_robust * W_sr_select`
  5. `Mip-inspired sampling-aware smoothing`
  6. `EDGS-style local reinitialization` 作为 deferred idea
- 已完成 plan 同步:
  - `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`
- 已完成文档验证:
  - `beautiful-mermaid-rs --validate-markdown < specs/long_lrm_style_post_refinement.md` 返回 `true`
  - 更新后的 flowchart 已成功渲染成 Unicode 图

### 2026-03-06 11:00 UTC

- 用户选择继续走选项 `1`.
- 本轮目标切换为:
  1. 做一轮 `spec vs current code` 的 gap review.
  2. 明确当前 `Long-LRM style post refinement` 新规格里,哪些能力已经有代码骨架,哪些仍未落地.
  3. 给出下一步最顺手的实现切入点,避免重复实现旧 `refinement_v2` 已有能力.
- 本轮边界:
  - 仍处于 explore / 规格校准阶段.
  - 不直接写实现代码.
  - 允许更新四文件与规划文档,用于沉淀 gap review 结论.
- 下一步动作:
  - 回读 `specs/long_lrm_style_post_refinement.md`.
  - 回读 `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`.
  - 对照 `src/refinement_v2/` 与 `scripts/refine_robust_v2.py` 做模块级映射.
  - 把结论追加到 `notes.md` 与 `task_plan.md`.

### 2026-03-06 11:18 UTC

- `spec vs current code` gap review 已完成.
- 核心判断已经明确:
  1. `refinement_v2` 当前不是空骨架,而是已经具备大部分主线能力.
  2. 旧命名里的 `phase1/stage2a/stage2b` 与新 spec 的 `W_robust/Stage 3A/Stage 3B` 存在明显一一映射关系.
  3. 当前真正缺的不是 `opacity/pruning` 或 `limited geometry`,而是:
     - 真实 external reference 输入链
     - `gaussian_fidelity_score`
     - `W_sr_select`
     - `W_final_sr`
     - `L_sampling_smooth`
     - `Phase 3S / Stage 3SR` 的显式阶段拆分
- 已确认的重要代码事实:
  - `config.py` 目前仍没有 `--pose/--intrinsics/--reference-video`.
  - `data_loader.py` 的 `super_resolved` 目前只是对 native GT 做双线性上采样.
  - `runner.py` 已有 patch supervision,但 patch 权重仍是全 1,因此还不是 selective SR.
  - `stage2b` 已有主体实现,但 gating 还没接到“Stage 3SR 跑完以后再决定”的新语义.
- 本轮产出:
  - 详细结论已追加到 `notes.md`.
- 建议后续真正实现顺序:
  1. 先补 direct file inputs + external reference data contract.
  2. 再拆出 `Phase 3S / Stage 3SR`.
  3. 再补 fidelity / selection / `L_sampling_smooth`.

---

## 2026-03-06 06:46 UTC: 继续落地 `joint_refinement_camera_gaussians_v2`

### 当前目标

- 延续已完成的 `Task 1-8` 基础骨架,继续把 `runner/state_io/CLI` 主线跑通.
- 在实现过程中持续执行语法检查、单测、轻集成测试.
- 如果测试暴露设计问题,就及时调整任务拆分和实现方案,而不是硬顶.

### 当前判断

- 当前最关键风险不在基础模块,而在刚补上的 `runner.py` / `state_io.py` / `scripts/refine_robust_v2.py` 及其测试是否真的能跑通.
- 需要先验证这些新增主线文件的实际状态,再决定是否继续推进后续 task.

### 下一步动作

1. 读取并核对 `runner/state` 相关文件当前内容.
2. 先跑 `py_compile` 做语法闸门.
3. 再跑新增 `runner/state` 测试.
4. 失败就先修结构问题,成功再扩大到 `tests/refinement_v2` 全量验证.

### 状态

**目前在执行验证闸门**: 先检查 `runner/state` 主线 patch 是否完整可运行, 然后进入修复与补齐阶段.

### 2026-03-06 06:49 UTC

- 已核对 `src/refinement_v2/runner.py`、`state_io.py`、`gaussian_adapter.py`、`diagnostics.py`、`scripts/refine_robust_v2.py` 与新增测试文件.
- 目前未见文件缺失或明显截断.
- 下一步进入语法检查闸门: 先跑 `py_compile`, 再跑新增 runner/state 测试.

### 2026-03-06 06:52 UTC

- `runner/state` 定向测试已执行, 当前失败 3 个.
- 根因初判: `stage2a/stage2b` 循环里 `prev_weight_map` 通过 EMA 复用时保留了上一轮 autograd graph, 导致第二轮 `backward()` 触发 `Trying to backward through the graph a second time`.
- 修复方向: 让权重图构造显式基于 detached residual, 并保证返回的 `weight_map` 本身不携带梯度图.

### 2026-03-06 06:58 UTC

- `tests/refinement_v2` 已全绿, 共 `23 passed`.
- 进入 Task 13 真实验证前, 发现当前 shell 的 `python3` 缺少 `omegaconf` / `plyfile`, 不适合执行真实 dataloader + PLY 读写链路.
- 下一步改用仓库 `.pixi` 环境做 dry-run / 实机轻验证.

### 2026-03-06 07:05 UTC

- 真实 dry-run 暴露配置层缺陷: `build_scene_bundle()` 之前只读取 demo 顶层 YAML,没有按 `config_path` 合并训练配置链.
- 已按 `sample.py` 的契约修正 `src/refinement_v2/data_loader.py`:
  - 支持解析并合并嵌套 `config_path`.
  - 支持把 `dataset_name` / `static_view_indices_fixed` / `set_manual_time_idx` / `target_index_subsample` 映射到 dataloader 配置.
  - 支持 `--view-id` 覆盖 demo 里的固定轨迹,便于直接验证 view `3/5`.
- 同时补了纯单元回归测试,锁住这类 demo config 加载问题.

### 2026-03-06 07:10 UTC

- 真实验证继续暴露本地资产差异: `configs/demo/lyra_static.yaml` 当前默认 `dataset_name=lyra_static_demo_generated_one`, 但本机现有资产目录是 `lyra_static_demo_generated`.
- 方案调整: 不直接硬改全局 demo YAML, 而是给 refinement CLI 增加 `--dataset-name` 覆盖能力.
- 这样 refinement 可在不破坏原有 demo 配置语义的前提下, 对接当前机器真实存在的数据资产.

### 2026-03-06 07:16 UTC

- 真实验证第三个阻塞点确认: provider 默认还会尝试读取 `depth` 与 `latents`, 而当前 refinement 实际只需要 `RGB + cam_view + intrinsics`.
- 已将 refinement 专用 dataloader 覆盖项进一步收紧:
  - `use_depth = False`
  - `load_latents = False`
- 这样可以避免 demo 资产缺少深度包时在 provider 内部无限重试,也减少不必要的 IO.

### 2026-03-06 07:28 UTC

- 已完成 `joint_refinement_camera_gaussians_v2` 当前任务清单的代码落地与验证收口.
- 最终验证结果:
  - `python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py tests/refinement_v2/*.py` 通过.
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2` 通过, 共 `25 passed`.
  - 真实验证已完成:
    - `outputs/refine_v2/view3_phase0_subset`
    - `outputs/refine_v2/view3_stage2a_subset`
    - `outputs/refine_v2/view5_stage2a_subset`
- 真实 subset 验证指标摘要:
  - view `3`: `PSNR 19.2813 -> 21.9934`, `residual_mean 0.066586 -> 0.043972`.
  - view `5`: `PSNR 19.0454 -> 24.2871`, `residual_mean 0.068201 -> 0.034461`.
- 本轮任务状态: **已完成**.

## 2026-03-06 07:34 UTC: 继续推进 `refinement_v2` 的可视化导出与全序列验证

### 当前目标

- 延续上一轮已完成的 subset 验证,继续落地两个未完成步骤:
  1. 导出 before/after 渲染结果,便于肉眼核对 ghosting 是否真的下降.
  2. 跑完整 target 序列的 `view 3` / `view 5` 真实验证.

### 当前判断

- 现在主线可运行性已经成立.
- 下一步最重要的不是继续空谈指标,而是把渲染导出补成正式产物,并把 subset 验证扩到完整 target 序列.

### 下一步动作

1. 检查仓库已有的视频导出工具与 `DiagnosticsWriter` 当前能力.
2. 实现 before/after 渲染帧与视频导出.
3. 补对应测试.
4. 回归通过后跑 `view 3` / `view 5` 全序列真实验证.

### 2026-03-06 07:42 UTC

- 已为 `refinement_v2` 补上 before/after 渲染导出能力:
  - `videos/baseline_render.mp4`
  - `videos/final_render.mp4`
  - `videos/gt_reference.mp4`
  - 对应首帧 PNG 快照
- 实现层面增加了双通道视频写出:
  - 优先 `imageio`
  - 缺失时回退系统 `ffmpeg`
  - `ffmpeg` 再按 `libx264 -> mpeg4` 顺序尝试编码器
- 已补测试锁住这条新契约.

### 2026-03-06 07:48 UTC

- 已完成 before/after 渲染导出落地:
  - `videos/baseline_render.mp4`
  - `videos/final_render.mp4`
  - `videos/gt_reference.mp4`
  - `renders_before_after/*_frame_0000.png`
- 已完成全量回归:
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2`
  - `26 passed in 4.28s`
- 已完成完整 target 序列真实验证:
  - `outputs/refine_v2/view3_stage2a_full`
  - `outputs/refine_v2/view5_stage2a_full`
- 全序列关键结果:
  - view `3`: `PSNR 19.6426 -> 22.9245`, `residual_mean 0.063627 -> 0.039699`.
  - view `5`: `PSNR 18.7739 -> 21.7612`, `residual_mean 0.072603 -> 0.047303`.
- 当前状态: **本轮继续任务已完成**.

## 2026-03-06 08:00 UTC: 继续推进 `refinement_v2` 的 opacity/pruning

### 当前目标

- 不再继续单纯拉长 `Stage 2A` 迭代数.
- 直接进入下一步质量强化,落地 `opacity/pruning`.
- 目标是让当前 refinement 不只会降 residual,还要开始主动清理低价值高斯,减少重影和雾状叠层.

### 当前判断

- 现有 `Stage 2A` 已证明主线有效,但 sharpness 改善不稳定.
- 这说明现在主要在做 appearance alignment,还没有真正处理“低置信度高斯堆积”这一层问题.
- 因此优先落地 `opacity/pruning`,比先盲目把 iter 拉高更值.

### 下一步动作

1. 回读 spec / plan 里关于 opacity/pruning 的约束.
2. 设计最小可落地版本:
   - 配置项
   - 高斯适配器内的 prune 逻辑
   - runner 中的触发时机与统计
3. 补单测和轻集成测试.
4. 回归通过后做一轮真实验证.

### 2026-03-06 08:09 UTC

- 已重新回读 `task_plan.md`、`notes.md`、`WORKLOG.md`、`LATER_PLANS.md` 与当前 `refinement_v2` 代码入口.
- 当前进入 `opacity/pruning` 的正式实现阶段.
- 执行策略改为:
  1. 先补 pruning 契约测试.
  2. 再补配置、控制器、高斯适配器、诊断输出与 runner 接线.
  3. 先跑定向回归,再跑 `tests/refinement_v2` 全量,最后做真实验证.
- 这一步的目标不是盲目增加迭代,而是验证 pruning 能否真正减少低价值高斯并改善重影.

### 状态

**目前在补 pruning 测试矩阵**: 先用测试锁住 warmup、频率、比例上限、诊断输出和 runner 触发时机, 再进入代码实现.

### 2026-03-06 08:16 UTC

- pruning 定向测试已按预期失败,共 8 个失败.
- 缺口已确认:
  1. `RefinementRunConfig` / CLI 还没有 pruning 开关与参数.
  2. `StageController` 还没有 `should_prune_now()`.
  3. `GaussianAdapter` 还没有候选收集与实际裁剪接口.
  4. `DiagnosticsWriter` 还没有 `write_prune_summary()`.
  5. `runner.run_stage2a()` 还没有 pruning 触发与 optimizer 重建.
- 下一步直接进入代码落地,优先完成配置、控制器、高斯适配器与 diagnostics,最后接 runner.

### 2026-03-06 08:23 UTC

- `python3 -m py_compile` 已通过.
- `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2` 已通过, 当前结果 `34 passed`.
- pruning 的配置层、控制层、高斯裁剪、diagnostics 输出与 runner 接线已经完成.
- 下一步进入真实验证,优先复用已经验证过的 `view 5` 全序列资产.
- 本轮真实验证重点不再看“能不能跑起来”,而是看:
  1. `num_gaussians` 是否下降.
  2. `opacity_lowconf_ratio` 是否下降.
  3. `PSNR / residual_mean` 是否继续保持在合理方向.
  4. `final_render` 相比 baseline 是否更干净.

### 2026-03-06 08:33 UTC

- `opacity/pruning` 代码落地已完成.
- 定向测试与全量 `tests/refinement_v2` 回归均已通过,当前结果 `34 passed`.
- 两轮真实验证已完成:
  - `outputs/refine_v2/view5_stage2a_prune_full`
  - `outputs/refine_v2/view3_stage2a_prune_full`
- 当前真实结论:
  - pruning 能稳定降低 `opacity_lowconf_ratio` 和 `num_gaussians`.
  - `PSNR / residual_mean / sharpness` 也都出现小幅正向改善.
  - 说明这一步不是“只会删点”,而是对当前重影问题确实有帮助.

### 状态

**目前已完成本轮 pruning 任务**: 代码、测试、真实验证、记录文件都已收口. 下一步默认进入 `patch-based supervision` 设计与实现.

### 2026-03-06 08:39 UTC

- 对 pruning diagnostics 做了最后一轮收口优化:
  - 不再保存完整 `pruned_indices`
  - 改为保存 `pruned_indices_preview`
- 已重新跑 `view 5` / `view 3` 真实验证,确认指标未偏移,同时 `pruning_summary.json` 已变成轻量文件.

### 状态

**目前已完成本轮 pruning 任务**: 代码、测试、真实验证与诊断输出体积优化都已完成.

## 2026-03-06 08:46 UTC: 继续推进 `patch-based supervision`

### 当前目标

- 接上 pruning 之后的下一个未完成质量项.
- 先做最小可落地的 patch-based supervision.
- 目标不是一次性做完整 SR 管线,而是先让 `stage2a` 支持基于 patch 的高分辨率/局部监督路径.

### 当前判断

- 现有 `refinement_v2` 仍然是全帧渲染 + 全帧监督.
- 这会把监督分辨率锁死在 native frame 上.
- 如果我们后面要接超分对等视频或更高分辨率参考图,必须先把 patch 路径打通.

### 下一步动作

1. 从 `Task 7` 规格提炼最小 contract.
2. 先补 patch supervision 测试.
3. 再实现 patch 采样、参考 patch 提取、patch loss 与 runner 接线.
4. 先跑 `tests/refinement_v2` 回归,再决定是否做真实资产验证.

### 状态

**目前在设计 patch supervision 契约**: 先把 patch window、reference 映射和 intrinsics 语义固定下来, 再进入测试驱动实现.

### 2026-03-06 09:00 UTC

- `patch-based supervision` 的第一版代码已经落地.
- 当前已完成:
  - `reference_mode/sr_scale/patch_size` 配置映射
  - `SceneBundle.reference_images/intrinsics_ref` 扩展
  - patch window 采样、reference patch 提取、patch intrinsics 偏移
  - `stage2a` 内的 patch loss 接线
- 定向测试结果:
  - `tests/refinement_v2/test_patch_supervision.py`
  - `tests/refinement_v2/test_data_loader.py`
  - `tests/refinement_v2/test_config.py`
  - 当前合计 `14 passed`
- 下一步进入全量回归与真实验证.

### 状态

**目前在做 patch supervision 回归验证**: 先确认不会破坏现有 pruning / stage2a 主线, 再决定真实验证参数.

### 2026-03-06 09:03 UTC

- 全量回归已通过:
  - `python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py tests/refinement_v2/*.py`
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2`
  - 当前结果 `41 passed`
- 下一步开始第一轮真实 patch 验证.
- 选择先跑 `view 5` 全序列,并与当前最强基线 `view5_stage2a_prune_full` 对比.
- 本轮参数选择:
  - 保持 `iters_stage2a = 5`
  - 保持 pruning 打开
  - 新增 `patch_size = 256`
  - `lambda_patch_rgb = 0.5`
  - `reference_mode = native`
- 目的:
  - 先验证 patch path 在真实资产上稳定可跑.
  - 再看它是否能继续压 residual 和改善清晰度.

### 2026-03-06 09:12 UTC

- `patch-based supervision` 第一版已经完成:
  - 配置、数据契约、patch 采样、patch intrinsics、runner 接线、测试都已落地.
- 全量回归通过:
  - `python3 -m py_compile ...`
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2`
  - 当前结果 `41 passed`
- 真实验证已完成:
  - `outputs/refine_v2/view5_stage2a_prune_patch_full`
  - `outputs/refine_v2/view5_stage2a_prune_patch_full_l025`
  - `outputs/refine_v2/view3_stage2a_prune_patch_full`
- 当前真实结论:
  - patch path 稳定可跑.
  - 它能继续小幅提升 `PSNR` 并压局部误差.
  - 对困难轨迹, `lambda_patch_rgb=0.25` 比 `0.5` 更稳.

### 状态

**目前已完成本轮 patch supervision 任务**: 下一步默认进入“真实外部 SR 参考图接入”或“继续推进 Stage 2B”二选一的后续阶段.

## 2026-03-06 继续 Stage 2B limited geometry
- 当前目标: 在 `refinement_v2` 中完成 Stage 2B 的可落地实现,先补测试约束,再接入 `means anchor` 与 `rotation regularization`,最后跑回归验证。
- 原因: Stage 2A 已经稳定,但 Stage 2B 还只是占位实现,没有把规格里的受限几何优化真正落地。
- 本轮先做: 1) 回读四文件刷新上下文; 2) 补充本轮行动记录到 `task_plan.md`; 3) 读取 Stage 2B 相关实现与测试; 4) 实现并验证。
- 当前状态: 准备进入测试先行阶段。

## 2026-03-06 准备补测试
- 下一步开始把 Stage 2B contract 落成测试,重点断言: 进入门控、loss 字段、means cap、rotation 归一化。

### 2026-03-06 跑 Stage 2B 定向测试
- 预期: 新加的 Stage 2B contract 测试会先失败,失败点会直接指导实现。

### 2026-03-06 定向测试失败总结
- 已确认 5 个缺口: 1) `should_enter_stage2b()` 过宽; 2) `StageHyperParams` 缺 `lambda_means_anchor`; 3) 缺 `lambda_rotation_reg`; 4) CLI 未暴露这两个参数; 5) `run_stage2b()` 未输出几何正则 loss。
- 下一步进入实现阶段,优先补配置与损失,再改 runner 与门控。

### 2026-03-06 重跑 Stage 2B 定向测试
- 目标: 验证门控、配置映射、geometry regularizer 和 metrics 落盘都已经补齐。

### 2026-03-06 定向测试通过,进入全量回归
- 已通过: `test_stage_controller / test_runner_stage2b / test_config` 共 16 项。
- 现在继续跑 `py_compile + tests/refinement_v2` 全量回归,确认没有破坏 Stage 2A / pruning / patch 主线。

### 2026-03-06 准备真实验证
- 目标: 基于当前最稳的 `patch + prune` 起点,补一轮真实 `Stage 2B` 验证。
- 优先选 `view 5`,因为它更难,也更能说明 limited geometry 是否真的有价值。

### 2026-03-06 开始真实 Stage 2B 验证
- 运行对象: `view 5` 全序列
- 起点参数: `prune + patch_size=256 + lambda_patch_rgb=0.25`
- 新增: `--enable-stage2b`, `--iters-stage2b 5`
- 输出目录: `outputs/refine_v2/view5_stage2b_prune_patch_full_l025`

### 2026-03-06 真实验证首轮失败
- 阻塞点不是 `Stage 2B` 逻辑,而是环境缺少 `flash_attn`.
- 先查是否存在可直接复用的 `pixi` / conda 运行环境,优先不改代码。

### 2026-03-06 尝试 pixi 环境
- 仓库推荐通过 `pixi` 运行,因此先验证 `pixi run python` 是否自带 `flash_attn`。

### 2026-03-06 使用 pixi 环境重跑真实验证
- 已清理前一轮因环境缺依赖留下的半成品目录。
- 本轮正式使用 `pixi run python` 执行真实 Stage 2B。

### 2026-03-06 修正 pixi 运行参数
- 上一轮失败原因是 `pixi` 子进程没有继承仓库 `PYTHONPATH`.
- 已补 `PYTHONPATH=$(pwd)` 后重跑同一条真实验证。

### 2026-03-06 真实 Stage 2B 跑完,开始读结果
- 重点确认: 是否进入 `stage2b`,以及相对 `view5_stage2a_prune_patch_full_l025` 的指标变化。

### 2026-03-06 根据真实 run 调整 Stage 2B 门控
- 原因: `view 5` 在 `residual_mean≈0.047` 时仍有明显局部重影,`0.05` 阈值过保守。
- 调整: `local_overlap_persistent` 阈值下调到 `0.045`。

### 2026-03-06 真实验证改为从已验证 Stage 2A 基线续跑
- 新输入高斯: `outputs/refine_v2/view5_stage2a_prune_patch_full_l025/gaussians/gaussians_stage2a.ply`
- 目的: 更直接评估 `Stage 2B` 相对当前最佳 Stage 2A 基线的增益。

### 2026-03-06 第二轮真实验证已跑完
- 现在读取 `view5_stage2b_from_stage2a_patch_l025` 的 diagnostics 与 `metrics_stage2b.json`。

### 2026-03-06 10:37 UTC

- `Stage 2B limited geometry` 已完成代码、测试和真实验证闭环.
- 本轮新增了 `means anchor` / `rotation regularization` / 更保守门控 / Stage 2B metrics 落盘.
- 全量回归结果:
  - `python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py tests/refinement_v2/*.py`
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2`
  - 当前结果 `47 passed`
- 真实验证最终采用“从已验证 Stage 2A 基线续跑 Stage 2B”的方式:
  - `outputs/refine_v2/view5_stage2b_from_stage2a_patch_l025`
- 关键真实结论:
  - `Stage 2B` 在 `view 5` 上有真实正收益.
  - 原始 `0.05` 进入阈值过保守,已按真实证据调整为 `0.045`.

### 状态

**目前已完成本轮 Stage 2B 任务**: 代码实现、定向测试、全量回归、真实验证和记录文件都已收口.

## 2026-03-06 继续任务: 显式化 Stage 2B 续跑 workflow
- 当前目标: 把“从已验证 Stage 2A 基线续跑 Stage 2B”收敛成明确的 CLI/workflow,避免每次手工拼 `gaussians_stage2a.ply` 和阶段跳转细节。
- 原因: 真实验证已经证明 Stage 2B 在这类 warm-start 路径上有价值,下一步应该把这条路径产品化。
- 本轮先做: 1) 回读四文件; 2) 设计 `start_stage` 语义; 3) 补测试; 4) 落地并回归。
- 当前状态: 准备分析当前脚本与 runner 的起始阶段控制点。

### 2026-03-06 开始验证 start_stage workflow
- 先跑 `test_config` 和 `test_runner_stage2b`,因为这两组最直接覆盖 CLI 映射与阶段跳转。

### 2026-03-06 start_stage 定向测试通过,进入全量回归
- 当前结果: `9 passed`。
- 下一步: `py_compile + tests/refinement_v2` 全量回归,然后用新的 `--start-stage stage2b` 做真实验证。

### 2026-03-06 使用新 CLI 做真实验证
- 命令改为显式 `--start-stage stage2b`,不再依赖手工多跑 1 轮 Stage 2A。
- 输出目录: `outputs/refine_v2/view5_stage2b_startstage_cli_l025`

### 2026-03-06 11:18 UTC: `Long-LRM style post refinement` gap review 收口

- 本轮 explore 结论已经补记完成,并正式落到 `notes.md` 与 `WORKLOG.md`。
- 当前最新判断:
  1. `refinement_v2` 已具备 `W_robust` / Stage 3A 等价主线 / opacity-pruning / Stage 3B 等价实现。
  2. 真正缺口集中在 external reference 输入链、`gaussian_fidelity_score`、`W_sr_select`、`W_final_sr`、`L_sampling_smooth`、以及 `Phase 3S / Stage 3SR` 的显式拆分。
  3. 后续如果转入实现,不该再重复补 `opacity/pruning` 主体,而该直接攻这几个缺口。
- 当前状态:
  - 这轮 gap review 已完成。
  - 下一步最合理动作是切到实现,先补 direct file inputs + external SR/reference 接入。

### 2026-03-06 新 CLI 真实验证完成
- 开始读取 `view5_stage2b_startstage_cli_l025` 并与旧的手工 warm-start 结果对比。

### 2026-03-06 11:18 UTC: gap review 最终尾记

- `Long-LRM style post refinement` 的本轮 gap review 已完成并收口。
- 对当前任务最关键的结论保持不变:
  1. `refinement_v2` 已经具备 `W_robust`、Stage 3A 等价主线、opacity/pruning、Stage 3B 等价实现。
  2. 真正未落地的是 external reference 输入链、`gaussian_fidelity_score`、`W_sr_select`、`W_final_sr`、`L_sampling_smooth`、`Phase 3S / Stage 3SR`。
  3. 后续若切回实现,优先从 external reference contract 开始,而不是重复补旧主线。

### 2026-03-06 11:32 UTC: 边界复核,避免和 `joint_refinement_camera_gaussians_v2` 串线

- 已按用户提醒重新核对边界。
- 现在明确区分两层语义:
  1. `specs/long_lrm_style_post_refinement.md` 与 `docs/plans/2026-03-06-long-lrm-style-post-refinement.md`
     - 是独立的规格与实现计划。
     - 当前讨论对象就是它们。
  2. `docs/plans/2026-03-06-joint-refinement-camera-gaussians-v2.md`
     - 是另一条更泛化的旧主线。
     - 即使另一个进程正在实现它,也不能把那边的实现进展直接当成这份 `Long-LRM` 计划已经完成。
- 修正后的解释口径:
  - 我前面提到的“已有主体能力”,只能解释为“共享代码库里可能已经出现可复用实现”。
  - 这不等于:
    - `Long-LRM` spec 已归档
    - `Long-LRM` plan 的对应 task 已正式完成
    - 或当前任务应转去跟随 `joint_v2` 的实现节奏
- 后续继续沿这条线时,我会只把 `Long-LRM` 文档当主参照,把 `joint_v2` 仅当背景或可复用代码来源。

### 2026-03-06 11:40 UTC: 基于最新代码现实重排 `Long-LRM` 后续路线

- 用户要求: 根据现在的代码情况,重新分析 `Long-LRM style post refinement` 这条线接下来应该怎么走。
- 本轮目标:
  1. 回读最新 `Long-LRM` spec / implementation plan。
  2. 核对共享代码里最新已经落下来的 external reference 能力。
  3. 重新给出只属于 `Long-LRM` 这条线的下一步顺序,避免继续混入 `joint_v2` 的一般性任务。
- 本轮边界:
  - 这是分析与路线校准,不是直接实现。
  - 允许更新四文件沉淀结论。
  - 输出里要明确区分:
    - 已有共享基础设施
    - `Long-LRM` 线独有未完成项

### 2026-03-06 11:45 UTC

- 已完成基于最新共享代码现实的复核。
- 核心更新:
  1. external reference contract 已不再是这条线的主要阻塞。
  2. `Long-LRM` 线现在的真正 critical path 已切换成:
     - 阶段拆分
     - renderer info 上抬
     - `gaussian_fidelity_score`
     - `W_sr_select`
     - `W_final_sr`
     - `L_sampling_smooth`
  3. Stage 3B 当前不该抢先推进,因为它依赖前面的 selective SR 阶段语义先成型。
- 结论已经追加到 `notes.md`。
- 如果下一步转入实现,建议直接从“显式拆出 Phase 3S / Stage 3SR”开始。

### 2026-03-06 11:52 UTC: 记录 `SplatSuRe + Mip-Splatting` 的方案约束并复核文档覆盖度

- 用户要求先把以下判断正式记录后再继续讨论:
  1. `SplatSuRe` 值得直接吸收:
     - per-Gaussian fidelity score
     - per-view SR weight map
     - LR + SR selective joint objective
  2. `SplatSuRe` 的合理顺序:
     - 先 Stage 3A 清 opacity / color
     - 再上 selective SR
  3. `Mip-Splatting` 值得参考,但不是当前主线:
     - 重点借 3D smoothing / sampling-frequency constraint
     - 用于更稳的 HR supervision / alias 控制
     - 不应被当作解决厚表面 / 双轮廓 / SR 假细节一致性的主方法
  4. 需要复核当前 spec / plan 是否已经完整吸收这些判断。
- 本轮动作:
  - 先把上述判断追加到 `notes.md`.
  - 再对照 `specs/long_lrm_style_post_refinement.md` 与 `docs/plans/2026-03-06-long-lrm-style-post-refinement.md` 做覆盖度核查。

### 2026-03-06 11:02 UTC

- 已完成 `Stage 2B` 显式 workflow 收口:
  - CLI 新增 `--start-stage {stage2a,stage2b}`
  - 新增 `bootstrap_stage2b_from_current_gaussians()`
  - `diagnostics.json` 新增 `start_stage` 与 `warm_start_stage2b`
- 回归结果:
  - `python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py tests/refinement_v2/*.py`
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2`
  - 当前结果 `48 passed`
- 真实验证已完成:
  - `outputs/refine_v2/view5_stage2b_startstage_cli_l025`
- 当前结论:
  - `--start-stage stage2b` 已可替代旧的手工 warm-start 绕法.
  - 旧手工路线略强,更像是因为它额外多跑了 1 轮 `Stage 2A`,不是新 CLI workflow 有问题.

### 状态

**目前已完成本轮 Stage 2B workflow 收口任务**: 代码、测试、规格同步、真实验证都已完成.

## 2026-03-06 继续推进 Stage 2B workflow 化
- 当前目标: 把“从已验证 Stage 2A 基线续跑 Stage 2B”做成显式 CLI/workflow,减少手工拼参数和阶段绕行。
- 原因: 这条路径已经在真实 `view 5` 上证明有效,现在需要把它工程化,方便后续稳定复用。
- 本轮计划: 1) 回读四文件刷新上下文; 2) 读取当前 runner/config/state 设计; 3) 先补测试约束; 4) 实现并回归; 5) 做一次真实验证。
- 当前状态: 准备进入 workflow 设计与测试阶段。

## 2026-03-06 修正推进方向
- 用户指出得对: `Stage 2B workflow/start_stage` 这一步已经完成。
- 本轮不再重复这条已完成分支。
- 立即切到下一条真正未完成主线: external reference / 外部 SR 视频输入契约。
- 目标: 让 `reference_images` 不再只是 native GT 上采样,而是可以显式载入外部处理后的视频/帧序列。
- 当前状态: 准备读取 `config.py`、`data_loader.py`、patch supervision 相关实现与规格,先收敛最小可落地 contract。

### 2026-03-06 跑 external reference 定向测试
- 目标: 先验证 CLI 映射、外部帧目录对齐、外部 intrinsics 覆盖三件事。

### 2026-03-06 external reference 定向测试通过,进入全量回归
- 当前结果: `13 passed`。
- 下一步: `py_compile + tests/refinement_v2` 全量回归,再做一次真实外部 reference 验证。

### 2026-03-06 准备真实 external reference 验证
- 验证路径: 现有 `assets/demo/static/diffusion_output_generated/5/rgb` 生成一个 2x 外部 reference 帧目录,然后用 `reference_path` 跑真实 refinement。

### 2026-03-06 开始真实 external reference refinement 验证
- 命令目标: 用 `--reference-path outputs/refine_v2/reference_view5_rgb_2x.mp4` 跑一轮最小 Stage 2A。
- 目的: 确认 patch supervision 已能真实消费外部 2x reference,而不是只在 `build_scene_bundle()` 里读到。

### 2026-03-06 external reference 真实验证已跑完
- 现在确认 `diagnostics.json` 与 `metrics_stage2a.json` 中是否出现 patch loss,以及 run 是否正常停在 `stage2a`。

### 2026-03-06 11:36 UTC

- external reference 输入契约已完成代码、测试与真实验证闭环.
- 本轮新增:
  - `--reference-path`
  - `--reference-intrinsics-path`
  - external 帧目录 / mp4 输入
  - external intrinsics override
  - 基于外部分辨率的 `sr_scale/intrinsics_ref` 自动推断
- 回归结果:
  - `python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py tests/refinement_v2/*.py`
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2`
  - 当前结果 `51 passed`
- 真实验证已完成:
  1. `build_scene_bundle()` 成功载入 external 2x mp4
  2. `outputs/refine_v2/view5_external_reference_2x_subset` 成功跑通最小 Stage 2A,并写出 patch loss

### 状态

**目前已完成本轮 external reference contract 任务**: 下一步应继续推进 full direct file inputs,或者继续做 selective SR 的 fidelity / selection / smoothing 三件套.

## 2026-03-06 更新 joint refinement 计划文档为最终完成版
- 当前目标: 把 `docs/plans/2026-03-06-joint-refinement-camera-gaussians-v2.md` 从施工期计划文档更新为最终完成版。
- 原因: 代码实现已经明显超出文档当前记录,如果不收口,后续会误判哪些任务还没做。
- 本轮会做: 1) 回读计划文档与最新 WORKLOG/notes; 2) 归纳已完成项与后续增量; 3) 直接更新计划文档; 4) 做一次文档一致性检查。
- 当前状态: 准备进入文档同步阶段。

### 2026-03-06 11:44 UTC
- 已将 `docs/plans/2026-03-06-joint-refinement-camera-gaussians-v2.md` 更新为最终完成版。
- 当前文档已明确: 原始 `Task 1 ~ Task 13` 全部完成,并补入 `start-stage stage2b` 与 external reference 两个后续增量。

### 状态

**目前已完成本轮计划文档同步任务**: 文档收口与一致性检查已完成。

## 2026-03-06 继续同步主规格到当前实现状态
- 当前目标: 更新 `specs/joint_refinement_camera_gaussians_v2.md`,让它与已经完成的代码、计划文档和真实验证结果保持一致。
- 原因: 计划文档已更新为最终完成版,主规格如果不跟进,后续会再次出现“代码已做完,规格还像未完成”的错位。
- 本轮会做: 1) 回读主规格相关段落; 2) 归纳需要同步的状态更新; 3) 修改规格; 4) 做一致性检查。
- 当前状态: 准备进入规格同步阶段。

### 2026-03-06 11:58 UTC
- 已将 `specs/joint_refinement_camera_gaussians_v2.md` 同步到当前实现状态。
- 当前规格已明确反映: `Task 1 ~ 13` 对应能力已完成,并补入 `--start-stage stage2b` 与 external reference contract。

### 状态

**目前已完成本轮主规格同步任务**: 主规格、计划文档与代码现实已重新对齐。

## 2026-03-06 12:10 UTC: 说明 `joint_refinement_camera_gaussians_v2` 的实际使用方式
- 当前目标: 回答用户“只跑 README Example 1 的两条命令是否足够使用 v2 增强部分”。
- 原因: 用户现在需要的是可直接执行的使用说明,而不是继续看规格文件。
- 本轮会做:
  1. 对照 `README.md` 的 Example 1 与当前 v2 refinement 入口。
  2. 提取最小可运行命令链,区分 baseline 与 enhanced pipeline。
  3. 明确 external reference / stage2b 等增强功能在什么时候需要额外参数。
- 当前状态: 正在读取 README、规格和脚本入口,准备整理成可执行说明。

### 2026-03-06 12:18 UTC
- 已完成 README Example 1 与 `joint_refinement_camera_gaussians_v2` 使用链路核对。
- 最终结论:
  1. README 两条命令只能得到 baseline 结果,不能自动触发 v2 refinement。
  2. 使用增强部分时,应在 `sample.py` 导出 `gaussians_orig/gaussians_0.ply` 后,再单独运行 `scripts/refine_robust_v2.py`。
  3. 当前仓库 `configs/demo/lyra_static.yaml` 已被本地调整,建议通过 CLI 显式覆盖 `dataset_name`,避免和 README 文案错位。
- 当前状态: 本轮说明已整理完成,准备向用户交付可执行命令示例。

## 2026-03-06 12:22 UTC: 产出独立的 `joint_refinement_camera_gaussians_v2` 使用文档
- 当前目标: 把刚确认过的使用方法整理成单独 md 文件,便于直接查阅和复制命令。
- 原因: README 只覆盖 baseline 两步,而 v2 增强需要第 3 步 refinement,单独文档更清晰。
- 本轮会做:
  1. 规划文档位置与标题。
  2. 写入 baseline / generated / external SR 三种用法。
  3. 复核路径、参数名和当前仓库配置差异。
- 当前状态: 准备创建独立使用说明文档。

### 2026-03-06 12:27 UTC
- 已完成独立使用说明文档落盘。
- 新增文件:
  - `docs/joint_refinement_camera_gaussians_v2_usage.md`
- 已确认:
  1. 文档内容与当前 `refine_robust_v2` CLI 一致。
  2. 示例路径对齐当前仓库已有输出结构。
  3. 已覆盖 baseline / generated / external SR 三类最常用场景。
- 当前状态: 本轮文档任务完成,可继续选择是否把该文档链接回 `README.md` 或相关 spec。

## 2026-03-06 12:31 UTC: 准备提交并推送当前 `joint_refinement_camera_gaussians_v2` 相关成果
- 当前目标: 按用户要求执行 git 提交并推送。
- 原因: 本轮代码、规格、计划文档和独立使用文档已经完成,现在需要把结果落到远端分支。
- 本轮会做:
  1. 检查 git 工作区状态与当前分支。
  2. 检查是否存在 submodule 改动或其他非本任务改动。
  3. 仅提交本任务相关文件,使用带 signoff 的 commit。
  4. 推送到当前跟踪分支。
- 当前状态: 正在做提交前检查。

### 2026-03-06 12:34 UTC
- 已完成提交前验证。
- 本次新鲜验证结果:
  - `python3 -m py_compile scripts/refine_robust_v2.py src/refinement_v2/*.py tests/refinement_v2/*.py` 通过
  - `PYTHONPATH="$(pwd)" pytest -q tests/refinement_v2` 通过,结果 `51 passed`
- 当前状态: 准备暂存本任务相关文件并创建带 signoff 的提交。

### 2026-03-06 12:35 UTC
- 已完成本任务相关文件的暂存与提交。
- 提交信息:
  - `84d7a36 Implement refinement v2 stage2b and reference workflow`
- 当前状态: 正在推送到 `origin/main`。

### 2026-03-06 12:36 UTC
- 已完成推送。
- 远端结果:
  - `origin/main`
  - `fd8f963..84d7a36  main -> main`
- 当前状态: 本轮“提交并推送”任务已完成。

## 2026-03-08 06:41 UTC: 在 A800 主机继续 `Long-LRM` selective SR 主线
- 当前目标: 从上次真正未完成的下一步继续, 把 `Phase 3S / Stage 3SR` 从现有 patch supervision 中显式拆出来, 为后续 `gaussian_fidelity_score`、`W_sr_select`、`W_final_sr`、`L_sampling_smooth` 铺路。
- 原因:
  1. `Stage 2B workflow` 与 external reference contract 都已经完成并推送。
  2. `Long-LRM` 线当前真正未落地的 critical path, 是 selective SR 的阶段语义和代码结构。
  3. 上次结论已经明确: 下一步应优先做“无行为变化”的阶段拆分, 而不是继续扩 direct file inputs。
- 本轮计划:
  1. 回读 `refinement_v2` 相关实现, 确认 patch supervision 当前嵌入点。
  2. 先做显式 `Phase 3S / Stage 3SR` 拆分, 尽量保持现有行为不变。
  3. 用测试锁住阶段顺序与兼容行为。
  4. 若拆分稳定, 再记录下一轮继续补 fidelity / selection / smoothing 的切口。
- 当前状态: 正在读取 `runner.py`、`stage_controller.py`、loss/diagnostics/tests, 准备进入结构拆分设计。

## 2026-03-08 07:04 UTC: 在 A800 上建立 `target_index_subsample = 4` 的 `stage2a` 基准
- 当前目标: 先在 A800 主机上建立 full-view native `stage2a` 的 `target_subsample=4` 基准, 为后续 selective SR / `Stage 3SR` 对照提供新的高 observation 密度基线。
- 原因:
  1. 旧 48G 主机的正式 full-view 基线只稳定到 `target_subsample=16`。
  2. 文档已经记录 `target_subsample=8` 在旧机器 full-view `Stage 2A` 上会 OOM, 因此 A800 的首个价值就是把 observation 密度推进到 `4`。
  3. 在继续做 `Long-LRM` selective SR 结构改造前, 先立住资源与质量基线更稳。
- 现象:
  1. 当前仓库已存在 `full_view_native_stage2a_fair_v2/v3` 结果, 但导出视频帧数仍是 `48`, 说明它们和 `sub16` 同档, 不是本轮要的 `sub4` 基准。
  2. 当前 A800 设备已确认是 `NVIDIA A800-SXM4-80GB`。
- 当前假设:
  1. `target_subsample=4` 会把 full-view observation 数从 `48` 提升到约 `186`, 显存和时长都会显著上升。
  2. A800 80GB 有机会支撑这档 `stage2a` native benchmark`, 但需要先做 smoke 验证避免直接长跑 OOM。
- 验证计划:
  1. 先跑一个 `sub4` 的最小 smoke, 验证 loader / render / backward / export 是否能完整通过。
  2. smoke 通过后, 复用当前 fair native 口径跑正式 `stage2a` benchmark, 记录 wall time、GPU 显存采样、`diagnostics.json` 和 `metrics_stage2a/stage3sr`。
  3. 跑完后把结果追加到 `notes.md`、`WORKLOG.md`、必要时更新 `docs/cmd.md`。
- 当前状态: 正在做 `sub4` smoke 前的命令口径确认与日志续档。

### 2026-03-08 07:04 UTC
- `sub4` smoke 已完成两轮最小验证:
  1. 默认 allocator
  2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 两轮结果一致:
  - 都在 `Phase 0` 的 `src/rendering/gs.py::_merge_chunk_meta()` OOM
  - 尚未进入 `stage2a` 迭代主体
- 本轮已建立的基准结论:
  - 当前 full-view native `target_subsample=4` 是 A800 80GB 上的容量边界, 不是可稳定运行基线
- 当前状态:
  - 已完成“建立 `sub4` benchmark 边界”这一步
  - 下一步如继续推进, 应优先处理 `render_meta` 内存占用, 或退回 `sub8/sub16` 做可运行对照

## [2026-03-09 04:04:00 UTC] 任务: 再次分析 add-refinement-v2-depth-anchor 可行性

### 目标
- 重新评估 `add-refinement-v2-depth-anchor` change 的技术可行性
- 基于当前代码与既有 OpenSpec 文档,区分现象、假设、验证依据、结论
- 给出是否值得继续推进、卡点在哪里、下一步应补什么证据

### 阶段
- [x] 阶段1: 收集历史上下文与 change 状态
- [ ] 阶段2: 回读最近计划、笔记、工作日志与风险记录
- [ ] 阶段3: 阅读 change artifacts 与相关代码路径
- [ ] 阶段4: 形成可行性判断与建议

### 当前状态
- 正在回读六文件与 OpenSpec change,准备定位上一次分析停在哪个问题上

### 2026-03-09 04:06 UTC
- [x] 阶段2: 回读最近计划、笔记、工作日志与风险记录
- [ ] 阶段3: 阅读 change artifacts 与相关代码路径
- [ ] 阶段4: 形成可行性判断与建议
- 目前判断: 六文件中尚未出现 `add-refinement-v2-depth-anchor` 的历史分析记录,因此需要把 OpenSpec artifacts 和现有 `refinement_v2` 主线直接对照。
- 当前状态: 正在读取 change 文档与代码入口,确认它与 selective SR / depth 现状的关系。

### 2026-03-09 04:18 UTC
- [x] 阶段3: 阅读 change artifacts 与相关代码路径
- [ ] 阶段4: 形成可行性判断与建议
- 关键静态发现:
  1. `Stage 2A / Stage 3SR` 当前只训练 `opacity/scales/colors`,不训练 `means/rotations`。
  2. renderer 已稳定返回 `alphas_pred` 与 `depths_pred`,且走的是 `gsplat` 的 `RGB+ED` 输出。
  3. `Phase 0 / Phase 1` 已经存在无 backward 的 baseline/eval render 路径,适合复用来捕获 depth anchor reference。
  4. `SceneBundle` 与 dataloader 当前不携带 depth,且 refinement 显式 `use_depth=false`,说明 baseline_render 版 V1 不需要先改数据契约。
- 当前状态: 正在基于这些证据整理最终可行性结论,并区分“防漂移”与“纠偏”边界。

### 2026-03-09 04:20 UTC
- [x] 阶段4: 形成可行性判断与建议
- 最终判断:
  1. `add-refinement-v2-depth-anchor` 在工程上可行。
  2. 它最适合被定义为 `anti-drift` 深度锚点,而不是 geometry correction 方案。
  3. 若直接把收益表述成“修复 baseline 厚表面/拉丝”,会超出当前证据。
  4. 最稳的 V1 落点应优先复用 `Phase 0 / Phase 1` 的 baseline eval render,并谨慎处理低 alpha 像素与 `opacity_sparse` 的耦合。
- 当前状态: 本轮可行性分析完成,可在后续决定是否回写 OpenSpec artifacts 或进入实现。

## [2026-03-09 04:31:00 UTC] 任务: 追踪当前 `moge_version` 默认行为的完整调用链

### 目标
- 沿当前仓库实际入口脚本追踪 `moge_version` 的默认值传播路径
- 区分 CLI 默认值、内建模型默认值、以及本地 checkpoint `auto` 检测三层语义
- 给用户一个可直接落地到自己命令上的判断依据

### 阶段
- [x] 阶段1: 回读六文件上下文
- [ ] 阶段2: 定位所有相关入口脚本与参数注册点
- [ ] 阶段3: 串联参数传递与加载逻辑
- [ ] 阶段4: 整理结论并给出按命令判断的方法

### 当前状态
- 正在读取入口脚本和参数注册位置,准备从 README 常用命令向下追到 `load_moge_model(...)`

### 2026-03-09 04:38 UTC
- [x] 阶段2: 定位所有相关入口脚本与参数注册点
- [x] 阶段3: 串联参数传递与加载逻辑
- [ ] 阶段4: 整理结论并给出按命令判断的方法
- 关键发现:
  1. README Example 1 第 1 步和 `scripts/bash/static_sdg.sh` 都没有显式传 `--moge_version`。
  2. `gen3c_single_image.py` / `gen3c_single_image_sdg.py` 的 parser 都复用 `add_moge_arguments(...)`,因此默认值统一是 `auto`。
  3. `gen3c_persistent.py` 复用 `gen3c_single_image.create_parser()`,所以默认行为同样继承 `auto`。
  4. `sample.py` 本身不加载 `MoGe`,只消费前一步产出的相机/视频结果。
  5. `auto` 在“未显式指定 repo_id”时会走 `Ruicheng/moge-2-vitl`,因此 README 常用命令实际默认是 `v2`。
- 当前状态: 正在整理最终答复,会按“你现在跑 README 命令时到底用哪个版本”来解释。

### 2026-03-09 04:40 UTC
- [x] 阶段4: 整理结论并给出按命令判断的方法
- 最终结论:
  1. `--moge_version` 的 CLI 默认值是 `auto`,不是字面上的 `v2`。
  2. 但在 README 常用命令、`scripts/bash/static_sdg.sh`、`gen3c_single_image.py`、`gen3c_single_image_sdg.py`、`gen3c_persistent.py` 这些入口上,如果你不额外传参,最终都会落到 `auto -> v2`。
  3. `sample.py` 本身不参与 `MoGe` 选择,它只消费前一步已经生成好的相机和视频产物。
  4. 只有当你显式传本地 checkpoint 或显式指定 `--moge_version v1` / `--moge_model_id Ruicheng/moge-vitl` 时,行为才会偏离这条默认链路。
- 当前状态: 本轮追踪完成。

## [2026-03-09 04:39:00 UTC] 任务: 将 auto_center_depth 改为默认开启

### 目标
- 将单图/SDG 相机轨迹路径中的 `auto_center_depth` 改为默认启用
- 保留一个显式关闭入口,避免完全失去旧行为
- 补充测试,验证默认值与显式关闭语义

### 阶段
- [ ] 阶段1: 回读相机参数入口与现有测试
- [ ] 阶段2: 修改 CLI 默认行为并保留显式关闭开关
- [ ] 阶段3: 补测试与必要文档同步
- [ ] 阶段4: 运行验证并记录结果

### 当前状态
- 正在核对 `auto_center_depth` 参数定义、调用路径和回归测试覆盖面

### 2026-03-09 04:42 UTC
- [x] 阶段1: 回读相机参数入口与现有测试
- [x] 阶段2: 修改 CLI 默认行为并保留显式关闭开关
- [x] 阶段3: 补测试与必要文档同步
- [ ] 阶段4: 运行验证并记录结果
- 已完成改动:
  1. `auto_center_depth` 改为默认开启
  2. 新增 `--no_auto_center_depth` 显式关闭入口
  3. README 已同步说明默认行为与关闭方式
- 当前状态: 正在运行针对性回归测试与语法验证。

### 2026-03-09 04:45 UTC
- [x] 阶段4: 运行验证并记录结果
- 验证结果:
  1. `python3 -m py_compile cosmos_predict1/diffusion/inference/inference_utils.py cosmos_predict1/diffusion/inference/gen3c_single_image.py` 通过
  2. `env PYTHONPATH="$(pwd)" pixi run pytest -q tests/test_camera_trajectory_center_depth.py` 通过,结果 `16 passed`
- 过程中遇到的环境问题:
  1. 直接用系统 `python3` 跑 pytest 时缺少 `warp`
  2. 直接 `pixi run pytest` 时缺少 `PYTHONPATH`
  3. 最终通过 `env PYTHONPATH="$(pwd)" pixi run pytest ...` 获得真实测试结果
- 当前状态: 本轮“默认启用 auto_center_depth”改动已完成并验证通过。
