# 任务计划: Phase C iter32 收口与后续 Phase C 主线推进

## 目标

- 先把 `Phase C hr32 lr0.5 sub8 iter32` 的真实结果同步回六文件与长期计划文档。
- 再基于更新后的 backlog, 继续下一轮 `Phase C` 主线实验。

## 阶段

- [x] 阶段1: 回读六文件、计划文档与真实实验产物
- [ ] 阶段2: 把 `iter32` 结果写回六文件与文档
- [ ] 阶段3: 启动下一轮 `Phase C` `sub8` 实验
- [ ] 阶段4: 回收结果并再次同步六文件与文档

## 关键问题

1. `Phase C` 现在是否已经跨过“成立性验证”门槛?
   - 是。磁盘证据显示它已经在 native `psnr` 上超过 `Phase A iter20`, 但 `residual_mean` 仍略高。
2. 下一轮最值得继续压的方向是什么?
   - 优先围绕 `lambda_lr_consistency≈0.5` 做近邻 sweep, 看能否在不丢掉 HR 指标的前提下把 native `residual_mean` 也压过去。
3. 当前是否还需要把 `Phase D` 当主 blocker?
   - 不需要。`phase0` 与更重的 `stage2a` 级真实 HR 导出 smoke 都已经完成。

## 做出的决定

- 先续档 `task_plan.md`, 避免继续在超过 1000 行的文件上追加。
- 先用 `diagnostics.json` 与 `metrics_stage3sr.json` 的真实结果更新文档, 再决定新实验方向。
- 后续实验继续统一使用 `--target-subsample 8`。
- 暂不回头做 `Stage 2B` 训练, 也不再加侧面视角补评。

## 遇到的错误

- 暂无新的代码错误。
- 本轮主要是状态收口与 backlog 重排, 不是 bug 修复。

## 状态

**目前在阶段2**
- 正在把 `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p5_sub8_iter32_20260310` 的结果同步到六文件与文档。
- 同时会把根目录里已回读的历史六文件移入 `archive/`, 降低后续检索噪音。

## 2026-03-10 08:55:30 UTC

### 阶段更新: `iter32` 收口完成, 开始下一轮 `Phase C` 近邻 sweep

- [x] 阶段2: 把 `iter32` 结果写回六文件与文档
- [ ] 阶段3: 启动下一轮 `Phase C` `sub8` 实验
- [ ] 阶段4: 回收结果并再次同步六文件与文档

### 当前行动

- 已完成 `iter32` 的文档和六文件同步。
- 现在开始按新的 backlog 跑第一条近邻点:
  - `lambda_hr_rgb=32`
  - `lambda_lr_consistency=0.6`
  - `target_subsample=8`
  - `iters-stage2a=32`
- 这一轮的目标不是再证明 `Phase C` 是否成立, 而是看 native `residual_mean` 能否继续压低。

## 2026-03-10 09:05:30 UTC

### 状态变更: `lr=0.6` 近邻实验被用户主动中断, 先基于已有中间证据回答策略问题

- [ ] 阶段3: 启动下一轮 `Phase C` `sub8` 实验
- [ ] 阶段4: 回收结果并再次同步六文件与文档

### 当前现象

- `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p6_sub8_iter32_20260310` 已进入 `stage3sr`, 但尚未自然结束。
- 用户在实验进行中提出新的策略问题:
  - 是否应该直接舍弃 LR consistency
  - `/workspace/FastGS` 的 `26+ PSNR` 应如何理解
  - `Phase E` 是否还没做

### 当前行动

- 先基于已落盘的中间指标和 `FastGS` 本地文档回答这三个问题。
- 回答后再决定是继续补完 `lr=0.6` / `lr=0.4` 实验, 还是先切出 `HR-only` 明确对照。

## 2026-03-10 09:07:30 UTC

### 行动变更: 暂停继续 `lr=0.6` 近邻点, 先跑 `HR-only` 明确对照

- [ ] 阶段3: 启动下一轮 `Phase C` `sub8` 实验
- [ ] 阶段4: 回收结果并再次同步六文件与文档

### 做出的决定

- 用户已经明确质疑 LR consistency 的必要性。
- 当前最有信息量的实验不再是继续补 `0.6`, 而是直接做:
  - `lambda_hr_rgb=32`
  - `lambda_lr_consistency=0.0`
  - `target_subsample=8`
  - `iters-stage2a=32`
- 这条 run 的意义是把“继续保留 LR”与“完全舍弃 LR”做一次干净的 apples-to-apples 对照。

### 当前行动

- 先确认 A800 空闲。
- 然后立即启动 `HR-only` 对照实验。
- 实验结束后, 直接和 `lr=0.5 iter32` 以及 `FastGS` 的 train-only 口径做分层对比。

## 2026-03-10 09:19:30 UTC

### 阶段推进: `HR-only` 对照已完成, 结论是否定“直接删 LR”, 继续补 `0.4` 近邻点

- [ ] 阶段3: 启动下一轮 `Phase C` `sub8` 实验
- [ ] 阶段4: 回收结果并再次同步六文件与文档

### 新证据

- `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p0_sub8_iter32_20260310` 已自然完成。
- 最终结果:
  - `psnr = 22.271083371818634`
  - `residual_mean = 0.049804605543613434`
  - `sharpness = 0.00288753816857934`
  - `psnr_hr = 21.283824302586495`
  - `residual_mean_hr = 0.04892662912607193`
- 相比 `lr=0.5 iter32`:
  - `psnr -1.883485`
  - `residual_mean +0.014887`
  - `sharpness -0.000304`
  - `psnr_hr -0.332138`

### 当前结论

- “完全舍弃 LR consistency” 不是当前正确方向。
- `LR consistency` 至少仍然在承担重要的 anti-hallucination / native 对齐作用。
- 因此下一条最该补的不是再次跑 `HR-only`, 而是补 `lambda_lr_consistency=0.4` 这一侧的近邻点, 看它是否比 `0.5` 更接近 frontier。

### 当前行动

- 立即启动:
  - `lambda_hr_rgb=32`
  - `lambda_lr_consistency=0.4`
  - `target_subsample=8`
  - `iters-stage2a=32`
- 这条 run 结束后, 再统一收口 `0.0 / 0.4 / 0.5 / 0.6(partial)` 的策略结论。

## 2026-03-10 09:24:30 UTC

### 方向切换: 用户要求先停止参数调优, 转入 `Phase E` 实现

- [ ] 阶段3: 实现 `Phase E` 代码与测试
- [ ] 阶段4: 跑验证并同步六文件与文档

### 当前现象

- `lr=0.4` 近邻实验已经启动, 但用户明确要求先不继续调参。
- 当前最高优先级已经从 `Phase C` 参数 frontier 切换为 `Phase E` 落地。

### 当前行动

- 立即停止正在运行的 `lr=0.4` 实验进程。
- 回读 `Phase E` 计划和现有 `Stage 2B limited geometry` 实现。
- 基于现有代码选择最小正确切口, 开始实现 `Phase E`。

## 2026-03-10 09:39:30 UTC

### 阶段推进: `Phase E` 代码与回归已通过, 开始补真实 smoke

- [x] 阶段3: 实现 `Phase E` 代码与测试
- [ ] 阶段4: 跑验证并同步六文件与文档

### 已完成验证

- `python3 -m py_compile` 已通过:
  - `src/refinement_v2/config.py`
  - `src/refinement_v2/losses.py`
  - `src/refinement_v2/stage_controller.py`
  - `src/refinement_v2/runner.py`
  - `tests/refinement_v2/test_runner_stage3b.py`
- 定向测试已通过:
  - `25 passed`
- 全量 `tests/refinement_v2` 已通过:
  - `117 passed`

### 当前行动

- 启动一条真实 `Phase E` smoke:
  - `--enable-stage3b`
  - `--stop-after stage3b`
  - `--target-subsample 8`
- 目标是确认真实资产下能实际到达 `phase_reached = stage3b`, 而不只是单测通过。

## 2026-03-10 09:49:30 UTC

### 阶段完成: `Phase E` 最小实现 + 回归 + 真实 smoke 已收口

- [x] 阶段3: 实现 `Phase E` 代码与测试
- [x] 阶段4: 跑验证并同步六文件与文档

### 本轮完成内容

- 已新增 `enable_stage3b` 配置和 `stop_after=stage3b` 调试入口。
- 已在 `Stage 3SR` 的 full-frame HR 主监督路径上补出 `stage3b`:
  - 允许有限 geometry 更新
  - 复用 HR 主监督 + LR consistency
  - 额外加入 `loss_means_anchor` / `loss_rotation_reg`
- 已跑通:
  - 定向测试 `25 passed`
  - 全量 `tests/refinement_v2` `117 passed`
  - 真实 smoke `outputs/refine_v2/phaseE_stage3b_smoke_sub8_20260310`

### 动态结论

- 真实 smoke 已确认:
  - `phase_reached = stage3b`
  - `stopped_reason = metrics_plateau`
  - 同一条 run 内:
    - `stage3sr` 末尾 `psnr = 19.628720`, `residual_mean = 0.064013`
    - `stage3b` 末尾 `psnr = 20.435591`, `residual_mean = 0.056794`
- 这说明 `Phase E` 已经不只是文档计划, 而是代码和真实资产上都能进入的阶段。

## 2026-03-10 10:05:00 UTC

### 阶段续做: `Phase E` 继续补独立 `stage3b` 超参数面

- [ ] 阶段3: 实现 `stage3b` 专属超参数与 CLI 映射
- [ ] 阶段4: 补测试、回归验证并同步六文件与文档

### 现象

- `Phase E` 的最小版 `stage3b` 已经完成, 代码和真实 smoke 都通过了。
- 但当前 `stage3b` 仍直接复用 `stage2b` 的关键超参数:
  - `iters_stage2b`
  - `lambda_means_anchor`
  - `lambda_rotation_reg`
  - `means_delta_cap`
- 这会让后续 `Phase E` 长跑实验难以单独调 `stage3b`, 也会把旧 `stage2b` 约束继续耦合进来。

### 当前决定

- 先不做新的参数 sweep。
- 先把 `stage3b` 拆成独立超参数面:
  - `iters_stage3b`
  - `means_delta_cap_stage3b`
  - `lambda_means_anchor_stage3b`
  - `lambda_rotation_reg_stage3b`
- 保持兼容:
  - 默认值继续继承现有 `stage2b` / 通用 geometry 配置语义
  - 不破坏已经跑通的最小版 `Phase E`

### 当前行动

- 修改 `config.py` / `runner.py` / `gaussian_adapter.py` 的 `stage3b` 参数读取路径。
- 补 `test_config.py` 和 `test_runner_stage3b.py` 的回归。
- 跑定向测试后再决定是否补全量 `tests/refinement_v2`。

## 2026-03-10 10:12:00 UTC

### 阶段推进: `stage3b` 独立超参数面已实现, 开始做全量回归与文档同步

- [x] 阶段3: 实现 `stage3b` 专属超参数与 CLI 映射
- [ ] 阶段4: 补测试、回归验证并同步六文件与文档

### 已完成实现

- `StageHyperParams` 已新增并兼容回填:
  - `iters_stage3b`
  - `lambda_means_anchor_stage3b`
  - `lambda_rotation_reg_stage3b`
  - `means_delta_cap_stage3b`
- CLI 已新增:
  - `--iters-stage3b`
  - `--lambda-means-anchor-stage3b`
  - `--lambda-rotation-reg-stage3b`
  - `--means-delta-cap`
  - `--means-delta-cap-stage3b`
- `runner.py` 已让 `stage3b` 读取独立 iteration / geometry regularizer / clamp 配置。
- `gaussian_adapter.py` 已让 `stage3b` 的位置 clamp 独立读取 `means_delta_cap_stage3b`。

### 已完成验证

- `python3 -m py_compile` 已通过。
- 定向测试已通过:
  - `tests/refinement_v2/test_config.py`
  - `tests/refinement_v2/test_gaussian_adapter.py`
  - `tests/refinement_v2/test_runner_stage3b.py`
  - `tests/refinement_v2/test_stage_controller.py`
- 当前结果: `30 passed`

### 当前行动

- 继续跑全量 `tests/refinement_v2`。
- 通过后同步 `docs/cmd.md`、长期计划文档和六文件。

## 2026-03-10 10:20:00 UTC

### 阶段完成: `Phase E` 的 `stage3b` 独立超参数面已落地并验证完成

- [x] 阶段3: 实现 `stage3b` 专属超参数与 CLI 映射
- [x] 阶段4: 补测试、回归验证并同步六文件与文档

### 最终结果

- 已新增并跑通:
  - `iters_stage3b`
  - `lambda_means_anchor_stage3b`
  - `lambda_rotation_reg_stage3b`
  - `means_delta_cap_stage3b`
- 已完成验证:
  - 定向测试 `30 passed`
  - 全量回归 `119 passed`
  - 真实 smoke `outputs/refine_v2/phaseE_stage3b_hparams_smoke_sub8_20260310`
- 真实 smoke 中已经确认新的 `stage3b` 专属参数真的进入运行时:
  - `iters_budget = 2`
  - `lambda_means_anchor_active = 0.02`
  - `lambda_rotation_reg_active = 0.02`
  - `means_delta_cap_active = 0.01`

### 当前结论

- `Phase E` 现在不仅有最小版 `stage3b`, 也已经有可独立实验的超参数面。
- 因此后续主线应直接进入更长的 `stage3b` apples-to-apples 对照, 不再需要回头补这层基础设施。

## 2026-03-10 10:28:00 UTC

### 阶段续做: 启动更长的 `Phase E stage3b` apples-to-apples 对照

- [ ] 阶段3: 启动更长的 `Phase E stage3b` 真实 run
- [ ] 阶段4: 回收结果并与 `Phase C hr32 lr0.5 iter32` 做对照

### 现象

- `stage3b` 的独立超参数面已经落地, 配置和 smoke 都已验证。
- 当前最缺的是更长的动态证据, 用来回答:
  - `stage3b` 的收益在更长 iter 下是否稳定存在
  - 它最终能否超过 `Phase C hr32 lr0.5 sub8 iter32`

### 当前决定

- 直接跑一条更长的真实 full-view `sub8` 对照。
- 口径保持和 `Phase C` 主线一致:
  - `lambda_hr_rgb=32`
  - `lambda_lr_consistency=0.5`
  - `target_subsample=8`
  - `reference_render_shard_views=1`
- 同时显式启用新的 `stage3b` 专属参数面:
  - `iters_stage3b=32`
  - `lambda_means_anchor_stage3b=0.02`
  - `lambda_rotation_reg_stage3b=0.02`
  - `means_delta_cap_stage3b=0.01`

### 当前行动

- 先确认 A800 空闲。
- 然后启动长跑目录:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_hr32_lr0p5_sub8_iter32_20260310`
- run 结束后, 直接回收 `diagnostics.json`、`metrics_stage3sr.json`、`metrics_stage3b.json` 做对照结论。

## 2026-03-10 10:41:00 UTC

### 阶段续做: 从 `stage3sr` 末态直接续跑 `stage3b` 做最小可证伪实验

- [ ] 阶段3: 运行 `stage3sr -> stage3b` 直接续跑实验
- [ ] 阶段4: 回收结果并判断是否需要改 gate / CLI

### 现象

- 更长的 `Phase E` apples-to-apples run 已完成, 但最终停在 `stage3sr`。
- 静态门槛与动态结果已经对上:
  - `src/refinement_v2/runner.py:1803` 使用 `residual_mean > 0.045` 作为 `local_overlap_persistent`
  - 本轮 `stage3sr` 最终 `residual_mean = 0.034917574375867844`
- `state/latest.pt` 已确认保存的是 `stage3sr` 末态, 且 diagnostics 里:
  - `stage3sr_completed = True`
  - `phase3s_completed = True`
  - `need_geometry = False`
  - `local_overlap_persistent = False`

### 当前决定

- 先不急着改全局 gate。
- 先做最小实验:
  - 从这条 run 的 `stage3sr` 末态直接接一段 `stage3b`
  - 只回答“末态继续放 geometry 还有没有收益”
- 这比直接改阈值更干净, 更容易解释。

### 当前行动

- 新建 continuation 目录:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_resume_from_stage3sr_hr32_lr0p5_sub8_iter32_20260310`
- 手动加载 `latest.pt`
- 重建 `prev_weight_map` 与 `phase3s` 选择图
- 直接运行 `run_stage3b()` 并导出最终结果

## 2026-03-10 11:02:00 UTC

### 阶段续做: 把 `stage3b` continuation 片段固化成正式 CLI 入口

- [ ] 阶段3: 实现 `start_stage=stage3b` 与 warm-start workflow
- [ ] 阶段4: 补测试、回归并用真实 continuation 结果同步文档

### 新证据

- 正常长跑 `outputs/refine_v2/full_view_sr_stage3b_phaseE_hr32_lr0p5_sub8_iter32_20260310` 没进入 `stage3b`:
  - `phase_reached = stage3sr`
  - `final.residual_mean = 0.034917574375867844`
- 从同一条 run 的 `stage3sr` 末态手动续跑 `stage3b` 后:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_resume_from_stage3sr_hr32_lr0p5_sub8_iter32_20260310`
  - `phase_reached = stage3b`
  - `final.psnr = 24.633636263443535`
  - `final.residual_mean = 0.03258064016699791`
  - `final_hr.psnr_hr = 21.93410154162478`
  - `final_hr.residual_mean_hr = 0.04774709418416023`
- 相比 gate 停在 `stage3sr` 的结果:
  - `psnr +0.479122`
  - `residual_mean -0.002337`
  - `psnr_hr +0.306722`
  - `residual_mean_hr -0.001824`

### 当前决定

- 这说明 `Phase E` 的核心问题已经从“有没有收益”变成了“如何把 continuation 路径变成正式入口”。
- 因此下一步不是继续手写片段, 而是实现:
  - `start_stage=stage3b`
  - 显式 warm-start `stage3b`
  - 让后续实验可以稳定复现

### 当前行动

- 修改 `config.py` 允许 `start_stage=stage3b`。
- 在 `runner.py` 中补 `bootstrap_stage3b_from_current_gaussians()` 或等价 workflow。
- 补 `test_runner_stage3b.py` 对应回归。

## 2026-03-10 11:10:00 UTC

### 阶段推进: 用正式 CLI 验证 `start_stage=stage3b --resume`

- [x] 阶段3: 实现 `start_stage=stage3b` 与 warm-start workflow
- [ ] 阶段4: 用真实 CLI continuation 验证并同步文档

### 已完成验证

- `python3 -m py_compile` 已通过
- 定向回归已通过:
  - `37 passed`
- 全量 `tests/refinement_v2` 已通过:
  - `122 passed`

### 当前行动

- 复制 `stage3sr` 末态的 `state/latest.pt` 到新目录。
- 用正式 CLI 启动:
  - `--resume`
  - `--start-stage stage3b`
  - `--enable-stage3b`
- 目标目录:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_cli_resume_from_stage3sr_hr32_lr0p5_sub8_iter32_20260310`

## 2026-03-10 11:22:00 UTC

### 阶段完成: `Phase E` 的 continuation workflow 已固化成正式 CLI 入口

- [x] 阶段3: 实现 `start_stage=stage3b` 与 warm-start workflow
- [x] 阶段4: 用真实 CLI continuation 验证并同步文档

### 最终结果

- 更长 auto-gate run 已确认停在 `stage3sr`, 原因是当前 geometry gate 未放行。
- 从同一条 `stage3sr` 末态继续接 `stage3b` 后, 指标继续改善:
  - 相比 gate 停在 `stage3sr`:
    - `psnr +0.479122`
    - `residual_mean -0.002337`
    - `psnr_hr +0.306722`
    - `residual_mean_hr -0.001824`
- 已新增并验证正式 CLI continuation:
  - `start_stage=stage3b`
  - `warm_start_stage3b`
  - `--resume`
- 回归结果:
  - 定向 `37 passed`
  - 全量 `122 passed`
- 真实 CLI continuation 目录:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_cli_resume_from_stage3sr_hr32_lr0p5_sub8_iter32_20260310`

### 当前结论

- `Phase E` 现在不仅成立, 而且已经具备正式 continuation workflow。
- 后续主线不再是补入口, 而是用这条入口去做 `stage3b` calibration, 再决定是否调整 auto gate。

## 2026-03-10 11:31:00 UTC

### 阶段续做: 进入 Phase E calibration

- [ ] 阶段5: 基于正式 CLI continuation 做 `stage3b` calibration
- [ ] 阶段6: 汇总 calibration 指标, 判断下一步先调 gate 还是继续扫超参数

### 当前背景

- `Phase E` 已完成正式 continuation workflow:
  - `--start-stage stage3b`
  - `--resume`
  - `warm_start_stage3b`
- 当前真正未完成的主线, 已切换为:
  - 用 `--target-subsample 8` 做 `stage3b` calibration
  - 先回答哪组 `stage3b` 超参数最值得继续

### 当前行动

- 回读 `docs/plans/2026-03-06-long-lrm-style-post-refinement.md` 与 `docs/cmd.md` 里的 `Phase E` 段落
- 选一到两组 continuation calibration 方案
- 直接在 A800 上启动真实 run, 然后汇总相对基线的增益与风险

## 2026-03-10 11:45:00 UTC

### 阶段完成: Phase E 第一轮 continuation calibration 已完成

- [x] 阶段5: 基于正式 CLI continuation 做 `stage3b` calibration
- [x] 阶段6: 汇总 calibration 指标, 判断下一步先调 gate 还是继续扫超参数

### 已验证结果

- 新 run:
  - `outputs/refine_v2/full_view_sr_stage3b_phaseE_cli_resume_from_stage3sr_hr32_lr0p5_sub8_iter64_20260310`
- 命令口径:
  - 正式 `--start-stage stage3b --resume`
  - `--target-subsample 8`
  - 只把 `iters_stage3b` 从 `32` 提到 `64`
  - 其余 `stage3b` regularizer 与 cap 保持不变
- 最终结果:
  - `phase_reached = stage3b`
  - `warm_start_stage3b = true`
  - `psnr = 24.962974696829484`
  - `residual_mean = 0.031111249700188637`
  - `psnr_hr = 22.124704905272846`
  - `residual_mean_hr = 0.046562712639570236`

### 对比结论

- 相比 `stage3b iter32` continuation:
  - `psnr +0.329351`
  - `residual_mean -0.001469`
  - `psnr_hr +0.190583`
  - `residual_mean_hr -0.001184`
- 相比 auto gate 停在 `stage3sr` 的 source run:
  - `psnr +0.808461`
  - `residual_mean -0.003806`
  - `psnr_hr +0.497326`
  - `residual_mean_hr -0.003009`
- 动态现象:
  - `metrics_stage3b.json` 最后 5 个点仍持续改善
- 当前假设:
  - `stage3b` 的主要瓶颈至少部分来自 iteration budget, `32 iter` 明显偏短
- 仍待验证:
  - `means_delta_cap_stage3b=0.01` 是否已经开始限制 geometry 继续改善
  - `lambda_means_anchor_stage3b / lambda_rotation_reg_stage3b` 是否还偏保守

### 下一步

- backlog 往前推进为:
  1. 在 `iters_stage3b>=64` 的前提下继续做 `means_delta_cap_stage3b` 与 regularizer calibration
  2. 再决定 auto gate 是否要跟着放宽

## 2026-03-14 00:00:00 UTC

### 新任务: 定位 `assets/demo/static/diffusion_output_generated_my` 生成链路

- [ ] 阶段1: 定位生成目标目录与 `rgb/*.mp4`、`pose/*.npz`、`intrinsics/*.npz` 的代码路径
- [ ] 阶段2: 验证 `pose` 与 `intrinsics` 的语义依据
- [ ] 阶段3: 验证 `0..5` 子目录分别代表什么
- [ ] 阶段4: 回写六文件并输出带行号结论

### 当前背景

- 用户要的不是泛泛描述, 而是:
  - 真实写文件的函数/调用链
  - `pose` 矩阵语义的证据
  - `intrinsics` 四元组语义的证据
  - `0..5` 子目录含义
- 本轮明确不改代码, 只做只读排查。

### 当前行动

- 先用 `rg` 与 `ast-grep` 搜索 `diffusion_output_generated_my`、`rgb`、`pose`、`intrinsics`、`.mp4`、`.npz` 的落盘路径。
- 然后回读被命中的导出函数、调用入口和相机数据来源。
- 最后把“已验证事实”和“推断”分开整理。

## 2026-03-14 00:20:00 UTC

### 阶段完成: `diffusion_output_generated_my` 生成链路已定位完毕

- [x] 阶段1: 定位生成目标目录与 `rgb/*.mp4`、`pose/*.npz`、`intrinsics/*.npz` 的代码路径
- [x] 阶段2: 验证 `pose` 与 `intrinsics` 的语义依据
- [x] 阶段3: 验证 `0..5` 子目录分别代表什么
- [x] 阶段4: 回写六文件并输出带行号结论

### 已验证结果

- 精确入口命令在 `README.md`:
  - `gen3c_single_image_sdg.py --video_save_folder assets/demo/static/diffusion_output_generated_my`
- 真实写文件脚本是:
  - `cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py`
- 真实 mp4 编码 helper 是:
  - `cosmos_predict1/utils/io.py::save_video`
- `pose.npz` 保存的是:
  - `generated_w2cs.inverse()`
  - 即 `c2w`
- `intrinsics.npz` 保存的是:
  - `K[0,0], K[1,1], K[0,2], K[1,2]`
  - 即 `[fx, fy, cx, cy]`
- `0..5` 目录含义已对应到固定轨迹:
  - `0 left`
  - `1 right`
  - `2 up`
  - `3 zoom_out`
  - `4 zoom_in`
  - `5 clockwise`

### 当前行动

- 把关键文件路径、行号、动态验证结果整理成最终答复。
- 明确区分“已验证事实”和“推断”。

## 2026-03-14 16:17:47 UTC

### 新任务: 收敛 diffusion 输出视频文件名, 避免把整段 prompt 直接拼进 `.mp4`

- [ ] 阶段1: 定位 `gen3c_single_image_sdg.py` 及相关 helper 的文件名生成链路
- [ ] 阶段2: 设计并实现安全、短且稳定的输出命名规则
- [ ] 阶段3: 补最小回归验证, 确认不再生成带空格和标点的长文件名
- [ ] 阶段4: 回写六文件并给出后续建议

### 关键问题

1. 当前长文件名是在哪一层生成的?
   - 待验证。需要先确认是入口脚本直接拼接, 还是下层 `save_video` / 命名 helper 做的默认行为。
2. 新命名规则应该优先保留什么?
   - 候选假设: 保留输入图片 stem 作为人类可读前缀, 再用短 slug 或截断后的 prompt 片段补语义, 最后必要时加 hash 保证稳定和避免冲突。

### 当前状态

**目前在阶段1**
- 正在用 `ast-grep` 和 `rg` 搜索 `.mp4` 输出命名链路。
- 目标是先拿到静态证据, 再做最小正确修复。

## 2026-03-14 16:26:00 UTC

### 阶段完成: sdg 输出文件名已改为短安全名, 并保留 legacy resume 兼容

- [x] 阶段1: 定位 `gen3c_single_image_sdg.py` 及相关 helper 的文件名生成链路
- [x] 阶段2: 设计并实现安全、短且稳定的输出命名规则
- [x] 阶段3: 补最小回归验证, 确认不再生成带空格和标点的长文件名
- [x] 阶段4: 回写六文件并给出后续建议

### 已完成内容

- 已确认根因代码路径:
  - `cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py`
  - `cosmos_predict1/diffusion/inference/gen3c_dynamic_sdg.py`
- 已把新命名逻辑抽到:
  - `cosmos_predict1/diffusion/inference/output_naming.py`
- 已接入规则:
  - 默认 `输入 stem + 短 prompt hash`
  - 显式 `--video_save_name` 优先
  - 已存在旧长文件名产物时, resume 继续兼容旧路径

### 验证结果

- `python3 -m py_compile ...` 已通过
- `python3 -m pytest tests/test_inference_output_naming.py -q` 已通过:
  - `6 passed in 0.02s`
- 用用户示例 prompt 实际演算:
  - `xhc_97e474c6`
  - `rgb/xhc_97e474c6.mp4`

### 当前状态

**本任务已完成**
- 接下来只需要在实际推理命令中直接使用当前脚本即可, 不必额外传长 prompt 到文件名。

## 2026-03-14 16:41:00 UTC

### 新任务: 把 google-t5 模型目录切到固定本地路径

- [ ] 阶段1: 定位 google-t5 当前模型路径解析链路
- [ ] 阶段2: 改成优先使用 /model/HuggingFace/google-t5/t5-11b
- [ ] 阶段3: 做最小验证, 确认不再回落到 ~/.cache/huggingface/hub
- [ ] 阶段4: 回写六文件并给出后续建议

### 关键问题

1. 当前是配置层写死了 cache 路径, 还是 loader 默认回落到了 HuggingFace cache?
   - 待验证。需要先确认 google-t5 的 from_pretrained 调用和参数传递链路。
2. 固定路径是否应该只改 google-t5, 还是抽象成通用模型根目录?
   - 当前先按用户明确要求处理 google-t5, 除非代码结构显示顺手抽成单点更干净。

### 当前状态

**目前在阶段1**
- 正在定位 google-t5 的配置来源、模型加载入口和路径拼接逻辑。
- 目标是做一刀准确改动, 而不是只在外层再叠一个补丁。

## 2026-03-14 16:58:00 UTC

### 阶段完成: google-t5 默认本地目录已统一到固定路径

- [x] 阶段1: 定位 google-t5 当前模型路径解析链路
- [x] 阶段2: 改成优先使用 /model/HuggingFace/google-t5/t5-11b
- [x] 阶段3: 做最小验证, 确认不再回落到 ~/.cache/huggingface/hub
- [x] 阶段4: 回写六文件并给出后续建议

### 现象

- 主 pipeline 已经显式传入固定目录。
- 但 `CosmosT5TextEncoder` 类自己的默认 `cache_dir` 仍然是 `~/.cache`。
- 这意味着只要有别的入口直接实例化 `CosmosT5TextEncoder()`, 就仍然可能回到 HuggingFace 默认缓存链路。

### 已验证结论

- 已把 `google-t5/t5-11b` 的默认模型名与默认目录抽成单点常量:
  - `DEFAULT_T5_MODEL_NAME`
  - `DEFAULT_T5_MODEL_DIR`
- `CosmosT5TextEncoder` 默认 `cache_dir` 已改为:
  - `/model/HuggingFace/google-t5/t5-11b`
- `BaseWorldGenerationPipeline` 现在也复用同一份常量, 不再自己写死另一份字符串。

### 验证结果

- 语法检查通过:
  - `python3 -m py_compile cosmos_predict1/auxiliary/t5_text_encoder.py cosmos_predict1/utils/base_world_generation_pipeline.py tests/test_t5_text_encoder.py`
- 定向测试通过:
  - `python3 -m pytest tests/test_t5_text_encoder.py -q`
  - `2 passed in 1.37s`

### 过程中遇到的测试干扰

- 首轮测试 collection 失败:
  - `ModuleNotFoundError: No module named 'transformers'`
- 第二轮测试失败:
  - 导入 `BaseWorldGenerationPipeline` 时经过 guardrail 链路触发 `ModuleNotFoundError: No module named 'better_profanity'`
- 这两条都已通过测试 stub 去耦解决, 没有改动生产逻辑。

### 当前状态

**本任务已完成**
- 当前代码已经把 google-t5 的默认加载目录收口到固定本地路径。
- 后续若还有新入口直接构造 `CosmosT5TextEncoder()`, 也会继承同样的固定目录行为。

## 2026-03-14 17:02:00 UTC

### 后续动作: 为 google-t5 路径修复创建独立 git 提交

- [ ] 阶段1: 识别工作区内哪些改动属于本次 google-t5 修复
- [ ] 阶段2: 仅暂存本次相关文件
- [ ] 阶段3: 创建独立 commit, 不混入其他未提交改动

### 当前状态

**目前在阶段1**
- 已观察到工作区还有 `sdg` 命名修复等其他未提交变更。
- 本次 commit 将只包含:
  - `cosmos_predict1/auxiliary/t5_text_encoder.py`
  - `cosmos_predict1/utils/base_world_generation_pipeline.py`
  - `tests/test_t5_text_encoder.py`
  - `task_plan.md`
  - `WORKLOG.md`
  - `ERRORFIX.md`

## 2026-03-14 17:08:00 UTC

### 新任务: 把 `gen3c_single_image_sdg.py` 的视频编码质量固定为 9

- [ ] 阶段1: 确认 `gen3c_single_image_sdg.py` 当前保存 mp4 的所有写入点
- [ ] 阶段2: 统一改为固定质量 9
- [ ] 阶段3: 做最小验证并回写六文件

### 当前状态

**目前在阶段1**
- 已确认该脚本有两处 `save_video(..., video_save_quality=8, ...)`。
- 用户要求只改 `gen3c_single_image_sdg.py`, 本轮不碰 `gen3c_dynamic_sdg.py`。

## 2026-03-14 17:10:00 UTC

### 阶段完成: `gen3c_single_image_sdg.py` 视频编码质量已固定为 9

- [x] 阶段1: 确认 `gen3c_single_image_sdg.py` 当前保存 mp4 的所有写入点
- [x] 阶段2: 统一改为固定质量 9
- [x] 阶段3: 做最小验证并回写六文件

### 已完成内容

- 在 `gen3c_single_image_sdg.py` 新增:
  - `SDG_VIDEO_SAVE_QUALITY = 9`
- 已把该脚本中两处 `save_video(...)` 调用都改为复用这一个常量。
- 明确没有改动:
  - `gen3c_dynamic_sdg.py`

### 验证结果

- 搜索确认:
  - `SDG_VIDEO_SAVE_QUALITY = 9`
  - 两处写视频路径都已改为 `video_save_quality=SDG_VIDEO_SAVE_QUALITY`
- 语法检查通过:
  - `python3 -m py_compile cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py`

### 当前状态

**本任务已完成**
- `gen3c_single_image_sdg.py` 现在固定以质量 9 写 mp4。
