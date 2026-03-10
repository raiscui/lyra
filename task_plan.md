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
