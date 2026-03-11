# LATER_PLANS

## 2026-03-10 03:48:00 UTC

- `Phase B` first validation 已完成:
  - `multi4` OOM
  - `multi2` 可跑但收益接近持平
- 当前更值得继续的后手顺序:
  1. 把 `fidelity_ratio_threshold / fidelity_sigmoid_k / fidelity_min_views` 暴露到 CLI,做小范围 calibration
  2. 如果 calibration 仍然无明显收益, 直接进入 `Phase C`, 不再继续堆 patch 数量

## 2026-03-10 06:10:00 UTC

- fidelity 参数打通与第一轮 calibration 已完成:
  - 已暴露 `--fidelity-ratio-threshold`
  - 已暴露 `--fidelity-sigmoid-k`
  - 已暴露 `--fidelity-min-views`
  - 已暴露 `--fidelity-opacity-threshold`
- 第一轮 calibration 结果:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseA_rho_thr1p1_sub8_iter20_20260310`
  - `selection_mean` 从 `0.01574` 提到 `0.04699`
  - 但最终仍未超过 `Phase A` rerun
- 当前后手顺序更新为:
  1. 直接推进 `Phase C: HR render + LR consistency`
  2. 如果未来还想继续压榨 `Phase A`, 再做 `fidelity_sigmoid_k` 或 `fidelity_min_views` 的小范围 sweep, 但优先级已经低于 `Phase C`

## 2026-03-10 06:35:00 UTC

- `Phase C` 方案和任务已按新共识改写到计划文档
- 当前正式后手顺序改成:
  1. 实现 `Phase C.1`: full-frame HR render 路径
  2. 实现 `Phase C.2`: HR -> LR consistency 路径
  3. 实现 `Phase C.3`: 重组 `Stage 3SR` 目标
  4. 实现 `Phase C.4`: diagnostics / metrics / tests
- `Phase A` calibration 与旧 `Continuation Task B/D/A` 现在都不是主线 blocker


## 2026-03-10 06:35:00 UTC

- `Phase C` 的最小真实 `sub8` smoke 已跑通:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_smoke_sub8_streamshard_20260310`
- 当前下一批最值得继续的任务:
  1. 用同一条 `stream-sharded full-frame HR` 路径把 `iters-stage2a` 提到真实对照长度, 观察是否能接近或超过当前 `Phase A`
  2. 评估 `lambda-hr-rgb` / `lambda-lr-consistency` 配比, 当前 smoke 中 `loss_lr_consistency` 仍明显主导
  3. 在 `Phase C` 已可跑的前提下, 再决定是否进入 `Phase D` 的 HR 导出主线


## 2026-03-10 07:05:00 UTC

- `Phase C` 的 `hr=32, lr=1.0, iter8` 已拿到第一轮长程证据:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr1_sub8_iter8_20260310`
- 下一轮最有信息量的两个方向:
  1. 固定 `lambda_hr_rgb=32`, 开始扫 `lambda_lr_consistency=0.5 / 0.25`
  2. 把当前 `hr=32, lr=1.0` 继续拉长到更接近 `Phase A iter20` 的长度, 做更公平对照

## 2026-03-10 07:47:33 UTC

- `Phase C` 的 `hr32, lr0.5, iter20, sub8` 长跑已经完成并收口:
  - `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p5_sub8_iter20_20260310`
- 当前后手顺序进一步更新为:
  1. 优先推进 `Phase D`, 让最终导出支持真正的 HR render
  2. 如果后面仍要继续压 `Phase C`, 优先做:
     - 更长 iter
     - `lambda_lr_consistency` 在 `0.5` 附近的进一步 sweep
  3. 不再默认回到 `1 iter` 的参数 sweep

## 2026-03-10 08:11:59 UTC

- `Phase D` 最小闭环已经落地并通过 `113 passed` 回归。
- 但真实 full-view `sub8` smoke 仍缺一条新的动态证据:
  - 需要在 A800 没有外部 `~49 GiB` 常驻占用时, 重跑一次:
    - `outputs/refine_v2/full_view_sr_stage3sr_phaseD_export_smoke_sub8_20260310_retry`
  - 目标是确认真实资产下也能完整生成:
    - `baseline_render_hr.mp4`
    - `gt_reference_hr.mp4`
    - `final_render_hr.mp4`
- 这条验证完成后, 再继续回到:
  1. `Phase C` 更长 iter
  2. `lambda_lr_consistency≈0.5` 附近的小范围继续实验

## 2026-03-10 08:13:02 UTC

- `Phase D` 已完成最小交付:
  - 当前若继续主线, 优先回到 `Phase C` 的质量推进
- 下一轮最有信息量的后手顺序改成:
  1. 继续基于 `hr=32, lr=0.5` 压更长 iter 或近邻 sweep
  2. 如果 `Phase C` 的 native gap 迟迟不再缩小, 再评估是否进入 `Phase E`
  3. `Phase B` 只在确认需要扩大 supervision coverage 时再回头做

## 2026-03-10 08:13:38 UTC

- 回滚一条过于保守的口径:
  - `Phase D` 不是“还没有真实 smoke 证据”
  - 它已经有 `outputs/refine_v2/phaseD_phase0_hr_export_smoke_20260310`
- 当前真正还没补到的动态验证是:
  - full-view
  - `--target-subsample 8`
  - `stop_after=stage2a`
  这一条更重的 HR 导出 smoke

## 2026-03-10 08:53:00 UTC

- `Phase D` 的 `phase0` 与更重的 `full-view + sub8 + stage2a` 级 HR 导出 smoke 都已经完成, 不再是当前 blocker。
- `Phase C hr32 lr0.5 sub8 iter32` 现在已经在 native `psnr` 上超过 `Phase A iter20`。
- 当前后手顺序更新为:
  1. 固定 `lambda_hr_rgb=32`, 继续做 `lambda_lr_consistency≈0.5` 的近邻 sweep, 优先:
     - `0.6`
     - `0.4`
  2. 如果 native `residual_mean` 仍始终压不过 `Phase A`, 再评估 `Phase E`
  3. `Phase B` 只在新的 objective 下重新暴露 coverage / 显存问题时再回头

## 2026-03-10 09:49:30 UTC

- `Phase E` 的最小版 `stage3b` 已经落地并完成真实 smoke:
  - `outputs/refine_v2/phaseE_stage3b_smoke_sub8_20260310`
- 当前值得继续的后手顺序更新为:
  1. 跑一条更长的真实 `stage3b` 对照, 与 `Phase C hr32 lr0.5 iter32` 做 apples-to-apples 比较
  2. 如果 `stage3b` 长跑仍持续改进, 再考虑是否给它拆出独立超参数面:
     - `iters_stage3b`
     - `means_delta_cap_stage3b`
     - `lambda_means_anchor_stage3b`
     - `lambda_rotation_reg_stage3b`
  3. `Phase C` 的 `0.4` / `0.6` 近邻 sweep 现在后移, 除非后面要重新审视 `Phase E` 的收益归因

## 2026-03-10 10:20:00 UTC

- 已完成并可从后续待办里移除:
  - 给 `stage3b` 拆独立超参数面
- 当前新的 `Phase E` 后手顺序:
  1. 用 `--target-subsample 8` 跑一条更长的真实 `stage3b` 对照, 明确带上:
     - `--iters-stage3b`
     - `--lambda-means-anchor-stage3b`
     - `--lambda-rotation-reg-stage3b`
     - `--means-delta-cap-stage3b`
  2. 用这条更长 run 和 `outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p5_sub8_iter32_20260310` 做 apples-to-apples 比较
  3. 如果长跑仍有持续收益, 再做 `Phase E` 内部 calibration
  4. 更后面的版本再评估 `stage3b` 与 densify / prune 的耦合

## 2026-03-10 11:22:00 UTC

- `Phase E` 当前最新 continuation 顺序更新为:
  1. 直接基于正式 CLI workflow 做 `stage3b` calibration:
     - `--start-stage stage3b`
     - `--resume`
     - `--iters-stage3b`
     - `--lambda-means-anchor-stage3b`
     - `--lambda-rotation-reg-stage3b`
     - `--means-delta-cap-stage3b`
  2. 在拿到 1-2 组 calibration 证据后, 再决定要不要放宽 auto gate 的 `residual_mean` 阈值
  3. 再后面才考虑 `stage3b` 与 densify / prune 的耦合
- 已可从“待补基础设施”中移除:
  - `start_stage=stage3b` continuation 入口
