# `refinement_v2` 常用命令速查

## 先记一条命名约定

- `stage2a` 仍然是对外主线阶段名
- `Stage 3A / Phase 3S / Stage 3SR` 是 `stage2a` 内部的可观测子阶段
- 所以:
  - `--stop-after stage2a` 仍然是正确写法
  - 但如果你开的是 `--stage2a-mode enhanced` 且 patch supervision 已配置
  - 最终 `diagnostics.json` 里的 `phase_reached` 很可能会是 `stage3sr`
- 常见产物要这样理解:
  - `metrics_stage2a.json` / `gaussians_stage2a.ply`
    - 对应 `stage2a` 对外阶段里的 native cleanup 落盘
  - `metrics_phase3s.json`
    - 对应 `Phase 3S` 的 fidelity / SR selection 诊断
  - `metrics_stage3sr.json` / `gaussians_stage3sr.ply`
    - 对应 `Stage 3SR` 的 selective SR patch supervision 落盘

## 当前主基线

- `Stage 2A` 现在要理解成:
  - `Stage 3A native cleanup`
  - `Phase 3S`
  - `Stage 3SR`
- 当前 48G 主机上, full-view 联合优化先固定:
  - `target_subsample=16`
  - 即 `8 frames/view * 6 views = 48 observations`
- `target_subsample=8` 的 `96 observations` 已在 full-view `Stage 2A` 上触发 OOM
- 所以更大 observation 密度留到:
  - A100
  - 或多卡
- 如果重跑 static demo 的 `MoGe v2 + auto center depth` 路线, 当前更接近 `v1` 体感的经验参数是:
  - `--translation_reference_depth_scale 0.35`

## full-view 正式推荐顺序

如果你当前主任务是"所有 view 一起优化一个 gaussian scene", 当前正式建议按这个顺序来:

1. 先固定 full-view native smoke
   - 这是当前 48G 主机上的正式起点
   - 推荐:
     - `target_subsample=16`
     - `48 observations`
   - 成功信号:
     - `phase_reached = stage2a`
2. 如果你要比较 external SR, 再跑 full-view SR smoke
   - 也继续固定:
     - `target_subsample=16`
   - 这条是和 native 同 observation 预算下的增强分支
   - 成功信号:
     - `phase_reached = stage3sr`
3. native / SR 这两条 smoke 都稳定后, 再决定要不要继续更长 smoke
4. 只有在上面都跑清楚后, 才继续考虑 `Stage 2B`

当前不推荐在 48G 主机上直接这样做:

- 一上来就切 `target_subsample=8`
- 一上来就把 `Stage 2B` 混进 full-view baseline
- 还没固定 native baseline, 就直接把结果归因给 SR 或别的增强项

## 1. 单路旧 `Stage 2A` 对照

```bash
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --dataset-name lyra_static_demo_generated \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --view-id 3 \
  --start-stage stage2a \
  --stage2a-mode legacy \
  --stop-after stage2a \
  --outdir outputs/refine_v2/view3_stage2a_legacy_compare
```

## 2. 单路 enhanced + native 对照

```bash
PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --dataset-name lyra_static_demo_generated \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --view-id 3 \
  --start-stage stage2a \
  --stage2a-mode enhanced \
  --patch-size 256 \
  --lambda-patch-rgb 0.25 \
  --lambda-sampling-smooth 0.0005 \
  --stop-after stage2a \
  --outdir outputs/refine_v2/view3_stage2a_enhanced_compare
```

## 3. 单路真实 SR 变体

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
  --outdir outputs/refine_v2/view3_stage3sr_real_sr_baseline_v1_fixed_cam_view
```

说明:

- 这条命令是 single-view 的 `SR variant`
- 如果要做严格的 `native vs SR` 单变量对比,正式 baseline 应该看:
  - `outputs/refine_v2/view3_stage3sr_native_reference_v1_fixed_cam_view`

## 4. 先生成 all-view `FlashVSR` reference

```bash
PYTHONPATH="$(pwd)" pixi run python3 scripts/run_flashvsr_reference.py \
  --input-root assets/demo/static/diffusion_output_generated \
  --output-root outputs/flashvsr_reference \
  --flashvsr-repo /workspace/FlashVSR-Pro \
  --runner local \
  --local-python /usr/local/miniconda3/envs/flashvsr/bin/python3 \
  --view-ids 5,0,1,2,3,4 \
  --scene-stem 00172 \
  --mode full \
  --debug-every 8
```

输出会位于:

- `outputs/flashvsr_reference/full_scale2x/<view>/rgb/00172.mp4`

## 5. full-view native smoke

```bash
PYTHONPATH="$(pwd)" pixi run python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --scene-stem 00172 \
  --view-ids 5,0,1,2,3,4 \
  --pose-root assets/demo/static/diffusion_output_generated \
  --intrinsics-root assets/demo/static/diffusion_output_generated \
  --rgb-root assets/demo/static/diffusion_output_generated \
  --target-subsample 8 \
  --iters-stage2a 2 \
  --save-every 1 \
  --stop-after stage2a \
  --outdir outputs/refine_v2/full_view_native_stage2a_smoke_sub8
```

已验证结果:

- `phase_reached = stage2a`
- `psnr = 19.4572 -> 20.3172`
- `residual_mean = 0.063859 -> 0.055581`

## 6. full-view SR smoke

```bash
PYTHONPATH="$(pwd)" pixi run python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --scene-stem 00172 \
  --view-ids 5,0,1,2,3,4 \
  --pose-root assets/demo/static/diffusion_output_generated \
  --intrinsics-root assets/demo/static/diffusion_output_generated \
  --rgb-root assets/demo/static/diffusion_output_generated \
  --reference-root outputs/flashvsr_reference/full_scale2x \
  --reference-mode super_resolved \
  --sr-scale 2.0 \
  --target-subsample 8 \
  --stage2a-mode enhanced \
  --patch-size 256 \
  --lambda-patch-rgb 0.25 \
  --lambda-sampling-smooth 0.0005 \
  --enable-pruning \
  --iters-stage2a 1 \
  --save-every 1 \
  --stop-after stage2a \
  --outdir outputs/refine_v2/full_view_sr_stage3sr_smoke_sub8
```

已验证结果:

- `phase_reached = stage3sr`
- `psnr = 19.4572 -> 20.3172`
- `residual_mean = 0.063859 -> 0.055581`
- `sr_selection_mean = 0.127218`

## 6A. full-view Phase C smoke

```bash
PYTHONPATH="$(pwd)" pixi run python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --scene-stem 00172 \
  --view-ids 5,0,1,2,3,4 \
  --pose-root assets/demo/static/diffusion_output_generated \
  --intrinsics-root assets/demo/static/diffusion_output_generated \
  --rgb-root assets/demo/static/diffusion_output_generated \
  --reference-root outputs/flashvsr_reference/full_scale2x \
  --reference-mode super_resolved \
  --sr-scale 2.0 \
  --target-subsample 8 \
  --stage2a-mode enhanced \
  --lambda-hr-rgb 0.5 \
  --lambda-lr-consistency 1.0 \
  --reference-render-shard-views 1 \
  --iters-stage2a 1 \
  --save-every 1 \
  --stop-after stage2a \
  --outdir outputs/refine_v2/full_view_sr_stage3sr_phaseC_smoke_sub8_streamshard_20260310
```

已验证结果:

- `phase_reached = stage3sr`
- `stage3sr_supervision_mode = full_frame_hr`
- `psnr = 17.8840 -> 18.8914`
- `residual_mean = 0.085080 -> 0.070438`

实现备注:

- 当前 `Phase C` 不能只做 `serial render`.
- 必须进一步走 `stream-sharded full-frame HR loss/backward`, 否则在真实 `sub8` 上仍会 OOM.

## 6B. full-view Phase C iter8 对照

```bash
PYTHONPATH="$(pwd)" pixi run python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --scene-stem 00172 \
  --view-ids 5,0,1,2,3,4 \
  --pose-root assets/demo/static/diffusion_output_generated \
  --intrinsics-root assets/demo/static/diffusion_output_generated \
  --rgb-root assets/demo/static/diffusion_output_generated \
  --reference-root outputs/flashvsr_reference/full_scale2x \
  --reference-mode super_resolved \
  --sr-scale 2.0 \
  --target-subsample 8 \
  --stage2a-mode enhanced \
  --lambda-hr-rgb 32 \
  --lambda-lr-consistency 1.0 \
  --reference-render-shard-views 1 \
  --iters-stage2a 8 \
  --save-every 2 \
  --stop-after stage2a \
  --outdir outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr1_sub8_iter8_20260310
```

已验证结果:

- `phase_reached = stage3sr`
- `stage3sr_supervision_mode = full_frame_hr`
- `psnr = 22.2530`
- `residual_mean = 0.044018`
- `psnr_hr = 20.2697`
- `residual_mean_hr = 0.059271`

补充结论:

- 当前 `Phase C` 的 `hr=8/16/32, lr=1.0` 在 `1 iter` smoke 上几乎没有可见差异。
- 所以如果后面还要继续做参数实验, 更值得优先:
  - 拉长 iter
  - 或开始动 `lambda-lr-consistency`

## 6C. full-view Phase C iter20 对照

```bash
PYTHONPATH="$(pwd)" pixi run python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --scene-stem 00172 \
  --view-ids 5,0,1,2,3,4 \
  --pose-root assets/demo/static/diffusion_output_generated \
  --intrinsics-root assets/demo/static/diffusion_output_generated \
  --rgb-root assets/demo/static/diffusion_output_generated \
  --reference-root outputs/flashvsr_reference/full_scale2x \
  --reference-mode super_resolved \
  --sr-scale 2.0 \
  --target-subsample 8 \
  --stage2a-mode enhanced \
  --lambda-hr-rgb 32 \
  --lambda-lr-consistency 0.5 \
  --reference-render-shard-views 1 \
  --iters-stage2a 20 \
  --save-every 4 \
  --stop-after stage2a \
  --outdir outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p5_sub8_iter20_20260310
```

已验证结果:

- `phase_reached = stage3sr`
- `stage3sr_supervision_mode = full_frame_hr`
- `psnr = 23.5465`
- `residual_mean = 0.037822`
- `sharpness = 0.002699`
- `psnr_hr = 21.2458`
- `residual_mean_hr = 0.051848`

对比结论:

- 相比 `6B` 的 `hr=32, lr=1.0, iter8`:
  - `psnr +1.2935`
  - `residual_mean -0.006196`
  - `psnr_hr +0.9761`
  - `residual_mean_hr -0.007423`
- 相比 `Phase A iter20`:
  - native 指标差距已经缩到:
    - `psnr -0.4545`
    - `residual_mean +0.003393`
  - 但 `Phase C` 已经额外拿到了:
    - `psnr_hr = 21.2458`
    - `residual_mean_hr = 0.051848`

当前判断:

- `Phase C` 现在已经不只是“能跑通”, 而是拿到了真正的长程收益证据。
- 如果继续主线 backlog, 下一步更值得转去 `Phase D`, 把最终导出也升级成真正的 HR 输出。

## 6D. Phase D phase0 HR export smoke

```bash
PYTHONPATH="$(pwd)" pixi run python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --scene-stem 00172 \
  --view-ids 5,0,1,2,3,4 \
  --pose-root assets/demo/static/diffusion_output_generated \
  --intrinsics-root assets/demo/static/diffusion_output_generated \
  --rgb-root assets/demo/static/diffusion_output_generated \
  --reference-root outputs/flashvsr_reference/full_scale2x \
  --reference-mode super_resolved \
  --sr-scale 2.0 \
  --target-subsample 8 \
  --stage2a-mode enhanced \
  --lambda-hr-rgb 32 \
  --lambda-lr-consistency 0.5 \
  --reference-render-shard-views 1 \
  --iters-stage2a 1 \
  --save-every 1 \
  --stop-after phase0 \
  --outdir outputs/refine_v2/phaseD_phase0_hr_export_smoke_20260310
```

已验证结果:

- `phase_reached = phase0`
- `native_hw = [704, 1280]`
- `reference_hw = [1408, 2560]`
- `diagnostics.json` 已额外包含:
  - `baseline_hr`
  - `final_hr`
  - `psnr_gain_hr`
  - `sharpness_gain_hr`
  - `residual_mean_hr_drop`

已确认真实落盘产物:

- videos:
  - `baseline_render.mp4`
  - `baseline_render_hr.mp4`
  - `final_render.mp4`
  - `final_render_hr.mp4`
  - `gt_reference.mp4`
  - `gt_reference_hr.mp4`
- snapshots:
  - `baseline_render_frame_0000.png`
  - `baseline_render_hr_frame_0000.png`
  - `final_render_frame_0000.png`
  - `final_render_hr_frame_0000.png`
  - `gt_reference_frame_0000.png`
  - `gt_reference_hr_frame_0000.png`

当前结论:

- `Phase D` 的最小交付已经落地:
  - native 导出继续保留
  - HR 导出现在也会默认一起落盘
- 这样就不需要再只盯着 native `final_render.mp4`.
- 现在可以直接看 reference-space 的 before / after 与 HR 指标。

## 6E. full-view Phase C iter32 对照

```bash
PYTHONPATH="$(pwd)" pixi run python3 scripts/refine_robust_v2.py \
  --config configs/demo/lyra_static.yaml \
  --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
  --scene-stem 00172 \
  --view-ids 5,0,1,2,3,4 \
  --pose-root assets/demo/static/diffusion_output_generated \
  --intrinsics-root assets/demo/static/diffusion_output_generated \
  --rgb-root assets/demo/static/diffusion_output_generated \
  --reference-root outputs/flashvsr_reference/full_scale2x \
  --reference-mode super_resolved \
  --sr-scale 2.0 \
  --target-subsample 8 \
  --stage2a-mode enhanced \
  --lambda-hr-rgb 32 \
  --lambda-lr-consistency 0.5 \
  --reference-render-shard-views 1 \
  --iters-stage2a 32 \
  --save-every 4 \
  --stop-after stage2a \
  --outdir outputs/refine_v2/full_view_sr_stage3sr_phaseC_hr32_lr0p5_sub8_iter32_20260310
```

已验证结果:

- `phase_reached = stage3sr`
- `stopped_reason = metrics_plateau`
- `psnr = 24.1546`
- `residual_mean = 0.034917`
- `sharpness = 0.003191`
- `psnr_hr = 21.6160`
- `residual_mean_hr = 0.049654`

已确认真实落盘产物:

- `videos/final_render.mp4`
- `videos/final_render_hr.mp4`
- `videos/baseline_render_hr.mp4`
- `diagnostics.json`
- `metrics_stage3sr.json`

对比结论:

- 相比 `6C` 的 `hr=32, lr=0.5, iter20`:
  - `psnr +0.6081`
  - `residual_mean -0.002905`
  - `sharpness +0.000493`
  - `psnr_hr +0.3702`
  - `residual_mean_hr -0.002194`
- 相比 `Phase A iter20`:
  - native `psnr +0.1536`
  - native `sharpness +0.000236`
  - native `residual_mean +0.000488`
- 也就是说:
  - `Phase C` 已经在 native `psnr` 上超过 `Phase A iter20`
  - 但 native `residual_mean` 还差最后一点点
  - 同时 `Phase C` 还保留了 `HR-space` 指标与 HR 导出产物

当前判断:

- `Phase C` 现在已经跨过了“到底能不能成立”的阶段。
- 后续主线不该再回到 `1 iter` sweep。
- 更值得继续压的是:
  1. `lambda_lr_consistency≈0.5` 附近的近邻 sweep
  2. native `residual_mean` 是否也能一起超过 `Phase A`
  3. 如果这条线开始平台化, 再评估 `Phase E`

## 7. 当前怎么理解 baseline

- 如果你现在做的是 single-view 对比:
  - 正式 baseline 还是 native reference 那条线
- 如果你现在做的是 full-view 联合优化:
  - 当前 48G 主机上的正式起点就是:
    - `target_subsample=16`
    - 先跑 full-view native smoke
    - 再按同 observation 预算比较 full-view SR smoke
  - 不建议跳过 native smoke 直接做 SR 归因
  - 也不建议在 baseline 还没固定前直接混入 `Stage 2B`
