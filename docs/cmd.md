
 - `Stage 2A` 内部的:
    - `Stage 3A native cleanup`
    - `Phase 3S`
    - `Stage 3SR`
  - `Stage 2B`



  旧 Stage 2A:

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

  现在的 Stage 2A:

```bash
  PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py \
    --config configs/demo/lyra_static.yaml \
    --dataset-name lyra_static_demo_generated \
    --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/
  lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
    --view-id 3 \
    --start-stage stage2a \
    --stage2a-mode enhanced \
    --patch-size 256 \
    --lambda-patch-rgb 0.25 \
    --lambda-sampling-smooth 0.0005 \
    --stop-after stage2a \
    --outdir outputs/refine_v2/view3_stage2a_enhanced_compare
```

  所以现在有 3 种东西要分清:

  - legacy
      - 旧 Stage 2A
  - enhanced + native
      - 新 Stage 2A 流程, 但 reference 还是 native
  - enhanced + external SR
      - 新 Stage 2A 流程, reference 是真实超分视频


用这条真正的 SR 版本命令:

```bash
  PYTHONPATH="$(pwd)" python3 scripts/refine_robust_v2.py \
    --config configs/demo/lyra_static.yaml \
    --dataset-name lyra_static_demo_generated \
    --gaussians outputs/demo/lyra_static/static_view_indices_fixed_5_0_1_2_3_4/
  lyra_static_demo_generated/gaussians_orig/gaussians_0.ply \
    --view-id 3 \
    --start-stage stage2a \
    --stage2a-mode enhanced \
    --reference-mode super_resolved \
    --reference-path outputs/flashvsr_reference/full_scale2x/3/rgb/00172.mp4 \
    --patch-size 256 \
    --lambda-patch-rgb 0.25 \
    --lambda-sampling-smooth 0.0005 \
    --stop-after stage2a \
    --outdir outputs/refine_v2/view3_stage2a_enhanced_flashvsr_compare
```

  当前已经验证通过的一版 `Stage 3SR` 主基线 v1:

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
  --outdir outputs/refine_v2/view3_stage3sr_real_sr_baseline_v1
```

说明:

- `--pose-path` 提供的是 raw pose / `c2w` 风格 `pose.npz`
- 当前代码会在 direct-input 路径内部自动转换成 provider 兼容的 `cam_view`
- 2026-03-07 修复该契约后, 同一组主线参数的重新验证输出为:
  - `outputs/refine_v2/view3_stage3sr_real_sr_baseline_v1_fixed_cam_view`
- 但如果你的问题是“超分到底比 native reference 强多少”,正式 baseline 不是上面这个目录.
  - 正式 baseline 是同一条 `enhanced + pruning + Stage 3A -> Phase 3S -> Stage 3SR` 主线,只是把 reference 保持为 native:
    - `outputs/refine_v2/view3_stage3sr_native_reference_v1_fixed_cam_view`
  - `view3_stage3sr_real_sr_baseline_v1_fixed_cam_view` 这个目录名里的 `baseline`,只是历史命名残留.
  - 语义上它应理解为:
    - `SR variant`

  这条命令的已验证结果:

  - `phase_reached = stage3sr`
  - `stopped_reason = metrics_plateau`
  - `PSNR: 15.9834 -> 18.4991`
  - `residual_mean: 0.09453 -> 0.05571`
  - `sharpness: 0.002266 -> 0.004078`
  - `scale_tail_ratio: 0.01946 -> 0.00747`
