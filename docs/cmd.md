# `refinement_v2` 常用命令速查

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
  --target-subsample 16 \
  --iters-stage2a 2 \
  --save-every 1 \
  --stop-after stage2a \
  --outdir outputs/refine_v2/full_view_native_stage2a_smoke_sub16
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
  --target-subsample 16 \
  --stage2a-mode enhanced \
  --patch-size 256 \
  --lambda-patch-rgb 0.25 \
  --lambda-sampling-smooth 0.0005 \
  --enable-pruning \
  --iters-stage2a 1 \
  --save-every 1 \
  --stop-after stage2a \
  --outdir outputs/refine_v2/full_view_sr_stage3sr_smoke_sub16
```

已验证结果:

- `phase_reached = stage3sr`
- `psnr = 19.4572 -> 20.3172`
- `residual_mean = 0.063859 -> 0.055581`
- `sr_selection_mean = 0.127218`

## 7. 当前怎么理解 baseline

- 如果你现在做的是 single-view 对比:
  - 正式 baseline 还是 native reference 那条线
- 如果你现在做的是 full-view 联合优化:
  - 当前 48G 主机上的正式起点就是:
    - `target_subsample=16`
    - full-view native smoke
    - 以及同 observation 预算下的 full-view SR smoke
