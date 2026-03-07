
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