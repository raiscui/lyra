"""配置层测试."""

from src.refinement_v2.config import load_effective_config_from_cli


def test_cli_mapping_uses_defaults() -> None:
    run_config, hparams = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
        ]
    )

    assert str(run_config.config_path) == "configs/demo/lyra_static.yaml"
    assert str(run_config.gaussians_path) == "outputs/demo/gaussians_0.ply"
    assert str(run_config.outdir) == "outputs/refine_v2/test"
    assert run_config.scene_stem is None
    assert run_config.pose_path is None
    assert run_config.intrinsics_path is None
    assert run_config.rgb_path is None
    assert run_config.pose_root is None
    assert run_config.intrinsics_root is None
    assert run_config.rgb_root is None
    assert run_config.scene_index == 0
    assert run_config.dataset_name is None
    assert run_config.view_ids is None
    assert run_config.frame_indices is None
    assert run_config.reference_path is None
    assert run_config.reference_root is None
    assert run_config.reference_intrinsics_path is None
    assert run_config.start_stage == "stage2a"
    assert run_config.stage2a_mode == "auto"
    assert run_config.render_devices is None
    assert run_config.enable_stage3b is False
    assert run_config.enable_stage2b is False
    assert run_config.dry_run is False
    assert hparams.weight_floor == 0.20
    assert hparams.iters_stage2a == 600
    assert hparams.iters_stage3b == 300
    assert hparams.lambda_means_anchor == 0.01
    assert hparams.lambda_means_anchor_stage3b == 0.01
    assert hparams.lambda_rotation_reg == 0.01
    assert hparams.lambda_rotation_reg_stage3b == 0.01
    assert hparams.means_delta_cap == 0.02
    assert hparams.means_delta_cap_stage3b == 0.02
    assert hparams.lambda_sampling_smooth == 5e-4
    assert hparams.sampling_radius_threshold == 1.5
    assert hparams.fidelity_ratio_threshold == 1.5
    assert hparams.fidelity_sigmoid_k == 6.0
    assert hparams.fidelity_min_views == 3
    assert hparams.fidelity_opacity_threshold == 1e-4


def test_cli_mapping_reads_bool_flags_and_frame_indices() -> None:
    run_config, hparams = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
            "--scene-index",
            "3",
            "--dataset-name",
            "lyra_static_demo_generated",
            "--view-id",
            "5",
            "--frame-indices",
            "1,3,7",
            "--start-stage",
            "stage2b",
            "--stage2a-mode",
            "enhanced",
            "--enable-stage3b",
            "--enable-stage2b",
            "--enable-pose-diagnostic",
            "--enable-joint-fallback",
            "--resume",
            "--dry-run",
            "--weight-floor",
            "0.3",
            "--iters-stage2a",
            "111",
        ]
    )

    assert run_config.scene_index == 3
    assert run_config.dataset_name == "lyra_static_demo_generated"
    assert run_config.view_id == "5"
    assert run_config.frame_indices == [1, 3, 7]
    assert run_config.start_stage == "stage2b"
    assert run_config.stage2a_mode == "enhanced"
    assert run_config.enable_stage3b is True
    assert run_config.enable_stage2b is True
    assert run_config.enable_pose_diagnostic is True
    assert run_config.enable_joint_fallback is True
    assert run_config.resume is True
    assert run_config.dry_run is True
    assert hparams.weight_floor == 0.3
    assert hparams.iters_stage2a == 111


def test_cli_mapping_reads_start_stage_stage3b() -> None:
    """确认 `start_stage=stage3b` 能稳定映射到配置对象."""

    run_config, _ = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
            "--start-stage",
            "stage3b",
            "--enable-stage3b",
        ]
    )

    assert run_config.start_stage == "stage3b"
    assert run_config.enable_stage3b is True


def test_cli_mapping_reads_render_devices() -> None:
    """确认多设备渲染入口能稳定映射到配置对象."""

    run_config, _ = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
            "--device",
            "cuda:0",
            "--render-devices",
            "cuda:0,cuda:1",
        ]
    )

    assert run_config.device == "cuda:0"
    assert run_config.render_devices == ["cuda:0", "cuda:1"]


def test_cli_mapping_reads_pruning_flags() -> None:
    """确认 pruning 开关和超参数都能从 CLI 正确映射."""

    run_config, hparams = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
            "--enable-pruning",
            "--opacity-prune-threshold",
            "0.07",
            "--prune-every",
            "5",
            "--prune-warmup-iters",
            "8",
            "--prune-max-fraction",
            "0.15",
            "--min-gaussians-to-keep",
            "64",
        ]
    )

    assert run_config.enable_pruning is True
    assert hparams.opacity_prune_threshold == 0.07
    assert hparams.prune_every == 5
    assert hparams.prune_warmup_iters == 8
    assert hparams.prune_max_fraction == 0.15
    assert hparams.min_gaussians_to_keep == 64


def test_cli_mapping_reads_patch_supervision_flags() -> None:
    """确认 patch supervision 相关参数能稳定映射到配置对象."""

    run_config, hparams = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
            "--reference-mode",
            "super_resolved",
            "--sr-scale",
            "2.0",
            "--patch-size",
            "256",
            "--sr-patches-per-view",
            "4",
            "--lambda-patch-rgb",
            "0.5",
            "--lambda-patch-perceptual",
            "0.1",
        ]
    )

    assert run_config.reference_mode == "super_resolved"
    assert run_config.sr_scale == 2.0
    assert hparams.patch_size == 256
    assert hparams.sr_patches_per_view == 4
    assert hparams.lambda_patch_rgb == 0.5
    assert hparams.lambda_patch_perceptual == 0.1


def test_cli_mapping_reads_fidelity_supervision_flags() -> None:
    """确认 fidelity 相关超参数能稳定映射到配置对象."""

    _, hparams = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
            "--fidelity-ratio-threshold",
            "1.8",
            "--fidelity-sigmoid-k",
            "9.5",
            "--fidelity-min-views",
            "5",
            "--fidelity-opacity-threshold",
            "0.02",
        ]
    )

    assert hparams.fidelity_ratio_threshold == 1.8
    assert hparams.fidelity_sigmoid_k == 9.5
    assert hparams.fidelity_min_views == 5
    assert hparams.fidelity_opacity_threshold == 0.02


def test_cli_mapping_reads_phase_c_full_frame_hr_flags() -> None:
    """确认 Phase C 的 full-frame HR 参数能稳定映射到配置对象."""

    _, hparams = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
            "--lambda-hr-rgb",
            "0.75",
            "--lambda-lr-consistency",
            "1.25",
            "--reference-render-shard-views",
            "3",
        ]
    )

    assert hparams.lambda_hr_rgb == 0.75
    assert hparams.lambda_lr_consistency == 1.25
    assert hparams.reference_render_shard_views == 3


def test_cli_mapping_reads_external_reference_flags() -> None:
    """确认 external reference 相关路径能稳定映射到配置对象."""

    run_config, _ = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
            "--reference-mode",
            "super_resolved",
            "--reference-path",
            "assets/demo/static/diffusion_output_generated/5/rgb",
            "--reference-intrinsics-path",
            "assets/demo/static/diffusion_output_generated/5/intrinsics/demo.npz",
        ]
    )

    assert str(run_config.reference_path) == "assets/demo/static/diffusion_output_generated/5/rgb"
    assert str(run_config.reference_intrinsics_path) == "assets/demo/static/diffusion_output_generated/5/intrinsics/demo.npz"


def test_cli_mapping_reads_direct_file_input_flags() -> None:
    """确认 direct file inputs 能稳定映射到配置对象."""

    run_config, _ = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
            "--pose-path",
            "assets/demo/static/diffusion_output_generated/3/pose/00172.npz",
            "--intrinsics-path",
            "assets/demo/static/diffusion_output_generated/3/intrinsics/00172.npz",
            "--rgb-path",
            "assets/demo/static/diffusion_output_generated/3/rgb/00172.mp4",
        ]
    )

    assert str(run_config.pose_path) == "assets/demo/static/diffusion_output_generated/3/pose/00172.npz"
    assert str(run_config.intrinsics_path) == "assets/demo/static/diffusion_output_generated/3/intrinsics/00172.npz"
    assert str(run_config.rgb_path) == "assets/demo/static/diffusion_output_generated/3/rgb/00172.mp4"


def test_cli_mapping_reads_full_view_root_flags() -> None:
    """确认 full-view root inputs 能稳定映射到配置对象."""

    run_config, _ = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
            "--scene-stem",
            "00172",
            "--view-ids",
            "5,0,1,2,3,4",
            "--pose-root",
            "assets/demo/static/diffusion_output_generated",
            "--intrinsics-root",
            "assets/demo/static/diffusion_output_generated",
            "--rgb-root",
            "assets/demo/static/diffusion_output_generated",
            "--reference-root",
            "outputs/flashvsr_reference/full_scale2x",
        ]
    )

    assert run_config.scene_stem == "00172"
    assert run_config.view_ids == ["5", "0", "1", "2", "3", "4"]
    assert str(run_config.pose_root) == "assets/demo/static/diffusion_output_generated"
    assert str(run_config.intrinsics_root) == "assets/demo/static/diffusion_output_generated"
    assert str(run_config.rgb_root) == "assets/demo/static/diffusion_output_generated"
    assert str(run_config.reference_root) == "outputs/flashvsr_reference/full_scale2x"


def test_cli_mapping_reads_stage2b_regularizer_flags() -> None:
    """确认 Stage 2B 的几何正则权重能从 CLI 映射进来."""

    _, hparams = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
            "--lambda-means-anchor",
            "0.2",
            "--lambda-rotation-reg",
            "0.05",
            "--means-delta-cap",
            "0.03",
        ]
    )

    assert hparams.lambda_means_anchor == 0.2
    assert hparams.lambda_rotation_reg == 0.05
    assert hparams.means_delta_cap == 0.03


def test_cli_mapping_reads_stage3b_dedicated_geometry_flags() -> None:
    """确认 `stage3b` 可以覆盖 Stage 2B 的共享 geometry 配置."""

    _, hparams = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
            "--iters-stage2b",
            "7",
            "--iters-stage3b",
            "11",
            "--lambda-means-anchor",
            "0.2",
            "--lambda-means-anchor-stage3b",
            "0.6",
            "--lambda-rotation-reg",
            "0.05",
            "--lambda-rotation-reg-stage3b",
            "0.12",
            "--means-delta-cap",
            "0.03",
            "--means-delta-cap-stage3b",
            "0.09",
        ]
    )

    assert hparams.iters_stage2b == 7
    assert hparams.iters_stage3b == 11
    assert hparams.lambda_means_anchor == 0.2
    assert hparams.lambda_means_anchor_stage3b == 0.6
    assert hparams.lambda_rotation_reg == 0.05
    assert hparams.lambda_rotation_reg_stage3b == 0.12
    assert hparams.means_delta_cap == 0.03
    assert hparams.means_delta_cap_stage3b == 0.09


def test_cli_mapping_reads_sampling_smooth_flags() -> None:
    """确认 sampling smooth 相关参数能从 CLI 映射进来."""

    _, hparams = load_effective_config_from_cli(
        [
            "--config",
            "configs/demo/lyra_static.yaml",
            "--gaussians",
            "outputs/demo/gaussians_0.ply",
            "--outdir",
            "outputs/refine_v2/test",
            "--lambda-sampling-smooth",
            "0.002",
            "--sampling-radius-threshold",
            "2.5",
        ]
    )

    assert hparams.lambda_sampling_smooth == 0.002
    assert hparams.sampling_radius_threshold == 2.5
