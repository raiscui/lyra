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
    assert run_config.scene_index == 0
    assert run_config.dataset_name is None
    assert run_config.frame_indices is None
    assert run_config.enable_stage2b is False
    assert run_config.dry_run is False
    assert hparams.weight_floor == 0.20
    assert hparams.iters_stage2a == 600


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
    assert run_config.enable_stage2b is True
    assert run_config.enable_pose_diagnostic is True
    assert run_config.enable_joint_fallback is True
    assert run_config.resume is True
    assert run_config.dry_run is True
    assert hparams.weight_floor == 0.3
    assert hparams.iters_stage2a == 111


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
            "--lambda-patch-rgb",
            "0.5",
            "--lambda-patch-perceptual",
            "0.1",
        ]
    )

    assert run_config.reference_mode == "super_resolved"
    assert run_config.sr_scale == 2.0
    assert hparams.patch_size == 256
    assert hparams.lambda_patch_rgb == 0.5
    assert hparams.lambda_patch_perceptual == 0.1
