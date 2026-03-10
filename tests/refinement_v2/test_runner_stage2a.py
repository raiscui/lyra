"""Stage 2A 测试."""

import json

import pytest
import torch

from src.refinement_v2.diagnostics import DiagnosticsWriter
from src.refinement_v2.runner import RefinementRunner
from src.refinement_v2.stage_controller import StageController
from tests.refinement_v2.helpers import FakeRenderer, build_gaussian_adapter, build_run_config, build_scene_bundle, build_stage_hparams


def test_stage2a_updates_colors_but_not_means(tmp_path) -> None:
    run_config = build_run_config(tmp_path)
    hparams = build_stage_hparams(iters_stage2a=4)
    gaussians = build_gaussian_adapter()
    colors_before = gaussians.colors.detach().clone()
    means_before = gaussians.means.detach().clone()

    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=gaussians,
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=FakeRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    runner.run_stage2a()

    assert torch.allclose(gaussians.means.detach(), means_before)
    assert not torch.allclose(gaussians.colors.detach(), colors_before)
    assert runner.diagnostics_state["stage2a_mode_resolved"] == "legacy"
    assert (run_config.outdir / "metrics_stage2a.json").exists()


def test_runner_wires_fidelity_hparams_into_weight_builder(tmp_path) -> None:
    """runner 初始化时应把 fidelity 参数完整传给 WeightBuilder."""

    run_config = build_run_config(tmp_path)
    hparams = build_stage_hparams(
        fidelity_ratio_threshold=1.9,
        fidelity_sigmoid_k=8.5,
        fidelity_min_views=5,
        fidelity_opacity_threshold=0.03,
    )

    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=FakeRenderer(),
    )

    assert runner.weight_builder.fidelity_ratio_threshold == 1.9
    assert runner.weight_builder.fidelity_sigmoid_k == 8.5
    assert runner.weight_builder.fidelity_min_views == 5
    assert runner.weight_builder.fidelity_opacity_threshold == 0.03


def test_runner_run_exports_final_render_artifacts_at_stage2a_stop(tmp_path) -> None:
    run_config = build_run_config(tmp_path, stop_after="stage2a")
    hparams = build_stage_hparams(iters_stage2a=2)

    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=FakeRenderer(),
    )

    summary = runner.run()
    diagnostics = json.loads((run_config.outdir / "diagnostics.json").read_text(encoding="utf-8"))

    assert summary["phase_reached"] == "stage2a"
    assert diagnostics["phase_reached"] == "stage2a"
    assert (run_config.outdir / "videos" / "final_render.mp4").exists()
    assert (run_config.outdir / "renders_before_after" / "final_render_frame_0000.png").exists()


def test_runner_run_exports_final_hr_render_artifacts_at_stage2a_stop(tmp_path) -> None:
    """super-resolved scene 在 stop_after=stage2a 时也应补出 HR after 导出."""

    run_config = build_run_config(
        tmp_path,
        stop_after="stage2a",
        reference_mode="super_resolved",
        sr_scale=2.0,
        stage2a_mode="enhanced",
    )
    hparams = build_stage_hparams(
        iters_stage2a=2,
        lambda_hr_rgb=0.5,
        lambda_lr_consistency=1.0,
        reference_render_shard_views=1,
    )

    runner = RefinementRunner(
        scene=build_scene_bundle(reference_mode="super_resolved", sr_scale=2.0),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=FakeRenderer(),
    )

    summary = runner.run()

    assert summary["phase_reached"] == "stage3sr"
    assert (run_config.outdir / "videos" / "final_render_hr.mp4").exists()
    assert (run_config.outdir / "renders_before_after" / "final_render_hr_frame_0000.png").exists()
    assert "final_hr" in summary
    assert "final_render_hr_video" in summary["artifacts"]


def test_stage2a_splits_native_cleanup_and_stage3sr_metrics(tmp_path) -> None:
    """开启 patch supervision 后, native cleanup 和 Stage 3SR 应分开留痕."""

    run_config = build_run_config(tmp_path, stage2a_mode="enhanced")
    hparams = build_stage_hparams(iters_stage2a=2, patch_size=4, lambda_patch_rgb=0.5)
    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=FakeRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    runner.run_stage2a()

    stage2a_metrics = json.loads((run_config.outdir / "metrics_stage2a.json").read_text(encoding="utf-8"))
    stage3sr_metrics = json.loads((run_config.outdir / "metrics_stage3sr.json").read_text(encoding="utf-8"))

    assert "loss_patch_rgb" not in stage2a_metrics[-1]
    assert "loss_patch_rgb" in stage3sr_metrics[-1]
    assert runner.diagnostics_state["stage2a_mode_resolved"] == "enhanced"


def test_stage2a_mode_legacy_skips_stage3sr_even_when_patch_configured(tmp_path) -> None:
    """显式 legacy 模式下,即便 patch 参数已给也不进入 Stage 3SR."""

    run_config = build_run_config(tmp_path, stage2a_mode="legacy")
    hparams = build_stage_hparams(iters_stage2a=2, patch_size=4, lambda_patch_rgb=0.5)
    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=FakeRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    runner.run_stage2a()

    assert runner.diagnostics_state["stage2a_mode_resolved"] == "legacy"
    assert runner.diagnostics_state["stage3sr_enabled"] is False
    assert "stage2a_mode_legacy_skipped_stage3sr_supervision" in runner.diagnostics_state.get("warnings", [])
    assert (run_config.outdir / "metrics_stage2a.json").exists()
    assert not (run_config.outdir / "metrics_stage3sr.json").exists()


def test_stage2a_mode_enhanced_requires_stage3sr_supervision(tmp_path) -> None:
    """显式 enhanced 模式下,缺少任何 Stage 3SR 监督都应直接报错."""

    run_config = build_run_config(tmp_path, stage2a_mode="enhanced")
    hparams = build_stage_hparams(iters_stage2a=2)
    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=FakeRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()

    with pytest.raises(RuntimeError, match="stage2a_mode=enhanced requires Stage 3SR supervision"):
        runner.run_stage2a()
