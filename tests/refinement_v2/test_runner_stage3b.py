"""Stage 3B 测试."""

import json

import pytest
import torch

from src.refinement_v2.diagnostics import DiagnosticsWriter
from src.refinement_v2.runner import RefinementRunner
from src.refinement_v2.stage_controller import StageController
from tests.refinement_v2.helpers import build_gaussian_adapter, build_run_config, build_scene_bundle, build_stage_hparams


class GeometryAwareRenderer:
    """让 Stage 3B 的几何参数拥有稳定梯度的渲染桩."""

    def render(self, gaussians: torch.Tensor, scene) -> dict[str, torch.Tensor]:
        batch_size, _, _ = gaussians.shape
        num_views = scene.gt_images.shape[1]
        height, width = scene.gt_images.shape[-2:]

        means_term = gaussians[:, :, 0:3].mean(dim=1)
        scales_term = gaussians[:, :, 4:7].mean(dim=1)
        rotation_term = gaussians[:, :, 7:10].mean(dim=1)

        # 让颜色显式依赖几何项.
        # 这样 `stage3b` 里的 HR / LR supervision 才会真的把梯度传到位置和旋转.
        rgb = torch.sigmoid(1.6 * means_term + 0.8 * scales_term + 0.6 * rotation_term)
        image = rgb[:, None, :, None, None].expand(batch_size, num_views, 3, height, width).clone()
        depth = torch.zeros(batch_size, num_views, 1, height, width, dtype=image.dtype, device=image.device)
        return {"images_pred": image, "depths_pred": depth}


def test_stage3b_records_geometry_regularizers_and_updates_geometry(tmp_path) -> None:
    """Stage 3B 应记录 geometry 正则,并允许受限更新位置与旋转."""

    run_config = build_run_config(tmp_path, enable_stage3b=True, stage2a_mode="enhanced")
    hparams = build_stage_hparams(
        iters_stage2a=2,
        iters_stage2b=1,
        iters_stage3b=4,
        lr_scale=0.15,
        lr_means=0.08,
        means_delta_cap=0.25,
        means_delta_cap_stage3b=0.01,
        lambda_means_anchor=0.0,
        lambda_means_anchor_stage3b=0.2,
        lambda_rotation_reg=0.0,
        lambda_rotation_reg_stage3b=0.1,
        lambda_hr_rgb=1.0,
        lambda_lr_consistency=0.5,
        reference_render_shard_views=1,
    )
    gaussians = build_gaussian_adapter()
    means_before = gaussians.means.detach().clone()
    rotations_before = gaussians.rotations.detach().clone()

    runner = RefinementRunner(
        scene=build_scene_bundle(reference_mode="super_resolved", sr_scale=2.0),
        gaussians=gaussians,
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=GeometryAwareRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    runner.run_stage2a()
    final_metrics = runner.run_stage3b()

    metrics_history = json.loads((run_config.outdir / "metrics_stage3b.json").read_text(encoding="utf-8"))
    last_metrics = metrics_history[-1]

    assert "loss_means_anchor" in final_metrics
    assert "loss_rotation_reg" in final_metrics
    assert "loss_means_anchor" in last_metrics
    assert "loss_rotation_reg" in last_metrics
    assert len(metrics_history) == 4
    assert final_metrics["iters_budget"] == 4
    assert final_metrics["lambda_means_anchor_active"] == 0.2
    assert final_metrics["lambda_rotation_reg_active"] == 0.1
    assert final_metrics["means_delta_cap_active"] == 0.01
    assert final_metrics["stage3sr_supervision_mode"] == "full_frame_hr"
    assert runner.diagnostics_state["stage3b_completed"] is True
    assert not torch.allclose(gaussians.means.detach(), means_before)
    assert not torch.allclose(gaussians.rotations.detach(), rotations_before)
    assert float((gaussians.means.detach() - gaussians.initial_means).abs().max().item()) <= 0.01 + 1e-8
    assert torch.allclose(
        gaussians.rotations.detach().norm(dim=-1),
        torch.ones_like(gaussians.rotations.detach().norm(dim=-1)),
        atol=1e-5,
    )


def test_run_auto_enters_stage3b_after_stage3sr_when_enabled(tmp_path) -> None:
    """开启 `enable_stage3b` 后, full run 应在 `stage3sr` 之后自动进入 `stage3b`."""

    run_config = build_run_config(
        tmp_path,
        enable_stage3b=True,
        stage2a_mode="enhanced",
        stop_after="stage3b",
    )
    hparams = build_stage_hparams(
        iters_stage2a=2,
        iters_stage2b=2,
        iters_stage3b=2,
        lr_scale=0.15,
        lr_means=0.08,
        means_delta_cap=0.01,
        means_delta_cap_stage3b=0.01,
        lambda_means_anchor=0.2,
        lambda_means_anchor_stage3b=0.2,
        lambda_rotation_reg=0.1,
        lambda_rotation_reg_stage3b=0.1,
        lambda_hr_rgb=1.0,
        lambda_lr_consistency=0.5,
        reference_render_shard_views=1,
    )

    runner = RefinementRunner(
        scene=build_scene_bundle(reference_mode="super_resolved", sr_scale=2.0),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=GeometryAwareRenderer(),
    )

    summary = runner.run()

    assert summary["phase_reached"] == "stage3b"
    assert (run_config.outdir / "metrics_stage3b.json").exists()
    assert not (run_config.outdir / "metrics_stage2b.json").exists()
    assert (run_config.outdir / "gaussians" / "gaussians_stage3b.ply").exists()


def test_run_with_start_stage_stage3b_skips_stage2a_and_runs_stage3b(tmp_path) -> None:
    """显式 `start_stage=stage3b` 时,应直接走 warm-start `stage3b` workflow."""

    run_config = build_run_config(
        tmp_path,
        start_stage="stage3b",
        enable_stage3b=True,
        stop_after="stage3b",
    )
    hparams = build_stage_hparams(
        iters_stage3b=3,
        lr_scale=0.15,
        lr_means=0.08,
        means_delta_cap_stage3b=0.01,
        lambda_means_anchor_stage3b=0.2,
        lambda_rotation_reg_stage3b=0.1,
        lambda_hr_rgb=1.0,
        lambda_lr_consistency=0.5,
        reference_render_shard_views=1,
    )

    runner = RefinementRunner(
        scene=build_scene_bundle(reference_mode="super_resolved", sr_scale=2.0),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=GeometryAwareRenderer(),
    )

    summary = runner.run()

    assert summary["start_stage"] == "stage3b"
    assert summary["warm_start_stage3b"] is True
    assert summary["phase_reached"] == "stage3b"
    assert (run_config.outdir / "metrics_stage3b.json").exists()
    assert (run_config.outdir / "metrics_phase3s.json").exists()
    assert not (run_config.outdir / "metrics_stage2a.json").exists()


def test_start_stage_stage3b_requires_enable_stage3b(tmp_path) -> None:
    """显式 `start_stage=stage3b` 但没打开 `enable_stage3b` 时,应直接报错."""

    run_config = build_run_config(
        tmp_path,
        start_stage="stage3b",
        enable_stage3b=False,
    )
    runner = RefinementRunner(
        scene=build_scene_bundle(reference_mode="super_resolved", sr_scale=2.0),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, build_stage_hparams()),
        hparams=build_stage_hparams(),
        renderer=GeometryAwareRenderer(),
    )

    with pytest.raises(RuntimeError, match="enable_stage3b"):
        runner.run()
