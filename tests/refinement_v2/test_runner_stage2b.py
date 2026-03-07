"""Stage 2B 测试."""

import json
from pathlib import Path

import torch

from src.refinement_v2.diagnostics import DiagnosticsWriter
from src.refinement_v2.runner import RefinementRunner
from src.refinement_v2.stage_controller import StageController
from tests.refinement_v2.helpers import FakeRenderer, build_gaussian_adapter, build_run_config, build_scene_bundle, build_stage_hparams


class GeometryAwareRenderer:
    """让 Stage 2B 的几何参数拥有稳定梯度的渲染桩."""

    def render(self, gaussians: torch.Tensor, scene) -> dict[str, torch.Tensor]:
        batch_size, _, _ = gaussians.shape
        num_views = scene.gt_images.shape[1]
        height, width = scene.gt_images.shape[-2:]

        means_term = gaussians[:, :, 0:3].mean(dim=1)
        scales_term = gaussians[:, :, 4:7].mean(dim=1)
        rotation_term = gaussians[:, :, 7:10].mean(dim=1)

        # 让颜色主要受 geometry 影响.
        # 这样 Stage 2B 的 `means / rotation / scale` 都能得到真实梯度.
        rgb = torch.sigmoid(1.6 * means_term + 0.8 * scales_term + 0.6 * rotation_term)
        image = rgb[:, None, :, None, None].expand(batch_size, num_views, 3, height, width).clone()
        depth = torch.zeros(batch_size, num_views, 1, height, width, dtype=image.dtype, device=image.device)
        return {"images_pred": image, "depths_pred": depth}


def test_stage2b_optimizer_uses_lower_lr_for_means_than_appearance(tmp_path) -> None:
    """位置学习率应该明显低于外观参数,避免 geometry 跑飞."""

    del tmp_path
    hparams = build_stage_hparams(
        lr_opacity=1e-2,
        lr_color=5e-3,
        lr_scale=1e-3,
        lr_means=1e-4,
    )
    gaussians = build_gaussian_adapter()
    gaussians.freeze_for_stage("stage2b")
    optimizer = gaussians.build_optimizer("stage2b", hparams)

    group_lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert group_lrs["means"] == hparams.lr_means
    assert group_lrs["means"] < group_lrs["colors"]
    assert group_lrs["means"] < group_lrs["opacity"]


def test_stage2b_respects_means_delta_cap(tmp_path) -> None:
    run_config = build_run_config(tmp_path, enable_stage2b=True)
    hparams = build_stage_hparams(iters_stage2b=3, means_delta_cap=0.005)
    gaussians = build_gaussian_adapter()

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
    runner.run_stage2b()

    max_delta = float((gaussians.means.detach() - gaussians.initial_means).abs().max().item())
    assert max_delta <= 0.005 + 1e-8
    assert (run_config.outdir / "metrics_stage2b.json").exists()


def test_stage2b_records_geometry_regularizers_and_updates_geometry(tmp_path) -> None:
    """Stage 2B 应记录 geometry 正则,并允许受限更新位置与旋转."""

    run_config = build_run_config(tmp_path, enable_stage2b=True)
    hparams = build_stage_hparams(
        iters_stage2b=4,
        lr_scale=0.15,
        lr_means=0.08,
        means_delta_cap=0.01,
        lambda_means_anchor=0.2,
        lambda_rotation_reg=0.1,
    )
    gaussians = build_gaussian_adapter()
    means_before = gaussians.means.detach().clone()
    rotations_before = gaussians.rotations.detach().clone()

    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=gaussians,
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=GeometryAwareRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    final_metrics = runner.run_stage2b()

    metrics_history = json.loads((run_config.outdir / "metrics_stage2b.json").read_text(encoding="utf-8"))
    last_metrics = metrics_history[-1]

    assert "loss_means_anchor" in final_metrics
    assert "loss_rotation_reg" in final_metrics
    assert "loss_means_anchor" in last_metrics
    assert "loss_rotation_reg" in last_metrics
    assert not torch.allclose(gaussians.means.detach(), means_before)
    assert not torch.allclose(gaussians.rotations.detach(), rotations_before)
    assert float((gaussians.means.detach() - gaussians.initial_means).abs().max().item()) <= 0.01 + 1e-8
    assert torch.allclose(
        gaussians.rotations.detach().norm(dim=-1),
        torch.ones_like(gaussians.rotations.detach().norm(dim=-1)),
        atol=1e-5,
    )


def test_run_with_start_stage_stage2b_skips_stage2a_optimizer_and_runs_stage2b(tmp_path) -> None:
    """当输入已经是 Stage 2A 基线时,应允许直接进入 Stage 2B workflow."""

    run_config = build_run_config(
        tmp_path,
        start_stage="stage2b",
        enable_stage2b=True,
        stop_after="stage2b",
    )
    hparams = build_stage_hparams(
        iters_stage2a=9,
        iters_stage2b=3,
        lr_scale=0.15,
        lr_means=0.08,
        means_delta_cap=0.01,
    )
    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=GeometryAwareRenderer(),
    )

    summary = runner.run()

    assert summary["start_stage"] == "stage2b"
    assert summary["warm_start_stage2b"] is True
    assert summary["phase_reached"] == "stage2b"
    assert (run_config.outdir / "metrics_stage2b.json").exists()
    assert not (run_config.outdir / "metrics_stage2a.json").exists()


def test_explicit_start_stage2b_bypasses_auto_gate_when_residual_is_already_low(tmp_path) -> None:
    """显式 `start_stage=stage2b` 时,不应再被 `need_geometry` 自动 gate 挡住."""

    run_config = build_run_config(
        tmp_path,
        start_stage="stage2b",
        enable_stage2b=True,
        stop_after="stage2b",
    )
    hparams = build_stage_hparams(
        iters_stage2b=2,
        lr_scale=0.1,
        lr_means=0.05,
        means_delta_cap=0.01,
    )
    scene = build_scene_bundle()
    scene.gt_images = torch.full_like(scene.gt_images, 0.1)
    scene.reference_images = scene.gt_images.clone()

    runner = RefinementRunner(
        scene=scene,
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=FakeRenderer(),
    )

    summary = runner.run()

    assert summary["start_stage"] == "stage2b"
    assert summary["phase_reached"] == "stage2b"
    assert summary["warm_start_stage2b"] is True
    assert (run_config.outdir / "metrics_stage2b.json").exists()


def test_start_stage2b_requires_enable_stage2b(tmp_path) -> None:
    """显式 `start_stage=stage2b` 但没打开 `enable_stage2b` 时,应直接报错."""

    run_config = build_run_config(
        tmp_path,
        start_stage="stage2b",
        enable_stage2b=False,
    )
    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, build_stage_hparams()),
        hparams=build_stage_hparams(),
        renderer=FakeRenderer(),
    )

    try:
        runner.run()
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected start_stage=stage2b without enable_stage2b to raise RuntimeError.")

    assert "enable_stage2b" in message


def test_restore_latest_state_rebuilds_adapter_when_state_and_ply_counts_differ(tmp_path) -> None:
    """resume 时若 state 比 `.ply` 保留了更多高斯,应直接按 state 重建 adapter."""

    run_config = build_run_config(
        tmp_path,
        start_stage="stage2b",
        enable_stage2b=True,
        stop_after="stage2b",
        resume=True,
    )
    hparams = build_stage_hparams(
        iters_stage2b=2,
        lr_scale=0.1,
        lr_means=0.05,
        means_delta_cap=0.01,
    )

    restored_tensor = build_gaussian_adapter(num_points=20).to_tensor().detach().clone()
    state_dir = Path(run_config.outdir) / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage_name": "stage3sr",
            "iter_idx": 0,
            "gaussians": restored_tensor,
            "diagnostics_state": {"phase_reached": "stage3sr"},
            "pose_delta": None,
        },
        state_dir / "latest.pt",
    )

    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(num_points=8),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=FakeRenderer(),
    )

    restored = runner.restore_latest_state()
    assert restored is True
    assert tuple(runner.gaussians.to_tensor().shape) == tuple(restored_tensor.shape)

    summary = runner.run()

    assert summary["start_stage"] == "stage2b"
    assert summary["phase_reached"] == "stage2b"
    assert summary["warm_start_stage2b"] is True
    assert (run_config.outdir / "metrics_stage2b.json").exists()
