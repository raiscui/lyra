"""depth anchor 回归测试."""

from __future__ import annotations

import json

import pytest
import torch

from src.refinement_v2.diagnostics import DiagnosticsWriter
from src.refinement_v2.losses import build_depth_anchor_valid_mask, compute_depth_anchor_loss
from src.refinement_v2.runner import RefinementRunner
from src.refinement_v2.stage_controller import StageController
from tests.refinement_v2.helpers import FakeRenderer, build_gaussian_adapter, build_run_config, build_scene_bundle, build_stage_hparams


class DepthAwareRenderer:
    """提供正深度与 alpha 的可微渲染桩."""

    def __init__(self) -> None:
        self.render_calls = 0
        self.view_counts: list[int] = []

    def render(self, gaussians: torch.Tensor, scene) -> dict[str, torch.Tensor]:
        self.render_calls += 1
        self.view_counts.append(scene.gt_images.shape[1])

        batch_size, _, _ = gaussians.shape
        num_views = scene.gt_images.shape[1]
        height, width = scene.gt_images.shape[-2:]

        color_mean = gaussians[:, :, 11:].mean(dim=1)
        opacity_mean = gaussians[:, :, 3].mean(dim=1).view(batch_size, 1, 1, 1, 1)
        view_bias = torch.arange(num_views, device=gaussians.device, dtype=gaussians.dtype).view(1, num_views, 1, 1, 1) / 10.0

        image = color_mean[:, None, :, None, None].expand(batch_size, num_views, 3, height, width).clone() + view_bias

        # depth 明确依赖 opacity.
        # 这样 depth anchor 在 appearance 阶段不是“算了一个常数”,而是真的有梯度路径.
        depth = (0.5 + opacity_mean + view_bias).expand(batch_size, num_views, 1, height, width).clone()
        alpha = torch.sigmoid(opacity_mean * 4.0).expand(batch_size, num_views, 1, height, width).clone()
        return {"images_pred": image, "depths_pred": depth, "alphas_pred": alpha}


class MissingDepthAfterCaptureRenderer(DepthAwareRenderer):
    """让 reference capture 成功,但 stage 内部故意缺失 pred depth."""

    def render(self, gaussians: torch.Tensor, scene) -> dict[str, torch.Tensor]:
        output = super().render(gaussians, scene)
        if self.render_calls >= 4:
            output.pop("depths_pred", None)
        return output


class ZeroAlphaRenderer(DepthAwareRenderer):
    """让 reference alpha 始终为 0, 验证空 mask 的 graceful fallback."""

    def render(self, gaussians: torch.Tensor, scene) -> dict[str, torch.Tensor]:
        output = super().render(gaussians, scene)
        output["alphas_pred"] = torch.zeros_like(output["depths_pred"])
        return output


class CaptureCountingRunner(RefinementRunner):
    """记录 baseline depth anchor 实际捕获了几次."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.depth_anchor_capture_calls = 0

    def _capture_depth_anchor_reference(self) -> None:
        self.depth_anchor_capture_calls += 1
        super()._capture_depth_anchor_reference()


def test_build_depth_anchor_valid_mask_respects_alpha_threshold() -> None:
    reference_depth = torch.tensor([[[[[1.0, 0.0], [2.0, 3.0]]]]], dtype=torch.float32)
    reference_alpha = torch.tensor([[[[[0.9, 0.9], [0.01, 0.5]]]]], dtype=torch.float32)

    valid_mask = build_depth_anchor_valid_mask(
        reference_depth,
        reference_alpha,
        alpha_threshold=0.05,
    )

    assert valid_mask.dtype == torch.bool
    assert valid_mask.tolist() == [[[[[True, False], [False, True]]]]]


def test_compute_depth_anchor_loss_is_scale_invariant() -> None:
    reference_depth = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]]]]], dtype=torch.float32)
    pred_depth = reference_depth * 7.0 + 3.0
    valid_mask = torch.ones_like(reference_depth, dtype=torch.bool)

    summary = compute_depth_anchor_loss(pred_depth, reference_depth, valid_mask)

    assert summary.skip_reason is None
    assert summary.valid_ratio == pytest.approx(1.0)
    assert float(summary.loss.item()) == pytest.approx(0.0, abs=1e-6)


def test_compute_depth_anchor_loss_skips_empty_mask_without_nan() -> None:
    reference_depth = torch.ones((1, 1, 1, 2, 2), dtype=torch.float32)
    pred_depth = torch.full_like(reference_depth, 2.0)
    valid_mask = torch.zeros_like(reference_depth, dtype=torch.bool)

    summary = compute_depth_anchor_loss(pred_depth, reference_depth, valid_mask)

    assert summary.skip_reason == "empty_valid_mask"
    assert summary.valid_ratio == 0.0
    assert torch.isfinite(summary.loss)
    assert float(summary.loss.item()) == 0.0


def test_stage2a_and_stage3sr_reuse_same_depth_anchor_reference(tmp_path) -> None:
    run_config = build_run_config(tmp_path, stage2a_mode="enhanced")
    hparams = build_stage_hparams(
        iters_stage2a=2,
        patch_size=4,
        lambda_patch_rgb=0.5,
        enable_depth_anchor=True,
        lambda_depth_anchor=0.25,
    )
    renderer = DepthAwareRenderer()
    runner = CaptureCountingRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=renderer,
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    runner.run_stage2a()

    stage2a_metrics = json.loads((run_config.outdir / "metrics_stage2a.json").read_text(encoding="utf-8"))
    stage3sr_metrics = json.loads((run_config.outdir / "metrics_stage3sr.json").read_text(encoding="utf-8"))

    assert runner.depth_anchor_capture_calls == 1
    assert runner.depth_anchor_reference is not None
    assert runner.diagnostics_state["depth_anchor_reference_ready"] is True
    assert stage2a_metrics[-1]["loss_depth_anchor"] >= 0.0
    assert stage2a_metrics[-1]["depth_anchor_skip_reason"] is None
    assert stage3sr_metrics[-1]["loss_depth_anchor"] >= 0.0
    assert stage3sr_metrics[-1]["depth_anchor_skip_reason"] is None


def test_depth_anchor_runs_under_multi_device_view_shards(tmp_path) -> None:
    run_config = build_run_config(tmp_path, render_devices=["cpu", "cpu"])
    hparams = build_stage_hparams(
        iters_stage2a=2,
        enable_depth_anchor=True,
        lambda_depth_anchor=0.25,
    )
    renderer = DepthAwareRenderer()
    runner = CaptureCountingRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=renderer,
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    runner.run_stage2a()

    stage2a_metrics = json.loads((run_config.outdir / "metrics_stage2a.json").read_text(encoding="utf-8"))

    assert runner.depth_anchor_capture_calls == 1
    assert stage2a_metrics[-1]["loss_depth_anchor"] >= 0.0
    assert stage2a_metrics[-1]["depth_anchor_skip_reason"] is None
    assert 3 not in renderer.view_counts
    assert set(renderer.view_counts) == {1, 2}


def test_stage2b_metrics_do_not_include_depth_anchor(tmp_path) -> None:
    run_config = build_run_config(tmp_path, enable_stage2b=True)
    hparams = build_stage_hparams(
        iters_stage2a=2,
        iters_stage2b=2,
        enable_depth_anchor=True,
        lambda_depth_anchor=0.25,
    )
    runner = CaptureCountingRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=DepthAwareRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    runner.run_stage2a()
    runner.run_stage2b()

    stage2b_metrics = json.loads((run_config.outdir / "metrics_stage2b.json").read_text(encoding="utf-8"))

    assert "loss_depth_anchor" not in stage2b_metrics[-1]
    assert "depth_anchor_skip_reason" not in stage2b_metrics[-1]


def test_depth_anchor_missing_pred_depth_warns_but_does_not_crash(tmp_path) -> None:
    run_config = build_run_config(tmp_path)
    hparams = build_stage_hparams(
        iters_stage2a=2,
        enable_depth_anchor=True,
        lambda_depth_anchor=0.25,
    )
    runner = CaptureCountingRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=MissingDepthAfterCaptureRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    metrics = runner.run_stage2a()

    assert metrics["depth_anchor_skip_reason"] == "pred_depth_missing"
    assert "stage2a_depth_anchor_skipped:pred_depth_missing" in runner.diagnostics_state.get("warnings", [])
    assert (run_config.outdir / "metrics_stage2a.json").exists()


def test_depth_anchor_empty_reference_mask_warns_but_does_not_crash(tmp_path) -> None:
    run_config = build_run_config(tmp_path)
    hparams = build_stage_hparams(
        iters_stage2a=2,
        enable_depth_anchor=True,
        lambda_depth_anchor=0.25,
    )
    runner = CaptureCountingRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=ZeroAlphaRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    metrics = runner.run_stage2a()

    assert runner.depth_anchor_reference is None
    assert metrics["depth_anchor_skip_reason"] == "empty_reference_mask"
    assert "depth_anchor_skipped:empty_reference_mask" in runner.diagnostics_state.get("warnings", [])
    assert (run_config.outdir / "metrics_stage2a.json").exists()


def test_stage2b_depth_anchor_disabled_path_still_runs_with_legacy_fake_renderer(tmp_path) -> None:
    """补一个最弱回归,确保旧 renderer 不会因为新字段直接炸掉."""

    run_config = build_run_config(tmp_path, enable_stage2b=True)
    hparams = build_stage_hparams(
        iters_stage2a=1,
        iters_stage2b=1,
        enable_depth_anchor=True,
        lambda_depth_anchor=0.25,
    )
    runner = CaptureCountingRunner(
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
    runner.run_stage2b()

    assert runner.diagnostics_state["depth_anchor_reference_skip_reason"] == "empty_reference_mask"
    assert (run_config.outdir / "metrics_stage2b.json").exists()
