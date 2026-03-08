"""patch-based supervision 测试."""

from __future__ import annotations

import json

import torch

from src.refinement_v2.diagnostics import DiagnosticsWriter
from src.refinement_v2.runner import RefinementRunner
from src.refinement_v2.stage_controller import StageController
from tests.refinement_v2.helpers import build_gaussian_adapter, build_run_config, build_stage_hparams


class RecordingRenderer:
    """记录 patch 渲染时收到的 scene 信息."""

    def __init__(self) -> None:
        self.calls: list[dict[str, torch.Tensor | tuple[int, int]]] = []

    def render(self, gaussians: torch.Tensor, scene) -> dict[str, torch.Tensor]:
        self.calls.append(
            {
                "intrinsics": scene.intrinsics.detach().clone(),
                "hw": scene.gt_images.shape[-2:],
            }
        )
        batch_size, _, _ = gaussians.shape
        num_views = scene.gt_images.shape[1]
        height, width = scene.gt_images.shape[-2:]

        # 这里保留可微性, 这样同一个 renderer 既能做 phase3s 诊断,
        # 也能直接跑 stage3sr 的优化循环测试.
        color_mean = gaussians[:, :, 11:].mean(dim=1)
        image = color_mean[:, None, :, None, None].expand(batch_size, num_views, 3, height, width).clone()
        depth = torch.zeros(
            scene.gt_images.shape[0],
            scene.gt_images.shape[1],
            1,
            *scene.gt_images.shape[-2:],
            dtype=image.dtype,
            device=image.device,
        )
        return {"images_pred": image, "depths_pred": depth}


class MetaRecordingRenderer(RecordingRenderer):
    """额外返回 dense render meta,用于验证 Phase 3S."""

    def render(self, gaussians: torch.Tensor, scene) -> dict[str, torch.Tensor]:
        output = super().render(gaussians, scene)
        batch_size, num_gaussians, _ = gaussians.shape
        num_views = scene.gt_images.shape[1]
        height, width = scene.gt_images.shape[-2:]
        radii_scalar = (
            torch.linspace(1.0, 2.0, steps=num_gaussians, dtype=gaussians.dtype)
            .view(1, 1, num_gaussians)
            .repeat(batch_size, num_views, 1)
        )
        radii = torch.stack([radii_scalar, radii_scalar * 0.8], dim=-1)
        tiles_per_gauss = (
            torch.linspace(2.0, 4.0, steps=num_gaussians, dtype=gaussians.dtype)
            .view(1, 1, num_gaussians)
            .repeat(batch_size, num_views, 1)
        )
        opacities = gaussians[:, :, 3].unsqueeze(1).expand(batch_size, num_views, num_gaussians).detach().clone()
        means_x = torch.linspace(0.5, max(width - 1.5, 0.5), steps=num_gaussians, dtype=gaussians.dtype)
        means_y = torch.linspace(0.5, max(height - 1.5, 0.5), steps=num_gaussians, dtype=gaussians.dtype)
        means2d = torch.stack([means_x, means_y], dim=-1).view(1, 1, num_gaussians, 2).repeat(batch_size, num_views, 1, 1)
        output["render_meta"] = {
            "radii": radii,
            "means2d": means2d,
            "opacities": opacities,
            "tiles_per_gauss": tiles_per_gauss,
        }
        return output


def _build_patch_scene(reference_mode: str = "native", sr_scale: float = 1.0):
    """构造一个最小 patch supervision scene."""

    gt_images = torch.zeros(1, 1, 3, 4, 4, dtype=torch.float32)
    reference_hw = 4 if reference_mode == "native" else 8
    reference_images = torch.arange(1 * 1 * 3 * reference_hw * reference_hw, dtype=torch.float32).view(1, 1, 3, reference_hw, reference_hw)
    cam_view = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4)
    intrinsics = torch.tensor([[[2.0, 2.0, 1.5, 1.5]]], dtype=torch.float32)
    intrinsics_ref = intrinsics.clone() if reference_mode == "native" else intrinsics * sr_scale

    from src.refinement_v2.data_loader import SceneBundle

    return SceneBundle(
        gt_images=gt_images,
        cam_view=cam_view,
        intrinsics=intrinsics,
        frame_indices=[0],
        scene_index=0,
        view_id="3",
        reference_images=reference_images,
        intrinsics_ref=intrinsics_ref,
        native_hw=(4, 4),
        reference_hw=(reference_hw, reference_hw),
        reference_mode=reference_mode,
        sr_scale=sr_scale,
    )


def test_sample_patch_windows_clamps_to_frame_bounds(tmp_path) -> None:
    """patch window 即使贴近边界也不能越界."""

    scene = _build_patch_scene(reference_mode="native")
    runner = RefinementRunner(
        scene=scene,
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(build_run_config(tmp_path).outdir),
        controller=StageController(build_run_config(tmp_path), build_stage_hparams(patch_size=2)),
        hparams=build_stage_hparams(patch_size=2),
        renderer=RecordingRenderer(),
    )

    residual_map = torch.zeros(1, 1, 1, 4, 4)
    residual_map[0, 0, 0, 3, 3] = 1.0
    patch_windows = runner.sample_patch_windows(residual_map)

    assert patch_windows.tolist() == [[[2, 2, 2, 2]]]


def test_gather_reference_patch_maps_native_window_to_sr_reference(tmp_path) -> None:
    """SR 模式下 native patch 坐标要正确映射到 reference 尺度."""

    scene = _build_patch_scene(reference_mode="super_resolved", sr_scale=2.0)
    runner = RefinementRunner(
        scene=scene,
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(build_run_config(tmp_path).outdir),
        controller=StageController(build_run_config(tmp_path, reference_mode="super_resolved", sr_scale=2.0), build_stage_hparams(patch_size=4)),
        hparams=build_stage_hparams(patch_size=4),
        renderer=RecordingRenderer(),
    )

    patch_windows_native = torch.tensor([[[1, 1, 2, 2]]], dtype=torch.long)
    patch_windows_ref = runner.map_patch_windows_to_reference(patch_windows_native)
    reference_patch = runner.gather_reference_patch(patch_windows_ref)

    assert patch_windows_ref.tolist() == [[[2, 2, 4, 4]]]
    assert torch.equal(reference_patch, scene.reference_images[:, :, :, 2:6, 2:6])


def test_render_patch_prediction_uses_intrinsics_ref_offsets(tmp_path) -> None:
    """patch 渲染必须使用 reference intrinsics 做局部主点偏移."""

    renderer = RecordingRenderer()
    run_config = build_run_config(tmp_path, reference_mode="super_resolved", sr_scale=2.0)
    hparams = build_stage_hparams(patch_size=4)
    scene = _build_patch_scene(reference_mode="super_resolved", sr_scale=2.0)

    runner = RefinementRunner(
        scene=scene,
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=renderer,
    )

    patch_windows_native = torch.tensor([[[1, 1, 2, 2]]], dtype=torch.long)
    pred_patch, reference_patch, patch_intrinsics = runner.render_patch_prediction(patch_windows_native)

    assert pred_patch.shape == reference_patch.shape
    assert torch.allclose(renderer.calls[-1]["intrinsics"], patch_intrinsics)
    assert torch.allclose(patch_intrinsics[0, 0], torch.tensor([4.0, 4.0, 1.0, 1.0]))


def test_native_reference_mode_reuses_same_patch_path(tmp_path) -> None:
    """native 模式下也应该走同一套 patch 路径,而不是另写特殊分支."""

    renderer = RecordingRenderer()
    run_config = build_run_config(tmp_path, reference_mode="native", sr_scale=1.0)
    hparams = build_stage_hparams(patch_size=2)
    scene = _build_patch_scene(reference_mode="native", sr_scale=1.0)

    runner = RefinementRunner(
        scene=scene,
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=renderer,
    )

    patch_windows_native = torch.tensor([[[1, 1, 2, 2]]], dtype=torch.long)
    pred_patch, reference_patch, patch_intrinsics = runner.render_patch_prediction(patch_windows_native)

    assert pred_patch.shape == (1, 1, 3, 2, 2)
    assert reference_patch.shape == (1, 1, 3, 2, 2)
    assert torch.allclose(patch_intrinsics[0, 0], torch.tensor([2.0, 2.0, 0.5, 0.5]))


def test_render_patch_prediction_with_render_devices_does_not_reuse_stale_intrinsics(tmp_path) -> None:
    """双设备打开时,连续 patch render 也必须看到新的局部内参."""

    renderer = RecordingRenderer()
    run_config = build_run_config(
        tmp_path,
        reference_mode="super_resolved",
        sr_scale=2.0,
        render_devices=["cpu", "cpu"],
    )
    hparams = build_stage_hparams(patch_size=4)
    scene = _build_patch_scene(reference_mode="super_resolved", sr_scale=2.0)

    runner = RefinementRunner(
        scene=scene,
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=renderer,
    )

    runner.render_patch_prediction(torch.tensor([[[0, 0, 2, 2]]], dtype=torch.long))
    runner.render_patch_prediction(torch.tensor([[[1, 1, 2, 2]]], dtype=torch.long))

    assert len(renderer.calls) == 2
    assert torch.allclose(renderer.calls[0]["intrinsics"][0, 0], torch.tensor([4.0, 4.0, 3.0, 3.0]))
    assert torch.allclose(renderer.calls[1]["intrinsics"][0, 0], torch.tensor([4.0, 4.0, 1.0, 1.0]))


def test_stage2a_with_patch_supervision_records_patch_losses(tmp_path) -> None:
    """开启 patch supervision 后, Stage 2A 指标里应包含 patch loss."""

    from tests.refinement_v2.helpers import FakeRenderer, build_scene_bundle

    run_config = build_run_config(tmp_path, reference_mode="native", sr_scale=1.0)
    hparams = build_stage_hparams(iters_stage2a=2, patch_size=4, lambda_patch_rgb=0.5)
    scene = build_scene_bundle()

    runner = RefinementRunner(
        scene=scene,
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=FakeRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    final_metrics = runner.run_stage2a()

    assert "loss_patch_rgb" in final_metrics
    assert (run_config.outdir / "metrics_stage3sr.json").exists()
    assert (run_config.outdir / "gaussian_fidelity_histogram.json").exists()
    assert (run_config.outdir / "sr_selection_stats.json").exists()


def test_phase3s_builds_fidelity_summary_from_render_meta(tmp_path) -> None:
    """Phase 3S 应该能消费 render meta 并写出非平凡的 selection map."""

    from tests.refinement_v2.helpers import build_scene_bundle

    run_config = build_run_config(tmp_path)
    hparams = build_stage_hparams()
    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=MetaRecordingRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    metrics = runner.run_phase3s_build_sr_selection()

    assert metrics["fidelity_mean"] > 0.0
    assert metrics["sr_selection_mean"] > 0.0
    assert runner.diagnostics_state["phase3s_completed"] is True
    assert "warnings" not in runner.diagnostics_state
    assert runner.sr_selection_map is not None
    assert not torch.allclose(runner.sr_selection_map, torch.ones_like(runner.sr_selection_map))
    assert (run_config.outdir / "metrics_phase3s.json").exists()
    assert (run_config.outdir / "gaussian_fidelity_histogram.json").exists()
    assert (run_config.outdir / "sr_selection_stats.json").exists()


def test_stage3sr_records_sampling_smooth_loss_metric(tmp_path) -> None:
    """Stage 3SR 应把 sampling smooth loss 正式记到指标里."""

    from tests.refinement_v2.helpers import build_scene_bundle

    run_config = build_run_config(tmp_path)
    hparams = build_stage_hparams(
        iters_stage2a=2,
        patch_size=4,
        lambda_patch_rgb=0.5,
        lambda_sampling_smooth=1e-2,
    )
    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=MetaRecordingRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    runner.run_phase3s_build_sr_selection()
    metrics = runner.run_stage3sr_selective_patch()
    stage3sr_metrics = json.loads((run_config.outdir / "metrics_stage3sr.json").read_text(encoding="utf-8"))

    assert "loss_sampling_smooth" in metrics
    assert metrics["loss_sampling_smooth"] > 0.0
    assert "loss_sampling_smooth" in stage3sr_metrics[-1]
