"""patch-based supervision 测试."""

from __future__ import annotations

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
        image = torch.zeros_like(scene.gt_images)
        depth = torch.zeros(scene.gt_images.shape[0], scene.gt_images.shape[1], 1, *scene.gt_images.shape[-2:], dtype=image.dtype)
        return {"images_pred": image, "depths_pred": depth}


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
