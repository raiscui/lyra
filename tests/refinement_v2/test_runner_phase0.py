"""Phase 0 / dry-run 测试."""

import torch

from src.refinement_v2.diagnostics import DiagnosticsWriter
from src.refinement_v2.runner import RefinementRunner
from src.refinement_v2.stage_controller import StageController
from tests.refinement_v2.helpers import FakeRenderer, build_gaussian_adapter, build_run_config, build_scene_bundle, build_stage_hparams


class ShardAwareRenderer:
    """记录 shard 调用并返回可验证顺序的 render 结果."""

    def __init__(self) -> None:
        self.view_counts: list[int] = []

    def render(self, gaussians: torch.Tensor, scene) -> dict[str, torch.Tensor]:
        batch_size, _, _ = gaussians.shape
        num_views = scene.gt_images.shape[1]
        height, width = scene.gt_images.shape[-2:]
        self.view_counts.append(num_views)

        view_values = scene.target_index.to(dtype=gaussians.dtype).view(batch_size, num_views, 1)
        image = view_values.view(batch_size, num_views, 1, 1, 1).expand(batch_size, num_views, 3, height, width).clone()
        depth = torch.zeros(batch_size, num_views, 1, height, width, dtype=image.dtype, device=image.device)

        num_gaussians = gaussians.shape[1]
        radii = view_values.expand(batch_size, num_views, num_gaussians)
        opacities = gaussians[:, :, 3].unsqueeze(1).expand(batch_size, num_views, num_gaussians)
        tiles_per_gauss = torch.ones_like(radii)
        means_x = view_values.expand(batch_size, num_views, num_gaussians)
        means_y = torch.zeros_like(means_x)
        means2d = torch.stack([means_x, means_y], dim=-1)
        return {
            "images_pred": image,
            "depths_pred": depth,
            "render_meta": {
                "radii": radii,
                "opacities": opacities,
                "tiles_per_gauss": tiles_per_gauss,
                "means2d": means2d,
                "width": width,
                "height": height,
            },
        }


class DifferentiableShardRenderer:
    """既能记录 shard 调用,又能给 Stage 2A 提供稳定梯度."""

    def __init__(self) -> None:
        self.view_counts: list[int] = []

    def render(self, gaussians: torch.Tensor, scene) -> dict[str, torch.Tensor]:
        batch_size, _, _ = gaussians.shape
        num_views = scene.gt_images.shape[1]
        height, width = scene.gt_images.shape[-2:]
        self.view_counts.append(num_views)

        color_mean = gaussians[:, :, 11:].mean(dim=1)
        view_bias = scene.target_index.to(dtype=gaussians.dtype).view(batch_size, num_views, 1, 1, 1) / 10.0
        image = color_mean[:, None, :, None, None].expand(batch_size, num_views, 3, height, width).clone() + view_bias
        depth = torch.zeros(batch_size, num_views, 1, height, width, dtype=image.dtype, device=image.device)
        return {"images_pred": image, "depths_pred": depth}


class EvaluationTrackingRunner(RefinementRunner):
    """记录评估路径和训练路径分别被调用了多少次."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eval_calls = 0
        self.current_calls = 0

    def _render_scene_for_evaluation(self, scene):
        self.eval_calls += 1
        return super()._render_scene_for_evaluation(scene)

    def render_current_scene(self):
        self.current_calls += 1
        return super().render_current_scene()


def test_run_phase0_only_outputs_diagnostics(tmp_path) -> None:
    run_config = build_run_config(tmp_path, dry_run=True)
    hparams = build_stage_hparams()
    diagnostics = DiagnosticsWriter(run_config.outdir)
    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=diagnostics,
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=FakeRenderer(),
    )

    summary = runner.run_phase0_only()

    assert summary["phase_reached"] == "phase0"
    assert summary["stopped_reason"] == "dry_run"
    assert (run_config.outdir / "diagnostics.json").exists()
    assert (run_config.outdir / "residual_maps" / "phase0_frame_0000.png").exists()
    assert (run_config.outdir / "videos" / "baseline_render.mp4").exists()
    assert (run_config.outdir / "videos" / "final_render.mp4").exists()
    assert (run_config.outdir / "videos" / "gt_reference.mp4").exists()
    assert "baseline_render_video" in summary["artifacts"]
    assert "final_render_video" in summary["artifacts"]


def test_render_scene_splits_view_shards_across_render_devices(tmp_path) -> None:
    """多设备渲染应按 view 维切 shard,再按原顺序拼回结果."""

    run_config = build_run_config(tmp_path, render_devices=["cpu", "cpu"])
    hparams = build_stage_hparams()
    renderer = ShardAwareRenderer()
    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=renderer,
    )

    render_output = runner.render_current_scene()
    view_values = render_output["images_pred"][0, :, 0, 0, 0].tolist()
    render_meta = render_output["render_meta"]

    assert renderer.view_counts == [2, 1]
    assert view_values == [0.0, 1.0, 2.0]
    assert render_meta["radii"].shape[:2] == (1, 3)
    assert render_meta["means2d"].shape[:3] == (1, 3, 16)


def test_phase0_phase1_phase3s_use_evaluation_render_path_under_multi_device(tmp_path) -> None:
    """无 backward 阶段在多设备下必须走 evaluation helper,避免主卡聚合."""

    run_config = build_run_config(tmp_path, render_devices=["cpu", "cpu"])
    hparams = build_stage_hparams()
    runner = EvaluationTrackingRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=ShardAwareRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    runner.run_phase3s_build_sr_selection()

    assert runner.eval_calls == 3
    assert runner.current_calls == 0
    assert runner.prev_weight_map is not None
    assert runner.prev_weight_map.device.type == "cpu"
    assert runner.sr_selection_map is not None
    assert runner.sr_selection_map.device.type == "cpu"


def test_stage2a_multi_device_never_falls_back_to_full_view_render(tmp_path) -> None:
    """真正的多设备 Stage 2A 不应再出现单次 3-view 整体渲染."""

    run_config = build_run_config(tmp_path, render_devices=["cpu", "cpu"])
    hparams = build_stage_hparams(iters_stage2a=2)
    renderer = DifferentiableShardRenderer()
    gaussians = build_gaussian_adapter()
    colors_before = gaussians.colors.detach().clone()
    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=gaussians,
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=renderer,
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    runner.run_stage2a()

    assert 3 not in renderer.view_counts
    assert set(renderer.view_counts) == {1, 2}
    assert not torch.allclose(gaussians.colors.detach(), colors_before)


def test_build_render_shards_uses_single_view_round_robin_for_cuda_devices(tmp_path) -> None:
    """真实 CUDA 多卡时, shard 应进一步收成单 view 轮转."""

    run_config = build_run_config(tmp_path)
    hparams = build_stage_hparams()
    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=build_gaussian_adapter(),
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=FakeRenderer(),
    )

    runner.render_devices = [torch.device("cuda:0"), torch.device("cuda:1")]

    assert runner._build_render_shards(5) == [
        (torch.device("cuda:0"), 0, 1),
        (torch.device("cuda:1"), 1, 2),
        (torch.device("cuda:0"), 2, 3),
        (torch.device("cuda:1"), 3, 4),
        (torch.device("cuda:0"), 4, 5),
    ]
