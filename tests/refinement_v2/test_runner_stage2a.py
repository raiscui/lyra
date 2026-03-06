"""Stage 2A 测试."""

import json

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
    assert (run_config.outdir / "metrics_stage2a.json").exists()


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
