"""Phase 0 / dry-run 测试."""

from src.refinement_v2.diagnostics import DiagnosticsWriter
from src.refinement_v2.runner import RefinementRunner
from src.refinement_v2.stage_controller import StageController
from tests.refinement_v2.helpers import FakeRenderer, build_gaussian_adapter, build_run_config, build_scene_bundle, build_stage_hparams


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
