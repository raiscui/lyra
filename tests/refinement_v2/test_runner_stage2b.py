"""Stage 2B 测试."""

from src.refinement_v2.diagnostics import DiagnosticsWriter
from src.refinement_v2.runner import RefinementRunner
from src.refinement_v2.stage_controller import StageController
from tests.refinement_v2.helpers import FakeRenderer, build_gaussian_adapter, build_run_config, build_scene_bundle, build_stage_hparams


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
