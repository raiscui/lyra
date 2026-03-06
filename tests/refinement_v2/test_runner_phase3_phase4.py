"""Phase 3 / Phase 4 测试."""

from src.refinement_v2.diagnostics import DiagnosticsWriter
from src.refinement_v2.runner import RefinementRunner
from src.refinement_v2.stage_controller import StageController
from tests.refinement_v2.helpers import FakeRenderer, build_gaussian_adapter, build_run_config, build_scene_bundle, build_stage_hparams


def test_phase3_and_phase4_outputs_expected_artifacts(tmp_path) -> None:
    run_config = build_run_config(
        tmp_path,
        enable_pose_diagnostic=True,
        enable_joint_fallback=True,
    )
    hparams = build_stage_hparams(iters_pose=2, iters_joint=2)
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
    runner.diagnostics_state["global_shift_detected"] = True
    runner.diagnostics_state["local_overlap_persistent"] = True

    runner.run_phase3_pose_only()
    runner.run_phase4_joint()

    assert (run_config.outdir / "pose" / "pose_delta_summary.json").exists()
    assert (run_config.outdir / "metrics_phase4.json").exists()
