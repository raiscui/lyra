"""`refine_robust_v2` 入口脚本.

当前阶段已经接入最小 CLI.
现在已经接入基础 runner 主线.
"""

from __future__ import annotations

from src.refinement_v2.data_loader import build_scene_bundle
from src.refinement_v2.diagnostics import DiagnosticsWriter
from src.refinement_v2.gaussian_adapter import GaussianAdapter
from src.refinement_v2.runner import RefinementRunner
from src.refinement_v2.stage_controller import StageController
from src.refinement_v2.config import load_effective_config_from_cli


def main(argv: list[str] | None = None) -> None:
    """运行 `refine_robust_v2` 的主入口.

    入口负责完成对象组装.
    具体阶段推进由 `RefinementRunner` 接管.
    """

    run_config, stage_hparams = load_effective_config_from_cli(argv)
    scene_bundle = build_scene_bundle(run_config)
    gaussians = GaussianAdapter.from_ply(run_config.gaussians_path)
    diagnostics = DiagnosticsWriter(run_config.outdir)
    controller = StageController(run_config, stage_hparams)
    runner = RefinementRunner(
        scene=scene_bundle,
        gaussians=gaussians,
        diagnostics=diagnostics,
        controller=controller,
        hparams=stage_hparams,
    )

    if run_config.resume:
        runner.restore_latest_state()

    if run_config.dry_run:
        runner.run_phase0_only()
        return

    runner.run()


if __name__ == "__main__":
    main()
