"""阶段控制器测试."""

from src.refinement_v2.config import RefinementRunConfig, StageHyperParams
from src.refinement_v2.stage_controller import StageController


def _build_run_config(**overrides):
    kwargs = {
        "config_path": "configs/demo/lyra_static.yaml",
        "gaussians_path": "outputs/demo/gaussians_0.ply",
        "outdir": "outputs/refine_v2/test",
    }
    kwargs.update(overrides)
    return RefinementRunConfig(**kwargs)


def test_should_stop_stage_when_ghosting_is_acceptable() -> None:
    controller = StageController(_build_run_config(), StageHyperParams())
    assert controller.should_stop_stage("stage2a", [{"ghosting_acceptable": True}]) is True


def test_pose_diagnostic_requires_explicit_flag() -> None:
    disabled = StageController(_build_run_config(), StageHyperParams())
    enabled = StageController(_build_run_config(enable_pose_diagnostic=True), StageHyperParams())

    diagnostics = {"global_shift_detected": True}
    assert disabled.should_enter_pose_diagnostic(diagnostics) is False
    assert enabled.should_enter_pose_diagnostic(diagnostics) is True


def test_stage2b_requires_explicit_enable_flag() -> None:
    """Stage 2B 默认不能偷偷进入,必须显式开启."""

    disabled = StageController(_build_run_config(enable_stage2b=False), StageHyperParams())
    diagnostics = {
        "need_geometry": True,
        "local_overlap_persistent": True,
    }

    assert disabled.should_enter_stage2b(diagnostics) is False


def test_stage2b_requires_local_overlap_and_blocks_global_shift() -> None:
    """Stage 2B 应该更像局部结构重叠,而不是整体错位."""

    enabled = StageController(_build_run_config(enable_stage2b=True), StageHyperParams())

    assert enabled.should_enter_stage2b({"need_geometry": True}) is False
    assert enabled.should_enter_stage2b(
        {
            "need_geometry": True,
            "local_overlap_persistent": True,
            "global_shift_detected": True,
        }
    ) is False
    assert enabled.should_enter_stage2b(
        {
            "need_geometry": True,
            "local_overlap_persistent": True,
            "global_shift_detected": False,
        }
    ) is True


def test_stage2b_stays_closed_when_ghosting_is_already_acceptable() -> None:
    """既然已经可接受,就不要再放开 geometry."""

    enabled = StageController(_build_run_config(enable_stage2b=True), StageHyperParams())
    diagnostics = {
        "need_geometry": True,
        "local_overlap_persistent": True,
        "ghosting_acceptable": True,
    }

    assert enabled.should_enter_stage2b(diagnostics) is False


def test_joint_fallback_requires_flag_and_evidence() -> None:
    disabled = StageController(_build_run_config(), StageHyperParams())
    enabled = StageController(_build_run_config(enable_joint_fallback=True), StageHyperParams())

    diagnostics = {
        "global_shift_detected": True,
        "local_overlap_persistent": True,
        "pose_diagnostic_ran": True,
    }
    assert disabled.should_enter_joint_fallback(diagnostics) is False
    assert enabled.should_enter_joint_fallback(diagnostics) is True


def test_should_prune_now_respects_warmup_and_interval() -> None:
    """pruning 只在 warmup 之后按固定步频触发."""

    controller = StageController(
        _build_run_config(enable_pruning=True),
        StageHyperParams(prune_warmup_iters=2, prune_every=2),
    )

    # warmup 期间以及未对齐步频时都不触发.
    assert controller.should_prune_now(1) is False
    assert controller.should_prune_now(2) is False
    assert controller.should_prune_now(3) is False

    # 1-based iteration 下,第 4、6 ... 轮开始触发.
    assert controller.should_prune_now(4) is True
    assert controller.should_prune_now(5) is False
    assert controller.should_prune_now(6) is True


def test_should_prune_now_requires_explicit_enable_flag() -> None:
    """默认主线不启用 pruning,避免改变既有基线行为."""

    controller = StageController(
        _build_run_config(enable_pruning=False),
        StageHyperParams(prune_warmup_iters=0, prune_every=1),
    )

    assert controller.should_prune_now(1) is False
