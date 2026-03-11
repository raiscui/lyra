"""阶段推进与停止条件控制器."""

from __future__ import annotations

from .config import RefinementRunConfig, StageHyperParams


class StageController:
    """把阶段切换规则集中到一个地方."""

    def __init__(self, run_config: RefinementRunConfig, hparams: StageHyperParams) -> None:
        self.run_config = run_config
        self.hparams = hparams

    def should_stop_stage(self, stage_name: str, metrics_history: list[dict]) -> bool:
        """判断当前阶段是否应该停止."""

        if not metrics_history:
            return False

        latest = metrics_history[-1]
        if latest.get("ghosting_acceptable", False):
            return True
        if latest.get("stop_now", False):
            return True

        patience = min(len(metrics_history), self.hparams.plateau_patience)
        if patience < self.hparams.plateau_patience:
            return False

        recent = metrics_history[-patience:]
        losses = [float(metric.get("loss_total", 0.0)) for metric in recent]
        return max(losses) - min(losses) < self.hparams.plateau_delta

    def should_enter_stage2b(self, diagnostics: dict) -> bool:
        """判断是否进入 limited geometry."""

        if not self.run_config.enable_stage2b:
            return False
        if not diagnostics.get("stage3a_completed", False):
            return False
        if diagnostics.get("stage3sr_enabled", False):
            if not diagnostics.get("phase3s_completed", False):
                return False
            if not diagnostics.get("stage3sr_completed", False):
                return False
        if diagnostics.get("ghosting_acceptable", False):
            return False
        if diagnostics.get("global_shift_detected", False):
            return False
        if diagnostics.get("weight_map_unstable", False):
            return False
        if diagnostics.get("geometry_overfit_risk", False):
            return False
        if not diagnostics.get("need_geometry", False):
            return False
        return diagnostics.get("local_overlap_persistent", False)

    def should_enter_stage3b(self, diagnostics: dict) -> bool:
        """判断是否进入 `Phase E` 风格的 SR-driven limited geometry.

        这条路径只在 `Stage 3SR` 已完成且 supervision 仍明确是 HR 主监督时打开.
        这样可以避免 geometry release 在不稳定的上游状态里过早发生.
        """

        if not self.run_config.enable_stage3b:
            return False
        if not diagnostics.get("stage3a_completed", False):
            return False
        if not diagnostics.get("stage3sr_enabled", False):
            return False
        if not diagnostics.get("phase3s_completed", False):
            return False
        if not diagnostics.get("stage3sr_completed", False):
            return False
        if diagnostics.get("stage3sr_supervision_mode") != "full_frame_hr":
            return False
        if diagnostics.get("ghosting_acceptable", False):
            return False
        if diagnostics.get("global_shift_detected", False):
            return False
        if diagnostics.get("weight_map_unstable", False):
            return False
        if diagnostics.get("geometry_overfit_risk", False):
            return False
        if not diagnostics.get("need_geometry", False):
            return False
        return diagnostics.get("local_overlap_persistent", False)

    def should_prune_now(self, iteration: int) -> bool:
        """判断当前 iteration 是否该触发 pruning.

        这里使用 1-based iteration.
        这样和日志里的第几轮更一致,也更不容易在 warmup 判断上出错.
        """

        if not self.run_config.enable_pruning:
            return False
        if self.hparams.prune_every <= 0:
            return False
        if iteration <= self.hparams.prune_warmup_iters:
            return False

        # warmup 之后按固定间隔触发.
        # 例如 warmup=2, every=2 时,会在 4/6/8... 轮触发.
        return (iteration - self.hparams.prune_warmup_iters) % self.hparams.prune_every == 0

    def should_enter_pose_diagnostic(self, diagnostics: dict) -> bool:
        """判断是否进入 tiny pose-only."""

        return self.run_config.enable_pose_diagnostic and diagnostics.get("global_shift_detected", False)

    def should_enter_joint_fallback(self, diagnostics: dict) -> bool:
        """判断是否进入 joint fallback."""

        return (
            self.run_config.enable_joint_fallback
            and diagnostics.get("global_shift_detected", False)
            and diagnostics.get("local_overlap_persistent", False)
            and diagnostics.get("pose_diagnostic_ran", False)
        )

    def summarize_stop_reason(self, diagnostics: dict) -> str:
        """把停止原因压成一个稳定字符串."""

        if diagnostics.get("ghosting_acceptable", False):
            return "ghosting_acceptable"
        if diagnostics.get("weight_map_unstable", False):
            return "weight_map_unstable"
        if diagnostics.get("geometry_overfit_risk", False):
            return "geometry_overfit_risk"
        if diagnostics.get("pose_gain_too_small", False):
            return "pose_gain_too_small"
        if diagnostics.get("joint_not_allowed", False):
            return "joint_not_allowed"
        return "metrics_plateau"
