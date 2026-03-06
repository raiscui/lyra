"""高斯适配器测试."""

import torch

from src.refinement_v2.config import StageHyperParams
from src.refinement_v2.gaussian_adapter import GaussianAdapter


def _build_gaussians(num_points: int = 8) -> torch.Tensor:
    gaussians = torch.zeros(num_points, 14)
    gaussians[:, 3] = 0.2
    gaussians[:, 4:7] = 0.1
    gaussians[:, 7] = 1.0
    gaussians[:, 11:] = 0.5
    return gaussians


def test_freeze_for_stage2a_disables_means_and_rotations() -> None:
    adapter = GaussianAdapter.from_tensor(_build_gaussians())
    adapter.freeze_for_stage("stage2a")

    assert adapter.opacity.requires_grad is True
    assert adapter.colors.requires_grad is True
    assert adapter.scales.requires_grad is True
    assert adapter.means.requires_grad is False
    assert adapter.rotations.requires_grad is False


def test_build_optimizer_only_contains_trainable_groups() -> None:
    adapter = GaussianAdapter.from_tensor(_build_gaussians())
    adapter.freeze_for_stage("stage2a")
    optimizer = adapter.build_optimizer("stage2a", StageHyperParams())

    group_names = {group["name"] for group in optimizer.param_groups}
    assert group_names == {"opacity", "colors", "scales"}


def test_summarize_gaussian_stats_contains_expected_keys() -> None:
    adapter = GaussianAdapter.from_tensor(_build_gaussians())
    stats = adapter.summarize_gaussian_stats()

    assert "num_gaussians" in stats
    assert "scale_tail_ratio" in stats
    assert "opacity_lowconf_ratio" in stats
