"""权重图构造测试."""

import torch

from src.refinement_v2.weight_builder import WeightBuilder


def test_zero_residual_produces_near_one_weights() -> None:
    builder = WeightBuilder(weight_floor=0.2)
    pred = torch.zeros(1, 2, 3, 4, 4)
    gt = torch.zeros(1, 2, 3, 4, 4)

    residual = builder.build_residual_map(pred, gt)
    weight = builder.build_weight_map(residual)

    assert weight.shape == (1, 2, 1, 4, 4)
    assert torch.allclose(weight, torch.ones_like(weight), atol=1e-5)


def test_extreme_residual_is_downweighted_but_respects_floor() -> None:
    builder = WeightBuilder(weight_floor=0.3, weight_tau=0.1)
    residual = torch.zeros(1, 1, 1, 2, 2)
    residual[..., 0, 0] = 10.0

    weight = builder.build_weight_map(residual)

    assert float(weight.min().item()) >= 0.3
    assert float(weight[..., 0, 0].item()) <= float(weight[..., 1, 1].item())


def test_ema_smooths_second_weight_map() -> None:
    builder = WeightBuilder(weight_floor=0.2, ema_decay=0.8)
    first_residual = torch.zeros(1, 1, 1, 2, 2)
    second_residual = torch.ones(1, 1, 1, 2, 2)

    first_weight = builder.build_weight_map(first_residual)
    second_weight = builder.build_weight_map(second_residual, prev_weight_map=first_weight)

    assert torch.all(second_weight <= first_weight)
    assert torch.all(second_weight >= 0.2)
