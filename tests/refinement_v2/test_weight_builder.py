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


def test_build_sr_selection_weight_projects_low_fidelity_gaussians_to_reference_grid() -> None:
    """低 fidelity Gaussian 应该被投影成非平凡的 reference 选择图."""

    builder = WeightBuilder()
    fidelity_score = torch.tensor([[0.1, 0.95]], dtype=torch.float32)
    render_meta = {
        "means2d": torch.tensor([[[[1.0, 1.0], [3.0, 3.0]]]], dtype=torch.float32),
        "radii": torch.tensor([[[[1.2, 1.0], [1.0, 0.8]]]], dtype=torch.float32),
        "opacities": torch.tensor([[[0.9, 0.9]]], dtype=torch.float32),
    }

    selection_map = builder.build_sr_selection_weight(
        render_meta=render_meta,
        fidelity_score=fidelity_score,
        native_hw=(4, 4),
        output_hw=(8, 8),
    )

    assert selection_map is not None
    assert selection_map.shape == (1, 1, 1, 8, 8)
    assert float(selection_map.mean().item()) > 0.0
    assert not torch.allclose(selection_map, torch.ones_like(selection_map))
    assert float(selection_map[0, 0, 0, 2, 2].item()) > float(selection_map[0, 0, 0, 6, 6].item())


def test_compute_gaussian_fidelity_score_supports_vector_radii() -> None:
    """真实 renderer 若返回双轴半径, 也应能正常产出 fidelity."""

    builder = WeightBuilder()
    render_meta = {
        "radii": torch.tensor([[[[1.2, 0.8], [0.0, 0.0]]]], dtype=torch.float32),
        "opacities": torch.tensor([[[0.9, 0.6]]], dtype=torch.float32),
        "tiles_per_gauss": torch.tensor([[[3.0, 1.0]]], dtype=torch.float32),
    }

    fidelity = builder.compute_gaussian_fidelity_score(render_meta)

    assert fidelity is not None
    assert fidelity.shape == (1, 2)
    assert float(fidelity[0, 0].item()) > 0.0
    assert float(fidelity[0, 1].item()) == 0.0


def test_combine_sr_weights_multiplies_robust_and_selection_maps() -> None:
    """最终 SR 权重应显式来自两张图的逐点乘法."""

    builder = WeightBuilder()
    w_robust = torch.tensor([[[[[1.0, 0.5], [0.25, 0.8]]]]], dtype=torch.float32)
    w_sr_select = torch.tensor([[[[[0.2, 0.4], [1.0, 0.5]]]]], dtype=torch.float32)

    combined = builder.combine_sr_weights(w_robust, w_sr_select)

    expected = torch.tensor([[[[[0.2, 0.2], [0.25, 0.4]]]]], dtype=torch.float32)
    assert torch.allclose(combined, expected)
