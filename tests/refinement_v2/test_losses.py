"""损失函数测试."""

import torch

from src.refinement_v2.losses import (
    compute_opacity_sparse_loss,
    compute_pose_regularization,
    compute_sampling_smooth_loss,
    compute_scale_tail_loss,
    compute_weighted_rgb_loss,
)


def test_weighted_rgb_loss_is_positive() -> None:
    pred = torch.ones(1, 1, 3, 2, 2)
    gt = torch.zeros(1, 1, 3, 2, 2)
    weight = torch.ones(1, 1, 1, 2, 2)

    loss = compute_weighted_rgb_loss(pred, gt, weight)
    assert float(loss.item()) > 0


def test_scale_tail_loss_only_penalizes_large_scales() -> None:
    scales = torch.tensor([[0.1, 0.1, 0.1], [0.4, 0.2, 0.1]], dtype=torch.float32)
    loss = compute_scale_tail_loss(scales, threshold=0.25)
    assert float(loss.item()) > 0


def test_opacity_sparse_loss_is_mean_opacity() -> None:
    opacity = torch.tensor([[0.2], [0.4]], dtype=torch.float32)
    loss = compute_opacity_sparse_loss(opacity)
    assert abs(float(loss.item()) - 0.3) < 1e-6


def test_pose_regularization_returns_l2_and_smooth_terms() -> None:
    pose_delta = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    pose_l2, pose_smooth = compute_pose_regularization(pose_delta)
    assert float(pose_l2.item()) > 0
    assert float(pose_smooth.item()) > 0


def test_sampling_smooth_loss_penalizes_small_scale_low_fidelity_gaussians() -> None:
    scales = torch.tensor([[0.1, 0.1, 0.1], [0.5, 0.4, 0.4]], dtype=torch.float32)
    fidelity_score = torch.tensor([[0.1, 0.9]], dtype=torch.float32)
    render_meta = {
        "radii": torch.tensor([[[0.3, 2.2], [0.4, 2.0]]], dtype=torch.float32),
        "opacities": torch.tensor([[[0.9, 0.9], [0.8, 0.8]]], dtype=torch.float32),
    }

    loss = compute_sampling_smooth_loss(
        scales=scales,
        fidelity_score=fidelity_score,
        render_meta=render_meta,
        radius_threshold=1.5,
    )

    assert float(loss.item()) > 0.0
