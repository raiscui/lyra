"""`refinement_v2` 使用的损失函数集合."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossSummary:
    """统一承载单次迭代损失项."""

    total: torch.Tensor
    loss_rgb_weighted: torch.Tensor
    loss_scale_tail: torch.Tensor
    loss_opacity_sparse: torch.Tensor
    loss_patch_rgb: torch.Tensor | None = None
    loss_patch_perceptual: torch.Tensor | None = None
    loss_means_anchor: torch.Tensor | None = None
    loss_rotation_reg: torch.Tensor | None = None
    loss_pose_l2: torch.Tensor | None = None
    loss_pose_smooth: torch.Tensor | None = None


def compute_weighted_rgb_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    weight_map: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """计算带权重的 RGB Charbonnier 损失."""

    if pred_rgb.shape != gt_rgb.shape:
        raise ValueError("pred_rgb and gt_rgb must have the same shape.")
    if weight_map.shape[:2] != pred_rgb.shape[:2]:
        raise ValueError("weight_map batch/view dimensions must match pred_rgb.")

    charbonnier = torch.sqrt((pred_rgb - gt_rgb) ** 2 + eps**2)
    weighted = charbonnier * weight_map
    return weighted.mean()


def compute_scale_tail_loss(scales: torch.Tensor, threshold: float = 0.25) -> torch.Tensor:
    """惩罚过大的高斯尺度尾部."""

    scale_max = scales.max(dim=-1).values
    return F.relu(scale_max - threshold).mean()


def compute_opacity_sparse_loss(opacity: torch.Tensor) -> torch.Tensor:
    """轻微压制整体透明度均值,减少雾状叠层."""

    return opacity.mean()


def compute_patch_perceptual_loss(
    pred_patch: torch.Tensor,
    reference_patch: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """最小版 perceptual patch loss.

    第一版先用不带权重的 Charbonnier 占位.
    这样接口先稳定下来,后续如果要接 LPIPS 或特征金字塔,不需要再改 runner 契约.
    """

    if pred_patch.shape != reference_patch.shape:
        raise ValueError("pred_patch and reference_patch must have the same shape.")

    return torch.sqrt((pred_patch - reference_patch) ** 2 + eps**2).mean()


def compute_means_anchor_loss(
    means: torch.Tensor,
    initial_means: torch.Tensor,
) -> torch.Tensor:
    """约束位置不要偏离初始高斯中心太远."""

    if means.shape != initial_means.shape:
        raise ValueError("means and initial_means must have the same shape.")
    return (means - initial_means).pow(2).sum(dim=-1).mean()


def compute_rotation_regularization_loss(
    rotations: torch.Tensor,
    initial_rotations: torch.Tensor,
) -> torch.Tensor:
    """约束 quaternion 不要无约束乱转.

    这里按单位 quaternion 的夹角相似度做约束.
    同时对 `q` 和 `-q` 视为同一朝向,避免符号翻转被误罚.
    """

    if rotations.shape != initial_rotations.shape:
        raise ValueError("rotations and initial_rotations must have the same shape.")

    rotations_normalized = F.normalize(rotations, dim=-1)
    initial_normalized = F.normalize(initial_rotations, dim=-1)
    cosine_similarity = (rotations_normalized * initial_normalized).sum(dim=-1).abs().clamp(max=1.0)
    return (1.0 - cosine_similarity).mean()


def compute_pose_regularization(
    pose_delta: torch.Tensor,
    smooth_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 pose 的 L2 与时间平滑正则."""

    pose_l2 = pose_delta.pow(2).mean()
    if pose_delta.shape[0] <= 1:
        pose_smooth = torch.zeros((), dtype=pose_delta.dtype, device=pose_delta.device)
    else:
        pose_smooth = (pose_delta[1:] - pose_delta[:-1]).pow(2).mean() * smooth_weight
    return pose_l2, pose_smooth
