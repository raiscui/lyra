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


@dataclass
class DepthAnchorLossSummary:
    """承载 depth anchor 的单次计算结果."""

    loss: torch.Tensor
    valid_ratio: float
    skip_reason: str | None = None


def _canonicalize_screen_radii(radii: torch.Tensor | None) -> torch.Tensor | None:
    """把投影半径统一压成可供损失项消费的标量半径.

    真实 renderer 现在可能返回 `[B, V, N, 2]` 的双轴半径.
    `sampling_smooth` 原来按 `[B, V, N]` 实现,这里先做兼容归一.
    """

    if not isinstance(radii, torch.Tensor):
        return None

    radii = radii.float()
    if radii.ndim == 4 and radii.shape[-1] == 2:
        return radii.amax(dim=-1)
    if radii.ndim == 4 and radii.shape[-1] == 1:
        return radii.squeeze(-1)
    return radii


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


def downsample_rgb_tensor(
    rgb_tensor: torch.Tensor,
    output_hw: tuple[int, int],
    *,
    mode: str = "bilinear",
    antialias: bool = True,
) -> torch.Tensor:
    """把 `[B, V, C, H, W]` RGB tensor 可微地下采样到目标分辨率.

    这里默认使用 `bilinear + align_corners=False + antialias=True`.
    这和 PyTorch 官方文档推荐的 downsampling 语义一致, 更接近常见图像库的结果.
    """

    if rgb_tensor.ndim != 5:
        raise ValueError(f"Expected rgb_tensor with shape [B, V, C, H, W], got {tuple(rgb_tensor.shape)}.")

    target_height, target_width = int(output_hw[0]), int(output_hw[1])
    if min(target_height, target_width) <= 0:
        raise ValueError(f"output_hw must be positive, got {output_hw}.")

    batch_size, num_views, num_channels, height, width = rgb_tensor.shape
    if (height, width) == (target_height, target_width):
        return rgb_tensor

    resize_kwargs: dict[str, object] = {}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        resize_kwargs["align_corners"] = False
    if mode in {"bilinear", "bicubic"}:
        resize_kwargs["antialias"] = antialias and (target_height < height or target_width < width)

    resized = F.interpolate(
        rgb_tensor.reshape(batch_size * num_views, num_channels, height, width),
        size=(target_height, target_width),
        mode=mode,
        **resize_kwargs,
    )
    return resized.reshape(batch_size, num_views, num_channels, target_height, target_width)


def compute_scale_tail_loss(scales: torch.Tensor, threshold: float = 0.25) -> torch.Tensor:
    """惩罚过大的高斯尺度尾部."""

    scale_max = scales.max(dim=-1).values
    return F.relu(scale_max - threshold).mean()


def compute_opacity_sparse_loss(opacity: torch.Tensor) -> torch.Tensor:
    """轻微压制整体透明度均值,减少雾状叠层."""

    return opacity.mean()


def _flatten_depth_tensor(depth: torch.Tensor) -> torch.Tensor:
    """把 `[B, V, 1, H, W]` depth/mask 压平成 `[B, V, H*W]`."""

    if depth.ndim != 5 or depth.shape[2] != 1:
        raise ValueError(f"Expected depth tensor with shape [B, V, 1, H, W], got {tuple(depth.shape)}.")
    return depth.reshape(depth.shape[0], depth.shape[1], -1)


def build_depth_anchor_valid_mask(
    reference_depth: torch.Tensor,
    reference_alpha: torch.Tensor | None = None,
    *,
    alpha_threshold: float = 0.0,
) -> torch.Tensor:
    """构造 depth anchor 的有效像素掩码.

    规则保持保守:
    1. 参考深度必须大于 0 且是有限值
    2. 如果 renderer 同时给了 alpha,再叠加一次可见性阈值
    """

    if reference_alpha is not None and reference_alpha.shape != reference_depth.shape:
        raise ValueError("reference_alpha must have the same shape as reference_depth.")

    valid_mask = (reference_depth > 0) & torch.isfinite(reference_depth)
    if reference_alpha is not None:
        valid_mask = valid_mask & torch.isfinite(reference_alpha) & (reference_alpha > alpha_threshold)
    return valid_mask


def normalize_depth(depth: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """按训练期同语义做尺度不变 depth 归一化."""

    depth_flat = _flatten_depth_tensor(depth)
    valid_mask_flat = _flatten_depth_tensor(valid_mask.to(dtype=depth.dtype))

    # 这里故意保持和训练期一致的“先乘 mask 再求 median”的语义.
    # 这样 refinement 与训练侧的尺度不变 depth 监督口径不会漂移.
    valid_count = valid_mask_flat.sum(dim=-1, keepdim=True).clamp_min(1.0)
    depth_valid = depth_flat * valid_mask_flat
    depth_median = torch.median(depth_valid, dim=-1, keepdim=True).values
    depth_centered = (depth_valid - depth_median) * valid_mask_flat
    depth_var = depth_centered.abs().sum(dim=-1, keepdim=True) / valid_count
    depth_var = depth_var.clamp(min=1e-3, max=1e3)
    return depth_centered / depth_var


def compute_depth_anchor_loss(
    pred_depth: torch.Tensor,
    reference_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> DepthAnchorLossSummary:
    """计算 refinement_v2 使用的 depth anchor loss."""

    zero_loss = pred_depth.new_zeros(())
    if pred_depth.shape != reference_depth.shape:
        return DepthAnchorLossSummary(loss=zero_loss, valid_ratio=0.0, skip_reason="shape_mismatch")
    if valid_mask.shape != reference_depth.shape:
        return DepthAnchorLossSummary(loss=zero_loss, valid_ratio=0.0, skip_reason="mask_shape_mismatch")

    effective_mask = valid_mask.bool() & torch.isfinite(pred_depth) & torch.isfinite(reference_depth)
    valid_ratio = float(effective_mask.float().mean().item())
    if valid_ratio <= 0.0:
        return DepthAnchorLossSummary(loss=zero_loss, valid_ratio=0.0, skip_reason="empty_valid_mask")

    pred_depth_norm = normalize_depth(pred_depth, effective_mask)
    reference_depth_norm = normalize_depth(reference_depth, effective_mask)
    valid_mask_flat = _flatten_depth_tensor(effective_mask.to(dtype=pred_depth.dtype))
    loss_depth = F.smooth_l1_loss(
        pred_depth_norm * valid_mask_flat,
        reference_depth_norm * valid_mask_flat,
    )
    return DepthAnchorLossSummary(loss=loss_depth, valid_ratio=valid_ratio)


def compute_sampling_smooth_loss(
    scales: torch.Tensor,
    fidelity_score: torch.Tensor | None,
    render_meta: dict[str, torch.Tensor] | None,
    radius_threshold: float = 1.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """约束低 fidelity 高斯不要收缩出不受支持的高频.

    第一版不直接复刻 `Mip-Splatting` 的 renderer-level 3D smoothing.
    这里只做一个更保守的训练期正则:
    - 如果某个 Gaussian 在当前 native 渲染里投影半径偏小
    - 且它本身 fidelity 也低
    - 那就不要让它的 3D scale 再继续收得过小
    """

    if fidelity_score is None or render_meta is None:
        return torch.zeros((), dtype=scales.dtype, device=scales.device)

    radii = render_meta.get("radii")
    opacities = render_meta.get("opacities")
    if not isinstance(radii, torch.Tensor) or not isinstance(opacities, torch.Tensor):
        return torch.zeros((), dtype=scales.dtype, device=scales.device)

    radii = _canonicalize_screen_radii(radii)
    if radii is None:
        return torch.zeros((), dtype=scales.dtype, device=scales.device)
    radii = radii.to(device=scales.device, dtype=scales.dtype)
    opacities = opacities.to(device=scales.device, dtype=scales.dtype)
    if fidelity_score.device != scales.device or fidelity_score.dtype != scales.dtype:
        fidelity_score = fidelity_score.to(device=scales.device, dtype=scales.dtype)
    if radii.ndim == 2:
        radii = radii.unsqueeze(0)
    if opacities.ndim == 2:
        opacities = opacities.unsqueeze(0)
    if fidelity_score.ndim == 1:
        fidelity_score = fidelity_score.unsqueeze(0)

    if radii.ndim != 3 or opacities.shape != radii.shape:
        return torch.zeros((), dtype=scales.dtype, device=scales.device)
    if fidelity_score.ndim != 2 or fidelity_score.shape[0] != radii.shape[0] or fidelity_score.shape[1] != radii.shape[2]:
        return torch.zeros((), dtype=scales.dtype, device=scales.device)

    visible = (radii > 0).to(dtype=scales.dtype)
    radius_deficit = F.relu(radius_threshold - radii) / max(radius_threshold, eps)
    opacity_gate = opacities.clamp(0.0, 1.0)
    unsupported_per_view = radius_deficit * visible * opacity_gate
    unsupported_per_gaussian = unsupported_per_view.sum(dim=1) / visible.sum(dim=1).clamp_min(1.0)

    low_fidelity = (1.0 - fidelity_score).clamp(0.0, 1.0)
    scale_max = scales.max(dim=-1).values.clamp_min(eps)
    return (low_fidelity * unsupported_per_gaussian / scale_max).mean()


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
