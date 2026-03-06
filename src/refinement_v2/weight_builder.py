"""残差与权重图构造逻辑."""

from __future__ import annotations

import torch


class WeightBuilder:
    """把 residual 稳定地映射为 soft trust weight."""

    def __init__(
        self,
        alpha_rgb: float = 1.0,
        alpha_perc: float = 0.0,
        q_low: float = 0.50,
        q_high: float = 0.90,
        weight_tau: float = 0.45,
        weight_floor: float = 0.20,
        ema_decay: float = 0.90,
    ) -> None:
        self.alpha_rgb = alpha_rgb
        self.alpha_perc = alpha_perc
        self.q_low = q_low
        self.q_high = q_high
        self.weight_tau = weight_tau
        self.weight_floor = weight_floor
        self.ema_decay = ema_decay

    def build_residual_map(self, pred_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> torch.Tensor:
        """从 RGB 预测和 GT 构造逐像素 residual 图."""

        if pred_rgb.shape != gt_rgb.shape:
            raise ValueError(f"pred_rgb shape {tuple(pred_rgb.shape)} does not match gt_rgb shape {tuple(gt_rgb.shape)}")

        if pred_rgb.ndim != 5:
            raise ValueError("Expected RGB tensors with shape [B, V, C, H, W].")

        residual_rgb = (pred_rgb - gt_rgb).abs().mean(dim=2, keepdim=True)
        return self.alpha_rgb * residual_rgb

    def build_weight_map(
        self,
        residual_map: torch.Tensor,
        prev_weight_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """把 residual 图映射成 soft trust weight."""

        if residual_map.ndim != 5:
            raise ValueError("Expected residual_map with shape [B, V, 1, H, W].")

        # 这里的权重图是鲁棒监督权重,语义上应该是
        # “下一轮 loss 的静态系数”,而不是可学习分支.
        # 因此必须先 detach,避免 EMA 把上一轮计算图带进下一轮 backward.
        residual_map = residual_map.detach()

        flattened = residual_map.flatten(start_dim=-2)
        q_low = torch.quantile(flattened, self.q_low, dim=-1, keepdim=True)
        q_high = torch.quantile(flattened, self.q_high, dim=-1, keepdim=True)

        q_low = q_low.unsqueeze(-1)
        q_high = q_high.unsqueeze(-1)
        normalized = (residual_map - q_low) / (q_high - q_low + 1e-8)
        normalized = normalized.clamp(0.0, 1.0)

        weight_raw = torch.exp(-normalized / self.weight_tau)
        weight_map = weight_raw.clamp(self.weight_floor, 1.0)

        if prev_weight_map is not None:
            if prev_weight_map.shape != weight_map.shape:
                raise ValueError("prev_weight_map must have the same shape as weight_map.")
            weight_map = self.ema_decay * prev_weight_map.detach() + (1.0 - self.ema_decay) * weight_map

        return weight_map.detach()

    def summarize_weight_stats(self, weight_map: torch.Tensor) -> dict[str, float]:
        """输出权重图的基础统计."""

        return {
            "weight_min": float(weight_map.min().item()),
            "weight_max": float(weight_map.max().item()),
            "weight_mean": float(weight_map.mean().item()),
            "weight_std": float(weight_map.std().item()),
        }
