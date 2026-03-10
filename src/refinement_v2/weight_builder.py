"""残差与权重图构造逻辑."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
        fidelity_ratio_threshold: float = 1.5,
        fidelity_sigmoid_k: float = 6.0,
        fidelity_min_views: int = 3,
        fidelity_opacity_threshold: float = 1e-4,
    ) -> None:
        self.alpha_rgb = alpha_rgb
        self.alpha_perc = alpha_perc
        self.q_low = q_low
        self.q_high = q_high
        self.weight_tau = weight_tau
        self.weight_floor = weight_floor
        self.ema_decay = ema_decay
        self.fidelity_ratio_threshold = fidelity_ratio_threshold
        self.fidelity_sigmoid_k = fidelity_sigmoid_k
        self.fidelity_min_views = fidelity_min_views
        self.fidelity_opacity_threshold = fidelity_opacity_threshold

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

    def _canonicalize_screen_radii(self, radii: torch.Tensor | None) -> torch.Tensor | None:
        """把 renderer 返回的投影半径统一压成标量半径.

        当前代码最初按 `[B, V, N]` 理解 `radii`.
        但真实 `gsplat`/本仓库渲染链路里,它可能已经是 `[B, V, N, 2]`,
        表示屏幕空间两个主切向方向上的投影半径.

        这里保守地取 `amax(dim=-1)`:
        - 可见性判断保持“任一方向有支撑就算可见”
        - 扩张 kernel 也更偏保守,不容易把真实支撑低估掉
        """

        if not isinstance(radii, torch.Tensor):
            return None

        radii = radii.detach().float()
        if radii.ndim == 4 and radii.shape[-1] == 2:
            return radii.amax(dim=-1)
        if radii.ndim == 4 and radii.shape[-1] == 1:
            return radii.squeeze(-1)
        return radii

    def _prepare_fidelity_inputs(
        self,
        render_meta: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """把 fidelity 需要的 meta 字段整理成统一 shape."""

        if render_meta is None:
            return None

        radii = self._canonicalize_screen_radii(render_meta.get("radii"))
        opacities = render_meta.get("opacities")
        if not isinstance(radii, torch.Tensor) or not isinstance(opacities, torch.Tensor):
            return None

        radii = radii.detach().float()
        opacities = opacities.detach().float()

        if radii.ndim == 2:
            radii = radii.unsqueeze(0)
        if opacities.ndim == 2:
            opacities = opacities.unsqueeze(0)

        if radii.ndim != 3 or opacities.shape != radii.shape:
            return None
        return radii, opacities

    def compute_gaussian_fidelity_diagnostics(
        self,
        render_meta: dict[str, torch.Tensor] | None,
        eps: float = 1e-6,
    ) -> dict[str, torch.Tensor] | None:
        """基于跨视图半径统计构造更接近 SplatSuRe 的 fidelity 诊断.

        当前最小实现保留三件核心语义:
        1. 只统计真正可见的 view
        2. fidelity 由 `rho = r_max / r_min` 决定
        3. `num_times_seen < fidelity_min_views` 时直接记为低 fidelity
        """

        prepared = self._prepare_fidelity_inputs(render_meta)
        if prepared is None:
            return None
        radii, opacities = prepared

        finite_mask = torch.isfinite(radii) & torch.isfinite(opacities)
        visible_mask = finite_mask & (radii > 0) & (opacities > self.fidelity_opacity_threshold)
        num_times_seen = visible_mask.sum(dim=1)
        has_support = num_times_seen > 0

        positive_inf = torch.full_like(radii, float("inf"))
        negative_inf = torch.full_like(radii, float("-inf"))
        radii_for_min = torch.where(visible_mask, radii, positive_inf)
        radii_for_max = torch.where(visible_mask, radii, negative_inf)

        r_min = radii_for_min.amin(dim=1)
        r_max = radii_for_max.amax(dim=1)
        r_min = torch.where(has_support, r_min, torch.zeros_like(r_min))
        r_max = torch.where(has_support, r_max, torch.zeros_like(r_max))

        rho = torch.where(
            has_support,
            r_max / r_min.clamp_min(eps),
            torch.zeros_like(r_max),
        )
        fidelity_score = torch.sigmoid((self.fidelity_ratio_threshold - rho) * self.fidelity_sigmoid_k)
        enough_views = num_times_seen >= self.fidelity_min_views
        fidelity_score = fidelity_score * enough_views.to(dtype=fidelity_score.dtype)

        argmax_view = radii_for_max.argmax(dim=1)
        argmax_view = torch.where(has_support, argmax_view, torch.full_like(argmax_view, -1))
        max_view_mask = visible_mask & torch.isclose(
            radii,
            r_max[:, None, :],
            rtol=1e-5,
            atol=1e-6,
        )

        return {
            "fidelity_score": fidelity_score.detach(),
            "r_min": r_min.detach(),
            "r_max": r_max.detach(),
            "rho": rho.detach(),
            "num_times_seen": num_times_seen.detach(),
            "argmax_view": argmax_view.detach(),
            "max_view_mask": max_view_mask.detach(),
        }

    def compute_gaussian_fidelity_score(
        self,
        render_meta: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor | None:
        """返回 per-Gaussian fidelity score.

        这里保留旧接口,方便 `Stage 3SR` / `L_sampling_smooth` 继续直接消费 tensor.
        更完整的 `Phase A` 诊断请走 `compute_gaussian_fidelity_diagnostics()`.
        """

        diagnostics = self.compute_gaussian_fidelity_diagnostics(render_meta)
        if diagnostics is None:
            return None
        return diagnostics["fidelity_score"]

    def summarize_fidelity_stats(self, fidelity_score: torch.Tensor | None) -> dict[str, float]:
        """输出 fidelity score 的基础统计."""

        if fidelity_score is None:
            return {}

        return {
            "fidelity_min": float(fidelity_score.min().item()),
            "fidelity_max": float(fidelity_score.max().item()),
            "fidelity_mean": float(fidelity_score.mean().item()),
            "fidelity_std": float(fidelity_score.std(unbiased=False).item()),
        }

    def summarize_fidelity_diagnostics(
        self,
        fidelity_diagnostics: dict[str, torch.Tensor] | None,
    ) -> dict[str, float]:
        """输出 Phase A 额外需要的跨视图统计."""

        if fidelity_diagnostics is None:
            return {}

        rho = fidelity_diagnostics.get("rho")
        num_times_seen = fidelity_diagnostics.get("num_times_seen")
        max_view_mask = fidelity_diagnostics.get("max_view_mask")
        if not all(isinstance(item, torch.Tensor) for item in [rho, num_times_seen, max_view_mask]):
            return {}

        rho = rho.float()
        num_times_seen = num_times_seen.float()
        max_view_mask = max_view_mask.float()
        return {
            "fidelity_rho_mean": float(rho.mean().item()),
            "fidelity_rho_max": float(rho.max().item()),
            "fidelity_num_seen_mean": float(num_times_seen.mean().item()),
            "fidelity_num_seen_min": float(num_times_seen.min().item()),
            "fidelity_num_seen_max": float(num_times_seen.max().item()),
            "fidelity_max_view_ties_mean": float(max_view_mask.sum(dim=1).mean().item()),
        }

    def build_sr_selection_weight(
        self,
        render_meta: dict[str, torch.Tensor] | None,
        fidelity_score: torch.Tensor | None,
        native_hw: tuple[int, int],
        output_hw: tuple[int, int] | None = None,
        fidelity_diagnostics: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """把 per-Gaussian fidelity 投影成 reference 尺度的选择图.

        第一版只做最保守的 `SplatSuRe-style selective SR`:
        - 低 fidelity 的 Gaussian 才值得引入 SR
        - 先把 Gaussian 中心投到像素网格
        - 再按投影半径做一次轻量 max-pool 扩张
        """

        if render_meta is None or fidelity_score is None:
            return None

        means2d = render_meta.get("means2d")
        radii = render_meta.get("radii")
        opacities = render_meta.get("opacities")
        if not all(isinstance(item, torch.Tensor) for item in [means2d, radii, opacities]):
            return None

        means2d = means2d.detach().float()
        radii = self._canonicalize_screen_radii(radii)
        if radii is None:
            return None
        opacities = opacities.detach().float()
        fidelity_score = fidelity_score.detach().float()

        if means2d.ndim == 3:
            means2d = means2d.unsqueeze(0)
        if radii.ndim == 2:
            radii = radii.unsqueeze(0)
        if opacities.ndim == 2:
            opacities = opacities.unsqueeze(0)
        if fidelity_score.ndim == 1:
            fidelity_score = fidelity_score.unsqueeze(0)

        if means2d.ndim != 4 or means2d.shape[-1] != 2:
            return None
        if radii.ndim != 3 or opacities.shape != radii.shape:
            return None

        batch_size, num_views, num_gaussians = radii.shape
        if means2d.shape[:3] != (batch_size, num_views, num_gaussians):
            return None
        if fidelity_score.shape != (batch_size, num_gaussians):
            return None

        output_height, output_width = output_hw or native_hw
        native_height, native_width = native_hw
        if min(output_height, output_width, native_height, native_width) <= 0:
            return None

        scale_x = float(output_width) / float(native_width)
        scale_y = float(output_height) / float(native_height)

        # 低 fidelity 区域更该吃 SR.
        # 当前实现优先把低 fidelity Gaussian 投到自己的 max-view 上.
        # 这样 selection 不会再对所有可见视角平均撒开.
        low_fidelity = (1.0 - fidelity_score).clamp(0.0, 1.0)
        visible_mask = radii > 0
        finite_mask = torch.isfinite(means2d).all(dim=-1)
        scatter_mask = visible_mask
        if isinstance(fidelity_diagnostics, dict):
            max_view_mask = fidelity_diagnostics.get("max_view_mask")
            if isinstance(max_view_mask, torch.Tensor):
                if max_view_mask.ndim == 2:
                    max_view_mask = max_view_mask.unsqueeze(0)
                if max_view_mask.shape == (batch_size, num_views, num_gaussians):
                    scatter_mask = scatter_mask & max_view_mask.to(device=means2d.device).bool()
        scatter_weight = low_fidelity[:, None, :] * opacities.clamp(0.0, 1.0) * scatter_mask.to(dtype=means2d.dtype)
        scatter_weight = scatter_weight * finite_mask.to(dtype=means2d.dtype)

        means_x = torch.round(means2d[..., 0] * scale_x).long().clamp(0, output_width - 1)
        means_y = torch.round(means2d[..., 1] * scale_y).long().clamp(0, output_height - 1)
        flat_index = means_y * output_width + means_x

        selection_flat = torch.zeros(
            batch_size,
            num_views,
            output_height * output_width,
            dtype=means2d.dtype,
            device=means2d.device,
        )
        selection_flat.scatter_reduce_(
            dim=2,
            index=flat_index,
            src=scatter_weight,
            reduce="amax",
            include_self=True,
        )
        selection_map = selection_flat.view(batch_size, num_views, 1, output_height, output_width)

        # 用 view 内典型投影半径做一次轻量扩张.
        # 这样 SR 权重不会只剩零星中心点,但也不会立刻退化成全图监督.
        radius_scale = max(scale_x, scale_y)
        pooled_views: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            pooled_batch_views: list[torch.Tensor] = []
            for view_index in range(num_views):
                positive_radii = radii[batch_index, view_index][visible_mask[batch_index, view_index]]
                if positive_radii.numel() == 0:
                    pooled_batch_views.append(selection_map[batch_index, view_index : view_index + 1])
                    continue

                expand_radius = int(
                    torch.round(positive_radii.median().mul(radius_scale).clamp(min=1.0, max=4.0)).item()
                )
                kernel_size = expand_radius * 2 + 1
                pooled_batch_views.append(
                    F.max_pool2d(
                        selection_map[batch_index, view_index : view_index + 1],
                        kernel_size=kernel_size,
                        stride=1,
                        padding=expand_radius,
                    )
                )
            pooled_views.append(torch.cat(pooled_batch_views, dim=0))

        return torch.stack(pooled_views, dim=0).clamp(0.0, 1.0)

    def combine_sr_weights(
        self,
        w_robust: torch.Tensor,
        w_sr_select: torch.Tensor,
    ) -> torch.Tensor:
        """组合 native robust 权重与 selective SR 权重."""

        if w_robust.shape != w_sr_select.shape:
            raise ValueError("w_robust and w_sr_select must have the same shape.")
        if w_robust.ndim != 5:
            raise ValueError("Expected SR weights with shape [B, V, 1, H, W].")

        return (w_robust.clamp(0.0, 1.0) * w_sr_select.clamp(0.0, 1.0)).clamp(0.0, 1.0)
