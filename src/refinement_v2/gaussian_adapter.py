"""高斯参数适配层.

这个模块负责把 `.ply` 和 refinement 中的可训练参数连接起来.
当前实现优先满足 Stage 2A / Stage 2B 的冻结与优化需求.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import StageHyperParams


class GaussianAdapter(nn.Module):
    """把高斯 tensor 包装成便于分阶段优化的适配器."""

    def __init__(
        self,
        gaussians: torch.Tensor,
        scale_tail_threshold: float = 0.25,
        opacity_low_threshold: float = 0.10,
    ) -> None:
        super().__init__()

        if gaussians.ndim == 3:
            if gaussians.shape[0] != 1:
                raise ValueError("GaussianAdapter only supports batch size 1.")
            gaussians = gaussians[0]

        if gaussians.ndim != 2 or gaussians.shape[1] != 14:
            raise ValueError(f"Expected gaussians with shape [N, 14], got {tuple(gaussians.shape)}.")

        gaussians = gaussians.detach().clone().float()
        self.means = nn.Parameter(gaussians[:, 0:3])
        self.opacity = nn.Parameter(gaussians[:, 3:4])
        self.scales = nn.Parameter(gaussians[:, 4:7])
        self.rotations = nn.Parameter(gaussians[:, 7:11])
        self.colors = nn.Parameter(gaussians[:, 11:14])

        self.register_buffer("initial_means", gaussians[:, 0:3].detach().clone())
        self.register_buffer("initial_rotations", gaussians[:, 7:11].detach().clone())
        self.scale_tail_threshold = scale_tail_threshold
        self.opacity_low_threshold = opacity_low_threshold

    @classmethod
    def from_tensor(
        cls,
        gaussians: torch.Tensor,
        scale_tail_threshold: float = 0.25,
        opacity_low_threshold: float = 0.10,
    ) -> "GaussianAdapter":
        """从内存中的 tensor 直接创建适配器."""

        return cls(
            gaussians=gaussians,
            scale_tail_threshold=scale_tail_threshold,
            opacity_low_threshold=opacity_low_threshold,
        )

    @classmethod
    def from_ply(
        cls,
        path: Path,
        compatible: bool = True,
        scale_tail_threshold: float = 0.25,
        opacity_low_threshold: float = 0.10,
    ) -> "GaussianAdapter":
        """从 `.ply` 文件读取高斯."""

        from plyfile import PlyData

        ply_data = PlyData.read(str(path))
        element = ply_data.elements[0]

        xyz = np.stack(
            (
                np.asarray(element["x"]),
                np.asarray(element["y"]),
                np.asarray(element["z"]),
            ),
            axis=1,
        )
        opacity = np.asarray(element["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        shs[:, 0] = np.asarray(element["f_dc_0"])
        shs[:, 1] = np.asarray(element["f_dc_1"])
        shs[:, 2] = np.asarray(element["f_dc_2"])

        scale_names = [prop.name for prop in element.properties if prop.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for index, attr_name in enumerate(scale_names):
            scales[:, index] = np.asarray(element[attr_name])

        rotation_names = [prop.name for prop in element.properties if prop.name.startswith("rot_")]
        rotations = np.zeros((xyz.shape[0], len(rotation_names)), dtype=np.float32)
        for index, attr_name in enumerate(rotation_names):
            rotations[:, index] = np.asarray(element[attr_name])

        gaussians = np.concatenate([xyz, opacity, scales, rotations, shs], axis=1)
        gaussians_torch = torch.from_numpy(gaussians).float()

        if compatible:
            gaussians_torch[:, 3:4] = torch.sigmoid(gaussians_torch[:, 3:4])
            gaussians_torch[:, 4:7] = torch.exp(gaussians_torch[:, 4:7])
            gaussians_torch[:, 11:] = 0.28209479177387814 * gaussians_torch[:, 11:] + 0.5

        return cls.from_tensor(
            gaussians=gaussians_torch,
            scale_tail_threshold=scale_tail_threshold,
            opacity_low_threshold=opacity_low_threshold,
        )

    def to_tensor(self) -> torch.Tensor:
        """导出为 `[1, N, 14]` 的高斯 tensor."""

        gaussians = torch.cat(
            [self.means, self.opacity, self.scales, self.rotations, self.colors],
            dim=-1,
        )
        return gaussians.unsqueeze(0)

    def copy_from_tensor(self, gaussians: torch.Tensor) -> None:
        """用外部 tensor 覆盖当前参数值."""

        if gaussians.ndim == 3:
            gaussians = gaussians[0]
        if gaussians.shape != (self.means.shape[0], 14):
            raise ValueError(f"Expected gaussian tensor with shape {(self.means.shape[0], 14)}, got {tuple(gaussians.shape)}")

        with torch.no_grad():
            self.means.copy_(gaussians[:, 0:3])
            self.opacity.copy_(gaussians[:, 3:4])
            self.scales.copy_(gaussians[:, 4:7])
            self.rotations.copy_(gaussians[:, 7:11])
            self.colors.copy_(gaussians[:, 11:14])

    def freeze_for_stage(self, stage_name: str) -> None:
        """按阶段切换参数冻结状态."""

        stage_to_trainable = {
            "stage2a": {"opacity", "scales", "colors"},
            "stage3a": {"opacity", "scales", "colors"},
            "stage3sr": {"opacity", "scales", "colors"},
            "stage2b": {"means", "opacity", "scales", "rotations", "colors"},
            "stage3b": {"means", "opacity", "scales", "rotations", "colors"},
            "phase3": set(),
            "phase4": {"means", "opacity", "scales", "rotations", "colors"},
        }
        trainable_names = stage_to_trainable.get(stage_name, {"opacity", "scales", "colors"})

        for name in ["means", "opacity", "scales", "rotations", "colors"]:
            parameter = getattr(self, name)
            parameter.requires_grad_(name in trainable_names)

    def build_optimizer(self, stage_name: str, hparams: StageHyperParams) -> torch.optim.Optimizer:
        """按阶段构造 Adam 优化器."""

        param_groups: list[dict] = []

        if self.opacity.requires_grad:
            param_groups.append({"params": [self.opacity], "lr": hparams.lr_opacity, "name": "opacity"})
        if self.colors.requires_grad:
            param_groups.append({"params": [self.colors], "lr": hparams.lr_color, "name": "colors"})
        if self.scales.requires_grad:
            param_groups.append({"params": [self.scales], "lr": hparams.lr_scale, "name": "scales"})
        if self.rotations.requires_grad:
            param_groups.append({"params": [self.rotations], "lr": hparams.lr_scale, "name": "rotations"})
        if self.means.requires_grad:
            param_groups.append({"params": [self.means], "lr": hparams.lr_means, "name": "means"})

        if not param_groups:
            raise ValueError(f"No trainable gaussian parameters are enabled for stage {stage_name}.")

        return torch.optim.Adam(param_groups)

    def collect_prune_candidates(self, threshold: float | None = None) -> torch.Tensor:
        """找出低 opacity 的 pruning 候选.

        这里只做识别,不直接修改参数.
        这样 runner 可以先看候选统计,再决定是否真正执行裁剪.
        """

        active_threshold = self.opacity_low_threshold if threshold is None else threshold
        return self.opacity.detach().squeeze(-1) < active_threshold

    def _apply_keep_mask(self, keep_mask: torch.Tensor) -> None:
        """按 keep mask 同步裁剪全部高斯参数与参考 buffer."""

        keep_mask = keep_mask.detach().to(device=self.means.device, dtype=torch.bool)
        if keep_mask.ndim != 1 or keep_mask.shape[0] != self.means.shape[0]:
            raise ValueError(f"Expected keep_mask with shape {(self.means.shape[0],)}, got {tuple(keep_mask.shape)}")
        if not torch.any(keep_mask):
            raise ValueError("Refusing to prune all gaussians.")

        # 逐个重建 Parameter,避免 optimizer 继续持有旧 tensor 引用.
        # `requires_grad` 继承原状态,这样 stage freeze 语义不会丢.
        parameter_names = ["means", "opacity", "scales", "rotations", "colors"]
        for parameter_name in parameter_names:
            old_parameter = getattr(self, parameter_name)
            new_parameter = nn.Parameter(old_parameter.detach()[keep_mask].clone())
            new_parameter.requires_grad_(old_parameter.requires_grad)
            setattr(self, parameter_name, new_parameter)

        # 这些初始参考 buffer 也必须同步裁掉.
        # 否则后续位置/旋转正则就会和当前参数数量错位.
        self.initial_means = self.initial_means.detach()[keep_mask].clone()
        self.initial_rotations = self.initial_rotations.detach()[keep_mask].clone()

    def prune_low_opacity(
        self,
        threshold: float | None = None,
        max_fraction: float = 0.02,
        min_gaussians_to_keep: int = 1,
    ) -> dict[str, float | int | bool | list[int]]:
        """裁掉一批低 opacity 高斯.

        裁剪策略保持保守:
        1. 只看 opacity.
        2. 每次最多裁掉固定比例.
        3. 永远保留至少 `min_gaussians_to_keep` 个高斯.
        """

        if max_fraction < 0:
            raise ValueError(f"max_fraction must be non-negative, got {max_fraction}.")

        threshold_value = self.opacity_low_threshold if threshold is None else threshold
        num_before = int(self.means.shape[0])
        min_keep = max(1, min(int(min_gaussians_to_keep), num_before))

        before_stats = self.summarize_gaussian_stats()
        candidate_mask = self.collect_prune_candidates(threshold=threshold_value)
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        candidate_count = int(candidate_indices.numel())

        max_prune_by_fraction = int(math.floor(num_before * max_fraction))
        max_prune_by_capacity = max(0, num_before - min_keep)
        prune_count = min(candidate_count, max_prune_by_fraction, max_prune_by_capacity)

        if prune_count > 0:
            opacity_values = self.opacity.detach().squeeze(-1)
            sorted_candidate_indices = candidate_indices[torch.argsort(opacity_values[candidate_indices])]
            prune_indices = sorted_candidate_indices[:prune_count]
            keep_mask = torch.ones(num_before, device=self.means.device, dtype=torch.bool)
            keep_mask[prune_indices] = False
            self._apply_keep_mask(keep_mask)
            pruned_index_list = prune_indices.detach().cpu().tolist()
        else:
            pruned_index_list = []

        after_stats = self.summarize_gaussian_stats()
        return {
            "threshold": float(threshold_value),
            "num_before": num_before,
            "num_after": int(self.means.shape[0]),
            "candidate_count": candidate_count,
            "candidate_fraction": float(candidate_count / max(num_before, 1)),
            "pruned_count": len(pruned_index_list),
            "max_prune_by_fraction": max_prune_by_fraction,
            "min_gaussians_to_keep": min_keep,
            "applied": bool(pruned_index_list),
            # 不把完整索引全量塞进摘要.
            # 大场景下一次 pruning 可能就是几千个索引,写全会让 diagnostics 过重.
            "pruned_indices_preview": pruned_index_list[:32],
            "opacity_lowconf_ratio_before": float(before_stats["opacity_lowconf_ratio"]),
            "opacity_lowconf_ratio_after": float(after_stats["opacity_lowconf_ratio"]),
        }

    def clamp_stage_constraints(self, stage_name: str, hparams: StageHyperParams) -> None:
        """在 optimizer step 后施加阶段约束."""

        with torch.no_grad():
            self.opacity.data.clamp_(1e-4, 1 - 1e-4)
            self.scales.data.clamp_(1e-4)

            rotation_norm = self.rotations.data.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self.rotations.data = self.rotations.data / rotation_norm

            if stage_name not in {"stage2b", "stage3b", "phase4"}:
                return
            if hparams.means_delta_cap <= 0:
                return

            min_means = self.initial_means - hparams.means_delta_cap
            max_means = self.initial_means + hparams.means_delta_cap
            self.means.data.clamp_(min=min_means, max=max_means)

    def summarize_gaussian_stats(self) -> dict[str, float]:
        """输出一份轻量高斯统计,供 diagnostics 使用."""

        with torch.no_grad():
            max_scale = self.scales.detach().max(dim=-1).values
            opacity = self.opacity.detach().squeeze(-1)

            scale_tail_ratio = float((max_scale > self.scale_tail_threshold).float().mean().item())
            opacity_lowconf_ratio = float((opacity < self.opacity_low_threshold).float().mean().item())

            return {
                "num_gaussians": int(self.means.shape[0]),
                "scale_mean": float(self.scales.detach().mean().item()),
                "opacity_mean": float(self.opacity.detach().mean().item()),
                "scale_tail_ratio": scale_tail_ratio,
                "opacity_lowconf_ratio": opacity_lowconf_ratio,
            }

    def export_ply(self, path: Path, compatible: bool = True) -> None:
        """把当前高斯导出为 `.ply`."""

        from plyfile import PlyData, PlyElement

        means = self.means.detach().cpu()
        opacity = self.opacity.detach().cpu()
        scales = self.scales.detach().cpu()
        rotations = self.rotations.detach().cpu()
        colors = self.colors.detach().cpu()

        mask = opacity.squeeze(-1) >= 0.005
        means = means[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        colors = colors[mask]

        if compatible:
            opacity = torch.logit(opacity.clamp(1e-6, 1 - 1e-6))
            scales = torch.log(scales.clamp_min(1e-8))
            colors = (colors - 0.5) / 0.28209479177387814

        xyzs = means.numpy()
        opacities = opacity.numpy()
        scales_np = scales.numpy()
        rotations_np = rotations.numpy()
        colors_np = colors.numpy()

        attributes = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        dtype_full = [(attribute, "f4") for attribute in attributes]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        concatenated = np.concatenate([xyzs, colors_np, opacities, scales_np, rotations_np], axis=1)
        elements[:] = list(map(tuple, concatenated))
        ply_element = PlyElement.describe(elements, "vertex")
        PlyData([ply_element]).write(str(path))
