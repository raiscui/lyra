"""`refinement_v2` 的主控执行器."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from .config import RefinementRunConfig, StageHyperParams
from .data_loader import SceneBundle
from .diagnostics import DiagnosticsWriter
from .gaussian_adapter import GaussianAdapter
from .losses import (
    compute_means_anchor_loss,
    compute_opacity_sparse_loss,
    compute_patch_perceptual_loss,
    compute_pose_regularization,
    compute_rotation_regularization_loss,
    compute_sampling_smooth_loss,
    compute_scale_tail_loss,
    compute_weighted_rgb_loss,
)
from .stage_controller import StageController
from .state_io import load_latest_state, save_state
from .weight_builder import WeightBuilder


class GaussianSceneRenderer:
    """基于现有 `GaussianRenderer` 的薄封装."""

    def __init__(self, scene: SceneBundle) -> None:
        height, width = scene.gt_images.shape[-2:]
        opt = SimpleNamespace(
            img_size=(height, width),
            znear=0.1,
            zfar=500.0,
            gs_view_chunk_size=1,
        )
        opt.get = lambda key, default=None: getattr(opt, key, default)

        from src.rendering.gs import GaussianRenderer

        self.renderer = GaussianRenderer(opt)

    def render(self, gaussians: torch.Tensor, scene: SceneBundle) -> dict[str, torch.Tensor]:
        """渲染当前高斯场景."""

        return self.renderer.render(gaussians, scene.cam_view, intrinsics=scene.intrinsics)


class RefinementRunner:
    """串起 `Phase 0 -> Phase 4` 的 refinement 主控."""

    def __init__(
        self,
        scene: SceneBundle,
        gaussians: GaussianAdapter,
        diagnostics: DiagnosticsWriter,
        controller: StageController,
        hparams: StageHyperParams,
        renderer: Any | None = None,
    ) -> None:
        self.controller = controller
        self.run_config: RefinementRunConfig = controller.run_config
        self.hparams = hparams
        self.device = self._resolve_device(self.run_config.device)

        self.scene = self._move_scene_to_device(scene, self.device)
        self.gaussians = gaussians.to(self.device)
        self.diagnostics = diagnostics
        self.renderer = renderer if renderer is not None else GaussianSceneRenderer(self.scene)
        self.weight_builder = WeightBuilder(
            alpha_rgb=hparams.alpha_rgb,
            alpha_perc=hparams.alpha_perc,
            q_low=hparams.q_low,
            q_high=hparams.q_high,
            weight_tau=hparams.weight_tau,
            weight_floor=hparams.weight_floor,
            ema_decay=hparams.ema_decay,
        )

        self.current_stage = "init"
        self.prev_weight_map: torch.Tensor | None = None
        self.pose_delta: torch.Tensor | None = None
        self.latest_render_meta: dict[str, Any] | None = None
        self.gaussian_fidelity_score: torch.Tensor | None = None
        self.sr_selection_map: torch.Tensor | None = None
        self.renderer_cache: dict[tuple[int, int], GaussianSceneRenderer] = {}
        self.diagnostics_state: dict[str, Any] = {
            "phase_reached": "init",
            "used_pose_refinement": False,
            "used_joint_fallback": False,
            "stage2a_mode_requested": self.run_config.stage2a_mode,
            "stage3sr_enabled": self._stage2a_should_run_stage3sr(),
        }
        self.visual_artifacts: dict[str, str] = {}

    def _resolve_device(self, device_name: str) -> torch.device:
        """解析并回退设备选择."""

        if device_name.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)

    def _move_scene_to_device(self, scene: SceneBundle, device: torch.device) -> SceneBundle:
        """把场景 tensor 迁移到目标设备."""

        return SceneBundle(
            gt_images=scene.gt_images.to(device),
            cam_view=scene.cam_view.to(device),
            intrinsics=scene.intrinsics.to(device),
            frame_indices=scene.frame_indices,
            scene_index=scene.scene_index,
            view_id=scene.view_id,
            target_index=scene.target_index.to(device) if isinstance(scene.target_index, torch.Tensor) else scene.target_index,
            file_name=scene.file_name,
            reference_images=scene.reference_images.to(device) if isinstance(scene.reference_images, torch.Tensor) else scene.reference_images,
            intrinsics_ref=scene.intrinsics_ref.to(device) if isinstance(scene.intrinsics_ref, torch.Tensor) else scene.intrinsics_ref,
            native_hw=scene.native_hw,
            reference_hw=scene.reference_hw,
            reference_mode=scene.reference_mode,
            sr_scale=scene.sr_scale,
        )

    def _get_renderer_for_scene(self, scene: SceneBundle) -> Any:
        """按目标分辨率选择可复用的 renderer."""

        if not isinstance(self.renderer, GaussianSceneRenderer):
            return self.renderer

        height, width = scene.gt_images.shape[-2:]
        base_hw = self.scene.gt_images.shape[-2:]
        if (height, width) == base_hw:
            return self.renderer

        cache_key = (height, width)
        if cache_key not in self.renderer_cache:
            self.renderer_cache[cache_key] = GaussianSceneRenderer(scene)
        return self.renderer_cache[cache_key]

    def render_scene(self, scene: SceneBundle) -> dict[str, torch.Tensor]:
        """渲染任意分辨率的场景视图."""

        renderer = self._get_renderer_for_scene(scene)
        render_output = renderer.render(self.gaussians.to_tensor(), scene)
        if "images_pred" not in render_output:
            raise KeyError("Renderer output must contain `images_pred`.")
        return render_output

    def render_current_scene(self) -> dict[str, torch.Tensor]:
        """渲染当前高斯场景."""

        return self.render_scene(self.scene)

    def _extract_render_meta(self, render_output: dict[str, torch.Tensor]) -> dict[str, Any] | None:
        """从 renderer 输出中提取可选的 dense meta."""

        render_meta = render_output.get("render_meta")
        if isinstance(render_meta, dict):
            return render_meta
        return None

    def _build_default_fidelity_score(self) -> torch.Tensor:
        """在 renderer 不提供 meta 时回退到全 1 fidelity."""

        return torch.ones(
            self.scene.gt_images.shape[0],
            self.gaussians.means.shape[0],
            dtype=self.gaussians.means.dtype,
            device=self.gaussians.means.device,
        )

    def _build_default_sr_selection_map(self, residual_map: torch.Tensor) -> torch.Tensor:
        """在缺少 meta 时回退到 reference 尺度的全 1 selection map."""

        reference_images = self._get_reference_images()
        return torch.ones(
            reference_images.shape[0],
            reference_images.shape[1],
            1,
            reference_images.shape[-2],
            reference_images.shape[-1],
            dtype=residual_map.dtype,
            device=residual_map.device,
        ).detach()

    def _write_phase3s_artifacts(
        self,
        stage_name: str,
        fidelity_score: torch.Tensor,
        sr_selection_map: torch.Tensor,
    ) -> None:
        """把 Phase 3S 的诊断产物落盘."""

        self.diagnostics.write_gaussian_fidelity_summary(fidelity_score)
        self.diagnostics.write_sr_selection_stats(sr_selection_map)
        for frame_id in range(sr_selection_map.shape[1]):
            self.diagnostics.save_sr_selection_map(stage_name, frame_id, sr_selection_map[:, frame_id])

    def _patch_supervision_configured(self) -> bool:
        """判断当前是否配置了 patch supervision 所需参数."""

        return self.hparams.patch_size > 0 and (
            self.hparams.lambda_patch_rgb > 0.0 or self.hparams.lambda_patch_perceptual > 0.0
        )

    def _stage2a_should_run_stage3sr(self) -> bool:
        """根据当前模式和参数,判断 Stage 2A 是否应进入增强链路."""

        if self.run_config.stage2a_mode == "legacy":
            return False
        return self._patch_supervision_configured()

    def _resolve_stage2a_mode(self) -> str:
        """把 `auto/legacy/enhanced` 解析成本轮真正执行的模式."""

        requested_mode = self.run_config.stage2a_mode
        if requested_mode == "legacy":
            return "legacy"
        if requested_mode == "enhanced":
            if not self._patch_supervision_configured():
                raise RuntimeError(
                    "stage2a_mode=enhanced requires patch supervision. "
                    "Please set --patch-size > 0 and enable at least one patch loss weight."
                )
            return "enhanced"
        if self._patch_supervision_configured():
            return "enhanced"
        return "legacy"

    def _get_reference_images(self) -> torch.Tensor:
        """统一解析当前 reference 图像张量."""

        if isinstance(self.scene.reference_images, torch.Tensor):
            return self.scene.reference_images
        return self.scene.gt_images

    def _resolve_patch_sizes(self) -> tuple[int, int, int]:
        """解析 native/reference patch 尺寸和缩放倍率."""

        reference_patch_size = int(self.hparams.patch_size)
        if reference_patch_size <= 0:
            raise ValueError("patch_size must be positive when patch supervision is enabled.")

        scale_value = float(self.scene.sr_scale)
        scale_int = int(round(scale_value))
        if scale_int <= 0 or abs(scale_value - scale_int) > 1e-6:
            raise ValueError(f"Patch supervision requires integer sr_scale, got {self.scene.sr_scale}.")
        if reference_patch_size % scale_int != 0:
            raise ValueError(
                f"patch_size {reference_patch_size} must be divisible by sr_scale {scale_int}."
            )

        native_patch_size = reference_patch_size // scale_int
        native_height, native_width = self.scene.gt_images.shape[-2:]
        reference_images = self._get_reference_images()
        reference_height, reference_width = reference_images.shape[-2:]
        if native_patch_size > native_height or native_patch_size > native_width:
            raise ValueError(
                f"native patch size {native_patch_size} exceeds native image size {(native_height, native_width)}."
            )
        if reference_patch_size > reference_height or reference_patch_size > reference_width:
            raise ValueError(
                f"reference patch size {reference_patch_size} exceeds reference image size {(reference_height, reference_width)}."
            )
        return native_patch_size, reference_patch_size, scale_int

    def sample_patch_windows(self, residual_map: torch.Tensor) -> torch.Tensor:
        """根据 residual 热点在 native 尺度采样 patch window."""

        native_patch_size, _, _ = self._resolve_patch_sizes()
        batch_size, num_views, _, height, width = residual_map.shape
        patch_windows = torch.zeros(batch_size, num_views, 4, dtype=torch.long, device=residual_map.device)
        residual_energy = residual_map.detach().mean(dim=2)

        for batch_index in range(batch_size):
            for view_index in range(num_views):
                hotspot_index = int(torch.argmax(residual_energy[batch_index, view_index]).item())
                hotspot_y = hotspot_index // width
                hotspot_x = hotspot_index % width
                top = min(max(hotspot_y - native_patch_size // 2, 0), height - native_patch_size)
                left = min(max(hotspot_x - native_patch_size // 2, 0), width - native_patch_size)
                patch_windows[batch_index, view_index] = torch.tensor(
                    [top, left, native_patch_size, native_patch_size],
                    dtype=torch.long,
                    device=residual_map.device,
                )
        return patch_windows

    def map_patch_windows_to_reference(self, patch_windows_native: torch.Tensor) -> torch.Tensor:
        """把 native patch window 映射到 reference 尺度."""

        _, _, scale_int = self._resolve_patch_sizes()
        patch_windows_ref = patch_windows_native.detach().clone()
        patch_windows_ref[..., 0:2] *= scale_int
        patch_windows_ref[..., 2:4] *= scale_int
        return patch_windows_ref

    def gather_reference_patch(self, patch_windows_ref: torch.Tensor) -> torch.Tensor:
        """从 reference 图像中提取 patch."""

        reference_images = self._get_reference_images()
        return self._gather_tensor_patch(reference_images, patch_windows_ref)

    def _gather_tensor_patch(self, source_tensor: torch.Tensor, patch_windows: torch.Tensor) -> torch.Tensor:
        """从任意 `[B, V, C, H, W]` 张量中抽取 patch."""

        if source_tensor.ndim != 5:
            raise ValueError("Expected source_tensor with shape [B, V, C, H, W].")

        patches: list[torch.Tensor] = []
        for batch_index in range(source_tensor.shape[0]):
            view_patches: list[torch.Tensor] = []
            for view_index in range(source_tensor.shape[1]):
                top, left, patch_height, patch_width = patch_windows[batch_index, view_index].tolist()
                view_patches.append(
                    source_tensor[batch_index, view_index, :, top : top + patch_height, left : left + patch_width]
                )
            patches.append(torch.stack(view_patches, dim=0))
        return torch.stack(patches, dim=0)

    def _resize_patch_tensor(self, patch_tensor: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
        """把 patch 张量调整到目标空间分辨率."""

        if patch_tensor.shape[-2:] == output_hw:
            return patch_tensor

        batch_size, num_views, channels, _, _ = patch_tensor.shape
        resized = F.interpolate(
            patch_tensor.reshape(batch_size * num_views, channels, *patch_tensor.shape[-2:]),
            size=output_hw,
            mode="bilinear",
            align_corners=False,
        )
        return resized.view(batch_size, num_views, channels, *output_hw)

    def build_patch_intrinsics(self, patch_windows_ref: torch.Tensor) -> torch.Tensor:
        """基于 reference intrinsics 构造 patch camera intrinsics."""

        base_intrinsics = self.scene.intrinsics_ref if isinstance(self.scene.intrinsics_ref, torch.Tensor) else self.scene.intrinsics
        patch_intrinsics = base_intrinsics.detach().clone()
        patch_intrinsics[..., 2] -= patch_windows_ref[..., 1].to(dtype=patch_intrinsics.dtype)
        patch_intrinsics[..., 3] -= patch_windows_ref[..., 0].to(dtype=patch_intrinsics.dtype)
        return patch_intrinsics

    def render_patch_prediction(
        self,
        patch_windows_native: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在 reference 尺度渲染 patch 并返回对应参考图."""

        patch_windows_ref = self.map_patch_windows_to_reference(patch_windows_native)
        reference_patch = self.gather_reference_patch(patch_windows_ref)
        patch_intrinsics = self.build_patch_intrinsics(patch_windows_ref)
        native_patch_size, reference_patch_size, _ = self._resolve_patch_sizes()

        patch_scene = SceneBundle(
            gt_images=torch.zeros_like(reference_patch),
            cam_view=self.scene.cam_view,
            intrinsics=patch_intrinsics,
            frame_indices=self.scene.frame_indices,
            scene_index=self.scene.scene_index,
            view_id=self.scene.view_id,
            target_index=self.scene.target_index,
            file_name=self.scene.file_name,
            reference_images=reference_patch,
            intrinsics_ref=patch_intrinsics,
            native_hw=(native_patch_size, native_patch_size),
            reference_hw=(reference_patch_size, reference_patch_size),
            reference_mode=self.scene.reference_mode,
            sr_scale=self.scene.sr_scale,
        )
        patch_render_output = self.render_scene(patch_scene)
        return patch_render_output["images_pred"], reference_patch, patch_intrinsics

    def _compute_psnr(self, pred_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> float:
        """计算一个稳定的 PSNR 指标."""

        mse = float(F.mse_loss(pred_rgb, gt_rgb).item())
        return 10.0 * math.log10(1.0 / (mse + 1e-8))

    def _compute_sharpness(self, pred_rgb: torch.Tensor) -> float:
        """用 Laplacian 方差近似衡量锐度."""

        gray = pred_rgb.mean(dim=2, keepdim=True)
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        gray_2d = gray.reshape(-1, 1, gray.shape[-2], gray.shape[-1])
        laplace = F.conv2d(gray_2d, kernel, padding=1)
        return float(laplace.var().item())

    def _compute_patch_losses(
        self,
        residual_map: torch.Tensor,
        reference_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """统一计算可选 patch supervision 损失.

        Stage 2A 和 Stage 2B 都复用这条逻辑.
        这样 patch 监督的行为只维护一份,不会因为阶段切换而漂移.
        """

        loss_patch_rgb = torch.zeros((), dtype=reference_tensor.dtype, device=reference_tensor.device)
        loss_patch_perceptual = torch.zeros((), dtype=reference_tensor.dtype, device=reference_tensor.device)
        if not self._patch_supervision_configured():
            return loss_patch_rgb, loss_patch_perceptual

        patch_windows_native = self.sample_patch_windows(residual_map)
        pred_patch, reference_patch, _ = self.render_patch_prediction(patch_windows_native)
        patch_windows_ref = self.map_patch_windows_to_reference(patch_windows_native)

        native_weight_map = self.prev_weight_map
        if native_weight_map is None:
            native_weight_map = torch.ones_like(residual_map)
        robust_patch_native = self._gather_tensor_patch(native_weight_map, patch_windows_native)
        robust_patch = self._resize_patch_tensor(robust_patch_native, pred_patch.shape[-2:])

        if self.sr_selection_map is None:
            sr_selection_patch = torch.ones_like(robust_patch)
        else:
            sr_selection_patch = self._gather_tensor_patch(self.sr_selection_map, patch_windows_ref).to(dtype=pred_patch.dtype)

        patch_weights = self.weight_builder.combine_sr_weights(robust_patch.to(dtype=pred_patch.dtype), sr_selection_patch)
        loss_patch_rgb = compute_weighted_rgb_loss(pred_patch, reference_patch, patch_weights)
        loss_patch_perceptual = compute_patch_perceptual_loss(pred_patch, reference_patch)
        return loss_patch_rgb, loss_patch_perceptual

    def _summarize_prediction(
        self,
        pred_rgb: torch.Tensor,
        residual_map: torch.Tensor | None = None,
        weight_map: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """汇总当前预测对应的指标."""

        summary = {
            "psnr": self._compute_psnr(pred_rgb, self.scene.gt_images),
            "sharpness": self._compute_sharpness(pred_rgb),
        }
        summary.update(self.gaussians.summarize_gaussian_stats())
        if residual_map is not None:
            summary["residual_mean"] = float(residual_map.mean().item())
        if weight_map is not None:
            summary.update(self.weight_builder.summarize_weight_stats(weight_map))
        return summary

    def _update_stage2b_diagnostics(self, final_metrics: dict[str, Any]) -> None:
        """把 Stage 2A 的结束状态压成是否进入 Stage 2B 的证据."""

        baseline = self.diagnostics_state.get("baseline", {})
        residual_mean = float(final_metrics.get("residual_mean", 0.0))
        baseline_psnr = float(baseline.get("psnr", final_metrics.get("psnr", 0.0)))
        baseline_scale_tail = float(baseline.get("scale_tail_ratio", final_metrics.get("scale_tail_ratio", 0.0)))
        baseline_opacity_lowconf = float(
            baseline.get("opacity_lowconf_ratio", final_metrics.get("opacity_lowconf_ratio", 0.0))
        )

        scale_tail_improved = float(final_metrics.get("scale_tail_ratio", 0.0)) < baseline_scale_tail - 1e-6
        opacity_improved = float(final_metrics.get("opacity_lowconf_ratio", 0.0)) < baseline_opacity_lowconf - 1e-6
        psnr_healthy = float(final_metrics.get("psnr", 0.0)) >= baseline_psnr - 0.10
        # 真实 `view 5` 里,`residual_mean` 落在 `0.046 ~ 0.048` 时,
        # 视觉上仍然能看到明显的局部双轮廓.
        # 因此这里不能把阈值卡得过死在 `0.05`.
        local_overlap_persistent = residual_mean > 0.045

        # 规格里建议“至少两条证据”再进入 Stage 2B.
        # 当前实现先用已有 diagnostics 能稳定算出来的四项做代理.
        stage2b_signal_count = sum(
            [
                int(scale_tail_improved),
                int(opacity_improved),
                int(psnr_healthy),
                int(local_overlap_persistent),
            ]
        )

        self.diagnostics_state["stage2b_signal_count"] = stage2b_signal_count
        self.diagnostics_state["scale_tail_improved"] = scale_tail_improved
        self.diagnostics_state["opacity_sparse_improved"] = opacity_improved
        self.diagnostics_state["psnr_healthy"] = psnr_healthy
        self.diagnostics_state["need_geometry"] = local_overlap_persistent and stage2b_signal_count >= 2
        self.diagnostics_state["local_overlap_persistent"] = local_overlap_persistent

    def _run_appearance_stage(
        self,
        *,
        stage_name: str,
        freeze_stage_name: str,
        include_patch_supervision: bool,
        allow_pruning: bool,
    ) -> dict[str, Any]:
        """运行一段 appearance-first 优化循环.

        这里把 Stage 3A 和 Stage 3SR 共有的主体 loop 抽到一起.
        两者的主要差别只剩:
        - 是否启用 patch supervision
        - 是否允许 pruning
        """

        self.current_stage = stage_name
        self.gaussians.freeze_for_stage(freeze_stage_name)
        optimizer = self.gaussians.build_optimizer(freeze_stage_name, self.hparams)

        final_metrics: dict[str, Any] = {}
        for iter_idx in range(self.hparams.iters_stage2a):
            render_output = self.render_current_scene()
            pred_rgb = render_output["images_pred"]
            render_meta = self._extract_render_meta(render_output)
            self.latest_render_meta = render_meta
            residual_map = self.weight_builder.build_residual_map(pred_rgb, self.scene.gt_images)
            self.prev_weight_map = self.weight_builder.build_weight_map(residual_map, self.prev_weight_map)

            loss_rgb = compute_weighted_rgb_loss(pred_rgb, self.scene.gt_images, self.prev_weight_map)
            loss_scale = compute_scale_tail_loss(self.gaussians.scales, self.hparams.scale_tail_threshold)
            loss_opacity = compute_opacity_sparse_loss(self.gaussians.opacity)
            if include_patch_supervision:
                loss_patch_rgb, loss_patch_perceptual = self._compute_patch_losses(residual_map, loss_rgb)
                loss_sampling_smooth = compute_sampling_smooth_loss(
                    scales=self.gaussians.scales,
                    fidelity_score=self.gaussian_fidelity_score,
                    render_meta=render_meta,
                    radius_threshold=self.hparams.sampling_radius_threshold,
                )
            else:
                loss_patch_rgb = torch.zeros((), dtype=loss_rgb.dtype, device=loss_rgb.device)
                loss_patch_perceptual = torch.zeros((), dtype=loss_rgb.dtype, device=loss_rgb.device)
                loss_sampling_smooth = torch.zeros((), dtype=loss_rgb.dtype, device=loss_rgb.device)

            loss_total = loss_rgb + self.hparams.lambda_scale_tail * loss_scale + self.hparams.lambda_opacity_sparse * loss_opacity
            if include_patch_supervision:
                loss_total = (
                    loss_total
                    + self.hparams.lambda_patch_rgb * loss_patch_rgb
                    + self.hparams.lambda_patch_perceptual * loss_patch_perceptual
                    + self.hparams.lambda_sampling_smooth * loss_sampling_smooth
                )

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.gaussians.clamp_stage_constraints(freeze_stage_name, self.hparams)

            final_metrics = self._summarize_prediction(pred_rgb, residual_map=residual_map, weight_map=self.prev_weight_map)
            final_metrics.update(
                {
                    "loss_total": float(loss_total.item()),
                    "loss_rgb_weighted": float(loss_rgb.item()),
                    "loss_scale_tail": float(loss_scale.item()),
                    "loss_opacity_sparse": float(loss_opacity.item()),
                }
            )
            if include_patch_supervision:
                final_metrics.update(
                    {
                        "loss_patch_rgb": float(loss_patch_rgb.item()),
                        "loss_patch_perceptual": float(loss_patch_perceptual.item()),
                        "loss_sampling_smooth": float(loss_sampling_smooth.item()),
                    }
                )

            iteration = iter_idx + 1
            if allow_pruning and self.controller.should_prune_now(iteration):
                prune_summary = self.gaussians.prune_low_opacity(
                    threshold=self.hparams.opacity_prune_threshold,
                    max_fraction=self.hparams.prune_max_fraction,
                    min_gaussians_to_keep=self.hparams.min_gaussians_to_keep,
                )
                self.diagnostics.write_prune_summary(iteration=iteration, summary=prune_summary)

                # 当前轮的 render/residual 仍来自 prune 前.
                # 但结构性统计已经变化,这里先把统计写回当轮 metrics.
                final_metrics.update(self.gaussians.summarize_gaussian_stats())
                self.diagnostics_state.setdefault("prune_history", []).append(prune_summary)
                self.diagnostics_state["last_prune"] = prune_summary

                if prune_summary["pruned_count"] > 0:
                    optimizer = self.gaussians.build_optimizer(freeze_stage_name, self.hparams)

            self._log_and_maybe_save(stage_name, final_metrics, residual_map, self.prev_weight_map, iter_idx=iter_idx)

            stage_history = self.diagnostics.stage_history.get(stage_name, [])
            if self.controller.should_stop_stage(stage_name, stage_history):
                break

        return final_metrics

    def run_stage3a_native_cleanup(
        self,
        *,
        stage_name: str = "stage3a",
        export_file_name: str = "gaussians_stage3a.ply",
    ) -> dict[str, Any]:
        """运行不含 SR patch 的 native cleanup."""

        final_metrics = self._run_appearance_stage(
            stage_name=stage_name,
            freeze_stage_name="stage3a",
            include_patch_supervision=False,
            allow_pruning=True,
        )
        self._safe_export_ply(export_file_name)
        self.diagnostics_state["stage3a_completed"] = True
        self.diagnostics_state["phase_reached"] = stage_name
        self.diagnostics_state["global_shift_detected"] = False
        return final_metrics

    def run_phase3s_build_sr_selection(self) -> dict[str, Any]:
        """基于 renderer meta 构造第一版 fidelity 与 selection 诊断."""

        self.current_stage = "phase3s"
        render_output = self.render_current_scene()
        pred_rgb = render_output["images_pred"]
        residual_map = self.weight_builder.build_residual_map(pred_rgb, self.scene.gt_images)
        render_meta = self._extract_render_meta(render_output)

        fidelity_score = self.weight_builder.compute_gaussian_fidelity_score(render_meta)
        if fidelity_score is None:
            fidelity_score = self._build_default_fidelity_score()
            self.diagnostics_state.setdefault("warnings", []).append("phase3s_missing_render_meta")

        sr_selection_map = self.weight_builder.build_sr_selection_weight(
            render_meta=render_meta,
            fidelity_score=fidelity_score,
            native_hw=self.scene.gt_images.shape[-2:],
            output_hw=self._get_reference_images().shape[-2:],
        )
        if sr_selection_map is None:
            sr_selection_map = self._build_default_sr_selection_map(residual_map)
            self.diagnostics_state.setdefault("warnings", []).append("phase3s_missing_sr_selection_meta")
        self.latest_render_meta = render_meta
        self.gaussian_fidelity_score = fidelity_score.detach()
        self.sr_selection_map = sr_selection_map.detach()

        metrics = self._summarize_prediction(pred_rgb, residual_map=residual_map, weight_map=self.prev_weight_map)
        metrics.update(self.weight_builder.summarize_fidelity_stats(fidelity_score))
        metrics["sr_selection_mean"] = float(sr_selection_map.mean().item())
        self._log_and_maybe_save("phase3s", metrics, residual_map, self.prev_weight_map, iter_idx=0)
        self._write_phase3s_artifacts("phase3s", fidelity_score, sr_selection_map)

        self.diagnostics_state["phase3s_completed"] = True
        self.diagnostics_state["phase_reached"] = "phase3s"
        return metrics

    def run_stage3sr_selective_patch(
        self,
        *,
        stage_name: str = "stage3sr",
        export_file_name: str = "gaussians_stage3sr.ply",
    ) -> dict[str, Any]:
        """运行 selective SR patch supervision 的最小闭环."""

        if not self._patch_supervision_configured():
            raise RuntimeError("Stage 3SR requires patch supervision to be enabled.")
        if not self.diagnostics_state.get("phase3s_completed", False):
            self.run_phase3s_build_sr_selection()

        final_metrics = self._run_appearance_stage(
            stage_name=stage_name,
            freeze_stage_name="stage3sr",
            include_patch_supervision=True,
            allow_pruning=False,
        )
        self._safe_export_ply(export_file_name)
        self.diagnostics_state["stage3sr_completed"] = True
        self.diagnostics_state["phase_reached"] = stage_name
        self.diagnostics_state["global_shift_detected"] = False
        self._update_stage2b_diagnostics(final_metrics)
        return final_metrics

    def bootstrap_stage2b_from_current_gaussians(self) -> dict[str, Any]:
        """把当前输入高斯视为“已完成 Stage 2A”的 warm start.

        这个入口用于显式支持:
        - 输入已经是 `gaussians_stage2a.ply`
        - 本轮只想继续做 `Stage 2B`

        这里不会再做新的 Stage 2A optimizer step.
        只会:
        1. 重新渲染当前高斯
        2. 更新权重图与统计
        3. 生成是否进入 `Stage 2B` 的 diagnostics
        """

        self.current_stage = "stage2a"
        render_output = self.render_current_scene()
        pred_rgb = render_output["images_pred"]
        residual_map = self.weight_builder.build_residual_map(pred_rgb, self.scene.gt_images)
        self.prev_weight_map = self.weight_builder.build_weight_map(residual_map, self.prev_weight_map)

        loss_rgb = compute_weighted_rgb_loss(pred_rgb, self.scene.gt_images, self.prev_weight_map)
        loss_scale = compute_scale_tail_loss(self.gaussians.scales, self.hparams.scale_tail_threshold)
        loss_opacity = compute_opacity_sparse_loss(self.gaussians.opacity)
        loss_patch_rgb, loss_patch_perceptual = self._compute_patch_losses(residual_map, loss_rgb)
        metrics = self._summarize_prediction(pred_rgb, residual_map=residual_map, weight_map=self.prev_weight_map)
        metrics.update(
            {
                "loss_total": float(loss_rgb.item()),
                "loss_rgb_weighted": float(loss_rgb.item()),
                "loss_scale_tail": float(loss_scale.item()),
                "loss_opacity_sparse": float(loss_opacity.item()),
                "loss_patch_rgb": float(loss_patch_rgb.item()),
                "loss_patch_perceptual": float(loss_patch_perceptual.item()),
            }
        )

        self.diagnostics_state["phase_reached"] = "stage2a"
        self.diagnostics_state["stage3a_completed"] = True
        self.diagnostics_state["warm_start_stage2b"] = True
        self.diagnostics_state["stage2a_bootstrap"] = metrics
        self._update_stage2b_diagnostics(metrics)
        return metrics

    def _safe_export_ply(self, file_name: str) -> None:
        """尽量导出高斯,导出失败时不打断主流程."""

        try:
            self.gaussians.export_ply(self.diagnostics.gaussian_dir / file_name)
        except Exception as exc:  # noqa: BLE001
            self.diagnostics_state.setdefault("warnings", []).append(f"export_ply_failed:{exc}")

    def _export_rgb_artifacts(
        self,
        name: str,
        rgb_tensor: torch.Tensor,
        *,
        save_snapshot: bool = True,
    ) -> dict[str, str]:
        """导出一组 RGB 可视化产物.

        这里统一导出:
        - `videos/<name>.mp4`
        - `renders_before_after/<name>_frame_0000.png`
        """

        artifacts: dict[str, str] = {}

        video_path = self.diagnostics.save_render_video(name, rgb_tensor)
        artifacts[f"{name}_video"] = str(video_path)

        if save_snapshot:
            snapshot_path = self.diagnostics.save_render_snapshot(name, rgb_tensor, frame_id=0)
            artifacts[f"{name}_frame"] = str(snapshot_path)

        self.visual_artifacts.update(artifacts)
        return artifacts

    def _export_baseline_visuals(self, pred_rgb: torch.Tensor) -> None:
        """导出 baseline 和参考 GT 视频."""

        # baseline 是“优化前”的核心对照.
        # 在 Phase 0 就立刻落盘,后面无论 stop 在哪一阶段都能拿到 before 结果.
        self._export_rgb_artifacts("baseline_render", pred_rgb)
        self._export_rgb_artifacts("gt_reference", self.scene.gt_images)

    def _log_and_maybe_save(
        self,
        stage_name: str,
        metrics: dict[str, Any],
        residual_map: torch.Tensor,
        weight_map: torch.Tensor | None = None,
        iter_idx: int = 0,
        save_state_now: bool = False,
    ) -> None:
        """统一处理 metrics、图像和状态保存."""

        self.diagnostics.log_stage_metrics(stage_name, metrics)
        self.diagnostics.save_residual_map(stage_name, iter_idx, residual_map)
        if weight_map is not None:
            self.diagnostics.save_weight_map(stage_name, iter_idx, weight_map)
        if save_state_now:
            save_state(
                self.diagnostics.state_dir,
                stage_name=stage_name,
                iter_idx=iter_idx,
                gaussians=self.gaussians,
                diagnostics_state=self.diagnostics_state,
                pose_delta=self.pose_delta,
            )

    def run_phase0(self) -> dict[str, Any]:
        """运行 baseline 渲染与诊断."""

        self.current_stage = "phase0"
        render_output = self.render_current_scene()
        pred_rgb = render_output["images_pred"]
        residual_map = self.weight_builder.build_residual_map(pred_rgb, self.scene.gt_images)
        baseline = self._summarize_prediction(pred_rgb, residual_map=residual_map)
        self._export_baseline_visuals(pred_rgb)

        metrics = dict(baseline)
        metrics["loss_total"] = float(residual_map.mean().item())
        self._log_and_maybe_save("phase0", metrics, residual_map, iter_idx=0, save_state_now=True)

        self.diagnostics_state["baseline"] = baseline
        self.diagnostics_state["phase_reached"] = "phase0"
        return baseline

    def run_phase0_only(self) -> dict[str, Any]:
        """只运行 Phase 0 并直接结束."""

        baseline = self.run_phase0()
        return self.export_final_outputs(baseline, override_stop_reason="dry_run")

    def run_phase1_prepare_weights(self) -> dict[str, Any]:
        """构造 Stage 2A 起始权重图."""

        self.current_stage = "phase1"
        render_output = self.render_current_scene()
        pred_rgb = render_output["images_pred"]
        residual_map = self.weight_builder.build_residual_map(pred_rgb, self.scene.gt_images)
        self.prev_weight_map = self.weight_builder.build_weight_map(residual_map, self.prev_weight_map)

        metrics = self._summarize_prediction(pred_rgb, residual_map=residual_map, weight_map=self.prev_weight_map)
        metrics["loss_total"] = float(residual_map.mean().item())
        self._log_and_maybe_save("phase1", metrics, residual_map, self.prev_weight_map, iter_idx=0)

        self.diagnostics_state["phase1"] = metrics
        self.diagnostics_state["phase_reached"] = "phase1"
        return metrics

    def run_stage2a(self) -> dict[str, Any]:
        """兼容旧入口的 Stage 2A.

        现在它内部会按新的边界拆成:
        1. native cleanup
        2. Phase 3S
        3. selective SR patch
        如果当前没有启用 patch supervision,则只执行 native cleanup.
        """

        resolved_mode = self._resolve_stage2a_mode()
        self.diagnostics_state["stage2a_mode_resolved"] = resolved_mode
        self.diagnostics_state["stage3sr_enabled"] = resolved_mode == "enhanced"
        final_metrics = self.run_stage3a_native_cleanup(
            stage_name="stage2a",
            export_file_name="gaussians_stage2a.ply",
        )
        if resolved_mode == "legacy":
            if self.run_config.stage2a_mode == "legacy" and self._patch_supervision_configured():
                self.diagnostics_state.setdefault("warnings", []).append("stage2a_mode_legacy_skipped_patch_supervision")
            self._update_stage2b_diagnostics(final_metrics)
            return final_metrics

        self.run_phase3s_build_sr_selection()
        return self.run_stage3sr_selective_patch()

    def run_stage2b(self) -> dict[str, Any]:
        """运行 limited geometry refinement."""

        self.current_stage = "stage2b"
        self.gaussians.freeze_for_stage("stage2b")
        optimizer = self.gaussians.build_optimizer("stage2b", self.hparams)

        final_metrics: dict[str, Any] = {}
        for iter_idx in range(self.hparams.iters_stage2b):
            render_output = self.render_current_scene()
            pred_rgb = render_output["images_pred"]
            residual_map = self.weight_builder.build_residual_map(pred_rgb, self.scene.gt_images)
            self.prev_weight_map = self.weight_builder.build_weight_map(residual_map, self.prev_weight_map)

            loss_rgb = compute_weighted_rgb_loss(pred_rgb, self.scene.gt_images, self.prev_weight_map)
            loss_scale = compute_scale_tail_loss(self.gaussians.scales, self.hparams.scale_tail_threshold)
            loss_opacity = compute_opacity_sparse_loss(self.gaussians.opacity)
            loss_patch_rgb, loss_patch_perceptual = self._compute_patch_losses(residual_map, loss_rgb)
            loss_means_anchor = compute_means_anchor_loss(self.gaussians.means, self.gaussians.initial_means)
            loss_rotation_reg = compute_rotation_regularization_loss(
                self.gaussians.rotations,
                self.gaussians.initial_rotations,
            )
            loss_total = loss_rgb + self.hparams.lambda_scale_tail * loss_scale + self.hparams.lambda_opacity_sparse * loss_opacity
            loss_total = (
                loss_total
                + self.hparams.lambda_patch_rgb * loss_patch_rgb
                + self.hparams.lambda_patch_perceptual * loss_patch_perceptual
                + self.hparams.lambda_means_anchor * loss_means_anchor
                + self.hparams.lambda_rotation_reg * loss_rotation_reg
            )

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.gaussians.clamp_stage_constraints("stage2b", self.hparams)

            final_metrics = self._summarize_prediction(pred_rgb, residual_map=residual_map, weight_map=self.prev_weight_map)
            final_metrics.update(
                {
                    "loss_total": float(loss_total.item()),
                    "loss_rgb_weighted": float(loss_rgb.item()),
                    "loss_scale_tail": float(loss_scale.item()),
                    "loss_opacity_sparse": float(loss_opacity.item()),
                    "loss_patch_rgb": float(loss_patch_rgb.item()),
                    "loss_patch_perceptual": float(loss_patch_perceptual.item()),
                    "loss_means_anchor": float(loss_means_anchor.item()),
                    "loss_rotation_reg": float(loss_rotation_reg.item()),
                }
            )
            self._log_and_maybe_save("stage2b", final_metrics, residual_map, self.prev_weight_map, iter_idx=iter_idx)

        self._safe_export_ply("gaussians_stage2b.ply")
        self.diagnostics_state["phase_reached"] = "stage2b"
        self.diagnostics_state["global_shift_detected"] = final_metrics.get("residual_mean", 0.0) > 0.08
        self.diagnostics_state["local_overlap_persistent"] = final_metrics.get("residual_mean", 0.0) > 0.03
        return final_metrics

    def run_phase3_pose_only(self) -> dict[str, Any]:
        """运行 tiny pose-only diagnostic."""

        self.current_stage = "phase3"
        num_views = self.scene.cam_view.shape[1]
        self.pose_delta = torch.zeros(num_views, 6, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([self.pose_delta], lr=self.hparams.lr_pose)

        metrics: dict[str, Any] = {}
        for iter_idx in range(self.hparams.iters_pose):
            loss_pose_l2, loss_pose_smooth = compute_pose_regularization(self.pose_delta)
            loss_total = loss_pose_l2 + loss_pose_smooth
            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            metrics = {
                "loss_total": float(loss_total.item()),
                "loss_pose_l2": float(loss_pose_l2.item()),
                "loss_pose_smooth": float(loss_pose_smooth.item()),
            }
            self.diagnostics.log_stage_metrics("phase3", metrics)

        pose_summary = {
            "pose_delta_norm": float(self.pose_delta.detach().norm().item()),
            "num_views": int(num_views),
        }
        self.diagnostics.save_pose_summary(pose_summary)
        self.diagnostics_state["phase_reached"] = "phase3"
        self.diagnostics_state["used_pose_refinement"] = True
        self.diagnostics_state["pose_diagnostic_ran"] = True
        return metrics

    def run_phase4_joint(self) -> dict[str, Any]:
        """运行 joint fallback."""

        self.current_stage = "phase4"
        self.gaussians.freeze_for_stage("phase4")
        optimizer = self.gaussians.build_optimizer("phase4", self.hparams)

        final_metrics: dict[str, Any] = {}
        for iter_idx in range(self.hparams.iters_joint):
            render_output = self.render_current_scene()
            pred_rgb = render_output["images_pred"]
            residual_map = self.weight_builder.build_residual_map(pred_rgb, self.scene.gt_images)
            self.prev_weight_map = self.weight_builder.build_weight_map(residual_map, self.prev_weight_map)

            loss_rgb = compute_weighted_rgb_loss(pred_rgb, self.scene.gt_images, self.prev_weight_map)
            loss_scale = compute_scale_tail_loss(self.gaussians.scales, self.hparams.scale_tail_threshold)
            loss_opacity = compute_opacity_sparse_loss(self.gaussians.opacity)
            loss_total = loss_rgb + self.hparams.lambda_scale_tail * loss_scale + self.hparams.lambda_opacity_sparse * loss_opacity

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.gaussians.clamp_stage_constraints("phase4", self.hparams)

            final_metrics = self._summarize_prediction(pred_rgb, residual_map=residual_map, weight_map=self.prev_weight_map)
            final_metrics.update({"loss_total": float(loss_total.item())})
            self._log_and_maybe_save("phase4", final_metrics, residual_map, self.prev_weight_map, iter_idx=iter_idx)

        self._safe_export_ply("gaussians_refined.ply")
        self.diagnostics_state["phase_reached"] = "phase4"
        self.diagnostics_state["used_joint_fallback"] = True
        return final_metrics

    def restore_latest_state(self) -> bool:
        """从最近一次保存状态恢复."""

        payload = load_latest_state(self.diagnostics.state_dir)
        if payload is None:
            return False

        state_gaussians = payload["gaussians"].to(self.device)

        # `export_ply()` 会按 opacity 过滤一部分低置信高斯.
        # 因此 resume workflow 下,磁盘上的 `.ply` 数量可能小于 `state/latest.pt`
        # 里保存的全量 tensor 数量. 这时不能再强行要求 shape 完全一致,
        # 而应该直接以 state 中的全量高斯重建 adapter.
        expected_shape = (self.gaussians.means.shape[0], 14)
        current_shape = tuple(state_gaussians.shape[-2:])
        if current_shape != expected_shape:
            self.gaussians = GaussianAdapter.from_tensor(
                state_gaussians,
                scale_tail_threshold=self.gaussians.scale_tail_threshold,
                opacity_low_threshold=self.gaussians.opacity_low_threshold,
            ).to(self.device)
        else:
            self.gaussians.copy_from_tensor(state_gaussians)

        self.diagnostics_state.update(payload.get("diagnostics_state", {}))
        pose_delta = payload.get("pose_delta")
        if isinstance(pose_delta, torch.Tensor):
            self.pose_delta = pose_delta.to(self.device)
        return True

    def _build_final_summary(
        self,
        final_metrics: dict[str, Any],
        override_stop_reason: str | None = None,
    ) -> dict[str, Any]:
        """构建最终 diagnostics 摘要."""

        baseline = self.diagnostics_state.get("baseline", {})
        return {
            "scene_id": self.scene.scene_index,
            "view_id": self.scene.view_id,
            "start_stage": self.run_config.start_stage,
            "phase_reached": self.diagnostics_state.get("phase_reached", self.current_stage),
            "stopped_reason": override_stop_reason or self.controller.summarize_stop_reason(self.diagnostics_state),
            "used_pose_refinement": self.diagnostics_state.get("used_pose_refinement", False),
            "used_joint_fallback": self.diagnostics_state.get("used_joint_fallback", False),
            "warm_start_stage2b": self.diagnostics_state.get("warm_start_stage2b", False),
            "baseline": baseline,
            "final": final_metrics,
            "artifacts": dict(self.visual_artifacts),
            "deltas": {
                "psnr_gain": float(final_metrics.get("psnr", 0.0) - baseline.get("psnr", 0.0)),
                "sharpness_gain": float(final_metrics.get("sharpness", 0.0) - baseline.get("sharpness", 0.0)),
                "scale_tail_drop": float(baseline.get("scale_tail_ratio", 0.0) - final_metrics.get("scale_tail_ratio", 0.0)),
            },
        }

    def export_final_outputs(
        self,
        final_metrics: dict[str, Any],
        override_stop_reason: str | None = None,
    ) -> dict[str, Any]:
        """保存最终高斯和 diagnostics 摘要."""

        # 结束前再次渲染当前高斯.
        # 这样无论 stop 在哪一阶段,都能得到统一命名的 after 视频.
        final_render_output = self.render_current_scene()
        final_pred_rgb = final_render_output["images_pred"]
        self._export_rgb_artifacts("final_render", final_pred_rgb)

        save_state(
            self.diagnostics.state_dir,
            stage_name=self.current_stage,
            iter_idx=0,
            gaussians=self.gaussians,
            diagnostics_state=self.diagnostics_state,
            pose_delta=self.pose_delta,
        )
        summary = self._build_final_summary(final_metrics, override_stop_reason=override_stop_reason)
        self.diagnostics.finalize(summary)
        return summary

    def run(self) -> dict[str, Any]:
        """运行完整 refinement 流程."""

        explicit_stage2b_start = self.run_config.start_stage == "stage2b"
        if explicit_stage2b_start and not self.run_config.enable_stage2b:
            raise RuntimeError("start_stage=stage2b requires enable_stage2b=True.")

        final_metrics = self.run_phase0()
        if self.run_config.stop_after == "phase0":
            return self.export_final_outputs(final_metrics)

        final_metrics = self.run_phase1_prepare_weights()
        if self.run_config.stop_after == "phase1":
            return self.export_final_outputs(final_metrics)

        if explicit_stage2b_start:
            final_metrics = self.bootstrap_stage2b_from_current_gaussians()
        else:
            final_metrics = self.run_stage2a()

        if self.run_config.stop_after == "stage2a":
            return self.export_final_outputs(final_metrics)

        # 显式 `start_stage=stage2b` 表示用户已经决定继续几何阶段.
        # 这种情况下不应再被自动 gate 拦住.
        if explicit_stage2b_start or self.controller.should_enter_stage2b(self.diagnostics_state):
            final_metrics = self.run_stage2b()
            if self.run_config.stop_after == "stage2b":
                return self.export_final_outputs(final_metrics)

        if self.controller.should_enter_pose_diagnostic(self.diagnostics_state):
            self.run_phase3_pose_only()
            if self.run_config.stop_after == "phase3":
                return self.export_final_outputs(final_metrics)

        if self.controller.should_enter_joint_fallback(self.diagnostics_state):
            final_metrics = self.run_phase4_joint()
            if self.run_config.stop_after == "phase4":
                return self.export_final_outputs(final_metrics)

        return self.export_final_outputs(final_metrics)
