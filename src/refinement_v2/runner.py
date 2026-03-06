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
    compute_opacity_sparse_loss,
    compute_patch_perceptual_loss,
    compute_pose_regularization,
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
        self.renderer_cache: dict[tuple[int, int], GaussianSceneRenderer] = {}
        self.diagnostics_state: dict[str, Any] = {
            "phase_reached": "init",
            "used_pose_refinement": False,
            "used_joint_fallback": False,
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

    def _patch_supervision_enabled(self) -> bool:
        """判断当前是否启用 patch supervision."""

        return self.hparams.patch_size > 0 and (
            self.hparams.lambda_patch_rgb > 0.0 or self.hparams.lambda_patch_perceptual > 0.0
        )

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
        reference_images = self.scene.reference_images if isinstance(self.scene.reference_images, torch.Tensor) else self.scene.gt_images
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

        reference_images = self.scene.reference_images if isinstance(self.scene.reference_images, torch.Tensor) else self.scene.gt_images
        patches: list[torch.Tensor] = []
        for batch_index in range(reference_images.shape[0]):
            view_patches: list[torch.Tensor] = []
            for view_index in range(reference_images.shape[1]):
                top, left, patch_height, patch_width = patch_windows_ref[batch_index, view_index].tolist()
                view_patches.append(
                    reference_images[batch_index, view_index, :, top : top + patch_height, left : left + patch_width]
                )
            patches.append(torch.stack(view_patches, dim=0))
        return torch.stack(patches, dim=0)

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
        """运行 appearance-first 的高斯优化."""

        self.current_stage = "stage2a"
        self.gaussians.freeze_for_stage("stage2a")
        optimizer = self.gaussians.build_optimizer("stage2a", self.hparams)

        final_metrics: dict[str, Any] = {}
        for iter_idx in range(self.hparams.iters_stage2a):
            render_output = self.render_current_scene()
            pred_rgb = render_output["images_pred"]
            residual_map = self.weight_builder.build_residual_map(pred_rgb, self.scene.gt_images)
            self.prev_weight_map = self.weight_builder.build_weight_map(residual_map, self.prev_weight_map)

            loss_rgb = compute_weighted_rgb_loss(pred_rgb, self.scene.gt_images, self.prev_weight_map)
            loss_scale = compute_scale_tail_loss(self.gaussians.scales, self.hparams.scale_tail_threshold)
            loss_opacity = compute_opacity_sparse_loss(self.gaussians.opacity)
            loss_patch_rgb = torch.zeros((), dtype=loss_rgb.dtype, device=loss_rgb.device)
            loss_patch_perceptual = torch.zeros((), dtype=loss_rgb.dtype, device=loss_rgb.device)
            if self._patch_supervision_enabled():
                patch_windows_native = self.sample_patch_windows(residual_map)
                pred_patch, reference_patch, _ = self.render_patch_prediction(patch_windows_native)
                patch_weights = torch.ones(
                    pred_patch.shape[0],
                    pred_patch.shape[1],
                    1,
                    pred_patch.shape[-2],
                    pred_patch.shape[-1],
                    dtype=pred_patch.dtype,
                    device=pred_patch.device,
                )
                loss_patch_rgb = compute_weighted_rgb_loss(pred_patch, reference_patch, patch_weights)
                loss_patch_perceptual = compute_patch_perceptual_loss(pred_patch, reference_patch)
            loss_total = loss_rgb + self.hparams.lambda_scale_tail * loss_scale + self.hparams.lambda_opacity_sparse * loss_opacity
            loss_total = (
                loss_total
                + self.hparams.lambda_patch_rgb * loss_patch_rgb
                + self.hparams.lambda_patch_perceptual * loss_patch_perceptual
            )

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.gaussians.clamp_stage_constraints("stage2a", self.hparams)

            final_metrics = self._summarize_prediction(pred_rgb, residual_map=residual_map, weight_map=self.prev_weight_map)
            final_metrics.update(
                {
                    "loss_total": float(loss_total.item()),
                    "loss_rgb_weighted": float(loss_rgb.item()),
                    "loss_scale_tail": float(loss_scale.item()),
                    "loss_opacity_sparse": float(loss_opacity.item()),
                }
            )
            if self._patch_supervision_enabled():
                final_metrics.update(
                    {
                        "loss_patch_rgb": float(loss_patch_rgb.item()),
                        "loss_patch_perceptual": float(loss_patch_perceptual.item()),
                    }
                )

            # pruning 放在 step 之后.
            # 这样本轮梯度先完整落到参数上,再做结构裁剪和 optimizer 重建.
            iteration = iter_idx + 1
            if self.controller.should_prune_now(iteration):
                prune_summary = self.gaussians.prune_low_opacity(
                    threshold=self.hparams.opacity_prune_threshold,
                    max_fraction=self.hparams.prune_max_fraction,
                    min_gaussians_to_keep=self.hparams.min_gaussians_to_keep,
                )
                self.diagnostics.write_prune_summary(iteration=iteration, summary=prune_summary)

                # 当前轮的 render/residual 仍来自 prune 前.
                # 但高斯统计已经发生变化,这里先把结构性统计更新到当轮 metrics.
                final_metrics.update(self.gaussians.summarize_gaussian_stats())
                self.diagnostics_state.setdefault("prune_history", []).append(prune_summary)
                self.diagnostics_state["last_prune"] = prune_summary

                if prune_summary["pruned_count"] > 0:
                    optimizer = self.gaussians.build_optimizer("stage2a", self.hparams)

            self._log_and_maybe_save("stage2a", final_metrics, residual_map, self.prev_weight_map, iter_idx=iter_idx)

            stage_history = self.diagnostics.stage_history.get("stage2a", [])
            if self.controller.should_stop_stage("stage2a", stage_history):
                break

        self._safe_export_ply("gaussians_stage2a.ply")
        self.diagnostics_state["phase_reached"] = "stage2a"
        self.diagnostics_state["need_geometry"] = final_metrics.get("residual_mean", 0.0) > 0.05
        self.diagnostics_state["global_shift_detected"] = False
        self.diagnostics_state["local_overlap_persistent"] = final_metrics.get("residual_mean", 0.0) > 0.05
        return final_metrics

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
            loss_total = loss_rgb + self.hparams.lambda_scale_tail * loss_scale + self.hparams.lambda_opacity_sparse * loss_opacity

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.gaussians.clamp_stage_constraints("stage2b", self.hparams)

            final_metrics = self._summarize_prediction(pred_rgb, residual_map=residual_map, weight_map=self.prev_weight_map)
            final_metrics.update({"loss_total": float(loss_total.item())})
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

        self.gaussians.copy_from_tensor(payload["gaussians"].to(self.device))
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
            "phase_reached": self.diagnostics_state.get("phase_reached", self.current_stage),
            "stopped_reason": override_stop_reason or self.controller.summarize_stop_reason(self.diagnostics_state),
            "used_pose_refinement": self.diagnostics_state.get("used_pose_refinement", False),
            "used_joint_fallback": self.diagnostics_state.get("used_joint_fallback", False),
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

        final_metrics = self.run_phase0()
        if self.run_config.stop_after == "phase0":
            return self.export_final_outputs(final_metrics)

        final_metrics = self.run_phase1_prepare_weights()
        if self.run_config.stop_after == "phase1":
            return self.export_final_outputs(final_metrics)

        final_metrics = self.run_stage2a()
        if self.run_config.stop_after == "stage2a":
            return self.export_final_outputs(final_metrics)

        if self.controller.should_enter_stage2b(self.diagnostics_state):
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
