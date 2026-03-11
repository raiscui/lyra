"""`refinement_v2` 的主控执行器."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from .config import RefinementRunConfig, StageHyperParams
from .data_loader import SceneBundle
from .diagnostics import DiagnosticsWriter
from .gaussian_adapter import GaussianAdapter
from .losses import (
    build_depth_anchor_valid_mask,
    compute_depth_anchor_loss,
    compute_opacity_sparse_loss,
    compute_patch_perceptual_loss,
    compute_pose_regularization,
    compute_sampling_smooth_loss,
    compute_scale_tail_loss,
    compute_stage3b_losses,
    compute_weighted_rgb_loss,
    downsample_rgb_tensor,
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


@dataclass
class DepthAnchorReference:
    """缓存一份不可变的 baseline depth anchor."""

    depth: torch.Tensor
    valid_mask: torch.Tensor
    valid_ratio: float
    source: str
    alpha: torch.Tensor | None = None


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
        self.render_devices = self._resolve_render_devices(self.run_config.render_devices)

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
            fidelity_ratio_threshold=hparams.fidelity_ratio_threshold,
            fidelity_sigmoid_k=hparams.fidelity_sigmoid_k,
            fidelity_min_views=hparams.fidelity_min_views,
            fidelity_opacity_threshold=hparams.fidelity_opacity_threshold,
        )

        self.current_stage = "init"
        self.prev_weight_map: torch.Tensor | None = None
        self.pose_delta: torch.Tensor | None = None
        self.latest_render_meta: dict[str, Any] | None = None
        self.gaussian_fidelity_score: torch.Tensor | None = None
        self.gaussian_fidelity_diagnostics: dict[str, torch.Tensor] | None = None
        self.sr_selection_map: torch.Tensor | None = None
        self.last_sr_patch_sets_used = 0
        self.depth_anchor_reference: DepthAnchorReference | None = None
        self.depth_anchor_capture_attempted = False
        self.renderer_cache: dict[tuple[int, int], GaussianSceneRenderer] = {}
        self.scene_shard_cache: dict[tuple[Any, ...], SceneBundle] = {}
        self.diagnostics_state: dict[str, Any] = {
            "phase_reached": "init",
            "used_pose_refinement": False,
            "used_joint_fallback": False,
            "stage2a_mode_requested": self.run_config.stage2a_mode,
            "stage3sr_enabled": self._stage2a_should_run_stage3sr(),
            "stage3b_enabled": self.run_config.enable_stage3b,
            "render_devices": [str(device) for device in self.render_devices],
            "depth_anchor_enabled": bool(self.hparams.enable_depth_anchor and self.hparams.lambda_depth_anchor > 0.0),
            "depth_anchor_weight": self.hparams.lambda_depth_anchor,
            "depth_anchor_source": self.hparams.depth_anchor_source,
            "depth_anchor_reference_ready": False,
            "depth_anchor_reference_valid_ratio": 0.0,
            "depth_anchor_reference_skip_reason": None,
            "depth_anchor_last_skip_reason": None,
            "depth_anchor_last_valid_ratio": 0.0,
            "depth_anchor_last_loss": 0.0,
        }
        self.visual_artifacts: dict[str, str] = {}

    def _resolve_device(self, device_name: str) -> torch.device:
        """解析并回退设备选择."""

        if device_name.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)

    def _resolve_render_devices(self, device_names: list[str] | None) -> list[torch.device]:
        """解析渲染设备列表.

        当前优化器与高斯参数仍固定在 `self.device`.
        `render_devices` 只负责把 view 维渲染负载分摊到多张卡.
        """

        if not device_names:
            return [self.device]

        resolved_devices: list[torch.device] = []
        for device_name in device_names:
            resolved_device = self._resolve_device(device_name)
            if resolved_device.type == "cuda":
                device_index = resolved_device.index if resolved_device.index is not None else 0
                if device_index >= torch.cuda.device_count():
                    raise ValueError(
                        f"render device `{device_name}` is out of range for {torch.cuda.device_count()} visible CUDA devices."
                    )
            resolved_devices.append(resolved_device)

        # 主设备始终放在第一位.
        # 这样 loss 汇总和 optimizer step 不需要改现有语义.
        if str(self.device) not in {str(device) for device in resolved_devices}:
            resolved_devices.insert(0, self.device)
        return resolved_devices

    def _move_scene_to_device(self, scene: SceneBundle, device: torch.device) -> SceneBundle:
        """把场景 tensor 迁移到目标设备."""

        return SceneBundle(
            gt_images=scene.gt_images.to(device),
            cam_view=scene.cam_view.to(device),
            intrinsics=scene.intrinsics.to(device),
            frame_indices=scene.frame_indices,
            scene_index=scene.scene_index,
            view_id=scene.view_id,
            view_ids=scene.view_ids,
            target_index=scene.target_index.to(device) if isinstance(scene.target_index, torch.Tensor) else scene.target_index,
            file_name=scene.file_name,
            reference_images=scene.reference_images.to(device) if isinstance(scene.reference_images, torch.Tensor) else scene.reference_images,
            intrinsics_ref=scene.intrinsics_ref.to(device) if isinstance(scene.intrinsics_ref, torch.Tensor) else scene.intrinsics_ref,
            native_hw=scene.native_hw,
            reference_hw=scene.reference_hw,
            reference_mode=scene.reference_mode,
            sr_scale=scene.sr_scale,
        )

    def _slice_scene_view_range(self, scene: SceneBundle, start_index: int, end_index: int) -> SceneBundle:
        """按 view 维切出一个 scene shard.

        full-view joint optimization 的 batch 仍是 `B=1`.
        多卡这里只沿着 `V` 维切,保持单个高斯场景共享.
        """

        target_index = scene.target_index[:, start_index:end_index] if isinstance(scene.target_index, torch.Tensor) else scene.target_index
        reference_images = (
            scene.reference_images[:, start_index:end_index] if isinstance(scene.reference_images, torch.Tensor) else scene.reference_images
        )
        intrinsics_ref = (
            scene.intrinsics_ref[:, start_index:end_index] if isinstance(scene.intrinsics_ref, torch.Tensor) else scene.intrinsics_ref
        )
        return SceneBundle(
            gt_images=scene.gt_images[:, start_index:end_index],
            cam_view=scene.cam_view[:, start_index:end_index],
            intrinsics=scene.intrinsics[:, start_index:end_index],
            frame_indices=scene.frame_indices[start_index:end_index],
            scene_index=scene.scene_index,
            view_id=scene.view_id,
            # `view_ids` 是 scene-level metadata,不是 flatten 后逐 observation 的列表.
            # 因此 shard 里继续保留原始多视角集合,避免 diagnostics 语义漂移.
            view_ids=scene.view_ids,
            target_index=target_index,
            file_name=scene.file_name,
            reference_images=reference_images,
            intrinsics_ref=intrinsics_ref,
            native_hw=scene.native_hw,
            reference_hw=scene.reference_hw,
            reference_mode=scene.reference_mode,
            sr_scale=scene.sr_scale,
        )

    def _get_scene_shard(self, scene: SceneBundle, device: torch.device, start_index: int, end_index: int) -> SceneBundle:
        """获取指定 view range 的缓存 shard."""

        height, width = scene.gt_images.shape[-2:]
        cache_key = (
            str(device),
            start_index,
            end_index,
            height,
            width,
            id(scene.gt_images),
            id(scene.cam_view),
            id(scene.intrinsics),
            id(scene.reference_images) if isinstance(scene.reference_images, torch.Tensor) else -1,
            id(scene.intrinsics_ref) if isinstance(scene.intrinsics_ref, torch.Tensor) else -1,
        )
        if cache_key not in self.scene_shard_cache:
            shard = self._slice_scene_view_range(scene, start_index, end_index)
            self.scene_shard_cache[cache_key] = self._move_scene_to_device(shard, device)
        return self.scene_shard_cache[cache_key]

    def _get_scene_device(self, scene: SceneBundle) -> torch.device:
        """读取 scene 当前所在设备."""

        return scene.gt_images.device

    def _get_gaussians_for_device(
        self,
        device: torch.device,
        gaussians_primary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """按目标设备拿到可参与 autograd 的高斯张量副本."""

        active_gaussians = self.gaussians.to_tensor() if gaussians_primary is None else gaussians_primary
        if active_gaussians.device == device:
            return active_gaussians
        return active_gaussians.to(device)

    def _build_render_shards(self, num_views: int) -> list[tuple[torch.device, int, int]]:
        """把 view 维均匀切到可用渲染设备上."""

        active_devices = self.render_devices[: min(len(self.render_devices), num_views)]
        num_active_devices = len(active_devices)
        if num_active_devices <= 1:
            return [(active_devices[0], 0, num_views)]

        # 真实 CUDA renderer 在 `render_meta` 合并阶段会为 shard 内全部 view
        # 额外保留一份 dense meta.
        # 对 full-view sub4 这类大 observation 任务来说, 即便已经做了双卡分片,
        # “每张卡一次吃 3 个 view” 仍然可能在 renderer 内部 OOM.
        # 因此多 CUDA 设备场景下进一步收成“单 view shard + 设备轮转”,
        # 用更多前向次数换稳定落地.
        if num_active_devices > 1 and any(device.type == "cuda" for device in active_devices):
            return [
                (active_devices[view_index % num_active_devices], view_index, view_index + 1)
                for view_index in range(num_views)
            ]

        shard_specs: list[tuple[torch.device, int, int]] = []
        start_index = 0
        base_views_per_device = num_views // num_active_devices
        extra_views = num_views % num_active_devices
        for device_index, device in enumerate(active_devices):
            shard_length = base_views_per_device + (1 if device_index < extra_views else 0)
            end_index = start_index + shard_length
            if shard_length > 0:
                shard_specs.append((device, start_index, end_index))
            start_index = end_index
        return shard_specs

    def _move_render_meta_to_device(
        self,
        render_meta: dict[str, Any] | None,
        device: torch.device,
        *,
        detach_tensors: bool = False,
    ) -> dict[str, Any] | None:
        """把 renderer meta 迁回主设备."""

        if not isinstance(render_meta, dict):
            return None

        moved_meta: dict[str, Any] = {}
        for key, value in render_meta.items():
            if isinstance(value, torch.Tensor):
                moved_tensor = value.detach() if detach_tensors else value
                moved_meta[key] = moved_tensor.to(device)
            else:
                moved_meta[key] = value
        return moved_meta

    def _merge_render_meta_shards(self, render_meta_shards: list[dict[str, Any] | None]) -> dict[str, Any] | None:
        """把多卡 view shard 的 meta 沿 view 维重新拼回去."""

        valid_meta_shards = [render_meta for render_meta in render_meta_shards if isinstance(render_meta, dict)]
        if not valid_meta_shards:
            return None

        merged_meta: dict[str, Any] = {}
        all_keys = set().union(*(render_meta.keys() for render_meta in valid_meta_shards))
        for key in all_keys:
            values = [render_meta.get(key) for render_meta in valid_meta_shards]
            tensor_values = [value for value in values if isinstance(value, torch.Tensor)]
            if len(tensor_values) == len(values):
                if tensor_values[0].ndim >= 2:
                    merged_meta[key] = torch.cat(tensor_values, dim=1)
                else:
                    merged_meta[key] = tensor_values[0]
                continue

            first_non_none = next((value for value in values if value is not None), None)
            if first_non_none is not None:
                merged_meta[key] = first_non_none

        return merged_meta

    def _render_scene_single_device(self, scene: SceneBundle) -> dict[str, torch.Tensor]:
        """沿用现有单设备渲染路径."""

        renderer = self._get_renderer_for_scene(scene)
        render_output = renderer.render(
            self._get_gaussians_for_device(self._get_scene_device(scene)),
            scene,
        )
        if "images_pred" not in render_output:
            raise KeyError("Renderer output must contain `images_pred`.")
        return render_output

    def _render_scene_shards(self, scene: SceneBundle) -> list[dict[str, Any]]:
        """渲染并返回每个 view shard 的原始输出.

        这个辅助函数不做任何聚合.
        调用方可以按自己的需要:
        - 聚到主卡
        - 聚到 CPU
        - 或者逐 shard 直接做 loss/backward
        """

        if scene.gt_images.shape[0] != 1:
            self.diagnostics_state.setdefault("warnings", []).append("multi_device_render_fallback_batch_gt_1")
            return []

        shard_specs = self._build_render_shards(scene.gt_images.shape[1])
        if len(shard_specs) <= 1:
            return []

        gaussians_primary = self.gaussians.to_tensor()
        shard_outputs: list[dict[str, Any]] = []
        for render_device, start_index, end_index in shard_specs:
            shard_scene = self._get_scene_shard(scene, render_device, start_index, end_index)
            renderer = self._get_renderer_for_scene(shard_scene)
            shard_output = renderer.render(
                self._get_gaussians_for_device(render_device, gaussians_primary),
                shard_scene,
            )
            if "images_pred" not in shard_output:
                raise KeyError("Renderer output must contain `images_pred`.")
            shard_outputs.append(
                {
                    "device": render_device,
                    "start_index": start_index,
                    "end_index": end_index,
                    "scene": shard_scene,
                    "output": shard_output,
                }
            )
        return shard_outputs

    def _use_multi_device_render(self, scene: SceneBundle) -> bool:
        """判断当前 scene 是否真的启用多设备渲染."""

        return len(self.render_devices) > 1 and scene.gt_images.shape[1] > 1

    def _render_scene_multi_device(
        self,
        scene: SceneBundle,
        *,
        gather_device: torch.device | None = None,
        detach_outputs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """按 view shard 把渲染负载切到多张设备上.

        这里仍由主设备持有参数与 optimizer.
        其它设备只负责前向渲染对应的 view 子集.
        """

        if scene.gt_images.shape[0] != 1:
            self.diagnostics_state.setdefault("warnings", []).append("multi_device_render_fallback_batch_gt_1")
            return self._render_scene_single_device(scene)

        shard_outputs = self._render_scene_shards(scene)
        if not shard_outputs:
            return self._render_scene_single_device(scene)

        target_device = self.device if gather_device is None else gather_device
        image_shards: list[torch.Tensor] = []
        alpha_shards: list[torch.Tensor] = []
        depth_shards: list[torch.Tensor] = []
        render_meta_shards: list[dict[str, Any] | None] = []

        for shard_payload in shard_outputs:
            shard_output = shard_payload["output"]
            image_tensor = shard_output["images_pred"].detach() if detach_outputs else shard_output["images_pred"]
            image_shards.append(image_tensor.to(target_device))
            if isinstance(shard_output.get("alphas_pred"), torch.Tensor):
                alpha_tensor = shard_output["alphas_pred"].detach() if detach_outputs else shard_output["alphas_pred"]
                alpha_shards.append(alpha_tensor.to(target_device))
            if isinstance(shard_output.get("depths_pred"), torch.Tensor):
                depth_tensor = shard_output["depths_pred"].detach() if detach_outputs else shard_output["depths_pred"]
                depth_shards.append(depth_tensor.to(target_device))
            render_meta_shards.append(
                self._move_render_meta_to_device(
                    shard_output.get("render_meta"),
                    target_device,
                    detach_tensors=detach_outputs,
                )
            )

        merged_output: dict[str, torch.Tensor | dict[str, Any]] = {
            "images_pred": torch.cat(image_shards, dim=1),
        }
        if alpha_shards:
            merged_output["alphas_pred"] = torch.cat(alpha_shards, dim=1)
        if depth_shards:
            merged_output["depths_pred"] = torch.cat(depth_shards, dim=1)

        merged_meta = self._merge_render_meta_shards(render_meta_shards)
        if merged_meta is not None:
            merged_output["render_meta"] = merged_meta

        return merged_output  # type: ignore[return-value]

    def _render_scene_for_evaluation(self, scene: SceneBundle) -> dict[str, torch.Tensor]:
        """渲染一个仅用于指标/诊断的 scene.

        多卡场景下把结果直接聚合到 CPU.
        这样可以避免 Phase 0 / Phase 1 / Phase 3S 这类无 backward 阶段
        在主卡上重新拼完整 view tensor 时再次 OOM.
        """

        if self._use_multi_device_render(scene):
            return self._render_scene_multi_device(scene, gather_device=torch.device("cpu"), detach_outputs=True)

        render_output = self._render_scene_single_device(scene)
        output_cpu: dict[str, torch.Tensor | dict[str, Any]] = {
            "images_pred": render_output["images_pred"].detach().cpu(),
        }
        if isinstance(render_output.get("alphas_pred"), torch.Tensor):
            output_cpu["alphas_pred"] = render_output["alphas_pred"].detach().cpu()
        if isinstance(render_output.get("depths_pred"), torch.Tensor):
            output_cpu["depths_pred"] = render_output["depths_pred"].detach().cpu()
        render_meta = self._move_render_meta_to_device(
            render_output.get("render_meta"),
            torch.device("cpu"),
            detach_tensors=True,
        )
        if render_meta is not None:
            output_cpu["render_meta"] = render_meta
        return output_cpu  # type: ignore[return-value]

    def _render_scene_serial_view_shards(
        self,
        scene: SceneBundle,
        *,
        views_per_shard: int,
        gather_device: torch.device | None = None,
        detach_outputs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """在单设备上沿 view 维串行分片渲染一个 scene.

        `Phase C` 的 full-frame HR render 很容易在单次前向里撞上 renderer 的显存峰值.
        这里主动把 reference-resolution scene 切成多个 view shard 串行跑,
        用更多前向次数换更低的单次峰值.
        """

        views_per_shard = max(1, int(views_per_shard))
        if scene.gt_images.shape[0] != 1 or scene.gt_images.shape[1] <= views_per_shard:
            if gather_device is None and not detach_outputs:
                return self._render_scene_single_device(scene)

            render_output = self._render_scene_single_device(scene)
            target_device = self._get_scene_device(scene) if gather_device is None else gather_device
            moved_output: dict[str, torch.Tensor | dict[str, Any]] = {
                "images_pred": (render_output["images_pred"].detach() if detach_outputs else render_output["images_pred"]).to(target_device)
            }
            if isinstance(render_output.get("alphas_pred"), torch.Tensor):
                alpha_tensor = render_output["alphas_pred"].detach() if detach_outputs else render_output["alphas_pred"]
                moved_output["alphas_pred"] = alpha_tensor.to(target_device)
            if isinstance(render_output.get("depths_pred"), torch.Tensor):
                depth_tensor = render_output["depths_pred"].detach() if detach_outputs else render_output["depths_pred"]
                moved_output["depths_pred"] = depth_tensor.to(target_device)
            render_meta = self._move_render_meta_to_device(
                render_output.get("render_meta"),
                target_device,
                detach_tensors=detach_outputs,
            )
            if render_meta is not None:
                moved_output["render_meta"] = render_meta
            return moved_output  # type: ignore[return-value]

        active_device = self._get_scene_device(scene)
        target_device = active_device if gather_device is None else gather_device
        gaussians_primary = self.gaussians.to_tensor()
        image_shards: list[torch.Tensor] = []
        alpha_shards: list[torch.Tensor] = []
        depth_shards: list[torch.Tensor] = []
        render_meta_shards: list[dict[str, Any] | None] = []

        for start_index in range(0, scene.gt_images.shape[1], views_per_shard):
            end_index = min(scene.gt_images.shape[1], start_index + views_per_shard)
            shard_scene = self._get_scene_shard(scene, active_device, start_index, end_index)
            renderer = self._get_renderer_for_scene(shard_scene)
            shard_output = renderer.render(
                self._get_gaussians_for_device(active_device, gaussians_primary),
                shard_scene,
            )
            if "images_pred" not in shard_output:
                raise KeyError("Renderer output must contain `images_pred`.")

            image_tensor = shard_output["images_pred"].detach() if detach_outputs else shard_output["images_pred"]
            image_shards.append(image_tensor.to(target_device))
            if isinstance(shard_output.get("alphas_pred"), torch.Tensor):
                alpha_tensor = shard_output["alphas_pred"].detach() if detach_outputs else shard_output["alphas_pred"]
                alpha_shards.append(alpha_tensor.to(target_device))
            if isinstance(shard_output.get("depths_pred"), torch.Tensor):
                depth_tensor = shard_output["depths_pred"].detach() if detach_outputs else shard_output["depths_pred"]
                depth_shards.append(depth_tensor.to(target_device))
            render_meta_shards.append(
                self._move_render_meta_to_device(
                    shard_output.get("render_meta"),
                    target_device,
                    detach_tensors=detach_outputs,
                )
            )

        merged_output: dict[str, torch.Tensor | dict[str, Any]] = {
            "images_pred": torch.cat(image_shards, dim=1),
        }
        if alpha_shards:
            merged_output["alphas_pred"] = torch.cat(alpha_shards, dim=1)
        if depth_shards:
            merged_output["depths_pred"] = torch.cat(depth_shards, dim=1)

        merged_meta = self._merge_render_meta_shards(render_meta_shards)
        if merged_meta is not None:
            merged_output["render_meta"] = merged_meta
        return merged_output  # type: ignore[return-value]


    def _iter_scene_single_device_view_shards(
        self,
        scene: SceneBundle,
        *,
        views_per_shard: int,
    ):
        """在单设备上按 view shard 逐块渲染.

        这个 helper 不会把所有 shard 再拼回一个大张量.
        适合 `Phase C` 这类 full-frame HR loss 需要边渲染边反传的路径.
        """

        views_per_shard = max(1, int(views_per_shard))
        active_device = self._get_scene_device(scene)
        gaussians_primary = self.gaussians.to_tensor()

        if scene.gt_images.shape[0] != 1:
            renderer = self._get_renderer_for_scene(scene)
            shard_output = renderer.render(
                self._get_gaussians_for_device(active_device, gaussians_primary),
                scene,
            )
            if "images_pred" not in shard_output:
                raise KeyError("Renderer output must contain `images_pred`.")
            yield {
                "start_index": 0,
                "end_index": scene.gt_images.shape[1],
                "scene": scene,
                "output": shard_output,
            }
            return

        for start_index in range(0, scene.gt_images.shape[1], views_per_shard):
            end_index = min(scene.gt_images.shape[1], start_index + views_per_shard)
            shard_scene = self._get_scene_shard(scene, active_device, start_index, end_index)
            renderer = self._get_renderer_for_scene(shard_scene)
            shard_output = renderer.render(
                self._get_gaussians_for_device(active_device, gaussians_primary),
                shard_scene,
            )
            if "images_pred" not in shard_output:
                raise KeyError("Renderer output must contain `images_pred`.")
            yield {
                "start_index": start_index,
                "end_index": end_index,
                "scene": shard_scene,
                "output": shard_output,
            }

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

        if self._use_multi_device_render(scene):
            return self._render_scene_multi_device(scene)
        return self._render_scene_single_device(scene)

    def render_current_scene(self) -> dict[str, torch.Tensor]:
        """渲染当前高斯场景."""

        return self.render_scene(self.scene)

    def _extract_render_meta(self, render_output: dict[str, torch.Tensor]) -> dict[str, Any] | None:
        """从 renderer 输出中提取可选的 dense meta."""

        render_meta = render_output.get("render_meta")
        if isinstance(render_meta, dict):
            return render_meta
        return None

    def _append_warning_once(self, warning: str) -> None:
        """只追加一次 warning,避免同类降级在每轮循环刷屏."""

        warnings = self.diagnostics_state.setdefault("warnings", [])
        if warning not in warnings:
            warnings.append(warning)

    def _depth_anchor_requested(self) -> bool:
        """判断这次运行是否真的请求了 depth anchor."""

        return bool(self.hparams.enable_depth_anchor and self.hparams.lambda_depth_anchor > 0.0)

    def _stage_uses_depth_anchor(self, stage_name: str) -> bool:
        """V1 只让 appearance 阶段启用 depth anchor."""

        return self._depth_anchor_requested() and stage_name in {"stage2a", "stage3sr"}

    def _set_depth_anchor_reference_status(
        self,
        *,
        ready: bool,
        valid_ratio: float = 0.0,
        skip_reason: str | None = None,
    ) -> None:
        """同步更新 reference 级别的 depth anchor 状态."""

        self.diagnostics_state["depth_anchor_reference_ready"] = ready
        self.diagnostics_state["depth_anchor_reference_valid_ratio"] = valid_ratio
        self.diagnostics_state["depth_anchor_reference_skip_reason"] = skip_reason

    def _capture_depth_anchor_reference(self) -> None:
        """捕获一份 immutable baseline depth anchor."""

        self.depth_anchor_capture_attempted = True
        source = self.hparams.depth_anchor_source
        self.diagnostics_state["depth_anchor_source"] = source

        # V1 只实现 baseline_render.
        # 其它来源先保留参数面,但不偷偷伪装成已经支持.
        if source != "baseline_render":
            reason = f"unsupported_source:{source}"
            self.depth_anchor_reference = None
            self._set_depth_anchor_reference_status(ready=False, skip_reason=reason)
            self._append_warning_once(f"depth_anchor_skipped:{reason}")
            return

        render_output = self._render_scene_for_evaluation(self.scene)
        reference_depth = render_output.get("depths_pred")
        reference_alpha = render_output.get("alphas_pred")
        if not isinstance(reference_depth, torch.Tensor):
            reason = "reference_depth_missing"
            self.depth_anchor_reference = None
            self._set_depth_anchor_reference_status(ready=False, skip_reason=reason)
            self._append_warning_once(f"depth_anchor_skipped:{reason}")
            return
        if reference_alpha is not None and not isinstance(reference_alpha, torch.Tensor):
            reference_alpha = None

        try:
            valid_mask = build_depth_anchor_valid_mask(
                reference_depth,
                reference_alpha,
                alpha_threshold=self.hparams.depth_anchor_alpha_threshold,
            )
        except ValueError as exc:
            reason = f"reference_invalid:{exc}"
            self.depth_anchor_reference = None
            self._set_depth_anchor_reference_status(ready=False, skip_reason=reason)
            self._append_warning_once(f"depth_anchor_skipped:{reason}")
            return

        valid_ratio = float(valid_mask.float().mean().item())
        if valid_ratio <= 0.0:
            reason = "empty_reference_mask"
            self.depth_anchor_reference = None
            self._set_depth_anchor_reference_status(ready=False, skip_reason=reason)
            self._append_warning_once(f"depth_anchor_skipped:{reason}")
            return

        self.depth_anchor_reference = DepthAnchorReference(
            depth=reference_depth.detach().cpu(),
            alpha=reference_alpha.detach().cpu() if isinstance(reference_alpha, torch.Tensor) else None,
            valid_mask=valid_mask.detach().cpu(),
            valid_ratio=valid_ratio,
            source=source,
        )
        self._set_depth_anchor_reference_status(ready=True, valid_ratio=valid_ratio, skip_reason=None)

    def _ensure_depth_anchor_reference(self, stage_name: str) -> None:
        """在进入 appearance loop 前按需捕获 baseline reference."""

        if not self._stage_uses_depth_anchor(stage_name):
            return
        if self.depth_anchor_reference is not None or self.depth_anchor_capture_attempted:
            return
        self._capture_depth_anchor_reference()

    def _compute_depth_anchor_loss_for_view_range(
        self,
        pred_depth: torch.Tensor | None,
        *,
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, float, str | None]:
        """对某个 view 范围计算 depth anchor loss."""

        if not isinstance(pred_depth, torch.Tensor):
            zero_loss = self.gaussians.opacity.new_zeros(())
            return zero_loss, 0.0, "pred_depth_missing"

        if self.depth_anchor_reference is None:
            zero_loss = pred_depth.new_zeros(())
            skip_reason = self.diagnostics_state.get("depth_anchor_reference_skip_reason") or "reference_unavailable"
            return zero_loss, 0.0, skip_reason

        reference_depth = self._slice_view_tensor(
            self.depth_anchor_reference.depth,
            start_index,
            end_index,
            device=pred_depth.device,
            dtype=pred_depth.dtype,
        )
        reference_valid_mask = self._slice_view_tensor(
            self.depth_anchor_reference.valid_mask,
            start_index,
            end_index,
            device=pred_depth.device,
            dtype=torch.bool,
        )
        if reference_depth is None or reference_valid_mask is None:
            zero_loss = pred_depth.new_zeros(())
            return zero_loss, 0.0, "reference_slice_missing"

        loss_summary = compute_depth_anchor_loss(pred_depth, reference_depth, reference_valid_mask)
        return loss_summary.loss, loss_summary.valid_ratio, loss_summary.skip_reason

    def _build_depth_anchor_metrics(
        self,
        *,
        stage_name: str,
        loss_value: float,
        valid_ratio: float,
        skip_reason: str | None,
    ) -> dict[str, Any]:
        """把当前轮 depth anchor 的诊断项整理成 metrics."""

        if skip_reason is not None:
            self._append_warning_once(f"{stage_name}_depth_anchor_skipped:{skip_reason}")

        self.diagnostics_state["depth_anchor_last_skip_reason"] = skip_reason
        self.diagnostics_state["depth_anchor_last_valid_ratio"] = valid_ratio
        self.diagnostics_state["depth_anchor_last_loss"] = loss_value
        return {
            "loss_depth_anchor": loss_value,
            "depth_anchor_active": skip_reason is None,
            "depth_anchor_valid_ratio": valid_ratio,
            "depth_anchor_source": self.hparams.depth_anchor_source,
            "depth_anchor_reference_ready": self.depth_anchor_reference is not None,
            "depth_anchor_skip_reason": skip_reason,
        }

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
        fidelity_diagnostics: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """把 Phase 3S 的诊断产物落盘."""

        self.diagnostics.write_gaussian_fidelity_summary(
            fidelity_score,
            fidelity_diagnostics=fidelity_diagnostics,
        )
        self.diagnostics.write_sr_selection_stats(sr_selection_map)
        for frame_id in range(sr_selection_map.shape[1]):
            self.diagnostics.save_sr_selection_map(stage_name, frame_id, sr_selection_map[:, frame_id])

    def _patch_supervision_configured(self) -> bool:
        """判断当前是否配置了 patch supervision 所需参数."""

        return self.hparams.patch_size > 0 and (
            self.hparams.lambda_patch_rgb > 0.0 or self.hparams.lambda_patch_perceptual > 0.0
        )

    def _full_frame_hr_supervision_configured(self, scene: SceneBundle | None = None) -> bool:
        """判断当前是否打开了 `Phase C` 风格的 full-frame HR supervision."""

        active_scene = self.scene if scene is None else scene
        if self.hparams.lambda_hr_rgb <= 0.0:
            return False
        if not isinstance(active_scene.reference_images, torch.Tensor):
            return False
        return active_scene.reference_images.shape[-2:] != active_scene.gt_images.shape[-2:]

    def _stage3sr_supervision_configured(self, scene: SceneBundle | None = None) -> bool:
        """判断 Stage 3SR 是否至少有一种监督模式可用."""

        return self._patch_supervision_configured() or self._full_frame_hr_supervision_configured(scene)

    def _resolve_stage3sr_supervision_mode(self, scene: SceneBundle | None = None) -> str:
        """解析当前 Stage 3SR 真正要走的监督模式."""

        active_scene = self.scene if scene is None else scene
        uses_patch = self._patch_supervision_configured()
        uses_full_frame_hr = self._full_frame_hr_supervision_configured(active_scene)
        if uses_patch and uses_full_frame_hr:
            raise RuntimeError(
                "Stage 3SR cannot enable patch supervision and full-frame HR supervision at the same time. "
                "Please choose one supervision mode."
            )
        if uses_full_frame_hr:
            return "full_frame_hr"
        if uses_patch:
            return "patch"
        return "none"

    def _stage2a_should_run_stage3sr(self) -> bool:
        """根据当前模式和参数,判断 Stage 2A 是否应进入增强链路."""

        if self.run_config.stage2a_mode == "legacy":
            return False
        try:
            return self._resolve_stage3sr_supervision_mode() != "none"
        except RuntimeError:
            return False

    def _resolve_stage2a_mode(self) -> str:
        """把 `auto/legacy/enhanced` 解析成本轮真正执行的模式."""

        requested_mode = self.run_config.stage2a_mode
        if requested_mode == "legacy":
            return "legacy"
        if requested_mode == "enhanced":
            if not self._stage3sr_supervision_configured():
                raise RuntimeError(
                    "stage2a_mode=enhanced requires Stage 3SR supervision. "
                    "Please enable either patch supervision, or full-frame HR supervision."
                )
            self._resolve_stage3sr_supervision_mode()
            return "enhanced"
        if self._stage3sr_supervision_configured():
            self._resolve_stage3sr_supervision_mode()
            return "enhanced"
        return "legacy"

    def _get_reference_images(
        self,
        scene: SceneBundle | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """统一拿到当前 scene 的 reference 图像,并按需要对齐设备/类型."""

        active_scene = self.scene if scene is None else scene
        reference_images = active_scene.reference_images if isinstance(active_scene.reference_images, torch.Tensor) else active_scene.gt_images
        if device is None and dtype is None:
            return reference_images

        target_device = reference_images.device if device is None else device
        target_dtype = reference_images.dtype if dtype is None else dtype
        if reference_images.device == target_device and reference_images.dtype == target_dtype:
            return reference_images
        return reference_images.to(device=target_device, dtype=target_dtype)

    def _get_gt_images(
        self,
        scene: SceneBundle | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """统一拿到当前 scene 的 GT 图像,并按需要对齐设备/类型."""

        active_scene = self.scene if scene is None else scene
        gt_images = active_scene.gt_images
        if device is None and dtype is None:
            return gt_images

        target_device = gt_images.device if device is None else device
        target_dtype = gt_images.dtype if dtype is None else dtype
        if gt_images.device == target_device and gt_images.dtype == target_dtype:
            return gt_images
        return gt_images.to(device=target_device, dtype=target_dtype)

    def _reference_space_enabled(self, scene: SceneBundle | None = None) -> bool:
        """判断当前 scene 是否真的存在独立的 HR reference 空间.

        `Phase D` 的关键不是“永远再导一份视频”.
        而是只在 reference 分辨率真的高于 native 时, 才补出 HR 导出与指标.
        """

        active_scene = self.scene if scene is None else scene
        reference_images = self._get_reference_images(active_scene)
        return tuple(reference_images.shape[-2:]) != tuple(active_scene.gt_images.shape[-2:])

    def _build_reference_render_scene(self, scene: SceneBundle | None = None) -> SceneBundle:
        """构造一个 reference 分辨率的整图渲染 scene.

        这里不再额外分配一份同尺寸零张量.
        直接复用 `reference_images` 作为 shape carrier 即可.
        renderer 只关心分辨率与 intrinsics, 不会把它当成监督目标去消费.
        """

        active_scene = self.scene if scene is None else scene
        reference_images = self._get_reference_images(active_scene)
        reference_intrinsics = (
            active_scene.intrinsics_ref if isinstance(active_scene.intrinsics_ref, torch.Tensor) else active_scene.intrinsics
        )
        reference_hw = (
            active_scene.reference_hw
            if active_scene.reference_hw is not None
            else (int(reference_images.shape[-2]), int(reference_images.shape[-1]))
        )
        return SceneBundle(
            gt_images=reference_images,
            cam_view=active_scene.cam_view,
            intrinsics=reference_intrinsics,
            frame_indices=active_scene.frame_indices,
            scene_index=active_scene.scene_index,
            view_id=active_scene.view_id,
            view_ids=active_scene.view_ids,
            target_index=active_scene.target_index,
            file_name=active_scene.file_name,
            reference_images=reference_images,
            intrinsics_ref=reference_intrinsics,
            native_hw=active_scene.native_hw,
            reference_hw=reference_hw,
            reference_mode=active_scene.reference_mode,
            sr_scale=active_scene.sr_scale,
        )

    def _render_reference_scene_for_training(self, scene: SceneBundle | None = None) -> dict[str, torch.Tensor]:
        """在 reference 分辨率渲染整图 HR output.

        默认会按 `reference_render_shard_views` 串行切 view,
        避免 full-frame HR render 一次性把显存峰值拉爆.
        """

        reference_scene = self._build_reference_render_scene(scene)
        if self._use_multi_device_render(reference_scene):
            return self.render_scene(reference_scene)
        return self._render_scene_serial_view_shards(
            reference_scene,
            views_per_shard=self.hparams.reference_render_shard_views,
        )

    def _render_reference_scene_for_evaluation(self, scene: SceneBundle | None = None) -> dict[str, torch.Tensor]:
        """在 reference 分辨率渲染一个只用于导出/指标的 HR scene.

        训练路径已经有了 `reference-space` scene builder.
        `Phase D` 这里补的是 evaluation/export 版本, 让最终产物也能直接落到 HR 空间.
        """

        reference_scene = self._build_reference_render_scene(scene)
        if self._use_multi_device_render(reference_scene):
            return self._render_scene_multi_device(reference_scene, gather_device=torch.device("cpu"), detach_outputs=True)
        return self._render_scene_serial_view_shards(
            reference_scene,
            views_per_shard=self.hparams.reference_render_shard_views,
            gather_device=torch.device("cpu"),
            detach_outputs=True,
        )

    def _resolve_patch_sizes(self, scene: SceneBundle | None = None) -> tuple[int, int, int]:
        """解析 native/reference patch 尺寸和缩放倍率."""

        active_scene = self.scene if scene is None else scene
        reference_patch_size = int(self.hparams.patch_size)
        if reference_patch_size <= 0:
            raise ValueError("patch_size must be positive when patch supervision is enabled.")

        scale_value = float(active_scene.sr_scale)
        scale_int = int(round(scale_value))
        if scale_int <= 0 or abs(scale_value - scale_int) > 1e-6:
            raise ValueError(f"Patch supervision requires integer sr_scale, got {active_scene.sr_scale}.")
        if reference_patch_size % scale_int != 0:
            raise ValueError(
                f"patch_size {reference_patch_size} must be divisible by sr_scale {scale_int}."
            )

        native_patch_size = reference_patch_size // scale_int
        native_height, native_width = active_scene.gt_images.shape[-2:]
        reference_images = self._get_reference_images(active_scene)
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

    def sample_patch_windows(self, residual_map: torch.Tensor, scene: SceneBundle | None = None) -> torch.Tensor:
        """根据 residual 热点在 native 尺度采样 patch window."""

        native_patch_size, _, _ = self._resolve_patch_sizes(scene)
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

    def _build_reference_supervision_weight_map(
        self,
        residual_map: torch.Tensor,
        *,
        native_weight_map: torch.Tensor | None = None,
        sr_selection_map: torch.Tensor | None = None,
        scene: SceneBundle | None = None,
    ) -> torch.Tensor | None:
        """把 native robust 权重与 reference selection 组合成 reference 尺度监督权重."""

        active_scene = self.scene if scene is None else scene
        active_sr_selection = self.sr_selection_map if sr_selection_map is None else sr_selection_map

        active_weight_map = self.prev_weight_map if native_weight_map is None else native_weight_map
        if active_weight_map is None:
            active_weight_map = torch.ones_like(residual_map)
        if active_weight_map.device != residual_map.device or active_weight_map.dtype != residual_map.dtype:
            active_weight_map = active_weight_map.to(device=residual_map.device, dtype=residual_map.dtype)

        reference_hw = self._get_reference_images(active_scene).shape[-2:]
        robust_weight_ref = self._resize_patch_tensor(active_weight_map, reference_hw)
        if active_sr_selection is None:
            return robust_weight_ref.detach()
        if active_sr_selection.device != robust_weight_ref.device or active_sr_selection.dtype != robust_weight_ref.dtype:
            active_sr_selection = active_sr_selection.to(device=robust_weight_ref.device, dtype=robust_weight_ref.dtype)
        return self.weight_builder.combine_sr_weights(robust_weight_ref, active_sr_selection).detach()

    def _build_sr_patch_priority_map(
        self,
        residual_map: torch.Tensor,
        *,
        native_weight_map: torch.Tensor | None = None,
        sr_selection_map: torch.Tensor | None = None,
        scene: SceneBundle | None = None,
    ) -> torch.Tensor | None:
        """把 reference 尺度监督权重复用成 patch 选窗优先级图."""

        return self._build_reference_supervision_weight_map(
            residual_map,
            native_weight_map=native_weight_map,
            sr_selection_map=sr_selection_map,
            scene=scene,
        )

    def sample_sr_patch_window_sets(
        self,
        residual_map: torch.Tensor,
        *,
        native_weight_map: torch.Tensor | None = None,
        sr_selection_map: torch.Tensor | None = None,
        scene: SceneBundle | None = None,
    ) -> torch.Tensor:
        """按 selection priority 选出多个 patch window.

        `Phase B` 的最小版先不直接上整图 HR render.
        而是把“单热点 residual patch”升级成“按 reference priority 选多个 patch”.
        这样可以复用现有 patch render 路径,同时明显扩大 SR supervision 覆盖面.
        """

        active_scene = self.scene if scene is None else scene
        fallback_windows = self.sample_patch_windows(residual_map, scene=active_scene)
        patch_sets_per_view = max(1, int(self.hparams.sr_patches_per_view))
        if patch_sets_per_view <= 1:
            return fallback_windows.unsqueeze(0)

        priority_map = self._build_sr_patch_priority_map(
            residual_map,
            native_weight_map=native_weight_map,
            sr_selection_map=sr_selection_map,
            scene=active_scene,
        )
        if priority_map is None:
            return fallback_windows.unsqueeze(0)

        native_patch_size, reference_patch_size, scale_int = self._resolve_patch_sizes(active_scene)
        batch_size, num_views, _, _, _ = priority_map.shape
        pooled_scores = F.avg_pool2d(
            priority_map.detach().mean(dim=2, keepdim=True).reshape(batch_size * num_views, 1, *priority_map.shape[-2:]),
            kernel_size=reference_patch_size,
            stride=scale_int,
        )
        score_height, score_width = pooled_scores.shape[-2:]
        if min(score_height, score_width) <= 0:
            return fallback_windows.unsqueeze(0)

        patch_window_sets = fallback_windows.unsqueeze(0).repeat(patch_sets_per_view, 1, 1, 1)
        score_map = pooled_scores.view(batch_size, num_views, score_height, score_width).clone()

        for batch_index in range(batch_size):
            for view_index in range(num_views):
                view_scores = score_map[batch_index, view_index]
                fallback_window = fallback_windows[batch_index, view_index]
                for patch_set_index in range(patch_sets_per_view):
                    flat_scores = view_scores.reshape(-1)
                    flat_index = int(torch.argmax(flat_scores).item())
                    best_score = float(flat_scores[flat_index].item())
                    if best_score <= 0.0:
                        if patch_set_index > 0:
                            patch_window_sets[patch_set_index, batch_index, view_index] = patch_window_sets[
                                patch_set_index - 1,
                                batch_index,
                                view_index,
                            ]
                        else:
                            patch_window_sets[patch_set_index, batch_index, view_index] = fallback_window
                        continue

                    top_native = flat_index // score_width
                    left_native = flat_index % score_width
                    patch_window_sets[patch_set_index, batch_index, view_index] = torch.tensor(
                        [top_native, left_native, native_patch_size, native_patch_size],
                        dtype=torch.long,
                        device=residual_map.device,
                    )

                    suppress_top = max(0, top_native - native_patch_size + 1)
                    suppress_bottom = min(score_height, top_native + native_patch_size)
                    suppress_left = max(0, left_native - native_patch_size + 1)
                    suppress_right = min(score_width, left_native + native_patch_size)
                    view_scores[suppress_top:suppress_bottom, suppress_left:suppress_right] = -1.0

        return patch_window_sets

    def map_patch_windows_to_reference(
        self,
        patch_windows_native: torch.Tensor,
        scene: SceneBundle | None = None,
    ) -> torch.Tensor:
        """把 native patch window 映射到 reference 尺度."""

        _, _, scale_int = self._resolve_patch_sizes(scene)
        patch_windows_ref = patch_windows_native.detach().clone()
        patch_windows_ref[..., 0:2] *= scale_int
        patch_windows_ref[..., 2:4] *= scale_int
        return patch_windows_ref

    def gather_reference_patch(
        self,
        patch_windows_ref: torch.Tensor,
        scene: SceneBundle | None = None,
    ) -> torch.Tensor:
        """从 reference 图像中提取 patch."""

        reference_images = self._get_reference_images(scene)
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

    def build_patch_intrinsics(self, patch_windows_ref: torch.Tensor, scene: SceneBundle | None = None) -> torch.Tensor:
        """基于 reference intrinsics 构造 patch camera intrinsics."""

        active_scene = self.scene if scene is None else scene
        base_intrinsics = active_scene.intrinsics_ref if isinstance(active_scene.intrinsics_ref, torch.Tensor) else active_scene.intrinsics
        patch_intrinsics = base_intrinsics.detach().clone()
        # `patch_windows_ref` 可能来自 CPU diagnostics 路径.
        # 这里必须显式对齐到 intrinsics 所在设备,否则 warm-start 的
        # `Stage 2B` 在 CUDA scene 下会因为跨设备减法直接报错.
        patch_offsets = patch_windows_ref.to(device=patch_intrinsics.device, dtype=patch_intrinsics.dtype)
        patch_intrinsics[..., 2] -= patch_offsets[..., 1]
        patch_intrinsics[..., 3] -= patch_offsets[..., 0]
        return patch_intrinsics

    def render_patch_prediction(
        self,
        patch_windows_native: torch.Tensor,
        scene: SceneBundle | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在 reference 尺度渲染 patch 并返回对应参考图."""

        active_scene = self.scene if scene is None else scene
        patch_windows_ref = self.map_patch_windows_to_reference(patch_windows_native, scene=active_scene)
        reference_patch = self.gather_reference_patch(patch_windows_ref, scene=active_scene)
        patch_intrinsics = self.build_patch_intrinsics(patch_windows_ref, scene=active_scene)
        native_patch_size, reference_patch_size, _ = self._resolve_patch_sizes(active_scene)

        patch_scene = SceneBundle(
            gt_images=torch.zeros_like(reference_patch),
            cam_view=active_scene.cam_view,
            intrinsics=patch_intrinsics,
            frame_indices=active_scene.frame_indices,
            scene_index=active_scene.scene_index,
            view_id=active_scene.view_id,
            view_ids=active_scene.view_ids,
            target_index=active_scene.target_index,
            file_name=active_scene.file_name,
            reference_images=reference_patch,
            intrinsics_ref=patch_intrinsics,
            native_hw=(native_patch_size, native_patch_size),
            reference_hw=(reference_patch_size, reference_patch_size),
            reference_mode=active_scene.reference_mode,
            sr_scale=active_scene.sr_scale,
        )
        # patch 渲染的目标只是局部监督.
        # 即便主场景启用了多卡,这里也直接在 patch scene 当前设备上单卡渲染,
        # 避免再次走整段 view 聚合,把小 patch 路径重新放大成主卡 OOM 风险点.
        patch_render_output = self._render_scene_single_device(patch_scene)
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
        *,
        scene: SceneBundle | None = None,
        native_weight_map: torch.Tensor | None = None,
        sr_selection_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """统一计算可选 patch supervision 损失.

        Stage 2A 和 Stage 2B 都复用这条逻辑.
        这样 patch 监督的行为只维护一份,不会因为阶段切换而漂移.
        """

        loss_patch_rgb = torch.zeros((), dtype=reference_tensor.dtype, device=reference_tensor.device)
        loss_patch_perceptual = torch.zeros((), dtype=reference_tensor.dtype, device=reference_tensor.device)
        self.last_sr_patch_sets_used = 0
        if not self._patch_supervision_configured():
            return loss_patch_rgb, loss_patch_perceptual

        active_scene = self.scene if scene is None else scene
        active_weight_map = self.prev_weight_map if native_weight_map is None else native_weight_map
        if active_weight_map is None:
            active_weight_map = torch.ones_like(residual_map)
        elif active_weight_map.device != residual_map.device or active_weight_map.dtype != residual_map.dtype:
            active_weight_map = active_weight_map.to(device=residual_map.device, dtype=residual_map.dtype)

        active_sr_selection = self.sr_selection_map if sr_selection_map is None else sr_selection_map
        patch_window_sets = self.sample_sr_patch_window_sets(
            residual_map,
            native_weight_map=active_weight_map,
            sr_selection_map=active_sr_selection,
            scene=active_scene,
        )
        self.last_sr_patch_sets_used = int(patch_window_sets.shape[0])

        for patch_windows_native in patch_window_sets:
            pred_patch, reference_patch, _ = self.render_patch_prediction(patch_windows_native, scene=active_scene)
            patch_windows_ref = self.map_patch_windows_to_reference(patch_windows_native, scene=active_scene)

            robust_patch_native = self._gather_tensor_patch(active_weight_map, patch_windows_native)
            robust_patch = self._resize_patch_tensor(robust_patch_native, pred_patch.shape[-2:]).to(
                device=pred_patch.device,
                dtype=pred_patch.dtype,
            )

            if active_sr_selection is None:
                # warm-start resume 可能还没有恢复 `sr_selection_map`.
                # fallback 权重也必须落在 patch render 的设备上,否则后续组合权重会再次跨设备报错.
                sr_selection_patch = torch.ones_like(robust_patch, device=pred_patch.device, dtype=pred_patch.dtype)
            else:
                if active_sr_selection.device != pred_patch.device or active_sr_selection.dtype != pred_patch.dtype:
                    active_sr_selection = active_sr_selection.to(device=pred_patch.device, dtype=pred_patch.dtype)
                sr_selection_patch = self._gather_tensor_patch(active_sr_selection, patch_windows_ref).to(
                    device=pred_patch.device,
                    dtype=pred_patch.dtype,
                )

            patch_weights = self.weight_builder.combine_sr_weights(
                robust_patch,
                sr_selection_patch,
            )
            loss_patch_rgb = loss_patch_rgb + compute_weighted_rgb_loss(pred_patch, reference_patch, patch_weights)
            loss_patch_perceptual = loss_patch_perceptual + compute_patch_perceptual_loss(pred_patch, reference_patch)

        patch_set_denominator = max(1, int(patch_window_sets.shape[0]))
        loss_patch_rgb = loss_patch_rgb / patch_set_denominator
        loss_patch_perceptual = loss_patch_perceptual / patch_set_denominator
        return loss_patch_rgb, loss_patch_perceptual

    def _resolve_geometry_stage_hparams(self, stage_name: str) -> dict[str, float | int]:
        """解析当前 geometry 阶段应使用的超参数.

        `stage2b` 继续沿用历史参数.
        `stage3b` 则优先读取独立超参数面,避免继续被旧 limited geometry 口径绑住.
        """

        if stage_name == "stage3b":
            return {
                "iters": int(self.hparams.iters_stage3b),
                "lambda_means_anchor": float(self.hparams.lambda_means_anchor_stage3b),
                "lambda_rotation_reg": float(self.hparams.lambda_rotation_reg_stage3b),
                "means_delta_cap": float(self.hparams.means_delta_cap_stage3b),
            }

        return {
            "iters": int(self.hparams.iters_stage2b),
            "lambda_means_anchor": float(self.hparams.lambda_means_anchor),
            "lambda_rotation_reg": float(self.hparams.lambda_rotation_reg),
            "means_delta_cap": float(self.hparams.means_delta_cap),
        }

    def _run_reference_supervised_stage(
        self,
        *,
        stage_name: str,
        export_file_name: str,
        geometry_stage: bool,
    ) -> dict[str, Any]:
        """运行 reference-space 主监督阶段.

        `geometry_stage=False` 时,它就是当前 `Phase C` 的 `stage3sr`.
        `geometry_stage=True` 时,它就是 `Phase E` 的 `stage3b`:
        在保持 HR 主监督 + LR consistency 的同时, 允许有限 geometry 更新.
        """

        self._ensure_depth_anchor_reference(stage_name)
        self.current_stage = stage_name
        train_stage_name = 'stage3b' if geometry_stage else 'stage3sr'
        self.gaussians.freeze_for_stage(train_stage_name)
        optimizer = self.gaussians.build_optimizer(train_stage_name, self.hparams)
        self.last_sr_patch_sets_used = 0
        geometry_hparams = self._resolve_geometry_stage_hparams(stage_name)

        final_metrics: dict[str, Any] = {}
        use_depth_anchor = self._stage_uses_depth_anchor(stage_name)
        total_iters = int(geometry_hparams["iters"]) if geometry_stage else self.hparams.iters_stage2a
        lambda_means_anchor = float(geometry_hparams["lambda_means_anchor"])
        lambda_rotation_reg = float(geometry_hparams["lambda_rotation_reg"])
        for iter_idx in range(total_iters):
            optimizer.zero_grad(set_to_none=True)

            # native render 仍然只负责 residual / robust weight 与可见性统计.
            # 如果当前阶段没开 depth anchor, 这一支不需要保留反传图.
            if use_depth_anchor:
                render_output = self.render_current_scene()
            else:
                with torch.no_grad():
                    render_output = self.render_current_scene()
            pred_rgb_native = render_output['images_pred']
            render_meta = self._extract_render_meta(render_output)
            self.latest_render_meta = render_meta

            gt_rgb = self._get_gt_images(device=pred_rgb_native.device, dtype=pred_rgb_native.dtype)
            gt_rgb_cpu = gt_rgb.detach().cpu()
            residual_map_native = self.weight_builder.build_residual_map(pred_rgb_native, gt_rgb)
            previous_weight_map = self.prev_weight_map
            if isinstance(previous_weight_map, torch.Tensor):
                previous_weight_map = previous_weight_map.to(device=pred_rgb_native.device, dtype=pred_rgb_native.dtype)
            self.prev_weight_map = self.weight_builder.build_weight_map(residual_map_native, previous_weight_map)

            loss_scale = compute_scale_tail_loss(self.gaussians.scales, self.hparams.scale_tail_threshold)
            loss_opacity = compute_opacity_sparse_loss(self.gaussians.opacity)
            if use_depth_anchor:
                loss_depth_anchor, depth_anchor_valid_ratio, depth_anchor_skip_reason = self._compute_depth_anchor_loss_for_view_range(
                    render_output.get('depths_pred'),
                    start_index=0,
                    end_index=self.scene.gt_images.shape[1],
                )
            else:
                loss_depth_anchor = torch.zeros((), dtype=pred_rgb_native.dtype, device=pred_rgb_native.device)
                depth_anchor_valid_ratio = 0.0
                depth_anchor_skip_reason = None

            native_metrics = self._summarize_prediction(
                pred_rgb_native,
                residual_map=residual_map_native,
                weight_map=self.prev_weight_map,
                gt_rgb=gt_rgb,
            )
            loss_sampling_smooth = compute_sampling_smooth_loss(
                scales=self.gaussians.scales,
                fidelity_score=self.gaussian_fidelity_score,
                render_meta=render_meta,
                radius_threshold=self.hparams.sampling_radius_threshold,
            )
            if geometry_stage:
                # `Phase E` 允许 geometry 参与优化,但仍要被 anchor / rotation reg 拉住.
                # 这样 SR 信息能推动结构,又不会一下子把点云拉飞.
                loss_means_anchor, loss_rotation_reg = compute_stage3b_losses(
                    self.gaussians.means,
                    self.gaussians.initial_means,
                    self.gaussians.rotations,
                    self.gaussians.initial_rotations,
                )
            else:
                loss_means_anchor = torch.zeros((), dtype=pred_rgb_native.dtype, device=pred_rgb_native.device)
                loss_rotation_reg = torch.zeros((), dtype=pred_rgb_native.dtype, device=pred_rgb_native.device)

            base_loss = (
                self.hparams.lambda_scale_tail * loss_scale
                + self.hparams.lambda_opacity_sparse * loss_opacity
                + self.hparams.lambda_depth_anchor * loss_depth_anchor
                + self.hparams.lambda_sampling_smooth * loss_sampling_smooth
                + lambda_means_anchor * loss_means_anchor
                + lambda_rotation_reg * loss_rotation_reg
            )
            base_loss.backward()

            del render_output
            if not use_depth_anchor:
                del pred_rgb_native

            reference_scene = self._build_reference_render_scene()
            total_views = max(1, int(reference_scene.gt_images.shape[1]))
            loss_hr_rgb_value = 0.0
            loss_lr_consistency_value = 0.0
            pred_rgb_hr_cpu_shards: list[torch.Tensor] = []
            pred_rgb_lr_cpu_shards: list[torch.Tensor] = []
            reference_rgb_cpu_shards: list[torch.Tensor] = []
            hr_residual_cpu_shards: list[torch.Tensor] = []
            lr_residual_cpu_shards: list[torch.Tensor] = []
            hr_weight_cpu_shards: list[torch.Tensor] = []

            for shard_payload in self._iter_scene_single_device_view_shards(
                reference_scene,
                views_per_shard=self.hparams.reference_render_shard_views,
            ):
                start_index = int(shard_payload['start_index'])
                end_index = int(shard_payload['end_index'])
                shard_scene = shard_payload['scene']
                shard_output = shard_payload['output']
                pred_rgb_hr_shard = shard_output['images_pred']
                reference_rgb_shard = self._get_reference_images(
                    shard_scene,
                    device=pred_rgb_hr_shard.device,
                    dtype=pred_rgb_hr_shard.dtype,
                )
                residual_map_native_shard = self._slice_view_tensor(
                    residual_map_native,
                    start_index,
                    end_index,
                    device=pred_rgb_hr_shard.device,
                    dtype=pred_rgb_hr_shard.dtype,
                )
                lr_weight_map_shard = self._slice_view_tensor(
                    self.prev_weight_map,
                    start_index,
                    end_index,
                    device=pred_rgb_hr_shard.device,
                    dtype=pred_rgb_hr_shard.dtype,
                )
                sr_selection_map_shard = self._slice_view_tensor(
                    self.sr_selection_map,
                    start_index,
                    end_index,
                    device=pred_rgb_hr_shard.device,
                    dtype=pred_rgb_hr_shard.dtype,
                )
                gt_rgb_shard = self._slice_view_tensor(
                    gt_rgb,
                    start_index,
                    end_index,
                    device=pred_rgb_hr_shard.device,
                    dtype=pred_rgb_hr_shard.dtype,
                )
                if residual_map_native_shard is None or lr_weight_map_shard is None or gt_rgb_shard is None:
                    raise RuntimeError('Phase C/Phase E shard preparation should never return None tensors.')

                hr_weight_map_shard = self._build_reference_supervision_weight_map(
                    residual_map_native_shard,
                    native_weight_map=lr_weight_map_shard,
                    sr_selection_map=sr_selection_map_shard,
                    scene=shard_scene,
                )
                if hr_weight_map_shard is None:
                    hr_weight_map_shard = torch.ones(
                        reference_rgb_shard.shape[0],
                        reference_rgb_shard.shape[1],
                        1,
                        reference_rgb_shard.shape[-2],
                        reference_rgb_shard.shape[-1],
                        device=pred_rgb_hr_shard.device,
                        dtype=pred_rgb_hr_shard.dtype,
                    )
                else:
                    hr_weight_map_shard = hr_weight_map_shard.to(device=pred_rgb_hr_shard.device, dtype=pred_rgb_hr_shard.dtype)

                loss_hr_rgb_shard = compute_weighted_rgb_loss(pred_rgb_hr_shard, reference_rgb_shard, hr_weight_map_shard)
                pred_rgb_lr_shard = downsample_rgb_tensor(pred_rgb_hr_shard, gt_rgb_shard.shape[-2:])
                loss_lr_consistency_shard = compute_weighted_rgb_loss(pred_rgb_lr_shard, gt_rgb_shard, lr_weight_map_shard)

                shard_view_fraction = float(end_index - start_index) / float(total_views)
                shard_loss_total = shard_view_fraction * (
                    self.hparams.lambda_hr_rgb * loss_hr_rgb_shard
                    + self.hparams.lambda_lr_consistency * loss_lr_consistency_shard
                )
                shard_loss_total.backward()

                loss_hr_rgb_value += float(loss_hr_rgb_shard.item()) * shard_view_fraction
                loss_lr_consistency_value += float(loss_lr_consistency_shard.item()) * shard_view_fraction
                pred_rgb_hr_cpu_shards.append(pred_rgb_hr_shard.detach().cpu())
                pred_rgb_lr_cpu_shards.append(pred_rgb_lr_shard.detach().cpu())
                reference_rgb_cpu_shards.append(reference_rgb_shard.detach().cpu())
                hr_residual_cpu_shards.append(
                    self.weight_builder.build_residual_map(pred_rgb_hr_shard, reference_rgb_shard).detach().cpu()
                )
                lr_residual_cpu_shards.append(
                    self.weight_builder.build_residual_map(pred_rgb_lr_shard, gt_rgb_shard).detach().cpu()
                )
                hr_weight_cpu_shards.append(hr_weight_map_shard.detach().cpu())

                del shard_output
                del pred_rgb_hr_shard
                del reference_rgb_shard
                del hr_weight_map_shard
                del pred_rgb_lr_shard
                del gt_rgb_shard

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.gaussians.clamp_stage_constraints(train_stage_name, self.hparams)

            pred_rgb_hr_cpu = torch.cat(pred_rgb_hr_cpu_shards, dim=1)
            pred_rgb_lr_cpu = torch.cat(pred_rgb_lr_cpu_shards, dim=1)
            reference_rgb_cpu = torch.cat(reference_rgb_cpu_shards, dim=1)
            hr_residual_map = torch.cat(hr_residual_cpu_shards, dim=1)
            lr_residual_map = torch.cat(lr_residual_cpu_shards, dim=1)
            hr_weight_map = torch.cat(hr_weight_cpu_shards, dim=1)
            loss_total_value = (
                self.hparams.lambda_hr_rgb * loss_hr_rgb_value
                + self.hparams.lambda_lr_consistency * loss_lr_consistency_value
                + self.hparams.lambda_scale_tail * float(loss_scale.item())
                + self.hparams.lambda_opacity_sparse * float(loss_opacity.item())
                + self.hparams.lambda_depth_anchor * float(loss_depth_anchor.item())
                + self.hparams.lambda_sampling_smooth * float(loss_sampling_smooth.item())
                + lambda_means_anchor * float(loss_means_anchor.item())
                + lambda_rotation_reg * float(loss_rotation_reg.item())
            )
            final_metrics = self._summarize_prediction(
                pred_rgb_lr_cpu,
                residual_map=lr_residual_map,
                weight_map=self.prev_weight_map,
                gt_rgb=gt_rgb_cpu,
            )
            final_metrics.update(
                {
                    'loss_total': loss_total_value,
                    'loss_hr_rgb': loss_hr_rgb_value,
                    'loss_lr_consistency': loss_lr_consistency_value,
                    'loss_scale_tail': float(loss_scale.item()),
                    'loss_opacity_sparse': float(loss_opacity.item()),
                    'loss_sampling_smooth': float(loss_sampling_smooth.item()),
                    'stage3sr_supervision_mode': 'full_frame_hr',
                    'psnr_hr': self._compute_psnr(pred_rgb_hr_cpu, reference_rgb_cpu),
                    'sharpness_hr': self._compute_sharpness(pred_rgb_hr_cpu),
                    'residual_mean_hr': float(hr_residual_map.mean().item()),
                    'weight_mean_hr': float(hr_weight_map.mean().item()),
                    'psnr_native_render': native_metrics['psnr'],
                    'sharpness_native_render': native_metrics['sharpness'],
                    'residual_mean_native_render': native_metrics.get('residual_mean', 0.0),
                }
            )
            if geometry_stage:
                final_metrics.update(
                    {
                        'loss_means_anchor': float(loss_means_anchor.item()),
                        'loss_rotation_reg': float(loss_rotation_reg.item()),
                        'iters_budget': total_iters,
                        'lambda_means_anchor_active': lambda_means_anchor,
                        'lambda_rotation_reg_active': lambda_rotation_reg,
                        'means_delta_cap_active': float(geometry_hparams['means_delta_cap']),
                    }
                )
            if use_depth_anchor:
                final_metrics.update(
                    self._build_depth_anchor_metrics(
                        stage_name=stage_name,
                        loss_value=float(loss_depth_anchor.item()),
                        valid_ratio=depth_anchor_valid_ratio,
                        skip_reason=depth_anchor_skip_reason,
                    )
                )

            self._log_and_maybe_save(stage_name, final_metrics, lr_residual_map, self.prev_weight_map, iter_idx=iter_idx)

            stage_history = self.diagnostics.stage_history.get(stage_name, [])
            if self.controller.should_stop_stage(stage_name, stage_history):
                break

        self._safe_export_ply(export_file_name)
        if geometry_stage:
            self.diagnostics_state['stage3b_completed'] = True
        else:
            self.diagnostics_state['stage3sr_completed'] = True
            self.diagnostics_state['stage3sr_supervision_mode'] = 'full_frame_hr'
        self.diagnostics_state['phase_reached'] = stage_name
        self.diagnostics_state['global_shift_detected'] = False
        self._update_stage2b_diagnostics(final_metrics)
        return final_metrics

    def _run_stage3sr_full_frame_hr(
        self,
        *,
        stage_name: str,
        export_file_name: str,
    ) -> dict[str, Any]:
        """运行 `Phase C` 风格的 full-frame HR supervision."""

        return self._run_reference_supervised_stage(
            stage_name=stage_name,
            export_file_name=export_file_name,
            geometry_stage=False,
        )

    def _summarize_prediction(
        self,
        pred_rgb: torch.Tensor,
        residual_map: torch.Tensor | None = None,
        weight_map: torch.Tensor | None = None,
        gt_rgb: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """汇总当前预测对应的指标."""

        target_rgb = self._get_gt_images(device=pred_rgb.device, dtype=pred_rgb.dtype) if gt_rgb is None else gt_rgb
        if target_rgb.device != pred_rgb.device or target_rgb.dtype != pred_rgb.dtype:
            target_rgb = target_rgb.to(device=pred_rgb.device, dtype=pred_rgb.dtype)
        summary = {
            "psnr": self._compute_psnr(pred_rgb, target_rgb),
            "sharpness": self._compute_sharpness(pred_rgb),
        }
        summary.update(self.gaussians.summarize_gaussian_stats())
        if residual_map is not None:
            summary["residual_mean"] = float(residual_map.mean().item())
        if weight_map is not None:
            summary.update(self.weight_builder.summarize_weight_stats(weight_map))
        return summary

    def _summarize_reference_prediction(
        self,
        pred_rgb_hr: torch.Tensor,
        scene: SceneBundle | None = None,
    ) -> dict[str, Any]:
        """汇总 reference-space 的 HR 指标.

        这里故意不复用 native 命名.
        这样最终 diagnostics 能直接把 `native-space` 和 `hr-space` 区分开.
        """

        reference_rgb = self._get_reference_images(scene, device=pred_rgb_hr.device, dtype=pred_rgb_hr.dtype)
        hr_residual_map = self.weight_builder.build_residual_map(pred_rgb_hr, reference_rgb)
        return {
            "psnr_hr": self._compute_psnr(pred_rgb_hr, reference_rgb),
            "sharpness_hr": self._compute_sharpness(pred_rgb_hr),
            "residual_mean_hr": float(hr_residual_map.mean().item()),
        }

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

    def _slice_view_tensor(
        self,
        tensor: torch.Tensor | None,
        start_index: int,
        end_index: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        """切 view shard,并按需要对齐设备和 dtype."""

        if tensor is None:
            return None

        tensor_shard = tensor[:, start_index:end_index]
        target_device = tensor_shard.device if device is None else device
        target_dtype = tensor_shard.dtype if dtype is None else dtype
        if tensor_shard.device == target_device and tensor_shard.dtype == target_dtype:
            return tensor_shard
        return tensor_shard.to(device=target_device, dtype=target_dtype)

    def _run_appearance_stage_multi_device(
        self,
        *,
        stage_name: str,
        freeze_stage_name: str,
        include_patch_supervision: bool,
        allow_pruning: bool,
    ) -> dict[str, Any]:
        """运行真正的多卡 appearance 优化循环.

        核心策略:
        1. 每张卡只渲染并保留自己负责的 view shard 计算图
        2. residual / diagnostics 先 detach 后聚到 CPU
        3. 全局权重图在 CPU 上按完整 residual 统一构造
        4. 再把对应 shard 的权重切回各卡做 loss/backward
        """

        self.current_stage = stage_name
        self.gaussians.freeze_for_stage(freeze_stage_name)
        optimizer = self.gaussians.build_optimizer(freeze_stage_name, self.hparams)
        gt_rgb_cpu = self._get_gt_images().detach().cpu()
        total_views = self.scene.gt_images.shape[1]

        final_metrics: dict[str, Any] = {}
        for iter_idx in range(self.hparams.iters_stage2a):
            optimizer.zero_grad(set_to_none=True)
            shard_outputs = self._render_scene_shards(self.scene)
            if not shard_outputs:
                raise RuntimeError("multi-device appearance stage requires at least two render shards.")

            pred_rgb_cpu_shards: list[torch.Tensor] = []
            residual_cpu_shards: list[torch.Tensor] = []
            render_meta_cpu_shards: list[dict[str, Any] | None] = []
            shard_records: list[dict[str, Any]] = []

            for shard_payload in shard_outputs:
                shard_scene = shard_payload["scene"]
                shard_output = shard_payload["output"]
                pred_rgb_shard = shard_output["images_pred"]
                render_meta_shard = self._extract_render_meta(shard_output)
                residual_map_shard = self.weight_builder.build_residual_map(pred_rgb_shard, shard_scene.gt_images)

                pred_rgb_cpu_shards.append(pred_rgb_shard.detach().cpu())
                residual_cpu_shards.append(residual_map_shard.detach().cpu())
                render_meta_cpu_shards.append(
                    self._move_render_meta_to_device(
                        render_meta_shard,
                        torch.device("cpu"),
                        detach_tensors=True,
                    )
                )
                shard_records.append(
                    {
                        "start_index": shard_payload["start_index"],
                        "end_index": shard_payload["end_index"],
                        "scene": shard_scene,
                        "pred_rgb": pred_rgb_shard,
                        "pred_depth": shard_output.get("depths_pred"),
                        "residual_map": residual_map_shard,
                        "render_meta": render_meta_shard,
                    }
                )

            pred_rgb_cpu = torch.cat(pred_rgb_cpu_shards, dim=1)
            residual_map_cpu = torch.cat(residual_cpu_shards, dim=1)
            previous_weight_map = self.prev_weight_map.detach().cpu() if isinstance(self.prev_weight_map, torch.Tensor) else None
            weight_map_cpu = self.weight_builder.build_weight_map(residual_map_cpu, previous_weight_map).detach().cpu()
            self.prev_weight_map = weight_map_cpu
            self.latest_render_meta = self._merge_render_meta_shards(render_meta_cpu_shards)

            loss_rgb_value = 0.0
            loss_depth_anchor_value = 0.0
            loss_patch_rgb_value = 0.0
            loss_patch_perceptual_value = 0.0
            loss_sampling_smooth_value = 0.0
            depth_anchor_valid_ratio_value = 0.0
            depth_anchor_any_active = False
            depth_anchor_skip_reasons: set[str] = set()

            for shard_record in shard_records:
                start_index = int(shard_record["start_index"])
                end_index = int(shard_record["end_index"])
                shard_scene = shard_record["scene"]
                pred_rgb_shard = shard_record["pred_rgb"]
                residual_map_shard = shard_record["residual_map"]
                render_meta_shard = shard_record["render_meta"]
                pred_depth_shard = shard_record.get("pred_depth")
                shard_view_fraction = float(end_index - start_index) / float(total_views)

                weight_map_shard = self._slice_view_tensor(
                    weight_map_cpu,
                    start_index,
                    end_index,
                    device=pred_rgb_shard.device,
                    dtype=pred_rgb_shard.dtype,
                )
                if weight_map_shard is None:
                    raise RuntimeError("weight_map_shard should never be None during multi-device appearance stage.")

                loss_rgb = compute_weighted_rgb_loss(pred_rgb_shard, shard_scene.gt_images, weight_map_shard)
                shard_loss_total = shard_view_fraction * loss_rgb
                loss_rgb_value += float(loss_rgb.item()) * shard_view_fraction

                if self._stage_uses_depth_anchor(stage_name):
                    loss_depth_anchor, depth_valid_ratio, depth_skip_reason = self._compute_depth_anchor_loss_for_view_range(
                        pred_depth_shard,
                        start_index=start_index,
                        end_index=end_index,
                    )
                    shard_loss_total = shard_loss_total + shard_view_fraction * self.hparams.lambda_depth_anchor * loss_depth_anchor
                    loss_depth_anchor_value += float(loss_depth_anchor.item()) * shard_view_fraction
                    depth_anchor_valid_ratio_value += depth_valid_ratio * shard_view_fraction
                    if depth_skip_reason is None:
                        depth_anchor_any_active = True
                    else:
                        depth_anchor_skip_reasons.add(depth_skip_reason)

                if include_patch_supervision:
                    sr_selection_map_shard = self._slice_view_tensor(
                        self.sr_selection_map,
                        start_index,
                        end_index,
                        device=pred_rgb_shard.device,
                        dtype=pred_rgb_shard.dtype,
                    )
                    loss_patch_rgb, loss_patch_perceptual = self._compute_patch_losses(
                        residual_map_shard,
                        loss_rgb,
                        scene=shard_scene,
                        native_weight_map=weight_map_shard,
                        sr_selection_map=sr_selection_map_shard,
                    )
                    fidelity_score = self.gaussian_fidelity_score
                    if isinstance(fidelity_score, torch.Tensor):
                        fidelity_score = fidelity_score.to(device=pred_rgb_shard.device, dtype=pred_rgb_shard.dtype)
                    scales_tensor = self.gaussians.scales
                    if scales_tensor.device != pred_rgb_shard.device:
                        scales_tensor = scales_tensor.to(pred_rgb_shard.device)
                    loss_sampling_smooth = compute_sampling_smooth_loss(
                        scales=scales_tensor,
                        fidelity_score=fidelity_score,
                        render_meta=render_meta_shard,
                        radius_threshold=self.hparams.sampling_radius_threshold,
                    )
                    shard_loss_total = shard_loss_total + shard_view_fraction * (
                        self.hparams.lambda_patch_rgb * loss_patch_rgb
                        + self.hparams.lambda_patch_perceptual * loss_patch_perceptual
                        + self.hparams.lambda_sampling_smooth * loss_sampling_smooth
                    )
                    loss_patch_rgb_value += float(loss_patch_rgb.item()) * shard_view_fraction
                    loss_patch_perceptual_value += float(loss_patch_perceptual.item()) * shard_view_fraction
                    loss_sampling_smooth_value += float(loss_sampling_smooth.item()) * shard_view_fraction

                shard_loss_total.backward()

            loss_scale = compute_scale_tail_loss(self.gaussians.scales, self.hparams.scale_tail_threshold)
            loss_opacity = compute_opacity_sparse_loss(self.gaussians.opacity)
            common_loss = self.hparams.lambda_scale_tail * loss_scale + self.hparams.lambda_opacity_sparse * loss_opacity
            common_loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.gaussians.clamp_stage_constraints(freeze_stage_name, self.hparams)

            final_metrics = self._summarize_prediction(
                pred_rgb_cpu,
                residual_map=residual_map_cpu,
                weight_map=weight_map_cpu,
                gt_rgb=gt_rgb_cpu,
            )
            final_metrics.update(
                {
                    "loss_total": (
                        loss_rgb_value
                        + self.hparams.lambda_scale_tail * float(loss_scale.item())
                        + self.hparams.lambda_opacity_sparse * float(loss_opacity.item())
                        + self.hparams.lambda_depth_anchor * loss_depth_anchor_value
                        + self.hparams.lambda_patch_rgb * loss_patch_rgb_value
                        + self.hparams.lambda_patch_perceptual * loss_patch_perceptual_value
                        + self.hparams.lambda_sampling_smooth * loss_sampling_smooth_value
                    ),
                    "loss_rgb_weighted": loss_rgb_value,
                    "loss_scale_tail": float(loss_scale.item()),
                    "loss_opacity_sparse": float(loss_opacity.item()),
                    }
                )
            if self._stage_uses_depth_anchor(stage_name):
                depth_skip_reason = None if depth_anchor_any_active else ",".join(sorted(depth_anchor_skip_reasons)) or "reference_unavailable"
                final_metrics.update(
                    self._build_depth_anchor_metrics(
                        stage_name=stage_name,
                        loss_value=loss_depth_anchor_value,
                        valid_ratio=depth_anchor_valid_ratio_value,
                        skip_reason=depth_skip_reason,
                    )
                )
            if include_patch_supervision:
                final_metrics.update(
                    {
                        "loss_patch_rgb": loss_patch_rgb_value,
                        "loss_patch_perceptual": loss_patch_perceptual_value,
                        "loss_sampling_smooth": loss_sampling_smooth_value,
                        "sr_patch_sets_used": int(self.last_sr_patch_sets_used),
                    }
                )

            if allow_pruning and self.controller.should_prune_now(iter_idx + 1):
                prune_summary = self.gaussians.prune_low_opacity(
                    threshold=self.hparams.opacity_prune_threshold,
                    max_fraction=self.hparams.prune_max_fraction,
                    min_gaussians_to_keep=self.hparams.min_gaussians_to_keep,
                )
                self.diagnostics.write_prune_summary(iteration=iter_idx + 1, summary=prune_summary)
                final_metrics.update(self.gaussians.summarize_gaussian_stats())
                self.diagnostics_state.setdefault("prune_history", []).append(prune_summary)
                self.diagnostics_state["last_prune"] = prune_summary

                if prune_summary["pruned_count"] > 0:
                    optimizer = self.gaussians.build_optimizer(freeze_stage_name, self.hparams)

            self._log_and_maybe_save(stage_name, final_metrics, residual_map_cpu, weight_map_cpu, iter_idx=iter_idx)

            stage_history = self.diagnostics.stage_history.get(stage_name, [])
            if self.controller.should_stop_stage(stage_name, stage_history):
                break

        return final_metrics

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

        self._ensure_depth_anchor_reference(stage_name)
        if self._use_multi_device_render(self.scene):
            return self._run_appearance_stage_multi_device(
                stage_name=stage_name,
                freeze_stage_name=freeze_stage_name,
                include_patch_supervision=include_patch_supervision,
                allow_pruning=allow_pruning,
            )

        self.current_stage = stage_name
        self.gaussians.freeze_for_stage(freeze_stage_name)
        optimizer = self.gaussians.build_optimizer(freeze_stage_name, self.hparams)

        final_metrics: dict[str, Any] = {}
        for iter_idx in range(self.hparams.iters_stage2a):
            render_output = self.render_current_scene()
            pred_rgb = render_output["images_pred"]
            render_meta = self._extract_render_meta(render_output)
            self.latest_render_meta = render_meta
            gt_rgb = self._get_gt_images(device=pred_rgb.device, dtype=pred_rgb.dtype)
            residual_map = self.weight_builder.build_residual_map(pred_rgb, gt_rgb)
            previous_weight_map = self.prev_weight_map
            if isinstance(previous_weight_map, torch.Tensor):
                previous_weight_map = previous_weight_map.to(device=pred_rgb.device, dtype=pred_rgb.dtype)
            self.prev_weight_map = self.weight_builder.build_weight_map(residual_map, previous_weight_map)

            loss_rgb = compute_weighted_rgb_loss(pred_rgb, gt_rgb, self.prev_weight_map)
            loss_scale = compute_scale_tail_loss(self.gaussians.scales, self.hparams.scale_tail_threshold)
            loss_opacity = compute_opacity_sparse_loss(self.gaussians.opacity)
            if self._stage_uses_depth_anchor(stage_name):
                loss_depth_anchor, depth_anchor_valid_ratio, depth_anchor_skip_reason = self._compute_depth_anchor_loss_for_view_range(
                    render_output.get("depths_pred"),
                    start_index=0,
                    end_index=self.scene.gt_images.shape[1],
                )
            else:
                loss_depth_anchor = torch.zeros((), dtype=loss_rgb.dtype, device=loss_rgb.device)
                depth_anchor_valid_ratio = 0.0
                depth_anchor_skip_reason = None
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

            loss_total = (
                loss_rgb
                + self.hparams.lambda_scale_tail * loss_scale
                + self.hparams.lambda_opacity_sparse * loss_opacity
                + self.hparams.lambda_depth_anchor * loss_depth_anchor
            )
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

            final_metrics = self._summarize_prediction(
                pred_rgb,
                residual_map=residual_map,
                weight_map=self.prev_weight_map,
                gt_rgb=gt_rgb,
            )
            final_metrics.update(
                {
                    "loss_total": float(loss_total.item()),
                    "loss_rgb_weighted": float(loss_rgb.item()),
                    "loss_scale_tail": float(loss_scale.item()),
                    "loss_opacity_sparse": float(loss_opacity.item()),
                }
            )
            if self._stage_uses_depth_anchor(stage_name):
                final_metrics.update(
                    self._build_depth_anchor_metrics(
                        stage_name=stage_name,
                        loss_value=float(loss_depth_anchor.item()),
                        valid_ratio=depth_anchor_valid_ratio,
                        skip_reason=depth_anchor_skip_reason,
                    )
                )
            if include_patch_supervision:
                final_metrics.update(
                    {
                        "loss_patch_rgb": float(loss_patch_rgb.item()),
                        "loss_patch_perceptual": float(loss_patch_perceptual.item()),
                        "loss_sampling_smooth": float(loss_sampling_smooth.item()),
                        "sr_patch_sets_used": int(self.last_sr_patch_sets_used),
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
        render_output = self._render_scene_for_evaluation(self.scene)
        pred_rgb = render_output["images_pred"]
        gt_rgb = self._get_gt_images(device=pred_rgb.device, dtype=pred_rgb.dtype)
        residual_map = self.weight_builder.build_residual_map(pred_rgb, gt_rgb)
        render_meta = self._extract_render_meta(render_output)

        fidelity_diagnostics = self.weight_builder.compute_gaussian_fidelity_diagnostics(render_meta)
        if fidelity_diagnostics is None:
            fidelity_score = self._build_default_fidelity_score()
            self.diagnostics_state.setdefault("warnings", []).append("phase3s_missing_render_meta")
            self.gaussian_fidelity_diagnostics = None
        else:
            fidelity_score = fidelity_diagnostics["fidelity_score"]
            self.gaussian_fidelity_diagnostics = {
                key: value.detach().cpu()
                for key, value in fidelity_diagnostics.items()
                if isinstance(value, torch.Tensor)
            }

        sr_selection_map = self.weight_builder.build_sr_selection_weight(
            render_meta=render_meta,
            fidelity_score=fidelity_score,
            fidelity_diagnostics=fidelity_diagnostics,
            native_hw=self.scene.gt_images.shape[-2:],
            output_hw=self._get_reference_images().shape[-2:],
        )
        if sr_selection_map is None:
            sr_selection_map = self._build_default_sr_selection_map(residual_map)
            self.diagnostics_state.setdefault("warnings", []).append("phase3s_missing_sr_selection_meta")
        self.latest_render_meta = render_meta
        self.gaussian_fidelity_score = fidelity_score.detach().cpu()
        self.sr_selection_map = sr_selection_map.detach().cpu()

        metrics = self._summarize_prediction(pred_rgb, residual_map=residual_map, weight_map=self.prev_weight_map, gt_rgb=gt_rgb)
        metrics.update(self.weight_builder.summarize_fidelity_stats(fidelity_score))
        metrics.update(self.weight_builder.summarize_fidelity_diagnostics(fidelity_diagnostics))
        metrics["sr_selection_mean"] = float(sr_selection_map.mean().item())
        self._log_and_maybe_save("phase3s", metrics, residual_map, self.prev_weight_map, iter_idx=0)
        self._write_phase3s_artifacts(
            "phase3s",
            fidelity_score,
            sr_selection_map,
            fidelity_diagnostics=fidelity_diagnostics,
        )

        self.diagnostics_state["phase3s_completed"] = True
        self.diagnostics_state["phase_reached"] = "phase3s"
        return metrics

    def run_stage3sr_selective_patch(
        self,
        *,
        stage_name: str = "stage3sr",
        export_file_name: str = "gaussians_stage3sr.ply",
    ) -> dict[str, Any]:
        """运行 Stage 3SR.

        当前这里会在 patch 模式和 `Phase C` full-frame HR 模式之间择一.
        """

        supervision_mode = self._resolve_stage3sr_supervision_mode()
        if supervision_mode == "none":
            raise RuntimeError("Stage 3SR requires either patch supervision or full-frame HR supervision.")
        if not self.diagnostics_state.get("phase3s_completed", False):
            self.run_phase3s_build_sr_selection()

        if supervision_mode == "full_frame_hr":
            return self._run_stage3sr_full_frame_hr(
                stage_name=stage_name,
                export_file_name=export_file_name,
            )

        final_metrics = self._run_appearance_stage(
            stage_name=stage_name,
            freeze_stage_name="stage3sr",
            include_patch_supervision=True,
            allow_pruning=False,
        )
        self._safe_export_ply(export_file_name)
        self.diagnostics_state["stage3sr_completed"] = True
        self.diagnostics_state["stage3sr_supervision_mode"] = "patch"
        self.diagnostics_state["phase_reached"] = stage_name
        self.diagnostics_state["global_shift_detected"] = False
        self._update_stage2b_diagnostics(final_metrics)
        return final_metrics

    def _bootstrap_geometry_from_current_gaussians(self, *, warm_start_key: str) -> dict[str, Any]:
        """把当前输入高斯视为 geometry continuation 的 warm start.

        这个入口同时服务:
        - `start_stage=stage2b`
        - `start_stage=stage3b`

        两者的共同需求都是:
        1. 不再重复跑前面的 appearance optimizer
        2. 先把当前高斯重新渲染一遍
        3. 重建后续 geometry 阶段需要的权重图与 diagnostics
        """

        self.current_stage = "stage2a"
        render_output = self._render_scene_for_evaluation(self.scene)
        pred_rgb = render_output["images_pred"]
        gt_rgb = self._get_gt_images(device=pred_rgb.device, dtype=pred_rgb.dtype)
        residual_map = self.weight_builder.build_residual_map(pred_rgb, gt_rgb)
        prev_weight_map = self.prev_weight_map
        if isinstance(prev_weight_map, torch.Tensor):
            prev_weight_map = prev_weight_map.to(device=pred_rgb.device, dtype=pred_rgb.dtype)
        self.prev_weight_map = self.weight_builder.build_weight_map(residual_map, prev_weight_map).detach().cpu()

        loss_rgb = compute_weighted_rgb_loss(pred_rgb, gt_rgb, self.prev_weight_map)
        loss_scale = compute_scale_tail_loss(self.gaussians.scales, self.hparams.scale_tail_threshold)
        loss_opacity = compute_opacity_sparse_loss(self.gaussians.opacity)
        loss_patch_rgb, loss_patch_perceptual = self._compute_patch_losses(residual_map, loss_rgb)
        metrics = self._summarize_prediction(pred_rgb, residual_map=residual_map, weight_map=self.prev_weight_map, gt_rgb=gt_rgb)
        metrics.update(
            {
                "loss_total": float(loss_rgb.item()),
                "loss_rgb_weighted": float(loss_rgb.item()),
                "loss_scale_tail": float(loss_scale.item()),
                "loss_opacity_sparse": float(loss_opacity.item()),
                "loss_patch_rgb": float(loss_patch_rgb.item()),
                "loss_patch_perceptual": float(loss_patch_perceptual.item()),
                "sr_patch_sets_used": int(self.last_sr_patch_sets_used),
            }
        )

        self.diagnostics_state["phase_reached"] = "stage2a"
        self.diagnostics_state["stage3a_completed"] = True
        self.diagnostics_state[warm_start_key] = True
        self.diagnostics_state["stage2a_bootstrap"] = metrics
        self._update_stage2b_diagnostics(metrics)
        return metrics

    def bootstrap_stage2b_from_current_gaussians(self) -> dict[str, Any]:
        """把当前输入高斯视为“已完成 Stage 2A”的 warm start."""

        return self._bootstrap_geometry_from_current_gaussians(warm_start_key="warm_start_stage2b")

    def bootstrap_stage3b_from_current_gaussians(self) -> dict[str, Any]:
        """把当前输入高斯视为“已完成 Stage 3SR”的 warm start.

        这条路径主要用于 continuation / resume:
        - 输入已经是 `gaussians_stage3sr.ply` 或 `latest.pt` 中的 `stage3sr` 末态
        - 只想继续接一段 `stage3b`
        """

        metrics = self._bootstrap_geometry_from_current_gaussians(warm_start_key="warm_start_stage3b")
        self.diagnostics_state["stage3sr_enabled"] = True
        self.diagnostics_state["stage3sr_completed"] = True
        self.diagnostics_state["stage3sr_supervision_mode"] = "full_frame_hr"
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
        self._export_rgb_artifacts("gt_reference", self._get_gt_images().detach().cpu())

        # 只有 reference 分辨率真的高于 native 时, 才补 `Phase D` 的 HR 产物.
        # 这样不会把 native-reference 场景的人为塞进一套重复导出.
        if not self._reference_space_enabled():
            return

        baseline_hr_render_output = self._render_reference_scene_for_evaluation()
        baseline_pred_rgb_hr = baseline_hr_render_output["images_pred"]
        self._export_rgb_artifacts("baseline_render_hr", baseline_pred_rgb_hr)
        self._export_rgb_artifacts("gt_reference_hr", self._get_reference_images(device=torch.device("cpu")))
        self.diagnostics_state["baseline_hr"] = self._summarize_reference_prediction(baseline_pred_rgb_hr)

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
        render_output = self._render_scene_for_evaluation(self.scene)
        pred_rgb = render_output["images_pred"]
        gt_rgb = self._get_gt_images(device=pred_rgb.device, dtype=pred_rgb.dtype)
        residual_map = self.weight_builder.build_residual_map(pred_rgb, gt_rgb)
        baseline = self._summarize_prediction(pred_rgb, residual_map=residual_map, gt_rgb=gt_rgb)
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
        render_output = self._render_scene_for_evaluation(self.scene)
        pred_rgb = render_output["images_pred"]
        gt_rgb = self._get_gt_images(device=pred_rgb.device, dtype=pred_rgb.dtype)
        residual_map = self.weight_builder.build_residual_map(pred_rgb, gt_rgb)
        prev_weight_map = self.prev_weight_map
        if isinstance(prev_weight_map, torch.Tensor):
            prev_weight_map = prev_weight_map.to(device=pred_rgb.device, dtype=pred_rgb.dtype)
        self.prev_weight_map = self.weight_builder.build_weight_map(residual_map, prev_weight_map).detach().cpu()

        metrics = self._summarize_prediction(pred_rgb, residual_map=residual_map, weight_map=self.prev_weight_map, gt_rgb=gt_rgb)
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
        3. Stage 3SR supervision
        如果当前没有启用任何 Stage 3SR supervision,则只执行 native cleanup.
        """

        resolved_mode = self._resolve_stage2a_mode()
        self.diagnostics_state["stage2a_mode_resolved"] = resolved_mode
        self.diagnostics_state["stage3sr_enabled"] = resolved_mode == "enhanced"
        final_metrics = self.run_stage3a_native_cleanup(
            stage_name="stage2a",
            export_file_name="gaussians_stage2a.ply",
        )
        if resolved_mode == "legacy":
            if self.run_config.stage2a_mode == "legacy" and self._stage3sr_supervision_configured():
                self.diagnostics_state.setdefault("warnings", []).append("stage2a_mode_legacy_skipped_stage3sr_supervision")
            self._update_stage2b_diagnostics(final_metrics)
            return final_metrics

        self.run_phase3s_build_sr_selection()
        return self.run_stage3sr_selective_patch()

    def run_stage2b(self) -> dict[str, Any]:
        """运行 limited geometry refinement."""

        self.current_stage = "stage2b"
        self.gaussians.freeze_for_stage("stage2b")
        optimizer = self.gaussians.build_optimizer("stage2b", self.hparams)
        geometry_hparams = self._resolve_geometry_stage_hparams("stage2b")
        lambda_means_anchor = float(geometry_hparams["lambda_means_anchor"])
        lambda_rotation_reg = float(geometry_hparams["lambda_rotation_reg"])

        final_metrics: dict[str, Any] = {}
        for iter_idx in range(int(geometry_hparams["iters"])):
            render_output = self.render_current_scene()
            pred_rgb = render_output["images_pred"]
            gt_rgb = self._get_gt_images(device=pred_rgb.device, dtype=pred_rgb.dtype)
            residual_map = self.weight_builder.build_residual_map(pred_rgb, gt_rgb)
            previous_weight_map = self.prev_weight_map
            if isinstance(previous_weight_map, torch.Tensor):
                previous_weight_map = previous_weight_map.to(device=pred_rgb.device, dtype=pred_rgb.dtype)
            self.prev_weight_map = self.weight_builder.build_weight_map(residual_map, previous_weight_map)

            loss_rgb = compute_weighted_rgb_loss(pred_rgb, gt_rgb, self.prev_weight_map)
            loss_scale = compute_scale_tail_loss(self.gaussians.scales, self.hparams.scale_tail_threshold)
            loss_opacity = compute_opacity_sparse_loss(self.gaussians.opacity)
            loss_patch_rgb, loss_patch_perceptual = self._compute_patch_losses(residual_map, loss_rgb)
            loss_means_anchor, loss_rotation_reg = compute_stage3b_losses(
                self.gaussians.means,
                self.gaussians.initial_means,
                self.gaussians.rotations,
                self.gaussians.initial_rotations,
            )
            loss_total = loss_rgb + self.hparams.lambda_scale_tail * loss_scale + self.hparams.lambda_opacity_sparse * loss_opacity
            loss_total = (
                loss_total
                + self.hparams.lambda_patch_rgb * loss_patch_rgb
                + self.hparams.lambda_patch_perceptual * loss_patch_perceptual
                + lambda_means_anchor * loss_means_anchor
                + lambda_rotation_reg * loss_rotation_reg
            )

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.gaussians.clamp_stage_constraints("stage2b", self.hparams)

            final_metrics = self._summarize_prediction(
                pred_rgb,
                residual_map=residual_map,
                weight_map=self.prev_weight_map,
                gt_rgb=gt_rgb,
            )
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
                    "iters_budget": int(geometry_hparams["iters"]),
                    "lambda_means_anchor_active": lambda_means_anchor,
                    "lambda_rotation_reg_active": lambda_rotation_reg,
                    "means_delta_cap_active": float(geometry_hparams["means_delta_cap"]),
                    "sr_patch_sets_used": int(self.last_sr_patch_sets_used),
                }
            )
            self._log_and_maybe_save("stage2b", final_metrics, residual_map, self.prev_weight_map, iter_idx=iter_idx)

        self._safe_export_ply("gaussians_stage2b.ply")
        self.diagnostics_state["phase_reached"] = "stage2b"
        self.diagnostics_state["global_shift_detected"] = final_metrics.get("residual_mean", 0.0) > 0.08
        self.diagnostics_state["local_overlap_persistent"] = final_metrics.get("residual_mean", 0.0) > 0.03
        return final_metrics

    def run_stage3b(self) -> dict[str, Any]:
        """运行 `Phase E` 的 SR-driven limited geometry.

        当前最小版只支持建立在 `Phase C` full-frame HR supervision 之上的 geometry release.
        这样实现最直接, 也最贴近“让 SR 真正影响结构”这个目标.
        """

        if self._resolve_stage3sr_supervision_mode() != "full_frame_hr":
            raise RuntimeError("stage3b currently requires full-frame HR supervision.")

        return self._run_reference_supervised_stage(
            stage_name="stage3b",
            export_file_name="gaussians_stage3b.ply",
            geometry_stage=True,
        )

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
            gt_rgb = self._get_gt_images(device=pred_rgb.device, dtype=pred_rgb.dtype)
            residual_map = self.weight_builder.build_residual_map(pred_rgb, gt_rgb)
            previous_weight_map = self.prev_weight_map
            if isinstance(previous_weight_map, torch.Tensor):
                previous_weight_map = previous_weight_map.to(device=pred_rgb.device, dtype=pred_rgb.dtype)
            self.prev_weight_map = self.weight_builder.build_weight_map(residual_map, previous_weight_map)

            loss_rgb = compute_weighted_rgb_loss(pred_rgb, gt_rgb, self.prev_weight_map)
            loss_scale = compute_scale_tail_loss(self.gaussians.scales, self.hparams.scale_tail_threshold)
            loss_opacity = compute_opacity_sparse_loss(self.gaussians.opacity)
            loss_total = loss_rgb + self.hparams.lambda_scale_tail * loss_scale + self.hparams.lambda_opacity_sparse * loss_opacity

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.gaussians.clamp_stage_constraints("phase4", self.hparams)

            final_metrics = self._summarize_prediction(
                pred_rgb,
                residual_map=residual_map,
                weight_map=self.prev_weight_map,
                gt_rgb=gt_rgb,
            )
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
        *,
        final_hr_metrics: dict[str, Any] | None = None,
        override_stop_reason: str | None = None,
    ) -> dict[str, Any]:
        """构建最终 diagnostics 摘要."""

        baseline = self.diagnostics_state.get("baseline", {})
        baseline_hr = self.diagnostics_state.get("baseline_hr")
        summary = {
            "scene_id": self.scene.scene_index,
            "view_id": self.scene.view_id,
            "view_ids": list(self.scene.view_ids) if self.scene.view_ids is not None else None,
            "start_stage": self.run_config.start_stage,
            "native_hw": list(self.scene.native_hw),
            "reference_hw": list(self.scene.reference_hw) if self.scene.reference_hw is not None else None,
            "phase_reached": self.diagnostics_state.get("phase_reached", self.current_stage),
            "stopped_reason": override_stop_reason or self.controller.summarize_stop_reason(self.diagnostics_state),
            "used_pose_refinement": self.diagnostics_state.get("used_pose_refinement", False),
            "used_joint_fallback": self.diagnostics_state.get("used_joint_fallback", False),
            "warm_start_stage2b": self.diagnostics_state.get("warm_start_stage2b", False),
            "warm_start_stage3b": self.diagnostics_state.get("warm_start_stage3b", False),
            "baseline": baseline,
            "final": final_metrics,
            "artifacts": dict(self.visual_artifacts),
            "depth_anchor": {
                "enabled": self.diagnostics_state.get("depth_anchor_enabled", False),
                "weight": self.diagnostics_state.get("depth_anchor_weight", 0.0),
                "source": self.diagnostics_state.get("depth_anchor_source"),
                "reference_ready": self.diagnostics_state.get("depth_anchor_reference_ready", False),
                "reference_valid_ratio": self.diagnostics_state.get("depth_anchor_reference_valid_ratio", 0.0),
                "reference_skip_reason": self.diagnostics_state.get("depth_anchor_reference_skip_reason"),
                "last_skip_reason": self.diagnostics_state.get("depth_anchor_last_skip_reason"),
                "last_valid_ratio": self.diagnostics_state.get("depth_anchor_last_valid_ratio", 0.0),
                "last_loss": self.diagnostics_state.get("depth_anchor_last_loss", 0.0),
            },
            "deltas": {
                "psnr_gain": float(final_metrics.get("psnr", 0.0) - baseline.get("psnr", 0.0)),
                "sharpness_gain": float(final_metrics.get("sharpness", 0.0) - baseline.get("sharpness", 0.0)),
                "scale_tail_drop": float(baseline.get("scale_tail_ratio", 0.0) - final_metrics.get("scale_tail_ratio", 0.0)),
            },
        }
        if isinstance(baseline_hr, dict):
            summary["baseline_hr"] = baseline_hr
        if isinstance(final_hr_metrics, dict):
            summary["final_hr"] = final_hr_metrics
        if isinstance(baseline_hr, dict) and isinstance(final_hr_metrics, dict):
            summary["deltas"].update(
                {
                    "psnr_gain_hr": float(final_hr_metrics.get("psnr_hr", 0.0) - baseline_hr.get("psnr_hr", 0.0)),
                    "sharpness_gain_hr": float(
                        final_hr_metrics.get("sharpness_hr", 0.0) - baseline_hr.get("sharpness_hr", 0.0)
                    ),
                    "residual_mean_hr_drop": float(
                        baseline_hr.get("residual_mean_hr", 0.0) - final_hr_metrics.get("residual_mean_hr", 0.0)
                    ),
                }
            )
        return summary

    def export_final_outputs(
        self,
        final_metrics: dict[str, Any],
        override_stop_reason: str | None = None,
    ) -> dict[str, Any]:
        """保存最终高斯和 diagnostics 摘要."""

        # 结束前再次渲染当前高斯.
        # 这样无论 stop 在哪一阶段,都能得到统一命名的 after 视频.
        final_render_output = self._render_scene_for_evaluation(self.scene)
        final_pred_rgb = final_render_output["images_pred"]
        self._export_rgb_artifacts("final_render", final_pred_rgb)
        final_hr_metrics: dict[str, Any] | None = None

        # `Phase D` 的最小交付是:
        # native after 继续保留
        # reference-space after 额外补出
        if self._reference_space_enabled():
            final_hr_render_output = self._render_reference_scene_for_evaluation()
            final_pred_rgb_hr = final_hr_render_output["images_pred"]
            self._export_rgb_artifacts("final_render_hr", final_pred_rgb_hr)
            final_hr_metrics = self._summarize_reference_prediction(final_pred_rgb_hr)

        save_state(
            self.diagnostics.state_dir,
            stage_name=self.current_stage,
            iter_idx=0,
            gaussians=self.gaussians,
            diagnostics_state=self.diagnostics_state,
            pose_delta=self.pose_delta,
        )
        summary = self._build_final_summary(
            final_metrics,
            final_hr_metrics=final_hr_metrics,
            override_stop_reason=override_stop_reason,
        )
        self.diagnostics.finalize(summary)
        return summary

    def run(self) -> dict[str, Any]:
        """运行完整 refinement 流程."""

        explicit_stage2b_start = self.run_config.start_stage == "stage2b"
        explicit_stage3b_start = self.run_config.start_stage == "stage3b"
        if explicit_stage2b_start and not self.run_config.enable_stage2b:
            raise RuntimeError("start_stage=stage2b requires enable_stage2b=True.")
        if explicit_stage3b_start and not self.run_config.enable_stage3b:
            raise RuntimeError("start_stage=stage3b requires enable_stage3b=True.")

        final_metrics = self.run_phase0()
        if self.run_config.stop_after == "phase0":
            return self.export_final_outputs(final_metrics)

        final_metrics = self.run_phase1_prepare_weights()
        if self.run_config.stop_after == "phase1":
            return self.export_final_outputs(final_metrics)

        if explicit_stage2b_start:
            final_metrics = self.bootstrap_stage2b_from_current_gaussians()
        elif explicit_stage3b_start:
            final_metrics = self.bootstrap_stage3b_from_current_gaussians()
        else:
            final_metrics = self.run_stage2a()

        if self.run_config.stop_after == "stage2a":
            return self.export_final_outputs(final_metrics)

        if explicit_stage3b_start:
            self.run_phase3s_build_sr_selection()
            final_metrics = self.run_stage3b()
            if self.run_config.stop_after == "stage3b":
                return self.export_final_outputs(final_metrics)

        stage3b_ran = False
        if not explicit_stage3b_start and self.controller.should_enter_stage3b(self.diagnostics_state):
            final_metrics = self.run_stage3b()
            stage3b_ran = True
            if self.run_config.stop_after == "stage3b":
                return self.export_final_outputs(final_metrics)

        # 显式 `start_stage=stage2b` 表示用户已经决定继续几何阶段.
        # 这种情况下不应再被自动 gate 拦住.
        if not explicit_stage3b_start and not stage3b_ran and (
            explicit_stage2b_start or self.controller.should_enter_stage2b(self.diagnostics_state)
        ):
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
