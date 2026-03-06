"""数据读取与标准化逻辑.

这里不直接参与优化.
它只负责把 provider 输出收束成 refinement 层统一消费的 `SceneBundle`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .config import RefinementRunConfig


@dataclass
class SceneBundle:
    """refinement 侧统一使用的场景数据结构."""

    gt_images: torch.Tensor
    cam_view: torch.Tensor
    intrinsics: torch.Tensor
    frame_indices: list[int]
    scene_index: int
    view_id: str | None
    target_index: torch.Tensor | None = None
    file_name: str | None = None
    reference_images: torch.Tensor | None = None
    intrinsics_ref: torch.Tensor | None = None
    native_hw: tuple[int, int] | None = None
    reference_hw: tuple[int, int] | None = None
    reference_mode: str = "native"
    sr_scale: float = 1.0


def _ensure_expected_ndim(tensor: torch.Tensor, expected_ndim: int) -> torch.Tensor:
    """按期望维度补齐 batch 维.

    这里只允许缺少一层 batch 维.
    如果维度差异超过 1,说明上游数据形状已经和约定不一致.
    """

    if tensor.ndim == expected_ndim:
        return tensor
    if tensor.ndim == expected_ndim - 1:
        return tensor.unsqueeze(0)
    raise ValueError(f"Expected tensor with {expected_ndim - 1} or {expected_ndim} dims, got {tensor.ndim}.")


def _normalize_frame_indices(
    num_frames: int,
    frame_indices: list[int] | None,
    target_subsample: int,
) -> list[int]:
    """把外部索引参数规范成最终使用的帧列表."""

    if frame_indices is not None:
        normalized = sorted({int(index) for index in frame_indices})
        for index in normalized:
            if index < 0 or index >= num_frames:
                raise IndexError(f"Frame index {index} is out of range for {num_frames} frames.")
        return normalized

    if target_subsample <= 0:
        raise ValueError("target_subsample must be a positive integer.")

    return list(range(0, num_frames, target_subsample))


def _slice_view_tensor(tensor: torch.Tensor | None, frame_indices: list[int]) -> torch.Tensor | None:
    """按视角/时间维裁剪 tensor."""

    if tensor is None:
        return None

    index_tensor = torch.tensor(frame_indices, dtype=torch.long, device=tensor.device)
    return torch.index_select(tensor, dim=1, index=index_tensor)


def _normalize_reference_mode(reference_mode: str) -> str:
    """校验并规范 reference mode."""

    normalized = str(reference_mode).strip().lower()
    if normalized not in {"native", "super_resolved"}:
        raise ValueError(f"Unsupported reference_mode: {reference_mode}")
    return normalized


def _normalize_sr_scale(reference_mode: str, sr_scale: float) -> int:
    """把 SR scale 规范成整数倍率."""

    if reference_mode == "native":
        return 1

    scale_value = float(sr_scale)
    scale_int = int(round(scale_value))
    if scale_int <= 0 or abs(scale_value - scale_int) > 1e-6:
        raise ValueError(f"super_resolved reference requires an integer sr_scale, got {sr_scale}.")
    return scale_int


def _build_reference_supervision(
    gt_images: torch.Tensor,
    intrinsics: torch.Tensor,
    reference_mode: str,
    sr_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], tuple[int, int], float]:
    """构造参考监督图像与对应 intrinsics."""

    normalized_mode = _normalize_reference_mode(reference_mode)
    native_hw = (int(gt_images.shape[-2]), int(gt_images.shape[-1]))
    scale_int = _normalize_sr_scale(normalized_mode, sr_scale)

    if normalized_mode == "native":
        return gt_images.clone(), intrinsics.clone(), native_hw, native_hw, 1.0

    batch_size, num_views, num_channels, height, width = gt_images.shape
    resized = F.interpolate(
        gt_images.reshape(batch_size * num_views, num_channels, height, width),
        size=(height * scale_int, width * scale_int),
        mode="bilinear",
        align_corners=False,
    )
    reference_images = resized.reshape(batch_size, num_views, num_channels, height * scale_int, width * scale_int)
    intrinsics_ref = intrinsics * float(scale_int)
    reference_hw = (int(reference_images.shape[-2]), int(reference_images.shape[-1]))
    return reference_images, intrinsics_ref, native_hw, reference_hw, float(scale_int)


def _resolve_config_source_path(base_config_path: Path, raw_path: str | Path) -> Path:
    """把配置里声明的路径解析成稳定的 `Path`.

    优先保留仓库根目录下已经存在的相对路径.
    如果找不到,再回退到“相对当前配置文件目录”的解析方式.
    """

    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path

    nested_path = base_config_path.parent / path
    if nested_path.exists():
        return nested_path
    return path


def _resolve_scene_config_paths(base_config_path: Path, root_config: dict[str, Any]) -> list[Path]:
    """解析 refinement 入口配置真正依赖的训练配置链."""

    nested_config_paths = root_config.get("config_path")
    if nested_config_paths is None:
        return [base_config_path]

    if isinstance(nested_config_paths, (str, Path)):
        return [_resolve_config_source_path(base_config_path, nested_config_paths)]

    if isinstance(nested_config_paths, (list, tuple)):
        return [_resolve_config_source_path(base_config_path, item) for item in nested_config_paths]

    raise TypeError("config_path must be a string or a list of strings.")


def _build_scene_loader_overrides(
    root_config: dict[str, Any],
    run_config: RefinementRunConfig,
) -> dict[str, Any]:
    """把 demo/refinement 入口层的覆盖项压成 dataloader 配置."""

    overrides: dict[str, Any] = {
        "batch_size": 1,
        "num_workers": 0,
        "num_train_images": 1,
        # refinement 只需要真实 RGB 和相机参数.
        # 这里主动关掉深度与 latent 读取,避免本地 demo 资产不完整时
        # dataloader 因为无关文件缺失而无限重试.
        "use_depth": False,
        "load_latents": False,
        # provider 的推理路径会直接访问这个键.
        # refinement 默认不手动指定 target frame,因此补成 None.
        "target_index_manual": None,
    }

    dataset_name = run_config.dataset_name or root_config.get("dataset_name")
    if dataset_name is not None:
        overrides["data_mode"] = [[str(dataset_name), 1]]

    if root_config.get("set_manual_time_idx") is not None:
        overrides["set_manual_time_idx"] = root_config.get("set_manual_time_idx")

    if root_config.get("target_index_subsample") is not None:
        overrides["target_index_subsample"] = int(root_config.get("target_index_subsample"))

    selected_view_ids = None
    if run_config.view_id is not None:
        selected_view_ids = [str(run_config.view_id)]
    elif root_config.get("static_view_indices_fixed") is not None:
        selected_view_ids = [str(item) for item in root_config.get("static_view_indices_fixed")]

    if selected_view_ids is not None:
        overrides["static_view_indices_fixed"] = selected_view_ids
        overrides["static_view_indices_sampling"] = "fixed"
        overrides["num_input_multi_views"] = len(selected_view_ids)

    if root_config.get("num_test_images") is not None:
        overrides["num_test_images"] = int(root_config.get("num_test_images"))
    else:
        overrides["num_test_images"] = run_config.scene_index + 1

    return overrides


def standardize_batch(
    batch: dict[str, Any],
    scene_index: int,
    view_id: str | None = None,
    frame_indices: list[int] | None = None,
    target_subsample: int = 1,
    reference_mode: str = "native",
    sr_scale: float = 1.0,
) -> SceneBundle:
    """把 provider batch 变成 refinement 使用的 `SceneBundle`."""

    required_keys = {"images_output", "cam_view", "intrinsics"}
    missing_keys = required_keys - set(batch.keys())
    if missing_keys:
        raise KeyError(f"Batch is missing required keys: {sorted(missing_keys)}")

    gt_images = _ensure_expected_ndim(batch["images_output"], expected_ndim=5).detach().clone()
    cam_view = _ensure_expected_ndim(batch["cam_view"], expected_ndim=4).detach().clone()
    intrinsics = _ensure_expected_ndim(batch["intrinsics"], expected_ndim=3).detach().clone()

    num_frames = gt_images.shape[1]
    selected_frame_indices = _normalize_frame_indices(num_frames, frame_indices, target_subsample)

    gt_images = _slice_view_tensor(gt_images, selected_frame_indices)
    cam_view = _slice_view_tensor(cam_view, selected_frame_indices)
    intrinsics = _slice_view_tensor(intrinsics, selected_frame_indices)
    reference_images, intrinsics_ref, native_hw, reference_hw, normalized_sr_scale = _build_reference_supervision(
        gt_images=gt_images,
        intrinsics=intrinsics,
        reference_mode=reference_mode,
        sr_scale=sr_scale,
    )

    target_index = batch.get("target_index")
    if isinstance(target_index, torch.Tensor):
        target_index = _ensure_expected_ndim(target_index, expected_ndim=2)
        target_index = _slice_view_tensor(target_index, selected_frame_indices)

    file_name = batch.get("file_name")
    if isinstance(file_name, (list, tuple)) and file_name:
        file_name = str(file_name[0])
    elif file_name is not None:
        file_name = str(file_name)

    return SceneBundle(
        gt_images=gt_images,
        cam_view=cam_view,
        intrinsics=intrinsics,
        frame_indices=selected_frame_indices,
        scene_index=scene_index,
        view_id=view_id,
        target_index=target_index,
        file_name=file_name,
        reference_images=reference_images,
        intrinsics_ref=intrinsics_ref,
        native_hw=native_hw,
        reference_hw=reference_hw,
        reference_mode=_normalize_reference_mode(reference_mode),
        sr_scale=normalized_sr_scale,
    )


def _load_scene_config(run_config: RefinementRunConfig):
    """读取 refinement 所依赖的完整场景配置.

    这里不能只读 demo 顶层 YAML.
    因为像 `configs/demo/lyra_static.yaml` 这类入口配置,真正的数据契约
    在 `config_path` 指向的训练配置链里.
    """

    from omegaconf import OmegaConf
    from src.models.utils.misc import load_and_merge_configs

    root_config = OmegaConf.load(str(run_config.config_path))
    root_config_dict = OmegaConf.to_container(root_config, resolve=True)
    if not isinstance(root_config_dict, dict):
        raise TypeError("Top-level refinement config must resolve to a mapping.")

    scene_config_paths = _resolve_scene_config_paths(run_config.config_path, root_config_dict)
    if len(scene_config_paths) == 1:
        scene_config = OmegaConf.load(str(scene_config_paths[0]))
    else:
        scene_config = load_and_merge_configs([str(path) for path in scene_config_paths])

    for key, value in _build_scene_loader_overrides(root_config_dict, run_config).items():
        scene_config[key] = value

    return scene_config


def build_scene_bundle(
    run_config: RefinementRunConfig,
    batch: dict[str, Any] | None = None,
    test_dataloader: Any | None = None,
):
    """从真实 dataloader 或外部 batch 构建 `SceneBundle`."""

    if batch is not None:
        return standardize_batch(
            batch=batch,
            scene_index=run_config.scene_index,
            view_id=run_config.view_id,
            frame_indices=run_config.frame_indices,
            target_subsample=run_config.target_subsample,
            reference_mode=run_config.reference_mode,
            sr_scale=run_config.sr_scale,
        )

    if test_dataloader is None:
        from src.models.data import get_multi_dataloader

        scene_config = _load_scene_config(run_config)
        try:
            _, test_dataloader = get_multi_dataloader(scene_config)
        except FileNotFoundError as exc:
            dataset_name = run_config.dataset_name or "config.dataset_name"
            raise FileNotFoundError(
                f"Failed to build dataloader because dataset root is missing. "
                f"Current dataset selection: {dataset_name}. "
                f"If the demo YAML points to a dataset variant that is not present locally, "
                f"please pass `--dataset-name` to override it."
            ) from exc

    for index, batch_item in enumerate(test_dataloader):
        if index != run_config.scene_index:
            continue
        return standardize_batch(
            batch=batch_item,
            scene_index=run_config.scene_index,
            view_id=run_config.view_id,
            frame_indices=run_config.frame_indices,
            target_subsample=run_config.target_subsample,
            reference_mode=run_config.reference_mode,
            sr_scale=run_config.sr_scale,
        )

    raise IndexError(f"Could not find scene index {run_config.scene_index} in the test dataloader.")
