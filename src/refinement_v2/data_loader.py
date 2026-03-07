"""数据读取与标准化逻辑.

这里不直接参与优化.
它只负责把 provider 输出收束成 refinement 层统一消费的 `SceneBundle`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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


def _load_image_tensor(path: Path) -> torch.Tensor:
    """读取单张 RGB 图片并转成 `[3, H, W]` tensor."""

    image = Image.open(path).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).contiguous()


def _load_reference_frames_from_directory(reference_path: Path) -> torch.Tensor:
    """从帧目录读取 `[T, 3, H, W]` 的 reference 序列."""

    supported_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    frame_paths = sorted(
        path for path in reference_path.iterdir() if path.is_file() and path.suffix.lower() in supported_suffixes
    )
    if not frame_paths:
        raise FileNotFoundError(f"No image frames were found under reference directory: {reference_path}")

    frames = [_load_image_tensor(frame_path) for frame_path in frame_paths]
    return torch.stack(frames, dim=0)


def _load_reference_frames_from_video(reference_path: Path) -> torch.Tensor:
    """从本地视频读取 `[T, 3, H, W]` 的 reference 序列.

    这里保持惰性导入 `imageio`.
    这样测试环境在只走目录输入时,不会被额外视频依赖拖住.
    """

    try:
        import imageio.v3 as iio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading a video reference requires `imageio`. "
            "Please use the pixi environment, or install `imageio` in the current python environment."
        ) from exc

    frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in iio.imiter(reference_path)]
    if not frames:
        raise FileNotFoundError(f"No frames could be read from reference video: {reference_path}")
    return torch.stack(frames, dim=0)


def _load_reference_frames(reference_path: Path) -> torch.Tensor:
    """读取外部 reference 图像序列.

    支持两类输入:
    1. 帧目录.
    2. 本地视频文件.
    """

    if reference_path.is_dir():
        return _load_reference_frames_from_directory(reference_path)
    if reference_path.is_file():
        return _load_reference_frames_from_video(reference_path)
    raise FileNotFoundError(f"reference_path does not exist: {reference_path}")


def _load_npz_array(npz_path: Path, preferred_keys: tuple[str, ...], tensor_name: str) -> np.ndarray:
    """从 `npz` 里按常见 key 顺序取出主数组.

    当前项目真实数据通常是:
    - `data`
    - `inds`

    这里优先取语义 key.
    如果没有,再回退到第一个数组.
    """

    with np.load(npz_path, allow_pickle=False) as payload:
        for key in preferred_keys:
            if key in payload:
                return np.asarray(payload[key], dtype=np.float32)

        if not payload.files:
            raise ValueError(f"{tensor_name} npz is empty: {npz_path}")
        return np.asarray(payload[payload.files[0]], dtype=np.float32)


def _load_npz_array_with_key(
    npz_path: Path,
    preferred_keys: tuple[str, ...],
    tensor_name: str,
) -> tuple[np.ndarray, str]:
    """从 `npz` 中同时取出数组和值来源的 key.

    direct input 的 pose 需要根据来源 key 判断是否还要做
    `provider` 契约下的 `cam_view` 变换.
    因此这里保留 key 信息,避免把已经是 `cam_view` 的输入再次变换.
    """

    with np.load(npz_path, allow_pickle=False) as payload:
        for key in preferred_keys:
            if key in payload:
                return np.asarray(payload[key], dtype=np.float32), key

        if not payload.files:
            raise ValueError(f"{tensor_name} npz is empty: {npz_path}")

        first_key = payload.files[0]
        return np.asarray(payload[first_key], dtype=np.float32), first_key


def _convert_direct_pose_to_cam_view(pose: torch.Tensor) -> torch.Tensor:
    """把 raw pose/c2w 序列转换成 provider 期望的 `cam_view`.

    `src/models/data/provider.py` 中的 dataloader 契约明确是:
    `cam_view = torch.inverse(c2ws).transpose(1, 2)`.
    direct input 路径也必须产出同一约定,否则 baseline render 和
    Stage 2A 优化会与 dataloader 路径分叉.
    """

    return torch.linalg.inv(pose).transpose(-1, -2).contiguous()


def _load_direct_pose_sequence(pose_path: Path) -> torch.Tensor:
    """读取并规范 direct input 模式下的相机位姿序列.

    外部 `pose.npz` 在项目当前资产里保存的是 raw pose / c2w.
    refinement 内部真正消费的是 provider 风格的 `cam_view`.
    因此默认会把 raw pose 转成 `inverse(c2w).T`.

    如果输入文件显式使用 `cam_view` 这个 key,则认为它已经满足
    renderer 契约,不再二次变换.
    """

    pose_np, source_key = _load_npz_array_with_key(
        pose_path,
        preferred_keys=("data", "pose", "cam_view", "c2w", "c2ws"),
        tensor_name="pose",
    )
    pose = torch.from_numpy(pose_np).float()
    if pose.ndim == 4 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.ndim != 3 or tuple(pose.shape[-2:]) != (4, 4):
        raise ValueError(f"pose_path must resolve to shape [T, 4, 4], got {tuple(pose.shape)}.")

    if source_key != "cam_view":
        pose = _convert_direct_pose_to_cam_view(pose)
    return pose


def _load_direct_intrinsics_sequence(intrinsics_path: Path) -> torch.Tensor:
    """读取 direct input 模式下的内参序列."""

    intrinsics_np, _ = _load_npz_array_with_key(
        intrinsics_path,
        preferred_keys=("data", "intrinsics", "K"),
        tensor_name="intrinsics",
    )
    intrinsics = torch.from_numpy(intrinsics_np).float()
    if intrinsics.ndim == 3 and intrinsics.shape[0] == 1:
        intrinsics = intrinsics[0]
    if intrinsics.ndim == 1 and intrinsics.shape[0] == 4:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.ndim != 2 or intrinsics.shape[-1] != 4:
        raise ValueError(f"intrinsics_path must resolve to shape [T, 4], got {tuple(intrinsics.shape)}.")
    return intrinsics


def _validate_direct_input_triplet(run_config: RefinementRunConfig) -> bool:
    """校验 direct file inputs 是否成套提供."""

    direct_inputs = {
        "pose_path": run_config.pose_path,
        "intrinsics_path": run_config.intrinsics_path,
        "rgb_path": run_config.rgb_path,
    }
    provided = {name: path for name, path in direct_inputs.items() if path is not None}
    if not provided:
        return False

    if len(provided) != len(direct_inputs):
        missing_names = [name for name, path in direct_inputs.items() if path is None]
        raise ValueError(
            "Direct file inputs require --pose-path, --intrinsics-path and --rgb-path together. "
            f"Missing: {', '.join(missing_names)}."
        )
    return True


def _build_direct_input_batch(run_config: RefinementRunConfig) -> dict[str, Any] | None:
    """从 direct file inputs 直接组装一个 provider-compatible batch."""

    if not _validate_direct_input_triplet(run_config):
        return None

    assert run_config.pose_path is not None
    assert run_config.intrinsics_path is not None
    assert run_config.rgb_path is not None

    rgb_frames = _load_reference_frames(run_config.rgb_path)
    cam_view = _load_direct_pose_sequence(run_config.pose_path)
    intrinsics = _load_direct_intrinsics_sequence(run_config.intrinsics_path)

    frame_count_summary = {
        "rgb": int(rgb_frames.shape[0]),
        "pose": int(cam_view.shape[0]),
        "intrinsics": int(intrinsics.shape[0]),
    }
    unique_lengths = set(frame_count_summary.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            "Direct file inputs must share the same frame count before refinement slicing. "
            f"Got {frame_count_summary}."
        )

    file_name = run_config.rgb_path.stem
    if run_config.rgb_path.is_dir():
        file_name = run_config.rgb_path.name

    return {
        "images_output": rgb_frames.unsqueeze(0),
        "cam_view": cam_view.unsqueeze(0),
        "intrinsics": intrinsics.unsqueeze(0),
        "file_name": [file_name],
    }


def _select_external_sequence(sequence: torch.Tensor, selected_frame_indices: list[int], sequence_name: str) -> torch.Tensor:
    """把外部序列对齐到 refinement 当前使用的帧列表.

    允许两种输入形态:
    1. 外部序列已经只包含当前选中的帧.
    2. 外部序列包含完整时序,此时按 `selected_frame_indices` 再裁一遍.
    """

    num_frames = int(sequence.shape[0])
    if num_frames == len(selected_frame_indices):
        return sequence

    max_index = max(selected_frame_indices) if selected_frame_indices else -1
    if num_frames > max_index:
        index_tensor = torch.tensor(selected_frame_indices, dtype=torch.long)
        return torch.index_select(sequence, dim=0, index=index_tensor)

    raise ValueError(
        f"{sequence_name} contains {num_frames} frames, which cannot satisfy requested indices {selected_frame_indices}."
    )


def _load_reference_intrinsics(reference_intrinsics_path: Path, selected_frame_indices: list[int]) -> torch.Tensor:
    """读取 reference intrinsics,并对齐到当前帧序列."""

    with np.load(reference_intrinsics_path) as payload:
        if "intrinsics" in payload:
            intrinsics_np = payload["intrinsics"]
        else:
            first_key = payload.files[0]
            intrinsics_np = payload[first_key]

    intrinsics = torch.from_numpy(np.asarray(intrinsics_np)).float()
    if intrinsics.ndim == 3 and intrinsics.shape[0] == 1:
        intrinsics = intrinsics[0]
    if intrinsics.ndim != 2 or intrinsics.shape[-1] != 4:
        raise ValueError(
            f"reference intrinsics must resolve to shape [T, 4] or [1, T, 4], got {tuple(intrinsics.shape)}."
        )

    intrinsics = _select_external_sequence(intrinsics, selected_frame_indices, "reference intrinsics")
    return intrinsics.unsqueeze(0)


def _resolve_external_reference_scale(
    native_hw: tuple[int, int],
    reference_hw: tuple[int, int],
    reference_mode: str,
    requested_sr_scale: float,
) -> float:
    """根据外部 reference 分辨率解析实际缩放倍率."""

    native_height, native_width = native_hw
    reference_height, reference_width = reference_hw

    if reference_mode == "native":
        if reference_hw != native_hw:
            raise ValueError(
                f"reference_mode=native requires reference size {native_hw}, got {reference_hw}."
            )
        return 1.0

    if reference_height % native_height != 0 or reference_width % native_width != 0:
        raise ValueError(
            f"External super_resolved reference must be an integer multiple of native size {native_hw}, got {reference_hw}."
        )

    scale_height = reference_height // native_height
    scale_width = reference_width // native_width
    if scale_height != scale_width:
        raise ValueError(
            f"External super_resolved reference must keep aspect ratio. Native={native_hw}, reference={reference_hw}."
        )

    actual_scale = float(scale_height)
    if actual_scale <= 1.0:
        raise ValueError(
            f"reference_mode=super_resolved requires scale > 1, got reference size {reference_hw} for native {native_hw}."
        )

    # `sr_scale=1.0` 视为“让系统自动从外部分辨率推断”.
    # 如果用户显式给了非 1 的值,就要求与真实尺寸一致.
    if abs(float(requested_sr_scale) - 1.0) > 1e-6 and abs(actual_scale - float(requested_sr_scale)) > 1e-6:
        raise ValueError(
            f"External reference scale mismatch: requested sr_scale={requested_sr_scale}, actual={actual_scale}."
        )
    return actual_scale


def _build_external_reference_supervision(
    gt_images: torch.Tensor,
    intrinsics: torch.Tensor,
    reference_path: Path,
    reference_mode: str,
    sr_scale: float,
    selected_frame_indices: list[int],
    reference_intrinsics_path: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], tuple[int, int], float]:
    """从外部 reference 数据源构造监督图像与 intrinsics."""

    normalized_mode = _normalize_reference_mode(reference_mode)
    native_hw = (int(gt_images.shape[-2]), int(gt_images.shape[-1]))

    reference_images = _load_reference_frames(reference_path)
    reference_images = _select_external_sequence(reference_images, selected_frame_indices, "reference images")
    reference_images = reference_images.unsqueeze(0).to(dtype=gt_images.dtype)

    reference_hw = (int(reference_images.shape[-2]), int(reference_images.shape[-1]))
    normalized_sr_scale = _resolve_external_reference_scale(
        native_hw=native_hw,
        reference_hw=reference_hw,
        reference_mode=normalized_mode,
        requested_sr_scale=sr_scale,
    )

    if reference_intrinsics_path is not None:
        intrinsics_ref = _load_reference_intrinsics(reference_intrinsics_path, selected_frame_indices)
        intrinsics_ref = intrinsics_ref.to(dtype=intrinsics.dtype)
    elif normalized_mode == "native":
        intrinsics_ref = intrinsics.clone()
    else:
        intrinsics_ref = intrinsics * float(normalized_sr_scale)

    return reference_images, intrinsics_ref, native_hw, reference_hw, float(normalized_sr_scale)


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
    selected_frame_indices: list[int],
    reference_path: Path | None = None,
    reference_intrinsics_path: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], tuple[int, int], float]:
    """构造参考监督图像与对应 intrinsics."""

    normalized_mode = _normalize_reference_mode(reference_mode)
    native_hw = (int(gt_images.shape[-2]), int(gt_images.shape[-1]))

    if reference_path is not None:
        return _build_external_reference_supervision(
            gt_images=gt_images,
            intrinsics=intrinsics,
            reference_path=reference_path,
            reference_mode=normalized_mode,
            sr_scale=sr_scale,
            selected_frame_indices=selected_frame_indices,
            reference_intrinsics_path=reference_intrinsics_path,
        )

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
    reference_path: Path | None = None,
    reference_intrinsics_path: Path | None = None,
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
        selected_frame_indices=selected_frame_indices,
        reference_path=reference_path,
        reference_intrinsics_path=reference_intrinsics_path,
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
            reference_path=run_config.reference_path,
            reference_intrinsics_path=run_config.reference_intrinsics_path,
        )

    direct_input_batch = _build_direct_input_batch(run_config)
    if direct_input_batch is not None:
        return standardize_batch(
            batch=direct_input_batch,
            scene_index=run_config.scene_index,
            view_id=run_config.view_id,
            frame_indices=run_config.frame_indices,
            target_subsample=run_config.target_subsample,
            reference_mode=run_config.reference_mode,
            sr_scale=run_config.sr_scale,
            reference_path=run_config.reference_path,
            reference_intrinsics_path=run_config.reference_intrinsics_path,
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
            reference_path=run_config.reference_path,
            reference_intrinsics_path=run_config.reference_intrinsics_path,
        )

    raise IndexError(f"Could not find scene index {run_config.scene_index} in the test dataloader.")
