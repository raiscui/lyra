"""相机轨迹中心深度估计与位移解耦回归测试."""

from __future__ import annotations

import argparse

import pytest
import torch

from cosmos_predict1.diffusion.inference.camera_utils import (
    estimate_trajectory_center_depth,
    generate_camera_trajectory,
)
from cosmos_predict1.diffusion.inference.gen3c_single_image import (
    _resolve_pipeline_offload_network,
    _resolve_translation_reference_depth,
)
from cosmos_predict1.diffusion.inference.inference_utils import add_camera_center_arguments


def _camera_positions_from_w2cs(generated_w2cs: torch.Tensor) -> torch.Tensor:
    """把 world-to-camera 轨迹还原成相机位置序列."""

    c2ws = torch.linalg.inv(generated_w2cs[0])
    return c2ws[:, :3, 3]


def _path_length(generated_w2cs: torch.Tensor) -> float:
    """计算整条轨迹的总位移长度."""

    positions = _camera_positions_from_w2cs(generated_w2cs)
    return float(torch.norm(positions[1:] - positions[:-1], dim=1).sum().item())


def test_estimate_trajectory_center_depth_prefers_center_crop_statistics() -> None:
    depth = torch.tensor(
        [
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 10.0, 10.0, 2.0],
            [2.0, 10.0, 10.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)
    mask = torch.ones_like(depth, dtype=torch.bool)

    estimated = estimate_trajectory_center_depth(
        depth,
        mask,
        depth_quantile=0.5,
        center_crop_ratio=0.5,
        fallback_depth=1.0,
    )

    assert estimated == pytest.approx(10.0)


def test_estimate_trajectory_center_depth_falls_back_to_full_frame_when_center_invalid() -> None:
    depth = torch.tensor(
        [
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 10.0, 10.0, 2.0],
            [2.0, 10.0, 10.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)
    mask = torch.ones_like(depth, dtype=torch.bool)
    mask[:, :, 1:3, 1:3] = False

    estimated = estimate_trajectory_center_depth(
        depth,
        mask,
        depth_quantile=0.5,
        center_crop_ratio=0.5,
        fallback_depth=1.0,
    )

    assert estimated == pytest.approx(2.0)


def test_estimate_trajectory_center_depth_supports_foreground_mask_mode() -> None:
    depth = torch.tensor(
        [
            [50.0, 50.0, 50.0, 50.0],
            [50.0, 3.0, 4.0, 50.0],
            [50.0, 5.0, 6.0, 50.0],
            [50.0, 50.0, 50.0, 50.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)
    mask = torch.zeros_like(depth, dtype=torch.bool)
    mask[:, :, 1:3, 1:3] = True

    estimated = estimate_trajectory_center_depth(
        depth,
        mask,
        mode="foreground_mask",
        depth_quantile=0.5,
        center_crop_ratio=0.5,
        fallback_depth=1.0,
    )

    assert estimated == pytest.approx(4.5)


def test_add_camera_center_arguments_accepts_mode_and_multiplier_alias() -> None:
    parser = argparse.ArgumentParser()
    add_camera_center_arguments(parser)

    args = parser.parse_args(
        [
            "--auto_center_depth",
            "--auto_center_depth_mode",
            "foreground_mask",
            "--auto_center_depth_multiplier",
            "0.8",
        ]
    )

    assert args.auto_center_depth is True
    assert args.auto_center_depth_mode == "foreground_mask"
    assert args.auto_center_depth_scale == pytest.approx(0.8)


def test_add_camera_center_arguments_accepts_translation_reference_depth_scale() -> None:
    parser = argparse.ArgumentParser()
    add_camera_center_arguments(parser)

    args = parser.parse_args(
        [
            "--translation_reference_depth_scale",
            "0.25",
        ]
    )

    assert args.translation_reference_depth == pytest.approx(1.0)
    assert args.translation_reference_depth_scale == pytest.approx(0.25)


def test_resolve_translation_reference_depth_prefers_scale_when_provided() -> None:
    args = argparse.Namespace(
        translation_reference_depth=1.0,
        translation_reference_depth_scale=0.25,
    )

    resolved = _resolve_translation_reference_depth(args, center_depth=12.0)

    assert resolved == pytest.approx(3.0)


def test_resolve_translation_reference_depth_uses_fixed_value_when_scale_missing() -> None:
    args = argparse.Namespace(
        translation_reference_depth=1.5,
        translation_reference_depth_scale=None,
    )

    resolved = _resolve_translation_reference_depth(args, center_depth=12.0)

    assert resolved == pytest.approx(1.5)


def test_resolve_pipeline_offload_network_disables_transformer_offload_on_multi_gpu() -> None:
    args = argparse.Namespace(
        num_gpus=2,
        offload_diffusion_transformer=True,
    )

    resolved = _resolve_pipeline_offload_network(args, need_latent=True)

    assert resolved is False


def test_resolve_pipeline_offload_network_keeps_requested_value_on_single_gpu() -> None:
    args = argparse.Namespace(
        num_gpus=1,
        offload_diffusion_transformer=True,
    )

    resolved = _resolve_pipeline_offload_network(args, need_latent=True)

    assert resolved is True


def test_generate_camera_trajectory_decouples_center_depth_from_path_scale() -> None:
    initial_w2c = torch.eye(4, dtype=torch.float32)
    initial_intrinsics = torch.eye(3, dtype=torch.float32)

    baseline_w2cs, _ = generate_camera_trajectory(
        trajectory_type="left",
        initial_w2c=initial_w2c,
        initial_intrinsics=initial_intrinsics,
        num_frames=5,
        movement_distance=0.3,
        camera_rotation="center_facing",
        center_depth=1.0,
        device="cpu",
    )
    auto_center_w2cs, _ = generate_camera_trajectory(
        trajectory_type="left",
        initial_w2c=initial_w2c,
        initial_intrinsics=initial_intrinsics,
        num_frames=5,
        movement_distance=0.3,
        camera_rotation="center_facing",
        center_depth=8.0,
        translation_reference_depth=1.0,
        device="cpu",
    )
    scaled_w2cs, _ = generate_camera_trajectory(
        trajectory_type="left",
        initial_w2c=initial_w2c,
        initial_intrinsics=initial_intrinsics,
        num_frames=5,
        movement_distance=0.3,
        camera_rotation="center_facing",
        center_depth=8.0,
        device="cpu",
    )

    baseline_path_length = _path_length(baseline_w2cs)
    auto_center_path_length = _path_length(auto_center_w2cs)

    assert auto_center_path_length == pytest.approx(baseline_path_length, rel=0.02)
    assert _path_length(scaled_w2cs) == pytest.approx(_path_length(baseline_w2cs) * 8.0, rel=1e-6)
    assert not torch.allclose(auto_center_w2cs[0, -1, :3, :3], baseline_w2cs[0, -1, :3, :3])
