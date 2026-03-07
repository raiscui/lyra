"""数据标准化测试."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.refinement_v2.config import RefinementRunConfig
from src.refinement_v2.data_loader import (
    _build_scene_loader_overrides,
    _resolve_scene_config_paths,
    build_scene_bundle,
    standardize_batch,
)


def test_standardize_batch_with_subsample() -> None:
    batch = {
        "images_output": torch.randn(1, 6, 3, 8, 8),
        "cam_view": torch.randn(1, 6, 4, 4),
        "intrinsics": torch.randn(1, 6, 4),
        "target_index": torch.arange(6).view(1, 6),
        "file_name": ["demo_scene"],
    }

    scene = standardize_batch(batch, scene_index=0, view_id="3", target_subsample=2)

    assert scene.frame_indices == [0, 2, 4]
    assert scene.gt_images.shape == (1, 3, 3, 8, 8)
    assert scene.cam_view.shape == (1, 3, 4, 4)
    assert scene.intrinsics.shape == (1, 3, 4)
    assert scene.file_name == "demo_scene"


def test_standardize_batch_with_explicit_frame_indices() -> None:
    batch = {
        "images_output": torch.randn(1, 5, 3, 8, 8),
        "cam_view": torch.randn(1, 5, 4, 4),
        "intrinsics": torch.randn(1, 5, 4),
    }

    scene = standardize_batch(
        batch,
        scene_index=2,
        view_id="5",
        frame_indices=[1, 3],
        target_subsample=1,
    )

    assert scene.scene_index == 2
    assert scene.view_id == "5"
    assert scene.frame_indices == [1, 3]
    assert scene.gt_images.shape[1] == 2


def test_resolve_scene_config_paths_uses_nested_config_chain(tmp_path) -> None:
    base_config_path = tmp_path / "demo.yaml"
    train_config_a = tmp_path / "train_a.yaml"
    train_config_b = tmp_path / "train_b.yaml"

    base_config_path.write_text("demo: true\n", encoding="utf-8")
    train_config_a.write_text("a: 1\n", encoding="utf-8")
    train_config_b.write_text("b: 2\n", encoding="utf-8")

    resolved_paths = _resolve_scene_config_paths(
        base_config_path,
        {"config_path": [str(train_config_a), str(train_config_b)]},
    )

    assert resolved_paths == [train_config_a, train_config_b]


def test_build_scene_loader_overrides_respects_cli_view_override(tmp_path) -> None:
    run_config = RefinementRunConfig(
        config_path=tmp_path / "demo.yaml",
        gaussians_path=tmp_path / "gaussians.ply",
        outdir=tmp_path / "refine_out",
        scene_index=2,
        dataset_name="lyra_static_demo_generated",
        view_id="5",
    )

    overrides = _build_scene_loader_overrides(
        {
            "dataset_name": "lyra_static_demo_generated_one",
            "static_view_indices_fixed": ["3"],
            "target_index_subsample": 4,
            "set_manual_time_idx": True,
        },
        run_config,
    )

    assert overrides["data_mode"] == [["lyra_static_demo_generated", 1]]
    assert overrides["static_view_indices_fixed"] == ["5"]
    assert overrides["num_input_multi_views"] == 1
    assert overrides["target_index_subsample"] == 4
    assert overrides["set_manual_time_idx"] is True
    assert overrides["num_test_images"] == 3
    assert overrides["use_depth"] is False
    assert overrides["load_latents"] is False
    assert overrides["target_index_manual"] is None


def test_standardize_batch_builds_super_resolved_reference_bundle() -> None:
    """SR 模式下应显式生成高分辨率参考图和缩放后的 intrinsics."""

    batch = {
        "images_output": torch.arange(1 * 4 * 3 * 6 * 6, dtype=torch.float32).view(1, 4, 3, 6, 6) / 255.0,
        "cam_view": torch.randn(1, 4, 4, 4),
        "intrinsics": torch.tensor([[[2.0, 3.0, 1.5, 2.5]]], dtype=torch.float32).repeat(1, 4, 1),
    }

    scene = standardize_batch(
        batch,
        scene_index=0,
        view_id="3",
        target_subsample=2,
        reference_mode="super_resolved",
        sr_scale=2.0,
    )

    assert scene.reference_mode == "super_resolved"
    assert scene.sr_scale == 2.0
    assert scene.native_hw == (6, 6)
    assert scene.reference_hw == (12, 12)
    assert scene.reference_images.shape == (1, 2, 3, 12, 12)
    assert torch.allclose(scene.intrinsics_ref, scene.intrinsics * 2.0)


def _write_reference_frames(frame_dir: Path, frame_values: list[int], size: tuple[int, int]) -> None:
    """写一个简单的 RGB 帧目录,便于测试外部 reference 输入."""

    frame_dir.mkdir(parents=True, exist_ok=True)
    height, width = size
    for frame_index, frame_value in enumerate(frame_values):
        frame = np.full((height, width, 3), frame_value, dtype=np.uint8)
        Image.fromarray(frame, mode="RGB").save(frame_dir / f"{frame_index:04d}.png")


def _write_direct_camera_inputs(
    pose_path: Path,
    intrinsics_path: Path,
    num_frames: int,
) -> None:
    """写一组和项目当前真实格式一致的 pose / intrinsics npz."""

    pose = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], repeats=num_frames, axis=0)
    pose[:, 0, 3] = np.arange(num_frames, dtype=np.float32)
    intrinsics = np.stack(
        [
            np.full(num_frames, 10.0, dtype=np.float32),
            np.full(num_frames, 11.0, dtype=np.float32),
            np.full(num_frames, 4.0, dtype=np.float32),
            np.full(num_frames, 5.0, dtype=np.float32),
        ],
        axis=-1,
    )
    frame_inds = np.arange(num_frames, dtype=np.int64)
    np.savez(pose_path, data=pose, inds=frame_inds)
    np.savez(intrinsics_path, data=intrinsics, inds=frame_inds)


def test_standardize_batch_loads_external_reference_directory_and_aligns_indices(tmp_path) -> None:
    """外部 reference 目录如果提供完整时序,应按 selected_frame_indices 对齐."""

    frame_dir = tmp_path / "reference_rgb"
    _write_reference_frames(frame_dir, frame_values=[10, 40, 90, 140], size=(12, 12))

    batch = {
        "images_output": torch.zeros(1, 4, 3, 6, 6, dtype=torch.float32),
        "cam_view": torch.randn(1, 4, 4, 4),
        "intrinsics": torch.tensor([[[2.0, 3.0, 1.5, 2.5]]], dtype=torch.float32).repeat(1, 4, 1),
    }

    scene = standardize_batch(
        batch=batch,
        scene_index=0,
        view_id="5",
        frame_indices=[1, 3],
        reference_mode="super_resolved",
        sr_scale=2.0,
        reference_path=frame_dir,
    )

    assert scene.reference_mode == "super_resolved"
    assert scene.reference_images.shape == (1, 2, 3, 12, 12)
    assert scene.reference_hw == (12, 12)
    assert scene.sr_scale == 2.0
    assert torch.allclose(scene.intrinsics_ref, scene.intrinsics * 2.0)

    selected_values = scene.reference_images[:, :, 0, 0, 0].mul(255.0).round().to(dtype=torch.int64)
    assert selected_values.tolist() == [[40, 140]]


def test_standardize_batch_uses_external_reference_intrinsics_override(tmp_path) -> None:
    """如果提供 external reference intrinsics,应优先使用该覆盖值."""

    frame_dir = tmp_path / "reference_rgb"
    _write_reference_frames(frame_dir, frame_values=[20, 60, 100, 180], size=(12, 12))

    intrinsics_np = np.array(
        [
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
            [40.0, 41.0, 42.0, 43.0],
        ],
        dtype=np.float32,
    )
    intrinsics_path = tmp_path / "reference_intrinsics.npz"
    np.savez(intrinsics_path, intrinsics=intrinsics_np)

    batch = {
        "images_output": torch.zeros(1, 4, 3, 6, 6, dtype=torch.float32),
        "cam_view": torch.randn(1, 4, 4, 4),
        "intrinsics": torch.tensor([[[2.0, 3.0, 1.5, 2.5]]], dtype=torch.float32).repeat(1, 4, 1),
    }

    scene = standardize_batch(
        batch=batch,
        scene_index=0,
        view_id="5",
        frame_indices=[0, 2],
        reference_mode="super_resolved",
        sr_scale=2.0,
        reference_path=frame_dir,
        reference_intrinsics_path=intrinsics_path,
    )

    expected_intrinsics = torch.tensor(
        [[[10.0, 11.0, 12.0, 13.0], [30.0, 31.0, 32.0, 33.0]]],
        dtype=torch.float32,
    )
    assert torch.allclose(scene.intrinsics_ref, expected_intrinsics)


def test_build_scene_bundle_supports_direct_file_inputs(tmp_path) -> None:
    """direct file inputs 应该绕过 provider,但仍保持 provider 的 `cam_view` 契约."""

    rgb_dir = tmp_path / "rgb"
    pose_path = tmp_path / "pose.npz"
    intrinsics_path = tmp_path / "intrinsics.npz"

    _write_reference_frames(rgb_dir, frame_values=[10, 40, 90, 140], size=(6, 8))
    _write_direct_camera_inputs(pose_path, intrinsics_path, num_frames=4)

    run_config = RefinementRunConfig(
        config_path=tmp_path / "demo.yaml",
        gaussians_path=tmp_path / "gaussians.ply",
        outdir=tmp_path / "refine_out",
        pose_path=pose_path,
        intrinsics_path=intrinsics_path,
        rgb_path=rgb_dir,
        frame_indices=[1, 3],
        view_id="3",
    )

    scene = build_scene_bundle(run_config)

    assert scene.frame_indices == [1, 3]
    assert scene.view_id == "3"
    assert scene.gt_images.shape == (1, 2, 3, 6, 8)
    assert scene.cam_view.shape == (1, 2, 4, 4)
    assert scene.intrinsics.shape == (1, 2, 4)
    assert scene.file_name == "rgb"

    selected_values = scene.gt_images[:, :, 0, 0, 0].mul(255.0).round().to(dtype=torch.int64)
    assert selected_values.tolist() == [[40, 140]]

    # provider 的契约是 `cam_view = inverse(c2w).T`.
    # 这里用两帧纯平移矩阵验证 direct path 也走同一约定.
    expected_cam_view = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-3.0, 0.0, 0.0, 1.0],
            ],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(scene.cam_view[0], expected_cam_view)


def test_build_scene_bundle_preserves_cam_view_inputs_without_double_conversion(tmp_path) -> None:
    """如果 pose.npz 本身已经是 `cam_view`,direct path 不应重复 inverse/transpose."""

    rgb_dir = tmp_path / "rgb"
    pose_path = tmp_path / "pose_cam_view.npz"
    intrinsics_path = tmp_path / "intrinsics.npz"

    _write_reference_frames(rgb_dir, frame_values=[10, 40, 90, 140], size=(6, 8))
    _write_direct_camera_inputs(intrinsics_path=intrinsics_path, pose_path=tmp_path / "unused_pose.npz", num_frames=4)

    cam_view = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], repeats=4, axis=0)
    cam_view[:, 3, 2] = np.arange(4, dtype=np.float32)
    np.savez(pose_path, cam_view=cam_view)

    run_config = RefinementRunConfig(
        config_path=tmp_path / "demo.yaml",
        gaussians_path=tmp_path / "gaussians.ply",
        outdir=tmp_path / "refine_out",
        pose_path=pose_path,
        intrinsics_path=intrinsics_path,
        rgb_path=rgb_dir,
        frame_indices=[1, 3],
        view_id="3",
    )

    scene = build_scene_bundle(run_config)

    expected_cam_view = torch.from_numpy(cam_view[[1, 3]]).float()
    assert torch.allclose(scene.cam_view[0], expected_cam_view)


def test_build_scene_bundle_rejects_partial_direct_file_inputs(tmp_path) -> None:
    """direct file inputs 必须成套提供,不能半套半套地混用."""

    run_config = RefinementRunConfig(
        config_path=tmp_path / "demo.yaml",
        gaussians_path=tmp_path / "gaussians.ply",
        outdir=tmp_path / "refine_out",
        rgb_path=tmp_path / "rgb.mp4",
    )

    try:
        build_scene_bundle(run_config)
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected partial direct file inputs to raise ValueError.")

    assert "--pose-path" in message
    assert "--intrinsics-path" in message
