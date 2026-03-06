"""数据标准化测试."""

import torch

from src.refinement_v2.config import RefinementRunConfig
from src.refinement_v2.data_loader import (
    _build_scene_loader_overrides,
    _resolve_scene_config_paths,
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
