"""`refinement_v2` 测试辅助工具."""

from __future__ import annotations

from pathlib import Path

import torch

from src.refinement_v2.config import RefinementRunConfig, StageHyperParams
from src.refinement_v2.data_loader import SceneBundle
from src.refinement_v2.gaussian_adapter import GaussianAdapter


class FakeRenderer:
    """一个可微的轻量渲染桩.

    它只用高斯颜色均值构造整帧图像.
    这样 tests 可以专注验证控制流和参数更新,不用依赖真实 GPU renderer.
    """

    def render(self, gaussians: torch.Tensor, scene: SceneBundle) -> dict[str, torch.Tensor]:
        batch_size, _, _ = gaussians.shape
        num_views = scene.gt_images.shape[1]
        height, width = scene.gt_images.shape[-2:]

        color_mean = gaussians[:, :, 11:].mean(dim=1)
        image = color_mean[:, None, :, None, None].expand(batch_size, num_views, 3, height, width).clone()
        depth = torch.zeros(batch_size, num_views, 1, height, width, dtype=image.dtype, device=image.device)
        return {"images_pred": image, "depths_pred": depth}


def build_run_config(tmp_path: Path, **overrides) -> RefinementRunConfig:
    """构造测试用运行配置."""

    kwargs = {
        "config_path": Path("configs/demo/lyra_static.yaml"),
        "gaussians_path": tmp_path / "gaussians_init.ply",
        "outdir": tmp_path / "refine_out",
        "device": "cpu",
    }
    kwargs.update(overrides)
    return RefinementRunConfig(**kwargs)


def build_stage_hparams(**overrides) -> StageHyperParams:
    """构造测试用超参数."""

    kwargs = {
        "iters_stage2a": 3,
        "iters_stage2b": 2,
        "iters_pose": 2,
        "iters_joint": 2,
        "plateau_patience": 50,
    }
    kwargs.update(overrides)
    return StageHyperParams(**kwargs)


def build_scene_bundle(reference_mode: str = "native", sr_scale: float = 1.0) -> SceneBundle:
    """构造一个小型 synthetic scene bundle.

    这里默认仍返回 native-reference 版本.
    但也允许测试直接切到 super-resolved reference, 复用同一套夹具.
    """

    gt_images = torch.full((1, 3, 3, 6, 6), 0.8, dtype=torch.float32)
    cam_view = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(1, 3, 1, 1)
    intrinsics = torch.tensor([[[1.0, 1.0, 0.5, 0.5]]], dtype=torch.float32).repeat(1, 3, 1)
    target_index = torch.arange(3).view(1, 3)
    scale_value = float(sr_scale)

    if reference_mode == "super_resolved":
        scale_int = int(round(scale_value))
        if scale_int <= 1 or abs(scale_value - scale_int) > 1e-6:
            raise ValueError(f"super_resolved test scene requires integer sr_scale > 1, got {sr_scale}.")
        reference_hw = (gt_images.shape[-2] * scale_int, gt_images.shape[-1] * scale_int)
        reference_images = torch.full((1, 3, 3, *reference_hw), 0.8, dtype=torch.float32)
        intrinsics_ref = intrinsics * scale_value
    else:
        reference_hw = (6, 6)
        reference_images = gt_images.clone()
        intrinsics_ref = intrinsics.clone()

    return SceneBundle(
        gt_images=gt_images,
        cam_view=cam_view,
        intrinsics=intrinsics,
        frame_indices=[0, 1, 2],
        scene_index=0,
        view_id="3",
        target_index=target_index,
        file_name="synthetic_scene",
        reference_images=reference_images,
        intrinsics_ref=intrinsics_ref,
        native_hw=(6, 6),
        reference_hw=reference_hw,
        reference_mode=reference_mode,
        sr_scale=scale_value,
    )


def build_gaussian_adapter(num_points: int = 16) -> GaussianAdapter:
    """构造一个可训练的 synthetic gaussian adapter."""

    gaussians = torch.zeros(num_points, 14, dtype=torch.float32)
    gaussians[:, 3] = 0.2
    gaussians[:, 4:7] = 0.1
    gaussians[:, 7] = 1.0
    gaussians[:, 11:] = 0.1
    return GaussianAdapter.from_tensor(gaussians)
