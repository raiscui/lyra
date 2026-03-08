"""`refinement_v2` 的配置层.

这一层只负责两件事:
1. 定义运行期会反复传递的 dataclass.
2. 把 CLI 参数稳定地映射成这些对象.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RefinementRunConfig:
    """描述一次 refinement 运行所需的外部参数."""

    config_path: Path
    gaussians_path: Path
    outdir: Path
    scene_stem: str | None = None
    pose_path: Path | None = None
    intrinsics_path: Path | None = None
    rgb_path: Path | None = None
    pose_root: Path | None = None
    intrinsics_root: Path | None = None
    rgb_root: Path | None = None
    scene_index: int = 0
    dataset_name: str | None = None
    view_id: str | None = None
    view_ids: list[str] | None = None
    reference_mode: str = "native"
    sr_scale: float = 1.0
    reference_path: Path | None = None
    reference_root: Path | None = None
    reference_intrinsics_path: Path | None = None
    frame_indices: list[int] | None = None
    target_subsample: int = 1
    start_stage: str = "stage2a"
    stage2a_mode: str = "auto"
    enable_stage2b: bool = False
    enable_pruning: bool = False
    enable_pose_diagnostic: bool = False
    enable_joint_fallback: bool = False
    stop_after: str | None = None
    device: str = "cuda"
    render_devices: list[str] | None = None
    mixed_precision: bool = False
    save_every: int = 50
    resume: bool = False
    dry_run: bool = False
    overwrite: bool = False


@dataclass
class StageHyperParams:
    """描述各阶段使用的默认超参数."""

    alpha_rgb: float = 1.0
    alpha_perc: float = 0.0
    q_low: float = 0.50
    q_high: float = 0.90
    weight_tau: float = 0.45
    weight_floor: float = 0.20
    ema_decay: float = 0.90
    iters_stage2a: int = 600
    iters_stage2b: int = 300
    iters_pose: int = 100
    iters_joint: int = 200
    lr_opacity: float = 1e-2
    lr_color: float = 5e-3
    lr_scale: float = 1e-3
    lr_means: float = 1e-4
    lr_pose: float = 3e-5
    patch_size: int = 0
    lambda_patch_rgb: float = 0.0
    lambda_patch_perceptual: float = 0.0
    lambda_sampling_smooth: float = 5e-4
    lambda_means_anchor: float = 0.01
    lambda_rotation_reg: float = 0.01
    means_delta_cap: float = 0.02
    scale_tail_threshold: float = 0.25
    sampling_radius_threshold: float = 1.5
    opacity_low_threshold: float = 0.10
    opacity_prune_threshold: float = 0.05
    lambda_scale_tail: float = 1e-2
    lambda_opacity_sparse: float = 1e-3
    prune_every: int = 2
    prune_warmup_iters: int = 2
    prune_max_fraction: float = 0.02
    min_gaussians_to_keep: int = 1
    plateau_patience: int = 10
    plateau_delta: float = 1e-4


def _parse_frame_indices(raw_value: str | None) -> list[int] | None:
    """把 `1,2,5` 这样的字符串转成整数列表."""

    if raw_value is None or raw_value.strip() == "":
        return None

    items = [item.strip() for item in raw_value.split(",")]
    frame_indices = [int(item) for item in items if item]
    return frame_indices or None


def _parse_view_ids(raw_value: str | None) -> list[str] | None:
    """把 `5,0,1,2,3,4` 这样的字符串转成 view id 列表."""

    if raw_value is None or raw_value.strip() == "":
        return None

    items = [item.strip() for item in raw_value.split(",")]
    view_ids = [item for item in items if item]
    return view_ids or None


def _parse_device_names(raw_value: str | None) -> list[str] | None:
    """把 `cuda:0,cuda:1` 这样的字符串转成设备名列表."""

    if raw_value is None or raw_value.strip() == "":
        return None

    items = [item.strip() for item in raw_value.split(",")]
    device_names = [item for item in items if item]
    return device_names or None


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器."""

    parser = argparse.ArgumentParser(
        description="Run robust gaussian refinement V2 on an exported gaussian scene."
    )

    parser.add_argument("--config", required=True, help="Path to the scene config YAML.")
    parser.add_argument("--gaussians", required=True, help="Path to the initial gaussian PLY.")
    parser.add_argument("--outdir", required=True, help="Output directory for diagnostics and results.")
    parser.add_argument(
        "--scene-stem",
        type=str,
        default=None,
        help="Optional scene stem used by the explicit full-view root inputs, e.g. 00172.",
    )
    parser.add_argument(
        "--pose-path",
        type=str,
        default=None,
        help="Optional direct pose npz input. When set together with --intrinsics-path and --rgb-path, the loader skips provider mode.",
    )
    parser.add_argument(
        "--intrinsics-path",
        type=str,
        default=None,
        help="Optional direct intrinsics npz input used together with --pose-path and --rgb-path.",
    )
    parser.add_argument(
        "--rgb-path",
        type=str,
        default=None,
        help="Optional direct RGB input path. Supports a frame directory or a local video file.",
    )
    parser.add_argument(
        "--pose-root",
        type=str,
        default=None,
        help="Optional multi-view pose root. Expected layout: <root>/<view_id>/pose/<scene_stem>.npz.",
    )
    parser.add_argument(
        "--intrinsics-root",
        type=str,
        default=None,
        help="Optional multi-view intrinsics root. Expected layout: <root>/<view_id>/intrinsics/<scene_stem>.npz.",
    )
    parser.add_argument(
        "--rgb-root",
        type=str,
        default=None,
        help="Optional multi-view RGB root. Expected layout: <root>/<view_id>/rgb/<scene_stem>.mp4 or frame dir.",
    )
    parser.add_argument("--scene-index", type=int, default=0, help="Scene index inside the test dataloader.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset registry override, e.g. lyra_static_demo_generated.",
    )
    parser.add_argument("--view-id", type=str, default=None, help="Optional logical view identifier.")
    parser.add_argument(
        "--view-ids",
        type=str,
        default=None,
        help="Optional comma-separated multi-view identifiers, e.g. 5,0,1,2,3,4.",
    )
    parser.add_argument(
        "--frame-indices",
        type=str,
        default=None,
        help="Comma-separated frame indices to refine, e.g. 0,4,8.",
    )
    parser.add_argument(
        "--reference-mode",
        type=str,
        default="native",
        choices=["native", "super_resolved"],
        help="Reference supervision mode. `super_resolved` uses scaled reference frames.",
    )
    parser.add_argument(
        "--sr-scale",
        type=float,
        default=1.0,
        help="Reference scale used when reference_mode=super_resolved.",
    )
    parser.add_argument(
        "--reference-path",
        type=str,
        default=None,
        help="Optional external reference source. Supports a frame directory or a local video file.",
    )
    parser.add_argument(
        "--reference-root",
        type=str,
        default=None,
        help="Optional multi-view external reference root. Expected layout: <root>/<view_id>/rgb/<scene_stem>.mp4 or frame dir.",
    )
    parser.add_argument(
        "--reference-intrinsics-path",
        type=str,
        default=None,
        help="Optional npz path that provides intrinsics for the external reference frames.",
    )
    parser.add_argument(
        "--target-subsample",
        type=int,
        default=1,
        help="Fallback frame stride when frame_indices is not provided.",
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        default="stage2a",
        choices=["stage2a", "stage2b"],
        help="Optimization entry stage after Phase 0 / Phase 1 preparation.",
    )
    parser.add_argument(
        "--stage2a-mode",
        type=str,
        default="auto",
        choices=["auto", "legacy", "enhanced"],
        help=(
            "How Stage 2A behaves internally. "
            "`auto` keeps the current compatible behavior, "
            "`legacy` forces native cleanup only, "
            "`enhanced` forces native cleanup + Phase 3S + Stage 3SR."
        ),
    )
    parser.add_argument("--enable-stage2b", action="store_true", help="Enable limited geometry refinement.")
    parser.add_argument(
        "--enable-pose-diagnostic",
        action="store_true",
        help="Enable tiny pose-only diagnostic stage.",
    )
    parser.add_argument(
        "--enable-pruning",
        action="store_true",
        help="Enable low-opacity pruning inside Stage 2A.",
    )
    parser.add_argument(
        "--enable-joint-fallback",
        action="store_true",
        help="Enable final joint fallback stage.",
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        default=None,
        choices=["phase0", "phase1", "stage2a", "stage2b", "phase3", "phase4"],
        help="Stop after a specific stage for debugging.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device string, e.g. cuda or cpu.")
    parser.add_argument(
        "--render-devices",
        type=str,
        default=None,
        help="Optional comma-separated render devices, e.g. cuda:0,cuda:1. The first device remains the optimizer/main device.",
    )
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision for refinement.")
    parser.add_argument("--save-every", type=int, default=50, help="Save intermediate state every N iterations.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest saved state.")
    parser.add_argument("--dry-run", action="store_true", help="Only run Phase 0 baseline diagnostics.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output directory.")

    parser.add_argument("--weight-floor", type=float, default=0.20)
    parser.add_argument("--weight-tau", type=float, default=0.45)
    parser.add_argument("--ema-decay", type=float, default=0.90)
    parser.add_argument("--iters-stage2a", type=int, default=600)
    parser.add_argument("--iters-stage2b", type=int, default=300)
    parser.add_argument("--iters-pose", type=int, default=100)
    parser.add_argument("--iters-joint", type=int, default=200)
    parser.add_argument("--lr-opacity", type=float, default=1e-2)
    parser.add_argument("--lr-color", type=float, default=5e-3)
    parser.add_argument("--lr-scale", type=float, default=1e-3)
    parser.add_argument("--lr-means", type=float, default=1e-4)
    parser.add_argument("--lr-pose", type=float, default=3e-5)
    parser.add_argument("--patch-size", type=int, default=0)
    parser.add_argument("--lambda-patch-rgb", type=float, default=0.0)
    parser.add_argument("--lambda-patch-perceptual", type=float, default=0.0)
    parser.add_argument("--lambda-sampling-smooth", type=float, default=5e-4)
    parser.add_argument("--lambda-means-anchor", type=float, default=0.01)
    parser.add_argument("--lambda-rotation-reg", type=float, default=0.01)
    parser.add_argument("--sampling-radius-threshold", type=float, default=1.5)
    parser.add_argument("--opacity-prune-threshold", type=float, default=0.05)
    parser.add_argument("--prune-every", type=int, default=2)
    parser.add_argument("--prune-warmup-iters", type=int, default=2)
    parser.add_argument("--prune-max-fraction", type=float, default=0.02)
    parser.add_argument("--min-gaussians-to-keep", type=int, default=1)

    return parser


def load_effective_config_from_cli(
    argv: list[str] | None = None,
) -> tuple[RefinementRunConfig, StageHyperParams]:
    """把 CLI 参数映射成运行配置和阶段超参数."""

    parser = build_parser()
    args = parser.parse_args(argv)

    run_config = RefinementRunConfig(
        config_path=Path(args.config),
        gaussians_path=Path(args.gaussians),
        outdir=Path(args.outdir),
        scene_stem=args.scene_stem,
        pose_path=Path(args.pose_path) if args.pose_path else None,
        intrinsics_path=Path(args.intrinsics_path) if args.intrinsics_path else None,
        rgb_path=Path(args.rgb_path) if args.rgb_path else None,
        pose_root=Path(args.pose_root) if args.pose_root else None,
        intrinsics_root=Path(args.intrinsics_root) if args.intrinsics_root else None,
        rgb_root=Path(args.rgb_root) if args.rgb_root else None,
        scene_index=args.scene_index,
        dataset_name=args.dataset_name,
        view_id=args.view_id,
        view_ids=_parse_view_ids(args.view_ids),
        reference_mode=args.reference_mode,
        sr_scale=args.sr_scale,
        reference_path=Path(args.reference_path) if args.reference_path else None,
        reference_root=Path(args.reference_root) if args.reference_root else None,
        reference_intrinsics_path=Path(args.reference_intrinsics_path) if args.reference_intrinsics_path else None,
        frame_indices=_parse_frame_indices(args.frame_indices),
        target_subsample=args.target_subsample,
        start_stage=args.start_stage,
        stage2a_mode=args.stage2a_mode,
        enable_stage2b=args.enable_stage2b,
        enable_pruning=args.enable_pruning,
        enable_pose_diagnostic=args.enable_pose_diagnostic,
        enable_joint_fallback=args.enable_joint_fallback,
        stop_after=args.stop_after,
        device=args.device,
        render_devices=_parse_device_names(args.render_devices),
        mixed_precision=args.mixed_precision,
        save_every=args.save_every,
        resume=args.resume,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    stage_hyper_params = StageHyperParams(
        weight_floor=args.weight_floor,
        weight_tau=args.weight_tau,
        ema_decay=args.ema_decay,
        iters_stage2a=args.iters_stage2a,
        iters_stage2b=args.iters_stage2b,
        iters_pose=args.iters_pose,
        iters_joint=args.iters_joint,
        lr_opacity=args.lr_opacity,
        lr_color=args.lr_color,
        lr_scale=args.lr_scale,
        lr_means=args.lr_means,
        lr_pose=args.lr_pose,
        patch_size=args.patch_size,
        lambda_patch_rgb=args.lambda_patch_rgb,
        lambda_patch_perceptual=args.lambda_patch_perceptual,
        lambda_sampling_smooth=args.lambda_sampling_smooth,
        lambda_means_anchor=args.lambda_means_anchor,
        lambda_rotation_reg=args.lambda_rotation_reg,
        sampling_radius_threshold=args.sampling_radius_threshold,
        opacity_prune_threshold=args.opacity_prune_threshold,
        prune_every=args.prune_every,
        prune_warmup_iters=args.prune_warmup_iters,
        prune_max_fraction=args.prune_max_fraction,
        min_gaussians_to_keep=args.min_gaussians_to_keep,
    )

    return run_config, stage_hyper_params
