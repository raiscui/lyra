"""运行 `FlashVSR-Pro` 生成 external SR reference.

这条脚本的定位是:
1. 先把 `diffusion_output_generated/*/rgb/*.mp4` 做成可复用的 SR reference.
2. 顺手导出逐帧对照图,方便肉眼判断问题是不是从 SR 阶段开始出现.
3. 输出结果继续给 `scripts/refine_robust_v2.py --reference-path ...` 使用.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.refinement_v2.flashvsr_reference import (
    FlashVsrRunConfig,
    discover_source_videos,
    ensure_runner_ready,
    parse_csv_ints,
    parse_csv_strings,
    plan_video_tasks,
    process_video_task,
)


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器."""

    parser = argparse.ArgumentParser(
        description="Run FlashVSR-Pro on diffusion_output_generated videos and export per-frame debug artifacts.",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="assets/demo/static/diffusion_output_generated",
        help="Root folder that contains view_id/rgb/*.mp4.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/flashvsr_reference",
        help="Root folder used to store FlashVSR outputs and debug artifacts.",
    )
    parser.add_argument(
        "--flashvsr-repo",
        type=str,
        required=True,
        help="Local checkout of https://github.com/LujiaJin/FlashVSR-Pro.",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="docker",
        choices=["docker", "local"],
        help="How to invoke FlashVSR-Pro. `docker` is the recommended path.",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default="flashvsr-pro:latest",
        help="Docker image name used when --runner=docker.",
    )
    parser.add_argument(
        "--docker-gpus",
        type=str,
        default="all",
        help="Value passed to `docker run --gpus`.",
    )
    parser.add_argument(
        "--local-python",
        type=str,
        default="python",
        help="Python executable used when --runner=local.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "tiny", "tiny-long"],
        help="FlashVSR-Pro inference mode.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Upscale factor forwarded to FlashVSR-Pro.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Precision forwarded to FlashVSR-Pro.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=10,
        help="Output quality forwarded to FlashVSR-Pro.",
    )
    parser.add_argument(
        "--view-ids",
        type=str,
        default=None,
        help="Optional comma-separated view ids, e.g. 0,3,5.",
    )
    parser.add_argument(
        "--scene-stem",
        type=str,
        default=None,
        help="Optional scene stem filter, e.g. 00172.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Forward --keep-audio to FlashVSR-Pro when input video has audio.",
    )
    parser.add_argument(
        "--tile-vae",
        action="store_true",
        help="When fallback tiling happens, also enable VAE tiling.",
    )
    parser.add_argument(
        "--disable-fallback-tiling",
        action="store_true",
        help="Disable the automatic fallback from full mode to tiled mode on OOM-like failures.",
    )
    parser.add_argument(
        "--fallback-tile-size",
        type=int,
        default=512,
        help="Tile size used by the automatic OOM fallback.",
    )
    parser.add_argument(
        "--fallback-overlap",
        type=int,
        default=128,
        help="Tile overlap used by the automatic OOM fallback.",
    )
    parser.add_argument(
        "--debug-frame-indices",
        type=str,
        default=None,
        help="Optional comma-separated debug frame indices, e.g. 0,8,16,24.",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=8,
        help="When --debug-frame-indices is omitted, dump every N-th frame plus the last frame.",
    )
    parser.add_argument(
        "--dump-all-debug-frames",
        action="store_true",
        help="Dump every frame for native/SR/compare directories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing SR videos instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only emit commands and manifest files without running FlashVSR-Pro.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """运行 `FlashVSR-Pro` reference 生成流程."""

    args = build_parser().parse_args(argv)
    view_ids = parse_csv_strings(args.view_ids)
    debug_frame_indices = parse_csv_ints(args.debug_frame_indices)

    run_config = FlashVsrRunConfig(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        flashvsr_repo=Path(args.flashvsr_repo),
        runner=args.runner,
        docker_image=args.docker_image,
        docker_gpus=args.docker_gpus,
        local_python=args.local_python,
        mode=args.mode,
        scale=args.scale,
        dtype=args.dtype,
        quality=args.quality,
        keep_audio=args.keep_audio,
        enable_tile_vae=args.tile_vae,
        enable_fallback_tiling=not args.disable_fallback_tiling,
        fallback_tile_size=args.fallback_tile_size,
        fallback_overlap=args.fallback_overlap,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        debug_every=args.debug_every,
        dump_all_debug_frames=args.dump_all_debug_frames,
        debug_frame_indices=debug_frame_indices,
    )

    ensure_runner_ready(run_config)
    source_paths = discover_source_videos(
        input_root=run_config.input_root,
        view_ids=view_ids,
        scene_stem=args.scene_stem,
    )
    tasks = plan_video_tasks(
        input_root=run_config.input_root,
        output_root=run_config.output_root,
        source_paths=source_paths,
        mode=run_config.mode,
        scale=run_config.scale,
    )

    summaries: list[dict[str, object]] = []
    print(f"将处理 {len(tasks)} 个视频。")
    for task in tasks:
        print(f"[FlashVSR] view={task.view_id} scene={task.scene_stem} -> {task.output_video_path}")
        summary = process_video_task(run_config=run_config, task=task)
        summaries.append(summary)
        print(
            "[FlashVSR] 完成: "
            f"status={summary['status']} "
            f"output={summary['output_video_path']} "
            f"manifest={task.manifest_path}"
        )

    aggregate_path = run_config.output_root / "flashvsr_reference_summary.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"汇总已写入: {aggregate_path}")


if __name__ == "__main__":
    main()
