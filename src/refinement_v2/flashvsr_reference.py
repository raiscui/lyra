"""`FlashVSR-Pro` reference 生成与排查工具.

这一层不碰 `refine_robust_v2.py` 的优化核心.
它只负责三件事:
1. 扫描待处理的 diffusion 输出视频.
2. 组装并执行 `FlashVSR-Pro` 的 docker / local 命令.
3. 导出逐帧排查产物,方便定位问题是从 SR 前还是 SR 后开始出现.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_FRAME_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
OOM_HINTS = (
    "out of memory",
    "cuda out of memory",
    "cuda error: out of memory",
    "cudnn_status_alloc_failed",
)

FLASHVSR_REQUIRED_MODEL_FILES = {
    "shared": (
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
    ),
    "full": ("Wan2.1_VAE.pth",),
    "tiny": ("TCDecoder.ckpt",),
    "tiny-long": ("TCDecoder.ckpt",),
}


@dataclass(frozen=True)
class FlashVsrRunConfig:
    """描述一次 `FlashVSR-Pro` 前置参考生成任务的运行参数."""

    input_root: Path
    output_root: Path
    flashvsr_repo: Path
    runner: str = "docker"
    docker_image: str = "flashvsr-pro:latest"
    docker_gpus: str = "all"
    local_python: str = "python"
    mode: str = "full"
    scale: float = 2.0
    dtype: str = "bf16"
    quality: int = 10
    keep_audio: bool = False
    enable_tile_vae: bool = False
    enable_fallback_tiling: bool = True
    fallback_tile_size: int = 512
    fallback_overlap: int = 128
    overwrite: bool = False
    dry_run: bool = False
    debug_every: int = 8
    dump_all_debug_frames: bool = False
    debug_frame_indices: list[int] | None = None


@dataclass(frozen=True)
class FlashVsrVideoTask:
    """描述一条待处理视频及其输出位置."""

    view_id: str
    scene_stem: str
    source_path: Path
    relative_source_path: Path
    output_video_path: Path
    debug_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class FlashVsrCommandResult:
    """保存一次 `FlashVSR-Pro` 调用的执行结果."""

    command: list[str]
    return_code: int
    stdout: str
    stderr: str
    tiled: bool
    log_path: Path

    @property
    def succeeded(self) -> bool:
        """返回这次命令是否成功."""

        return self.return_code == 0


def _format_scale_tag(scale: float) -> str:
    """把 `2.0` 这种缩放倍率稳定转成目录名片段."""

    if float(scale).is_integer():
        return f"{int(scale)}x"
    return f"{str(scale).replace('.', 'p')}x"


def build_run_tag(mode: str, scale: float) -> str:
    """构建本轮 SR 输出目录标签."""

    return f"{mode}_scale{_format_scale_tag(scale)}"


def parse_csv_strings(raw_value: str | None) -> list[str] | None:
    """把逗号分隔字符串转成字符串列表."""

    if raw_value is None or raw_value.strip() == "":
        return None
    values = [item.strip() for item in raw_value.split(",")]
    normalized_values = [item for item in values if item]
    return normalized_values or None


def parse_csv_ints(raw_value: str | None) -> list[int] | None:
    """把逗号分隔字符串转成整数列表."""

    if raw_value is None or raw_value.strip() == "":
        return None
    return [int(item) for item in parse_csv_strings(raw_value) or []]


def discover_source_videos(
    input_root: Path,
    view_ids: list[str] | None = None,
    scene_stem: str | None = None,
) -> list[Path]:
    """扫描 `diffusion_output_generated/*/rgb/*.mp4` 形式的视频."""

    normalized_input_root = Path(input_root)
    if not normalized_input_root.exists():
        raise FileNotFoundError(f"FlashVSR input root does not exist: {normalized_input_root}")

    allowed_view_ids = set(view_ids or [])
    source_paths: list[Path] = []
    for view_dir in sorted(path for path in normalized_input_root.iterdir() if path.is_dir()):
        if allowed_view_ids and view_dir.name not in allowed_view_ids:
            continue

        rgb_dir = view_dir / "rgb"
        if not rgb_dir.is_dir():
            continue

        for video_path in sorted(path for path in rgb_dir.iterdir() if path.is_file()):
            if video_path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
                continue
            if scene_stem is not None and video_path.stem != scene_stem:
                continue
            source_paths.append(video_path)

    if not source_paths:
        raise FileNotFoundError(
            "No source videos were found for FlashVSR-Pro. "
            f"input_root={normalized_input_root}, view_ids={view_ids}, scene_stem={scene_stem}"
        )
    return source_paths


def plan_video_tasks(
    *,
    input_root: Path,
    output_root: Path,
    source_paths: list[Path],
    mode: str,
    scale: float,
) -> list[FlashVsrVideoTask]:
    """把源视频列表映射成稳定的输出目录布局."""

    run_tag = build_run_tag(mode, scale)
    tasks: list[FlashVsrVideoTask] = []
    for source_path in source_paths:
        relative_source_path = source_path.relative_to(input_root)
        view_id = relative_source_path.parts[0]
        scene_stem = source_path.stem

        output_video_path = output_root / run_tag / view_id / "rgb" / f"{scene_stem}.mp4"
        debug_dir = output_root / run_tag / view_id / "debug" / scene_stem
        manifest_path = output_root / run_tag / view_id / "manifests" / f"{scene_stem}.json"

        tasks.append(
            FlashVsrVideoTask(
                view_id=view_id,
                scene_stem=scene_stem,
                source_path=source_path,
                relative_source_path=relative_source_path,
                output_video_path=output_video_path,
                debug_dir=debug_dir,
                manifest_path=manifest_path,
            )
        )
    return tasks


def ensure_flashvsr_repo_layout(flashvsr_repo: Path) -> None:
    """确认 `FlashVSR-Pro` 仓库目录具备基本结构."""

    infer_path = flashvsr_repo / "infer.py"
    dockerfile_path = flashvsr_repo / "Dockerfile"
    if not infer_path.is_file():
        raise FileNotFoundError(f"`FlashVSR-Pro` repo is missing infer.py: {infer_path}")
    if not dockerfile_path.is_file():
        raise FileNotFoundError(f"`FlashVSR-Pro` repo is missing Dockerfile: {dockerfile_path}")


def ensure_flashvsr_models(flashvsr_repo: Path, mode: str) -> None:
    """确认 `FlashVSR-Pro` 模型文件已就位."""

    model_root = flashvsr_repo / "models" / "FlashVSR-v1.1"
    required_files = list(FLASHVSR_REQUIRED_MODEL_FILES["shared"]) + list(FLASHVSR_REQUIRED_MODEL_FILES[mode])
    missing_files = [file_name for file_name in required_files if not (model_root / file_name).is_file()]
    if missing_files:
        missing_summary = ", ".join(missing_files)
        raise FileNotFoundError(
            "FlashVSR-Pro model weights are incomplete. "
            f"model_root={model_root}, missing={missing_summary}"
        )


def ensure_runner_ready(run_config: FlashVsrRunConfig) -> None:
    """在真正执行前检查 runner 所需前置条件."""

    ensure_flashvsr_repo_layout(run_config.flashvsr_repo)
    if run_config.dry_run:
        return

    ensure_flashvsr_models(run_config.flashvsr_repo, run_config.mode)

    if run_config.runner == "docker" and shutil.which("docker") is None:
        raise RuntimeError(
            "Docker runner was requested, but `docker` is not available in PATH. "
            "Please install Docker, or switch to `--runner local` after activating the FlashVSR environment."
        )


def _build_flashvsr_infer_args(
    *,
    input_path: str,
    output_path: str,
    run_config: FlashVsrRunConfig,
    tiled: bool,
) -> list[str]:
    """构造 `infer.py` 的参数列表."""

    command = [
        "python",
        "infer.py",
        "-i",
        input_path,
        "-o",
        output_path,
        "--mode",
        run_config.mode,
        "--scale",
        str(run_config.scale),
        "--dtype",
        run_config.dtype,
        "--quality",
        str(run_config.quality),
    ]
    if run_config.keep_audio:
        command.append("--keep-audio")
    if tiled:
        command.extend(
            [
                "--tile-dit",
                "--tile-size",
                str(run_config.fallback_tile_size),
                "--overlap",
                str(run_config.fallback_overlap),
            ]
        )
        if run_config.enable_tile_vae:
            command.append("--tile-vae")
    return command


def build_flashvsr_command(
    *,
    run_config: FlashVsrRunConfig,
    task: FlashVsrVideoTask,
    tiled: bool,
) -> list[str]:
    """根据 runner 类型构造最终执行命令."""

    if run_config.runner == "docker":
        container_input_path = f"/data/input/{task.relative_source_path.as_posix()}"
        relative_output_path = task.output_video_path.relative_to(run_config.output_root).as_posix()
        container_output_path = f"/data/output/{relative_output_path}"
        infer_args = _build_flashvsr_infer_args(
            input_path=container_input_path,
            output_path=container_output_path,
            run_config=run_config,
            tiled=tiled,
        )
        return [
            "docker",
            "run",
            "--rm",
            "--gpus",
            run_config.docker_gpus,
            "-v",
            f"{run_config.flashvsr_repo.resolve()}:/workspace/FlashVSR-Pro",
            "-v",
            f"{run_config.input_root.resolve()}:/data/input:ro",
            "-v",
            f"{run_config.output_root.resolve()}:/data/output",
            run_config.docker_image,
            *infer_args,
        ]

    infer_args = _build_flashvsr_infer_args(
        input_path=str(task.source_path.resolve()),
        output_path=str(task.output_video_path.resolve()),
        run_config=run_config,
        tiled=tiled,
    )
    infer_args[0] = run_config.local_python
    return infer_args


def _write_command_log(
    *,
    task: FlashVsrVideoTask,
    command: list[str],
    stdout: str,
    stderr: str,
    tiled: bool,
) -> Path:
    """把单次命令的 stdout / stderr 落盘."""

    task.debug_dir.mkdir(parents=True, exist_ok=True)
    log_name = "flashvsr_tiled.log" if tiled else "flashvsr_full.log"
    log_path = task.debug_dir / log_name
    payload = [
        "# command",
        " ".join(command),
        "",
        "# stdout",
        stdout.rstrip(),
        "",
        "# stderr",
        stderr.rstrip(),
        "",
    ]
    log_path.write_text("\n".join(payload), encoding="utf-8")
    return log_path


def _run_single_command(
    *,
    run_config: FlashVsrRunConfig,
    task: FlashVsrVideoTask,
    command: list[str],
    tiled: bool,
) -> FlashVsrCommandResult:
    """执行一次 `FlashVSR-Pro` 命令并返回完整结果."""

    if run_config.dry_run:
        stdout = "dry-run: command not executed"
        stderr = ""
        log_path = _write_command_log(task=task, command=command, stdout=stdout, stderr=stderr, tiled=tiled)
        return FlashVsrCommandResult(
            command=command,
            return_code=0,
            stdout=stdout,
            stderr=stderr,
            tiled=tiled,
            log_path=log_path,
        )

    completed = subprocess.run(
        command,
        cwd=run_config.flashvsr_repo if run_config.runner == "local" else None,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    log_path = _write_command_log(task=task, command=command, stdout=stdout, stderr=stderr, tiled=tiled)
    return FlashVsrCommandResult(
        command=command,
        return_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        tiled=tiled,
        log_path=log_path,
    )


def _looks_like_oom_failure(command_result: FlashVsrCommandResult) -> bool:
    """判断失败是否像是显存不足."""

    if command_result.succeeded:
        return False
    merged_output = f"{command_result.stdout}\n{command_result.stderr}".lower()
    return any(token in merged_output for token in OOM_HINTS)


def _list_frame_paths(frame_dir: Path) -> list[Path]:
    """列出帧目录里可读取的图片文件."""

    frame_paths = [
        path for path in sorted(frame_dir.iterdir()) if path.is_file() and path.suffix.lower() in SUPPORTED_FRAME_SUFFIXES
    ]
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found under {frame_dir}")
    return frame_paths


def _load_frame_count(video_or_dir: Path) -> int:
    """返回视频或帧目录的总帧数."""

    if video_or_dir.is_dir():
        return len(_list_frame_paths(video_or_dir))

    iio = _import_imageio_v3()
    frame_count = 0
    for frame_count, _frame in enumerate(iio.imiter(video_or_dir), start=1):
        pass
    if frame_count == 0:
        raise RuntimeError(f"No frames could be read from {video_or_dir}")
    return frame_count


def _choose_debug_frame_indices(
    *,
    frame_count: int,
    explicit_indices: list[int] | None,
    debug_every: int,
    dump_all: bool,
) -> list[int]:
    """确定本轮要导出的对照帧索引."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")

    if dump_all:
        return list(range(frame_count))

    if explicit_indices is not None:
        unique_indices = sorted({index for index in explicit_indices if 0 <= index < frame_count})
        if not unique_indices:
            raise ValueError(
                f"Explicit debug frame indices {explicit_indices} do not overlap valid range [0, {frame_count - 1}]."
            )
        return unique_indices

    if debug_every <= 0:
        return [0, frame_count - 1] if frame_count > 1 else [0]

    chosen_indices = list(range(0, frame_count, debug_every))
    last_index = frame_count - 1
    if chosen_indices[-1] != last_index:
        chosen_indices.append(last_index)
    return chosen_indices


def _import_imageio_v3():
    """惰性导入 `imageio.v3`, 让 dry-run 不依赖视频后端."""

    try:
        import imageio.v3 as iio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Extracting debug frames requires `imageio`. "
            "Please use the pixi environment, or install `imageio` in the current python environment."
        ) from exc
    return iio


def _load_selected_frames(video_or_dir: Path, selected_indices: list[int]) -> dict[int, np.ndarray]:
    """按指定帧索引读取图像,避免把整段视频一次性塞进内存."""

    selected_index_set = set(selected_indices)
    selected_frames: dict[int, np.ndarray] = {}

    if video_or_dir.is_dir():
        frame_paths = _list_frame_paths(video_or_dir)
        for frame_index in selected_indices:
            if frame_index >= len(frame_paths):
                raise IndexError(f"Frame index {frame_index} is out of range for directory {video_or_dir}.")
            selected_frames[frame_index] = np.asarray(Image.open(frame_paths[frame_index]).convert("RGB"))
        return selected_frames

    iio = _import_imageio_v3()
    for frame_index, frame in enumerate(iio.imiter(video_or_dir)):
        if frame_index not in selected_index_set:
            continue
        selected_frames[frame_index] = np.asarray(frame, dtype=np.uint8)
        if len(selected_frames) == len(selected_index_set):
            break

    missing_indices = [index for index in selected_indices if index not in selected_frames]
    if missing_indices:
        raise IndexError(f"Failed to load requested frame indices {missing_indices} from {video_or_dir}.")
    return selected_frames


def _save_debug_frame(image_array: np.ndarray, output_path: Path) -> None:
    """保存单帧 RGB 图像."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array, mode="RGB").save(output_path)


def _build_compare_strip(native_frame: np.ndarray, sr_frame: np.ndarray) -> np.ndarray:
    """构造一个左侧 bicubic-native,右侧 SR 的对照条图."""

    sr_image = Image.fromarray(sr_frame, mode="RGB")
    native_bicubic = Image.fromarray(native_frame, mode="RGB").resize(sr_image.size, resample=Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (sr_image.width * 2, sr_image.height))
    canvas.paste(native_bicubic, (0, 0))
    canvas.paste(sr_image, (sr_image.width, 0))
    return np.asarray(canvas, dtype=np.uint8)


def export_debug_frames(
    *,
    source_video: Path,
    sr_video: Path,
    debug_dir: Path,
    explicit_indices: list[int] | None,
    debug_every: int,
    dump_all: bool,
) -> dict[str, object]:
    """导出 native / SR / compare 三组逐帧对照图."""

    native_frame_count = _load_frame_count(source_video)
    sr_frame_count = _load_frame_count(sr_video)
    selected_indices = _choose_debug_frame_indices(
        frame_count=min(native_frame_count, sr_frame_count),
        explicit_indices=explicit_indices,
        debug_every=debug_every,
        dump_all=dump_all,
    )

    native_frames = _load_selected_frames(source_video, selected_indices)
    sr_frames = _load_selected_frames(sr_video, selected_indices)

    native_frame_dir = debug_dir / "native_frames"
    sr_frame_dir = debug_dir / "sr_frames"
    compare_frame_dir = debug_dir / "compare_frames"

    for frame_index in selected_indices:
        frame_name = f"frame_{frame_index:04d}.png"
        _save_debug_frame(native_frames[frame_index], native_frame_dir / frame_name)
        _save_debug_frame(sr_frames[frame_index], sr_frame_dir / frame_name)
        compare_strip = _build_compare_strip(native_frames[frame_index], sr_frames[frame_index])
        _save_debug_frame(compare_strip, compare_frame_dir / frame_name)

    return {
        "native_frame_count": native_frame_count,
        "sr_frame_count": sr_frame_count,
        "selected_indices": selected_indices,
        "native_frame_dir": str(native_frame_dir),
        "sr_frame_dir": str(sr_frame_dir),
        "compare_frame_dir": str(compare_frame_dir),
    }


def write_manifest(task: FlashVsrVideoTask, payload: dict[str, object]) -> None:
    """把单个视频任务的结果摘要保存成 JSON."""

    task.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    task.manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def process_video_task(
    *,
    run_config: FlashVsrRunConfig,
    task: FlashVsrVideoTask,
) -> dict[str, object]:
    """执行一个视频任务,必要时自动从 full 回退到 tiled."""

    task.output_video_path.parent.mkdir(parents=True, exist_ok=True)
    task.debug_dir.mkdir(parents=True, exist_ok=True)

    skipped = task.output_video_path.exists() and not run_config.overwrite
    attempt_results: list[FlashVsrCommandResult] = []
    debug_summary: dict[str, object] | None = None

    if skipped:
        status = "skipped_existing"
    else:
        full_command = build_flashvsr_command(run_config=run_config, task=task, tiled=False)
        attempt_results.append(
            _run_single_command(
                run_config=run_config,
                task=task,
                command=full_command,
                tiled=False,
            )
        )

        needs_fallback = (
            run_config.enable_fallback_tiling
            and _looks_like_oom_failure(attempt_results[-1])
            and run_config.mode == "full"
        )
        if needs_fallback:
            tiled_command = build_flashvsr_command(run_config=run_config, task=task, tiled=True)
            attempt_results.append(
                _run_single_command(
                    run_config=run_config,
                    task=task,
                    command=tiled_command,
                    tiled=True,
                )
            )

        last_result = attempt_results[-1]
        status = "succeeded" if last_result.succeeded else "failed"
        if not last_result.succeeded:
            summary = build_task_summary(task=task, status=status, debug_summary=None, attempt_results=attempt_results)
            write_manifest(task, summary)
            raise RuntimeError(
                "FlashVSR-Pro inference failed. "
                f"view={task.view_id}, scene={task.scene_stem}, log={last_result.log_path}"
            )

    if task.output_video_path.exists():
        debug_summary = export_debug_frames(
            source_video=task.source_path,
            sr_video=task.output_video_path,
            debug_dir=task.debug_dir,
            explicit_indices=run_config.debug_frame_indices,
            debug_every=run_config.debug_every,
            dump_all=run_config.dump_all_debug_frames,
        )

    summary = build_task_summary(task=task, status=status, debug_summary=debug_summary, attempt_results=attempt_results)
    write_manifest(task, summary)
    return summary


def build_task_summary(
    *,
    task: FlashVsrVideoTask,
    status: str,
    debug_summary: dict[str, object] | None,
    attempt_results: list[FlashVsrCommandResult],
) -> dict[str, object]:
    """构造单任务 JSON 摘要."""

    return {
        "status": status,
        "view_id": task.view_id,
        "scene_stem": task.scene_stem,
        "source_path": str(task.source_path),
        "output_video_path": str(task.output_video_path),
        "debug_dir": str(task.debug_dir),
        "debug_summary": debug_summary,
        "attempts": [
            {
                "tiled": attempt_result.tiled,
                "return_code": attempt_result.return_code,
                "log_path": str(attempt_result.log_path),
                "command": attempt_result.command,
            }
            for attempt_result in attempt_results
        ],
    }
