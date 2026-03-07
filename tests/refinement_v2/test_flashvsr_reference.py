"""`FlashVSR-Pro` reference 生成链路测试."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.refinement_v2.flashvsr_reference import (
    FlashVsrCommandResult,
    FlashVsrRunConfig,
    FlashVsrVideoTask,
    _choose_debug_frame_indices,
    _looks_like_oom_failure,
    build_flashvsr_command,
    build_run_tag,
    discover_source_videos,
    ensure_flashvsr_models,
    ensure_flashvsr_repo_layout,
    export_debug_frames,
    plan_video_tasks,
)


def _touch(path: Path) -> None:
    """创建一个空文件,方便测试目录扫描逻辑."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _build_fake_flashvsr_repo(repo_root: Path, mode: str = "full") -> Path:
    """构造一个最小可验证的 `FlashVSR-Pro` 假仓库."""

    _touch(repo_root / "infer.py")
    _touch(repo_root / "Dockerfile")
    model_root = repo_root / "models" / "FlashVSR-v1.1"
    _touch(model_root / "diffusion_pytorch_model_streaming_dmd.safetensors")
    _touch(model_root / "LQ_proj_in.ckpt")
    if mode == "full":
        _touch(model_root / "Wan2.1_VAE.pth")
    else:
        _touch(model_root / "TCDecoder.ckpt")
    return repo_root


def _write_frame_dir(frame_dir: Path, values: list[int], hw: tuple[int, int]) -> None:
    """写一个小型帧目录,用于测试逐帧导出."""

    frame_dir.mkdir(parents=True, exist_ok=True)
    height, width = hw
    for frame_index, value in enumerate(values):
        image = np.full((height, width, 3), value, dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(frame_dir / f"{frame_index:04d}.png")


def test_discover_source_videos_filters_by_view_and_scene(tmp_path) -> None:
    input_root = tmp_path / "diffusion_output_generated"
    _touch(input_root / "0" / "rgb" / "00172.mp4")
    _touch(input_root / "1" / "rgb" / "00172.mp4")
    _touch(input_root / "1" / "rgb" / "00854.mp4")

    source_paths = discover_source_videos(input_root=input_root, view_ids=["1"], scene_stem="00172")

    assert source_paths == [input_root / "1" / "rgb" / "00172.mp4"]


def test_plan_video_tasks_preserves_view_layout(tmp_path) -> None:
    input_root = tmp_path / "diffusion_output_generated"
    source_path = input_root / "3" / "rgb" / "00172.mp4"
    _touch(source_path)

    tasks = plan_video_tasks(
        input_root=input_root,
        output_root=tmp_path / "outputs",
        source_paths=[source_path],
        mode="full",
        scale=2.0,
    )

    assert len(tasks) == 1
    assert tasks[0].output_video_path == tmp_path / "outputs" / "full_scale2x" / "3" / "rgb" / "00172.mp4"
    assert tasks[0].manifest_path == tmp_path / "outputs" / "full_scale2x" / "3" / "manifests" / "00172.json"


def test_build_flashvsr_command_for_docker_mounts_expected_roots(tmp_path) -> None:
    repo_root = _build_fake_flashvsr_repo(tmp_path / "FlashVSR-Pro")
    input_root = tmp_path / "diffusion_output_generated"
    source_path = input_root / "5" / "rgb" / "00172.mp4"
    _touch(source_path)

    run_config = FlashVsrRunConfig(
        input_root=input_root,
        output_root=tmp_path / "outputs",
        flashvsr_repo=repo_root,
        runner="docker",
        mode="full",
    )
    task = plan_video_tasks(
        input_root=input_root,
        output_root=run_config.output_root,
        source_paths=[source_path],
        mode=run_config.mode,
        scale=run_config.scale,
    )[0]

    command = build_flashvsr_command(run_config=run_config, task=task, tiled=True)

    assert command[:4] == ["docker", "run", "--rm", "--gpus"]
    assert "/data/input/5/rgb/00172.mp4" in command
    assert "/data/output/full_scale2x/5/rgb/00172.mp4" in command
    assert "--tile-dit" in command
    assert "--tile-size" in command
    assert "512" in command
    assert "128" in command


def test_build_flashvsr_command_for_local_uses_selected_python(tmp_path) -> None:
    repo_root = _build_fake_flashvsr_repo(tmp_path / "FlashVSR-Pro", mode="tiny")
    input_root = tmp_path / "diffusion_output_generated"
    source_path = input_root / "2" / "rgb" / "00172.mp4"
    _touch(source_path)

    run_config = FlashVsrRunConfig(
        input_root=input_root,
        output_root=tmp_path / "outputs",
        flashvsr_repo=repo_root,
        runner="local",
        local_python="python3.11",
        mode="tiny",
    )
    task = plan_video_tasks(
        input_root=input_root,
        output_root=run_config.output_root,
        source_paths=[source_path],
        mode=run_config.mode,
        scale=run_config.scale,
    )[0]

    command = build_flashvsr_command(run_config=run_config, task=task, tiled=False)

    assert command[0] == "python3.11"
    assert command[1:3] == ["infer.py", "-i"]
    assert str(source_path.resolve()) in command
    assert str(task.output_video_path.resolve()) in command


def test_export_debug_frames_from_frame_directories(tmp_path) -> None:
    native_dir = tmp_path / "native_frames_src"
    sr_dir = tmp_path / "sr_frames_src"
    debug_dir = tmp_path / "debug"
    _write_frame_dir(native_dir, values=[10, 40, 70, 100], hw=(4, 6))
    _write_frame_dir(sr_dir, values=[20, 80, 140, 200], hw=(8, 12))

    summary = export_debug_frames(
        source_video=native_dir,
        sr_video=sr_dir,
        debug_dir=debug_dir,
        explicit_indices=[0, 2],
        debug_every=8,
        dump_all=False,
    )

    assert summary["selected_indices"] == [0, 2]
    assert (debug_dir / "native_frames" / "frame_0000.png").exists()
    assert (debug_dir / "sr_frames" / "frame_0002.png").exists()
    assert (debug_dir / "compare_frames" / "frame_0002.png").exists()

    compare_image = Image.open(debug_dir / "compare_frames" / "frame_0002.png")
    assert compare_image.size == (24, 8)


def test_choose_debug_frame_indices_keeps_last_frame() -> None:
    selected_indices = _choose_debug_frame_indices(
        frame_count=10,
        explicit_indices=None,
        debug_every=4,
        dump_all=False,
    )

    assert selected_indices == [0, 4, 8, 9]


def test_looks_like_oom_failure_detects_memory_errors(tmp_path) -> None:
    command_result = FlashVsrCommandResult(
        command=["python", "infer.py"],
        return_code=1,
        stdout="",
        stderr="RuntimeError: CUDA out of memory while running full mode",
        tiled=False,
        log_path=tmp_path / "flashvsr_full.log",
    )

    assert _looks_like_oom_failure(command_result) is True


def test_flashvsr_repo_validation_checks_required_files(tmp_path) -> None:
    repo_root = _build_fake_flashvsr_repo(tmp_path / "FlashVSR-Pro", mode="full")

    ensure_flashvsr_repo_layout(repo_root)
    ensure_flashvsr_models(repo_root, "full")


def test_build_run_tag_formats_scale_stably() -> None:
    assert build_run_tag("full", 2.0) == "full_scale2x"
    assert build_run_tag("full", 1.5) == "full_scale1p5x"
