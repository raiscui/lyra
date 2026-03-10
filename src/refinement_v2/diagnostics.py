"""诊断产物与可视化输出逻辑."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


class DiagnosticsWriter:
    """统一管理 refinement 的指标与可视化产物输出."""

    def __init__(self, outdir: Path) -> None:
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.residual_dir = self.outdir / "residual_maps"
        self.weight_dir = self.outdir / "weight_maps"
        self.sr_selection_dir = self.outdir / "sr_selection_maps"
        self.render_dir = self.outdir / "renders_before_after"
        self.histogram_dir = self.outdir / "histograms"
        self.pose_dir = self.outdir / "pose"
        self.prune_dir = self.outdir / "pruning"
        self.state_dir = self.outdir / "state"
        self.gaussian_dir = self.outdir / "gaussians"
        self.video_dir = self.outdir / "videos"

        for directory in [
            self.residual_dir,
            self.weight_dir,
            self.sr_selection_dir,
            self.render_dir,
            self.histogram_dir,
            self.pose_dir,
            self.prune_dir,
            self.state_dir,
            self.gaussian_dir,
            self.video_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self.history: list[dict[str, Any]] = []
        self.stage_history: dict[str, list[dict[str, Any]]] = {}
        self.prune_history: list[dict[str, Any]] = []

    def log_stage_metrics(self, stage_name: str, metrics: dict[str, Any]) -> None:
        """记录某个 stage 的单次指标."""

        stage_metrics = {key: self._to_jsonable(value) for key, value in dict(metrics).items()}
        stage_metrics["stage_name"] = stage_name
        self.history.append(stage_metrics)
        self.stage_history.setdefault(stage_name, []).append(stage_metrics)

        metrics_path = self.outdir / f"metrics_{stage_name}.json"
        metrics_path.write_text(
            json.dumps(self.stage_history[stage_name], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_weight_map(self, stage_name: str, frame_id: int, weight_map: torch.Tensor) -> None:
        """保存权重图 PNG."""

        output_path = self.weight_dir / f"{stage_name}_frame_{frame_id:04d}.png"
        self._save_tensor_map(weight_map, output_path)

    def save_residual_map(self, stage_name: str, frame_id: int, residual_map: torch.Tensor) -> None:
        """保存 residual 图 PNG."""

        output_path = self.residual_dir / f"{stage_name}_frame_{frame_id:04d}.png"
        self._save_tensor_map(residual_map, output_path)

    def save_sr_selection_map(self, stage_name: str, frame_id: int, selection_map: torch.Tensor) -> None:
        """保存 SR selection map PNG."""

        output_path = self.sr_selection_dir / f"{stage_name}_frame_{frame_id:04d}.png"
        self._save_tensor_map(selection_map, output_path)

    def write_gaussian_fidelity_summary(
        self,
        fidelity_score: torch.Tensor,
        fidelity_diagnostics: dict[str, torch.Tensor] | None = None,
    ) -> Path:
        """把 fidelity score 的摘要和直方图落盘.

        第一版先保存:
        - 基础统计
        - 固定 10 bins 直方图
        这样后续接真正的 `W_sr_select` 时, 先有稳定诊断证据可看.
        """

        score_cpu = fidelity_score.detach().float().cpu().flatten()
        if score_cpu.numel() == 0:
            raise ValueError("fidelity_score must contain at least one element.")

        histogram = torch.histc(score_cpu, bins=10, min=0.0, max=1.0).to(dtype=torch.int64)
        payload = {
            "num_gaussians": int(score_cpu.numel()),
            "fidelity_min": float(score_cpu.min().item()),
            "fidelity_max": float(score_cpu.max().item()),
            "fidelity_mean": float(score_cpu.mean().item()),
            "fidelity_std": float(score_cpu.std(unbiased=False).item()),
            "bins": histogram.tolist(),
        }

        if isinstance(fidelity_diagnostics, dict):
            rho = fidelity_diagnostics.get("rho")
            num_times_seen = fidelity_diagnostics.get("num_times_seen")
            max_view_mask = fidelity_diagnostics.get("max_view_mask")
            if isinstance(rho, torch.Tensor):
                rho_cpu = rho.detach().float().cpu()
                payload.update(
                    {
                        "rho_min": float(rho_cpu.min().item()),
                        "rho_max": float(rho_cpu.max().item()),
                        "rho_mean": float(rho_cpu.mean().item()),
                        "rho_std": float(rho_cpu.std(unbiased=False).item()),
                    }
                )
            if isinstance(num_times_seen, torch.Tensor):
                seen_cpu = num_times_seen.detach().float().cpu()
                payload.update(
                    {
                        "num_times_seen_min": float(seen_cpu.min().item()),
                        "num_times_seen_max": float(seen_cpu.max().item()),
                        "num_times_seen_mean": float(seen_cpu.mean().item()),
                    }
                )
            if isinstance(max_view_mask, torch.Tensor):
                max_view_cpu = max_view_mask.detach().float().cpu()
                payload["max_view_counts"] = max_view_cpu.sum(dim=(0, 2)).to(dtype=torch.int64).tolist()

        output_path = self.outdir / "gaussian_fidelity_histogram.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def write_sr_selection_stats(self, selection_map: torch.Tensor) -> Path:
        """把 SR selection map 的基础统计落盘."""

        selection_cpu = selection_map.detach().float().cpu()
        payload = {
            "selection_min": float(selection_cpu.min().item()),
            "selection_max": float(selection_cpu.max().item()),
            "selection_mean": float(selection_cpu.mean().item()),
            "selection_std": float(selection_cpu.std(unbiased=False).item()),
        }
        output_path = self.outdir / "sr_selection_stats.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def save_pose_summary(self, summary: dict[str, Any]) -> None:
        """保存 pose 诊断摘要."""

        pose_path = self.pose_dir / "pose_delta_summary.json"
        pose_path.write_text(
            json.dumps({key: self._to_jsonable(value) for key, value in summary.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_prune_summary(self, iteration: int, summary: dict[str, Any]) -> Path:
        """保存单次 pruning 摘要,同时维护一份聚合历史.

        单次文件适合快速定位“第几轮删了多少”.
        聚合文件适合后处理脚本整体读取.
        """

        payload = {"iteration": iteration}
        payload.update({key: self._to_jsonable(value) for key, value in summary.items()})
        self.prune_history.append(payload)

        prune_path = self.prune_dir / f"prune_iter_{iteration:04d}.json"
        prune_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary_path = self.prune_dir / "pruning_summary.json"
        summary_path.write_text(
            json.dumps(self.prune_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return prune_path

    def save_render_video(self, name: str, rgb_tensor: torch.Tensor, fps: int = 8) -> Path:
        """把渲染结果导出为 mp4 视频.

        这里统一接受 `[B, T, C, H, W]` 或 `[T, C, H, W]`.
        refinement 主线后续只需要调用这一层,不用再关心视频编码细节.
        """

        output_path = self.video_dir / f"{name}.mp4"
        video_frames = self._tensor_to_video_frames(rgb_tensor)
        self._write_video(output_path, video_frames, fps=fps)
        return output_path

    def save_render_snapshot(self, name: str, rgb_tensor: torch.Tensor, frame_id: int = 0) -> Path:
        """导出某个渲染序列的单帧快照.

        这类静态快照适合快速肉眼浏览.
        不需要每次都打开 mp4.
        """

        output_path = self.render_dir / f"{name}_frame_{frame_id:04d}.png"
        frame_uint8 = self._extract_video_frame(rgb_tensor, frame_id)
        Image.fromarray(frame_uint8, mode="RGB").save(output_path)
        return output_path

    def finalize(self, summary: dict[str, Any]) -> None:
        """写出最终的 diagnostics 和聚合 metrics."""

        diagnostics_path = self.outdir / "diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(self._to_jsonable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        metrics_path = self.outdir / "metrics.json"
        metrics_path.write_text(
            json.dumps(self._to_jsonable(self.history), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_tensor_map(self, tensor: torch.Tensor, output_path: Path) -> None:
        """把 2D/3D tensor 归一化后保存为单通道 PNG."""

        tensor_cpu = tensor.detach().float().cpu()
        while tensor_cpu.ndim > 2:
            tensor_cpu = tensor_cpu[0]
        if tensor_cpu.ndim != 2:
            raise ValueError(f"Expected a 2D tensor map, got {tuple(tensor.shape)}")

        array = tensor_cpu.numpy()
        array = array - array.min()
        max_value = array.max()
        if max_value > 0:
            array = array / max_value
        array_uint8 = (array * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(array_uint8, mode="L").save(output_path)

    def _tensor_to_video_frames(self, tensor: torch.Tensor) -> np.ndarray:
        """把 RGB tensor 规范成 `imageio` 可写入的视频帧数组."""

        tensor_cpu = tensor.detach().float().cpu()
        if tensor_cpu.ndim == 5:
            tensor_cpu = tensor_cpu[0]
        elif tensor_cpu.ndim == 3:
            tensor_cpu = tensor_cpu.unsqueeze(0)

        if tensor_cpu.ndim != 4:
            raise ValueError(f"Expected RGB video tensor with 3/4/5 dims, got {tuple(tensor.shape)}")
        if tensor_cpu.shape[1] != 3:
            raise ValueError(f"Expected RGB video tensor in [T, 3, H, W], got {tuple(tensor_cpu.shape)}")

        tensor_cpu = tensor_cpu.clamp(0.0, 1.0)
        video_np = tensor_cpu.permute(0, 2, 3, 1).numpy()
        return (video_np * 255.0).round().clip(0, 255).astype(np.uint8)

    def _extract_video_frame(self, tensor: torch.Tensor, frame_id: int = 0) -> np.ndarray:
        """从 RGB 视频 tensor 中取出某一帧,返回 `uint8` 图像."""

        video_frames = self._tensor_to_video_frames(tensor)
        if frame_id < 0 or frame_id >= video_frames.shape[0]:
            raise IndexError(f"frame_id {frame_id} is out of range for {video_frames.shape[0]} frames.")
        return video_frames[frame_id]

    def _write_video(self, output_path: Path, video_frames: np.ndarray, fps: int = 8) -> None:
        """把 `uint8[T,H,W,3]` 帧数组写成 mp4.

        优先使用 `imageio`.
        如果当前解释器没有 `imageio`,就回退到系统 `ffmpeg`.
        """

        try:
            import imageio

            imageio.mimwrite(output_path, video_frames, fps=fps)
            return
        except ModuleNotFoundError:
            pass

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError("Saving mp4 requires either `imageio` or a system `ffmpeg` binary.")

        _, height, width, channels = video_frames.shape
        if channels != 3:
            raise ValueError(f"Expected RGB video frames with 3 channels, got {channels}.")

        # 直接通过 stdin 喂 rawvideo.
        # 这样不用在磁盘上先落一堆临时 PNG,测试和真实运行都更干净.
        # 不同环境的 ffmpeg 可用编码器不完全一致.
        # 因此这里按顺序尝试,优先 `libx264`,再退到更常见的 `mpeg4`.
        encoder_candidates = ["libx264", "mpeg4"]
        last_error = ""
        for encoder_name in encoder_candidates:
            command = [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                encoder_name,
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]

            result = subprocess.run(
                command,
                input=video_frames.tobytes(),
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                return

            last_error = result.stderr.decode("utf-8", errors="ignore").strip()

        raise RuntimeError(f"ffmpeg failed to encode video: {last_error}")

    def _to_jsonable(self, value: Any) -> Any:
        """把常见 tensor / ndarray 递归转成 JSON 可序列化对象."""

        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(item) for item in value]
        return value
