"""最小状态保存与恢复逻辑."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .gaussian_adapter import GaussianAdapter


def save_state(
    state_dir: Path,
    stage_name: str,
    iter_idx: int,
    gaussians: GaussianAdapter,
    diagnostics_state: dict[str, Any],
    pose_delta: torch.Tensor | None = None,
) -> Path:
    """把当前最小可恢复状态保存到 `latest.pt`."""

    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "latest.pt"

    payload = {
        "stage_name": stage_name,
        "iter_idx": iter_idx,
        "gaussians": gaussians.to_tensor().detach().cpu(),
        "diagnostics_state": diagnostics_state,
        "pose_delta": pose_delta.detach().cpu() if pose_delta is not None else None,
    }
    torch.save(payload, state_path)
    return state_path


def load_latest_state(state_dir: Path) -> dict[str, Any] | None:
    """从 `latest.pt` 读取状态,不存在时返回 `None`."""

    state_path = Path(state_dir) / "latest.pt"
    if not state_path.exists():
        return None
    return torch.load(state_path, map_location="cpu")
