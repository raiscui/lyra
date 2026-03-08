"""MoGe 加载器与 v2 focal 校正回归测试."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

from cosmos_predict1.diffusion.inference.inference_utils import (
    load_moge_model,
    maybe_apply_moge_focal_correction,
)


class _FakeMoGeModel:
    """用最小假模型验证加载逻辑,避免测试依赖真实大权重."""

    def __init__(self, **model_config):
        self.model_config = model_config
        self.loaded_state_dict: dict | None = None
        self.loaded_strict: bool | None = None
        self.device = None
        self.eval_called = False

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.loaded_state_dict = state_dict
        self.loaded_strict = strict
        return None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def _install_fake_moge_modules(monkeypatch) -> None:
    """把 `moge.model.v1/v2` 替换成可观测的假实现."""

    moge_module = types.ModuleType("moge")
    moge_model_module = types.ModuleType("moge.model")
    moge_v1_module = types.ModuleType("moge.model.v1")
    moge_v2_module = types.ModuleType("moge.model.v2")
    moge_v1_module.MoGeModel = _FakeMoGeModel
    moge_v2_module.MoGeModel = _FakeMoGeModel

    monkeypatch.setitem(sys.modules, "moge", moge_module)
    monkeypatch.setitem(sys.modules, "moge.model", moge_model_module)
    monkeypatch.setitem(sys.modules, "moge.model.v1", moge_v1_module)
    monkeypatch.setitem(sys.modules, "moge.model.v2", moge_v2_module)


def _write_fake_checkpoint(path: Path, *, encoder_config) -> None:
    """按 v1 / v2 的结构约定落一个最小 checkpoint."""

    checkpoint = {
        "model_config": {
            "encoder": encoder_config,
            "decoder": "fake-decoder",
        },
        "model": {
            "weight": torch.tensor([1.0]),
        },
    }
    torch.save(checkpoint, path)


def test_load_moge_model_auto_detects_v1_from_local_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _install_fake_moge_modules(monkeypatch)
    checkpoint_path = tmp_path / "moge_v1.pt"
    _write_fake_checkpoint(checkpoint_path, encoder_config="vitl")

    moge_model, moge_version = load_moge_model(
        moge_version="auto",
        moge_model_id=None,
        moge_checkpoint_path=str(checkpoint_path),
        hf_local_files_only=True,
        device=torch.device("cpu"),
    )

    assert moge_version == "v1"
    assert isinstance(moge_model, _FakeMoGeModel)
    assert moge_model.loaded_strict is True
    assert moge_model.device == torch.device("cpu")
    assert moge_model.eval_called is True


def test_load_moge_model_auto_detects_v2_from_local_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _install_fake_moge_modules(monkeypatch)
    checkpoint_path = tmp_path / "moge_v2.pt"
    _write_fake_checkpoint(checkpoint_path, encoder_config={"name": "vitl"})

    moge_model, moge_version = load_moge_model(
        moge_version="auto",
        moge_model_id=None,
        moge_checkpoint_path=str(checkpoint_path),
        hf_local_files_only=True,
        device=torch.device("cpu"),
    )

    assert moge_version == "v2"
    assert isinstance(moge_model, _FakeMoGeModel)
    assert moge_model.loaded_strict is False
    assert moge_model.device == torch.device("cpu")
    assert moge_model.eval_called is True


def test_load_moge_model_rejects_requested_version_mismatch(monkeypatch, tmp_path: Path) -> None:
    _install_fake_moge_modules(monkeypatch)
    checkpoint_path = tmp_path / "moge_v2.pt"
    _write_fake_checkpoint(checkpoint_path, encoder_config={"name": "vitl"})

    try:
        load_moge_model(
            moge_version="v1",
            moge_model_id=None,
            moge_checkpoint_path=str(checkpoint_path),
            hf_local_files_only=True,
            device=torch.device("cpu"),
        )
    except ValueError as exc:
        assert "requested v1" in str(exc)
        assert "is v2" in str(exc)
    else:
        raise AssertionError("expected version mismatch to raise ValueError")


def test_load_moge_model_uses_built_in_default_repo_for_requested_version(monkeypatch, tmp_path: Path) -> None:
    _install_fake_moge_modules(monkeypatch)
    checkpoint_paths = {
        "Ruicheng/moge-vitl": tmp_path / "moge_v1.pt",
        "Ruicheng/moge-2-vitl": tmp_path / "moge_v2.pt",
    }
    _write_fake_checkpoint(checkpoint_paths["Ruicheng/moge-vitl"], encoder_config="vitl")
    _write_fake_checkpoint(checkpoint_paths["Ruicheng/moge-2-vitl"], encoder_config={"name": "vitl"})

    captured_repo_ids: list[str] = []

    huggingface_hub_module = types.ModuleType("huggingface_hub")

    def _fake_hf_hub_download(*, repo_id: str, repo_type: str, filename: str, local_files_only: bool):
        captured_repo_ids.append(repo_id)
        assert repo_type == "model"
        assert filename == "model.pt"
        assert local_files_only is True
        return str(checkpoint_paths[repo_id])

    huggingface_hub_module.hf_hub_download = _fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub_module)

    _, loaded_v1 = load_moge_model(
        moge_version="v1",
        moge_model_id=None,
        moge_checkpoint_path=None,
        hf_local_files_only=True,
        device=torch.device("cpu"),
    )
    _, loaded_auto = load_moge_model(
        moge_version="auto",
        moge_model_id=None,
        moge_checkpoint_path=None,
        hf_local_files_only=True,
        device=torch.device("cpu"),
    )

    assert loaded_v1 == "v1"
    assert loaded_auto == "v2"
    assert captured_repo_ids == [
        "Ruicheng/moge-vitl",
        "Ruicheng/moge-2-vitl",
    ]


def test_maybe_apply_moge_focal_correction_only_scales_v2_focal_terms() -> None:
    intrinsics = torch.tensor(
        [
            [790.0, 0.0, 640.0],
            [0.0, 773.0, 352.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    adjusted = maybe_apply_moge_focal_correction(
        intrinsics,
        loaded_moge_version="v2",
        moge_v2_focal_scale=1.04637,
    )

    assert torch.isclose(adjusted[0, 0], intrinsics[0, 0] * 1.04637)
    assert torch.isclose(adjusted[1, 1], intrinsics[1, 1] * 1.04637)
    assert adjusted[0, 2].item() == intrinsics[0, 2].item()
    assert adjusted[1, 2].item() == intrinsics[1, 2].item()
    assert intrinsics[0, 0].item() == 790.0
    assert intrinsics[1, 1].item() == 773.0


def test_maybe_apply_moge_focal_correction_keeps_v1_unchanged() -> None:
    intrinsics = torch.tensor(
        [
            [827.0, 0.0, 640.0],
            [0.0, 809.0, 352.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    adjusted = maybe_apply_moge_focal_correction(
        intrinsics,
        loaded_moge_version="v1",
        moge_v2_focal_scale=1.04637,
    )

    assert torch.equal(adjusted, intrinsics)
