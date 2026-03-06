"""`opacity/pruning` 相关测试."""

from __future__ import annotations

import json

import torch

from src.refinement_v2.diagnostics import DiagnosticsWriter
from src.refinement_v2.runner import RefinementRunner
from src.refinement_v2.stage_controller import StageController
from tests.refinement_v2.helpers import FakeRenderer, build_gaussian_adapter, build_run_config, build_scene_bundle, build_stage_hparams


def _build_adapter_with_opacity(opacity_values: list[float]):
    """构造带指定 opacity 的高斯适配器."""

    adapter = build_gaussian_adapter(num_points=len(opacity_values))
    with torch.no_grad():
        adapter.opacity[:, 0] = torch.tensor(opacity_values, dtype=adapter.opacity.dtype)
    return adapter


def test_collect_prune_candidates_marks_low_opacity_gaussians() -> None:
    """低 opacity 高斯应该先被识别出来,再决定是否真正裁剪."""

    adapter = _build_adapter_with_opacity([0.9, 0.01, 0.02, 0.5])
    candidates = adapter.collect_prune_candidates(threshold=0.05)

    assert candidates.tolist() == [False, True, True, False]


def test_prune_low_opacity_respects_max_fraction() -> None:
    """一次 pruning 不能超过设定比例上限."""

    adapter = _build_adapter_with_opacity([0.9, 0.01, 0.02, 0.03])
    summary = adapter.prune_low_opacity(
        threshold=0.05,
        max_fraction=0.25,
        min_gaussians_to_keep=1,
    )

    assert summary["candidate_count"] == 3
    assert summary["pruned_count"] == 1
    assert summary["num_after"] == 3
    assert adapter.means.shape[0] == 3
    assert torch.allclose(adapter.opacity.detach().squeeze(-1), torch.tensor([0.9, 0.02, 0.03]))


def test_prune_low_opacity_respects_minimum_keep() -> None:
    """保护线要生效,不能把高斯裁到空."""

    adapter = _build_adapter_with_opacity([0.001, 0.002, 0.003])
    summary = adapter.prune_low_opacity(
        threshold=0.05,
        max_fraction=1.0,
        min_gaussians_to_keep=2,
    )

    assert summary["pruned_count"] == 1
    assert summary["num_after"] == 2


def test_stage2a_pruning_reduces_gaussian_count_and_writes_summary(tmp_path) -> None:
    """runner 开启 pruning 后,要真的裁掉高斯并留下诊断证据."""

    run_config = build_run_config(tmp_path, enable_pruning=True)
    hparams = build_stage_hparams(
        iters_stage2a=2,
        prune_warmup_iters=0,
        prune_every=1,
        prune_max_fraction=0.5,
        opacity_prune_threshold=0.05,
        min_gaussians_to_keep=1,
    )
    gaussians = _build_adapter_with_opacity([0.9, 0.01, 0.02, 0.03, 0.6, 0.7])

    runner = RefinementRunner(
        scene=build_scene_bundle(),
        gaussians=gaussians,
        diagnostics=DiagnosticsWriter(run_config.outdir),
        controller=StageController(run_config, hparams),
        hparams=hparams,
        renderer=FakeRenderer(),
    )

    runner.run_phase0()
    runner.run_phase1_prepare_weights()
    runner.run_stage2a()

    prune_summary_path = run_config.outdir / "pruning" / "pruning_summary.json"
    prune_history = json.loads(prune_summary_path.read_text(encoding="utf-8"))

    assert gaussians.means.shape[0] == 3
    assert prune_summary_path.exists()
    assert prune_history[0]["pruned_count"] == 3
