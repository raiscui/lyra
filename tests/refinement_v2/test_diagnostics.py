"""诊断输出测试."""

import json

import torch

from src.refinement_v2.diagnostics import DiagnosticsWriter


def test_diagnostics_writer_outputs_expected_files(tmp_path) -> None:
    writer = DiagnosticsWriter(tmp_path)
    writer.log_stage_metrics("stage2a", {"loss_total": 1.0, "psnr": 20.0})
    writer.save_weight_map("stage2a", 0, torch.ones(1, 1, 1, 4, 4))
    writer.save_residual_map("stage2a", 0, torch.zeros(1, 1, 1, 4, 4))
    writer.save_render_video("baseline_render", torch.zeros(1, 2, 3, 4, 4))
    writer.save_render_snapshot("baseline_render", torch.zeros(1, 2, 3, 4, 4))
    writer.finalize({"phase_reached": "stage2a"})

    assert (tmp_path / "metrics_stage2a.json").exists()
    assert (tmp_path / "weight_maps" / "stage2a_frame_0000.png").exists()
    assert (tmp_path / "residual_maps" / "stage2a_frame_0000.png").exists()
    assert (tmp_path / "videos" / "baseline_render.mp4").exists()
    assert (tmp_path / "renders_before_after" / "baseline_render_frame_0000.png").exists()
    assert (tmp_path / "diagnostics.json").exists()

    summary = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    assert summary["phase_reached"] == "stage2a"


def test_diagnostics_writer_outputs_prune_summary(tmp_path) -> None:
    """确认 pruning 摘要会落到独立目录,便于后续肉眼核对."""

    writer = DiagnosticsWriter(tmp_path)
    writer.write_prune_summary(
        iteration=4,
        summary={
            "num_before": 8,
            "num_after": 6,
            "candidate_count": 3,
            "pruned_count": 2,
            "opacity_lowconf_ratio_before": 0.5,
            "opacity_lowconf_ratio_after": 0.1666667,
        },
    )

    prune_path = tmp_path / "pruning" / "prune_iter_0004.json"
    summary_path = tmp_path / "pruning" / "pruning_summary.json"

    assert prune_path.exists()
    assert summary_path.exists()

    payload = json.loads(prune_path.read_text(encoding="utf-8"))
    history = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["iteration"] == 4
    assert payload["pruned_count"] == 2
    assert history[0]["num_after"] == 6
