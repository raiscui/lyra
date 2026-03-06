"""状态读写测试."""

from src.refinement_v2.state_io import load_latest_state, save_state
from tests.refinement_v2.helpers import build_gaussian_adapter


def test_save_and_load_latest_state(tmp_path) -> None:
    adapter = build_gaussian_adapter()
    state_path = save_state(
        state_dir=tmp_path / "state",
        stage_name="stage2a",
        iter_idx=3,
        gaussians=adapter,
        diagnostics_state={"phase_reached": "stage2a"},
    )

    payload = load_latest_state(tmp_path / "state")

    assert state_path.exists()
    assert payload is not None
    assert payload["stage_name"] == "stage2a"
    assert payload["iter_idx"] == 3
