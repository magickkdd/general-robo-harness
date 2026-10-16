"""One real episode through the whole stack with the zero-cost RandomBrain."""
from pathlib import Path
import tempfile

import pytest

from harness.brain import make_brain
from harness.envs import get_adapter, make_cell
from harness.runner import Budget, run_batch


@pytest.mark.slow
def test_random_episode_runs_to_completion():
    out = Path(tempfile.mkdtemp())
    res = run_batch(lambda: get_adapter("maniskill"),
                    [make_cell("PickCube-v1", "panda")],
                    lambda: make_brain("random"), [0],
                    out_dir=out, budget=Budget(max_brain_calls=400))
    assert len(res) == 1
    r = res[0]
    assert r.status in ("success", "max_steps")
    assert r.steps > 0
    assert (out / "episodes" / "panda_PickCube-v1_0.json").exists()
    from harness.report import write_report
    assert write_report(res, out).exists()
