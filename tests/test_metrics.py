from harness.metrics import aggregate, totals
from harness.runner import EpisodeResult


def _r(status="success", steps=10, embodiment="panda", task="PickCube-v1", tok=5):
    return EpisodeResult(backend="maniskill", task=task, embodiment=embodiment, seed=0,
                         status=status, steps=steps, brain_calls=steps, tokens_in=tok)


def test_aggregate_rates():
    rs = [_r(), _r("max_steps"), _r(status="success", task="PushCube-v1")]
    agg = aggregate(rs)
    assert agg["panda/PickCube-v1"]["success_rate"] == 0.5
    assert agg["panda/PickCube-v1"]["episodes"] == 2
    assert agg["panda/PushCube-v1"]["successes"] == 1


def test_totals_counts_statuses():
    t = totals([_r(), _r("error")])
    assert t["statuses"] == {"success": 1, "max_steps": 0, "budget": 0, "error": 1}
    assert t["success_rate"] == 0.5


def test_empty():
    assert totals([])["episodes"] == 0
