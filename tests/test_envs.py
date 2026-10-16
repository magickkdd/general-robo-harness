import numpy as np
import pytest

from harness.envs import EMBODIMENTS, SUPPORTED, TASKS, make_cell, get_adapter
from harness.envs.base import clip_action


def test_registry_consistency():
    assert set(TASKS) and set(EMBODIMENTS)
    for e, t in SUPPORTED:
        assert e in EMBODIMENTS and t in TASKS


def test_make_cell_rejects_unknown():
    with pytest.raises(ValueError):
        make_cell("Nope-v1", "panda")
    with pytest.raises(ValueError):
        make_cell("PickCube-v1", "nope")


def test_clip_action():
    assert clip_action([2, -2, 0, 0.5]).tolist() == [1.0, -1.0, 0.0, 0.5]
    with pytest.raises(AssertionError):
        clip_action([1, 2, 3])


def test_get_adapter_unknown():
    with pytest.raises(ValueError):
        get_adapter("nope")
