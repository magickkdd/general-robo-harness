"""VLM brain wiring against a mocked OpenAI client (no network, no cost)."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.brain.vlm_api import VLMBrain


class _FakeResp:
    class _U:
        prompt_tokens = 11
        completion_tokens = 7
    usage = _U()

    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return _FakeResp(self.replies.pop(0))


class _FakeClient:
    def __init__(self, replies):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(replies)})()


def _brain(replies):
    b = VLMBrain.__new__(VLMBrain)
    b.name = "fake"
    b.client = _FakeClient(replies)
    b.model = "fake-model"
    b.temperature = 0.0
    b.max_tokens = 10
    b.max_repair = 1
    b.thinking = True
    b.send_images = True
    return b


def _obs():
    from harness.envs.base import Obs
    return Obs(rgb=np.zeros((8, 8, 3), dtype=np.uint8), state_text="ee at origin",
               instruction="pick the cube")


def test_call_structure_and_parse():
    b = _brain(['{"dx": 0.1, "dy": 0, "dz": 0.2, "skill": "none", "object": ""}'])
    out = b.act(_obs(), 0, "", 0.0)
    kw = b.client.chat.completions.calls[0]
    assert kw["model"] == "fake-model"
    msg = kw["messages"][1]
    assert msg["content"][0]["type"] == "image_url"
    assert msg["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "pick the cube" in msg["content"][1]["text"]
    assert out.skill == "none" and out.move == (0.1, 0.0, 0.2)
    assert out.usage.tokens_in == 11 and out.usage.brain_calls == 1


def test_skill_parse():
    b = _brain(['{"dx": 0, "dy": 0, "dz": 0, "skill": "pick", "object": "cube_red"}'])
    out = b.act(_obs(), 0, "", 0.0)
    assert out.skill == "pick" and out.arg == "cube_red"


def test_repair_retry_on_garbage_then_noop_fallback():
    b = _brain(["garbage", "still garbage"])
    out = b.act(_obs(), 0, "", 0.0)
    assert len(b.client.chat.completions.calls) == 2  # 1 + max_repair
    assert out.skill == "none" and out.move == (0.0, 0.0, 0.0)  # no-op fallback
    assert "fallback" in out.raw_text
