"""Brain registry. vlm_api covers any OpenAI-compatible endpoint; random for smoke."""
from __future__ import annotations

import numpy as np

from .base import Brain, BrainOutput, Usage  # noqa: F401
from .vlm_api import VLMBrain, load_api_key


class RandomBrain(Brain):
    """Uniform random canonical actions; for pipeline smoke tests, zero cost."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs, step_idx, last_action, last_reward) -> BrainOutput:
        names = [r.name for r in obs.readings if r.position is not None]
        if names and self.rng.random() < 0.3:
            return BrainOutput(skill=str(self.rng.choice(["pick", "place"])),
                               arg=str(self.rng.choice(names)))
        mv = self.rng.uniform(-0.5, 0.5, 3)
        return BrainOutput(skill="none", move=tuple(float(v) for v in mv))


def make_brain(kind: str, **kw) -> Brain:
    if kind == "random":
        return RandomBrain(seed=kw.get("seed", 0))
    if kind == "vlm":
        return VLMBrain(
            model=kw["model"],
            base_url=kw["base_url"],
            api_key=kw["api_key"],
            temperature=kw.get("temperature", 0.0),
            max_tokens=kw.get("max_tokens", 8192),
            timeout_s=kw.get("timeout_s", 240.0),
            max_repair=kw.get("max_repair", 1),
            thinking=kw.get("thinking", True),
            send_images=kw.get("send_images", "all"),
        )
    raise ValueError(f"unknown brain kind {kind}; known: random, vlm")
