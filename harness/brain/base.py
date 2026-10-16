"""Brain interface: anything that maps an observation to a canonical action."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..envs.base import Obs


@dataclass
class Usage:
    brain_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def add(self, other: "Usage") -> None:
        self.brain_calls += other.brain_calls
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out


@dataclass
class BrainOutput:
    skill: str = "none"       # "none" (move by `move`) | "pick" | "place"
    arg: str = ""             # object name when skill != "none"
    move: tuple = (0.0, 0.0, 0.0)  # [dx, dy, dz] applied when skill == "none"
    raw_text: str = ""        # model raw response (first call)
    reasoning: str = ""       # model thinking trace when the API provides one
    usage: Usage = field(default_factory=Usage)
    # every failed parse: {"stage": "first"|"repairN", "error": str, "raw": str}
    parse_failures: list = field(default_factory=list)


class Brain(ABC):
    name: str = "brain"

    @abstractmethod
    def act(self, obs: Obs, step_idx: int, last_action: str, last_reward: float) -> BrainOutput: ...
