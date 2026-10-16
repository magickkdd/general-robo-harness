"""Canonical observation/action types and the environment adapter interface.

The harness speaks one canonical protocol regardless of backend:
  - action: [dx, dy, dz, gripper] in [-1, 1] (end-effector delta + gripper open/close)
  - observation: RGB image + proprioceptive state text + task instruction
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Obs:
    rgb: np.ndarray          # H x W x 3 uint8, third-person camera
    state_text: str          # proprioception + task-relevant state, human/VLM readable
    instruction: str         # task instruction in natural language
    wrist_rgb: np.ndarray | None = None  # optional wrist-mounted camera view
    readings: list = field(default_factory=list)  # perception.ObjectReading list


@dataclass
class StepResult:
    obs: Obs
    reward: float
    success: bool


@dataclass
class SkillResult:
    """Outcome of one motion-primitive invocation (protocol v2)."""
    status: str                 # "ok" | "failed" | "aborted"
    detail: str = ""
    steps: int = 0              # sim steps consumed
    reward: float = 0.0
    env_success: bool = False
    obs: object | None = None
    trace: list = field(default_factory=list)


@dataclass
class TaskCell:
    """One (task, embodiment) cell of the benchmark matrix."""
    task: str
    embodiment: str
    max_steps: int = 200
    gt_state: bool = False  # privileged ground-truth object state, no images (debug)


class EnvAdapter(ABC):
    """Backend adapter. One instance runs one (task, embodiment) cell at a time."""

    @abstractmethod
    def open(self, cell: TaskCell, seed: int) -> None: ...

    @abstractmethod
    def reset(self, seed: int) -> Obs: ...

    @abstractmethod
    def render(self) -> np.ndarray: ...

    @abstractmethod
    def close(self) -> None: ...

    def execute_skill(self, name: str, arg: str) -> "SkillResult":
        raise NotImplementedError(f"adapter does not implement skill {name!r}")

    def move_base(self, move) -> "StepResult":
        """Protocol v2: dx/dy/dz drive the mobile base. Fixed-base adapters
        ignore it. Must be implemented by every adapter."""
        raise NotImplementedError("adapter does not implement move_base")


def clip_action(a) -> np.ndarray:
    """Canonical action vector [dx, dy, dz, gripper], each in [-1, 1]."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    assert a.shape == (4,), f"canonical action must be 4-dim, got {a.shape}"
    return np.clip(a, -1.0, 1.0)


STEP_TARGET_M = 0.1  # 1.0 canonical = 0.1 m EE-delta target (controller clip limit)

