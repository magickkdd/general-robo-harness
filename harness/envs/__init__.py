"""Benchmark registry. Currently one backend: maniskill."""
from __future__ import annotations

from .base import EnvAdapter, Obs, StepResult, TaskCell  # noqa: F401
from .maniskill import EMBODIMENTS, SUPPORTED, TASKS, ManiSkillAdapter, make_cell  # noqa: F401

BACKENDS = {"maniskill": ManiSkillAdapter}


def get_adapter(backend: str) -> EnvAdapter:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend}; known: {list(BACKENDS)}")
    return BACKENDS[backend]()
