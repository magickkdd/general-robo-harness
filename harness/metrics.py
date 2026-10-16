"""Aggregate EpisodeResults into per-cell and total metrics."""
from __future__ import annotations

from collections import defaultdict

from .runner import EpisodeResult


def _cell_key(r: EpisodeResult) -> str:
    return f"{r.embodiment}/{r.task}"


def aggregate(results: list[EpisodeResult]) -> dict:
    cells: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        cells[_cell_key(r)].append(r)
    out = {}
    for key, rs in sorted(cells.items()):
        n = len(rs)
        succ = sum(1 for r in rs if r.status == "success")
        out[key] = {
            "episodes": n,
            "successes": succ,
            "success_rate": round(succ / n, 3) if n else 0.0,
            "mean_steps": round(sum(r.steps for r in rs) / n, 1) if n else 0.0,
            "brain_calls": sum(r.brain_calls for r in rs),
            "tokens_in": sum(r.tokens_in for r in rs),
            "tokens_out": sum(r.tokens_out for r in rs),
            "wall_s": round(sum(r.wall_s for r in rs), 1),
            "errors": sum(1 for r in rs if r.status == "error"),
        }
    return out


def totals(results: list[EpisodeResult]) -> dict:
    n = len(results)
    succ = sum(1 for r in results if r.status == "success")
    return {
        "episodes": n,
        "successes": succ,
        "success_rate": round(succ / n, 3) if n else 0.0,
        "brain_calls": sum(r.brain_calls for r in results),
        "tokens_in": sum(r.tokens_in for r in results),
        "tokens_out": sum(r.tokens_out for r in results),
        "wall_s": round(sum(r.wall_s for r in results), 1),
        "statuses": {s: sum(1 for r in results if r.status == s)
                     for s in ("success", "max_steps", "budget", "error")},
    }
