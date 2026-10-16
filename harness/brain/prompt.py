"""Prompt construction and strict skill-JSON parsing/repair for VLM brains.

Protocol v2: the brain moves with dx/dy/dz deltas and invokes motion
primitives (skills) instead of commanding the gripper directly.
"""
from __future__ import annotations

import json
import re

from ..envs.base import Obs

SKILLS = ("none", "pick", "place")

SYSTEM_PROMPT = """You are the brain of a robot arm operating a tabletop scene. \
At each step you receive proprioceptive state (and camera images when available) \
and choose ONE action.

Action format (strict JSON, no other text):
{"dx": <float>, "dy": <float>, "dz": <float>, "skill": "none"|"pick"|"place", \
"object": "<name>"}

- dx, dy, dz: mobile-base movement in [-1, 1], applied only when \
skill = "none". Fixed-arm robots have no base: this is ignored - leave 0.
- skill:
  - "none": no manipulation action this step (base moves if the robot has \
one). All arm motion happens inside pick/place.
  - "pick": grasp <object> and lift it. Invoke it when the object is centered \
in the jaw (check the images and delta_to_ee first); dx/dy/dz are ignored.
  - "place": lower the held object onto <object>, release, and retreat. \
Requires holding something; dx/dy/dz are ignored.
- object: the name exactly as it appears in the state text (required when \
skill is not "none").
- The result of each skill invocation is reported in the state text as \
"last skill" - read it before deciding.

State conventions:
- Positions in the state text are world-frame meters. delta_to_ee is the object \
position minus your end-effector position, and dist is the euclidean distance.
- If an object is "not visible now; last seen ...", it is occluded right now; \
the coordinates are from that sighting. Obstacles are reported the same way: \
keep your path clear of them.

Rules:
- Reasoning cap: at most ~60 words. State only the key delta numbers and your \
choice, then output the JSON immediately. Long deliberation gets truncated and \
leaves no answer at all - if you notice yourself deliberating, stop and reply.
- Look at the images AND the reported 3D positions before deciding.
- Reply with ONLY the JSON object."""


def build_user_prompt(obs: Obs, step_idx: int, last_action: str, last_reward: float) -> str:
    lines = [f"Task: {obs.instruction}", "", obs.state_text]
    if last_action:
        lines += ["", f"Last action: {last_action} | reward: {last_reward:+.3f}"]
    lines += ["", f"Step {step_idx}. Reply with the action JSON only."]
    return "\n".join(lines)


def parse_action_json(text: str) -> tuple[dict | None, str]:
    """Extract a v2 action from a model response.

    Returns ({move: [dx,dy,dz], skill: str, object: str} or None, error).
    """
    m = re.search(r"\{[^{}]*\}?", text, re.DOTALL)
    if not m:
        return None, "no JSON object found in response"
    raw = m.group(0)
    err = ""
    d = None
    for cand in (raw, raw + "}"):  # second handles truncation mid-object
        try:
            d = json.loads(cand)
            break
        except json.JSONDecodeError as e:
            err = f"invalid JSON: {e}"
            d = None
    if d is None:
        return None, err
    if not isinstance(d, dict):
        return None, "JSON is not an object"
    skill = d.get("skill", "none")
    if skill not in SKILLS:
        return None, f"unknown skill: {skill!r}"
    try:
        move = [float(d.get(k, 0.0)) for k in ("dx", "dy", "dz")]
    except (TypeError, ValueError):
        return None, "non-numeric dx/dy/dz"
    if not all(-1.5 <= v <= 1.5 for v in move):
        return None, f"dx/dy/dz out of range: {move}"
    obj = d.get("object")
    obj = str(obj) if obj is not None else ""
    if skill != "none" and not obj:
        return None, f"missing 'object' for skill {skill!r}"
    return {"move": move, "skill": skill, "object": obj}, ""


def repair_prompt(text: str, error: str) -> str:
    return (
        f"Your previous reply could not be parsed: {error}\n"
        f"Previous reply: {text[:500]}\n"
        "Reply again with ONLY the JSON object: "
        '{"dx": <float>, "dy": <float>, "dz": <float>, '
        '"skill": "none"|"pick"|"place", "object": "<name>"} '
        '("object" required when skill is not "none").'
    )
