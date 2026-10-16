"""harness: cross-embodiment, cross-VLM benchmark harness.

Axes:
  envs/    -- benchmark adapters (ManiSkill3 first; tasks x embodiments)
  brain/   -- VLM brains via OpenAI-compatible APIs
  runner   -- episode loop, budget cap, trajectory logging
"""

__version__ = "1.0.0"
