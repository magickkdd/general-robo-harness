"""VLM brain over any OpenAI-compatible chat-completions API (vision-capable model).

Resolution of credentials: pass --api-key-env to name the env var; .env in the
working directory is loaded if python-dotenv is available, otherwise export first.
"""
from __future__ import annotations

import base64
import io
import os

from openai import OpenAI
from PIL import Image

from ..envs.base import Obs
from .base import Brain, BrainOutput, Usage
from .prompt import SYSTEM_PROMPT, build_user_prompt, parse_action_json, repair_prompt


def _b64_png(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class VLMBrain(Brain):
    """One VLM decision per step: image + state -> action JSON."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,  # thinking peaks reach ~16k chars (~5k tokens); headroom
        timeout_s: float = 240.0,  # thinking calls measured 41-176 s; 60 s was too tight
        max_repair: int = 1,
        thinking: bool = True,       # False: gateway-side thinking disabled (fast)
        send_images: str = "all",    # all (base+wrist) | base | none (text-only)
    ) -> None:
        self.name = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_repair = max_repair
        self.thinking = thinking
        self.send_images = send_images

    @staticmethod
    def _reasoning_of(message) -> str:
        rc = getattr(message, "reasoning_content", None)
        if not rc:
            extra = getattr(message, "model_extra", None) or {}
            rc = extra.get("reasoning_content")
        return rc or ""

    def _chat(self, user_prompt: str, obs: Obs) -> tuple[str, Usage, str]:
        usage = Usage()
        content: list[dict] = []
        if self.send_images != "none":
            content.append(
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{_b64_png(obs.rgb)}"}})
            if self.send_images == "all" and obs.wrist_rgb is not None:
                content.append(
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_b64_png(obs.wrist_rgb)}"}})
        content.append({"type": "text", "text": user_prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            **({} if self.thinking else {"extra_body": {"thinking": {"type": "disabled"}}}),
        )
        usage.brain_calls = 1
        if resp.usage is not None:
            usage.tokens_in = resp.usage.prompt_tokens or 0
            usage.tokens_out = resp.usage.completion_tokens or 0
        message = resp.choices[0].message
        return (message.content or ""), usage, self._reasoning_of(message)

    def act(self, obs: Obs, step_idx: int, last_action: str, last_reward: float) -> BrainOutput:
        user_prompt = build_user_prompt(obs, step_idx, last_action, last_reward)
        if self.send_images == "none":
            user_prompt = ("Input: proprioceptive state text only "
                           "(no camera images this run).\n\n" + user_prompt)
        elif self.send_images == "all" and obs.wrist_rgb is not None:
            user_prompt = ("Image 1: third-person camera. "
                           "Image 2: wrist camera mounted on the gripper.\n\n" + user_prompt)
        else:
            user_prompt = "Image 1: third-person camera.\n\n" + user_prompt
        text, usage, reasoning = self._chat(user_prompt, obs)
        parsed, err = parse_action_json(text)
        failures = []
        if parsed is None:  # root cause of the repair loop: first-round raw + error
            failures.append({"stage": "first", "error": err, "raw": text})
        tries = 0
        while parsed is None and tries < self.max_repair:
            text2, u2, rc2 = self._chat(repair_prompt(text, err), obs)
            usage.add(u2)
            reasoning = (reasoning + "\n" + rc2).strip() if (rc2 or reasoning) else ""
            parsed, err = parse_action_json(text2)
            text = text2
            tries += 1
            if parsed is None:
                failures.append({"stage": f"repair{tries}", "error": err, "raw": text2})
        if parsed is None:
            # last-resort deterministic fallback: no motion, no skill
            parsed = {"move": [0.0, 0.0, 0.0], "skill": "none", "object": ""}
            err = f"{err} (fallback: no-op)"
        out = BrainOutput(skill=parsed["skill"], arg=parsed["object"],
                          move=tuple(parsed["move"]), raw_text=text,
                          reasoning=reasoning, usage=usage, parse_failures=failures)
        out.raw_text += f" | parse: {err}" if err else ""
        return out


def load_api_key(env_var: str | None) -> str:
    """Load .env from cwd when possible, then resolve the key."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    candidates = [env_var] if env_var else ["LLM_API_KEY", "DEEPSEEK_API_KEY", "SENSENOVA_API_KEY"]
    for name in candidates:
        val = os.environ.get(name)
        if val:
            return val
    raise RuntimeError(f"no API key found; set one of {candidates} or --api-key-env")
