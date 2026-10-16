"""Episode runner: env x brain loop with budget caps and trajectory logging."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .envs.base import EnvAdapter, TaskCell
from .brain.base import Brain
from .perception import to_json


class BudgetExhausted(Exception):
    pass


@dataclass
class Budget:
    max_episodes: int = 10**9
    max_brain_calls: int = 10**9
    max_tokens: int = 10**9
    episodes_done: int = 0
    brain_calls: int = 0
    tokens: int = 0

    def claim_episode(self) -> None:
        if self.episodes_done >= self.max_episodes:
            raise BudgetExhausted("episode budget exhausted")
        self.episodes_done += 1

    def claim_usage(self, brain_calls: int, tokens: int) -> None:
        self.brain_calls += brain_calls
        self.tokens += tokens
        if self.brain_calls > self.max_brain_calls:
            raise BudgetExhausted(f"brain-call budget exhausted ({self.brain_calls} calls)")
        if self.tokens > self.max_tokens:
            raise BudgetExhausted(f"token budget exhausted ({self.tokens} tokens)")


@dataclass
class EpisodeResult:
    backend: str
    task: str
    embodiment: str
    seed: int
    status: str = "max_steps"  # success | max_steps | budget | error
    steps: int = 0
    reward_sum: float = 0.0
    brain_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_s: float = 0.0
    error: str = ""
    extra: dict = field(default_factory=dict)


def run_episode(
    adapter: EnvAdapter,
    cell: TaskCell,
    brain: Brain,
    seed: int,
    budget: Budget,
    out_dir: Path | None = None,
    record_every: int = 0,
) -> EpisodeResult:
    """One episode. Env/brain errors are contained here and reported in status."""
    res = EpisodeResult(
        backend="maniskill", task=cell.task, embodiment=cell.embodiment, seed=seed
    )
    budget.claim_episode()
    frames_dir = None
    if out_dir is not None:
        frames_dir = out_dir / "frames" / f"{cell.embodiment}_{cell.task}_{seed}"
        (frames_dir / "obs_base").mkdir(parents=True, exist_ok=True)
        (frames_dir / "obs_wrist").mkdir(exist_ok=True)
    t0 = time.time()
    steps: list[dict] = []
    try:
        obs = adapter.reset(seed)
        if frames_dir is not None and record_every > 0:
            _save_frame(frames_dir / "f0000.png", adapter.render())
        last_action, last_reward = "", 0.0
        for step in range(adapter.max_steps):
            # persist exactly what the brain sees this step (+ segmentation overlays)
            if frames_dir is not None:
                _save_frame(frames_dir / "obs_base" / f"{step:04d}.png", obs.rgb)
                if obs.wrist_rgb is not None:
                    _save_frame(frames_dir / "obs_wrist" / f"{step:04d}.png", obs.wrist_rgb)
                if obs.readings:
                    (frames_dir / "obs_overlay").mkdir(exist_ok=True)
                    from .perception import draw_overlays
                    base_r = [r for r in obs.readings if r.camera == "base_camera"]
                    wrist_r = [r for r in obs.readings if r.camera == "hand_camera"]
                    if base_r:
                        _save_frame(frames_dir / "obs_overlay" / f"{step:04d}_base.png",
                                    draw_overlays(obs.rgb, base_r))
                    if wrist_r and obs.wrist_rgb is not None:
                        _save_frame(frames_dir / "obs_overlay" / f"{step:04d}_wrist.png",
                                    draw_overlays(obs.wrist_rgb, wrist_r))
            t0b = time.time()
            out = brain.act(obs, step, last_action, last_reward)
            t_brain = time.time() - t0b
            budget.claim_usage(out.usage.brain_calls, out.usage.tokens_in + out.usage.tokens_out)
            res.brain_calls += out.usage.brain_calls
            res.tokens_in += out.usage.tokens_in
            res.tokens_out += out.usage.tokens_out
            t0e = time.time()
            if out.skill != "none":
                sk = adapter.execute_skill(out.skill, out.arg)
                t_env = time.time() - t0e
                res.steps += sk.steps
                res.reward_sum += sk.reward
                action_desc = f"{out.skill} {out.arg}".strip()
                steps.append({"step": step, "action": action_desc,
                              "reward": round(sk.reward, 4),
                              "success": sk.env_success,
                              "skill_status": sk.status, "skill_detail": sk.detail,
                              "sim_steps": sk.steps,
                              "state_text": obs.state_text,
                              "perception": to_json(obs.readings),
                              "raw": out.raw_text[-300:],
                              **({"reasoning": out.reasoning} if out.reasoning else {}),
                              **({"parse_failures": out.parse_failures}
                                 if out.parse_failures else {}),
                              "t_brain_s": round(t_brain, 2),
                              "t_env_s": round(t_env, 3),
                              **({"trace": sk.trace} if sk.trace else {})})
                last_action, last_reward = action_desc, sk.reward
                done = sk.env_success
                new_obs = sk.obs
            else:
                sr = adapter.move_base(list(out.move))
                t_env = time.time() - t0e
                action_desc = f"move base ({out.move[0]:+.2f},{out.move[1]:+.2f},{out.move[2]:+.2f})"
                steps.append({"step": step, "action": action_desc,
                              "reward": round(sr.reward, 4), "success": sr.success,
                              "state_text": obs.state_text,
                              "perception": to_json(obs.readings),
                              "raw": out.raw_text[-300:],
                              **({"reasoning": out.reasoning} if out.reasoning else {}),
                              **({"parse_failures": out.parse_failures}
                                 if out.parse_failures else {}),
                              "t_brain_s": round(t_brain, 2),
                              "t_env_s": round(t_env, 3)})
                last_action, last_reward = action_desc, sr.reward
                done = sr.success
                new_obs = sr.obs
            if frames_dir is not None and record_every > 0 and (step + 1) % record_every == 0:
                _save_frame(frames_dir / f"f{step + 1:04d}.png", adapter.render())
            if done:
                res.status = "success"
                break
            obs = new_obs
        else:
            res.status = "max_steps"
    except BudgetExhausted as e:
        res.status = "budget"
        res.error = str(e)
    except Exception as e:  # env or brain failure: contained, batch continues
        res.status = "error"
        res.error = f"{type(e).__name__}: {e}"
    res.wall_s = round(time.time() - t0, 2)
    if frames_dir is not None and record_every > 0:
        video = encode_video(frames_dir, out_dir / "videos" / f"{frames_dir.name}.mp4")
        if video is not None:
            res.extra["video"] = str(video)
    if out_dir is not None:
        ep_dir = out_dir / "episodes"
        ep_dir.mkdir(parents=True, exist_ok=True)
        name = f"{cell.embodiment}_{cell.task}_{seed}"
        (ep_dir / f"{name}.json").write_text(json.dumps(
            {"result": asdict(res), "steps": steps}, ensure_ascii=False, indent=1))
    return res


def _save_frame(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(rgb).save(path)


def encode_video(frames_dir: Path, out_path: Path, fps: int = 10) -> Path | None:
    """Encode f*.png frames from an episode into an mp4. No other processing."""
    try:
        import imageio_ffmpeg
    except ImportError:
        print("[video] imageio-ffmpeg not installed; keeping PNG frames only")
        return None
    frames = sorted(frames_dir.glob("f*.png"))
    if not frames:
        return None
    from PIL import Image
    w, h = Image.open(frames[0]).size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(out_path), (w, h), fps=fps, quality=7)
    writer.send(None)
    for f in frames:
        writer.send(np.asarray(Image.open(f)))
    writer.close()
    return out_path


def run_batch(
    make_adapter,
    cells: list[TaskCell],
    make_brain,
    seeds: list[int],
    out_dir: Path,
    budget: Budget | None = None,
    record_every: int = 0,
) -> list[EpisodeResult]:
    """Run cells x seeds, appending results. Budget exhaustion stops the batch cleanly."""
    budget = budget or Budget()
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[EpisodeResult] = []
    adapter = make_adapter()
    try:
        current: TaskCell | None = None
        for cell in cells:
            if current != cell:
                adapter.open(cell, seeds[0])  # env build + first reset
                current = cell
            for seed in seeds:
                brain = make_brain()
                r = run_episode(adapter, cell, brain, seed, budget,
                                out_dir=out_dir, record_every=record_every)
                results.append(r)
                print(f"[run] {cell.embodiment:14s} {cell.task:22s} seed={seed} "
                      f"-> {r.status:10s} steps={r.steps:3d} calls={r.brain_calls} "
                      f"tok={r.tokens_in + r.tokens_out} {r.wall_s}s")
                if r.status == "budget":
                    return results
    finally:
        adapter.close()
    return results
