"""ManiSkill3 adapter: tasks x embodiments over a canonical EE-delta protocol.

Headless defaults are set BEFORE sapien is imported. On this machine there is no
NVIDIA Vulkan ICD, so we fall back to lavapipe (software rendering); if an NVIDIA
ICD exists we leave Vulkan device selection untouched.
"""
from __future__ import annotations

import os


def _default_vk_icd() -> str | None:
    for p in (
        "/etc/vulkan/icd.d/nvidia_icd.json",
        "/usr/share/vulkan/icd.d/nvidia_icd.json",
    ):
        if os.path.exists(p):
            return p
    lvp = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
    return lvp if os.path.exists(lvp) else None


_icd = _default_vk_icd()
if _icd:
    os.environ.setdefault("VK_ICD_FILENAMES", _icd)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: F401, E402
import numpy as np  # noqa: E402
import sapien  # noqa: E402
import torch  # noqa: E402
from mani_skill.agents.registration import register_agent  # noqa: E402
from mani_skill.agents.robots.panda.panda import Panda  # noqa: E402
from mani_skill.sensors.camera import CameraConfig  # noqa: E402

from ..perception import ObjectReading, format_object_lines, perceive  # noqa: E402
from .base import (EnvAdapter, Obs, SkillResult, StepResult, TaskCell,  # noqa: E402
                   STEP_TARGET_M, clip_action)

# Wrist camera support. panda has no built-in wrist cam, so we register a subclass
# with ManiSkill's own (commented) hand_camera mount pose. fetch already ships one
# (fetch_hand). xarm6_robotiq has no calibrated wrist pose yet.
PANDA_HAND_CAM = CameraConfig(
    uid="hand_camera",
    pose=sapien.Pose(
        p=[0.0464982, -0.0200011, 0.0360011],
        q=[0, 0.7071068, 0, 0.7071068],  # 135 deg about +Y: steeper than the 120 deg
    ),                             # of v1.0, keeps fingers + cube in frame at grasp
    width=128,
    height=128,
    fov=1.57,
    near=0.01,
    far=100,
    entity_uid="panda_hand",
)


@register_agent()
class PandaWithWristCam(Panda):
    uid = "panda_wrist"

    @property
    def _sensor_configs(self):
        return [PANDA_HAND_CAM]


AGENT_UIDS = {"panda": "panda_wrist", "xarm6_robotiq": "xarm6_robotiq", "fetch": "fetch"}
WRIST_SENSOR = {"panda": "hand_camera", "fetch": "fetch_hand"}

# task id -> natural-language instruction shown to the brain
TASKS = {
    "PickCube-v1": "Pick up the cube and hold it in the air.",
    "StackCube-v1": "Pick up the red cube and stack it on top of the green cube.",
    "PushCube-v1": "Push the cube until it reaches the target position marked on the table.",
    "PullCube-v1": "Pull the cube until it reaches the target position marked on the table.",
    "PegInsertionSide-v1": "Insert the peg into the hole from the side.",
}

EMBODIMENTS = ["panda", "xarm6_robotiq", "fetch"]

# verified capability matrix (probed on sapien 3.0.3 / mani-skill 3.0.1)
SUPPORTED: set[tuple[str, str]] = {
    (r, t)
    for r in EMBODIMENTS
    for t in TASKS
    if not (r == "fetch" and t == "PegInsertionSide-v1")
    and not (r == "xarm6_robotiq" and t == "PegInsertionSide-v1")
}


class ManiSkillAdapter(EnvAdapter):
    def __init__(self) -> None:
        self.env = None
        self._cell: TaskCell | None = None
        self._raw_obs = None
        self._t = 0                             # steps since episode reset
        self._last_seen: dict[str, tuple] = {}  # object name -> (pos, step seen)
        self._episode_events: list[str] = []    # in-episode grasp/release log
        self._prev_holding = False
        self._last_held = ""
        self._jaw_target = 1.0                  # gripper slot value (-1 close, 1 open)
        self._base_cmd = None                   # [forward vel, yaw rate] for mobile bases
        self._last_skill_report = "none yet"
        self._skill_reward = 0.0
        self._env_success = False
        # canonical action -> backend action layout (fetch combines body/base/arm/gripper)
        self._layout: list[tuple[str, int]] = []  # (subcontroller key, dim) in action order

    # -- setup -------------------------------------------------------------

    def open(self, cell: TaskCell, seed: int = 0) -> None:
        if (cell.embodiment, cell.task) not in SUPPORTED:
            raise ValueError(f"unsupported cell: {cell.embodiment} x {cell.task}")
        self._cell = cell
        self.env = gym.make(
            cell.task,
            obs_mode="rgbd",
            render_mode="rgb_array",
            robot_uids=AGENT_UIDS[cell.embodiment],
            control_mode="pd_ee_delta_pos",
            sim_backend="cpu",
            render_backend="cpu",
        )
        self.reset(seed)

    @property
    def max_steps(self) -> int:
        spec_steps = self.env.spec.max_episode_steps if self.env.spec else None
        vals = [int(v) for v in (spec_steps, self._cell.max_steps) if v]
        return min(vals)

    def _build_layout(self) -> None:
        """Map canonical [dx,dy,dz,grip] into the backend action vector.

        ManiSkill's CombinedController exposes `action_mapping`: {key: (start, end)}
        slices over the flat action vector. Single (non-combined) controllers have
        none and are canonical.
        """
        mapping = getattr(self.env.unwrapped.agent.controller, "action_mapping", None)
        if not mapping:
            assert self.env.action_space.shape == (4,)
            self._layout = []
            return
        self._layout = [(k, end - start, start) for k, (start, end) in mapping.items()]
        assert sum(d for _, d, _ in self._layout) == self.env.action_space.shape[0]

    # -- episode API ---------------------------------------------------------

    def _fix_custom_uid_init(self) -> None:
        """ManiSkill's TableSceneBuilder hard-codes initial poses per robot uid;
        our panda_wrist uid matches no branch, leaving the arm at all-zero qpos.
        Re-apply the rest keyframe and the panda base pose after every env reset."""
        if self.env.unwrapped.robot_uids != "panda_wrist":
            return
        agent = self.env.unwrapped.agent
        agent.reset(agent.keyframes["rest"].qpos)
        agent.robot.set_pose(sapien.Pose(p=[-0.615, 0, 0]))

    def reset(self, seed: int) -> Obs:
        self._raw_obs, _ = self.env.reset(seed=seed)
        self._t = 0
        self._last_seen.clear()
        self._episode_events.clear()
        self._prev_holding = False
        self._last_held = ""
        self._jaw_target = 1.0
        self._base_cmd = None
        self._last_skill_report = "none yet"
        self._fix_custom_uid_init()
        self._build_layout()
        return self._obs()

    def move_base(self, move) -> StepResult:
        """All current embodiments are fixed-base: base commands are no-ops.
        Mobile embodiments (e.g. fetch with a base controller) override this."""
        return StepResult(obs=self._obs(), reward=0.0, success=False)

    def _sim_step(self, act4) -> StepResult:
        self._t += 1
        act = self._to_backend(clip_action(act4))
        self._raw_obs, reward, terminated, truncated, info = self.env.step(act)
        self._skill_reward += float(reward)
        self._env_success = self._env_success or bool(
            np.asarray(info.get("success", False)).reshape(-1)[0])
        success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
        return StepResult(obs=self._obs(), reward=float(reward), success=success)

    def render(self) -> np.ndarray:
        img = self.env.render()
        return self._to_uint8(img)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    # -- skills (protocol v2) -------------------------------------------------

    def execute_skill(self, name: str, arg: str) -> SkillResult:
        self._skill_reward = 0.0
        self._env_success = False
        t0 = self._t
        if name == "pick":
            r = self._skill_pick(arg)
        elif name == "place":
            r = self._skill_place(arg)
        else:
            r = SkillResult("failed", f"unknown skill {name!r}")
        r.steps = self._t - t0
        r.reward = self._skill_reward
        r.env_success = self._env_success
        r.obs = self._obs()
        self._last_skill_report = f"{name} {arg} -> {r.status} ({r.detail})"
        return r

    def _tcp(self) -> np.ndarray:
        agent = self.env.unwrapped.agent
        return np.asarray(agent.tcp.pose.p[0] if agent.tcp.pose.p.ndim > 1
                          else agent.tcp.pose.p, dtype=float)

    def _target_pos(self, name):
        """Live object position: GT actor pose first, else last perception."""
        for canon, actor in _gt_actors(self.env.unwrapped).items():
            if canon == name:
                return np.asarray(_gt_pos(actor), dtype=float)
        pos = self._last_seen.get(name)
        return np.asarray(pos[0], dtype=float) if pos else None

    def _force_on(self, name) -> float:
        env = self.env.unwrapped
        keys = {name}
        for canon, actor in _gt_actors(env).items():
            if canon == name:
                keys.add(actor.name)
        return sum(v for k, v in self._contact_forces().items()
                   if any(k in key or key in k for key in keys))

    def _touched_objects(self) -> set:
        """Canonical names of task objects currently in contact with the robot."""
        env = self.env.unwrapped
        canon_by_actor = {a.name: canon for canon, a in _gt_actors(env).items()}
        out = set()
        for k in self._contact_forces():
            if k in canon_by_actor:
                out.add(canon_by_actor[k])
                continue
            for canon, a in _gt_actors(env).items():
                if a.name in k or k in a.name:
                    out.add(canon)
        return out

    def _touching(self, name) -> bool:
        """A contact pair with the named object exists (any force magnitude)."""
        return name in self._touched_objects()

    def _base_homing_cmd(self, target_xy) -> list:
        """Unicycle command for PDBaseForwardVel layouts: [forward m/s, yaw rad/s]."""
        root = self.env.unwrapped.agent.robot.pose
        p = np.asarray(root.p[0] if root.p.ndim > 1 else root.p, dtype=float)[:2]
        q = np.asarray(root.q[0] if root.q.ndim > 1 else root.q, dtype=float)
        yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                               1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        d = np.asarray(target_xy, dtype=float) - p
        ang = float(np.arctan2(d[1], d[0]) - yaw)
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        v = float(np.clip(0.8 * float(np.linalg.norm(d))
                          * (1.0 if abs(ang) < 0.6 else 0.25), -0.5, 0.5))
        w = float(np.clip(2.0 * ang, -2.0, 2.0))
        return [v, w]

    def _servo(self, target, max_steps, tol, trace) -> bool:
        """Closed-loop move toward target at sim speed (target may be a callable
        homing on a live pose). Mobile bases home in when the arm stalls short
        of the target. Returns True when within tol."""
        has_base = any(k == "base" for k, _, _ in self._layout)
        prev_dist, stall = None, 0
        for _ in range(max_steps):
            if self._t >= self.max_steps:
                self._base_cmd = None
                return False
            t = target() if callable(target) else np.asarray(target, dtype=float)
            ee = self._tcp()
            dist = float(np.linalg.norm(t - ee))
            if dist < tol:
                self._base_cmd = None
                return True
            self._base_cmd = None
            if has_base and dist > 0.15:
                if prev_dist is not None and abs(dist - prev_dist) < 0.005:
                    stall += 1
                else:
                    stall = 0
                if stall >= 2:  # arm saturated short of the target: drive the base
                    self._base_cmd = self._base_homing_cmd(t[:2])
            prev_dist = dist
            d = np.clip((t - ee) / STEP_TARGET_M, -1.0, 1.0)
            sr = self._sim_step([float(d[0]), float(d[1]), float(d[2]),
                                 self._jaw_target])
            trace.append({"t": self._t,
                          "ee": [round(float(v), 3) for v in self._tcp()],
                          "jaw": round(self._jaw_target, 2),
                          **({"base": [round(v, 2) for v in self._base_cmd]}
                             if self._base_cmd else {})})
            if sr.success:
                self._base_cmd = None
                return True
        self._base_cmd = None
        return False

    def _skill_pick(self, name) -> SkillResult:
        trace = []
        if self._target_pos(name) is None:
            return SkillResult("failed", f"object {name!r} unknown")
        self._servo(lambda: self._target_pos(name) + np.array([0.0, 0.0, 0.06]),
                    25, 0.02, trace)                                    # pre-grasp
        self._servo(lambda: self._target_pos(name), 12, 0.015, trace)   # descend
        grasp_ok = False
        for _ in range(8):  # normalized delta slot: the PD closes gradually
            if self._t >= self.max_steps:
                break
            self._jaw_target = -1.0
            sr = self._sim_step([0.0, 0.0, 0.0, -1.0])
            trace.append({"t": self._t, "jaw": -1.0})
            if sr.success:
                return SkillResult("ok", "grasped (env success)", trace=trace)
            if self._touching(name):
                grasp_ok = True
                break
        lift_z = self._tcp()[2] + 0.06   # absolute height: avoids chasing a moving target
        lifted = self._servo(lambda: np.array([self._target_pos(name)[0],
                                               self._target_pos(name)[1], lift_z]),
                             18, 0.015, trace)                            # lift
        ok = grasp_ok and lifted
        detail = ("grasped and lifted" if ok else
                  "closed but lift incomplete" if grasp_ok else
                  "object not secured in jaw")
        return SkillResult("ok" if ok else "failed", detail, trace=trace)

    def _skill_place(self, name) -> SkillResult:
        trace = []
        if self._target_pos(name) is None:
            return SkillResult("failed", f"object {name!r} unknown")
        if not self._prev_holding:
            return SkillResult("failed", "nothing is held (call pick first)")
        self._servo(lambda: self._target_pos(name) + np.array([0.0, 0.0, 0.08]),
                    30, 0.02, trace)                                    # above target
        placed = self._servo(lambda: self._target_pos(name) + np.array([0.0, 0.0, 0.045]),
                             12, 0.008, trace)                          # lower to stack height
        for _ in range(3):                                              # open
            if self._t >= self.max_steps:
                break
            self._jaw_target = 1.0
            sr = self._sim_step([0.0, 0.0, 0.0, 1.0])
            trace.append({"t": self._t, "jaw": 1.0})
            if sr.success:
                return SkillResult("ok", "released (env success)", trace=trace)
        ret = self._servo(lambda: self._tcp() + np.array([0.0, 0.0, 0.05]),
                          10, 0.015, trace)                             # retreat
        ok = placed and ret
        detail = ("released and retreated" if ok else
                  "lowered but release incomplete")
        return SkillResult("ok" if ok else "failed", detail, trace=trace)

    # -- internals -----------------------------------------------------------

    def _contact_forces(self) -> dict[str, float]:
        """Robot-side contact force per touched body (F/T-sensor analogue).

        Generic: any non-robot body counts - task objects, fixtures, the table.
        Robot detection is by articulation-link component, so this works for
        any robot and any scene without task-specific registration.
        """
        try:
            contacts = self.env.unwrapped.scene.get_contacts()
        except Exception:
            return {}
        forces: dict[str, float] = {}
        for c in contacts:
            robot_hit, others = False, []
            for b in c.bodies:
                if isinstance(b, sapien.physx.PhysxArticulationLinkComponent):
                    robot_hit = True  # a link of some robot articulation
                    continue
                ent = getattr(b, "entity", None)
                nm = (getattr(ent, "name", "") if ent is not None else "") \
                    or (getattr(b, "name", "") or "")
                others.append(nm.removeprefix("scene-0_"))
            if not robot_hit or not others:
                continue  # object-object contact: not robot force sensing
            f = sum(float(np.linalg.norm(np.asarray(pt.impulse)))
                    for pt in c.points) / 0.002  # sim dt 1/500 s
            for nm in others:
                forces[nm] = forces.get(nm, 0.0) + f
        return forces

    def _to_backend(self, a: np.ndarray) -> np.ndarray:
        if not self._layout:  # canonical already
            return a
        out = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        for key, dim, off in self._layout:
            if key == "arm" and dim >= 3:
                out[off : off + 3] = a[:3]
            elif key == "gripper":
                out[off] = a[3]
            elif key == "base" and self._base_cmd is not None:
                out[off : off + dim] = np.asarray(self._base_cmd, dtype=np.float32)
        return out


    def _obs(self) -> Obs:
        agent = self.env.unwrapped.agent
        ee = np.asarray(agent.tcp.pose.p[0] if agent.tcp.pose.p.ndim > 1 else agent.tcp.pose.p)
        grip = self._gripper_openness()
        sd = self._raw_obs["sensor_data"]
        wrist_data = sd.get(WRIST_SENSOR.get(self._cell.embodiment, ""))
        if getattr(self._cell, "gt_state", False):
            # privileged debug mode: ground-truth object positions, no CV perception
            env = self.env.unwrapped
            readings = [ObjectReading(name, pos, "gt", 0)
                        for name, pos in _gt_objects(env).items()]
            gp = getattr(env, "goal_region", None)  # PushCube/PullCube target marker
            if gp is not None:
                readings.append(ObjectReading("target", _gt_pos(gp), "gt", 0))
        else:
            readings = perceive(
                self._to_uint8(sd["base_camera"]["rgb"]), sd["base_camera"]["depth"],
                self._to_uint8(wrist_data["rgb"]) if wrist_data is not None else None,
                wrist_data["depth"] if wrist_data is not None else None,
                self._raw_obs["sensor_param"], self._cell.task,
                wrist_uid=WRIST_SENSOR.get(self._cell.embodiment, "hand_camera"),
            )
        for r in readings:  # last-known-position memory: remember / annotate stale
            if r.position is not None:
                self._last_seen[r.name] = (r.position, self._t)
            elif r.name in self._last_seen:
                pos, t0 = self._last_seen[r.name]
                r.last_seen_pos, r.last_seen_age = pos, self._t - t0
        touched = self._touched_objects()
        holding = bool(touched)
        held_name = ""
        if holding:
            best, bd = None, 0.06
            for r in readings:
                if r.position is None:
                    continue
                d = float(np.linalg.norm(np.asarray(r.position) - ee))
                if d < bd:
                    best, bd = r.name, d
            held_name = best or "object"
        if holding and not self._prev_holding:
            self._episode_events.append(f"s{self._t}: grasped {held_name}")
        elif not holding and self._prev_holding:
            self._episode_events.append(
                f"s{self._t}: released {self._last_held} "
                f"at ({ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}) m")
        self._prev_holding, self._last_held = holding, (held_name or self._last_held)
        state_lines = [
            f"end-effector tcp (mid-fingertip grasp point, m): "
            f"x={ee[0]:.3f} y={ee[1]:.3f} z={ee[2]:.3f}",
            "base: fixed (dx/dy/dz ignored)",
            f"held: {held_name if holding else 'none'}",
            *format_object_lines(readings, ee),
            f"last skill: {self._last_skill_report}",
            "history: " + ("; ".join(self._episode_events[-3:]) if self._episode_events
                           else "none yet"),
        ]
        rgb = self._to_uint8(sd["base_camera"]["rgb"])
        wrist_rgb = self._to_uint8(wrist_data["rgb"]) if wrist_data is not None else None
        return Obs(rgb=rgb, state_text="\n".join(state_lines),
                   instruction=TASKS[self._cell.task], wrist_rgb=wrist_rgb,
                   readings=readings)

    def _gripper_openness(self) -> float:
        agent = self.env.unwrapped.agent
        subs = getattr(agent.controller, "controllers", None)
        gripper = (subs or {}).get("gripper") or (subs or {}).get("gripper_active")
        joints = getattr(gripper, "joints", None) if gripper is not None else None
        if not joints:
            return 0.0
        q = agent.robot.get_qpos()[0]
        vals = []
        for j in joints:
            lo, hi = (float(x) for x in j.limits.reshape(-1)[-2:])
            vals.append(float(np.clip((float(q[j.active_index]) - lo) / max(hi - lo, 1e-6), 0, 1)))
        return float(np.mean(vals))

    @staticmethod
    def _to_uint8(t) -> np.ndarray:
        a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        a = np.asarray(a).reshape(-1, *a.shape[-3:]) if a.ndim > 3 else a
        a = a[0] if a.ndim == 4 else a
        if a.dtype != np.uint8:
            a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
        return a


# (canonical object name, env attribute); actors have .pose, some tasks may
# expose raw pose tensors instead
_GT_OBJECT_ATTRS = (("cube", "cube"), ("cube", "obj"),
                    ("cube_red", "cubeA"), ("cube_green", "cubeB"))


def _gt_pos(o) -> tuple:
    p = o.pose.p if hasattr(o, "pose") else o
    return tuple(float(v) for v in np.asarray(p).reshape(-1))


def _gt_actors(env) -> dict:
    return {name: getattr(env, attr) for name, attr in _GT_OBJECT_ATTRS
            if getattr(env, attr, None) is not None}


def _gt_objects(env) -> dict:
    return {name: _gt_pos(actor) for name, actor in _gt_actors(env).items()}


def make_cell(task: str, embodiment: str, max_steps: int | None = None,
              gt_state: bool = False) -> TaskCell:
    if task not in TASKS:
        raise ValueError(f"unknown task {task}; known: {list(TASKS)}")
    if embodiment not in EMBODIMENTS:
        raise ValueError(f"unknown embodiment {embodiment}; known: {EMBODIMENTS}")
    kw = {"max_steps": max_steps} if max_steps else {}
    return TaskCell(task=task, embodiment=embodiment, gt_state=gt_state, **kw)
