from __future__ import annotations

"""MuJoCo geometry-based differential-drive navigation environments.

``MujocoNavEnv`` is a standard Gymnasium environment for demos and notebooks.
``MujocoBatchEnv`` adapts several independent scenes to the tensor interface
used by AIRC-ROS2-Istruction's PPO rollout code.
"""

from dataclasses import dataclass
import math
from typing import Any, Optional

import gymnasium as gym
import mujoco
import numpy as np
import torch
from gymnasium import spaces


@dataclass
class MujocoNavConfig:
    dt: float = 0.1
    patch_meters: float = 10.0
    ray_max_gap: float = 0.6
    arena_size: float = 12.0
    wall_thickness: float = 0.25
    obstacle_count: int = 10
    obstacle_min_size: float = 0.35
    obstacle_max_size: float = 1.0
    robot_radius: float = 0.22
    goal_radius: float = 0.3
    max_episode_steps: int = 600
    vx_max: float = 0.6
    omega_max: float = 1.5
    vx_forward_only: bool = True
    reward_collision: float = 2.0
    reward_progress: float = 0.1
    reward_success: float = 2.0
    reward_jerk: float = 1e-6
    reward_jerk_omega: float = 1e-6

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "MujocoNavConfig":
        obs = cfg.get("obs", {}) or {}
        sim = cfg.get("sim", {}) or {}
        lim = cfg.get("limits", {}) or {}
        rew = cfg.get("reward", {}) or {}
        world = cfg.get("world", {}) or {}
        return cls(
            dt=float(sim.get("dt", 0.1)),
            patch_meters=float(obs.get("patch_meters", 10.0)),
            ray_max_gap=float(obs.get("ray_max_gap", 0.6)),
            arena_size=float(world.get("arena_size", 12.0)),
            wall_thickness=float(world.get("wall_thickness", 0.25)),
            obstacle_count=int(world.get("obstacle_count", 10)),
            obstacle_min_size=float(world.get("obstacle_min_size", 0.35)),
            obstacle_max_size=float(world.get("obstacle_max_size", 1.0)),
            robot_radius=float(world.get("robot_radius", 0.22)),
            goal_radius=float(sim.get("task_point_success_radius_m", 0.3)),
            max_episode_steps=int(sim.get("max_episode_steps", 600)),
            vx_max=float(lim.get("vx_max", 0.6)),
            omega_max=float(lim.get("omega_max", 1.5)),
            vx_forward_only=bool(lim.get("vx_forward_only", True)),
            reward_collision=float(rew.get("reward_collision", 2.0)),
            reward_progress=float(rew.get("reward_progress", 0.1)),
            reward_success=float(rew.get("reward_success", 2.0)),
            reward_jerk=float(rew.get("reward_jerk", 1e-6)),
            reward_jerk_omega=float(rew.get("reward_jerk_omega", 1e-6)),
        )


def _scene_xml(cfg: MujocoNavConfig) -> str:
    half = cfg.arena_size / 2.0
    wall = cfg.wall_thickness
    r = cfg.robot_radius
    # Display-only car geometry fits inside the existing circular footprint.
    # Group 2 is excluded from LiDAR and all parts disable contact response.
    wheels = "\n".join(
        f'<geom name="wheel_{axle}_{side}" type="cylinder" '
        f'size="{0.23*r} {0.10*r}" pos="{x*r} {y*r} {-0.12+0.23*r}" '
        'quat="0.70710678 0.70710678 0 0" material="tire_mat"/>'
        for axle, x in (("front", 0.46), ("rear", -0.46))
        for side, y in (("left", 0.64), ("right", -0.64))
    )
    obstacle_geoms = "\n".join(
        f'<geom name="obstacle_{i}" type="box" size="0.5 0.5 0.45" '
        'pos="0 0 0.45" group="1" material="obstacle_mat"/>'
        for i in range(cfg.obstacle_count)
    )
    return f'''<mujoco model="airc_ros2_istruction_random_nav">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  <asset>
    <material name="floor_mat" rgba="0.82 0.87 0.92 1"/>
    <material name="wall_mat" rgba="0.12 0.14 0.17 1"/>
    <material name="obstacle_mat" rgba="0.48 0.50 0.52 1"/>
    <material name="robot_mat" rgba="0.04 0.42 0.72 1" specular="0.6" shininess="0.5"/>
    <material name="tire_mat" rgba="0.045 0.055 0.07 1"/>
    <material name="deck_mat" rgba="0.72 0.78 0.83 1"/>
    <material name="sensor_mat" rgba="0.08 0.12 0.17 1"/>
    <material name="light_mat" rgba="1 0.90 0.55 1" emission="0.5"/>
    <material name="goal_mat" rgba="0.16 0.68 0.38 0.8"/>
  </asset>
  <default>
    <default class="car_visual">
      <geom group="2" contype="0" conaffinity="0"/>
    </default>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="{half + 2} {half + 2} 0.1" material="floor_mat" group="0"/>
    <geom name="wall_north" type="box" size="{half} {wall / 2} 0.5" pos="0 {half} 0.5" material="wall_mat" group="1"/>
    <geom name="wall_south" type="box" size="{half} {wall / 2} 0.5" pos="0 {-half} 0.5" material="wall_mat" group="1"/>
    <geom name="wall_east" type="box" size="{wall / 2} {half} 0.5" pos="{half} 0 0.5" material="wall_mat" group="1"/>
    <geom name="wall_west" type="box" size="{wall / 2} {half} 0.5" pos="{-half} 0 0.5" material="wall_mat" group="1"/>
    {obstacle_geoms}
    <body name="robot" mocap="true" pos="0 0 0.12" childclass="car_visual">
      <geom name="robot_body" type="box" size="{0.72*r} {0.51*r} {0.15*r}" pos="0 0 {-0.12+0.44*r}" material="robot_mat"/>
      <geom name="robot_deck" type="box" size="{0.45*r} {0.39*r} {0.07*r}" pos="{-0.10*r} 0 {-0.12+0.66*r}" material="deck_mat"/>
      <geom name="robot_lidar" type="cylinder" size="{0.19*r} {0.10*r}" pos="{-0.12*r} 0 {-0.12+0.83*r}" material="sensor_mat"/>
      <geom name="robot_front" type="box" size="{0.18*r} {0.06*r} {0.018*r}" pos="{0.42*r} 0 {-0.12+0.61*r}" material="light_mat"/>
      <geom name="headlight_left" type="box" size="{0.02*r} {0.10*r} {0.06*r}" pos="{0.74*r} {0.30*r} {-0.12+0.46*r}" material="light_mat"/>
      <geom name="headlight_right" type="box" size="{0.02*r} {0.10*r} {0.06*r}" pos="{0.74*r} {-0.30*r} {-0.12+0.46*r}" material="light_mat"/>
      {wheels}
    </body>
    <site name="goal" type="cylinder" size="{cfg.goal_radius} 0.015" pos="0 0 0.02" material="goal_mat"/>
    <camera name="top" mode="fixed" pos="0 0 {cfg.arena_size * 1.7}" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>'''


class MujocoNavEnv(gym.Env):
    """Single randomized square arena with a differential-drive robot and 2D LiDAR."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: Optional[MujocoNavConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or MujocoNavConfig()
        self.render_mode = render_mode
        self.n_rays = max(8, int(math.ceil(2 * math.pi * self.cfg.patch_meters / self.cfg.ray_max_gap)))
        self.model = mujoco.MjModel.from_xml_string(_scene_xml(self.cfg))
        self.data = mujoco.MjData(self.model)
        self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        self.robot_mocap_id = int(self.model.body_mocapid[self.robot_body_id])
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self.obstacle_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{i}")
            for i in range(self.cfg.obstacle_count)
        ], dtype=np.int32)
        self._ray_groups = np.zeros(6, dtype=np.uint8)
        self._ray_groups[1] = 1
        self._ray_hit = np.zeros(1, dtype=np.int32)
        self._limits = np.array([
            [0.0 if self.cfg.vx_forward_only else -self.cfg.vx_max, self.cfg.vx_max],
            [-self.cfg.omega_max, self.cfg.omega_max],
        ], dtype=np.float32)
        self.action_space = spaces.Box(self._limits[:, 0], self._limits[:, 1], dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.n_rays + 7,), dtype=np.float32)
        self.pos = np.zeros(2, dtype=np.float64)
        self.yaw = 0.0
        self.goal = np.zeros(2, dtype=np.float64)
        self.prev_cmd = np.zeros(2, dtype=np.float64)
        self.prev_prev_cmd = np.zeros(2, dtype=np.float64)
        self.steps = 0
        self._renderer = None
        self._viewer = None

    def get_limits(self) -> np.ndarray:
        return self._limits.copy()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        half = self.cfg.arena_size / 2 - self.cfg.robot_radius - 0.5
        obstacle_centers: list[tuple[float, float, float, float]] = []
        for i in range(self.cfg.obstacle_count):
            for _ in range(100):
                hx = float(self.np_random.uniform(self.cfg.obstacle_min_size, self.cfg.obstacle_max_size)) / 2
                hy = float(self.np_random.uniform(self.cfg.obstacle_min_size, self.cfg.obstacle_max_size)) / 2
                x, y = self.np_random.uniform(-half, half, size=2)
                if all(math.hypot(x - ox, y - oy) > hx + hy + ohx + ohy + 0.25
                       for ox, oy, ohx, ohy in obstacle_centers):
                    obstacle_centers.append((float(x), float(y), hx, hy))
                    break
            else:
                obstacle_centers.append((0.0, 0.0, 0.0, 0.0))
            x, y, hx, hy = obstacle_centers[-1]
            self.model.geom_pos[self.obstacle_ids[i]] = (x, y, 0.45)
            self.model.geom_size[self.obstacle_ids[i]] = (hx, hy, 0.45)
        self.pos = self._sample_free_point(obstacle_centers, half)
        self.goal = self._sample_free_point(obstacle_centers, half, avoid=self.pos, min_dist=3.0)
        self.yaw = float(self.np_random.uniform(-math.pi, math.pi))
        self.prev_cmd.fill(0)
        self.prev_prev_cmd.fill(0)
        self.steps = 0
        self._sync_mujoco()
        self.model.site_pos[self.goal_site_id] = (self.goal[0], self.goal[1], 0.025)
        mujoco.mj_forward(self.model, self.data)
        return self._observe(), {"goal": self.goal.astype(np.float32)}

    def _sample_free_point(self, obstacles, half, avoid=None, min_dist=0.0):
        for _ in range(1000):
            p = self.np_random.uniform(-half, half, size=2)
            clear = all(abs(p[0] - x) > hx + self.cfg.robot_radius + 0.15 or
                        abs(p[1] - y) > hy + self.cfg.robot_radius + 0.15
                        for x, y, hx, hy in obstacles)
            if clear and (avoid is None or np.linalg.norm(p - avoid) >= min_dist):
                return p.astype(np.float64)
        return np.zeros(2, dtype=np.float64)

    def _sync_mujoco(self):
        self.data.mocap_pos[self.robot_mocap_id] = (self.pos[0], self.pos[1], 0.12)
        self.data.mocap_quat[self.robot_mocap_id] = (math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2))
        mujoco.mj_forward(self.model, self.data)

    def _observe(self):
        rays = np.empty(self.n_rays, dtype=np.float32)
        origin = np.array((self.pos[0], self.pos[1], 0.16), dtype=np.float64)
        for i in range(self.n_rays):
            angle = self.yaw + 2 * math.pi * i / self.n_rays
            direction = np.array((math.cos(angle), math.sin(angle), 0), dtype=np.float64)
            distance = mujoco.mj_ray(self.model, self.data, origin, direction,
                                     self._ray_groups, 1, self.robot_body_id, self._ray_hit)
            rays[i] = self.cfg.patch_meters if distance < 0 else min(distance, self.cfg.patch_meters)
        delta = self.goal - self.pos
        dist = float(np.linalg.norm(delta))
        goal_angle = math.atan2(delta[1], delta[0]) - self.yaw
        vx_center = (self._limits[0, 0] + self._limits[0, 1]) / 2
        vx_half = max((self._limits[0, 1] - self._limits[0, 0]) / 2, 1e-9)
        tail = np.array([
            math.sin(goal_angle), math.cos(goal_angle),
            (self.prev_cmd[0] - vx_center) / vx_half,
            self.prev_cmd[1] / self.cfg.omega_max,
            (self.prev_cmd[0] - self.prev_prev_cmd[0]) / (2 * vx_half),
            (self.prev_cmd[1] - self.prev_prev_cmd[1]) / (2 * self.cfg.omega_max),
            min(dist / self.cfg.patch_meters, 1.0),
        ], dtype=np.float32)
        return np.concatenate((np.clip(rays / self.cfg.patch_meters, 0, 1), tail)).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(2)
        action = np.clip(action, self._limits[:, 0], self._limits[:, 1])
        vx, omega = float(action[0]), float(action[1])
        old_pos = self.pos.copy()
        old_dist = float(np.linalg.norm(self.goal - old_pos))
        next_yaw = (self.yaw + omega * self.cfg.dt + math.pi) % (2 * math.pi) - math.pi
        next_pos = self.pos + vx * self.cfg.dt * np.array((math.cos(next_yaw), math.sin(next_yaw)))
        collided = self._collides(next_pos)
        if not collided:
            self.pos = next_pos
            self.yaw = next_yaw
        new_dist = float(np.linalg.norm(self.goal - self.pos))
        success = new_dist <= self.cfg.goal_radius
        scale = max(self.cfg.vx_max * self.cfg.dt, 1e-9)
        progress = (old_dist - new_dist) / scale
        jerk_v = (vx - 2 * self.prev_cmd[0] + self.prev_prev_cmd[0]) / max(self.cfg.vx_max, 1e-9)
        jerk_w = (omega - 2 * self.prev_cmd[1] + self.prev_prev_cmd[1]) / max(self.cfg.omega_max, 1e-9)
        reward = self.cfg.reward_progress * progress
        reward -= self.cfg.reward_collision * float(collided) * abs(vx) / max(self.cfg.vx_max, 1e-9)
        reward -= self.cfg.reward_jerk * min(jerk_v * jerk_v / 16, 1)
        reward -= self.cfg.reward_jerk_omega * min(jerk_w * jerk_w / 16, 1)
        if success:
            reward += self.cfg.reward_success
        self.prev_prev_cmd[:] = self.prev_cmd
        self.prev_cmd[:] = (vx, omega)
        self.steps += 1
        terminated = bool(collided or success)
        truncated = self.steps >= self.cfg.max_episode_steps
        self._sync_mujoco()
        obs = self._observe()
        info = {"collided": bool(collided), "success": bool(success), "distance": new_dist}
        return obs, float(reward), terminated, bool(truncated), info

    def _collides(self, p):
        half = self.cfg.arena_size / 2 - self.cfg.wall_thickness / 2
        if abs(p[0]) + self.cfg.robot_radius >= half or abs(p[1]) + self.cfg.robot_radius >= half:
            return True
        for geom_id in self.obstacle_ids:
            x, y, _ = self.model.geom_pos[geom_id]
            hx, hy, _ = self.model.geom_size[geom_id]
            if hx > 0 and abs(p[0] - x) < hx + self.cfg.robot_radius and abs(p[1] - y) < hy + self.cfg.robot_radius:
                return True
        return False

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer
                self._viewer = viewer.launch_passive(self.model, self.data)
                with self._viewer.lock():
                    self._viewer.cam.lookat[:] = (0, 0, 0)
                    self._viewer.cam.distance = self.cfg.arena_size * 1.25
                    self._viewer.cam.azimuth = 90
                    self._viewer.cam.elevation = -65
            self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=720, width=960)
            self._renderer.update_scene(self.data, camera="top")
            return self._renderer.render()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class MujocoBatchEnv:
    """CPU MuJoCo scenes exposed as the batched Torch API expected by PPO."""

    def __init__(self, env_config: dict[str, Any], n_envs: int, device: str = "cpu", seed: Optional[int] = None):
        self.cfg = MujocoNavConfig.from_dict(env_config)
        self.B = int(n_envs)
        if self.B < 1:
            raise ValueError("n_envs must be at least 1")
        self.device = torch.device(device)
        self.envs = [MujocoNavEnv(self.cfg) for _ in range(self.B)]
        self.n_rays = self.envs[0].n_rays
        self._limits = torch.as_tensor(self.envs[0].get_limits(), device=self.device)
        self._seed = seed
        self._episode_returns = np.zeros(self.B, dtype=np.float64)
        self._episode_lengths = np.zeros(self.B, dtype=np.int64)
        self.last_reset_cuts = 0

    def get_limits(self):
        return self._limits

    def reset(self):
        self.last_reset_cuts = int(np.count_nonzero(self._episode_lengths))
        self._episode_returns.fill(0)
        self._episode_lengths.fill(0)
        obs = []
        for i, env in enumerate(self.envs):
            seed = None if self._seed is None else int(self._seed) + i
            obs.append(env.reset(seed=seed)[0])
        self._seed = None
        return torch.as_tensor(np.stack(obs), device=self.device)

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().to("cpu").numpy()
        obs, rewards, dones, collided, success, timeout = [], [], [], [], [], []
        episodes = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            o, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            self._episode_returns[i] += r
            self._episode_lengths[i] += 1
            if done:
                outcome = 'success' if info['success'] else 'collision' if info['collided'] else 'timeout'
                episodes.append({'outcome': outcome, 'return': float(self._episode_returns[i]),
                                 'length': int(self._episode_lengths[i])})
                self._episode_returns[i] = 0
                self._episode_lengths[i] = 0
                o = env.reset()[0]
            obs.append(o)
            rewards.append(r)
            dones.append(done)
            collided.append(info["collided"])
            success.append(info["success"])
            timeout.append(truncated)
        info = {
            "episodes": episodes,
            "collided": torch.as_tensor(collided, device=self.device),
            "success": torch.as_tensor(success, device=self.device),
            "timeout": torch.as_tensor(timeout, device=self.device),
        }
        return (
            torch.as_tensor(np.stack(obs), device=self.device),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones, dtype=torch.bool, device=self.device),
            info,
        )

    def close(self):
        for env in self.envs:
            env.close()
