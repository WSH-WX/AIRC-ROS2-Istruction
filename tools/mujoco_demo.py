from __future__ import annotations

"""Open the randomized MuJoCo navigation scene for an interactive demo."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from env.mujoco_env import MujocoNavConfig, MujocoNavEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--controller", choices=("goal", "random"), default="goal")
    parser.add_argument("--env-config", type=Path, default=Path("config/env_config.json"))
    args = parser.parse_args()

    with args.env_config.open("r", encoding="utf-8") as f:
        cfg = MujocoNavConfig.from_dict(json.load(f))
    env = MujocoNavEnv(cfg, render_mode="human")
    try:
        obs, info = env.reset(seed=args.seed)
        print(f"seed={args.seed} goal={info['goal'].round(2).tolist()} rays={env.n_rays}")
        finished = 0
        while finished < args.episodes:
            if args.controller == "random":
                action = env.action_space.sample()
            else:
                delta = env.goal - env.pos
                angle = math.atan2(delta[1], delta[0]) - env.yaw
                angle = (angle + math.pi) % (2 * math.pi) - math.pi
                omega = np.clip(2.0 * angle, -cfg.omega_max, cfg.omega_max)
                vx = cfg.vx_max * max(0.0, math.cos(angle))
                action = np.array((vx, omega), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                finished += 1
                print(f"episode={finished} success={info['success']} collision={info['collided']} reward={reward:.3f}")
                if finished < args.episodes:
                    obs, info = env.reset(seed=args.seed + finished)
                    print(f"seed={args.seed + finished} goal={info['goal'].round(2).tolist()}")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
