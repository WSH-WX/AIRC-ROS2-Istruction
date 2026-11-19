"""Evaluate a frozen AIRC-ROS2-Istruction policy on reproducible MuJoCo episodes."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from env.mujoco_env import MujocoNavConfig, MujocoNavEnv
from eval.ckpt_loader import resolve_ckpt_configs, build_policy_for_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--seed', type=int, default=10000,
                        help='Episode i uses seed + i; matches the 64-episode evaluation')
    parser.add_argument('--fps', type=float, default=10,
                        help='Displayed control steps per second; 0 disables pacing')
    parser.add_argument('--headless', action='store_true', help='Evaluate without a window or pacing')
    parser.add_argument('--device', default='cpu', help='Inference device; CPU is sufficient for one robot')
    args = parser.parse_args()
    if args.episodes < 1 or args.fps < 0:
        parser.error('--episodes must be positive and --fps nonnegative')
    if not args.ckpt.is_file():
        parser.error(f'Checkpoint not found: {args.ckpt}')
    torch.set_num_threads(2)
    train, config, models, key, source = resolve_ckpt_configs(args.ckpt)
    device = torch.device(args.device)
    env = MujocoNavEnv(MujocoNavConfig.from_dict(config),
                       render_mode=None if args.headless else 'human')
    results = []
    try:
        policy = build_policy_for_eval(train, models, key, env.observation_space.shape[0], device)
        payload = torch.load(args.ckpt, map_location=device, weights_only=False)
        policy.load_state_dict(payload['policy'], strict=True)
        policy.eval()
        policy.requires_grad_(False)
        limits = torch.as_tensor(env.get_limits(), device=device)
        print(f'[test] checkpoint={args.ckpt} step={payload.get("step")} config={source}', flush=True)
        print('[test] Frozen policy, deterministic actions, no optimizer or training output', flush=True)
        with torch.inference_mode():
            for episode in range(args.episodes):
                seed = args.seed + episode
                obs, _ = env.reset(seed=seed)
                total = 0.0
                if not args.headless:
                    env.render()
                for step in range(env.cfg.max_episode_steps):
                    started = time.monotonic()
                    if not args.headless and not env._viewer.is_running():
                        print('[test] Window closed', flush=True)
                        return
                    action = policy.act_deterministic(
                        torch.as_tensor(obs, device=device).unsqueeze(0), limits
                    ).squeeze(0).cpu().numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    if not args.headless:
                        env.render()
                        if args.fps > 0:
                            time.sleep(max(0, 1 / args.fps - (time.monotonic() - started)))
                    if terminated or truncated:
                        break
                outcome = 'success' if info['success'] else 'collision' if info['collided'] else 'timeout'
                results.append({'seed': seed, 'outcome': outcome, 'steps': step + 1, 'reward': total})
                print(f'[test] {episode+1}/{args.episodes} seed={seed} {outcome} '
                      f'steps={step+1} sim_seconds={(step+1)*env.cfg.dt:.1f} reward={total:.3f}', flush=True)
        summary = {key: sum(r['outcome'] == key for r in results)
                   for key in ('success', 'collision', 'timeout')}
        summary['episodes'] = len(results)
        summary['success_rate'] = summary['success'] / len(results)
        summary['mean_reward'] = float(np.mean([r['reward'] for r in results]))
        print('[test] ' + json.dumps(summary), flush=True)
    except KeyboardInterrupt:
        print('\n[test] Interrupted; checkpoint unchanged', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
