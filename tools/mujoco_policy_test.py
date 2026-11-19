"""Evaluate a frozen AIRC-ROS2-Istruction policy on reproducible MuJoCo episodes."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from env.mujoco_env import MujocoNavConfig, MujocoNavEnv
from eval.ckpt_loader import resolve_ckpt_configs, build_policy_for_eval


def save_evaluation(output, checkpoint, summary, results, scenes, arena_size):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(json.dumps({
        'checkpoint': str(checkpoint.resolve()), 'summary': summary,
        'episodes': results, 'trajectories': scenes,
    }, indent=2), encoding='utf-8')
    cols = min(3, len(scenes))
    rows = (len(scenes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 6.5 * rows), squeeze=False)
    half = arena_size / 2
    for ax, scene in zip(axes.flat, scenes):
        for x, y, hx, hy in scene['obstacles']:
            ax.add_patch(Rectangle((x-hx, y-hy), 2*hx, 2*hy, color='gray'))
        path = np.asarray(scene['positions'])
        ax.plot(path[:, 0], path[:, 1], color='#2166ac', linewidth=1.5)
        ax.scatter(*path[0], color='green', s=35, zorder=3)
        ax.scatter(*scene['goal'], color='red', marker='*', s=160, zorder=3)
        ax.add_patch(Circle(scene['goal'], scene['goal_radius'], fill=False, color='red', linewidth=.8))
        ax.set(xlim=(-half, half), ylim=(-half, half), aspect='equal',
               title=f"Seed {scene['seed']}: {scene['outcome']}, {scene['steps']} steps")
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(scenes):]:
        ax.set_visible(False)
    n = summary['episodes']
    fig.suptitle('AIRC-ROS2-Istruction deterministic evaluation | '
                 f"success {summary['success']}/{n}, collision {summary['collision']}/{n}, "
                 f"timeout {summary['timeout']}/{n}")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / 'trajectories.png', dpi=150)
    plt.close(fig)
    print(f'[test] Saved {output / "results.json"} and {output / "trajectories.png"}', flush=True)


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
    parser.add_argument('--output', type=Path, help='Save results.json and trajectories.png to this directory')
    parser.add_argument('--plot-episodes', type=int, default=6, help='Plot the first N episodes (default: 6)')
    args = parser.parse_args()
    if args.episodes < 1 or args.fps < 0 or args.plot_episodes < 1:
        parser.error('--episodes and --plot-episodes must be positive and --fps nonnegative')
    if not args.ckpt.is_file():
        parser.error(f'Checkpoint not found: {args.ckpt}')
    torch.set_num_threads(2)
    train, config, models, key, source = resolve_ckpt_configs(args.ckpt)
    device = torch.device(args.device)
    env = MujocoNavEnv(MujocoNavConfig.from_dict(config),
                       render_mode=None if args.headless else 'human')
    results = []
    scenes = []
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
                initial_distance = float(np.linalg.norm(env.goal - env.pos))
                positions = [env.pos.copy()]
                velocities = []
                scene = {
                    'seed': seed, 'goal': env.goal.tolist(), 'goal_radius': env.cfg.goal_radius,
                    'obstacles': [[*env.model.geom_pos[i, :2].tolist(),
                                   *env.model.geom_size[i, :2].tolist()] for i in env.obstacle_ids],
                } if args.output and episode < args.plot_episodes else None
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
                    positions.append(env.pos.copy())
                    velocities.append(float(action[0]))
                    total += reward
                    if not args.headless:
                        env.render()
                        if args.fps > 0:
                            time.sleep(max(0, 1 / args.fps - (time.monotonic() - started)))
                    if terminated or truncated:
                        break
                outcome = 'success' if info['success'] else 'collision' if info['collided'] else 'timeout'
                results.append({'seed': seed, 'outcome': outcome, 'steps': step + 1, 'reward': total,
                                'return_': total, 'initial_distance': initial_distance,
                                'final_distance': info['distance'],
                                'path_length': float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
                                'mean_vx': float(np.mean(velocities))})
                if scene is not None:
                    scene.update(outcome=outcome, steps=step+1,
                                 positions=np.asarray(positions).tolist())
                    scenes.append(scene)
                print(f'[test] {episode+1}/{args.episodes} seed={seed} {outcome} '
                      f'steps={step+1} sim_seconds={(step+1)*env.cfg.dt:.1f} reward={total:.3f}', flush=True)
        summary = {key: sum(r['outcome'] == key for r in results)
                   for key in ('success', 'collision', 'timeout')}
        summary['episodes'] = len(results)
        summary['success_rate'] = summary['success'] / len(results)
        summary['mean_reward'] = float(np.mean([r['reward'] for r in results]))
        summary['mean_return'] = summary['mean_reward']
        summary['mean_vx'] = float(np.mean([r['mean_vx'] for r in results]))
        summary['mean_distance_reduction'] = float(np.mean([
            r['initial_distance'] - r['final_distance'] for r in results]))
        success_steps = [r['steps'] for r in results if r['outcome'] == 'success']
        summary['success_mean_seconds'] = float(np.mean(success_steps) * env.cfg.dt) if success_steps else None
        print('[test] ' + json.dumps(summary), flush=True)
        if args.output:
            save_evaluation(args.output, args.ckpt, summary, results, scenes, env.cfg.arena_size)
    except KeyboardInterrupt:
        print('\n[test] Interrupted; checkpoint unchanged', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
