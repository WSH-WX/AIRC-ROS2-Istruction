"""Plot training.log scalars without requiring a GPU or importing the trainer."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PANELS = [
    ('Mean reward per step', ['learner/mean_reward']),
    ('Completed episode outcomes (fraction)', ['episode/success_rate', 'episode/collision_rate', 'episode/timeout_rate']),
    ('Completed episode return', ['episode/mean_return']),
    ('Completed episode length (steps)', ['episode/mean_length']),
    ('Successful episode time (seconds)', ['episode/success_mean_seconds']),
    ('Collisions per environment step', ['rollout/step_collision_rate', 'rollout/collision_rate']),
    ('Episode counts per log interval', ['rollout/completed_episodes', 'rollout/reset_cut_episodes']),
    ('PPO losses', ['ppo/policy_loss', 'ppo/value_loss', 'ppo/total_loss']),
    ('Approximate KL', ['ppo/approx_kl']),
    ('Clip fraction', ['ppo/clip_fraction']),
    ('Value explained variance (before update)', ['ppo/explained_variance']),
    ('Gaussian entropy', ['ppo/entropy']),
    ('Gaussian standard deviation', ['ppo/std_vx', 'ppo/std_omega']),
    ('Gradient norm before clipping', ['ppo/grad_norm']),
    ('Throughput (environment steps/second)', ['rollout/fps']),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Run directory containing training.log')
    parser.add_argument('--output', type=Path, help='Output directory; defaults to RUN/plots')
    parser.add_argument('--window', type=int, default=1, help='Trailing smoothing window in logged points')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be >= 1')
    series = defaultdict(dict)
    skipped = 0
    with (args.run / 'training.log').open(encoding='utf-8') as source:
        for line in source:
            try:
                row = json.loads(line)
                step = int(row['step'])
                for key, value in row['metrics'].items():
                    value = float(value)
                    if math.isfinite(value):
                        # Last record wins if a resumed run repeats a step.
                        series[f"{row['scope']}/{key}"][step] = value
            except (ValueError, TypeError, KeyError):
                skipped += 1
    if skipped:
        print(f'[plot] Skipped {skipped} malformed records (possibly an unfinished log line).')
    panels = []
    for title, keys in PANELS:
        available = [key for key in keys if series.get(key)]
        if 'rollout/step_collision_rate' in available:
            available.remove('rollout/collision_rate') if 'rollout/collision_rate' in available else None
        if available:
            panels.append((title, available))
    if not panels:
        parser.error('No supported finite metrics found in training.log')
    output = args.output or args.run / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as target:
        writer = csv.writer(target)
        writer.writerow(['metric', 'environment_step', 'value'])
        for key in sorted(series):
            for step, value in sorted(series[key].items()):
                writer.writerow([key, step, value])
    rows = (len(panels) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 3.5 * rows), squeeze=False)
    for ax, (title, keys) in zip(axes.flat, panels):
        for key in keys:
            points = sorted(series[key].items())
            x, y = zip(*points)
            smooth = [sum(y[max(0, i-args.window+1):i+1]) / min(i+1, args.window) for i in range(len(y))]
            if args.window > 1:
                ax.plot(x, y, alpha=.2, linewidth=.7)
            ax.plot(x, smooth, label=key.split('/')[-1], linewidth=1.3, marker='.' if len(x) < 10 else None)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Environment steps')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f'{args.run.name} | training metrics | smoothing={args.window}', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    path = output / 'training_metrics.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f'[plot] Saved {path} and {output / "metrics.csv"}')
    if not series.get('ppo/value_loss'):
        print('[plot] No PPO loss records: old logs cannot recover missing losses; restart/resume training with the updated trainer.')
    if not series.get('episode/success_rate'):
        print('[plot] No completed-episode rates recorded; no success rate has been inferred.')


if __name__ == '__main__':
    main()
