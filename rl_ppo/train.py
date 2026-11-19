from __future__ import annotations

"""Entry point for PPO training using randomized MuJoCo navigation scenes.

Usage:
    python -m rl_ppo.train --fresh [--tag NAME]
    python -m rl_ppo.train --fresh --resume <path-or-tag> [--tag NAME] [--opt]
    python -m rl_ppo.train --resume <path-or-tag>

train_config / env_config / model_config are always read from config/ (for
--fresh modes) or from the target run directory (for pure --resume).
"""

import os
import re
import shutil
import time
import argparse
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch

from .buffer import RolloutBuffer
from .writer import close_writer, get_writer, init_writer
from env import load_json_config
from env.mujoco_env import MujocoBatchEnv
from models import PPOPolicy, build_encoder, resolve_model_entry


DEFAULT_TRAIN_CONFIG = os.path.join("config", "train_config.json")
_STEP_CKPT_PATTERN = re.compile(r"^step-(\d+)\.pt$")
_RUN_DIR_PATTERN = re.compile(r"^(\d{8}-\d{6})(?:-(.+))?$")


def load_train_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = load_json_config(path) if path else {}
    cfg.setdefault("device", "cuda:0")
    cfg.setdefault("env_config", "env_config.json")
    cfg.setdefault("model_config", "model_config.json")
    cfg.setdefault("model", "circular_attn")
    samp = cfg.setdefault("sampling", {})
    samp.setdefault("batch_env", 16)
    samp.setdefault("rollout_len", 128)
    samp.setdefault("reset_each_rollout", True)
    ppo = cfg.setdefault("ppo", {})
    ppo.setdefault("gamma", 0.99)
    ppo.setdefault("gae_lambda", 0.95)
    ppo.setdefault("clip_range", 0.2)
    ppo.setdefault("lr", 3e-4)
    ppo.setdefault("value_lr", 3e-4)
    ppo.setdefault("entropy_coef", 0.0)
    ppo.setdefault("value_coef", 0.5)
    ppo.setdefault("max_grad_norm", 0.5)
    ppo.setdefault("epochs", 4)
    ppo.setdefault("minibatch_size", 2048)
    ppo.setdefault("amp", True)
    ppo.setdefault("amp_bf16", True)
    ppo.setdefault("bootstrap", True)
    ppo.setdefault("log_std_min", -5.0)
    ppo.setdefault("log_std_max", 2.0)
    ppo.setdefault("collision_done", True)

    run = cfg.setdefault("run", {})
    run.setdefault("total_env_steps", 2_000_000)
    run.setdefault("ckpt_dir", "runs")
    run.setdefault("log_interval", 20000)
    run.setdefault("save_interval", 200000)
    run.setdefault("resume_as_additional", False)
    return cfg


def _extract_seed_from_train_or_env(train_cfg: Dict[str, Any], env_cfg: Dict[str, Any]) -> Optional[int]:
    try:
        run = train_cfg.get("run", {}) or {}
        seed_v = run.get("seed", None)
        if seed_v in (None, "", "null"):
            seed_v = None
        if seed_v is not None:
            return int(seed_v)
        sim = env_cfg.get("sim", {}) or {}
        s2 = sim.get("seed", None)
        return None if s2 in (None, "", "null") else int(s2)
    except Exception:
        return None


def _resolve_sibling_cfg_path(train_cfg_path: str, key: str,
                              train_cfg: Dict[str, Any]) -> Optional[str]:
    val = train_cfg.get(key, None)
    if not val:
        return None
    if os.path.isabs(val):
        return val
    return os.path.join(os.path.dirname(os.path.abspath(train_cfg_path)), val)


def _resolve_env_cfg_path(train_cfg_path: str, train_cfg: Dict[str, Any]) -> Optional[str]:
    return _resolve_sibling_cfg_path(train_cfg_path, "env_config", train_cfg)


def _resolve_model_cfg_path(train_cfg_path: str, train_cfg: Dict[str, Any]) -> Optional[str]:
    return _resolve_sibling_cfg_path(train_cfg_path, "model_config", train_cfg)


def _setup_fresh_run_dir(runs_root: str, tag: Optional[str],
                         train_cfg_src: str, env_cfg_src: str,
                         model_cfg_src: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{ts}-{tag}" if tag else ts
    run_dir = os.path.join(runs_root, name)
    os.makedirs(run_dir, exist_ok=False)
    shutil.copy2(train_cfg_src, os.path.join(run_dir, "train_config.json"))
    shutil.copy2(env_cfg_src, os.path.join(run_dir, "env_config.json"))
    shutil.copy2(model_cfg_src, os.path.join(run_dir, "model_config.json"))
    return run_dir


def _pick_latest_ckpt(run_dir: str) -> str:
    latest = os.path.join(run_dir, "latest.pt")
    if os.path.isfile(latest):
        return latest
    candidates = []
    for name in os.listdir(run_dir):
        m = _STEP_CKPT_PATTERN.match(name)
        if m:
            candidates.append((int(m.group(1)), os.path.join(run_dir, name)))
    if not candidates:
        raise FileNotFoundError(f"No latest.pt or step-<N>.pt in {run_dir}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def _find_run_dir_by_tag(runs_root: str, tag: str) -> Optional[str]:
    if not os.path.isdir(runs_root):
        return None
    matches = []
    for name in os.listdir(runs_root):
        m = _RUN_DIR_PATTERN.match(name)
        if not m or m.group(2) != tag:
            continue
        full = os.path.join(runs_root, name)
        if os.path.isdir(full):
            matches.append(full)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def _resolve_resume_target(value: str, runs_root: Optional[str]) -> Tuple[str, str]:
    if os.path.isfile(value):
        return os.path.dirname(os.path.abspath(value)), os.path.abspath(value)
    if os.path.isdir(value):
        run_dir = os.path.abspath(value)
        return run_dir, _pick_latest_ckpt(run_dir)
    if runs_root is None:
        raise FileNotFoundError(f"--resume target {value!r} is not an existing file or directory")
    run_dir = _find_run_dir_by_tag(runs_root, value)
    if run_dir is None:
        raise FileNotFoundError(
            f"--resume target {value!r}: not a path, and no `*-{value}` folder under {runs_root}"
        )
    return run_dir, _pick_latest_ckpt(run_dir)


def _load_checkpoint(ckpt_file: str, policy: PPOPolicy,
                     opt_pi: Optional[torch.optim.Optimizer]) -> int:
    payload = torch.load(ckpt_file, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint format at {ckpt_file}")
    step_v = int(payload.get("step", payload.get("global_step", 0)))
    state = payload.get("policy", None)
    if state is None:
        keys = list(payload.keys())
        if all(isinstance(k, str) for k in keys) and any(
            k.startswith(("encoder", "mu", "value", "log_std")) for k in keys
        ):
            state = payload
    if state is not None:
        policy.load_state_dict(state, strict=False)
    if opt_pi is not None and "opt" in payload:
        try:
            opt_pi.load_state_dict(payload["opt"])
        except Exception:
            print("[PPO] Warning: failed to load optimizer state; continuing without it.")
    return step_v


def _save_checkpoint(run_dir: str, global_step: int,
                     policy: PPOPolicy, opt_pi: torch.optim.Optimizer) -> str:
    step_path = os.path.join(run_dir, f"step-{int(global_step)}.pt")
    torch.save({
        "step": int(global_step),
        "policy": policy.state_dict(),
        "opt": opt_pi.state_dict(),
    }, step_path)
    shutil.copy2(step_path, os.path.join(run_dir, "latest.pt"))
    return step_path


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true",
                        help="Create a new run folder under <ckpt_dir>/<timestamp>[-<tag>]/")
    parser.add_argument("--resume", type=str, default=None,
                        help="Run dir, step-<N>.pt file, or a tag (matches the latest `*-<tag>` folder by mtime)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Name the new run folder (only valid with --fresh)")
    parser.add_argument("--opt", action="store_true",
                        help="Also load optimizer state on warm-start (only valid with --fresh --resume; pure --resume always loads it)")
    parser.add_argument("--port", type=int, default=6006,
                        help="TensorBoard server port (default: 6006)")
    parser.add_argument("--render", action="store_true",
                        help="Show the first PPO training environment in a MuJoCo window")
    parser.add_argument("--render-every", type=int, default=1,
                        help="Update the viewer every N vector steps (default: 1)")
    parser.add_argument("--render-fps", type=float, default=0.0,
                        help="Cap viewer updates per second; 0 runs without pacing")
    parser.add_argument("--total-env-steps", type=int, default=None,
                        help="Override the target environment steps, saved to the run config")
    args = parser.parse_args()
    if not args.fresh and not args.resume:
        parser.error("at least one of --fresh or --resume is required")
    if args.tag and not args.fresh:
        parser.error("--tag is only valid with --fresh")
    if args.opt and not (args.fresh and args.resume):
        parser.error("--opt is only valid with --fresh --resume")
    if args.render_every < 1 or args.render_fps < 0:
        parser.error("--render-every must be >= 1 and --render-fps must be >= 0")
    if args.total_env_steps is not None and args.total_env_steps < 1:
        parser.error("--total-env-steps must be >= 1")
    return args


def main():
    args = _parse_cli()

    if args.fresh:
        if not os.path.isfile(DEFAULT_TRAIN_CONFIG):
            raise FileNotFoundError(f"train_config not found: {DEFAULT_TRAIN_CONFIG}")
        external_cfg = load_train_config(DEFAULT_TRAIN_CONFIG)
        external_env_cfg_src = _resolve_env_cfg_path(DEFAULT_TRAIN_CONFIG, external_cfg)
        if not external_env_cfg_src or not os.path.isfile(external_env_cfg_src):
            raise FileNotFoundError(f"env_config not found: {external_env_cfg_src}")
        external_model_cfg_src = _resolve_model_cfg_path(DEFAULT_TRAIN_CONFIG, external_cfg)
        if not external_model_cfg_src or not os.path.isfile(external_model_cfg_src):
            raise FileNotFoundError(f"model_config not found: {external_model_cfg_src}")
        runs_root = external_cfg["run"]["ckpt_dir"]

        resume_ckpt: Optional[str] = None
        if args.resume:
            _, resume_ckpt = _resolve_resume_target(args.resume, runs_root)

        run_dir = _setup_fresh_run_dir(
            runs_root, args.tag,
            DEFAULT_TRAIN_CONFIG, external_env_cfg_src, external_model_cfg_src,
        )
        if resume_ckpt:
            print(f"[PPO] Fresh run at {run_dir} (warm-starting from {resume_ckpt})")
        else:
            print(f"[PPO] Fresh run at {run_dir}")
    else:
        runs_root_for_tag: Optional[str] = None
        if not (os.path.isfile(args.resume) or os.path.isdir(args.resume)):
            if not os.path.isfile(DEFAULT_TRAIN_CONFIG):
                raise FileNotFoundError(
                    f"Cannot resolve --resume {args.resume!r} as a path; "
                    f"tag lookup requires {DEFAULT_TRAIN_CONFIG}"
                )
            runs_root_for_tag = load_train_config(DEFAULT_TRAIN_CONFIG)["run"]["ckpt_dir"]
        run_dir, resume_ckpt = _resolve_resume_target(args.resume, runs_root_for_tag)
        print(f"[PPO] Resuming in {run_dir} from {resume_ckpt}")

    train_cfg_path = os.path.join(run_dir, "train_config.json")
    env_cfg_path = os.path.join(run_dir, "env_config.json")
    model_cfg_path = os.path.join(run_dir, "model_config.json")
    if not os.path.isfile(train_cfg_path):
        raise FileNotFoundError(f"Missing train_config.json in {run_dir}")
    if not os.path.isfile(env_cfg_path):
        raise FileNotFoundError(f"Missing env_config.json in {run_dir}")
    if not os.path.isfile(model_cfg_path):
        raise FileNotFoundError(f"Missing model_config.json in {run_dir}")
    cfg = load_train_config(train_cfg_path)
    if args.total_env_steps is not None:
        import json
        cfg["run"]["total_env_steps"] = args.total_env_steps
        with open(train_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    env_cfg = load_json_config(env_cfg_path)
    model_configs = load_json_config(model_cfg_path)

    device = torch.device(
        cfg.get("device", "cuda:0") if (cfg.get("device", "cuda:0") == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    ppo_cfg = cfg.get("ppo", {}) or {}

    B_env = int((cfg.get("sampling", {}) or {}).get("batch_env", 256))
    T_roll = int((cfg.get("sampling", {}) or {}).get("rollout_len", 128))
    reset_each_rollout = bool((cfg.get("sampling", {}) or {}).get("reset_each_rollout", True))
    run_seed = _extract_seed_from_train_or_env(cfg, env_cfg)
    if run_seed is not None:
        torch.manual_seed(int(run_seed))
    env = MujocoBatchEnv(env_cfg, n_envs=B_env, device=str(device), seed=run_seed)
    obs = env.reset()
    if args.render:
        env.envs[0].render_mode = "human"
        env.envs[0].render()
        print("[PPO] MuJoCo viewer: environment 0, actions from the training policy")
    vec_dim = int(obs.shape[1])
    act_dim = 2

    model_key = str(cfg.get("model", "circular_attn"))
    model_entry = resolve_model_entry(model_configs, model_key)
    encoder = build_encoder(model_entry, vec_dim=vec_dim)
    policy = PPOPolicy(
        encoder=encoder,
        action_dim=act_dim,
        log_std_min=float((cfg.get("ppo", {}) or {}).get("log_std_min", -5.0)),
        log_std_max=float((cfg.get("ppo", {}) or {}).get("log_std_max", 2.0)),
    ).to(device)
    print(f"[PPO] model={model_key!r} ({model_entry.get('name')}) | vec_dim={vec_dim} | feature_dim={encoder.feature_dim}")
    ppo_cfg = cfg.get("ppo", {}) or {}
    opt_pi = torch.optim.Adam([
        {"params": list(policy.encoder.parameters()) + list(policy.mu.parameters()) + [policy.log_std], "lr": float(ppo_cfg.get("lr", 3e-4))},
        {"params": policy.value.parameters(), "lr": float(ppo_cfg.get("value_lr", ppo_cfg.get("lr", 3e-4)))}
    ])

    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    use_amp = bool(ppo_cfg.get("amp", True)) and use_cuda
    prefer_bf16 = bool(ppo_cfg.get("amp_bf16", True))
    amp_dtype = torch.bfloat16 if (use_amp and prefer_bf16 and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    global_step = 0
    if resume_ckpt is not None:
        load_opt = (not args.fresh) or args.opt
        loaded_step = _load_checkpoint(resume_ckpt, policy, opt_pi if load_opt else None)
        if args.fresh:
            opt_str = "policy+opt" if args.opt else "policy only"
            print(f"[PPO] Warm-started ({opt_str}) from src step={loaded_step:,}; counter reset to 0")
        else:
            global_step = loaded_step
            print(f"[PPO] Loaded step={global_step:,}")
    log_interval = int((cfg.get("run", {}) or {}).get("log_interval", 20000))
    save_interval = int((cfg.get("run", {}) or {}).get("save_interval", 200000))

    total_env_steps = int(cfg["run"]["total_env_steps"])
    if bool((cfg.get("run", {}) or {}).get("resume_as_additional", False)) and global_step > 0:
        total_env_steps = int(global_step) + int(cfg["run"]["total_env_steps"])
        print(f"[PPO] resume_as_additional=True => target_total_env_steps={total_env_steps:,} (loaded {global_step:,} + add {int(cfg['run']['total_env_steps']):,})")
    clip_eps = float(ppo_cfg.get("clip_range", 0.2))
    ent_coef = float(ppo_cfg.get("entropy_coef", 0.0))
    vf_coef = float(ppo_cfg.get("value_coef", 0.5))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
    epochs = int(ppo_cfg.get("epochs", 4))
    mb_size = int(ppo_cfg.get("minibatch_size", 2048))
    gamma = float(ppo_cfg.get("gamma", 0.99))
    lam = float(ppo_cfg.get("gae_lambda", 0.95))

    limits = env.get_limits()              # [A, 2] action bounds (lo, hi)
    limits_b = limits.unsqueeze(0).expand(B_env, -1, -1)

    init_writer(
        logdir=cfg["run"]["ckpt_dir"],
        name=os.path.basename(os.path.normpath(run_dir)),
        port=args.port,
        log="training",
    )
    writer = get_writer()

    roll_reward = torch.zeros((), device=device)
    roll_collide = torch.zeros((), device=device)
    roll_steps_total = 0
    episode_records = []
    reset_cuts = 0
    # Sample-weighted minibatch diagnostics over the whole logging interval.
    diagnostic_sums = torch.zeros(7, device=device)
    diagnostic_samples = 0
    ev_variance_sum = torch.zeros(2, device=device)
    ev_samples = 0

    last_log_step = global_step
    last_log_time = time.time()
    last_save_step = global_step
    render_step = 0
    last_render_time = time.monotonic()

    while global_step < total_env_steps:
        if reset_each_rollout:
            obs = env.reset()
            reset_cuts += env.last_reset_cuts
        buf = RolloutBuffer(T_roll, B_env, obs_dim=obs.shape[1], act_dim=act_dim, device=device)
        with torch.no_grad():
            for _ in range(T_roll):
                out = policy.act(obs, limits_b)
                logp = out.logp
                act = out.action
                _, _, v = policy._core(obs)
                next_obs, reward_t, done_t, info = env.step(act)
                render_step += 1
                if args.render and render_step % args.render_every == 0:
                    viewer = env.envs[0]._viewer
                    if not viewer.is_running():
                        _save_checkpoint(run_dir, global_step + B_env, policy, opt_pi)
                        close_writer()
                        env.close()
                        print("[PPO] Viewer closed; checkpoint saved, training stopped")
                        return
                    if args.render_fps > 0:
                        remaining = 1.0 / args.render_fps - (time.monotonic() - last_render_time)
                        if remaining > 0:
                            time.sleep(remaining)
                    env.envs[0].render()
                    last_render_time = time.monotonic()
                d = done_t
                buf.add(
                    obs=obs.detach(),
                    act=act.detach(),
                    logp=logp.detach(),
                    rew=reward_t.detach(),
                    done=d.detach().to(torch.float32),
                    val=v.detach(),
                    limits=limits_b.detach(),
                )

                roll_reward += reward_t.sum()
                roll_collide += info["collided"].to(torch.float32).sum()
                roll_steps_total += B_env
                episode_records.extend(info['episodes'])

                global_step += B_env
                obs = next_obs

        if bool(ppo_cfg.get("bootstrap", True)):
            with torch.no_grad():
                _, _, last_v = policy._core(obs)
                last_v = last_v.view(B_env, 1)
            buf.compute_gae(last_v, gamma=gamma, lam=lam, count_timeout_as_done=True)
        else:
            buf.compute_mc_returns(gamma=gamma)
        with torch.no_grad():
            count = buf.returns.numel()
            ev_variance_sum += torch.stack([
                buf.returns.var(unbiased=False),
                (buf.returns - buf.values).var(unbiased=False),
            ]) * count
            ev_samples += count
        for _ in range(epochs):
            for mb in buf.minibatches(mb_size):
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    new_logp, ent, v_pred = policy.evaluate_actions(mb.obs, mb.actions, mb.limits)
                    ratio = (new_logp - mb.logp).exp().clamp(0.0, 10.0)
                    surr1 = ratio * mb.advantages
                    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * mb.advantages
                    pg_loss = -torch.min(surr1, surr2).mean()
                    v_loss = 0.5 * (mb.returns - v_pred).pow(2).mean()
                    ent_bonus = ent.mean()
                    loss = pg_loss + vf_coef * v_loss - ent_coef * ent_bonus

                opt_pi.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt_pi)
                else:
                    loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_grad_norm, error_if_nonfinite=True)
                with torch.no_grad():
                    log_ratio = (new_logp - mb.logp).float()
                    approx_kl = (torch.expm1(log_ratio) - log_ratio).mean()
                    clip_fraction = (log_ratio.exp().sub(1).abs() > clip_eps).float().mean()
                    metrics = torch.stack([pg_loss.detach().float(), v_loss.detach().float(),
                                           loss.detach().float(), ent_bonus.detach().float(),
                                           approx_kl, clip_fraction, grad_norm.detach().float()])
                    if not torch.isfinite(metrics).all():
                        raise FloatingPointError('Non-finite PPO loss/diagnostic; update stopped')
                    diagnostic_sums += metrics * mb.obs.shape[0]
                    diagnostic_samples += mb.obs.shape[0]
                if scaler.is_enabled():
                    scaler.step(opt_pi)
                    scaler.update()
                else:
                    opt_pi.step()
        if global_step - last_log_step >= log_interval or global_step >= total_env_steps:
            now = time.time()
            fps = (global_step - last_log_step) / max(1e-3, now - last_log_time)
            n_steps = max(1, roll_steps_total)
            names = ('policy_loss', 'value_loss', 'total_loss', 'entropy',
                     'approx_kl', 'clip_fraction', 'grad_norm')
            diagnostics = dict(zip(names, (diagnostic_sums / max(1, diagnostic_samples)).tolist()))
            variance, residual = (ev_variance_sum / max(1, ev_samples)).tolist()
            if variance > 1e-12:
                diagnostics['explained_variance'] = 1.0 - residual / variance
            std = policy.log_std.detach().clamp(policy._log_std_min, policy._log_std_max).exp().tolist()
            diagnostics.update(std_vx=std[0], std_omega=std[1])
            rollout_metrics = {'collision_rate': (roll_collide / n_steps).item(),
                               'step_collision_rate': (roll_collide / n_steps).item(),
                               'completed_episodes': len(episode_records),
                               'reset_cut_episodes': reset_cuts, 'fps': fps}
            episode_metrics = {}
            if episode_records:
                n = len(episode_records)
                episode_metrics = {f'{outcome}_rate': sum(e['outcome'] == outcome for e in episode_records) / n
                                   for outcome in ('success', 'collision', 'timeout')}
                episode_metrics['mean_return'] = sum(e['return'] for e in episode_records) / n
                episode_metrics['mean_length'] = sum(e['length'] for e in episode_records) / n
                success_lengths = [e['length'] for e in episode_records if e['outcome'] == 'success']
                if success_lengths:
                    episode_metrics['success_mean_seconds'] = sum(success_lengths) / len(success_lengths) * env.cfg.dt
            if writer is not None:
                writer.log({"mean_reward": (roll_reward / n_steps).item()},
                           step=global_step, scope="learner")
                writer.log(rollout_metrics, step=global_step, scope='rollout')
                writer.log(diagnostics, step=global_step, scope='ppo')
                if episode_metrics:
                    writer.log(episode_metrics, step=global_step, scope='episode')
            outcome_text = ('no completed episodes' if not episode_metrics else
                            f"success={episode_metrics['success_rate']:.1%} "
                            f"collision={episode_metrics['collision_rate']:.1%} "
                            f"timeout={episode_metrics['timeout_rate']:.1%}")
            print(f"[PPO] step={global_step:,} | fps={fps:.1f} | reward={(roll_reward/n_steps).item():.4f} | "
                  f"pi_loss={diagnostics['policy_loss']:.4f} v_loss={diagnostics['value_loss']:.4f} "
                  f"KL={diagnostics['approx_kl']:.4f} clip={diagnostics['clip_fraction']:.1%} | "
                  f"episodes={len(episode_records)} {outcome_text} reset_cuts={reset_cuts}", flush=True)
            roll_reward.zero_()
            roll_collide.zero_()
            roll_steps_total = 0
            episode_records.clear()
            reset_cuts = 0
            diagnostic_sums.zero_()
            diagnostic_samples = 0
            ev_variance_sum.zero_()
            ev_samples = 0
            last_log_step = global_step
            last_log_time = now

        if global_step - last_save_step >= save_interval:
            _save_checkpoint(run_dir, global_step, policy, opt_pi)
            last_save_step = global_step

    _save_checkpoint(run_dir, global_step, policy, opt_pi)
    close_writer()
    env.close()


if __name__ == "__main__":
    main()
