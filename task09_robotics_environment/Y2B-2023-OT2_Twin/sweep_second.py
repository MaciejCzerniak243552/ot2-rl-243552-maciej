from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from sim_wrapper import OT2Env

# Defaults aligned with the "Second" section of benchmark.ipynb
DEFAULT_TOTAL_TIMESTEPS = 300_000
DEFAULT_EVAL_EPISODES = 50
DEFAULT_BASE_SEED = 42
RESULTS_CSV = "sweep_results.csv"
WANDB_ENTITY = "242621-breda-university-of-applied-sciences"
WANDB_PROJECT = "ot2-rl-243552-300k_2"
os.environ['WANDB_API_KEY'] = "b1e375dd07d0792bb5601ffbb8b45cf2f84f5d20"


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(seed: Optional[int] = None, render_mode: Optional[str] = None) -> Monitor:
    env = OT2Env(render_mode=render_mode)
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------

def build_sweep_config() -> Dict[str, object]:
    """Bayesian sweep for SAC (mirrors the notebook Second section)."""
    return {
        "method": "bayes",
        "metric": {
            "name": "eval/mean_reward",
            "goal": "maximize",
        },
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "max": 5e-3,
                "min": 2e-6,
            },
            "gamma": {"values": [0.98, 0.99]},
            "batch_size": {"values": [256, 512]},
            "tau": {"values": [0.005, 0.01]},
            "ent_coef": {"values": ["auto_1.0"]},
            "learning_starts": {"values": [1000, 5000]},
            "train_freq": {"values": [1, 2]},
            "gradient_steps": {"values": [2, 4]},
            "target_entropy": {"values": ["auto", -2]},
        },
    }


# ---------------------------------------------------------------------------
# Training/evaluation for one sweep run
# ---------------------------------------------------------------------------

def sweep_train(
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = DEFAULT_BASE_SEED,
):
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        config={
            "total_timesteps": total_timesteps,
            "algo": "SAC",
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "batch_size": 256,
            "tau": 0.005,
            "ent_coef": "auto",
            "learning_starts": 1_000,
            "train_freq": 1,
            "gradient_steps": 1,
            "target_entropy": "auto",
        },
        sync_tensorboard=True,
        monitor_gym=True,
        settings=wandb.Settings(symlink=False),
    )
    config = wandb.config

    os.makedirs("models", exist_ok=True)

    env = make_env(seed=seed)
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        batch_size=config.batch_size,
        tau=config.tau,
        ent_coef=config.ent_coef,
        learning_starts=config.learning_starts,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        seed=seed,
        tensorboard_log=f"runs/{run.id}",
    )

    callbacks = [WandbCallback(log="all", verbose=1)]

    start_time = time.time()
    model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
    train_time = time.time() - start_time

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=eval_episodes,
        deterministic=True,
    )

    final_dists: List[float] = []
    successes = 0
    base_env = env.env if hasattr(env, "env") else env
    tol = getattr(base_env, "success_threshold", 0.001)

    for _ in range(eval_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        info: Dict[str, object] = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

        dist = info.get("distance", None)
        if dist is not None:
            dist = float(dist)
            final_dists.append(dist)
            if dist < tol:
                successes += 1

    mean_final_distance = float(np.mean(final_dists)) if final_dists else float("nan")
    std_final_distance = float(np.std(final_dists)) if final_dists else float("nan")
    success_rate = successes / eval_episodes if eval_episodes > 0 else float("nan")

    print(
        f"[SAC] mean_reward={mean_reward:.2f} +/- {std_reward:.2f}, "
        f"mean_final_dist={mean_final_distance:.4f} +/- {std_final_distance:.4f}, "
        f"success_rate={success_rate:.2%}, train_time={train_time/60:.1f} min"
    )

    wandb.define_metric("eval/mean_final_distance", summary="min")
    wandb.define_metric("eval/std_final_distance", summary="min")
    wandb.define_metric("eval/success_rate", summary="max")
    wandb.define_metric("eval/success_count", summary="max")
    wandb.define_metric("eval/mean_reward", summary="max")
    wandb.define_metric("eval/std_reward", summary="min")

    wandb.log(
        {
            "eval/mean_reward": mean_reward,
            "eval/std_reward": std_reward,
            "eval/mean_final_distance": mean_final_distance,
            "eval/std_final_distance": std_final_distance,
            "eval/success_rate": success_rate,
            "eval/success_count": successes,
            "train_time_sec": train_time,
        },
        step=model.num_timesteps,
    )

    run.summary["eval/mean_reward"] = mean_reward
    run.summary["eval/std_reward"] = std_reward
    run.summary["eval/mean_final_distance"] = mean_final_distance
    run.summary["eval/std_final_distance"] = std_final_distance
    run.summary["eval/success_rate"] = success_rate
    run.summary["eval/success_count"] = successes
    run.summary["train_time_sec"] = train_time

    result_row = {
        "run_id": run.id,
        "algorithm": "SAC",
        "train_steps": int(config.total_timesteps),
        "eval_episodes": eval_episodes,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_final_distance": mean_final_distance,
        "std_final_distance": std_final_distance,
        "success_rate": success_rate,
        "success_count": successes,
        "train_time_sec": train_time,
        "model_path": f"models/{run.id}_final.zip",
    }

    file_exists = os.path.exists(RESULTS_CSV)
    pd.DataFrame([result_row]).to_csv(
        RESULTS_CSV,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
    )

    model.save(result_row["model_path"])
    env.close()
    wandb.finish()


# ---------------------------------------------------------------------------
# Sweep launcher
# ---------------------------------------------------------------------------

def launch_sweep(
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    count: int = 10,
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
):
    sweep_id = wandb.sweep(build_sweep_config(), project=project, entity=entity)
    wandb.agent(sweep_id, function=lambda: sweep_train(total_timesteps=total_timesteps), count=count)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAC sweep runner for OT2Env")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Run a single SAC sweep trial (wandb agent entrypoint).")
    run_p.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    run_p.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    run_p.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)

    sweep_p = sub.add_parser("launch", help="Create sweep and launch agents.")
    sweep_p.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    sweep_p.add_argument("--count", type=int, default=10)
    sweep_p.add_argument("--project", type=str, default=WANDB_PROJECT)
    sweep_p.add_argument("--entity", type=str, default=WANDB_ENTITY)

    args = parser.parse_args()

    if args.cmd == "launch":
        launch_sweep(
            total_timesteps=args.timesteps,
            count=args.count,
            project=args.project,
            entity=args.entity,
        )
    else:
        sweep_train(
            total_timesteps=getattr(args, "timesteps", DEFAULT_TOTAL_TIMESTEPS),
            eval_episodes=getattr(args, "eval_episodes", DEFAULT_EVAL_EPISODES),
            seed=getattr(args, "seed", DEFAULT_BASE_SEED),
        )


if __name__ == "__main__":
    main()
