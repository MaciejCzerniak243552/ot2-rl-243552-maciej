"""
Rerun the best "once-success" config from a W&B sweep with longer training and multiple seeds.

Selection rule:
- Filter runs with success_count > 0; pick the one with the lowest eval/mean_final_distance.
- If none succeeded, fall back to the lowest eval/mean_final_distance overall.

Then train the picked config for longer (default 600k) across a few seeds, keeping entropy
from collapsing by using an entropy coefficient floor when the sweep used "auto".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import wandb
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from ot2_gym_wrapper import OT2Env


ENTITY = "242621-breda-university-of-applied-sciences"
PROJECT = "ot2-rl-243552-200k"
SWEEP_ID = "kowftx1t"

# How long to rerun and how many seeds
TOTAL_TIMESTEPS = 600_000
SEEDS = (0, 1, 2)
N_EVAL_EPISODES = 100

# Entropy floor to keep exploration alive when the sweep used "auto"
ENTROPY_FLOOR = "auto_0.5"

ALGORITHMS: Dict[str, object] = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "A2C": A2C,
}


@dataclass
class BestRun:
    name: str
    run_id: str
    algo: str
    cfg: Dict[str, object]
    success_count: int
    mean_final_distance: float


def make_env(seed: int | None = None):
    """Factory for a fresh monitored OT2 environment."""
    env = OT2Env(render=False, max_steps=1000, tol=0.001)
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def pick_best_run() -> BestRun:
    """Pick the best run: any success first, then lowest mean_final_distance."""
    api = wandb.Api()
    sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")

    rows: List[Dict[str, object]] = []
    for run in sweep.runs:
        cfg = {k: v for k, v in run.config.items() if not k.startswith("_")}
        summary = run.summary
        rows.append(
            {
                "run_id": run.id,
                "name": run.name,
                "algo": cfg.get("algo", "SAC"),
                "cfg": cfg,
                "success_count": summary.get("eval/success_count", 0),
                "mean_final_distance": summary.get("eval/mean_final_distance", np.inf),
            }
        )

    df = pd.DataFrame(rows)
    # Successful runs first; if none, pick smallest distance overall
    successful = df[df["success_count"] > 0]
    if not successful.empty:
        best_row = successful.sort_values("mean_final_distance").iloc[0]
    else:
        best_row = df.sort_values("mean_final_distance").iloc[0]

    return BestRun(
        name=best_row["name"],
        run_id=best_row["run_id"],
        algo=best_row["algo"],
        cfg=best_row["cfg"],
        success_count=int(best_row["success_count"]),
        mean_final_distance=float(best_row["mean_final_distance"]),
    )


def prepare_hparams(cfg: Dict[str, object]) -> Dict[str, object]:
    """Extract and cast only SAC-supported hyperparameters from a sweep config."""
    allowed = {
        "learning_rate",
        "gamma",
        "batch_size",
        "ent_coef",
        "learning_starts",
        "tau",
        "train_freq",
        "gradient_steps",
        "buffer_size",
        "use_sde",
        "sde_sample_freq",
        "target_entropy",
    }
    h: Dict[str, object] = {}
    # Keep exploration alive if sweep used plain "auto"
    ent_coef = cfg.get("ent_coef", "auto")
    h["ent_coef"] = ENTROPY_FLOOR if str(ent_coef) == "auto" else ent_coef

    for key in allowed:
        if key not in cfg:
            continue
        val = cfg[key]
        if key in ("batch_size", "learning_starts", "train_freq", "gradient_steps", "buffer_size", "sde_sample_freq"):
            h[key] = int(val)
        elif key in ("learning_rate", "gamma", "tau"):
            h[key] = float(val)
        else:
            h[key] = val
    return h


def train_from_cfg(best: BestRun, seeds: Iterable[int] = SEEDS, total_timesteps: int = TOTAL_TIMESTEPS):
    algo_cls = ALGORITHMS.get(best.algo, SAC)
    hparams = prepare_hparams(best.cfg)

    for seed in seeds:
        run_name = f"{best.name}-retrain-seed{seed}"
        run = wandb.init(
            entity=ENTITY,
            project=f"{PROJECT}-followup",
            name=run_name,
            config={**hparams, "base_run": best.run_id, "algo": best.algo, "seed": seed},
            sync_tensorboard=True,
            monitor_gym=True,
            settings=wandb.Settings(symlink=False),
        )

        env = make_env(seed=seed)
        eval_env = make_env(seed=seed + 10_000)

        model = algo_cls(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            tensorboard_log=os.path.join("runs", run.id),
            **hparams,
        )

        model.learn(total_timesteps=total_timesteps)

        # Eval on a fresh env/seed
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True)
        final_dists, successes = [], 0
        for _ in range(N_EVAL_EPISODES):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
            final_dists.append(info["dist"])
            successes += info["dist"] < eval_env.env.tol

        mean_final_distance = float(np.mean(final_dists))
        success_rate = successes / N_EVAL_EPISODES

        wandb.define_metric("eval/mean_final_distance", summary="min")
        wandb.define_metric("eval/success_rate", summary="max")
        wandb.define_metric("eval/success_count", summary="max")

        wandb.log(
            {
                "eval/mean_reward": mean_reward,
                "eval/mean_final_distance": mean_final_distance,
                "eval/success_rate": success_rate,
                "eval/success_count": successes,
            },
            step=model.num_timesteps,
        )

        run.summary["eval/mean_final_distance"] = mean_final_distance
        run.summary["eval/success_rate"] = success_rate
        run.summary["eval/success_count"] = successes

        model.save(f"models/{run.id}_full")
        env.close()
        eval_env.close()
        wandb.finish()


if __name__ == "__main__":
    best = pick_best_run()
    print(
        f"Picked run {best.name} ({best.run_id}) "
        f"algo={best.algo} success_count={best.success_count} "
        f"mean_final_distance={best.mean_final_distance:.6f}"
    )
    train_from_cfg(best)
