#!/usr/bin/env python

import argparse
from clearml import Task

import os
import time
import numpy as np
import pandas as pd
from typing import Optional
import wandb
from wandb.integration.sb3 import WandbCallback
from sim_wrapper import OT2Env
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# ----------------- W&B / general config -----------------

ENTITY = "242621-breda-university-of-applied-sciences"
PROJECT = "ot2-rl-243552-full"   # you can change this

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

EVAL_EPISODES = 50

ALGORITHMS = {
    "SAC": SAC,
}

def make_env(seed):
    env = OT2Env(render=False, max_episode_steps=1000)
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env

def evaluate_final_distance(model, episodes=EVAL_EPISODES, max_episode_steps=1000, seed=None):
    env = make_env(seed=seed)  # or seed=None to avoid fixed seeding
    final_distances = []

    for i in range(episodes):
        obs, info = env.reset(seed=None if seed is None else seed + i)
        step = 0
        last_distance = None

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            last_distance = info.get("distance")
            step += 1
            if terminated or truncated or step >= max_episode_steps:
                break

        if last_distance is not None:
            final_distances.append(last_distance)

    env.close()
    if not final_distances:
        return np.nan, np.nan
    return float(np.mean(final_distances)), float(np.std(final_distances))


# ----------------- Training function -----------------


def train(train_steps: int, base_seed: int, task: Optional[Task] = None) -> pd.DataFrame:
    """
    Main training loop over ALGORITHMS.
    Logs to W&B, evaluates, and (optionally) uploads models as ClearML artifacts.
    """
    results = []

    for algo_idx, (algo_name, AlgoClass) in enumerate(ALGORITHMS.items(), start=1):
        print(f"=== [{algo_idx}/{len(ALGORITHMS)}] Training {algo_name} ===")

        # --- W&B run for this algorithm ---
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            name=f"full_{algo_name}_{train_steps}",
            config={
                "algorithm": algo_name,
                "train_steps": train_steps,
                "eval_episodes": EVAL_EPISODES,
            },
            reinit=True,  # allow multiple runs in one process
        )

        # Create fresh training env
        env = make_env(seed=base_seed)

        model = AlgoClass(
            "MlpPolicy",
            env,
            device="cuda",
            verbose=1,
            seed=base_seed,
        )

        callbacks = [WandbCallback(log="all", verbose=1)]

        start_time = time.time()
        model.learn(total_timesteps=train_steps, callback=callbacks)
        train_time = time.time() - start_time
        env.close()

        # Save model locally
        model_path = os.path.join(
            MODELS_DIR,
            f"{algo_name}_{train_steps}_steps"
        )
        model.save(model_path)
        print(f"[{algo_name}] Saved model to {model_path}.zip")

        # Optionally upload to ClearML as artifact
        if task is not None:
            try:
                task.upload_artifact(
                    name=f"{algo_name}_model_{train_steps}",
                    artifact_object=model_path + ".zip",
                )
                print(f"[{algo_name}] Uploaded model to ClearML artifacts.")
            except Exception as e:
                print(f"[{algo_name}] Failed to upload artifact to ClearML: {e}")

        # --- Evaluation (reward-based) ---
        print(f"[{algo_name}] Evaluating on {EVAL_EPISODES} episodes (reward)...")
        eval_env_seed = None if base_seed is None else base_seed + 1
        eval_env = make_env(seed=eval_env_seed)
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
        )

        # --- Evaluation (distance-based) ---
        print(f"[{algo_name}] Evaluating final distance to target...")
        mean_dist, std_dist = evaluate_final_distance(
            model,
            episodes=EVAL_EPISODES,
            max_episode_steps=1000,
            seed=base_seed,
        )

        successes = 0

        # If env is Monitor(OT2ReachEnv), unwrap once
        base_env = eval_env.env if hasattr(eval_env, "env") else eval_env
        tol = base_env.success_threshold  # same as your client requirement threshold

        # Success rate loop uses a fresh env to avoid reusing closed one
        success_env_seed = None if base_seed is None else base_seed + 2
        success_env = make_env(seed=success_env_seed)
        for i in range(EVAL_EPISODES):
            reset_seed = None if base_seed is None else success_env_seed + i
            obs, info = success_env.reset(seed=reset_seed)
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = success_env.step(action)
                if terminated or truncated:
                    break

            dist = info["distance"]  # OT2ReachEnv must put this in info
            if dist < tol:
                successes += 1

        success_env.close()
        eval_env.close()

        success_rate = successes / EVAL_EPISODES

        print(
            f"[{algo_name}] mean_reward={mean_reward:.2f} +/- {std_reward:.2f} (std), "
            f"mean_final_dist={mean_dist:.4f} +/- {std_dist:.4f} (std), "
            f"success_rate={success_rate:.2%}, train_time={train_time/60:.1f} min"
        )

        # --- Log to W&B ---
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
                "eval/mean_final_distance": mean_dist,
                "eval/std_final_distance": std_dist,
                "eval/success_rate": success_rate,
                "eval/success_count": successes,
                "train/train_time_sec": train_time,
            },
            step=train_steps,
        )

        # Summary (shows in W&B tables)
        run.summary["eval/mean_reward"] = mean_reward
        run.summary["std_reward"] = std_reward
        run.summary["eval/mean_final_distance"] = mean_dist
        run.summary["eval/std_final_distance"] = std_dist
        run.summary["eval/success_count"] = successes
        run.summary["eval/success_rate"] = success_rate
        run.summary["train_time_sec"] = train_time
        run.summary["model_path"] = model_path + ".zip"

        results.append(
            {
                "run_id": run.id,
                "algorithm": algo_name,
                "train_steps": train_steps,
                "eval_episodes": EVAL_EPISODES,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "mean_final_distance": mean_dist,
                "std_final_distance": std_dist,
                "success_count": successes,
                "success_rate": success_rate,
                "train_time_sec": train_time,
                "model_path": model_path + ".zip",
            }
        )

        run.finish()

    # ---------- Aggregate & results ----------

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(
        ["mean_final_distance", "mean_reward"],
        ascending=[True, False],  # smaller distance is better; higher reward is better
    )

    print("=== Benchmark Results (sorted by distance, then reward) ===")
    print(df_sorted.to_string(index=False))

    wandb.finish()
    return df_sorted


# ----------------- ClearML wrapper / entry point -----------------


def main():
    # ---------- Command-line arguments (so ClearML can edit them) ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_steps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_steps = args.train_steps
    base_seed = args.seed

    # ---------- ClearML task init ----------
    task = Task.init(
        project_name="OT2-RL/243552-Maciej",   # you can rename the project if you want
        task_name=f"SAC_train_{train_steps}",
    )

    # Use the course docker image & default queue
    task.set_base_docker("deanis/2023y2b-rl:latest")
    task.execute_remotely(queue_name="default")  # sends this job to the ADSAI server

    # ---------- Code below runs on the remote worker ----------
    df_sorted = train(train_steps=train_steps, base_seed=base_seed, task=task)
    print("Training finished. Best results:")
    print(df_sorted.head())


if __name__ == "__main__":
    main()