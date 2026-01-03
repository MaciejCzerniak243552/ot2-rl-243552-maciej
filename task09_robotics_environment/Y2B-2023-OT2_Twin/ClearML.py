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

EVAL_EPISODES = 100

def evaluate_episode_stats(
    model,
    episodes=EVAL_EPISODES,
    max_episode_steps=1000,
    seed=None,
    success_threshold=0.001,
):
    """
    Roll out episodes and collect richer stats: final/min distance, steps-to-done,
    termination/truncation rates, and threshold hits at 10mm/5mm/1mm.
    """
    env = make_env(seed=seed)
    finals, mins, steps_list = [], [], []
    terminations, truncations = 0, 0
    hits_10mm, hits_5mm, hits_1mm = 0, 0, 0

    for i in range(episodes):
        obs, info = env.reset(seed=None if seed is None else seed + i)
        min_d = info.get("distance", float("inf"))
        final_d = None
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            d = info.get("distance")
            if d is not None:
                final_d = d
                if d < min_d:
                    min_d = d
            steps += 1
            if terminated or truncated or steps >= max_episode_steps:
                terminations += int(terminated)
                truncations += int(truncated or (steps >= max_episode_steps))
                break

        if final_d is None:
            final_d = min_d
        finals.append(final_d)
        mins.append(min_d)
        steps_list.append(steps)

        if final_d < 0.01:
            hits_10mm += 1
        if final_d < 0.005:
            hits_5mm += 1
        if final_d < success_threshold:
            hits_1mm += 1

    env.close()

    stats = {
        "mean_final": float(np.mean(finals)) if finals else np.nan,
        "std_final": float(np.std(finals)) if finals else np.nan,
        "mean_min": float(np.mean(mins)) if mins else np.nan,
        "std_min": float(np.std(mins)) if mins else np.nan,
        "mean_steps": float(np.mean(steps_list)) if steps_list else np.nan,
        "termination_rate": terminations / episodes if episodes else 0.0,
        "truncation_rate": truncations / episodes if episodes else 0.0,
        "success_rate_10mm": hits_10mm / episodes if episodes else 0.0,
        "success_rate_5mm": hits_5mm / episodes if episodes else 0.0,
        "success_rate_1mm": hits_1mm / episodes if episodes else 0.0,
    }
    return stats

def make_env(seed: Optional[int]):
    """
    Create a monitored OT2Env.
    Seeding is done via env.reset(seed=seed), which is Gymnasium-compliant.
    """
    env = OT2Env(render=False, max_episode_steps=1000)
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env

def evaluate_final_distance(model, episodes=EVAL_EPISODES, max_episode_steps=1000, seed=None):
    """
    Roll out a number of episodes and record the final distance to the goal.
    """

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


def train(
    train_steps: int,
    base_seed: int,
    task: Optional[Task] = None,
    args: Optional[argparse.Namespace] = None,
) -> pd.DataFrame:
    """
    SAC training.
    Logs to W&B, evaluates, and (optionally) uploads models as ClearML artifacts.
    """
    os.environ['WANDB_API_KEY'] = 'b1e375dd07d0792bb5601ffbb8b45cf2f84f5d20'

    algo_name = "SAC"
    results = []

    print(f"=== Training {algo_name} ===")

    # --- W&B run for this algorithm ---
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"full_{algo_name}_{train_steps}",
        config={
            "algorithm": algo_name,
            "train_steps": train_steps,
            "eval_episodes": EVAL_EPISODES,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "tau": args.tau,
            "ent_coef": args.ent_coef,
            "learning_starts": args.learning_starts,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "target_entropy": args.target_entropy,
        } if args is not None else None,
        reinit=True,  # allow multiple runs in one process
    )

    # Create fresh training env
    env = make_env(seed=base_seed)

    # handle "auto" vs numeric target_entropy
    if args is not None:
        if isinstance(args.target_entropy, str) and args.target_entropy.startswith("auto"):
            target_entropy = args.target_entropy
        else:
            target_entropy = float(args.target_entropy)
        learning_rate = args.learning_rate
        gamma = args.gamma
        batch_size = args.batch_size
        tau = args.tau
        ent_coef = args.ent_coef
        learning_starts = args.learning_starts
        train_freq = args.train_freq
        gradient_steps = args.gradient_steps
    else:
        # Fallback to SAC defaults if args not provided
        target_entropy = "auto_0.5"
        learning_rate = 3e-4
        gamma = 0.99
        batch_size = 256
        tau = 0.005
        ent_coef = "auto"
        learning_starts = 1000
        train_freq = 1
        gradient_steps = 1

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=base_seed,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        tau=tau,
        ent_coef=ent_coef,
        learning_starts=learning_starts,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_entropy=target_entropy,
    )

    callbacks = [WandbCallback(log="all", verbose=1)]

    # -------- Incremental Training + Periodic Saving --------
    save_every = 500_000     # how many steps per chunk
    num_chunks = train_steps // save_every
    remainder = train_steps % save_every

    os.makedirs(f"models/{run.id}", exist_ok=True)

    start_time = time.time()

    for i in range(num_chunks):
        print(f"[{algo_name}] Training chunk {i+1}/{num_chunks} ...")

        model.learn(
            total_timesteps=save_every,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False,         # <-- important
            tb_log_name=f"runs/{run.id}",      # <-- keeps TB logs consistent
        )

        # Save checkpoint
        checkpoint_path = f"models/{run.id}/{save_every*(i+1)}"
        model.save(checkpoint_path)
        print(f"[{algo_name}] Saved checkpoint: {checkpoint_path}.zip")


        # Optionally upload to ClearML as artifact
        if task is not None:
            try:
                task.upload_artifact(
                    name=f"{algo_name}_checkpoint_{save_every*(i+1)}",
                    artifact_object=checkpoint_path + ".zip",
                )
                print(f"[{algo_name}] Uploaded checkpoint to ClearML: {checkpoint_path}.zip")
            except Exception as e:
                print(f"[{algo_name}] Failed to upload checkpoint artifact: {e}")

    # Train leftover steps (or all steps if less than save_every)
    if remainder > 0 or num_chunks == 0:
        leftover = remainder if remainder > 0 else train_steps
        print(f"[{algo_name}] Training final chunk of {leftover} steps ...")
        model.learn(
            total_timesteps=leftover,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=f"runs/{run.id}",
        )
        checkpoint_path = f"models/{run.id}/{save_every*num_chunks + leftover}"
        model.save(checkpoint_path)
        print(f"[{algo_name}] Saved checkpoint: {checkpoint_path}.zip")
        if task is not None:
            try:
                task.upload_artifact(
                    name=f"{algo_name}_checkpoint_{save_every*num_chunks + leftover}",
                    artifact_object=checkpoint_path + ".zip",
                )
                print(f"[{algo_name}] Uploaded checkpoint to ClearML: {checkpoint_path}.zip")
            except Exception as e:
                print(f"[{algo_name}] Failed to upload checkpoint artifact: {e}")

    train_time = time.time() - start_time
    env.close()

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
    print(f"[{algo_name}] Evaluating distance and stability metrics...")
    base_env = eval_env.env if hasattr(eval_env, "env") else eval_env
    stats = evaluate_episode_stats(
        model,
        episodes=EVAL_EPISODES,
        max_episode_steps=1000,
        seed=base_seed,
        success_threshold=base_env.success_threshold,
    )

    eval_env.close()

    print(
        f"[{algo_name}] mean_reward={mean_reward:.2f} +/- {std_reward:.2f} (std), "
        f"mean_final_dist={stats['mean_final']:.4f} +/- {stats['std_final']:.4f} (std), "
        f"mean_min_dist={stats['mean_min']:.4f}, "
        f"success_rate_1mm={stats['success_rate_1mm']:.2%}, "
        f"success_rate_5mm={stats['success_rate_5mm']:.2%}, "
        f"success_rate_10mm={stats['success_rate_10mm']:.2%}, "
        f"train_time={train_time/60:.1f} min"
    )

    # --- Log to W&B ---
    wandb.define_metric("eval/mean_final_distance", summary="min")
    wandb.define_metric("eval/std_final_distance", summary="min")
    wandb.define_metric("eval/mean_min_distance", summary="min")
    wandb.define_metric("eval/std_min_distance", summary="min")
    wandb.define_metric("eval/success_rate", summary="max")
    wandb.define_metric("eval/success_rate_5mm", summary="max")
    wandb.define_metric("eval/success_rate_10mm", summary="max")
    wandb.define_metric("eval/success_count", summary="max")
    wandb.define_metric("eval/mean_episode_len", summary="min")
    wandb.define_metric("eval/truncation_rate", summary="min")
    wandb.define_metric("eval/mean_reward", summary="max")
    wandb.define_metric("eval/std_reward", summary="min")

    wandb.log(
        {
            "eval/mean_reward": mean_reward,
            "eval/std_reward": std_reward,
            "eval/mean_final_distance": stats["mean_final"],
            "eval/std_final_distance": stats["std_final"],
            "eval/mean_min_distance": stats["mean_min"],
            "eval/std_min_distance": stats["std_min"],
            "eval/success_rate": stats["success_rate_1mm"],
            "eval/success_rate_5mm": stats["success_rate_5mm"],
            "eval/success_rate_10mm": stats["success_rate_10mm"],
            "eval/success_count": int(stats["success_rate_1mm"] * EVAL_EPISODES),
            "eval/mean_episode_len": stats["mean_steps"],
            "eval/truncation_rate": stats["truncation_rate"],
            "train/train_time_sec": train_time,
        },
        step=train_steps,
    )

    # Summary (shows in W&B tables)
    run.summary["eval/mean_reward"] = mean_reward
    run.summary["std_reward"] = std_reward
    run.summary["eval/mean_final_distance"] = stats["mean_final"]
    run.summary["eval/std_final_distance"] = stats["std_final"]
    run.summary["eval/mean_min_distance"] = stats["mean_min"]
    run.summary["eval/std_min_distance"] = stats["std_min"]
    run.summary["eval/success_count"] = int(stats["success_rate_1mm"] * EVAL_EPISODES)
    run.summary["eval/success_rate"] = stats["success_rate_1mm"]
    run.summary["eval/success_rate_5mm"] = stats["success_rate_5mm"]
    run.summary["eval/success_rate_10mm"] = stats["success_rate_10mm"]
    run.summary["eval/mean_episode_len"] = stats["mean_steps"]
    run.summary["eval/truncation_rate"] = stats["truncation_rate"]
    run.summary["train_time_sec"] = train_time
    run.summary["last_checkpoint"] = f"models/{run.id}/{save_every*num_chunks + (remainder if remainder > 0 else (train_steps if num_chunks == 0 else 0))}.zip"
    run.summary["checkpoints"] = [
        f"models/{run.id}/{save_every*(i+1)}.zip" for i in range(num_chunks)
    ] + (
        [f"models/{run.id}/{save_every*num_chunks + (remainder if remainder > 0 else train_steps)}.zip"]
        if (remainder > 0 or num_chunks == 0)
        else []
    )

    results.append(
        {
            "run_id": run.id,
            "algorithm": algo_name,
            "train_steps": train_steps,
            "eval_episodes": EVAL_EPISODES,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_final_distance": stats["mean_final"],
            "std_final_distance": stats["std_final"],
            "mean_min_distance": stats["mean_min"],
            "std_min_distance": stats["std_min"],
            "success_rate": stats["success_rate_1mm"],
            "success_rate_5mm": stats["success_rate_5mm"],
            "success_rate_10mm": stats["success_rate_10mm"],
            "train_time_sec": train_time,
            "checkpoints": [
                f"models/{run.id}/{save_every*(i+1)}.zip"
                for i in range(num_chunks)
            ]
        }
    )

    run.finish()
    wandb.finish()

    # ---------- Aggregate & results ----------

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(
        ["mean_final_distance", "mean_reward"],
        ascending=[True, False],  # smaller distance is better; higher reward is better
    )

    print("=== Benchmark Results (sorted by distance, then reward) ===")
    print(df_sorted.to_string(index=False))

    return df_sorted


# ----------------- ClearML wrapper / entry point -----------------


def main():
    # ---------- Command-line arguments (so ClearML can edit them) ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_steps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--ent_coef", type=str, default="auto")   # can also be "auto_1.0"
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--train_freq", type=int, default=1)       # SB3 allows (freq, unit) too, but int works
    parser.add_argument("--gradient_steps", type=int, default=1)

    # target_entropy can be "auto" or a number, so parse as string then convert later
    parser.add_argument("--target_entropy", type=str, default="auto_0.5")
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

    task.set_script(
        repository="https://github.com/MaciejCzerniak243552/ot2-rl-243552-maciej.git",
        branch="main",
        working_dir="task09_robotics_environment/Y2B-2023-OT2_Twin",
        entry_point="ClearML.py",
    )

    task.execute_remotely(queue_name="default")  # sends this job to the ADSAI server

    # ---------- Code below runs on the remote worker ----------
    df_sorted = train(train_steps=train_steps, base_seed=base_seed, task=task)
    print("Training finished. Best results:")
    print(df_sorted.head())


if __name__ == "__main__":
    main()
