#!/usr/bin/env python

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from clearml import Task
import numpy as np
import pandas as pd
from typing import Optional
import wandb
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper import OT2Env
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ----------------- W&B / general config -----------------

ENTITY = "242621-breda-university-of-applied-sciences"
PROJECT = "ot2-rl-243552-full"   # you can change this

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
TENSORBOARD_DIR = "tb"
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

EVAL_EPISODES = 50
CHECKPOINT_FREQ = 200_000


def ensure_tensorboard():
    if importlib.util.find_spec("tensorboard") is not None:
        return
    print("TensorBoard not found; installing for W&B rollout metrics...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "TensorBoard is required for W&B rollout metrics. "
            "Install it in the ClearML worker environment."
        ) from exc


def make_env(seed=None):
    env = OT2Env(render_mode=None, max_episode_steps=1000)
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env

def evaluate_final_distance(
    model,
    episodes=EVAL_EPISODES,
    max_episode_steps=1000,
    seed=None,
    obs_rms=None,
):
    env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv, seed=seed)
    env = VecNormalize(env, training=False, norm_obs=True, norm_reward=False, clip_obs=10)
    if obs_rms is not None:
        env.obs_rms = obs_rms
    final_distances = []

    for _ in range(episodes):
        obs = env.reset()
        step = 0
        last_info = None

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            last_info = infos[0]
            step += 1
            if bool(dones[0]) or step >= max_episode_steps:
                break

        if last_info is not None:
            last_distance = last_info.get("distance")
            if last_distance is not None:
                final_distances.append(last_distance)

    env.close()
    if not final_distances:
        return np.nan, np.nan
    return float(np.mean(final_distances)), float(np.std(final_distances))

def log_artifacts(model_path, artifact_name, task=None):
    if wandb.run is not None:
        try:
            artifact = wandb.Artifact(name=artifact_name, type="model")
            artifact.add_file(model_path + ".zip")
            wandb.log_artifact(artifact)
            print(f"[{artifact_name}] Uploaded model to W&B artifacts.")
        except Exception as e:
            print(f"[{artifact_name}] Failed to upload artifact to W&B: {e}")

    if task is not None:
        try:
            task.upload_artifact(
                name=artifact_name,
                artifact_object=model_path + ".zip",
            )
            print(f"[{artifact_name}] Uploaded model to ClearML artifacts.")
        except Exception as e:
            print(f"[{artifact_name}] Failed to upload artifact to ClearML: {e}")

class CheckpointArtifactCallback(BaseCallback):
    def __init__(self, save_freq, save_dir, algo_name, task=None, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.algo_name = algo_name
        self.task = task
        self.last_saved_step = 0

    def _save_checkpoint(self, step):
        model_path = os.path.join(self.save_dir, f"{self.algo_name}_{step}_steps")
        self.model.save(model_path)
        if self.verbose > 0:
            print(f"[{self.algo_name}] Saved checkpoint to {model_path}.zip")
        log_artifacts(model_path, f"{self.algo_name}_checkpoint_{step}", self.task)

    def _on_step(self):
        if self.save_freq <= 0:
            return True
        if self.num_timesteps % self.save_freq != 0:
            return True
        self.last_saved_step = self.num_timesteps
        self._save_checkpoint(self.last_saved_step)
        return True


# ----------------- Training function -----------------


def train(train_steps: int, base_seed: int, task: Optional[Task] = None) -> pd.DataFrame:
    """
    Main training loop over the SAC algorithm.
    Logs to W&B, evaluates, and (optionally) uploads models as ClearML artifacts.
    """
    results = []

    algo_name = "SAC"
    AlgoClass = SAC
    print(f"=== Training {algo_name} ===")

    # --- W&B run for this algorithm ---
    ensure_tensorboard()
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
    else:
        wandb.login()
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"full_{algo_name}_{train_steps}",
        config={
            "algorithm": algo_name,
            "train_steps": train_steps,
            "eval_episodes": EVAL_EPISODES,
        },
        sync_tensorboard=True,  # capture rollout metrics from SB3 logger
        reinit=True,  # allow multiple runs in one process
    )

    # Create fresh training env
    train_env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv, seed=base_seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10)

    model = AlgoClass(
        "MlpPolicy",
        train_env,
        device="cuda",
        verbose=1,
        seed=base_seed,
        tensorboard_log=TENSORBOARD_DIR,
    )

    checkpoint_callback = CheckpointArtifactCallback(
        save_freq=CHECKPOINT_FREQ,
        save_dir=MODELS_DIR,
        algo_name=algo_name,
        task=task,
        verbose=1,
    )
    callbacks = [WandbCallback(log="all", verbose=1), checkpoint_callback]

    start_time = time.time()
    model.learn(
        total_timesteps=train_steps,
        callback=callbacks,
        log_interval=1,
        tb_log_name=f"{algo_name}_{run.id}",
    )
    train_time = time.time() - start_time
    obs_rms = train_env.obs_rms
    train_env.close()

    # Save final model (if not already saved on the last checkpoint)
    final_step = model.num_timesteps
    model_path = os.path.join(
        MODELS_DIR,
        f"{algo_name}_{final_step}_steps"
    )
    if final_step != checkpoint_callback.last_saved_step:
        model.save(model_path)
        print(f"[{algo_name}] Saved model to {model_path}.zip")
        log_artifacts(model_path, f"{algo_name}_final_{final_step}", task)

    # --- Evaluation (reward-based) ---
    print(f"[{algo_name}] Evaluating on {EVAL_EPISODES} episodes (reward)...")
    eval_env_seed = None if base_seed is None else base_seed + 1
    eval_env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv, seed=eval_env_seed)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10)
    if obs_rms is not None:
        eval_env.obs_rms = obs_rms
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
        obs_rms=obs_rms,
    )

    successes = 0

    success_threshold = getattr(
        eval_env.unwrapped, "success_threshold", getattr(eval_env, "success_threshold", None)
    )

    # Success rate loop uses a fresh env to avoid reusing closed one
    success_env_seed = None if base_seed is None else base_seed + 2
    success_env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv, seed=success_env_seed)
    success_env = VecNormalize(success_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10)
    if obs_rms is not None:
        success_env.obs_rms = obs_rms
    for _ in range(EVAL_EPISODES):
        obs = success_env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = success_env.step(action)
            if bool(dones[0]):
                break

        dist = infos[0].get("distance", np.nan)  # OT2ReachEnv must put this in info
        if success_threshold is not None and dist < success_threshold:
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
    wandb.define_metric("rollout/ep_len_mean", summary="max")
    wandb.define_metric("rollout/ep_rew_mean", summary="max")

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
    wandb_params = {"wandb_api_key": "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"}
    task.connect(wandb_params, name="wandb")
    if wandb_params["wandb_api_key"]:
        os.environ["WANDB_API_KEY"] = wandb_params["wandb_api_key"]

    # Use the course docker image & default queue
    task.set_base_docker("deanis/2023y2b-rl:latest")
    task.execute_remotely(queue_name="default")  # sends this job to the ADSAI server

    # ---------- Code below runs on the remote worker ----------
    df_sorted = train(train_steps=train_steps, base_seed=base_seed, task=task)
    print("Training finished. Best results:")
    print(df_sorted.head())


if __name__ == "__main__":
    main()
