import argparse
import logging
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch as th

import chess_env
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

TOTAL_TIMESTEPS = 500_000
DEFAULT_RESUME_MODEL = "chess_fetch_dense_v3_ready.zip"
FALLBACK_MODEL = "models/sac-FetchPickAndPlace-v4.zip"

os.makedirs("logs", exist_ok=True)
progress_logger = logging.getLogger("training_progress")
progress_logger.setLevel(logging.INFO)
if not progress_logger.handlers:
    handler = logging.FileHandler("logs/training_progress_detailed.log")
    handler.setFormatter(logging.Formatter("%(asctime)s - [PROGRESS] - %(message)s"))
    progress_logger.addHandler(handler)


def resolve_default_model_path():
    if os.path.exists(DEFAULT_RESUME_MODEL):
        return DEFAULT_RESUME_MODEL
    return FALLBACK_MODEL


class DetailedLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.successes.append(info.get("is_success", 0.0))

        if self.num_timesteps % 2000 == 0 and self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_len = np.mean(self.episode_lengths[-100:])
            mean_success = np.mean(self.successes[-100:])
            progress_logger.info(
                "Step %d | Avg Reward: %.2f | Avg Length: %.1f steps | Success Rate: %.1f%%",
                self.num_timesteps,
                mean_reward,
                mean_len,
                mean_success * 100.0,
            )
        return True


class SuccessRateEvalCallback(EvalCallback):
    def __init__(self, *args, success_save_path="./checkpoints/", name="eval", **kwargs):
        super().__init__(*args, **kwargs)
        self.success_save_path = success_save_path
        self._best_success_rate = -1.0
        self.name = name

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.evaluations_successes:
            latest = float(np.mean(self.evaluations_successes[-1]))
            if latest > self._best_success_rate:
                self._best_success_rate = latest
                os.makedirs(self.success_save_path, exist_ok=True)
                save_path = os.path.join(self.success_save_path, f"best_model_{self.name}")
                self.model.save(save_path)
                if self.verbose >= 1:
                    print(
                        f"[DenseSuccessCallback - {self.name}] New best: {latest*100:.1f}% -> {save_path}.zip"
                    )
                progress_logger.info(
                    "*** NEW BEST EVAL (%s) *** | Success: %.1f%% -> saved to %s.zip",
                    self.name.upper(),
                    latest * 100.0,
                    save_path,
                )
        return result


def make_env(num_envs=1):
    def _init():
        per_env_curriculum_steps = max(TOTAL_TIMESTEPS // max(num_envs, 1), 1)
        env = gym.make("ChessFetchDense-v0", total_curriculum_steps=per_env_curriculum_steps, num_envs=num_envs)
        return Monitor(env)

    return _init


def make_eval_env(scenario_name, num_envs=1):
    def _init():
        per_env_curriculum_steps = max(TOTAL_TIMESTEPS // max(num_envs, 1), 1)
        env = gym.make(
            "ChessFetchDense-v0",
            force_scenario=scenario_name,
            curriculum_progress_override=1.0,
            total_curriculum_steps=per_env_curriculum_steps,
            num_envs=num_envs,
        )
        return Monitor(env)

    return _init


def fine_tune_dense(base_model_path=None, num_envs=4):
    user_supplied_model = base_model_path is not None
    base_model_path = base_model_path or resolve_default_model_path()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dense_refinement_{timestamp}"

    print(f"--- Starting Dense Refinement: {run_name} ---")
    progress_logger.info(
        "Starting training run: %s with %d parallel environments from %s.",
        run_name,
        num_envs,
        base_model_path,
    )

    env_fns = [make_env(num_envs=num_envs) for _ in range(num_envs)]
    try:
        train_env = SubprocVecEnv(env_fns)
        progress_logger.info("Using SubprocVecEnv with %d environments.", num_envs)
    except Exception as exc:
        progress_logger.warning(
            "SubprocVecEnv unavailable in this environment (%s). Falling back to DummyVecEnv.",
            exc,
        )
        print(f"SubprocVecEnv unavailable ({exc}). Falling back to DummyVecEnv.")
        train_env = DummyVecEnv(env_fns)
    eval_transit = DummyVecEnv([make_eval_env("transit", num_envs=num_envs)])
    eval_descend = DummyVecEnv([make_eval_env("descend", num_envs=num_envs)])
    eval_ascend = DummyVecEnv([make_eval_env("ascend", num_envs=num_envs)])

    success_save_path = f"./checkpoints/{run_name}/"

    cb_transit = SuccessRateEvalCallback(
        eval_transit,
        success_save_path=success_save_path,
        log_path=f"./logs/{run_name}/transit/",
        name="transit",
        eval_freq=max(10_000 // num_envs, 1),
        n_eval_episodes=50,
        deterministic=True,
        render=False,
        verbose=1,
    )
    cb_descend = SuccessRateEvalCallback(
        eval_descend,
        success_save_path=success_save_path,
        log_path=f"./logs/{run_name}/descend/",
        name="descend",
        eval_freq=max(10_000 // num_envs, 1),
        n_eval_episodes=50,
        deterministic=True,
        render=False,
        verbose=1,
    )
    cb_ascend = SuccessRateEvalCallback(
        eval_ascend,
        success_save_path=success_save_path,
        log_path=f"./logs/{run_name}/ascend/",
        name="ascend",
        eval_freq=max(10_000 // num_envs, 1),
        n_eval_episodes=50,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = CallbackList(
        [
            DetailedLoggingCallback(),
            cb_transit,
            cb_descend,
            cb_ascend,
        ]
    )

    print(f"Loading weights from {base_model_path}...")
    progress_logger.info("Loading base weights from %s", base_model_path)

    try:
        model = SAC.load(
            base_model_path,
            env=train_env,
            custom_objects={"ent_coef": "auto"},
            learning_rate=3e-4 if "FetchPickAndPlace" in base_model_path else 1e-5,
            batch_size=512,
            verbose=1,
        )
    except Exception as exc:
        if user_supplied_model or base_model_path == FALLBACK_MODEL:
            raise

        progress_logger.warning(
            "Failed to load preferred resume checkpoint %s (%s). Falling back to %s.",
            base_model_path,
            exc,
            FALLBACK_MODEL,
        )
        print(
            f"Preferred resume checkpoint failed to load ({base_model_path}). Falling back to {FALLBACK_MODEL}."
        )
        base_model_path = FALLBACK_MODEL
        model = SAC.load(
            base_model_path,
            env=train_env,
            custom_objects={"ent_coef": "auto"},
            learning_rate=3e-4 if "FetchPickAndPlace" in base_model_path else 1e-5,
            batch_size=512,
            verbose=1,
        )
    model.tensorboard_log = f"./logs/{run_name}/tensorboard/"

    target_ent_coef = 0.001
    model.target_entropy = -3.0
    if model.log_ent_coef is not None:
        with th.no_grad():
            model.log_ent_coef.data.fill_(float(np.log(target_ent_coef)))
        model.ent_coef_tensor = th.tensor(float(target_ent_coef), device=model.device)
        model.ent_coef_optimizer = None
        model.log_ent_coef = None
        model.ent_coef = target_ent_coef
        model.target_entropy = -3.0
    else:
        model.ent_coef = target_ent_coef
        model.ent_coef_tensor = th.tensor(float(target_ent_coef), device=model.device)

    print("Clearing old replay buffer...")
    model.replay_buffer.reset()
    model.learning_starts = 2000

    print(f"Beginning {TOTAL_TIMESTEPS:,} steps of Dense Fine-Tuning...")
    progress_logger.info("Beginning %d steps of training...", TOTAL_TIMESTEPS)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=True,
        progress_bar=True,
    )

    final_path = f"chess_fetch_dense_{timestamp}.zip"
    model.save(final_path)
    print(f"Dense refinement complete! Model saved to {final_path}")
    progress_logger.info("Training complete! Final model saved to %s", final_path)

    train_env.close()
    eval_transit.close()
    eval_descend.close()
    eval_ascend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--envs", type=int, default=4)
    args = parser.parse_args()
    fine_tune_dense(base_model_path=args.model, num_envs=args.envs)
