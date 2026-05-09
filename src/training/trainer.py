import os
import glob
from datetime import datetime
import gymnasium as gym
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.utils.config import load_config
from src.training.callbacks import (
    DetailedLoggingCallback,
    HERFineTuneWarmupCallback,
    SuccessRateEvalCallback,
    CombinedSuccessCallback,
    progress_logger
)
import src.chess_env  # Registers ChessFetchTask-v0

class SACTrainer:
    """
    Orchestrates the SAC training process for RoboChess.
    
    This class handles the creation of vectorized training environments,
    stratified evaluation environments, callback management, and model
    persistence (checkpoints and final saves).
    """

    def __init__(
        self,
        num_envs=None,
        fresh_start=False,
        debug=False,
        fixed_drift=False,
        scenario=None,
        reset_num_timesteps=True,
        reset_entropy=True,
    ):
        """
        Initializes the trainer.

        Args:
            num_envs (int): Number of parallel environments for training.
            fresh_start (bool): If True, starts from base weights even if checkpoints exist.
            debug (bool): If True, enables debug logging in all created environments.
            fixed_drift (bool): If True, bypasses the drift curriculum.
            scenario (str | None): If set, force all training episodes to one scenario.
            reset_num_timesteps (bool): If False, continue the loaded model's
                timestep counter for fine-tuning.
            reset_entropy (bool): If True, reset SAC entropy for extra exploration.
        """
        if scenario not in {None, "transit", "descend", "ascend"}:
            raise ValueError(f"Unknown training scenario: {scenario}")

        self.train_cfg = load_config("training")
        self.env_cfg = load_config("env")
        
        self.num_envs = num_envs or self.train_cfg.get("num_envs", 12)
        self.fresh_start = fresh_start
        self.debug = debug
        self.fixed_drift = fixed_drift
        self.scenario = scenario
        self.reset_num_timesteps = reset_num_timesteps
        self.reset_entropy = reset_entropy
        
        self.total_timesteps = self.train_cfg.get("total_timesteps", 1_000_000)
        self.base_model_path = self.train_cfg.get("base_model")
        self.target_curriculum_total = self.train_cfg.get("target_curriculum_total", 500_000)
        
    def find_best_checkpoint(self, checkpoint_dir="./checkpoints"):
        """
        Scans the checkpoints directory for the latest best_model_transit.zip.
        Used to automatically resume training from the most recent successful run.
        """
        transit_checkpoints = sorted(
            glob.glob(f"{checkpoint_dir}/pure_movement_v6_*/best_model_transit.zip"),
            key=os.path.getmtime
        )
        if transit_checkpoints:
            best = transit_checkpoints[-1]
            progress_logger.info("Auto-selected best transit checkpoint: %s", best)
            return best
        return self.base_model_path

    def make_env(self, curriculum_steps):
        """
        Factory function for creating training environments.
        Wraps each env with a Monitor for metric tracking.
        """
        def _init():
            return Monitor(gym.make("ChessFetchTask-v0", 
                                    drift_curriculum_steps=curriculum_steps,
                                    force_scenario=self.scenario,
                                    debug=self.debug,
                                    fixed_drift=self.fixed_drift))
        return _init

    def make_eval_env(self, scenario_name):
        """
        Factory function for creating evaluation environments.
        Locks the environment to a specific scenario and applies a fixed drift limit.
        """
        def _init():
            return Monitor(gym.make("ChessFetchTask-v0", 
                                    force_scenario=scenario_name,
                                    force_drift_limit=self.env_cfg.get("eval_drift_limit", 0.005),
                                    debug=self.debug))
        return _init

    def train(self, model_path=None):
        """
        Executes the full training loop.
        
        Steps:
        1. Resolves the starting model (Fresh vs Checkpoint).
        2. Vectorizes 12 training environments.
        3. Sets up 3 separate evaluation environments (Transit, Descend, Ascend).
        4. Configures a suite of callbacks for logging, saving, and combined success monitoring.
        5. Resets the entropy coefficient to allow exploration on the new reward surface.
        6. Calls model.learn().
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_suffix = f"_{self.scenario}" if self.scenario else ""
        run_name = f"pure_movement_v6{scenario_suffix}_{timestamp}"
        
        # Calculate per-worker curriculum steps to ensure target_total is met
        drift_curriculum_steps = self.target_curriculum_total // self.num_envs

        if model_path is None:
            model_path = self.base_model_path
            if not self.fresh_start:
                model_path = self.find_best_checkpoint()

        scenario_label = self.scenario or "mixed"
        print(f"--- Starting V6 Pure Movement Training: {run_name} {model_path} | scenario={scenario_label} ---")
        progress_logger.info(
            "Starting V6 run: %s with %d envs from %s (scenario=%s, reset_num_timesteps=%s, reset_entropy=%s).",
            run_name, self.num_envs, model_path, scenario_label, self.reset_num_timesteps, self.reset_entropy
        )
        progress_logger.info("Drift curriculum completes in %d steps per worker (%d total).", 
                             drift_curriculum_steps, self.target_curriculum_total)

        # Environment Vectorization
        train_env = SubprocVecEnv([self.make_env(drift_curriculum_steps) for _ in range(self.num_envs)])
        eval_transit = DummyVecEnv([self.make_eval_env("transit")])
        eval_descend = DummyVecEnv([self.make_eval_env("descend")])
        eval_ascend = DummyVecEnv([self.make_eval_env("ascend")])

        success_save_path = f"./checkpoints/{run_name}/"
        eval_freq_total = self.train_cfg.get("eval_freq", 10000)
        eval_freq = max(eval_freq_total // self.num_envs, 1) # Convert total steps to per-process frequency
        n_eval_episodes = self.train_cfg.get("n_eval_episodes", 50)

        # Callback Orchestration
        cb_transit = SuccessRateEvalCallback(eval_transit, success_save_path=success_save_path, name="transit", eval_freq=eval_freq, n_eval_episodes=n_eval_episodes)
        cb_descend = SuccessRateEvalCallback(eval_descend, success_save_path=success_save_path, name="descend", eval_freq=eval_freq, n_eval_episodes=n_eval_episodes)
        cb_ascend  = SuccessRateEvalCallback(eval_ascend, success_save_path=success_save_path, name="ascend", eval_freq=eval_freq, n_eval_episodes=n_eval_episodes)

        callback_items = [
            DetailedLoggingCallback(),
        ]
        if not self.reset_num_timesteps:
            callback_items.append(HERFineTuneWarmupCallback())

        callback_items.extend([
            cb_transit,
            cb_descend,
            cb_ascend,
            CombinedSuccessCallback([cb_transit, cb_descend, cb_ascend], success_save_path),
        ])
        callbacks = CallbackList(callback_items)

        # Model Loading and Hyperparameter Injection
        model = SAC.load(
            model_path, 
            env=train_env, 
            learning_rate=float(self.train_cfg.get("learning_rate", 5e-5)), 
            batch_size=self.train_cfg.get("batch_size", 512), 
            target_entropy=float(self.train_cfg.get("target_entropy", -4.0)),
            buffer_size=300000, # Reduce buffer size to prevent OOM
            verbose=1
        )
        model.tensorboard_log = f"./logs/{run_name}/tensorboard/"
        model.learning_starts = self.train_cfg.get("learning_starts", 10_000)
        
        if self.fresh_start:
            model.replay_buffer.reset()
        
        if self.reset_entropy:
            # The pretrained weights have very low entropy (~0.003), which can
            # hinder exploration on a changed reward surface.
            initial_ent_coef = float(self.train_cfg.get("initial_ent_coef", 0.1))
            ent_coef_lr = float(self.train_cfg.get("ent_coef_lr", 1e-3))

            model.log_ent_coef = th.log(th.ones(1) * initial_ent_coef).to(model.device)
            model.log_ent_coef = th.nn.Parameter(model.log_ent_coef, requires_grad=True)
            model.ent_coef_optimizer = th.optim.Adam([model.log_ent_coef], lr=ent_coef_lr)
        else:
            print("Keeping loaded SAC entropy coefficient for conservative fine-tuning.")
            progress_logger.info("Keeping loaded SAC entropy coefficient for conservative fine-tuning.")
        
        print(f"Beginning {self.total_timesteps:,} steps...")
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=self.reset_num_timesteps,
            progress_bar=True,
        )

        final_path = f"chess_fetch_pure_v6{scenario_suffix}_{timestamp}.zip"
        model.save(final_path)
        print(f"Training complete! Saved to {final_path}")
        print(f"Eval checkpoints saved under {success_save_path}")
        print("For one-model full-sequence eval, prefer best_model_combined.zip from that folder when it exists.")
        if self.scenario:
            print(
                f"Note: this is a {self.scenario}-specialized model. "
                f"For full pick-and-place eval, pass it as --{self.scenario}-model "
                "with a stable combined model in --model."
            )
        
        train_env.close()
        eval_transit.close()
        eval_descend.close()
        eval_ascend.close()
