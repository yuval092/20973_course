import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from src.utils.logger import setup_logger
from src.utils.config import load_config

# Setup the progress logger
progress_logger = setup_logger("training_progress", "logs/training_progress_detailed.log")

class DetailedLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        train_cfg = load_config("training")
        self.log_freq = train_cfg.get("log_freq", 2000)
        self.moving_avg_window = train_cfg.get("moving_avg_window", 100)
        self.scenario_log_window = train_cfg.get("scenario_log_window", 30)
        self.scenario_min_episodes = train_cfg.get("scenario_min_episodes", 10)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.scenario_successes = {"transit": [], "descend": [], "ascend": []}
        self.last_log_step = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                success = info.get("is_success", 0.0)
                self.successes.append(success)
                scenario = info.get("scenario")
                if scenario in self.scenario_successes:
                    self.scenario_successes[scenario].append(success)

        if (self.num_timesteps // self.log_freq) > (self.last_log_step // self.log_freq) and self.episode_rewards:
            self.last_log_step = self.num_timesteps
            mean_reward = np.mean(self.episode_rewards[-self.moving_avg_window:])
            mean_len = np.mean(self.episode_lengths[-self.moving_avg_window:])
            mean_success = np.mean(self.successes[-self.moving_avg_window:])
            
            parts = []
            for sc, slist in self.scenario_successes.items():
                if len(slist) >= self.scenario_min_episodes:
                    parts.append(f"{sc[:4]}={np.mean(slist[-self.scenario_log_window:])*100:.0f}%")
            scenario_str = " | " + " | ".join(parts) if parts else ""

            progress_logger.info(
                "Step %d | Reward: %.2f | Len: %.1f | Success: %.1f%%%s",
                self.num_timesteps, mean_reward, mean_len,
                mean_success * 100.0, scenario_str
            )
        return True

class HERFineTuneWarmupCallback(BaseCallback):
    """
    For fine-tuning HER models without a saved replay buffer.

    Keeping reset_num_timesteps=False lets the loaded policy act immediately, but
    it can also make SB3 train before HER has a complete episode to sample from.
    This callback pauses gradient updates until the first new episode finishes.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.original_gradient_steps = None
        self.released = False

    def _on_training_start(self) -> None:
        if hasattr(self.model.replay_buffer, "ep_length"):
            self.original_gradient_steps = self.model.gradient_steps
            self.model.gradient_steps = 0
            message = "HER fine-tune warmup: delaying gradient updates until the first complete episode."
            print(message)
            progress_logger.info(message)
        else:
            self.released = True

    def _on_step(self) -> bool:
        if self.released:
            return True

        ep_length = getattr(self.model.replay_buffer, "ep_length", None)
        has_complete_episode = ep_length is not None and bool(np.any(ep_length > 0))
        saw_episode_info = any("episode" in info for info in self.locals.get("infos", []))

        if has_complete_episode or saw_episode_info:
            self.model.gradient_steps = self.original_gradient_steps
            self.released = True
            message = "HER fine-tune warmup complete: gradient updates enabled."
            print(message)
            progress_logger.info(message)

        return True

class SuccessRateEvalCallback(EvalCallback):
    def __init__(self, *args, success_save_path="./checkpoints/", name="eval", **kwargs):
        super().__init__(*args, **kwargs)
        self.success_save_path = success_save_path
        self._best_success_rate = -1.0
        self.last_mean_success = -1.0
        self.name = name

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self._is_success_buffer:
                latest = float(np.mean(self._is_success_buffer))
                self.last_mean_success = latest
                
                os.makedirs(self.success_save_path, exist_ok=True)
                latest_save_path = os.path.join(self.success_save_path, f"latest_model_{self.name}")
                self.model.save(latest_save_path)
                
                if latest > self._best_success_rate:
                    self._best_success_rate = latest
                    save_path = os.path.join(self.success_save_path, f"best_model_{self.name}")
                    self.model.save(save_path)
                    progress_logger.info(
                        "*** NEW BEST EVAL (%s) *** | Success: %.1f%% -> saved to %s.zip",
                        self.name.upper(), latest * 100.0, save_path
                    )
        return result

class CombinedSuccessCallback(BaseCallback):
    """Saves model when the average eval success across all 3 scenarios improves."""
    def __init__(self, eval_callbacks, save_path, verbose=0):
        super().__init__(verbose)
        self.eval_callbacks = eval_callbacks
        self.save_path = save_path
        self._best_combined = -1.0

    def _on_step(self):
        # Use last_mean_success (current performance), not historical _best_success_rate
        rates = [cb.last_mean_success for cb in self.eval_callbacks
                 if cb.last_mean_success >= 0]
        if len(rates) == 3:
            combined = float(np.mean(rates))
            
            latest_path = os.path.join(self.save_path, "latest_model_combined")
            self.model.save(latest_path)
            
            if combined > self._best_combined:
                self._best_combined = combined
                path = os.path.join(self.save_path, "best_model_combined")
                self.model.save(path)
                progress_logger.info(
                    "*** NEW BEST COMBINED *** | Avg: %.1f%% (T=%.0f%% D=%.0f%% A=%.0f%%) -> %s.zip",
                    combined * 100.0,
                    rates[0] * 100.0, rates[1] * 100.0, rates[2] * 100.0, path
                )
        return True
