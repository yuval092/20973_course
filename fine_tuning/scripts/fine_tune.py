import os
import numpy as np
import gymnasium as gym
import chess_env
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

class SuccessRateEvalCallback(EvalCallback):
    """
    Extends EvalCallback to additionally save the best checkpoint
    by is_success rate rather than mean reward.
    """
    def __init__(self, *args, success_save_path: str = './checkpoints/', **kwargs):
        super().__init__(*args, **kwargs)
        self.success_save_path  = success_save_path
        self._best_success_rate = -1.0

    def _on_step(self) -> bool:
        result = super()._on_step()
        if len(self.evaluations_successes) > 0:
            latest = float(np.mean(self.evaluations_successes[-1]))
            if latest > self._best_success_rate:
                self._best_success_rate = latest
                os.makedirs(self.success_save_path, exist_ok=True)
                # best_model.zip is the symbolic link to the actual timestamped file
                save_path = os.path.join(self.success_save_path, 'best_model')
                self.model.save(save_path)
                if self.verbose >= 1:
                    print(f'[SuccessCallback] New best success rate: {latest*100:.1f}% -> {save_path}.zip')
        return result

def make_env():
    return gym.make('ChessFetch-v0')

def fine_tune():
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"fine_tune_{timestamp}"
    
    print(f"Starting run: {run_name}")
    
    # Ensure directories exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    train_env = DummyVecEnv([make_env])
    eval_env  = DummyVecEnv([make_env])

    # Save specifically for THIS run
    success_save_path = f'./checkpoints/{run_name}/'
    log_path = f'./logs/{run_name}/'
    
    eval_callback = SuccessRateEvalCallback(
        eval_env,
        success_save_path=success_save_path,
        log_path=log_path,
        eval_freq=5000,
        n_eval_episodes=30,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # ALWAYS load from the original base model for a fresh start
    model_path = 'models/sac-FetchPickAndPlace-v4.zip'
    
    print(f"Loading ORIGINAL pretrained model from {model_path}...")
    model = SAC.load(
        model_path,
        env=train_env,
        learning_rate=1e-4,
        batch_size=512,
        verbose=1,
    )
    model.tensorboard_log = None
    # Reset learning_starts to collect fresh data for this new geometry
    model.learning_starts = model.num_timesteps + 1000

    print(f"Starting fine-tuning for 300,000 steps with {train_env.envs[0].spec.max_episode_steps} step limit...")
    model.learn(
        total_timesteps=300_000,
        callback=eval_callback,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    final_model_path = f'chess_fetch_{timestamp}.zip'
    model.save(final_model_path)
    print(f'Fine-tuning complete. Final model saved as {final_model_path}')

if __name__ == "__main__":
    fine_tune()
