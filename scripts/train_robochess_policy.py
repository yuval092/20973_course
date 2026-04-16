"""
Fine-tune the TQC+HER FetchPickAndPlace checkpoint on the RoboChess scene.
"""

import argparse
import os
import json
import numpy as np
import torch
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from sb3_contrib import TQC
from src.env.chess_manipulation_train_env import ChessManipulationTrainEnv

# ── Hyperparameters ──────────────────────────────────────────────────
EVAL_FREQ       = 10_000    
CHECKPOINT_FREQ = 20_000    
N_HER_GOALS     = 4         
LEARNING_RATE   = 1e-4      
BUFFER_SIZE     = 1_000_000
BATCH_SIZE      = 256
TAU             = 0.005
GAMMA           = 0.98

def make_env(stage: int, seed: int, rank: int):
    def _init():
        env = ChessManipulationTrainEnv(curriculum_stage=stage, seed=seed + rank)
        return env
    return _init

def should_advance_stage(metrics: dict, current_stage: int) -> bool:
    """Gate advancement on eval metrics (Section 7.2)."""
    thresholds = {
        1: {"grasp_rate": 0.95, "place_rate": 0.90},
        2: {"grasp_rate": 0.95, "place_rate": 0.90},
        3: {"grasp_rate": 0.92, "place_rate": 0.87},
        4: {"grasp_rate": 0.90, "place_rate": 0.85, "clutter_rate": 0.05},
        5: {"place_rate": 0.85},
        6: {"place_rate": 0.80},
    }
    if current_stage not in thresholds:
        return False
    t = thresholds[current_stage]
    for key, val in t.items():
        if key == "clutter_rate":
            if metrics.get(key, 1.0) > val:
                return False
        else:
            if metrics.get(key, 0.0) < val:
                return False
    return True

def main(args):
    np.random.seed(args.seed)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # ── Environment setup ────────────────────────────────────────────
    if args.n_envs > 1:
        vec_env = SubprocVecEnv([
            make_env(args.stage, args.seed, i) for i in range(args.n_envs)
        ])
    else:
        vec_env = DummyVecEnv([make_env(args.stage, args.seed, 0)])
        
    vec_env = VecMonitor(vec_env, filename=f"logs/training_stage{args.stage}")

    eval_env = ChessManipulationTrainEnv(curriculum_stage=args.stage, seed=9999)

    # ── Load pretrained checkpoint ───────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Check if we are loading from hub or local
    from src.config import HF_REPO_ID, HF_FILENAME
    if args.checkpoint == HF_REPO_ID:
        from huggingface_sb3 import load_from_hub
        checkpoint_path = load_from_hub(HF_REPO_ID, HF_FILENAME)
    else:
        checkpoint_path = args.checkpoint

    model = TQC.load(
        checkpoint_path,
        env       = vec_env,
        verbose   = 1,
        learning_rate = LEARNING_RATE,
        buffer_size   = BUFFER_SIZE,
        batch_size    = BATCH_SIZE,
        tau           = TAU,
        gamma         = GAMMA,
        replay_buffer_class = HerReplayBuffer,
        replay_buffer_kwargs = {
            "n_sampled_goal": N_HER_GOALS,
            "goal_selection_strategy": "future",
        },
        device = "auto",
    )
    model._last_obs = None   

    # ── Callbacks ────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq    = CHECKPOINT_FREQ,
        save_path    = f"checkpoints/stage{args.stage}/",
        name_prefix  = "tqc_robochess",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = f"checkpoints/stage{args.stage}/best/",
        log_path             = f"logs/eval_stage{args.stage}/",
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = 50,
        deterministic        = True,
    )

    # ── Training loop ────────────────────────────────────────────────
    current_stage = args.stage
    steps_per_stage = args.total_timesteps

    # Note: run_eval_suite would be imported from scripts.eval_robochess_policy
    # We'll define a stub or import it if we create it in time.
    from scripts.eval_robochess_policy import run_eval_suite

    while current_stage <= 7:
        print(f"\n{'='*60}")
        print(f"  Training Stage {current_stage}")
        print(f"{'='*60}")

        model.learn(
            total_timesteps    = steps_per_stage,
            reset_num_timesteps= False,
            callback           = [checkpoint_cb, eval_cb],
            log_interval       = 10,
            tb_log_name        = f"stage{current_stage}",
        )

        metrics = run_eval_suite(model, stage=current_stage, n_episodes=100, seed=9999)
        print(f"Stage {current_stage} eval metrics: {metrics}")

        with open(f"logs/stage{current_stage}_final_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        if should_advance_stage(metrics, current_stage):
            print(f"✓ Stage {current_stage} gates met — advancing to Stage {current_stage+1}")
            current_stage += 1
            vec_env.env_method("set_curriculum_stage", current_stage)
            eval_env.set_curriculum_stage(current_stage)
        else:
            print(f"✗ Stage {current_stage} gates NOT met — continuing training")
            model.learning_rate = max(model.learning_rate * 0.95, 1e-5)

    model.save("checkpoints/final/tqc_robochess_final")
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to our hub model ID
    parser.add_argument("--checkpoint",       default="sb3/tqc-FetchPickAndPlace-v1")
    parser.add_argument("--stage",            type=int,   default=1)
    parser.add_argument("--total-timesteps",  type=int,   default=500_000)
    parser.add_argument("--n-envs",           type=int,   default=4)
    parser.add_argument("--seed",             type=int,   default=42)
    main(parser.parse_args())
