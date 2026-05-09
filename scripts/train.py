import sys
import os
import argparse

# Ensure src is importable
sys.path.append(os.getcwd())

from src.training.trainer import SACTrainer

def main():
    parser = argparse.ArgumentParser(description="Train the RoboChess agent using SAC.")
    parser.add_argument("--envs", type=int, help="Number of parallel environments.")
    parser.add_argument("--model", type=str, help="Path to a specific model .zip to start from.")
    parser.add_argument("--fresh", action="store_true", help="Start training from a fresh model (base weights).")
    parser.add_argument("--timesteps", type=int, help="Total timesteps to train.")
    parser.add_argument("--config", type=str, help="Path to config file (optional).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging in the environment.")
    parser.add_argument("--fixed-drift", action="store_true", help="Bypass the horizontal drift curriculum and lock the constraint to the final 5mm radius.")
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Continue the loaded model timestep counter, so pretrained policy actions are used immediately.",
    )
    parser.add_argument(
        "--keep-entropy",
        action="store_true",
        help="Keep the loaded SAC entropy coefficient for conservative fine-tuning.",
    )
    parser.add_argument(
        "--scenario",
        choices=["transit", "descend", "ascend", "all"],
        default="all",
        help="Focus training on one movement scenario. Use 'descend' to fine-tune HOVER_Z descent.",
    )
    
    args = parser.parse_args()
    
    # Note: SACTrainer loads configs from src/utils/config.py
    # If we wanted to support a custom --config path for YAMLs, we'd need to modify load_config.
    # For now, we follow the trainer's internal config loading.
    
    scenario = None if args.scenario == "all" else args.scenario
    trainer = SACTrainer(
        num_envs=args.envs,
        fresh_start=args.fresh,
        debug=args.debug,
        fixed_drift=args.fixed_drift,
        scenario=scenario,
        reset_num_timesteps=not args.finetune,
        reset_entropy=not args.keep_entropy,
    )
    
    if args.timesteps:
        trainer.total_timesteps = args.timesteps
        
    trainer.train(model_path=args.model)

if __name__ == "__main__":
    main()
