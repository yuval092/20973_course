import sys
import os
import logging
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.bootstrap import SystemBootstrapper
from src.config import SCENE_XML, LOCAL_MODEL_PATH

logging.basicConfig(level=logging.INFO)

def main():
    print(f"Loading scene: {SCENE_XML}")
    mj_model, mj_data = SystemBootstrapper.load_scene(SCENE_XML)
    
    from src.env.chess_pick_place_env import ChessPickPlaceEnv
    env = ChessPickPlaceEnv(mj_model, mj_data)
    
    print(f"Loading model from: {LOCAL_MODEL_PATH}")
    try:
        model = SystemBootstrapper.load_rl_policy(env, LOCAL_MODEL_PATH)
        print(f"Model loaded: {type(model).__name__}")
        
        # Test prediction
        env.set_target("w_pawn_1", [1.18, 0.75, 0.45])
        obs = env.get_obs()
        action, _ = model.predict(obs, deterministic=True)
        print(f"Sample prediction successful: {action}")
        
        print("RL model OK.")
        return 0
    except Exception as e:
        print(f"RL model verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
