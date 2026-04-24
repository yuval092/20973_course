import gymnasium as gym
import numpy as np
from PIL import Image
import os
import chess_env

def save_visual(scenario_name, filename):
    env = gym.make('ChessFetchDense-v0', force_scenario=scenario_name, render_mode='rgb_array')
    obs, _ = env.reset()
    
    # Step a few times to let it settle visually
    for _ in range(5):
        env.step(np.array([0, 0, 0, 0], dtype=np.float32))
        
    pixels = env.render()
    img = Image.fromarray(pixels)
    img.save(filename)
    print(f"Saved {filename}")
    env.close()

if __name__ == "__main__":
    os.makedirs('logs/visuals', exist_ok=True)
    save_visual('transit', 'logs/visuals/scenario_transit.png')
    save_visual('descend', 'logs/visuals/scenario_descend.png')
    save_visual('ascend', 'logs/visuals/scenario_ascend.png')
