import os
import glob
import numpy as np
import gymnasium as gym
from PIL import Image
import chess_env
from stable_baselines3 import SAC

def generate_montage(model_path, n_images=6, filename='final_grandmaster_montage.png'):
    print(f"Loading model from {model_path}...")
    
    # Setup environment with explicit render mode
    env = gym.make('ChessFetch-v0', render_mode='rgb_array')
    model = SAC.load(model_path, env=env)
    
    frames = []
    
    for i in range(n_images):
        print(f"Capturing scenario {i+1}/{n_images}...")
        obs, _ = env.reset()
        # Run for 120 steps to show the movement near completion
        for _ in range(120):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
        
        # Take a snapshot
        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))
        else:
            print("Warning: Render returned None")
    
    if not frames:
        print("No frames captured.")
        return

    # Create a grid montage (2 rows, 3 columns)
    cols = 3
    rows = (len(frames) + cols - 1) // cols
    w, h = frames[0].size
    montage = Image.new('RGB', (w * cols, h * rows))
    
    for idx, frame in enumerate(frames):
        r, c = divmod(idx, cols)
        montage.paste(frame, (c * w, r * h))
        
    montage.save(filename)
    print(f"Final montage saved to {os.path.abspath(filename)}")
    env.close()

if __name__ == "__main__":
    import os
    model = 'chess_fetch_20260421_152804.zip'
    if os.path.exists(model):
        generate_montage(model)
    else:
        print("Final model not found.")
