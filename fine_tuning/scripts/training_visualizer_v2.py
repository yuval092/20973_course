import os
import numpy as np
import gymnasium as gym
from PIL import Image
import chess_env
from stable_baselines3 import SAC

def generate_montage(model_path, n_images=4, filename='dense_refinement_progress.png'):
    print(f"Loading model from {model_path}...")
    env = gym.make('ChessFetchDense-v0', render_mode='rgb_array')
    
    # Custom objects to fix the ent_coef mismatch during load
    custom_objects = {"ent_coef": 0.001}
    try:
        model = SAC.load(model_path, env=env, custom_objects=custom_objects)
    except:
        # If that fails, try without custom objects but with manual fix
        model = SAC.load(model_path, env=env)
        
    frames = []
    for i in range(n_images):
        print(f"Capturing scenario {i+1}/{n_images}...")
        obs, _ = env.reset()
        for _ in range(120): # Run longer to see final placement
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
        frames.append(Image.fromarray(env.render()))
    
    # Create montage
    w, h = frames[0].size
    montage = Image.new('RGB', (w * n_images, h))
    for i, f in enumerate(frames):
        montage.paste(f, (i * w, 0))
    
    montage.save(filename)
    print(f"Montage saved to {os.path.abspath(filename)}")
    env.close()

if __name__ == "__main__":
    import glob
    runs = glob.glob('checkpoints/dense_refinement_*/best_model.zip')
    if runs:
        runs.sort()
        generate_montage(runs[-1])
    else:
        print("No dense model found.")
