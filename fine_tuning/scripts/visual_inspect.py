import mujoco
import os
import numpy as np
from PIL import Image
import gymnasium as gym
import chess_env

def render_to_file(filename='chess_board_visual.png'):
    print(f"Creating environment and rendering to {filename}...")
    env = gym.make('ChessFetch-v0', render_mode='rgb_array')
    obs, _ = env.reset()
    
    # Grab the pixels from the environment's internal camera
    pixels = env.render()
    
    # Save image
    img = Image.fromarray(pixels)
    img.save(filename)
    print(f"Image saved to {os.path.abspath(filename)}")
    env.close()

def launch_viewer():
    # Keep the raw XML viewer option for those with displays
    import mujoco.viewer
    import time
    print("Attempting to launch interactive viewer (requires display)...")
    xml_path = os.path.abspath('chess_env/assets/pick_and_place.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < 10:
                mujoco.mj_step(model, data)
                viewer.sync()
    except Exception as e:
        print(f"Interactive viewer failed: {e}")

if __name__ == "__main__":
    try:
        # 1. Always render the 'real' reset state to PNG
        render_to_file()
        
        # 2. Try interactive viewer if DISPLAY exists
        if "DISPLAY" in os.environ:
            launch_viewer()
        else:
            print("No DISPLAY detected, skipping interactive viewer.")
            
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
