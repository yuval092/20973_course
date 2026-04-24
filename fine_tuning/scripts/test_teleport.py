import gymnasium as gym
import numpy as np
import gymnasium_robotics

env = gym.make('FetchPickAndPlace-v4', max_episode_steps=100)
obs, info = env.reset()

print("Initial grip pos:", obs['observation'][:3])

# Try to teleport mocap to a new position
new_pos = obs['observation'][:3] + np.array([0.2, 0.2, 0.2])
env.unwrapped.data.mocap_pos[0] = new_pos

# Step with zero action
for i in range(10):
    obs, r, term, trunc, info = env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    print(f"Step {i} grip pos:", obs['observation'][:3])
