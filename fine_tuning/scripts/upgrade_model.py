import os
import torch
import numpy as np
from stable_baselines3 import SAC
import gymnasium as gym
import chess_env

def upgrade_model(old_path, new_path):
    print(f"Upgrading {old_path} -> {new_path}")
    
    # We need a dummy env with the OLD shape (25) to load the model
    # I'll temporarily patch the environment code to return 25
    # or just use any 25-dim environment like FetchPickAndPlace-v2
    # but simpler: load it with custom_objects to bypass some checks
    
    # Create the NEW env (26 dims)
    new_env = gym.make('ChessFetchDense-v0')
    
    # Instantiate the new model (26 dims)
    new_model = SAC('MultiInputPolicy', new_env, verbose=1)
    new_params = new_model.policy.state_dict()
    
    # Load old parameters
    # SAC saves a .zip which contains policy.pth or similares
    # We can use SAC.load but we need to satisfy the env check.
    # Hack: create a dummy env wrapper that claims to have 25 dims
    class LegacyEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Dict({
                'observation': gym.spaces.Box(-np.inf, np.inf, (25,), np.float64),
                'achieved_goal': gym.spaces.Box(-np.inf, np.inf, (3,), np.float64),
                'desired_goal': gym.spaces.Box(-np.inf, np.inf, (3,), np.float64)
            })
            self.action_space = gym.spaces.Box(-1, 1, (4,), np.float32)
            self.metadata = {}
            self.render_mode = None
        def reset(self, seed=None): return {}, {}
        def step(self, action): return {}, 0, False, False, {}
        def close(self): pass

    old_model = SAC.load(old_path, env=LegacyEnv())
    old_params = old_model.policy.state_dict()
    
    print("Mapping weights...")
    for name, param in old_params.items():
        if name in new_params:
            if param.shape == new_params[name].shape:
                new_params[name].copy_(param)
            else:
                print(f"  Resizing layer {name}: {param.shape} -> {new_params[name].shape}")
                # This is likely the first layer of the MLP
                # Shape is [hidden, input]
                # input is concatenated [achieved(3), desired(3), obs(25/26)]
                # Old concat length = 3+3+25 = 31
                # New concat length = 3+3+26 = 32
                # We copy [:, :31] and the 32nd input is the Scenario ID
                new_params[name][:, :param.shape[1]].copy_(param)
                # Zero out the rest or leave as random? Zeros is safer.
                new_params[name][:, param.shape[1]:].zero_()
        else:
            print(f"  Skipping {name}")

    new_model.policy.load_state_dict(new_params)
    
    # Sync target networks
    new_model.critic_target.load_state_dict(new_model.critic.state_dict())
    
    new_model.save(new_path)
    print(f"Upgrade complete! Saved to {new_path}")

if __name__ == "__main__":
    latest = "chess_fetch_dense_20260423_235832.zip"
    upgrade_model(latest, "chess_fetch_dense_v3_ready.zip")
