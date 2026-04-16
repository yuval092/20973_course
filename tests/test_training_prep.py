import mujoco
import numpy as np
import pytest
import chess
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.config import SCENE_XML, N_SUBSTEPS, TABLE_HEIGHT
from src.env.chess_manipulation_train_env import ChessManipulationTrainEnv
from src.env.episode_sampler import EpisodeSampler, Episode
from src.control.execution_controller import ExecutionController

def test_env_obs_shape():
    """Validate that the existing ChessPickPlaceEnv observation is correct (Section 2.4)."""
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
    # Use a real piece name from the model
    env.set_target("w_pawn_1", np.array([1.075, 0.715, 0.422]))
    obs = env.get_obs()

    assert obs["observation"].shape == (26,), f"Expected (26,), got {obs['observation'].shape}"
    assert obs["achieved_goal"].shape == (3,)
    assert obs["desired_goal"].shape == (3,)
    assert not np.any(np.isnan(obs["observation"])), "NaN in observation"

def test_train_env_obs_shape():
    """Validate ChessManipulationTrainEnv observation shape (Section 3)."""
    env = ChessManipulationTrainEnv(curriculum_stage=1, seed=42)
    obs, _ = env.reset()
    
    # 26 (base) + 18 (context) = 44
    assert obs["observation"].shape == (44,)
    assert obs["achieved_goal"].shape == (3,)
    assert obs["desired_goal"].shape == (3,)

def test_episode_sampler_stage_1():
    rng = np.random.default_rng(42)
    sampler = EpisodeSampler(rng, curriculum_stage=1)
    ep = sampler.sample()
    
    assert ep.piece_type == "pawn"
    assert ep.piece_name.startswith("w_pawn_")
    assert ep.src_square in ["d4", "e4", "d5", "e5"]
    assert ep.dst_square in ["d4", "e4", "d5", "e5"]

def test_reward_fn_basic():
    from src.env.reward_fn import compute_reward
    achieved = np.array([1.0, 1.0, 0.422])
    desired = np.array([1.0, 1.0, 0.422])
    info = {
        "phase": "placement",
        "grip_pos": np.array([1.0, 1.0, 0.422]),
        "up_z": 1.0,
        "table_height": 0.421
    }
    reward = compute_reward(achieved, desired, info)
    # Should have placement bonus
    assert reward >= 50.0
