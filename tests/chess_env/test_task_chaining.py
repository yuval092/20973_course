import gymnasium as gym
import numpy as np
import pytest
import src.chess_env

def test_transition_validate():
    env = gym.make("ChessFetchTask-v0", render_mode=None)
    env.reset()
    diag = env.unwrapped.transition_validate()
    assert "grip_speed_mm_s" in diag
    assert "is_velocity_ok" in diag
    assert diag["error_from_nominal_mm"] is None
    
    # Test with nominal position
    nominal_pos = np.array(diag["grip_pos"]) + np.array([0.001, 0, 0]) # 1mm away
    diag_with_nominal = env.unwrapped.transition_validate(nominal_exit_pos=nominal_pos)
    assert pytest.approx(diag_with_nominal["error_from_nominal_mm"], abs=1e-3) == 1.0
    
    env.close()

def test_soft_reset_flow():
    env = gym.make("ChessFetchTask-v0", render_mode=None)
    env.reset()
    
    # Simulate a success state
    uw = env.unwrapped
    uw.current_scenario = "transit"
    nominal_xy = np.array([0.5, 0.5])
    exit_wp = np.array([0.5, 0.5, 0.550])
    next_goal = np.array([0.5, 0.5, 0.430]) # descend
    
    obs, info = uw.soft_reset(
        new_scenario="descend",
        new_goal_pos=next_goal,
        nominal_exit_pos=exit_wp,
        nominal_xy=nominal_xy
    )
    
    assert isinstance(obs, dict)
    assert "observation" in obs
    assert "halt_steps" in info
    assert "align_steps" in info
    assert uw.current_scenario == "descend"
    assert np.allclose(uw.goal, next_goal)
    assert uw.episode_steps == 0
    env.close()

def test_soft_reset_finger_validation():
    env = gym.make("ChessFetchTask-v0", render_mode=None)
    env.reset()
    uw = env.unwrapped
    
    nominal_xy = np.array([0.5, 0.5])
    exit_wp = np.array([0.5, 0.5, 0.550])
    next_goal = np.array([0.5, 0.5, 0.430])
    
    # This should pass without error with the current physics
    obs, info = uw.soft_reset(
        new_scenario="descend",
        new_goal_pos=next_goal,
        nominal_exit_pos=exit_wp,
        nominal_xy=nominal_xy
    )
    assert isinstance(info, dict)
    env.close()
