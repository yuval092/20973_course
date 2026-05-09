import numpy as np
import pytest
from src.chess_env.waypoints import validate_chain, derive_goal_pos, exit_waypoint

def test_validate_chain_valid():
    # Should not raise
    validate_chain(["transit", "descend", "ascend"])

def test_validate_chain_invalid():
    with pytest.raises(ValueError, match="Invalid chain transition"):
        validate_chain(["transit", "ascend"])

def test_derive_goal_pos():
    cell = np.array([0.4, 0.4])
    goal = derive_goal_pos("descend", cell)
    assert np.allclose(goal, [0.4, 0.4, 0.430]) # GRASP_Z

def test_derive_goal_pos_all_scenarios():
    cell = np.array([0.4, 0.4])
    assert np.allclose(derive_goal_pos("transit", cell), [0.4, 0.4, 0.550])
    assert np.allclose(derive_goal_pos("descend", cell), [0.4, 0.4, 0.430])
    assert np.allclose(derive_goal_pos("ascend", cell), [0.4, 0.4, 0.550])
    with pytest.raises(ValueError, match="Unknown scenario"):
        derive_goal_pos("invalid", cell)

def test_validate_chain_invalid_transitions():
    # Vertical to Horizontal at wrong height
    with pytest.raises(ValueError, match="Invalid chain transition"):
        validate_chain(["descend", "transit"])
    
    # Horizontal to Vertical lift from wrong height
    with pytest.raises(ValueError, match="Invalid chain transition"):
        validate_chain(["transit", "ascend"])
        
    # Repeated vertical
    with pytest.raises(ValueError, match="Invalid chain transition"):
        validate_chain(["descend", "descend"])

def test_derive_goal_pos_error():
    with pytest.raises(ValueError, match="Unknown scenario"):
        derive_goal_pos("invalid_scenario", np.array([0, 0]))

def test_exit_waypoint_error():
    with pytest.raises(ValueError, match="Unknown scenario"):
        exit_waypoint("invalid_scenario", np.array([0, 0]))

def test_exit_waypoint_all_scenarios():
    cell = np.array([0.5, 0.5])
    assert np.allclose(exit_waypoint("transit", cell), [0.5, 0.5, 0.550])
    assert np.allclose(exit_waypoint("descend", cell), [0.5, 0.5, 0.430])
    assert np.allclose(exit_waypoint("ascend", cell), [0.5, 0.5, 0.550])
    with pytest.raises(ValueError, match="Unknown scenario"):
        exit_waypoint("invalid", cell)

def test_chain_shortcuts_validity():
    from src.chess_env.waypoints import CHAIN_SHORTCUTS, validate_chain
    for chain in CHAIN_SHORTCUTS.values():
        validate_chain(chain)
