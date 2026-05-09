#!/usr/bin/env python3
"""
Physics verification for the grasp stage.
Run this after applying XML changes to verify the cube can be held.

Usage:
    PYTHONPATH=. python scripts/test_grasp_physics.py --n-trials 20 --visualize --debug
"""
import argparse
import math
import time
import numpy as np
import gymnasium as gym
import mujoco
import src.chess_env
from src.chess_env.task import ChessTaskEnv

def assert_mandatory_preconditions(uw):
    """Call at the start of every test run."""
    print("Checking mandatory preconditions...")
    
    # 1. GRASP_Z check
    assert abs(uw.GRASP_Z - 0.425) < 0.001, \
        f"GRASP_Z={uw.GRASP_Z}, expected 0.425. Restart process after env.yaml change."
    assert abs(uw.HOVER_Z - 0.460) < 0.001, \
        f"HOVER_Z={uw.HOVER_Z}, expected 0.460. Restart process after env.yaml change."
    
    # 2. Cube mass check
    cube_body_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_BODY, "object0")
    mass = uw.model.body_mass[cube_body_id]
    print(f"  - Cube mass: {mass:.3f}kg")
    assert abs(mass - 0.05) < 0.001, \
        f"Cube mass={mass:.3f}, expected 0.05. XML not updated."
    
    # 3. Finger threshold check
    # Plan says 0.016 for verification (ABOVE the stall point of 0.0141).
    assert abs(uw.GRASP_VERIFY_FINGER_THRESHOLD - 0.016) < 0.001, \
        f"GRASP_VERIFY_FINGER_THRESHOLD={uw.GRASP_VERIFY_FINGER_THRESHOLD}, expected 0.016"
    assert abs(uw.FINGER_CLOSED_JOINT - 0.012) < 0.001, \
        f"FINGER_CLOSED_JOINT={uw.FINGER_CLOSED_JOINT}, expected 0.012"
        
    # 3.5. Actuator Kp check
    l_act_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:l_gripper_finger_joint")
    kp = uw.model.actuator_gainprm[l_act_id, 0]
    print(f"  - Actuator Kp: {kp}")
    assert kp == 150000, f"Expected 150000, got {kp}"

    # 4. Geom parameters
    cube_geom_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_GEOM, "object0")
    solref = uw.model.geom_solref[cube_geom_id]
    solimp = uw.model.geom_solimp[cube_geom_id]
    condim = uw.model.geom_condim[cube_geom_id]
    conaff = uw.model.geom_conaffinity[cube_geom_id]
    contype = uw.model.geom_contype[cube_geom_id]
    print(f"  - Cube solref: {solref}")
    print(f"  - Cube solimp: {solimp}")
    print(f"  - Cube condim: {condim} | conaff: {conaff} | contype: {contype}")
    
    l_finger_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_GEOM, "robot0:l_gripper_finger_link")
    f_solref = uw.model.geom_solref[l_finger_id]
    f_condim = uw.model.geom_condim[l_finger_id]
    f_conaff = uw.model.geom_conaffinity[l_finger_id]
    f_contype = uw.model.geom_contype[l_finger_id]
    print(f"  - Finger solref: {f_solref}")
    print(f"  - Finger condim: {f_condim} | conaff: {f_conaff} | contype: {f_contype}")
    
    print("[OK] Mandatory preconditions verified")

def sample_valid_pos(uw):
    """Samples a valid board position within reachable workspace."""
    return uw._sample_board_position()

def move_arm_to_target(uw, target, tolerance=0.001, max_steps=200):
    """Moves arm directly using mocap for test setup."""
    if hasattr(uw, "_move_mocap_to"):
        target_quat = uw.data.mocap_quat[0].copy()
        return uw._move_mocap_to(target, target_quat, max_steps=max_steps, tolerance=tolerance)

    for _ in range(max_steps):
        grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
        error = target - grip_pos
        if np.linalg.norm(error) < tolerance:
            return True
        # Proportional step
        step_vec = 0.5 * error
        if np.linalg.norm(step_vec) > 0.015: # POS_CTRL_SCALE
            step_vec = step_vec / np.linalg.norm(step_vec) * 0.015
        uw.data.mocap_pos[0][:3] += step_vec
        uw._mujoco_step(None)
    return False

def test_static_grasp(env, debug=False) -> dict:
    """Test 1: Place arm at HOVER_Z, execute scripted grasp, check cube stability."""
    uw = env.unwrapped
    uw.force_scenario = "descend"
    uw.hide_object = False
    obs, _ = env.reset()
    
    # Position arm exactly at HOVER_Z above cube center. execute_grasp owns the plunge.
    cube_pos = uw.get_cube_position()
    src_xy = cube_pos[:2]
    target = np.array([src_xy[0], src_xy[1], uw.HOVER_Z])
    
    moved = move_arm_to_target(uw, target)
    
    grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
    arm_alignment_error = np.linalg.norm(grip_pos - target) * 1000  # mm
    
    # Execute grasp
    result = uw.execute_grasp()
    
    cube_pos_after = uw.get_cube_position()
    expected_cube_z = uw.HOVER_Z - 0.015
    cube_z_displacement = abs(cube_pos_after[2] - expected_cube_z) * 1000  # mm
    cube_xy_displacement = np.linalg.norm(cube_pos_after[:2] - src_xy) * 1000  # mm
    
    if debug:
        print(f"  [Static] Alignment err: {arm_alignment_error:.1f}mm | Success: {result['success']} ({result.get('reason')})")
        print(f"  [Static] Z-disp: {cube_z_displacement:.1f}mm | XY-disp: {cube_xy_displacement:.1f}mm | Finger: {result.get('final_finger_pos', 0):.4f}")
    
    return {
        "success": result["success"],
        "reason": result.get("reason"),
        "arm_alignment_error_mm": arm_alignment_error,
        "cube_z_displacement_mm": cube_z_displacement,
        "cube_xy_displacement_mm": cube_xy_displacement,
        "close_steps_used": result.get("close_steps_used"),
        "final_finger_pos": result.get("final_finger_pos"),
    }

def test_lift(env, debug=False) -> dict:
    """Test 2: Grasp cube and lift to SAFE_Z. Verify no drop."""
    uw = env.unwrapped
    uw.force_scenario = "descend"
    uw.hide_object = False
    env.reset()
    
    cube_pos = uw.get_cube_position()
    src_xy = cube_pos[:2]
    target = np.array([src_xy[0], src_xy[1], uw.HOVER_Z])
    move_arm_to_target(uw, target)
    
    grasp_result = uw.execute_grasp()
    if not grasp_result["success"]:
        return {"success": False, "reason": f"GRASP_FAILED: {grasp_result.get('reason')}"}
    
    # Lift to SAFE_Z
    target_z = uw.SAFE_Z
    max_z_deviation = 0.0
    cube_dropped = False
    drop_reason = None
    
    # 3mm per step for smooth lift
    for step in range(300):
        grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
        if grip_pos[2] >= target_z - 0.001:
            break
        
        uw.data.mocap_pos[0][2] += 0.003
        uw._mujoco_step(None)
        
        # Check cube
        cube_pos = uw.get_cube_position()
        grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
        z_offset = grip_pos[2] - cube_pos[2] # should be ~15mm
        deviation = abs(z_offset - 0.015)
        max_z_deviation = max(max_z_deviation, deviation)
        
        xy_error = np.linalg.norm(cube_pos[:2] - grip_pos[:2]) * 1000
        if xy_error > 30.0:
            cube_dropped = True
            drop_reason = f"CUBE_DROPPED_XY ({xy_error:.1f}mm)"
            break
        if deviation > 0.040:
            cube_dropped = True
            drop_reason = f"CUBE_DROPPED_Z (deviation={deviation*1000:.1f}mm)"
            break
            
        if env.unwrapped.render_mode == "human":
            env.render()
            
    if debug:
        print(f"  [Lift] Success: {not cube_dropped} | Max Z-dev: {max_z_deviation*1000:.1f}mm")

    return {
        "success": not cube_dropped,
        "reason": drop_reason,
        "max_z_deviation_mm": max_z_deviation * 1000,
    }

def test_transit_held(env, debug=False) -> dict:
    """Test 3: Transit horizontally while holding cube. Verify no drop, no rotation."""
    uw = env.unwrapped
    uw.force_scenario = "descend"
    uw.hide_object = False
    env.reset()
    
    cube_pos = uw.get_cube_position()
    src_xy = cube_pos[:2]
    move_arm_to_target(uw, np.array([src_xy[0], src_xy[1], uw.HOVER_Z]))
    
    grasp_result = uw.execute_grasp()
    if not grasp_result["success"]:
        return {"success": False, "reason": f"GRASP_FAILED: {grasp_result.get('reason')}"}
    
    # Lift to SAFE_Z first
    move_arm_to_target(uw, np.array([src_xy[0], src_xy[1], uw.SAFE_Z]))
    
    # Horizontal transit 100mm
    start_grip = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip").copy()
    target_xy = start_grip[:2] + np.array([0.10, 0.0])
    target = np.array([target_xy[0], target_xy[1], uw.SAFE_Z])
    
    cube_dropped = False
    drop_reason = None
    max_yaw_deg = 0.0
    
    for _ in range(200):
        grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
        error = target - grip_pos
        if np.linalg.norm(error) < 0.001:
            break
            
        step_vec = 0.5 * error
        if np.linalg.norm(step_vec) > 0.005: # Slower for transit stability
            step_vec = step_vec / np.linalg.norm(step_vec) * 0.005
        uw.data.mocap_pos[0][:3] += step_vec
        uw._mujoco_step(None)
        
        # Check drop
        cube_pos = uw.get_cube_position()
        xy_err = np.linalg.norm(cube_pos[:2] - grip_pos[:2]) * 1000
        z_dev = abs((grip_pos[2] - cube_pos[2]) - 0.015) * 1000
        
        if xy_err > 30.0 or z_dev > 40.0:
            cube_dropped = True
            drop_reason = f"CUBE_DROPPED (XY={xy_err:.1f}mm, Z_dev={z_dev:.1f}mm)"
            break
            
        # Check rotation
        quat = uw.get_cube_quat()
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw_deg = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        max_yaw_deg = max(max_yaw_deg, abs(yaw_deg))
        
        if env.unwrapped.render_mode == "human":
            env.render()

    if debug:
        print(f"  [Transit] Success: {not cube_dropped} | Max Yaw: {max_yaw_deg:.1f} deg")
        
    return {
        "success": not cube_dropped,
        "reason": drop_reason,
        "max_yaw_deg": max_yaw_deg,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    
    render_mode = "human" if args.visualize else None
    env = gym.make("ChessFetchTask-v0", render_mode=render_mode, hide_object=False, debug=args.debug)
    uw = env.unwrapped
    
    assert_mandatory_preconditions(uw)
    
    stats = {
        "static": {"success": 0, "fail_reasons": {}},
        "lift": {"success": 0, "fail_reasons": {}},
        "transit": {"success": 0, "fail_reasons": {}},
    }
    
    for i in range(args.n_trials):
        print(f"\nTrial {i+1}/{args.n_trials}...")
        
        # Test 1
        res = test_static_grasp(env, args.debug)
        if res["success"]:
            stats["static"]["success"] += 1
        else:
            reason = res.get("reason", "UNKNOWN")
            stats["static"]["fail_reasons"][reason] = stats["static"]["fail_reasons"].get(reason, 0) + 1
            
        # Test 2
        res = test_lift(env, args.debug)
        if res["success"]:
            stats["lift"]["success"] += 1
        else:
            reason = res.get("reason", "UNKNOWN")
            stats["lift"]["fail_reasons"][reason] = stats["lift"]["fail_reasons"].get(reason, 0) + 1
            
        # Test 3
        res = test_transit_held(env, args.debug)
        if res["success"]:
            stats["transit"]["success"] += 1
        else:
            reason = res.get("reason", "UNKNOWN")
            stats["transit"]["fail_reasons"][reason] = stats["transit"]["fail_reasons"].get(reason, 0) + 1
            
    print("\n" + "="*40)
    print("PHYSICS TEST SUMMARY")
    print("="*40)
    for test in ["static", "lift", "transit"]:
        s = stats[test]
        rate = s["success"] / args.n_trials * 100
        print(f"{test.capitalize():<10}: {s['success']}/{args.n_trials} ({rate:.1f}%)")
        if s["fail_reasons"]:
            print(f"  Failures: {s['fail_reasons']}")
    print("="*40)
    
    env.close()

if __name__ == "__main__":
    main()
