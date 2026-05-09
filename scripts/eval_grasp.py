#!/usr/bin/env python3
"""
Full pick sequence evaluation: home → transit → descend → GRASP → ascend → transit(home).

IMPORTANT: This script does NOT use ChainEpisodeRunner. It uses explicit while-loops
because the GRASP step must be inserted between DESCEND and ASCEND.

Usage:
    PYTHONPATH=. python scripts/eval_grasp.py \
        --model checkpoints/pure_movement_v6_20260504_090639/best_model_combined.zip \
        --n-episodes 50 --drift-limit 0.010 --debug --visualize --wait
"""
import argparse
import math
import time
import numpy as np
import gymnasium as gym
import mujoco
import src.chess_env
from stable_baselines3 import SAC

SAFE_Z   = 0.550
GRASP_Z  = 0.425
HOVER_Z  = 0.460
TABLE_Z  = 0.400
CUBE_H   = 0.030


def reset_episode_timelimit(env):
    """Reset the TimeLimit wrapper step counter. No helper function exists — traverse inline."""
    curr = env
    while hasattr(curr, "env"):
        if hasattr(curr, "_elapsed_steps"):
            curr._elapsed_steps = 0
            break
        curr = curr.env


def run_scenario_loop(env, model, initial_obs, max_steps=500, debug=False, label="", delay=0.0):
    """
    Run one RL scenario with an explicit step loop.
    Returns: {"outcome": "success"|"crash"|"timeout", "steps": int, "obs": np.ndarray,
              "crash_reason": str|None}
    """
    obs = initial_obs
    done = False
    steps = 0
    outcome = "timeout"
    crash_reason = None
    visualize = (env.unwrapped.render_mode == "human")

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        if visualize:
            env.render()
            if delay > 0:
                time.sleep(delay)

        if debug:
            uw = env.unwrapped
            g = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
            # print(f"  [{label} step {steps}] grip=({g[0]*1000:.1f}, {g[1]*1000:.1f}, {g[2]*1000:.1f})mm")

        if info.get("is_success"):
            outcome = "success"
            break
        if terminated and not info.get("is_success"):
            outcome = "crash"
            crash_reason = info.get("crash_reason", "UNKNOWN_CRASH")
            break

    return {"outcome": outcome, "steps": steps, "obs": obs, "crash_reason": crash_reason}


def scripted_recover_to_hover(env, xy: np.ndarray, debug=False, delay=0.0) -> dict:
    """
    Recovery path for the old descend policy. If RL fails to stop at HOVER_Z,
    move there with the same mocap-aware scripted helper used by execute_grasp().
    """
    uw = env.unwrapped
    target = np.array([xy[0], xy[1], uw.HOVER_Z])

    uw.data.qvel[:] = 0.0
    uw.data.qacc[:] = 0.0
    mujoco.mj_forward(uw.model, uw.data)

    uw.grasp_mode = False
    uw.finger_target_joint = uw.FINGER_OPEN_JOINT
    target_quat = uw.data.mocap_quat[0].copy()
    ok = uw._move_mocap_to(target, target_quat, max_steps=250, tolerance=0.003)

    if env.unwrapped.render_mode == "human" and delay > 0:
        time.sleep(delay)

    grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip").copy()
    err_mm = float(np.linalg.norm(grip_pos - target) * 1000)

    if not ok:
        return {"success": False, "reason": f"SCRIPTED_HOVER_FAILED ({err_mm:.1f}mm)"}

    uw.current_scenario = "descend"
    uw.goal_pos = target.copy()
    uw.goal = target.copy()
    uw.tube_center_xy = xy.copy()
    uw.episode_steps = 0

    if debug:
        print(f"  [DESCEND RECOVERY] scripted hover reached, err={err_mm:.1f}mm")

    return {"success": True, "error_mm": err_mm, "obs": uw._get_obs()}


def run_one_pick_sequence(env, model, home_pos, debug=False, delay=0.0, drift_limit=0.010):
    """
    Runs the full pick sequence for one episode.
    Returns a dict with per-step results and grasp quality metrics.
    """
    uw = env.unwrapped
    results = {}

    # Choose src_xy for this episode
    src_xy = uw._sample_board_position()[:2]

    # ── Step 1: Initialize ─────────────────────────────────────────────────────
    uw.force_start_pos = home_pos.copy()
    uw.force_cube_pos  = np.array([src_xy[0], src_xy[1], TABLE_Z + CUBE_H / 2])
    uw.force_scenario  = "transit"
    uw.hide_object     = False
    obs, _ = env.reset()
    
    # Clear overrides immediately after reset so they don't persist into future episodes
    uw.force_start_pos = None
    uw.force_cube_pos  = None
    
    # Apply drift limit
    uw.force_drift_limit = drift_limit

    if debug:
        print(f"\n[Episode] src_xy=({src_xy[0]*1000:.1f}, {src_xy[1]*1000:.1f})mm")

    # ── Step 2: Transit (home → src_xy at SAFE_Z) ─────────────────────────────
    transit_res = run_scenario_loop(env, model, obs, label="TRANSIT", delay=delay)
    results["transit_to_src"] = transit_res
    if transit_res["outcome"] != "success":
        return results

    # ── Transition: transit → descend ─────────────────────────────────────────
    obs, trans_info = uw.soft_reset(
        new_scenario="descend",
        new_goal_pos=np.array([src_xy[0], src_xy[1], uw.HOVER_Z]),
        nominal_exit_pos=np.array([src_xy[0], src_xy[1], SAFE_Z]),
        nominal_xy=src_xy,
    )
    reset_episode_timelimit(env)

    # ── Step 3: Descend (SAFE_Z -> HOVER_Z at src_xy) ────────────────────────
    descend_res = run_scenario_loop(env, model, obs, label="DESCEND", delay=delay)
    results["descend"] = descend_res
    if descend_res["outcome"] != "success":
        recovery = scripted_recover_to_hover(env, src_xy, debug=debug, delay=delay)
        results["descend_recovery"] = recovery
        if not recovery["success"]:
            return results
        descend_res["original_outcome"] = descend_res["outcome"]
        descend_res["original_crash_reason"] = descend_res.get("crash_reason")
        descend_res["outcome"] = "success"
        descend_res["recovered"] = True
        obs = recovery["obs"]

    # ── Step 4: GRASP (scripted — no soft_reset needed) ───────────────────────
    grasp_result = uw.execute_grasp()
    results["grasp"] = grasp_result
    if not grasp_result["success"]:
        if debug:
            print(f"  [GRASP FAILED] {grasp_result['reason']}")
        return results

    # ── Transition: GRASP → ascend (grasp_mode=True preserved) ───────────────
    obs, trans_info = uw.soft_reset(
        new_scenario="ascend",
        new_goal_pos=np.array([src_xy[0], src_xy[1], SAFE_Z]),
        nominal_exit_pos=np.array([src_xy[0], src_xy[1], uw.HOVER_Z]),
        nominal_xy=src_xy,
    )
    reset_episode_timelimit(env)
    assert uw.grasp_mode, "grasp_mode was reset during soft_reset"

    # ── Step 5: Ascend (HOVER_Z -> SAFE_Z, cube held) ────────────────────────
    ascend_res = run_scenario_loop(env, model, obs, label="ASCEND", delay=delay)
    results["ascend"] = ascend_res
    if ascend_res["outcome"] != "success":
        return results

    # ── Transition: ascend → transit (toward home) ────────────────────────────
    obs, trans_info = uw.soft_reset(
        new_scenario="transit",
        new_goal_pos=np.array([home_pos[0], home_pos[1], SAFE_Z]),
        nominal_exit_pos=np.array([src_xy[0], src_xy[1], SAFE_Z]),
        nominal_xy=src_xy,
    )
    reset_episode_timelimit(env)

    # ── Step 6: Transit to Home (carrying cube) ───────────────────────────────
    transit_home_res = run_scenario_loop(env, model, obs, label="TRANSIT_HOME", delay=delay)
    results["transit_to_home"] = transit_home_res

    # ── Evaluate grasp quality ─────────────────────────────────────────────────
    if transit_home_res["outcome"] == "success":
        results["grasp_quality"] = evaluate_grasp_quality(uw, src_xy)

    return results


def evaluate_grasp_quality(uw, src_xy: np.ndarray) -> dict:
    """Measures cube position/orientation drift at the end of the sequence."""
    cube_pos  = uw.get_cube_position()
    grip_pos  = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
    cube_quat = uw.get_cube_quat()

    # Measure drift relative to gripper (how much it moved INSIDE the fingers)
    xy_drift  = float(np.linalg.norm(cube_pos[:2] - grip_pos[:2])) * 1000
    # Cube hangs ~15mm below grip site while held in mid-air
    z_error   = float(abs(cube_pos[2] - (grip_pos[2] - 0.015))) * 1000

    w, x, y, z = cube_quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.degrees(math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    euler_deg = np.array([roll, pitch, yaw])

    return {
        "cube_xy_drift_mm": xy_drift,
        "cube_z_error_mm": z_error,
        "cube_max_rotation_deg": float(np.max(np.abs(euler_deg))),
        "cube_euler_xyz_deg": euler_deg.tolist(),
    }


def print_summary(all_results: list):
    """Print the standardised EVAL_GRASP report."""
    n = len(all_results)
    keys = ["transit_to_src", "descend", "grasp", "ascend", "transit_to_home"]
    print(f"\n=== EVAL_GRASP RESULTS ({n} episodes) ===\n")
    print("Step success rates:")
    for k in keys:
        success = sum(
            1 for r in all_results
            if k in r and (r[k].get("outcome") == "success" or r[k].get("success"))
        )
        print(f"  {k:<22}: {success}/{n} ({100*success//n}%)")

    quality_episodes = [r["grasp_quality"] for r in all_results if "grasp_quality" in r]
    if quality_episodes:
        print(f"\nGrasp quality (successful episodes, n={len(quality_episodes)}):")
        for metric in ["cube_xy_drift_mm", "cube_z_error_mm", "cube_max_rotation_deg"]:
            vals = [q[metric] for q in quality_episodes]
            print(f"  {metric:<30}: mean={np.mean(vals):.1f}  max={np.max(vals):.1f}")

    recoveries = sum(1 for r in all_results if r.get("descend", {}).get("recovered"))
    if recoveries:
        print(f"\nScripted descend recoveries: {recoveries}/{n}")
            
    print("\nFailure breakdown:")
    failures = {}
    for r in all_results:
        # Find the first step that didn't succeed
        for k in keys:
            if k not in r: continue
            res = r[k]
            if res.get("outcome") != "success" and not res.get("success"):
                reason = res.get("crash_reason") or res.get("reason") or "TIMEOUT"
                key = f"{k.upper()}_{reason}"
                failures[key] = failures.get(key, 0) + 1
                break
    for k, v in failures.items():
        print(f"  {k:<40}: {v}")
        
    print("\n=== DONE ===")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True, help="Path to SAC model .zip")
    p.add_argument("--n-episodes",  type=int, default=50)
    p.add_argument("--drift-limit", type=float, default=0.010,
                   help="Radial limit for drift during evaluation")
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--visualize",   action="store_true")
    p.add_argument("--delay",       type=float, default=0.0)
    p.add_argument("--wait",        action="store_true")
    args = p.parse_args()

    render_mode = "human" if args.visualize else None
    env = gym.make("ChessFetchTask-v0", render_mode=render_mode, hide_object=False, debug=args.debug)
    model = SAC.load(args.model, env=env)

    uw = env.unwrapped
    # Verify GRASP_Z
    assert abs(uw.GRASP_Z - 0.425) < 0.001
    assert abs(uw.HOVER_Z - HOVER_Z) < 0.001

    home_xy  = np.array(uw.env_cfg.get("home_position_xy", [0.680, 0.2641]))
    home_pos = np.array([home_xy[0], home_xy[1], SAFE_Z])

    all_results = []
    for ep in range(args.n_episodes):
        print(f"Running episode {ep+1}/{args.n_episodes}...")
        result = run_one_pick_sequence(env, model, home_pos, debug=args.debug,
                                       delay=args.delay, drift_limit=args.drift_limit)
        all_results.append(result)
        if args.wait:
            input(f"  Episode {ep+1}/{args.n_episodes} done. Press Enter...")

    print_summary(all_results)
    env.close()


if __name__ == "__main__":
    main()
