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
import os
import sys
import time
import numpy as np
import gymnasium as gym
import mujoco

sys.path.append(os.getcwd())

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


def select_model(models: dict, scenario: str):
    """Return the scenario-specific model when one is provided."""
    return models.get(scenario) or models["default"]


def scripted_recover_to_hover(env, xy: np.ndarray, holding_cube=False, debug=False, delay=0.0) -> dict:
    """
    Recovery path for the old descend policy. If RL fails to stop at HOVER_Z,
    glide there through small physics-simulated mocap targets.
    """
    uw = env.unwrapped
    target = np.array([xy[0], xy[1], uw.HOVER_Z])
    zero_action = np.zeros(4)
    max_step_m = 0.002
    max_steps = 450
    tolerance = 0.003

    uw.data.qvel[:] = 0.0
    uw.data.qacc[:] = 0.0
    mujoco.mj_forward(uw.model, uw.data)

    uw.grasp_mode = bool(holding_cube)
    uw.finger_target_joint = uw.FINGER_CLOSED_JOINT if holding_cube else uw.FINGER_OPEN_JOINT
    target_quat = uw.data.mocap_quat[0].copy()

    ok = False
    steps_used = 0
    for step in range(max_steps):
        grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip").copy()
        error = target - grip_pos
        err_norm = float(np.linalg.norm(error))
        if err_norm < tolerance:
            ok = True
            break

        step_vec = error * 0.35
        step_norm = float(np.linalg.norm(step_vec))
        if step_norm > max_step_m:
            step_vec = step_vec / step_norm * max_step_m

        next_target = grip_pos + step_vec
        uw._set_action(zero_action)
        uw._assert_mocap_target(next_target, target_quat)
        uw._mujoco_step(None)

        if env.unwrapped.render_mode == "human":
            env.render()
            if delay > 0:
                time.sleep(delay)
        steps_used = step + 1

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
        print(f"  [DESCEND RECOVERY] smooth hover reached, err={err_mm:.1f}mm, steps={steps_used}")

    return {"success": True, "error_mm": err_mm, "steps": steps_used, "obs": uw._get_obs()}


def sample_distinct_xy(uw, src_xy: np.ndarray) -> np.ndarray:
    """Sample a destination far enough from the source to be a meaningful move."""
    dst_xy = uw._sample_board_position()[:2]
    for _ in range(100):
        if np.linalg.norm(dst_xy - src_xy) >= uw.MIN_GOAL_DIST:
            return dst_xy
        dst_xy = uw._sample_board_position()[:2]
    return dst_xy


def run_one_pick_sequence(env, models, home_pos, debug=False, delay=0.0,
                          drift_limit=0.010, allow_recovery=False):
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
    transit_res = run_scenario_loop(env, select_model(models, "transit"), obs, label="TRANSIT", delay=delay)
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
    descend_res = run_scenario_loop(env, select_model(models, "descend"), obs, label="DESCEND", delay=delay)
    results["descend"] = descend_res
    if descend_res["outcome"] != "success":
        if not allow_recovery:
            return results
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
    ascend_res = run_scenario_loop(env, select_model(models, "ascend"), obs, label="ASCEND", delay=delay)
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
    transit_home_res = run_scenario_loop(env, select_model(models, "transit"), obs, label="TRANSIT_HOME", delay=delay)
    results["transit_to_home"] = transit_home_res

    # ── Evaluate grasp quality ─────────────────────────────────────────────────
    if transit_home_res["outcome"] == "success":
        results["grasp_quality"] = evaluate_grasp_quality(uw, src_xy)

    return results


def run_one_pick_place_sequence(env, models, home_pos, debug=False, delay=0.0,
                                drift_limit=0.010, allow_recovery=False):
    """
    Runs a full pick-and-place episode:
    home -> source -> grasp -> destination -> place -> ascend -> home.
    """
    uw = env.unwrapped
    results = {}

    src_xy = uw._sample_board_position()[:2]
    dst_xy = sample_distinct_xy(uw, src_xy)
    results["src_xy"] = src_xy.copy()
    results["dst_xy"] = dst_xy.copy()

    uw.force_start_pos = home_pos.copy()
    uw.force_cube_pos = np.array([src_xy[0], src_xy[1], TABLE_Z + CUBE_H / 2])
    uw.force_scenario = "transit"
    uw.hide_object = False
    obs, _ = env.reset()

    uw.force_start_pos = None
    uw.force_cube_pos = None
    uw.force_drift_limit = drift_limit

    if debug:
        print(
            f"\n[Episode] src_xy=({src_xy[0]*1000:.1f}, {src_xy[1]*1000:.1f})mm "
            f"dst_xy=({dst_xy[0]*1000:.1f}, {dst_xy[1]*1000:.1f})mm"
        )

    transit_res = run_scenario_loop(env, select_model(models, "transit"), obs, label="TRANSIT_SRC", delay=delay)
    results["transit_to_src"] = transit_res
    if transit_res["outcome"] != "success":
        return results

    obs, _ = uw.soft_reset(
        new_scenario="descend",
        new_goal_pos=np.array([src_xy[0], src_xy[1], uw.HOVER_Z]),
        nominal_exit_pos=np.array([src_xy[0], src_xy[1], SAFE_Z]),
        nominal_xy=src_xy,
    )
    reset_episode_timelimit(env)

    descend_res = run_scenario_loop(env, select_model(models, "descend"), obs, label="DESCEND_SRC", delay=delay)
    results["descend_src"] = descend_res
    if descend_res["outcome"] != "success":
        if not allow_recovery:
            return results
        recovery = scripted_recover_to_hover(env, src_xy, holding_cube=False, debug=debug, delay=delay)
        results["descend_src_recovery"] = recovery
        if not recovery["success"]:
            return results
        descend_res["original_outcome"] = descend_res["outcome"]
        descend_res["original_crash_reason"] = descend_res.get("crash_reason")
        descend_res["outcome"] = "success"
        descend_res["recovered"] = True
        obs = recovery["obs"]

    grasp_result = uw.execute_grasp()
    results["grasp"] = grasp_result
    if not grasp_result["success"]:
        if debug:
            print(f"  [GRASP FAILED] {grasp_result['reason']}")
        return results

    obs, _ = uw.soft_reset(
        new_scenario="ascend",
        new_goal_pos=np.array([src_xy[0], src_xy[1], SAFE_Z]),
        nominal_exit_pos=np.array([src_xy[0], src_xy[1], uw.HOVER_Z]),
        nominal_xy=src_xy,
    )
    reset_episode_timelimit(env)

    ascend_src_res = run_scenario_loop(env, select_model(models, "ascend"), obs, label="ASCEND_SRC", delay=delay)
    results["ascend_src"] = ascend_src_res
    if ascend_src_res["outcome"] != "success":
        return results

    obs, _ = uw.soft_reset(
        new_scenario="transit",
        new_goal_pos=np.array([dst_xy[0], dst_xy[1], SAFE_Z]),
        nominal_exit_pos=np.array([src_xy[0], src_xy[1], SAFE_Z]),
        nominal_xy=src_xy,
    )
    reset_episode_timelimit(env)

    transit_dst_res = run_scenario_loop(env, select_model(models, "transit"), obs, label="TRANSIT_DST", delay=delay)
    results["transit_to_dst"] = transit_dst_res
    if transit_dst_res["outcome"] != "success":
        return results

    obs, _ = uw.soft_reset(
        new_scenario="descend",
        new_goal_pos=np.array([dst_xy[0], dst_xy[1], uw.HOVER_Z]),
        nominal_exit_pos=np.array([dst_xy[0], dst_xy[1], SAFE_Z]),
        nominal_xy=dst_xy,
    )
    reset_episode_timelimit(env)

    descend_dst_res = run_scenario_loop(env, select_model(models, "descend"), obs, label="DESCEND_DST", delay=delay)
    results["descend_dst"] = descend_dst_res
    if descend_dst_res["outcome"] != "success":
        if not allow_recovery:
            return results
        recovery = scripted_recover_to_hover(env, dst_xy, holding_cube=True, debug=debug, delay=delay)
        results["descend_dst_recovery"] = recovery
        if not recovery["success"]:
            return results
        descend_dst_res["original_outcome"] = descend_dst_res["outcome"]
        descend_dst_res["original_crash_reason"] = descend_dst_res.get("crash_reason")
        descend_dst_res["outcome"] = "success"
        descend_dst_res["recovered"] = True
        obs = recovery["obs"]

    place_result = uw.execute_place(dst_xy)
    results["place"] = place_result
    if not place_result["success"]:
        if debug:
            print(f"  [PLACE FAILED] {place_result['reason']}")
        return results

    obs, _ = uw.soft_reset(
        new_scenario="ascend",
        new_goal_pos=np.array([dst_xy[0], dst_xy[1], SAFE_Z]),
        nominal_exit_pos=np.array([dst_xy[0], dst_xy[1], uw.HOVER_Z]),
        nominal_xy=dst_xy,
    )
    reset_episode_timelimit(env)

    ascend_dst_res = run_scenario_loop(env, select_model(models, "ascend"), obs, label="ASCEND_DST", delay=delay)
    results["ascend_dst"] = ascend_dst_res
    if ascend_dst_res["outcome"] != "success":
        return results

    obs, _ = uw.soft_reset(
        new_scenario="transit",
        new_goal_pos=np.array([home_pos[0], home_pos[1], SAFE_Z]),
        nominal_exit_pos=np.array([dst_xy[0], dst_xy[1], SAFE_Z]),
        nominal_xy=dst_xy,
    )
    reset_episode_timelimit(env)

    transit_home_res = run_scenario_loop(env, select_model(models, "transit"), obs, label="TRANSIT_HOME", delay=delay)
    results["transit_to_home"] = transit_home_res
    if transit_home_res["outcome"] == "success":
        results["placement_quality"] = evaluate_place_quality(uw, dst_xy)

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


def evaluate_place_quality(uw, dst_xy: np.ndarray) -> dict:
    """Measures final cube placement after release."""
    cube_pos = uw.get_cube_position()
    xy_error = float(np.linalg.norm(cube_pos[:2] - dst_xy[:2])) * 1000
    z_error = float(abs(cube_pos[2] - (TABLE_Z + CUBE_H / 2))) * 1000
    return {
        "place_xy_error_mm": xy_error,
        "place_z_error_mm": z_error,
        "final_cube_pos": cube_pos.tolist(),
    }


def print_summary(all_results: list, has_place: bool = False):
    """Print the standardised EVAL_GRASP report."""
    n = len(all_results)
    if has_place:
        keys = [
            "transit_to_src",
            "descend_src",
            "grasp",
            "ascend_src",
            "transit_to_dst",
            "descend_dst",
            "place",
            "ascend_dst",
            "transit_to_home",
        ]
    else:
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

    placement_episodes = [r["placement_quality"] for r in all_results if "placement_quality" in r]
    if placement_episodes:
        print(f"\nPlacement quality (successful episodes, n={len(placement_episodes)}):")
        for metric in ["place_xy_error_mm", "place_z_error_mm"]:
            vals = [q[metric] for q in placement_episodes]
            print(f"  {metric:<30}: mean={np.mean(vals):.1f}  max={np.max(vals):.1f}")

    recoveries = sum(
        1
        for r in all_results
        for k in ("descend", "descend_src", "descend_dst")
        if r.get(k, {}).get("recovered")
    )
    if recoveries:
        print(f"\nScripted descend recoveries: {recoveries}")
            
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
    p.add_argument("--transit-model", type=str,
                   help="Optional SAC model to use for transit scenarios")
    p.add_argument("--descend-model", type=str,
                   help="Optional SAC model to use for descend scenarios")
    p.add_argument("--ascend-model", type=str,
                   help="Optional SAC model to use for ascend scenarios")
    p.add_argument("--n-episodes",  type=int, default=50)
    p.add_argument("--drift-limit", type=float, default=0.010,
                   help="Radial limit for drift during evaluation")
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--visualize",   action="store_true")
    p.add_argument("--delay",       type=float, default=0.0)
    p.add_argument("--wait",        action="store_true")
    p.add_argument("--place",       action="store_true",
                   help="Run full pick-and-place instead of pick-and-carry-to-home")
    p.add_argument("--allow-recovery", action="store_true",
                   help="After RL descend fails, use scripted hover recovery. Off by default for plan-faithful evaluation.")
    args = p.parse_args()

    render_mode = "human" if args.visualize else None
    env = gym.make("ChessFetchTask-v0", render_mode=render_mode, hide_object=False, debug=args.debug)
    models = {"default": SAC.load(args.model, env=env)}
    if args.transit_model:
        models["transit"] = SAC.load(args.transit_model, env=env)
    if args.descend_model:
        models["descend"] = SAC.load(args.descend_model, env=env)
    if args.ascend_model:
        models["ascend"] = SAC.load(args.ascend_model, env=env)

    uw = env.unwrapped
    uw.transition_render_delay = args.delay
    # Verify GRASP_Z
    assert abs(uw.GRASP_Z - 0.425) < 0.001
    assert abs(uw.HOVER_Z - HOVER_Z) < 0.001

    home_xy  = np.array(uw.env_cfg.get("home_position_xy", [0.680, 0.2641]))
    home_pos = np.array([home_xy[0], home_xy[1], SAFE_Z])

    all_results = []
    for ep in range(args.n_episodes):
        print(f"Running episode {ep+1}/{args.n_episodes}...")
        if args.place:
            result = run_one_pick_place_sequence(
                env, models, home_pos, debug=args.debug,
                delay=args.delay, drift_limit=args.drift_limit,
                allow_recovery=args.allow_recovery,
            )
        else:
            result = run_one_pick_sequence(
                env, models, home_pos, debug=args.debug,
                delay=args.delay, drift_limit=args.drift_limit,
                allow_recovery=args.allow_recovery,
            )
        all_results.append(result)
        if args.wait:
            input(f"  Episode {ep+1}/{args.n_episodes} done. Press Enter...")

    print_summary(all_results, has_place=args.place)
    env.close()


if __name__ == "__main__":
    main()
