#!/usr/bin/env python3
"""
Evaluates model performance on multi-scenario chains.
Usage: python scripts/eval_sequence.py --model models/latest_model.zip --chain pick --n-episodes 100
"""

import argparse
import logging
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import src.chess_env
from src.chess_env.waypoints import (
    validate_chain, derive_goal_pos, exit_waypoint, CHAIN_SHORTCUTS,
    SCENARIO_EXIT_Z
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

class ChainEpisodeRunner:
    """
    Runs a sequence of RL scenarios back-to-back on a single environment instance.
    Handles halt gating, waypoint alignment, soft resets, and per-scenario metrics.
    """
    def __init__(self, env, model, chain, drift_limit, debug=False,
                 delay=0.0, wait=False):
        self.env = env
        self.model = model
        self.chain = chain
        self.drift_limit = drift_limit
        self.debug = debug
        self.delay = delay
        self.wait = wait

    def run_one_chain(self, waypoints: list[np.ndarray]) -> list:
        """
        Executes one full chain episode.
        waypoints: List of nominal XY coordinates for each scenario.
        """
        uw = self.env.unwrapped
        results = []
        
        # Initial Reset
        obs, _ = self.env.reset()
        uw.force_drift_limit = self.drift_limit

        for i, scenario in enumerate(self.chain):
            cell_xy = waypoints[i]
            goal_pos = derive_goal_pos(scenario, cell_xy)
            
            # Setup for the current scenario
            if i == 0:
                uw.current_scenario = scenario
                uw.goal_pos = goal_pos.copy()
                uw.goal = goal_pos.copy()
                if scenario in {"descend", "ascend"}:
                    uw.tube_center_xy = cell_xy.copy()
                else:
                    uw.tube_center_xy = None
                obs = uw._get_obs()

            scen_result = {
                "scenario": scenario,
                "steps": 0,
                "reward": 0.0,
                "outcome": None,
                "crash_reason": None,
                "final_dist_mm": None,
                "final_speed_mm_s": None,
                "transition": None
            }

            if self.debug:
                log.info(f"\n[SCENARIO {i+1}/{len(self.chain)}: {scenario}]  goal={goal_pos}")

            done = False
            ep_reward = 0.0
            ep_steps = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)

                if self.env.unwrapped.render_mode == "human":
                    self.env.render()

                if self.delay > 0:
                    time.sleep(self.delay)

                if self.debug and (ep_steps % 5 == 0 or terminated or truncated):
                    grip_pos = obs["observation"][:3]
                    grip_vel = obs["observation"][20:23]
                    dist_mm = np.linalg.norm(grip_pos - goal_pos) * 1000
                    speed_mm = np.linalg.norm(grip_vel) * 1000
                    drift_str = ""
                    if uw.tube_center_xy is not None:
                        drift = np.linalg.norm(grip_pos[:2] - uw.tube_center_xy) * 1000
                        drift_str = f" drift={drift:5.1f}mm/{self.drift_limit*1000:2.0f}mm"
                    log.info(f"  step {ep_steps+1:3d}: pos=({grip_pos[0]:.3f},{grip_pos[1]:.3f},{grip_pos[2]:.3f})"
                             f" dist={dist_mm:6.1f}mm speed={speed_mm:5.1f}mm/s{drift_str}")

                ep_reward += reward
                ep_steps += 1
                done = terminated or truncated

            scen_result["steps"] = ep_steps
            scen_result["reward"] = ep_reward
            grip_pos = obs["observation"][:3]
            grip_vel = obs["observation"][20:23]
            scen_result["final_dist_mm"] = np.linalg.norm(grip_pos - goal_pos) * 1000
            scen_result["final_speed_mm_s"] = np.linalg.norm(grip_vel) * 1000

            if info.get("is_success"):
                scen_result["outcome"] = "success"
            elif info.get("crash_reason"):
                scen_result["outcome"] = "crash"
                scen_result["crash_reason"] = info["crash_reason"]
            else:
                scen_result["outcome"] = "timeout"

            if self.debug:
                log.info(f"  OUTCOME: {scen_result['outcome'].upper()}  "
                         f"steps={ep_steps}  reward={ep_reward:.1f}  "
                         f"dist={scen_result['final_dist_mm']:.1f}mm  "
                         f"speed={scen_result['final_speed_mm_s']:.1f}mm/s")

            results.append(scen_result)

            if scen_result["outcome"] != "success":
                break

            if i == len(self.chain) - 1:
                break

            # ── TRANSITION ───────────────────────────────────────────────────────
            nom_exit = exit_waypoint(scenario, cell_xy)
            pre_diag = uw.transition_validate(nominal_exit_pos=nom_exit)
            
            next_scenario = self.chain[i + 1]
            next_cell_xy = waypoints[i + 1]
            next_goal = derive_goal_pos(next_scenario, next_cell_xy)

            try:
                obs, trans_info = uw.soft_reset(
                    new_scenario=next_scenario,
                    new_goal_pos=next_goal,
                    nominal_exit_pos=nom_exit,
                    nominal_xy=next_cell_xy if next_scenario in {"descend", "ascend"} else None,
                )
                
                # Reset TimeLimit wrapper
                curr = self.env
                while hasattr(curr, "env"):
                    if hasattr(curr, "_elapsed_steps"):
                        curr._elapsed_steps = 0
                        break
                    curr = curr.env
                
                post_grip = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
                post_err = np.linalg.norm(post_grip - nom_exit) * 1000
                
                trans_diag = {
                    "halt_speed": pre_diag["grip_speed_mm_s"],
                    "halt_steps": trans_info["halt_steps"],
                    "align_steps": trans_info["align_steps"],
                    "align_error": post_err,
                }
                scen_result["transition"] = trans_diag
                
                if self.debug:
                    log.info(f"  [TRANSITION {i+1}\u2192{i+2}]")
                    log.info(f"    HALT:  hold_steps={trans_info['halt_steps']}  final_speed={pre_diag['grip_speed_mm_s']:.2f}mm/s \u2713")
                    log.info(f"    ALIGN: settle_steps={trans_info['align_steps']}  final_error={post_err:.1f}mm \u2713")

            except Exception as e:
                log.error(f"  TRANSITION FAILED: {e}")
                scen_result["outcome"] = "transition_failed"
                break

        if self.wait:
            input("\nChain episode complete. Press Enter to continue...")

        return results

def print_summary(chain, all_results, drift_limit):
    n_episodes = len(all_results)
    if n_episodes == 0: return

    log.info("\n" + "="*80)
    log.info(f"Chain: {' \u2192 '.join(chain)}  ({n_episodes} episodes)")
    log.info(f"Drift limit: {drift_limit*1000:.0f}mm")
    log.info("-"*80)

    # Scenario-level stats
    scen_stats = {s: {"success": 0, "dist": [], "speed": [], "steps": []} for s in chain}
    chain_successes = 0
    failed_at = [0] * (len(chain) + 1)
    crash_reasons = {}

    for res_list in all_results:
        if len(res_list) == len(chain) and all(r["outcome"] == "success" for r in res_list):
            chain_successes += 1
        
        failed_idx = len(res_list)
        if res_list[-1]["outcome"] != "success":
            failed_at[failed_idx] += 1
            reason = res_list[-1].get("crash_reason") or res_list[-1]["outcome"].upper()
            crash_reasons[reason] = crash_reasons.get(reason, 0) + 1

        for r in res_list:
            s = r["scenario"]
            if r["outcome"] == "success":
                scen_stats[s]["success"] += 1
                scen_stats[s]["dist"].append(r["final_dist_mm"])
                scen_stats[s]["speed"].append(r["final_speed_mm_s"])
                scen_stats[s]["steps"].append(r["steps"])

    log.info("Scenario-level breakdown:")
    for s in chain:
        stats = scen_stats[s]
        if stats["success"] > 0:
            avg_dist = np.mean(stats["dist"])
            avg_speed = np.mean(stats["speed"])
            avg_steps = np.mean(stats["steps"])
            log.info(f"  {s:8s}: {stats['success']:3d}/{n_episodes:3d} succeeded | "
                     f"avg_dist={avg_dist:5.1f}mm | avg_speed={avg_speed:4.1f}mm/s | avg_steps={avg_steps:4.1f}")
        else:
            log.info(f"  {s:8s}:   0/{n_episodes:3d} succeeded")

    log.info("\nChain-level results:")
    log.info(f"  Full chain success rate: {chain_successes/n_episodes*100:5.1f}%   "
             f"({chain_successes}/{n_episodes} scenarios succeeded)")
    
    for i, s in enumerate(chain):
        count = failed_at[i+1]
        if count > 0:
            log.info(f"  Failed at {s:8s}: {count:3d} / {n_episodes:3d} ({count/n_episodes*100:5.1f}%)")

    if crash_reasons:
        reason_str = ", ".join([f"{k} x{v}" for k, v in crash_reasons.items()])
        log.info(f"\nFailure breakdown: {reason_str}")

    # Transition diagnostics
    trans_errors = []
    trans_halt_steps = []
    trans_align_steps = []
    for res_list in all_results:
        for r in res_list:
            if r.get("transition"):
                trans_errors.append(r["transition"]["align_error"])
                trans_halt_steps.append(r["transition"]["halt_steps"])
                trans_align_steps.append(r["transition"]["align_steps"])
    
    if trans_errors:
        log.info("\nTransition diagnostics (successful transitions only):")
        log.info(f"  avg halt_steps={np.mean(trans_halt_steps):.1f} | "
                 f"avg align_steps={np.mean(trans_align_steps):.1f} | "
                 f"avg align_error={np.mean(trans_errors):.2f}mm (max={np.max(trans_errors):.2f}mm)")
    
    log.info("="*80 + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True)
    p.add_argument("--chain",       required=True)
    p.add_argument("--n-episodes",  type=int, default=10)
    p.add_argument("--drift-limit", type=float, default=0.010)
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--visualize",   action="store_true")
    p.add_argument("--delay",       type=float, default=0.0)
    p.add_argument("--wait",        action="store_true")
    args = p.parse_args()

    chain = CHAIN_SHORTCUTS.get(args.chain, args.chain.split(","))
    validate_chain(chain)

    render_mode = "human" if args.visualize else None
    env = gym.make("ChessFetchTask-v0", render_mode=render_mode, force_scenario=chain[0])
    model = SAC.load(args.model, env=env)
    
    runner = ChainEpisodeRunner(
        env, model, chain, args.drift_limit, 
        debug=args.debug, delay=args.delay, wait=args.wait
    )

    all_results = []
    for ep in range(args.n_episodes):
        if not args.debug:
            print(f"Running episode {ep+1}/{args.n_episodes}...", end="\r", flush=True)
        
        # Pre-sample waypoints
        src_xy = env.unwrapped._sample_board_position()[:2]
        dst_xy = env.unwrapped._sample_board_position()[:2]
        while np.linalg.norm(dst_xy - src_xy) < 0.10: # MIN_GOAL_DIST
            dst_xy = env.unwrapped._sample_board_position()[:2]
            
        # Strategy for waypoints in chain:
        # We need to know which cell to target for each scenario step.
        waypoints = []
        
        if args.chain == "full_move":
            # src, src, src, dst, dst, dst
            waypoints = [src_xy]*3 + [dst_xy]*3
        elif args.chain == "pick":
            # src, src, src
            waypoints = [src_xy]*3
        elif args.chain == "place":
            # dst, dst, dst
            waypoints = [dst_xy]*3
        elif args.chain == "vertical":
            # dst, dst
            waypoints = [dst_xy]*2
        else:
            # Generic logic: new transit = new cell, then hold that cell for descend/ascend
            current_target = src_xy
            pts = [dst_xy]
            # Add some extra points just in case it's a long custom chain
            for _ in range(len(chain)):
                pts.append(env.unwrapped._sample_board_position()[:2])
            
            pt_idx = 0
            for s in chain:
                if s == "transit":
                    current_target = pts[pt_idx]
                    pt_idx += 1
                waypoints.append(current_target)

        res = runner.run_one_chain(waypoints)
        all_results.append(res)

    print("\nDone.")
    print_summary(chain, all_results, args.drift_limit)
    env.close()

if __name__ == "__main__":
    main()
