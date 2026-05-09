# Implementation Plan: Scenario Chaining

## Guiding Principles

1. **Evaluation only** — no changes to training code, rewards, or env config
2. **Minimal env changes** — add two new methods (`soft_reset`, `transition_validate`); don't modify existing step/reset logic
3. **Test incrementally** — transit-only chains first, then add descend/ascend when V6 model ready
4. **Instrument everything** — every transition logs halt steps, alignment error, velocity, and finger state
5. **Nominal positions always** — tube_center = nominal cell XY, never the arm's actual landed position

---

## Files Overview

| File | Action | Description |
|:---|:---|:---|
| `scripts/eval_sequence.py` | **Create** | Main chaining evaluation script |
| `src/chess_env/task.py` | **Add methods** | `soft_reset()` + `transition_validate()` |
| `src/chess_env/waypoints.py` | **Create** | Goal position derivation utilities |
| `configs/env.yaml` | **No change** | All needed constants already exist |

---

## Step 1: Add `soft_reset` to `ChessTaskEnv` (task.py)

The `soft_reset` method orchestrates the full four-phase transition:
1. Complete halt (hold actions + active arm-joint zeroing)
2. Waypoint alignment (scripted settle to exact nominal position)
3. Gripper state transition (scripted open/close if needed)
4. State update (scenario, goal, tube_center = nominal_xy)

```python
import mujoco

# Fetch MuJoCo DOF layout (verified dynamically):
#   qvel[ 0: 3] slide(3), qvel[3] torso(1), qvel[4:6] head(2),
#   qvel[6:13] arm(7), qvel[13:15] fingers(2), qvel[15+] object0:joint
ROBOT_DOF = 15               # slide(3)+torso(1)+head(2)+arm(7)+fingers(2)
HALT_VEL_THRESHOLD = 0.0005  # 0.5mm/s
HALT_HOLD_MAX_STEPS = 100
ALIGN_TOLERANCE_M = 0.003    # 3mm
ALIGN_MAX_STEPS = 200
ALIGN_GAIN = 0.8
ALIGN_MAX_STEP_M = 0.005     # 5mm per settle step


def soft_reset(self, new_scenario: str, new_goal_pos: np.ndarray,
               nominal_xy: np.ndarray | None = None) -> dict:
    """
    Transitions to a new scenario without teleporting the arm.
    Executes: complete halt → waypoint alignment → gripper transition → state update.

    Args:
        new_scenario:  "transit", "descend", or "ascend"
        new_goal_pos:  np.array([x, y, z]) — the goal for the new scenario
        nominal_xy:    np.array([x, y]) — the nominal chess cell position.
                       Used as tube_center for descend/ascend. Must be provided
                       for descend/ascend scenarios.

    Returns:
        obs: Fresh observation dict for the new scenario.
    """
    # ── Phase 1: Complete Halt ──────────────────────────────────────────────

    # Step 1a: Issue hold actions until speed drops
    zero_action = np.zeros(4)
    for hold_step in range(HALT_HOLD_MAX_STEPS):
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        speed = float(np.linalg.norm(grip_vel))
        if speed < HALT_VEL_THRESHOLD:
            break
        self._set_action(zero_action)
        self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # Step 1b: Actively zero robot joint velocities (robot DOF only — NOT object joint)
    self.data.qvel[:ROBOT_DOF] = 0.0
    self.data.qacc[:ROBOT_DOF] = 0.0
    mujoco.mj_forward(self.model, self.data)

    # Step 1c: Confirm halt
    grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
    speed = float(np.linalg.norm(grip_vel))
    if speed >= HALT_VEL_THRESHOLD:
        raise RuntimeError(
            f"soft_reset HALT_FAILED: speed={speed*1000:.3f}mm/s >= "
            f"threshold={HALT_VEL_THRESHOLD*1000:.1f}mm/s"
        )

    # ── Phase 2: Waypoint Alignment ─────────────────────────────────────────

    # The exit waypoint of the current scenario (before transitioning state)
    # WARNING: Do NOT use new_goal_pos! If transitioning from transit to descend, 
    # new_goal_pos is at GRASP_Z. We must align to the top of the tube (SAFE_Z) first.
    align_target = nominal_exit_pos.copy()  # Passed from the eval script!

    converged = False
    for align_step in range(ALIGN_MAX_STEPS):
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        error = align_target - grip_pos
        dist = float(np.linalg.norm(error))
        if dist < ALIGN_TOLERANCE_M:
            # Zero robot velocities after settling — robot DOF only, not object
            self.data.qvel[:ROBOT_DOF] = 0.0
            self.data.qacc[:ROBOT_DOF] = 0.0
            mujoco.mj_forward(self.model, self.data)
            converged = True
            break
        step_vec = ALIGN_GAIN * error
        if np.linalg.norm(step_vec) > ALIGN_MAX_STEP_M:
            step_vec = step_vec / np.linalg.norm(step_vec) * ALIGN_MAX_STEP_M
        self.data.mocap_pos[0][:3] += step_vec
        self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    if not converged:
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        dist_mm = float(np.linalg.norm(align_target - grip_pos)) * 1000
        raise RuntimeError(
            f"soft_reset ALIGN_FAILED: did not converge after {ALIGN_MAX_STEPS} steps. "
            f"final_error={dist_mm:.1f}mm, target={align_target}"
        )

    # ── Phase 3: State Update ───────────────────────────────────────────────

    prev_scenario = self.current_scenario
    self.current_scenario = new_scenario
    self.goal_pos = new_goal_pos.copy()
    self.goal = self.goal_pos.copy()  # Critical: _build_phase9_observation() reads self.goal,
                                      # not self.goal_pos. Without this, the model receives
                                      # an obs that still points to the previous scenario's goal.
    self.episode_steps = 0
    
    # Critical: Reset the Gymnasium TimeLimit wrapper so the chain doesn't timeout
    curr_env = self
    while hasattr(curr_env, "env"):
        if hasattr(curr_env, "_elapsed_steps"):
            curr_env._elapsed_steps = 0
        curr_env = curr_env.env

    if new_scenario in {"descend", "ascend"}:
        if nominal_xy is None:
            raise ValueError(
                f"soft_reset: nominal_xy is required for scenario '{new_scenario}'. "
                f"Pass the chess cell XY position (not the arm's actual position)."
            )
        self.tube_center_xy = nominal_xy.copy()  # NOMINAL, not grip_pos[:2]
    else:
        self.tube_center_xy = None

    # Re-enforce vertical orientation
    self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT)

    # ── Phase 4: Gripper Transition ─────────────────────────────────────────

    dummy_action = np.zeros(4)

    needs_open = (new_scenario == "descend" and prev_scenario != "descend")
    needs_close = (new_scenario in {"ascend", "transit"} and prev_scenario == "descend")

    if needs_open:
        self.finger_target_joint = self.FINGER_OPEN_JOINT
        for _ in range(50):
            self._set_action(dummy_action)
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
    elif needs_close:
        self.finger_target_joint = self.FINGER_CLOSED_JOINT
        for _ in range(50):
            self._set_action(dummy_action)
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # Finger validation (same as _reset_sim Phase 3)
    l_pos = self._utils.get_joint_qpos(
        self.model, self.data, "robot0:l_gripper_finger_joint"
    ).item()
    if abs(l_pos - self.finger_target_joint) > 0.0005:
        raise RuntimeError(
            f"soft_reset FINGER_VALIDATION_FAILED: "
            f"actual={l_pos:.6f}, target={self.finger_target_joint:.6f}"
        )

    return self._get_obs()
```

---

## Step 2: Add `transition_validate` helper (task.py)

A diagnostic utility called before `soft_reset` to log arm state at transition time.

```python
def transition_validate(self, nominal_exit_pos: np.ndarray | None = None,
                        vel_threshold: float = HALT_VEL_THRESHOLD) -> dict:
    """
    Returns arm state diagnostics at a scenario transition point.

    Returns:
        {
            "grip_pos": [x, y, z],
            "grip_speed_mm_s": float,
            "is_velocity_ok": bool,
            "error_from_nominal_mm": float | None,
            "finger_state": float,
        }
    """
    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip").copy()
    speed = float(np.linalg.norm(grip_vel))
    l_finger = self._utils.get_joint_qpos(
        self.model, self.data, "robot0:l_gripper_finger_joint"
    ).item()

    result = {
        "grip_pos": grip_pos.tolist(),
        "grip_speed_mm_s": speed * 1000,
        "is_velocity_ok": speed < vel_threshold,
        "error_from_nominal_mm": None,
        "finger_state": l_finger,
    }
    if nominal_exit_pos is not None:
        result["error_from_nominal_mm"] = float(
            np.linalg.norm(grip_pos - nominal_exit_pos) * 1000
        )
    return result
```

---

## Step 3: Create `src/chess_env/waypoints.py`

A small utility module encoding valid scenario transitions and goal derivation rules.

```python
"""
Waypoint derivation for scenario chains.
"""

import numpy as np
from src.chess_env.config_loader import load_env_cfg

_cfg = load_env_cfg()
SAFE_Z  = _cfg["safe_z"]    # 0.550m
GRASP_Z = _cfg["grasp_z"]   # 0.430m

SCENARIO_EXIT_Z = {
    "transit": SAFE_Z,
    "descend": GRASP_Z,
    "ascend":  SAFE_Z,
}
SCENARIO_ENTRY_Z = {
    "transit": SAFE_Z,
    "descend": SAFE_Z,
    "ascend":  GRASP_Z,
}

VALID_TRANSITIONS = {
    ("transit", "transit"): True,
    ("transit", "descend"): True,
    ("descend", "ascend"):  True,
    ("ascend",  "transit"): True,
    ("ascend",  "descend"): True,
    ("descend", "transit"): False,  # GRASP_Z → SAFE_Z: wrong height
    ("transit", "ascend"):  False,  # SAFE_Z → GRASP_Z: arm not at grasp height
    ("descend", "descend"): False,  # Semantically invalid
    ("ascend",  "ascend"):  False,  # Semantically invalid
}

CHAIN_SHORTCUTS = {
    "full_move": ["transit", "descend", "ascend", "transit", "descend", "ascend"],
    "pick":      ["transit", "descend", "ascend"],
    "place":     ["transit", "descend", "ascend"],
    "vertical":  ["descend", "ascend"],
}


def validate_chain(chain: list[str]) -> None:
    """Raises ValueError if any transition in the chain is physically invalid."""
    for i in range(len(chain) - 1):
        key = (chain[i], chain[i + 1])
        if not VALID_TRANSITIONS.get(key, False):
            raise ValueError(
                f"Invalid chain transition at position {i}: "
                f"{chain[i]} → {chain[i+1]}. "
                f"Exit Z of '{chain[i]}' = {SCENARIO_EXIT_Z[chain[i]]*100:.1f}cm, "
                f"but '{chain[i+1]}' requires entry Z = "
                f"{SCENARIO_ENTRY_Z[chain[i+1]]*100:.1f}cm."
            )


def derive_goal_pos(scenario: str, cell_xy: np.ndarray) -> np.ndarray:
    """
    Returns the goal_pos for a scenario at the given chess cell XY.
    Always uses nominal cell coordinates.
    """
    if scenario == "transit":
        return np.array([cell_xy[0], cell_xy[1], SAFE_Z])
    elif scenario == "descend":
        return np.array([cell_xy[0], cell_xy[1], GRASP_Z])
    elif scenario == "ascend":
        return np.array([cell_xy[0], cell_xy[1], SAFE_Z])
    else:
        raise ValueError(f"Unknown scenario: '{scenario}'")


def exit_waypoint(scenario: str, cell_xy: np.ndarray) -> np.ndarray:
    """Returns the nominal exit waypoint for a scenario (what to align to after success)."""
    z = SCENARIO_EXIT_Z[scenario]
    return np.array([cell_xy[0], cell_xy[1], z])
```

---

## Step 4: Create `scripts/eval_sequence.py`

The main evaluation script. Key structure:

```python
#!/usr/bin/env python3
"""
Evaluates model performance on multi-scenario chains.
Usage: python scripts/eval_sequence.py --model models/latest_model.zip --chain transit,transit
"""

import argparse
import logging
import time
import numpy as np
from stable_baselines3 import SAC
from src.chess_env.waypoints import validate_chain, derive_goal_pos, exit_waypoint, CHAIN_SHORTCUTS

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True)
    p.add_argument("--chain",       required=True)
    p.add_argument("--n-episodes",  type=int, default=100)
    p.add_argument("--drift-limit", type=float, default=0.010)  # 10mm — V6 curriculum standard
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--visualize",   action="store_true")
    p.add_argument("--delay",       type=float, default=0.0)
    p.add_argument("--wait",        action="store_true",
                   help="Pause after each episode (press Enter to continue)")
    return p.parse_args()


def wait_for_enter(prompt="Press Enter to continue..."):
    try:
        input(prompt)
    except EOFError:
        pass


def run_one_chain(env, model, chain, dst_xy_list, args):
    """
    Run one chain episode. Returns list of per-scenario result dicts.
    dst_xy_list: nominal chess cell XY for each scenario.
    """
    uw = env.unwrapped
    results = []

    obs, _ = env.reset()

    for i, scenario in enumerate(chain):
        cell_xy = dst_xy_list[i]
        goal_pos = derive_goal_pos(scenario, cell_xy)
        # Override env's goal with the correct one for this scenario
        uw.goal_pos = goal_pos.copy()
        if scenario in {"descend", "ascend"}:
            uw.tube_center_xy = cell_xy.copy()

        scen_result = {
            "scenario": scenario,
            "steps": 0,
            "reward": 0.0,
            "outcome": None,
            "crash_reason": None,
            "final_dist_mm": None,
            "final_speed_mm_s": None,
        }

        done = False
        ep_reward = 0.0
        ep_steps = 0

        if args.debug:
            log.info(f"[SCENARIO {i+1}/{len(chain)}: {scenario}] "
                     f"goal={goal_pos}  tube_center={getattr(uw, 'tube_center_xy', None)}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if args.delay > 0:
                time.sleep(args.delay)

            if args.debug and ep_steps % 5 == 0:
                grip_pos = obs["observation"][:3]
                grip_vel = obs["observation"][20:23]
                dist_mm = np.linalg.norm(grip_pos - goal_pos) * 1000
                speed_mm = np.linalg.norm(grip_vel) * 1000
                drift_str = ""
                if hasattr(uw, "tube_center_xy") and uw.tube_center_xy is not None:
                    drift = np.linalg.norm(grip_pos[:2] - uw.tube_center_xy) * 1000
                    drift_str = f" drift={drift:.1f}mm/{uw.eval_drift_limit*1000:.0f}mm"
                log.debug(f"  step {ep_steps+1:3d}: pos=({grip_pos[0]:.3f},{grip_pos[1]:.3f},{grip_pos[2]:.3f})"
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

        if args.debug:
            log.info(f"  OUTCOME: {scen_result['outcome'].upper()}  "
                     f"steps={ep_steps}  reward={ep_reward:.1f}  "
                     f"dist={scen_result['final_dist_mm']:.1f}mm  "
                     f"speed={scen_result['final_speed_mm_s']:.1f}mm/s")
            if scen_result["crash_reason"]:
                log.info(f"  CRASH_REASON: {scen_result['crash_reason']}")

        results.append(scen_result)

        if scen_result["outcome"] != "success":
            break

        if i == len(chain) - 1:
            break

        # ── TRANSITION ───────────────────────────────────────────────────────
        pre_diag = uw.transition_validate(nominal_exit_pos=exit_waypoint(scenario, cell_xy))
        if args.debug:
            log.info(f"  [PRE-TRANSITION diag] "
                     f"pos={[f'{v:.4f}' for v in pre_diag['grip_pos']]}  "
                     f"speed={pre_diag['grip_speed_mm_s']:.2f}mm/s  "
                     f"error_from_nominal={pre_diag['error_from_nominal_mm']:.1f}mm")

        next_scenario = chain[i + 1]
        next_cell_xy = dst_xy_list[i + 1]
        next_goal = derive_goal_pos(next_scenario, next_cell_xy)

        obs = uw.soft_reset(
            new_scenario=next_scenario,
            new_goal_pos=next_goal,
            nominal_xy=next_cell_xy if next_scenario in {"descend", "ascend"} else None,
        )

        if args.debug:
            post_grip = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
            post_err = np.linalg.norm(post_grip - exit_waypoint(next_scenario, next_cell_xy)) * 1000
            log.info(f"  [POST-TRANSITION] pos=({post_grip[0]:.4f},{post_grip[1]:.4f},{post_grip[2]:.4f})"
                     f"  align_error={post_err:.1f}mm  "
                     f"tube_center={getattr(uw, 'tube_center_xy', None)}")

    return results
```

---

## Step 5: Validation Tests

Run these in order before any 100-episode experiments.

### Test A: Chain validity rejection
```bash
python scripts/eval_sequence.py --model models/latest_model.zip --chain "transit,ascend" --n-episodes 1
# Expected: ValueError — "Invalid chain transition: transit → ascend"
```

### Test B: Transit-only chain (current model works for transit)
```bash
python scripts/eval_sequence.py --model models/latest_model.zip --chain "transit,transit" \
    --n-episodes 20 --debug
# Expected: ~100% success. Verify transition logs show halt + alignment working.
```

### Test C: Visualized transit chain for visual inspection
```bash
python scripts/eval_sequence.py --model models/latest_model.zip --chain "transit,transit" \
    --n-episodes 5 --visualize --delay 0.02 --wait --debug
# Expected: Watch each episode. After each chain, viewer shows final arm state.
# Press Enter to advance to next episode.
```

### Test D: Transit → descend (alignment diagnostic)
```bash
python scripts/eval_sequence.py --model models/latest_model.zip --chain "transit,descend" \
    --n-episodes 5 --debug
# Expected: transit succeeds. Transition logs show halt + alignment.
# Descend will TUBE_BREACH (pre-V6 model can't do 5mm precision).
# The transition itself should be clean — validate alignment error < 3mm.
```

### Test E: Alignment error distribution
```bash
python scripts/eval_sequence.py --model models/latest_model.zip --chain "transit,transit" \
    --n-episodes 50 --debug 2>&1 | grep "align_error"
# Expected: distribution of align_error values after transit success
# Should be 1–5mm range; median closer to 2–3mm
```

---

## Execution Order

1. Add `soft_reset` and `transition_validate` to `task.py`
2. Create `src/chess_env/waypoints.py`
3. Create `scripts/eval_sequence.py`
4. Run validation tests A → E in order
5. Run Tier 1: `transit,transit` (20+ episodes) to establish baseline
6. Run Tier 1: `transit,descend` to see alignment error distribution
7. Wait for V6 model training completion
8. Run Tier 1–4 chains with V6 model
9. Document results; set minimum quality thresholds for game controller

---

## What This Stage Does NOT Cover

- Actual grasping (picking up the cube) — no cube in simulation yet
- Releasing the piece at the destination
- Home position transitions
- Integration with real hardware
- The game controller state machine

These are deferred until chaining validation is complete with ≥70% success on the
full-move chain with the V6 model.
