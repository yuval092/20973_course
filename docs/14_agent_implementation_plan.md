# Actuator Enforcement & Production Certification Implementation Plan

**Target Audience:** AI Implementation Agent
**Objective:** Refactor the RoboChess `Grasp Stage` pipeline to achieve 100% reliability. You will implement a Hybrid "Macro/Micro" control paradigm: RL for macro-navigation and scripted proportional controllers for micro-manipulation (the "Danger Zone").

---

## 1. Core Paradigm Shift

We have empirically proven that enforcing perfect verticality ("Crane Mode") during the RL phase causes a 20% failure rate because the robot cannot physically reach the board corners without tilting. Furthermore, softening the `weld` constraint to prevent "wrist snap" during lifting caused physical lag and trajectory drift.

**The Solution:**
1.  **Allow RL to Tilt:** Remove vertical enforcement during the RL phase.
2.  **Raise the Target:** The RL policy targets a safe `HOVER_Z` (0.460m) instead of `GRASP_Z` (0.425m).
3.  **Scripted Danger Zone:** Once hovering, a high-precision script takes over to fix the rotation, align perfectly over the cube, plunge vertically, grasp, and retract back to `HOVER_Z`.
4.  **Stiff Weld:** Because the scripted retract eliminates the "wrist snap," we can revert the physical weld to be extremely stiff, eliminating drift.

---

## 2. Implementation Tasks

Execute the following tasks sequentially.

### Task 1: Physical & Environmental Hardening
**Files to modify:** `chess_env/assets/shared.xml`, `configs/env.yaml`, `src/chess_env/simulation.py`

1.  **Stiffen the Weld:** In `shared.xml`, revert the `<weld>` constraint connecting `robot0:mocap` to `robot0:gripper_link` back to `solref="0.01 1"`.
2.  **Define Hover Height:** In `configs/env.yaml`, add a new configuration parameter `hover_z: 0.460`.
3.  **Remove RL Handcuffs:** In `src/chess_env/simulation.py`, locate the `_set_action` method. Remove the code that explicitly enforces `VERTICAL_QUAT` on the mocap every step. *Note: Keep the rotational delta at zero (`rot_ctrl = np.zeros(4)`), just remove the absolute `set_mocap_quat` override.*

### Task 2: Task Environment Preparation
**File to modify:** `src/chess_env/task.py`

1.  **Initialize `HOVER_Z`:** In the `__init__` method, load `self.HOVER_Z = self.env_cfg.get("hover_z", 0.460)`.
2.  **Update Descend Target:** In `_reset_sim` and `soft_reset`, modify the goal generation for the `descend` scenario so the target Z-height is `self.HOVER_Z` instead of `self.GRASP_Z` or `self.SAFE_Z`.
3.  **Add `settle_mocap` Utility:** Create a new helper method inside the class:
    ```python
    def _settle_mocap(self, max_steps=50, tolerance=0.001):
        """Runs the physics engine until the physical grip_pos matches the mocap_pos."""
        zero_action = np.zeros(4)
        for _ in range(max_steps):
            grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            mocap_pos = self.data.mocap_pos[0][:3]
            if np.linalg.norm(grip_pos - mocap_pos) < tolerance:
                break
            self._set_action(zero_action)
            self._mujoco_step(None)
    ```

### Task 3: The Scripted `execute_grasp()` Pipeline
**File to modify:** `src/chess_env/task.py`

Completely rewrite the `execute_grasp` method. It must return the standard `dict` with a `"success"` boolean. Implement the following sub-phases in exact order:

**Phase 0: Halt & Settle**
*   Zero out the arm's velocity (`qvel` and `qacc` for `[:15]`).
*   Run `_set_action(zeros)` for 15 steps to let the stiff weld equilibrate.

**Phase 1: Vertical Correction**
*   *Algorithm:* The arm is currently at `HOVER_Z` but might be tilted. Force the `mocap` orientation to `self.VERTICAL_QUAT` using `self._utils.set_mocap_quat`.
*   Run the `_settle_mocap()` utility so the physical arm swings into a perfect vertical position in mid-air.

**Phase 2: Perfect XY Align & Rotation Safety Check**
*   *Algorithm:* Get the cube's exact real-world position using `self.get_cube_position()`.
*   Check the cube's orientation (quaternion). Convert it to Euler yaw. If the yaw is > 25 degrees (approx `0.43` radians), the cube is diagonal and the fingers will stub. Return `{"success": False, "reason": "CUBE_ROTATED"}` immediately.
*   Update `self.data.mocap_pos[0][:2]` to exactly match the cube's `[:2]` coordinates.
*   Run the `_settle_mocap()` utility. The arm is now perfectly centered over the cube in strict Crane Mode.

**Phase 3: The Plunge**
*   *Algorithm:* Move `mocap_pos[0][2]` down toward `self.GRASP_Z` in tiny increments (e.g., 1mm per step).
*   After each 1mm increment, call `_set_action(zeros)` and `self._mujoco_step(None)`.
*   Do this until the physical `grip_pos[2]` reaches `self.GRASP_Z`.

**Phase 4: Grasp**
*   *Algorithm:* Set `self.grasp_mode = True` and `self.finger_target_joint = self.FINGER_CLOSED_JOINT`.
*   Run `_set_action(zeros)` and `self._mujoco_step(None)` for `self.GRASP_CLOSE_STEPS` (150 steps).
*   Implement the Early Abort logic: if fingers are fully closed (`< 0.003`) after 30 steps, the cube was missed. Return `{"success": False, "reason": "MISSED_CUBE"}`.

**Phase 5: Hold Settle & Verify**
*   Run 50 steps of `_set_action(zeros)` to let the grasp physics settle.
*   Perform the standard checks: ensure the cube's XY hasn't drifted more than 15mm from the grip site.

**Phase 6: The Retract**
*   *Algorithm:* Move `mocap_pos[0][2]` up toward `self.HOVER_Z` in tiny increments (e.g., 1mm per step).
*   After each increment, step the physics. This slowly lifts the cube, mathematically eliminating the "wrist snap" shock.
*   Once at `HOVER_Z`, the script ends and returns `{"success": True, ...}`. The RL policy will take over for the `ASCEND` scenario.

---

## 3. Critical Warnings & Edge Cases

*   **Mocap Update Order:** In MuJoCo, `_set_action` calls `mocap_set_action`, which often resets the mocap position to the current body position. When manually updating `data.mocap_pos` (like in the Plunge or Retract phases), you MUST apply the manual update *after* calling `_set_action` but *before* calling `self._mujoco_step(None)`.
*   **Zero Velocity:** Never run a scripted sequence without explicitly forcing `qvel` and `qacc` to `0.0` at the very beginning. Residual RL momentum will cause the arm to jitter and explode on contact.
*   **Do Not Shrink the Board:** We are relying entirely on the arm's natural tilt to reach the board edges. Do not modify `table_center_xy` or `table_half_x` unless physically impossible limits are hit during testing.
