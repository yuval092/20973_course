# Soft Reset Reference: Phase-by-Phase Specification

## What Is Soft Reset?

`soft_reset` is a controlled transition between two RL scenarios. When a scenario
succeeds, the next scenario starts from a well-defined state. Without soft_reset,
the new scenario would start from a random initial state, wasting time and causing
distribution shift.

Soft reset does NOT call `env.reset()`. The physics simulation continues running.
Only the scenario context (goal, reward function, success condition) changes.

---

## The Four Phases

### Phase 1: HALT (100 steps)

**Purpose:** Stop the arm from moving before the scenario transition.

**Mechanism:**
```python
for _ in range(HALT_STEPS):  # 100 steps
    zero_action = np.zeros(4)  # [dx=0, dy=0, dz=0, gripper_ignored]
    env.step(zero_action)
```

During halt:
- The arm's inertia is dissipated by MuJoCo's damping
- `_set_action` is called with zero positional change
- Fingers: if `grasp_mode=False`, teleport continues at current target. If
  `grasp_mode=True`, the actuator holds at `FINGER_CLOSED_JOINT` — the cube is held

**How long to halt:** 100 steps × 20ms/step = 2 seconds simulated time. Arm speed
after halt is typically < 1mm/s.

**What can go wrong:**
- Cube drops during halt if grasp_mode=True and contact force is marginal. Check
  `_check_cube_held()` during halt when grasp_mode=True.
- Arm doesn't fully stop in 100 steps if the weld constraint is resonating. The
  align phase corrects this.

---

### Phase 2: ALIGN (up to 200 steps, gain=0.8)

**Purpose:** Move the arm to the nominal exit position of the outgoing scenario.

**Mechanism:**
```python
for step in range(ALIGN_STEPS):  # 200 steps
    grip_pos = get_site_xpos("robot0:grip")
    error = nominal_exit_pos - grip_pos  # 3D error vector
    if np.linalg.norm(error) < ALIGN_TOLERANCE:  # 1mm
        break
    delta = ALIGN_GAIN * error  # 0.8 * error
    delta = np.clip(delta, -POS_CTRL_SCALE, POS_CTRL_SCALE)  # max 15mm/step
    set_mocap_pos(current_mocap_pos + delta)
    mujoco_step()
```

**What `nominal_exit_pos` is for each transition:**

| Outgoing Scenario | nominal_exit_pos |
|:---|:---|
| transit → descend | `[src_xy[0], src_xy[1], SAFE_Z]` |
| descend → (GRASP) | n/a — no soft_reset |
| (GRASP) → ascend | `[src_xy[0], src_xy[1], GRASP_Z]` |
| ascend → transit | `[src_xy[0], src_xy[1], SAFE_Z]` |
| transit → transit | `[prev_dst_xy[0], prev_dst_xy[1], SAFE_Z]` |
| transit → descend (place) | `[dst_xy[0], dst_xy[1], SAFE_Z]` |
| (PLACE) → ascend | `[dst_xy[0], dst_xy[1], PLACE_Z]` |
| ascend → transit (return) | `[dst_xy[0], dst_xy[1], SAFE_Z]` |

**Why align matters:** The new scenario's policy was trained from a distribution
of starting positions. If the arm is 8mm off from the nominal exit, the new policy
may take 50+ steps to correct this. Aligning first puts the arm in the policy's
training distribution immediately.

**Cube behavior during align:** If `grasp_mode=True`, the cube hangs in the gripper
during alignment. The align moves the arm (and thus the cube) slowly. The
`_check_cube_held()` check should run each step if implemented.

---

### Phase 3: STATE UPDATE

**Purpose:** Switch the environment's internal scenario context.

**What changes:**
```python
self.current_scenario = new_scenario
self.goal_pos = new_goal_pos
# The reward function, success condition, and observation now reflect new_scenario
```

**What does NOT change:**
- Physics state (qpos, qvel, mocap_pos)
- `self.grasp_mode` — NEVER reset here
- `self.finger_target_joint` — NOT reset here if grasp_mode=True
- Cube position

**The time-limit wrapper:** After soft_reset, the time-limit wrapper must be reset
so the new scenario gets its full step budget. There is no pre-defined helper
function — the caller must traverse the wrapper chain inline:
```python
curr = env
while hasattr(curr, "env"):
    if hasattr(curr, "_elapsed_steps"):
        curr._elapsed_steps = 0
        break
    curr = curr.env
```

---

### Phase 4: GRIPPER TRANSITION (50 steps)

**Purpose:** Set the finger target for the new scenario and let physics settle.

**For each scenario:**

| Incoming Scenario | finger_target | Action |
|:---|:---|:---|
| transit (no cube) | `FINGER_OPEN_JOINT` | Teleport open |
| descend (no cube) | `FINGER_OPEN_JOINT` | Teleport open |
| ascend (no cube) | `FINGER_OPEN_JOINT` | Teleport open (was closed in prior stage? No — was always open) |
| ascend (with cube) | `FINGER_CLOSED_JOINT` | **SKIPPED** (grasp_mode=True) |
| transit (with cube) | `FINGER_CLOSED_JOINT` | **SKIPPED** (grasp_mode=True) |

**grasp_mode guard (Bug C fix):**
```python
if not self.grasp_mode:
    self.finger_target_joint = finger_target_for_new_scenario
    # Teleport fingers to new target
    self._utils.set_joint_qpos(model, data, "robot0:l_gripper_finger_joint",
                                self.finger_target_joint)
    self._utils.set_joint_qpos(model, data, "robot0:r_gripper_finger_joint",
                                self.finger_target_joint)
    # Run 50 settle steps
    for _ in range(GRIP_TRANSITION_STEPS):
        mujoco_step()
# else: grasp_mode=True — fingers stay actuator-driven, no teleportation
```

---

## Per-Transition Specifications

### T1: transit → descend

**Trigger:** Transit succeeds (arm within 10mm of `[src_xy, SAFE_Z]`).

**State entering soft_reset:**
- Arm near `[src_xy, SAFE_Z]`, speed < 50mm/s
- grasp_mode = False
- finger_target = FINGER_OPEN_JOINT
- Cube: at src_xy on table (not yet picked)

**Phase 1 (HALT):** Zero action × 100 steps. Arm speed → ~0.
**Phase 2 (ALIGN):** Target = `[src_xy[0], src_xy[1], SAFE_Z]`. Typically 0–3mm
  correction. May take 5–20 steps.
**Phase 3 (STATE):** `current_scenario = "descend"`. `goal_pos = [src_xy, GRASP_Z]`.
**Phase 4 (GRIP):** finger_target = FINGER_OPEN_JOINT (already open). Teleport
  is idempotent. 50 settle steps.

**State exiting soft_reset:**
- Arm at `[src_xy, SAFE_Z]` ± 1mm
- Speed < 1mm/s
- Scenario: descend
- Goal: `[src_xy, GRASP_Z]`

---

### T2: GRASP → ascend

**Trigger:** `execute_grasp()` returns `success=True`.

**State entering soft_reset:**
- Arm at `[src_xy, GRASP_Z]` ± 1mm, speed ≈ 0 (hold settle ended)
- grasp_mode = True
- finger_target = FINGER_CLOSED_JOINT (set by execute_grasp)
- Cube: held, ~15mm below grip site

**Phase 1 (HALT):** Zero action × 100 steps. Arm already stationary. Cube held.
  `_check_cube_held()` runs each step to detect unexpected drop.
**Phase 2 (ALIGN):** Target = `[src_xy, GRASP_Z]`. Arm already there. < 5 steps.
**Phase 3 (STATE):** `current_scenario = "ascend"`. `goal_pos = [src_xy, SAFE_Z]`.
  `grasp_mode` NOT touched.
**Phase 4 (GRIP):** grasp_mode=True → SKIPPED. Fingers stay actuator-driven.

**State exiting soft_reset:**
- Arm at `[src_xy, GRASP_Z]` ± 1mm
- Scenario: ascend
- Goal: `[src_xy, SAFE_Z]`
- grasp_mode: True
- Cube: held

**Critical invariant:** `self.grasp_mode` must survive the state update unchanged.
Verify by adding: `assert uw.grasp_mode == True` after soft_reset returns.

---

### T3: ascend → transit (after pick, cube held)

**Trigger:** Ascend succeeds (arm within 10mm of `[src_xy, SAFE_Z]`).

**State entering soft_reset:**
- Arm near `[src_xy, SAFE_Z]`, speed < 50mm/s
- grasp_mode = True
- Cube: held, ~15mm below grip, at ~0.535m altitude

**Phase 1 (HALT):** Zero × 100. Arm stops. Cube held.
**Phase 2 (ALIGN):** Target = `[src_xy, SAFE_Z]`. 0–3mm correction.
**Phase 3 (STATE):** `current_scenario = "transit"`. `goal_pos = [next_xy, SAFE_Z]`.
  `grasp_mode` NOT touched.
**Phase 4 (GRIP):** SKIPPED (grasp_mode=True).

**State exiting soft_reset:**
- Arm at `[src_xy, SAFE_Z]`
- Scenario: transit
- grasp_mode: True
- Cube: held at ~0.535m (120mm above table, 90mm above chess pieces)

---

### T4: transit → descend (place leg, cube held)

**Trigger:** Transit to `dst_xy` succeeds (arm within 10mm of `[dst_xy, SAFE_Z]`).

**State entering soft_reset:**
- Arm near `[dst_xy, SAFE_Z]`, speed < 50mm/s
- grasp_mode = True
- Cube: held

**Phase 1 (HALT):** Zero × 100. Arm stops.
**Phase 2 (ALIGN):** Target = `[dst_xy, SAFE_Z]`.
**Phase 3 (STATE):** `current_scenario = "descend"`. `goal_pos = [dst_xy, PLACE_Z]`.
  Note: `force_target_z = PLACE_Z` override needed in `_reset_sim`. Without this,
  `_reset_sim` defaults to `GRASP_Z` for descend goals.
**Phase 4 (GRIP):** SKIPPED (grasp_mode=True). Cube still held during descent.

---

### T5: PLACE → ascend (after release)

**Trigger:** `execute_place()` returns `success=True`. Cube on board at dst_xy.

**State entering soft_reset:**
- Arm at `[dst_xy, PLACE_Z]` ± 1mm, speed ≈ 0
- grasp_mode = False (reset by execute_place)
- finger_target = FINGER_OPEN_JOINT (reset by execute_place)
- Cube: on board at dst_xy, not held

**Phase 1 (HALT):** Zero × 100. Arm already stationary.
**Phase 2 (ALIGN):** Target = `[dst_xy, PLACE_Z]`. Already there.
**Phase 3 (STATE):** `current_scenario = "ascend"`. `goal_pos = [dst_xy, SAFE_Z]`.
**Phase 4 (GRIP):** grasp_mode=False → teleport open. Already open (idempotent).

---

### T6: ascend → transit (return to home, no cube)

**Trigger:** Ascend from `dst_xy` succeeds.

**State entering soft_reset:**
- Arm near `[dst_xy, SAFE_Z]`, speed < 50mm/s
- grasp_mode = False
- No cube held

Standard transit soft_reset. `goal_pos = HOME_POS`.

---

## grasp_mode Lifecycle

```
env.reset()
    └─ _reset_sim() sets grasp_mode = False

execute_grasp()
    └─ sets grasp_mode = True (Sub-Phase 2 start)

soft_reset() [GRASP → ascend]
    └─ does NOT touch grasp_mode
    └─ grasp_mode stays True

step() [ascend, transit with cube held]
    └─ reads grasp_mode for _set_action and _check_cube_held

execute_place()
    └─ sets grasp_mode = False (finger open, cube released)

soft_reset() [PLACE → ascend]
    └─ does NOT touch grasp_mode
    └─ grasp_mode stays False

env.reset() [next episode]
    └─ _reset_sim() sets grasp_mode = False
```

**Rule:** `grasp_mode` is only written by:
1. `_reset_sim()` → always False
2. `execute_grasp()` → True
3. `execute_place()` → False

It is never written by `soft_reset()`, `step()`, or the RL policy.

---

## Bug C: soft_reset Phase 4 Teleports Fingers Into Cube

**The bug (before fix):**

In the current `soft_reset` Phase 4:
```python
self.finger_target_joint = FINGER_OPEN_JOINT  # or some scenario default
self._utils.set_joint_qpos(model, data, "robot0:l_gripper_finger_joint",
                            self.finger_target_joint)
```

If called when `grasp_mode=True` (GRASP→ascend transition), this teleports the
fingers open while the cube is between them. The cube is now unsupported and
falls. Additionally, MuJoCo generates a large repulsive force from the fingers
snapping open against the cube geometry.

**The fix:**
```python
# Phase 4: Gripper Transition
if not self.grasp_mode:
    self.finger_target_joint = finger_target_for_scenario(self.current_scenario)
    self._utils.set_joint_qpos(model, data, "robot0:l_gripper_finger_joint",
                                self.finger_target_joint)
    self._utils.set_joint_qpos(model, data, "robot0:r_gripper_finger_joint",
                                self.finger_target_joint)
    self._utils.set_joint_qvel(model, data, "robot0:l_gripper_finger_joint", 0.0)
    self._utils.set_joint_qvel(model, data, "robot0:r_gripper_finger_joint", 0.0)
    for _ in range(GRIP_TRANSITION_STEPS):
        self._mujoco_step(None)
# else: grasp_mode=True → fingers stay actuator-driven, no teleportation
```

---

## soft_reset Implementation Checklist

- [ ] Phase 1: 100 halt steps with zero action
- [ ] Phase 2: Up to 200 align steps targeting nominal_exit_pos, gain=0.8, 1mm tolerance
- [ ] Phase 3: Set current_scenario, goal_pos; do NOT touch grasp_mode
- [ ] Phase 4: Guard with `if not self.grasp_mode:`
- [ ] **Bug 10 fix:** The existing `soft_reset` code (task.py lines 496–503) has a
  `FINGER_VALIDATION` block after Phase 4 that raises `RuntimeError` if the finger
  position deviates from `finger_target_joint` by > 0.0005. With `grasp_mode=True`,
  fingers are at j≈0.014 but `finger_target_joint=0.0` (CLOSED), giving deviation
  0.014 >> 0.0005. This raises `RuntimeError` on every GRASP→ASCEND transition.
  **Fix:** wrap the entire FINGER_VALIDATION block with `if not self.grasp_mode:`
- [ ] After soft_reset: caller resets timelimit wrapper via inline traversal (no helper function)
- [ ] After soft_reset: assert grasp_mode is expected value if debug mode
- [ ] During halt/align: run `_check_cube_held()` if grasp_mode=True

---

*Next: [09 — Test Plan](./09_test_plan.md)*
