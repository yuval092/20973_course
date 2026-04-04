# Arm Movement Stages

This document describes the current staged manipulation pipeline in
`src/control/execution_controller.py`.

The current design goals are:

- keep the motion path debuggable by splitting it into explicit stages,
- avoid overriding piece physics during gameplay,
- keep the gripper state stage-controlled and deterministic,
- use the RL model only where it helps without letting it destabilize contact,
- make failures local and recoverable instead of monolithic.

## Physics Policy

During gameplay, the controller does not directly rewrite chess-piece poses,
velocities, or orientations.

What is still non-physical:

- `ChessPickPlaceEnv.reset_robot_pose()` writes robot state directly, but it is
  only used to establish the canonical home pose during environment setup.

What is physical during normal arm execution:

- all staged arm motion uses MuJoCo stepping,
- the gripper is actuated through the robot actuators,
- released pieces settle under gravity,
- retreat and homing are physical moves, not teleports.

## RL Versus Scripted Control

The controller is intentionally hybrid.

Scripted control owns:

- stage sequencing,
- source/destination targets,
- gripper opening/closing,
- release, clearance, and homing,
- retry behavior after failed pickup acquisition.

RL currently contributes only to the higher-clearance travel stages:

- `PREHOVER_SRC`
- `PREHOVER_DEST`

The low-altitude contact stages are currently scripted for determinism:

- `DESCEND_SRC`
- `CLOSE_GRIPPER_ONLY`
- `LIFT_VERIFY`
- `DESCEND_DEST`

This is deliberate. The contact-critical stages were the least stable part of
the pipeline, so the controller now prioritizes repeatability there.

## Stage Summary

### 1. `HOME_RESET`

Purpose:
Return the arm to the canonical home area before starting a pickup attempt.

How it works:

- Forces the gripper open.
- If the gripper is already near home, it only settles.
- Otherwise it performs a physical retreat to the cached home grip pose.

Important note:
This is not a simulator reset during gameplay. It is a staged physical move.

### 2. `PREHOVER_SRC`

Purpose:
Move above the live source piece at safe height.

How it works:

- Targets `[src_x, src_y, Z_SAFE]`.
- Uses waypoint guidance blended with RL if the RL proposal is aligned.
- Keeps the gripper open.

Why it exists:
This covers most of the source approach distance while avoiding nearby pieces.

### 3. `PREGRASP_NARROW`

Purpose:
Set a deterministic pre-grasp finger opening before the final source approach.

How it works:

- Uses direct gripper actuation.
- Does not move the arm.

Why it exists:
This reduces variability before the low-altitude scripted pickup stages.

### 4. `DESCEND_SRC`

Purpose:
Bring the gripper down to a source approach height above the piece.

How it works:

- Re-reads the live piece pose.
- Descends to `piece_z + source_descend_offset`.
- Uses scripted Cartesian motion with capped action magnitude.
- Holds the pre-grasp aperture explicitly.

Why it exists:
This is an approach stage, not the actual grasp. The goal is to arrive close to
the piece without shoving it.

### 5. `CLOSE_GRIPPER_ONLY`

Purpose:
Establish contact and close the fingers around the source piece.

How it works:

- First re-centers above the current live piece pose at a short clearance height.
- Then performs a slow vertical close sequence.
- Uses direct gripper actuation instead of RL.
- Validates close-stage geometry before allowing the lift stage.

Why it exists:
Pickup contact is the least reliable phase, so it is isolated for logging and retry.

### 6. `LIFT_VERIFY`

Purpose:
Lift the piece and prove that the grasp is real.

How it works:

- Moves back toward source safe height with the gripper held closed.
- Requires the piece to follow the gripper.
- Fails if the fingers close but the piece stays on the board.

Why it exists:
This is the first point where the controller can distinguish a real grasp from
an empty close.

### 7. `PREHOVER_DEST`

Purpose:
Transport the grasped piece above the destination at safe height.

How it works:

- Moves toward `[dest_x, dest_y, Z_SAFE]`.
- Keeps the gripper closed.
- Uses waypoint guidance blended with RL if aligned.

Why it exists:
Most lateral travel happens away from the board surface and neighboring pieces.

### 8. `DESCEND_DEST`

Purpose:
Bring the grasped piece down toward the destination placement height.

How it works:

- Uses scripted, capped Cartesian motion.
- Holds the gripper closed.
- Monitors whether the piece continues to follow.

Why it exists:
Placement descent benefits more from repeatability than from aggressive policy motion.

### 9. `OPEN_GRIPPER_ONLY`

Purpose:
Release the piece.

How it works:

- Opens the fingers through direct actuator control.
- Does not change the arm target.

Why it exists:
Release is kept separate from retreat so contact problems are easier to diagnose.

### 10. `POST_RELEASE_SETTLE`

Purpose:
Wait for the released piece to separate from the gripper and settle under gravity.

How it works:

- Holds the arm still.
- Steps MuJoCo without pose overrides.
- Requires:
  - open fingers,
  - destination XY/Z proximity,
  - low piece velocity,
  - sufficient gripper-piece separation.

Why it exists:
This prevents the controller from dragging a just-released piece into the retreat.

### 11. `POST_RELEASE_CLEARANCE`

Purpose:
Lift vertically away from the released piece before moving laterally.

How it works:

- Moves upward to a clearance corridor.
- Keeps the gripper open.
- Fails if the released piece is disturbed beyond tolerance.

Why it exists:
Sideways retreat too early is a common way to re-contact the placed piece.

### 12. `RETURN_HOME`

Purpose:
Return to the home pose after the release is physically safe.

How it works:

- Moves laterally at the elevated clearance height.
- Keeps the gripper open.
- Continues to monitor the released piece for disturbance.

Why it exists:
Homing should be predictable and should not sweep through the board plane.

### 13. `FINAL_PLACEMENT_SETTLE`

Purpose:
Give the final placement time to stabilize after the arm is clear.

How it works:

- Steps physics only.
- Requires XY/Z tolerance, low speed, and upright stability.
- Fails if the piece remains tipped, elevated, or sunk.

Why it exists:
This catches placements that only look correct at release time.

## Retry Behavior

Pickup acquisition currently has one explicit recovery path.

If `CLOSE_GRIPPER_ONLY` or `LIFT_VERIFY` fails:

- the controller opens the gripper,
- lets the scene settle,
- re-observes the live source pose,
- rebuilds the source stage targets,
- retries from `HOME_RESET`.

This retry loop is intended to recover from small physical nudges during source acquisition.

## Known Weak Spot

The current weakest part of the system is still source acquisition on out-of-policy
pieces and squares. The controller is now fully physical in this area, but the
grasp primitive is still being tuned.

The main remaining challenge is:

- reliably establishing a physical pinch on chess-piece proxies without shoving
  the piece away before lift.
