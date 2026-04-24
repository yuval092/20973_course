# Phase 9: Waypoint Mapping

This document maps the user's 11-step complete chess movement flow to the 3 underlying RL Sub-Tasks (Scenarios) we are training the Cerebellum to perform.

## The 3 RL Sub-Tasks (The "Cerebellum" Skills)
The RL model is only responsible for moving the arm from point A to point B smoothly, without violating the constraints of the airspace.
*   **Scenario A (Transit):** Move horizontally at `SAFE_Z`. Constraint: Do not drop altitude (Virtual Floor).
*   **Scenario B (Elevator Down):** Move vertically from `SAFE_Z` down to `GRASP_Z`/`RELEASE_Z`. Constraint: Do not drift horizontally (Virtual Tube).
*   **Scenario C (Elevator Up):** Move vertically from `GRASP_Z`/`RELEASE_Z` up to `SAFE_Z`. Constraint: Do not drift horizontally (Virtual Tube).

---

## Mapping the 11-Step Flow

### 1. HOME_RESET
*   **Action:** The arm begins by resetting to its default, neutral home position (air, in the center of the board).
*   **Executor:** RL Model (Scenario A: Transit)

### 2. PREHOVER_SRC
*   **Action:** The arm moves to hover directly above the piece to be picked up (Point A).
*   **Executor:** RL Model (Scenario A: Transit)

### 3. PREGRASP_NARROW
*   **Action:** The gripper partially closes to match the width of the target piece before descending.
*   **Executor:** Scripted Python Controller. (Sets gripper actuator to 0.02m).

### 4. DESCEND_SRC
*   **Action:** The arm lowers down over the piece to the appropriate grasping height.
*   **Executor:** RL Model (Scenario B: Elevator Down). Target = `GRASP_Z`.

### 5. CLOSE_GRIPPER_ONLY
*   **Action:** The gripper fingers close to firmly grasp the piece.
*   **Executor:** Scripted Python Controller.

### 6. LIFT_VERIFY
*   **Action:** The arm lifts the piece up and validates that the grasp was successful.
*   **Executor:** RL Model (Scenario C: Elevator Up). Start = `GRASP_Z`, Target = `SAFE_Z`. Python script verifies Z-height of piece.

### 7. PREHOVER_DEST
*   **Action:** The arm carries the piece through the air to hover over the target square (Point B).
*   **Executor:** RL Model (Scenario A: Transit).

### 8. DESCEND_DEST
*   **Action:** The arm lowers the piece down to the board surface.
*   **Executor:** RL Model (Scenario B: Elevator Down). Target = `RELEASE_Z`.

### 9. OPEN_GRIPPER_ONLY
*   **Action:** The gripper fingers open to release the piece.
*   **Executor:** Scripted Python Controller.

### 10. POST_RELEASE_CLEARANCE
*   **Action:** The arm lifts straight up to safely clear the placed piece without knocking it over.
*   **Executor:** RL Model (Scenario C: Elevator Up). Start = `RELEASE_Z`, Target = `SAFE_Z`.

### 11. RETURN_HOME
*   **Action:** The arm transits back to its neutral home position.
*   **Executor:** RL Model (Scenario A: Transit).
