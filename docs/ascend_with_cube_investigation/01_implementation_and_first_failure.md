# 01. Initial Implementation and First Failure

This document details the first phase of the `ascend_with_cube` project, where we moved from "pure movement" (empty gripper) to physical interaction.

## 1. The Design: Scripted Grasping
To test the model's ability to carry a payload, we implemented a new scenario in `src/chess_env/task.py`. Unlike previous scenarios where the piece was hidden, this scenario required the physical presence of a 3x3x3 cm cube.

### Code Changes:
1.  **Scripted Grasp Phase:** We added a ramp-up phase during the environment reset. Before the RL model takes control, the arm spawns at `GRASP_Z` directly over the cube. The gripper fingers are then gradually closed over 20 physics steps, increasing the force from 0N to **15N**.
2.  **Scenario Logic:** A new ID `ascend_with_cube` was added. This locks the gripper closed (`action[3] = -1.0`) and sets the goal to the cruise altitude (`SAFE_Z = 0.55m`).

## 2. The Execution: What We Saw
When we ran the first evaluation (`scripts/eval.py`), we observed a **0% success rate**. 

### Observations from Debug Logs:
By using the `--debug` flag, we printed the exact coordinates of the `robot0:grip` and the `object0` at every step. We saw the following sequence:
1.  **Step 0:** The arm is perfectly positioned over the cube. The grasp is successful.
2.  **Step 1:** The arm begins to lift. The cube moves up to `z = 0.415m`.
3.  **Steps 2-5:** The cube's (X, Y) coordinates suddenly diverge from the gripper's (X, Y). 
4.  **Step 6:** The cube's Z coordinate drops back to the table surface (`z = 0.400m`), while the arm continues to the goal altitude.

## 3. The Diagnosis: Lateral Jitter
The debug logs revealed a hidden behavior in our trained SAC model. Although the model reached the vertical goal, it did not move in a straight line. It applied rapid, high-frequency "wiggles" in the X and Y directions.

**How sure are we?**
We are **100% certain**. The `[DEBUG TASK] Step` logs showed lateral velocity spikes at precisely the moment the cube slipped. These wiggles are invisible to the naked eye at 50FPS but are massive in the physics engine's 500Hz calculation loop. The lateral inertia from these wiggles simply overcame the friction of the fingers, causing the cube to slide out the open sides of the gripper.
