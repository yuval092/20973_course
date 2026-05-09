# Executive Summary: Ascend with Cube Failure Investigation

## 1. Overview
This document summarizes the results of an independent evaluation into the 0% success rate of the `ascend_with_cube` scenario. While the previous investigation (Documents 01-04) identified "lateral jitter" as the primary cause, our deep dive reveals a multi-layered failure stemming from simulation geometry, settlement precision, and reward miscalculation.

## 2. Key Findings
*   **The "Ghost Grasp" (Primary Blocker):** A 0.9mm offset in the robotic finger geoms causes a 0.4mm physical gap when the gripper is "closed" at the target force. The arm is lifting nothing; the cube never leaves the table.
*   **Settlement Drift:** The environment reset allows for up to 3mm of misalignment. In a task where the piece is only 30mm wide and the gripper gap is tight, this randomness makes consistent grasping physically impossible without retraining the model to "hunt" for the piece.
*   **Coupling Disproved:** We successfully demonstrated vertical lifts to 0.55m with all lateral jitter masked out. The claim that wiggles are mathematically required for altitude is an artifact of failed success checks, not neural network coupling.
*   **Pneumatic Hammer Effect:** The high-frequency actions (jitter) create accelerations exceeding **1.5 m/s²**. In a rigid simulation, these act as impulsive forces that eject the cube once contact is finally established (the "explosion" bug).

## 3. Conclusion on Suggested Solution
The suggested solution (retraining with a lateral velocity penalty) is **directionally correct but technically insufficient**. Retraining a model in an environment with a physical gap and 3mm start error will only result in a model that "smoothly fails." 

**Retraining should only occur after the environment's physical and kinematic foundations are corrected.**

## 4. Summary Table of Experiments
| Configuration | Success Rate | Avg Final Z | Avg Max Accel | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 0.0% | 0.5465m | 1.53 m/s² | **Total Failure** (Cube stays on table) |
| **Fixed Geometry Only** | 0.0% | 0.5467m | 1.49 m/s² | **Failure** (Jitter knocks cube away) |
| **Full Masking (No Jitter)** | 0.0% | 0.5490m | 0.04 m/s² | **Failure** (Misalignment at start) |
| **Manual "Super Glue"** | **100%** | **0.5512m** | 1.45 m/s² | **Success** (Proves model *can* lift) |
