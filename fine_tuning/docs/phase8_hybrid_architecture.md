# Phase 8: Hybrid Architecture and Dense Reward Fine-Tuning - Final Specification

This document is the definitive technical blueprint for the RoboChess movement system. All coordinates and mathematical constants have been verified against the physical constraints of the MuJoCo environment.

---

## 1. Physical Environment & Reachability Reference
These values are the "Source of Truth" for all scripts.

*   **Table Surface Height:** `Z = 0.400 m`
*   **Board Center:** `X = 0.880 m, Y = 0.264 m`
*   **Board Half-Extents:** `0.320 m`
*   **Sampling Range (with 4cm margin):** 
    *   `X: [0.600, 1.160]`
    *   `Y: [-0.016, 0.544]`
*   **Robot Base Anchor:** `X = 0.287, Y = 0.264` (Base is perfectly centered on Y-axis).
*   **Safety Gap:** `11.3 cm` between robot front and table edge.

---

## 2. Waypoint Controller & State Machine

The Python controller will manage the following deterministic sequence for every move.

### Waypoint Z-Altitudes
| Waypoint | Z-Height | Description |
| :--- | :--- | :--- |
| `SAFE_Z` | `0.550 m` | 15cm above table. Clears all standing pieces. |
| `GRASP_Z` | `0.425 m` | Gripper center aligned with cube center-of-mass. |
| `RELEASE_Z` | `0.435 m` | 1cm "Soft-Drop" height to avoid board impact. |

### Movement Sequence Logic
1.  **Stage 1 (Hover):** Request Target `[SRC_X, SRC_Y, SAFE_Z]`. 
2.  **Stage 2 (Descend):** Request Target `[SRC_X, SRC_Y, GRASP_Z]`. 
3.  **Stage 3 (Grasp):** Send Scripted Action `[0, 0, 0, -1]` (Close Gripper). Wait 0.5s.
4.  **Stage 4 (Verification):** Check `obs['observation'][5]` (Object Z). If `< 0.420`, **ABORT** (Grasp failed).
5.  **Stage 5 (Lift):** Request Target `[SRC_X, SRC_Y, SAFE_Z]`.
6.  **Stage 6 (Transit):** Request Target `[DST_X, DST_Y, SAFE_Z]`.
7.  **Stage 7 (Deliver):** Request Target `[DST_X, DST_Y, RELEASE_Z]`.
8.  **Stage 8 (Release):** Send Scripted Action `[0, 0, 0, 1]` (Open Gripper). Wait 0.5s.
9.  **Stage 9 (Home):** Request Target `[0.700, 0.264, 0.550]` (Neutral rest position).

---

## 3. Dense Reward Mathematical Specification

The reward function `r` used for fine-tuning the **Cerebellum** (Movement Engine).

### The Formula
$$r = R_{success} + R_{accuracy} + P_{smoothness} + P_{dampening} + P_{collision}$$

### The Components
*   **$R_{success}$:** `+1.0` if `dist < 0.02m`. (Main goal signal).
*   **$R_{accuracy}$:** `0.5 * exp(-dist / 0.01)`. (Provides sub-centimeter precision pressure).
*   **$P_{smoothness}$:** `-0.01 * ||action[0:3]||^2`. (Penalizes jerky/jittery motor commands).
*   **$P_{dampening}$:** If `dist < 0.05m`: `-0.1 * ||gripper_velocity||`. (Forced deceleration on approach).
*   **$P_{collision}$:** If `gripper_Z < 0.405m`: `-10.0` and **Terminate**. (Instills absolute floor fear).

---

## 4. Training Protocol (The "Style Transfer" Run)

To refine the existing 96% successful Grandmaster weights without damaging them:

*   **Base Weights:** `models/final_grandmaster_model.zip`.
*   **Learning Rate:** `1e-5` (Constant).
*   **Policy:** `MultiInputPolicy` (SAC).
*   **Entropy (`ent_coef`):** `0.001` (Low exploration; focus on perfection).
*   **Steps:** `300,000` (Approx 2 hours).
*   **Validation:** Use `SuccessRateEvalCallback` with 20mm threshold.

---

## 5. Success Criteria (Post-Training)
The model is considered ready for Phase 9 (Chess AI integration) only if:
1.  **Overall Success Rate:** `> 98%`.
2.  **Corner Success Rate:** `> 95%`.
3.  **Mean Terminal Error:** `< 8 mm`.
4.  **Table Collisions:** `0` across 200 evaluation episodes.
