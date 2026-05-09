# Master Controller Logic: Orchestrating the "Staccato" Pipeline

The Master Controller (`eval.py` or a dedicated `game.py` script) is the orchestrator of the entire chess move. It acts as the state machine that transitions between the RL Motor Primitives and the scripted actions, enforcing the "Stillness Validation" firewall.

## 1. The State Machine Sequence
A complete Pick-and-Place sequence for a chess move (e.g., e2 to e4) follows this strict state progression:

1.  **Settle:** Arm moves over the starting piece (`GRASP_Z` + margin) and waits.
2.  **Scripted Grasp:** Close fingers gradually (20 steps).
3.  **Stillness Validation:** Wait until velocity is near zero.
4.  **Ascend (RL):** Load `Ascend` model. Execute until `is_success` (reaches `SAFE_Z`).
5.  **Stillness Validation:** Wait.
6.  **Transit (RL):** Load `Transit` model. Execute until `is_success` (reaches Goal XY).
7.  **Stillness Validation:** Wait.
8.  **Descend (RL):** Load `Descend` model. Execute until `is_success` (reaches `GRASP_Z` AND velocity is near zero).
9.  **Scripted Release:** Open fingers gradually (20 steps).
10. **Stillness Validation:** Wait.
11. **Retract:** Move arm back up to `SAFE_Z` to clear the board.

## 2. The "Stillness Validation" Firewall
This is the core innovation of the Staccato pipeline. Between every major phase, the Controller injects a `0.2s` (10 physics steps) pause where it sends `[0, 0, 0, prev_grip]` actions.

*   **Why:** In MuJoCo, changing a control target or loading a new RL model's first action often creates a sudden discontinuity (a "step"). If the arm is already moving at 0.5 m/s when this happens, the discontinuity translates into a massive acceleration impulse. 
*   **The Firewall:** By forcing the velocity to near zero (`< 0.05 m/s`) before transitioning, we reset the momentum vector. Any discontinuity from the new model starts from a zero-energy state, preventing the "Micro-Bounce" and subsequent payload ejection.

## 3. The Sentry & Recovery Logic
The Master Controller also monitors the environment's `info` dictionary every step. It acts as the "Sentry" checking the Omniscient Critic's flags.

*   **Drop Detection:** If `info['is_held'] == False` at any point during `Ascend`, `Transit`, or `Descend`.
*   **Shift Detection:** If `info['shift_distance'] > 0.005m`.

### Retry Logic (The Real-World Benefit)
If the Sentry detects a Drop or a massive Shift, the Staccato pipeline makes recovery much simpler than a monolithic RL model.

*   **If Dropped during Ascend/Descend:** The piece is still roughly in the same vertical column. The Controller can abort the RL run, execute a Scripted Release, retract to `SAFE_Z`, and retry the grasp.
*   **If Dropped during Transit:** The chess state is corrupted (the piece fell on the board). The Controller must flag an "Operator Intervention" or switch to a completely different "Board Recovery" vision model. By knowing exactly *which* primitive failed, debugging and physical recovery are isolated.
