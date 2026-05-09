# Elastic Footprint & Curriculum Synchronization

## The Geometry of Success
The Fetch gripper has a finger-to-finger width of **33mm** when closed. A chess cell is **70mm** wide. To prevent the fingers from hitting adjacent pieces (30mm wide), we calculated the physical collision point.

**The Golden Formula**: `Drift + 33mm < 55mm` (Distance to adjacent piece edge).
*   **Max Safe Drift**: **22mm**.
*   **Target Precision**: We set the goal to **10mm (1cm)**. This provides a massive **12mm safety buffer** while maintaining a professional, centered look.

## The Solution: Elastic Boundaries
We synchronized the **Safety Footprint** with the **Success Curriculum**.

1.  **Start State**: Curriculum limit is **100mm**. The arm can move anywhere on the board. Success is easy to hit.
2.  **Linear Decay**: Over 500,000 steps, the limit shrinks from 100mm to **10mm**.
3.  **Safety Check**: The `TUBE_BREACH` crash is now calculated as:
    ```python
    if center_drift > current_curriculum_limit:
        terminate_with_crash("TUBE_BREACH")
    ```

## Why This Works
*   **Exploration**: At the start of training, the model "explores" the vertical space without fear of crashing.
*   **Reward Gradient**: By adding an `XY_REWARD_WEIGHT`, the agent feels a "pull" toward the center, rather than just hitting a wall.
*   **Success Continuity**: The model always maintains a high success rate because the precision required never exceeds its current proven capability.

## Visualization of the "Squeeze"
| Training Step | Drift Limit | Precision Level |
| :--- | :--- | :--- |
| 0 | 100 mm | Rough Approximation |
| 100,000 | 82 mm | Scenario Understanding |
| 250,000 | 55 mm | Cell Containment |
| 400,000 | 28 mm | Physical Safety Mark (22mm) |
| 500,000 | 10 mm | **Production Certified** |

---
*Next: [Absolute Actuator Enforcement](./03_actuator_enforcement.md)*
