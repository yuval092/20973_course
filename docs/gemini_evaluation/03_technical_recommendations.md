# Technical Recommendations: The Path Forward

To achieve stable chess piece manipulation, we recommend the following sequence of fixes. **Do not retrain the model until Phase 1 and 2 are complete.**

## Phase 1: Environment Correction (Zero Cost)
These changes fix the math and physics parameters without requiring new training data.

1.  **Correct Grasp Width:** Update `src/chess_env/task.py` to use a `GRASP_WIDTH_CLOSED` of `0.0136`. This accounts for the geom offset and provides the intended gripping force.
2.  **Soften Contacts:** Modify `chess_env/assets/pick_and_place.xml`. Change the `solref` from `0.005` to `0.02` and `solimp` to `0.8 0.95 0.001`. This allows the simulation to "absorb" high-frequency jitter without exploding.
3.  **Tighten Settlement:** Reduce `settle_tolerance` in `configs/physics.yaml` from `0.003` to `0.0005`. This ensures the arm is actually over the cube before the grasp phase begins.

## Phase 2: Reward Surface Refinement
If retraining is still desired to improve movement quality, we recommend a more robust reward shaping approach than the one suggested in Document 04.

1.  **Action Variance Penalty (Instead of Velocity):** 
    *   *Why:* Velocity penalties can make the arm sluggish. 
    *   *How:* Penalize $\sum |a_t - a_{t-1}|^2$. This directly targets high-frequency jitter (acceleration/jerk) while allowing for high-speed, smooth movement.
2.  **Center-Line Drift Penalty:**
    *   *How:* Apply a penalty based on the $L_2$ distance of the gripper from the `tube_center_xy` during `ascend` and `descend`.
    *   *Why:* This provides a smooth gradient for the model to stay centered, rather than just a terminal crash penalty.

## Phase 3: Fine-Tuning (The Cerebellum Update)
Once the physics and rewards are corrected, perform a short fine-tuning run (100k - 200k steps).

*   **Curriculum:** Use the corrected environment.
*   **Result:** The model will learn a new "motor primitive" for lifting that is inherently smooth and centered, utilizing the now-functional physical contacts.

## Summary Checklist
- [ ] Fix geom offset in `task.py`
- [ ] Update `solref/solimp` in XML
- [ ] Set `settle_tolerance` to 0.5mm
- [ ] Replace Velocity Penalty with Action Variance Penalty
- [ ] Run 200k step fine-tuning
