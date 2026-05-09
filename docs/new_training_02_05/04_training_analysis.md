# Training Progress Analysis (May 2nd Refinement)

## The "5mm Crisis" (Lessons Learned)
During the initial V6 run, we attempted a **5mm** precision target.
*   **The Problem**: At Step 500,000, the curriculum narrowed to 5mm. Because the model was not penalized for XY drift during the first 400k steps, it developed "lazy" centering habits.
*   **The Result**: The model hit a "Precision Wall." Success rates collapsed as it could not physically survive the 5mm tube without an explicit centering reward.
*   **The Fix**: We identified that 10mm is more than sufficient for chess cells. We added an `XY_REWARD_WEIGHT: 2.0` to explicitly teach the model to stay centered.

## Live Status (New Refined Run)
The refined training run is designed to avoid the "Precision Wall" by teaching centering from Step 1.

### 1. Success Rate Expectations
*   **Vertical Scenarios**: Should maintain near **100% success** throughout the curriculum, as the `XY_REWARD` will pull the arm into the center long before the tube becomes tight.
*   **Transit Scenario**: Expected success rate of **90%+** at the 10mm standard.

### 2. Forensic Log Analysis
Refined debugging now monitors:
*   **XY Drift Penalty**: Verified that `reward` is now slightly lower when off-center, providing the necessary gradient for learning.
*   **10mm Boundary**: `TUBE_BREACH` is now calibrated to a physically achievable 1cm target.

## Why Refined Models Succeed
By aligning the **Reward Function** with the **Curriculum Boundary**, the model no longer "wanders" blind. It actively seeks the center of the cell, making the 10mm certification a natural outcome of the learning process rather than a fatal obstacle.

## System Constraints & Stability
On machines with 8GB RAM, 15 parallel environments plus a 1M-step replay buffer can trigger the OOM killer. We have optimized the training for stability:
*   **Worker Count**: Reduced to **8** environments.
*   **Buffer Size**: Optimized to **300,000** transitions to fit within physical RAM while maintaining learning quality.

---
*Next: [Current Project Status & Roadmap](./05_project_status.md)*
