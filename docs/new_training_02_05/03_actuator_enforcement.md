# Absolute Actuator Enforcement & Scripted Transitions

## The "Software Cerebellum" Concept
In this architecture, the RL model (the "Cerebellum") is responsible only for **point-to-point movement**. It is NOT responsible for gripper manipulation. The high-level Python State Machine handles all gripper logic.

## Technical Implementation
We implemented **Absolute Enforcement** at the MuJoCo simulation level (`simulation.py`).

### 1. Scripted Transitions (The Reset Phase)
During the environment reset, a 50-step loop is executed:
*   **DESCEND**: Starts Closed $\rightarrow$ Scripts to **Open** (0.0181 joint).
*   **ASCEND**: Starts Open $\rightarrow$ Scripts to **Closed** (0.0000 joint).
*   **TRANSIT**: Remains **Closed**.

During these 50 steps, the arm is held still while the fingers move. This ensures the physics engine "settles" and velocities are zeroed before Step 1 of the RL phase.

### 2. SIM-Level Locking
During every step of the RL episode, the `_set_action` method forces the finger joints:
```python
# Force the joint position directly (Deterministic Locking)
self._utils.set_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint", target_qpos)
self._utils.set_joint_qpos(self.model, self.data, "robot0:r_gripper_finger_joint", target_qpos)

# Zero joint velocities for fingers
self._utils.set_joint_qvel(self.model, self.data, "robot0:l_gripper_finger_joint", 0.0)
self._utils.set_joint_qvel(self.model, self.data, "robot0:r_gripper_finger_joint", 0.0)
```
This makes the fingers **effectively immovable** by the RL policy, preventing "actuator creep" or jerky noise.

## Mechanical Fuses (Crash Detection)
If the arm collides with a piece or the table, the physics engine will attempt to push the fingers away from their target positions. We monitor this in `task.py`:
```python
# Terminate if fingers deviate > 3mm from enforced target
if abs(actual_finger_qpos - target_qpos) > 0.003:
    terminate_with_crash("FINGER_FAULT")
```
This serves as a reliable collision detector without needing complex tactile sensors.

---
*Next: [Training Progress Analysis](./04_training_analysis.md)*
