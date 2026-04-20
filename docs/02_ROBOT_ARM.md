# 02 ROBOT ARM: Control, Policy, and Workspace

The Fetch robotic arm is the primary actor in RoboChess. Understanding how it moves and why it is constrained is key to understanding the project's physical behavior.

## 🤖 The Hardware: Fetch Manipulator
- **Type**: 7-Degree of Freedom (7-DOF) robotic arm.
- **End-Effector**: A parallel-jaw gripper. It has two fingers driven by Position PID actuators (`kp="30000"`).
- **Mocap & IK Control**: While the arm has 7 physical joints, we do not actuate them directly with motors. Instead, the simulation features an invisible, magical "Mocap" (Motion Capture) body tracking at the tip of the gripper. A rigid **Weld Equality Constraint** is defined between the Mocap body and the robot's end-effector. 

To move the arm to a Cartesian coordinate (X,Y,Z), the `ExecutionController` teleports the invisible Mocap body. The underlying MuJoCo engine mathematically calculates the forces required on all 7 joints to pull the heavy physical arm towards the Mocap target, essentially solving Inverse Kinematics recursively.

### Key Concept: Joint Space vs. Cartesian Space
- **Joint Space**: The specific native rotational angles of the 7 mechanical motors.
- **Cartesian Space**: A physical point in 3D scale (X, Y, Z). Our logic exclusively works in Cartesian space and relies on the Mocap Weld constraint to handle the joint space translation.

## 🧠 The "Brain": TQC Reinforcement Learning
We use a pretrained **Truncated Quantile Critics (TQC)** model. TQC is a state-of-the-art RL algorithm that is particularly good at "goal-conditioned" tasks like picking up objects.

### 1. Goal-Conditioned RL
The policy takes a "Desired Goal" (where we want the piece to go) and the "Achieved Goal" (where the piece is now) as inputs. It then outputs the velocity actions needed to minimize the distance between them.

### 2. The Observation Vector (26 Dimensions)
To make a decision, the RL model looks at:
- **Robot State**: Position and velocity of the gripper fingers.
- **Object State**: Position, velocity, and rotation of the chess piece.
- **Relative State**: Distance between the gripper and the piece.

### 3. The Action Space (4 Dimensions)
The model outputs 4 numbers every step:
- **[X, Y, Z]**: Target displacement for the gripper.
- **[Gripper]**: Open (>0) or Closed (<0).

## 📍 The "Sweet Spot": Policy Workspace
A critical discovery in this project is that the RL policy was trained in a specific area relative to the robot's base.

- **Policy Center**: `[1.3419, 0.7491, 0.575]` (The robot's home position).
- **Policy Radius**: `0.15 meters`.
- **Vertical Range**: `0.41m to 0.75m`.

### Board Reachability
Because the chessboard is large and the policy radius is small, only a subset of squares can be moved using the physical arm:
- **The Sweet Spot**: File **F**, Ranks **3, 4, 5, and 6**.
- **Fallbacks**: For moves outside this 0.15m radius, the `GameOrchestrator` automatically uses **Teleportation** to maintain game flow without crashing the physics engine.

## 🚀 Advanced Movement: Guided Actions
The `ExecutionController` doesn't just blindly follow the RL model. It uses **Guided Actions**:
1. It calculates a "scripted" direction towards the goal.
2. It queries the RL model for its suggested action.
3. If the RL action aligns with the scripted goal, it applies a blend of both.
4. If the RL model suggests moving away from the goal, the controller overrides it with the scripted path. 
*This prevents the robot from performing erratic "spinning" motions sometimes seen in RL policies.*
