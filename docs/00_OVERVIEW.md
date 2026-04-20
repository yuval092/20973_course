# 00 OVERVIEW: Robotic Chess with MuJoCo

Welcome to **RoboChess**, a sophisticated integration of classical game logic, modern reinforcement learning (RL), and high-fidelity physics simulation. This project enables a **Fetch robotic arm** to physically play chess against a human or an AI engine (Stockfish) within the **MuJoCo** (Multi-Joint dynamics with Contact) physics environment.

## 🎯 Project Goals

The primary objective of RoboChess is to bridge the gap between "perfect" logical chess and the "messy" physical world. Unlike standard chess apps where moves are just bit-flips, in RoboChess:
- Pieces have mass, friction, and inertia.
- The robot must physically grasp, lift, and place pieces.
- High-gain forces or poor planning can knock over pieces, leading to physical "game over" states.

## 🧱 Key Concepts for Beginners

### 1. What is MuJoCo?
MuJoCo is a physics engine designed for robotics and biomechanics. Unlike "game" physics engines (like PhysX), MuJoCo is optimized for **accuracy**, continuous-time formulation, and rigid-body constraints. It handles complex contacts, joint properties, and multi-body dynamics with extreme precision. 

**Visual vs. Physical Geometries**: In RoboChess, you will see a beautiful environment with detailed meshes (like knights and bishops) and a stylized table. However, MuJoCo maintains a hidden "physical layer" consisting of simplified bounding shapes (like cylinders and box arrays) and bitwise collision masks. The graphics you see are often purely "holograms" covering an underlying geometric simulation.

### 2. The Fetch Robot
The robot used in this simulation is the **Fetch**, a mobile manipulator with a 7-degree-of-freedom (7-DOF) arm and a parallel-jaw gripper. It is a standard research robot. In RoboChess, we focus on the arm's ability to reach specific coordinates (Cartesian control) and manipulate pieces.

### 3. Reinforcement Learning (RL) & TQC
The robot doesn't just follow a hard-coded path for every move. We use a pretrained RL model called **TQC (Truncated Quantile Critics)**. 
- **The Policy**: Think of this as the "brain" that has practiced picking up blocks millions of times in simulation.
- **Goal-Conditioned**: We give the policy a target (e.g., "move piece X to square E4"), and it outputs the tiny motor forces needed to get there.

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- MuJoCo (`pip install mujoco`)
- Gymnasium Robotics (`pip install gymnasium-robotics`)
- Chess logic (`pip install python-chess`)

### Execution
To start the interactive GUI:
```bash
python3 main.py
```

### Controls
1.  **Human Move**: Click a piece on the board, then click the destination square.
2.  **AI Move**: The robot will automatically move for the Black pieces.
3.  **Step-by-Step Mode**: Enable this to manually trigger the robot's next logical turn or physics step.
4.  **Hint**: Asks Stockfish for the best move in the current position.

## 🛠 Project Structure
-   `src/`: The core source code (refactored for OOP and PEP8).
-   `src/settings/runtime.yaml`: The single source of truth for the spatial configuration, physics tolerances, and board geometry (loaded via `src/config.py`). Everything from table height to gripper parameters scales off this file.
-   `scripts/`: Utilities for XML generation (`generate_xml.py`) and scene verification.
-   `tests/`: Comprehensive test suites verifying logic and physical boundary integrity.
-   `docs/`: Detailed technical documentation (which you are reading now).
