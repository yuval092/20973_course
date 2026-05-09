# RoboChess Fine-Tuning: Instructional Context

This project is a Reinforcement Learning (RL) framework for fine-tuning a Fetch robotic arm to perform high-precision chess piece manipulation within a MuJoCo simulation.

## 🎯 Project Overview
The goal is to move a robotic arm between board positions without collisions, handling vertical transitions (`ascend`/`descend`) and horizontal movement (`transit`). It uses **Stable Baselines 3**'s implementation of **Soft Actor-Critic (SAC)**.

### Core Stack
- **Language:** Python
- **Physics Engine:** MuJoCo (via `gymnasium-robotics`)
- **RL Framework:** Stable Baselines 3 (SAC)
- **Configuration:** YAML-driven (`configs/`)

## 🏗 System Architecture
The system employs a **Hybrid Architecture** where a high-level Python State Machine handles task sequencing, and the RL model acts as the "cerebellum" for point-to-point movement.

### The "Holding Object" Trick
To simplify learning, the environment observation space is modified to map `object_pos` directly to the `grip_pos`. This tricks the pretrained RL model into thinking it is always carrying a payload, leveraging its stable "Phase 2" transport logic.

### Specialized Scenarios
Training is divided into three logical phases to maximize success rates:
1.  **Transit:** Horizontal movement at a safe cruise altitude.
2.  **Descend:** Precise vertical movement to the board surface.
3.  **Ascend:** Vertical lift after a piece is "grasped".

## 🚀 Key Commands

### Setup & Verification
```bash
# Verify MuJoCo and physics constants
python scripts/verify_physics.py
```

### Training
```bash
# Train with default settings (auto-detects models/latest_model.zip)
python scripts/train.py --envs 12

# Start a fresh training run (ignoring existing models)
python scripts/train.py --fresh --timesteps 1000000
```

### Evaluation & Visualization
```bash
# Run a full evaluation suite with visualization
python scripts/eval.py --visualize

# Watch the agent in a specific scenario
python scripts/visualize.py --scenario transit --delay 0.05
```

## 📂 Directory Structure
- `src/`: Core logic.
    - `chess_env/`: Custom Gymnasium environment (`ChessFetchTask-v0`).
    - `training/`: `SACTrainer` and RL-specific logic.
    - `utils/`: Configuration loaders and physics utilities.
- `scripts/`: Unified command-line entry points.
- `configs/`: Centralized YAML configurations.
    - `env.yaml`: Observation space and reward parameters.
    - `physics.yaml`: Table heights, gripper widths, and limits.
    - `training.yaml`: RL hyperparameters.
- `docs/`: Technical deep-dives (Architecture, Reward Shaping, etc.).
- `checkpoints/`: Timestamped training outputs and best models.

## 🛠 Development Conventions
- **Config First:** Never hardcode physical constants; use `configs/physics.yaml`.
- **Surgical Changes:** When modifying the environment, ensure compatibility with the "Holding Object" mapping in `src/chess_env/task.py`.
- **Validation:** Always run `scripts/verify_physics.py` after modifying `physics.yaml` to ensure simulation stability.
