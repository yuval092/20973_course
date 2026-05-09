# RoboChess Fine-Tuning

This project provides a robust, configuration-driven Reinforcement Learning environment for fine-tuning a Fetch robotic arm to manipulate chess pieces using MuJoCo and Stable Baselines 3.

## 🚀 Quick Start
1.  **Verify Setup:** `python scripts/verify_physics.py`
2.  **Run Evaluation:** `python scripts/eval.py --visualize`
3.  **Watch Agent:** `python scripts/visualize.py --scenario transit --delay 0.05`

## 📂 Project Structure
*   `configs/`: Centralized YAML configurations (`env.yaml`, `physics.yaml`, `training.yaml`).
*   `src/`: Core logic (Environment definitions, Trainer, Utilities).
*   `scripts/`: Unified command-line entry points.
*   `docs/`: Detailed technical documentation.
*   `models/`: Pretrained models and training saves.

## 📖 Documentation
For a deep dive into the system, please refer to the following documents:

1.  **[System Architecture](docs/01_system_architecture.md)**: High-level design and the "Holding Object" trick.
2.  **[Physics and Kinematics](docs/02_physics_and_kinematics.md)**: Absolute coordinates, altitudes, and drift limits.
3.  **[Environment Specification](docs/03_environment_specification.md)**: Details on reset logic and observation space.
4.  **[Reward Shaping](docs/04_reward_shaping.md)**: Explanation of the dense reward formula and curriculum.
5.  **[Configuration Guide](docs/05_configuration_guide.md)**: Master reference for all YAML parameters.
6.  **[Scripts Usage](docs/06_scripts_usage.md)**: Detailed command-line flag reference.

## 🛠 Features
*   **Standardized Model:** Automatically uses `models/latest_model.zip`.
*   **Robust Debug Mode:** Use the `--debug` flag on any script for verbose logging.
*   **Safety Guaranteed:** Implicit "Virtual Tubes" and "Virtual Floors" prevent collisions.
*   **Drift Curriculum:** Linear tightening of horizontal constraints to stabilize learning.
