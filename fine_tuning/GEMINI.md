# RoboChess Fine-Tuning: Gemini Context

This project focuses on fine-tuning a robotic movement engine (Fetch arm) for high-precision chess piece manipulation within a MuJoCo simulation. It builds upon `gymnasium-robotics` and uses `stable-baselines3` (SAC) for reinforcement learning.

## Project Overview

*   **Goal:** Refine a "Cerebellum" (movement engine) to move blocks on a 64cm x 64cm chess board with >98% success rate and <8mm terminal error.
*   **Technologies:** Python, MuJoCo, Gymnasium Robotics, Stable Baselines3 (SAC), NumPy.
*   **Architecture:**
    *   **Custom Environments:** `ChessFetch-v0` and `ChessFetchDense-v0` (registered in `chess_env/`).
    *   **Hybrid Control:** A waypoint-based state machine (Hover -> Descend -> Grasp -> Lift -> Transit -> Deliver) combined with an RL-trained movement policy.
    *   **Reward Strategy:** Transitioning from sparse rewards to a complex dense reward function (Success + Accuracy + Smoothness + Dampening - Collision) for surgical refinement.

## Key Directories and Files

*   `chess_env/`: Core environment implementation.
    *   `chess_fetch_env.py`: Custom board geometry and uniform sampling logic.
    *   `chess_fetch_dense_env.py`: Implements the dense reward function specified in Phase 8.
    *   `assets/`: Modified MuJoCo XML files for the robot and board.
*   `scripts/`: Automation and utility scripts.
    *   `fine_tune.py`: Standard fine-tuning script.
    *   `fine_tune_dense.py`: "Style Transfer" script for dense reward refinement using a low learning rate.
    *   `visual_inspect.py`: Tool for manual verification and rendering.
    *   `verify_env.py` / `verify_model.py`: Diagnostic scripts for validating environment physics and model performance.
*   `docs/`:
    *   `phase8_hybrid_architecture.md`: The "Source of Truth" for physical constants, waypoint logic, and reward formulas.
*   `models/`: Pretrained weights and fine-tuned checkpoints.

## Building and Running

### Setup
```bash
# Recommended to use the existing .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training
*   **Standard Fine-Tuning:**
    ```bash
    python scripts/fine_tune.py
    ```
*   **Dense Reward Refinement:**
    ```bash
    python scripts/fine_tune_dense.py
    ```

### Evaluation & Visualization
*   **Visual Inspection:**
    ```bash
    python scripts/visual_inspect.py
    ```
*   **Check Coordinate Mapping:**
    ```bash
    python scripts/coordinate_mapping.py
    ```
*   **Training Visualizer:**
    ```bash
    python scripts/training_visualizer_v2.py
    ```

## Development Conventions

*   **Environment Constants:** Always refer to `docs/phase8_hybrid_architecture.md` for ground-truth values (Table Z = 0.400, Board Center = [0.880, 0.264]).
*   **Checkpoints:** Use `SuccessRateEvalCallback` to save the best models based on success rate rather than raw reward.
*   **Safety First:** Any collision with the table surface (Z < 0.405) should result in terminal failure and high penalty.
*   **Determinism:** Ensure `np_random` is used for all sampling within environments to maintain seed reproducibility.
