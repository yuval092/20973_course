# Scripts Usage: Command Line Reference

This document describes the unified entry points in the `scripts/` directory.

## 1. `train.py`
The primary interface for training the SAC model. It handles vectorized environments and automatic checkpoint resumption.

**Example:**
```bash
python scripts/train.py --envs 12 --fresh
```

*   `--envs`: Number of parallel worker processes.
*   `--fresh`: Discards existing checkpoints and starts from base weights.
*   `--debug`: Enables verbose environment logs (not recommended for long training).

## 2. `eval.py`
Quantitative performance measurement tool. Runs the model on specific scenarios and generates a success report.

**Example:**
```bash
python scripts/eval.py --scenarios transit,descend --n-episodes 50
```

*   `--model`: Path to a specific `.zip` model (defaults to `models/latest_model.zip`).
*   `--visualize`: Opens a MuJoCo window to watch the evaluation.
*   `--debug`: Prints per-step rewards and coordinates to the console.

## 3. `verify_physics.py`
Automated "Smoke Test" for the system. Runs 4 critical tests to ensure the physics and XMLs are healthy.

**Example:**
```bash
python scripts/verify_physics.py --debug
```

Tests include:
1.  **XML Integrity:** Assets load and key sites exist.
2.  **Static Stability:** A piece on the table doesn't move when the arm is idle.
3.  **Kinematic Reachability:** The arm can reach its high and low waypoints without collision.
4.  **Teleport Stability:** The object-hiding logic works correctly.

## 4. `visualize.py`
Qualitative rendering tool for debugging specific movements in slow-motion.

**Example:**
```bash
python scripts/visualize.py --scenario descend --delay 0.05
```

*   `--delay`: Adds a pause between steps (in seconds) for easier observation.
*   `--episodes`: Number of episodes to run (default is infinite loop).
*   `--debug`: Prints raw mocap action values and gripper positions.
