# RoboChess Scripts

This directory contains the primary entry points for interacting with the RoboChess fine-tuning system.

## Entry Points

### 1. `train.py`
Used to start or resume training sessions.

**Usage:**
```bash
python scripts/train.py [OPTIONS]
```

**Options:**
- `--envs`: (int) Number of parallel Gymnasium environments to run. Defaults to the value in `configs/training.yaml`.
- `--fresh`: (flag) If set, ignores existing checkpoints and starts training from the base model weights.
- `--timesteps`: (int) Override the total number of timesteps to train.
- `--config`: (str) Path to a specific training configuration file (future support).
- `--debug`: (flag) Enable verbose debug logging in the environment.

---

### 2. `eval.py`
Comprehensive evaluation tool for measuring agent performance across different movement phases.

**Usage:**
```bash
python scripts/eval.py [OPTIONS]
```

**Options:**
- `--model`: (str) Path to the `.zip` model file. Defaults to `models/latest_model.zip`.
- `--scenarios`: (list) Comma-separated scenarios (e.g., `transit,descend`) or `all`.
- `--n-episodes`: (int) Number of episodes to run per scenario. Default: 100.
- `--visualize`: (flag) If set, opens a MuJoCo window to render the evaluation.
- `--deterministic`: (flag) Use deterministic actions. Default: True.
- `--debug`: (flag) Enable verbose debug logging in the environment.

**Output:**
Prints a summary table with success rates and average rewards for each scenario.

---

### 3. `verify_physics.py`
Diagnostic script to ensure the simulation environment is healthy. It runs checks on:
- XML model compilation.
- Static stability (no-action drift check).
- Gripper kinematic reachability.
- Teleportation stability (hiding the object correctly).

**Usage:**
```bash
python scripts/verify_physics.py [OPTIONS]
```

**Options:**
- `--debug`: (flag) Enable verbose physics logging during tests.

---

### 4. `visualize.py`
A simple utility for visual inspection of the agent's behavior. Unlike `eval.py`, this script runs a continuous loop on a single scenario for observation.

**Usage:**
```bash
python scripts/visualize.py [OPTIONS]
```

**Options:**
- `--model`: (str) Path to the model. Defaults to `models/latest_model.zip`.
- `--scenario`: The scenario to visualize (e.g., `descend`).
- `--delay`: (float) Delay in seconds between steps (for slow-motion observation).
- `--debug`: (flag) Enable verbose debug logging in the environment.
