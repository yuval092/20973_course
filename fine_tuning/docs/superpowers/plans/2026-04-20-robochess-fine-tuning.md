# RoboChess Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune a SAC+HER policy to move a single chess piece on an enlarged board in a custom MuJoCo environment.

**Architecture:** Extend `FetchPickAndPlaceEnv` from `gymnasium-robotics`, enlarging the table in XML and overriding the goal/object sampler. Fine-tune using `stable-baselines3` from a pretrained checkpoint.

**Tech Stack:** Python, MuJoCo, Gymnasium-Robotics, Stable-Baselines3, Hugging Face Hub, NumPy.

---

## Status Tracking

- [x] **Phase 1: Environment Setup**
- [x] **Phase 2: Resize Board**
- [x] **Phase 3: Custom Environment Class**
- [x] **Phase 4: Verification**
- [x] **Phase 5: Zero-Shot Test**
- [x] **Phase 6: Fine-Tuning**
- [x] **Phase 7: Final Evaluation**

---

### Phase 1: Environment Setup

**Files:**
- Create: `chess_env/__init__.py` (empty for now)
- Create: `chess_env/assets/` directory
- Create: `scripts/download_model.py`

- [ ] **Step 1: Download the pretrained model**
  
```python
import os
from huggingface_hub import hf_hub_download

# Create directories
os.makedirs('models', exist_ok=True)

# Download sac-FetchPickAndPlace-v4.zip
model_path = hf_hub_download(
    repo_id='IntelliGrow/FetchPickAndPlace-v4',
    filename='sac-FetchPickAndPlace-v4.zip',
    local_dir='models'
)
print(f'Model downloaded to: {model_path}')
```

Run: `python scripts/download_model.py`
Expected: `models/sac-FetchPickAndPlace-v4.zip` exists.

- [ ] **Step 2: Verify model compatibility**

```python
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC

gym.register_envs(gymnasium_robotics)
env = gym.make('FetchPickAndPlace-v4')
model = SAC.load('models/sac-FetchPickAndPlace-v4.zip', env=env)
print('Action space:', env.action_space)
print('Obs space:   ', env.observation_space)
env.close()
print('Model loaded successfully.')
```

Run: `python scripts/verify_model.py`
Expected: "Model loaded successfully." printed.

- [ ] **Step 3: Copy XML asset**

```bash
python -c "import gymnasium_robotics, os, shutil; base = os.path.dirname(gymnasium_robotics.__file__); src = os.path.join(base, 'envs', 'assets', 'fetch', 'pick_and_place.xml'); os.makedirs('chess_env/assets', exist_ok=True); shutil.copy(src, 'chess_env/assets/pick_and_place.xml'); print('Copied XML to chess_env/assets/')"
```

Run the command.
Expected: `chess_env/assets/pick_and_place.xml` exists.

---

### Phase 2: Resize Board

**Files:**
- Modify: `chess_env/assets/pick_and_place.xml`

- [ ] **Step 1: Update table size in XML**
Find `<geom name="table0" ... size="0.25 0.35 0.02" ... />` and change to `size="0.25 0.25 0.02"`. (Note: XML might use `table0` or `table`).

- [ ] **Step 2: Visual inspection**

```python
import mujoco
import mujoco.viewer
import os

xml_path = os.path.abspath('chess_env/assets/pick_and_place.xml')
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    import time
    start = time.time()
    while viewer.is_running() and time.time() - start < 5:
        mujoco.mj_step(model, data)
        viewer.sync()
```

Run: `python scripts/visual_inspect.py`
Expected: Viewer opens showing the table and robot.

---

### Phase 3: Custom Environment Class

**Files:**
- Create: `chess_env/chess_fetch_env.py`
- Modify: `chess_env/__init__.py`

- [ ] **Step 1: Implement ChessFetchEnv**
Implement the class as described in Step 3.3 of the game plan, ensuring all 4 fixes (monkey-patch, joint indexing, goal guard, mj_forward) are included.

- [ ] **Step 2: Register environment**
Add registration code to `chess_env/__init__.py`.

```python
from gymnasium.envs.registration import register

register(
    id='ChessFetch-v0',
    entry_point='chess_env.chess_fetch_env:ChessFetchEnv',
    max_episode_steps=100,
)
```

---

### Phase 4: Verification

**Files:**
- Create: `scripts/verify_env.py`

- [ ] **Step 1: Run Gymnasium check_env**
- [ ] **Step 2: Verify goal boundaries**
- [ ] **Step 3: Verify arm reachability (>= 90%)**

---

### Phase 5: Zero-Shot Test

**Files:**
- Create: `scripts/zero_shot_eval.py`

- [ ] **Step 1: Run evaluation**
Evaluate `sac-FetchPickAndPlace-v4.zip` on `ChessFetch-v0` for 100 episodes. Record success rate.

---

### Phase 6: Fine-Tuning

**Files:**
- Create: `scripts/fine_tune.py`

- [ ] **Step 1: Setup SuccessRateEvalCallback**
- [ ] **Step 2: Configure SAC with learning_rate=1e-4**
- [ ] **Step 3: Run learning for 300,000 steps**

---

### Phase 7: Final Evaluation

**Files:**
- Create: `scripts/final_eval.py`

- [ ] **Step 1: Evaluate best model**
- [ ] **Step 2: Generate per-region breakdown report**
