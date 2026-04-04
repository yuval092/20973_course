# RoboChess Fine-Tuning — Full Implementation Specification

> **Document type:** Implementation handoff for the fine-tuning agent  
> **Companion to:** `FINETUNE_FETCH_CHESS_PLAN.md`, `High-Level-Design.md`, `Low-Level-Design.md`  
> **Scope:** Everything needed to go from zero training code to a deployed chess-manipulation policy, including cloud training options for underpowered hardware, every known edge case, and recovery procedures for common failure modes

---

## Table of Contents

1. [Why Fine-Tuning Is Necessary — Evidence From Logs](#1-why-fine-tuning-is-necessary)
2. [Repository Changes Required Before Training](#2-repository-changes-required-before-training)
3. [Training Environment Design](#3-training-environment-design)
4. [Observation Vector Specification](#4-observation-vector-specification)
5. [Action Space Specification](#5-action-space-specification)
6. [Reward Function Specification](#6-reward-function-specification)
7. [Curriculum Design](#7-curriculum-design)
8. [Domain Randomization](#8-domain-randomization)
9. [Reset and Episode Generation](#9-reset-and-episode-generation)
10. [Training Procedure](#10-training-procedure)
11. [Evaluation Protocol](#11-evaluation-protocol)
12. [Integration Back Into Main Pipeline](#12-integration-back-into-main-pipeline)
13. [File Manifest and Deliverables](#13-file-manifest-and-deliverables)
14. [First Milestone Definition](#14-first-milestone-definition)
15. [Compute Options — Training on a Weak PC or in the Cloud](#15-compute-options)
16. [Platform-Specific Setup Guides](#16-platform-specific-setup-guides)
17. [Runtime and Environment Edge Cases](#17-runtime-and-environment-edge-cases)
18. [Training Stability and Divergence Recovery](#18-training-stability-and-divergence-recovery)
19. [Checkpoint and Compatibility Edge Cases](#19-checkpoint-and-compatibility-edge-cases)
20. [Experiment Tracking](#20-experiment-tracking)
21. [Debugging and Profiling](#21-debugging-and-profiling)

---

## 1. Why Fine-Tuning Is Necessary — Evidence From Logs

Before writing any code, the agent must understand **why** the current setup fails. Every log file (logs1–logs28) contains the same class of warning on every single AI turn:

```
WARNING: OUT_OF_POLICY_WORKSPACE:
  src_xy_dist=0.319m exceeds trained radius 0.150m
  src_z=0.413m outside trained z-range [0.420, 0.750]
  dest_xy_dist=0.269m exceeds trained radius 0.150m
  dest_z=0.413m outside trained z-range [0.420, 0.750]
```

The `FetchPickAndPlace-v4` training environment places the object within approximately 15 cm of the arm's rest position. The RoboChess board spans 40 × 40 cm, meaning that:

- **All 64 squares are out-of-distribution** from the perspective of source distance
- **Piece z-height (≈ 0.413 m) is below the policy's trained floor (0.420 m)** by 7 mm — enough to prevent gripper-piece contact on descent
- The gripper slides off the target during `CLOSE_GRIPPER_ONLY` (see `xy_dist=301.3mm` in logs27), which is a direct consequence of the policy producing actions appropriate for a nearby object but applied to a piece that is geometrically far from its expectation

Additionally, graveyard cells at x ≈ 0.82 are outside the physical reachability envelope `[0.90, 1.48]`. The system currently falls back to teleportation for those. Fine-tuning will not help unreachable cells — those must first be **repositioned** (see Section 2.2).

No amount of scripted wrapper code will fix a distribution mismatch this large. The policy must be trained on data that looks like the actual RoboChess scene.

---

## 2. Repository Changes Required Before Training

These are blocking prerequisites. Do not begin training until all of them are resolved.

### 2.1 Fix the 7 mm Table Height Gap

**Problem:** Chess pieces rest at z ≈ 0.413 m. The policy's trained grasp z-range floor is 0.420 m. The gripper never makes contact.

**Fix:** In `src/assets/chess_world.xml` (or the `scripts/generate_xml.py` script that produces it), raise the chessboard surface by **8–10 mm**. Target piece rest height: z ≈ 0.422–0.424 m. This brings all 64 squares inside the trained z-range floor with a small margin.

Also update `config.py`:
```python
# Before
TABLE_HEIGHT = 0.413          # m, current piece rest Z
# After
TABLE_HEIGHT = 0.422          # m, after board height fix
GRASP_HEIGHT = TABLE_HEIGHT + 0.015  # 15 mm above rest = descend target
```

Verify with `scripts/headless_run.py` that a single-piece pick succeeds after this change before proceeding.

### 2.2 Move Graveyard Cells Inside the Reachable Workspace

**Problem:** Current graveyard origin at x ≈ 0.82 is outside `reach_x_min = 0.90`. The arm cannot reach it.

**Fix:** In `config.py`, redefine graveyard layout so all 16 cells (8 per side) fall within the reachable envelope:

```python
# Workspace envelope (from logs): x=[0.90, 1.48], y=[0.30, 0.95]
# Chessboard occupies approximately x=[1.005, 1.405], y=[0.435, 0.835]
# Available space: x=[0.90, 1.005) left strip, and (1.405, 1.48] right strip

WHITE_GRAVEYARD_ORIGIN = np.array([0.92, 0.44, TABLE_HEIGHT])  # left strip
BLACK_GRAVEYARD_ORIGIN = np.array([0.92, 0.63, TABLE_HEIGHT])  # left strip, shifted Y
GRAVEYARD_COLS = 4
GRAVEYARD_ROWS = 4
GRAVEYARD_SPACING = 0.032   # 32 mm between cells, tight but fits in 0.90–1.00 strip
```

If the left strip is too narrow to fit 4 columns, use a 2 × 8 layout instead. The constraint is strict: **every graveyard cell must be reachable by the scripted waypoint controller before fine-tuning begins**.

### 2.3 Unify Collision and Visual Geometry in scripts/generate_xml.py

**Problem:** The scene currently has visible cylinder geoms and separate hidden collision proxies. The policy learns contact dynamics on the hidden shapes but receives observations from the visible shapes. This is an unnecessary domain gap.

**Fix in `scripts/generate_xml.py`:**
- Use a **single geom per piece** that serves as both the visual representation and the collision geometry
- Keep cylinder primitives (not meshes) — they are sufficient and contacts are well-conditioned
- Confirm that `condim=4`, friction, and damping settings are identical for all piece bodies
- Remove any `group="1"` / `group="2"` separation used to hide collision proxies from the viewer

After the change, run a quick simulation (no policy needed) and verify:
1. Pieces remain upright with no solver violations in the console
2. `up_z` reported by `board_observer.py` is ≥ 0.999 at start for all 32 pieces
3. The scripted controller can still complete a simple move

### 2.4 Validate That the Existing `ChessPickPlaceEnv` Observation Is Correct

Before adding any training code, unit-test the observation vector:

```python
# tests/test_env_obs.py
import mujoco, numpy as np
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.config import SCENE_XML, N_SUBSTEPS

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)

env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
env.set_target("w_pawn_1", np.array([1.075, 0.715, 0.422]))
obs = env._get_obs()

assert obs["observation"].shape == (25,), f"Expected (25,), got {obs['observation'].shape}"
assert obs["achieved_goal"].shape == (3,)
assert obs["desired_goal"].shape == (3,)
assert not np.any(np.isnan(obs["observation"])), "NaN in observation"
```

If any assertion fails, fix the env before writing training code.

---

## 3. Training Environment Design

### 3.1 Architecture Decision

Do **not** train through `main.py`. Create a self-contained environment that:
- resets to a random manipulation episode
- is runnable headlessly in parallel
- exposes the full SB3 GoalEnv interface
- produces dense reward each step

The existing `ChessPickPlaceEnv` is an inference wrapper — it exposes `set_target()` to be driven by the execution controller. The training environment needs to own the episode lifecycle.

### 3.2 File: `src/env/chess_manipulation_train_env.py`

```python
"""
ChessManipulationTrainEnv
─────────────────────────
A standalone GoalEnv for fine-tuning the Fetch pick-and-place policy
on the RoboChess scene. This environment owns the full episode lifecycle:
  - reset() samples a random legal manipulation episode
  - step() applies the Fetch action and returns dense reward
  - compute_reward() is separable (required for HER)

It does NOT use the staged scripted controller. The policy is expected
to output raw Cartesian deltas + gripper command, end-to-end, just as
it was trained in FetchPickAndPlace-v4.

Observation:
  A concatenation of the base 25-dim Fetch obs (preserved for checkpoint
  warm-start compatibility) and an 18-dim chess-scene context vector.
  Total: 43-dim observation.

Goal:
  achieved_goal: current XYZ of the target piece (3-dim)
  desired_goal:  target XYZ where the piece should be placed (3-dim)
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from src.config import (
    TABLE_HEIGHT, GRASP_HEIGHT, BOARD_ORIGIN, SQUARE_SIZE,
    REACH_X, REACH_Y, REACH_Z, N_SUBSTEPS, SCENE_XML,
    PIECE_NAMES, PIECE_INITIAL_POS, GOAL_TOLERANCE
)
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.env.episode_sampler import EpisodeSampler


OBS_DIM      = 25   # base Fetch observation (preserved for warm-start)
CONTEXT_DIM  = 18   # chess-scene context (see Section 4)
TOTAL_OBS    = OBS_DIM + CONTEXT_DIM  # 43


class ChessManipulationTrainEnv(gym.Env):
    """
    Wraps the MuJoCo chess scene for RL training.
    Compatible with stable-baselines3 GoalEnv via a flat Dict obs space.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        curriculum_stage: int = 1,
        max_steps: int = 200,
        seed: int | None = None,
    ):
        super().__init__()
        self.curriculum_stage = curriculum_stage
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        # Load MuJoCo scene fresh for each env instance (for VecEnv safety)
        self._model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self._data  = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Inner env provides the obs builder and physics step
        self._env = ChessPickPlaceEnv(self._model, self._data, n_substeps=N_SUBSTEPS)

        # Episode sampler drives curriculum (see Section 7 and 9)
        self._sampler = EpisodeSampler(self._rng, curriculum_stage)

        # Spaces
        obs_low  = np.full(TOTAL_OBS, -np.inf, dtype=np.float32)
        obs_high = np.full(TOTAL_OBS, np.inf,  dtype=np.float32)
        goal_low  = np.array([REACH_X[0], REACH_Y[0], REACH_Z[0]], dtype=np.float32)
        goal_high = np.array([REACH_X[1], REACH_Y[1], REACH_Z[1]], dtype=np.float32)

        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(obs_low, obs_high, dtype=np.float32),
            "achieved_goal": spaces.Box(goal_low, goal_high, dtype=np.float32),
            "desired_goal":  spaces.Box(goal_low, goal_high, dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # Episode state
        self._step_count    = 0
        self._episode       = None   # EpisodeSampler.Episode namedtuple
        self._phase         = "approach"
        self._grasped       = False
        self._prev_piece_xy = None

    # ────────────────────────────────────────────────
    # Core GoalEnv interface
    # ────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample an episode from the curriculum
        self._episode = self._sampler.sample()

        # Reset physics to initial layout, then teleport the target piece
        # to the episode's source square
        self._reset_physics()
        self._teleport_piece(self._episode.piece_name, self._episode.src_pos)

        # Reset the inner Fetch env (arm back to home position, gripper open)
        self._env.set_target(self._episode.piece_name, self._episode.dest_pos)
        inner_obs = self._env._get_obs()

        self._step_count    = 0
        self._phase         = "approach"
        self._grasped       = False
        self._prev_piece_xy = self._get_piece_xy()

        obs = self._build_obs(inner_obs)
        return obs, {}

    def step(self, action: np.ndarray):
        # Apply one physics step through the inner env
        inner_obs, _, _, _, info = self._env.step(action)

        self._step_count += 1
        self._update_phase(inner_obs)

        obs        = self._build_obs(inner_obs)
        achieved   = obs["achieved_goal"]
        desired    = obs["desired_goal"]
        reward     = float(self.compute_reward(achieved, desired, info))
        terminated = self._is_success(achieved, desired)
        truncated  = self._step_count >= self.max_steps

        info.update({
            "is_success":        terminated,
            "phase":             self._phase,
            "grasped":           self._grasped,
            "placement_error_m": np.linalg.norm(achieved - desired),
            "up_z":              self._get_up_z(),
        })

        self._prev_piece_xy = self._get_piece_xy()
        return obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> float:
        """
        Dense shaped reward. Must be separable (no self state) for HER compatibility.
        See Section 6 for full specification.
        """
        # This is a stub — see Section 6 for the full implementation.
        # The actual reward function is implemented in reward_fn.py and
        # called here so it can also be imported standalone by HER.
        from src.env.reward_fn import compute_reward
        return compute_reward(achieved_goal, desired_goal, info)

    def render(self):
        pass  # Headless training. Use mujoco.viewer separately.

    # ────────────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────────────

    def _reset_physics(self):
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

    def _teleport_piece(self, piece_name: str, pos: np.ndarray):
        body = self._data.body(piece_name)
        body.xpos[:] = pos
        body.xquat[:] = [1, 0, 0, 0]   # upright
        # Zero velocity
        jnt_id = self._model.body(piece_name).jntadr[0]
        self._data.qvel[jnt_id * 6 : jnt_id * 6 + 6] = 0
        mujoco.mj_forward(self._model, self._data)

    def _build_obs(self, inner_obs: dict) -> dict:
        base_obs = inner_obs["observation"].astype(np.float32)      # (25,)
        context  = self._build_context().astype(np.float32)         # (18,)
        return {
            "observation":   np.concatenate([base_obs, context]),
            "achieved_goal": inner_obs["achieved_goal"].astype(np.float32),
            "desired_goal":  inner_obs["desired_goal"].astype(np.float32),
        }

    def _build_context(self) -> np.ndarray:
        # See Section 4 for the full 18-dim specification
        from src.env.context_builder import build_context
        return build_context(self._model, self._data, self._episode, self._grasped)

    def _update_phase(self, inner_obs: dict):
        piece_pos  = inner_obs["achieved_goal"]
        grip_pos   = inner_obs["observation"][0:3]
        piece_lift = piece_pos[2] - TABLE_HEIGHT

        if not self._grasped:
            if piece_lift > 0.01:          # 1 cm above table → grasped
                self._grasped = True
                self._phase   = "transport"
            else:
                self._phase   = "approach"
        else:
            self._phase = "placement"

    def _is_success(self, achieved: np.ndarray, desired: np.ndarray) -> bool:
        xy_err = np.linalg.norm(achieved[:2] - desired[:2])
        z_err  = abs(achieved[2] - desired[2])
        up_z   = self._get_up_z()
        return (xy_err < 0.010) and (z_err < 0.010) and (up_z >= 0.966)

    def _get_piece_xy(self) -> np.ndarray:
        if self._episode is None:
            return np.zeros(2)
        return self._data.body(self._episode.piece_name).xpos[:2].copy()

    def _get_up_z(self) -> float:
        if self._episode is None:
            return 1.0
        q = self._data.body(self._episode.piece_name).xquat
        return float(1.0 - 2.0 * (q[1]**2 + q[2]**2))
```

---

## 4. Observation Vector Specification

The total observation is **43-dimensional**: 25 preserved base dims + 18 chess-context dims.

### 4.1 Base 25-Dim Vector (Preserved From Pretrained Policy)

This must remain **identical** to the `FetchPickAndPlace-v4` layout. Any change breaks checkpoint warm-start.

| Indices | Content | Source in MuJoCo |
|---------|---------|-----------------|
| 0–2   | Gripper XYZ (end-effector site) | `data.site("robot0:grip_site").xpos` |
| 3–5   | Target piece XYZ | `data.body(piece_name).xpos` |
| 6–8   | Piece pos relative to gripper | indices 3:6 − indices 0:3 |
| 9–10  | Finger joint positions (L, R) | `data.qpos[finger_l_id]`, `data.qpos[finger_r_id]` |
| 11–13 | Piece orientation (Euler XYZ) | quaternion → Euler, `data.body(piece_name).xquat` |
| 14–16 | Piece linear velocity | `data.body(piece_name).cvel[3:6]` |
| 17–19 | Piece angular velocity | `data.body(piece_name).cvel[0:3]` |
| 20–22 | Gripper linear velocity | `data.site_xvelp("robot0:grip_site")` or equivalent |
| 23–24 | Finger joint velocities (L, R) | `data.qvel[finger_l_id]`, `data.qvel[finger_r_id]` |

### 4.2 Chess-Context 18-Dim Vector

This is appended after the base 25 dims. The policy's new layers will learn to use this. The pretrained layers still warm-start because their input is the same.

| Indices (in context) | Content | How Computed |
|---------------------|---------|-------------|
| 0–2 | Destination XYZ (desired goal, repeated for convenience in the context) | `episode.dest_pos` |
| 3–5 | Vector: piece → destination | `dest_pos − piece_pos` |
| 6   | Normalized board file of source square (0–1) | `(file_idx) / 7.0` |
| 7   | Normalized board rank of source square (0–1) | `(rank_idx) / 7.0` |
| 8   | Normalized board file of dest square (0–1) | `(file_idx) / 7.0` |
| 9   | Normalized board rank of dest square (0–1) | `(rank_idx) / 7.0` |
| 10  | Transport distance (src→dest, normalized) | `dist / 0.40` (0.40 m = max board diagonal) |
| 11  | `is_grasped` flag | 1.0 if piece lifted > 1 cm, else 0.0 |
| 12  | `is_capture` flag | 1.0 if episode involves capture else 0.0 |
| 13  | Piece type encoding (pawn=0, rook=1, knight=2, bishop=3, queen=4, king=5, normalized to 0–1) | `piece_type / 5.0` |
| 14–15 | Closest neighbor piece XY offset (from source square) | XY of nearest other piece body − src_pos XY, clamped to ±0.15 |
| 16–17 | Second-closest neighbor piece XY offset | Same, second nearest |

**Design rationale:**
- Indices 0–13 are cheap to compute and give the policy most of what it needs
- Indices 14–17 give local clutter awareness without encoding the full board
- Do not encode all 32 piece positions — that adds 96 dims for marginal gain in Stage 1–3

### 4.3 File: `src/env/context_builder.py`

```python
import numpy as np
import mujoco
from src.config import PIECE_NAMES, BOARD_ORIGIN, SQUARE_SIZE, TABLE_HEIGHT

PIECE_TYPE_MAP = {
    "pawn": 0, "rook": 1, "knight": 2,
    "bishop": 3, "queen": 4, "king": 5,
}

def build_context(model, data, episode, is_grasped: bool) -> np.ndarray:
    ctx = np.zeros(18, dtype=np.float64)

    piece_pos  = data.body(episode.piece_name).xpos.copy()
    dest_pos   = episode.dest_pos
    src_square = episode.src_square   # e.g. "e2"
    dst_square = episode.dst_square

    # 0–2: destination position
    ctx[0:3] = dest_pos

    # 3–5: piece → destination vector
    ctx[3:6] = dest_pos - piece_pos

    # 6–9: normalized board coordinates
    ctx[6]  = (ord(src_square[0]) - ord('a')) / 7.0
    ctx[7]  = (int(src_square[1]) - 1) / 7.0
    ctx[8]  = (ord(dst_square[0]) - ord('a')) / 7.0
    ctx[9]  = (int(dst_square[1]) - 1) / 7.0

    # 10: normalized transport distance
    ctx[10] = np.linalg.norm(dest_pos[:2] - episode.src_pos[:2]) / 0.40

    # 11: is_grasped
    ctx[11] = 1.0 if is_grasped else 0.0

    # 12: is_capture
    ctx[12] = 1.0 if episode.is_capture else 0.0

    # 13: piece type
    piece_type_str = episode.piece_name.split("_")[1]   # e.g. "pawn"
    ctx[13] = PIECE_TYPE_MAP.get(piece_type_str, 0) / 5.0

    # 14–17: two nearest neighbor offsets
    neighbors = _get_neighbor_offsets(data, episode.piece_name, piece_pos)
    ctx[14:16] = np.clip(neighbors[0], -0.15, 0.15) if len(neighbors) > 0 else 0.0
    ctx[16:18] = np.clip(neighbors[1], -0.15, 0.15) if len(neighbors) > 1 else 0.0

    return ctx.astype(np.float32)


def _get_neighbor_offsets(data, target_name, target_pos):
    dists = []
    for name in PIECE_NAMES:
        if name == target_name:
            continue
        pos = data.body(name).xpos.copy()
        if pos[2] < TABLE_HEIGHT - 0.05:    # below table = hidden spare, skip
            continue
        d = np.linalg.norm(pos[:2] - target_pos[:2])
        dists.append((d, pos[:2] - target_pos[:2]))
    dists.sort(key=lambda x: x[0])
    return [v for (_, v) in dists[:2]]
```

---

## 5. Action Space Specification

Keep the existing Fetch-compatible Cartesian delta action space **exactly**. Do not change it.

```
action[0] = Δx  (end-effector displacement, world frame, clipped to ±0.05 m per step)
action[1] = Δy
action[2] = Δz
action[3] = gripper command  (-1 = close, +1 = open)
```

**Action scaling note:** The existing `ChessPickPlaceEnv.step()` uses the same scaling as `FetchPickAndPlace-v4`. Do not change it. If the policy struggles to traverse large board distances, the solution is a longer episode horizon and curriculum (not scaling).

**Why not change it:** Any change to action semantics invalidates the warm-start from the pretrained checkpoint. The checkpoint is the most valuable asset in this pipeline. Preserve it.

---

## 6. Reward Function Specification

### 6.1 Phase-Gated Architecture

The reward function gates its terms based on the current manipulation phase. This prevents conflicting signals (e.g., the approach reward pulling the gripper away from the piece once it is grasped).

```
Phase = "approach"   → gripper not near piece, piece not lifted
Phase = "transport"  → piece lifted, not yet at destination
Phase = "placement"  → piece near destination, being lowered
```

### 6.2 File: `src/env/reward_fn.py`

```python
"""
Separable dense reward function for HER compatibility.

compute_reward() must not access env state. All information it needs
must be passed through the `info` dict. The training env populates
info with everything listed below.

Required info keys (populated by ChessManipulationTrainEnv.step):
  info["phase"]              : str, "approach" | "transport" | "placement"
  info["grasped"]            : bool
  info["grip_pos"]           : np.ndarray(3,)
  info["piece_pos"]          : np.ndarray(3,)  (= achieved_goal)
  info["up_z"]               : float
  info["finger_qpos"]        : np.ndarray(2,)
  info["neighbor_displaced"] : bool, True if any non-target piece moved > 5 mm
  info["piece_dropped"]      : bool, True if piece was lifted then fell below table + 5mm
  info["timed_out"]          : bool
"""

import numpy as np

# ─── Reward weights ───────────────────────────────────────────────
W_APPROACH       =  2.0    # reward per 1 cm gripper-to-piece reduction
W_GRASP_BONUS    = 10.0    # one-time bonus when grasp detected
W_LIFT_BONUS     =  5.0    # one-time bonus when piece lifted > 2 cm
W_TRANSPORT      =  3.0    # reward per 1 cm piece-to-dest reduction
W_PLACEMENT      = 50.0    # large bonus for final success
W_TILT_PENALTY   = -1.0    # per step while up_z < 0.966 during transport
W_DROP_PENALTY   = -20.0   # one-time, piece dropped after lift
W_CLUTTER_PEN    = -5.0    # one-time, displaced neighbor piece
W_TIMEOUT_PEN    = -2.0    # flat penalty on truncation
FINGER_CLOSED_THR = 0.005  # m, finger qpos below this = closed

# ─── Placement tolerance (must match is_success in env) ───────────
XY_TOL = 0.010   # m
Z_TOL  = 0.010   # m
UP_Z_THR = 0.966 # cos(15°) ≈ 0.966


def compute_reward(
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    info: dict,
) -> float:
    """
    Dense shaped reward. Separable — all state accessed via `info`.
    """
    phase    = info.get("phase", "approach")
    grasped  = info.get("grasped", False)
    grip_pos = np.asarray(info.get("grip_pos", achieved_goal))
    up_z     = info.get("up_z", 1.0)

    reward = 0.0

    # ── Approach phase ──────────────────────────────────────────────
    if phase == "approach":
        dist_grip_to_piece = np.linalg.norm(grip_pos - achieved_goal)
        # Reward proportional to closeness (negative distance, shifted)
        reward += W_APPROACH * max(0.0, 0.30 - dist_grip_to_piece)

        # Detect grasp
        finger_qpos = np.asarray(info.get("finger_qpos", [1.0, 1.0]))
        fingers_closed = np.all(np.abs(finger_qpos) < FINGER_CLOSED_THR)
        piece_above_table = achieved_goal[2] > info.get("table_height", 0.422) + 0.008
        if fingers_closed and piece_above_table:
            reward += W_GRASP_BONUS  # one-time (phase gate prevents double-award)

    # ── Transport phase ──────────────────────────────────────────────
    elif phase == "transport":
        dist_piece_to_dest = np.linalg.norm(achieved_goal - desired_goal)
        reward += W_TRANSPORT * max(0.0, 0.60 - dist_piece_to_dest)

        # Lift bonus (piece significantly above table)
        piece_height = achieved_goal[2] - info.get("table_height", 0.422)
        if piece_height > 0.02:
            reward += W_LIFT_BONUS * min(piece_height / 0.10, 1.0)

        # Tilt penalty during transport
        if up_z < UP_Z_THR:
            reward += W_TILT_PENALTY

    # ── Placement phase ──────────────────────────────────────────────
    elif phase == "placement":
        xy_err = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        z_err  = abs(achieved_goal[2] - desired_goal[2])
        reward += W_TRANSPORT * max(0.0, 0.60 - np.linalg.norm(achieved_goal - desired_goal))

        # Success bonus
        if xy_err < XY_TOL and z_err < Z_TOL and up_z >= UP_Z_THR:
            reward += W_PLACEMENT

        # Tilt penalty during placement
        if up_z < UP_Z_THR:
            reward += W_TILT_PENALTY

    # ── Global penalties (any phase) ────────────────────────────────
    if info.get("piece_dropped", False):
        reward += W_DROP_PENALTY

    if info.get("neighbor_displaced", False):
        reward += W_CLUTTER_PEN

    if info.get("timed_out", False):
        reward += W_TIMEOUT_PEN

    return float(reward)
```

### 6.3 Reward Calibration Notes

- **W_PLACEMENT = 50.0** is intentionally large relative to the shaped terms. This ensures that the policy learns to optimize the final placement outcome, not just the intermediate distance reductions.
- The approach reward `W_APPROACH * max(0, 0.30 − dist)` saturates at 0.30 m and gives 0 beyond that. This prevents the policy from being rewarded for arbitrary movement before it reaches the piece.
- Do not use a pure sparse reward (`0 if success, −1 otherwise`) for fine-tuning from a partially-trained checkpoint. The shaped reward will accelerate convergence dramatically.
- After convergence on shaped reward, optionally run a final phase with sparse-only reward to sharpen the policy.

---

## 7. Curriculum Design

### 7.1 Stage Definitions

| Stage | Name | Source squares | Dest squares | Clutter | Piece types | Advancement gate |
|-------|------|---------------|-------------|---------|-------------|-----------------|
| 1 | Central single-piece | d4, e4, d5, e5 (center 4) | Any of center 4 | None | Pawn only | Grasp ≥ 95%, Place ≥ 90% on 200-ep eval |
| 2 | Central region | c3–f6 (4×4 grid) | c3–f6 | None | Pawn, Knight | Grasp ≥ 95%, Place ≥ 90% |
| 3 | Full board, no clutter | All 64 squares | All 64 squares | None | All 6 types | Grasp ≥ 92%, Place ≥ 87% |
| 4 | Full board, passive clutter | All 64 squares | All 64 squares | 4–8 random pieces on board | All 6 types | Grasp ≥ 90%, Place ≥ 85%, Clutter rate < 5% |
| 5 | Legal board states | Sampled from real games | Legal moves | 16–32 pieces | All 6 types | Success ≥ 85% on legal-position eval suite |
| 6 | Captures + graveyard | Legal captures | Graveyard cells | Full board | All 6 types | Success ≥ 80% on capture eval suite |
| 7 | Sequential (game sim) | Full game trace | Full game trace | Full board | All 6 types | ≥ 10 consecutive moves, no drop |

### 7.2 Stage Advancement Logic

```python
# scripts/train_robochess_policy.py (partial)

def should_advance_stage(metrics: dict, current_stage: int) -> bool:
    """
    Gate advancement on eval metrics, not step count.
    `metrics` is the output of eval_robochess_policy on the held-out suite.
    """
    thresholds = {
        1: {"grasp_rate": 0.95, "place_rate": 0.90},
        2: {"grasp_rate": 0.95, "place_rate": 0.90},
        3: {"grasp_rate": 0.92, "place_rate": 0.87},
        4: {"grasp_rate": 0.90, "place_rate": 0.85, "clutter_rate": 0.05},
        5: {"place_rate": 0.85},
        6: {"place_rate": 0.80},
    }
    if current_stage not in thresholds:
        return False
    t = thresholds[current_stage]
    for key, val in t.items():
        if key == "clutter_rate":
            if metrics.get(key, 1.0) > val:  # clutter rate must be BELOW threshold
                return False
        else:
            if metrics.get(key, 0.0) < val:
                return False
    return True
```

### 7.3 Board Region Definitions

```python
# src/env/episode_sampler.py — regions
from src.config import square_to_xyz  # converts "e4" → np.array([x,y,z])

REGIONS = {
    1: ["d4", "e4", "d5", "e5"],
    2: [f"{f}{r}" for f in "cdef" for r in "3456"],   # 4×4 = 16 squares
    3: [f"{f}{r}" for f in "abcdefgh" for r in "12345678"],  # all 64
}
```

---

## 8. Domain Randomization

Apply **after** Stage 1 converges. Too early, it prevents convergence.

### 8.1 Parameters and Ranges

| Parameter | Nominal | Randomization range | When to enable |
|-----------|---------|--------------------|--------------------|
| Piece mass | per-type (20–30 g) | ±20% | Stage 2 |
| Piece friction (slide) | 0.7 | [0.5, 0.9] | Stage 2 |
| Piece friction (torsion) | 0.01 | [0.005, 0.02] | Stage 2 |
| Piece damping | 10.0 | [5.0, 20.0] | Stage 3 |
| Initial piece offset | 0 mm | ±3 mm XY, ±0 mm Z | Stage 2 |
| Initial orientation noise | 0° | ±3° roll/pitch | Stage 3 |
| Gripper initial pose noise | home | ±5 mm XYZ | Stage 3 |
| Table friction | 1.0 | [0.8, 1.2] | Stage 4 |

### 8.2 How to Apply

```python
# In ChessManipulationTrainEnv.reset() after _reset_physics():
if self.curriculum_stage >= 2:
    self._randomize_piece_properties()

def _randomize_piece_properties(self):
    for piece_name in PIECE_NAMES:
        body_id = self._model.body(piece_name).id
        # Mass randomization
        nominal_mass = self._model.body_mass[body_id]
        self._model.body_mass[body_id] = nominal_mass * self._rng.uniform(0.80, 1.20)
        # Friction randomization (geom-level)
        geom_id = self._model.body_geomadr[body_id]
        if geom_id >= 0:
            self._model.geom_friction[geom_id, 0] = self._rng.uniform(0.5, 0.9)
    mujoco.mj_forward(self._model, self._data)
```

---

## 9. Reset and Episode Generation

### 9.1 File: `src/env/episode_sampler.py`

```python
"""
EpisodeSampler
──────────────
Generates manipulation episodes for the training environment.
Each episode defines exactly one pick-and-place task:
  - which piece to pick
  - from which square (src)
  - to which square (dst)
  - whether it's a capture

Episodes are sourced from three buckets:
  A) Random legal board states from python-chess
  B) Adversarial states (edge files, crowded neighborhoods, long transports)
  C) Evaluation seeds (fixed, for regression tracking)

Bucket A: 70%
Bucket B: 20%
Bucket C: 10% (only during eval — not during training)
"""

from dataclasses import dataclass
import numpy as np
import chess
from src.config import square_to_xyz, PIECE_NAMES, TABLE_HEIGHT
from src.env.board_state_gen import generate_random_board_state

@dataclass
class Episode:
    piece_name:  str          # MuJoCo body name, e.g. "w_pawn_3"
    src_square:  str          # UCI square, e.g. "e2"
    dst_square:  str          # UCI square, e.g. "e4"
    src_pos:     np.ndarray   # MuJoCo XYZ of source square
    dest_pos:    np.ndarray   # MuJoCo XYZ of destination square
    is_capture:  bool
    piece_type:  str          # "pawn", "rook", etc.


class EpisodeSampler:
    def __init__(self, rng: np.random.Generator, curriculum_stage: int):
        self._rng   = rng
        self._stage = curriculum_stage
        self._board = chess.Board()

    def sample(self) -> Episode:
        bucket = self._rng.choice(["legal", "adversarial"], p=[0.70, 0.30])
        if bucket == "legal":
            return self._sample_legal()
        else:
            return self._sample_adversarial()

    def _sample_legal(self) -> Episode:
        """Sample from a random legal board position."""
        from src.env.board_state_gen import sample_board_state
        board, piece_map = sample_board_state(self._rng, stage=self._stage)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self._sample_legal()   # recursion guard

        move = self._rng.choice(legal_moves)
        src  = chess.square_name(move.from_square)
        dst  = chess.square_name(move.to_square)

        piece = board.piece_at(move.from_square)
        piece_type = chess.piece_name(piece.piece_type)
        color  = "w" if piece.color == chess.WHITE else "b"
        # Map to MuJoCo body name via piece_map dict
        piece_name = piece_map.get((color, piece_type, src))
        if piece_name is None:
            return self._sample_legal()

        return Episode(
            piece_name = piece_name,
            src_square = src,
            dst_square = dst,
            src_pos    = square_to_xyz(src),
            dest_pos   = square_to_xyz(dst),
            is_capture = board.is_capture(move),
            piece_type = piece_type,
        )

    def _sample_adversarial(self) -> Episode:
        """
        Adversarial buckets:
          - edge file (a or h column) pickups
          - back rank (rank 1 or 8) pickups
          - knight move transport (L-shaped, ≥ 1 full square diagonal)
          - long transport (≥ 4 squares)
          - crowded source (≥ 2 neighbors within 1 square radius)
        """
        # Choose an adversarial pattern
        pattern = self._rng.choice(["edge_file", "long_transport", "back_rank"])
        # ... implementation mirrors _sample_legal() but with constrained sampling
        # This is left as an implementation task. The important thing is that
        # these patterns are represented — exactly HOW they're sampled is flexible.
        return self._sample_legal()  # fallback stub
```

### 9.2 Fixed Evaluation Seeds

Create a file `scripts/eval_seeds.json` with 100 fixed episodes per eval category:

```json
{
  "unit_manipulation": [
    {"piece": "w_pawn_1", "src": "e2", "dst": "e4", "stage": 1},
    {"piece": "b_knight_1", "src": "g8", "dst": "f6", "stage": 2},
    ...
  ],
  "legal_positions": [
    {"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
     "move": "e7e5", "expected_piece": "b_pawn_5"},
    ...
  ],
  "sequential": [
    {"game_pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5 ..."}
  ]
}
```

These seeds must be generated once and committed. They allow regression comparison across training runs.

---

## 10. Training Procedure

### 10.1 File: `scripts/train_robochess_policy.py`

```python
"""
Fine-tune the TQC+HER FetchPickAndPlace checkpoint on the RoboChess scene.

Usage:
  python scripts/train_robochess_policy.py \
    --checkpoint sb3/tqc-FetchPickAndPlace-v1 \
    --stage 1 \
    --total-timesteps 500000 \
    --n-envs 4 \
    --seed 42

The script:
  1. Loads the pretrained TQC checkpoint
  2. Wraps it with ChessManipulationTrainEnv
  3. Fine-tunes using TQC + HER
  4. Evaluates every EVAL_FREQ steps
  5. Saves the best checkpoint on validation success rate
  6. Advances the curriculum stage when eval gates are met
"""

import argparse, os, json, numpy as np
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from sb3_contrib import TQC
from src.env.chess_manipulation_train_env import ChessManipulationTrainEnv
from scripts.eval_robochess_policy import run_eval_suite

# ── Hyperparameters ──────────────────────────────────────────────────
EVAL_FREQ       = 10_000    # steps between evaluations
CHECKPOINT_FREQ = 20_000    # steps between checkpoint saves
N_HER_GOALS     = 4         # HER future goal sampling ratio
LEARNING_RATE   = 1e-4      # reduce from default 3e-4 for fine-tuning stability
BUFFER_SIZE     = 1_000_000
BATCH_SIZE      = 256
TAU             = 0.005
GAMMA           = 0.98


def make_env(stage: int, seed: int, rank: int):
    def _init():
        env = ChessManipulationTrainEnv(curriculum_stage=stage, seed=seed + rank)
        return env
    return _init


def main(args):
    np.random.seed(args.seed)

    # ── Environment setup ────────────────────────────────────────────
    vec_env = SubprocVecEnv([
        make_env(args.stage, args.seed, i) for i in range(args.n_envs)
    ])
    vec_env = VecMonitor(vec_env, filename=f"logs/training_stage{args.stage}")

    eval_env = ChessManipulationTrainEnv(curriculum_stage=args.stage, seed=9999)

    # ── Load pretrained checkpoint ───────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model = TQC.load(
        args.checkpoint,
        env       = vec_env,
        verbose   = 1,
        # Override hyperparameters for fine-tuning:
        learning_rate = LEARNING_RATE,
        buffer_size   = BUFFER_SIZE,
        batch_size    = BATCH_SIZE,
        tau           = TAU,
        gamma         = GAMMA,
        replay_buffer_class = HerReplayBuffer,
        replay_buffer_kwargs = {
            "n_sampled_goal": N_HER_GOALS,
            "goal_selection_strategy": "future",
        },
        device = "auto",
    )
    # Do NOT reset_num_timesteps — preserves learning rate schedule and
    # buffer statistics from pretraining.
    model._last_obs = None   # force reset of internal obs buffer

    # ── Observation normalization ────────────────────────────────────
    # The pretrained model was trained WITHOUT VecNormalize.
    # Do not add it here unless you are prepared to retrain from scratch.
    # If you choose to add VecNormalize, unfreeze stats from the start
    # and monitor that the running mean/var does not collapse.

    # ── Callbacks ────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq    = CHECKPOINT_FREQ,
        save_path    = f"checkpoints/stage{args.stage}/",
        name_prefix  = "tqc_robochess",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = f"checkpoints/stage{args.stage}/best/",
        log_path             = f"logs/eval_stage{args.stage}/",
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = 50,
        deterministic        = True,
    )

    # ── Training loop ────────────────────────────────────────────────
    current_stage = args.stage
    steps_per_stage = args.total_timesteps

    while current_stage <= 7:
        print(f"\n{'='*60}")
        print(f"  Training Stage {current_stage}")
        print(f"{'='*60}")

        model.learn(
            total_timesteps    = steps_per_stage,
            reset_num_timesteps= False,
            callback           = [checkpoint_cb, eval_cb],
            log_interval       = 10,
            tb_log_name        = f"stage{current_stage}",
        )

        # Evaluate on held-out suite
        metrics = run_eval_suite(model, stage=current_stage, n_episodes=200, seed=9999)
        print(f"Stage {current_stage} eval metrics: {metrics}")

        # Save metrics
        with open(f"logs/stage{current_stage}_final_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        if should_advance_stage(metrics, current_stage):
            print(f"✓ Stage {current_stage} gates met — advancing to Stage {current_stage+1}")
            current_stage += 1
            # Update curriculum in all envs
            vec_env.env_method("set_curriculum_stage", current_stage)
            eval_env.curriculum_stage = current_stage
        else:
            print(f"✗ Stage {current_stage} gates NOT met — continuing training")
            # Optionally: decay LR slightly on plateau
            model.learning_rate = max(model.learning_rate * 0.95, 1e-5)

    model.save("checkpoints/final/tqc_robochess_final")
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       default="sb3/tqc-FetchPickAndPlace-v1")
    parser.add_argument("--stage",            type=int,   default=1)
    parser.add_argument("--total-timesteps",  type=int,   default=500_000)
    parser.add_argument("--n-envs",           type=int,   default=4)
    parser.add_argument("--seed",             type=int,   default=42)
    main(parser.parse_args())
```

### 10.2 Exact Package Versions to Lock

The `FetchPickAndPlace-v1` checkpoint was saved under OpenAI Gym. SB3 will warn about this. The loading is compatible with Gymnasium via the patch in `stable_baselines3.common.vec_env.patch_gym`.

Pin these in `requirements.txt`:

```
mujoco>=3.1.0
gymnasium>=0.29.0
gymnasium-robotics>=1.2.0
stable-baselines3>=2.1.0
sb3-contrib>=2.1.0            # provides TQC
huggingface-sb3>=3.0
python-chess>=1.10.0
numpy>=1.24.0,<2.0.0          # CRITICAL: SB3 has issues with NumPy 2.x
```

> **NumPy 2.x warning:** The logs show `Gym has been unmaintained since 2022 and does not support NumPy 2.0`. Pin `numpy<2.0.0` explicitly. This is not a deprecation warning — it causes silent numerical errors in SB3's internal buffer handling.

### 10.3 Required Logging Metrics

Log these at every `EVAL_FREQ` boundary. They feed the curriculum advancement gates.

| Metric | Description | Target (Stage 3+) |
|--------|-------------|------------------|
| `grasp_rate` | Fraction of episodes where piece was lifted > 1 cm | ≥ 0.92 |
| `place_rate` | Fraction of episodes meeting XY ≤ 10 mm, Z ≤ 10 mm, up_z ≥ 0.966 | ≥ 0.87 |
| `drop_rate` | Fraction where piece was dropped after lift | ≤ 0.05 |
| `tilt_rate` | Fraction where final up_z < 0.966 | ≤ 0.10 |
| `clutter_rate` | Fraction where a non-target piece moved > 5 mm | ≤ 0.05 |
| `mean_xy_err_mm` | Mean XY placement error in mm | ≤ 8 mm |
| `mean_z_err_mm` | Mean Z placement error in mm | ≤ 8 mm |
| `success_by_file` | Per-file (a–h) success rate dict | Visualize only |
| `success_by_piece_type` | Per piece-type success rate dict | Visualize only |
| `mean_episode_steps` | Average steps to success or timeout | Trend monitor |

---

## 11. Evaluation Protocol

### 11.1 File: `scripts/eval_robochess_policy.py`

```python
"""
Evaluation script for the fine-tuned RoboChess policy.

Three evaluation suites:
  1. Unit Manipulation Suite  — fixed source/dest per piece type and board region
  2. Legal Position Suite     — sampled legal chess positions
  3. Sequential Game Suite    — consecutive AI turns, no teleport fallback

Run standalone:
  python scripts/eval_robochess_policy.py \
    --checkpoint checkpoints/stage3/best/best_model.zip \
    --suite all \
    --n-episodes 200
"""

import json, numpy as np
from sb3_contrib import TQC
from src.env.chess_manipulation_train_env import ChessManipulationTrainEnv


def run_eval_suite(
    model,
    stage:      int = 3,
    n_episodes: int = 200,
    seed:       int = 9999,
) -> dict:
    metrics = {}
    metrics.update(_run_unit_suite(model, n_episodes, seed))
    metrics.update(_run_legal_suite(model, n_episodes, seed))
    return metrics


def _run_unit_suite(model, n_episodes, seed) -> dict:
    """
    Fixed source/dest tasks, one per piece type × region combination.
    Always uses deterministic=True.
    """
    env = ChessManipulationTrainEnv(curriculum_stage=3, seed=seed)
    rng = np.random.default_rng(seed)

    successes, grasps, drops, tilts, xy_errs, z_errs = [], [], [], [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        done = False
        grasped_this_ep = False
        info_final = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            grasped_this_ep = grasped_this_ep or info.get("grasped", False)
            done = terminated or truncated
            info_final = info

        successes.append(float(info_final.get("is_success", False)))
        grasps.append(float(grasped_this_ep))
        drops.append(float(info_final.get("piece_dropped", False)))
        tilts.append(float(info_final.get("up_z", 1.0) < 0.966))
        err_m = info_final.get("placement_error_m", 0.0)
        xy_errs.append(err_m * 1000)   # to mm
        z_errs.append(abs(
            np.array(obs["achieved_goal"])[2] - np.array(obs["desired_goal"])[2]
        ) * 1000)

    return {
        "grasp_rate":    float(np.mean(grasps)),
        "place_rate":    float(np.mean(successes)),
        "drop_rate":     float(np.mean(drops)),
        "tilt_rate":     float(np.mean(tilts)),
        "mean_xy_err_mm": float(np.mean(xy_errs)),
        "mean_z_err_mm":  float(np.mean(z_errs)),
    }


def _run_legal_suite(model, n_episodes, seed) -> dict:
    """
    Randomly sampled legal chess positions. Measures manipulation
    success rate across the diversity of real game states.
    """
    # Mirror of _run_unit_suite but using stage=5 (legal positions)
    env = ChessManipulationTrainEnv(curriculum_stage=5, seed=seed + 1000)
    # ... same measurement loop ...
    return {}  # placeholder — implement identically to _run_unit_suite


def run_sequential_suite(model, n_games: int = 10, seed: int = 9999) -> dict:
    """
    Sequential game suite. Run N complete game traces using Stockfish.
    Fail a game if: piece dropped, piece tipped, coordinate mismatch > 10 mm,
    non-target piece moved > 5 mm, or wrong destination.

    This suite is the FINAL deployment gate. A policy passes only if it
    completes all N games without requiring teleport fallback.
    """
    # Implementation requires Stockfish + the full game loop from main.py
    # but with teleport_fallback disabled.
    # Key difference from main.py: on RL failure, log and fail the game
    # instead of falling back to teleportation.
    results = {"games_attempted": n_games, "games_completed": 0, "failure_reasons": []}
    # ... implement using ChessGameManager + learned policy only ...
    return results
```

### 11.2 Evaluation Decision Tree

```
                    ┌────────────────────────────────────┐
                    │   Run Sequential Suite (10 games)   │
                    └────────────────┬───────────────────┘
                                     │
                    ┌────────────────▼───────────────────┐
                    │ All 10 games complete without drop/ │
                    │ tilt / drift / wrong destination?   │
                    └────────┬──────────────┬────────────┘
                             │ YES          │ NO
                    ┌────────▼──────┐  ┌───▼──────────────────────┐
                    │  Deploy pure  │  │  Is grasp rate ≥ 90%?    │
                    │  RL policy    │  └────┬──────────────┬───────┘
                    └───────────────┘       │ YES          │ NO
                                   ┌────────▼──┐  ┌───────▼──────────┐
                                   │  Hybrid:  │  │ Continue training │
                                   │  learned  │  │ on Stage 5–6      │
                                   │  grasp +  │  └──────────────────┘
                                   │  scripted │
                                   │  transport│
                                   └───────────┘
```

---

## 12. Integration Back Into Main Pipeline

### 12.1 Policy Loading in `main.py`

Replace the existing pretrained model load with a conditional:

```python
# src/config.py — add:
FINETUNED_MODEL_PATH = "checkpoints/final/tqc_robochess_final"   # set to None to use pretrained
PRETRAINED_MODEL_PATH = "sb3/tqc-FetchPickAndPlace-v1"

# main.py — model loading section:
model_path = FINETUNED_MODEL_PATH or PRETRAINED_MODEL_PATH
logger.info(f"Loading model: {model_path}")
model = TQC.load(model_path, env=env)
```

### 12.2 Keep All Existing Validators

Do not remove or weaken these:
- `runtime_guard.py` `validate_board_state()` — keeps `StabilityError` on tipped pieces
- `board_observer.py` `verify_piece()` and `full_board_scan()`
- `execution_controller.py` `OUT_OF_POLICY_WORKSPACE` warning (keep as informational)

### 12.3 Disable Teleport Fallback During Evaluation

Add a flag to `main.py`:

```python
# main.py
DISABLE_TELEPORT_FALLBACK = os.environ.get("NO_TELEPORT", "0") == "1"

# In the AI move execution block:
if not result.success:
    if DISABLE_TELEPORT_FALLBACK:
        logger.error(f"RL failed and teleport fallback disabled. Game ended.")
        raise RuntimeError("Policy failure in no-fallback evaluation mode.")
    else:
        logger.warning(f"RL failed — falling back to teleportation.")
        _teleport_piece(...)
```

Run integration evaluations with `NO_TELEPORT=1 python main.py` to measure true policy reliability.

### 12.4 Hybrid Controller Option

If the fine-tuned policy reliably grasps but drops during transport, implement a hybrid:

```python
# In execution_controller.py, ExecutionController.execute_op():

def execute_op_hybrid(self, op: PickPlaceOp, viewer=None) -> OpResult:
    """
    Hybrid mode:
      - Use the learned policy for APPROACH + CLOSE_GRIPPER stages
      - Switch to scripted waypoint control for LIFT + TRANSPORT + PLACE stages
    
    This is useful if grasp reliability is ≥ 90% but transport has drift.
    """
    # Phase 1: Run learned policy until grasp detected (piece lifted > 1 cm)
    grasp_result = self._run_policy_until_grasp(op)
    if not grasp_result.grasped:
        return OpResult(success=False, ...)

    # Phase 2: Scripted transport using the existing staged waypoint controller
    return self._scripted_transport(op, from_current_grip_pos=grasp_result.grip_pos)
```

---

## 13. File Manifest and Deliverables

### 13.1 New Files to Create

| File | Purpose |
|------|---------|
| `src/env/chess_manipulation_train_env.py` | Training GoalEnv (Section 3) |
| `src/env/context_builder.py` | Chess-scene context vector (Section 4) |
| `src/env/reward_fn.py` | Separable dense reward function (Section 6) |
| `src/env/episode_sampler.py` | Curriculum-aware episode generator (Section 9) |
| `src/env/board_state_gen.py` | Legal board position generator from python-chess |
| `scripts/train_robochess_policy.py` | Main training script (Section 10) |
| `scripts/eval_robochess_policy.py` | Evaluation script (Section 11) |
| `scripts/eval_seeds.json` | 100 fixed evaluation seeds per suite |
| `scripts/run_sequential_eval.py` | Runs the sequential game suite |
| `tests/test_train_env.py` | Unit tests for the training env |
| `tests/test_reward_fn.py` | Unit tests for the reward function |

### 13.2 Files to Modify

| File | Modification |
|------|-------------|
| `src/config.py` | Add TABLE_HEIGHT fix, graveyard repositioning, FINETUNED_MODEL_PATH |
| `scripts/generate_xml.py` | Unify collision/visual geometry (Section 2.3) |
| `src/assets/chess_world.xml` | Board height +8 mm (Section 2.1) |
| `main.py` | Add DISABLE_TELEPORT_FALLBACK flag, conditional model path |
| `requirements.txt` | Pin numpy<2.0.0, add sb3-contrib |

### 13.3 Checkpoint Management Plan

```
checkpoints/
├── stage1/
│   ├── tqc_robochess_0_steps.zip
│   ├── tqc_robochess_20000_steps.zip
│   └── best/
│       └── best_model.zip          ← best stage-1 checkpoint
├── stage2/
│   └── best/best_model.zip
├── ...
└── final/
    └── tqc_robochess_final.zip     ← deployed checkpoint
```

Always promote from the `best/` checkpoint within a stage, never from an intermediate checkpoint. Tag checkpoints with their stage and eval metrics in a `checkpoint_registry.json`.

### 13.4 Failure Analysis Deliverables

After each stage evaluation, produce:
1. **Success rate heatmap** over the 8×8 board — which squares are problematic
2. **Failure mode breakdown** — what fraction fail at grasp vs. lift vs. transport vs. placement
3. **Example failure logs** — 5 representative failed episodes per failure mode with step-by-step obs/action traces
4. **Comparative table** against the scripted controller baseline on the same eval seeds

---

## 14. First Milestone Definition

The first concrete milestone is intentionally narrow. Achieving it validates the entire pipeline before scaling.

### Milestone 1 Definition

**Name:** Single-Piece Central Grasp (Stage 1)

**Conditions:**
- Curriculum Stage 1 only (center 4 squares: c3–f6 / 4×4 region)
- Pawn and Knight piece types only
- No clutter pieces on the board
- Policy evaluated on 200 fixed evaluation seeds
- Deterministic inference (`deterministic=True`)

**Success criteria:**

| Metric | Threshold |
|--------|-----------|
| Grasp rate (piece lifted > 1 cm) | ≥ 95% |
| Placement success rate | ≥ 90% |
| Mean XY placement error | ≤ 8 mm |
| Tilt rate (up_z < 0.966 at placement) | ≤ 5% |
| Drop rate | ≤ 3% |

**Validation method:**
```bash
python scripts/eval_robochess_policy.py \
  --checkpoint checkpoints/stage1/best/best_model.zip \
  --suite unit \
  --n-episodes 200 \
  --seed 9999
```

**Expected training cost:**  
~200k–500k timesteps with 4 parallel envs on a modern workstation (CPU). With GPU, faster convergence in the policy network updates but MuJoCo physics is CPU-bound.

**Gate to proceed:**  
If Milestone 1 is not met after 1M timesteps of training, do not expand the curriculum. Instead:
1. Check that the 7 mm table height fix was applied
2. Check that the grasp geometry is correct (run `scripts/headless_run.py`)
3. Check that the observation vector has no NaNs or mis-scaled dims
4. Consider reducing `max_steps` to force more episodic diversity in the buffer

---

## Appendix A: Board Coordinate Reference

```
Square  │ MuJoCo X  │ MuJoCo Y  │ Notes
────────┼───────────┼───────────┼────────────────────────
a1      │ 1.005     │ 0.435     │ White queenside rook start
e4      │ 1.145     │ 0.715     │ Common pawn push square
h8      │ 1.405     │ 0.855     │ Black kingside corner
d1      │ 1.145     │ 0.435     │ White queen start
```

> These coordinates are derived from `config.py: BOARD_ORIGIN` and `SQUARE_SIZE=0.05 m`. Verify them against the actual MJCF scene before training.

---

## Appendix B: Troubleshooting Common Failures

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Gripper slides off piece during close | Table height below policy z-floor | Apply Section 2.1 fix |
| Policy circles around piece but never descends | Observation NaN or wrong indices | Run `test_env_obs.py` |
| Grasp rate 0% from the start | HER not configured, or reward always 0 | Check `replay_buffer_class` arg and reward fn |
| Piece drifts during transport | Friction too low, or damping too high | Apply narrow domain randomization (Section 8) |
| CUDA OOM during training | Batch size too large for VRAM | Reduce `BATCH_SIZE` to 128 |
| Curriculum never advances | Eval gates too strict for stage | Verify eval uses correct stage env and deterministic=True |
| Policy loss spikes to NaN mid-training | Learning rate too high after checkpoint load | Reduce `LEARNING_RATE` to 3e-5, restart from last checkpoint |
| `mujoco.FatalError: gladLoadGL error` | OpenGL unavailable on headless server | Set `MUJOCO_GL=osmesa` or `MUJOCO_GL=egl` before training |
| `SubprocVecEnv` hangs on macOS | macOS fork safety issue | Use `DummyVecEnv` on macOS or set `multiprocessing.set_start_method("spawn")` |
| HER buffer loads but replay crashes | Obs shape mismatch between saved buffer and new env | Do not load a pretrained HER buffer — start fresh |
| `UserWarning: You loaded a model trained on OpenAI Gym` | Checkpoint was saved under old Gym API | Safe to ignore — SB3's patch_gym handles it. Add `filterwarnings("ignore")` if it clutters logs |

> **Note:** The `robot0:mocap` startup crash previously described here has been fixed in `board_observer.py`. No action needed.

---

*End of Implementation Specification — continues with Sections 15–21 and Appendices C–E below*

---

## 15. Compute Options

MuJoCo training is CPU-bound for the physics simulation and GPU-bound for the neural network updates. The two halves scale independently, which matters when choosing where to train.

### 15.1 What "Weak PC" Means in This Context

A PC is effectively too weak if it has **fewer than 4 physical CPU cores, less than 16 GB RAM, or no CUDA-capable GPU**. For reference, a Stage 1 training run (500k timesteps, 4 parallel envs) takes approximately:

| Hardware | Estimated wall-clock time for Stage 1 |
|----------|--------------------------------------|
| 4-core CPU, no GPU (laptop) | 8–14 hours |
| 8-core CPU, no GPU | 4–7 hours |
| 8-core CPU + GTX 1080 (8 GB VRAM) | 2–4 hours |
| 8-core CPU + RTX 3090 (24 GB VRAM) | 1–2 hours |
| 16-core CPU + A100 (40 GB VRAM) | 30–60 minutes |

If you are on the first row, do not attempt full training locally. Use one of the cloud options below.

### 15.2 Can You Train Locally With Fewer Resources?

**Yes, with adjustments.** These settings reduce wall-clock time on weak hardware:

```python
# For a weak local machine (4-core, no GPU):
N_ENVS         = 2           # instead of 4 — reduces RAM pressure
BATCH_SIZE     = 128         # instead of 256 — reduces per-update cost
BUFFER_SIZE    = 200_000     # instead of 1M — fits in RAM
EVAL_FREQ      = 5_000       # more frequent evals to catch divergence early
MAX_STEPS      = 150         # shorter episodes reduce env overhead
device         = "cpu"       # explicit
```

Also set:
```bash
export OMP_NUM_THREADS=2     # prevent MuJoCo from oversubscribing CPU
export MKL_NUM_THREADS=2
```

With these settings, Stage 1 should complete in 10–16 hours on a 4-core laptop. It will work — just slowly.

### 15.3 Cloud Platform Options

These are all viable for RoboChess fine-tuning. MuJoCo runs headlessly via EGL/OSMesa on every major cloud provider without a display.

---

#### Option A: Google Colab (Free Tier)

**Best for:** Testing the pipeline end-to-end before committing to paid compute.

**What you get:** T4 GPU (16 GB VRAM), ~12 CPU cores, ~13 GB RAM. Sessions time out after ~12 hours of inactivity and at most ~24 hours of total runtime.

**Limitations:**
- Disk is ephemeral — checkpoints vanish when the session ends unless you mount Google Drive
- Session disconnects mid-run are common on the free tier
- Colab Pro / Pro+ ($10–$50/month) gives priority GPU access and longer runtimes — strongly recommended for any real training

**Setup for Colab:**
```python
# Cell 1: Install dependencies
!pip install mujoco gymnasium gymnasium-robotics stable-baselines3 sb3-contrib \
             huggingface-sb3 python-chess "numpy<2.0.0" --quiet

# Cell 2: Mount Google Drive for persistent checkpoints
from google.colab import drive
drive.mount('/content/drive')
CHECKPOINT_DIR = "/content/drive/MyDrive/robochess_checkpoints"

# Cell 3: Clone the repo
!git clone https://github.com/yuval092/RoboChess.git
%cd RoboChess

# Cell 4: Set headless rendering (CRITICAL — Colab has no display)
import os
os.environ["MUJOCO_GL"] = "egl"      # use EGL (GPU-backed, fastest)
# If EGL fails, fall back to: os.environ["MUJOCO_GL"] = "osmesa"

# Cell 5: Verify MuJoCo works headlessly
import mujoco
print(mujoco.__version__)   # should print without error

# Cell 6: Start training with frequent checkpointing
!python scripts/train_robochess_policy.py \
    --checkpoint sb3/tqc-FetchPickAndPlace-v1 \
    --stage 1 \
    --total-timesteps 500000 \
    --n-envs 4 \
    --seed 42 \
    --checkpoint-dir {CHECKPOINT_DIR}
```

**Colab-specific watchouts:**
- Do not use `SubprocVecEnv` on Colab — it frequently deadlocks due to forking limitations. Use `DummyVecEnv` or set `multiprocessing.set_start_method("spawn")` at the top of your script before any imports.
- Colab kills the runtime if GPU memory exceeds the limit silently. Monitor with `!nvidia-smi` in a separate cell.
- Use `tqdm` for progress bars with `position=0, leave=True` — otherwise the output scrolls excessively.

---

#### Option B: Kaggle Notebooks (Free)

**Best for:** Free GPU access with longer session limits than Colab free tier.

**What you get:** P100 GPU (16 GB VRAM) or T4 GPU, up to **30 GPU hours per week**, 9-hour session limit, 20 GB disk (not persistent between sessions).

**Limitations:**
- No outbound internet by default in notebook sessions (can be enabled in settings)
- Git cloning works but `pip install` from GitHub may need internet enabled
- 30 h/week is enough for Stage 1 and Stage 2 but not a full curriculum run

**Setup differences from Colab:**
```python
# Kaggle already has most packages; install just the gaps:
!pip install sb3-contrib gymnasium-robotics python-chess --quiet

# Headless rendering — same as Colab:
import os
os.environ["MUJOCO_GL"] = "egl"

# Save outputs to /kaggle/working/ for download after session
```

Kaggle does not support Google Drive mounting. Download checkpoints at the end of each session or push them to a GitHub release.

---

#### Option C: RunPod (Paid, Cheapest GPU Rental)

**Best for:** Persistent training without session timeouts. Ideal for Stage 3–7 curriculum runs.

**What you get:** On-demand or spot GPU instances. RTX 3090 at ~$0.30–0.45/hour, A100 at ~$1.50–2.50/hour. Persistent storage volumes available.

**Why this is the recommended paid option:** Persistent disk (you choose the volume size) means checkpoints survive between sessions. SSH access means you can run `tmux` and walk away.

**Setup:**
```bash
# 1. Create a RunPod account at runpod.io
# 2. Deploy a pod with:
#    - Template: "RunPod Pytorch 2.x" (has CUDA pre-installed)
#    - GPU: RTX 3090 or A100
#    - Persistent volume: 20 GB minimum (attach at /workspace)
# 3. SSH into the pod:

# Install dependencies
pip install mujoco gymnasium gymnasium-robotics stable-baselines3 \
            sb3-contrib huggingface-sb3 python-chess "numpy<2.0.0"

# Clone repo to persistent volume
cd /workspace
git clone https://github.com/yuval092/RoboChess.git
cd RoboChess

# Set headless rendering
export MUJOCO_GL=egl

# Run training in tmux so it survives SSH disconnect
tmux new-session -d -s train
tmux send-keys -t train "python scripts/train_robochess_policy.py \
    --stage 1 --total-timesteps 500000 --n-envs 8 --seed 42" Enter

# Monitor
tmux attach -t train
```

**Cost estimate for full curriculum (Stages 1–6):**
- RTX 3090, ~4M total timesteps: approximately 8–12 hours → **$3–$6 total**
- A100 40 GB, same: approximately 3–5 hours → **$5–$10 total**

This is the cheapest realistic option for a complete training run.

---

#### Option D: Vast.ai (Cheapest Spot Market)

**Best for:** Maximum GPU for minimum cost, with tolerance for interruption.

**What you get:** A marketplace where individuals rent out their GPU rigs. RTX 4090 instances available for ~$0.20–0.35/hour. Quality varies.

**Risk:** Instances can be reclaimed mid-training. You must checkpoint frequently and design the training script to resume from the last checkpoint automatically.

The training script in Section 10 already supports resumption via `reset_num_timesteps=False` and staged checkpointing. To make it fully resumable, add:

```python
# At the start of train_robochess_policy.py:
import glob, os

def find_latest_checkpoint(stage: int, checkpoint_dir: str) -> str | None:
    """Find the most recent checkpoint for the given stage."""
    pattern = os.path.join(checkpoint_dir, f"stage{stage}", "tqc_robochess_*_steps.zip")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None

# In main():
resume_from = find_latest_checkpoint(args.stage, args.checkpoint_dir)
if resume_from:
    print(f"Resuming from checkpoint: {resume_from}")
    model = TQC.load(resume_from, env=vec_env, ...)
else:
    print(f"Starting from pretrained: {args.checkpoint}")
    model = TQC.load(args.checkpoint, env=vec_env, ...)
```

---

#### Option E: Lambda Labs Cloud (Affordable, High Quality)

**Best for:** More reliable than Vast.ai, less expensive than AWS/GCP for single-GPU jobs.

**What you get:** A100 80 GB at ~$1.29/hour, H100 at ~$2.49/hour. Persistent storage. SSH access. Very clean Ubuntu environments.

**No free tier**, but credit promotions are common. Recommended over RunPod for Stage 5–7 training where reliability matters more than cost per hour.

Setup is identical to RunPod. Lambda Labs instances come with PyTorch pre-installed; add MuJoCo and SB3 manually.

---

#### Option F: Modal (Serverless GPU, Developer-Friendly)

**Best for:** Running eval sweeps or short training bursts without managing a persistent instance.

Modal executes Python functions in cloud containers on demand. You pay only for actual GPU time used (billed by the second).

```python
# scripts/modal_train.py
import modal

stub = modal.Stub("robochess-finetune")
image = modal.Image.debian_slim().pip_install(
    "mujoco", "gymnasium", "gymnasium-robotics",
    "stable-baselines3", "sb3-contrib", "python-chess", "numpy<2.0.0"
)

@stub.function(
    image   = image,
    gpu     = modal.gpu.A10G(),
    timeout = 3600,                     # 1 hour
    volumes = {"/checkpoints": modal.Volume.from_name("robochess-checkpoints")},
)
def train_stage1():
    import os
    os.environ["MUJOCO_GL"] = "egl"
    # import and call your train_robochess_policy.main() here
    from scripts.train_robochess_policy import main, Args
    main(Args(stage=1, total_timesteps=200_000, n_envs=4, seed=42))

@stub.local_entrypoint()
def run():
    train_stage1.remote()
```

```bash
modal run scripts/modal_train.py
```

Cost: A10G (24 GB VRAM) at ~$0.54/hour. Good for one-off runs but not the cheapest for long continuous training.

---

#### Option G: Google Cloud / AWS / Azure Spot Instances

**Best for:** Teams, repeatability, and compliance-sensitive environments.

These are the most expensive options per GPU-hour but offer the most flexibility, SLA guarantees, and tooling. Not recommended unless you already have cloud credits or organizational requirements.

Spot instances (GCP: preemptible, AWS: spot, Azure: spot VMs) reduce cost by 60–90% at the risk of termination. The resumption logic from Option D applies here as well.

Rough GPU costs (on-demand, not spot):
- GCP: V100 ~$2.48/h, A100 ~$3.67/h
- AWS: p3.2xlarge (V100) ~$3.06/h, p4d.xlarge (A100) ~$3.91/h
- Azure: NC6s_v3 (V100) ~$3.06/h

---

### 15.4 Decision Guide: Which Platform to Use

```
Do you have a machine with ≥ 8 cores and any CUDA GPU?
├── YES → Train locally. Use Section 10 settings as-is.
│          If GPU has < 8 GB VRAM: set BATCH_SIZE=128, N_ENVS=4.
│
└── NO → Is this a quick pipeline test (Stage 1 only)?
         ├── YES → Google Colab free tier. Set MUJOCO_GL=egl.
         │          Use DummyVecEnv. Expect 3–5 h for Stage 1.
         │
         └── NO → Is cost your primary constraint?
                  ├── YES → RunPod (RTX 3090, ~$0.35/h).
                  │          Full curriculum: $5–10 total.
                  │
                  └── NO → Lambda Labs (A100, ~$1.29/h).
                             More reliable, faster, cleaner.
                             Full curriculum: $15–25 total.
```

---

## 16. Platform-Specific Setup Guides

### 16.1 Headless Rendering — The Most Common Blocker on Servers

MuJoCo requires an OpenGL context even for headless physics simulation (when no viewer is launched). On servers and cloud instances there is no display. You must explicitly choose a software or EGL renderer.

**Option 1: EGL (preferred — GPU-backed, fastest)**
```bash
export MUJOCO_GL=egl
pip install mujoco  # EGL support is built in since MuJoCo 2.3.x
```
EGL requires a CUDA-capable GPU present in the system. It is the fastest option and the one to use on RunPod, Lambda, and any cloud instance with a GPU.

**Option 2: OSMesa (CPU software renderer — no GPU needed)**
```bash
sudo apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3
export MUJOCO_GL=osmesa
```
OSMesa is slower but works on CPU-only machines (Colab CPU runtime, GitHub Actions CI, etc.). Performance is adequate for training since the viewer is not used.

**Option 3: Xvfb virtual framebuffer (legacy, avoid)**
```bash
sudo apt-get install -y xvfb
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
```
This works but adds overhead and complexity. Only use it if EGL and OSMesa both fail.

**Verification:**
```python
import os
os.environ["MUJOCO_GL"] = "egl"   # or "osmesa"
import mujoco
model = mujoco.MjModel.from_xml_string("<mujoco/>")
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print("MuJoCo headless OK")
```

---

### 16.2 Windows-Specific Issues

MuJoCo 3.x supports Windows but with several caveats.

**Problem 1: `SubprocVecEnv` deadlocks on Windows**

Windows uses `spawn` (not `fork`) for multiprocessing, which requires all code to be inside `if __name__ == "__main__":` guards.

```python
# scripts/train_robochess_policy.py — Windows fix
import multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main(parse_args())
```

Alternatively, just use `DummyVecEnv` on Windows. It is slower but eliminates the multiprocessing issue entirely.

**Problem 2: Path separators in SCENE_XML**

MuJoCo's XML parser is sensitive to backslashes on Windows. In `config.py`:
```python
# Bad on Windows:
SCENE_XML = "src\\assets\\chess_world.xml"

# Good — use pathlib:
from pathlib import Path
SCENE_XML = str(Path(__file__).parent / "assets" / "chess_world.xml")
```

**Problem 3: EGL unavailable on Windows**

EGL on Windows requires a specific driver setup. Use the built-in WGL renderer instead:
```bash
set MUJOCO_GL=wgl
```
WGL is Windows-native and works without additional setup when a GPU is present.

**Problem 4: NumPy and SB3 on Windows with conda**

If using Anaconda on Windows, `numpy<2.0.0` may conflict with the conda numpy package. Use `pip` directly inside the conda env, not `conda install numpy`:
```bash
conda create -n robochess python=3.11
conda activate robochess
pip install mujoco gymnasium gymnasium-robotics stable-baselines3 \
            sb3-contrib python-chess "numpy<2.0.0"
```

---

### 16.3 macOS-Specific Issues

**Problem 1: MuJoCo rendering on Apple Silicon**

MuJoCo 3.x supports Apple Silicon natively. Use the metal renderer:
```bash
# No special environment variable needed on macOS — MuJoCo auto-selects
# If you get rendering errors, try:
export MUJOCO_GL=osmesa
```

**Problem 2: `SubprocVecEnv` on macOS**

macOS changed the default multiprocessing start method to `spawn` in Python 3.8+. Add to the top of your training script:
```python
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
```

Or simply use `DummyVecEnv` with `n_envs=1` on macOS and accept the slower training speed.

**Problem 3: No CUDA on Apple Silicon**

Apple Silicon Macs have no CUDA. Training runs on CPU (Metal MPS is not yet supported by PyTorch for SB3's use case). Use `device="cpu"` explicitly:
```python
model = TQC.load(..., device="cpu")
```
Stage 1 training on an M2 Pro (10-core) takes approximately 5–8 hours. Reasonable for pipeline testing; use cloud for full curriculum.

---

### 16.4 Docker Container Setup

For reproducible environments across machines and cloud providers, use a container:

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-pip git \
    libosmesa6-dev libgl1-mesa-glx libglfw3 \
    libstockfish    \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    mujoco \
    gymnasium \
    gymnasium-robotics \
    "stable-baselines3>=2.1.0" \
    "sb3-contrib>=2.1.0" \
    "huggingface-sb3>=3.0" \
    "python-chess>=1.10.0" \
    "numpy<2.0.0" \
    wandb tensorboard

ENV MUJOCO_GL=egl

WORKDIR /workspace
COPY . /workspace/RoboChess
WORKDIR /workspace/RoboChess

ENTRYPOINT ["python", "scripts/train_robochess_policy.py"]
```

```bash
# Build and run
docker build -t robochess-train .
docker run --gpus all \
    -v $(pwd)/checkpoints:/workspace/RoboChess/checkpoints \
    robochess-train \
    --stage 1 --total-timesteps 500000 --n-envs 8
```

This Docker setup works identically on RunPod, Lambda Labs, and your local machine with an NVIDIA GPU and the NVIDIA Container Toolkit installed.

---

## 17. Runtime and Environment Edge Cases

### 17.1 MuJoCo Physics Instability (NaN Explosion)

**Symptom:** Training suddenly stops with `nan` in loss metrics, or `mujoco.MujocoException: b'Physics error found.'`

**Causes and fixes:**

| Cause | Fix |
|-------|-----|
| Piece penetrates the table (contact solver diverges) | Increase `<option solver="Newton" iterations="100"/>` in MJCF |
| Domain randomization sets friction to 0 | Clamp `friction > 0.01` in `_randomize_piece_properties()` |
| Two pieces teleported to overlapping positions | Add a 5 mm minimum separation check in `EpisodeSampler.sample()` |
| Action scaling produces unrealistically large Cartesian steps | Clip actions to `[-1, 1]` before passing to env (SB3 does this but verify your wrapper doesn't bypass it) |

**Recovery:** Add a NaN guard in `ChessManipulationTrainEnv.step()`:

```python
def step(self, action):
    # Sanitize action
    action = np.nan_to_num(np.clip(action, -1.0, 1.0), nan=0.0)
    
    try:
        result = self._env.step(action)
    except Exception as e:
        # Physics crash — return a zero observation and large negative reward
        # This prevents the training loop from crashing
        logger.warning(f"Physics exception in step: {e}. Resetting.")
        obs, _ = self.reset()
        return obs, -10.0, False, True, {"physics_crash": True}
    
    # Check for NaN in observation
    if np.any(np.isnan(result[0]["observation"])):
        logger.warning("NaN in observation — resetting episode.")
        obs, _ = self.reset()
        return obs, -10.0, False, True, {"obs_nan": True}
    
    return result
```

---

### 17.2 Episode Reachability Failures

**Symptom:** The episode sampler generates a source or destination square that the arm physically cannot reach, causing the policy to time out on every episode.

**Prevention:** Add a reachability check in `EpisodeSampler.sample()`:

```python
from src.config import REACH_X, REACH_Y

def _is_reachable(pos: np.ndarray) -> bool:
    """Check if a board position is within the arm's reachable envelope."""
    return (REACH_X[0] <= pos[0] <= REACH_X[1] and
            REACH_Y[0] <= pos[1] <= REACH_Y[1])

def sample(self) -> Episode:
    ep = self._sample_legal()
    max_retries = 20
    for _ in range(max_retries):
        if _is_reachable(ep.src_pos) and _is_reachable(ep.dest_pos):
            return ep
        ep = self._sample_legal()
    raise RuntimeError(
        f"Could not sample a reachable episode after {max_retries} retries. "
        f"Check graveyard positions and board layout against REACH_X/REACH_Y."
    )
```

---

### 17.3 `SubprocVecEnv` Hanging or Deadlocking

**Symptom:** Training starts, a few episodes run, then the process hangs indefinitely with no output.

**Causes:**
- A child process raises an exception inside `reset()` or `step()` — the parent waits for a result that never comes
- MuJoCo's shared memory system conflicts across processes on some Linux kernels

**Fixes:**
```python
# Add a timeout to subprocess communication
from stable_baselines3.common.vec_env import SubprocVecEnv

vec_env = SubprocVecEnv(
    [make_env(stage, seed, i) for i in range(n_envs)],
    start_method="forkserver",   # more stable than "fork" on Linux
)

# If it still hangs, use DummyVecEnv:
from stable_baselines3.common.vec_env import DummyVecEnv
vec_env = DummyVecEnv([make_env(stage, seed, i) for i in range(n_envs)])
```

`DummyVecEnv` runs all environments serially in one process. It is ~2–3× slower but never deadlocks.

---

### 17.4 Piece Permanently Stuck Against a Neighbor

**Symptom:** During an episode, the target piece gets wedged between the gripper and a clutter piece and cannot be moved to the destination.

**Detection:** Add a "stuck" check in `step()`:

```python
STUCK_THRESHOLD_MM = 1.0    # piece moved less than 1 mm over the last N steps
STUCK_CHECK_WINDOW = 30     # check every 30 steps

def _is_stuck(self) -> bool:
    if self._step_count % STUCK_CHECK_WINDOW != 0:
        return False
    current_xy = self._get_piece_xy()
    if self._prev_piece_xy is not None:
        drift = np.linalg.norm(current_xy - self._prev_piece_xy) * 1000
        if self._grasped and drift < STUCK_THRESHOLD_MM:
            return True
    return False
```

If `_is_stuck()` returns `True`, terminate the episode with a large negative reward and reset. This prevents the policy from wasting thousands of steps on an unrecoverable state.

---

### 17.5 Gripper Finger Asymmetry After Reset

**Symptom:** After a reset, the two gripper fingers are not symmetric (`qpos[finger_l] ≠ qpos[finger_r]` by more than 1 mm). This can happen if a piece was jammed between the fingers at the end of the previous episode.

**Fix in `ChessManipulationTrainEnv.reset()`:**

```python
def _reset_physics(self):
    mujoco.mj_resetData(self._model, self._data)
    # Explicitly set both finger joints to the open position
    finger_l_id = self._model.joint("robot0:l_gripper_finger_joint").id
    finger_r_id = self._model.joint("robot0:r_gripper_finger_joint").id
    self._data.qpos[finger_l_id] = 0.05   # open position
    self._data.qpos[finger_r_id] = 0.05
    self._data.qvel[finger_l_id] = 0.0
    self._data.qvel[finger_r_id] = 0.0
    mujoco.mj_forward(self._model, self._data)
```

---

### 17.6 Clutter Pieces Falling Off the Board

**Symptom:** During Stage 4+ episodes (with passive clutter), a clutter piece falls off the edge of the board during training, landing on the floor. The physics simulation continues but the board state is corrupted.

**Prevention:** In `EpisodeSampler._place_clutter_pieces()`, ensure all clutter pieces are placed in the center of their squares (not near the edge), and add a per-step clutter sanity check:

```python
def _check_clutter_integrity(self) -> bool:
    """Return False if any clutter piece has fallen below the table."""
    for name in self._episode.clutter_pieces:
        z = self._data.body(name).xpos[2]
        if z < TABLE_HEIGHT - 0.02:   # 2 cm below table = fallen off
            return False
    return True
```

If this returns `False`, terminate the episode immediately.

---

### 17.7 python-chess Board Generation Producing Unreachable Moves

**Symptom:** `EpisodeSampler._sample_legal()` returns an episode with a legal move that is physically unreachable (e.g., involves a piece starting on a square outside the arm's workspace, because the graveyard was not yet repositioned).

**Root cause:** This can happen if the graveyard fix from Section 2.2 was not fully applied. Verify that after the fix, **all** graveyard cells satisfy `_is_reachable()` before starting training.

**Additional guard:** During episode sampling for capture moves, explicitly verify the graveyard destination:

```python
if episode.is_capture:
    graveyard_pos = self._get_next_graveyard_cell(episode.piece_type)
    if not _is_reachable(graveyard_pos):
        # Graveyard is misconfigured — skip capture episodes entirely
        logger.error("Graveyard cell unreachable. Fix config.py graveyard positions.")
        return self._sample_legal()   # resample a non-capture episode
```

---

### 17.8 HER Replay Buffer Running Out of Valid Episodes

**Symptom:** Training loss is very high but constant; the policy makes no progress. TensorBoard shows `ep_rew_mean` stuck at a large negative constant.

**Likely cause:** The HER replay buffer has been filled entirely with failed episodes (no successes to relabel from). This happens when the initial policy is so far off-distribution that it never achieves even partial progress, starving HER of positive signal.

**Fix:** Add a "success kickstart" phase at the very start of Stage 1. Before running any random policy rollouts, manually populate the buffer with a small number of scripted-controller episodes that are known to succeed:

```python
# At the start of train_robochess_policy.main(), before model.learn():
from scripts.collect_scripted_demos import collect_demos

if args.stage == 1 and not resume_from:
    print("Collecting scripted demos to seed the replay buffer...")
    collect_demos(
        model      = model,
        env        = vec_env,
        n_episodes = 50,         # 50 successful scripted demos
        use_scripted_controller = True,
    )
    print("Buffer seeded. Starting RL training.")
```

The `collect_demos` script runs the existing `ExecutionController` (scripted waypoints) on Stage 1 episodes and stores the transitions directly into `model.replay_buffer`. This gives HER real successful trajectories to relabel from, breaking the cold-start problem.

---

## 18. Training Stability and Divergence Recovery

### 18.1 Signs of a Healthy Training Run

At the start of fine-tuning from the pretrained checkpoint, you should see:

- **First 10k steps:** `ep_rew_mean` is negative but not at the worst-case floor (policy is doing *something* from the checkpoint warm-start)
- **10k–50k steps:** `ep_rew_mean` increases steadily; `grasp_rate` climbs from baseline toward 50–70%
- **50k–200k steps:** Grasp rate reaches 80–90%; placement rate begins climbing
- **200k–500k steps:** Both metrics plateau near the Stage 1 advancement gates

If `ep_rew_mean` stays flat for more than 50k consecutive steps, something is wrong. See below.

---

### 18.2 Divergence: `ep_rew_mean` Suddenly Drops to Floor

**What happened:** The policy gradient update caused a catastrophic weight update. The policy now outputs near-random actions.

**Recovery steps:**
1. Load the last good checkpoint (the one before the drop — set `CHECKPOINT_FREQ=5000` to maximize granularity)
2. Reduce learning rate: multiply by 0.3 (e.g., `3e-4 → 9e-5`)
3. Increase `tau` slightly: `0.005 → 0.01` (faster target network update can smooth the gradient)
4. If divergence repeats, add gradient clipping:
   ```python
   model = TQC.load(..., policy_kwargs={"optimizer_kwargs": {"eps": 1e-5}})
   # SB3 does not expose gradient clipping directly in TQC — patch it:
   model.actor.optimizer.defaults["max_norm"] = 1.0
   ```

---

### 18.3 Plateaued Grasp Rate — Stuck Below 60%

**What this looks like:** After 200k steps, grasp rate never exceeds 55–60%. Placement rate is negligible (can't place what you can't grasp).

**Diagnosis checklist:**
1. Run `scripts/headless_run.py` on a single central-board episode. Does the scripted controller still succeed after your geometry changes? If not, a geometry or config regression was introduced.
2. Check that `GRASP_HEIGHT` in `config.py` is at least 5 mm above `TABLE_HEIGHT`. If they are equal, the descend target is exactly on the table surface and the gripper never contacts the piece.
3. Verify the observation vector with `test_env_obs.py`. Print `obs["observation"][3:6]` (piece position) during an episode and confirm it tracks the piece.
4. Check that `W_GRASP_BONUS` is being triggered. Add a temporary log line: `if phase == "approach" and fingers_closed and piece_above_table: print("GRASP BONUS TRIGGERED")`.

**Fixes:**
- If the problem is geometry: re-run `scripts/generate_xml.py` and verify the MJCF before resuming
- If the problem is reward: check that the grasp bonus is phase-gated correctly and is not being awarded every step
- If the problem is observation: re-verify indices 0–10 against the FetchPickAndPlace-v4 source

---

### 18.4 Good Grasp Rate But Poor Placement (Stage 2+)

**What this looks like:** Grasp rate ≥ 90%, but placement rate stays below 50%. The piece is picked up but then dropped or lands far from the goal.

**Causes and fixes:**

| Cause | Fix |
|-------|-----|
| Transport reward signal is too weak relative to grasp reward | Increase `W_TRANSPORT` from 3.0 → 5.0 |
| HER is not relabeling transport trajectories properly | Verify `goal_selection_strategy="future"` — `"final"` is worse for long-horizon transport |
| Gripper command is releasing the piece mid-transport | The policy is still reverting to the pretrained "release at goal" behavior on goals that are near the piece's current position. Increase `max_steps` to give the policy more time after grasp |
| Piece tipping during transport | Increase `W_TILT_PENALTY` from −1.0 → −3.0 |
| Destination far from training distribution | Check curriculum stage — do not jump to full board before Stage 2 gates are met |

---

### 18.5 Training Diverges on a Specific Curriculum Stage

**What this looks like:** The policy passes Stage 2 comfortably but destabilizes on Stage 3 (full board).

**Cause:** The distribution shift between Stage 2 (central 4×4 region) and Stage 3 (all 64 squares) is large enough to cause catastrophic forgetting.

**Fix — Gradual curriculum blending:**
Instead of switching stages abruptly, blend them:

```python
class GradualCurriculumEnv(ChessManipulationTrainEnv):
    """
    Instead of advancing stages discretely, blend the current and next stage
    over a transition window of N episodes.
    """
    def __init__(self, *args, blend_window: int = 10_000, **kwargs):
        super().__init__(*args, **kwargs)
        self._blend_window = blend_window
        self._blend_step   = 0
        self._next_stage   = self.curriculum_stage + 1

    def reset(self, **kwargs):
        # Probability of drawing from the next stage increases linearly
        p_next = min(1.0, self._blend_step / self._blend_window)
        if self._rng.random() < p_next and self._next_stage <= 7:
            self._sampler._stage = self._next_stage
        else:
            self._sampler._stage = self.curriculum_stage
        self._blend_step += 1
        return super().reset(**kwargs)
```

This creates a smooth transition over 10k episodes rather than an abrupt jump.

---

### 18.6 Learning Rate Schedule Recommendations

For fine-tuning from a pretrained checkpoint, the learning rate schedule matters significantly more than when training from scratch.

```
Steps 0 → 50k:     LR = 1e-4     (conservative warm-up, preserve pretrained features)
Steps 50k → 200k:  LR = 3e-4     (accelerate once stability is confirmed)
Steps 200k → 400k: LR = 1e-4     (back off as policy converges)
Steps 400k+:       LR = 3e-5     (fine-grained convergence)
```

Implement this as a linear schedule:

```python
from stable_baselines3.common.utils import get_linear_fn

# In train_robochess_policy.py:
lr_schedule = get_linear_fn(start=1e-4, end=3e-5, end_fraction=0.8)
model = TQC.load(..., learning_rate=lr_schedule)
```

---

## 19. Checkpoint and Compatibility Edge Cases

### 19.1 Loading a TQC Checkpoint Into a Different Observation Shape

**Symptom:** After adding the 18-dim chess context (Section 4.2), you get:
```
ValueError: Error: the input observations were inconsistent with the actor network (43 != 25)
```

**What's happening:** The saved checkpoint has actor/critic network weights shaped for a 25-dim input. You cannot directly load it into a 43-dim env.

**Solution — Weight surgery:**

```python
# scripts/expand_checkpoint.py
"""
Loads the pretrained 25-dim TQC checkpoint and expands the first layer of
the actor and critic networks to accept 43-dim input.
The original 25-dim weights are preserved in their original positions.
The 18 new dimensions are initialized to near-zero random weights.
"""

import torch
import numpy as np
from sb3_contrib import TQC
from src.env.chess_manipulation_train_env import ChessManipulationTrainEnv

OLD_OBS_DIM = 25
NEW_OBS_DIM = 43
DELTA       = NEW_OBS_DIM - OLD_OBS_DIM   # = 18

# Load old policy
old_env   = ...   # a env with obs_dim=25 (original ChessPickPlaceEnv)
old_model = TQC.load("sb3/tqc-FetchPickAndPlace-v1", env=old_env)

# Create new model with the expanded env
new_env   = ChessManipulationTrainEnv(curriculum_stage=1)
new_model = TQC("MultiInputPolicy", new_env, verbose=1)

# Transfer weights, expanding the first layer
def expand_first_layer(old_weight: torch.Tensor, delta: int) -> torch.Tensor:
    """
    Expand a weight tensor along the input dimension.
    Shape: (out_features, old_in) → (out_features, old_in + delta)
    New columns are initialized to 1/100th of the original weight std.
    """
    out_features, old_in = old_weight.shape
    std  = old_weight.std().item()
    new_weight = torch.zeros(out_features, old_in + delta)
    new_weight[:, :old_in] = old_weight
    new_weight[:, old_in:] = torch.randn(out_features, delta) * (std / 100)
    return new_weight

# Apply to actor and critic first layers
with torch.no_grad():
    for net in [new_model.actor, new_model.critic]:
        first_layer = list(net.parameters())[0]
        old_first   = list(old_model.actor.parameters())[0]  # adjust for critic
        first_layer.data = expand_first_layer(old_first.data, DELTA)

new_model.save("checkpoints/expanded/tqc_expanded_43dim")
print("Expanded checkpoint saved.")
```

After this surgery, the first training steps may be slightly unstable as the new weights settle, but the original features are preserved.

> **Alternative:** If weight surgery feels risky, a simpler approach is to train a **projection layer** that maps the 43-dim obs down to 25-dim before it reaches the pretrained actor. This is an adapter pattern that requires zero modification to the checkpoint:
>
> ```python
> class ProjectedObsWrapper(gym.ObservationWrapper):
>     """Maps 43-dim obs → 25-dim by ignoring the context during the initial phase."""
>     def observation(self, obs):
>         obs["observation"] = obs["observation"][:25]
>         return obs
> ```
> Use this wrapper during Stage 1, then switch to the full 43-dim obs starting from Stage 2 after the checkpoint has adapted to the chess scene.

---

### 19.2 SB3 Version Mismatch When Loading Checkpoints

**Symptom:**
```
KeyError: 'use_sde' or similar missing key when loading checkpoint
```

SB3 checkpoints are version-sensitive. The `tqc-FetchPickAndPlace-v1` checkpoint was saved under an older version. When loading across versions, SB3 sometimes raises missing-key errors.

**Fix:** Load with `custom_objects` to override stale parameters:

```python
custom_objects = {
    "learning_rate":      1e-4,
    "lr_schedule":        lambda _: 1e-4,
    "clip_range":         lambda _: 0.2,
    "n_steps":            2048,
    "batch_size":         256,
}
model = TQC.load(checkpoint_path, env=env, custom_objects=custom_objects)
```

This tells SB3 to use the provided values for any parameters missing from the checkpoint rather than raising an error.

---

### 19.3 HER Buffer Cannot Be Saved and Reloaded

**Symptom:** When using `save_replay_buffer=True` in `CheckpointCallback`, the buffer save succeeds but loading it on resume fails with an observation shape mismatch.

**Cause:** The HER buffer stores observations at the shape they were created with. If you change the observation space between runs (e.g., during checkpoint expansion), the buffer is incompatible.

**Fix:** Never try to reload a HER replay buffer across observation shape changes. Treat the buffer as a training artifact that is rebuilt from scratch each run. Use:

```python
# Always start with a fresh buffer, regardless of whether the policy is resumed
model.replay_buffer.reset()
```

This is safe because HER buffers fill quickly (within the first few thousand steps) and the policy weights carry the real knowledge — not the buffer.

---

### 19.4 Checkpoint Corruption Due to Interrupted Save

**Symptom:** Loading a checkpoint raises `EOFError` or `zipfile.BadZipFile`.

SB3 saves checkpoints as zip archives. An interrupted save produces a partial file that cannot be opened.

**Prevention:** Save to a temporary file and atomically rename:

```python
# In CheckpointCallback subclass:
import tempfile, os, shutil

def _on_step(self) -> bool:
    if self.n_calls % self.save_freq == 0:
        tmp_path = self.save_path + f"/tmp_{self.num_timesteps}.zip"
        final_path = self.save_path + f"/tqc_robochess_{self.num_timesteps}_steps"
        self.model.save(tmp_path)
        shutil.move(tmp_path, final_path + ".zip")   # atomic on POSIX systems
    return True
```

---

### 19.5 Gymnasium vs. Gym API Compatibility

The logs show:
```
UserWarning: You loaded a model that was trained using OpenAI Gym.
We strongly recommend transitioning to Gymnasium...
```

This warning is safe to suppress — SB3's `patch_gym` module translates between APIs automatically. However, it becomes a **real error** if you do any of the following:

1. Mix `gym.spaces` and `gymnasium.spaces` in the same code — always use one or the other. This project uses Gymnasium.
2. Return a 5-tuple `(obs, reward, done, truncated, info)` from `step()` in a wrapper that inherits from the old 4-tuple `gym.Env` API. Always return the 5-tuple Gymnasium format.
3. Use `env.seed(42)` instead of `env.reset(seed=42)`. The old `seed()` method does not exist in Gymnasium.

To suppress the warning cleanly:
```python
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*trained using OpenAI Gym.*",
    category=UserWarning,
)
```

---

## 20. Experiment Tracking

### 20.1 TensorBoard (Built-In, Zero Config)

SB3 writes TensorBoard logs automatically when `tensorboard_log` is set:

```python
model = TQC.load(
    ...,
    tensorboard_log = "runs/robochess",
)
model.learn(..., tb_log_name=f"stage{stage}_seed{seed}")
```

Launch the dashboard:
```bash
tensorboard --logdir runs/robochess --port 6006
```

On a remote cloud machine, forward the port:
```bash
# On your local machine:
ssh -L 6006:localhost:6006 user@your-cloud-machine
# Then open http://localhost:6006 in your browser
```

**Key scalars to monitor in TensorBoard:**

| Scalar | What it tells you |
|--------|------------------|
| `rollout/ep_rew_mean` | Average reward per episode — primary health indicator |
| `rollout/ep_len_mean` | Average episode length — should decrease as policy improves |
| `train/actor_loss` | Should decrease steadily; spike = instability |
| `train/critic_loss` | Should decrease; high variance = reward function issue |
| `train/ent_coef` | Entropy coefficient — if it collapses to 0, policy is becoming deterministic prematurely |
| `train/learning_rate` | Confirms your schedule is being applied |
| Custom: `eval/grasp_rate` | Log this manually at each eval using `model.logger.record()` |
| Custom: `eval/place_rate` | Same |

---

### 20.2 Weights & Biases (Recommended for Multi-Run Comparison)

W&B is the best tool for comparing multiple training runs (different seeds, stages, hyperparameters). It is free for personal use.

```bash
pip install wandb
wandb login   # one-time setup, enter your API key from wandb.ai
```

Integrate into the training script:

```python
# scripts/train_robochess_policy.py — add at the top of main():
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WandbCallback(BaseCallback):
    def __init__(self, stage: int, run, verbose: int = 0):
        super().__init__(verbose)
        self._run   = run
        self._stage = stage

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            self._run.log({
                "timestep":      self.num_timesteps,
                "ep_rew_mean":   self.locals.get("infos", [{}])[0].get("episode", {}).get("r", 0),
                "stage":         self._stage,
            }, step=self.num_timesteps)
        return True

# In main(), before model.learn():
run = wandb.init(
    project = "robochess-finetune",
    name    = f"stage{args.stage}_seed{args.seed}",
    config  = {
        "stage":           args.stage,
        "total_timesteps": args.total_timesteps,
        "n_envs":          args.n_envs,
        "seed":            args.seed,
        "learning_rate":   LEARNING_RATE,
        "batch_size":      BATCH_SIZE,
        "n_her_goals":     N_HER_GOALS,
    },
    sync_tensorboard = True,   # auto-sync SB3 TensorBoard logs to W&B
)
wandb_cb = WandbCallback(stage=args.stage, run=run)

# Add to callbacks list:
callbacks = [checkpoint_cb, eval_cb, wandb_cb]
model.learn(..., callback=callbacks)

run.finish()
```

**W&B features especially useful for this project:**

- **Sweep:** Automatically run hyperparameter searches over learning rate, batch size, and HER goal count. Define a sweep config:
  ```yaml
  # sweeps/stage1_sweep.yaml
  program: scripts/train_robochess_policy.py
  method: bayes   # Bayesian optimization
  metric:
    name: eval/place_rate
    goal: maximize
  parameters:
    learning_rate:
      values: [1e-5, 3e-5, 1e-4, 3e-4]
    batch_size:
      values: [128, 256]
    n_her_goals:
      values: [2, 4, 8]
  ```
  ```bash
  wandb sweep sweeps/stage1_sweep.yaml
  wandb agent <sweep_id>
  ```

- **Artifacts:** Store checkpoints as W&B artifacts so they are versioned and downloadable from any machine:
  ```python
  artifact = wandb.Artifact(f"checkpoint_stage{stage}", type="model")
  artifact.add_file("checkpoints/stage1/best/best_model.zip")
  run.log_artifact(artifact)
  ```

---

### 20.3 Minimal Logging Without W&B or TensorBoard

If you are on a constrained environment (Colab free tier, no internet), log to CSV:

```python
import csv, time, os

class CSVLogger:
    def __init__(self, path: str):
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "timestep", "stage",
                             "grasp_rate", "place_rate", "ep_rew_mean"])

    def log(self, timestep, stage, grasp_rate, place_rate, ep_rew_mean):
        with open(self._path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                timestep, stage, grasp_rate, place_rate, ep_rew_mean
            ])
```

---

## 21. Debugging and Profiling

### 21.1 Visualizing What the Policy Is Doing

During debugging, you want to watch the policy actually move the arm. The training env is headless, but you can launch a dedicated visualization session alongside training:

```python
# scripts/visualize_policy.py
"""
Load a checkpoint and run it with the MuJoCo viewer active.
Run this SEPARATELY from training — do not mix viewer and training.

Usage:
  python scripts/visualize_policy.py \
    --checkpoint checkpoints/stage1/best/best_model.zip \
    --stage 1 \
    --n-episodes 10
"""
import mujoco.viewer
from sb3_contrib import TQC
from src.env.chess_manipulation_train_env import ChessManipulationTrainEnv

def main(args):
    env = ChessManipulationTrainEnv(curriculum_stage=args.stage, seed=42)
    model = TQC.load(args.checkpoint, env=env)

    with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
        for ep in range(args.n_episodes):
            obs, _ = env.reset()
            done = False
            step = 0
            print(f"\n--- Episode {ep+1}: {env._episode.piece_name} "
                  f"{env._episode.src_square} → {env._episode.dst_square} ---")
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                viewer.sync()
                done = term or trunc
                step += 1

            print(f"Result: {'SUCCESS' if info['is_success'] else 'FAILURE'} "
                  f"| Steps: {step} | Grasp: {info['grasped']} "
                  f"| up_z: {info['up_z']:.3f}")
```

---

### 21.2 Profiling Training Throughput

Bottlenecks in RL training with MuJoCo are almost always in the environment (physics stepping), not the network. To identify where time is spent:

```python
# Add to train_robochess_policy.py during debugging:
import cProfile, pstats, io

def profile_training_step(model, vec_env, n_steps: int = 1000):
    pr = cProfile.Profile()
    pr.enable()
    model.learn(total_timesteps=n_steps, reset_num_timesteps=False)
    pr.disable()

    s  = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)   # top 20 functions by cumulative time
    print(s.getvalue())
```

Typical findings:
- `mujoco.mj_step` takes 60–80% of environment time → this is normal and irreducible
- `_build_context()` takes > 5% → optimize `_get_neighbor_offsets()` (cache body IDs)
- `SubprocVecEnv` IPC takes > 20% → switch to `DummyVecEnv` or reduce `n_envs`

**Optimize `_get_neighbor_offsets()` for production:**
```python
# Pre-compute body IDs once in __init__ instead of looking them up by name each step
def __init__(self, ...):
    ...
    self._piece_body_ids = {
        name: self._model.body(name).id for name in PIECE_NAMES
    }

def _get_neighbor_offsets(self, target_name, target_pos):
    dists = []
    for name, body_id in self._piece_body_ids.items():
        if name == target_name:
            continue
        pos = self._data.xpos[body_id]
        if pos[2] < TABLE_HEIGHT - 0.05:
            continue
        d = np.linalg.norm(pos[:2] - target_pos[:2])
        dists.append((d, pos[:2] - target_pos[:2]))
    dists.sort(key=lambda x: x[0])
    return [v for (_, v) in dists[:2]]
```

---

### 21.3 Inspecting the Replay Buffer

If you suspect the buffer contains corrupted or off-distribution data:

```python
# Inspect the most recent 100 transitions in the buffer
buf = model.replay_buffer
idx_start = max(0, buf.pos - 100)
idx_end   = buf.pos

obs_sample    = buf.observations["observation"][idx_start:idx_end]
action_sample = buf.actions[idx_start:idx_end]
reward_sample = buf.rewards[idx_start:idx_end]

print(f"Obs range:    [{obs_sample.min():.3f}, {obs_sample.max():.3f}]")
print(f"Action range: [{action_sample.min():.3f}, {action_sample.max():.3f}]")
print(f"Reward range: [{reward_sample.min():.3f}, {reward_sample.max():.3f}]")
print(f"NaN in obs:   {np.any(np.isnan(obs_sample))}")
print(f"NaN in reward:{np.any(np.isnan(reward_sample))}")
```

If `obs_sample.max()` is on the order of 1e6 or reward range is wildly asymmetric, the reward function or observation builder has a bug.

---

### 21.4 Quick Smoke Test Before Any Full Training Run

Before committing to a long training run (hours), always run this 60-second smoke test:

```bash
python scripts/train_robochess_policy.py \
    --stage 1 \
    --total-timesteps 1000 \
    --n-envs 2 \
    --seed 0
```

This should:
1. Load the checkpoint without error
2. Initialize the envs without error
3. Run 1000 steps without NaN, crash, or hang
4. Print a non-zero `ep_rew_mean` (even if negative — it should not be exactly 0.0 or −inf)
5. Save a checkpoint to `checkpoints/stage1/`

If all five conditions are met, proceed to the full run. If any fails, fix it before burning compute hours.

---

## Appendix C: Pre-Training Checklist

Run through this list in order before starting any training. Check each item off before proceeding.

```
PRE-TRAINING CHECKLIST
═══════════════════════════════════════════════════════════

Prerequisites
─────────────
[ ] Table height fix applied (chess_world.xml, TABLE_HEIGHT = 0.422)
[ ] Graveyard cells verified reachable (all satisfy _is_reachable())
[ ] Collision/visual geometry unified in scripts/generate_xml.py
[ ] scripts/headless_run.py completes without error (single move, no viewer)
[ ] All 32 pieces initialized with up_z >= 0.999 (board_observer scan)
[ ] robot0:mocap bug fixed in board_observer.py ✓ (already done)
[ ] requirements.txt pinned (numpy<2.0.0, sb3-contrib, etc.)
[ ] MUJOCO_GL set to egl or osmesa (for headless environments)

Environment Validation
──────────────────────
[ ] test_env_obs.py passes (obs shape (43,), no NaN)
[ ] test_reward_fn.py passes (reward not always 0)
[ ] EpisodeSampler produces reachable episodes (no reachability warnings)
[ ] 10 manual reset+step cycles complete without physics crash
[ ] DummyVecEnv(n_envs=2) smoke test runs for 500 steps

Compute Setup
─────────────
[ ] Checkpoint dir writable and on persistent storage
[ ] TensorBoard or W&B configured
[ ] eval_seeds.json committed (100 fixed seeds per suite)
[ ] If cloud: SSH tunnel or notebook ready for TensorBoard viewing
[ ] If cloud: tmux session created for background training

Training Configuration
──────────────────────
[ ] LEARNING_RATE = 1e-4 (not default 3e-4)
[ ] HerReplayBuffer with goal_selection_strategy="future"
[ ] CHECKPOINT_FREQ = 5000 (granular enough to recover from divergence)
[ ] EVAL_FREQ = 10000
[ ] 60-second smoke test passed (1000 timesteps, n_envs=2)

Only after all boxes are checked: start the full training run.
```

---

## Appendix D: Cloud Cost Reference (April 2026)

All prices are approximate and change frequently. Verify at provider's pricing page before committing.

| Platform | GPU | VRAM | Cost/hr | Est. Stage 1 Time | Est. Stage 1 Cost | Notes |
|----------|-----|------|---------|-------------------|-------------------|-------|
| Colab (free) | T4 | 16 GB | $0 | 3–5 h | $0 | Session timeout; free tier throttled |
| Colab Pro | T4/A100 | 16–40 GB | ~$0.45 effective | 2–4 h | $1–2 | More reliable, longer sessions |
| Kaggle | P100/T4 | 16 GB | $0 | 4–6 h | $0 | 30 h/week cap |
| RunPod | RTX 3090 | 24 GB | ~$0.40 | 2–3 h | $0.80–1.20 | Persistent storage available |
| RunPod | A100 40G | 40 GB | ~$1.50 | 1–1.5 h | $1.50–2.25 | Best value for full curriculum |
| Lambda Labs | A100 80G | 80 GB | ~$1.29 | 45–90 min | $1–2 | Reliable, clean environment |
| Vast.ai | RTX 4090 | 24 GB | ~$0.25 | 1.5–2 h | $0.40–0.50 | Spot market, can be reclaimed |
| Modal | A10G | 24 GB | ~$0.54 | 2–3 h | $1–1.60 | Serverless, per-second billing |
| GCP (spot) | A100 | 40 GB | ~$1.10 | 1–1.5 h | $1.10–1.65 | Most overhead to set up |
| AWS (spot) | V100 | 16 GB | ~$1.00 | 2–3 h | $2–3 | Familiar tooling, higher cost |

**Full curriculum (Stages 1–6) cost estimate:**
- ~4–6M total timesteps expected across all stages
- RTX 3090 at RunPod: **$5–10 total**
- A100 at Lambda Labs: **$12–20 total**

---

## Appendix E: Recommended Reading

These resources are directly applicable to the techniques used in this project.

**Hindsight Experience Replay (HER)**  
Andrychowicz et al., 2017. *Hindsight Experience Replay.*  
The foundational paper for HER. Read Section 3 carefully — the "final" vs. "future" goal selection strategies are described here and the choice matters significantly for manipulation tasks.

**Soft Actor-Critic (SAC)**  
Haarnoja et al., 2018. *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.*  
TQC (which this project uses) is a direct extension of SAC. Understanding SAC's entropy regularization explains why `ent_coef` collapsing is a warning sign.

**Truncated Quantile Critics (TQC)**  
Kuznetsov et al., 2020. *Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics.*  
The algorithm this project uses. TQC is more stable than SAC on manipulation tasks because it reduces Q-value overestimation — the main failure mode for sparse-reward goal-conditioned environments.

**Curriculum Learning for RL**  
Bengio et al., 2009. *Curriculum Learning.*  
The theoretical foundation for the staged curriculum in Section 7. The key result: ordering training examples from simple to complex accelerates learning compared to random sampling.

**Fetch Manipulation Environments**  
Plappert et al., 2018. *Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research.*  
Describes the exact Fetch environments used for pretraining. Section 3 explains the 25-dim observation format that must be preserved for warm-start compatibility.

**MuJoCo Documentation**  
https://mujoco.readthedocs.io/en/stable/  
The `mujoco.MjModel` and `mujoco.MjData` field reference (Section: *API Reference*) is essential for understanding body pose access, contact detection, and physics parameter tuning.

**Stable-Baselines3 Documentation**  
https://stable-baselines3.readthedocs.io/  
Specifically: the `HerReplayBuffer` API, `TQC` usage, and the `EvalCallback` API. The SB3 docs are well-maintained and include working examples for GoalEnv training.

---

*End of Implementation Specification v2*
