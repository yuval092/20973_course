"""
ChessManipulationTrainEnv (Section 3)
─────────────────────────────────────
A standalone GoalEnv for fine-tuning the Fetch pick-and-place policy
on the RoboChess scene.
"""

from __future__ import annotations

import logging
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from src.config import (
    TABLE_HEIGHT, N_SUBSTEPS, SCENE_XML, REACHABLE_X_MIN, REACHABLE_X_MAX,
    REACHABLE_Y_MIN, REACHABLE_Y_MAX, REACHABLE_Z_MIN, REACHABLE_Z_MAX,
    GOAL_TOLERANCE
)
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.env.episode_sampler import EpisodeSampler

logger = logging.getLogger(__name__)

OBS_DIM      = 26   # base Fetch observation (including time feature)
CONTEXT_DIM  = 18   # chess-scene context
TOTAL_OBS    = OBS_DIM + CONTEXT_DIM  # 44


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

        # Load MuJoCo scene fresh for each env instance
        self._model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self._data  = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Inner env provides the obs builder and physics step
        self._env = ChessPickPlaceEnv(self._model, self._data, n_substeps=N_SUBSTEPS)

        # Episode sampler drives curriculum
        self._sampler = EpisodeSampler(self._rng, curriculum_stage)

        # Spaces
        obs_low  = np.full(TOTAL_OBS, -np.inf, dtype=np.float32)
        obs_high = np.full(TOTAL_OBS, np.inf,  dtype=np.float32)
        
        # Reachable bounds from config
        goal_low  = np.array([REACHABLE_X_MIN, REACHABLE_Y_MIN, REACHABLE_Z_MIN], dtype=np.float32)
        goal_high = np.array([REACHABLE_X_MAX, REACHABLE_Y_MAX, REACHABLE_Z_MAX], dtype=np.float32)

        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(obs_low, obs_high, dtype=np.float32),
            "achieved_goal": spaces.Box(goal_low, goal_high, dtype=np.float32),
            "desired_goal":  spaces.Box(goal_low, goal_high, dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # Episode state
        self._step_count    = 0
        self._episode       = None   
        self._phase         = "approach"
        self._grasped       = False
        self._prev_piece_xy = None

    def set_curriculum_stage(self, stage: int):
        """Update the curriculum stage dynamically."""
        self.curriculum_stage = stage
        self._sampler._stage = stage

    # ────────────────────────────────────────────────
    # Core GoalEnv interface
    # ────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._sampler._rng = self._rng

        # Sample an episode from the curriculum
        self._episode = self._sampler.sample()

        # Reset physics to initial layout, then teleport the target piece
        # to the episode's source square
        self._reset_physics()
        self._teleport_piece(self._episode.piece_name, self._episode.src_pos)

        # Reset the inner Fetch env (arm back to home position, gripper open)
        self._env.set_target(self._episode.piece_name, self._episode.dest_pos)
        inner_obs = self._env.get_obs()

        self._step_count    = 0
        self._phase         = "approach"
        self._grasped       = False
        self._prev_piece_xy = self._get_piece_xy()
        self._lifted_max_z  = 0.0
        self._initial_piece_positions = self._snapshot_all_piece_positions()

        obs = self._build_obs(inner_obs)
        return obs, {}

    def step(self, action: np.ndarray):
        # Apply one physics step through the inner env
        inner_obs, _, _, info = self._env.step(action)

        self._step_count += 1
        self._update_phase(inner_obs)
        
        piece_pos = inner_obs["achieved_goal"]
        self._lifted_max_z = max(self._lifted_max_z, float(piece_pos[2]))

        obs        = self._build_obs(inner_obs)
        achieved   = obs["achieved_goal"]
        desired    = obs["desired_goal"]
        
        # Check for penalties
        neighbor_displaced = self._check_neighbor_displaced()
        piece_dropped = False
        if self._grasped and piece_pos[2] < TABLE_HEIGHT + 0.005 and self._lifted_max_z > TABLE_HEIGHT + 0.02:
            piece_dropped = True

        # Populate info for reward function
        info.update({
            "phase":             self._phase,
            "grasped":           self._grasped,
            "grip_pos":          inner_obs["observation"][0:3],
            "piece_pos":         piece_pos,
            "up_z":              self._get_up_z(),
            "finger_qpos":       inner_obs["observation"][9:11],
            "table_height":      TABLE_HEIGHT,
            "neighbor_displaced": neighbor_displaced,
            "piece_dropped":      piece_dropped,
            "timed_out":         self._step_count >= self.max_steps,
        })

        reward     = float(self.compute_reward(achieved, desired, info))
        terminated = self._is_success(achieved, desired)
        truncated  = self._step_count >= self.max_steps

        info.update({
            "is_success":        terminated,
            "placement_error_m": np.linalg.norm(achieved - desired),
        })

        self._prev_piece_xy = self._get_piece_xy()
        return obs, reward, terminated, truncated, info

    def _snapshot_all_piece_positions(self) -> dict[str, np.ndarray]:
        from src.config import PIECE_NAMES
        snapshot = {}
        for name in PIECE_NAMES:
            try:
                snapshot[name] = self._data.body(name).xpos.copy()
            except KeyError:
                continue
        return snapshot

    def _check_neighbor_displaced(self) -> bool:
        """Return True if any piece other than the target moved > 5 mm."""
        target_name = self._episode.piece_name
        for name, initial_pos in self._initial_piece_positions.items():
            if name == target_name:
                continue
            # Skip pieces that are below table (hidden spares)
            if initial_pos[2] < TABLE_HEIGHT - 0.05:
                continue
            current_pos = self._data.body(name).xpos
            if np.linalg.norm(current_pos[:2] - initial_pos[:2]) > 0.005:
                return True
        return False

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> float:
        """
        Dense shaped reward. Must be separable (no self state) for HER compatibility.
        """
        from src.env.reward_fn import compute_reward
        return compute_reward(achieved_goal, desired_goal, info)

    def render(self):
        pass  # Headless training.

    # ────────────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────────────

    def _reset_physics(self):
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

    def _teleport_piece(self, piece_name: str, pos: np.ndarray):
        try:
            body = self._data.body(piece_name)
            jnt_id = self._model.body(piece_name).jntadr[0]
            if jnt_id != -1:
                self._data.joint(jnt_id).qpos[:3] = pos
                self._data.joint(jnt_id).qpos[3:7] = [1, 0, 0, 0]   # upright
                self._data.joint(jnt_id).qvel[:] = 0
            mujoco.mj_forward(self._model, self._data)
        except KeyError:
            logger.error(f"Piece {piece_name} not found for teleport")

    def _build_obs(self, inner_obs: dict) -> dict:
        base_obs = inner_obs["observation"].astype(np.float32)      # (25,)
        context  = self._build_context().astype(np.float32)         # (18,)
        return {
            "observation":   np.concatenate([base_obs, context]),
            "achieved_goal": inner_obs["achieved_goal"].astype(np.float32),
            "desired_goal":  inner_obs["desired_goal"].astype(np.float32),
        }

    def _build_context(self) -> np.ndarray:
        from src.env.context_builder import build_context
        return build_context(self._model, self._data, self._episode, self._grasped)

    def _update_phase(self, inner_obs: dict):
        piece_pos  = inner_obs["achieved_goal"]
        piece_lift = piece_pos[2] - TABLE_HEIGHT

        if not self._grasped:
            if piece_lift > 0.01:          # 1 cm above table → grasped
                self._grasped = True
                self._phase   = "transport"
            else:
                self._phase   = "approach"
        else:
            # Once grasped, we transition to placement when near destination XY
            dist_to_dest_xy = np.linalg.norm(piece_pos[:2] - self._episode.dest_pos[:2])
            if dist_to_dest_xy < 0.03:
                self._phase = "placement"
            else:
                self._phase = "transport"

    def _is_success(self, achieved: np.ndarray, desired: np.ndarray) -> bool:
        xy_err = np.linalg.norm(achieved[:2] - desired[:2])
        z_err  = abs(achieved[2] - desired[2])
        up_z   = self._get_up_z()
        # Criteria from plan Section 3.2
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
