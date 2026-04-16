"""
Custom GoalEnv wrapper that bridges the MuJoCo chess scene to the
pretrained FetchPickAndPlace SAC+HER policy.

This wrapper presents the same observation/action interface
as the pretrained checkpoint used by this project:
  - Observation dict: {observation: (26,), achieved_goal: (3,), desired_goal: (3,)}
  - Action space: Box(-1, 1, shape=(4,))  [dx, dy, dz, gripper]
  - Reward: sparse (-1 if not at goal, 0 if at goal)

The action application pipeline mirrors _set_action() from the training env:
  1. Scale position delta by 0.05
  2. Apply as mocap position offset
  3. Mirror gripper command to both finger actuators
  4. Step physics for n_substeps sub-steps
"""

import logging
import time
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium_robotics.utils import mujoco_utils

from src.config import (
    N_SUBSTEPS, ACTION_SCALE, GOAL_TOLERANCE,
    GRIPPER_OPEN, GRIPPER_CLOSED,
    L_FINGER_ACTUATOR, R_FINGER_ACTUATOR, GRIP_SITE,
    BOARD_CENTER, SQUARE_SIZE, WORKSPACE_XY_MARGIN,
    WORKSPACE_Z_MIN, WORKSPACE_Z_MAX, VIEWER_STEP_DELAY_SEC,
    FETCH_INIT_GRIP, FETCH_POLICY_HORIZON,
)
from src.env.obs_wrapper import reconstruct_obs

logger = logging.getLogger(__name__)

IDLE_PIECE_FREEJOINT_DAMPING = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 200.0], dtype=np.float64)
ACTIVE_PIECE_FREEJOINT_DAMPING = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float64)


class ChessPickPlaceEnv(gym.Env):
    """
    Wraps the MuJoCo chess scene so the pretrained FetchPickAndPlace
    policy can operate on individual chess pieces.

    Usage:
        env = ChessPickPlaceEnv(mj_model, mj_data)
        env.set_target("w_pawn_1", goal_xyz)
        obs = env.get_obs()
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    """

    def __init__(self, mj_model, mj_data, n_substeps=N_SUBSTEPS):
        super().__init__()
        self.model = mj_model
        self.data = mj_data
        self.n_substeps = n_substeps

        # Resolve actuator IDs once
        self._l_actuator_id = self.model.actuator(L_FINGER_ACTUATOR).id
        self._r_actuator_id = self.model.actuator(R_FINGER_ACTUATOR).id

        # Current operation target (set per operation)
        self._target_body_name = None
        self._target_site_name = None
        self._goal = None
        self._elapsed_steps = 0
        self._policy_horizon = FETCH_POLICY_HORIZON
        self._robot_home_joint_state = None
        self._home_mocap_pos = None
        self._home_mocap_quat = None
        self._home_grip_pos = None
        self._piece_mesh_geom_ids = self._resolve_piece_mesh_geom_ids()
        self._piece_joint_dofadrs = self._resolve_piece_joint_dofadrs()
        self._piece_mesh_contact_masks = {
            name: (
                int(self.model.geom_contype[geom_id]),
                int(self.model.geom_conaffinity[geom_id]),
            )
            for name, geom_id in self._piece_mesh_geom_ids.items()
        }
        for piece_name in self._piece_mesh_geom_ids:
            self.set_piece_mesh_collision_enabled(piece_name, enabled=False)
            self.set_piece_active_damping(piece_name, active=False)

        # board_half_extent = 4 * SQUARE_SIZE
        board_half_extent = 4 * SQUARE_SIZE
        self._mocap_min = np.array([
            BOARD_CENTER[0] - board_half_extent - WORKSPACE_XY_MARGIN,
            BOARD_CENTER[1] - board_half_extent - WORKSPACE_XY_MARGIN,
            WORKSPACE_Z_MIN,
        ], dtype=np.float64)
        self._mocap_max = np.array([
            BOARD_CENTER[0] + board_half_extent + WORKSPACE_XY_MARGIN,
            BOARD_CENTER[1] + board_half_extent + WORKSPACE_XY_MARGIN,
            WORKSPACE_Z_MAX,
        ], dtype=np.float64)

        # ─── Define spaces to match SB3 TQC checkpoint ────────────────────
        low = np.full(26, -np.inf, dtype=np.float32)
        low[-1] = 0.0
        high = np.full(26, np.inf, dtype=np.float32)
        high[-1] = 1.0

        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low, high, dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # ─── Fetch-Compatible Robot Initialization ────────────────────
        self.reset_robot_pose()

    def _capture_piece_joint_state(self):
        """Snapshot all chess-piece freejoint states before robot-only resets."""
        piece_state = {}
        for body_idx in range(self.model.nbody):
            body = self.model.body(body_idx)
            name = body.name
            if not name or not name.startswith(("w_", "b_")):
                continue
            joint_id = body.jntadr[0]
            if joint_id < 0:
                continue
            qpos_addr = self.model.jnt_qposadr[joint_id]
            qvel_addr = self.model.jnt_dofadr[joint_id]
            piece_state[name] = (
                self.data.qpos[qpos_addr:qpos_addr + 7].copy(),
                self.data.qvel[qvel_addr:qvel_addr + 6].copy(),
            )
        return piece_state

    def _restore_piece_joint_state(self, piece_state) -> None:
        """Restore chess-piece freejoint states after robot homing."""
        for name, (qpos, qvel) in piece_state.items():
            joint_id = self.model.body(name).jntadr[0]
            qpos_addr = self.model.jnt_qposadr[joint_id]
            qvel_addr = self.model.jnt_dofadr[joint_id]
            self.data.qpos[qpos_addr:qpos_addr + 7] = qpos
            self.data.qvel[qvel_addr:qvel_addr + 6] = qvel

    def reset_robot_pose(self):
        """
        Reset the robot to the same startup pose used by FetchPickAndPlace.

        The pretrained policy expects the gripper to begin from Fetch's
        canonical post-reset hover pose, not from MuJoCo's raw XML defaults.
        """
        piece_state = self._capture_piece_joint_state()
        if self._robot_home_joint_state is not None:
            for joint_name, joint_qpos, joint_qvel in self._robot_home_joint_state:
                joint_id = self.model.joint(joint_name).id
                qpos_addr = self.model.jnt_qposadr[joint_id]
                qvel_addr = self.model.jnt_dofadr[joint_id]
                self.data.qpos[qpos_addr] = joint_qpos
                self.data.qvel[qvel_addr] = joint_qvel
            self.data.mocap_pos[:] = self._home_mocap_pos
            self.data.mocap_quat[:] = self._home_mocap_quat
            if self.model.nu > 0:
                self.data.ctrl[:] = 0.0
            self._restore_piece_joint_state(piece_state)
            mujoco.mj_forward(self.model, self.data)
            logger.debug(f"Fetch robot reset | grip_xyz={self.get_grip_pos().round(4)}")
            return

        for joint_name, value in (
            ("robot0:slide0", 0.405),
            ("robot0:slide1", 0.48),
            ("robot0:slide2", 0.0),
        ):
            joint_id = self.model.joint(joint_name).id
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = value

        mujoco_utils.reset_mocap_welds(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        gripper_target = np.asarray(FETCH_INIT_GRIP, dtype=np.float64)
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)

        self.data.mocap_pos[0] = gripper_target
        self.data.mocap_quat[0] = gripper_rotation
        
        # Teleport pieces to a safe distance to avoid collisions 
        # during the homing motion.
        safe_pos = np.array([2.0, 2.0, 0.5])
        for name in piece_state:
            joint_id = self.model.body(name).jntadr[0]
            self.data.joint(joint_id).qpos[:3] = safe_pos
        
        for _ in range(10):
            mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

        # Robot homing should not disturb the manually placed chess setup.
        self._restore_piece_joint_state(piece_state)
        mujoco.mj_forward(self.model, self.data)

        self._robot_home_joint_state = []
        for joint_idx in range(self.model.njnt):
            joint_name = self.model.joint(joint_idx).name
            if not joint_name or not joint_name.startswith("robot0:"):
                continue
            qpos_addr = self.model.jnt_qposadr[joint_idx]
            qvel_addr = self.model.jnt_dofadr[joint_idx]
            self._robot_home_joint_state.append(
                (
                    joint_name,
                    float(self.data.qpos[qpos_addr]),
                    float(self.data.qvel[qvel_addr]),
                )
            )
        self._home_mocap_pos = self.data.mocap_pos.copy()
        self._home_mocap_quat = self.data.mocap_quat.copy()
        self._home_grip_pos = self.get_grip_pos()
        logger.debug(f"Fetch robot reset | grip_xyz={self.get_grip_pos().round(4)}")

    # ─── Configuration ───────────────────────────────────────────────

    def _resolve_piece_mesh_geom_ids(self) -> dict[str, int]:
        """Map each chess-piece body name to its visible mesh geom id."""
        mesh_geom_ids = {}
        for body_idx in range(self.model.nbody):
            body = self.model.body(body_idx)
            name = body.name
            if not name or not name.startswith(("w_", "b_")) or "spare" in name:
                continue
            geomnum = int(np.asarray(body.geomnum).item())
            geomadr = int(np.asarray(body.geomadr).item())
            for geom_offset in range(geomnum):
                geom_id = geomadr + geom_offset
                if self.model.geom_type[geom_id] in {mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_CYLINDER}:
                    mesh_geom_ids[name] = geom_id
                    break
        return mesh_geom_ids

    def _resolve_piece_joint_dofadrs(self) -> dict[str, int]:
        """Map each chess-piece body name to the start of its freejoint DOFs."""
        joint_dofadrs = {}
        for body_idx in range(self.model.nbody):
            body = self.model.body(body_idx)
            name = body.name
            if not name or not name.startswith(("w_", "b_")) or "spare" in name:
                continue
            joint_id = int(np.asarray(body.jntadr).item())
            if joint_id < 0:
                continue
            joint_dofadrs[name] = int(self.model.jnt_dofadr[joint_id])
        return joint_dofadrs

    def set_target(self, piece_name, goal_pos):
        """
        Configure which chess piece is the current manipulation target
        and where it should be placed.

        Args:
            piece_name: MuJoCo body name of the piece (e.g. "w_pawn_1")
            goal_pos: Target XYZ position as numpy array of shape (3,)
        """
        self._target_body_name = piece_name
        self._target_site_name = f"{piece_name}_site"
        self._goal = np.asarray(goal_pos, dtype=np.float64).copy()
        self._elapsed_steps = 0
        for other_piece in self._piece_joint_dofadrs:
            self.set_piece_active_damping(other_piece, active=(other_piece == piece_name))
        self.set_piece_mesh_collision_enabled(piece_name, enabled=True)

    def set_piece_mesh_collision_enabled(self, piece_name: str, enabled: bool) -> None:
        """Enable or disable the target piece's tall mesh collision proxy."""
        geom_id = self._piece_mesh_geom_ids.get(piece_name)
        if geom_id is None:
            return
        contype, conaffinity = self._piece_mesh_contact_masks[piece_name]
        self.model.geom_contype[geom_id] = contype if enabled else 0
        self.model.geom_conaffinity[geom_id] = conaffinity if enabled else 0

    def set_piece_active_damping(self, piece_name: str, active: bool) -> None:
        """Switch a piece between transport-friendly and idle-stable damping."""
        dofadr = self._piece_joint_dofadrs.get(piece_name)
        if dofadr is None:
            return
        damping = ACTIVE_PIECE_FREEJOINT_DAMPING if active else IDLE_PIECE_FREEJOINT_DAMPING
        self.model.dof_damping[dofadr:dofadr + 6] = damping

    @staticmethod
    def _expand_fetch_action(action):
        """
        Match FetchPickAndPlace action preprocessing exactly.

        The pretrained policy outputs 4D actions:
          [dx, dy, dz, gripper]

        The canonical Fetch env converts that into:
          [pos_delta(3), fixed_rot_quat(4), gripper(2)]
        """
        pos_ctrl = action[:3] * ACTION_SCALE
        rot_ctrl = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        gripper_ctrl = np.array([action[3], action[3]], dtype=np.float64)
        return np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

    # ─── Observation ─────────────────────────────────────────────────

    def get_obs(self):
        """
        Build and return the observation dict matching the
        FetchPickAndPlace training format.

        Returns:
            dict with keys: observation (26,), achieved_goal (3,), desired_goal (3,)
        """
        assert self._target_body_name is not None, "Call set_target() first"
        assert self._target_site_name is not None, "Call set_target() first"
        assert self._goal is not None, "Call set_target() first"

        obs_vec = reconstruct_obs(
            self.model, self.data,
            self._target_body_name,
            self._target_site_name,
            self.n_substeps,
            self.remaining_time_feature,
        )

        return {
            'observation': obs_vec,
            'achieved_goal': self.data.site(self._target_site_name).xpos.copy(),
            'desired_goal': self._goal.copy(),
        }

    # ─── Action Execution ────────────────────────────────────────────

    def step(self, action, viewer=None, debug=False, gripper_target=None):
        """
        Apply a 4D action to the simulation, exactly as the training
        environment does.

        The action application pipeline from fetch_env.py:
          1. pos_ctrl = action[:3] * 0.05
          2. mocap_pos += pos_ctrl
          3. ctrl[l_finger] = action[3]
          4. ctrl[r_finger] = action[3]
          5. mj_step × n_substeps

        Args:
            action: numpy array of shape (4,) — [dx, dy, dz, gripper]
            viewer: optional MuJoCo viewer to sync during stepping
            debug: enable detailed execution logs
            gripper_target: optional deterministic finger target in joint
                space. When provided, it overrides the action's gripper
                component after mocap processing and before physics stepping.

        Returns:
            obs: observation dict
            reward: float (sparse: 0.0 or -1.0)
            terminated: bool (always False — continuing task)
            info: dict with additional info
        """
        action = np.asarray(action, dtype=np.float64).copy()
        assert action.shape == (4,), f"Expected action shape (4,), got {action.shape}"

        # 1. Expand 4D policy action to the canonical Fetch mocap action.
        fetch_action = self._expand_fetch_action(action)
        pos_ctrl = fetch_action[:3]

        # 2. Apply action exactly like Fetch:
        #    - gripper actuators via ctrl_set_action()
        #    - mocap reset + delta via mocap_set_action()
        old_mocap = self.data.mocap_pos[0].copy()
        mujoco_utils.ctrl_set_action(self.model, self.data, fetch_action)
        mujoco_utils.mocap_set_action(self.model, self.data, fetch_action)
        unclipped_mocap = self.data.mocap_pos[0].copy()
        self.data.mocap_pos[0] = np.clip(unclipped_mocap, self._mocap_min, self._mocap_max)
        if gripper_target is not None:
            self.set_gripper_target(gripper_target)

        if debug:
            logger.debug(
                "[Env Step] Action Predict | "
                f"dx,dy,dz: {pos_ctrl.round(4)} | grip_ctrl: {action[3]:.4f}"
            )
            logger.debug(f"[Env Step] Mocap Before: {old_mocap.round(4)} | Target Mocap: {self.data.mocap_pos[0].round(4)}")
            if not np.allclose(unclipped_mocap, self.data.mocap_pos[0]):
                logger.debug(
                    "[Env Step] Mocap target clipped to workspace bounds: "
                    f"requested={unclipped_mocap.round(4)} "
                    f"clipped={self.data.mocap_pos[0].round(4)}"
                )

        # 4. Step physics
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps += 1

        if debug:
            logger.debug(f"[Env Step] Grip Post-step: {self.get_grip_pos().round(4)} | Piece Post-step: {self.get_piece_pos().round(4)}")

        # 5. Sync viewer if provided
        if viewer is not None:
            viewer.sync()
            if VIEWER_STEP_DELAY_SEC > 0.0:
                time.sleep(VIEWER_STEP_DELAY_SEC)

        # 6. Build observation
        obs = self.get_obs()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])

        info = {
            'is_success': reward == 0.0,
            'piece_pos': obs['achieved_goal'].copy(),
            'goal_pos': obs['desired_goal'].copy(),
        }

        return obs, reward, False, info

    # ─── Reward ──────────────────────────────────────────────────────

    @staticmethod
    def compute_reward(achieved_goal, desired_goal):
        """
        Sparse reward matching FetchPickAndPlace training.
        Returns 0.0 if within tolerance, -1.0 otherwise.
        """
        d = np.linalg.norm(achieved_goal - desired_goal)
        return 0.0 if d < GOAL_TOLERANCE else -1.0

    # ─── Gripper Control ─────────────────────────────────────────────

    def force_gripper_open(self):
        """
        Force the gripper to fully open position.
        Used as a safety measure after placement to ensure piece release.
        """
        self.set_gripper_target(GRIPPER_OPEN)

    def force_gripper_closed(self):
        """Force the gripper to fully closed position."""
        self.set_gripper_target(GRIPPER_CLOSED)

    def set_gripper_target(self, opening):
        """
        Set both finger actuator targets to a joint-space opening value.
        The finger joints themselves are limited to [0, 0.05].
        """
        opening = float(np.clip(opening, 0.0, 0.05))
        self.data.ctrl[self._l_actuator_id] = opening
        self.data.ctrl[self._r_actuator_id] = opening
        logger.debug(
            "Gripper target set | "
            f"opening={opening:.4f} | "
            f"ctrl_l={self.data.ctrl[self._l_actuator_id]:.4f} | "
            f"ctrl_r={self.data.ctrl[self._r_actuator_id]:.4f}"
        )

    # ─── Utilities ───────────────────────────────────────────────────

    def get_grip_pos(self):
        """Return current end-effector position."""
        return self.data.site(GRIP_SITE).xpos.copy()

    def get_piece_pos(self):
        """Return current target piece position."""
        assert self._target_body_name is not None
        return self.data.body(self._target_body_name).xpos.copy()

    def get_piece_linear_velocity(self):
        """Return the target piece linear velocity from its freejoint state."""
        assert self._target_body_name is not None
        joint_id = self.model.body(self._target_body_name).jntadr[0]
        qvel_addr = self.model.jnt_dofadr[joint_id]
        return self.data.qvel[qvel_addr:qvel_addr + 3].copy()

    def get_gripper_finger_qpos(self):
        """Return the left/right finger joint positions."""
        l_joint_id = self.model.joint("robot0:l_gripper_finger_joint").id
        r_joint_id = self.model.joint("robot0:r_gripper_finger_joint").id
        return np.array(
            [
                self.data.qpos[self.model.jnt_qposadr[l_joint_id]],
                self.data.qpos[self.model.jnt_qposadr[r_joint_id]],
            ],
            dtype=np.float64,
        )

    @property
    def target_body_name(self):
        return self._target_body_name

    @property
    def goal(self):
        return self._goal

    @property
    def home_grip_pos(self):
        assert self._home_grip_pos is not None
        return self._home_grip_pos.copy()

    @property
    def remaining_time_feature(self):
        remaining = 1.0 - (self._elapsed_steps / float(self._policy_horizon))
        return float(np.clip(remaining, 0.0, 1.0))

    @property
    def policy_horizon(self):
        return self._policy_horizon
