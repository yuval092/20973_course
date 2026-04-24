"""
Custom GoalEnv wrapper for the MuJoCo chess environment.

This module provides a bridge between the MuJoCo chess scene and the pretrained
FetchPickAndPlace reinforcement learning policy. It handles observation
reconstruction, action application, and coordinate mapping to ensure
compatibility with the Fetch robot's expected interface.
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
    GRIPPER_OPEN,
    L_FINGER_ACTUATOR, R_FINGER_ACTUATOR, GRIP_SITE,
    BOARD_CENTER, SQUARE_SIZE, WORKSPACE_XY_MARGIN,
    WORKSPACE_Z_MIN, WORKSPACE_Z_MAX, VIEWER_STEP_DELAY_SEC,
    FETCH_INIT_GRIP, FETCH_POLICY_HORIZON,
    ROBOT_SLIDE_X, ROBOT_SLIDE_Y, ROBOT_SLIDE_Z,
)
from src.env.obs_wrapper import ObservationReconstructor

logger = logging.getLogger(__name__)

# Damping constants for chess piece freejoints.
IDLE_PIECE_FREEJOINT_DAMPING = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 200.0], dtype=np.float64)
ACTIVE_PIECE_FREEJOINT_DAMPING = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float64)


class ChessPickPlaceEnv(gym.Env):
    """
    Wraps the MuJoCo chess scene for Fetch robot operation.

    This environment presents an interface compatible with pretrained Fetch
    policies, including specific observation vectors and action scaling.
    """

    def __init__(self, mj_model, mj_data, n_substeps=N_SUBSTEPS):
        """
        Initialize the pick-and-place environment.

        Args:
            mj_model: MuJoCo model.
            mj_data: MuJoCo data.
            n_substeps: Number of physics sub-steps per action step.
        """
        super().__init__()
        self.model = mj_model
        self.data = mj_data
        self.n_substeps = n_substeps

        # Actuators are the robot left and right fingers.
        self._left_finger_id = self.model.actuator(L_FINGER_ACTUATOR).id
        self._right_finger_id = self.model.actuator(R_FINGER_ACTUATOR).id

        self._target_body_name = None
        self._target_site_name = None
        self._goal = None
        self._elapsed_steps = 0
        self._policy_horizon = FETCH_POLICY_HORIZON
        self._robot_home_joint_state = None
        self._home_mocap_pos = None
        self._home_mocap_quat = None
        self._home_grip_pos = None

        # Cache piece information.
        self._piece_mesh_geom_ids_map  = self._build_piece_mesh_geom_id_map()
        self._piece_joint_dofadrs = self._build_piece_joint_dofadrs_map()
        self._piece_mesh_contact_masks = self._resolve_piece_mesh_contact_masks()
        self._set_all_pieces_idle()

        # Settle the scene.
        for _ in range(200):
            mujoco.mj_step(self.model, self.data, self.n_substeps)

        # Define workspace bounds.
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

        # Define Gym spaces.
        low = np.full(25, -np.inf, dtype=np.float32)
        high = np.full(25, np.inf, dtype=np.float32)

        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low, high, dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # Initialize robot pose.
        self.reset_robot_pose()

    def _capture_piece_joint_state(self):
        """
        Snapshot all chess-piece freejoint states.

        Returns:
            Dictionary mapping piece names to (qpos, qvel) tuples.
        """
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

    def _restore_piece_joint_state(self, piece_state):
        """
        Restore chess-piece freejoint states.

        Args:
            piece_state: Dictionary mapping piece names to (qpos, qvel) tuples.
        """
        for name, (qpos, qvel) in piece_state.items():
            joint_id = self.model.body(name).jntadr[0]
            qpos_addr = self.model.jnt_qposadr[joint_id]
            qvel_addr = self.model.jnt_dofadr[joint_id]
            self.data.qpos[qpos_addr:qpos_addr + 7] = qpos
            self.data.qvel[qvel_addr:qvel_addr + 6] = qvel

    def reset_robot_pose(self):
        """
        Reset the robot to its canonical Fetch hover pose.
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
            return

        # Initialize specific Fetch joints if no home state exists.
        for joint_name, value in (
            ("robot0:slide0", ROBOT_SLIDE_X),
            ("robot0:slide1", ROBOT_SLIDE_Y),
            ("robot0:slide2", ROBOT_SLIDE_Z),
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
        for _ in range(10):
            mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

        self._restore_piece_joint_state(piece_state)
        mujoco.mj_forward(self.model, self.data)

        # Snapshot the home state.
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

    def _build_piece_mesh_geom_id_map(self):
        """Scan every MujoCo body in the model and build a dict mapping each name (e.g 'w_pawn_a2') to its
        mesh geometry ID, essentialy building a table for which shape in the physics model belongs to each piece."""
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
                if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh_geom_ids[name] = geom_id
                    break
        return mesh_geom_ids

    def _build_piece_joint_dofadrs_map(self):
        """Map piece body names to their freejoint DOF addresses. which is the index into MujoCo's internal
        state arrays where you read and write the piece's position and velocity."""
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

    def _resolve_piece_mesh_contact_masks(self):
        """Map piece body names to their original (contype, conaffinity) collision masks."""
        return {
            name: (
                int(self.model.geom_contype[geom_id]),
                int(self.model.geom_conaffinity[geom_id]),
            )
            for name, geom_id in self._piece_mesh_geom_ids_map.items()
        }

    def _set_all_pieces_idle(self):
        """Disable collision and set high damping on every piece."""
        for piece_name in self._piece_mesh_geom_ids_map:
            self.set_piece_mesh_collision_enabled(piece_name, enabled=False)
            self.set_piece_active_damping(piece_name, active=False)

    def set_target(self, piece_name, goal_pos):
        """
        Configure the current manipulation target and goal.

        Args:
            piece_name: Name of the chess piece body.
            goal_pos: Target XYZ position.
        """
        self._target_body_name = piece_name
        self._target_site_name = f"{piece_name}_site"
        self._goal = np.asarray(goal_pos, dtype=np.float64).copy()
        self._elapsed_steps = 0
        for other_piece in self._piece_joint_dofadrs:
            self.set_piece_active_damping(other_piece, active=(other_piece == piece_name))
        self.set_piece_mesh_collision_enabled(piece_name, enabled=True)

    def set_piece_mesh_collision_enabled(self, piece_name, enabled):
        """Enable or disable collision proxies for a piece."""
        geom_id = self._piece_mesh_geom_ids_map.get(piece_name)
        if geom_id is None:
            return
        contype, conaffinity = self._piece_mesh_contact_masks[piece_name]
        self.model.geom_contype[geom_id] = contype if enabled else 0
        self.model.geom_conaffinity[geom_id] = conaffinity if enabled else 0

    def set_piece_active_damping(self, piece_name, active):
        """Switch a piece between active and idle damping states."""
        dofadr = self._piece_joint_dofadrs.get(piece_name)
        if dofadr is None:
            return
        damping = ACTIVE_PIECE_FREEJOINT_DAMPING if active else IDLE_PIECE_FREEJOINT_DAMPING
        self.model.dof_damping[dofadr:dofadr + 6] = damping

    def _expand_fetch_action(self, action):
        """Expand a 4D policy action into the canonical Fetch mocap action."""
        pos_ctrl = action[:3] * ACTION_SCALE
        rot_ctrl = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        gripper_ctrl = np.array([action[3], action[3]], dtype=np.float64)
        return np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

    def get_obs(self):
        """
        Build and return the observation dictionary.

        Returns:
            Dictionary with 'observation', 'achieved_goal', and 'desired_goal'.
        """
        assert self._target_body_name is not None, "Call set_target() first"
        assert self._goal is not None, "Call set_target() first"

        obs_vec = ObservationReconstructor.reconstruct_obs(
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

    def step(self, action, viewer=None, gripper_target=None):
        """
        Apply a 4D action to the simulation.

        Args:
            action: 4D numpy array [dx, dy, dz, gripper].
            viewer: Optional MuJoCo viewer.
            gripper_target: Optional deterministic gripper opening value.

        Returns:
            (obs, reward, terminated, info) tuple.
        """
        action = np.asarray(action, dtype=np.float64).copy()
        fetch_action = self._expand_fetch_action(action)

        # Apply action.
        mujoco_utils.ctrl_set_action(self.model, self.data, fetch_action)
        mujoco_utils.mocap_set_action(self.model, self.data, fetch_action)
        
        # Clip mocap to workspace.
        self.data.mocap_pos[0] = np.clip(self.data.mocap_pos[0], self._mocap_min, self._mocap_max)
        
        if gripper_target is not None:
            self.set_gripper_target(gripper_target)

        # Step physics.
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps += 1

        # Sync viewer.
        if viewer is not None:
            viewer.sync()
            if VIEWER_STEP_DELAY_SEC > 0.0:
                time.sleep(VIEWER_STEP_DELAY_SEC)

        # Build results.
        obs = self.get_obs()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])

        info = {
            'is_success': reward == 0.0,
            'piece_pos': obs['achieved_goal'].copy(),
            'goal_pos': obs['desired_goal'].copy(),
        }

        return obs, reward, False, info

    @staticmethod
    def compute_reward(achieved_goal, desired_goal):
        """
        Sparse reward calculation.

        Returns:
            0.0 if within tolerance, -1.0 otherwise.
        """
        d = np.linalg.norm(achieved_goal - desired_goal)
        return 0.0 if d < GOAL_TOLERANCE else -1.0

    def force_gripper_open(self):
        """Safety helper to force gripper open."""
        self.set_gripper_target(GRIPPER_OPEN)

    def set_gripper_target(self, opening):
        """
        Set finger joint targets.

        Args:
            opening: Target joint position in meters.
        """
        opening = float(np.clip(opening, 0.0, 0.05))
        self.data.ctrl[self._left_finger_id] = opening
        self.data.ctrl[self._right_finger_id] = opening

    def get_grip_pos(self):
        """Return current end-effector position."""
        return self.data.site(GRIP_SITE).xpos.copy()

    def get_piece_pos(self):
        """Return current target piece position."""
        assert self._target_body_name is not None
        return self.data.body(self._target_body_name).xpos.copy()

    def get_piece_linear_velocity(self):
        """Return target piece linear velocity."""
        assert self._target_body_name is not None
        joint_id = self.model.body(self._target_body_name).jntadr[0]
        qvel_addr = self.model.jnt_dofadr[joint_id]
        return self.data.qvel[qvel_addr:qvel_addr + 3].copy()

    def get_gripper_finger_qpos(self):
        """Return finger joint positions."""
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
        """Name of the current target piece."""
        return self._target_body_name

    @property
    def home_grip_pos(self):
        """Robot home position."""
        assert self._home_grip_pos is not None
        return self._home_grip_pos.copy()

    @property
    def remaining_time_feature(self):
        """Normalized remaining time for the current task."""
        remaining = 1.0 - (self._elapsed_steps / float(self._policy_horizon))
        return float(np.clip(remaining, 0.0, 1.0))
