"""
Custom GoalEnv wrapper for the MuJoCo chess environment.

This module provides a bridge between the MuJoCo chess scene and the pretrained
FetchPickAndPlace reinforcement learning policy. It handles observation
reconstruction, action application, and coordinate mapping to ensure
compatibility with the Fetch robot's expected interface.
"""

import logging
import time
from typing import NamedTuple
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
    ROBOT_SLIDE_X, ROBOT_SLIDE_Y, ROBOT_SLIDE_Z
)
from src.env.obs_wrapper import ObservationReconstructor

logger = logging.getLogger(__name__)

HIGH_MOVEMENT_RESISTANCE = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 200.0], dtype=np.float64)
LOW_MOVEMENT_RESISTANCE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float64)
GRIPPER_DOWNWARD_ANGLE = (1.0, 0.0, 1.0, 0.0)

class GripperAction(NamedTuple):
    dx: float
    dy: float
    dz: float
    finger_command: float


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
        self._home_gripper_target_pos = None
        self._home_gripper_target_angle = None
        self._home_grip_pos = None

        # Cache piece information.
        self._piece_3d_model_id_map = self._build_piece_3d_model_id_map()
        self._piece_movement_resistance_indices = self._build_piece_movement_resistance_indices_map()
        self._piece_collision_filters_map = self._build_piece_collision_filters_map()

        self._freeze_all_pieces()
        self._advance_simulation(steps=200)

        self._mocap_min, self._mocap_max = self._calculate_board_boundaries()
        self.observation_space, self.action_space = self._build_robot_io_spaces()

        self.reset_robot_pose()

    def _capture_piece_joint_state(self):
        """
        Snapshot the position and velocity of every chess piece.

        Returns:
            Dictionary mapping piece names to (position, velocity) tuples.
        """
        piece_state = {}
        for body_idx in range(self.model.nbody):
            body = self.model.body(body_idx)
            name = body.name
            if not self._is_piece_name(name):
                continue
            joint_id = body.jntadr[0]
            if joint_id < 0:
                continue
            position_index = self.model.jnt_qposadr[joint_id]
            velocity_index = self.model.jnt_dofadr[joint_id]
            piece_state[name] = (
                self.data.qpos[position_index:position_index + 7].copy(),
                self.data.qvel[velocity_index:velocity_index + 6].copy(),
            )
        return piece_state

    def _restore_piece_joint_state(self, piece_state):
        """
        Restore the position and velocity of every chess piece to a previous snapshot.

        Args:
            piece_state: Dictionary mapping piece names to (position, velocity) tuples,
                         as returned by _capture_piece_joint_state().
        """
        for name, (qpos, qvel) in piece_state.items():
            joint_id = self.model.body(name).jntadr[0]
            position_index = self.model.jnt_qposadr[joint_id]
            velocity_index = self.model.jnt_dofadr[joint_id]
            self.data.qpos[position_index:position_index + 7] = qpos
            self.data.qvel[velocity_index:velocity_index + 6] = qvel

    def reset_robot_pose(self):
        """
        Reset the robot to its canonical Fetch hover pose.
        """
        piece_state = self._capture_piece_joint_state()
        if self._robot_home_joint_state is None:
            self._initialize_home_pose()
        else:
            self._restore_home_pose()
        self._restore_piece_joint_state(piece_state)
        mujoco.mj_forward(self.model, self.data)

    def _initialize_home_pose(self):
        self._move_robot_to_home_position()
        self._snapshot_home_pose()

    def _move_robot_to_home_position(self):
        self._set_arm_base_position()
        self._set_gripper_home_target()

    def _set_joint_position(self, joint_name, value):
        joint_id = self.model.joint(joint_name).id
        position_index = self.model.jnt_qposadr[joint_id]
        self.data.qpos[position_index] = value

    def _set_joint_velocity(self, joint_name, value):
        joint_id = self.model.joint(joint_name).id
        velocity_index = self.model.jnt_dofadr[joint_id]
        self.data.qvel[velocity_index] = value

    def _get_joint_position(self, joint_name):
        joint_id = self.model.joint(joint_name).id
        position_index = self.model.jnt_qposadr[joint_id]
        return float(self.data.qpos[position_index])

    def _get_joint_velocity(self, joint_name):
        joint_id = self.model.joint(joint_name).id
        velocity_index = self.model.jnt_dofadr[joint_id]
        return float(self.data.qvel[velocity_index])

    def _set_arm_base_position(self):
        self._set_joint_position("robot0:slide0", ROBOT_SLIDE_X)
        self._set_joint_position("robot0:slide1", ROBOT_SLIDE_Y)
        self._set_joint_position("robot0:slide2", ROBOT_SLIDE_Z)
        # After teleporting the arm base, resync the invisible gripper target so it
        # doesn't try to pull the arm back to where it was before.
        mujoco_utils.reset_mocap_welds(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _set_gripper_home_target(self):
        self.data.mocap_pos[0] = np.asarray(FETCH_INIT_GRIP, dtype=np.float64)
        self.data.mocap_quat[0] = np.array(GRIPPER_DOWNWARD_ANGLE, dtype=np.float64)
        self._advance_simulation(steps=10)

    def _snapshot_home_pose(self):
        self._robot_home_joint_state = []
        for joint_idx in range(self.model.njnt):
            joint_name = self.model.joint(joint_idx).name
            if not joint_name or not joint_name.startswith("robot0:"):
                continue  # Skip non-robot joints such as pieces.
            self._robot_home_joint_state.append((
                joint_name,
                self._get_joint_position(joint_name),
                self._get_joint_velocity(joint_name),
            ))
        self._home_gripper_target_pos = self.data.mocap_pos.copy() # commanded target, not necessarily where the gripper lands.
        self._home_gripper_target_angle = self.data.mocap_quat.copy()
        self._home_grip_pos = self.get_grip_pos() # actual physical position after physics, may differ from target.


    def _restore_home_pose(self):
        """Teleport all robot joints back to their home position and cancel any active motor commands."""
        for joint_name, joint_qpos, joint_qvel in self._robot_home_joint_state:
            self._set_joint_position(joint_name, joint_qpos)
            self._set_joint_velocity(joint_name, joint_qvel)
        self.data.mocap_pos[:] = self._home_gripper_target_pos
        self.data.mocap_quat[:] = self._home_gripper_target_angle
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0  # Cancel any leftover motor command (e.g. old "close gripper").

    def _is_piece_name(self, name):
        return bool(name) and name.startswith(("w_", "b_"))
    
    def _is_active_piece_name(self, name):
        return self._is_piece_name(name) and "spare" not in name

    def _find_body_shape_id(self, body, shape_type):
        shape_count = int(np.asarray(body.geomnum).item())
        first_shape_idx = int(np.asarray(body.geomadr).item())
        for offset in range(shape_count):
            shape_id = first_shape_idx + offset
            if self.model.geom_type[shape_id] == shape_type:
                return shape_id
        return None

    def _build_piece_3d_model_id_map(self):
        """Build a lookup table so we can enable/disable piece-to-piece collisions per piece
        without scanning the model every time. Maps each piece name to its 3D model shape ID."""
        piece_3d_model_ids = {}
        for body_idx in range(self.model.nbody):
            body = self.model.body(body_idx)
            if not self._is_active_piece_name(body.name):
                continue
            shape_id = self._find_body_shape_id(body, mujoco.mjtGeom.mjGEOM_MESH)
            if shape_id is not None:
                piece_3d_model_ids[body.name] = shape_id
        return piece_3d_model_ids

    def _build_piece_movement_resistance_indices_map(self):
        """Build a lookup table so we can freeze or unfreeze individual pieces without scanning the model every time.
        Maps each piece name to its index in MuJoCo's damping array, which controls how much it resists movement."""
        resistance_indices = {}
        for body_idx in range(self.model.nbody):
            body = self.model.body(body_idx)
            if not self._is_active_piece_name(body.name):
                continue
            joint_id = int(np.asarray(body.jntadr).item())
            if joint_id < 0:
                continue
            resistance_indices[body.name] = int(self.model.jnt_dofadr[joint_id])
        return resistance_indices

    def _build_piece_collision_filters_map(self):
        """Save each piece's original collision on/off settings so we can restore them
        when re enabling piece to piece collision after it was temporarily disabled."""
        return {
            name: (
                int(self.model.geom_contype[geom_id]),
                int(self.model.geom_conaffinity[geom_id]),
            )
            for name, geom_id in self._piece_3d_model_id_map.items()
        }

    def _build_robot_io_spaces(self):
        low = np.full(25, -np.inf, dtype=np.float32)
        high = np.full(25, np.inf, dtype=np.float32)
        observation_space = spaces.Dict({
            'observation': spaces.Box(low, high, dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        })
        action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        return observation_space, action_space

    def _calculate_board_boundaries(self):
        board_half_extent = 4 * SQUARE_SIZE
        mocap_min = np.array([
            BOARD_CENTER[0] - board_half_extent - WORKSPACE_XY_MARGIN,
            BOARD_CENTER[1] - board_half_extent - WORKSPACE_XY_MARGIN,
            WORKSPACE_Z_MIN,
        ], dtype=np.float64)
        mocap_max = np.array([
            BOARD_CENTER[0] + board_half_extent + WORKSPACE_XY_MARGIN,
            BOARD_CENTER[1] + board_half_extent + WORKSPACE_XY_MARGIN,
            WORKSPACE_Z_MAX,
        ], dtype=np.float64)
        return mocap_min, mocap_max

    def _advance_simulation(self, steps: int):
        """Run the physics engine for the given number of steps without applying any new action."""
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data, self.n_substeps)

    def _freeze_all_pieces(self):
        """Disable collision and lock every piece in place so they don't drift during setup."""
        for piece_name in self._piece_3d_model_id_map:
            self.set_piece_collision_enabled(piece_name, enabled=False)
            self.set_piece_movement_resistance(piece_name, low=False)

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
        for other_piece in self._piece_movement_resistance_indices:
            self.set_piece_movement_resistance(other_piece, low=(other_piece == piece_name))
        self.set_piece_collision_enabled(piece_name, enabled=True)

    def set_piece_collision_enabled(self, piece_name, enabled):
        """Turn on or off whether this piece's 3D shape can physically collide with other pieces."""
        geom_id = self._piece_3d_model_id_map.get(piece_name)
        if geom_id is None:
            return
        collision_group, collision_targets = self._piece_collision_filters_map[piece_name]
        self.model.geom_contype[geom_id] = collision_group if enabled else 0
        self.model.geom_conaffinity[geom_id] = collision_targets if enabled else 0

    def set_piece_movement_resistance(self, piece_name, low):
        """Control whether a piece can move freely or is locked in place.
        low=True means the piece can be pushed around; low=False locks it so physics won't knock it over."""
        resistance_index = self._piece_movement_resistance_indices.get(piece_name)
        if resistance_index is None:
            return
        resistance = LOW_MOVEMENT_RESISTANCE if low else HIGH_MOVEMENT_RESISTANCE
        self.model.dof_damping[resistance_index:resistance_index + 6] = resistance

    def _format_action_for_gripper_controller(self, action:GripperAction):
        """Convert the policy's 4D action [dx, dy, dz, finger_command] into the 9-value
        format MuJoCo's gripper controller expects: 3 movement + 4 orientation + 2 fingers.
        Orientation is always locked to straight down."""
        position_delta = np.array([action.dx, action.dy, action.dz]) * ACTION_SCALE
        gripper_downward_angle = np.array(GRIPPER_DOWNWARD_ANGLE, dtype=np.float64)
        finger_commands = np.array([action.finger_command, action.finger_command], dtype=np.float64)
        return np.concatenate([position_delta, gripper_downward_angle, finger_commands])

    def get_obs(self):
        """
        Build what the RL policy needs to see each step.

        Requires set_target() to have been called first to specify which piece
        is being moved (target_body_name) and the attachment point on that piece
        the gripper aims for (target_site_name).

        Returns a dict with:
          - 'observation': 25-float sensor vector (joint states, gripper pos, goal, time left).
          - 'achieved_goal': where the piece currently is (XYZ).
          - 'desired_goal': where we want the piece to end up (XYZ).
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

    def step(self, action: GripperAction, viewer=None, gripper_target=None):
        """
        Apply one gripper action to the simulation.

        Args:
            action: GripperAction with dx, dy, dz movement and finger_command.
            viewer: Optional MuJoCo viewer.
            gripper_target: Optional deterministic gripper opening value.

        Returns:
            (obs, reward, terminated, info) tuple.
        """
        gripper_command = self._format_action_for_gripper_controller(action)
        self._apply_gripper_command(gripper_command)

        # Clamp anchor to board boundaries so the robot won't fly off the edge.
        self.data.mocap_pos[0] = np.clip(self.data.mocap_pos[0], self._mocap_min, self._mocap_max)

        if gripper_target is not None:
            self.set_gripper_target(gripper_target)

        # Run physics — each call is one full substep so the control signal is reapplied every iteration.
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps += 1
        self._sync_viewer(viewer)
        return self._build_step_result()

    def _apply_gripper_command(self, gripper_command):
        mujoco_utils.ctrl_set_action(self.model, self.data, gripper_command)
        mujoco_utils.mocap_set_action(self.model, self.data, gripper_command)

    def _sync_viewer(self, viewer: mujoco.viewer.Handle | None):
        """Push the latest simulation state to the viewer window and optionally
        slow down playback so a human can follow what the robot is doing."""
        if viewer is not None:
            viewer.sync()
            if VIEWER_STEP_DELAY_SEC > 0.0:
                time.sleep(VIEWER_STEP_DELAY_SEC)

    def _build_step_result(self):
        """Return the standard Gym step tuple (obs, reward, terminated, info).
        terminated is always False — this env never ends on its own, the caller decides when to stop."""
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
        diff = np.linalg.norm(achieved_goal - desired_goal)
        return 0.0 if diff < GOAL_TOLERANCE else -1.0

    def force_gripper_open(self):
        """Safety helper to force gripper open."""
        self.set_gripper_target(GRIPPER_OPEN)

    def set_gripper_target(self, width: float):
        """
        Set how wide the fingers should open, in meters.
        0.0 = fully closed (gripping), 0.05 = fully open.
        """
        width = float(np.clip(width, 0.0, 0.05))
        self.data.ctrl[self._left_finger_id] = width
        self.data.ctrl[self._right_finger_id] = width

    def get_grip_pos(self):
        """Return current end effector position."""
        return self.data.site(GRIP_SITE).xpos.copy()

    def get_piece_pos(self):
        """Return current target piece position."""
        assert self._target_body_name is not None
        return self.data.body(self._target_body_name).xpos.copy()

    def get_piece_linear_velocity(self):
        """Return target piece linear velocity."""
        assert self._target_body_name is not None
        joint_id = self.model.body(self._target_body_name).jntadr[0]
        velocity_index = self.model.jnt_dofadr[joint_id]
        return self.data.qvel[velocity_index:velocity_index + 3].copy()

    def get_finger_joint_positions(self):
        """Return how open each finger is in meters — [left_finger, right_finger]."""
        return np.array([
            self._get_joint_position("robot0:l_gripper_finger_joint"),
            self._get_joint_position("robot0:r_gripper_finger_joint"),
        ], dtype=np.float64)

    @property
    def target_body_name(self):
        """Name of the current target piece."""
        return self._target_body_name

    @property
    def home_grip_pos(self):
        """Physical position of the grip site when the arm is at its home hover pose."""
        assert self._home_grip_pos is not None
        return self._home_grip_pos.copy()

    @property
    def remaining_time_feature(self):
        """Normalized remaining time for the current task."""
        remaining = 1.0 - (self._elapsed_steps / float(self._policy_horizon))
        return float(np.clip(remaining, 0.0, 1.0))
