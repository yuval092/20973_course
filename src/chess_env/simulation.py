import os
import numpy as np
import mujoco
import gymnasium_robotics.envs.fetch.pick_and_place as _fpp_module
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv
from src.utils.config import load_config

class ChessSimulationEnv(MujocoFetchPickAndPlaceEnv):
    """
    Modular MuJoCo environment for RoboChess fine-tuning.
    
    This class extends the base FetchPickAndPlace-v4 environment to support
    a larger 64cm x 64cm chess board and configuration-driven physics constants.
    It handles low-level simulation tasks like model loading, action scaling,
    and board sampling.
    """

    def __init__(self, debug=False, **kwargs):
        """
        Initializes the simulation environment.

        Args:
            debug (bool): If True, enables verbose logging of gripper positions and control setpoints.
            **kwargs: Additional arguments passed to the parent MujocoFetchPickAndPlaceEnv.
        """
        # Load configurations from YAML files
        self.env_cfg = load_config("env")
        self.physics_cfg = load_config("physics")
        self.debug = debug

        # --- Table Geometry ---
        # We use a larger table (64cm x 64cm) than the standard Fetch task.
        self.TABLE_CENTER_XY = np.array(self.env_cfg["table_center_xy"])
        self.TABLE_HALF_X    = self.env_cfg["table_half_x"]
        self.TABLE_HALF_Y    = self.env_cfg["table_half_y"]
        self.TABLE_SURFACE_Z = self.env_cfg["table_surface_z"]
        self.TABLE_Z         = self.TABLE_SURFACE_Z # Alias for backward compatibility
        self.EDGE_MARGIN     = self.env_cfg["edge_margin"]
  # Safety margin to avoid edge collisions
        self.MIN_GOAL_DIST   = self.env_cfg["min_goal_dist"] # Min distance between start and goal
        self.CUBE_HEIGHT     = self.env_cfg.get("cube_height", 0.030) # Height of the chess piece (cube)
        self.MAX_GRIPPER_WIDTH = self.env_cfg.get("max_gripper_width", 0.05)

        # --- Physics Parameters ---
        # Number of physics steps to run before an episode starts to let the arm settle
        self.ENV_SETUP_STEPS = self.physics_cfg.get("env_setup_steps", 10)
        # Max attempts to find a valid board position for sampling goals
        self.MAX_GOAL_RETRIES = self.physics_cfg.get("max_goal_retries", 100)
        # Scaling factor for the relative mocap actions (limits movement speed)
        self.POS_CTRL_SCALE = self.physics_cfg.get("pos_ctrl_scale", 0.015)

        # Force the gripper to point straight down (verticality constraint)
        raw_quat = np.array(self.physics_cfg["vertical_quat"])
        self.VERTICAL_QUAT = raw_quat / np.linalg.norm(raw_quat)

        # --- Model Injection ---
        # We override the model XML path before calling super().__init__ to use our custom assets.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        asset_path = os.path.join(project_root, 'chess_env', 'assets', 'pick_and_place.xml')

        _original_path = _fpp_module.MODEL_XML_PATH
        _fpp_module.MODEL_XML_PATH = asset_path
        try:
            super().__init__(**kwargs)
            # Adjust the robot's default base position to center it on the larger board
            init_qpos = self.physics_cfg.get("initial_qpos", [-0.05, 0.00])
            self.initial_qpos[0] = init_qpos[0]
            self.initial_qpos[1] = init_qpos[1]
        finally:
            _fpp_module.MODEL_XML_PATH = _original_path

    def _sample_board_position(self):
        """
        Samples a random (X, Y) coordinate within the board boundaries.
        Returns coordinates at the table surface altitude.
        """
        low_x  = self.TABLE_CENTER_XY[0] - self.TABLE_HALF_X + self.EDGE_MARGIN
        high_x = self.TABLE_CENTER_XY[0] + self.TABLE_HALF_X - self.EDGE_MARGIN
        low_y  = self.TABLE_CENTER_XY[1] - self.TABLE_HALF_Y + self.EDGE_MARGIN
        high_y = self.TABLE_CENTER_XY[1] + self.TABLE_HALF_Y - self.EDGE_MARGIN
        x = self.np_random.uniform(low_x, high_x)
        y = self.np_random.uniform(low_y, high_y)
        return np.array([x, y, self.TABLE_SURFACE_Z])

    def _sample_goal(self):
        """
        Samples a goal position that is at least MIN_GOAL_DIST away from the object.
        Used by the parent class during reset.
        """
        obj_joint_id = self.model.joint("object0:joint").id
        qpos_start = self.model.jnt_qposadr[obj_joint_id]
        obj_pos = self.data.qpos[qpos_start : qpos_start + 2]
        for _ in range(self.MAX_GOAL_RETRIES):
            goal = self._sample_board_position()
            dist_xy = np.linalg.norm(goal[:2] - obj_pos)
            if dist_xy >= self.MIN_GOAL_DIST:
                return goal
        return goal

    def _reset_sim(self):
        """
        Resets the simulation state. Samples a new object position on the board.
        """
        result = super()._reset_sim()
        obj_pos = self._sample_board_position()
        obj_pos[2] += self.CUBE_HEIGHT / 2.0  # Adjust for cube's center-of-mass
        
        obj_joint_id = self.model.joint("object0:joint").id
        qpos_start   = self.model.jnt_qposadr[obj_joint_id]
        dof_start    = self.model.jnt_dofadr[obj_joint_id]
        
        # Place cube and zero its velocity
        self.data.qpos[qpos_start : qpos_start + 3] = obj_pos
        self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
        self.data.qvel[dof_start  : dof_start  + 6] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        return result

    def _set_action(self, action):
        """
        Converts the RL model output [dx, dy, dz, gripper] into MuJoCo mocap control.
        
        The position actions are scaled to ensure smooth movement, and the orientation
        is locked to point vertically downward.
        
        Finger positions are absolutely enforced at the simulation level based on
        finger_target_joint.
        """
        assert action.shape == (4,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        
        # 1. Scaled relative movement (limits max displacement per step)
        pos_ctrl *= self.POS_CTRL_SCALE

        # 2. ZERO DELTA ROTATION (Maintains vertical orientation)
        rot_ctrl = np.zeros(4)

        # 3. Finger Enforcement
        # Retrieve target joint position (to be set by Task class)
        target_qpos = getattr(self, "finger_target_joint", 0.0)
        
        if getattr(self, 'grasp_mode', False):
            # Actuator-driven mode: set ctrl target only.
            # Let MuJoCo's Kp controller drive the fingers with contact physics enabled.
            # The cube can push back against the fingers; contact forces are computed.
            self.data.ctrl[0] = target_qpos
            self.data.ctrl[1] = target_qpos
            # IMPORTANT: do NOT call set_joint_qpos or set_joint_qvel here.
        else:
            # Teleport mode: absolute enforcement (no cube contact during movement phases).
            self.data.ctrl[0] = target_qpos
            self.data.ctrl[1] = target_qpos
            
            # Force the joint position directly
            self._utils.set_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint", target_qpos)
            self._utils.set_joint_qpos(self.model, self.data, "robot0:r_gripper_finger_joint", target_qpos)
            
            # Zero joint velocities for fingers
            self._utils.set_joint_qvel(self.model, self.data, "robot0:l_gripper_finger_joint", 0.0)
            self._utils.set_joint_qvel(self.model, self.data, "robot0:r_gripper_finger_joint", 0.0)
        
        # Apply mocap position update
        mocap_action = np.concatenate([pos_ctrl, rot_ctrl])
        self._utils.mocap_set_action(self.model, self.data, mocap_action)

        # Enforce crane mode (vertical orientation) every step
        self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT)

    def _env_setup(self, initial_qpos):
        """
        Initial setup of the environment. Forces vertical orientation and lets 
        physics settle before the first episode starts.
        """
        super()._env_setup(initial_qpos)
        # Force crane orientation immediately on setup
        self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT)
        for _ in range(self.ENV_SETUP_STEPS):
            mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
