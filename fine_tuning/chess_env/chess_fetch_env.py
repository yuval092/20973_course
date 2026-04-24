import os
import numpy as np
import mujoco
import gymnasium_robotics.envs.fetch.pick_and_place as _fpp_module
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv

class ChessFetchEnv(MujocoFetchPickAndPlaceEnv):
    """
    Extends FetchPickAndPlace-v4 with a larger board.
    Goals and object spawns are sampled uniformly over the full board surface.
    No chess-rule constraints are applied.
    Observation and action spaces are IDENTICAL to the original.
    """

    # ── Board geometry ─────────────────────────────────────────
    # These must match the values in your modified XML exactly.
    TABLE_CENTER_XY = np.array([0.88, 0.2641])  # from <body pos="0.88 0.2641 0.2" name="table0">
    TABLE_HALF_X    = 0.32                    # from geom size
    TABLE_HALF_Y    = 0.32                    # from geom size
    TABLE_SURFACE_Z = 0.4                     # pos[2] + size[2] = 0.2 + 0.2
    EDGE_MARGIN     = 0.04                    # 4 cm safety margin

    # Minimum XY distance between the spawned object and its goal.
    MIN_GOAL_DIST = 0.10                      # 10 cm

    def __init__(self, **kwargs):
        # ── FIX: safe model_path injection ───────────────────────
        # Monkey-patch the module-level variable so the parent
        # constructor naturally picks up our modified XML.
        _original_path = _fpp_module.MODEL_XML_PATH
        # Use absolute path to avoid issues with current working directory
        _fpp_module.MODEL_XML_PATH = os.path.join(
            os.path.dirname(__file__), 'assets', 'pick_and_place.xml'
        )
        try:
            super().__init__(**kwargs)
            
            # --- FIX: Home Position (Phantom Table Bug) ---
            # The original FetchPickAndPlace initial_qpos is hardcoded
            # to point the robot at X=1.3, Y=0.75. 
            # We override it here to center the robot over our new board.
            
            # initial_qpos[0] = X-slide, [1] = Y-slide
            # Robot base link is at X=0.2869, Y=0.2641
            # To center it at our table (Y=0.2641), we set Y-slide to 0.0.
            # To bring it closer to the table (X=0.75), we set X-slide to 0.0 (Robot X=0.2869).
            self.initial_qpos[0] = 0.00
            self.initial_qpos[1] = 0.00
            # ----------------------------------------------
            
        finally:
            # Always restore the original path
            _fpp_module.MODEL_XML_PATH = _original_path

    # ── Internal helpers ────────────────────────────────────────
    def _sample_board_position(self):
        """Return a random XY position uniformly sampled on the board."""
        low_x  = self.TABLE_CENTER_XY[0] - self.TABLE_HALF_X + self.EDGE_MARGIN
        high_x = self.TABLE_CENTER_XY[0] + self.TABLE_HALF_X - self.EDGE_MARGIN
        low_y  = self.TABLE_CENTER_XY[1] - self.TABLE_HALF_Y + self.EDGE_MARGIN
        high_y = self.TABLE_CENTER_XY[1] + self.TABLE_HALF_Y - self.EDGE_MARGIN
        x = self.np_random.uniform(low_x, high_x)
        y = self.np_random.uniform(low_y, high_y)
        return np.array([x, y, self.TABLE_SURFACE_Z])

    # ── Override: goal sampling ──────────────────────────────────
    def _sample_goal(self):
        """
        Sample a goal position anywhere on the board surface.
        The goal is always on the table (never in the air).
        Ensures the goal is at least MIN_GOAL_DIST away from the object.
        """
        # Find the current object position from MuJoCo data
        obj_joint_id = self.model.joint("object0:joint").id
        qpos_start = self.model.jnt_qposadr[obj_joint_id]
        obj_pos = self.data.qpos[qpos_start : qpos_start + 2]  # XY only

        for _ in range(100):
            goal = self._sample_board_position()
            dist_xy = np.linalg.norm(goal[:2] - obj_pos)
            if dist_xy >= self.MIN_GOAL_DIST:
                return goal
        return goal  # fallback

    # ── Override: object reset ───────────────────────────────────
    def _reset_sim(self):
        """
        Reset the simulation and place the block at a random board position.
        """
        # Let the parent handle the arm and gripper reset.
        result = super()._reset_sim()

        # Sample a random position for the object.
        # We sample exactly once to ensure reset(seed) is deterministic.
        obj_pos = self._sample_board_position()

        # ── FIX: joint indexing ───────────────────────────────────
        # Use jnt_qposadr and jnt_dofadr for the true starting index.
        obj_joint_id = self.model.joint("object0:joint").id
        qpos_start   = self.model.jnt_qposadr[obj_joint_id]
        dof_start    = self.model.jnt_dofadr[obj_joint_id]

        self.data.qpos[qpos_start : qpos_start + 3] = obj_pos   # XYZ
        self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]  # quaternion (upright)
        self.data.qvel[dof_start  : dof_start  + 6] = 0.0       # zero velocity

        # ── FIX: call mj_forward ─────────────────────────────────
        # Ensure derived state is up-to-date after manual qpos write.
        mujoco.mj_forward(self.model, self.data)

        return result
