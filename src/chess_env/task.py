import logging
import os
import mujoco
import numpy as np
from src.chess_env.simulation import ChessSimulationEnv
from src.utils.config import load_config

class ChessTaskEnv(ChessSimulationEnv):
    """
    Config-driven Reinforcement Learning environment for RoboChess movement.
    
    This class implements the V6 "Pure Movement" logic, handling scenario-based
    task definitions (Transit, Descend, Ascend), reward shaping, terminal conditions,
    and a drift curriculum to stabilize training.
    
    It employs the "Holding Object" trick (Phase 9 architecture) to leverage
    pretrained transport weights without requiring piece-grasping physics.
    """

    def __init__(self, force_scenario=None, drift_curriculum_steps=None, force_drift_limit=None, hide_object=True, debug=False, fixed_drift=False, sample_debug_freq=None, **kwargs):
        """
        Initializes the task environment.

        Args:
            force_scenario (str): Lock the environment into 'transit', 'descend', or 'ascend'.
            drift_curriculum_steps (int): Total steps over which the drift limit tightens.
            force_drift_limit (float): Override the curriculum with a fixed radial drift limit.
            hide_object (bool): If True, teleports the piece to a hidden location (used for pure movement).
            debug (bool): Enables verbose per-step state logging.
            fixed_drift (bool): If True, bypasses the curriculum and locks the drift limit to its end value.
            sample_debug_freq (int): If set, enables debug mode every N episodes for one episode.
        """
        # Consume legacy params to avoid gym warnings
        kwargs.pop('total_curriculum_steps', None)
        kwargs.pop('num_envs', None)
        kwargs.pop('curriculum_progress_override', None)
        
        # Load configurations
        self.env_cfg = load_config("env")
        self.physics_cfg = load_config("physics")

        self.force_scenario = force_scenario
        self.force_drift_limit = force_drift_limit
        self.hide_object = hide_object
        self.base_debug = debug
        self.debug = debug
        self.fixed_drift = fixed_drift
        self.sample_debug_freq = sample_debug_freq or self.env_cfg.get("sample_debug_freq", None)
        
        # Initialization Debug
        if debug:
            print(f"[DEBUG INIT] Pid {os.getpid()} | Scenario: {force_scenario} | CurriculumSteps: {drift_curriculum_steps} | ForceDrift: {force_drift_limit} | Fixed: {fixed_drift}")
        
        self.current_scenario = None
        self.tube_center_xy = None
        self.gripper_forced_state = None
        self.goal_pos = None
        self.episode_steps = 0
        self.total_env_steps = 0
        self.episode_number = 0

        # Initialize base simulation
        super().__init__(debug=debug, **kwargs)

        # --- Task Constants ---
        self.CUBE_HEIGHT = self.env_cfg["cube_height"]
        self.CUBE_Z = self.env_cfg["cube_z"]
        self.GRASP_Z = self.env_cfg["grasp_z"]
        self.SAFE_Z = self.env_cfg["safe_z"]
        self.SUCCESS_THRESHOLD = self.env_cfg["success_threshold"]
        self.SUCCESS_BONUS = self.env_cfg["success_bonus"]
        self.DRIFT_LIMIT_START = self.env_cfg["drift_limit_start"]
        self.DRIFT_LIMIT_END = self.env_cfg["drift_limit_end"]
        self.DRIFT_CURRICULUM_STEPS = drift_curriculum_steps or self.env_cfg.get("drift_curriculum_steps", 500_000 // 15)
        self.FLOOR_LIMIT = self.env_cfg["floor_limit"]
        self.CRASH_PENALTY = self.env_cfg["crash_penalty"]
        self.HIDDEN_OBJECT_POS = np.array(self.env_cfg["hidden_object_pos"])
        self.MAX_SETTLE_STEPS = self.physics_cfg["max_settle_steps"]
        self.SETTLE_TOLERANCE = self.physics_cfg["settle_tolerance"]
        self.SETTLE_GAIN = self.physics_cfg["settle_gain"]

        # --- Reward & Precision Tuning ---
        self.STABILITY_VEL_THRESHOLD = self.env_cfg["stability_vel_threshold"]
        self.BRAKING_DIST = self.env_cfg["braking_dist"]
        self.FLOOR_PROXIMITY_THRESHOLD = self.env_cfg["floor_proximity_threshold"]
        self.Z_REWARD_WEIGHT = self.env_cfg["z_reward_weight"]
        self.XY_REWARD_WEIGHT = self.env_cfg.get("xy_reward_weight", 2.0)
        self.JITTER_PENALTY_WEIGHT = self.env_cfg["jitter_penalty_weight"]
        self.FLOOR_PENALTY = self.env_cfg["floor_penalty"]
        self.BRAKING_REWARD_WEIGHT = self.env_cfg["braking_reward_weight"]
        self.DIST_REWARD_WEIGHT = self.env_cfg["dist_reward_weight"]
        self.SETTLE_STEPS_FINAL = self.physics_cfg["settle_steps_final"]
        self.HALT_VEL_THRESHOLD = self.env_cfg.get("halt_vel_threshold", 0.0005)

        # --- Actuator Enforcement ---
        self.FINGER_OPEN_JOINT = self.env_cfg["finger_open_joint"]
        self.FINGER_CLOSED_JOINT = self.env_cfg["finger_closed_joint"]
        self.FINGER_OUTER_OFFSET = self.env_cfg["finger_outer_offset"]
        self.finger_target_joint = self.FINGER_CLOSED_JOINT

        # --- Grasp Stage State ---
        self.grasp_mode = False
        
        # Home Position
        home_xy = self.env_cfg.get("home_position_xy", [0.680, 0.2641])
        self.HOME_POS = np.array([home_xy[0], home_xy[1], self.SAFE_Z])

        # Grasp Thresholds
        self.GRASP_CONTACT_APPROACH_TOLERANCE = self.env_cfg.get("grasp_contact_approach_tolerance", 0.001)
        self.GRASP_CLOSE_STEPS = self.env_cfg.get("grasp_close_steps", 150)
        self.GRASP_HOLD_STEPS = self.env_cfg.get("grasp_hold_steps", 50)
        self.GRASP_VERIFY_XY_THRESHOLD = self.env_cfg.get("grasp_verify_xy_threshold", 0.015)
        self.GRASP_VERIFY_Z_THRESHOLD = self.env_cfg.get("grasp_verify_z_threshold", 0.020)
        self.GRASP_VERIFY_FINGER_THRESHOLD = self.env_cfg.get("grasp_verify_finger_threshold", 0.012)
        self.CUBE_HELD_XY_LIMIT = self.env_cfg.get("cube_held_xy_limit", 0.030)
        self.CUBE_HELD_Z_LIMIT = self.env_cfg.get("cube_held_z_limit", 0.020)

        # Overrides for evaluation
        self.force_start_pos = None
        self.force_cube_pos = None

        # --- Logger Setup ---
        log_cfg = self.env_cfg.get("logging", {})
        log_level = logging.DEBUG if debug else getattr(logging, log_cfg.get("level", "INFO"))
        log_dir = log_cfg.get("log_dir", "logs/env_debug")
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger(f"chess_task_{os.getpid()}")
        self.logger.setLevel(logging.DEBUG) # Always set to DEBUG, let handler filter if needed
        if not self.logger.handlers:
            handler = logging.FileHandler(os.path.join(log_dir, f"env_{os.getpid()}.log"))
            handler.setFormatter(logging.Formatter("%(asctime)s - [TASK] - %(message)s"))
            handler.setLevel(logging.DEBUG)
            self.logger.addHandler(handler)
        
        # Initial level based on sampled debug
        if self.sample_debug_freq is not None:
            self.logger.setLevel(logging.DEBUG if (self.episode_number % self.sample_debug_freq == 0) else logging.INFO)

    def _build_phase9_observation(self):
        """
        Constructs a 25-dimensional observation vector.
        
        Implements the 'Holding Object' trick: by mapping object_pos to grip_pos,
        the model is convinced it is always carrying a piece, forcing it to
        use its stable transport (Phase 2) logic.
        
        Vector Mapping:
            0-2:   Gripper Position
            3-5:   Object Position (Mapped to Gripper Position)
            6-8:   Object-to-Goal Relative Position
            9-10:  Gripper Finger State (Masked/Zero)
            11-13: Scenario ID (One-Hot: Transit, Descend, Ascend)
            14-19: Object Velocity (Masked/Zero)
            20-22: Gripper Velocity
            23-24: Gripper Finger Velocity (Masked/Zero)
        """
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()

        scenario_map = {
            "transit": [1.0, 0.0, 0.0],
            "descend": [0.0, 1.0, 0.0],
            "ascend": [0.0, 0.0, 1.0]
        }
        scenario_id_vec = scenario_map.get(self.current_scenario, [0.0, 0.0, 0.0])
        
        # Apply the architectural trick
        target_pos_for_model = grip_pos.copy()

        if hasattr(self, 'goal') and self.goal is not None and self.goal.shape == (3,):
            relative_dist_for_model = self.goal.copy() - grip_pos
        else:
            relative_dist_for_model = np.zeros(3)

        obs = np.concatenate(
            [
                grip_pos,                 # 0-2
                target_pos_for_model,     # 3-5
                relative_dist_for_model,  # 6-8
                np.zeros(2),              # 9-10
                scenario_id_vec,          # 11-13
                np.zeros(3),              # 14-16
                np.zeros(3),              # 17-19
                grip_velp,                # 20-22
                np.zeros(2),              # 23-24
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": grip_pos.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _get_obs(self):
        """Returns the current observation dictionary."""
        return self._build_phase9_observation()

    def get_cube_position(self) -> np.ndarray:
        """Returns the current 3D world position of the cube."""
        obj_joint_id = self.model.joint("object0:joint").id
        qpos_start = self.model.jnt_qposadr[obj_joint_id]
        return self.data.qpos[qpos_start : qpos_start + 3].copy()

    def get_cube_quat(self) -> np.ndarray:
        """Returns the quaternion orientation of the cube (w, x, y, z)."""
        obj_joint_id = self.model.joint("object0:joint").id
        qpos_start = self.model.jnt_qposadr[obj_joint_id]
        return self.data.qpos[qpos_start + 3 : qpos_start + 7].copy()

    def _sample_goal(self):
        """Returns the goal sampled during reset_sim."""
        return self.goal_pos.copy()

    def _is_success(self, achieved_goal, desired_goal):
        """
        Determines if the gripper is within the success threshold of the goal.
        Checks both 2D (XY) and height (Z) precision.
        """
        achieved_goal = np.asarray(achieved_goal)
        desired_goal  = np.asarray(desired_goal)
        d_xy = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        d_z = abs(achieved_goal[2] - desired_goal[2])

        return float(d_xy < self.SUCCESS_THRESHOLD and d_z < self.SUCCESS_THRESHOLD)

    def _set_gripper_state(self):
        """Physically sets the joint positions of the fingers based on the target state."""
        self._utils.set_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint", self.finger_target_joint)
        self._utils.set_joint_qpos(self.model, self.data, "robot0:r_gripper_finger_joint", self.finger_target_joint)

    def _check_cube_held(self, grip_pos: np.ndarray) -> tuple[bool, str | None]:
        """
        Checks that the cube is still within the gripper.
        Only called when grasp_mode=True and current_scenario is 'ascend' or 'transit'.
        """
        cube_pos = self.get_cube_position()
        xy_error = np.linalg.norm(cube_pos[:2] - grip_pos[:2])
        # Cube CoM is ~15mm below grip site when held at center; allow tolerance
        z_error = abs(cube_pos[2] - (grip_pos[2] - 0.015))
        
        if xy_error > self.CUBE_HELD_XY_LIMIT:
            return False, f"CUBE_DROPPED_XY (err={xy_error*1000:.1f}mm > limit={self.CUBE_HELD_XY_LIMIT*1000:.0f}mm)"
        if z_error > self.CUBE_HELD_Z_LIMIT:
            return False, f"CUBE_DROPPED_Z (err={z_error*1000:.1f}mm > limit={self.CUBE_HELD_Z_LIMIT*1000:.0f}mm)"
        return True, None

    def _settle_arm_to_start(self, arm_start_pos):
        """
        Uses a gain-based settle loop to move the arm to the starting mocap position.
        Prevents physics 'explosions' caused by sudden joints teleportation.
        """
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        mocap_offset = self.data.mocap_pos[0].copy() - grip_pos
        self.data.mocap_pos[0] = arm_start_pos + mocap_offset
        mujoco.mj_forward(self.model, self.data)

        last_error_norm = None
        for _ in range(self.MAX_SETTLE_STEPS):
            grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            error = arm_start_pos - grip_pos
            last_error_norm = np.linalg.norm(error)
            if last_error_norm < self.SETTLE_TOLERANCE:
                break

            self.data.mocap_pos[0] += error * self.SETTLE_GAIN
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

        # Zero velocities to ensure a stable episode start
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
    def _reset_sim(self):
        """
        Executes a complex reset sequence with scripted gripper transitions:
        1. Samples start/goal positions.
        2. Teleports object to a hidden or board location.
        3. Settles the arm at the starting waypoint with fingers CLOSED (stability).
        4. Executes scenario-specific scripted gripper transitions (Open/Close).
        5. Validates final finger state before handing over to RL model.
        """
        self.episode_steps = 0
        self.episode_number += 1

        # Periodic Sampled Debugging
        if self.sample_debug_freq is not None:
            self.debug = self.base_debug or (self.episode_number % self.sample_debug_freq == 0)
            if self.debug:
                self.logger.setLevel(logging.DEBUG)
            else:
                self.logger.setLevel(logging.INFO)

        if self.force_scenario is not None:
            self.current_scenario = self.force_scenario
        else:
            self.current_scenario = self.np_random.choice(["transit", "descend", "ascend"])

        # NEW: reset grasp mode on every reset
        self.grasp_mode = False

        start_pos = self._sample_board_position()
        start_xy = start_pos[:2]

        # Define scenario-specific start/goal configurations
        if self.current_scenario == "transit":
            goal_pos = self._sample_board_position()
            while np.linalg.norm(goal_pos[:2] - start_xy) < self.MIN_GOAL_DIST:
                goal_pos = self._sample_board_position()
            self.tube_center_xy = None
            
            # Support force_start_pos (Home Position override)
            if self.force_start_pos is not None:
                arm_start_pos = self.force_start_pos.copy()
            else:
                arm_start_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
                
            self.goal_pos = np.array([goal_pos[0], goal_pos[1], self.SAFE_Z])
        elif self.current_scenario == "descend":
            self.tube_center_xy = start_xy.copy()
            arm_start_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
            self.goal_pos = np.array([start_xy[0], start_xy[1], self.GRASP_Z])
        elif self.current_scenario == "ascend":
            self.tube_center_xy = start_xy.copy()
            arm_start_pos = np.array([start_xy[0], start_xy[1], self.GRASP_Z])
            self.goal_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])

        # Reset parent simulation
        super(ChessSimulationEnv, self)._reset_sim()

        # Place object
        obj_joint_id = self.model.joint("object0:joint").id
        qpos_start = self.model.jnt_qposadr[obj_joint_id]
        dof_start = self.model.jnt_dofadr[obj_joint_id]
        
        if self.hide_object:
            self.data.qpos[qpos_start : qpos_start + 3] = self.HIDDEN_OBJECT_POS
            self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
            self.data.qvel[dof_start : dof_start + 6] = 0.0
        elif self.force_cube_pos is not None:
            # Support force_cube_pos override
            self.data.qpos[qpos_start : qpos_start + 3] = self.force_cube_pos[:3]
            self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
            self.data.qvel[dof_start : dof_start + 6] = 0.0
        else:
            self.data.qpos[qpos_start : qpos_start + 2] = start_xy
            self.data.qpos[qpos_start + 2] = self.TABLE_Z + (self.CUBE_HEIGHT / 2.0)
            self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
            self.data.qvel[dof_start : dof_start + 6] = 0.0
            
        mujoco.mj_forward(self.model, self.data)
        
        # --- PHASE 1: Settle arm CLOSED for stability ---
        self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT)
        self.finger_target_joint = self.FINGER_CLOSED_JOINT
        self._set_gripper_state()
        self._settle_arm_to_start(arm_start_pos)
        
        # --- PHASE 2: Scripted Transitions ---
        # Dummy action for _set_action calls during transitions
        dummy_action = np.zeros(4)

        if self.current_scenario == "descend":
            # Descend: closed --> opened (scripted transition before RL)
            self.finger_target_joint = self.FINGER_OPEN_JOINT
            for _ in range(200):
                self._set_action(dummy_action)
                self._mujoco_step(None)
        elif self.current_scenario == "ascend":
            # Ascend: start opened --> closed (scripted transition before RL)
            # First open them
            self.finger_target_joint = self.FINGER_OPEN_JOINT
            self._set_gripper_state() # Force immediately
            for _ in range(100): # Let physics settle in open state
                self._set_action(dummy_action)
                self._mujoco_step(None)
            
            # Then close them scripted
            self.finger_target_joint = self.FINGER_CLOSED_JOINT
            for _ in range(200):
                self._set_action(dummy_action)
                self._mujoco_step(None)
        else:
            # Transit: stay closed
            self.finger_target_joint = self.FINGER_CLOSED_JOINT
            for _ in range(200):
                self._set_action(dummy_action)
                self._mujoco_step(None)

        # --- PHASE 3: Final Validation ---
        l_pos = self._utils.get_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint").item()
        if abs(l_pos - self.finger_target_joint) > 0.0005:
            self.logger.error(f"Reset Failed: Finger joint at {l_pos:.6f}, target {self.finger_target_joint:.6f} (Scenario: {self.current_scenario})")
            return False 
        
        if self.debug:
            obj_pos_now = self.data.qpos[qpos_start : qpos_start + 3]
            grip_pos_now = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            arm_joints = ["robot0:shoulder_pan_joint", "robot0:shoulder_lift_joint",
                          "robot0:upperarm_roll_joint", "robot0:elbow_flex_joint",
                          "robot0:forearm_roll_joint", "robot0:wrist_flex_joint",
                          "robot0:wrist_roll_joint"]
            joint_qpos = []
            for jname in arm_joints:
                try:
                    joint_qpos.append(self._utils.get_joint_qpos(self.model, self.data, jname).item())
                except Exception:
                    joint_qpos.append(float('nan'))
            jq_str = ", ".join([f"{x:.4f}" for x in joint_qpos])
            tube_str = f"{self.tube_center_xy}" if self.tube_center_xy is not None else "N/A"
            msg = (
                f"\n{'='*80}\n"
                f"[EPISODE {self.episode_number} START] Scenario: {self.current_scenario.upper()}\n"
                f"  GripPos:    [{', '.join([f'{x:.4f}' for x in grip_pos_now])}]\n"
                f"  Goal:       [{', '.join([f'{x:.4f}' for x in self.goal_pos])}]\n"
                f"  TubeCenter: {tube_str}\n"
                f"  ObjPos:     [{', '.join([f'{x:.4f}' for x in obj_pos_now])}]\n"
                f"  ArmJoints:  [{jq_str}]\n"
                f"{'='*80}"
            )
            print(msg)
            self.logger.debug(msg)

        return True

    def transition_validate(self, nominal_exit_pos: np.ndarray | None = None) -> dict:
        """
        Returns arm state diagnostics at a scenario transition point.
        """
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip").copy()
        speed = float(np.linalg.norm(grip_vel))
        l_finger = self._utils.get_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint").item()

        result = {
            "grip_pos": grip_pos.tolist(),
            "grip_speed_mm_s": speed * 1000,
            "is_velocity_ok": speed < self.HALT_VEL_THRESHOLD,
            "error_from_nominal_mm": None,
            "finger_state": l_finger,
        }
        if nominal_exit_pos is not None:
            result["error_from_nominal_mm"] = float(np.linalg.norm(grip_pos - nominal_exit_pos) * 1000)
        return result
    def execute_grasp(self) -> dict:
        """
        Scripted GRASP phase. Executes immediately after DESCEND success.
        Returns a plain dict (NOT a dataclass). All call sites use result["success"].
        
        Phases:
          0: Zero velocity (IMMEDIATELY) + short settle (15 steps)
          0.5: Pre-conditions + cube sanity + cube rotation check
          1: Contact Approach (fine-descent to GRASP_Z, 1mm/step)
          2: Finger Close (actuator-driven, 150 steps, with early-abort)
          3: Hold Settle (50 steps)
          4: Final Verification
        """
        ROBOT_DOF = 15
        dummy_action = np.zeros(4)
        result = {
            "success": False,
            "reason": None,
            "pre_grasp_cube_xy": None,
            "post_grasp_cube_pos": None,
            "post_grasp_cube_quat": None,
            "final_xy_error_mm": 0.0,
            "final_z_error_mm": 0.0,
            "final_finger_pos": 0.0,
            "close_steps_used": 0,
        }
        
        # ── Phase 0: ZERO VELOCITY FIRST, then short settle ──────────────────
        # CRITICAL: Zero qvel/qacc BEFORE running any steps. The arm arrives with
        # residual velocity from the RL policy. Running 50 simulation steps BEFORE
        # zeroing velocity causes the arm to oscillate near GRASP_Z, triggering
        # proximity contact explosions that launch the cube (~5% of full pipeline).
        # By zeroing first, the arm is stationary from step 1 of settle.
        self.data.qvel[:ROBOT_DOF] = 0.0
        self.data.qacc[:ROBOT_DOF] = 0.0
        mujoco.mj_forward(self.model, self.data)
        
        zero_action = np.zeros(4)
        for _ in range(15):  # Short settle (arm already stationary — just equilibrate weld)
            self._set_action(zero_action)
            self._mujoco_step(None)
        
        # ── Precondition Checks ───────────────────────────────────────────────
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        speed = float(np.linalg.norm(grip_vel))
        if speed > 0.005:  # > 5mm/s after halt — something is wrong
            result["reason"] = f"PRECONDITION_SPEED ({speed*1000:.2f}mm/s after halt)"
            return result
        
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        if abs(grip_pos[2] - self.GRASP_Z) > 0.010:
            result["reason"] = f"PRECONDITION_Z (grip at {grip_pos[2]*1000:.1f}mm, GRASP_Z={self.GRASP_Z*1000:.1f}mm)"
            return result
        
        cube_pos = self.get_cube_position()
        result["pre_grasp_cube_xy"] = cube_pos[:2].copy()
        xy_to_cube = np.linalg.norm(cube_pos[:2] - grip_pos[:2])
        if xy_to_cube > 0.010:
            result["reason"] = f"PRECONDITION_XY (cube is {xy_to_cube*1000:.1f}mm from grip site)"
            return result
        
        l_finger = self._utils.get_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint").item()
        if l_finger < self.FINGER_OPEN_JOINT - 0.002:
            result["reason"] = f"PRECONDITION_FINGERS_NOT_OPEN (finger={l_finger:.4f})"
            return result
        
        # ── Sub-Phase 1: Contact Approach (Fine-Descent) ──────────────────────
        # CRITICAL: DO NOT SKIP THIS PHASE. DESCEND success threshold is 10mm, so
        # the arm can succeed at GRASP_Z + 9mm = 0.434m. Without fine-descent, the
        # cube overlap is only 48% instead of 78%, reducing pipeline grasp success
        # from ~84% to ~78%. This phase is the single most impactful step.
        #
        # CRITICAL ORDER: _set_action calls mocap_set_action → reset_mocap2body_xpos,
        # which resets mocap_pos to the current body position (erasing any direct
        # modification). step_vec MUST be applied AFTER _set_action or it will be
        # silently erased before the physics step.
        # Correct order: (1) _set_action(zero_action), (2) mocap_pos += step_vec, (3) _mujoco_step()
        target_z = self.GRASP_Z
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        if abs(grip_pos[2] - target_z) > self.GRASP_CONTACT_APPROACH_TOLERANCE:
            fine_target = np.array([grip_pos[0], grip_pos[1], target_z])
            for _ in range(50):
                grip_pos_now = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
                error = fine_target - grip_pos_now
                if np.linalg.norm(error) < self.GRASP_CONTACT_APPROACH_TOLERANCE:
                    break
                step_vec = 0.5 * error
                if np.linalg.norm(step_vec) > 0.003:
                    step_vec = step_vec / np.linalg.norm(step_vec) * 0.003  # 3mm/step max
                # CRITICAL ORDER: _set_action calls mocap_set_action → reset_mocap2body_xpos,
                # which resets mocap_pos to the current *body* position (erasing any direct
                # modification). step_vec MUST be applied AFTER _set_action or it will be
                # silently erased before the physics step, causing the arm to never move.
                self._set_action(zero_action)           # 1. Reset mocap_pos → body pos
                self.data.mocap_pos[0][:3] += step_vec  # 2. Add step_vec on top
                self._mujoco_step(None)                 # 3. Physics sees correct target
            else:
                grip_pos_now = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
                if abs(grip_pos_now[2] - target_z) > 0.005:
                    result["reason"] = f"CONTACT_APPROACH_FAILED (z_err={abs(grip_pos_now[2]-target_z)*1000:.1f}mm)"
                    return result
        
        # ── Sub-Phase 2: Finger Close (Actuator-Driven) ────────────────────────
        self.grasp_mode = True
        self.finger_target_joint = self.FINGER_CLOSED_JOINT
        close_steps = 0
        
        for step in range(self.GRASP_CLOSE_STEPS):
            self._set_action(zero_action)
            self._mujoco_step(None)
            close_steps += 1
            
            if step % 10 == 0:
                cube_now = self.get_cube_position()
                if cube_now[2] > self.GRASP_Z + 0.050:
                    result["reason"] = f"CUBE_EXPLOSION_DURING_CLOSE (cube_z={cube_now[2]*1000:.1f}mm)"
                    return result
            
            # Early-abort FINGER_CLOSED_EMPTY: if fingers nearly fully closed after >30 steps
            # with no cube resistance, the cube is not in the finger path → abort immediately.
            # Without early abort, the full 150 steps elapse uselessly.
            if step > 30:
                l_finger_now = self._utils.get_joint_qpos(
                    self.model, self.data, "robot0:l_gripper_finger_joint"
                ).item()
                if l_finger_now < 0.003:  # Fingers closed fully — no cube contact
                    result["reason"] = f"FINGER_CLOSED_EMPTY_EARLY (finger={l_finger_now:.4f} after {step} steps)"
                    result["close_steps_used"] = step + 1
                    return result
            
            # NOTE: fingers WITH cube stall at j≈0.0143, never reach 0.012 or below.
        
        result["close_steps_used"] = close_steps
        
        # ── Sub-Phase 3: Grasp Verification ───────────────────────────────────
        cube_pos = self.get_cube_position()
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        l_finger = self._utils.get_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint").item()
        
        xy_error = float(np.linalg.norm(cube_pos[:2] - grip_pos[:2])) * 1000
        # Z check: cube is on table during grasp (CoM ≈ 0.415m), grip site at GRASP_Z=0.425.
        # Gap is ~10mm. GRASP_VERIFY_Z_THRESHOLD=20mm covers this. Do NOT use (grip_pos[2]-0.015)
        # here — the 15mm hang only occurs when cube is in mid-air, not resting on table.
        z_error  = float(abs(cube_pos[2] - grip_pos[2])) * 1000
        
        result["final_xy_error_mm"] = xy_error
        result["final_z_error_mm"] = z_error
        result["final_finger_pos"] = l_finger
        
        if xy_error > self.GRASP_VERIFY_XY_THRESHOLD * 1000:
            result["reason"] = f"VERIFY_XY_FAILED ({xy_error:.1f}mm > {self.GRASP_VERIFY_XY_THRESHOLD*1000:.0f}mm)"
            return result
        if z_error > self.GRASP_VERIFY_Z_THRESHOLD * 1000:
            result["reason"] = f"VERIFY_Z_FAILED ({z_error:.1f}mm > {self.GRASP_VERIFY_Z_THRESHOLD*1000:.0f}mm)"
            return result
        if l_finger > self.GRASP_VERIFY_FINGER_THRESHOLD:
            result["reason"] = f"VERIFY_FINGERS_FAILED (finger={l_finger:.4f} > threshold={self.GRASP_VERIFY_FINGER_THRESHOLD})"
            return result
        
        # ── Sub-Phase 4: Hold Settle ───────────────────────────────────────────
        for _ in range(self.GRASP_HOLD_STEPS):
            self._set_action(zero_action)
            self._mujoco_step(None)
        
        cube_pos_final = self.get_cube_position()
        grip_pos_final = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        final_xy = float(np.linalg.norm(cube_pos_final[:2] - grip_pos_final[:2])) * 1000
        
        if final_xy > self.GRASP_VERIFY_XY_THRESHOLD * 1000:
            result["reason"] = f"CUBE_DRIFTED_AFTER_HOLD ({final_xy:.1f}mm)"
            return result
        
        result["success"] = True
        result["post_grasp_cube_pos"] = cube_pos_final.tolist()
        result["post_grasp_cube_quat"] = self.get_cube_quat().tolist()
        return result

    def soft_reset(self, new_scenario: str, new_goal_pos: np.ndarray,
                   nominal_exit_pos: np.ndarray,
                   nominal_xy: np.ndarray | None = None) -> tuple[dict, dict]:
        """
        Transitions to a new scenario without teleporting the arm.
        Executes: complete halt -> waypoint alignment -> gripper transition -> state update.

        Note: The caller is responsible for resetting environment wrappers (like TimeLimit).
        
        Returns:
            (obs, info): obs is the new observation, info contains transition diagnostics.
        """
        # ROBOT_DOF = 15 is correct for Fetch: 3 slides + 1 torso + 2 head + 7 arm + 2 fingers
        ROBOT_DOF = 15
        HALT_HOLD_MAX_STEPS = 100
        ALIGN_TOLERANCE_M = 0.003
        ALIGN_MAX_STEPS = 200
        ALIGN_GAIN = 0.8
        ALIGN_MAX_STEP_M = 0.005

        # Phase 1: Halt (Driving arm to a dead stop)
        zero_action = np.zeros(4)
        halt_steps = 0
        for _ in range(HALT_HOLD_MAX_STEPS):
            grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
            if np.linalg.norm(grip_vel) < self.HALT_VEL_THRESHOLD:
                break
            self._set_action(zero_action)
            self._mujoco_step(None)
            halt_steps += 1

        # Active velocity zeroing (Robot only, preserves object physics)
        self.data.qvel[:ROBOT_DOF] = 0.0
        self.data.qacc[:ROBOT_DOF] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Confirm halt (Plan Step 1c)
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        speed = float(np.linalg.norm(grip_vel))
        if speed >= self.HALT_VEL_THRESHOLD:
            raise RuntimeError(
                f"soft_reset HALT_FAILED: speed={speed*1000:.3f}mm/s >= "
                f"threshold={self.HALT_VEL_THRESHOLD*1000:.1f}mm/s"
            )

        # Phase 2: Waypoint Alignment (Physics-simulated smooth movement)
        converged = False
        align_steps = 0
        for loop_step in range(ALIGN_MAX_STEPS):
            grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            error = nominal_exit_pos - grip_pos
            dist = np.linalg.norm(error)
            if dist < ALIGN_TOLERANCE_M:
                self.data.qvel[:ROBOT_DOF] = 0.0
                self.data.qacc[:ROBOT_DOF] = 0.0
                mujoco.mj_forward(self.model, self.data)
                converged = True
                break
            # Proportional step toward target
            step_vec = ALIGN_GAIN * error
            if np.linalg.norm(step_vec) > ALIGN_MAX_STEP_M:
                step_vec = step_vec / np.linalg.norm(step_vec) * ALIGN_MAX_STEP_M
            self.data.mocap_pos[0][:3] += step_vec
            self._mujoco_step(None)
            align_steps = loop_step + 1

        if not converged:
            raise RuntimeError(f"soft_reset ALIGN_FAILED: did not converge to {nominal_exit_pos}")

        # Phase 3: State Update
        prev_scenario = self.current_scenario
        self.current_scenario = new_scenario
        self.goal_pos = new_goal_pos.copy()
        self.goal = self.goal_pos.copy() # Critical for observation desync
        self.episode_steps = 0
        
        if new_scenario in {"descend", "ascend"}:
            if nominal_xy is None:
                raise ValueError(f"nominal_xy required for {new_scenario}")
            self.tube_center_xy = nominal_xy.copy()
        else:
            self.tube_center_xy = None

        # Re-enforce vertical orientation
        self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT)

        # Phase 4: Gripper Transition (Scripted finger movement)
        if not self.grasp_mode:
            needs_open = (new_scenario == "descend" and prev_scenario != "descend")
            needs_close = (new_scenario in {"ascend", "transit"} and prev_scenario == "descend")
            dummy_action = np.zeros(4)

            if needs_open:
                self.finger_target_joint = self.FINGER_OPEN_JOINT
                for _ in range(50):
                    self._set_action(dummy_action)
                    self._mujoco_step(None)
            elif needs_close:
                self.finger_target_joint = self.FINGER_CLOSED_JOINT
                for _ in range(50):
                    self._set_action(dummy_action)
                    self._mujoco_step(None)

            # Finger validation (same as _reset_sim Phase 3)
            l_pos = self._utils.get_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint"
            ).item()
            if abs(l_pos - self.finger_target_joint) > 0.0005:
                raise RuntimeError(
                    f"soft_reset FINGER_VALIDATION_FAILED: "
                    f"actual={l_pos:.6f}, target={self.finger_target_joint:.6f}"
                )
        # else: grasp_mode=True -> fingers stay actuator-driven, no teleportation or validation needed

        info = {
            "halt_steps": halt_steps,
            "align_steps": align_steps,
        }
        return self._get_obs(), info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dense reward shaping:
        1. Distance: Negative L2 norm to goal.
        2. Z-Error: Penalty for height inaccuracy.
        3. Braking: Penalty for high velocity when near the goal.
        """
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)
        scalar_input = achieved_goal.ndim == 1
        if scalar_input:
            achieved_goal, desired_goal = achieved_goal[None, :], desired_goal[None, :]

        dist_to_goal = np.linalg.norm(achieved_goal - desired_goal, axis=1)
        reward = -self.DIST_REWARD_WEIGHT * dist_to_goal
        
        z_err = np.abs(achieved_goal[:, 2] - desired_goal[:, 2])
        reward -= self.Z_REWARD_WEIGHT * z_err

        xy_err = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        reward -= self.XY_REWARD_WEIGHT * xy_err

        grip_velp = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        g_speed = np.linalg.norm(grip_velp)
        
        braking_mask = (dist_to_goal < self.BRAKING_DIST).astype(np.float32)
        reward -= self.BRAKING_REWARD_WEIGHT * g_speed * braking_mask

        return float(reward[0]) if scalar_input else reward

    def step(self, action):
        """
        Executes a physics step:
        1. Forces gripper target based on scenario state machine.
        2. Enforces absolute finger position via Simulation-level actuator override.
        3. Checks for tube breaches (including footprint) and finger faults.
        """
        self.total_env_steps += 1
        self.episode_steps += 1
        
        # 1. State Machine Enforcement
        if self.current_scenario == "descend" and not self.grasp_mode:
            self.finger_target_joint = self.FINGER_OPEN_JOINT
        elif self.current_scenario != "descend":
            self.finger_target_joint = self.FINGER_CLOSED_JOINT
            
        action_copy = action.copy()
        action_copy[3] = -1.0 # Standardize; simulation.py will override with finger_target_joint
        
        action_copy = np.clip(action_copy, self.action_space.low, self.action_space.high)
        self._set_action(action_copy)
        self._mujoco_step(action_copy)
        self._step_callback()

        obs = self._get_obs()
        gripper_pos = obs["observation"][0:3]
        gripper_vel = obs["observation"][20:23]
        
        terminated, truncated = False, False
        crashed = False
        crash_reason = None
        
        # 2. Finger Fault Check (Production Certification)
        l_finger = self._utils.get_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint").item()
        finger_fault = False
        if not self.grasp_mode:
            finger_fault = abs(l_finger - self.finger_target_joint) > 0.003
            if finger_fault:
                crashed = True
                crash_reason = f"FINGER_FAULT (actual={l_finger:.4f}, target={self.finger_target_joint:.4f})"
        
        # NEW: Cube drop check (only during grasp_mode in ascend/transit)
        if self.grasp_mode and self.current_scenario in {"ascend", "transit"}:
            cube_held, drop_reason = self._check_cube_held(gripper_pos)
            if not cube_held:
                crashed = True
                crash_reason = drop_reason

        # 3. Collision & Drift Detection
        current_drift_limit = self.DRIFT_LIMIT_END
        if not self.fixed_drift and self.force_drift_limit is None:
            dp = min(self.total_env_steps / self.DRIFT_CURRICULUM_STEPS, 1.0)
            current_drift_limit = self.DRIFT_LIMIT_START - (self.DRIFT_LIMIT_START - self.DRIFT_LIMIT_END) * dp
        elif self.force_drift_limit is not None:
            current_drift_limit = self.force_drift_limit

        if self.current_scenario == "transit":
            if gripper_pos[2] < self.FLOOR_LIMIT:
                crashed = True
                crash_reason = f"FLOOR_HIT (z={gripper_pos[2]:.4f} < limit={self.FLOOR_LIMIT:.4f})"
        elif self.current_scenario in {"descend", "ascend"}:
            drift = np.linalg.norm(gripper_pos[:2] - self.tube_center_xy)
            drift_outer = drift + self.FINGER_OUTER_OFFSET
            
            # The curriculum drift limit now governs the center.
            # Footprint check is implicitly governed by curriculum.
            if drift > current_drift_limit:
                crashed = True
                crash_reason = f"TUBE_BREACH (center={drift:.4f} > limit={current_drift_limit:.4f})"
            elif gripper_pos[2] < self.TABLE_SURFACE_Z:
                crashed = True
                crash_reason = f"TABLE_HIT (z={gripper_pos[2]:.4f} < surface={self.TABLE_SURFACE_Z:.4f})"

        if self.debug:
            info_log = { "fault": finger_fault, "limit": f"{current_drift_limit:.4f}" }
            msg = (f"[DEBUG TASK] Ep{self.episode_number} Step {self.episode_steps:03d} | "
                   f"Grip: [{', '.join([f'{x:.4f}' for x in gripper_pos])}] | "
                   f"Target: [{', '.join([f'{x:.4f}' for x in self.goal_pos])}] | "
                   f"Fingers: {l_finger:.4f} | Info: {info_log}")
            self.logger.debug(msg)
        
        reward = self.compute_reward(gripper_pos, self.goal_pos, {})
        reward -= self.JITTER_PENALTY_WEIGHT * np.linalg.norm(action_copy[:3]) ** 2 
        
        if gripper_pos[2] < self.FLOOR_LIMIT + self.FLOOR_PROXIMITY_THRESHOLD:
            reward += self.FLOOR_PENALTY

        is_near = self._is_success(gripper_pos, self.goal_pos)
        is_stable = np.linalg.norm(gripper_vel) < self.STABILITY_VEL_THRESHOLD
        success = float(is_near and is_stable)
        
        if crashed:
            terminated = True
            reward = self.CRASH_PENALTY
            success = 0.0

        if success:
            terminated = True
            reward += self.SUCCESS_BONUS

        if self.debug and terminated:
            outcome = f"CRASH ({crash_reason})" if crashed else ("SUCCESS" if success else "TIMEOUT")
            msg = (f"\n[EPISODE {self.episode_number} END] Scenario={self.current_scenario} | "
                   f"Outcome={outcome} | Reward={reward:.2f}\n")
            self.logger.debug(msg)

        info = {
            "is_success": float(success),
            "scenario": self.current_scenario,
            "crash_reason": crash_reason
        }
        return obs, reward, terminated, truncated, info
