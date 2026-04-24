import logging
import os

import mujoco
import numpy as np

from chess_env.chess_fetch_env import ChessFetchEnv

os.makedirs("logs/env_debug", exist_ok=True)


class ChessFetchDenseEnv(ChessFetchEnv):
    TABLE_Z = 0.400
    CUBE_Z = 0.410
    GRASP_Z = 0.410
    RELEASE_Z = 0.415
    SAFE_Z = 0.550

    SUCCESS_THRESHOLD = 0.010
    SOFT_LANDING_DIST = 0.050
    SOFT_LANDING_VEL = 0.010
    SUCCESS_BONUS = 10.0
    SOFT_LANDING_BONUS = 2.0
    TRANSIT_DISTANCE_SCALE = 0.1
    ELEVATOR_DISTANCE_SCALE = 0.1
    TRANSIT_ALTITUDE_ERROR_SCALE = 20.0
    TRANSIT_WARNING_SCALE = 40.0

    CURRICULUM_DRIFT_START = 0.080
    CURRICULUM_DRIFT_END = 0.035
    CURRICULUM_FLOOR_START = 0.450
    CURRICULUM_FLOOR_END = 0.470

    FLOOR_WARNING_Z = SAFE_Z - 0.03
    HIDDEN_OBJECT_POS = np.array([2.0, 2.0, TABLE_Z])
    OPEN_FINGER_POS = 0.05
    CLOSED_FINGER_POS = 0.008
    MAX_SETTLE_STEPS = 500
    SETTLE_TOLERANCE = 0.005
    SETTLE_GAIN = 0.3

    def __init__(
        self,
        force_scenario=None,
        total_curriculum_steps=300_000,
        num_envs=1,
        curriculum_progress_override=None,
        **kwargs,
    ):
        self.force_scenario = force_scenario
        # Each env sees 1/num_envs of total steps, so its local curriculum must scale accordingly
        self.total_curriculum_steps = max(int(total_curriculum_steps // num_envs), 1)
        self.curriculum_progress_override = curriculum_progress_override

        self.current_scenario = None
        self.tube_center_xy = None
        self.gripper_forced_state = None
        self.goal_pos = None
        self.episode_steps = 0
        self.total_env_steps = 0
        self.current_drift_limit = self.CURRICULUM_DRIFT_START
        self.current_floor_limit = self.CURRICULUM_FLOOR_START

        self.logger = logging.getLogger(f"env_logger_{os.getpid()}")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.FileHandler(f"logs/env_debug/env_{os.getpid()}.log")
            handler.setFormatter(logging.Formatter("%(asctime)s - [ENV] - %(message)s"))
            self.logger.addHandler(handler)

        super().__init__(**kwargs)

    def _curriculum_progress(self):
        if self.curriculum_progress_override is not None:
            return float(np.clip(self.curriculum_progress_override, 0.0, 1.0))
        return float(np.clip(self.total_env_steps / self.total_curriculum_steps, 0.0, 1.0))

    def _update_curriculum_limits(self):
        progress = self._curriculum_progress()
        self.current_drift_limit = self.CURRICULUM_DRIFT_START + (
            self.CURRICULUM_DRIFT_END - self.CURRICULUM_DRIFT_START
        ) * progress
        self.current_floor_limit = self.CURRICULUM_FLOOR_START + (
            self.CURRICULUM_FLOOR_END - self.CURRICULUM_FLOOR_START
        ) * progress

    def _apply_observation_mask(self, obs):
        if self.current_scenario != "transit":
            return obs

        # Mask object position and kinematics in Transit
        obs["observation"][3:9] = 0.0
        obs["observation"][11:20] = 0.0
        obs["achieved_goal"] = obs["observation"][0:3].copy()
        return obs

    def _build_phase9_observation(self):
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

        # Scenario ID: Transit=0.0, Descend=0.5, Ascend=1.0
        scenario_map = {"transit": 0.0, "descend": 0.5, "ascend": 1.0}
        scenario_id = scenario_map.get(self.current_scenario, -1.0)

        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
                [scenario_id], # Index 25
            ]
        )

        phase9_obs = {
            "observation": obs.copy(),
            "achieved_goal": grip_pos.copy(),
            "desired_goal": self.goal.copy(),
        }
        return self._apply_observation_mask(phase9_obs)

    def _get_obs(self):
        return self._build_phase9_observation()

    def _sample_goal(self):
        return self.goal_pos.copy()

    def _is_success(self, achieved_goal, desired_goal):
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)
        
        d_xy = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        d_z = abs(achieved_goal[2] - desired_goal[2])
        
        if self.current_scenario == "transit":
            # Loosen Z threshold for transit to 20mm due to oscillation at height
            return float(d_xy < self.SUCCESS_THRESHOLD and d_z < 0.020)
        else:
            # Standard 10mm accuracy for others
            d = np.linalg.norm(achieved_goal - desired_goal)
            return float(d < self.SUCCESS_THRESHOLD)

    def _set_gripper_state(self):
        finger_pos = self.OPEN_FINGER_POS
        if self.gripper_forced_state == "closed":
            finger_pos = self.CLOSED_FINGER_POS

        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint", finger_pos
        )
        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:r_gripper_finger_joint", finger_pos
        )

    def _settle_arm_to_start(self, arm_start_pos):
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

        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return last_error_norm

    def _reset_sim(self):
        super(ChessFetchEnv, self)._reset_sim()

        self.episode_steps = 0
        self._update_curriculum_limits()

        if self.force_scenario is not None:
            self.current_scenario = self.force_scenario
        else:
            self.current_scenario = self.np_random.choice(["transit", "descend", "ascend"])

        start_pos = self._sample_board_position()
        start_xy = start_pos[:2]

        if self.current_scenario == "transit":
            goal_pos = self._sample_board_position()
            while np.linalg.norm(goal_pos[:2] - start_xy) < self.MIN_GOAL_DIST:
                goal_pos = self._sample_board_position()

            goal_xy = goal_pos[:2]
            self.tube_center_xy = None
            arm_start_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
            self.goal_pos = np.array([goal_xy[0], goal_xy[1], self.SAFE_Z])
            object_pos = self.HIDDEN_OBJECT_POS.copy()
            self.gripper_forced_state = self.np_random.choice(["open", "closed"])
        elif self.current_scenario == "descend":
            self.tube_center_xy = start_xy.copy()
            arm_start_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
            self.goal_pos = np.array(
                [
                    start_xy[0],
                    start_xy[1],
                    self.np_random.choice([self.GRASP_Z, self.RELEASE_Z]),
                ]
            )
            object_pos = np.array([start_xy[0], start_xy[1], self.CUBE_Z])
            self.gripper_forced_state = "open"
        elif self.current_scenario == "ascend":
            self.tube_center_xy = start_xy.copy()
            # Start slightly higher (0.415 instead of 0.410) to avoid initial physics overlap with cube center
            arm_start_pos = np.array([start_xy[0], start_xy[1], 0.415])
            self.goal_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
            object_pos = np.array([start_xy[0], start_xy[1], self.CUBE_Z])
            self.gripper_forced_state = self.np_random.choice(["open", "closed"])
        else:
            raise ValueError(f"Unsupported scenario: {self.current_scenario}")

        self.logger.info(
            "--- EPISODE START | Scenario: %s | Start: %s | Goal: %s | DriftLimit: %.4f | FloorLimit: %.4f ---",
            self.current_scenario.upper(),
            arm_start_pos,
            self.goal_pos,
            self.current_drift_limit,
            self.current_floor_limit,
        )

        obj_joint_id = self.model.joint("object0:joint").id
        qpos_start = self.model.jnt_qposadr[obj_joint_id]
        dof_start = self.model.jnt_dofadr[obj_joint_id]

        self.data.qpos[qpos_start : qpos_start + 3] = self.HIDDEN_OBJECT_POS
        self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
        self.data.qvel[dof_start : dof_start + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        settle_error = self._settle_arm_to_start(arm_start_pos)
        self._set_gripper_state()

        self.data.qpos[qpos_start : qpos_start + 3] = object_pos
        self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
        self.data.qvel[dof_start : dof_start + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        if settle_error is not None and settle_error >= self.SETTLE_TOLERANCE:
            self.logger.warning(
                "Reset settle stopped with residual error %.4fm in scenario %s",
                settle_error,
                self.current_scenario,
            )

        return True

    def _ensure_info_batch(self, info, batch_size):
        if isinstance(info, dict):
            return [info] * batch_size

        if isinstance(info, np.ndarray):
            return [item if isinstance(item, dict) else {} for item in info.tolist()]

        if isinstance(info, (list, tuple)):
            return [item if isinstance(item, dict) else {} for item in info]

        return [{} for _ in range(batch_size)]

    def _build_phase9_info(self, dist_to_goal=None, grip_velocity=None):
        info = {
            "scenario": self.current_scenario,
            "tube_center_xy": None if self.tube_center_xy is None else self.tube_center_xy.copy(),
            "current_drift_limit": self.current_drift_limit,
            "current_floor_limit": self.current_floor_limit,
        }
        if dist_to_goal is not None:
            info["phase9_goal_distance"] = dist_to_goal
        if grip_velocity is not None:
            info["phase9_grip_velocity"] = grip_velocity
        return info

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)

        scalar_input = achieved_goal.ndim == 1
        if scalar_input:
            achieved_goal = achieved_goal[None, :]
            desired_goal = desired_goal[None, :]

        dist_to_goal = np.linalg.norm(achieved_goal - desired_goal, axis=1)
        reward = -0.1 * dist_to_goal
        
        # Scenario-aware success bonus
        for i in range(len(dist_to_goal)):
            if self._is_success(achieved_goal[i], desired_goal[i]):
                reward[i] += self.SUCCESS_BONUS

        info_batch = self._ensure_info_batch(info, achieved_goal.shape[0])
        for idx, meta in enumerate(info_batch):
            scenario = meta.get("scenario")
            if scenario == "transit":
                z = achieved_goal[idx, 2]
                reward[idx] -= 5.0 * max(0.0, self.SAFE_Z - z)
                if z < self.FLOOR_WARNING_Z:
                    reward[idx] -= 20.0 * (self.FLOOR_WARNING_Z - z)
                floor_limit = float(meta.get("current_floor_limit", self.current_floor_limit))
                if z < floor_limit:
                    reward[idx] -= 10.0
            elif scenario in {"descend", "ascend"}:
                tube_center_xy = meta.get("tube_center_xy")
                if tube_center_xy is None:
                    continue

                tube_center_xy = np.asarray(tube_center_xy, dtype=np.float64)
                drift = np.linalg.norm(achieved_goal[idx, :2] - tube_center_xy)
                reward[idx] -= 5.0 * max(0.0, drift - 0.005)
                drift_limit = float(meta.get("current_drift_limit", self.current_drift_limit))
                if drift > drift_limit:
                    reward[idx] -= 10.0

        if scalar_input:
            return float(reward[0])
        return reward

    def step(self, action):
        self.episode_steps += 1
        self.total_env_steps += 1
        self._update_curriculum_limits()

        action_copy = action.copy()
        if self.gripper_forced_state == "open":
            action_copy[3] = 1.0
        elif self.gripper_forced_state == "closed":
            action_copy[3] = -1.0

        action_copy = np.clip(action_copy, self.action_space.low, self.action_space.high)
        self._set_action(action_copy)
        self._mujoco_step(action_copy)
        self._step_callback()

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        terminated = False
        truncated = False

        gripper_pos = obs["observation"][0:3]
        grip_velocity = np.linalg.norm(obs["observation"][20:23])
        dist_to_goal = np.linalg.norm(gripper_pos - self.goal_pos)
        info = self._build_phase9_info(dist_to_goal=dist_to_goal, grip_velocity=grip_velocity)

        reward = self.compute_reward(gripper_pos, self.goal_pos, info)

        if dist_to_goal < self.SOFT_LANDING_DIST:
            # Penalize moving too fast when close to the goal to encourage a soft landing
            reward -= 5.0 * grip_velocity

        reward -= 0.01 * np.linalg.norm(action_copy[0:3]) ** 2
        info["is_success"] = float(self._is_success(gripper_pos, self.goal_pos))

        if self.current_scenario == "transit":
            if gripper_pos[2] < self.current_floor_limit:
                terminated = True
                info["is_success"] = 0.0
                reward -= 10.0
                self.logger.warning(
                    "TERMINATED (Transit): Floor collision. Z = %.4fm (Min %.3fm) at step %d",
                    gripper_pos[2],
                    self.current_floor_limit,
                    self.episode_steps,
                )
        elif self.current_scenario in {"descend", "ascend"}:
            drift = np.linalg.norm(gripper_pos[:2] - self.tube_center_xy)
            info["phase9_tube_drift"] = drift
            if drift > self.current_drift_limit:
                terminated = True
                info["is_success"] = 0.0
                reward -= 10.0
                self.logger.warning(
                    "TERMINATED (%s): Tube Drift = %.4fm (Max %.4fm) at step %d",
                    self.current_scenario.capitalize(),
                    drift,
                    self.current_drift_limit,
                    self.episode_steps,
                )

        if info["is_success"] == 1.0:
            terminated = True
            self.logger.info(
                "SUCCESS: Goal reached in %d steps. Dist = %.4fm",
                self.episode_steps,
                dist_to_goal,
            )
        elif terminated or truncated:
            self.logger.debug(
                "Episode ended without success at step %d. Final dist: %.4fm",
                self.episode_steps,
                dist_to_goal,
            )

        obs = self._apply_observation_mask(obs)
        return obs, reward, terminated, truncated, info
