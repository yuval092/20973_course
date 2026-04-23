"""
Observation wrapper for the MuJoCo chess environment.

This module provides functionality to reconstruct the 26D observation vector
expected by the pretrained reinforcement learning models used in this project.
The observation vector mirrors the Gymnasium-Robotics Fetch environment.
"""

import numpy as np
from gymnasium_robotics.utils import mujoco_utils
from gymnasium_robotics.utils import rotations

from src.config import GRIP_SITE


class ObservationReconstructor:
    """
    Handles the reconstruction of observation vectors from MuJoCo state.

    This class encapsulates the logic for mapping raw MuJoCo data into the
    format required by the RL policy.
    """

    @staticmethod
    def reconstruct_obs(model, data, target_body_name, target_site_name, n_substeps, time_feature):
        """
        Reconstruct the 26D observation vector.

        Args:
            model: MuJoCo model.
            data: MuJoCo data.
            target_body_name: Name of the chess piece body to observe.
            target_site_name: Name of the piece site to observe.
            n_substeps: Number of physics sub-steps per action step.
            time_feature: Normalized remaining-time feature in [0, 1].

        Returns:
            numpy array of shape (26,) with the observation vector.
        """
        dt = n_substeps * model.opt.timestep

        grip_pos = data.site(GRIP_SITE).xpos.copy()
        grip_velp = mujoco_utils.get_site_xvelp(model, data, GRIP_SITE) * dt

        robot_joint_names = tuple(
            model.joint(i).name for i in range(model.njnt) if model.joint(i).name
        )
        robot_qpos, robot_qvel = mujoco_utils.robot_get_obs(model, data, robot_joint_names)

        obj_pos = data.site(target_site_name).xpos.copy()
        obj_rel_pos = obj_pos - grip_pos
        gripper_state = robot_qpos[-2:].copy()
        obj_rot = rotations.mat2euler(data.site(target_site_name).xmat.reshape(3, 3))
        obj_velp = mujoco_utils.get_site_xvelp(model, data, target_site_name) * dt
        obj_velr = mujoco_utils.get_site_xvelr(model, data, target_site_name) * dt
        obj_velp -= grip_velp
        gripper_vel = robot_qvel[-2:] * dt

        obs = np.concatenate([
            grip_pos,
            obj_pos,
            obj_rel_pos,
            gripper_state,
            obj_rot,
            obj_velp,
            obj_velr,
            grip_velp,
            gripper_vel,
        ]).astype(np.float32)

        return obs
