"""
Observation wrapper that reconstructs the 26D observation vector
expected by the pretrained checkpoint used by this project.

The first 25 dimensions mirror Gymnasium-Robotics Fetch exactly:
    0-2:   Gripper position
    3-5:   Object position
    6-8:   Object-Gripper relative position
    9-10:  Gripper finger joint positions
    11-13: Object rotation (Euler XYZ)
    14-16: Object linear velocity relative to gripper, scaled by dt
    17-19: Object angular velocity, scaled by dt
    20-22: Gripper linear velocity, scaled by dt
    23-24: Gripper finger joint velocities, scaled by dt
    25:    Dummy time feature appended by the RL Zoo training setup
"""

import numpy as np
from gymnasium_robotics.utils import mujoco_utils
from gymnasium_robotics.utils import rotations

from src.config import GRIP_SITE


def reconstruct_obs(model, data, target_body_name, target_site_name, n_substeps, time_feature):
    """
    Reconstructs the 26D observation vector expected by the
    pretrained TQC checkpoint.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_body_name: Name of the chess piece body to observe
        target_site_name: Name of the piece site to observe, matching object0 in Fetch
        n_substeps: Number of physics sub-steps per action step (20)
        time_feature: Normalized remaining-time feature in [0, 1]

    Returns:
        numpy array of shape (26,) with the observation vector
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
        [float(np.clip(time_feature, 0.0, 1.0))],
    ]).astype(np.float32)

    return obs
