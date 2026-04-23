"""
Unit tests for the observation wrapper.

This module verifies that the observation reconstruction logic correctly
extracts and scales features from the Mujoco simulation state into a 
format suitable for the RL policy.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.env.obs_wrapper import ObservationReconstructor


class TestObsWrapper:
    """
    Test suite for observation processing and reconstruction.
    """

    def test_reconstruct_obs_shape(self):
        """
        Verify that reconstruct_obs returns a vector of the correct shape and values.
        """
        model = MagicMock()
        model.opt.timestep = 0.002
        
        # Mock joint indices
        model.jnt_qposadr = MagicMock()
        model.jnt_qposadr.__getitem__.side_effect = lambda x: x
        model.jnt_dofadr = MagicMock()
        model.jnt_dofadr.__getitem__.side_effect = lambda x: x
        
        data = MagicMock()
        
        # Mock data.site('robot0:grip').xpos
        mock_grip_site = MagicMock()
        mock_grip_site.xpos = np.array([1.0, 2.0, 3.0])
        
        # Mock object site
        mock_obj_site = MagicMock()
        mock_obj_site.xpos = np.array([4.0, 5.0, 6.0])
        mock_obj_site.xmat = np.eye(3).reshape(-1)
        mock_obj_site.id = 1
        
        # Site side effect to handle both grip and target site
        def site_side_effect(name):
            if name == 'robot0:grip':
                return mock_grip_site
            return mock_obj_site
        data.site.side_effect = site_side_effect
        
        # Mock data.body(target_body_name).xpos, xquat, cvel
        mock_obj_body = MagicMock()
        mock_obj_body.xpos = np.array([4.0, 5.0, 6.0])
        mock_obj_body.xmat = np.eye(3).reshape(-1)
        mock_obj_body.xquat = np.array([1.0, 0.0, 0.0, 0.0])
        # cvel: 0-2 angular, 3-5 linear
        mock_obj_body.cvel = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        def body_side_effect(name):
            if name == 'target':
                return mock_obj_body
            return None
        data.body.side_effect = body_side_effect
        
        data.qpos = np.array([0.01, 0.02])
        data.qvel = np.array([0.11, 0.22])
        
        mock_joint = MagicMock()
        mock_joint.qpos = np.array([0.01])
        mock_joint.qvel = np.array([0.11])
        data.joint.side_effect = lambda name: mock_joint
        
        with patch('src.env.obs_wrapper.mujoco_utils') as mock_utils:
            mock_utils.get_site_xvelp.return_value = np.array([0.04, 0.05, 0.06])
            mock_utils.get_site_xvelr.return_value = np.array([0.01, 0.02, 0.03])
            mock_utils.robot_get_obs.return_value = (
                np.array([0.0] * 13 + [0.01, 0.02]),
                np.array([0.0] * 13 + [0.11, 0.22]),
            )

            n_substeps = 20
            obs = ObservationReconstructor.reconstruct_obs(
                model,
                data,
                "target",
                "target_site",
                n_substeps,
                time_feature=0.75,
            )
            # Observation vector should be 25-dimensional
            assert obs.shape == (25,)

            dt = n_substeps * 0.002  # 0.04

            # Verify relative position (indices 6-8)
            # obj_pos (4,5,6) - grip_pos (1,2,3) = (3,3,3)
            np.testing.assert_allclose(obs[6:9], [3.0, 3.0, 3.0])

            # Verify relative velocity (indices 14-16)
            # The test mock returns the same velocity vector for grip and object,
            # so the relative translational velocity should be zero.
            expected_rel_vel = np.zeros(3)
            np.testing.assert_allclose(obs[14:17], expected_rel_vel)

            # Verify grip_vel_lin_scaled (indices 20-22)
            expected_grip_vel = np.array([0.04, 0.05, 0.06]) * dt
            np.testing.assert_allclose(obs[20:23], expected_grip_vel)

            root.destroy()

