import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.env.obs_wrapper import reconstruct_obs

def test_reconstruct_obs_shape():
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
    data.site.side_effect = lambda name: mock_grip_site if name == 'robot0:grip' else None
    
    # Mock data.body(target_body_name).xpos, xquat, cvel
    mock_obj_body = MagicMock()
    mock_obj_body.xpos = np.array([4.0, 5.0, 6.0])
    mock_obj_body.xmat = np.eye(3).reshape(-1)
    mock_obj_body.xquat = np.array([1.0, 0.0, 0.0, 0.0])
    # cvel: 0-2 angular, 3-5 linear
    mock_obj_body.cvel = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    data.body.side_effect = lambda name: mock_obj_body if name == 'target' else None
    mock_obj_site = MagicMock()
    mock_obj_site.xpos = np.array([4.0, 5.0, 6.0])
    mock_obj_site.xmat = np.eye(3).reshape(-1)
    mock_obj_site.id = 1
    data.site.side_effect = lambda name: mock_grip_site if name == 'robot0:grip' else mock_obj_site
    
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
        obs = reconstruct_obs(
            model,
            data,
            "target",
            "target_site",
            n_substeps,
            time_feature=0.75,
        )

        assert obs.shape == (26,)

        dt = n_substeps * 0.002  # 0.04

        # Verify relative position (6-8)
        # obj_pos (4,5,6) - grip_pos (1,2,3) = (3,3,3)
        np.testing.assert_allclose(obs[6:9], [3.0, 3.0, 3.0])

        # Verify relative velocity (14-16)
        # The test mock returns the same velocity vector for grip and object,
        # so the relative translational velocity should be zero.
        expected_rel_vel = np.zeros(3)
        np.testing.assert_allclose(obs[14:17], expected_rel_vel)

        # Verify grip_vel_lin_scaled (20-22)
        expected_grip_vel = np.array([0.04, 0.05, 0.06]) * dt
        np.testing.assert_allclose(obs[20:23], expected_grip_vel)
        assert obs[25] == pytest.approx(0.75)

if __name__ == "__main__":
    pytest.main([__file__])
