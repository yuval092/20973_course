"""
Tests for the ExecutionController class.

This module verifies the robot arm's execution logic, including reachability
checks, policy workspace validation, stage target generation, and various
sub-operation components like arm retraction and piece settling.
"""

import numpy as np
from src.control.execution_controller import ExecutionController
from src.config import (
    Z_GRASP,
    Z_SAFE,
    CLOSE_DESCEND_OFFSET,
    FINAL_SETTLE_STEPS,
    PLACEMENT_TOLERANCE,
)


class TestExecutionController:
    """
    Test suite for the ExecutionController and its internal logic.
    """

    class DummyBody:
        """Dummy body class for mocking MuJoCo body."""
        def __init__(self, xpos, xquat=None):
            self.xpos = xpos
            self.xquat = xquat if xquat is not None else np.array([1.0, 0.0, 0.0, 0.0])

    class DummyData:
        """Dummy data class for mocking MuJoCo data."""
        def __init__(self, body):
            self._body = body

        def body(self, _name):
            return self._body

    class DummyEnv:
        """Dummy environment class for mocking robot environment."""
        def __init__(self, body=None):
            self.data = TestExecutionController.DummyData(body)
            self.model = object()
            self.target_body_name = "piece"
            self.force_gripper_open_calls = 0
            self.reset_robot_pose_calls = 0
            self._home_grip_pos = np.array([1.25, 0.75, 0.55])
            self._grip_pos = np.array([1.20, 0.70, 0.50])

        def get_piece_pos(self):
            return self.data.body("piece").xpos.copy()

        def get_piece_linear_velocity(self):
            return np.array([0.0, 0.0, 0.0])

        def force_gripper_open(self):
            self.force_gripper_open_calls += 1

        def reset_robot_pose(self):
            self.reset_robot_pose_calls += 1

        @property
        def home_grip_pos(self):
            return self._home_grip_pos.copy()

        def get_grip_pos(self):
            return self._grip_pos.copy()

        def get_obs(self):
            return {"observation": np.zeros(1)}

        def get_finger_joint_positions(self):
            return np.array([0.05, 0.05])

    def test_reachability_accepts_in_bounds_goal(self):
        """
        Verify that reachable coordinates are accepted.
        """
        assert ExecutionController._is_reachable(np.array([1.20, 0.60, 0.50]))

    def test_reachability_rejects_out_of_bounds_goal(self):
        """
        Verify that unreachable coordinates are rejected.
        """
        assert not ExecutionController._is_reachable(np.array([1.60, 0.60, 0.50]))

    def test_policy_workspace_rejects_far_back_rank_square(self):
        """
        Verify that squares too far from the robot are flagged.
        """
        controller = ExecutionController(None, None)
        # Point reachable but far away from FETCH_INIT_GRIP [1.3419, 0.7491, 0.575]
        # [0.9, 1.18] is ~0.61m away, exceeding 0.5m radius
        issues = controller._policy_compatibility_issues(
            np.array([0.9, 1.18, Z_GRASP]),
            np.array([1.255, 0.675, Z_GRASP]),
        )
        assert any("src_xy_dist" in issue for issue in issues)

    def test_policy_workspace_accepts_midboard_move(self):
        """
        Verify that mid-board moves are within the policy workspace.
        """
        controller = ExecutionController(None, None)
        # Mid-board point near FETCH_INIT_GRIP
        issues = controller._policy_compatibility_issues(
            np.array([1.255, 0.675, 0.421]),
            np.array([1.205, 0.775, 0.421]),
        )
        assert issues == []

    def test_preflight_detects_policy_issues_before_motion(self):
        """
        Verify that pre-flight checks identify potential workspace issues.
        """
        class DummyOp:
            piece_name = "b_knight_2"
            target_square = "g8"
            dest_square = "f6"

        # Point reachable but far away from DummyEnv's home [1.25, 0.75, 0.55]
        # [0.9, 1.18] is ~0.55m away, exceeding 0.5m radius
        body = self.DummyBody(np.array([0.9, 1.18, Z_GRASP]))
        env = self.DummyEnv(body)
        controller = ExecutionController(None, env)
        preflight = controller._preflight_op(DummyOp())

        assert preflight.policy_issues
        assert preflight.failure_result is None

    def test_stage_targets_split_move_into_explicit_steps(self):
        """
        Verify that a move is correctly decomposed into stages.
        """
        controller = ExecutionController(None, None)
        src = np.array([1.255, 0.675, Z_GRASP])
        dest = np.array([1.205, 0.775, Z_GRASP])
        live_piece = np.array([1.255, 0.675, Z_GRASP - 0.0001])

        targets = controller._build_stage_targets(src, dest, live_piece)

        assert list(targets.keys()) == controller.STAGES
        np.testing.assert_allclose(targets["PREHOVER_SRC"], [1.255, 0.675, Z_SAFE])
        assert targets["PREGRASP_NARROW"] is None
        np.testing.assert_allclose(
            targets["DESCEND_SRC"],
            [1.255, 0.675, live_piece[2] + controller._source_descend_offset_z],
        )
        np.testing.assert_allclose(targets["LIFT_VERIFY"], [1.255, 0.675, Z_SAFE])
        np.testing.assert_allclose(targets["PREHOVER_DEST"], [1.205, 0.775, Z_SAFE])
        np.testing.assert_allclose(
            targets["DESCEND_DEST"],
            [1.205, 0.775, dest[2] - CLOSE_DESCEND_OFFSET]
        )
        assert targets["POST_RELEASE_SETTLE"] is None
        assert targets["POST_RELEASE_CLEARANCE"][2] >= Z_SAFE

    def test_descend_dest_goal_uses_live_carry_offset_when_available(self):
        """
        Verify that stage goal resolution uses current piece position.
        """
        class CustomEnv(self.DummyEnv):
            def get_piece_pos(self):
                return np.array([1.205, 0.775, 0.533])

            def get_grip_pos(self):
                return np.array([1.205, 0.775, 0.580])

        controller = ExecutionController(None, CustomEnv())
        src = np.array([1.255, 0.675, Z_GRASP])
        dest = np.array([1.205, 0.775, Z_GRASP])
        goal = controller._resolve_stage_goal("DESCEND_DEST", {}, src, dest)
        # Bug Fix Verification: It now uses dest[2] directly
        np.testing.assert_allclose(
            goal,
            [1.205, 0.775, dest[2] - CLOSE_DESCEND_OFFSET]
        )

    def test_retract_arm_moves_to_home_without_hard_reset(self):
        """
        Verify that retracting the arm correctly moves to home position.
        """
        env = self.DummyEnv()
        controller = ExecutionController(None, env)
        recorded = []

        def fake_move(target_pos, viewer=None, gripper_action=1.0, **_kwargs):
            recorded.append((target_pos.copy(), gripper_action))
            return True, 12

        controller._move_gripper_to = fake_move
        controller._settle = lambda viewer=None, steps=None: None

        class DummyViewer:
            def sync(self):
                return None

        viewer = DummyViewer()
        success, steps_taken = controller._retract_arm(viewer=viewer)

        assert success is True
        assert steps_taken == 24
        assert controller.env.force_gripper_open_calls == 1
        assert controller.env.reset_robot_pose_calls == 0
        np.testing.assert_allclose(recorded[0][0], [1.20, 0.70, recorded[0][0][2]])
        np.testing.assert_allclose(recorded[1][0], controller.env.home_grip_pos)
        assert recorded[0][1] == 1.0
        assert recorded[1][1] == 1.0

    def test_compose_guided_action_blends_aligned_rl_signal(self):
        """
        Verify that RL signals are blended when aligned with the goal.
        """
        class DummyPolicy:
            def predict(self, _obs, deterministic=True):
                return np.array([0.2, 0.4, 0.6, -1.0]), None

        env = self.DummyEnv()
        controller = ExecutionController(DummyPolicy(), env)
        action = controller._compose_guided_action(
            np.array([0.05, 0.0, 0.0]),
            use_rl=True
        )

        assert action.shape == (4,)
        assert 0.7 < action[0] <= 1.0
        assert action[3] == 0.0

    def test_compose_guided_action_rejects_misaligned_rl_signal(self):
        """
        Verify that RL signals are ignored if they misalign with the goal.
        """
        class DummyPolicy:
            def predict(self, _obs, deterministic=True):
                return np.array([-1.0, 0.0, 0.0, -1.0]), None

        env = self.DummyEnv()
        controller = ExecutionController(DummyPolicy(), env)
        action = controller._compose_guided_action(
            np.array([0.05, 0.0, 0.0]),
            use_rl=True
        )

        np.testing.assert_allclose(action, [1.0, 0.0, 0.0, 0.0])

    def test_finalize_placement_reports_failure_when_piece_never_settles(self):
        """
        Verify that placement fails if the piece remains unstable.
        """
        body = self.DummyBody(
            np.array([1.10, 0.70, Z_GRASP + PLACEMENT_TOLERANCE * 3])
        )
        env = self.DummyEnv(body)
        controller = ExecutionController(None, env)
        controller._sync_viewer = lambda viewer=None: None

        import src.control.execution_controller as execution_controller_module

        original_step = execution_controller_module.mujoco.mj_step
        execution_controller_module.mujoco.mj_step = lambda _model, _data: None
        try:
            success, steps_taken = controller._finalize_placement(
                np.array([1.10, 0.70, Z_GRASP]),
                viewer=None
            )
        finally:
            execution_controller_module.mujoco.mj_step = original_step

        assert success is False
        assert steps_taken == FINAL_SETTLE_STEPS

    def test_released_piece_issue_flags_vertical_drag_even_when_jitter_is_tolerated(self):
        """
        Verify that vertical displacement issues are flagged after release.
        """
        body = self.DummyBody(np.array([1.10, 0.70, Z_GRASP + 0.08]))
        env = self.DummyEnv(body)
        controller = ExecutionController(None, env)
        issue = controller._released_piece_issue(
            np.array([1.10, 0.70, Z_GRASP]),
            tolerate_contact_jitter=True,
        )

        assert issue is not None
        assert "z_error" in issue

    def test_settle_released_piece_succeeds_even_when_close_to_gripper(self):
        """
        Verify that settling succeeds even if the piece is still near the gripper 
        (since the arm hasn't lifted yet).
        """
        class CustomEnv(self.DummyEnv):
            def get_piece_pos(self):
                return np.array([1.10, 0.70, Z_GRASP])

            def get_grip_pos(self):
                return np.array([1.10, 0.70, Z_GRASP + 0.005])

        env = CustomEnv()
        controller = ExecutionController(None, env)
        controller._sync_viewer = lambda viewer=None: None

        import src.control.execution_controller as execution_controller_module

        original_step = execution_controller_module.mujoco.mj_step
        execution_controller_module.mujoco.mj_step = lambda _model, _data: None
        try:
            success, _steps_taken = controller._settle_released_piece(
                np.array([1.10, 0.70, Z_GRASP]),
                viewer=None,
                max_steps=3
            )
        finally:
            execution_controller_module.mujoco.mj_step = original_step

        assert success is True
