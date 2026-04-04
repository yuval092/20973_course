import numpy as np

from src.control.execution_controller import ExecutionController
from src.config import (
    Z_GRASP,
    Z_SAFE,
    GRASP_DESCEND_OFFSET,
    CLOSE_DESCEND_OFFSET,
    FINAL_SETTLE_STEPS,
    PLACEMENT_TOLERANCE,
)


def test_reachability_accepts_in_bounds_goal():
    assert ExecutionController._is_reachable(np.array([1.20, 0.60, 0.50]))


def test_reachability_rejects_out_of_bounds_goal():
    assert not ExecutionController._is_reachable(np.array([1.60, 0.60, 0.50]))


def test_policy_workspace_rejects_far_back_rank_square():
    controller = ExecutionController(None, None)
    issues = controller._policy_compatibility_issues(
        np.array([1.305, 0.575, Z_GRASP]),
        np.array([1.255, 0.675, Z_GRASP]),
    )
    assert any("src_xy_dist" in issue for issue in issues)


def test_policy_workspace_accepts_midboard_move():
    controller = ExecutionController(None, None)
    issues = controller._policy_compatibility_issues(
        np.array([1.255, 0.675, 0.421]),
        np.array([1.205, 0.775, 0.421]),
    )
    assert issues == []


def test_preflight_detects_policy_issues_before_motion():
    class DummyBody:
        def __init__(self, xpos):
            self.xpos = xpos

    class DummyData:
        def body(self, _name):
            return DummyBody(np.array([1.305, 0.575, Z_GRASP]))

    class DummyEnv:
        def __init__(self):
            self.data = DummyData()

    class DummyOp:
        piece_name = "b_knight_2"
        target_square = "g8"
        dest_square = "f6"

    controller = ExecutionController(None, DummyEnv())
    preflight = controller._preflight_op(DummyOp())

    assert preflight.policy_issues
    assert preflight.failure_result is None


def test_stage_targets_split_move_into_explicit_steps():
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
    np.testing.assert_allclose(targets["DESCEND_DEST"], [1.205, 0.775, live_piece[2] + GRASP_DESCEND_OFFSET])
    assert targets["POST_RELEASE_SETTLE"] is None
    assert targets["POST_RELEASE_CLEARANCE"][2] >= Z_SAFE


def test_descend_dest_goal_uses_live_carry_offset_when_available():
    class DummyEnv:
        def get_piece_pos(self):
            return np.array([1.205, 0.775, 0.533])

        def get_grip_pos(self):
            return np.array([1.205, 0.775, 0.580])

    controller = ExecutionController(None, DummyEnv())
    src = np.array([1.255, 0.675, Z_GRASP])
    dest = np.array([1.205, 0.775, Z_GRASP])
    goal = controller._resolve_stage_goal("DESCEND_DEST", {}, src, dest)
    np.testing.assert_allclose(goal, [1.205, 0.775, Z_GRASP + 0.047 + CLOSE_DESCEND_OFFSET])


def test_retract_arm_moves_to_home_without_hard_reset():
    class DummyEnv:
        def __init__(self):
            self.force_gripper_open_calls = 0
            self.reset_robot_pose_calls = 0
            self._home_grip_pos = np.array([1.25, 0.75, 0.55])
            self._grip_pos = np.array([1.20, 0.70, 0.50])

        def force_gripper_open(self):
            self.force_gripper_open_calls += 1

        def reset_robot_pose(self):
            self.reset_robot_pose_calls += 1

        @property
        def home_grip_pos(self):
            return self._home_grip_pos.copy()

        def get_grip_pos(self):
            return self._grip_pos.copy()

    controller = ExecutionController(None, DummyEnv())
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


def test_compose_guided_action_blends_aligned_rl_signal():
    class DummyPolicy:
        def predict(self, _obs, deterministic=True):
            return np.array([0.2, 0.4, 0.6, -1.0]), None

    class DummyEnv:
        def get_obs(self):
            return {"observation": np.zeros(1)}

    controller = ExecutionController(DummyPolicy(), DummyEnv())
    action = controller._compose_guided_action(np.array([0.05, 0.0, 0.0]), use_rl=True)

    assert action.shape == (4,)
    assert 0.7 < action[0] <= 1.0
    assert action[3] == 0.0


def test_compose_guided_action_rejects_misaligned_rl_signal():
    class DummyPolicy:
        def predict(self, _obs, deterministic=True):
            return np.array([-1.0, 0.0, 0.0, -1.0]), None

    class DummyEnv:
        def get_obs(self):
            return {"observation": np.zeros(1)}

    controller = ExecutionController(DummyPolicy(), DummyEnv())
    action = controller._compose_guided_action(np.array([0.05, 0.0, 0.0]), use_rl=True)

    np.testing.assert_allclose(action, [1.0, 0.0, 0.0, 0.0])


def test_finalize_placement_reports_failure_when_piece_never_settles():
    class DummyBody:
        def __init__(self):
            self.xpos = np.array([1.10, 0.70, Z_GRASP + PLACEMENT_TOLERANCE * 3])
            self.xquat = np.array([1.0, 0.0, 0.0, 0.0])

    class DummyData:
        def __init__(self, body):
            self._body = body

        def body(self, _name):
            return self._body

    class DummyEnv:
        def __init__(self):
            self.model = object()
            self.data = DummyData(DummyBody())
            self.target_body_name = "piece"

        def get_piece_pos(self):
            return self.data.body("piece").xpos.copy()

        def get_piece_linear_velocity(self):
            return np.array([0.0, 0.0, 0.0])

    controller = ExecutionController(None, DummyEnv())
    controller._sync_viewer = lambda viewer=None: None

    import src.control.execution_controller as execution_controller_module

    original_step = execution_controller_module.mujoco.mj_step
    execution_controller_module.mujoco.mj_step = lambda _model, _data: None
    try:
        success, steps_taken = controller._finalize_placement(np.array([1.10, 0.70, Z_GRASP]), viewer=None)
    finally:
        execution_controller_module.mujoco.mj_step = original_step

    assert success is False
    assert steps_taken == FINAL_SETTLE_STEPS


def test_released_piece_issue_flags_vertical_drag_even_when_jitter_is_tolerated():
    class DummyEnv:
        def get_piece_pos(self):
            return np.array([1.10, 0.70, Z_GRASP + 0.08])

        def get_piece_linear_velocity(self):
            return np.array([0.0, 0.0, 0.0])

    controller = ExecutionController(None, DummyEnv())
    issue = controller._released_piece_issue(
        np.array([1.10, 0.70, Z_GRASP]),
        tolerate_contact_jitter=True,
    )

    assert issue is not None
    assert "z_error" in issue


def test_settle_released_piece_requires_piece_to_clear_the_gripper():
    class DummyEnv:
        def __init__(self):
            self.model = object()
            self.data = object()

        def get_piece_pos(self):
            return np.array([1.10, 0.70, Z_GRASP])

        def get_piece_linear_velocity(self):
            return np.array([0.0, 0.0, 0.0])

        def get_grip_pos(self):
            return np.array([1.10, 0.70, Z_GRASP + 0.01])

        def get_gripper_finger_qpos(self):
            return np.array([0.05, 0.05])

    controller = ExecutionController(None, DummyEnv())
    controller._sync_viewer = lambda viewer=None: None

    import src.control.execution_controller as execution_controller_module

    original_step = execution_controller_module.mujoco.mj_step
    execution_controller_module.mujoco.mj_step = lambda _model, _data: None
    try:
        success, _steps_taken = controller._settle_released_piece(np.array([1.10, 0.70, Z_GRASP]), viewer=None, max_steps=3)
    finally:
        execution_controller_module.mujoco.mj_step = original_step

    assert success is False
