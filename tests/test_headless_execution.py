import mujoco

from src.config import SCENE_XML, N_SUBSTEPS
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.logic.operation_planner import PickPlaceOp


class DummyViewer:
    def sync(self):
        return None


def _build_env_and_controller():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
    controller = ExecutionController(None, env)
    return env, controller, DummyViewer()


def test_headless_close_stage_no_longer_false_fails_on_valid_pregrasp():
    env, controller, viewer = _build_env_and_controller()
    src = controller.get_square_pos("g8")
    dest = controller.get_square_pos("f6")
    env.set_target("b_knight_2", src)

    for stage in ["HOME_RESET", "PREHOVER_SRC", "PREGRASP_NARROW", "DESCEND_SRC"]:
        goal = controller._resolve_stage_goal(
            stage,
            controller._build_stage_targets(src, dest, env.data.body("b_knight_2").xpos.copy()),
            src,
            dest,
        )
        success, _ = controller._execute_stage(stage, "b_knight_2", goal, viewer)
        assert success, stage

    success, _ = controller._execute_stage("CLOSE_GRIPPER_ONLY", "b_knight_2", None, viewer)
    assert success


def test_headless_lift_verify_confirms_piece_follow_after_close():
    env, controller, viewer = _build_env_and_controller()
    src = controller.get_square_pos("g8")
    dest = controller.get_square_pos("f6")
    env.set_target("b_knight_2", src)

    for stage in ["HOME_RESET", "PREHOVER_SRC", "PREGRASP_NARROW", "DESCEND_SRC", "CLOSE_GRIPPER_ONLY"]:
        goal = controller._resolve_stage_goal(
            stage,
            controller._build_stage_targets(src, dest, env.data.body("b_knight_2").xpos.copy()),
            src,
            dest,
        )
        success, _ = controller._execute_stage(stage, "b_knight_2", goal, viewer)
        assert success, stage

    goal = controller._resolve_stage_goal(
        "LIFT_VERIFY",
        controller._build_stage_targets(src, dest, env.data.body("b_knight_2").xpos.copy()),
        src,
        dest,
    )
    success, _ = controller._execute_stage("LIFT_VERIFY", "b_knight_2", goal, viewer)
    assert success


def test_headless_execute_op_places_piece_on_board_upright():
    env, controller, viewer = _build_env_and_controller()
    op = PickPlaceOp(target_square="g8", dest_square="f6", piece_name="b_knight_2")

    result = controller.execute_op(op, viewer=viewer)

    piece = env.data.body("b_knight_2")
    dest = controller.get_square_pos("f6")
    xy_error = ((piece.xpos[0] - dest[0]) ** 2 + (piece.xpos[1] - dest[1]) ** 2) ** 0.5
    z_error = abs(piece.xpos[2] - dest[2])
    up_z = 1.0 - 2.0 * (piece.xquat[1] ** 2 + piece.xquat[2] ** 2)

    assert result.success, result.details
    assert xy_error < 0.02
    assert z_error < 0.02
    assert up_z > 0.966


def test_headless_sequential_knight_moves_succeed_on_visible_path():
    env, controller, viewer = _build_env_and_controller()

    first = PickPlaceOp(target_square="g8", dest_square="f6", piece_name="b_knight_2")
    first_result = controller.execute_op(first, viewer=viewer)
    assert first_result.success, first_result.details

    piece_after_first = env.data.body("b_knight_2").xpos.copy()
    nominal_f6 = controller.get_square_pos("f6")
    first_xy_error = ((piece_after_first[0] - nominal_f6[0]) ** 2 + (piece_after_first[1] - nominal_f6[1]) ** 2) ** 0.5
    assert first_xy_error < 0.02

    second = PickPlaceOp(target_square="f6", dest_square="e4", piece_name="b_knight_2")
    second_result = controller.execute_op(second, viewer=viewer)
    assert second_result.success, second_result.details

    piece_after_second = env.data.body("b_knight_2")
    nominal_e4 = controller.get_square_pos("e4")
    xy_error = ((piece_after_second.xpos[0] - nominal_e4[0]) ** 2 + (piece_after_second.xpos[1] - nominal_e4[1]) ** 2) ** 0.5
    up_z = 1.0 - 2.0 * (piece_after_second.xquat[1] ** 2 + piece_after_second.xquat[2] ** 2)

    assert xy_error < 0.02
    assert up_z > 0.966
