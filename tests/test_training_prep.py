import chess
import mujoco
import pytest

from src.config import N_SUBSTEPS, SCENE_XML
from src.env.chess_manipulation_train_env import ChessManipulationTrainEnv
from src.game_runtime import initialize_square_to_piece
from src.logic.operation_planner import OperationPlanner
from src.training.config import FineTuneConfig
from src.training.tasks import ManipulationTask, TaskSampler, task_from_move


def _build_mapping():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, initialize_square_to_piece(model, data)


def test_task_from_move_builds_expected_fields():
    board = chess.Board()
    square_to_piece = {"e2": "w_pawn_5"}
    move = chess.Move.from_uci("e2e4")

    task = task_from_move(board, move, square_to_piece)

    assert task.move_uci == "e2e4"
    assert task.piece_name == "w_pawn_5"
    assert task.source_square == "e2"
    assert task.dest_square == "e4"
    assert task.is_capture is False


def test_task_sampler_board_from_sequence_applies_moves():
    sampler = TaskSampler()
    board = sampler.board_from_sequence(["e2e4", "e7e5"])

    assert board.piece_at(chess.E4).symbol() == "P"
    assert board.piece_at(chess.E5).symbol() == "p"


def test_task_sampler_first_legal_task_returns_deterministic_move():
    sampler = TaskSampler()
    board = chess.Board()
    square_to_piece = initialize_square_to_piece(*_build_mapping()[:2])

    task = sampler.sample_first_legal_task(board, square_to_piece)

    assert task.move_uci == "g1h3"


def test_task_sampler_tasks_from_openings_returns_one_per_opening():
    sampler = TaskSampler(opening_sequences=[["e2e4"], ["d2d4", "d7d5"]])
    _, _, square_to_piece = _build_mapping()

    tasks = sampler.tasks_from_openings(square_to_piece)

    assert len(tasks) == 2
    assert all(isinstance(task, ManipulationTask) for task in tasks)


def test_finetune_config_has_curriculum():
    config = FineTuneConfig.load_default()
    assert len(config.curriculum) >= 3
    assert config.curriculum[0].name == "center_opening"


def test_training_env_configure_task_returns_obs_and_sets_info():
    model, data, square_to_piece = _build_mapping()
    env = ChessManipulationTrainEnv(model, data, square_to_piece, n_substeps=N_SUBSTEPS)
    board = chess.Board()
    task = task_from_move(board, chess.Move.from_uci("e2e4"), square_to_piece)

    obs = env.configure_task(task)
    info = env.get_task_info()

    assert obs["desired_goal"].shape == (3,)
    assert info["move_uci"] == "e2e4"
    assert info["piece_name"] == "w_pawn_5"


def test_training_env_get_task_info_requires_configured_task():
    model, data, square_to_piece = _build_mapping()
    env = ChessManipulationTrainEnv(model, data, square_to_piece, n_substeps=N_SUBSTEPS)

    with pytest.raises(Exception, match="No manipulation task configured"):
        env.get_task_info()


def test_training_env_rejects_piece_mismatch():
    model, data, square_to_piece = _build_mapping()
    env = ChessManipulationTrainEnv(model, data, square_to_piece, n_substeps=N_SUBSTEPS)
    task = ManipulationTask(
        fen=chess.Board().fen(),
        move_uci="e2e4",
        piece_name="w_pawn_4",
        source_square="e2",
        dest_square="e4",
        is_capture=False,
    )

    with pytest.raises(Exception, match="Task piece mismatch"):
        env.configure_task(task)
