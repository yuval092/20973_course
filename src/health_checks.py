"""Pluggable runtime validation hooks for RoboChess."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

import numpy as np

from src.config import (
    FETCH_INIT_GRIP,
    PLACEMENT_TOLERANCE,
    PIECE_FOLLOW_TOLERANCE,
    REACHABLE_X_MAX,
    REACHABLE_X_MIN,
    REACHABLE_Y_MAX,
    REACHABLE_Y_MIN,
    REACHABLE_Z_MAX,
    REACHABLE_Z_MIN,
)
from src.exceptions import (
    ArmStateError,
    ExecutionError,
    MappingIntegrityError,
    NumericalStabilityError,
    SceneIntegrityError,
)
from src.logging_utils import log_event
from src.observation.board_observer import BoardObserver
from src.runtime_guard import validate_board_state

logger = logging.getLogger(__name__)


class CheckHook(str, Enum):
    """Named points in the runtime flow where checks can execute."""

    PROGRAM_START = "program_start"
    POST_SCENE_LOAD = "post_scene_load"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    ARM_STAGE_START = "arm_stage_start"
    ARM_STAGE_END = "arm_stage_end"


@dataclass
class CheckContext:
    """Context made available to all runtime checks."""

    hook: CheckHook
    systems: Any
    viewer: Any = None
    move_uci: Optional[str] = None
    stage: Optional[str] = None
    op: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


class RuntimeCheck:
    """Base class for runtime validation hooks."""

    name = "runtime_check"
    hooks: tuple[CheckHook, ...] = ()

    def run(self, context: CheckContext) -> None:
        raise NotImplementedError


class RuntimeCheckRegistry:
    """Registry that dispatches checks at named hook points."""

    def __init__(self, checks: Iterable[RuntimeCheck]):
        self._checks = tuple(checks)

    def run(self, hook: CheckHook, context: CheckContext) -> None:
        for check in self._checks:
            if hook not in check.hooks:
                continue
            log_event(logger, logging.DEBUG, "runtime_check_start", check=check.name, hook=hook.value)
            check.run(context)
            log_event(logger, logging.DEBUG, "runtime_check_ok", check=check.name, hook=hook.value)


class SceneAssetsCheck(RuntimeCheck):
    """Verify that required robot assets exist in the MuJoCo scene."""

    name = "scene_assets"
    hooks = (CheckHook.POST_SCENE_LOAD,)

    def run(self, context: CheckContext) -> None:
        model = context.systems.mj_model
        required_bodies = ("w_king", "b_king")
        required_sites = ("robot0:grip",)
        required_actuators = ("robot0:l_gripper_finger_joint", "robot0:r_gripper_finger_joint")

        for body_name in required_bodies:
            try:
                model.body(body_name)
            except KeyError as exc:
                raise SceneIntegrityError(f"Required body '{body_name}' missing from scene.") from exc

        for site_name in required_sites:
            try:
                model.site(site_name)
            except KeyError as exc:
                raise SceneIntegrityError(f"Required site '{site_name}' missing from scene.") from exc

        for actuator_name in required_actuators:
            try:
                model.actuator(actuator_name)
            except KeyError as exc:
                raise SceneIntegrityError(f"Required actuator '{actuator_name}' missing from scene.") from exc


class MappingIntegrityCheck(RuntimeCheck):
    """Verify that the square mapping is one-to-one and matches board piece count."""

    name = "mapping_integrity"
    hooks = (CheckHook.PROGRAM_START, CheckHook.TURN_START, CheckHook.TURN_END)

    def run(self, context: CheckContext) -> None:
        square_to_piece = context.systems.square_to_piece
        board_piece_count = len(context.systems.manager.board.piece_map())

        if len(square_to_piece) != len(set(square_to_piece.values())):
            raise MappingIntegrityError("Square mapping contains duplicate piece assignments.")
        if len(square_to_piece) != board_piece_count:
            raise MappingIntegrityError(
                f"Square mapping count {len(square_to_piece)} does not match board piece count {board_piece_count}."
            )


class BoardAgreementCheck(RuntimeCheck):
    """Verify that piece positions and identities match the logical board."""

    name = "board_agreement"
    hooks = (CheckHook.PROGRAM_START, CheckHook.TURN_START, CheckHook.TURN_END)

    def run(self, context: CheckContext) -> None:
        systems = context.systems
        validate_board_state(
            systems.mj_model,
            systems.mj_data,
            systems.manager.board,
            systems.square_to_piece,
            systems.controller,
        )


class ArmHomePoseCheck(RuntimeCheck):
    """Verify that the gripper returns close to the configured home pose."""

    name = "arm_home_pose"
    hooks = (CheckHook.PROGRAM_START, CheckHook.TURN_END)

    def __init__(self, tolerance: float = PLACEMENT_TOLERANCE):
        self.tolerance = tolerance

    def run(self, context: CheckContext) -> None:
        actual = context.systems.env.get_grip_pos()
        expected = context.systems.arm_home_grip
        error = float(np.linalg.norm(actual - expected))
        if error > self.tolerance:
            raise ArmStateError(
                f"Gripper not at home pose (error={error:.4f}, expected={expected.round(4)}, actual={actual.round(4)})."
            )


class RobotWorkspaceCheck(RuntimeCheck):
    """Verify that the gripper remains inside the configured reachable workspace."""

    name = "robot_workspace"
    hooks = (CheckHook.PROGRAM_START, CheckHook.TURN_START, CheckHook.TURN_END)

    def run(self, context: CheckContext) -> None:
        grip = context.systems.env.get_grip_pos()
        if not (
            REACHABLE_X_MIN <= grip[0] <= REACHABLE_X_MAX
            and REACHABLE_Y_MIN <= grip[1] <= REACHABLE_Y_MAX
            and REACHABLE_Z_MIN <= grip[2] <= REACHABLE_Z_MAX
        ):
            raise ArmStateError(f"Gripper outside workspace: {grip.round(4)}")


class FiniteStateCheck(RuntimeCheck):
    """Verify that key MuJoCo state arrays contain only finite values."""

    name = "finite_state"
    hooks = (
        CheckHook.POST_SCENE_LOAD,
        CheckHook.PROGRAM_START,
        CheckHook.TURN_START,
        CheckHook.TURN_END,
        CheckHook.ARM_STAGE_START,
        CheckHook.ARM_STAGE_END,
    )

    def run(self, context: CheckContext) -> None:
        data = context.systems.mj_data
        arrays = {"qpos": data.qpos, "qvel": data.qvel, "ctrl": data.ctrl}
        for name, array in arrays.items():
            if array.size and not np.isfinite(array).all():
                raise NumericalStabilityError(f"Non-finite values detected in {name}.")


class PieceObserverCheck(RuntimeCheck):
    """Run the physical board observer for tipped or buried pieces."""

    name = "piece_observer"
    hooks = (CheckHook.PROGRAM_START, CheckHook.TURN_START, CheckHook.TURN_END)

    def run(self, context: CheckContext) -> None:
        systems = context.systems
        BoardObserver(systems.mj_model, systems.mj_data).verify_stability(
            active_piece_names=set(systems.square_to_piece.values())
        )


class StageGoalReachabilityCheck(RuntimeCheck):
    """Verify that stage goals are reachable before stage execution begins."""

    name = "stage_goal_reachability"
    hooks = (CheckHook.ARM_STAGE_START,)

    def run(self, context: CheckContext) -> None:
        goal = context.extra.get("goal")
        if goal is None:
            return
        if not context.systems.controller._is_reachable(goal):
            raise ArmStateError(f"Stage {context.stage} goal is unreachable: {np.asarray(goal).round(4)}")


class StageOutcomeCheck(RuntimeCheck):
    """Fail immediately when an arm stage reports unsuccessful completion."""

    name = "stage_outcome"
    hooks = (CheckHook.ARM_STAGE_END,)

    def run(self, context: CheckContext) -> None:
        success = context.extra.get("success")
        if success is False:
            raise ExecutionError(f"Arm stage failed: {context.stage}")


class StageGripAttachmentCheck(RuntimeCheck):
    """Verify that close/lift stages keep the target piece attached to the gripper."""

    name = "stage_grip_attachment"
    hooks = (CheckHook.ARM_STAGE_END,)

    def run(self, context: CheckContext) -> None:
        if context.stage not in {"CLOSE_GRIPPER_ONLY", "LIFT_VERIFY"}:
            return
        if context.extra.get("success") is not True:
            return

        systems = context.systems
        grip = systems.env.get_grip_pos()
        piece = systems.env.get_piece_pos()
        distance = float(np.linalg.norm(grip - piece))
        if distance > PIECE_FOLLOW_TOLERANCE * 1.5:
            raise ArmStateError(
                f"Target piece detached from gripper after {context.stage} (distance={distance:.4f})."
            )


def build_default_check_registry() -> RuntimeCheckRegistry:
    """Create the default runtime validation suite."""
    return RuntimeCheckRegistry(
        checks=[
            SceneAssetsCheck(),
            MappingIntegrityCheck(),
            BoardAgreementCheck(),
            ArmHomePoseCheck(),
            RobotWorkspaceCheck(),
            FiniteStateCheck(),
            PieceObserverCheck(),
            StageGoalReachabilityCheck(),
            StageOutcomeCheck(),
            StageGripAttachmentCheck(),
        ]
    )
