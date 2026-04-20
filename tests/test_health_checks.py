"""
Unit tests for the health check system.

This module verifies that various runtime health checks (mapping integrity, 
arm state, numerical stability, etc.) correctly identify and report system failures.
"""

import mujoco
import numpy as np
import pytest

from src.config import N_SUBSTEPS, SCENE_XML
from src.control.execution_controller import ExecutionController
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.exceptions import (
    ArmStateError, ExecutionError, MappingIntegrityError, 
    NumericalStabilityError, SceneIntegrityError
)
from src.models import GameSystems
from src.bootstrap import SystemBootstrapper
from src.health_checks import (
    ArmHomePoseCheck, CheckContext, CheckHook, FiniteStateCheck,
    MappingIntegrityCheck, RobotWorkspaceCheck, RuntimeCheck,
    RuntimeCheckRegistry, SceneAssetsCheck, StageGoalReachabilityCheck,
    StageOutcomeCheck, build_default_check_registry
)
from src.logic.chess_manager import ChessGameManager
from src.logic.operation_planner import OperationPlanner, PickPlaceOp


class TestHealthChecks:
    """
    Test suite for system health checks and runtime validation.
    """

    def _build_systems(self):
        """
        Helper to initialize the full game system context for testing.
        """
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        env = ChessPickPlaceEnv(model, data, n_substeps=N_SUBSTEPS)
        controller = ExecutionController(None, env)
        manager = ChessGameManager()
        planner = OperationPlanner()
        square_to_piece = SystemBootstrapper.initialize_square_to_piece(model, data)
        
        return GameSystems(
            mj_model=model,
            mj_data=data,
            env=env,
            rl_model=None,
            manager=manager,
            planner=planner,
            controller=controller,
            square_to_piece=square_to_piece,
            arm_home_grip=env.get_grip_pos().copy(),
            check_registry=build_default_check_registry(),
        )

    def test_mapping_integrity_rejects_duplicate_piece_assignment(self):
        """
        Verify that MappingIntegrityCheck detects duplicate piece entries.
        """
        systems = self._build_systems()
        systems.square_to_piece["e2"] = systems.square_to_piece["d2"]
        check = MappingIntegrityCheck()

        with pytest.raises(MappingIntegrityError, match="duplicate"):
            check.run(CheckContext(hook=CheckHook.TURN_START, systems=systems))

    def test_mapping_integrity_rejects_count_mismatch(self):
        """
        Verify that MappingIntegrityCheck detects missing pieces in the mapping.
        """
        systems = self._build_systems()
        systems.square_to_piece.pop("e2")
        check = MappingIntegrityCheck()

        with pytest.raises(MappingIntegrityError, match="does not match"):
            check.run(CheckContext(hook=CheckHook.TURN_START, systems=systems))

    def test_arm_home_pose_check_rejects_offset_gripper(self):
        """
        Verify that ArmHomePoseCheck detects when the arm is not at its home position.
        """
        systems = self._build_systems()
        systems.arm_home_grip = systems.arm_home_grip + np.array([0.05, 0.0, 0.0])
        check = ArmHomePoseCheck()

        with pytest.raises(ArmStateError, match="not at home pose"):
            check.run(CheckContext(hook=CheckHook.TURN_END, systems=systems))

    def test_robot_workspace_check_rejects_out_of_bounds_gripper(self):
        """
        Verify that RobotWorkspaceCheck detects gripper positions outside valid bounds.
        """
        systems = self._build_systems()
        # Mocking the gripper position to be far outside the workspace
        systems.env.get_grip_pos = lambda: np.array([2.0, 2.0, 2.0])
        check = RobotWorkspaceCheck()

        with pytest.raises(ArmStateError, match="outside workspace"):
            check.run(CheckContext(hook=CheckHook.TURN_END, systems=systems))

    def test_finite_state_check_rejects_nan_state(self):
        """
        Verify that FiniteStateCheck detects NaN values in physics state.
        """
        systems = self._build_systems()
        systems.mj_data.qpos[0] = np.nan
        check = FiniteStateCheck()

        with pytest.raises(NumericalStabilityError, match="qpos"):
            check.run(CheckContext(hook=CheckHook.TURN_START, systems=systems))

    def test_scene_assets_check_passes_for_real_scene(self):
        """
        Verify that SceneAssetsCheck succeeds for a standard loaded scene.
        """
        systems = self._build_systems()
        check = SceneAssetsCheck()
        check.run(CheckContext(hook=CheckHook.POST_SCENE_LOAD, systems=systems))

    def test_scene_assets_check_raises_when_required_body_missing(self):
        """
        Verify that SceneAssetsCheck fails when a critical body is missing from the model.
        """
        systems = self._build_systems()

        class BrokenModel:
            """Mock model that hides a specific body."""
            def __init__(self, real_model):
                self._real = real_model

            def body(self, name):
                if name == "w_king":
                    raise KeyError(name)
                return self._real.body(name)

            def site(self, name):
                return self._real.site(name)

            def actuator(self, name):
                return self._real.actuator(name)

        systems.mj_model = BrokenModel(systems.mj_model)
        check = SceneAssetsCheck()

        with pytest.raises(SceneIntegrityError, match="w_king"):
            check.run(CheckContext(hook=CheckHook.POST_SCENE_LOAD, systems=systems))

    def test_runtime_check_registry_dispatches_only_matching_hook(self):
        """
        Verify that the registry only executes checks registered for a specific hook.
        """
        calls = []

        class TurnEndOnlyCheck(RuntimeCheck):
            name = "turn_end_only"
            hooks = (CheckHook.TURN_END,)

            def run(self, context):
                calls.append(context.hook)

        registry = RuntimeCheckRegistry([TurnEndOnlyCheck()])
        systems = self._build_systems()

        registry.run(CheckHook.TURN_START, CheckContext(hook=CheckHook.TURN_START, systems=systems))
        registry.run(CheckHook.TURN_END, CheckContext(hook=CheckHook.TURN_END, systems=systems))

        assert calls == [CheckHook.TURN_END]

    def test_execution_controller_emits_stage_events(self):
        """
        Verify that the execution controller triggers hook events for each manipulation stage.
        """
        systems = self._build_systems()
        events = []
        systems.controller.set_stage_event_handler(
            lambda hook, **payload: events.append((hook, payload["stage"]))
        )

        result = systems.controller.execute_op(
            PickPlaceOp("e4", "e5", "w_pawn_5"), viewer=None
        )

        assert result.success
        starts = [e for e in events if e[0] == CheckHook.ARM_STAGE_START]
        ends = [e for e in events if e[0] == CheckHook.ARM_STAGE_END]
        assert len(starts) == len(systems.controller.STAGES)
        assert len(ends) == len(systems.controller.STAGES)

    def test_stage_goal_reachability_reads_goal_from_context_extra(self):
        """
        Verify that StageGoalReachabilityCheck detects unreachable goals passed via context.
        """
        systems = self._build_systems()
        check = StageGoalReachabilityCheck()
        unreachable_goal = np.array([9.0, 9.0, 9.0])

        with pytest.raises(ArmStateError, match="unreachable"):
            check.run(
                CheckContext(
                    hook=CheckHook.ARM_STAGE_START,
                    systems=systems,
                    stage="PREHOVER_DEST",
                    extra={"goal": unreachable_goal},
                )
            )

    def test_stage_outcome_check_raises_on_unsuccessful_stage(self):
        """
        Verify that StageOutcomeCheck detects and reports a failed stage.
        """
        systems = self._build_systems()
        check = StageOutcomeCheck()

        with pytest.raises(ExecutionError, match="Arm stage failed"):
            check.run(
                CheckContext(
                    hook=CheckHook.ARM_STAGE_END,
                    systems=systems,
                    stage="DESCEND_SRC",
                    extra={"success": False},
                )
            )
