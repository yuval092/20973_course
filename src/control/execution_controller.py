"""
Execution Controller — Staged Cartesian Manipulation

Executes pick-and-place operations using explicit, debuggable stages:

  1. HOME_RESET
  2. PREHOVER_SRC
  3. PREGRASP_NARROW
  4. DESCEND_SRC
  5. CLOSE_GRIPPER_ONLY
  6. LIFT_VERIFY
  7. PREHOVER_DEST
  8. DESCEND_DEST
  9. OPEN_GRIPPER_ONLY
 10. POST_RELEASE_SETTLE
 11. POST_RELEASE_CLEARANCE
 12. RETURN_HOME
 13. FINAL_PLACEMENT_SETTLE

The stage split makes it much easier to localize failures than the
previous monolithic phase rollout. Each stage has a concrete target and
its own success condition.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import mujoco

from src.config import (
    BOARD_CENTER, SQUARE_SIZE, Z_SAFE, Z_GRASP, ACTION_SCALE, TABLE_HEIGHT,
    MAX_STEPS_PER_PHASE, SETTLE_STEPS, RETRACT_STEPS,
    PLACEMENT_TOLERANCE, GRIPPER_OPEN, GRIPPER_CLOSED,
    WHITE_GRAVEYARD_ORIGIN, BLACK_GRAVEYARD_ORIGIN,
    GRAVEYARD_SPACING, GRAVEYARD_COLS,
    REACHABLE_X_MIN, REACHABLE_X_MAX,
    REACHABLE_Y_MIN, REACHABLE_Y_MAX,
    REACHABLE_Z_MIN, REACHABLE_Z_MAX,
    FETCH_INIT_GRIP, POLICY_XY_RADIUS, POLICY_Z_MIN, POLICY_Z_MAX,
    VIEWER_STEP_DELAY_SEC, GRIPPER_ACTUATION_STEPS,
    GRIP_CONTACT_TOLERANCE, PIECE_FOLLOW_TOLERANCE, PIECE_DRIFT_TOLERANCE,
    RELEASE_SETTLE_STEPS, RELEASE_VELOCITY_TOLERANCE,
    RELEASE_GRIPPER_OPEN_TOLERANCE, RELEASE_CLEARANCE_MARGIN,
    FINAL_SETTLE_STEPS,
    PREGRASP_GRIPPER_OPENING, GRASP_DESCEND_OFFSET, CLOSE_DESCEND_OFFSET,
    CLOSE_DESCEND_STEPS,
)
from src.env.chess_pick_place_env import ChessPickPlaceEnv
from src.health_checks import CheckHook

logger = logging.getLogger(__name__)


@dataclass
class OpResult:
    """Outcome of a single pick-and-place operation."""
    success: bool
    steps_taken: int
    final_error_mm: float
    phases_completed: int
    details: str = ""


@dataclass
class OpPreflight:
    """Resolved positions and cheap validation results before arm motion."""

    actual_piece_pos: np.ndarray
    src_pos: np.ndarray
    dest_pos: np.ndarray
    stage_targets: dict[str, Optional[np.ndarray]]
    policy_issues: list[str]
    failure_result: Optional[OpResult] = None


class ExecutionController:
    """
    Executes PickPlaceOp operations using explicit stage-by-stage motion
    and gripper commands.
    """

    STAGES = [
        "HOME_RESET",
        "PREHOVER_SRC",
        "PREGRASP_NARROW",
        "DESCEND_SRC",
        "CLOSE_GRIPPER_ONLY",
        "LIFT_VERIFY",
        "PREHOVER_DEST",
        "DESCEND_DEST",
        "OPEN_GRIPPER_ONLY",
        "POST_RELEASE_SETTLE",
        "POST_RELEASE_CLEARANCE",
        "RETURN_HOME",
        "FINAL_PLACEMENT_SETTLE",
    ]
    _source_descend_offset_z = 0.03

    # Graveyard counters for grid positioning
    _reach_min = np.array([REACHABLE_X_MIN, REACHABLE_Y_MIN, REACHABLE_Z_MIN], dtype=np.float64)
    _reach_max = np.array([REACHABLE_X_MAX, REACHABLE_Y_MAX, REACHABLE_Z_MAX], dtype=np.float64)
    _policy_center = np.array(FETCH_INIT_GRIP, dtype=np.float64)
    _progress_log_interval = 10

    def __init__(self, rl_model, env: ChessPickPlaceEnv):
        """
        Args:
            rl_model: Pretrained SAC+HER model from stable-baselines3
            env: ChessPickPlaceEnv wrapping the MuJoCo chess scene
        """
        self.rl_model = rl_model
        self.env = env
        self._white_graveyard_count = 0
        self._black_graveyard_count = 0
        self._policy_center = np.array(FETCH_INIT_GRIP, dtype=np.float64)
        if getattr(env, "home_grip_pos", None) is not None:
            self._policy_center = env.home_grip_pos.copy()
        self._pregrasp_piece_rest_z = Z_GRASP
        self._placement_goal = None
        self._stage_event_handler = None

    def set_stage_event_handler(self, handler) -> None:
        """Register a callback invoked at stage boundaries."""
        self._stage_event_handler = handler

    def _emit_stage_event(self, phase: CheckHook, **payload) -> None:
        """Emit a stage lifecycle event to the registered callback."""
        if self._stage_event_handler is not None:
            self._stage_event_handler(phase, **payload)

    # ─── Position Computation ────────────────────────────────────────

    @staticmethod
    def get_square_pos(square_name):
        """
        Convert a chess square name (e.g. 'e4') to MuJoCo world coordinates.

        Returns:
            numpy array [x, y, z] at the piece center-of-mass height
        """
        file_idx = ord(square_name[0]) - ord('a')    # 0-7
        rank_idx = int(square_name[1:]) - 1           # 0-7

        start_x = BOARD_CENTER[0] - 4 * SQUARE_SIZE
        # Rank 8 is placed on the robot-facing side so the AI-controlled
        # black pieces start within the Fetch arm workspace.
        start_y = BOARD_CENTER[1] + 4 * SQUARE_SIZE

        # File a maps to high-X (human's LEFT from White's perspective)
        # and file h maps to low-X (human's RIGHT).  This follows the
        # standard chess convention for board orientation.
        x = start_x + (7 - file_idx + 0.5) * SQUARE_SIZE
        y = start_y - (rank_idx + 0.5) * SQUARE_SIZE

        return np.array([x, y, Z_GRASP])

    def get_pos(self, square_name):
        """
        Get world position for a square name or graveyard zone.

        Graveyard positions use a grid layout to prevent stacking.
        """
        if square_name == "white_graveyard":
            return self._next_graveyard_pos("white")
        if square_name == "black_graveyard":
            return self._next_graveyard_pos("black")
        return self.get_square_pos(square_name)

    def _next_graveyard_pos(self, color):
        """Allocate the next grid position in a graveyard zone."""
        if color == "white":
            idx = self._white_graveyard_count
            self._white_graveyard_count += 1
            origin = np.array(WHITE_GRAVEYARD_ORIGIN, dtype=np.float64)
        else:
            idx = self._black_graveyard_count
            self._black_graveyard_count += 1
            origin = np.array(BLACK_GRAVEYARD_ORIGIN, dtype=np.float64)

        col = idx % GRAVEYARD_COLS
        row = idx // GRAVEYARD_COLS
        offset = np.array([
            col * GRAVEYARD_SPACING,
            row * GRAVEYARD_SPACING,
            0.0
        ])
        return origin + offset

    # ─── Operation Execution ─────────────────────────────────────────

    def execute_op(self, op, viewer=None):
        """
        Execute a single PickPlaceOp using the guided-policy strategy.

        Args:
            op: PickPlaceOp describing what piece to move and where
            viewer: Optional MuJoCo viewer for visualization

        Returns:
            OpResult with success/failure and metrics
        """
        preflight = self._preflight_op(op)
        actual_piece_pos = preflight.actual_piece_pos
        src_pos = preflight.src_pos
        dest_pos = preflight.dest_pos
        stage_targets = preflight.stage_targets
        self._placement_goal = dest_pos.copy()

        if preflight.policy_issues:
            logger.warning("  OUT_OF_POLICY_WORKSPACE: " + "; ".join(preflight.policy_issues))
        if preflight.failure_result is not None:
            logger.warning(f"  {preflight.failure_result.details}")
            return preflight.failure_result

        logger.info(
            "Executing op: "
            f"piece={op.piece_name} "
            f"src_square={op.target_square} "
            f"dest_square={op.dest_square}"
        )
        logger.debug(
            "  Arm Input | "
            f"piece_current_xyz={actual_piece_pos.round(4)} | "
            f"src_xyz={src_pos.round(4)} | "
            f"dest_xyz={dest_pos.round(4)}"
        )
        logger.debug(
            "  Arm Input | "
            f"pre_hover_src_xyz={stage_targets['PREHOVER_SRC'].round(4)} | "
            f"descend_src_xyz={stage_targets['DESCEND_SRC'].round(4)} | "
            f"lift_verify_xyz={stage_targets['LIFT_VERIFY'].round(4)} | "
            f"pre_hover_dest_xyz={stage_targets['PREHOVER_DEST'].round(4)} | "
            f"descend_dest_xyz={stage_targets['DESCEND_DEST'].round(4)}"
        )
        logger.debug(
            "  Arm Input | "
            f"grip_current_xyz={self.env.get_grip_pos().round(4)} | "
            f"reach_bounds_min={self._reach_min.round(4)} | "
            f"reach_bounds_max={self._reach_max.round(4)}"
        )

        self.env.set_target(op.piece_name, src_pos)

        total_steps = 0
        stages_completed = 0
        failed_stage = None

        pick_retry_count = 0
        stage_index = 0
        while stage_index < len(self.STAGES):
            stage = self.STAGES[stage_index]
            goal = self._resolve_stage_goal(stage, stage_targets, src_pos, dest_pos)
            self._emit_stage_event(CheckHook.ARM_STAGE_START, stage=stage, op=op, goal=goal, viewer=viewer)
            logger.info(f"=== Starting Stage {stage} ===")
            logger.debug(
                "  Stage State | "
                f"target_piece={op.piece_name} | "
                f"piece_current_xyz={self.env.get_piece_pos().round(4)} | "
                f"grip_current_xyz={self.env.get_grip_pos().round(4)} | "
                f"goal_xyz={(goal.round(4) if goal is not None else 'n/a')}"
            )

            success, steps_taken = self._execute_stage(stage, op.piece_name, goal, viewer)
            self._emit_stage_event(
                CheckHook.ARM_STAGE_END,
                stage=stage,
                op=op,
                goal=goal,
                viewer=viewer,
                success=success,
                steps_taken=steps_taken,
            )
            total_steps += steps_taken

            if success:
                stages_completed += 1
                logger.debug(
                    f"  Stage {stage} SUCCESS after {steps_taken} steps. "
                    f"Grip={self.env.get_grip_pos().round(4)} | "
                    f"Piece={self.env.get_piece_pos().round(4)}"
                )
                stage_index += 1
                continue

            if stage in {"CLOSE_GRIPPER_ONLY", "LIFT_VERIFY"} and pick_retry_count < 2:
                pick_retry_count += 1
                logger.warning(
                    "  Pick acquisition failed; retrying from live piece pose | "
                    f"attempt={pick_retry_count + 1} | "
                    f"piece={self.env.get_piece_pos().round(4)}"
                )
                self.env.force_gripper_open()
                self._settle(viewer, steps=SETTLE_STEPS)
                actual_piece_pos = self.env.data.body(op.piece_name).xpos.copy()
                src_pos = actual_piece_pos.copy()
                self._pregrasp_piece_rest_z = float(actual_piece_pos[2])
                stage_targets = self._build_stage_targets(src_pos, dest_pos, actual_piece_pos)
                stages_completed = 0
                stage_index = self.STAGES.index("HOME_RESET")
                continue

            failed_stage = stage
            logger.warning(
                f"  Stage {stage} FAILED after {steps_taken} steps. "
                f"Grip={self.env.get_grip_pos().round(4)} | "
                f"Piece={self.env.get_piece_pos().round(4)}"
            )
            break

        if failed_stage is not None and failed_stage != "RETURN_HOME":
            self._retract_arm(viewer)

        # ── Evaluate final placement ──
        final_pos = self.env.data.body(op.piece_name).xpos.copy()
        final_xy_error = np.linalg.norm(final_pos[:2] - dest_pos[:2])
        final_z_error = abs(float(final_pos[2]) - float(dest_pos[2]))
        final_error_mm = final_xy_error * 1000
        stability_issue = self._placement_stability_issue(op.piece_name, dest_pos)

        success = (
            failed_stage is None
            and final_xy_error < PLACEMENT_TOLERANCE
            and final_z_error < PLACEMENT_TOLERANCE
            and stability_issue is None
        )
        status = "SUCCESS" if success else "FAILED"
        detail_suffix = (
            f"xy_error={final_xy_error * 1000:.1f}mm, "
            f"z_error={final_z_error * 1000:.1f}mm"
        )
        if stability_issue is not None:
            detail_suffix += f", stability={stability_issue}"

        result = OpResult(
            success=success,
            steps_taken=total_steps,
            final_error_mm=final_error_mm,
            phases_completed=stages_completed,
            details=f"{status}: {detail_suffix}, "
                    f"stages={stages_completed}/{len(self.STAGES)}, "
                    f"steps={total_steps}",
        )

        logger.info(f"  Result: {result.details}")
        return result

    def _preflight_op(self, op) -> OpPreflight:
        """Resolve source/destination poses and validate cheap invariants."""
        actual_piece_pos = self.env.data.body(op.piece_name).xpos.copy()
        # Pick up from the live MuJoCo pose, not the nominal square center.
        # After earlier moves the piece can be a centimeter or two off-center,
        # and targeting the stale square coordinate causes the gripper to miss.
        src_pos = actual_piece_pos.copy()
        if "graveyard" in op.target_square:
            src_pos = self.get_pos(op.target_square)
        dest_pos = self.get_pos(op.dest_square)
        self._pregrasp_piece_rest_z = float(actual_piece_pos[2])
        stage_targets = self._build_stage_targets(src_pos, dest_pos, actual_piece_pos)
        policy_issues = self._policy_compatibility_issues(src_pos, dest_pos)

        unreachable_stages = [
            stage for stage, goal in stage_targets.items()
            if goal is not None and not self._is_reachable(goal)
        ]
        failure_result = None
        if unreachable_stages:
            detail = (
                "UNREACHABLE: "
                f"stages={','.join(unreachable_stages)} "
                f"src={src_pos.round(3)} dest={dest_pos.round(3)} "
                f"reach_x=[{REACHABLE_X_MIN:.2f},{REACHABLE_X_MAX:.2f}] "
                f"reach_y=[{REACHABLE_Y_MIN:.2f},{REACHABLE_Y_MAX:.2f}] "
                f"reach_z=[{REACHABLE_Z_MIN:.2f},{REACHABLE_Z_MAX:.2f}]"
            )
            failure_result = OpResult(
                success=False,
                steps_taken=0,
                final_error_mm=float("inf"),
                phases_completed=0,
                details=detail,
            )

        return OpPreflight(
            actual_piece_pos=actual_piece_pos,
            src_pos=src_pos,
            dest_pos=dest_pos,
            stage_targets=stage_targets,
            policy_issues=policy_issues,
            failure_result=failure_result,
        )

    def _resolve_stage_goal(self, stage, stage_targets, src_pos, dest_pos):
        if stage == "PREHOVER_DEST":
            target_xy = dest_pos[:2] - self._current_carry_xy_offset()
            target_z = float(stage_targets["PREHOVER_DEST"][2])
            return np.array([target_xy[0], target_xy[1], target_z], dtype=np.float64)
        if stage == "DESCEND_SRC":
            piece_pos = self.env.get_piece_pos()
            return np.array(
                [piece_pos[0], piece_pos[1], piece_pos[2] + self._source_descend_offset_z],
                dtype=np.float64,
            )
        if stage == "DESCEND_DEST":
            carry_offset = self._current_carry_offset()
            return np.array(
                [
                    dest_pos[0] - carry_offset[0],
                    dest_pos[1] - carry_offset[1],
                    dest_pos[2] - carry_offset[2] + CLOSE_DESCEND_OFFSET,
                ],
                dtype=np.float64,
            )
        return stage_targets.get(stage)

    def _current_carry_xy_offset(self) -> np.ndarray:
        """Estimate the live XY offset between the carried piece and the grip site."""
        if not hasattr(self.env, "get_grip_pos"):
            return np.zeros(2, dtype=np.float64)
        piece_pos = self.env.get_piece_pos()
        grip_pos = self.env.get_grip_pos()
        if piece_pos[2] <= Z_GRASP + 0.004:
            return np.zeros(2, dtype=np.float64)
        return np.asarray(piece_pos[:2] - grip_pos[:2], dtype=np.float64)

    def _current_carry_offset(self) -> np.ndarray:
        """Estimate the live XYZ offset between the carried piece and grip site."""
        if not hasattr(self.env, "get_grip_pos"):
            return np.zeros(3, dtype=np.float64)
        piece_pos = self.env.get_piece_pos()
        grip_pos = self.env.get_grip_pos()
        if piece_pos[2] <= Z_GRASP + 0.004:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(piece_pos - grip_pos, dtype=np.float64)

    def _placement_stability_issue(self, piece_name, dest_pos):
        piece_body = self.env.data.body(piece_name)
        piece_pos = piece_body.xpos.copy()
        quat = piece_body.xquat.copy()
        up_z = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)

        if up_z < 0.966:
            return f"tipped(up_z={up_z:.3f})"
        if piece_pos[2] > dest_pos[2] + PLACEMENT_TOLERANCE:
            return f"elevated(z={piece_pos[2]:.3f})"
        if piece_pos[2] < dest_pos[2] - PLACEMENT_TOLERANCE:
            return f"sunk(z={piece_pos[2]:.3f})"
        return None

    @classmethod
    def _is_reachable(cls, goal):
        goal = np.asarray(goal, dtype=np.float64)
        return np.all(goal >= cls._reach_min) and np.all(goal <= cls._reach_max)

    def _policy_compatibility_issues(self, src_pos, dest_pos):
        issues = []
        grip_pos = self._policy_center

        def append_if_outside(name, pos):
            xy_dist = np.linalg.norm(np.asarray(pos[:2]) - grip_pos[:2])
            z = float(pos[2])
            if xy_dist > POLICY_XY_RADIUS:
                issues.append(
                    f"{name}_xy_dist={xy_dist:.3f}m exceeds trained radius {POLICY_XY_RADIUS:.3f}m"
                )
            if not self._is_reachable(pos):
                issues.append(
                    f"{name}_xyz={np.asarray(pos).round(3)} outside workspace bounds"
                )
            if z < POLICY_Z_MIN or z > POLICY_Z_MAX:
                issues.append(
                    f"{name}_z={z:.3f}m outside trained z-range [{POLICY_Z_MIN:.3f}, {POLICY_Z_MAX:.3f}]"
                )

        append_if_outside("src", src_pos)
        append_if_outside("dest", dest_pos)
        append_if_outside("acquire", np.array([src_pos[0], src_pos[1], Z_SAFE], dtype=np.float64))
        append_if_outside("transit", np.array([dest_pos[0], dest_pos[1], Z_SAFE], dtype=np.float64))
        return issues

    def _build_stage_targets(self, src_pos, dest_pos, live_piece_pos=None):
        piece_z = float(live_piece_pos[2]) if live_piece_pos is not None else Z_GRASP
        source_approach_z = piece_z + self._source_descend_offset_z
        dest_approach_z = piece_z + GRASP_DESCEND_OFFSET
        home_goal = np.array(FETCH_INIT_GRIP, dtype=np.float64)
        if self.env is not None and getattr(self.env, "home_grip_pos", None) is not None:
            home_goal = self.env.home_grip_pos
        clearance_z = self._release_clearance_z(dest_pos, home_goal)
        
        # Use Z_SAFE for all transit moves.
        transit_z = Z_SAFE
            
        return {
            "HOME_RESET": None,
            "PREHOVER_SRC": np.array([src_pos[0], src_pos[1], transit_z], dtype=np.float64),
            "PREGRASP_NARROW": None,
            "DESCEND_SRC": np.array([src_pos[0], src_pos[1], source_approach_z], dtype=np.float64),
            "CLOSE_GRIPPER_ONLY": None,
            "LIFT_VERIFY": np.array([src_pos[0], src_pos[1], transit_z], dtype=np.float64),
            "PREHOVER_DEST": np.array([dest_pos[0], dest_pos[1], transit_z], dtype=np.float64),
            "DESCEND_DEST": np.array([dest_pos[0], dest_pos[1], dest_approach_z], dtype=np.float64),
            "OPEN_GRIPPER_ONLY": None,
            "POST_RELEASE_SETTLE": None,
            "POST_RELEASE_CLEARANCE": np.array([dest_pos[0], dest_pos[1], clearance_z], dtype=np.float64),
            "RETURN_HOME": home_goal,
            "FINAL_PLACEMENT_SETTLE": None,
        }

    def _execute_stage(self, stage, piece_name, goal, viewer=None):
        if stage == "HOME_RESET":
            self.env.force_gripper_open()
            home_goal = self.env.home_grip_pos
            current_grip = self.env.get_grip_pos()
            steps_taken = 0
            if np.linalg.norm(current_grip - home_goal) > 0.01:
                retreat_z = self._release_clearance_z(current_grip, home_goal)
                success, steps_taken = self._move_gripper_to(
                    home_goal,
                    viewer=viewer,
                    gripper_opening=GRIPPER_OPEN,
                    transit_z=retreat_z,
                )
                if not success:
                    return False, steps_taken
            self._settle(viewer, steps=SETTLE_STEPS)
            return True, steps_taken

        if stage == "PREHOVER_SRC":
            self.env.set_target(piece_name, goal)
            return self._move_gripper_to(goal, viewer=viewer, gripper_opening=GRIPPER_OPEN, use_rl=True)

        if stage == "PREGRASP_NARROW":
            return self._set_gripper_aperture(PREGRASP_GRIPPER_OPENING, viewer=viewer)

        if stage == "DESCEND_SRC":
            self.env.set_target(piece_name, goal)
            self._pregrasp_piece_rest_z = float(self.env.get_piece_pos()[2])
            return self._move_gripper_to(
                goal,
                viewer=viewer,
                gripper_opening=PREGRASP_GRIPPER_OPENING,
                max_piece_drift=PIECE_DRIFT_TOLERANCE * 4,
                max_cartesian_action=0.25,
                rl_vertical_only=True,
                use_rl=False,
            )

        if stage == "CLOSE_GRIPPER_ONLY":
            return self._actuate_gripper(close=True, viewer=viewer, stabilize_piece=False)

        if stage == "LIFT_VERIFY":
            self.env.set_target(piece_name, goal)
            return self._move_gripper_to(
                goal,
                viewer=viewer,
                gripper_opening=GRIPPER_CLOSED,
                require_piece_follow=True,
                max_cartesian_action=0.3,
                use_rl=False,
            )

        if stage == "PREHOVER_DEST":
            self.env.set_target(piece_name, goal)
            return self._move_gripper_to(
                goal,
                viewer=viewer,
                gripper_opening=GRIPPER_CLOSED,
                require_piece_follow=True,
                use_rl=True,
            )

        if stage == "DESCEND_DEST":
            self.env.set_target(piece_name, goal)
            success, steps_taken = self._move_gripper_to(
                goal,
                viewer=viewer,
                gripper_opening=GRIPPER_CLOSED,
                require_piece_follow=True,
                max_piece_drift=PIECE_DRIFT_TOLERANCE * 3,
                max_cartesian_action=0.25,
                rl_vertical_only=True,
                use_rl=False,
            )
            piece_pos = self.env.get_piece_pos()
            
            if piece_pos[2] < TABLE_HEIGHT + 0.01:
                return False, steps_taken
                
            xy_error = np.linalg.norm(piece_pos[:2] - self._placement_goal[:2])
            z_error = abs(float(piece_pos[2]) - float(self._placement_goal[2]))
            if xy_error <= PLACEMENT_TOLERANCE and z_error <= PLACEMENT_TOLERANCE:
                return True, steps_taken
            if abs(piece_pos[2] - self._placement_goal[2]) <= 0.03 and piece_pos[2] <= self._placement_goal[2] + 0.03:
                align_goal = self.env.get_grip_pos().copy()
                align_goal[:2] += self._placement_goal[:2] - piece_pos[:2]
                align_success, align_steps = self._move_gripper_to(
                    align_goal,
                    viewer=viewer,
                    gripper_opening=GRIPPER_CLOSED,
                    require_piece_follow=False,
                    transit_z=align_goal[2],
                    max_cartesian_action=0.2,
                    use_rl=False,
                )
                steps_taken += align_steps
                piece_pos = self.env.get_piece_pos()
                xy_error = np.linalg.norm(piece_pos[:2] - self._placement_goal[:2])
                z_error = abs(float(piece_pos[2]) - float(self._placement_goal[2]))
                return bool(align_success and xy_error <= PLACEMENT_TOLERANCE and z_error <= PLACEMENT_TOLERANCE), steps_taken
            return success, steps_taken

        if stage == "OPEN_GRIPPER_ONLY":
            success, steps_taken = self._actuate_gripper(close=False, viewer=viewer)
            if success:
                self.env.set_piece_mesh_collision_enabled(piece_name, enabled=False)
            return success, steps_taken

        if stage == "POST_RELEASE_SETTLE":
            return self._settle_released_piece(goal_pos=self._placement_goal, viewer=viewer)

        if stage == "POST_RELEASE_CLEARANCE":
            return self._move_gripper_to(
                goal,
                viewer=viewer,
                gripper_opening=GRIPPER_OPEN,
                transit_z=goal[2],
                released_piece_goal=self._placement_goal,
                tolerate_released_piece_jitter=True,
            )

        if stage == "RETURN_HOME":
            return self._return_home_after_release(
                goal,
                viewer=viewer,
                released_piece_goal=self._placement_goal,
            )

        if stage == "FINAL_PLACEMENT_SETTLE":
            return self._finalize_placement(self._placement_goal, viewer=viewer)

        return False, 0

    # ─── Arm Retraction ──────────────────────────────────────────────

    def _retract_arm(self, viewer=None):
        """
        Return the arm to the canonical home area after an aborted stage.
        """
        self.env.force_gripper_open()
        home_goal = self.env.home_grip_pos
        retreat_z = self._release_clearance_z(self.env.get_grip_pos(), home_goal)
        lift_goal = np.array([self.env.get_grip_pos()[0], self.env.get_grip_pos()[1], retreat_z], dtype=np.float64)
        lift_success, lift_steps = self._move_gripper_to(
            lift_goal,
            viewer=viewer,
            gripper_opening=GRIPPER_OPEN,
            transit_z=retreat_z,
        )
        return_success, return_steps = self._move_gripper_to(
            home_goal,
            viewer=viewer,
            gripper_opening=GRIPPER_OPEN,
            transit_z=retreat_z,
        )
        success = lift_success and return_success
        steps_taken = lift_steps + return_steps
        self._settle(viewer, steps=SETTLE_STEPS)
        logger.debug(
            "  Arm retreat home | "
            f"success={success} | "
            f"steps={steps_taken} | "
            f"grip_xyz={self.env.get_grip_pos().round(4)} | "
            f"home_xyz={home_goal.round(4)}"
        )
        return success, steps_taken

    @staticmethod
    def _release_clearance_z(dest_pos, home_goal):
        dest_z = float(dest_pos[2])
        home_z = float(home_goal[2])
        return min(
            REACHABLE_Z_MAX - 0.01,
            max(Z_SAFE + RELEASE_CLEARANCE_MARGIN, dest_z + RELEASE_CLEARANCE_MARGIN, home_z + 0.02),
        )

    def _settle_released_piece(self, goal_pos, viewer=None, max_steps=RELEASE_SETTLE_STEPS):
        """
        Hold the open gripper stationary until the released piece is back on
        its support surface and no longer jittering from contact.
        """
        goal_pos = np.asarray(goal_pos, dtype=np.float64)
        goal_z = float(goal_pos[2])
        target_body_name = getattr(self.env, "target_body_name", None)
        if target_body_name and "pawn" in target_body_name:
            max_steps = max(max_steps, FINAL_SETTLE_STEPS)

        for steps_taken in range(1, max_steps + 1):
            mujoco.mj_step(self.env.model, self.env.data)
            self._sync_viewer(viewer)

            current_pos = self.env.get_piece_pos()
            xy_error = np.linalg.norm(current_pos[:2] - goal_pos[:2])
            z_error = abs(float(current_pos[2]) - goal_z)
            piece_speed = np.linalg.norm(self.env.get_piece_linear_velocity())
            piece_to_grip = np.linalg.norm(current_pos - self.env.get_grip_pos())
            if (
                xy_error <= PLACEMENT_TOLERANCE
                and z_error <= PLACEMENT_TOLERANCE
                and piece_speed <= RELEASE_VELOCITY_TOLERANCE * 3.0
                and piece_to_grip >= GRIP_CONTACT_TOLERANCE * 0.5
            ):
                return True, steps_taken

        current_pos = self.env.get_piece_pos()
        xy_error = np.linalg.norm(current_pos[:2] - goal_pos[:2])
        z_error = abs(float(current_pos[2]) - goal_z)
        piece_to_grip = np.linalg.norm(current_pos - self.env.get_grip_pos())
        if (
            xy_error <= PLACEMENT_TOLERANCE
            and z_error <= PLACEMENT_TOLERANCE
            and piece_to_grip >= GRIP_CONTACT_TOLERANCE * 0.5
        ):
            return True, max_steps

        logger.warning(
            "    Released piece did not fully separate before retreat | "
            f"piece={current_pos.round(4)} | "
            f"grip={self.env.get_grip_pos().round(4)}"
        )
        return False, max_steps

    def _return_home_after_release(
        self,
        home_goal,
        viewer=None,
        released_piece_goal=None,
    ):
        """Lift clear of the released piece, then return laterally to home."""
        home_goal = np.asarray(home_goal, dtype=np.float64)
        retreat_z = self._release_clearance_z(released_piece_goal, home_goal)
        lift_goal = np.array([self.env.get_grip_pos()[0], self.env.get_grip_pos()[1], retreat_z], dtype=np.float64)
        lift_success, lift_steps = self._move_gripper_to(
            lift_goal,
            viewer=viewer,
            gripper_opening=GRIPPER_OPEN,
            transit_z=retreat_z,
            released_piece_goal=released_piece_goal,
            tolerate_released_piece_jitter=True,
        )
        return_success, return_steps = self._move_gripper_to(
            home_goal,
            viewer=viewer,
            gripper_opening=GRIPPER_OPEN,
            transit_z=retreat_z,
            released_piece_goal=released_piece_goal,
            tolerate_released_piece_jitter=True,
        )
        return lift_success and return_success, lift_steps + return_steps

    def _move_gripper_to(
        self,
        target_pos,
        viewer=None,
        gripper_opening=GRIPPER_OPEN,
        require_piece_follow=False,
        max_piece_drift=None,
        transit_z=None,
        released_piece_goal=None,
        tolerate_released_piece_jitter=False,
        max_cartesian_action=1.0,
        rl_vertical_only=False,
        use_rl=False,
    ):
        """
        Script the gripper to a Cartesian target pose using small Fetch-style
        deltas. Optionally keep the gripper closed and verify that the piece
        follows the gripper during transport.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        steps_taken = 0
        initial_piece_pos = self.env.get_piece_pos()
        transit_z = float(Z_SAFE if transit_z is None else transit_z)
        waypoints = [
            np.array([self.env.get_grip_pos()[0], self.env.get_grip_pos()[1], transit_z], dtype=np.float64),
            np.array([target_pos[0], target_pos[1], transit_z], dtype=np.float64),
            target_pos,
        ]

        for waypoint in waypoints:
            logger.debug(
                "    Waypoint start | "
                f"target={waypoint.round(4)} | "
                f"grip={self.env.get_grip_pos().round(4)} | "
                f"piece={self.env.get_piece_pos().round(4)} | "
                f"gripper_opening={gripper_opening:.3f}"
            )
            stall_steps = 0
            best_dist = float('inf')
            
            for _ in range(RETRACT_STEPS * 3):
                grip_pos = self.env.get_grip_pos()
                delta = waypoint - grip_pos
                dist = np.linalg.norm(delta)
                if dist < 0.006:
                    logger.debug(
                        "    Waypoint reached | "
                        f"target={waypoint.round(4)} | "
                        f"grip={grip_pos.round(4)} | "
                        f"piece={self.env.get_piece_pos().round(4)} | "
                        f"steps_taken={steps_taken}"
                    )
                    break
                    
                if dist < best_dist - 0.001:
                    best_dist = dist
                    stall_steps = 0
                else:
                    stall_steps += 1
                    
                if stall_steps >= 50:
                    logger.warning(f"    Stall detected: no progress towards waypoint for 50 steps (dist={dist:.4f}m)")
                    return False, steps_taken

                action = self._compose_guided_action(
                    delta,
                    max_cartesian_action=max_cartesian_action,
                    rl_vertical_only=rl_vertical_only,
                    use_rl=use_rl,
                )
                self.env.step(
                    action,
                    viewer=viewer,
                    debug=False,
                    gripper_target=gripper_opening,
                )
                steps_taken += 1

                if steps_taken == 1 or steps_taken % self._progress_log_interval == 0:
                    piece_pos = self.env.get_piece_pos()
                    grip_pos = self.env.get_grip_pos()
                    piece_to_grip = np.linalg.norm(piece_pos - grip_pos)
                    piece_drift = np.linalg.norm(piece_pos[:2] - initial_piece_pos[:2])
                    logger.debug(
                        "    Waypoint progress | "
                        f"target={waypoint.round(4)} | "
                        f"grip={grip_pos.round(4)} | "
                        f"piece={piece_pos.round(4)} | "
                        f"remaining={np.linalg.norm(waypoint - grip_pos):.4f}m | "
                        f"piece_to_grip={piece_to_grip:.4f}m | "
                        f"piece_xy_drift={piece_drift:.4f}m | "
                        f"action_xyz={action[:3].round(3)}"
                    )

                if max_piece_drift is not None:
                    piece_drift = np.linalg.norm(self.env.get_piece_pos()[:2] - initial_piece_pos[:2])
                    if piece_drift > max_piece_drift:
                        logger.warning(
                            "    Stage drift guard triggered: "
                            f"piece_xy_drift={piece_drift:.4f}m > {max_piece_drift:.4f}m"
                        )
                        return False, steps_taken
                if released_piece_goal is not None:
                    if not tolerate_released_piece_jitter:
                        released_piece_issue = self._released_piece_issue(
                            released_piece_goal,
                            tolerate_contact_jitter=False,
                        )
                        if released_piece_issue is not None:
                            logger.warning(f"    Released piece disturbed during retreat: {released_piece_issue}")
                            return False, steps_taken

        success = self._validate_motion_stage(
            target_pos,
            require_piece_follow=require_piece_follow,
            initial_piece_pos=initial_piece_pos,
            max_piece_drift=max_piece_drift,
        )
        return success, steps_taken

    def _actuate_gripper(self, close, viewer=None, stabilize_piece=False):
        if close:
            return self._close_gripper_with_descent(viewer=viewer, stabilize_piece=stabilize_piece)
        target = 0.0 if close else GRIPPER_OPEN
        self._set_gripper_aperture(target, viewer=viewer)
        piece_to_grip = np.linalg.norm(self.env.get_piece_pos() - self.env.get_grip_pos())
        success = self._gripper_is_open() if not close else piece_to_grip < GRIP_CONTACT_TOLERANCE
        return success, GRIPPER_ACTUATION_STEPS

    def _close_gripper_with_descent(self, viewer=None, stabilize_piece=False):
        grip_pos = self.env.get_grip_pos()
        piece_pos = self.env.get_piece_pos()
        align_z = max(float(grip_pos[2]), float(piece_pos[2]) + 0.05)
        align_goal = np.array([piece_pos[0], piece_pos[1], align_z], dtype=np.float64)
        align_success, steps_taken = self._move_gripper_to(
            align_goal,
            viewer=viewer,
            gripper_opening=PREGRASP_GRIPPER_OPENING,
            transit_z=align_z,
            max_cartesian_action=0.2,
        )
        if not align_success:
            return False, steps_taken

        desired_contact_gap = GRIP_CONTACT_TOLERANCE * 0.95
        for _ in range(CLOSE_DESCEND_STEPS):
            grip_pos = self.env.get_grip_pos()
            piece_pos = self.env.get_piece_pos()
            piece_to_grip = np.linalg.norm(piece_pos - grip_pos)
            z_descent = -0.05 if piece_to_grip > desired_contact_gap else 0.0

            action = np.zeros(4, dtype=np.float64)
            action[2] = z_descent
            self.env.step(
                action,
                viewer=viewer,
                debug=False,
                gripper_target=GRIPPER_CLOSED,
            )
            steps_taken += 1

            piece_pos = self.env.get_piece_pos()
            grip_pos = self.env.get_grip_pos()
            xy_dist = np.linalg.norm(piece_pos[:2] - grip_pos[:2])
            piece_to_grip = np.linalg.norm(piece_pos - grip_pos)
            finger_qpos = self.env.get_gripper_finger_qpos()
            if steps_taken == 1 or steps_taken % self._progress_log_interval == 0:
                logger.debug(
                    "    Close progress | "
                    f"step={steps_taken}/{CLOSE_DESCEND_STEPS} | "
                    f"piece={piece_pos.round(4)} | "
                    f"grip={grip_pos.round(4)} | "
                    f"xy_dist={xy_dist:.4f}m | "
                    f"piece_to_grip={piece_to_grip:.4f}m | "
                    f"finger_qpos={finger_qpos.round(5)}"
                )
            if xy_dist < GRIP_CONTACT_TOLERANCE and piece_to_grip < desired_contact_gap:
                logger.debug(
                    "    Close precondition met | "
                    f"step={steps_taken} | "
                    f"xy_dist={xy_dist:.4f}m | "
                    f"piece_to_grip={piece_to_grip:.4f}m"
                )
                break

        self._settle(viewer, steps=SETTLE_STEPS)
        piece_pos = self.env.get_piece_pos()
        grip_pos = self.env.get_grip_pos()
        xy_dist = np.linalg.norm(piece_pos[:2] - grip_pos[:2])
        piece_to_grip = np.linalg.norm(piece_pos - grip_pos)
        finger_qpos = self.env.get_gripper_finger_qpos()
        # The grip site sits noticeably above the piece COM even in a valid
        # pinch pose, so vertical site-to-piece gap is not a reliable close
        # criterion here. Use lateral alignment plus nearly-closed fingers,
        # then let LIFT_VERIFY prove whether the piece is actually captured.
        fingers_closed = np.max(np.abs(finger_qpos)) < 0.002
        success = xy_dist < GRIP_CONTACT_TOLERANCE and piece_to_grip < desired_contact_gap and fingers_closed
        logger.debug(
            "    Close summary | "
            f"grip={grip_pos.round(4)} | "
            f"piece={piece_pos.round(4)} | "
            f"xy_dist={xy_dist:.4f}m | "
            f"piece_to_grip={piece_to_grip:.4f}m | "
            f"finger_qpos={finger_qpos.round(5)} | "
            f"fingers_closed={fingers_closed} | "
            f"success={success}"
        )
        return success, steps_taken

    def _set_gripper_aperture(self, target_opening, viewer=None):
        self.env.set_gripper_target(target_opening)
        settle_steps = GRIPPER_ACTUATION_STEPS
        if target_opening >= GRIPPER_OPEN - 1e-6:
            settle_steps = max(GRIPPER_ACTUATION_STEPS, RELEASE_SETTLE_STEPS * 3)
        self._settle(viewer, steps=settle_steps)
        logger.debug(
            "    Gripper aperture set | "
            f"target={target_opening:.4f} | "
            f"actual_qpos={self.env.get_gripper_finger_qpos().round(5)}"
        )
        return True, settle_steps

    def _gripper_is_open(self):
        """Return whether both finger joints are near the configured open pose."""
        return bool(np.min(self.env.get_gripper_finger_qpos()) >= GRIPPER_OPEN - RELEASE_GRIPPER_OPEN_TOLERANCE)

    def _released_piece_issue(self, goal_pos, tolerate_contact_jitter=False):
        """Detect whether the released piece is being disturbed during retreat."""
        goal_pos = np.asarray(goal_pos, dtype=np.float64)
        piece_pos = self.env.get_piece_pos()
        xy_error = np.linalg.norm(piece_pos[:2] - goal_pos[:2])
        z_error = abs(float(piece_pos[2]) - float(goal_pos[2]))
        piece_speed = np.linalg.norm(self.env.get_piece_linear_velocity())
        xy_limit = PLACEMENT_TOLERANCE * (2.0 if tolerate_contact_jitter else 1.0)
        z_limit = PLACEMENT_TOLERANCE * (2.0 if tolerate_contact_jitter else 1.0)
        speed_limit = RELEASE_VELOCITY_TOLERANCE * (5.0 if tolerate_contact_jitter else 3.0)
        if xy_error > xy_limit:
            return f"xy_error={xy_error * 1000:.1f}mm"
        if z_error > z_limit:
            return f"z_error={z_error * 1000:.1f}mm"
        if not tolerate_contact_jitter and piece_speed > speed_limit:
            return f"piece_speed={piece_speed:.4f}m/s"
        return None

    def _finalize_placement(self, goal_pos, viewer=None, max_steps=FINAL_SETTLE_STEPS):
        """Let gravity complete the placement after the gripper has cleared."""
        goal_pos = np.asarray(goal_pos, dtype=np.float64)
        steps_taken = 0
        for steps_taken in range(1, max_steps + 1):
            mujoco.mj_step(self.env.model, self.env.data)
            self._sync_viewer(viewer)
            piece_pos = self.env.get_piece_pos()
            xy_error = np.linalg.norm(piece_pos[:2] - goal_pos[:2])
            z_error = abs(float(piece_pos[2]) - float(goal_pos[2]))
            piece_speed = np.linalg.norm(self.env.get_piece_linear_velocity())
            stability_issue = self._placement_stability_issue(self.env.target_body_name, goal_pos)
            if (
                xy_error <= PLACEMENT_TOLERANCE
                and z_error <= PLACEMENT_TOLERANCE
                and piece_speed <= RELEASE_VELOCITY_TOLERANCE
                and stability_issue is None
            ):
                return True, steps_taken
        piece_pos = self.env.get_piece_pos()
        xy_error = np.linalg.norm(piece_pos[:2] - goal_pos[:2])
        z_error = abs(float(piece_pos[2]) - float(goal_pos[2]))
        piece_speed = np.linalg.norm(self.env.get_piece_linear_velocity())
        stability_issue = self._placement_stability_issue(self.env.target_body_name, goal_pos)
        success = (
            xy_error <= PLACEMENT_TOLERANCE
            and z_error <= PLACEMENT_TOLERANCE
            and stability_issue is None
        )
        if not success:
            logger.warning(
                "    Final placement settle incomplete | "
                f"piece={piece_pos.round(4)} | "
                f"goal={goal_pos.round(4)} | "
                f"xy_error={xy_error * 1000:.1f}mm | "
                f"z_error={z_error * 1000:.1f}mm | "
                f"piece_speed={piece_speed:.4f}m/s | "
                f"stability_issue={stability_issue or 'none'}"
            )
        return success, steps_taken

    def _compose_guided_action(
        self,
        delta,
        max_cartesian_action=1.0,
        rl_vertical_only=False,
        use_rl=False,
    ):
        """
        Blend waypoint guidance with deterministic RL inference.

        The scripted term guarantees progress toward the stage target. The RL
        term is only mixed in when a policy is available and its proposal is
        aligned with the current waypoint direction.
        """
        delta = np.asarray(delta, dtype=np.float64)
        scripted = np.zeros(4, dtype=np.float64)
        scripted[:3] = np.clip(delta / ACTION_SCALE, -max_cartesian_action, max_cartesian_action)
        scripted[3] = 0.0
        if not use_rl or self.rl_model is None:
            return scripted

        try:
            obs = self.env.get_obs()
            rl_action, _ = self.rl_model.predict(obs, deterministic=True)
            rl_action = np.asarray(rl_action, dtype=np.float64)
        except Exception:
            return scripted

        if rl_action.shape != (4,) or np.linalg.norm(delta) < 0.02:
            return scripted

        alignment = float(np.dot(rl_action[:3], scripted[:3]))
        if alignment <= 0.0:
            return scripted

        blended = scripted.copy()
        blended[:3] = np.clip(
            0.7 * scripted[:3] + 0.3 * rl_action[:3],
            -max_cartesian_action,
            max_cartesian_action,
        )
        if rl_vertical_only:
            blended[:2] = scripted[:2]
        blended[3] = 0.0
        return blended

    @staticmethod
    def _sync_viewer(viewer=None):
        """Sync the viewer if present, otherwise behave as a no-op."""
        if viewer is None:
            return
        viewer.sync()
        if VIEWER_STEP_DELAY_SEC > 0.0:
            time.sleep(VIEWER_STEP_DELAY_SEC)

    def _validate_motion_stage(self, target_pos, require_piece_follow, initial_piece_pos=None, max_piece_drift=None):
        grip_pos = self.env.get_grip_pos()
        piece_pos = self.env.get_piece_pos()
        grip_ok = np.linalg.norm(grip_pos - target_pos) < 0.02
        drift_ok = True
        piece_drift = None
        if initial_piece_pos is not None and max_piece_drift is not None:
            piece_drift = np.linalg.norm(piece_pos[:2] - initial_piece_pos[:2])
            drift_ok = piece_drift <= max_piece_drift
        if not require_piece_follow:
            logger.debug(
                "    Stage validation | "
                f"target={target_pos.round(4)} | "
                f"grip={grip_pos.round(4)} | "
                f"piece={piece_pos.round(4)} | "
                f"grip_ok={grip_ok} | "
                f"drift_ok={drift_ok} | "
                f"piece_xy_drift={(piece_drift if piece_drift is not None else 'N/A')}"
            )
            return grip_ok and drift_ok

        piece_xy_to_grip = np.linalg.norm(piece_pos[:2] - grip_pos[:2])
        piece_z_to_grip = abs(float(piece_pos[2] - grip_pos[2]))
        piece_lifted = piece_pos[2] > Z_GRASP + 0.004
        piece_supported_at_goal = False
        if self._placement_goal is not None:
            piece_supported_at_goal = (
                np.linalg.norm(piece_pos[:2] - self._placement_goal[:2]) <= PLACEMENT_TOLERANCE
                and abs(float(piece_pos[2] - self._placement_goal[2])) <= PLACEMENT_TOLERANCE
            )
        success = (
            grip_ok
            and drift_ok
            and piece_xy_to_grip < PIECE_FOLLOW_TOLERANCE
            and piece_z_to_grip < 0.10
            and (piece_lifted or piece_supported_at_goal)
        )
        logger.debug(
            "    Stage validation | "
            f"target={target_pos.round(4)} | "
            f"grip={grip_pos.round(4)} | "
            f"piece={piece_pos.round(4)} | "
            f"grip_ok={grip_ok} | "
            f"drift_ok={drift_ok} | "
            f"piece_xy_to_grip={piece_xy_to_grip:.4f}m | "
            f"piece_z_to_grip={piece_z_to_grip:.4f}m | "
            f"piece_lifted={piece_lifted} | "
            f"piece_supported_at_goal={piece_supported_at_goal} | "
            f"piece_xy_drift={(piece_drift if piece_drift is not None else 'N/A')} | "
            f"success={success}"
        )
        return success

    def _settle(self, viewer=None, steps=SETTLE_STEPS):
        for _ in range(steps):
            mujoco.mj_step(self.env.model, self.env.data)
            if viewer:
                viewer.sync()
                if VIEWER_STEP_DELAY_SEC > 0.0:
                    time.sleep(VIEWER_STEP_DELAY_SEC)
