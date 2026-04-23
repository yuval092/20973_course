# 03 — Class Hierarchy and Structure

## Full Class Hierarchy

```
object
├── Exception
│   └── RoboChessError                          (src/exceptions.py)
│       ├── RuntimeCheckError
│       │   ├── ArmStateError
│       │   ├── MappingIntegrityError
│       │   ├── NumericalStabilityError
│       │   └── SceneIntegrityError
│       ├── AIEngineError
│       ├── RLModelError
│       ├── SceneLoadError
│       ├── ExecutionError
│       ├── StabilityError
│       ├── BoardStateError
│       ├── PieceLookupError
│       └── PolicyWorkspaceError
│
├── str + Enum
│   └── CheckHook                               (src/health_checks.py)
│
├── gym.Env
│   └── ChessPickPlaceEnv                       (src/env/chess_pick_place_env.py)
│
├── tk.Frame
│   └── BoardPanel                              (src/gui/board_panel.py)
│
├── Data classes (plain Python, no inheritance)
│   ├── GameSystems                             (src/models.py)
│   ├── MoveResult                              (src/models.py)
│   ├── GameState                               (src/models.py)
│   ├── PickPlaceOp                             (src/logic/operation_planner.py)
│   ├── OpResult                                (src/control/execution_controller.py)
│   └── OpPreflight                             (src/control/execution_controller.py)
│
├── Service classes (plain Python, no inheritance)
│   ├── ConfigManager                           (src/config.py)
│   ├── EventLogger                             (src/logging_utils.py)
│   ├── ChessGameManager                        (src/logic/chess_manager.py)
│   ├── OperationPlanner                        (src/logic/operation_planner.py)
│   ├── ObservationReconstructor                (src/env/obs_wrapper.py)
│   ├── BoardObserver                           (src/observation/board_observer.py)
│   ├── ExecutionController                     (src/control/execution_controller.py)
│   ├── RuntimeGuard                            (src/runtime_guard.py)
│   ├── SystemBootstrapper                      (src/bootstrap.py)
│   ├── GameOrchestrator                        (src/game_runtime.py)
│   ├── GameLoop                                (src/game_loop.py)
│   ├── RoboChessApp                            (src/app.py)
│   └── RuntimeCheckRegistry                    (src/health_checks.py)
│
└── RuntimeCheck (abstract base)                (src/health_checks.py)
    ├── SceneAssetsCheck
    ├── MappingIntegrityCheck
    ├── BoardAgreementCheck
    ├── ArmHomePoseCheck
    ├── RobotWorkspaceCheck
    ├── FiniteStateCheck
    ├── PieceObserverCheck
    ├── StageGoalReachabilityCheck
    ├── StageOutcomeCheck
    └── StageGripAttachmentCheck
```

---

## Exception Hierarchy (Mermaid)

```mermaid
classDiagram
    class RoboChessError
    class RuntimeCheckError
    class ArmStateError
    class MappingIntegrityError
    class NumericalStabilityError
    class SceneIntegrityError
    class AIEngineError
    class RLModelError
    class SceneLoadError
    class ExecutionError
    class StabilityError
    class BoardStateError
    class PieceLookupError
    class PolicyWorkspaceError

    RoboChessError <|-- RuntimeCheckError
    RoboChessError <|-- AIEngineError
    RoboChessError <|-- RLModelError
    RoboChessError <|-- SceneLoadError
    RoboChessError <|-- ExecutionError
    RoboChessError <|-- StabilityError
    RoboChessError <|-- BoardStateError
    RoboChessError <|-- PieceLookupError
    RoboChessError <|-- PolicyWorkspaceError
    RuntimeCheckError <|-- ArmStateError
    RuntimeCheckError <|-- MappingIntegrityError
    RuntimeCheckError <|-- NumericalStabilityError
    RuntimeCheckError <|-- SceneIntegrityError
```

---

## Data Model Class Diagram (Mermaid)

```mermaid
classDiagram
    class GameSystems {
        +MjModel mj_model
        +MjData mj_data
        +ChessPickPlaceEnv env
        +SAC rl_model
        +ChessGameManager manager
        +OperationPlanner planner
        +ExecutionController controller
        +dict square_to_piece
        +ndarray arm_home_grip
        +RuntimeCheckRegistry check_registry
        +dict captured_count
    }

    class MoveResult {
        +bool success
        +str message
        +Exception error
    }

    class GameState {
        +Board board
        +bool is_game_over
        +bool is_check
        +bool is_checkmate
        +bool is_stalemate
        +bool is_insufficient_material
        +bool is_fifty_moves
        +bool is_threefold
    }

    class PickPlaceOp {
        +str target_square
        +str dest_square
        +str piece_name
        +bool is_capture
        +int promotion
    }

    class OpResult {
        +bool success
        +int steps_taken
        +float final_error_mm
        +int phases_completed
        +str details
    }

    class OpPreflight {
        +ndarray actual_piece_pos
        +ndarray src_pos
        +ndarray dest_pos
        +dict stage_targets
        +list policy_issues
        +OpResult failure_result
    }
```

---

## Module Dependency Graph (Mermaid)

```mermaid
graph TD
    main["main.py"] --> app["src/app.py"]
    app --> bootstrap["src/bootstrap.py"]
    app --> game_loop["src/game_loop.py"]
    app --> board_panel["src/gui/board_panel.py"]
    app --> game_runtime["src/game_runtime.py"]
    app --> runtime_guard["src/runtime_guard.py"]

    bootstrap --> chess_env["src/env/chess_pick_place_env.py"]
    bootstrap --> exec_ctrl["src/control/execution_controller.py"]
    bootstrap --> chess_mgr["src/logic/chess_manager.py"]
    bootstrap --> op_planner["src/logic/operation_planner.py"]
    bootstrap --> models["src/models.py"]
    bootstrap --> health_checks["src/health_checks.py"]
    bootstrap --> config["src/config.py"]

    game_loop --> game_runtime
    game_loop --> chess_mgr
    game_loop --> op_planner
    game_loop --> models

    game_runtime --> exec_ctrl
    game_runtime --> runtime_guard
    game_runtime --> config

    exec_ctrl --> chess_env
    exec_ctrl --> health_checks
    exec_ctrl --> config

    chess_env --> obs_wrapper["src/env/obs_wrapper.py"]
    chess_env --> config

    health_checks --> board_observer["src/observation/board_observer.py"]
    health_checks --> runtime_guard
    health_checks --> config

    runtime_guard --> board_observer
    runtime_guard --> config

    board_panel --> game_loop

    chess_mgr --> config
    op_planner --> config
    board_observer --> config

    config --> logging_utils["src/logging_utils.py"]
    bootstrap --> logging_utils
    health_checks --> logging_utils
    exec_ctrl --> logging_utils

    exceptions["src/exceptions.py"] -.-> config
    exceptions -.-> chess_mgr
    exceptions -.-> exec_ctrl
    exceptions -.-> health_checks
    exceptions -.-> runtime_guard
    exceptions -.-> game_runtime
    exceptions -.-> bootstrap
```

---

## Execution Stage State Machine (Mermaid)

```mermaid
stateDiagram-v2
    [*] --> HOME_RESET

    HOME_RESET --> PREHOVER_SRC : success
    HOME_RESET --> FAILED : failure

    PREHOVER_SRC --> DESCEND_SRC : success
    PREHOVER_SRC --> FAILED : failure

    note right of DESCEND_SRC
        Calls _set_gripper_aperture(PREGRASP)
        at entry (absorbs old PREGRASP_NARROW stage)
    end note

    DESCEND_SRC --> CLOSE_GRIPPER_ONLY : success
    DESCEND_SRC --> FAILED : failure

    CLOSE_GRIPPER_ONLY --> LIFT_VERIFY : success
    CLOSE_GRIPPER_ONLY --> PREHOVER_SRC : pick_retry < 2
    CLOSE_GRIPPER_ONLY --> FAILED : max_retries

    LIFT_VERIFY --> PREHOVER_DEST : success
    LIFT_VERIFY --> PREHOVER_SRC : pick_retry < 2
    LIFT_VERIFY --> FAILED : max_retries

    PREHOVER_DEST --> DESCEND_DEST : success
    PREHOVER_DEST --> FAILED : failure

    DESCEND_DEST --> OPEN_GRIPPER_ONLY : success
    DESCEND_DEST --> FAILED : failure

    OPEN_GRIPPER_ONLY --> RELEASE_AND_CLEAR : success
    OPEN_GRIPPER_ONLY --> FAILED : failure

    note right of RELEASE_AND_CLEAR
        Combined settle + clearance:
        raise arm while piece settles
    end note

    RELEASE_AND_CLEAR --> SETTLE_AND_HOME : success
    RELEASE_AND_CLEAR --> FAILED : failure

    note right of SETTLE_AND_HOME
        Combined return-home + final verify:
        interleave settle steps during transit
    end note

    SETTLE_AND_HOME --> [*] : OpResult returned

    FAILED --> RETRACT_ARM : if not SETTLE_AND_HOME
    RETRACT_ARM --> [*] : OpResult(success=False)
```

**Retry logic**: `CLOSE_GRIPPER_ONLY` and `LIFT_VERIFY` failures loop back to `PREHOVER_SRC` (not HOME_RESET) up to 2 times. The arm re-hovers above the source piece and retries the descent without travelling all the way home. On retry, the actual piece position is re-read and stage targets are rebuilt.

---

## CheckHook Dispatch Table

```mermaid
classDiagram
    class RuntimeCheckRegistry {
        +run(hook, context)
    }

    class SceneAssetsCheck {
        hooks: POST_SCENE_LOAD
    }
    class MappingIntegrityCheck {
        hooks: PROGRAM_START, TURN_START, TURN_END
    }
    class BoardAgreementCheck {
        hooks: PROGRAM_START, TURN_START, TURN_END
    }
    class ArmHomePoseCheck {
        hooks: PROGRAM_START, TURN_START, TURN_END
    }
    class RobotWorkspaceCheck {
        hooks: PROGRAM_START, TURN_START, TURN_END
    }
    class FiniteStateCheck {
        hooks: POST_SCENE_LOAD, PROGRAM_START, TURN_START, TURN_END, ARM_STAGE_START, ARM_STAGE_END
    }
    class PieceObserverCheck {
        hooks: PROGRAM_START, TURN_START, TURN_END
    }
    class StageGoalReachabilityCheck {
        hooks: ARM_STAGE_START
    }
    class StageOutcomeCheck {
        hooks: ARM_STAGE_END
    }
    class StageGripAttachmentCheck {
        hooks: ARM_STAGE_END
    }

    RuntimeCheckRegistry --> SceneAssetsCheck
    RuntimeCheckRegistry --> MappingIntegrityCheck
    RuntimeCheckRegistry --> BoardAgreementCheck
    RuntimeCheckRegistry --> ArmHomePoseCheck
    RuntimeCheckRegistry --> RobotWorkspaceCheck
    RuntimeCheckRegistry --> FiniteStateCheck
    RuntimeCheckRegistry --> PieceObserverCheck
    RuntimeCheckRegistry --> StageGoalReachabilityCheck
    RuntimeCheckRegistry --> StageOutcomeCheck
    RuntimeCheckRegistry --> StageGripAttachmentCheck
```

---

## `GameSystems` Component Wiring (Mermaid)

```mermaid
classDiagram
    class GameSystems {
        mj_model, mj_data
        env: ChessPickPlaceEnv
        rl_model: SAC
        manager: ChessGameManager
        planner: OperationPlanner
        controller: ExecutionController
        square_to_piece: dict
        arm_home_grip: ndarray
        check_registry: RuntimeCheckRegistry
        captured_count: dict
    }

    GameSystems --> ChessPickPlaceEnv : env
    GameSystems --> ChessGameManager : manager
    GameSystems --> OperationPlanner : planner
    GameSystems --> ExecutionController : controller
    GameSystems --> RuntimeCheckRegistry : check_registry

    ExecutionController --> ChessPickPlaceEnv : uses env
    ChessPickPlaceEnv --> ObservationReconstructor : uses
    RuntimeCheckRegistry --> BoardObserver : uses
    RuntimeCheckRegistry --> RuntimeGuard : uses
```

---

## Observation Vector Layout

```
Index  Dim  Feature
─────  ───  ───────────────────────────────────────────────────────
 0–2    3   Gripper site XYZ (world frame)
 3–5    3   Target piece site XYZ (world frame)
 6–8    3   Piece position relative to gripper (piece_pos - grip_pos)
 9–10   2   Gripper finger joint positions [l, r]
11–13   3   Piece orientation (Euler angles from rotation matrix)
14–16   3   Piece linear velocity relative to gripper (scaled by dt)
17–19   3   Piece angular velocity (scaled by dt)
20–22   3   Gripper site linear velocity (scaled by dt)
23–24   2   Gripper finger joint velocities (scaled by dt)
   25   1   Remaining time feature in [0, 1]
─────  ───
Total: 26 dimensions, dtype float32
```

`dt = n_substeps × model.opt.timestep`

---

## Action Space

The policy outputs a 4D action vector: `[dx, dy, dz, gripper]`.

| Dim | Meaning | Range | Applied as |
|-----|---------|-------|-----------|
| 0 | X displacement | [-1, 1] | `dx × ACTION_SCALE` added to mocap_pos |
| 1 | Y displacement | [-1, 1] | `dy × ACTION_SCALE` added to mocap_pos |
| 2 | Z displacement | [-1, 1] | `dz × ACTION_SCALE` added to mocap_pos |
| 3 | Gripper | [-1, 1] | Mapped to finger actuator targets |

The gripper dim is unused in the direct `step()` path when `gripper_target` is specified explicitly by the controller.

---

## `square_to_piece` Mapping Lifecycle

```
Bootstrap
  └─ initialize_square_to_piece()
       └─ Scans all bodies, computes file/rank from XY position
       └─ Populates square_to_piece = {"e2": "w_pawn_5", ...}

Per PickPlaceOp
  ├─ Before execution: square_to_piece read for piece lookup
  └─ After execution: _update_square_mapping()
       ├─ Graveyard → pop target_square entry
       └─ Normal move → move entry from target_square to dest_square

Per Promotion
  └─ square_to_piece[dest_square] = spare_piece_name

Health checks at TURN_START / TURN_END
  └─ MappingIntegrityCheck: no duplicates, count matches board
  └─ BoardAgreementCheck: positions match expected positions
  └─ ArmHomePoseCheck: arm at home before and after each turn
  └─ RobotWorkspaceCheck: gripper within reachability bounds
  └─ FiniteStateCheck: no NaN/Inf in simulation state
  └─ PieceObserverCheck: pieces upright, not buried, not displaced
```
