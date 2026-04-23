# 00 — Project Overview

## Purpose

RoboChess is a simulation in which a **Fetch robotic arm** physically plays chess against a human inside a **MuJoCo** physics environment. The human plays White using a Tkinter graphical interface; the AI plays Black, choosing moves with **Stockfish** and executing them physically by driving the robot arm with a pretrained **Reinforcement Learning (RL) policy**.

Unlike ordinary chess software, every piece has mass, friction, and inertia. The robot must grasp, lift, carry, and place each piece precisely. Poor timing, an over-aggressive descent, or an unstable grasp will cause pieces to tip, slide, or fall — all of which are physics failures, not logic failures.

---

## Key Concepts

### MuJoCo
MuJoCo (Multi-Joint dynamics with Contact) is a high-fidelity rigid-body physics engine used for robotics research. It is not a game engine; it solves continuous-time dynamics equations at each timestep. In RoboChess it runs with a **1ms timestep** and **40 substeps per controller step**, giving the simulation fine-grained physical accuracy.

**Visual vs. physical geometry**: The chess pieces you see (knight figures, bishops, etc.) are decorative meshes. The physics engine operates on simplified **box hitboxes** for board interaction and collision masks to control what collides with what. Box hitboxes are used instead of cylinders because cylinders can roll on contact — the SAC policy was fine-tuned from a `FetchPickAndPlace-v4` cube-trained model and expects non-rolling contact surfaces.

### The Fetch Robot
Fetch is a mobile manipulation robot. In this simulation only the arm and gripper are used. The arm is controlled via a **mocap body** (a tracked Cartesian anchor the joints follow) rather than direct joint commands. The gripper has two fingers driven by position actuators.

### The RL Policy
A fine-tuned **SAC** (Soft Actor-Critic) model is used for arm control. It was fine-tuned from `sac-FetchPickAndPlace-v4` (a HuggingFace pretrained model) on a larger chess-specific board. It takes a **26-dimensional observation vector** and outputs a **4-dimensional action** `[dx, dy, dz, gripper]`. In RoboChess the policy is the **primary driver of motion** for all major transits. The `ExecutionController` acts as a high-level stage manager. During key transit stages (like moving to the source piece or moving to the destination square), the RL policy is given 100% control to reach the target waypoint naturally. Scripted Cartesian motions are reserved only for precise, small-scale maneuvers (like the final 3cm vertical descent to grasp a piece) where RL policies often struggle with millimeter-perfect rigid contacts.

**Local model**: The model (`chess_fetch_20260421_095337.zip`) is loaded from a local file path configured in `runtime.yaml` via `SAC.load(path, env=env)` from `stable_baselines3`. No `sb3_contrib` dependency is needed.

### Stockfish
Stockfish is a UCI chess engine. It is used only for **move selection** — it plays Black. It does not control the robot. The time limit per move is configurable via `stockfish_time_limit` in `runtime.yaml`.

---

## Overall Architecture (Six Layers)

```
┌──────────────────────────────────────────────────┐
│  1. Entrypoint & Runtime (main.py, app.py,        │
│     bootstrap.py)                                  │
├──────────────────────────────────────────────────┤
│  2. GUI (gui/board_panel.py)                      │
├──────────────────────────────────────────────────┤
│  3. Game Orchestration (game_loop.py,             │
│     game_runtime.py)                              │
├──────────────────────────────────────────────────┤
│  4. Chess Logic (logic/chess_manager.py,          │
│     logic/operation_planner.py)                   │
├──────────────────────────────────────────────────┤
│  5. Physical Execution (control/execution_        │
│     controller.py, env/chess_pick_place_env.py)   │
├──────────────────────────────────────────────────┤
│  6. Health Checks & Guards (health_checks.py,     │
│     runtime_guard.py, observation/)               │
└──────────────────────────────────────────────────┘
```

Each layer depends only on the layers below it; the GUI never calls the physics engine directly.

---

## Main Components

| Component | File | Role |
|-----------|------|------|
| `RoboChessApp` | `src/app.py` | Main app; owns threading; wires GUI ↔ game worker |
| `SystemBootstrapper` | `src/bootstrap.py` | Loads scene, model, wires all subsystems into `GameSystems` |
| `GameLoop` | `src/game_loop.py` | Validates moves, delegates to `GameOrchestrator`, pushes to board |
| `GameOrchestrator` | `src/game_runtime.py` | Decides arm vs teleport; dispatches `PickPlaceOp` list |
| `ExecutionController` | `src/control/execution_controller.py` | 10-stage robotic pick-and-place |
| `ChessPickPlaceEnv` | `src/env/chess_pick_place_env.py` | Gym wrapper around MuJoCo; exposes obs, step, gripper API |
| `ObservationReconstructor` | `src/env/obs_wrapper.py` | Builds the 26D obs vector for the RL policy |
| `ChessGameManager` | `src/logic/chess_manager.py` | Owns the chess board and Stockfish engine |
| `OperationPlanner` | `src/logic/operation_planner.py` | Converts a `chess.Move` into `PickPlaceOp` list |
| `BoardPanel` | `src/gui/board_panel.py` | Tkinter board UI; handles clicks and renders state |
| `RuntimeGuard` | `src/runtime_guard.py` | Validates physical vs logical board agreement |
| `BoardObserver` | `src/observation/board_observer.py` | Detects tipped, buried, or displaced pieces |
| `RuntimeCheckRegistry` | `src/health_checks.py` | Runs pluggable checks at named lifecycle hooks |
| `ConfigManager` | `src/config.py` | Loads and validates `runtime.yaml`; exports constants |

---

## Runtime Flow

### Startup Sequence

1. `main.py` calls `src.app.main()`.
2. `RoboChessApp.run()` calls `SystemBootstrapper.bootstrap_game_systems()`:
   a. Load `chess_world.xml` into MuJoCo → `(MjModel, MjData)`.
   b. Create `ChessPickPlaceEnv` (settle physics for 200 steps, snapshot home pose).
   c. Load RL model from `LOCAL_MODEL_PATH` using `SAC.load(...)`.
   d. Build `square_to_piece` mapping by scanning piece body positions.
   e. Validate 32 pieces found; run `POST_SCENE_LOAD` and `PROGRAM_START` health checks.
   f. Wire stage event handler so arm stages trigger health checks.
3. `GameLoop` is created from `GameSystems`.
4. Tkinter root window is created; `BoardPanel` is attached.
5. A daemon thread starts running `game_worker`.

### Turn Loop

```
game_worker thread (daemon)
│
├─ Open MuJoCo passive viewer
│
└─ while game not over:
    │
    ├─ [White turn] Block on move_queue.get(timeout=0.1)
    │   ├─ UCI string received → GameLoop.submit_move(uci)
    │   │   ├─ Validate legal move (ChessGameManager)
    │   │   ├─ OperationPlanner.generate_operations()
    │   │   ├─ GameOrchestrator.execute_ops()  ← arm moves
    │   │   └─ ChessGameManager.push_move()
    │   └─ None received (step-mode trigger) → execute_ai_turn()
    │
    ├─ [Black turn] GameLoop.execute_ai_turn()
    │   ├─ ChessGameManager.get_ai_move()  ← Stockfish
    │   ├─ OperationPlanner.generate_operations()
    │   ├─ GameOrchestrator.execute_ops()  ← arm moves
    │   └─ ChessGameManager.push_move()
    │
    └─ root.after(0, panel.refresh)  ← GUI update (thread-safe)
```

### Physical Execution of One Move

When `GameOrchestrator.execute_ops()` runs a single `PickPlaceOp`:

1. If the operation is a capture, the arm physically picks up the captured piece and moves it to the graveyard.
2. If the operation is a normal move or promotion, the arm physically picks and places the piece(s).
3. `ExecutionController.execute_op()` executes all movements:
   - Preflight: resolve positions, compute stage targets.
   - Execute 10 stages in order (see `01_high_level_design.md`).
   - Each stage emits `ARM_STAGE_START` / `ARM_STAGE_END` health check hooks.
   - On pick failure (CLOSE_GRIPPER_ONLY or LIFT_VERIFY) → retry up to 2 times from PREHOVER_SRC.
   - Evaluate final placement: XY error, Z error, tilt check.
   - Return `OpResult`.

### Threading Safety

- The GUI (Tkinter) runs on the **main thread**. All GUI updates from the worker thread go through `root.after(0, callback)`.
- The move queue (`queue.Queue`) carries UCI strings from the GUI to the worker.
- `stop_event` (`threading.Event`) signals shutdown.
- `step_mode_event` controls whether Black auto-plays or waits for a manual trigger.

---

## Physics Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| `physics_timestep` | 1ms | Fine-grained accuracy for contact resolution |
| Substeps | 40 per action | Stability for high-gain robot control |
| Integrator | Euler | Stable for stiff robot joints |
| `noslip_iterations` | 100 | Prevents piece drift after placement |
| Piece mass | 0.25 kg | Heavy enough to resist robot-induced turbulence |
| Freejoint damping | 0.05 N·s/m | Uniform, physically realistic friction/mass keeps pieces stable |
| Piece mesh collision | enabled | Fully rigid body physics at all times |
| Piece hitbox | box per piece | Cannot roll on contact; matches cube policy training distribution |

---

## Environment Geometry

| Parameter | Value | Notes |
|-----------|-------|-------|
| Robot base | X=0.2869, Y=0.2641, Z=0 | Fetch arm anchor in world frame |
| Home slide pose | X=0.0, Y=0.0 (relative) | Initial joint values |
| Board center | X=0.88, Y=0.2641, Z=0.4 | Centred on robot Y-axis |
| Board size | 0.64m × 0.64m | Half-extents 0.32 |
| Square size | 0.08m | 0.64 / 8 |
| Board near edge | X=0.56 | 3cm gap from robot front |
| Board far edge | X=1.20 | Within arm reach |

## Local Model Note

The RL policy (`chess_fetch_20260421_095337.zip`) is loaded from `local_model_path` in `runtime.yaml` using `SAC.load()`. Fine-tuned from `sac-FetchPickAndPlace-v4`. The workspace check (`_policy_compatibility_issues`) should be treated as informational. If a target exceeds hard reachability bounds (`REACHABLE_X_MIN/MAX`, `REACHABLE_Y_MIN/MAX`, `REACHABLE_Z_MIN/MAX`), the operation fails.
