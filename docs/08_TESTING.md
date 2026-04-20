# 08 TESTING: Verifying Physics and Logic

RoboChess includes a comprehensive suite of tests to ensure that code refactoring doesn't break the delicate balance of physics and game logic.

## 🧪 Running the Tests

To run the entire test suite, use `pytest` from the project root:
```bash
PYTHONPATH=. pytest tests/
```

## 📂 Test Categories

### 1. Logic Tests (`tests/test_chess_manager.py`, `tests/test_operation_planner.py`)
- **Focus**: Pure Python logic without physics.
- **Checks**: "Is this move legal?", "Does a capture correctly generate two sub-operations?", "Does castling move both pieces?"

### 2. Physical Unit Tests (`tests/test_execution_controller.py`, `tests/test_obs_wrapper.py`)
- **Focus**: Math and coordinate transformations.
- **Checks**: "Are the stage targets calculated correctly?", "Does the observation vector have the correct 26 dimensions?", "Is the target reachable?"

### 3. Integration Tests (`tests/test_headless_gameplay.py`)
- **Focus**: End-to-end moves in a "Headless" (no GUI) MuJoCo environment.
- **Checks**: "Does the robot successfully move a piece from E2 to E4?", "Does the board state update correctly after a robotic move?"

### 4. Stability Tests (`tests/test_drift_and_stability.py`)
- **Focus**: Physics "edge cases."
- **Checks**: "Does the arm knock over pieces while passing by?", "Does the simulation stay stable over multiple turns?"

## 🧱 Mocking and Simulation
Because running a full physics step is slow, some tests use **Mocks**:
- **`DummyEnv`**: A fake environment that returns "Success" instantly so we can test logical flow without waiting for the robot to move. 
- **`DummyBody`**: A fake MuJoCo object used to test geometric math.

**Limitations of Mocks**: Mocks do not simulate bitwise collision overlaps (`contype` & `conaffinity`) or solver limits. Therefore, passing a logic test does not guarantee a piece won't mathematically explode in the actual MuJoCo solver. This is why full end-to-end headless integration tests are mandatory before merging changes to `generate_xml.py` or `fetch.xml`.

## 🛡️ The "Sweet Spot" in Testing
As documented in `02_ROBOT_ARM.md`, the RL policy is only stable within a 0.15m radius.
- **Policy-Safe Tests**: These tests use moves like `f4 to f5` to verify the physical arm.
- **Fallback Tests**: These tests verify that the system correctly switches to **Teleportation** for moves that are too far away for the current robot model.
