# 05 HEALTH CHECKS: Maintaining Simulation Stability

In a high-fidelity simulator like MuJoCo, things can go wrong very quickly. A single frame of "interpenetration" (where two solid objects overlap) can lead to infinite forces and cause pieces to explode across the virtual room. 

The `RuntimeCheckRegistry` provides a "safety net" by running specific tests at critical hooks.

## 🪝 Critical Hooks
Checks are triggered at these logical points:
1.  **`PROGRAM_START`**: Checks if required assets (meshes, XML) are present.
2.  **`TURN_START`**: Verifies the logical chess board matches the physical pieces.
3.  **`ARM_STAGE_START`**: Checks if the robot can actually reach the target coordinate before it tries.
4.  **`ARM_STAGE_END`**: Verifies if the stage succeeded (e.g., "Is the piece still in the gripper?").
5.  **`TURN_END`**: Checks for fallen pieces or drift.

## 🩺 The Registry of Checks

### 1. Finite State Check (`FiniteStateCheck`)
Ensures that no value in the simulation has become `NaN` (Not a Number) or `Inf` (Infinity). 

**The Physics of NaN**: In a rigid-body simulator, if two highly rigid objects (like the robot's metal base and a solid table) accidentally spawn overlapping each other, the physics engine will register a massive negative distance. To resolve this, it calculates a separation force. If the overlap is deep enough, that calculated force approaches infinity, immediately overflowing the floating-point variables in the C++ engine to `NaN`. If the physics "explodes" like this, the simulation is permanently corrupted. This check catches it immediately.

### 2. Piece Stability Check (`BoardObserver`)
Checks every piece on the board for:
-   **Tilt**: If a piece is tilted more than 15 degrees, it is considered "fallen."
-   **Height**: If a piece is floating too high or has fallen through the table.

### 3. Gripper State Check (`GripperOpenCheck`)
Before the robot starts a move, it must have its "hands" open. If the gripper is stuck or flailing, this check raises an `ArmStateError`.

### 4. Mapping Integrity Check (`MappingIntegrityCheck`)
Ensures that the system hasn't "lost" a piece. It verifies that for every piece on the chess board, there is exactly one corresponding body in the MuJoCo world.

## ⚠️ Why These Checks Matter for Beginners
In many coding projects, an error just prints a message. In RoboChess:
1.  A health check failure raises a specific **Exception** (e.g., `StabilityError`).
2.  The `RuntimeGuard` catches this exception.
3.  The simulation **Freezes** (stops time).
4.  The developer can then look at the screen and see exactly *where* the robot was and *why* the piece fell.

Without these checks, a bug in the physics might look like a bug in the AI, making development impossible.
