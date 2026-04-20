# 07 FUTURE ADDITIONS: Roadmap for Improvements

RoboChess is a robust prototype, but there is always room to make the physics more realistic and the AI more intelligent.

## 🚀 Near-Term Improvements

### 1. Adaptive Physics (Friction & Masks)
Currently, all pieces have the same friction and collision configurations.
- **Improvement**: Dynamically adjust `solimp` and `solref` modifiers in real-time depending on whether a piece is currently being gripped vs standing free.
- **Improvement**: Implement dynamic bitwise `contype` masking for human hands. If we implement a VR glove to play against the robot, assigning the human hands `contype="8"` could dynamically allow or prevent interference with the robot's hardware while allowing shared physics interaction with the pieces.

### 2. Multi-Agent Support
Currently, the Black pieces are moved by a simple AI loop.
- **Improvement**: Deploy a second robot arm on the opposite side of the table. This would require coordinating two `ExecutionController` instances and handling arm-to-arm collisions.

### 3. Advanced Collision Avoidance
The robot currently moves in straight lines between stage waypoints.
- **Improvement**: Use a path-planning algorithm like **RRT (Rapidly-exploring Random Trees)** to steer the arm *around* tall pieces (like the Queen) during transit, rather than just relying on a high `Z_SAFE` clearance.

## 🧠 Advanced Concepts

### 4. Sim-to-Real Transfer
The dream of any robotics project is to run on a physical robot.
- **Goal**: Make the simulation so accurate that the Python code could be plugged into a real Fetch robot and "just work."
- **Requirements**: Domain randomization (randomly varying light, mass, and friction) so the RL policy doesn't become too reliant on the "perfect" virtual world.

### 5. Piece Recovery (Self-Healing)
- **Goal**: If the robot knocks a piece over, it should detect the "Fallen" state and use the arm to upright the piece instead of just freezing the game.
- **Physics**: This would require a very advanced RL policy trained specifically for "non-prehensile manipulation" (poking and nudging objects).

### 6. Dynamic Camera Angles
- **Goal**: Add virtual cameras to the robot's "head" in MuJoCo and use Computer Vision (CV) to find the pieces instead of using the "perfect" ground-truth coordinates from the simulation.
- **Benefit**: This makes the project much more realistic and closer to how real-world AI robots operate.
