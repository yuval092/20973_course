# 06 BUG SUMMARY: Resolved Physics and Robotic Movement Issues

This document records the major "physical" bugs encountered during the development of RoboChess and how they were squashed. 

## 🐛 Bug 1: The Initialization Flail
- **The Issue**: When the game started, the robot arm would sometimes vibrate violently, knocking over nearby pawns.
- **The Physics**: This was caused by high "Control Gains." When the simulation starts, the robot tries to reach its home position instantly. The resulting force was so large that the physics engine couldn't handle it in a single step.
- **The Fix**: Increased `settle_steps` to `50`. This gives the simulation 0.1 seconds of "calm time" at startup to let the vibrations dissipate before the first move is allowed.

## 🐛 Bug 2: The 10mm Drop (Double Compensation)
- **The Issue**: Every time the robot placed a piece, it would drop it from 1cm in the air, causing it to bounce and tip.
- **The Physics**: This was a logic error in coordinate frames. The code was subtracting the piece height *twice*—once when calculating the target and once when executing the descent.
- **The Fix**: Refactored `DESCEND_DEST` to target the absolute board height plus a tiny downward pressure offset (`CLOSE_DESCEND_OFFSET`).

## 🐛 Bug 3: The Ghost Success (Missing Validation)
- **The Issue**: The robot would sometimes miss a piece entirely, but the console would report "Move Successful."
- **The Physics**: The state machine was finishing its stages but wasn't checking if the *result* was what we wanted.
- **The Fix**: Added `FINAL_PLACEMENT_SETTLE` which performs a full geometric check of the piece's final XYZ coordinates after the arm has returned home.

## 🐛 Bug 4: The Floating Graveyards
- **The Issue**: Captured pieces were appearing in the sky above the graveyard.
- **The Physics**: In the XML generator script, the graveyard height was being added to the table height. However, the graveyard height was already defined as an absolute world coordinate.
- **The Fix**: Corrected `scripts/generate_xml.py` to use the graveyard height directly.

## 🧠 Lessons Learned
For beginners in robotics:
1.  **Coordinates are relative**: Always be sure if a number is "relative to the table" or "relative to the world origin."
2.  **Simulation needs dampening**: Virtual objects have no natural friction to stop vibrations; you must explicitly provide "settling time."
3.  **Trust but Verify**: Never assume an actuator reached its target just because you sent the command. Always check the sensor data.
