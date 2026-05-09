# 02. Physics Modification Attempts (The "Iron Grip")

After diagnosing that lateral jitter was causing the cube to slip, our next instinct was to "brute force" the physics to hold the cube more firmly.

## 1. The Strategy: XML Overhaul
We hypothesized that if the friction was high enough, the cube would stay put regardless of the arm's wiggles. We performed several surgical edits to `chess_env/assets/robot.xml` and `pick_and_place.xml`.

### Experiments Performed:
1.  **Massive Friction:** We increased the sliding friction of the gripper fingers and the cube from `2.0` to **`100.0`**. 
2.  **Contact Stiffening:** We modified the `solref` and `solimp` parameters. These constants control how "springy" or "soft" a contact is. We made them extremely rigid to simulate an "Iron Grip."
3.  **Mass Reduction:** We reduced the cube's mass from `0.5kg` to `0.05kg` to minimize its lateral inertia.
4.  **Specific Contact Pairs:** We added a `<contact>` block to the XML to force a specific, high-friction interaction only between the fingers and the cube.

## 2. The Result: Solver Explosions
None of these changes worked. In fact, they made the environment **less stable**.

### What Happened:
*   **The "Ejection" Bug:** When we applied massive friction and rigid contacts, the high-frequency wiggles from the arm forced the gripper fingers and the cube to overlap by microscopic amounts. 
*   **Physics Explosion:** Because the contacts were now "unbreakable" and rigid, the MuJoCo solver saw this overlap as a massive violation of physics. To resolve it, it applied an infinite repulsive force.
*   **The Result:** Instead of dropping, the cube would literally **explode** out of the gripper, shooting meters into the air or through the floor.

## 3. The Lesson Learned
Physics modifications treat the **symptom**, not the **cause**. 
The "Iron Grip" failed because it tried to fight the laws of inertia ($F=ma$). As long as the model produces high-frequency acceleration ($a$), the resulting force will eventually break any simulated friction limit. We realized we could not fix this in the XML; we had to fix it in the model's brain.
