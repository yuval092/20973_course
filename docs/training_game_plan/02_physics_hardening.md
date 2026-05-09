# Physics Hardening: The "Last Mile" of Manipulation

In MuJoCo, the transition from pure kinematics (moving the arm in empty space) to physical interaction (grasping, moving, and releasing a payload) exposes the limitations of rigid-body simulations. To bridge this gap, we must perform "Physics Hardening"—modifying the simulation parameters to prioritize contact stability over perfect realism.

## 1. The Grasp Width Bug (The 0.4mm Gap)
The most critical issue identified during the investigation was the **Geometric Offset**.

*   **The Issue:** The target joint position of `0.0145m` resulted in a physical distance of `0.0154m` from the finger surface to the center. Since the chess piece (cube) is `30mm` wide (15mm half-width), this left a **0.4mm air gap** on both sides. The gripper was closing, but it was lifting nothing but air.
*   **The Fix:** Update `GRASP_WIDTH_CLOSED` in `src/chess_env/task.py` to **`0.0136m`**.
*   **The Math:** This value provides exactly `0.5mm` of penetration. At `kp=60000`, this generates exactly **30N of Normal Force**, which translates to 15N of static friction ($\mu=0.5$).

## 2. "Rubber Finger" Contacts (solref and solimp)
The SAC model produces high-frequency "jitter" (1.5 m/s² lateral acceleration). In a perfectly rigid simulation, this acts as a "Pneumatic Hammer," constantly breaking contact with the cube ($N=0$) and eliminating friction, causing the cube to slide or "explode" due to infinite repulsive forces.

*   **The Fix:** Modify `chess_env/assets/pick_and_place.xml`. Change `solref` from `0.005` to **`0.02`** and `solimp` to **`0.8 0.95 0.001`**.
*   **The Effect:** This makes the fingers behave like they have 2mm thick rubber pads. The simulation can now absorb high-frequency, microscopic wiggles without reflecting that energy as an impulsive strike. The fingers will maintain continuous contact, ensuring friction is always active.

## 3. Contact Dimensionality (condim)
In MuJoCo, `condim="4"` (the default) calculates sliding friction and spin friction around the normal axis, but ignores rolling friction.

*   **The Fix:** Change the cube and finger contact `condim` to **`6`** in `pick_and_place.xml`.
*   **The Effect:** This enables full 6D friction (including torsional/spinning friction). It prevents the cube from imperceptibly rotating or "walking" within the fingers, a primary cause of slips during horizontal transit.

## 4. Payload Inertia Modification
*   **The Fix:** Reduce the cube mass from `0.05kg` to **`0.02kg`**.
*   **The Effect:** Lower mass means lower lateral inertia ($F=ma$). During rapid horizontal movement (`Transit`), the cube generates less outward force against the friction holding it in place.

## 5. Damping for Slippage
*   **The Fix:** Increase the `damping` value on the `object0:joint` from `0.1` to **`1.0`**.
*   **The Effect:** This acts as "viscous air." If the cube does begin to slide, the damping resists its acceleration, buying the friction simulation more time to "catch" it before it falls out.

## 6. Settlement Precision Reduction
*   **The Fix:** Reduce `settle_tolerance` in `configs/physics.yaml` from `0.003` to **`0.0005`** (0.5mm).
*   **The Effect:** Before closing the fingers or starting a move, the arm must settle. A 3mm error meant the arm might not be centered over the cube. 0.5mm ensures perfect alignment for the 0.5mm penetration grasp math.

## 7. The Rejected Finger Geometry ("Lips")
We explored modifying the flat finger boxes into "U-shaped" cradles with vertical lips to mechanically block the cube from sliding out.

*   **Why it was rejected:** While excellent for transit, vertical lips pose a massive snagging risk during the **Release** phase. If the arm hasn't settled perfectly or the lips are sharp 90-degree boxes, opening the fingers could tilt or drag the cube across the board, ruining the precise placement. We decided to rely on hardened friction and soft contacts instead.
