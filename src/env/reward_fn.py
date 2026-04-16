"""
Separable dense reward function for HER compatibility (Section 6).
"""

import numpy as np

# ─── Reward weights ───────────────────────────────────────────────
W_APPROACH       =  2.0    # reward per 1 cm gripper-to-piece reduction
W_GRASP_BONUS    = 10.0    # one-time bonus when grasp detected
W_LIFT_BONUS     =  5.0    # one-time bonus when piece lifted > 2 cm
W_TRANSPORT      =  3.0    # reward per 1 cm piece-to-dest reduction
W_PLACEMENT      = 50.0    # large bonus for final success
W_TILT_PENALTY   = -1.0    # per step while up_z < 0.966 during transport
W_DROP_PENALTY   = -20.0   # one-time, piece dropped after lift
W_CLUTTER_PEN    = -5.0    # one-time, displaced neighbor piece
W_TIMEOUT_PEN    = -2.0    # flat penalty on truncation
FINGER_CLOSED_THR = 0.005  # m, finger qpos below this = closed

# ─── Placement tolerance (must match is_success in env) ───────────
XY_TOL = 0.010   # m
Z_TOL  = 0.010   # m
UP_Z_THR = 0.966 # cos(15°) ≈ 0.966


def compute_reward(
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    info: dict,
) -> float:
    """
    Dense shaped reward. Separable — all state accessed via `info`.
    """
    phase    = info.get("phase", "approach")
    grip_pos = np.asarray(info.get("grip_pos", achieved_goal))
    up_z     = info.get("up_z", 1.0)
    table_height = info.get("table_height", 0.422)

    reward = 0.0

    # ── Approach phase ──────────────────────────────────────────────
    if phase == "approach":
        dist_grip_to_piece = np.linalg.norm(grip_pos - achieved_goal)
        # Reward proportional to closeness (negative distance, shifted)
        reward += W_APPROACH * max(0.0, 0.30 - dist_grip_to_piece)

        # Detect grasp
        finger_qpos = np.asarray(info.get("finger_qpos", [1.0, 1.0]))
        fingers_closed = np.all(np.abs(finger_qpos) < FINGER_CLOSED_THR)
        piece_above_table = achieved_goal[2] > table_height + 0.008
        if fingers_closed and piece_above_table:
            reward += W_GRASP_BONUS  # one-time (phase gate in env prevents double-award)

    # ── Transport phase ──────────────────────────────────────────────
    elif phase == "transport":
        dist_piece_to_dest = np.linalg.norm(achieved_goal - desired_goal)
        reward += W_TRANSPORT * max(0.0, 0.60 - dist_piece_to_dest)

        # Lift bonus (piece significantly above table)
        piece_height = achieved_goal[2] - table_height
        if piece_height > 0.02:
            reward += W_LIFT_BONUS * min(piece_height / 0.10, 1.0)

        # Tilt penalty during transport
        if up_z < UP_Z_THR:
            reward += W_TILT_PENALTY

    # ── Placement phase ──────────────────────────────────────────────
    elif phase == "placement":
        xy_err = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        z_err  = abs(achieved_goal[2] - desired_goal[2])
        reward += W_TRANSPORT * max(0.0, 0.60 - np.linalg.norm(achieved_goal - desired_goal))

        # Success bonus
        if xy_err < XY_TOL and z_err < Z_TOL and up_z >= UP_Z_THR:
            reward += W_PLACEMENT

        # Tilt penalty during placement
        if up_z < UP_Z_THR:
            reward += W_TILT_PENALTY

    # ── Global penalties (any phase) ────────────────────────────────
    if info.get("piece_dropped", False):
        reward += W_DROP_PENALTY

    if info.get("neighbor_displaced", False):
        reward += W_CLUTTER_PEN

    if info.get("timed_out", False):
        reward += W_TIMEOUT_PEN

    return float(reward)
