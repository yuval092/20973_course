# Configuration Guide: YAML Reference

This document serves as a master reference for all parameters in the `configs/` directory.

## 1. `env.yaml` (Task and Geometry)

| Key | Unit | Description |
| :--- | :--- | :--- |
| `cube_height` | m | The vertical height of a chess piece (0.030m = 3cm). |
| `grasp_z` | m | Target altitude for picking up a piece (0.430m). |
| `safe_z` | m | Target altitude for horizontal transit (0.550m). |
| `success_threshold` | m | Distance within which a goal is considered reached (0.010m). |
| `stability_vel_threshold`| m/s | Maximum velocity to consider the arm "stationary" at the goal (0.05). |
| `drift_limit_start` | m | Starting radius of the Virtual Tube curriculum (0.100m). |
| `drift_limit_end` | m | Final radius of the Virtual Tube curriculum (0.015m). |
| `eval_drift_limit` | m | Radial limit used during evaluation (0.035m - matches cell size). |
| `crash_penalty` | float | Reward value applied upon collision (-500.0). |
| `hidden_object_pos` | [x,y,z]| Stable floor coordinate for object-hiding [2.0, 2.0, 0.015]. |

## 2. `training.yaml` (Algorithm and Reporting)

| Key | Unit | Description |
| :--- | :--- | :--- |
| `total_timesteps` | int | Total simulation steps for a full training run (1,000,000). |
| `num_envs` | int | Number of parallel worker environments (12). |
| `learning_rate` | float | Adam optimizer step size (5e-5). |
| `eval_freq` | steps | Total steps across all envs between evaluations (10,000). |
| `log_freq` | steps | Frequency of progress updates to the console (2,000). |
| `moving_avg_window` | episodes| Window size for mean reward/success statistics (100). |

## 3. `physics.yaml` (MuJoCo Constants)

| Key | Unit | Description |
| :--- | :--- | :--- |
| `settle_steps` | int | Initial physics steps for internal reset logic (25). |
| `settle_steps_final` | int | Physics steps run *after* object restoration during reset (25). |
| `max_settle_steps` | int | Limit for the gain-based arm settle loop (500). |
| `pos_ctrl_scale` | m | Scaling factor for RL actions (0.015m per step). |
| `vertical_quat` | quat | [1, 0, 1, 0] - Locks gripper to point straight down. |
