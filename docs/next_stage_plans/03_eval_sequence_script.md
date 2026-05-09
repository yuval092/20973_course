# Evaluation Script Specification: eval_sequence.py

## Purpose

`scripts/eval_sequence.py` evaluates the model's ability to execute multi-scenario
chains without full resets between scenarios. It is the primary tool for validating
chained behavior before committing to a full game controller.

---

## Command-Line Interface

```bash
python scripts/eval_sequence.py \
    --model      models/latest_model.zip     \  # SAC model to evaluate
    --chain      "transit,descend,ascend"    \  # Comma-separated scenario sequence
    --n-episodes 100                         \  # Episodes to run per chain
    --drift-limit 0.010                      \  # Tube drift limit for descend/ascend (V6 standard)
    --debug                                  \  # Verbose per-step and per-transition logging
    --visualize                              \  # Render the simulation
    --delay      0.0                         \  # Visualization delay in seconds per step
    --wait                                      # Pause after each chain episode (keypress)
```

### Flag Details

**`--debug`**
Enables verbose logging at three levels:
- Per-step: gripper position, velocity, distance to goal, drift, finger state, model output
- Per-transition: halt steps, final speed, alignment steps, final alignment error, finger state
- Per-episode: full chain result table, per-scenario metrics

**`--visualize`**
Opens the MuJoCo viewer during evaluation. Same as `eval.py --visualize`. Can be used
without `--debug` for visual inspection.

**`--delay <seconds>`**
Sleep `delay` seconds between each env.step() call. Useful for slowing down visualization
for human inspection (e.g., `--delay 0.05` for roughly real-time, `--delay 0.2` for slow-mo).

**`--wait`**
After each completed chain episode (success or failure), pause execution and wait for
the user to press Enter before continuing to the next episode. Useful for inspecting the
arm's final state in the viewer. Inspired by `scripts/visualize.py`. When combined with
`--visualize`, shows the arm frozen in its end state until the user continues.

**Predefined chain shortcuts:**
```bash
--chain full_move   # Equivalent to: transit,descend,ascend,transit,descend,ascend
--chain pick        # Equivalent to: transit,descend,ascend
--chain place       # Equivalent to: transit,descend,ascend
--chain vertical    # Equivalent to: descend,ascend
```

---

## Output Format

### Console (per-episode debug sample with `--debug`)

```
================================================================================
[CHAIN EP 3/100]  transit → descend → ascend
  Move: src=(0.312, 0.489) → dst=(0.521, 0.633)
================================================================================
[SCENARIO 1/3: transit]  goal=( 0.521,  0.633,  0.550)
  step   1: pos=( 0.319,  0.496,  0.550) dist=211.3mm speed= 18.4mm/s drift=N/A
  step   5: pos=( 0.381,  0.532,  0.549) dist=148.2mm speed= 22.1mm/s drift=N/A
  step  10: pos=( 0.444,  0.578,  0.550) dist= 83.7mm speed= 19.8mm/s drift=N/A
  step  15: pos=( 0.504,  0.617,  0.550) dist= 26.1mm speed= 11.3mm/s drift=N/A
  step  18: pos=( 0.516,  0.628,  0.550) dist=  7.3mm speed=  1.9mm/s drift=N/A
  OUTCOME: SUCCESS  steps=18  reward=312.4  dist=7.3mm  speed=1.9mm/s
  ──────────────────────────────────────────────────────────────────────────────
  [TRANSITION 1→2]
    HALT:  hold_steps=6   final_speed=0.28mm/s ✓
    ALIGN: settle_steps=34  final_error=1.8mm   converged=True ✓
    GRIPPER: closed→open  validated=True ✓
    Handoff pos=(0.522, 0.634, 0.550)  error_from_nominal=1.8mm  tube_center=(0.521, 0.633)
  ──────────────────────────────────────────────────────────────────────────────
[SCENARIO 2/3: descend]  goal=( 0.521,  0.633,  0.430)  tube_center=(0.521, 0.633)
  step   1: pos=( 0.522,  0.634,  0.550) dist=120.3mm speed=  0.1mm/s drift= 1.8mm/5.0mm ✓
  step   5: pos=( 0.522,  0.634,  0.501) dist= 71.3mm speed= 14.2mm/s drift= 1.8mm/5.0mm ✓
  step  11: pos=( 0.522,  0.634,  0.432) dist=  2.2mm speed=  1.1mm/s drift= 1.8mm/5.0mm ✓
  OUTCOME: SUCCESS  steps=11  reward=189.3  dist=2.2mm  speed=1.1mm/s
  ...
================================================================================
[CHAIN EP 3 RESULT]: SUCCESS  (3/3 scenarios)  total_reward=671.3  total_steps=42
================================================================================
```

### Summary (end of all episodes)

```
Chain: transit → descend → ascend  (100 episodes)
Model: models/latest_model.zip
Drift limit: 5mm

Scenario-level breakdown:
  transit:  100/100 succeeded | avg_dist= 7.1mm | avg_speed= 2.4mm/s | avg_steps=18
  descend:   94/100 succeeded | avg_dist= 2.8mm | avg_speed= 0.9mm/s | avg_steps=11
  ascend:    89/100 succeeded | avg_dist= 3.4mm | avg_speed= 1.1mm/s | avg_steps=10

Chain-level results:
  Full chain success rate:  89.0%   (all 3 scenarios succeeded)
  Failed at transit:         0 / 100   (0.0%)
  Failed at descend:         6 / 100   (6.0%)  [TUBE_BREACH x5, TIMEOUT x1]
  Failed at ascend:          5 / 100   (5.0%)  [TIMEOUT x4, TUBE_BREACH x1]

Transition diagnostics (successful transitions only):
  transit → descend: avg halt_steps=5.2  avg align_error=2.1mm  max_align_error=4.8mm
  descend → ascend:  avg halt_steps=3.1  avg align_error=0.4mm  max_align_error=1.2mm

Avg total chain reward: 682.4
```

---

## Internal Architecture

### ChainEpisodeRunner class

```python
class ChainEpisodeRunner:
    """
    Runs a sequence of RL scenarios back-to-back on a single environment instance.
    Handles halt gating, waypoint alignment, soft resets, and per-scenario metrics.
    """
    def __init__(self, env, model, chain, drift_limit, debug=False,
                 delay=0.0, wait=False):
        self.env = env
        self.model = model
        self.chain = chain          # ["transit", "descend", "ascend"]
        self.drift_limit = drift_limit
        self.debug = debug
        self.delay = delay
        self.wait = wait

    def run_one_chain(self, dst_positions: list[np.ndarray]) -> ChainResult:
        """
        Executes one full chain episode.

        Args:
            dst_positions: List of nominal XY goal positions for each scenario.
                           Length must equal len(self.chain).
                           For transit: the destination cell XY.
                           For descend/ascend: the cell XY (arm was aligned there).
        """
```

### Episode Flow

```
1. env.reset() — full reset for the first scenario
   - First scenario always uses env.reset() (arm teleports to canonical start)
   - tube_center for descend/ascend scenarios uses NOMINAL position, not arm position

2. RL loop — run model.predict() steps until terminated or truncated
   - At each step: apply --delay, log if --debug
   - Collect: steps, reward, final_pos, final_vel, outcome

3. If outcome != SUCCESS → abort chain, record "failed at scenario N", break

4. If last scenario → record success, break

5. TRANSITION SEQUENCE (for all but the last scenario):
   a. COMPLETE HALT:
      - Issue zero-movement hold actions until speed < 0.5mm/s (up to 100 steps)
      - Actively zero arm qvel/qacc (arm joints only, not object)
      - Confirm halt — abort chain if speed still >= 0.5mm/s
   b. WAYPOINT ALIGNMENT:
      - Run scripted settle loop to nominal exit waypoint
      - Converge within 3mm or abort chain
      - Zero arm qvel after convergence
   c. GRIPPER TRANSITION:
      - Execute scripted 50-step open/close if state change needed
      - Validate finger position
   d. SOFT RESET:
      - Update scenario, goal_pos, tube_center = nominal_xy
      - Return fresh observation

6. If --wait: pause for keypress before next episode

7. Repeat from step 2 for the next scenario
```

---

## Scenario Combinations to Test

Ordered from simplest to most complex. Each tier requires the previous tier to pass first.

### Tier 1: Two-Scenario Chains (Baseline)

| Chain | Tests | Physical meaning |
|:---|:---|:---|
| `transit,transit` | Two consecutive transits | Carry across board twice |
| `transit,descend` | Transit precision → descend | Approach cell + lower to piece |
| `descend,ascend` | Vertical round-trip | Grasp position validation |
| `ascend,transit` | Ascend precision → transit | Lift piece + carry |

### Tier 2: Three-Scenario Chains (Core)

| Chain | Tests | Physical meaning |
|:---|:---|:---|
| `transit,descend,ascend` | Full pick sequence | Arrive, lower, lift |
| `ascend,transit,descend` | Full carry sequence | Lift from A, move to B, lower at B |
| `descend,ascend,transit` | Vertical + departure | Probe position, then leave |

### Tier 3: Four-Scenario Chains (Complex)

| Chain | Tests | Physical meaning |
|:---|:---|:---|
| `transit,descend,ascend,transit` | Pick and carry | Full piece transport (no release) |
| `ascend,transit,descend,ascend` | Carry + placement | Lift, move, lower, re-lift |

### Tier 4: Full Move Chain (Production Target)

| Chain | Tests | Physical meaning |
|:---|:---|:---|
| `transit,descend,ascend,transit,descend,ascend` | Complete chess move | Pick at src + place at dst |

---

## Goal Position Strategy

The script pre-samples destination positions at episode start, then uses the same
nominal positions throughout the chain:

```python
# At episode start: pre-sample the move destinations
src_xy = sample_board_position()       # Where piece starts
dst_xy = sample_board_position()       # Destination (ensure min separation from src)

# Each scenario uses NOMINAL positions — not achieved positions
goals_by_scenario = {
    "transit":  np.array([dst_xy[0], dst_xy[1], SAFE_Z]),
    "descend":  np.array([dst_xy[0], dst_xy[1], GRASP_Z]),
    "ascend":   np.array([dst_xy[0], dst_xy[1], SAFE_Z]),
}
# tube_center = dst_xy always, set in soft_reset via nominal_xy
```

For multi-transit chains where each transit goes to a different cell:
```python
# Chain: transit(A→B), transit(B→C) — each transit has its own destination
waypoints = [sample_board_position() for _ in range(len(chain) + 1)]
```

---

## Metrics to Collect

### Per-scenario (within a chain)
- `success_rate`: fraction of episodes where this scenario succeeded
- `crash_rate`: fraction that crashed, with breakdown by crash type
- `timeout_rate`: fraction that timed out
- `avg_steps`: mean steps to succeed (successes only)
- `avg_final_dist_mm`: mean gripper-to-goal distance at termination
- `avg_final_speed_mm_s`: mean gripper speed at termination

### Per-transition (between scenarios)
- `halt_steps_used`: hold steps the halt gate needed
- `halt_final_speed_mm_s`: speed after hold loop (before active zeroing)
- `align_steps_used`: settle loop steps until convergence
- `align_final_error_mm`: final distance from nominal waypoint after alignment
- `align_converged`: whether alignment converged within budget

### Per-chain
- `full_chain_success_rate`: fraction where all scenarios succeeded
- `failure_scenario_index`: which scenario (1-indexed) caused failure
- `failure_breakdown`: crash_type → count per failed scenario

---

## Max Episode Steps Per Scenario

Each scenario in a chain gets its own `max_steps` budget from the environment's
`max_episode_steps=200` setting, applied per-scenario. A 3-scenario chain can take
up to 600 total steps.

---
*Next: [04 — Risk Analysis](./04_risk_analysis.md)*
