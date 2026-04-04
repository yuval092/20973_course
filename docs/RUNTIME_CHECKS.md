# Runtime Checks

## Purpose

Runtime checks are explicit invariants that execute at named hook points in the application flow. They replace ad hoc validation with a testable, modular mechanism.

## Hook Points

Current hook names:

- `program_start`
- `post_scene_load`
- `turn_start`
- `turn_end`
- `arm_stage_start`
- `arm_stage_end`

The runtime currently dispatches:

- `post_scene_load`
- `program_start`
- `turn_start`
- `turn_end`
- arm stage lifecycle events emitted by `ExecutionController`

## API

Each check is a class that extends `RuntimeCheck` and implements:

```python
def run(self, context: CheckContext) -> None:
    ...
```

If a check fails, it raises an exception.

## Current Default Checks

- `SceneAssetsCheck`
- `MappingIntegrityCheck`
- `BoardAgreementCheck`
- `ArmHomePoseCheck`
- `RobotWorkspaceCheck`
- `FiniteStateCheck`
- `PieceObserverCheck`
- `StageGoalReachabilityCheck`
- `StageOutcomeCheck`
- `StageGripAttachmentCheck`

## Extending

To add a new check:

1. define a `RuntimeCheck` subclass
2. assign its `hooks`
3. register it in `build_default_check_registry()`

Checks should be:

- deterministic
- cheap enough to run at their hook point
- specific in the exception they raise
- covered by tests
