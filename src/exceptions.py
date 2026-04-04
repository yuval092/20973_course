class RoboChessError(Exception):
    """Base class for all RoboChess errors."""
    pass


class RuntimeCheckError(RoboChessError):
    """Raised when a runtime validation hook detects an invariant violation."""
    pass


class AIEngineError(RoboChessError):
    """Raised when Stockfish is missing or fails."""
    pass


class RLModelError(RoboChessError):
    """Raised when the RL model fails to load."""
    pass


class SceneLoadError(RoboChessError):
    """Raised when the MuJoCo scene is missing or fails to load."""
    pass


class ExecutionError(RoboChessError):
    """Raised when the robot fails to reach a target."""
    pass


class StabilityError(RoboChessError):
    """Raised when a piece is tipped or displaced."""
    pass


class BoardStateError(RoboChessError):
    """Raised when the physical board no longer matches the chess state."""
    pass


class PieceLookupError(RoboChessError):
    """Raised when an expected MuJoCo piece body or joint is unavailable."""
    pass


class ArmStateError(RuntimeCheckError):
    """Raised when the robot arm is not in the expected physical state."""
    pass


class MappingIntegrityError(RuntimeCheckError):
    """Raised when square-to-piece mappings are internally inconsistent."""
    pass


class NumericalStabilityError(RuntimeCheckError):
    """Raised when NaNs or infinities appear in the simulation state."""
    pass


class SceneIntegrityError(RuntimeCheckError):
    """Raised when the MuJoCo scene is missing required robot or piece assets."""
    pass
