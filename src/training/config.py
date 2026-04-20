"""
Configuration models for RoboChess fine-tuning workflows.

This module defines the configuration structure for fine-tuning the robot's
chess manipulation capabilities. It includes curriculum stages and the
top-level configuration loader that reads from YAML files.
"""

import yaml

from src.config import SETTINGS_DIR


class CurriculumStage:
    """
    A named curriculum stage for chess manipulation training.

    This class defines the parameters for a specific stage in the training
    curriculum, such as whether to include captures or clutter.
    """

    def __init__(self, name, description, max_source_rank_span,
                 include_captures=False, include_clutter=False):
        """
        Initialize a CurriculumStage instance.

        Args:
            name: The unique name of the stage.
            description: A short description of the stage's goal.
            max_source_rank_span: Maximum board distance for task sampling.
            include_captures: Whether to include capture moves in this stage.
            include_clutter: Whether to include non-target pieces on the board.
        """
        self.name = name
        self.description = description
        self.max_source_rank_span = max_source_rank_span
        self.include_captures = include_captures
        self.include_clutter = include_clutter


class FineTuneConfig:
    """
    Top-level fine-tuning configuration used by training scripts.

    This class holds all global settings for a fine-tuning run, including
    repository information, rollout horizons, and the curriculum itself.
    """

    def __init__(self, repo_id, checkpoint_filename, rollout_horizon,
                 eval_games, deterministic_eval, curriculum):
        """
        Initialize a FineTuneConfig instance.

        Args:
            repo_id: The ID of the model repository.
            checkpoint_filename: The filename of the model checkpoint to load.
            rollout_horizon: The number of steps to rollout per episode.
            eval_games: The number of games to play during evaluation.
            deterministic_eval: Whether to use deterministic actions in eval.
            curriculum: A tuple of CurriculumStage instances.
        """
        self.repo_id = repo_id
        self.checkpoint_filename = checkpoint_filename
        self.rollout_horizon = rollout_horizon
        self.eval_games = eval_games
        self.deterministic_eval = deterministic_eval
        self.curriculum = curriculum

    @classmethod
    def load_default(cls):
        """
        Load the default fine-tuning configuration from YAML.

        Reads the 'training.yaml' file from the settings directory and
        constructs a FineTuneConfig instance with its contents.

        Returns:
            FineTuneConfig: The loaded configuration instance.
        """
        path = SETTINGS_DIR / "training.yaml"
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        curriculum = tuple(CurriculumStage(**item) for item in raw["curriculum"])
        return cls(
            repo_id=raw["repo_id"],
            checkpoint_filename=raw["checkpoint_filename"],
            rollout_horizon=raw["rollout_horizon"],
            eval_games=raw["eval_games"],
            deterministic_eval=raw["deterministic_eval"],
            curriculum=curriculum,
        )
