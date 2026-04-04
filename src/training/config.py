"""Configuration models for RoboChess fine-tuning workflows."""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from src.config import SETTINGS_DIR


@dataclass(frozen=True)
class CurriculumStage:
    """A named curriculum stage for chess manipulation training."""

    name: str
    description: str
    max_source_rank_span: int
    include_captures: bool = False
    include_clutter: bool = False


@dataclass(frozen=True)
class FineTuneConfig:
    """Top-level fine-tuning configuration used by training scripts."""

    repo_id: str
    checkpoint_filename: str
    rollout_horizon: int
    eval_games: int
    deterministic_eval: bool
    curriculum: tuple[CurriculumStage, ...]

    @classmethod
    def load_default(cls) -> "FineTuneConfig":
        """Load the default fine-tuning configuration from YAML."""
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
