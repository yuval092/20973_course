#!/usr/bin/env python3
"""Print the default fine-tuning configuration for RoboChess."""

from src.training.config import FineTuneConfig


def main() -> None:
    config = FineTuneConfig.load_default()
    print(config)


if __name__ == "__main__":
    main()
