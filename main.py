"""RoboChess CLI entrypoint."""

import logging

from src.game_runtime import main
from src.runtime_guard import freeze_on_exception


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Change the level to DEBUG when investigating controller stages or env steps.


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        freeze_on_exception(exc)
