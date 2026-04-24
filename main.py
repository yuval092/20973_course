"""
Main entry point for the RoboChess application.
This file initializes logging and starts the game runtime, ensuring that
any unhandled exceptions are caught and handled by the runtime guard.
"""

import logging
from src.app import main as start_game
from src.runtime_guard import RuntimeGuard


class RoboChessLauncher:
    """
    A class responsible for bootstrapping and launching the RoboChess application.
    """

    def __init__(self):
        """
        Initializes the launcher and configures the logging system.
        """
        self._setup_logging()

    def _setup_logging(self):
        """
        Configures the basic logging settings for the application.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        # Change the level to DEBUG when investigating controller stages or env steps.

    def run(self):
        """
        Starts the game runtime and manages exceptions using the runtime guard.
        """
        try:
            start_game()
        except Exception as exc:
            RuntimeGuard.freeze_on_exception(exc)


def main():
    """
    Main execution function for the RoboChess CLI.
    """
    launcher = RoboChessLauncher()
    launcher.run()


if __name__ == "__main__":
    main()
