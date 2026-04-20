"""
This script verifies that the MuJoCo model can be successfully loaded from the MJCF scene XML file.
It checks for any syntax or configuration errors in the XML that would prevent MuJoCo from initializing.
"""

import sys
from pathlib import Path
import mujoco

# Ensure project root is in sys.path for internal imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import SCENE_XML  # noqa: E402


class MuJoCoValidator:
    """
    A class used to validate the MuJoCo scene configuration.
    """

    def __init__(self, xml_path):
        """
        Initializes the MuJoCoValidator with a specific XML path.

        Args:
            xml_path: Path to the MJCF XML file to validate.
        """
        self.xml_path = xml_path

    def validate(self):
        """
        Attempts to load the MuJoCo model from the specified XML path.

        Returns:
            0 if successful, 1 if an exception occurs.
        """
        try:
            mujoco.MjModel.from_xml_path(self.xml_path)
            print("Successfully loaded MuJoCo model!")
            return 0
        except Exception as exc:
            print(f"Failed to load MuJoCo model: {exc}")
            return 1


def main():
    """
    Executes the MuJoCo validation process and exits with the appropriate status code.
    """
    validator = MuJoCoValidator(SCENE_XML)
    return validator.validate()


if __name__ == "__main__":
    sys.exit(main())
