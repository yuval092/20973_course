"""Load the generated MJCF scene and report whether MuJoCo accepts it."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import SCENE_XML  # noqa: E402


def main() -> int:
    """Return a process status code after attempting to load the scene."""
    try:
        mujoco.MjModel.from_xml_path(SCENE_XML)
        print("Successfully loaded MuJoCo model!")
        return 0
    except Exception as exc:
        print(f"Failed to load MuJoCo model: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
