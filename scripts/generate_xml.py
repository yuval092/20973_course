"""Generate the MuJoCo chess_world.xml scene file.

This script constructs the MJCF XML that defines the chessboard, pieces,
graveyards, labels, and lighting for the RoboChess simulation. It imports
board-geometry constants from ``src.config`` so there is a single source of
truth for positions and dimensions.

Run from the project root::

    python scripts/generate_xml.py

The generated file is written to ``src/assets/chess_world.xml``.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    BLACK_GRAVEYARD_ORIGIN,
    BOARD_CENTER,
    GRAVEYARD_PLATFORM_HALF_EXTENTS,
    GRAVEYARD_PLATFORM_HEIGHT,
    SQUARE_SIZE,
    TABLE_HEIGHT,
    WHITE_GRAVEYARD_ORIGIN,
    Z_GRASP,
)


LABEL_STROKES = {
    "A": [((0.1, 0.0), (0.22, 1.0)), ((0.78, 0.0), (0.9, 1.0)), ((0.22, 0.44), (0.78, 0.56)), ((0.22, 0.88), (0.78, 1.0))],
    "B": [((0.1, 0.0), (0.22, 1.0)), ((0.22, 0.88), (0.7, 1.0)), ((0.22, 0.44), (0.68, 0.56)),
          ((0.22, 0.0), (0.7, 0.12)), ((0.68, 0.56), (0.8, 0.88)), ((0.68, 0.12), (0.8, 0.44))],
    "C": [((0.22, 0.88), (0.82, 1.0)), ((0.1, 0.12), (0.22, 0.88)), ((0.22, 0.0), (0.82, 0.12))],
    "D": [((0.1, 0.0), (0.22, 1.0)), ((0.22, 0.88), (0.68, 1.0)), ((0.22, 0.0), (0.68, 0.12)),
          ((0.68, 0.12), (0.8, 0.88))],
    "E": [((0.1, 0.0), (0.22, 1.0)), ((0.22, 0.88), (0.82, 1.0)), ((0.22, 0.44), (0.74, 0.56)),
          ((0.22, 0.0), (0.82, 0.12))],
    "F": [((0.1, 0.0), (0.22, 1.0)), ((0.22, 0.88), (0.82, 1.0)), ((0.22, 0.44), (0.74, 0.56))],
    "G": [((0.22, 0.88), (0.82, 1.0)), ((0.1, 0.12), (0.22, 0.88)), ((0.22, 0.0), (0.82, 0.12)),
          ((0.6, 0.28), (0.82, 0.4)), ((0.7, 0.28), (0.82, 0.72))],
    "H": [((0.1, 0.0), (0.22, 1.0)), ((0.78, 0.0), (0.9, 1.0)), ((0.22, 0.44), (0.78, 0.56))],
    "1": [((0.44, 0.0), (0.56, 1.0)), ((0.28, 0.82), (0.56, 1.0)), ((0.24, 0.0), (0.76, 0.12))],
    "2": [((0.18, 0.88), (0.82, 1.0)), ((0.7, 0.56), (0.82, 0.88)), ((0.18, 0.44), (0.82, 0.56)),
          ((0.18, 0.12), (0.3, 0.44)), ((0.18, 0.0), (0.82, 0.12))],
    "3": [((0.18, 0.88), (0.82, 1.0)), ((0.18, 0.44), (0.82, 0.56)), ((0.18, 0.0), (0.82, 0.12)),
          ((0.7, 0.12), (0.82, 0.88))],
    "4": [((0.18, 0.56), (0.7, 0.68)), ((0.18, 0.56), (0.3, 1.0)), ((0.7, 0.0), (0.82, 1.0))],
    "5": [((0.18, 0.88), (0.82, 1.0)), ((0.18, 0.56), (0.3, 0.88)), ((0.18, 0.44), (0.82, 0.56)),
          ((0.7, 0.12), (0.82, 0.44)), ((0.18, 0.0), (0.82, 0.12))],
    "6": [((0.18, 0.88), (0.82, 1.0)), ((0.18, 0.12), (0.3, 0.88)), ((0.18, 0.44), (0.82, 0.56)),
          ((0.7, 0.12), (0.82, 0.44)), ((0.18, 0.0), (0.82, 0.12))],
    "7": [((0.18, 0.88), (0.82, 1.0)), ((0.62, 0.0), (0.74, 0.88))],
    "8": [((0.18, 0.88), (0.82, 1.0)), ((0.18, 0.44), (0.82, 0.56)), ((0.18, 0.0), (0.82, 0.12)),
          ((0.18, 0.12), (0.3, 0.88)), ((0.7, 0.12), (0.82, 0.88))],
}


def _append_label(xml, name, glyph, center_x, center_y, z, scale=0.028, depth=0.0012):
    """Render a simple block-stroke character as a visible board label."""
    rgba = "0.87 0.18 0.18 0.95"
    for idx, ((x0, y0), (x1, y1)) in enumerate(LABEL_STROKES[glyph]):
        cx = center_x + ((x0 + x1) / 2.0 - 0.5) * scale
        cy = center_y + ((y0 + y1) / 2.0 - 0.5) * scale
        sx = max((x1 - x0) * scale / 2.0, 0.0008)
        sy = max((y1 - y0) * scale / 2.0, 0.0008)
        xml.append(
            f'        <geom name="{name}_{idx}" type="box" size="{sx:.4f} {sy:.4f} {depth / 2:.4f}" '
            f'pos="{cx:.3f} {cy:.3f} {z:.4f}" rgba="{rgba}" contype="0" conaffinity="0"/>'
        )


def _file_x(file_idx: int) -> float:
    """Compute the X coordinate for file index ``a=0`` through ``h=7``."""
    start_x = BOARD_CENTER[0] - 4 * SQUARE_SIZE
    return start_x + (7 - file_idx + 0.5) * SQUARE_SIZE


def _rank_y(rank_idx: int) -> float:
    """Compute the Y coordinate for rank index ``1=0`` through ``8=7``."""
    start_y = BOARD_CENTER[1] + 4 * SQUARE_SIZE
    return start_y - (rank_idx + 0.5) * SQUARE_SIZE


def _append_file_labels(xml: list[str], prefix: str, y: float, z: float) -> None:
    """Render file labels along one board edge.

    Both edges use the same file-to-square mapping so opposite sides of the
    board agree on which file label belongs to which column.
    """
    for file_idx, glyph in enumerate("ABCDEFGH"):
        board_file_idx = file_idx
        _append_label(xml, f"{prefix}_{glyph.lower()}", glyph, _file_x(board_file_idx), y, z)


def generate_chess_world() -> str:
    """Build the full MuJoCo XML string for the chess scene."""
    xml: list[str] = []
    xml.append('<mujoco model="fetch_chess">')
    xml.append('    <include file="fetch.xml"/>')
    xml.append('    <visual>')
    xml.append('        <quality shadowsize="0"/>')
    xml.append('    </visual>')
    xml.append('    <asset>')
    xml.append('        <material name="white_square" rgba="0.96 0.87 0.70 1"/>')
    xml.append('        <material name="black_square" rgba="0.55 0.27 0.07 1"/>')
    xml.append('        <material name="white_piece" rgba="1 1 1 1"/>')
    xml.append('        <material name="black_piece" rgba="0.1 0.1 0.1 1"/>')
    for piece_type in ("pawn", "rook", "knight", "bishop", "queen", "king"):
        xml.append(f'        <mesh file="chess_{piece_type}.stl" name="chess_{piece_type}_mesh"/>')
    xml.append('    </asset>')
    xml.append('    <worldbody>')
    xml.append('        <light pos="0 0 2" dir="0 0 -1" directional="true"/>')

    table_half_height = TABLE_HEIGHT / 2
    xml.append(
        f'        <geom name="table" type="box" size="0.5 0.5 {table_half_height}" '
        f'pos="{BOARD_CENTER[0]} {BOARD_CENTER[1]} {table_half_height}" '
        f'rgba="0.6 0.6 0.6 1" contype="1" conaffinity="1"/>'
    )
    xml.append(
        f'        <geom name="safety_floor" type="plane" '
        f'size="2 2 0.1" pos="{BOARD_CENTER[0]:.3f} {BOARD_CENTER[1]:.3f} 0.35" '
        f'rgba="0 0 0 0" contype="1" conaffinity="1" />'
    )

    square_gap = 0.004
    square_half_size = (SQUARE_SIZE - square_gap) / 2
    square_thickness = 0.001
    square_half_thickness = square_thickness / 2
    start_y = BOARD_CENTER[1] + 4 * SQUARE_SIZE

    board_half = 4 * SQUARE_SIZE
    underlay_half_thickness = 0.0007
    underlay_z = TABLE_HEIGHT - underlay_half_thickness
    board_collision_half_extent = board_half + 0.040
    # Keep the collision top flush with the visible board surface so pieces
    # start exactly supported instead of spawning in slight penetration.
    board_collision_half_thickness = 0.012
    board_collision_z = TABLE_HEIGHT + 0.001 - board_collision_half_thickness
    xml.append(
        f'        <geom name="board_collision" type="box" '
        f'size="{board_collision_half_extent:.4f} {board_collision_half_extent:.4f} {board_collision_half_thickness:.4f}" '
        f'pos="{BOARD_CENTER[0]:.3f} {BOARD_CENTER[1]:.3f} {board_collision_z:.4f}" '
        f'rgba="0 0 0 0" condim="4" solimp="0.99 0.99 0.001" solref="0.01 1" friction="0.8 0.02 0.002" />'
    )
    xml.append(
        f'        <geom name="board_underlay" type="box" '
        f'size="{board_half:.4f} {board_half:.4f} {underlay_half_thickness:.4f}" '
        f'pos="{BOARD_CENTER[0]:.3f} {BOARD_CENTER[1]:.3f} {underlay_z:.4f}" '
        f'rgba="0.30 0.17 0.08 1" contype="0" conaffinity="0"/>'
    )
    frame_half_thickness = 0.003
    frame_z = TABLE_HEIGHT + frame_half_thickness
    frame_width = 0.012
    outer_half = board_half + frame_width
    frame_strip_half = frame_width / 2
    xml.append(
        f'        <geom name="board_frame_north" type="box" '
        f'size="{outer_half:.4f} {frame_strip_half:.4f} {frame_half_thickness:.4f}" '
        f'pos="{BOARD_CENTER[0]:.3f} {BOARD_CENTER[1] + board_half + frame_strip_half:.3f} {frame_z:.4f}" '
        f'rgba="0.36 0.18 0.04 1" contype="0" conaffinity="0"/>'
    )
    xml.append(
        f'        <geom name="board_frame_south" type="box" '
        f'size="{outer_half:.4f} {frame_strip_half:.4f} {frame_half_thickness:.4f}" '
        f'pos="{BOARD_CENTER[0]:.3f} {BOARD_CENTER[1] - board_half - frame_strip_half:.3f} {frame_z:.4f}" '
        f'rgba="0.36 0.18 0.04 1" contype="0" conaffinity="0"/>'
    )
    xml.append(
        f'        <geom name="board_frame_west" type="box" '
        f'size="{frame_strip_half:.4f} {board_half:.4f} {frame_half_thickness:.4f}" '
        f'pos="{BOARD_CENTER[0] - board_half - frame_strip_half:.3f} {BOARD_CENTER[1]:.3f} {frame_z:.4f}" '
        f'rgba="0.36 0.18 0.04 1" contype="0" conaffinity="0"/>'
    )
    xml.append(
        f'        <geom name="board_frame_east" type="box" '
        f'size="{frame_strip_half:.4f} {board_half:.4f} {frame_half_thickness:.4f}" '
        f'pos="{BOARD_CENTER[0] + board_half + frame_strip_half:.3f} {BOARD_CENTER[1]:.3f} {frame_z:.4f}" '
        f'rgba="0.36 0.18 0.04 1" contype="0" conaffinity="0"/>'
    )

    for i in range(8):
        for j in range(8):
            x = _file_x(i)
            y = _rank_y(j)
            name = f"square_{i}_{j}"
            material = "white_square" if (i + j) % 2 == 1 else "black_square"
            z = TABLE_HEIGHT + square_half_thickness
            xml.append(
                f'        <geom name="{name}" type="box" '
                f'size="{square_half_size} {square_half_size} {square_half_thickness}" '
                f'pos="{x:.3f} {y:.3f} {z:.4f}" material="{material}" contype="0" conaffinity="0"/>'
            )

    label_z = TABLE_HEIGHT + 0.0045
    file_label_far_y = start_y + 0.040
    file_label_near_y = start_y - 8 * SQUARE_SIZE - 0.040
    _append_file_labels(xml, "label_file", file_label_far_y, z=label_z)
    _append_file_labels(xml, "label_file_near", file_label_near_y, z=label_z)

    rank_label_x = _file_x(0) + 0.07
    for rank_idx in range(8):
        y = _rank_y(rank_idx)
        _append_label(xml, f"label_rank_{rank_idx + 1}", str(rank_idx + 1), rank_label_x, y, label_z)

    hitbox_half_extent_xy = 0.014
    hitbox_half_extent_z = 0.012
    # Spawn pieces exactly at grasp height to minimize drop energy during
    # initial simulation settling (prevents lateral drift).
    # Board collision is thickened internally to avoid penetration.
    piece_z = Z_GRASP
    back_line = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]

    mesh_z_offsets = {
        "pawn": 0.0040,
        "rook": 0.0050,
        "knight": 0.0050,
        "bishop": 0.0050,
        "queen": 0.0060,
        "king": 0.0060,
    }

    def add_piece(name, file_idx, rank_idx, color, piece_type):
        x = _file_x(file_idx)
        y = _rank_y(rank_idx)
        material = "white_piece" if color == "white" else "black_piece"
        mesh_z = mesh_z_offsets[piece_type]
        xml.append(f'        <body name="{name}" pos="{x:.3f} {y:.3f} {piece_z:.4f}">')
        xml.append('            <joint type="free" damping="5.0"/>')
        xml.append(
            f'            <geom type="mesh" mesh="chess_{piece_type}_mesh" material="{material}" '
            f'pos="0 0 {mesh_z:.4f}" mass="0" contype="2" conaffinity="2" friction="1.2 0.02 0.002"/>'
        )
        xml.append(
            f'            <geom type="cylinder" size="{hitbox_half_extent_xy} '
            f'{hitbox_half_extent_z}" rgba="1 0 0 0" contype="1" conaffinity="1" condim="3" '
            f'mass="0.08" solimp="0.95 0.99 0.001" solref="0.01 1" friction="1.2 0.02 0.002"/>'
        )
        xml.append(f'            <site name="{name}_site" pos="0 0 0" size="0.02 0.02 0.02" rgba="0 0 0 0"/>')
        xml.append('        </body>')

    for i in range(8):
        add_piece(f"w_pawn_{i + 1}", i, 1, "white", "pawn")
        p_type = back_line[i]
        name = f"w_{p_type}" if p_type in ["king", "queen"] else f"w_{p_type}_{1 if i < 4 else 2}"
        add_piece(name, i, 0, "white", p_type)

    for i in range(8):
        add_piece(f"b_pawn_{i + 1}", i, 6, "black", "pawn")
        p_type = back_line[i]
        name = f"b_{p_type}" if p_type in ["king", "queen"] else f"b_{p_type}_{1 if i < 4 else 2}"
        add_piece(name, i, 7, "black", p_type)

    spare_types = ["queen", "rook", "bishop", "knight"]

    def add_spare_piece(name, x, y, color, piece_type):
        material = "white_piece" if color == "white" else "black_piece"
        mesh_z = mesh_z_offsets[piece_type]
        xml.append(f'        <body name=\"{name}\" pos=\"{x:.3f} {y:.3f} -0.0500\">')
        xml.append('            <joint type="free" damping="5.0"/>')
        xml.append(
            f'            <geom type="mesh" mesh="chess_{piece_type}_mesh" material=\"{material}\" rgba="0 0 0 0" '
            f'pos="0 0 {mesh_z:.4f}" mass="0" contype="0" conaffinity="0" friction="1.2 0.02 0.002"/>'
        )
        xml.append(
            f'            <geom type="cylinder" size=\"{hitbox_half_extent_xy} '
            f'{hitbox_half_extent_z}\" rgba="0 0 0 0" contype="0" conaffinity="0" condim="3" '
            f'mass="0.08" solimp="0.95 0.99 0.001" solref="0.01 1" friction="1.2 0.02 0.002"/>'
        )
        xml.append('        </body>')

    # Promotion spares remain hidden below the table until a pawn is swapped
    # out for its promoted replacement during runtime.
    # Generate 2 of each spare type per color
    for count in (1, 2):
        for idx, p_type in enumerate(spare_types):
            x_offset = (idx * 0.1 - 0.15) + (count - 1) * 0.4
            add_spare_piece(f"w_spare_{p_type}_{count}", BOARD_CENTER[0] + x_offset, BOARD_CENTER[1] + 0.3, "white", p_type)
            add_spare_piece(f"b_spare_{p_type}_{count}", BOARD_CENTER[0] + x_offset, BOARD_CENTER[1] - 0.3, "black", p_type)

    graveyard_half_x, graveyard_half_y, graveyard_half_z = GRAVEYARD_PLATFORM_HALF_EXTENTS
    graveyard_top_z = TABLE_HEIGHT + GRAVEYARD_PLATFORM_HEIGHT
    graveyard_body_z = graveyard_top_z - graveyard_half_z
    tray_center_dx = 0.075
    tray_center_dy = 0.075

    xml.append(
        f'        <body name="white_graveyard" '
        f'pos="{WHITE_GRAVEYARD_ORIGIN[0] + tray_center_dx:.3f} '
        f'{WHITE_GRAVEYARD_ORIGIN[1] + tray_center_dy:.3f} {graveyard_body_z:.3f}">'
    )
    xml.append(
        f'            <geom type="box" size="{graveyard_half_x:.3f} {graveyard_half_y:.3f} '
        f'{graveyard_half_z:.3f}" rgba="0.5 0.5 0.5 0.7"/>'
    )
    xml.append('        </body>')
    xml.append(
        f'        <body name="black_graveyard" '
        f'pos="{BLACK_GRAVEYARD_ORIGIN[0] + tray_center_dx:.3f} '
        f'{BLACK_GRAVEYARD_ORIGIN[1] + tray_center_dy:.3f} {graveyard_body_z:.3f}">'
    )
    xml.append(
        f'            <geom type="box" size="{graveyard_half_x:.3f} {graveyard_half_y:.3f} '
        f'{graveyard_half_z:.3f}" rgba="0.5 0.5 0.5 0.7"/>'
    )
    xml.append('        </body>')
    xml.append('    </worldbody>')
    xml.append('</mujoco>')
    return "\n".join(xml)


if __name__ == "__main__":
    output_path = PROJECT_ROOT / "src" / "assets" / "chess_world.xml"
    output_path.write_text(generate_chess_world(), encoding="utf-8")
    print(f"Generated {output_path}")
