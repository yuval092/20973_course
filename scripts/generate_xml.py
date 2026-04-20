"""
Generate the MuJoCo chess_world.xml scene file.

This script constructs the MJCF XML that defines the chessboard, pieces,
graveyards, labels, and lighting for the RoboChess simulation. It imports
board-geometry constants from src.config so there is a single source of
truth for positions and dimensions.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
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
    "A": [((0.1, 0.0), (0.22, 1.0)), ((0.78, 0.0), (0.9, 1.0)),
          ((0.22, 0.44), (0.78, 0.56)), ((0.22, 0.88), (0.78, 1.0))],
    "B": [((0.1, 0.0), (0.22, 1.0)), ((0.22, 0.88), (0.7, 1.0)),
          ((0.22, 0.44), (0.68, 0.56)), ((0.22, 0.0), (0.7, 0.12)),
          ((0.68, 0.56), (0.8, 0.88)), ((0.68, 0.12), (0.8, 0.44))],
    "C": [((0.22, 0.88), (0.82, 1.0)), ((0.1, 0.12), (0.22, 0.88)),
          ((0.22, 0.0), (0.82, 0.12))],
    "D": [((0.1, 0.0), (0.22, 1.0)), ((0.22, 0.88), (0.68, 1.0)),
          ((0.22, 0.0), (0.68, 0.12)), ((0.68, 0.12), (0.8, 0.88))],
    "E": [((0.1, 0.0), (0.22, 1.0)), ((0.22, 0.88), (0.82, 1.0)),
          ((0.22, 0.44), (0.74, 0.56)), ((0.22, 0.0), (0.82, 0.12))],
    "F": [((0.1, 0.0), (0.22, 1.0)), ((0.22, 0.88), (0.82, 1.0)),
          ((0.22, 0.44), (0.74, 0.56))],
    "G": [((0.22, 0.88), (0.82, 1.0)), ((0.1, 0.12), (0.22, 0.88)),
          ((0.22, 0.0), (0.82, 0.12)), ((0.6, 0.28), (0.82, 0.4)),
          ((0.7, 0.28), (0.82, 0.72))],
    "H": [((0.1, 0.0), (0.22, 1.0)), ((0.78, 0.0), (0.9, 1.0)),
          ((0.22, 0.44), (0.78, 0.56))],
    "1": [((0.44, 0.0), (0.56, 1.0)), ((0.28, 0.82), (0.56, 1.0)),
          ((0.24, 0.0), (0.76, 0.12))],
    "2": [((0.18, 0.88), (0.82, 1.0)), ((0.7, 0.56), (0.82, 0.88)),
          ((0.18, 0.44), (0.82, 0.56)), ((0.18, 0.12), (0.3, 0.44)),
          ((0.18, 0.0), (0.82, 0.12))],
    "3": [((0.18, 0.88), (0.82, 1.0)), ((0.18, 0.44), (0.82, 0.56)),
          ((0.18, 0.0), (0.82, 0.12)), ((0.7, 0.12), (0.82, 0.88))],
    "4": [((0.18, 0.56), (0.7, 0.68)), ((0.18, 0.56), (0.3, 1.0)),
          ((0.7, 0.0), (0.82, 1.0))],
    "5": [((0.18, 0.88), (0.82, 1.0)), ((0.18, 0.56), (0.3, 0.88)),
          ((0.18, 0.44), (0.82, 0.56)), ((0.7, 0.12), (0.82, 0.44)),
          ((0.18, 0.0), (0.82, 0.12))],
    "6": [((0.18, 0.88), (0.82, 1.0)), ((0.18, 0.12), (0.3, 0.88)),
          ((0.18, 0.44), (0.82, 0.56)), ((0.7, 0.12), (0.82, 0.44)),
          ((0.18, 0.0), (0.82, 0.12))],
    "7": [((0.18, 0.88), (0.82, 1.0)), ((0.62, 0.0), (0.74, 0.88))],
    "8": [((0.18, 0.88), (0.82, 1.0)), ((0.18, 0.44), (0.82, 0.56)),
          ((0.18, 0.0), (0.82, 0.12)), ((0.18, 0.12), (0.3, 0.88)),
          ((0.7, 0.12), (0.82, 0.88))],
}


class XMLGenerator:
    """
    Constructs the MuJoCo XML scene for RoboChess.
    """

    def __init__(self):
        """
        Initialize the XMLGenerator.
        """
        self.xml = []
        self.mesh_z_offsets = {
            "pawn": 0.0040,
            "rook": 0.0050,
            "knight": 0.0050,
            "bishop": 0.0050,
            "queen": 0.0060,
            "king": 0.0060,
        }
        self.hitbox_half_extent_xy = 0.014
        self.hitbox_half_extent_z = 0.012

    def _file_x(self, file_idx):
        """
        Compute the X coordinate for file index a=0 through h=7.

        Args:
            file_idx: Integer index of the chess file.

        Returns:
            float: World X coordinate.
        """
        start_x = BOARD_CENTER[0] - 4 * SQUARE_SIZE
        return start_x + (7 - file_idx + 0.5) * SQUARE_SIZE

    def _rank_y(self, rank_idx):
        """
        Compute the Y coordinate for rank index 1=0 through 8=7.

        Args:
            rank_idx: Integer index of the chess rank.

        Returns:
            float: World Y coordinate.
        """
        start_y = BOARD_CENTER[1] + 4 * SQUARE_SIZE
        return start_y - (rank_idx + 0.5) * SQUARE_SIZE

    def _append_label(self, name, glyph, center_x, center_y, z, scale=0.028, depth=0.0012):
        """
        Render a simple block-stroke character as a visible board label.

        Args:
            name: Base name for the label geoms.
            glyph: Character to render.
            center_x: Center X coordinate.
            center_y: Center Y coordinate.
            z: Z height.
            scale: Size scale.
            depth: Thickness of the label.
        """
        rgba = "0.87 0.18 0.18 0.95"
        for idx, ((x0, y0), (x1, y1)) in enumerate(LABEL_STROKES[glyph]):
            cx = center_x + ((x0 + x1) / 2.0 - 0.5) * scale
            cy = center_y + ((y0 + y1) / 2.0 - 0.5) * scale
            sx = max((x1 - x0) * scale / 2.0, 0.0008)
            sy = max((y1 - y0) * scale / 2.0, 0.0008)
            self.xml.append(
                f'        <geom name="{name}_{idx}" type="box" '
                f'size="{sx:.4f} {sy:.4f} {depth / 2:.4f}" '
                f'pos="{cx:.3f} {cy:.3f} {z:.4f}" rgba="{rgba}" '
                'contype="0" conaffinity="0"/>'
            )

    def _append_file_labels(self, prefix, y, z):
        """
        Render file labels along one board edge.

        Args:
            prefix: Name prefix for the labels.
            y: Y coordinate for the labels.
            z: Z height for the labels.
        """
        for file_idx, glyph in enumerate("ABCDEFGH"):
            self._append_label(f"{prefix}_{glyph.lower()}", glyph,
                               self._file_x(file_idx), y, z)

    def _add_piece(self, name, file_idx, rank_idx, color, piece_type):
        """
        Add a single chess piece body to the XML.

        Args:
            name: Body name.
            file_idx: Board file index.
            rank_idx: Board rank index.
            color: 'white' or 'black'.
            piece_type: Type of piece (e.g., 'pawn').
        """
        x = self._file_x(file_idx)
        y = self._rank_y(rank_idx)
        material = "white_piece" if color == "white" else "black_piece"
        mesh_z = self.mesh_z_offsets[piece_type]
        self.xml.append(f'        <body name="{name}" pos="{x:.3f} {y:.3f} {Z_GRASP:.4f}">')
        self.xml.append('            <joint type="free" damping="5.0"/>')
        self.xml.append(
            f'            <geom type="mesh" mesh="chess_{piece_type}_mesh" '
            f'material="{material}" pos="0 0 {mesh_z:.4f}" mass="0" '
            'contype="2" conaffinity="2" friction="1.2 0.02 0.002"/>'
        )
        self.xml.append(
            f'            <geom type="cylinder" size="{self.hitbox_half_extent_xy} '
            f'{self.hitbox_half_extent_z}" rgba="1 0 0 0" contype="1" '
            'conaffinity="5" condim="3" mass="0.08" solimp="0.95 0.99 0.001" '
            'solref="0.01 1" friction="1.2 0.02 0.002"/><!-- conaffinity=5 (1|4) allows collision with table(1) and safety_floor(4) -->'
        )
        self.xml.append(f'            <site name="{name}_site" pos="0 0 0" '
                        'size="0.02 0.02 0.02" rgba="0 0 0 0"/>')
        self.xml.append('        </body>')

    def _add_spare_piece(self, name, x, y, color, piece_type):
        """
        Add a spare piece body (hidden below table).

        Args:
            name: Body name.
            x: World X coordinate.
            y: World Y coordinate.
            color: 'white' or 'black'.
            piece_type: Type of piece.
        """
        material = "white_piece" if color == "white" else "black_piece"
        mesh_z = self.mesh_z_offsets[piece_type]
        self.xml.append(f'        <body name="{name}" pos="{x:.3f} {y:.3f} -0.0500">')
        self.xml.append('            <joint type="free" damping="5.0"/>')
        self.xml.append(
            f'            <geom type="mesh" mesh="chess_{piece_type}_mesh" '
            f'material="{material}" rgba="0 0 0 0" pos="0 0 {mesh_z:.4f}" '
            'mass="0" contype="0" conaffinity="0" friction="1.2 0.02 0.002"/>'
        )
        self.xml.append(
            f'            <geom type="cylinder" size="{self.hitbox_half_extent_xy} '
            f'{self.hitbox_half_extent_z}" rgba="0 0 0 0" contype="0" '
            'conaffinity="0" condim="3" mass="0.08" solimp="0.95 0.99 0.001" '
            'solref="0.01 1" friction="1.2 0.02 0.002"/>'
        )
        self.xml.append('        </body>')

    def generate(self):
        """
        Build the full MuJoCo XML string for the chess scene.

        Returns:
            str: The generated XML content.
        """
        self.xml = []
        self.xml.append('<mujoco model="fetch_chess">')
        self.xml.append('    <include file="fetch.xml"/>')
        self.xml.append('    <visual>')
        self.xml.append('        <quality shadowsize="0"/>')
        self.xml.append('    </visual>')
        self.xml.append('    <asset>')
        self.xml.append('        <material name="white_square" rgba="0.96 0.87 0.70 1"/>')
        self.xml.append('        <material name="black_square" rgba="0.55 0.27 0.07 1"/>')
        self.xml.append('        <material name="white_piece" rgba="1 1 1 1"/>')
        self.xml.append('        <material name="black_piece" rgba="0.1 0.1 0.1 1"/>')
        for pt in ("pawn", "rook", "knight", "bishop", "queen", "king"):
            self.xml.append(f'        <mesh file="chess_{pt}.stl" name="chess_{pt}_mesh"/>')
        self.xml.append('    </asset>')
        self.xml.append('    <worldbody>')
        self.xml.append('        <light pos="0 0 2" dir="0 0 -1" directional="true"/>')

        table_half_h = TABLE_HEIGHT / 2
        self.xml.append(
            f'        <geom name="table" type="box" size="0.5 0.5 {table_half_h}" '
            f'pos="{BOARD_CENTER[0]} {BOARD_CENTER[1]} {table_half_h}" '
            'rgba="0.6 0.6 0.6 1" contype="0" conaffinity="0"/>'
        )
        # Safety floor prevents pieces from falling into oblivion if pushed off the table.
        # We use a box instead of a plane to avoid it being an infinite collision surface.
        # contype=4/conaffinity=4 ensures it ONLY collides with geoms that have a matching bit in their affinity (like our pieces = 5)
        # This prevents the robot arm base (contype=1) from exploding by intersecting with it.
        self.xml.append(
            f'        <geom name="safety_floor" type="box" size="2 2 0.01" '
            f'pos="{BOARD_CENTER[0]:.3f} {BOARD_CENTER[1]:.3f} 0.35" '
            'rgba="0 0 0 0" contype="4" conaffinity="4" />'
        )

        sq_gap = 0.004
        sq_half_size = (SQUARE_SIZE - sq_gap) / 2
        sq_thickness = 0.001
        sq_half_thickness = sq_thickness / 2
        board_half = 4 * SQUARE_SIZE

        col_half_extent = board_half + 0.040
        col_half_thick = 0.012
        col_z = TABLE_HEIGHT + 0.001 - col_half_thick
        # condim=4 adds torsional friction support, important for pieces so they don't spin endlessly on the board
        # solimp & solref govern contact softness and bounce dynamics, standard Gym configurations
        self.xml.append(
            f'        <geom name="board_collision" type="box" '
            f'size="{col_half_extent:.4f} {col_half_extent:.4f} {col_half_thick:.4f}" '
            f'pos="{BOARD_CENTER[0]:.3f} {BOARD_CENTER[1]:.3f} {col_z:.4f}" '
            'rgba="0 0 0 0" condim="4" solimp="0.99 0.99 0.001" solref="0.01 1" '
            'friction="0.8 0.02 0.002" />'
        )
        underlay_half_thick = 0.0007
        underlay_z = TABLE_HEIGHT - underlay_half_thick
        self.xml.append(
            f'        <geom name="board_underlay" type="box" '
            f'size="{board_half:.4f} {board_half:.4f} {underlay_half_thick:.4f}" '
            f'pos="{BOARD_CENTER[0]:.3f} {BOARD_CENTER[1]:.3f} {underlay_z:.4f}" '
            'rgba="0.30 0.17 0.08 1" contype="0" conaffinity="0"/>'
        )

        self._add_frame_geoms(board_half)

        for i in range(8):
            for j in range(8):
                x = self._file_x(i)
                y = self._rank_y(j)
                material = "white_square" if (i + j) % 2 == 1 else "black_square"
                self.xml.append(
                    f'        <geom name="square_{i}_{j}" type="box" '
                    f'size="{sq_half_size} {sq_half_size} {sq_half_thickness}" '
                    f'pos="{x:.3f} {y:.3f} {TABLE_HEIGHT + sq_half_thickness:.4f}" '
                    f'material="{material}" contype="0" conaffinity="0"/>'
                )

        label_z = TABLE_HEIGHT + 0.0045
        start_y = BOARD_CENTER[1] + 4 * SQUARE_SIZE
        self._append_file_labels("label_file", start_y + 0.040, label_z)
        self._append_file_labels("label_file_near", start_y - 8*SQUARE_SIZE - 0.040, label_z)

        rank_x = self._file_x(0) + 0.07
        for r_idx in range(8):
            self._append_label(f"label_rank_{r_idx + 1}", str(r_idx + 1),
                               rank_x, self._rank_y(r_idx), label_z)

        self._add_initial_pieces()
        self._add_all_spares()
        self._add_graveyards()

        self.xml.append('    </worldbody>')
        self.xml.append('</mujoco>')
        return "\n".join(self.xml)

    def _add_frame_geoms(self, board_half):
        """Add the board frame geoms."""
        frame_half_thick = 0.003
        frame_z = TABLE_HEIGHT + frame_half_thick
        frame_w = 0.012
        outer_half = board_half + frame_w
        f_strip_half = frame_w / 2
        bc = BOARD_CENTER
        self.xml.append(
            f'        <geom name="board_frame_north" type="box" '
            f'size="{outer_half:.4f} {f_strip_half:.4f} {frame_half_thick:.4f}" '
            f'pos="{bc[0]:.3f} {bc[1] + board_half + f_strip_half:.3f} {frame_z:.4f}" '
            'rgba="0.36 0.18 0.04 1" contype="0" conaffinity="0"/>'
        )
        self.xml.append(
            f'        <geom name="board_frame_south" type="box" '
            f'size="{outer_half:.4f} {f_strip_half:.4f} {frame_half_thick:.4f}" '
            f'pos="{bc[0]:.3f} {bc[1] - board_half - f_strip_half:.3f} {frame_z:.4f}" '
            'rgba="0.36 0.18 0.04 1" contype="0" conaffinity="0"/>'
        )
        self.xml.append(
            f'        <geom name="board_frame_west" type="box" '
            f'size="{f_strip_half:.4f} {board_half:.4f} {frame_half_thick:.4f}" '
            f'pos="{bc[0] - board_half - f_strip_half:.3f} {bc[1]:.3f} {frame_z:.4f}" '
            'rgba="0.36 0.18 0.04 1" contype="0" conaffinity="0"/>'
        )
        self.xml.append(
            f'        <geom name="board_frame_east" type="box" '
            f'size="{f_strip_half:.4f} {board_half:.4f} {frame_half_thick:.4f}" '
            f'pos="{bc[0] + board_half + f_strip_half:.3f} {bc[1]:.3f} {frame_z:.4f}" '
            'rgba="0.36 0.18 0.04 1" contype="0" conaffinity="0"/>'
        )

    def _add_initial_pieces(self):
        """Add all initial chess pieces to their starting positions."""
        back_line = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
        for i in range(8):
            self._add_piece(f"w_pawn_{i + 1}", i, 1, "white", "pawn")
            p_type = back_line[i]
            name = f"w_{p_type}" if p_type in ["king", "queen"] else f"w_{p_type}_{1 if i < 4 else 2}"
            self._add_piece(name, i, 0, "white", p_type)

            self._add_piece(f"b_pawn_{i + 1}", i, 6, "black", "pawn")
            p_type = back_line[i]
            name = f"b_{p_type}" if p_type in ["king", "queen"] else f"b_{p_type}_{1 if i < 4 else 2}"
            self._add_piece(name, i, 7, "black", p_type)

    def _add_all_spares(self):
        """Add all promotion spare pieces."""
        spare_types = ["queen", "rook", "bishop", "knight"]
        for count in (1, 2):
            for idx, p_type in enumerate(spare_types):
                x_off = (idx * 0.1 - 0.15) + (count - 1) * 0.4
                self._add_spare_piece(f"w_spare_{p_type}_{count}",
                                      BOARD_CENTER[0] + x_off,
                                      BOARD_CENTER[1] + 0.3, "white", p_type)
                self._add_spare_piece(f"b_spare_{p_type}_{count}",
                                      BOARD_CENTER[0] + x_off,
                                      BOARD_CENTER[1] - 0.3, "black", p_type)

    def _add_graveyards(self):
        """Add the graveyard platform bodies."""
        gx, gy, gz = GRAVEYARD_PLATFORM_HALF_EXTENTS
        top_z = GRAVEYARD_PLATFORM_HEIGHT
        body_z = top_z - gz
        dx, dy = 0.075, 0.075

        for name, origin in [("white_graveyard", WHITE_GRAVEYARD_ORIGIN),
                             ("black_graveyard", BLACK_GRAVEYARD_ORIGIN)]:
            self.xml.append(
                f'        <body name="{name}" pos="{origin[0] + dx:.3f} '
                f'{origin[1] + dy:.3f} {body_z:.3f}">'
            )
            self.xml.append(
                f'            <geom type="box" size="{gx:.3f} {gy:.3f} {gz:.3f}" '
                'rgba="0.5 0.5 0.5 0.7"/>'
            )
            self.xml.append('        </body>')


def main():
    """
    Main entry point for the XML generation script.
    """
    generator = XMLGenerator()
    output_path = PROJECT_ROOT / "src" / "assets" / "chess_world.xml"
    output_path.write_text(generator.generate(), encoding="utf-8")
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
