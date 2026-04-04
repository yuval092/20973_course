"""Generate improved Staunton-style STL meshes for chess-piece visuals.

The meshes are **visual-only**.  RoboChess uses the existing invisible box
proxy for collision and manipulation (``contype="1" conaffinity="1"``), so
changing these files does **not** change grasp / contact behaviour.

Each piece is built from stacked cylinders, cones and spheres — all
approximated by triangulated surfaces.  The result is a recognisable
Staunton silhouette that is visually much better than the old box-stacking
approach, while remaining completely self-contained (no external asset
downloads required).

Run from the project root::

    python scripts/generate_chess_piece_meshes.py
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parents[1] / "src" / "assets"

# Number of facets around the circumference for cylinder/cone/sphere
# primitives.  Higher = smoother but larger files.
SEGMENTS = 24


# ─── Primitive mesh generators ───────────────────────────────────────


def _cylinder_facets(
    base_z: float,
    top_z: float,
    base_r: float,
    top_r: float | None = None,
    segments: int = SEGMENTS,
) -> list[tuple]:
    """Generate a cylinder (or cone frustum) centred on Z axis.

    Returns a list of (v1, v2, v3) triangle tuples.
    """
    if top_r is None:
        top_r = base_r
    facets: list[tuple] = []
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * ((i + 1) % segments) / segments
        bx0, by0 = base_r * math.cos(a0), base_r * math.sin(a0)
        bx1, by1 = base_r * math.cos(a1), base_r * math.sin(a1)
        tx0, ty0 = top_r * math.cos(a0), top_r * math.sin(a0)
        tx1, ty1 = top_r * math.cos(a1), top_r * math.sin(a1)
        # Side quads (two triangles)
        facets.append(((bx0, by0, base_z), (bx1, by1, base_z), (tx1, ty1, top_z)))
        facets.append(((bx0, by0, base_z), (tx1, ty1, top_z), (tx0, ty0, top_z)))
        # Bottom cap
        facets.append(((0, 0, base_z), (bx1, by1, base_z), (bx0, by0, base_z)))
        # Top cap
        facets.append(((0, 0, top_z), (tx0, ty0, top_z), (tx1, ty1, top_z)))
    return facets


def _sphere_facets(
    center_z: float,
    radius: float,
    rings: int = 10,
    segments: int = SEGMENTS,
) -> list[tuple]:
    """Generate a UV sphere centred on the Z axis."""
    facets: list[tuple] = []
    for j in range(rings):
        phi0 = math.pi * j / rings
        phi1 = math.pi * (j + 1) / rings
        for i in range(segments):
            theta0 = 2.0 * math.pi * i / segments
            theta1 = 2.0 * math.pi * ((i + 1) % segments) / segments
            v00 = (
                radius * math.sin(phi0) * math.cos(theta0),
                radius * math.sin(phi0) * math.sin(theta0),
                center_z + radius * math.cos(phi0),
            )
            v01 = (
                radius * math.sin(phi0) * math.cos(theta1),
                radius * math.sin(phi0) * math.sin(theta1),
                center_z + radius * math.cos(phi0),
            )
            v10 = (
                radius * math.sin(phi1) * math.cos(theta0),
                radius * math.sin(phi1) * math.sin(theta0),
                center_z + radius * math.cos(phi1),
            )
            v11 = (
                radius * math.sin(phi1) * math.cos(theta1),
                radius * math.sin(phi1) * math.sin(theta1),
                center_z + radius * math.cos(phi1),
            )
            if j > 0:
                facets.append((v00, v01, v11))
            if j < rings - 1:
                facets.append((v00, v11, v10))
    return facets


def _box_facets(center, size):
    """Generate a box (cuboid) mesh for details like the king's cross."""
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    vertices = [
        (cx - sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz),
        (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz),
        (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz),
        (cx - sx, cy + sy, cz + sz),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    ]
    return [(vertices[a], vertices[b], vertices[c]) for a, b, c in faces]


# ─── STL writer ──────────────────────────────────────────────────────


def _write_binary_stl(path: Path, facets):
    """Write a binary STL file from a list of triangle tuples."""
    header = b"RoboChess chess piece mesh".ljust(80, b"\0")
    payload = bytearray(header)
    payload.extend(struct.pack("<I", len(facets)))
    for tri in facets:
        payload.extend(struct.pack("<3f", 0.0, 0.0, 0.0))  # normal placeholder
        for vx, vy, vz in tri:
            payload.extend(struct.pack("<3f", vx, vy, vz))
        payload.extend(struct.pack("<H", 0))  # attribute byte count
    path.write_bytes(payload)


# ─── Piece geometries ────────────────────────────────────────────────
# All pieces are centred on the XY origin.  The Z origin (z=0) is at
# the grasp centre; negative Z extends downward toward the base.


def _pawn_facets() -> list[tuple]:
    """Pawn: wide base, tapered stem, round ball on top."""
    f: list[tuple] = []
    # Base platform
    f.extend(_cylinder_facets(-0.016, -0.012, 0.014))
    # Stem — gentle taper
    f.extend(_cylinder_facets(-0.012, 0.004, 0.009, 0.006))
    # Collar ring
    f.extend(_cylinder_facets(0.002, 0.006, 0.010))
    # Head ball
    f.extend(_sphere_facets(0.012, 0.007, rings=8))
    return f


def _rook_facets() -> list[tuple]:
    """Rook: wide base, columnar body, battlements on top."""
    f: list[tuple] = []
    # Base platform
    f.extend(_cylinder_facets(-0.017, -0.012, 0.015))
    # Lower body
    f.extend(_cylinder_facets(-0.012, 0.006, 0.010, 0.011))
    # Upper collar
    f.extend(_cylinder_facets(0.006, 0.010, 0.012))
    # Battlement rim
    f.extend(_cylinder_facets(0.010, 0.018, 0.013))
    # Inner hollow suggestion (slightly recessed top)
    f.extend(_cylinder_facets(0.016, 0.018, 0.009))
    # Crenellations (small boxes around the rim)
    for i in range(6):
        angle = 2.0 * math.pi * i / 6
        cx = 0.011 * math.cos(angle)
        cy = 0.011 * math.sin(angle)
        f.extend(_box_facets((cx, cy, 0.022), (0.006, 0.006, 0.006)))
    return f


def _knight_facets() -> list[tuple]:
    """Knight: wide base, angled body suggesting a horse head profile."""
    f: list[tuple] = []
    # Base platform
    f.extend(_cylinder_facets(-0.017, -0.012, 0.015))
    # Lower body
    f.extend(_cylinder_facets(-0.012, 0.002, 0.010, 0.009))
    # Neck — tilted cylinder approximated by offset tapering
    f.extend(_cylinder_facets(0.002, 0.014, 0.009, 0.007))
    # Head block — angled forward
    f.extend(_box_facets((0.003, 0.0, 0.018), (0.016, 0.012, 0.012)))
    # Snout
    f.extend(_box_facets((0.010, 0.0, 0.014), (0.012, 0.008, 0.006)))
    # Ears
    f.extend(_box_facets((-0.002, 0.0, 0.026), (0.006, 0.004, 0.006)))
    return f


def _bishop_facets() -> list[tuple]:
    """Bishop: wide base, tapered body, mitre-shaped top with slot."""
    f: list[tuple] = []
    # Base platform
    f.extend(_cylinder_facets(-0.017, -0.012, 0.015))
    # Lower body
    f.extend(_cylinder_facets(-0.012, 0.002, 0.010, 0.008))
    # Collar
    f.extend(_cylinder_facets(0.000, 0.004, 0.011))
    # Mitre taper
    f.extend(_cylinder_facets(0.004, 0.020, 0.010, 0.004))
    # Mitre ball tip
    f.extend(_sphere_facets(0.023, 0.004, rings=6))
    # Slot across mitre (thin box)
    f.extend(_box_facets((0.0, 0.0, 0.016), (0.016, 0.002, 0.008)))
    return f


def _queen_facets() -> list[tuple]:
    """Queen: wide base, elegant taper, crown with small ball."""
    f: list[tuple] = []
    # Base platform
    f.extend(_cylinder_facets(-0.018, -0.012, 0.016))
    # Lower body — gentle taper
    f.extend(_cylinder_facets(-0.012, 0.004, 0.011, 0.009))
    # Collar ring
    f.extend(_cylinder_facets(0.002, 0.006, 0.012))
    # Upper body — elegant taper
    f.extend(_cylinder_facets(0.006, 0.018, 0.010, 0.006))
    # Crown rim
    f.extend(_cylinder_facets(0.016, 0.020, 0.010))
    # Crown points (small cones around the rim)
    for i in range(8):
        angle = 2.0 * math.pi * i / 8
        cx = 0.008 * math.cos(angle)
        cy = 0.008 * math.sin(angle)
        f.extend(_box_facets((cx, cy, 0.024), (0.003, 0.003, 0.006)))
    # Crown ball
    f.extend(_sphere_facets(0.028, 0.004, rings=6))
    return f


def _king_facets() -> list[tuple]:
    """King: tallest piece, wide base, prominent cross on top."""
    f: list[tuple] = []
    # Base platform
    f.extend(_cylinder_facets(-0.018, -0.012, 0.016))
    # Lower body
    f.extend(_cylinder_facets(-0.012, 0.004, 0.011, 0.009))
    # Collar
    f.extend(_cylinder_facets(0.002, 0.006, 0.012))
    # Upper body — straight column
    f.extend(_cylinder_facets(0.006, 0.020, 0.010, 0.008))
    # Crown rim
    f.extend(_cylinder_facets(0.018, 0.022, 0.010))
    # Cross — vertical bar
    f.extend(_box_facets((0.0, 0.0, 0.032), (0.004, 0.004, 0.018)))
    # Cross — horizontal bar
    f.extend(_box_facets((0.0, 0.0, 0.036), (0.016, 0.004, 0.004)))
    return f


# ─── Entry point ─────────────────────────────────────────────────────

PIECE_GENERATORS = {
    "pawn": _pawn_facets,
    "rook": _rook_facets,
    "knight": _knight_facets,
    "bishop": _bishop_facets,
    "queen": _queen_facets,
    "king": _king_facets,
}


def main():
    """Generate all 6 chess piece STL files to the assets directory."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for piece_name, generator in PIECE_GENERATORS.items():
        facets = generator()
        path = ASSET_DIR / f"chess_{piece_name}.stl"
        _write_binary_stl(path, facets)
        print(f"Generated {path}  ({len(facets)} triangles)")


if __name__ == "__main__":
    main()
