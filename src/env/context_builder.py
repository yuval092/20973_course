import numpy as np
import mujoco
from src.config import PIECE_NAMES, BOARD_CENTER, SQUARE_SIZE, TABLE_HEIGHT

PIECE_TYPE_MAP = {
    "pawn": 0, "rook": 1, "knight": 2,
    "bishop": 3, "queen": 4, "king": 5,
}

def build_context(model, data, episode, is_grasped: bool) -> np.ndarray:
    """Build the 18-dim chess-scene context vector (Section 4)."""
    ctx = np.zeros(18, dtype=np.float32)

    piece_pos  = data.body(episode.piece_name).xpos.copy()
    dest_pos   = episode.dest_pos
    src_square = episode.src_square   # e.g. "e2"
    dst_square = episode.dst_square

    # 0–2: destination position
    ctx[0:3] = dest_pos.astype(np.float32)

    # 3–5: piece → destination vector
    ctx[3:6] = (dest_pos - piece_pos).astype(np.float32)

    # 6–9: normalized board coordinates
    ctx[6]  = (ord(src_square[0]) - ord('a')) / 7.0
    ctx[7]  = (int(src_square[1]) - 1) / 7.0
    ctx[8]  = (ord(dst_square[0]) - ord('a')) / 7.0
    ctx[9]  = (int(dst_square[1]) - 1) / 7.0

    # 10: normalized transport distance
    ctx[10] = np.linalg.norm(dest_pos[:2] - episode.src_pos[:2]) / 0.40

    # 11: is_grasped
    ctx[11] = 1.0 if is_grasped else 0.0

    # 12: is_capture
    ctx[12] = 1.0 if episode.is_capture else 0.0

    # 13: piece type
    # Expecting piece_name like "w_pawn_1" or "b_king"
    parts = episode.piece_name.split("_")
    piece_type_str = parts[1] if len(parts) > 1 else "pawn"
    ctx[13] = PIECE_TYPE_MAP.get(piece_type_str, 0) / 5.0

    # 14–17: two nearest neighbor offsets
    neighbors = _get_neighbor_offsets(data, episode.piece_name, piece_pos)
    if len(neighbors) > 0:
        ctx[14:16] = np.clip(neighbors[0], -0.15, 0.15).astype(np.float32)
    if len(neighbors) > 1:
        ctx[16:18] = np.clip(neighbors[1], -0.15, 0.15).astype(np.float32)

    return ctx


def _get_neighbor_offsets(data, target_name, target_pos):
    dists = []
    # PIECE_NAMES should be available in config
    for name in PIECE_NAMES:
        if name == target_name:
            continue
        try:
            pos = data.body(name).xpos.copy()
        except KeyError:
            continue
            
        if pos[2] < TABLE_HEIGHT - 0.05:    # below table = hidden spare, skip
            continue
        d = np.linalg.norm(pos[:2] - target_pos[:2])
        dists.append((d, pos[:2] - target_pos[:2]))
    dists.sort(key=lambda x: x[0])
    return [v for (_, v) in dists[:2]]
