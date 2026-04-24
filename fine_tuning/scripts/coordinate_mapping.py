import numpy as np

class ChessCoordinateMapper:
    """
    Maps chess notation (e.g., 'a1', 'h8') to 0.5m x 0.5m board world coordinates.
    """
    
    # These MUST match the values in your ChessFetchEnv exactly.
    TABLE_CENTER_XY = np.array([0.88, 0.2641])
    TABLE_HALF_X    = 0.32
    TABLE_HALF_Y    = 0.32
    TABLE_SURFACE_Z = 0.4
    
    # Actual playable board area is slightly smaller than the table
    # Standard board size (8x70mm) is 0.56m. Let's center it.
    PLAYABLE_SIZE = 0.56
    
    def __init__(self):
        # Top-left corner of the board (a8)
        self.top_left = self.TABLE_CENTER_XY - (self.PLAYABLE_SIZE / 2.0)
        
    def notation_to_world(self, notation: str):
        """Convert 'a1' -> [X, Y, Z]"""
        if len(notation) != 2:
            raise ValueError("Notation must be 2 characters (e.g., 'a1')")
            
        col_char = notation[0].lower()
        row_char = notation[1]
        
        if col_char not in "abcdefgh" or row_char not in "12345678":
            raise ValueError("Invalid chess notation.")
            
        col_idx = ord(col_char) - ord('a')   # a=0, h=7
        row_idx = int(row_char) - 1          # 1=0, 8=7
        
        # In our MuJoCo scene:
        # X+ points toward the front of the robot (table is at 1.3)
        # Y+ points toward the left of the robot (table is at 0.75)
        
        # We define a1 as the corner farthest from the robot's base (higher X) 
        # and to the robot's right (lower Y). Let's align logically:
        # File (a-h) maps to Y (left to right from robot's POV)
        # Rank (1-8) maps to X (close to far from robot's POV)
        
        square_size = self.PLAYABLE_SIZE / 8.0
        
        # Y mapping: +Y is left in MuJoCo. So 'a' (left) should be max Y.
        y = self.TABLE_CENTER_XY[1] + (self.PLAYABLE_SIZE / 2.0) - (col_idx + 0.5) * square_size
        
        # X mapping: '1' is close (-X from table center), '8' is far (+X)
        x = self.TABLE_CENTER_XY[0] - (self.PLAYABLE_SIZE / 2.0) + (row_idx + 0.5) * square_size
        
        return np.array([x, y, self.TABLE_SURFACE_Z])

    def print_grid(self):
        """Print the world coordinates for every square."""
        print(f"{'Sq':4} | {'X':6} | {'Y':6} | {'Z':6}")
        print("-" * 30)
        for rank in "87654321":
            for file in "abcdefgh":
                pos = self.notation_to_world(file + rank)
                print(f"{file+rank:4} | {pos[0]:.3f} | {pos[1]:.3f} | {pos[2]:.3f}")

if __name__ == "__main__":
    mapper = ChessCoordinateMapper()
    print("Example Mappings:")
    for sq in ["a1", "h8", "e4", "d5"]:
        print(f"{sq}: {mapper.notation_to_world(sq)}")
    print("\nFull Grid:")
    mapper.print_grid()
