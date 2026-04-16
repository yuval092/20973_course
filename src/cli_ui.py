from src.game_loop import GameLoop
import chess
import logging

logger = logging.getLogger("robo_chess")

class CliUI:
    def __init__(self, game_loop: GameLoop):
        self.game_loop = game_loop

    def print_banner(self):
        print("\n" + "=" * 50)
        print("  ROBOCHESS — Human (White) vs AI (Black)")
        print("  Enter moves in UCI format (e.g., e2e4)")
        print("  Commands: quit, exit, hint, suggest")
        print("=" * 50 + "\n")

    def print_state(self):
        state = self.game_loop.get_state()
        if state.is_game_over:
            print("\n" + "=" * 50)
            print("  GAME OVER")
            if state.is_checkmate:
                winner = "Black" if state.board.turn == chess.WHITE else "White"
                print(f"  Checkmate! {winner} wins.")
            elif state.is_stalemate:
                print("  Stalemate — draw.")
            elif state.is_insufficient_material:
                print("  Draw — insufficient material.")
            elif state.is_fifty_moves:
                print("  Draw — 50-move rule.")
            elif state.is_threefold:
                print("  Draw — threefold repetition.")
            else:
                print(f"  Result: {state.board.result()}")
            print("=" * 50)
            return

        if state.is_check:
            print("\n  *** CHECK! ***\n")

        print(f"\n{'White' if state.board.turn == chess.WHITE else 'Black'}'s turn")
        print(state.board)

    def prompt_move(self) -> str:
        """Prompt the user for a move. Return 'quit' if the user wants to exit."""
        while True:
            move_uci = input("Enter move: ").strip()

            if move_uci.lower() in ("quit", "exit"):
                return "quit"

            if move_uci.lower() in ("hint", "suggest"):
                try:
                    hint = self.game_loop.request_hint()
                    print(f"Engine suggestion: {hint}")
                except Exception as e:
                    print(f"Engine suggestion failed: {e}")
                continue
                
            if not self.game_loop.systems.manager.validate_move(move_uci):
                print(f"Illegal move: {move_uci}. Try again.")
                continue

            return move_uci
