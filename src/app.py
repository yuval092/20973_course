"""
Interactive application entrypoint and UI loop for RoboChess.

This module provides the RoboChessApp class, which manages the main
application window, threading for physics, and the integration
between the GUI and game logic.
"""

import logging
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import messagebox

import chess
import mujoco
import mujoco.viewer

from src.bootstrap import SystemBootstrapper
from src.game_loop import GameLoop
from src.gui.board_panel import BoardPanel
from src.logging_utils import log_event
from src.runtime_guard import RuntimeGuard
from src.game_runtime import GameOrchestrator

logger = logging.getLogger("robo_chess")


class RoboChessApp:
    """Main application class for the RoboChess interactive GUI."""

    def __init__(self):
        """Initialize the RoboChess application."""
        self.systems = None
        self.game_loop = None
        self.root = None
        self.panel = None
        self.move_queue = queue.Queue()
        self.physics_step_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.step_mode_event = threading.Event()

    def print_game_banner(self):
        """Print the startup banner shown when the interactive game launches."""
        print("\n" + "=" * 50)
        print("  ROBOCHESS — Human (White) vs AI (Black)")
        print("  Enter moves in UCI format (e.g., e2e4)")
        print("  Commands: quit, exit, hint, suggest")
        print("=" * 50 + "\n")

    def print_game_over(self, board):
        """Print the game-over summary for the current board result."""
        print("\n" + "=" * 50)
        print("  GAME OVER")
        print(self._build_result_string(board))
        print("=" * 50)

    def _build_result_string(self, board):
        """Build a human-readable result string for game over."""
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            return f"Checkmate! {winner} wins."
        if board.is_stalemate():
            return "Stalemate — draw."
        if board.is_insufficient_material():
            return "Draw — insufficient material."
        if board.can_claim_fifty_moves():
            return "Draw — 50-move rule."
        if board.can_claim_threefold_repetition():
            return "Draw — threefold repetition."
        return f"Game Over. Result: {board.result()}"

    def on_gui_move(self, uci):
        """Callback for moves initiated from the GUI."""
        self.move_queue.put(uci)

    def on_physics_step(self):
        """Callback for manual physics steps requested by the GUI."""
        self.physics_step_queue.put("step")

    def on_close(self):
        """Handle window closure by stopping the worker thread."""
        self.stop_event.set()
        if self.root:
            self.root.destroy()

    def _game_worker(self):
        """The background thread running physical execution and logic."""
        try:
            with mujoco.viewer.launch_passive(self.systems.mj_model,
                                             self.systems.mj_data) as viewer:
                GameOrchestrator.sync_viewer(viewer, self.systems.mj_model,
                                           self.systems.mj_data)
                self.print_game_banner()
                
                while (not self.stop_event.is_set() and viewer.is_running() and
                       not self.systems.manager.board.is_game_over(claim_draw=True)):
                    
                    # 1. Drain Physics Step Queue
                    while not self.physics_step_queue.empty():
                        logger.info("Physics step manually triggered.")
                        self.physics_step_queue.get()
                        mujoco.mj_step(self.systems.mj_model, self.systems.mj_data)
                        self.root.after(0, self.panel.refresh)
                        GameOrchestrator.sync_viewer(viewer, self.systems.mj_model,
                                                   self.systems.mj_data)

                    # 2. Query Game Loop & Execute Turns

                    try:
                        res = None
                        if self.systems.manager.board.turn == chess.WHITE:
                            try:
                                uci = self.move_queue.get(timeout=0.1)
                            except queue.Empty:
                                continue
                                
                            if uci is None:
                                if not self.step_mode_event.is_set():
                                    continue
                                self.root.after(0, self.panel.lock_board)
                                res = self.game_loop.execute_ai_turn(viewer)
                                self.root.after(0, self.panel.unlock_board)
                            else:
                                self.root.after(0, self.panel.lock_board)
                                res = self.game_loop.submit_move(uci, viewer=viewer)
                                self.root.after(0, self.panel.unlock_board)
                        else:  # Black turn (AI)
                            if self.step_mode_event.is_set():
                                try:
                                    uci = self.move_queue.get(timeout=0.1)
                                except queue.Empty:
                                    continue
                                if uci is not None:
                                    continue
                                    
                            self.root.after(0, self.panel.lock_board)
                            res = self.game_loop.execute_ai_turn(viewer)
                            self.root.after(0, self.panel.unlock_board)
                    except Exception:
                        logger.error("Main loop error: %s", traceback.format_exc())
                        break
                    
                    if res is None:
                        continue
                        
                    # 3. Post-Turn UI Refresh & Error Handling
                    self.root.after(0, self.panel.refresh)
                    
                    if res.success:
                        self.root.after(0, self.panel.append_history, res.message)
                    elif res.error:
                        logger.error("Execution Error: %s", traceback.format_exc())
                        self.root.after(0, lambda e=res.error:
                                        messagebox.showerror("Execution Error", str(e)))
                        RuntimeGuard.freeze_on_exception(
                            res.error, 
                            viewer=viewer, 
                            mj_model=self.systems.mj_model, 
                            mj_data=self.systems.mj_data
                        )
                        break
                    else:
                        self.root.after(0, self.panel.set_status, f"Failed: {res.message}")

                if self.stop_event.is_set() or not viewer.is_running():
                    return
                    
                self.print_game_over(self.systems.manager.board)
                result_msg = self._build_result_string(self.systems.manager.board)
                self.root.after(0, lambda msg=result_msg:
                              messagebox.showinfo("Game Over", msg))
                self.root.after(0, self.panel.refresh)
                
                while not self.stop_event.is_set() and viewer.is_running():
                    time.sleep(0.5)
        finally:
            if self.systems and self.systems.manager:
                self.systems.manager.close()

    def _setup_app(self):
        """Set up the application systems and game loop."""
        self.systems = SystemBootstrapper.bootstrap_game_systems()
        log_event(logger, logging.INFO, "viewer_launch")
        
        self.game_loop = GameLoop(self.systems)
        self.root = tk.Tk()
        self.root.title("RoboChess")
        
        self.panel = BoardPanel(
            self.root, self.game_loop, 
            on_move_callback=self.on_gui_move, 
            step_mode_event=self.step_mode_event,
            physics_step_callback=self.on_physics_step
        )
        self.panel.pack(fill=tk.BOTH, expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_game(self):
        """Run the interactive RoboChess application."""
        self._setup_app()

        thread = threading.Thread(target=self._game_worker, daemon=True)
        thread.start()
        self.root.mainloop()


def main():
    """Application entry point."""
    app = RoboChessApp()
    app.start_game()


if __name__ == "__main__":
    main()
