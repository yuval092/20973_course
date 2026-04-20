"""
GUI board panel for the robot chess player.

This module provides a Tkinter-based user interface for displaying the
chess board, managing user interactions, and visualizing the game state.
"""

import tkinter as tk
from tkinter import messagebox
import chess


class BoardPanel(tk.Frame):
    """
    Tkinter frame containing the chess board and control buttons.

    This class handles rendering the board, processing mouse clicks for moves,
    and interfacing with the game loop to request hints or trigger AI moves.
    """

    def __init__(self, parent, game_loop, on_move_callback=None, 
                 step_mode_event=None, physics_step_callback=None):
        """
        Initialize the board panel.

        Args:
            parent: The parent Tkinter widget.
            game_loop: The GameLoop instance to interact with.
            on_move_callback: Optional callback for move events.
            step_mode_event: Optional threading event for step-mode toggle.
            physics_step_callback: Optional callback for manual physics steps.
        """
        super().__init__(parent)
        self.parent = parent
        self.game_loop = game_loop
        self.on_move_callback = on_move_callback
        self.step_mode_event = step_mode_event
        self.physics_step_callback = physics_step_callback
        
        self.selected_square = None
        self.auto_play_var = tk.BooleanVar(value=False)
        self.pieces = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        """Construct the UI widgets."""
        # Top toolbar.
        toolbar = tk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self._create_button(toolbar, "Hint", self._request_hint).pack(side=tk.LEFT, padx=2)
        self._create_button(toolbar, "Step (Logic)", self._step_auto).pack(side=tk.LEFT, padx=2)
        self._create_button(toolbar, "Step (Physics)", self._step_physics).pack(side=tk.LEFT, padx=2)
        
        tk.Checkbutton(
            toolbar, 
            text="Step-by-Step Mode", 
            variable=self.auto_play_var,
            command=self._on_step_mode_toggle
        ).pack(side=tk.LEFT, padx=2)
        
        # Board grid.
        self.board_frame = tk.Frame(self, bg='black')
        self.board_frame.pack(side=tk.TOP, pady=10)
        self.squares = {}
        
        for rank in range(7, -1, -1):
            for file in range(8):
                sq_index = chess.square(file, rank)
                color = '#eee' if (rank + file) % 2 != 0 else '#8B4513'
                btn = tk.Button(self.board_frame, text="", font=("Courier", 24),
                                width=2, height=1, bg=color, activebackground='#aaa',
                                command=lambda s=sq_index: self._on_square_click(s))
                btn.grid(row=7-rank, column=file)
                self.squares[sq_index] = btn
                
        # Status bar.
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
        
        # Move history list.
        self.history_list = tk.Listbox(self, height=5)
        self.history_list.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    def _on_square_click(self, square):
        """
        Handle clicks on board squares.

        Args:
            square: The chess.Square index that was clicked.
        """
        state = self.game_loop.get_state()
        if state.is_game_over:
            return
            
        if self.selected_square is None:
            piece = state.board.piece_at(square)
            if piece and piece.color == state.board.turn:
                self.selected_square = square
                self.squares[square].configure(bg='#ffb')
        else:
            move_uci = chess.square_name(self.selected_square) + chess.square_name(square)
            
            # Auto-promotion to Queen.
            piece = state.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN:
                if (state.board.turn == chess.WHITE and chess.square_rank(square) == 7) or \
                   (state.board.turn == chess.BLACK and chess.square_rank(square) == 0):
                    move_uci += "q"
                    
            try:
                move = chess.Move.from_uci(move_uci)
                if move in state.board.legal_moves:
                    if self.on_move_callback:
                        self.on_move_callback(move_uci)
                else:
                    self.status_var.set("Illegal move")
            except ValueError:
                self.status_var.set("Invalid move format")
                
            self.selected_square = None
            self.refresh()

    def _request_hint(self):
        """Request a hint from the chess engine."""
        try:
            hint = self.game_loop.request_hint()
            self.status_var.set(f"Hint: {hint}")
        except Exception as e:
            messagebox.showerror("Engine Error", str(e))

    def _step_auto(self):
        """Trigger an automatic turn execution."""
        if self.on_move_callback:
            self.on_move_callback(None)

    def _step_physics(self):
        """Trigger a manual physics step."""
        if self.physics_step_callback:
            self.physics_step_callback()

    def _on_step_mode_toggle(self):
        """Handle toggle of the step-by-step mode."""
        if self.step_mode_event:
            if self.auto_play_var.get():
                self.step_mode_event.set()
            else:
                self.step_mode_event.clear()

    def set_status(self, msg):
        """Update the status bar text."""
        self.status_var.set(msg)

    def lock_board(self):
        """Disable interaction with the board squares."""
        for btn in self.squares.values():
            btn.configure(state=tk.DISABLED)

    def unlock_board(self):
        """Enable interaction with the board squares."""
        for btn in self.squares.values():
            btn.configure(state=tk.NORMAL)

    def append_history(self, msg):
        """
        Add a message to the history listbox.

        Args:
            msg: The string message to append.
        """
        self.history_list.insert(tk.END, msg)
        self.history_list.yview(tk.END)

    def refresh(self):
        """Update the visual state of the board from the current game state."""
        state = self.game_loop.get_state()
        board = state.board
        
        # Update colors and piece symbols.
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, rank)
                color = '#eee' if (rank + file) % 2 != 0 else '#8B4513'
                self.squares[sq].configure(bg=color, text="")
                
                piece = board.piece_at(sq)
                if piece:
                    self.squares[sq].configure(text=self.pieces[piece.symbol()])
                    
        # Update status message.
        st = "White to move" if board.turn == chess.WHITE else "Black to move"
        if state.is_check:
            st += " | CHECK!"
        if state.is_game_over:
            st = "GAME OVER"
        self.status_var.set(st)

    def _create_button(self, parent, text, command):
        """Helper to create a standard button."""
        return tk.Button(parent, text=text, command=command)
