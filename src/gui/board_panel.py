import tkinter as tk
from tkinter import messagebox
import chess
from src.game_loop import GameLoop

class BoardPanel(tk.Frame):
    def __init__(self, parent, game_loop: GameLoop, on_move_callback=None):
        super().__init__(parent)
        self.parent = parent
        self.game_loop = game_loop
        self.on_move_callback = on_move_callback
        self.selected_square = None
        self.auto_play = tk.BooleanVar(value=False)
        self.pieces = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk_btn(toolbar, "Hint", self._request_hint).pack(side=tk.LEFT, padx=2)
        ttk_btn(toolbar, "Next Turn (AutoPlay Step)", self._step_auto).pack(side=tk.LEFT, padx=2)
        tk.Checkbutton(toolbar, text="Auto-Play Mode", variable=self.auto_play).pack(side=tk.LEFT, padx=2)
        
        # Board
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
                
        # Status
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
        
        # Move History
        self.history_list = tk.Listbox(self, height=5)
        self.history_list.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    def _on_square_click(self, square: int):
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
            
            # Auto-promotion to Queen for simplicity in GUI
            if state.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                if (state.board.turn == chess.WHITE and chess.square_rank(square) == 7) or \
                   (state.board.turn == chess.BLACK and chess.square_rank(square) == 0):
                    move_uci += "q"
                    
            if state.board.parse_san(move_uci) in state.board.legal_moves:
                if self.on_move_callback:
                    self.on_move_callback(move_uci)
            else:
                self.status_var.set("Illegal move")
                
            self.selected_square = None
            self.refresh()

    def _request_hint(self):
        try:
            hint = self.game_loop.request_hint()
            self.status_var.set(f"Hint: {hint}")
        except Exception as e:
            messagebox.showerror("Engine Error", str(e))

    def _step_auto(self):
        if self.on_move_callback:
            self.on_move_callback(None) # Signal AI turn or auto fallback

    def append_history(self, msg: str):
        self.history_list.insert(tk.END, msg)
        self.history_list.yview(tk.END)

    def refresh(self):
        state = self.game_loop.get_state()
        board = state.board
        
        # Reset colors
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, rank)
                color = '#eee' if (rank + file) % 2 != 0 else '#8B4513'
                self.squares[sq].configure(bg=color, text="")
                
                piece = board.piece_at(sq)
                if piece:
                    self.squares[sq].configure(text=self.pieces[piece.symbol()])
                    # Color black pieces explicitly if needed, but unicode covers it.
                    
        # Update status
        st = "White to move" if board.turn == chess.WHITE else "Black to move"
        if state.is_check:
            st += " | CHECK!"
        if state.is_game_over:
            st = "GAME OVER"
        self.status_var.set(st)

def ttk_btn(parent, text, command):
    return tk.Button(parent, text=text, command=command)
