# sudoku_game.py - Core Sudoku Game Logic & Constraint Management

import numpy as np
import random
from typing import List, Tuple, Optional, Set
from config import game_config, DIFFICULTY_LEVELS

class SudokuGame:
    """Core Sudoku game engine with constraint validation"""
    
    def __init__(self, difficulty: str = 'medium'):
        """
        Initialize Sudoku game
        
        Args:
            difficulty: 'easy', 'medium', or 'hard'
        """
        self.board_size = game_config.BOARD_SIZE
        self.subgrid_size = game_config.SUBGRID_SIZE
        self.board = np.zeros((9, 9), dtype=np.int32)
        self.original_board = np.zeros((9, 9), dtype=np.int32)
        self.solution = np.zeros((9, 9), dtype=np.int32)
        self.difficulty = difficulty
        
        self._generate_puzzle()
    
    def _generate_puzzle(self):
        """Generate a random valid Sudoku puzzle"""
        # Generate complete valid solution
        self.solution = self._generate_complete_board()
        
        # Copy to playable board
        self.board = self.solution.copy()
        
        # Remove numbers based on difficulty
        givens = DIFFICULTY_LEVELS.get(self.difficulty, 30)
        cells_to_remove = 81 - givens
        
        cells = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(cells)
        
        for i, j in cells[:cells_to_remove]:
            self.board[i, j] = 0
        
        self.original_board = self.board.copy()
    
    def _generate_complete_board(self) -> np.ndarray:
        """Generate a valid complete 9x9 Sudoku board using backtracking"""
        board = np.zeros((9, 9), dtype=np.int32)
        self._solve_backtrack(board)
        return board
    
    def _solve_backtrack(self, board: np.ndarray) -> bool:
        """Backtracking solver for board generation"""
        for i in range(9):
            for j in range(9):
                if board[i, j] == 0:
                    # Try digits 1-9 in random order
                    digits = list(range(1, 10))
                    random.shuffle(digits)
                    
                    for digit in digits:
                        if self.is_valid_move(board, i, j, digit):
                            board[i, j] = digit
                            if self._solve_backtrack(board):
                                return True
                            board[i, j] = 0
                    return False
        return True
    
    def is_valid_move(self, board: np.ndarray, row: int, col: int, digit: int) -> bool:
        """
        Check if placing digit at (row, col) is valid
        
        Args:
            board: Sudoku board
            row: Row index (0-8)
            col: Column index (0-8)
            digit: Digit to place (1-9)
        
        Returns:
            True if move is valid, False otherwise
        """
        if digit < 1 or digit > 9:
            return False
        
        # Check row constraint
        if digit in board[row, :]:
            return False
        
        # Check column constraint
        if digit in board[:, col]:
            return False
        
        # Check 3x3 subgrid constraint
        subgrid_row, subgrid_col = (row // 3) * 3, (col // 3) * 3
        subgrid = board[subgrid_row:subgrid_row+3, subgrid_col:subgrid_col+3]
        if digit in subgrid:
            return False
        
        return True
    
    def place_digit(self, row: int, col: int, digit: int,
                    force: bool = False) -> bool:
        """
        Place a digit on the board
        
        Args:
            row: Row index
            col: Column index
            digit: Digit to place (1-9 or 0 to clear)
            force: If True, skip validation (still can't modify givens)
        
        Returns:
            True if placement is valid, False otherwise
        """
        # Can't modify given cells
        if self.original_board[row, col] != 0:
            return False
        
        # Allow clearing cells
        if digit == 0:
            self.board[row, col] = 0
            return True
        
        if force or self.is_valid_move(self.board, row, col, digit):
            self.board[row, col] = digit
            return True
        
        return False
    
    def get_possible_values(self, row: int, col: int) -> Set[int]:
        """
        Get all valid digits that can be placed at (row, col)
        
        Args:
            row: Row index
            col: Column index
        
        Returns:
            Set of valid digits (1-9)
        """
        if self.board[row, col] != 0:
            return set()
        
        possible = set(range(1, 10))
        
        # Remove values in row
        possible -= set(self.board[row, :])
        
        # Remove values in column
        possible -= set(self.board[:, col])
        
        # Remove values in 3x3 subgrid
        subgrid_row, subgrid_col = (row // 3) * 3, (col // 3) * 3
        subgrid = self.board[subgrid_row:subgrid_row+3, subgrid_col:subgrid_col+3]
        possible -= set(subgrid[subgrid > 0])
        
        return possible
    
    def is_complete(self) -> bool:
        """Check if board is completely filled"""
        return not (self.board == 0).any()
    
    def is_solved(self) -> bool:
        """Check if board is solved correctly"""
        if not self.is_complete():
            return False
        
        return np.array_equal(self.board, self.solution)
    
    def get_conflicts(self) -> List[Tuple[int, int]]:
        """
        Get list of cells with conflicts (duplicates in row/col/subgrid)
        
        Returns:
            List of (row, col) tuples with conflicts
        """
        conflicts = []
        
        for i in range(9):
            for j in range(9):
                if self.board[i, j] == 0:
                    continue
                
                digit = self.board[i, j]
                
                # Check row for duplicates
                row_count = np.sum(self.board[i, :] == digit)
                if row_count > 1:
                    conflicts.append((i, j))
                    continue
                
                # Check column for duplicates
                col_count = np.sum(self.board[:, j] == digit)
                if col_count > 1:
                    conflicts.append((i, j))
                    continue
                
                # Check subgrid for duplicates
                subgrid_row, subgrid_col = (i // 3) * 3, (j // 3) * 3
                subgrid = self.board[subgrid_row:subgrid_row+3, subgrid_col:subgrid_col+3]
                if np.sum(subgrid == digit) > 1:
                    conflicts.append((i, j))
        
        return conflicts
    
    def solve_hint(self, row: int, col: int) -> Optional[int]:
        """
        Get a hint for a specific cell
        
        Args:
            row: Row index
            col: Column index
        
        Returns:
            Correct digit from solution, or None if cell is given/already filled
        """
        if self.board[row, col] != 0 or self.original_board[row, col] != 0:
            return None
        
        return int(self.solution[row, col])
    
    def reset(self):
        """Reset board to original state"""
        self.board = self.original_board.copy()
    
    def get_board_state(self) -> np.ndarray:
        """Get current board state as 9x9 array"""
        return self.board.copy()
    
    def get_encoded_state(self) -> np.ndarray:
        """
        Get board state as 9x9x10 tensor for neural network
        Encodes: digit presence at each position + empty cell indicator
        
        Returns:
            9x9x10 array where each cell has 10 channels (0-9)
        """
        state = np.zeros((9, 9, 10), dtype=np.float32)
        
        for i in range(9):
            for j in range(9):
                digit = self.board[i, j]
                if digit == 0:
                    state[i, j, 0] = 1.0  # Empty cell
                else:
                    state[i, j, digit] = 1.0  # Digit present
        
        return state
    
    def copy(self) -> 'SudokuGame':
        """Create a deep copy of the game state"""
        game_copy = SudokuGame.__new__(SudokuGame)
        game_copy.board = self.board.copy()
        game_copy.original_board = self.original_board.copy()
        game_copy.solution = self.solution.copy()
        game_copy.board_size = self.board_size
        game_copy.subgrid_size = self.subgrid_size
        game_copy.difficulty = self.difficulty
        return game_copy


class SudokuValidator:
    """Static validation utilities"""
    
    @staticmethod
    def is_valid_board(board: np.ndarray) -> bool:
        """Check if board satisfies all Sudoku constraints"""
        # Check rows
        for i in range(9):
            row = board[i, :]
            if len(set(row[row > 0])) != len(row[row > 0]):
                return False
        
        # Check columns
        for j in range(9):
            col = board[:, j]
            if len(set(col[col > 0])) != len(col[col > 0]):
                return False
        
        # Check 3x3 subgrids
        for subgrid_row in range(0, 9, 3):
            for subgrid_col in range(0, 9, 3):
                subgrid = board[subgrid_row:subgrid_row+3, subgrid_col:subgrid_col+3]
                subgrid_flat = subgrid[subgrid > 0]
                if len(set(subgrid_flat)) != len(subgrid_flat):
                    return False
        
        return True
