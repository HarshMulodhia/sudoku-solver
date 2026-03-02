"""Deterministic Backtracking Solver for Sudoku"""

import time
import numpy as np
from typing import Optional, Tuple, List, Set

from sudoku_game import SudokuGame


class BacktrackingSolver:
    """Deterministic constraint-propagation + backtracking Sudoku solver.

    This solver first applies naked-single constraint propagation to fill
    cells that have only one possible value, then falls back to recursive
    backtracking with MRV (Minimum Remaining Values) heuristic for the
    remaining cells.
    """

    def __init__(self):
        self.steps: int = 0
        self.backtracks: int = 0
        self.solve_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, game: SudokuGame) -> bool:
        """Solve the puzzle in-place on *game.board*.

        Args:
            game: A ``SudokuGame`` instance whose ``board`` will be
                  modified to contain the solution (if one exists).

        Returns:
            ``True`` if the puzzle was solved, ``False`` otherwise.
        """
        self.steps = 0
        self.backtracks = 0

        start = time.perf_counter()
        solved = self._solve(game.board)
        self.solve_time = time.perf_counter() - start

        return solved

    def get_metrics(self) -> dict:
        """Return timing / step metrics from the last ``solve`` call."""
        return {
            "steps": self.steps,
            "backtracks": self.backtracks,
            "solve_time": self.solve_time,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_possible(board: np.ndarray, row: int, col: int) -> Set[int]:
        """Return the set of valid digits for an empty cell."""
        possible = set(range(1, 10))
        possible -= set(board[row, :].tolist())
        possible -= set(board[:, col].tolist())
        sr, sc = (row // 3) * 3, (col // 3) * 3
        possible -= set(board[sr:sr + 3, sc:sc + 3].flatten().tolist())
        possible.discard(0)
        return possible

    @staticmethod
    def _has_duplicate(board: np.ndarray) -> bool:
        """Return True if any row/col/box has duplicate non-zero digits."""
        for i in range(9):
            row = board[i, :][board[i, :] > 0]
            if len(row) != len(set(row.tolist())):
                return True
            col = board[:, i][board[:, i] > 0]
            if len(col) != len(set(col.tolist())):
                return True
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                box = board[br:br + 3, bc:bc + 3].flatten()
                box = box[box > 0]
                if len(box) != len(set(box.tolist())):
                    return True
        return False

    def _propagate(self, board: np.ndarray) -> bool:
        """Fill naked singles until no more progress or a contradiction."""
        if self._has_duplicate(board):
            return False
        changed = True
        while changed:
            changed = False
            for i in range(9):
                for j in range(9):
                    if board[i, j] != 0:
                        continue
                    possible = self._get_possible(board, i, j)
                    if len(possible) == 0:
                        return False  # contradiction
                    if len(possible) == 1:
                        board[i, j] = possible.pop()
                        self.steps += 1
                        changed = True
        return True

    def _find_mrv_cell(self, board: np.ndarray) -> Optional[Tuple[int, int, Set[int]]]:
        """Find the empty cell with the fewest candidates (MRV heuristic)."""
        best: Optional[Tuple[int, int, Set[int]]] = None
        best_len = 10
        for i in range(9):
            for j in range(9):
                if board[i, j] == 0:
                    possible = self._get_possible(board, i, j)
                    if len(possible) < best_len:
                        best_len = len(possible)
                        best = (i, j, possible)
        return best

    def _solve(self, board: np.ndarray) -> bool:
        """Recursive backtracking with constraint propagation."""
        # Save a snapshot so we can restore on backtrack
        snapshot = board.copy()

        if not self._propagate(board):
            np.copyto(board, snapshot)
            return False

        # Check if solved
        if not (board == 0).any():
            return True

        cell = self._find_mrv_cell(board)
        if cell is None:
            return False

        row, col, candidates = cell
        for digit in sorted(candidates):
            self.steps += 1
            board_copy = board.copy()
            board[row, col] = digit

            if self._solve(board):
                return True

            self.backtracks += 1
            np.copyto(board, board_copy)

        np.copyto(board, snapshot)
        return False
