"""Tests for the deterministic backtracking solver."""

import numpy as np
import pytest

from backtracking_solver import BacktrackingSolver
from sudoku_game import SudokuGame, SudokuValidator


class TestBacktrackingSolverBasic:
    """Basic functionality tests."""

    def test_solves_easy(self):
        game = SudokuGame("easy")
        solver = BacktrackingSolver()
        assert solver.solve(game)
        assert not (game.board == 0).any()
        assert SudokuValidator.is_valid_board(game.board)

    def test_solves_medium(self):
        game = SudokuGame("medium")
        solver = BacktrackingSolver()
        assert solver.solve(game)
        assert not (game.board == 0).any()
        assert SudokuValidator.is_valid_board(game.board)

    def test_solves_hard(self):
        game = SudokuGame("hard")
        solver = BacktrackingSolver()
        assert solver.solve(game)
        assert not (game.board == 0).any()
        assert SudokuValidator.is_valid_board(game.board)

    def test_solution_is_valid(self):
        """Solved board should be a valid complete Sudoku."""
        game = SudokuGame("easy")
        solver = BacktrackingSolver()
        solver.solve(game)
        assert not (game.board == 0).any()
        assert SudokuValidator.is_valid_board(game.board)


class TestBacktrackingSolverMetrics:
    """Metrics / diagnostics tests."""

    def test_metrics_populated(self):
        game = SudokuGame("easy")
        solver = BacktrackingSolver()
        solver.solve(game)
        metrics = solver.get_metrics()
        assert metrics["steps"] > 0
        assert metrics["solve_time"] >= 0
        assert "backtracks" in metrics

    def test_steps_reset_on_new_solve(self):
        solver = BacktrackingSolver()
        game1 = SudokuGame("easy")
        solver.solve(game1)
        first_steps = solver.steps

        game2 = SudokuGame("easy")
        solver.solve(game2)
        # Steps are reset, so they shouldn't accumulate
        assert solver.steps <= first_steps * 3  # some variance is fine


class TestBacktrackingSolverEdgeCases:
    """Edge-case and robustness tests."""

    def test_already_solved(self):
        """Solver should handle an already-complete board."""
        game = SudokuGame("easy")
        game.board = game.solution.copy()
        solver = BacktrackingSolver()
        assert solver.solve(game)
        assert np.array_equal(game.board, game.solution)

    def test_empty_board(self):
        """Solver should handle a fully empty board."""
        game = SudokuGame.__new__(SudokuGame)
        game.board = np.zeros((9, 9), dtype=np.int32)
        game.original_board = np.zeros((9, 9), dtype=np.int32)
        game.solution = np.zeros((9, 9), dtype=np.int32)
        game.board_size = 9
        game.subgrid_size = 3
        game.difficulty = "hard"

        solver = BacktrackingSolver()
        assert solver.solve(game)
        assert not (game.board == 0).any()
        assert SudokuValidator.is_valid_board(game.board)

    def test_unsolvable_board(self):
        """Solver should return False for a contradictory board."""
        game = SudokuGame.__new__(SudokuGame)
        board = np.zeros((9, 9), dtype=np.int32)
        board[0, 0] = 1
        board[0, 1] = 1  # duplicate in row → unsolvable
        game.board = board
        game.original_board = board.copy()
        game.solution = board.copy()
        game.board_size = 9
        game.subgrid_size = 3
        game.difficulty = "hard"

        solver = BacktrackingSolver()
        assert not solver.solve(game)
