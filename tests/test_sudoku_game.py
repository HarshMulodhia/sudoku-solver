"""Tests for SudokuGame core logic."""

import numpy as np
import pytest

from sudoku_game import SudokuGame, SudokuValidator
from config import DIFFICULTY_LEVELS


class TestSudokuGameInit:
    """Test game initialization and puzzle generation."""

    def test_board_shape(self):
        game = SudokuGame("easy")
        assert game.board.shape == (9, 9)

    def test_solution_shape(self):
        game = SudokuGame("easy")
        assert game.solution.shape == (9, 9)

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_difficulty_givens(self, difficulty):
        game = SudokuGame(difficulty)
        expected_givens = DIFFICULTY_LEVELS[difficulty]
        actual_givens = int(np.sum(game.original_board > 0))
        assert actual_givens == expected_givens

    def test_solution_is_complete(self):
        game = SudokuGame("easy")
        assert not (game.solution == 0).any()

    def test_solution_is_valid(self):
        game = SudokuGame("easy")
        assert SudokuValidator.is_valid_board(game.solution)


class TestSudokuGameMoves:
    """Test move validation and placement."""

    def test_valid_move_check(self):
        game = SudokuGame("easy")
        board = np.zeros((9, 9), dtype=np.int32)
        assert game.is_valid_move(board, 0, 0, 1)

    def test_row_conflict(self):
        game = SudokuGame("easy")
        board = np.zeros((9, 9), dtype=np.int32)
        board[0, 0] = 5
        assert not game.is_valid_move(board, 0, 1, 5)

    def test_col_conflict(self):
        game = SudokuGame("easy")
        board = np.zeros((9, 9), dtype=np.int32)
        board[0, 0] = 5
        assert not game.is_valid_move(board, 1, 0, 5)

    def test_box_conflict(self):
        game = SudokuGame("easy")
        board = np.zeros((9, 9), dtype=np.int32)
        board[0, 0] = 5
        assert not game.is_valid_move(board, 1, 1, 5)

    def test_cannot_modify_given_cell(self):
        game = SudokuGame("easy")
        # Find a given cell
        given = np.argwhere(game.original_board > 0)[0]
        assert not game.place_digit(given[0], given[1], 1)

    def test_place_and_clear(self):
        game = SudokuGame("easy")
        # Find an empty cell
        empty = np.argwhere(game.board == 0)[0]
        row, col = empty[0], empty[1]
        possible = game.get_possible_values(row, col)
        if possible:
            digit = list(possible)[0]
            assert game.place_digit(row, col, digit)
            assert game.board[row, col] == digit
            assert game.place_digit(row, col, 0)
            assert game.board[row, col] == 0

<<<<<<< HEAD
=======
    def test_force_place_invalid_digit(self):
        """force=True allows placing a digit that violates constraints"""
        game = SudokuGame("easy")
        empty = np.argwhere(game.board == 0)[0]
        row, col = empty[0], empty[1]
        # Pick a digit already present in the row (guaranteed conflict)
        row_vals = set(game.board[row, :]) - {0}
        if row_vals:
            conflict_digit = list(row_vals)[0]
            # Without force, placement is rejected
            assert not game.place_digit(row, col, conflict_digit)
            # With force, placement is accepted
            assert game.place_digit(row, col, conflict_digit, force=True)
            assert game.board[row, col] == conflict_digit

    def test_force_cannot_modify_given(self):
        """force=True still cannot modify given (original) cells"""
        game = SudokuGame("easy")
        given = np.argwhere(game.original_board > 0)[0]
        assert not game.place_digit(given[0], given[1], 1, force=True)

>>>>>>> 9e839a1566699572ecf80c319b1a8ea47ee095ac

class TestSudokuGameState:
    """Test state queries."""

    def test_get_possible_values_for_filled(self):
        game = SudokuGame("easy")
        given = np.argwhere(game.original_board > 0)[0]
        assert game.get_possible_values(given[0], given[1]) == set()

    def test_get_possible_values_for_empty(self):
        game = SudokuGame("easy")
        empty = np.argwhere(game.board == 0)[0]
        possible = game.get_possible_values(empty[0], empty[1])
        assert len(possible) > 0
        assert all(1 <= d <= 9 for d in possible)

    def test_is_complete_false(self):
        game = SudokuGame("easy")
        assert not game.is_complete()

    def test_encoded_state_shape(self):
        game = SudokuGame("easy")
        state = game.get_encoded_state()
        assert state.shape == (9, 9, 10)
        assert state.dtype == np.float32

    def test_encoded_state_one_hot(self):
        game = SudokuGame("easy")
        state = game.get_encoded_state()
        # Each cell should have exactly one channel set to 1
        for i in range(9):
            for j in range(9):
                assert np.sum(state[i, j]) == 1.0

    def test_get_conflicts_on_valid_board(self):
        game = SudokuGame("easy")
        conflicts = game.get_conflicts()
        assert len(conflicts) == 0

    def test_reset(self):
        game = SudokuGame("easy")
        empty = np.argwhere(game.board == 0)[0]
        possible = game.get_possible_values(empty[0], empty[1])
        if possible:
            game.place_digit(empty[0], empty[1], list(possible)[0])
        game.reset()
        assert np.array_equal(game.board, game.original_board)

    def test_copy(self):
        game = SudokuGame("easy")
        game_copy = game.copy()
        assert np.array_equal(game.board, game_copy.board)
        # Mutating copy should not affect original
        empty = np.argwhere(game_copy.board == 0)[0]
        possible = game_copy.get_possible_values(empty[0], empty[1])
        if possible:
            game_copy.place_digit(empty[0], empty[1], list(possible)[0])
            assert not np.array_equal(game.board, game_copy.board)

    def test_solve_hint(self):
        game = SudokuGame("easy")
        empty = np.argwhere(game.board == 0)[0]
        hint = game.solve_hint(empty[0], empty[1])
        assert hint is not None
        assert 1 <= hint <= 9
        assert hint == int(game.solution[empty[0], empty[1]])


class TestSudokuValidator:
    """Test static validator."""

    def test_valid_board(self):
        game = SudokuGame("easy")
        assert SudokuValidator.is_valid_board(game.solution)

    def test_invalid_row(self):
        board = np.zeros((9, 9), dtype=np.int32)
        board[0, 0] = 1
        board[0, 1] = 1
        assert not SudokuValidator.is_valid_board(board)

    def test_invalid_col(self):
        board = np.zeros((9, 9), dtype=np.int32)
        board[0, 0] = 1
        board[1, 0] = 1
        assert not SudokuValidator.is_valid_board(board)

    def test_invalid_box(self):
        board = np.zeros((9, 9), dtype=np.int32)
        board[0, 0] = 1
        board[1, 1] = 1
        assert not SudokuValidator.is_valid_board(board)

    def test_empty_board_valid(self):
        board = np.zeros((9, 9), dtype=np.int32)
        assert SudokuValidator.is_valid_board(board)
