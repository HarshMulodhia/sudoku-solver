"""Tests for pygame_ui highlighting and theme logic"""

import pytest
import sys
import numpy as np
import time as _time
from unittest.mock import patch, MagicMock

# Mock pygame before importing pygame_ui
sys.modules['pygame'] = MagicMock()
sys.modules['pygame.font'] = MagicMock()
sys.modules['pygame.display'] = MagicMock()

from config import ui_config, DARK_THEME, LIGHT_THEME
from sudoku_game import SudokuGame
from pygame_ui import SudokuUI, Button


@pytest.fixture
def ui():
    """Create a SudokuUI with mocked pygame for testing logic"""
    game = SudokuGame(difficulty='easy')
    with patch.object(SudokuUI, '__init__', lambda self, g: None):
        ui = SudokuUI.__new__(SudokuUI)
        ui.game = game
        ui.selected_cell = None
        ui.hover_cell = None
        ui.colors = ui_config.get_theme()
        ui.text_cache = {}
        ui.mode = 'manual'
        ui.start_time = _time.time()
        ui.score = 0
        ui.undo_stack = []
    return ui


class TestIsHighlighted:
    """Tests for the is_highlighted method"""

    def test_no_selection_returns_false(self, ui):
        """No cell should be highlighted when nothing is selected"""
        ui.selected_cell = None
        assert ui.is_highlighted(0, 0) is False
        assert ui.is_highlighted(4, 4) is False
        assert ui.is_highlighted(8, 8) is False

    def test_same_row_highlighted(self, ui):
        """Cells in the same row as the selected cell should be highlighted"""
        ui.selected_cell = (3, 4)
        # Same row (row 3), different columns
        assert ui.is_highlighted(3, 0) is True
        assert ui.is_highlighted(3, 8) is True
        assert ui.is_highlighted(3, 2) is True

    def test_same_col_highlighted(self, ui):
        """Cells in the same column as selected cell should be highlighted"""
        ui.selected_cell = (3, 4)
        # Same column (col 4), different rows
        assert ui.is_highlighted(0, 4) is True
        assert ui.is_highlighted(8, 4) is True
        assert ui.is_highlighted(6, 4) is True

    def test_same_box_highlighted(self, ui):
        """Cells in the same 3x3 box as selected cell should be highlighted"""
        ui.selected_cell = (4, 4)  # Center of middle box (rows 3-5, cols 3-5)
        # Same box cells
        assert ui.is_highlighted(3, 3) is True
        assert ui.is_highlighted(3, 5) is True
        assert ui.is_highlighted(5, 3) is True
        assert ui.is_highlighted(5, 5) is True

    def test_unrelated_cell_not_highlighted(self, ui):
        """Cells outside the row, column, and box should not be highlighted"""
        ui.selected_cell = (4, 4)  # Center of middle box
        # Cell not in same row, col, or box
        assert ui.is_highlighted(0, 0) is False
        assert ui.is_highlighted(7, 7) is False
        assert ui.is_highlighted(0, 8) is False

    def test_selected_cell_itself_highlighted(self, ui):
        """The selected cell itself is in same row/col, so is_highlighted is True"""
        ui.selected_cell = (4, 4)
        assert ui.is_highlighted(4, 4) is True

    def test_corner_selection(self, ui):
        """Test highlighting when a corner cell is selected"""
        ui.selected_cell = (0, 0)
        # Same row
        assert ui.is_highlighted(0, 5) is True
        # Same col
        assert ui.is_highlighted(5, 0) is True
        # Same box (0,0 box)
        assert ui.is_highlighted(1, 1) is True
        assert ui.is_highlighted(2, 2) is True
        # Not related
        assert ui.is_highlighted(5, 5) is False

    def test_bottom_right_selection(self, ui):
        """Test highlighting with bottom-right corner selected"""
        ui.selected_cell = (8, 8)
        # Same row
        assert ui.is_highlighted(8, 0) is True
        # Same col
        assert ui.is_highlighted(0, 8) is True
        # Same box (rows 6-8, cols 6-8)
        assert ui.is_highlighted(6, 6) is True
        assert ui.is_highlighted(7, 7) is True
        # Not related
        assert ui.is_highlighted(4, 4) is False


class TestToggleTheme:
    """Tests for the toggle_theme method"""

    def test_toggle_switches_to_light(self, ui):
        """Toggling from dark mode should switch to light theme"""
        ui_config.dark_mode = True
        ui.colors = ui_config.get_theme()
        ui.toggle_theme()
        assert ui_config.dark_mode is False
        assert ui.colors == LIGHT_THEME

    def test_toggle_switches_to_dark(self, ui):
        """Toggling from light mode should switch back to dark theme"""
        ui_config.dark_mode = False
        ui.colors = ui_config.get_theme()
        ui.toggle_theme()
        assert ui_config.dark_mode is True
        assert ui.colors == DARK_THEME

    def test_toggle_clears_text_cache(self, ui):
        """Toggling theme should clear the text cache"""
        ui.text_cache = {"some_key": "some_value"}
        ui_config.dark_mode = True
        ui.toggle_theme()
        assert ui.text_cache == {}

    def test_double_toggle_restores_theme(self, ui):
        """Toggling twice should restore the original theme"""
        ui_config.dark_mode = True
        original = ui_config.get_theme()
        ui.toggle_theme()
        ui.toggle_theme()
        assert ui_config.dark_mode is True
        assert ui.colors == original


class TestButton:
    """Tests for the Button class"""

    def _make_button(self, x, y, w, h, text, active=False):
        """Create a Button with a real-rect-like object for testing"""
        btn = Button(x, y, w, h, text, active=active)
        # Replace mocked pygame.Rect with a simple implementation
        class SimpleRect:
            def __init__(self, rx, ry, rw, rh):
                self.x, self.y, self.w, self.h = rx, ry, rw, rh
                self.center = (rx + rw // 2, ry + rh // 2)
            def collidepoint(self, pos):
                return (self.x <= pos[0] <= self.x + self.w and
                        self.y <= pos[1] <= self.y + self.h)
        btn.rect = SimpleRect(x, y, w, h)
        return btn

    def test_is_clicked_inside(self):
        btn = self._make_button(10, 10, 100, 30, "Test")
        assert btn.is_clicked((50, 25))

    def test_is_clicked_outside(self):
        btn = self._make_button(10, 10, 100, 30, "Test")
        assert not btn.is_clicked((200, 200))

    def test_update_hover(self):
        btn = self._make_button(10, 10, 100, 30, "Test")
        btn.update_hover((50, 25))
        assert btn.hovered is True
        btn.update_hover((200, 200))
        assert btn.hovered is False

    def test_active_state(self):
        btn = self._make_button(10, 10, 100, 30, "Test", active=True)
        assert btn.active is True


class TestSetMode:
    """Tests for mode switching"""

    def test_set_mode_manual(self, ui):
        ui.btn_manual = Button(0, 0, 10, 10, "M", active=False)
        ui.btn_rl = Button(0, 0, 10, 10, "R", active=False)
        ui.btn_backtrack = Button(0, 0, 10, 10, "B", active=False)
        ui.mode_buttons = [ui.btn_manual, ui.btn_rl, ui.btn_backtrack]
        ui.set_mode('manual')
        assert ui.mode == 'manual'
        assert ui.btn_manual.active is True
        assert ui.btn_rl.active is False

    def test_set_mode_rl(self, ui):
        ui.btn_manual = Button(0, 0, 10, 10, "M", active=False)
        ui.btn_rl = Button(0, 0, 10, 10, "R", active=False)
        ui.btn_backtrack = Button(0, 0, 10, 10, "B", active=False)
        ui.mode_buttons = [ui.btn_manual, ui.btn_rl, ui.btn_backtrack]
        ui.set_mode('rl')
        assert ui.mode == 'rl'
        assert ui.btn_rl.active is True
        assert ui.btn_manual.active is False


class TestScoreAndTimer:
    """Tests for score computation and timer"""

    def test_compute_score_empty_board(self, ui):
        """Score should be 0 when no extra digits placed"""
        assert ui.compute_score() == 0

    def test_compute_score_correct_placement(self, ui):
        """Score increases when correct digit placed"""
        game = ui.game
        empty = np.argwhere(game.board == 0)[0]
        row, col = int(empty[0]), int(empty[1])
        correct = int(game.solution[row, col])
        game.place_digit(row, col, correct, force=True)
        assert ui.compute_score() >= 1

    def test_compute_score_wrong_placement(self, ui):
        """Wrong digit should not increase score"""
        game = ui.game
        empty = np.argwhere(game.board == 0)[0]
        row, col = int(empty[0]), int(empty[1])
        correct = int(game.solution[row, col])
        wrong = (correct % 9) + 1
        game.place_digit(row, col, wrong, force=True)
        assert ui.compute_score() == 0

    def test_get_elapsed_format(self, ui):
        """Elapsed time should be in mm:ss format"""
        elapsed = ui.get_elapsed()
        assert len(elapsed) == 5
        assert elapsed[2] == ':'

    def test_reset_timer(self, ui):
        """reset_timer should reset start_time and score"""
        ui.score = 10
        ui.reset_timer()
        assert ui.score == 0


class TestIsDigitHighlighted:
    """Tests for the is_digit_highlighted method"""

    def test_no_selection_returns_false(self, ui):
        """No digit highlight when nothing is selected"""
        ui.selected_cell = None
        assert ui.is_digit_highlighted(0, 0) is False

    def test_selected_empty_cell_returns_false(self, ui):
        """No digit highlight when selected cell is empty"""
        empty = np.argwhere(ui.game.board == 0)[0]
        ui.selected_cell = (int(empty[0]), int(empty[1]))
        # Any other cell should not be digit-highlighted
        assert ui.is_digit_highlighted(0, 0) is False

    def test_same_digit_highlighted(self, ui):
        """Cells with the same digit as the selected cell are highlighted"""
        # Find a given cell with a non-zero digit
        given = np.argwhere(ui.game.original_board > 0)
        row0, col0 = int(given[0][0]), int(given[0][1])
        digit = int(ui.game.board[row0, col0])
        ui.selected_cell = (row0, col0)

        # Find another cell with the same digit
        matches = np.argwhere(ui.game.board == digit)
        if len(matches) > 1:
            row1, col1 = int(matches[1][0]), int(matches[1][1])
            assert ui.is_digit_highlighted(row1, col1)

    def test_different_digit_not_highlighted(self, ui):
        """Cells with different digits are not digit-highlighted"""
        given = np.argwhere(ui.game.original_board > 0)
        row0, col0 = int(given[0][0]), int(given[0][1])
        digit = int(ui.game.board[row0, col0])
        ui.selected_cell = (row0, col0)

        # Find a cell with a different non-zero digit
        for idx in range(len(given)):
            r, c = int(given[idx][0]), int(given[idx][1])
            if ui.game.board[r, c] != digit:
                assert not ui.is_digit_highlighted(r, c)
                break


class TestSetDifficulty:
    """Tests for the set_difficulty method"""

    def test_set_difficulty_easy(self, ui):
        ui.btn_easy = Button(0, 0, 10, 10, "E", active=False)
        ui.btn_medium = Button(0, 0, 10, 10, "M", active=False)
        ui.btn_hard = Button(0, 0, 10, 10, "H", active=False)
        ui.difficulty_buttons = [ui.btn_easy, ui.btn_medium, ui.btn_hard]
        ui.set_difficulty('easy')
        assert ui.btn_easy.active is True
        assert ui.btn_medium.active is False
        assert ui.btn_hard.active is False

    def test_set_difficulty_hard(self, ui):
        ui.btn_easy = Button(0, 0, 10, 10, "E", active=False)
        ui.btn_medium = Button(0, 0, 10, 10, "M", active=False)
        ui.btn_hard = Button(0, 0, 10, 10, "H", active=False)
        ui.difficulty_buttons = [ui.btn_easy, ui.btn_medium, ui.btn_hard]
        ui.set_difficulty('hard')
        assert ui.btn_hard.active is True
        assert ui.btn_easy.active is False


class TestUndoStack:
    """Tests for the undo stack"""

    def test_undo_stack_initially_empty(self, ui):
        """Undo stack should be empty on init"""
        assert ui.undo_stack == []

    def test_undo_stack_records_moves(self, ui):
        """Appending to undo stack records moves"""
        ui.undo_stack.append((0, 0, 0))
        assert len(ui.undo_stack) == 1
        assert ui.undo_stack[0] == (0, 0, 0)

    def test_undo_stack_pop(self, ui):
        """Popping from undo stack returns last move"""
        ui.undo_stack.append((1, 2, 3))
        ui.undo_stack.append((4, 5, 6))
        row, col, prev = ui.undo_stack.pop()
        assert (row, col, prev) == (4, 5, 6)
        assert len(ui.undo_stack) == 1
