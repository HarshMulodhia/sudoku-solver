"""Tests for pygame_ui highlighting and theme logic"""

import pytest
import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Mock pygame before importing pygame_ui
sys.modules['pygame'] = MagicMock()
sys.modules['pygame.font'] = MagicMock()
sys.modules['pygame.display'] = MagicMock()

from config import ui_config, DARK_THEME, LIGHT_THEME
from sudoku_game import SudokuGame
from pygame_ui import SudokuUI


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
