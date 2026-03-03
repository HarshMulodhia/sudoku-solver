# config.py - Central Configuration for Sudoku RL Solver

from dataclasses import dataclass
from typing import Tuple

@dataclass
class GameConfig:
    """Game settings"""
    BOARD_SIZE: int = 9
    SUBGRID_SIZE: int = 3
    NUM_DIFFICULTIES: int = 3  # Easy, Medium, Hard
    
@dataclass
class RLConfig:
    """Reinforcement Learning settings"""
    # Network Architecture
    INPUT_SHAPE: Tuple[int, ...] = (9, 9, 10)  # Board state representation
    OUTPUT_SIZE: int = 81 * 9  # 81 cells × 9 possible digits
    HIDDEN_LAYERS: list = None
<<<<<<< HEAD
    
    # Learning Parameters
    LEARNING_RATE: float = 0.0001
    GAMMA: float = 0.99  # Discount factor
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.01
    EPSILON_DECAY: float = 0.9995
    BATCH_SIZE: int = 32
    MEMORY_SIZE: int = 100000
    
    # Training
    TARGET_UPDATE_FREQ: int = 1000
    TRAINING_EPISODES: int = 1000
    MAX_STEPS: int = 100
    
    def __post_init__(self):
        if self.HIDDEN_LAYERS is None:
            self.HIDDEN_LAYERS = [256, 256, 128]
=======
    CONV_CHANNELS: list = None
    
    # Learning Parameters
    LEARNING_RATE: float = 0.0005
    GAMMA: float = 0.99  # Discount factor
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_DECAY: float = 0.995
    BATCH_SIZE: int = 128
    MEMORY_SIZE: int = 50000
    
    # Training
    TARGET_UPDATE_FREQ: int = 100
    TRAINING_EPISODES: int = 2000
    MAX_STEPS: int = 81
    
    def __post_init__(self):
        if self.HIDDEN_LAYERS is None:
            self.HIDDEN_LAYERS = [256, 128, 64]
        if self.CONV_CHANNELS is None:
            self.CONV_CHANNELS = [16, 32, 64]
>>>>>>> 9e839a1566699572ecf80c319b1a8ea47ee095ac

@dataclass
class ThemeColors:
    """Color palette for a UI theme"""
    COLOR_BG: Tuple[int, ...]
    COLOR_CELL_DEFAULT: Tuple[int, ...]
    COLOR_GIVEN: Tuple[int, ...]
    COLOR_CELL_HOVER: Tuple[int, ...]
    COLOR_CELL_SELECTED: Tuple[int, ...]
    COLOR_SOLVED: Tuple[int, ...]
    COLOR_INVALID: Tuple[int, ...]
    COLOR_BORDER: Tuple[int, ...]
    COLOR_ACCENT: Tuple[int, ...]
    COLOR_TEXT: Tuple[int, ...]
    COLOR_PANEL_BG: Tuple[int, ...]
    COLOR_BOX_BORDER: Tuple[int, ...]
    COLOR_CELL_HIGHLIGHT: Tuple[int, ...]
    COLOR_BOX_ALT_BG: Tuple[int, ...]

DARK_THEME = ThemeColors(
    COLOR_BG              = (12, 14, 28),
    COLOR_CELL_DEFAULT    = (18, 22, 42),
    COLOR_GIVEN           = (24, 30, 55),
    COLOR_CELL_HOVER      = (30, 50, 90),
    COLOR_CELL_SELECTED   = (0, 200, 255),
    COLOR_SOLVED          = (80, 230, 255),
    COLOR_INVALID         = (255, 70, 120),
    COLOR_BORDER          = (45, 100, 180),
    COLOR_ACCENT          = (0, 220, 255),
    COLOR_TEXT            = (200, 215, 240),
    COLOR_PANEL_BG        = (15, 20, 38),
    COLOR_BOX_BORDER      = (0, 150, 255),
    COLOR_CELL_HIGHLIGHT  = (25, 45, 90),
    COLOR_BOX_ALT_BG      = (22, 28, 50),
)

LIGHT_THEME = ThemeColors(
    COLOR_BG              = (245, 247, 252),
    COLOR_CELL_DEFAULT    = (255, 255, 255),
    COLOR_GIVEN           = (232, 238, 248),
    COLOR_CELL_HOVER      = (210, 225, 250),
    COLOR_CELL_SELECTED   = (50, 110, 220),
    COLOR_SOLVED          = (30, 90, 195),
    COLOR_INVALID         = (220, 55, 55),
    COLOR_BORDER          = (175, 190, 215),
    COLOR_ACCENT          = (40, 95, 200),
    COLOR_TEXT            = (25, 30, 45),
    COLOR_PANEL_BG        = (238, 242, 250),
    COLOR_BOX_BORDER      = (35, 75, 160),
    COLOR_CELL_HIGHLIGHT  = (205, 220, 245),
    COLOR_BOX_ALT_BG      = (240, 243, 250),
)

@dataclass
class UIConfig:
    """Pygame UI settings"""
    # Window
    WINDOW_WIDTH: int = 1200
    WINDOW_HEIGHT: int = 900
    FPS: int = 60
    
    # Board Display
    BOARD_X: int = 50
    BOARD_Y: int = 50
    CELL_SIZE: int = 80
    BORDER_WIDTH: int = 2
    SUBGRID_BORDER_WIDTH: int = 4
    
    # Theme mode (True = dark, False = light)
    dark_mode: bool = True
    
    # Animations
    ANIMATION_DURATION: int = 300  # ms
    PARTICLE_LIFETIME: int = 1000  # ms
    
    # Font
    FONT_SIZE_CELL: int = 36
    FONT_SIZE_BUTTON: int = 14
    
    def get_theme(self) -> ThemeColors:
        """Return the active color theme based on dark_mode flag"""
        return DARK_THEME if self.dark_mode else LIGHT_THEME

@dataclass
class RewardConfig:
    """Reward shaping settings"""
<<<<<<< HEAD
    VALID_MOVE_REWARD: float = 10.0
    INVALID_MOVE_PENALTY: float = -5.0
    COMPLETION_REWARD: float = 100.0
    ROW_CONFLICT_PENALTY: float = -1.0
    COL_CONFLICT_PENALTY: float = -1.0
    BOX_CONFLICT_PENALTY: float = -1.0
    DUPLICATE_PENALTY: float = -10.0
=======
    VALID_MOVE_REWARD: float = 1.0
    INVALID_MOVE_PENALTY: float = -10.0
    COMPLETION_REWARD: float = 200.0
    ROW_CONFLICT_PENALTY: float = -5.0
    COL_CONFLICT_PENALTY: float = -5.0
    BOX_CONFLICT_PENALTY: float = -5.0
    DUPLICATE_PENALTY: float = -10.0
    CORRECT_MOVE_REWARD: float = 10.0
    WRONG_MOVE_PENALTY: float = -10.0
>>>>>>> 9e839a1566699572ecf80c319b1a8ea47ee095ac

# Instantiate configs
game_config = GameConfig()
rl_config = RLConfig()
ui_config = UIConfig()
reward_config = RewardConfig()

# Puzzle difficulties (number of givens)
DIFFICULTY_LEVELS = {
    'easy': 40,
    'medium': 30,
    'hard': 20
}
