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
    
    # Colors (Blue–Cyan AI Dashboard)
    COLOR_BG : Tuple[int, ...]              = (8, 12, 20)     # overall background (very dark)
    COLOR_CELL_DEFAULT : Tuple[int, ...]    = (10, 16, 28)    # empty cells (darkest surface)
    COLOR_GIVEN : Tuple[int, ...]           = (14, 22, 38)    # given cells (still dark, slightly lifted)
    COLOR_CELL_HOVER : Tuple[int, ...]      = (18, 38, 66)    # hover surface
    COLOR_CELL_SELECTED : Tuple[int, ...]   = (0, 220, 255)   # selection accent (bright cyan)
    COLOR_SOLVED : Tuple[int, ...]          = (80, 244, 255)  # user/agent digits (cyan)
    COLOR_INVALID : Tuple[int, ...]         = (255, 77, 166)  # error (kept visible vs blue theme)
    COLOR_BORDER : Tuple[int, ...]          = (33, 120, 210)  # thin grid lines (blue)
    COLOR_ACCENT : Tuple[int, ...]          = (0, 235, 255)   # HUD accent cyan (panels/outlines)
    COLOR_TEXT : Tuple[int, ...]            = (214, 226, 245) # off-white (panel text)
    COLOR_PANEL_BG : Tuple[int, ...]        = (12, 18, 30)    # info panel background
    COLOR_BOX_BORDER : Tuple[int, ...]      = (0, 180, 255)   # 3×3 box outline
    
    # Animations
    ANIMATION_DURATION: int = 300  # ms
    PARTICLE_LIFETIME: int = 1000  # ms
    
    # Font
    FONT_SIZE_CELL: int = 36
    FONT_SIZE_BUTTON: int = 14

@dataclass
class RewardConfig:
    """Reward shaping settings"""
    VALID_MOVE_REWARD: float = 10.0
    INVALID_MOVE_PENALTY: float = -5.0
    COMPLETION_REWARD: float = 100.0
    ROW_CONFLICT_PENALTY: float = -1.0
    COL_CONFLICT_PENALTY: float = -1.0
    BOX_CONFLICT_PENALTY: float = -1.0
    DUPLICATE_PENALTY: float = -10.0

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
